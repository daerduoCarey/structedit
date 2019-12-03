"""
    The main StructEdit network architecture
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from chamfer_distance import ChamferDistance
from data import Tree
from utils import linear_assignment, load_pts, transform_pc_batch, get_surface_reweighting_batch
from quaternion import qmul
from utils import compute_box_diff


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = x + self.mlp2mu(encode)

        if self.probabilistic:
            logvar = x + self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class BoxEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxEncoder, self).__init__()
        self.mlp_skip = nn.Linear(10, feature_size)
        self.mlp1 = nn.Linear(10, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, box_input):
        net = torch.relu(self.mlp1(box_input))
        net = torch.relu(self.mlp_skip(box_input) + self.mlp2(net))
        return net


class BoxDiffEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDiffEncoder, self).__init__()
        self.mlp_skip = nn.Linear(10, feature_size)
        self.mlp1 = nn.Linear(10, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, box_input):
        net = torch.relu(self.mlp1(box_input))
        net = torch.relu(self.mlp_skip(box_input) + self.mlp2(net))
        return net


class SymmetricChildEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, symmetric_type):
        super(SymmetricChildEncoder, self).__init__()

        print(f'Using Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type

        self.child_op = nn.Linear(feature_size + Tree.num_sem, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.second_norm = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)
        self.skip_op = nn.Linear(feature_size + Tree.num_sem, feature_size)

    def forward(self, child_feats):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        feat_size = child_feats.shape[2]

        # sum over child features (in larger feature space, using hidden_size)
        skip_feats = self.skip_op(child_feats)
        child_feats = self.child_op(child_feats)

        if self.symmetric_type == 'max':
            parent_feat = torch.relu(child_feats.max(dim=1)[0])
            skip_feat = torch.relu(skip_feats.max(dim=1)[0])
        elif self.symmetric_type == 'sum':
            parent_feat = torch.relu(child_feats.sum(dim=1))
            skip_feat = torch.relu(skip_feats.sum(dim=1))
        elif self.symmetric_type == 'avg':
            parent_feat = torch.relu(child_feats.sum(dim=1) / child_feats.size(1))
            skip_feat = torch.relu(skip_feats.sum(dim=1) / skip_feats.size(1))
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        # back to standard feature space size
        parent_feat = torch.relu(skip_feat + self.second_norm(self.second(parent_feat)))
        return parent_feat


class SymmetricChildDiffEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, symmetric_type):
        super(SymmetricChildDiffEncoder, self).__init__()

        print(f'Using Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type

        self.child_op = nn.Linear(3 * feature_size + Tree.num_sem + 4, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.second_norm = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)
        self.skip_op = nn.Linear(3 * feature_size + Tree.num_sem + 4, feature_size)

    def forward(self, child_feats):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        feat_size = child_feats.shape[2]

        # sum over child features (in larger feature space, using hidden_size)
        skip_feats = self.skip_op(child_feats)
        child_feats = self.child_op(child_feats)

        if self.symmetric_type == 'max':
            parent_feat = torch.relu(child_feats.max(dim=1)[0])
            skip_feat = torch.relu(skip_feats.max(dim=1)[0])
        elif self.symmetric_type == 'sum':
            parent_feat = torch.relu(child_feats.sum(dim=1))
            skip_feat = torch.relu(skip_feats.sum(dim=1))
        elif self.symmetric_type == 'avg':
            parent_feat = torch.relu(child_feats.sum(dim=1) / child_feats.size(1))
            skip_feat = torch.relu(skip_feats.sum(dim=1) / skip_feats.size(1))
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        # back to standard feature space size
        parent_feat = torch.relu(skip_feat + self.second_norm(self.second(parent_feat)))
        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, variational=False, probabilistic=True):
        super(RecursiveEncoder, self).__init__()
        self.conf = config

        self.box_encoder = BoxEncoder(config.feature_size, config.hidden_size)
        self.box_diff_encoder = BoxDiffEncoder(config.feature_size, config.hidden_size)
        self.child_encoder = SymmetricChildEncoder(config.feature_size, config.hidden_size, config.node_symmetric_type)

        self.child_diff_encoder = SymmetricChildDiffEncoder(config.feature_size, config.hidden_size, config.node_symmetric_type)

        if variational:
            self.sample_encoder = Sampler(config.feature_size, config.hidden_size, probabilistic)

    def apply_box_diff(self, box, box_diff):
        center = box[0, :3] + box_diff[0, :3]
        shape = box[0, 3:6] + box_diff[0, 3:6]
        q = qmul(box_diff[0, 6:], box[0, 6:])
        return torch.cat([center, shape, q]).view(1, -1)

    def encode_node(self, node):
        ret = self.box_encoder(node.get_box_quat())
        node.box_feature = ret

        if not node.is_leaf:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(child), child.get_semantic_one_hot()], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            # get feature of current node (parent of the children)
            ret = self.child_encoder(child_feats)

        # keep track of the feature for each subtree
        node.feature = ret
        return ret

    def encode_node_diff(self, node, diff):
        child_feats = []

        for i, cnode in enumerate(node.children):
            cdiff = diff.children[i]
            if cdiff.node_type == 'DEL':
                subtree_feat = cnode.feature
                one_hot_feat = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32, device=cnode.feature.device)
                cur_node_feat = self.box_encoder(cnode.get_box_quat())
                diff_feat = torch.zeros_like(cnode.feature)
            elif cdiff.node_type == 'SAME':
                subtree_feat = self.encode_node_diff(cnode, cdiff)
                one_hot_feat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=subtree_feat.device)
                cur_node_feat = self.box_encoder(cnode.get_box_quat())
                diff_feat = self.box_diff_encoder(cdiff.box_diff)
            elif cdiff.node_type == 'LEAF':
                cur_node_feat = self.box_encoder(cnode.get_box_quat())
                diff_feat = self.box_diff_encoder(cdiff.box_diff)
                subtree_feat = torch.zeros_like(cur_node_feat)
                one_hot_feat = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=subtree_feat.device)
            else:
                raise ValueError('Unknown diffnode type %s within editing range! [DEL/SAME only]' % cdiff.node_type)
            cur_child_feat = torch.cat([subtree_feat, cur_node_feat, diff_feat, cnode.get_semantic_one_hot(), one_hot_feat], dim=1)
            child_feats.append(cur_child_feat.unsqueeze(1))

        for i in range(len(node.children), len(diff.children)):
            cdiff = diff.children[i]
            if cdiff.node_type == 'ADD':
                subtree_feat = self.encode_node(cdiff.subnode)
                cur_node_feat = torch.zeros_like(subtree_feat)
                cur_node2_feat = self.box_encoder(cdiff.subnode.get_box_quat())
                one_hot_feat = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32, device=subtree_feat.device)
            else:
                raise ValueError('Unknown diffnode type %s outside editing range! [ADD only]' % cdiff.node_type)
            cur_child_feat = torch.cat([subtree_feat, cur_node_feat, cur_node2_feat, cdiff.subnode.get_semantic_one_hot(), one_hot_feat], dim=1)
            child_feats.append(cur_child_feat.unsqueeze(1))

        if len(child_feats) == 0:
            return torch.zeros(1, self.conf.feature_size, dtype=torch.float32, device=self.conf.device)
        else:
            child_feats = torch.cat(child_feats, dim=1)
            return self.child_diff_encoder(child_feats)

    def encode_tree(self, obj):
        return self.encode_node(obj.root)

    def encode_tree_diff(self, obj, diff):
        root_latent = self.encode_node_diff(obj.root, diff)
        return self.sample_encoder(root_latent)


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)
        self.skip = nn.Linear(feature_size, 1)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = self.skip(input_feature) + self.mlp2(output)
        return output


class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size*2, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.skip = nn.Linear(feature_size*2, feature_size)

    def forward(self, diff_feat, input_feat):
        feat = torch.cat([diff_feat, input_feat], dim=1)
        feat2 = torch.relu(self.mlp1(feat))
        output = torch.relu(self.skip(feat) + self.mlp2(feat2))
        return output


class BoxDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.center = nn.Linear(hidden_size, 3)
        self.size = nn.Linear(hidden_size, 3)
        self.quat = nn.Linear(hidden_size, 4)
        self.center_skip = nn.Linear(feature_size, 3)
        self.size_skip = nn.Linear(feature_size, 3)
        self.quat_skip = nn.Linear(feature_size, 4)

    def forward(self, feat):
        hidden = torch.relu(self.mlp(feat))
        center = torch.tanh(self.center_skip(feat) + self.center(hidden))
        size = torch.sigmoid(self.size_skip(feat) + self.size(hidden)) * 2
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = (self.quat_skip(feat) + self.quat(hidden)).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, quat], dim=1)
        return vector


class BoxDiffDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDiffDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.center = nn.Linear(hidden_size, 3)
        self.size = nn.Linear(hidden_size, 3)
        self.quat = nn.Linear(hidden_size, 4)
        self.center_skip = nn.Linear(feature_size, 3)
        self.size_skip = nn.Linear(feature_size, 3)
        self.quat_skip = nn.Linear(feature_size, 4)

    def forward(self, feat):
        hidden = torch.relu(self.mlp(feat))
        feat = torch.relu(self.mlp(feat))
        center = self.center_skip(feat) + self.center(hidden)
        size = self.size_skip(feat) + self.size(hidden)
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = (self.quat_skip(feat) + self.quat(hidden)).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, quat], dim=1)
        return vector


class NodeDiffFeatureExtractor(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeDiffFeatureExtractor, self).__init__()
        self.mlp1 = nn.Linear(4 * feature_size + Tree.num_sem, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.skip = nn.Linear(4 * feature_size + Tree.num_sem, feature_size)

    def forward(self, x):
        hidden = torch.relu(self.mlp1(x))
        return torch.relu(self.skip(x) + self.mlp2(hidden))


class NodeDiffClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeDiffClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 4)
        self.skip = nn.Linear(feature_size, 4)

    def forward(self, x):
        hidden = torch.relu(self.mlp1(x))
        return self.skip(x) + self.mlp2(hidden)


class ConcatChildDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num):
        super(ConcatChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.mlp_parent = nn.Linear(feature_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child = nn.Linear(hidden_size, feature_size)
        self.norm_child = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(-1, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # node features
        child_feats = self.norm_child(self.mlp_child(parent_feature.view(-1, self.hidden_size)))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits


class AddChildDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num):
        super(AddChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.mlp_parent = nn.Linear(feature_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child = nn.Linear(hidden_size, feature_size)
        self.norm_child = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(-1, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # node features
        child_feats = self.norm_child(self.mlp_child(parent_feature.view(-1, self.hidden_size)))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits


class RecursiveDecoder(nn.Module):

    def __init__(self, config):
        super(RecursiveDecoder, self).__init__()

        self.conf = config

        self.box_decoder = BoxDecoder(config.feature_size, config.hidden_size)
        self.box_diff_decoder = BoxDiffDecoder(config.feature_size, config.hidden_size)
        self.child_decoder = ConcatChildDecoder(config.feature_size, config.hidden_size, config.max_child_num)
        self.sample_decoder = SampleDecoder(config.feature_size, config.hidden_size)
        self.leaf_classifier = LeafClassifier(config.feature_size, config.hidden_size)
        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.ceLoss = nn.CrossEntropyLoss(reduction='none')

        self.node_diff_feature_extractor = NodeDiffFeatureExtractor(config.feature_size, config.hidden_size)
        self.node_diff_classifier = NodeDiffClassifier(config.feature_size, config.hidden_size)
        self.add_child_decoder = AddChildDecoder(config.feature_size, config.hidden_size, config.max_child_num)

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))

    def boxLossEstimator(self, box_feature, gt_box_feature):
        pred_box_pc = transform_pc_batch(self.unit_cube, box_feature)
        with torch.no_grad():
            pred_reweight = get_surface_reweighting_batch(box_feature[:, 3:6], self.unit_cube.size(0))
        gt_box_pc = transform_pc_batch(self.unit_cube, gt_box_feature)
        with torch.no_grad():
            gt_reweight = get_surface_reweighting_batch(gt_box_feature[:, 3:6], self.unit_cube.size(0))
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
        loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
        loss = (loss1 + loss2) / 2
        return loss

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    def apply_box_diff(self, box, box_diff):
        center = box[0, :3] + box_diff[0, :3]
        shape = box[0, 3:6] + box_diff[0, 3:6]
        q = qmul(box_diff[0, 6:], box[0, 6:])
        return torch.cat([center, shape, q]).view(1, -1)

    # decode a part node
    def decode_node(self, node_latent, max_depth, full_label, is_leaf=False):
        if node_latent.shape[0] != 1:
            raise ValueError('Node decoding does not support batch_size > 1.')

        is_leaf_logit = self.leaf_classifier(node_latent)
        node_is_leaf = is_leaf_logit.item() > 0

        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True

        # decode the current part box
        box = self.box_decoder(node_latent)

        if node_is_leaf or is_leaf:
            ret = Tree.Node(is_leaf=True, \
                    full_label=full_label, label=full_label.split('/')[-1])
            ret.set_from_box_quat(box.view(-1))
            return ret
        else:
            child_feats, child_sem_logits, child_exists_logit = \
                    self.child_decoder(node_latent)

            child_sem_logits = child_sem_logits.cpu().numpy().squeeze()

            # children
            child_nodes = []
            for ci in range(child_feats.shape[1]):
                if child_exists_logit[:, ci, :].item() > 0:
                    if full_label not in Tree.part_non_leaf_sem_names:
                        print('WARNING: predicting nonzero children for a node with leaf semantics, ignoring the children')
                        continue
                    idx = np.argmax(child_sem_logits[ci, Tree.part_name2cids[full_label]])
                    idx = Tree.part_name2cids[full_label][idx]
                    child_full_label = Tree.part_id2name[idx]
                    child_nodes.append(self.decode_node(\
                            child_feats[:, ci, :], max_depth-1, child_full_label, \
                            is_leaf=(child_full_label not in Tree.part_non_leaf_sem_names)))

            ret = Tree.Node(is_leaf=len(child_nodes) == 0, children=child_nodes, \
                    full_label=full_label, label=full_label.split('/')[-1])
            ret.set_from_box_quat(box.view(-1))
            return ret

    def decode_tree_diff(self, z, obj):
        feat = self.sample_decoder(diff_feat=z, input_feat=obj.root.feature)
        ret = self.decode_node_diff(z=feat, z_skip=z, obj_node=obj.root)
        ret.box_diff = self.box_diff_decoder(feat)
        return ret

    def decode_node_diff(self, z, z_skip, obj_node):
        # DEL/SAME/LEAF
        ret = Tree.DiffNode('SAME')
        if len(obj_node.children) > 0:
            type_valid_ids = [0, 1, 3]
            for i, cnode in enumerate(obj_node.children):
                feat = torch.cat(
                    [z, z_skip, cnode.box_feature, cnode.feature, cnode.get_semantic_one_hot()], dim=1)
                feat = self.node_diff_feature_extractor(feat)
                pred_type = self.node_diff_classifier(feat)
                pred_type = type_valid_ids[pred_type[0, type_valid_ids].argmax().item()]
                if pred_type == 0:  # SAME
                    cdiff = self.decode_node_diff(z=feat, z_skip=z_skip, obj_node=cnode)
                    cdiff.box_diff = self.box_diff_decoder(feat)
                    ret.children.append(cdiff)
                elif pred_type == 1:    # DEL
                    cdiff = Tree.DiffNode('DEL')
                    ret.children.append(cdiff)
                else:   # LEAF
                    cdiff = Tree.DiffNode('LEAF')
                    cdiff.box_diff = self.box_diff_decoder(feat)
                    ret.children.append(cdiff)

        # ADD
        add_child_feats, add_child_sem_logits, add_child_exists_logits = self.add_child_decoder(z)

        feature_size = add_child_feats.size(2)
        num_part = add_child_feats.size(1)
        add_child_boxes = self.box_decoder(add_child_feats.view(-1, feature_size))
        add_child_sem_logits = add_child_sem_logits.cpu().numpy().squeeze()

        for i in range(num_part):
            if add_child_exists_logits[0, i].item() > 0:
                if obj_node.full_label not in Tree.part_non_leaf_sem_names:
                    print('WARNING: predicting nonzero children for a node with leaf semantics, ignoring the children')
                    continue
                cdiff = Tree.DiffNode('ADD')
                idx = np.argmax(add_child_sem_logits[i, Tree.part_name2cids[obj_node.full_label]])
                idx = Tree.part_name2cids[obj_node.full_label][idx]
                child_full_label = Tree.part_id2name[idx]
                cdiff.subnode = self.decode_node(add_child_feats[:, i], self.conf.max_tree_depth, \
                        child_full_label, is_leaf=(child_full_label not in Tree.part_non_leaf_sem_names))
                ret.children.append(cdiff)

        return ret

    def tree_diff_recon_loss(self, z, obj, gt_diff):
        feat = self.sample_decoder(diff_feat=z, input_feat=obj.root.feature)
        losses = self.node_diff_recon_loss(z=feat, z_skip=z, obj_node=obj.root, gt_diff=gt_diff)
        return losses

    def node_diff_recon_loss(self, z, z_skip, obj_node, gt_diff):
        # initialize all losses to zeros
        box_loss = torch.zeros(1, device=z.device)
        is_leaf_loss = torch.zeros(1, device=z.device)
        child_exists_loss = torch.zeros(1, device=z.device)
        semantic_loss = torch.zeros(1, device=z.device)
        diffnode_type_loss = torch.zeros(1, device=z.device)
        diffnode_box_loss = torch.zeros(1, device=z.device)

        # DEL/SAME/LEAF
        node_diff_types = []; gt_node_diff_types = [];
        for i, cnode in enumerate(obj_node.children):
            cdiff = gt_diff.children[i]
            node_diff_feat = self.node_diff_feature_extractor(torch.cat(
                [z, z_skip, cnode.box_feature, cnode.feature, cnode.get_semantic_one_hot()], dim=1))
            node_diff_types.append(self.node_diff_classifier(node_diff_feat))
            gt_node_diff_types.append(cdiff.get_node_type_id())
            if cdiff.node_type == 'SAME':
                child_losses = self.node_diff_recon_loss(
                    z=node_diff_feat, z_skip=z_skip, obj_node=cnode, gt_diff=cdiff)
                diffnode_type_loss += child_losses['diffnode_type']
                diffnode_box_loss += child_losses['diffnode_box']
                box_loss += child_losses['box']
                is_leaf_loss += child_losses['leaf']
                child_exists_loss += child_losses['exists']
                semantic_loss += child_losses['semantic']
            if cdiff.node_type == 'LEAF' or cdiff.node_type == 'SAME':
                box2 = self.apply_box_diff(cnode.get_box_quat(), self.box_diff_decoder(node_diff_feat))
                gt_box2 = self.apply_box_diff(cnode.get_box_quat(), cdiff.box_diff)
                diffnode_box_loss += self.boxLossEstimator(box2, gt_box2).sum()

        # diff node loss
        if len(obj_node.children) > 0:
            node_diff_types = torch.cat(node_diff_types, dim=0)
            gt_node_diff_types = torch.tensor(gt_node_diff_types, dtype=torch.int64, device=node_diff_types.device)
            diffnode_type_loss += self.ceLoss(node_diff_types, gt_node_diff_types).sum()

        # ADD
        add_child_feats, add_child_sem_logits, add_child_exists_logits = self.add_child_decoder(z)

        add_child_boxes = self.box_decoder(add_child_feats.view(-1, self.conf.feature_size))
        num_pred = add_child_boxes.size(0)

        child_exists_gt = torch.zeros_like(add_child_exists_logits)

        with torch.no_grad():
            add_child_gt_boxes = []
            for i in range(len(obj_node.children), len(gt_diff.children)):
                add_child_gt_boxes.append(gt_diff.children[i].subnode.get_box_quat())

            num_gt = len(add_child_gt_boxes)
            if num_gt > 0:
                add_child_gt_boxes = torch.cat(add_child_gt_boxes, dim=0)
                pred_tiled = add_child_boxes.unsqueeze(0).repeat(num_gt, 1, 1)
                gt_tiled = add_child_gt_boxes.unsqueeze(1).repeat(1, num_pred, 1)
                dmat = self.boxLossEstimator(gt_tiled.view(-1, 10), pred_tiled.view(-1, 10)).view(-1, num_gt, num_pred)
                _, matched_gt_idx, matched_pred_idx = linear_assignment(dmat)

        if num_gt > 0:
            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_box_gt = []
            child_box_pred = []
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_diff.children[matched_gt_idx[i]+len(obj_node.children)].subnode.get_semantic_id())
                child_sem_pred_logits.append(add_child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_box_gt.append(gt_diff.children[matched_gt_idx[i]+len(obj_node.children)].subnode.get_box_quat())
                child_box_pred.append(add_child_boxes[matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

                # train add node subtree
                child_losses = self.node_recon_loss(add_child_feats[:, matched_pred_idx[i], :], \
                        gt_diff.children[matched_gt_idx[i]+len(obj_node.children)].subnode)
                box_loss += child_losses['box']
                is_leaf_loss += child_losses['leaf']
                child_exists_loss += child_losses['exists']
                semantic_loss += child_losses['semantic']

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, device=add_child_sem_logits.device)
            semantic_loss += self.ceLoss(child_sem_pred_logits, child_sem_gt_labels).sum()

        # train unused boxes to zeros
        unmatched_boxes = []
        for i in range(num_pred):
            if num_gt == 0 or i not in matched_pred_idx:
                unmatched_boxes.append(add_child_boxes[i, 3:6].view(1, -1))
        if len(unmatched_boxes) > 0:
            unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
            box_loss += unmatched_boxes.pow(2).sum() * 0.01

        # train exist scores
        child_exists_loss += F.binary_cross_entropy_with_logits(input=add_child_exists_logits, target=child_exists_gt, reduction='none').sum()

        return {'box': box_loss, 'leaf': is_leaf_loss, 'exists': child_exists_loss, 'semantic': semantic_loss, \
                'diffnode_type': diffnode_type_loss, 'diffnode_box': diffnode_box_loss}

    # use gt structure, compute the reconstruction losses
    def node_recon_loss(self, node_latent, gt_node):
        if gt_node.is_leaf:
            box = self.box_decoder(node_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))
            return {'box': box_loss, 'leaf': is_leaf_loss, 'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss)}
        else:
            child_feats, child_sem_logits, child_exists_logits = \
                    self.child_decoder(node_latent)

            # generate box prediction for each child
            feature_len = node_latent.size(1)
            child_pred_boxes = self.box_decoder(child_feats.view(-1, feature_len))
            num_child_parts = child_pred_boxes.size(0)

            # perform hungarian matching between pred boxes and gt boxes
            with torch.no_grad():
                child_gt_boxes = torch.cat([child_node.get_box_quat() for child_node in gt_node.children], dim=0)
                num_gt = child_gt_boxes.size(0)

                child_pred_boxes_tiled = child_pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

                dist_mat = self.boxLossEstimator(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

            # train the current node to be non-leaf
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # train the current node box to gt
            box = self.box_decoder(node_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_box_gt = []
            child_box_pred = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_box_gt.append(gt_node.children[matched_gt_idx[i]].get_box_quat())
                child_box_pred.append(child_pred_boxes[matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, \
                    device=child_sem_pred_logits.device)
            semantic_loss = self.ceLoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

            # train unused boxes to zeros
            unmatched_boxes = []
            for i in range(num_child_parts):
                if i not in matched_pred_idx:
                    unmatched_boxes.append(child_pred_boxes[i, 3:6].view(1, -1))
            if len(unmatched_boxes) > 0:
                unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
                unused_box_loss = unmatched_boxes.pow(2).sum() * 0.01
            else:
                unused_box_loss = 0.0

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # calculate children + aggregate losses
            for i in range(len(matched_gt_idx)):
                child_losses = self.node_recon_loss(\
                        child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]])
                box_loss = box_loss + child_losses['box']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']

        return {'box': box_loss + unused_box_loss, 'leaf': is_leaf_loss, 'exists': child_exists_loss, 'semantic': semantic_loss}

