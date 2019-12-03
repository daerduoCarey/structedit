"""
    From StructureNet: This file defines the Hierarchy of Graph Tree class and PartNet data loader.
    For StructEdit: We provide new classes for computing/loading shape differences.
                    We don't use the edge information or point-cloud shapes in the StructEdit project.
"""

import sys
import os
import json
import torch
import numpy as np
from torch.utils import data
from pyquaternion import Quaternion
from collections import namedtuple
from utils import one_hot, compute_box_diff, boxLoss, linear_assignment, apply_box_diff
import trimesh
from progressbar import ProgressBar
from copy import deepcopy

# store a part hierarchy of graphs for a shape
class Tree(object):

    # global object category information
    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = None
    root_sem = None

    @ staticmethod
    def load_category_info(cat):
        with open(os.path.join('../stats/part_semantics/', cat+'.txt'), 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if '/' in y:
                    Tree.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        Tree.num_sem = len(Tree.part_name2id) + 1
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)
        Tree.root_sem = Tree.part_id2name[1]


    # store a part node in the tree
    class Node(object):

        def __init__(self, part_id=0, is_leaf=False, box=None, label=None, children=None, edges=None, full_label=None, geo=None, geo_feat=None):
            self.is_leaf = is_leaf          # store True if the part is a leaf node
            self.part_id = part_id          # part_id in result_after_merging.json of PartNet
            self.box = box                  # box parameter for all nodes
            self.geo = geo                  # 1 x 1000 x 3 point cloud
            self.geo_feat = geo_feat        # 1 x 100 geometry feature
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            self.children = [] if children is None else children
                                            # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges
                                            # all of its children relationships;
                                            # each entry is a tuple <part_a, part_b, type, params, dist>
            """
                Here defines the edges format:
                    part_a, part_b:
                        Values are the order in self.children (e.g. 0, 1, 2, 3, ...).
                        This is an directional edge for A->B.
                        If an edge is commutative, you may need to manually specify a B->A edge.
                        For example, an ADJ edge is only shown A->B,
                        there is no edge B->A in the json file.
                    type:
                        Four types considered in StructureNet: ADJ, ROT_SYM, TRANS_SYM, REF_SYM.
                    params:
                        There is no params field for ADJ edge;
                        For ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle;
                        For TRANS_SYM edge, 0-2 translation vector;
                        For REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers,
                            3-5 unit normal direction of the reflection plane.
                    dist:
                        For ADJ edge, it's the closest distance between two parts;
                        For SYM edge, it's the chamfer distance after matching part B to part A.
            """

        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]

        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            out[0, Tree.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)

        def get_box_quat(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir /= np.linalg.norm(xdir)
            ydir = box[9:]
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
            rotmat = np.vstack([xdir, ydir, zdir]).T
            q = Quaternion(matrix=rotmat)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            box_quat = np.hstack([center, size, quat]).astype(np.float32)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)

        def set_from_box_quat(self, box_quat):
            device = box_quat.device
            box_quat = box_quat.cpu().detach().numpy().squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
            rotmat = q.rotation_matrix
            box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1).to(device=device)

        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)

            for child_node in self.children:
                child_node.to(device)

            return self

        def _to_str(self, level, pid, detailed=False):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}'
            if detailed and self.box is not None:
                out_str += ' Box('+';'.join([str(item) for item in self.box.squeeze().cpu().numpy()])+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx, detailed)

            #if detailed and len(self.edges) > 0:
            #    for edge in self.edges:
            #        if 'params' in edge:
            #            edge = edge.copy() # so the original parameters don't get changed
            #            edge['params'] = edge['params'].cpu().numpy()
            #        out_str += '  |'*(level) + '  ├' + 'Edge(' + str(edge) + ')\n'

            return out_str

        def __str__(self, detailed=False):
            return self._to_str(0, 0, detailed)

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        # max. child count of any node in the subtree
        def max_child_count(self):
            node_child_count = []
            for node in self.depth_first_traversal():
                if node.children is None:
                    node_child_count.append(0)
                else:
                    node_child_count.append(len(node.children))

            return max(node_child_count)

        def child_adjacency(self, typed=False, max_children=None):
            if max_children is None:
                adj = torch.zeros(len(self.children), len(self.children))
            else:
                adj = torch.zeros(max_children, max_children)

            if typed:
                edge_types = ['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM']

            for edge in self.edges:
                if typed:
                    edge_type_index = edge_types.index(edge['type'])
                    adj[edge['part_a'], edge['part_b']] = edge_type_index
                    adj[edge['part_b'], edge['part_a']] = edge_type_index
                else:
                    adj[edge['part_a'], edge['part_b']] = 1
                    adj[edge['part_b'], edge['part_a']] = 1

            return adj

        def geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_geos = []; out_nodes = [];
            for node in nodes:
                if not leafs_only or node.is_leaf:
                    out_geos.append(node.geo)
                    out_nodes.append(node)
            return out_geos, out_nodes

        def boxes(self, per_node=False, leafs_only=False):
            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.box)

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes

        def graph(self, leafs_only=False):
            part_boxes = []
            part_geos = []
            edges = []
            part_ids = []
            part_sems = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue

                    part_boxes.append(child.box)
                    part_geos.append(child.geo)
                    part_ids.append(child.part_id)
                    part_sems.append(child.full_label)

                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                for edge in node.edges:
                    if leafs_only and not (
                            node.children[edge['part_a']].is_leaf and
                            node.children[edge['part_b']].is_leaf):
                        continue
                    edges.append(edge.copy())
                    edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]

                box_index_offset += child_count

            return part_boxes, part_geos, edges, part_ids, part_sems

        def edge_tensors(self, edge_types, device, type_onehot=True):
            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor(
                [[e['part_a'], e['part_b']] for e in self.edges] + [[e['part_b'], e['part_a']] for e in self.edges],
                device=device, dtype=torch.long).view(1, num_edges*2, 2)

            # get edge type as tensor
            edge_type = torch.tensor([edge_types.index(edge['type']) for edge in self.edges], device=device, dtype=torch.long)
            if type_onehot:
                edge_type = one_hot(inp=edge_type, label_count=len(edge_types)).transpose(0, 1).view(1, num_edges, len(edge_types)).to(dtype=torch.float32)
            else:
                edge_type = edge_type.view(1, num_edges)
            edge_type = torch.cat([edge_type, edge_type], dim=1) # add edges in other direction (symmetric adjacency)

            return edge_type, edge_indices

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt

    # shape-diff
    class DiffNode(object):

        def __init__(self, node_type):
            self.node_type = node_type      # 'DEL', 'SAME', 'ADD'
            self.box_diff = None            # only exists for leaf DiffNode
            self.subnode = None             # for 'ADD' diffnode only (refer to a Node)
            self.children = []              # for 'SAME' diffnode only ('DEL' diffnode has no children)
            self.device = 'cpu'

        def get_node_type_id(self):
            if self.node_type == 'SAME':
                return 0
            elif self.node_type == 'DEL':
                return 1
            elif self.node_type == 'ADD':
                return 2
            elif self.node_type == 'LEAF':
                return 3
            else:
                raise ValueError('Unknown DiffNode type %s!' % self.node_type)

        def get_node_type_one_hot(self):
            if self.node_type == 'SAME':
                return torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
            elif self.node_type == 'DEL':
                return torch.tensor([[0, 1, 0, 0]], dtype=torch.float32, device=self.device)
            elif self.node_type == 'ADD':
                return torch.tensor([[0, 0, 1, 0]], dtype=torch.float32, device=self.device)
            elif self.node_type == 'LEAF':
                return torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            else:
                raise ValueError('Unknown DiffNode type %s!' % self.node_type)

        def _to_str(self, level, pid, detailed):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' [DiffNode: %s] ' % self.node_type
            if self.box_diff is None or not detailed:
                out_str += '\n'
            else:
                out_str += ' BoxDiff(%s) \n' % str(self.box_diff.squeeze().cpu().numpy().tolist())

            if self.node_type == 'SAME' and len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx, detailed)

            if self.node_type == 'ADD':
                out_str += self.subnode._to_str(level+1, 0, detailed)

            return out_str

        def __str__(self, detailed=False):
            return self._to_str(0, 0, detailed)

        def to(self, device):
            self.device = device
            if self.box_diff is not None:
                self.box_diff = self.box_diff.to(device)
            if len(self.children) > 0:
                for cnode in self.children:
                    cnode.to(device)
            if self.subnode is not None:
                self.subnode.to(device)

    # functions for class Tree
    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def depth_first_traversal(self):
        return self.root.depth_first_traversal()

    def max_child_count(self):
        return self.root.max_child_count()

    def boxes(self, per_node=False, leafs_only=False):
        return self.root.boxes(per_node=per_node, leafs_only=leafs_only)

    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    def free(self):
        for node in self.depth_first_traversal():
            del node.geo
            del node.geo_feat
            del node.box
            del node


    # calculate shape-diff (target - source: change source to target)
    @staticmethod
    def compute_shape_diff(source, target, device=None):
        dnode = Tree.DiffNode('SAME')

        # get box_diff for every node
        source_box_quat = source.get_box_quat().squeeze().cpu().numpy()
        target_box_quat = target.get_box_quat().squeeze().cpu().numpy()
        dnode.box_diff = torch.tensor(compute_box_diff(source_box_quat, target_box_quat), dtype=torch.float32, device=source.box.device).view(1, -1)

        if source.is_leaf:
            if target.is_leaf:
                dnode.node_type = 'LEAF'
            else:
                for cnode in target.children:
                    cdiffnode = Tree.DiffNode('ADD')
                    cdiffnode.subnode = cnode
                    dnode.children.append(cdiffnode)
        else:
            if target.is_leaf:
                for cnode in source.children:
                    dnode.children.append(Tree.DiffNode('DEL'))
            else:
                source_sem = set([node.label for node in source.children])
                target_sem = set([node.label for node in target.children])
                intersect_sem = set.intersection(source_sem, target_sem)

                source_cnodes_per_sem = dict()
                for node_id, cnode in enumerate(source.children):
                    if cnode.label in intersect_sem:
                        if cnode.label not in source_cnodes_per_sem:
                            source_cnodes_per_sem[cnode.label] = []
                        source_cnodes_per_sem[cnode.label].append(node_id)

                target_cnodes_per_sem = dict()
                for node_id, cnode in enumerate(target.children):
                    if cnode.label in intersect_sem:
                        if cnode.label not in target_cnodes_per_sem:
                            target_cnodes_per_sem[cnode.label] = []
                        target_cnodes_per_sem[cnode.label].append(node_id)

                matched_source_ids = []; matched_target_ids = []; matched_source2target = dict();
                unmatched_source_ids = set(range(len(source.children)))
                unmatched_target_ids = set(range(len(target.children)))
                for sem in intersect_sem:
                    source_boxes = torch.cat([source.children[cid].get_box_quat() for cid in source_cnodes_per_sem[sem]], dim=0)
                    target_boxes = torch.cat([target.children[cid].get_box_quat() for cid in target_cnodes_per_sem[sem]], dim=0)

                    num_source = source_boxes.size(0)
                    num_target = target_boxes.size(0)

                    source_boxes_tiled = source_boxes.unsqueeze(dim=1).repeat(1, num_target, 1)
                    target_boxes_tiled = target_boxes.unsqueeze(dim=0).repeat(num_source, 1, 1)

                    dmat = boxLoss(source_boxes_tiled.view(-1, 10), target_boxes_tiled.view(-1, 10)).view(-1, num_source, num_target).cpu()
                    _, cur_matched_source_ids, cur_matched_target_ids = linear_assignment(dmat)

                    for i in range(len(cur_matched_source_ids)):
                        source_node_id = source_cnodes_per_sem[sem][cur_matched_source_ids[i]]
                        matched_source_ids.append(source_node_id)
                        unmatched_source_ids.remove(source_node_id)
                        target_node_id = target_cnodes_per_sem[sem][cur_matched_target_ids[i]]
                        matched_target_ids.append(target_node_id)
                        unmatched_target_ids.remove(target_node_id)
                        matched_source2target[source_node_id] = target_node_id

                for node_id, cnode in enumerate(source.children):
                    if node_id in unmatched_source_ids:
                        dnode.children.append(Tree.DiffNode('DEL'))
                    else:
                        dnode.children.append(Tree.compute_shape_diff(source.children[node_id], target.children[matched_source2target[node_id]]))

                for i in unmatched_target_ids:
                    cdiffnode = Tree.DiffNode('ADD')
                    cdiffnode.subnode = target.children[i]
                    dnode.children.append(cdiffnode)

        return dnode

    # apply shape-diff (obj + diff)
    @staticmethod
    def apply_shape_diff(obj, diff):
        ret = Tree.Node()
        ret.part_id = obj.part_id
        ret.label = obj.label
        ret.full_label = obj.full_label

        box1 = obj.get_box_quat().squeeze().cpu().numpy()
        box_diff = diff.box_diff.squeeze().cpu().numpy()
        box2 = torch.tensor(apply_box_diff(box1, box_diff), dtype=torch.float32, device=obj.box.device).view(1, -1)
        ret.set_from_box_quat(box2)

        # for node_id, cnode in enumerate(obj.children):
        for i in range(min(len(obj.children), len(diff.children))):
            if diff.children[i].node_type == 'SAME' or diff.children[i].node_type == 'LEAF':
                ret.children.append(Tree.apply_shape_diff(obj.children[i], diff.children[i]))
            elif diff.children[i].node_type == 'DEL':
                pass
            else:
                raise ValueError('Invalid diffnode type %s within the editing range [can only be SAME or DEL]' % diff.children[node_id].node_type)

        for i in range(len(obj.children), len(diff.children)):
            if diff.children[i].node_type == 'ADD':
                assert diff.children[i].subnode is not None, 'ADD argument has no subnode!'
                ret.children.append(diff.children[i].subnode)
            else:
                raise ValueError('Invalid diffnode type %s out of the editing range [can only be ADD]' % diff.children[i].node_type)

        ret.is_leaf = len(ret.children) == 0
        return ret


# extend torch.data.Dataset class for PartNet
class PartNetDataset(data.Dataset):

    def __init__(self, root, object_list, data_features, load_geo=False):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

    def __getitem__(self, index):
        if 'object' in self.data_features:
            obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'), \
                    load_geo=self.load_geo)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.object_names)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, load_geo=False):
        if load_geo:
            geo_fn = fn.replace('_hier', '_geo').replace('json', 'npz')
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)

            if load_geo:
                node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.geo is not None:
                node_json['geo'] = node.geo.cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(deepcopy(edge))
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        with open(fn, 'w') as f:
            json.dump(obj_json, f)


# extend torch.data.Dataset class for PartNet ShapeDiff
class PartNetShapeDiffDataset(data.Dataset):

    def __init__(self, data_dir, object_list, data_features, topk, metric, self_is_neighbor):
        print('PartNetShapeDiffDataset: ', topk, metric)
        self.data_dir = data_dir
        self.data_features = data_features
        self.topk = topk
        self.metric = metric

        if isinstance(object_list, str):
            with open(os.path.join(self.data_dir, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.neighbors = []
        pbar = ProgressBar()
        for name in pbar(self.object_names):
            fn = os.path.join(self.data_dir, 'neighbors_'+metric, os.path.splitext(os.path.basename(object_list))[0], name+'.npy')
            self.neighbors.append(self.load_neighbors(fn=fn, topk=self.topk, name=name, self_is_neighbor=self_is_neighbor))

    def __getitem__(self, index):
        if 'object' in self.data_features or 'diff' in self.data_features:
            obj = PartNetDataset.load_object(os.path.join(self.data_dir, self.object_names[index]+'.json'))

        if 'diff' in self.data_features or 'name2' in self.data_features or 'object2' in self.data_features:
            name2 = self.neighbors[index][np.random.randint(min(self.topk, len(self.neighbors[index])))]
            obj2 = PartNetDataset.load_object(os.path.join(self.data_dir, name2+'.json'))
            diff = Tree.compute_shape_diff(obj.root, obj2.root)

        if 'neighbor_diffs' in self.data_features or 'neighbor_objs' in self.data_features:
            neighbor_diffs, neighbor_objs, neighbor_names = [], [], []
            for neighbor_name in self.neighbors[index]:
                neighbor_obj = PartNetDataset.load_object(os.path.join(self.data_dir, neighbor_name+'.json'))
                neighbor_objs.append(neighbor_obj)
                neighbor_diffs.append(Tree.compute_shape_diff(obj.root, neighbor_obj.root))
                neighbor_names.append(neighbor_name)

        if 'neighbor_names_only' in self.data_features:
            neighbor_names = self.neighbors[index]

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'object2':
                data_feats = data_feats + (obj2,)
            elif feat == 'diff':
                data_feats = data_feats + (diff,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            elif feat == 'name2':
                data_feats = data_feats + (name2,)
            elif feat == 'neighbor_objs':
                data_feats = data_feats + (neighbor_objs,)
            elif feat == 'neighbor_diffs':
                data_feats = data_feats + (neighbor_diffs,)
            elif feat == 'neighbor_names' or feat == 'neighbor_names_only':
                data_feats = data_feats + (neighbor_names,)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.object_names)

    @staticmethod
    def load_neighbors(fn, topk, name, self_is_neighbor):
        neighbor_info = np.load(fn, allow_pickle=True).item()
        idx = np.argsort(neighbor_info['dists'])[:topk+1]

        if not self_is_neighbor:
            idx = [neighbor_info['names'][i] for i in idx if neighbor_info['names'][i] != name]
        else:
            idx = [neighbor_info['names'][i] for i in idx]

        return idx[:topk]


# extend torch.data.Dataset class for SynChair ShapeDiff
class SynChairShapeDiffDataset(data.Dataset):

    def __init__(self, data_dir, object_list, data_features, topk):
        print('SynChairShapeDiffDataset: ', topk, data_dir)
        self.data_dir = data_dir
        self.data_features = data_features
        self.topk = topk

        if isinstance(object_list, str):
            with open(os.path.join(self.data_dir, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.neighbors = []
        bar = ProgressBar()
        for name in bar(self.object_names):
            self.neighbors.append(self.load_neighbors(name))

    def __getitem__(self, index):
        if 'object' in self.data_features or 'diff' in self.data_features:
            obj = PartNetDataset.load_object(os.path.join(self.data_dir, self.object_names[index]+'.json'))

        if 'diff' in self.data_features or 'name2' in self.data_features or 'object2' in self.data_features:
            name2 = self.neighbors[index][np.random.randint(self.topk)]
            obj2 = PartNetDataset.load_object(os.path.join(self.data_dir, name2+'.json'))
            diff = Tree.compute_shape_diff(obj.root, obj2.root)

        if 'neighbor_diffs' in self.data_features or 'neighbor_objs' in self.data_features:
            neighbor_diffs = []; neighbor_objs = []; neighbor_names = [];
            for neighbor_name in self.neighbors[index]:
                neighbor_obj = PartNetDataset.load_object(os.path.join(self.data_dir, neighbor_name+'.json'))
                neighbor_objs.append(neighbor_obj)
                neighbor_diffs.append(Tree.compute_shape_diff(obj.root, neighbor_obj.root))
                neighbor_names.append(neighbor_name)

        if 'neighbor_names_only' in self.data_features:
            neighbor_names = self.neighbors[index];

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'object2':
                data_feats = data_feats + (obj2,)
            elif feat == 'diff':
                data_feats = data_feats + (diff,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            elif feat == 'name2':
                data_feats = data_feats + (name2,)
            elif feat == 'neighbor_objs':
                data_feats = data_feats + (neighbor_objs,)
            elif feat == 'neighbor_diffs':
                data_feats = data_feats + (neighbor_diffs,)
            elif feat == 'neighbor_names' or feat == 'neighbor_names_only':
                data_feats = data_feats + (neighbor_names,)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.object_names)

    @staticmethod
    def load_neighbors(name):
        idx = int(name) // 96
        neighbors = ['%05d'%i for i in range(idx*96, (idx+1)*96)]
        return neighbors


# extend torch.data.Dataset class for SynShapes ShapeDiff
class SynShapesDiffDataset(data.Dataset):

    def __init__(self, object_list, data_features, topk):
        print('SynShapesDiffDataset: ', topk)
        self.data_dirs = ['../../data/syn_chair_structedit',
                '../../data/syn_sofa_structedit', 
                '../../data/syn_stool_structedit']
        print(self.data_dirs)
        self.data_features = data_features
        self.topk = topk

        if isinstance(object_list, str):
            with open(os.path.join(self.data_dirs[0], object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.neighbors = []
        bar = ProgressBar()
        for name in bar(self.object_names):
            self.neighbors.append(self.load_neighbors(name))

    def __getitem__(self, index):
        cur_data_dir = self.data_dirs[np.random.randint(len(self.data_dirs))]

        if 'object' in self.data_features or 'diff' in self.data_features:
            obj = PartNetDataset.load_object(os.path.join(cur_data_dir, self.object_names[index]+'.json'))

        if 'diff' in self.data_features or 'name2' in self.data_features or 'object2' in self.data_features:
            name2 = self.neighbors[index][np.random.randint(self.topk)]
            obj2 = PartNetDataset.load_object(os.path.join(cur_data_dir, name2+'.json'))
            diff = Tree.compute_shape_diff(obj.root, obj2.root)

        if 'neighbor_diffs' in self.data_features or 'neighbor_objs' in self.data_features:
            neighbor_diffs = []; neighbor_objs = []; neighbor_names = [];
            for neighbor_name in self.neighbors[index]:
                neighbor_obj = PartNetDataset.load_object(os.path.join(cur_data_dir, neighbor_name+'.json'))
                neighbor_objs.append(neighbor_obj)
                neighbor_diffs.append(Tree.compute_shape_diff(obj.root, neighbor_obj.root))
                neighbor_names.append(neighbor_name)

        if 'neighbor_names_only' in self.data_features:
            neighbor_names = self.neighbors[index];

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'object2':
                data_feats = data_feats + (obj2,)
            elif feat == 'diff':
                data_feats = data_feats + (diff,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            elif feat == 'name2':
                data_feats = data_feats + (name2,)
            elif feat == 'neighbor_objs':
                data_feats = data_feats + (neighbor_objs,)
            elif feat == 'neighbor_diffs':
                data_feats = data_feats + (neighbor_diffs,)
            elif feat == 'neighbor_names' or feat == 'neighbor_names_only':
                data_feats = data_feats + (neighbor_names,)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.object_names)

    @staticmethod
    def load_neighbors(name):
        idx = int(name) // 96
        neighbors = ['%05d'%i for i in range(idx*96, (idx+1)*96)]
        return neighbors

