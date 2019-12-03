"""
    This is the main tester script for box-shape reconstruction evaluation.
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from data import PartNetDataset, Tree
from chamfer_distance import ChamferDistance

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.ckpt_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path

# load object category information
Tree.load_category_info(conf.category)

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)
print(conf.data_path, conf.model_version, conf.model_epoch, conf.category)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
result_dir = os.path.join(conf.result_path, conf.exp_name + '_recon')
if os.path.exists(result_dir):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % result_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(result_dir)

# create a new directory to store eval results
os.makedirs(result_dir)

# create models
encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = models.RecursiveDecoder(conf)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.ckpt_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# create dataset and data loader
data_features = ['object', 'name']
dataset = PartNetDataset(conf.data_path, conf.test_dataset, data_features, load_geo=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_feats)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# load unit cube pc
unit_cube = torch.from_numpy(utils.load_pts('cube.pts')).to(device)

def boxLoss(box_feature, gt_box_feature):
    pred_box_pc = utils.transform_pc_batch(unit_cube, box_feature)
    pred_reweight = utils.get_surface_reweighting_batch(box_feature[:, 3:6], unit_cube.size(0))
    gt_box_pc = utils.transform_pc_batch(unit_cube, gt_box_feature)
    gt_reweight = utils.get_surface_reweighting_batch(gt_box_feature[:, 3:6], unit_cube.size(0))
    dist1, dist2 = chamferLoss(gt_box_pc, pred_box_pc)
    loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
    loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
    loss = (loss1 + loss2) / 2
    return loss

def compute_struct_diff(gt_node, pred_node):
    if gt_node.is_leaf:
        if pred_node.is_leaf:
            return 0
        else:
            return len(pred_node.boxes())-1
    else:
        if pred_node.is_leaf:
            return len(gt_node.boxes())-1
        else:
            gt_sem = set([node.label for node in gt_node.children])
            pred_sem = set([node.label for node in pred_node.children])
            intersect_sem = set.intersection(gt_sem, pred_sem)

            gt_cnodes_per_sem = dict()
            for node_id, gt_cnode in enumerate(gt_node.children):
                if gt_cnode.label in intersect_sem:
                    if gt_cnode.label not in gt_cnodes_per_sem:
                        gt_cnodes_per_sem[gt_cnode.label] = []
                    gt_cnodes_per_sem[gt_cnode.label].append(node_id)

            pred_cnodes_per_sem = dict()
            for node_id, pred_cnode in enumerate(pred_node.children):
                if pred_cnode.label in intersect_sem:
                    if pred_cnode.label not in pred_cnodes_per_sem:
                        pred_cnodes_per_sem[pred_cnode.label] = []
                    pred_cnodes_per_sem[pred_cnode.label].append(node_id)

            matched_gt_idx = []; matched_pred_idx = []; matched_gt2pred = np.zeros((conf.max_child_num), dtype=np.int32)
            for sem in intersect_sem:
                gt_boxes = torch.cat([gt_node.children[cid].get_box_quat() for cid in gt_cnodes_per_sem[sem]], dim=0).to(device)
                pred_boxes = torch.cat([pred_node.children[cid].get_box_quat() for cid in pred_cnodes_per_sem[sem]], dim=0).to(device)

                num_gt = gt_boxes.size(0)
                num_pred = pred_boxes.size(0)

                if num_gt == 1 and num_pred == 1:
                    cur_matched_gt_idx = [0]
                    cur_matched_pred_idx = [0]
                else:
                    gt_boxes_tiled = gt_boxes.unsqueeze(dim=1).repeat(1, num_pred, 1)
                    pred_boxes_tiled = pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                    dmat = boxLoss(gt_boxes_tiled.view(-1, 10), pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_pred).cpu()
                    _, cur_matched_gt_idx, cur_matched_pred_idx = utils.linear_assignment(dmat)

                for i in range(len(cur_matched_gt_idx)):
                    matched_gt_idx.append(gt_cnodes_per_sem[sem][cur_matched_gt_idx[i]])
                    matched_pred_idx.append(pred_cnodes_per_sem[sem][cur_matched_pred_idx[i]])
                    matched_gt2pred[gt_cnodes_per_sem[sem][cur_matched_gt_idx[i]]] = pred_cnodes_per_sem[sem][cur_matched_pred_idx[i]]

            struct_diff = 0.0;
            for i in range(len(gt_node.children)):
                if i not in matched_gt_idx:
                    struct_diff += len(gt_node.children[i].boxes())

            for i in range(len(pred_node.children)):
                if i not in matched_pred_idx:
                    struct_diff += len(pred_node.children[i].boxes())

            for i in range(len(matched_gt_idx)):
                gt_id = matched_gt_idx[i]
                pred_id = matched_pred_idx[i]
                cur_struct_diff = compute_struct_diff(gt_node.children[gt_id], pred_node.children[pred_id])
                struct_diff += cur_struct_diff
                pred_node.children[pred_id].part_id = gt_node.children[gt_id].part_id

            return struct_diff

# test over all test shapes
num_batch = len(dataloader)
chamfer_dists = []
structure_dists = []
with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        obj = batch[data_features.index('object')][0]
        obj.to(device)
        obj_name = batch[data_features.index('name')][0]

        root_code_and_kld = encoder.encode_structure(obj)
        root_code = root_code_and_kld[:, :conf.feature_size]
        recon_obj = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)
        loss = decoder.structure_recon_loss(z=root_code, gt_tree=obj)
        print('[%d/%d] ' % (batch_ind, num_batch), obj_name, loss)

        # structure diff and edge accuracy
        sd = compute_struct_diff(obj.root, recon_obj.root)
        sd = sd / len(obj.root.boxes())
        structure_dists.append(sd)

        # save original and reconstructed object
        os.mkdir(os.path.join(result_dir, obj_name))
        orig_output_filename = os.path.join(result_dir, obj_name, 'orig.json')
        recon_output_filename = os.path.join(result_dir, obj_name, 'recon.json')
        PartNetDataset.save_object(obj=obj, fn=orig_output_filename)
        PartNetDataset.save_object(obj=recon_obj, fn=recon_output_filename)

        # chamfer distance
        ori_obbs_np = torch.cat([item.view(1, -1) for item in obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
        ori_mesh_v, ori_mesh_f = utils.gen_obb_mesh(ori_obbs_np)
        ori_pc_sample = utils.sample_pc(ori_mesh_v, ori_mesh_f)
        print(ori_pc_sample.shape)
        recon_obbs_np = torch.cat([item.view(1, -1) for item in recon_obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
        recon_mesh_v, recon_mesh_f = utils.gen_obb_mesh(recon_obbs_np)
        recon_pc_sample = utils.sample_pc(recon_mesh_v, recon_mesh_f)
        print(recon_pc_sample.shape)
        cd1, cd2 = chamferLoss(torch.tensor(ori_pc_sample, dtype=torch.float32).view(1, -1, 3), 
                torch.tensor(recon_pc_sample, dtype=torch.float32).view(1, -1, 3))
        cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
        chamfer_dists.append(cd)

        stat_filename = os.path.join(result_dir, obj_name, 'stats.txt')
        with open(stat_filename, 'w') as stat_file:
            print(f'box pc chamfer distance: {cd}', file=stat_file)
            print(f'structure distance: {sd}', file=stat_file)

    avg_chamfer_dist = float(sum(chamfer_dists) / float(len(chamfer_dists)))
    avg_structure_dist = float(sum(structure_dists) / float(len(structure_dists)))

    dataset_stat_filename = os.path.join(result_dir, 'stats.txt')
    with open(dataset_stat_filename, 'w') as stat_file:
        print(f'average chamfer distance: {avg_chamfer_dist}', file=stat_file)
        print(f'average structure distance: {avg_structure_dist}', file=stat_file)

