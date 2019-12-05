"""
    This script compute PartNet ground-truth structure-distance neighbors
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from data import PartNetDataset, Tree
from chamfer_distance import ChamferDistance
from progressbar import ProgressBar

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()
data_path = '../data/partnetdata/chair_hier'
dataset_fn = 'train_no_other_less_than_10_parts.txt'
out_dir = os.path.join(data_path, 'neighbors_sd', os.path.splitext(dataset_fn)[0])

if os.path.exists(out_dir):
    response = input('output directory "%s" already exists, overwrite? (y/n) ' % out_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(out_dir)

# create a new directory to store eval results
os.makedirs(out_dir)

# create dataset and data loader
data_features = ['object', 'name']
dataset = PartNetDataset(data_path, dataset_fn, data_features, load_geo=False)

# parameters
device = 'cpu'
max_child_num = 10

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

            matched_gt_idx = []; matched_pred_idx = []; matched_gt2pred = np.zeros((max_child_num), dtype=np.int32)
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

            struct_diff = 0.0
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

# enumerate over all training shapes
num_shape = len(dataset)
objs = []
names = []
bar = ProgressBar()
for i in bar(range(num_shape)):
    obj, name = dataset[i]
    objs.append(obj)
    names.append(name)

bar = ProgressBar()
for i in bar(range(num_shape)):
    obj1 = objs[i]
    dists = np.zeros((num_shape), dtype=np.float32)
    for j in range(num_shape):
        obj2 = objs[j]
        sd = compute_struct_diff(obj1.root, obj2.root)
        dists[j] = sd / len(obj1.root.boxes())

    np.save(
        os.path.join(out_dir, names[i]+'.npy'),
        {'dists': dists, 'names': names})

