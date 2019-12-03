"""
    This script tests the shape difference/deltas/edits reconstruction
"""

import os
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from progressbar import ProgressBar
from chamfer_distance import ChamferDistance
from config import add_eval_args
from data import PartNetDataset, PartNetShapeDiffDataset, Tree
from eval_utils import compute_recon_numbers
import utils
import shutil

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity
chamfer_loss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=-1)
parser.add_argument('--baseline_dir', type=str, help='structurenet baseline result directory')
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.ckpt_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.category = conf.category
eval_conf.data_path = conf.data_path
if hasattr(conf, 'self_is_neighbor'):
    eval_conf.self_is_neighbor = conf.self_is_neighbor

# load object category information
if conf.category is not None:
    Tree.load_category_info(conf.category)

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)
print(conf.data_path, conf.category, conf.model_version, conf.baseline_dir, conf.self_is_neighbor, conf.shapediff_topk, conf.shapediff_metric)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {device}')

# load unit cube pc
unit_cube = torch.from_numpy(utils.load_pts('cube.pts')).to(device)

# check if eval results already exist. If so, delete it.
result_dir = os.path.join(conf.result_path, conf.exp_name + '_recon')
if os.path.exists(result_dir):
    response = input('Eval results directory "%s" already exists, detele it? (y/n) ' % result_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(result_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# create models
encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = models.RecursiveDecoder(conf)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
_ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.ckpt_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True,
    device=device)

# create dataset and data loader
data_features = ['object', 'name', 'neighbor_diffs', 'neighbor_objs', 'neighbor_names']
dataset = PartNetShapeDiffDataset(
    conf.data_path, conf.test_dataset, data_features, conf.shapediff_topk, conf.shapediff_metric, conf.self_is_neighbor)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()


def geometry_dist(obj, recon_obj):
    ori_obbs_np = torch.cat([item.view(1, -1) for item in obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
    ori_mesh_v, ori_mesh_f = utils.gen_obb_mesh(ori_obbs_np)
    ori_pc_sample = utils.sample_pc(ori_mesh_v, ori_mesh_f)
    recon_obbs_np = torch.cat([item.view(1, -1) for item in recon_obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
    recon_mesh_v, recon_mesh_f = utils.gen_obb_mesh(recon_obbs_np)
    recon_pc_sample = utils.sample_pc(recon_mesh_v, recon_mesh_f)
    cd1, cd2 = chamfer_loss(
        torch.tensor(ori_pc_sample, dtype=torch.float32).view(1, -1, 3),
        torch.tensor(recon_pc_sample, dtype=torch.float32).view(1, -1, 3))
    cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
    return cd

def box_dist(box_feature, gt_box_feature):
    pred_box_pc = utils.transform_pc_batch(unit_cube, box_feature)
    pred_reweight = utils.get_surface_reweighting_batch(box_feature[:, 3:6], unit_cube.size(0))
    gt_box_pc = utils.transform_pc_batch(unit_cube, gt_box_feature)
    gt_reweight = utils.get_surface_reweighting_batch(gt_box_feature[:, 3:6], unit_cube.size(0))
    dist1, dist2 = chamfer_loss(gt_box_pc, pred_box_pc)
    loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
    loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
    loss = (loss1 + loss2) / 2
    return loss

def struct_dist(gt_node, pred_node):
    if gt_node.is_leaf:
        if pred_node.is_leaf:
            return 0
        else:
            return len(pred_node.boxes()) - 1
    else:
        if pred_node.is_leaf:
            return len(gt_node.boxes()) - 1
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

            matched_gt_idx = []
            matched_pred_idx = []
            matched_gt2pred = np.zeros((100), dtype=np.int32)
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
                    dmat = box_dist(gt_boxes_tiled.view(-1, 10), pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_pred).cpu()
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
                struct_diff += struct_dist(gt_node.children[gt_id], pred_node.children[pred_id])
                pred_node.children[pred_id].part_id = gt_node.children[gt_id].part_id

            return struct_diff

# test over all shapes
with torch.no_grad():

    # if -1, mean testing until the last
    if conf.end_id < 0:
        conf.end_id = len(dataset)

    print('reconstructing edited shapes ...')
    pbar = ProgressBar()
    for i in pbar(range(conf.start_id, conf.end_id)):
        obj, obj_name, neighbor_diffs, neighbor_objs, neighbor_names = dataset[i]
        # print('[%d/%d] %s' % (i, conf.end_id, obj_name))

        cur_res_dir = os.path.join(result_dir, obj_name)
        if not os.path.exists(cur_res_dir):
            os.makedirs(cur_res_dir)

        obj.to(device)
        encoder.encode_tree(obj)

        for ni in range(len(neighbor_diffs)):
            neighbor_name = neighbor_names[ni]
            neighbor_fn = 'neighbor_%02d_%s' % (ni, neighbor_name)

            neighbor_obj = neighbor_objs[ni]
            neighbor_obj.to(device)

            neighbor_diff = neighbor_diffs[ni]
            neighbor_diff.to(device)
            with open(os.path.join(cur_res_dir, neighbor_fn+'.orig.diff'), 'w') as fout:
                fout.write(str(neighbor_diff))

            root_code = encoder.encode_tree_diff(obj, neighbor_diff)
            recon_diff = decoder.decode_tree_diff(root_code, obj)
            with open(os.path.join(cur_res_dir, neighbor_fn+'.recon.diff'), 'w') as fout:
                fout.write(str(recon_diff))

            recon_neighbor = Tree(Tree.apply_shape_diff(obj.root, recon_diff))
            PartNetDataset.save_object(recon_neighbor, os.path.join(cur_res_dir, neighbor_fn+'.recon.json'))

            cd = geometry_dist(neighbor_obj, recon_neighbor)
            sd = struct_dist(neighbor_obj.root, recon_neighbor.root)
            sd = sd / len(neighbor_obj.root.boxes())

            with open(os.path.join(cur_res_dir, neighbor_fn+'.stats'), 'w') as fout:
                fout.write('cd: %f\nsd: %f\n' % (cd, sd))

    print('computing stats ...')
    compute_recon_numbers(
        in_dir=result_dir, baseline_dir=conf.baseline_dir, shapediff_topk=conf.shapediff_topk)

