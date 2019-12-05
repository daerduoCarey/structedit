"""
    This script tests Edit Transfer performance of the StructureNet baseline over the SynShapes synthetic dataset
    with quantitative evaluations
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from chamfer_distance import ChamferDistance
import utils
from config import add_eval_args
from data import PartNetDataset, Tree
from progressbar import ProgressBar

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity
chamfer_loss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--num_tuples', type=int)
parser.add_argument('--data_path1', type=str)
parser.add_argument('--data_path2', type=str)
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.ckpt_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path
eval_conf.category = conf.category

# load object category information
if conf.category is not None:
    Tree.load_category_info(conf.category)
    print(conf.category)

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)
print(conf.data_path1)
print(conf.data_path2)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {device}')

# load unit cube pc
unit_cube = torch.from_numpy(utils.load_pts('cube.pts')).to(device)

# check if eval results already exist. If so, delete it.
result_dir = os.path.join(conf.result_path, conf.exp_name + '_consis_'+conf.data_path1.split('/')[-1]+'_'+conf.data_path2.split('/')[-1])
if os.path.exists(result_dir):
    response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
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
_ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.ckpt_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

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

# load tuples from file
tuple_fn = '../stats/synchair_edittransfer_eval_list.txt'
print(tuple_fn)
with open(tuple_fn, 'r') as fin:
    all_tuples = [l.rstrip().split() for l in fin.readlines()]
    print(len(all_tuples))

# test over all tuples
with torch.no_grad():
    bar = ProgressBar()
    tot_cd = 0; tot_sd = 0; tot_cnt = 0;
    for i in bar(range(conf.num_tuples)):
        cur_res_dir = os.path.join(result_dir, '%06d' % i)
        os.mkdir(cur_res_dir)

        # load tuple (A, B, C, D)
        name_A, name_B, name_C, name_D = all_tuples[i]
        obj_A = PartNetDataset.load_object(os.path.join(conf.data_path1, name_A+'.json')).to(device)
        PartNetDataset.save_object(obj_A, os.path.join(cur_res_dir, 'obj_A.json'))
        obj_B = PartNetDataset.load_object(os.path.join(conf.data_path1, name_B+'.json')).to(device)
        PartNetDataset.save_object(obj_B, os.path.join(cur_res_dir, 'obj_B.json'))
        obj_C = PartNetDataset.load_object(os.path.join(conf.data_path2, name_C+'.json')).to(device)
        PartNetDataset.save_object(obj_C, os.path.join(cur_res_dir, 'obj_C.json'))
        obj_D = PartNetDataset.load_object(os.path.join(conf.data_path2, name_D+'.json')).to(device)
        PartNetDataset.save_object(obj_D, os.path.join(cur_res_dir, 'obj_D.json'))

        # compute recon_D
        featA = encoder.encode_structure(obj_A)
        featB = encoder.encode_structure(obj_B)
        featC = encoder.encode_structure(obj_C)
        featD = featB - featA + featC
        recon_D = decoder.decode_structure(z=featD, max_depth=conf.max_tree_depth)
        PartNetDataset.save_object(recon_D, os.path.join(cur_res_dir, 'recon_D.json'))

        # compute dist between D and Dp
        cd = geometry_dist(obj_D, recon_D)
        sd = struct_dist(obj_D.root, recon_D.root) / len(obj_D.root.boxes())
        with open(os.path.join(cur_res_dir, 'stats.txt'), 'w') as fout:
            fout.write('cd: %f\nsd: %f\n' % (cd, sd))

        tot_cd += cd
        tot_sd += sd
        tot_cnt += 1

avg_cd = tot_cd / tot_cnt
avg_sd = tot_sd / tot_cnt
print('Avg cd: %f' % avg_cd)
print('Avg sd: %f' % avg_sd)
with open(os.path.join(result_dir, 'stats.txt'), 'w') as fout:
    fout.write('cd: %f\nsd: %f\n' % (avg_cd, avg_sd))

