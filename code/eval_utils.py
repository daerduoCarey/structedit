import os
import sys
import numpy as np
import torch
from progressbar import ProgressBar
from chamfer_distance import ChamferDistance
from data import PartNetDataset, PartNetShapeDiffDataset
import utils


def compute_recon_numbers(in_dir, baseline_dir, shapediff_topk):
    topk_cd = np.zeros((shapediff_topk), dtype=np.float32)
    topk_sd = np.zeros((shapediff_topk), dtype=np.float32)
    baseline_topk_cd = np.zeros((shapediff_topk), dtype=np.float32)
    baseline_topk_sd = np.zeros((shapediff_topk), dtype=np.float32)
    topk_cnt = np.zeros((shapediff_topk), dtype=np.int32)
    for anno_id in os.listdir(in_dir):
        if '.' not in anno_id:
            cur_dir = os.path.join(in_dir, anno_id)
            for item in os.listdir(cur_dir):
                if item.endswith('.stats'):
                    nid = int(item.split('.')[0].split('_')[1])
                    neighbor_anno_id = item.split('.')[0].split('_')[2]
                    with open(os.path.join(cur_dir, item), 'r') as fin:
                        topk_cd[nid] += float(fin.readline().rstrip().split()[-1])
                        topk_sd[nid] += float(fin.readline().rstrip().split()[-1])
                    with open(os.path.join(baseline_dir, neighbor_anno_id, 'stats.txt'), 'r') as fin:
                        baseline_topk_cd[nid] += float(fin.readline().rstrip().split()[-1])
                        baseline_topk_sd[nid] += float(fin.readline().rstrip().split()[-1])
                    topk_cnt[nid] += 1

    topk_cd /= topk_cnt
    topk_sd /= topk_cnt
    baseline_topk_cd /= topk_cnt
    baseline_topk_sd /= topk_cnt
    print('ours cd mean: %.5f' % np.mean(topk_cd))
    print('ours sd mean: %.5f' % np.mean(topk_sd))
    print('structurenet cd mean: %.5f' % np.mean(baseline_topk_cd))
    print('structurenet sd mean: %.5f' % np.mean(baseline_topk_sd))

    with open(os.path.join(in_dir, 'stats.txt'), 'w') as fout:
        fout.write('ours cd mean: %.5f\n' % np.mean(topk_cd))
        fout.write('ours sd mean: %.5f\n' % np.mean(topk_sd))
        fout.write('structurenet cd mean: %.5f\n' % np.mean(baseline_topk_cd))
        fout.write('structurenet sd mean: %.5f\n' % np.mean(baseline_topk_sd))
        for i in range(shapediff_topk):
            fout.write('%d %d %.5f %.5f %.5f %.5f\n' % (i, topk_cnt[i], topk_cd[i], topk_sd[i], baseline_topk_cd[i], baseline_topk_sd[i]))


def compute_gen_cd_numbers(in_dir, data_path, object_list, shapediff_topk, shapediff_metric, self_is_neighbor, tot_shape):
    chamfer_loss = ChamferDistance()

    data_features = ['object', 'name', 'neighbor_diffs', 'neighbor_objs', 'neighbor_names']
    dataset = PartNetShapeDiffDataset(
        data_path, object_list, data_features, shapediff_topk, shapediff_metric, self_is_neighbor)

    tot_gen = 100
    bar = ProgressBar()
    quality = 0.0; coverage = 0.0;
    for i in bar(range(tot_shape)):
        obj, obj_name, neighbor_diffs, neighbor_objs, neighbor_names = dataset[i]

        mat = np.zeros((shapediff_topk, tot_gen), dtype=np.float32)
        gt_pcs = []
        for ni in range(shapediff_topk):
            obbs_np = torch.cat([item.view(1, -1) for item in neighbor_objs[ni].boxes(leafs_only=True)], dim=0).cpu().numpy()
            mesh_v, mesh_f = utils.gen_obb_mesh(obbs_np)
            pc_sample = utils.sample_pc(mesh_v, mesh_f)
            gt_pcs.append(np.expand_dims(pc_sample, axis=0))
        gt_pcs = np.concatenate(gt_pcs, axis=0)
        gt_pcs = torch.from_numpy(gt_pcs).float().cuda()

        for i in range(tot_gen):
            obj = PartNetDataset.load_object(os.path.join(in_dir, obj_name, 'obj2-%03d.json'%i))
            obbs_np = torch.cat([item.view(1, -1) for item in obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
            mesh_v, mesh_f = utils.gen_obb_mesh(obbs_np)
            gen_pc = utils.sample_pc(mesh_v, mesh_f)
            gen_pc = np.tile(np.expand_dims(gen_pc, axis=0), [shapediff_topk, 1, 1])
            gen_pc = torch.from_numpy(gen_pc).float().cuda()
            d1, d2 = chamfer_loss(gt_pcs.cuda(), gen_pc)
            mat[:, i] = (d1.sqrt().mean(dim=1) + d2.sqrt().mean(dim=1)).cpu().numpy() / 2

        quality += mat.min(axis=0).mean()
        coverage += mat.min(axis=1).mean()
        np.save(os.path.join(in_dir, obj_name, 'cd_stats.npy'), mat)

    quality /= tot_shape
    coverage /= tot_shape
    print('mean cd quality: %.5f' % quality)
    print('mean cd coverage: %.5f' % coverage)
    print('q + c: %.5f' % (quality + coverage))
    with open(os.path.join(in_dir, 'neighbor_%s_cd_stats.txt'%shapediff_metric), 'w') as fout:
        fout.write('mean cd quality: %.5f\n' % quality)
        fout.write('mean cd coverage: %.5f\n' % coverage)
        fout.write('q + c: %.5f\n' % (quality + coverage))


def compute_gen_sd_numbers(in_dir, data_path, object_list, shapediff_topk, shapediff_metric, self_is_neighbor, tot_shape):
    chamfer_loss = ChamferDistance()
    unit_cube = torch.from_numpy(utils.load_pts('cube.pts'))

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
                    gt_boxes = torch.cat([gt_node.children[cid].get_box_quat() for cid in gt_cnodes_per_sem[sem]], dim=0)
                    pred_boxes = torch.cat([pred_node.children[cid].get_box_quat() for cid in pred_cnodes_per_sem[sem]], dim=0)

                    num_gt = gt_boxes.size(0)
                    num_pred = pred_boxes.size(0)

                    if num_gt == 1 and num_pred == 1:
                        cur_matched_gt_idx = [0]
                        cur_matched_pred_idx = [0]
                    else:
                        gt_boxes_tiled = gt_boxes.unsqueeze(dim=1).repeat(1, num_pred, 1)
                        pred_boxes_tiled = pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                        dmat = box_dist(gt_boxes_tiled.view(-1, 10), pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_pred)
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

                return struct_diff

    # create dataset and data loader
    data_features = ['object', 'name', 'neighbor_diffs', 'neighbor_objs', 'neighbor_names']
    dataset = PartNetShapeDiffDataset(
        data_path, object_list, data_features, shapediff_topk, shapediff_metric, self_is_neighbor)

    tot_gen = 100
    bar = ProgressBar()
    quality = 0.0; coverage = 0.0;
    for i in bar(range(tot_shape)):
        obj, obj_name, neighbor_diffs, neighbor_objs, neighbor_names = dataset[i]

        mat1 = np.zeros((shapediff_topk, tot_gen), dtype=np.float32)
        mat2 = np.zeros((shapediff_topk, tot_gen), dtype=np.float32)
        for j in range(tot_gen):
            gen_obj = PartNetDataset.load_object(os.path.join(in_dir, obj_name, 'obj2-%03d.json'%j))
            for ni in range(shapediff_topk):
                sd = struct_dist(neighbor_objs[ni].root, gen_obj.root)
                mat1[ni, j] = sd / len(neighbor_objs[ni].root.boxes())
                mat2[ni, j] = sd / len(gen_obj.root.boxes())

        quality += mat2.min(axis=0).mean()
        coverage += mat1.min(axis=1).mean()
        np.save(os.path.join(in_dir, obj_name, 'sd_mat1_stats.npy'), mat1)
        np.save(os.path.join(in_dir, obj_name, 'sd_mat2_stats.npy'), mat2)

    quality /= tot_shape
    coverage /= tot_shape
    print('mean sd quality: ', quality)
    print('mean sd coverage: ', coverage)
    print('q + c: %.5f' % (quality + coverage))
    with open(os.path.join(in_dir, 'neighbor_%s_sd_stats.txt'%shapediff_metric), 'w') as fout:
        fout.write('mean sd quality: %f\n' % quality)
        fout.write('mean sd coverage: %f\n' % coverage)
        fout.write('q + c: %.5f\n' % (quality + coverage))

