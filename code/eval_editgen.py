"""
    This script tests the shape difference/deltas/edits free generation
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
from progressbar import ProgressBar
from config import add_eval_args
from data import PartNetDataset, Tree
from eval_utils import compute_gen_cd_numbers
from eval_utils import compute_gen_sd_numbers
import utils

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--num_gen', type=int, default=100, help='num generation per input shape')
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.ckpt_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.category = conf.category
eval_conf.data_path = conf.data_path
if hasattr(conf, 'self_is_neighbor'):
    eval_conf.self_is_neighbor = conf.self_is_neighbor
if not hasattr(conf, 'latent_size'):
    conf.latent_size = conf.feature_size

# load object category information
if conf.category is not None:
    Tree.load_category_info(conf.category)

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it.
result_dir = os.path.join(conf.result_path, conf.exp_name + '_editgen')
if os.path.exists(result_dir):
     response = input('eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
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
    strict=True,
    device=device)

# create dataset and data loader
data_features = ['object', 'name']
dataset = PartNetDataset(conf.data_path, conf.test_dataset, data_features)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# test over all test shapes
with torch.no_grad():
    pbar = ProgressBar()
    print('generating edits ...')
    for i in pbar(range(len(dataset))):
        obj, obj_name = dataset[i]
        # print('[%d/%d] %s' % (i, conf.num_to_eval, obj_name))

        cur_res_dir = os.path.join(result_dir, obj_name)
        if not os.path.exists(cur_res_dir):
            os.mkdir(cur_res_dir)

        obj.to(device)
        PartNetDataset.save_object(obj, os.path.join(cur_res_dir, 'cond_obj.json'))

        encoder.encode_tree(obj)

        zs = torch.randn(conf.num_gen, conf.latent_size).to(device)

        for i in range(conf.num_gen):
            recon_diff = decoder.decode_tree_diff(zs[i:i+1], obj)
            recon_obj2 = Tree(Tree.apply_shape_diff(obj.root, recon_diff))

            PartNetDataset.save_object(recon_obj2, os.path.join(cur_res_dir, 'obj2-%03d.json' % i))

            with open(os.path.join(cur_res_dir, 'diff-%03d.txt' % i), 'w') as fout:
                fout.write(str(recon_diff))

    print('computing chamfer distance stats ...')
    compute_gen_cd_numbers(
        in_dir=result_dir, data_path=conf.data_path, object_list=conf.test_dataset,
        shapediff_topk=conf.shapediff_topk, shapediff_metric=conf.shapediff_metric,
        self_is_neighbor=conf.self_is_neighbor, tot_shape=len(dataset))

    print('computing structure distance stats ...')
    compute_gen_sd_numbers(
        in_dir=result_dir, data_path=conf.data_path, object_list=conf.test_dataset,
        shapediff_topk=conf.shapediff_topk, shapediff_metric=conf.shapediff_metric,
        self_is_neighbor=conf.self_is_neighbor, tot_shape=len(dataset))

