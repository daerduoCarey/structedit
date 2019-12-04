"""
    For StructEdit Project: sample a small uniform guassian in a shape local neighborhood space
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
from progressbar import ProgressBar
from eval_utils import compute_gen_cd_numbers
from eval_utils import compute_gen_sd_numbers

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--num_gen', type=int, default=100, help='how many shapes to generate?')
parser.add_argument('--sigma', type=float, help='sigma for local gaussian')
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.ckpt_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.category = conf.category
eval_conf.data_path = conf.data_path

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
result_dir = os.path.join(conf.result_path, conf.exp_name+'_editgen_sigma_%f'%conf.sigma)
if os.path.exists(result_dir):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % result_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(result_dir)

# create a new directory to store eval results
os.makedirs(result_dir)

# dataset
data_features = ['object', 'name']
dataset = PartNetDataset(conf.data_path, conf.test_dataset, data_features, load_geo=False)

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

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# generate shapes
with torch.no_grad():
    bar = ProgressBar()
    for i in bar(range(len(dataset))):
        obj, obj_name = dataset[i]
        cur_result_dir = os.path.join(result_dir, obj_name)
        os.mkdir(cur_result_dir)
        obj.to(device)
        cond_code = encoder.encode_structure(obj)
        for j in range(conf.num_gen):
            code = cond_code + torch.randn(1, conf.feature_size).cuda() * conf.sigma
            obj = decoder.decode_structure(z=code, max_depth=conf.max_tree_depth)
            output_filename = os.path.join(cur_result_dir, 'obj2-%03d.json'%j)
            PartNetDataset.save_object(obj=obj, fn=output_filename)
    
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

