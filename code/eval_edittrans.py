"""
    This script tests the shape difference/deltas/edits edit transfer (call from vis_edittrans.ipynb)
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from data import PartNetDataset, PartNetShapeDiffDataset, Tree
from progressbar import ProgressBar

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
parser.add_argument('--shapeA', type=int, help='data id')
parser.add_argument('--shapeB', type=int, help='data id')
parser.add_argument('--shapeC', type=int, help='data id')
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

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
result_dir = os.path.join(conf.result_path, conf.exp_name + '_edittrans_%s_%s_%s' % (conf.shapeA, conf.shapeB, conf.shapeC))
if os.path.exists(result_dir):
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

# set models to evaluation mode
for m in models:
    m.eval()

# test over all test shapes
with torch.no_grad():
    objA = PartNetDataset.load_object(os.path.join(conf.data_path, '%s.json'%conf.shapeA))
    objB = PartNetDataset.load_object(os.path.join(conf.data_path, '%s.json'%conf.shapeB))
    objC = PartNetDataset.load_object(os.path.join(conf.data_path, '%s.json'%conf.shapeC))
    
    encoder.encode_tree(objA)
    diffAB = Tree.compute_shape_diff(objA.root, objB.root)
    code = encoder.encode_tree_diff(objA, diffAB)

    encoder.encode_tree(objC)
    recon_obj_diff = decoder.decode_tree_diff(code, objC)
    recon_obj = Tree(Tree.apply_shape_diff(objC.root, recon_obj_diff))

    PartNetDataset.save_object(objA, os.path.join(result_dir, 'shapeA.json'))
    PartNetDataset.save_object(objB, os.path.join(result_dir, 'shapeB.json'))
    PartNetDataset.save_object(objC, os.path.join(result_dir, 'shapeC.json'))
    PartNetDataset.save_object(recon_obj, os.path.join(result_dir, 'output.json'))

