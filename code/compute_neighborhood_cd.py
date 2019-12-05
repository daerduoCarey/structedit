"""
    This script compute PartNet ground-truth chamfer-distance neighbors
"""

import os
import sys
import shutil
import numpy as np
import torch
import utils
from data import PartNetDataset
from chamfer_distance import ChamferDistance
from progressbar import ProgressBar

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()
data_path = '../data/partnetdata/chair_hier'
dataset_fn = 'train_no_other_less_than_10_parts.txt'
out_dir = os.path.join(data_path, 'neighbors_cd', os.path.splitext(dataset_fn)[0])

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
device = 'cuda:0'

# enumerate over all training shapes
num_shape = len(dataset)
n_points = 2048

print('Creating point clouds ...')
pcs = np.zeros((num_shape, n_points, 3), dtype=np.float32)
objs = []
names = []
bar = ProgressBar()
for i in bar(range(num_shape)):
    obj, name = dataset[i]
    objs.append(obj)
    names.append(name)
    obbs = torch.cat([item.view(1, -1) for item in obj.boxes(leafs_only=True)], dim=0).cpu().numpy()
    mesh_v, mesh_f = utils.gen_obb_mesh(obbs)
    pcs[i] = utils.sample_pc(mesh_v, mesh_f, n_points=n_points)

pcs = torch.tensor(pcs, dtype=torch.float32, device=device)

#np.save(os.path.join(out_dir, 'sd-%d.npy'%shape_id), sd_mat)
print('Computing distance ...')
bar = ProgressBar()
for i in bar(range(num_shape)):
    pc1 = pcs[i:i+1].repeat(num_shape, 1, 1)
    cd1, cd2 = chamferLoss(pc1, pcs)
    dists = ((cd1.sqrt().mean(1) + cd2.sqrt().mean(1)) / 2).cpu().numpy()

    np.save(
        os.path.join(out_dir, names[i]+'.npy'),
        {'dists': dists, 'names': names})

