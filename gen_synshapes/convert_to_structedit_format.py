import os
import sys
from commons import check_mkdir
from progressbar import ProgressBar
import json
import utils
import numpy as np

in_dir = sys.argv[1]
out_dir = sys.argv[2]
check_mkdir(out_dir)

def load_boxes(fn):
    boxes = [];
    with open(fn, 'r') as fin:
        for l in fin.readlines():
            d = np.array([float(item) for item in l.rstrip().split()], dtype=np.float32)
            boxes.append(np.reshape(d, [1, 4, 4])[:, :, :3])
    boxes = np.concatenate(boxes, axis=0)
    xdir = boxes[:, 1, :] - boxes[:, 0, :]
    xlen = np.sqrt((xdir**2).sum(axis=1, keepdims=True))
    xdir /= xlen
    ydir = boxes[:, 2, :] - boxes[:, 0, :]
    ylen = np.sqrt((ydir**2).sum(axis=1, keepdims=True))
    ydir /= ylen
    zdir = boxes[:, 3, :] - boxes[:, 0, :]
    zlen = np.sqrt((zdir**2).sum(axis=1, keepdims=True))
    boxes[:, 1, 0] = xlen.flatten() * 2
    boxes[:, 1, 1] = ylen.flatten() * 2
    boxes[:, 1, 2] = zlen.flatten() * 2
    boxes[:, 2, :] = xdir
    boxes[:, 3, :] = ydir
    return boxes

idx = 0
def convert_to_structurenet_format(csg, boxes):
    global idx
    sem_name = csg['name']
    if 'box_id' in csg:
        out = dict({
            'box_id': csg['box_id'], 
            'id': idx,
            'label': sem_name,
            'box': boxes[csg['box_id']].flatten().tolist(),
            })
        idx += 1
        v, f = utils.gen_obb_mesh(np.reshape(boxes[csg['box_id']], [1, -1]))
        pc = utils.sample_pc(v, f)
    else:
        children = []; cpcs = [];
        for citem in csg['parts']:
            cnode, cpc = convert_to_structurenet_format(citem, boxes)
            children.append(cnode);
            cpcs.append(cpc)
        pc = np.concatenate(cpcs, axis=0)
        box = utils.fit_box(pc)
        out = dict({
            'id': idx,
            'label': sem_name,
            'box': box.tolist(),
            'children': children,
            })
    return out, pc


bar = ProgressBar()
for item in bar(os.listdir(in_dir)):
    if item.endswith('csg.json'):
        idx = 0
        fn = item.split('.')[0]
        boxes = load_boxes(os.path.join(in_dir, fn+'.boxes.txt'))
        with open(os.path.join(in_dir, fn+'.csg.json'), 'r') as fin:
            csg = json.load(fin)
        tree, _ = convert_to_structurenet_format(csg, boxes)
        with open(os.path.join(out_dir, fn+'.json'), 'w') as fout:
            json.dump(tree, fout)

