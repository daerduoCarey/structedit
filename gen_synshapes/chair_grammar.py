import os
import sys
from utils import *
import random
from subprocess import call

from chair_legs import make_legs
from chair_back import make_back
from chair_seat import make_seat
from chair_armrests import make_armrests
from chair_stretchers import make_stretchers


def make_sofa(args, idx):
    csg = {'name': 'chair', 'parts': []};
    chair = [];
    box_id = 0

    back_parts, child_csg, box_id = make_back(args, box_id)
    csg['parts'].append(child_csg)
    chair += back_parts

    seat_parts, child_csg, box_id = make_seat(args, box_id)
    csg['parts'].append(child_csg)
    chair += seat_parts

    if args.hasArmrests: 
        armrest_parts, child_csg, box_id = make_armrests(args, box_id)
        csg['parts'] += child_csg
        chair += armrest_parts

    # normalize into unit sphere
    chair = normalize_shape(chair)

    export_settings(os.path.join(args.out_dir, '%05d.boxes.txt' % idx), chair)
    export_csg(os.path.join(args.out_dir, '%05d.csg.json' % idx), csg)
    with open(os.path.join(args.out_dir, '%05d.args.txt' % idx), 'w') as fout:
        fout.write(str(args))
    meshes = settings_to_meshes(chair)
    export_meshes(os.path.join(args.out_dir, '%05d.obj' % idx), meshes)


def make_stool(args, idx):
    csg = {'name': 'chair', 'parts': []};
    chair = [];
    box_id = 0

    seat_parts, child_csg, box_id = make_seat(args, box_id)
    csg['parts'].append(child_csg)
    chair += seat_parts

    leg_parts, leg_csg, box_id = make_legs(args, box_id)
    chair += leg_parts

    if args.hasStretchers: 
        stretcher_parts, child_csg, box_id = make_stretchers(args, box_id)
        leg_csg['parts'] += child_csg
        chair += stretcher_parts

    csg['parts'].append(leg_csg)

    # normalize into unit sphere
    chair = normalize_shape(chair)

    export_settings(os.path.join(args.out_dir, '%05d.boxes.txt' % idx), chair)
    export_csg(os.path.join(args.out_dir, '%05d.csg.json' % idx), csg)
    with open(os.path.join(args.out_dir, '%05d.args.txt' % idx), 'w') as fout:
        fout.write(str(args))
    meshes = settings_to_meshes(chair)
    export_meshes(os.path.join(args.out_dir, '%05d.obj' % idx), meshes)


def make_chair(args, idx):
    csg = {'name': 'chair', 'parts': []};
    chair = [];
    box_id = 0

    back_parts, child_csg, box_id = make_back(args, box_id)
    csg['parts'].append(child_csg)
    chair += back_parts

    seat_parts, child_csg, box_id = make_seat(args, box_id)
    csg['parts'].append(child_csg)
    chair += seat_parts

    if args.hasArmrests: 
        armrest_parts, child_csg, box_id = make_armrests(args, box_id)
        csg['parts'] += child_csg
        chair += armrest_parts

    leg_parts, leg_csg, box_id = make_legs(args, box_id)
    chair += leg_parts

    if args.hasStretchers: 
        stretcher_parts, child_csg, box_id = make_stretchers(args, box_id)
        leg_csg['parts'] += child_csg
        chair += stretcher_parts

    csg['parts'].append(leg_csg)

    # normalize into unit sphere
    chair = normalize_shape(chair)

    export_settings(os.path.join(args.out_dir, '%05d.boxes.txt' % idx), chair)
    export_csg(os.path.join(args.out_dir, '%05d.csg.json' % idx), csg)
    with open(os.path.join(args.out_dir, '%05d.args.txt' % idx), 'w') as fout:
        fout.write(str(args))
    meshes = settings_to_meshes(chair)
    export_meshes(os.path.join(args.out_dir, '%05d.obj' % idx), meshes)

