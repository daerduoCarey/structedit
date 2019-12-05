from utils import *
import random

def make_seat(args, box_id):
    x_min = 0
    x_max = args.seatWidth
    y_min = args.legHeight
    y_max = args.legHeight + args.seatHeight * (1 + args.mode * 1.5)
    z_min = 0
    z_max = args.seatDepth
    out = [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]
    csg = {'name': 'chair_seat', 'box_id': box_id}
    return out, csg, box_id + 1
