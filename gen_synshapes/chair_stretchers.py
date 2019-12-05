from utils import *
import random
import math

def make_X_stretchers(args, box_id):
    out = []
    parts = []

    left_leg_x = args.seatWidth - args.legWidth / 2
    right_leg_x = args.legWidth / 2
    front_leg_z = args.seatDepth - args.legWidth /2
    back_leg_z = args.legWidth /2

    center_x = (left_leg_x + right_leg_x) / 2
    center_z = (front_leg_z + back_leg_z) / 2

    # one bar
    y = args.stretcherHeight + args.stretcherWidth / 2
    delta_y = args.stretcherWidth / 2
    delta_y_div_sqrt_two = delta_y / math.sqrt(2)
    setting = np.array([[center_x, y, center_z, 1],
                        [left_leg_x, y, front_leg_z, 1],
                        [center_x, y + delta_y, center_z, 1],
                        [center_x - delta_y_div_sqrt_two, y, center_z + delta_y_div_sqrt_two, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    setting = np.array([[center_x, y, center_z, 1],
                        [right_leg_x, y, front_leg_z, 1],
                        [center_x, y + delta_y, center_z, 1],
                        [center_x - delta_y_div_sqrt_two, y, center_z - delta_y_div_sqrt_two, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    return out, parts, box_id

def make_H_stretchers(args, box_id):
    out = []
    parts = []

    left_leg_x = args.seatWidth - args.legWidth / 2
    right_leg_x = args.legWidth / 2
    front_leg_z = args.seatDepth - args.legWidth / 2
    back_leg_z = args.legWidth / 2

    # one bar
    y = args.stretcherHeight + args.stretcherWidth / 2
    delta_y = args.stretcherWidth / 2
    setting = np.array([[left_leg_x, y, (front_leg_z + back_leg_z) / 2, 1],
                        [left_leg_x, y, front_leg_z, 1],
                        [left_leg_x, y + delta_y, (front_leg_z + back_leg_z) / 2, 1],
                        [left_leg_x - delta_y, y, (front_leg_z + back_leg_z) / 2, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    setting = np.array([[right_leg_x, y, (front_leg_z + back_leg_z) / 2, 1],
                        [right_leg_x, y, front_leg_z, 1],
                        [right_leg_x, y + delta_y, (front_leg_z + back_leg_z) / 2, 1],
                        [right_leg_x - delta_y, y, (front_leg_z + back_leg_z) / 2, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    y = args.stretcherHeight + args.stretcherWidth / 2
    setting = np.array([[(left_leg_x + right_leg_x) / 2, y, (front_leg_z + back_leg_z) / 2, 1],
                        [left_leg_x, y, (front_leg_z + back_leg_z) / 2, 1],
                        [(left_leg_x + right_leg_x) / 2, y + delta_y, (front_leg_z + back_leg_z) / 2, 1],
                        [(left_leg_x + right_leg_x) / 2, y, (front_leg_z + back_leg_z) / 2 + delta_y, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    return out, parts, box_id

def make_O_stretchers(args, box_id):
    out = []
    parts = []

    left_leg_x = args.seatWidth - args.legWidth / 2
    right_leg_x = args.legWidth / 2
    front_leg_z = args.seatDepth - args.legWidth /2
    back_leg_z = args.legWidth /2

    # one bar
    y = args.stretcherHeight + args.stretcherWidth / 2
    delta_y = args.stretcherWidth / 2
    setting = np.array([[left_leg_x, y, (front_leg_z + back_leg_z) / 2, 1],
                        [left_leg_x, y, front_leg_z, 1],
                        [left_leg_x, y + delta_y, (front_leg_z + back_leg_z) / 2, 1],
                        [left_leg_x - delta_y, y, (front_leg_z + back_leg_z) / 2, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    setting = np.array([[right_leg_x, y, (front_leg_z + back_leg_z) / 2, 1],
                        [right_leg_x, y, front_leg_z, 1],
                        [right_leg_x, y + delta_y, (front_leg_z + back_leg_z) / 2, 1],
                        [right_leg_x - delta_y, y, (front_leg_z + back_leg_z) / 2, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    y = args.stretcherHeight + args.stretcherWidth / 2
    setting = np.array([[(left_leg_x + right_leg_x) / 2, y, front_leg_z, 1],
                        [right_leg_x, y, front_leg_z, 1],
                        [(left_leg_x + right_leg_x) / 2, y + delta_y, front_leg_z, 1],
                        [(left_leg_x + right_leg_x) / 2, y, front_leg_z + delta_y, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    # another bar
    setting = np.array([[(left_leg_x + right_leg_x) / 2, y, back_leg_z, 1],
                        [right_leg_x, y, back_leg_z, 1],
                        [(left_leg_x + right_leg_x) / 2, y + delta_y, back_leg_z, 1],
                        [(left_leg_x + right_leg_x) / 2, y, back_leg_z + delta_y, 1]], dtype=np.float32)
    out.append(setting)
    parts.append({'name': 'bar_stretcher', 'box_id': box_id})
    box_id += 1

    return out, parts, box_id

def make_stretchers(args, box_id):
    args.stretcherWidth = args.legWidth * (args.mode + 1) * 0.6
    args.stretcherHeight = 0.03
    args.stretcherHeight2 = 0.03

    if args.stretcher_type == 'O':
        return make_O_stretchers(args, box_id)
    elif args.stretcher_type == 'X':
        return make_X_stretchers(args, box_id)
    else:
        return make_H_stretchers(args, box_id)

