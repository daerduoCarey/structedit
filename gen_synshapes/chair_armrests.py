from utils import *
import random

def make_T_armrests(args, box_id):
    out = []
    right_arm = {'name': 'chair_arm', 'parts': []}

    # right armrest
    x_min = -args.armWidth / 2
    x_max = args.armWidth / 2
    y_min = args.legHeight + args.seatHeight + args.armHeightLoc
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc + args.armHeight
    z_min = 0
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    right_arm['parts'].append({'name': 'armrest', 'box_id': box_id})
    box_id += 1

    # right armsupport
    x_min = (args.armWidth - args.armSupportWidth) / 2 - args.armWidth / 2
    x_max = (args.armWidth - args.armSupportWidth) / 2 + args.armSupportWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc
    z_min = args.backDepth + args.armSupportLoc
    z_max = args.backDepth + args.armSupportLoc + args.armSupportDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    right_arm['parts'].append({'name': 'armsupport', 'box_id': box_id})
    box_id += 1

    left_arm = {'name': 'chair_arm', 'parts': []}

    # left armrest
    x_min = args.seatWidth - args.armWidth / 2
    x_max = args.armWidth + args.seatWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight + args.armHeightLoc
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc + args.armHeight
    z_min = 0
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    left_arm['parts'].append({'name': 'armrest', 'box_id': box_id})
    box_id += 1

    # left armsupport
    x_min = (args.armWidth - args.armSupportWidth) / 2 + args.seatWidth - args.armWidth / 2
    x_max = (args.armWidth - args.armSupportWidth) / 2 + args.armSupportWidth + args.seatWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc
    z_min = args.backDepth + args.armSupportLoc
    z_max = args.backDepth + args.armSupportLoc + args.armSupportDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    left_arm['parts'].append({'name': 'armsupport', 'box_id': box_id})
    box_id += 1

    return out, [left_arm, right_arm], box_id

def make_7_armrests(args, box_id):
    out = []
    right_arm = {'name': 'chair_arm', 'parts': []}

    # right armrest
    x_min = -args.armWidth / 2
    x_max = args.armWidth / 2
    y_min = args.legHeight + args.seatHeight + args.armHeightLoc
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc + args.armHeight
    z_min = 0
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    right_arm['parts'].append({'name': 'armrest', 'box_id': box_id})
    box_id += 1

    # right armsupport
    x_min = (args.armWidth - args.armSupportWidth) / 2 - args.armWidth / 2
    x_max = (args.armWidth - args.armSupportWidth) / 2 + args.armSupportWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc
    z_min = args.armDepth - args.armSupportDepth
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    right_arm['parts'].append({'name': 'armsupport', 'box_id': box_id})
    box_id += 1

    left_arm = {'name': 'chair_arm', 'parts': []}

    # left armrest
    x_min = args.seatWidth - args.armWidth / 2
    x_max = args.armWidth + args.seatWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight + args.armHeightLoc
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc + args.armHeight
    z_min = 0
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    left_arm['parts'].append({'name': 'armrest', 'box_id': box_id})
    box_id += 1

    # left armsupport
    x_min = (args.armWidth - args.armSupportWidth) / 2 + args.seatWidth - args.armWidth / 2
    x_max = (args.armWidth - args.armSupportWidth) / 2 + args.armSupportWidth + args.seatWidth - args.armWidth / 2
    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.armHeightLoc
    z_min = args.armDepth - args.armSupportDepth
    z_max = args.armDepth
    out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
    left_arm['parts'].append({'name': 'armsupport', 'box_id': box_id})
    box_id += 1

    return out, [left_arm, right_arm], box_id

def make_armrests(args, box_id):
    args.armWidth = args.legWidth * (args.mode + 1) * 0.7
    args.armDepth = args.seatDepth / 2
    args.armHeight = args.legWidth * (args.mode + 1) * 0.3
    args.armHeightLoc = args.backHeight / 2
    args.armSupportLoc = (args.armDepth-args.backDepth) / 2
    args.armSupportDepth = args.legWidth * (args.mode + 1) * 0.4
    args.armSupportWidth = args.legWidth * (args.mode + 1) * 0.4

    if args.arm_type == 'T':
        args.armWidth *= 1.5
        return make_T_armrests(args, box_id)
    else:
        return make_7_armrests(args, box_id)

