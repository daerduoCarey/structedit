from utils import *
import random

def make_simple_back(args, box_id):
    x_min = (args.seatWidth - args.backWidth) / 2
    x_max = (args.seatWidth - args.backWidth) / 2 + args.backWidth
    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.backHeight
    z_min = 0
    z_max = args.backDepth
    out = [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]
    csg = {'name': 'chair_back', 'parts': [{'name': 'back_surface', 'parts': [{'name': 'back_single_surface', 'box_id': box_id}]}]}
    return out, csg, box_id + 1

def make_back_frame(args, box_id):
    out = []

    csg = {'name': 'back_frame', 'parts': []}

    # top-bar
    if args.hasBackTopBar:
        x_min = (args.seatWidth - args.backWidth) / 2
        x_max = (args.seatWidth - args.backWidth) / 2 + args.backWidth
        y_min = args.legHeight + args.seatHeight + args.backHeight - args.backTopBarHeight
        y_max = args.legHeight + args.seatHeight + args.backHeight
        z_min = 0
        z_max = args.backDepth
        out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
        csg['parts'].append({'name': 'back_frame_horizontal_bar', 'box_id': box_id})
        box_id += 1

    # right side-bar
    if args.hasBackSideBar:
        x_min = (args.seatWidth - args.backWidth) / 2
        x_max = (args.seatWidth - args.backWidth) / 2 + args.backSideBarWidth
        y_min = args.legHeight + args.seatHeight
        y_max = args.legHeight + args.seatHeight + args.backHeight - args.backTopBarHeight
        z_min = 0
        z_max = args.backDepth
        out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
        csg['parts'].append({'name': 'back_frame_vertical_bar', 'box_id': box_id})
        box_id += 1
        
        # left side-bar
        x_min = (args.seatWidth - args.backWidth) / 2 + args.backWidth - args.backSideBarWidth
        x_max = (args.seatWidth - args.backWidth) / 2 + args.backWidth
        y_min = args.legHeight + args.seatHeight
        y_max = args.legHeight + args.seatHeight + args.backHeight - args.backTopBarHeight
        z_min = 0
        z_max = args.backDepth
        out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
        csg['parts'].append({'name': 'back_frame_vertical_bar', 'box_id': box_id})
        box_id += 1

    return out, csg, box_id

def make_T_back(args, box_id):
    out, back_frame_csg, box_id = make_back_frame(args, box_id)

    # add vertical bars
    x_range_min = (args.seatWidth - args.backWidth) / 2 + args.backSideBarWidth
    x_range_max = (args.seatWidth - args.backWidth) / 2 + args.backWidth - args.backSideBarWidth
    x_grid_size = (x_range_max - x_range_min) / (args.backNumSurfaceBars + 1)

    y_min = args.legHeight + args.seatHeight
    y_max = args.legHeight + args.seatHeight + args.backHeight - args.backTopBarHeight
    z_min = (args.backDepth - args.backSurfaceBarDepth) / 2
    z_max = (args.backDepth - args.backSurfaceBarDepth) / 2 + args.backSurfaceBarDepth

    surface_csg = {'name': 'back_surface', 'parts': []}

    for i in range(args.backNumSurfaceBars):
        x_min = x_range_min + x_grid_size*(i+1) - args.backSurfaceBarWidth / 2
        x_max = x_range_min + x_grid_size*(i+1) + args.backSurfaceBarWidth / 2
        out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
        surface_csg['parts'].append({'name': 'back_surface_vertical_bar', 'box_id': box_id+i})

    csg = {'name': 'chair_back', 'parts': [surface_csg, back_frame_csg]}
    
    return out, csg, box_id + args.backNumSurfaceBars

def make_H_back(args, box_id):
    out, back_frame_csg, box_id = make_back_frame(args, box_id)

    # add horizontal bars
    y_range_min = args.legHeight + args.seatHeight
    y_range_max = args.legHeight + args.seatHeight + args.backHeight - args.backTopBarHeight
    y_grid_size = (y_range_max - y_range_min) / (args.backNumSurfaceBars + 1)

    x_min = (args.seatWidth - args.backWidth) / 2 + args.backSideBarWidth
    x_max = (args.seatWidth - args.backWidth) / 2 + args.backWidth - args.backSideBarWidth
    z_min = (args.backDepth - args.backSurfaceBarDepth) / 2
    z_max = (args.backDepth - args.backSurfaceBarDepth) / 2 + args.backSurfaceBarDepth

    surface_csg = {'name': 'back_surface', 'parts': []}

    for i in range(args.backNumSurfaceBars):
        y_min = y_range_min + y_grid_size*(i+1) - args.backSurfaceBarWidth / 2
        y_max = y_range_min + y_grid_size*(i+1) + args.backSurfaceBarWidth / 2
        out.append(create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max))
        surface_csg['parts'].append({'name': 'back_surface_horizontal_bar', 'box_id': box_id+i})

    csg = {'name': 'chair_back', 'parts': [surface_csg, back_frame_csg]}
    
    return out, csg, box_id + args.backNumSurfaceBars

def make_back(args, box_id):
    args.backDepth *= (args.mode * 1.2 + 1)
    args.backHeight *= (args.mode * 0.8 + 1)
    if args.back_type == 'simple':
        return make_simple_back(args, box_id)
    elif args.back_type == 'H':
        return make_H_back(args, box_id)
    else:
        return make_T_back(args, box_id)

