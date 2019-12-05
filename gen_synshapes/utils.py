import os
import sys
import numpy as np
import json
import random
import trimesh
from sklearn.decomposition import PCA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', 'code'))
from pyquaternion import Quaternion

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()
    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
    f_arr = np.vstack(faces)
    v_arr = np.vstack(vertices)
    mesh = dict()
    mesh['v'] = v_arr
    mesh['f'] = f_arr
    return mesh

def export_obj(out, mesh):
    v = mesh['v']; f = mesh['f'];
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def get_quaternion_from_axis_angle(axis, angle):
    return Quaternion(axis=axis, angle=angle)

def get_quaternion_from_xy_axes(x, y):
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.vstack([x, y, z]).T
    return Quaternion(matrix=R)

def get_rot_mat_from_quaternion(q):
    return np.array(q.transformation_matrix, dtype=np.float32)

# center: numpy array of length 3
# size: numpy array of length 3
# q: numpy array of length 4 for quaternion
# output: mesh
#           v       --> vertices
#           f       --> faces
#           setting --> 4 x 4 numpy array containing the world coordinates for the cube center (0, 0, 0, 1) 
#                       and three local axes (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)
def gen_cuboid(center, size, q):
    cube_mesh = load_obj('cube.obj')
    cube_v = cube_mesh['v']
    cube_f = cube_mesh['f']
    n_vert = cube_v.shape[0]
    n_face = cube_f.shape[1]
    cube_v = np.concatenate([cube_v, np.ones((n_vert, 1))], axis=1)
    cube_control_v = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    S = np.array([[size[0], 0, 0, 0], [0, size[1], 0, 0], [0, 0, size[2], 0], [0, 0, 0, 1]], dtype=np.float32)
    R = q.transformation_matrix
    T = np.array([[1, 0, 0, center[0]], [0, 1, 0, center[1]], [0, 0, 1, center[2]], [0, 0, 0, 1]], dtype=np.float32)
    rot = T.dot(R).dot(S)
    cube_v = rot.dot(cube_v.T).T
    cube_control_v = rot.dot(cube_control_v.T).T
    mesh = dict()
    mesh['v'] = cube_v
    mesh['f'] = cube_f
    mesh['setting'] = cube_control_v
    return mesh

def assemble_meshes(mesh_list):
    n_vert = 0
    verts = []; faces = [];
    for mesh in mesh_list:
        verts.append(mesh['v'])
        faces.append(mesh['f']+n_vert)
        n_vert += mesh['v'].shape[0]
    vert_arr = np.vstack(verts)
    face_arr = np.vstack(faces)
    mesh = dict()
    mesh['v'] = vert_arr
    mesh['f'] = face_arr
    return mesh

def export_settings(out_fn, setting_list):
    with open(out_fn, 'w') as fout:
        for setting in setting_list:
            for i in range(4):
                for j in range(4):
                    fout.write('%f ' % setting[i, j])
            fout.write('\n')

def export_csg(out_fn, csg):
    with open(out_fn, 'w') as fout:
        json.dump(csg, fout)

def export_meshes(out, mesh_list):
    with open(out, 'w') as fout:
        n_vert = 0
        verts = []; faces = [];
        for idx, mesh in enumerate(mesh_list):
            fout.write('\ng %d\n' % idx)
            for i in range(mesh['v'].shape[0]):
                fout.write('v %f %f %f\n' % (mesh['v'][i, 0], mesh['v'][i, 1], mesh['v'][i, 2]))
            for i in range(mesh['f'].shape[0]):
                fout.write('f %d %d %d\n' % (mesh['f'][i, 0]+n_vert, mesh['f'][i, 1]+n_vert, mesh['f'][i, 2]+n_vert))
            n_vert += mesh['v'].shape[0]

def gen_cuboid_from_setting(setting):
    R = np.array([setting[1] - setting[0], 
                  setting[2] - setting[0],
                  setting[3] - setting[0],
                  setting[0]], dtype=np.float32).T
    cube_mesh = load_obj('cube.obj')
    cube_v = cube_mesh['v']
    cube_f = cube_mesh['f']
    n_vert = cube_v.shape[0]
    cube_v = np.concatenate([cube_v, np.ones((n_vert, 1))], axis=1)
    mesh = dict()
    mesh['v'] = R.dot(cube_v.T).T
    mesh['f'] = cube_f
    mesh['setting'] = setting
    return mesh

def settings_to_meshes(settings):
    meshes = []
    for setting in settings:
        meshes.append(gen_cuboid_from_setting(setting))
    return meshes

def create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max):
    setting = np.array([[(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2, 1],
                        [x_max, (y_min+y_max)/2, (z_min+z_max)/2, 1],
                        [(x_min+x_max)/2, y_max, (z_min+z_max)/2, 1],
                        [(x_min+x_max)/2, (y_min+y_max)/2, z_max, 1]], dtype=np.float32)
    return setting

def create_rotate_45_setting(x_min, x_max, y_min, y_max, z_min, z_max):
    l1 = (x_max - x_min) / 2 / np.sqrt(2)
    l2 = (z_max - z_min) / 2 / np.sqrt(2)
    setting = np.array([[(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2, 1],
                        [(x_min+x_max)/2+l1, (y_min+y_max)/2, (z_min+z_max)/2+l1, 1],
                        [(x_min+x_max)/2, y_max, (z_min+z_max)/2, 1],
                        [(x_min+x_max)/2-l2, (y_min+y_max)/2, (z_min+z_max)/2+l2, 1]], dtype=np.float32)
    return setting

def normalize_shape(settings):
    mesh = assemble_meshes(settings_to_meshes(settings))
    pts = sample_pc(mesh['v'][:, :3], mesh['f'], n_points=200)
    center = np.mean(pts, axis=0)
    pts -= center
    scale = np.sqrt(np.max(np.sum(pts**2, axis=1)))
    T = np.array([[1, 0, 0, -center[0]],
                  [0, 1, 0, -center[1]],
                  [0, 0, 1, -center[2]],
                  [0, 0, 0, 1]], dtype=np.float32)
    S = np.array([[1.0/scale, 0, 0, 0],
                  [0, 1.0/scale, 0, 0],
                  [0, 0, 1.0/scale, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    rot_mat = S.dot(T)
    new_settings = []
    for setting in settings:
        new_settings.append(rot_mat.dot(setting.T).T)
    return new_settings

def random_rotate(settings):
    rotation_angle = random.random() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval, 0],
                                [0, 1, 0, 0],
                                [-sinval, 0, cosval, 0],
                                [0, 0, 0, 1]], dtype=np.float32)
    new_settings = []
    for setting in settings:
        new_settings.append(rotation_matrix.dot(setting.T).T)
    return new_settings

def gen_obb_mesh(obbs):
    # load cube
    cube_mesh = load_obj('cube.obj')
    cube_v = cube_mesh['v']
    cube_f = cube_mesh['f']

    all_v = []; all_f = []; vid = 0;
    for pid in range(obbs.shape[0]):
        p = obbs[pid, :]
        center = p[0: 3]
        lengths = p[3: 6]
        dir_1 = p[6: 9]
        dir_2 = p[9: ]

        dir_1 = dir_1/np.linalg.norm(dir_1)
        dir_2 = dir_2/np.linalg.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)
        dir_3 = dir_3/np.linalg.norm(dir_3)

        v = np.array(cube_v, dtype=np.float32)
        f = np.array(cube_f, dtype=np.int32)
        rot = np.vstack([dir_1, dir_2, dir_3])
        v *= lengths
        v = np.matmul(v, rot)
        v += center

        all_v.append(v)
        all_f.append(f+vid)
        vid += v.shape[0]

    all_v = np.vstack(all_v)
    all_f = np.vstack(all_f)
    return all_v, all_f

def sample_pc(v, f, n_points=10000):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

def fit_box(points):
    pca = PCA()
    pca.fit(points)
    pcomps = pca.components_

    points_local = np.matmul(pcomps, points.transpose()).transpose()

    all_max = points_local.max(axis=0)
    all_min = points_local.min(axis=0)
    
    center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
    size = all_max - all_min

    xdir = pcomps[0, :]
    ydir = pcomps[1, :]

    return np.hstack([center, size, xdir, ydir]).astype(np.float32)

