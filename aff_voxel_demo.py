from __future__ import division, print_function

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import trimesh
import json
import argparse
import sys
import os

def plot_voxel(voxel):
  # voxel = np.load('/home/atabak/tmp/ycb_sample/fromHisDataset/02691156_fff513f407e00e85a9ced22d91ad7027_view019_gt_rotvox_samescale_128.npz')
  # voxel = voxel['voxel']
  # with open('rotated_mesh.binvox', 'rb') as f:
  #   m1 = binvox_rw.read_as_3d_array(f)
  #
  # voxel=m1.data
  from skimage.measure import block_reduce
  ds_voxel = block_reduce(voxel, (4, 4, 4))
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  #ax.set_aspect('equal')

  # ax.voxels(voxel, edgecolor="k")
  ax.voxels(ds_voxel, edgecolor="k", facecolors=[1, 0, 0, 0.05])
  # ax.view_init(90, 270)
  ax.view_init(0, 180)
  plt.draw() 
  plt.show()
  # for angle in range(0, 360):
  #   ax.view_init(0, angle)
  #   plt.draw()
  #   plt.pause(.001)


def mesh_to_voxel(dict_path):

  #mesh = trimesh.load(dict_info['model'].split('YCB_Video_Dataset/YCB_Video_Dataset/')[1])
  #mesh = trimesh.load(dict_info['model'].replace('media', 'mnt'))
  dict_info = np.load(dict_path)
  mesh = trimesh.load(str(dict_info['model_path']))
  rot = np.array(dict_info['rot'])
  RT = np.zeros((4, 4))
  RT_aux = np.zeros((4, 4))
  RT[:3, :3] = rot
  RT[3, 3] = 1.
  mesh.apply_transform(RT)

  # 90 around z
  rot_90_z = np.array([(0.0, 1.0, 0.0),
                       (-1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0)])
  # 90 around y
  rot_90_y = np.array([(0.0, 0.0, 1.0),
                       (0.0, 1.0, 0.0),
                       (-1.0, 0.0, 0.0)])


  RT_aux[3, 3] = 1.
  RT_aux[:3, :3] = rot_90_z
  mesh.apply_transform(RT_aux)
  RT_aux[:3, :3] = rot_90_y
  mesh.apply_transform(RT_aux)

  RT_aux[3,3] = 1.
  is_watertight = False
  attempts = 0
  while (is_watertight == False and attempts < 10):
    is_watertight = trimesh.repair.fill_holes(mesh)
    attempts += 1

  meshvoxel = trimesh.voxel.local_voxelize(mesh, (0., 0., 0.), pitch=0.25/129, radius=64)[0] #25cm devided in 129 voxels

  voxel_path = dict_path.replace('metadata', 'voxel')#.replace('png', 'npz').replace('media', 'mnt')
  voxel_dir = os.path.dirname(voxel_path)
  if not os.path.exists(voxel_dir):
    os.makedirs(voxel_dir)

  np.savez(voxel_path, voxel=meshvoxel)


if __name__ == "__main__":
  trimesh.util.attach_to_log()
  parser = argparse.ArgumentParser()
  parser.add_argument('--division_num', type=int, default=0,
                      help='division number 0<=division_num<60')
  
  args = parser.parse_args()
  base_dir = "/mnt/hdd/aff_render/metadata"
  target_dir = os.path.join(base_dir, '{:04d}'.format(args.division_num))
  if os.path.exists(target_dir):
    for info in os.listdir(target_dir):
      target_path = os.path.join(target_dir, info)
      mesh_to_voxel(target_path)

