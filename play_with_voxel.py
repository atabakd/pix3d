from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import trimesh
import json
import binvox_rw

def plot_voxel(voxel_path):
  voxel = np.load('/home/atabak/ycb_sample/fromHisDataset/02691156_fff513f407e00e85a9ced22d91ad7027_view019_gt_rotvox_samescale_128.npz')
  voxel = voxel['voxel']
  # with open('rotated_mesh_1.binvox', 'rb') as f:
  #   m1 = binvox_rw.read_as_3d_array(f)
  # voxel=m1.data
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.set_aspect('equal')

  ax.voxels(voxel, edgecolor="k")
  # ax.view_init(90, 270)
  ax.view_init(0, 0)
  plt.draw()

  plt.show()
  # for angle in range(0, 360):
  #   ax.view_init(0, angle)
  #   plt.draw()
  #   plt.pause(.001)
  
def mesh_to_voxel(mesh_path):
  
  # load a file by name or from a buffer
  mesh = trimesh.load(mesh_path)
  with open('/home/atabak/ycb_sample/{:04d}.json'.format(0), 'r') as j_file:
    data_list = json.load(j_file)
  dict_info = data_list[0]
  rot = np.array(dict_info['rot_mat'])
  trans = np.array(dict_info['trans_mat'])
  RT = np.zeros((4,4))
  rot_aux = np.array([(1.0, 0.0, 0.0),
             (0.0, -1.0, 0.0),
             (0.0, 0.0, -1.0)])
  RT_aux = np.zeros((4,4))
  RT[:3,:3] = rot#*rot_aux
  RT_aux[:3,:3] = rot_aux#*rot_aux
  RT[:3,3] = trans
  RT[3,3] = 1.
  RT_aux[3,3] = 1.
  mesh.apply_transform(RT)
  mesh.apply_transform(RT_aux)
  print(mesh.is_watertight)
  # mesh.show()
  is_watertight = False
  while (is_watertight == False):
  #   print("repairing mesh...")
    is_watertight = trimesh.repair.fill_holes(mesh)
    
  mesh.export("rotated_mesh.obj", file_type='obj')
  new_mesh = trimesh.load("rotated_mesh.obj")
  new_mesh.show()
  
  
if __name__ == "__main__":
  trimesh.util.attach_to_log()
  plot_voxel(2)
  # mesh_to_voxel('/home/atabak/ycb_sample/009_gelatin_box/textured.obj')