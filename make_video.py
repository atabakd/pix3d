from __future__ import print_function, division

import cv2
import os
import numpy as np
import glob
import skimage
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import trimesh
from skimage.measure import block_reduce
import matplotlib.animation as manimation


def get_obj_dirs(obj_root):
  obj_names = os.listdir(obj_root)
  return [os.path.join(obj_root, obj_name) for obj_name in obj_names]


def get_samples(obj_path):
  fractions = os.listdir(obj_path)
  fraction = fractions[np.random.randint(len(fractions))]
  obj_frac_path = os.path.join(obj_path, fraction)
  options = [opt_name for opt_name in glob.glob('*_depth.png')]


def get_single_sample(obj_root):
  obj_names = os.listdir(obj_root)
  obj_dirs = [os.path.join(obj_root, obj_name) for obj_name in obj_names]
  rand_obj_dir = obj_dirs[np.random.randint(len(obj_dirs))]
  fractions = os.listdir(rand_obj_dir)
  fraction = fractions[np.random.randint(len(fractions))]
  obj_frac_path = os.path.join(rand_obj_dir, fraction)
  sample_choices = [opt_name for opt_name in
                    glob.glob(os.path.join(obj_frac_path, '*_depth.png'))]
  while len(sample_choices) == 0:
    rand_obj_dir = obj_dirs[np.random.randint(len(obj_dirs))]
    fractions = os.listdir(rand_obj_dir)
    fraction = fractions[np.random.randint(len(fractions))]
    obj_frac_path = os.path.join(rand_obj_dir, fraction)
    sample_choices = [opt_name for opt_name in
                    glob.glob(os.path.join(obj_frac_path, '*_depth.png'))]

  sample_depth = sample_choices[np.random.randint(len(sample_choices))]
  sample_normal = sample_depth.replace('depth', "normal")
  sample_rgb = sample_depth.replace('depth', "rgb")
  sample_spherical = sample_depth.replace('depth.png', "spherical.npz")
  sample_voxel = sample_depth.replace('depth.png', "voxel_rot.npz")
  sample_dict = dict()
  sample_dict["depth"] = skimage.io.imread(sample_depth, as_gray=True)
  sample_dict["rgb"] = cv2.imread(sample_rgb)
  sample_dict["normal"] = cv2.imread(sample_normal)
  sample_dict["sph"] = np.load(sample_spherical)["obj_spherical"]
  sample_dict["sph_depth"] = \
              np.load(sample_spherical)["depth_spherical_centered"]
  sample_dict["voxel"] = np.load(sample_voxel)["voxel"]
  return sample_dict





if __name__== "__main__":
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata = dict(title='Movie Test', artist='Matplotlib',
                                  comment='Movie support!')
  writer = FFMpegWriter(fps=5, metadata=metadata)

  fig= plt.figure()
  w_in_inches = 19.2
  h_in_inches = 10.8
  fig.set_size_inches(w_in_inches, h_in_inches, True)
  with writer.saving(fig, "writer_test.mp4", dpi=100):
    for i in range(100):
      print(i)
      while True:
        try:
          sample_dict = get_single_sample(os.path.join("/mnt", "hamming"))
          break
        except PermissionError:
          print("permission problem!")

      axes11 = plt.subplot(231)
      axes11.set_title("colour image")
      axes12 = plt.subplot(232)
      axes12.set_title("rendered depth")
      axes13 = plt.subplot(233)
      axes13.set_title("rendered normals")
      axes21 = plt.subplot(234)
      axes21.set_title("full spherical map")
      axes22 = plt.subplot(235)
      axes22.set_title("partial spherical map")
      axes23 = plt.subplot(236, projection='3d')
      axes23.set_title("estimated for simulation")
      axes11.imshow(cv2.cvtColor(sample_dict["rgb"], cv2.COLOR_BGR2RGB))
      axes12.imshow(sample_dict["depth"], cmap="gray")
      axes13.imshow(sample_dict["normal"])
      axes21.imshow(sample_dict["sph"])
      axes22.imshow(sample_dict["sph_depth"])

      ds_voxel = block_reduce(sample_dict["voxel"], (2, 2, 2))
      #ax = axes[1,2].projection('3d')
      axes23.set_xlabel("x")
      axes23.set_ylabel("y")
      axes23.set_zlabel("z")
      axes23.set_aspect('equal')

      # ax.voxels(voxel, edgecolor="k")
      axes23.voxels(ds_voxel , edgecolor="k", facecolors=[1, 0, 0, 0.05])
      # ax.view_init(90, 270)
      axes23.view_init(0, 180)
      plt.draw()
      writer.grab_frame()
