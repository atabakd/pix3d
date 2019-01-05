from __future__ import division, print_function
import numpy as np
import cv2
from scipy.io import loadmat
import json
from collections import OrderedDict
import os
import glob
from skimage import io

def get_unoccluded(label_path, distances, factor_depth, save_mask_img = False):
  label_img = cv2.imread(label_path, 0)
  unique_objs = np.unique(label_img)
  unoccluded_objs = list()
  masks_path = list()
  minmax_vals = list()
  kernel = np.ones((3, 3), np.uint8)
  idx = 0
  for obj in unique_objs[1:]:  # skip background by [1:]
    occluded = False
    obj_mask = label_img == obj
    obj_mask_dilated = cv2.dilate(obj_mask.astype(np.uint8), kernel)
    for test_obj in unique_objs[1:]:
      if occluded: continue # once occluded, always occluded
      if test_obj == obj: continue  # skip self occlusion
      test_obj_mask = label_img == test_obj
      if np.sum(np.logical_and(test_obj_mask, obj_mask_dilated)) > 1:
        if distances[obj] > distances[test_obj]:
          occluded = True

    if not occluded:
      unoccluded_objs.append(obj)
      masked_img_dir = os.path.dirname(
      label_path.replace("YCB_Video_Dataset/YCB_Video_Dataset", "YCB_Video_Dataset/Generated_YCB_Video_Dataset").replace("data", "mask"))
      label_filename = os.path.basename(label_path)
      mask_path = os.path.join(masked_img_dir,
                               label_filename.replace(".png", "-{:02d}.png".format(idx)).replace("label", "mask"))
      masks_path.append(mask_path)
      depth_img = io.imread(label_path.replace("label", "depth")).astype(np.float)
      depth_img /= factor_depth
      depth_mask = np.where(obj_mask, depth_img, 0.)
      depth_max_white_hole = np.where(depth_mask==0., np.iinfo(np.uint16).max, depth_mask)
      minmax_vals.append((depth_max_white_hole.min(), depth_mask.max()))

      idx += 1
      if save_mask_img:
        real_img = cv2.imread(label_path.replace("label", "color"))
        masked_img = np.where(obj_mask[..., np.newaxis], real_img, 0)
        if not os.path.exists(masked_img_dir):
          os.makedirs(masked_img_dir)

        cv2.imwrite(mask_path, masked_img)
        bin_mask_path = mask_path.replace("mask", "bin_mask")
        if not os.path.exists(os.path.dirname(bin_mask_path)):
          os.makedirs(os.path.dirname(bin_mask_path))


        cv2.imwrite(bin_mask_path, obj_mask.astype(np.uint8)*255)


  return unoccluded_objs, masks_path, minmax_vals


def rt_from_label(label_path):
  label_mat = loadmat(label_path.replace("label.png", "meta.mat"))
  distances = {k[0]: v[2, 3] for k, v in zip(label_mat['cls_indexes'], label_mat['poses'].transpose(2, 0, 1))}
  unoccluded_objs, masks_path, minmax_vals = get_unoccluded(label_path, distances, label_mat['factor_depth'])
  obj_pose_dict = dict()
  for obj, mask_path, minmax_val in zip(unoccluded_objs, masks_path, minmax_vals):
    obj_ind = np.where(label_mat['cls_indexes'] == obj)[0]
    obj_pose_dict[obj] = (label_mat['poses'][..., obj_ind], mask_path, minmax_val)

  return obj_pose_dict


def dump_json(folder_name):
  base = "/media/hdd/YCBvideo"
  classes = np.loadtxt(os.path.join(base, "YCB_Video_toolbox/classes.txt"), dtype=object)
  with open(folder_name+".json", "w+") as j_file:
    try:
      data = json.load(j_file)
    except ValueError:
      data = list()

    num_files = len(glob.glob("/media/hdd/YCBvideo/YCB_Video_Dataset/YCB_Video_Dataset/data/" + folder_name + "/*-label.png"))
    for label_num in range(1, num_files+1):
      label_path = os.path.join(base, "YCB_Video_Dataset/YCB_Video_Dataset/data", folder_name, "{:06d}-label.png".format(label_num))
      obj_pose_dict = rt_from_label(label_path)
      for obj in obj_pose_dict.keys():
        obj_dict = OrderedDict()
        obj_dict["img"] = label_path
        obj_dict["mask"] = obj_pose_dict[obj][1]
        obj_dict["depth_minmax"] = obj_pose_dict[obj][2]
        obj_dict["model"] = "/media/hdd/YCBvideo/YCB_Video_Dataset/YCB_Video_Dataset/models/" + \
                            classes[obj - 1] + "/textured.obj"
        obj_dict["rot_mat"] = obj_pose_dict[obj][0][:, :-1].squeeze(2).tolist()
        obj_dict["trans_mat"] = obj_pose_dict[obj][0][:, -1].squeeze(1).tolist()
        data.append(obj_dict)

    json.dump(data, j_file, indent=2)


# https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
def depth_to_3d(depth, depth_intrinsics, depth_factor):
  """Transform a uint16 depth image into a point cloud with one point for each
  pixel in the image, using the camera transform for a camera
  centred at cx, cy with field of view fx, fy.

  depth is a 2-D ndarray with shape (rows, cols) containing
  depths in uint16. The result is a 3-D array with
  shape (rows, cols, 3). Pixels with invalid depth in the input have
  NaN for the z-coordinate in the result.

  """
  cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
  fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
  rows, cols = depth.shape
  c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
  valid = (depth >= 0) & (depth <= np.iinfo(np.uint16).max)
  z = np.where(valid, depth*1.0 / depth_factor, np.nan)
  x = np.where(valid, z * (c - cx) / fx, 0)
  y = np.where(valid, z * (r - cy) / fy, 0)
  return np.dstack((x, y, z))


# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
def xyz_2_uv(xyz, rgb_intrinsics, RT):
  """Projects 3d points , using the camera transform for a camera
  centred at cx, cy with field of view fx, fy.

  depth is a 2-D ndarray with shape (rows, cols) containing
  depths from 1 to 254 inclusive. The result is a 3-D array with
  shape (rows, cols, 3). Pixels with invalid depth in the input have
  NaN for the z-coordinate in the result.

  """
  fundamental_matrix = rgb_intrinsics * RT
  rows, cols = xyz.shape[:-1]
  xyz1 = xyz[..., np.newaxis]
  xyz1[..., -1] = 1.
  uv1 = np.ones_like(xyz)
  for r in range(rows):
    for c in range(cols):
      uv1[r,c,:] = fundamental_matrix * xyz1[r, c, ...]

  return uv1[..., :-1]

  xyz1 = xyz[..., np.newaxis]
  xyz1[..., -1] = 1.
  uv1 = fundamental_matrix * np.transpose(xyz1, [2, 0, 1])




if __name__ == "__main__":
  from joblib import Parallel, delayed
  Parallel(n_jobs=6)(delayed(dump_json)("{:04d}".format(i)) for i in range(60))
  # for i in range(60):
  #   dump_json("{:04d}".format(i))
  # from joblib import Parallel, delayed
  # Parallel(n_jobs=6)(delayed(dump_json)("{:04d}".format(i)) for i in range(60))
  # for i in range(60):
  #   write_mask("{:04d}".format(i))
