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
      depth_max_white_hole = np.where(depth_mask <= 0.05, np.iinfo(np.uint16).max, depth_mask)
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


def rt_from_label(label_path, label_mat):
  # label_mat = loadmat(label_path.replace("label.png", "meta.mat"))
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
      
  if int(folder_name) < 60:
    depth_intrinsics = np.array([[567.6188, 0, 310.0724], [0, 568.1618, 242.7912], [0, 0, 1.]])
    rgb_intrinsics = np.array([[1066.778 * 0.5, 0, 320 + 312.9869], [0, 1067.487 * 0.5, 240 + 241.3109], [0., 0, 1]])
    rotation_depth2rgb = np.array([[0.9997563, 0.02131301, -0.005761033],
                                  [-0.02132165, 0.9997716, -0.001442874],
                                  [0.005728965, 0.001565357, 0.9999824]])
    trans_depth2rgb = np.array([0.02627148, -0.0001685539, 0.0002760285])
  else:
    depth_intrinsics = np.array([[576.3624, 0, 319.2682], [0, 576.7067, 243.8256], [0, 0, 1.]])
    rgb_intrinsics = np.array([[1077.836 * 0.5, 0, 320 + 323.7872], [0, 1078.189 * 0.5, 240 + 279.6921], [0., 0, 1]])
    rotation_depth2rgb = np.array([[0.9999962, -0.002468782, 0.001219765],
                                  [0.002466791, 0.9999956, 0.001631345],
                                  [0.001223787, 0.00162833, 0.9999979]])
    trans_depth2rgb = np.array([0.02640966, -9.9086e-05, 0.0002972445])
    

    num_files = len(glob.glob("/media/hdd/YCBvideo/YCB_Video_Dataset/YCB_Video_Dataset/data/" + folder_name + "/*-label.png"))
    for label_num in range(1, num_files+1):
      label_path = os.path.join(base, "YCB_Video_Dataset/YCB_Video_Dataset/data", folder_name, "{:06d}-label.png".format(label_num))
      label_mat = loadmat(label_path.replace("label.png", "meta.mat"))
      obj_pose_dict = rt_from_label(label_path, label_mat)
      for obj in obj_pose_dict.keys():
        obj_dict = OrderedDict()
        obj_dict["img"] = label_path
        obj_dict["mask"] = obj_pose_dict[obj][1]
        obj_dict["depth_minmax"] = obj_pose_dict[obj][2]
        obj_dict["model"] = "/media/hdd/YCBvideo/YCB_Video_Dataset/YCB_Video_Dataset/models/" + \
                            classes[obj - 1] + "/textured.obj"
        obj_dict["rot_mat"] = obj_pose_dict[obj][0][:, :-1].squeeze(2).tolist()
        obj_dict["trans_mat"] = obj_pose_dict[obj][0][:, -1].squeeze(1).tolist()
        obj_dict["focal_length"] = label_mat["focal_length"]
        data.append(obj_dict)

    json.dump(data, j_file, indent=2)



if __name__ == "__main__":
  from joblib import Parallel, delayed
  Parallel(n_jobs=6)(delayed(dump_json)("{:04d}".format(i)) for i in range(60))
  # for i in range(60):
  #   dump_json("{:04d}".format(i))
  # from joblib import Parallel, delayed
  # Parallel(n_jobs=6)(delayed(dump_json)("{:04d}".format(i)) for i in range(60))
  # for i in range(60):
  #   write_mask("{:04d}".format(i))
