from __future__ import division, print_function

import cv2
import numpy as np
from scipy.io import loadmat
from skimage import io
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


# dep = cv2.imread("depth/0000/000001-depth-001.png", 0)
# cv2.imwrite("depth.png", 255-dep)

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
  from skimage.transform import rescale, resize, downscale_local_mean
  # depth = resize(depth, (640/4, 480/4))

  cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
  # cx, cy = -9.13, 2.79
  fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
  rows, cols = depth.shape
  c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
  valid = (depth >= 0) & (depth <= np.iinfo(np.uint16).max)
  z = np.where(valid, depth*1.0 / depth_factor, np.nan)
  x = np.where(valid, z * (c - cx) / fx, 0)
  y = np.where(valid, z * (r - cy) / fy, 0)
  return np.dstack((x, y, z))


# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
def project_to_rgb(xyz, rgb_intrinsics, rotation_matrix, translation_vec):
  rotation_vec = cv2.Rodrigues(rotation_matrix)[0]
  img = cv2.projectPoints(xyz.reshape(-1, 3), rotation_vec, translation_vec, rgb_intrinsics, None)[0].squeeze(
    1)  # .reshape(160,120, -1)
  xi = np.arange(320, 960)
  yi = np.arange(240, 720)
  xi, yi = np.meshgrid(xi, yi)
  fixed_depth = griddata(img, point_3d[..., 2].reshape(-1, 1), (xi, yi), method='linear')
  return fixed_depth



if __name__ == "__main__":
  # from mpl_toolkits.mplot3d import Axes3D

  depth_path = '/home/atabak/pix3d/for_rui/for_rui/000001-depth.png'
  depth_img = io.imread(depth_path).astype(np.float)
  depth_intrinsics = np.array([[567.6188, 0, 310.0724], [0, 568.1618, 242.7912], [0, 0, 1.]])
  meta = label_mat = loadmat(depth_path.replace("depth.png", "meta.mat"))
  point_3d = depth_to_3d(depth_img, depth_intrinsics, meta['factor_depth'])
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # for i in range(480):
  #   for j in range(640):
  # filter_idxs = np.where(np.logical_and(point_3d[..., 2].reshape(-1, 1)<0.6,point_3d[..., 2].reshape(-1, 1)>0.4))[0]
  # x = point_3d[..., 0].reshape(-1, 1)[filter_idxs]
  # y = point_3d[..., 1].reshape(-1, 1)[filter_idxs]
  # z = point_3d[..., 2].reshape(-1, 1)[filter_idxs]
  # ax.scatter(x, y, z, alpha=0.1)
  # ax.set_xlabel('X Label')
  # ax.set_ylabel('Y Label')
  # ax.set_zlabel('Z Label')
  # plt.show()
  rgb_intrinsics = np.array([[1066.778*0.5, 0, 320+312.9869], [0, 1067.487*0.5, 240+241.3109], [0., 0, 1]])
  rotation_matrix = np.array([[0.9997563, 0.02131301, -0.005761033],
                              [-0.02132165, 0.9997716, -0.001442874],
                              [0.005728965, 0.001565357, 0.9999824]])
  trans_vec = np.array([0.02627148, -0.0001685539, 0.0002760285])
  zi = project_to_rgb(point_3d, rgb_intrinsics, rotation_matrix, trans_vec)
  
  plt.imshow(zi.squeeze())
  plt.show()


