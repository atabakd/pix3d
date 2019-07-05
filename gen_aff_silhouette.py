from __future__ import division, print_function
import numpy as np
import cv2
import os
import fnmatch


def get_silhouette(render_img):
  # render_img = cv2.imread(img_path)
  mean_img = render_img.mean(axis=-1, keepdims=True)
  sil_img = mean_img > 10
  new_sil_img = np.broadcast_to(sil_img, (480, 480, 3))
  return new_sil_img*255

if __name__ == "__main__":
  root_dir = "/mnt/hdd/aff_render/render/"
  out_dir = "/mnt/hdd/genre_test_aff/"
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  height, width, channel = 480, 640, 3
  half_diff_height_width = (width-height)//2
  for root, dirnames, filenames in os.walk("/mnt/hdd/aff_render/render"):
    for idx, filename in enumerate(fnmatch.filter(filenames, '*.png')):
      render_orig_img = cv2.imread(os.path.join(root, filename))
      img_tmp = np.zeros([width, width, channel], dtype=np.int)
      img_tmp[half_diff_height_width:-half_diff_height_width, ...] = render_orig_img
      render_resized_img = cv2.resize(render_orig_img, (480, 480), interpolation=cv2.INTER_AREA)
      cv2.imwrite(os.path.join(out_dir, '{}-'.format(idx) + filename.replace('.png', '_rgb.png')), render_resized_img)
      cv2.imwrite(os.path.join(out_dir, '{}-'.format(idx) + filename.replace('.png', '_silhouette.png')),
                  get_silhouette(render_resized_img))












