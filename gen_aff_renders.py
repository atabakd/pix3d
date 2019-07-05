from __future__ import division, print_function
import json
import subprocess
import os
import numpy as np
# import bpy

def gen_render_for_division(gen_idx):
  aff_objects_root_dir = '/mnt/hdd/Affordance_Models'
  aff_objects = os.listdir(aff_objects_root_dir)
  aff_objects_full_path = list()
  for aff_object in aff_objects:
    aff_object_full_path = os.path.join(aff_objects_root_dir, aff_object, aff_object.split('google')[0][:-1], 'google_512k', 'textured.obj')
    aff_objects_full_path.append(aff_object_full_path)

  model_path = aff_objects_full_path[gen_idx % 11]
  rand_division = np.random.randint(0, 60)
  rand_idx = np.random.randint(0, 1e3)  # randomly only looking at the first 1000 instances


  com = ' ../blender-2.79b-linux-glibc219-x86_64/blender --background --verbose 0 --python aff_demo.py -- --anno_idx ' + \
  str(rand_idx) + ' --division_num ' + str(rand_division) + ' --model_path ' + model_path
  # p = subprocess.Popen([com], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  # out, err = p.communicate()
  subprocess.call([com], shell=True)
  # import pdb;pdb.set_trace()

if __name__ == '__main__':
  #gen_render_for_division(23)

  from joblib import Parallel, delayed
  Parallel(n_jobs=4)(delayed(gen_render_for_division)(i) for i in range(110))
