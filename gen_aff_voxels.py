from __future__ import division, print_function
import subprocess

def gen_voxels(num):
  com = 'python aff_voxel_demo.py --division_num {}'.format(num)
  subprocess.call([com], shell=True)

if __name__ == '__main__':

  from joblib import Parallel, delayed
  Parallel(n_jobs=11)(delayed(gen_voxels)(i) for i in range(0, 60))


#  gen_voxels(10)
