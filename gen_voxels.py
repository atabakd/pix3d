from __future__ import division, print_function
import subprocess

def gen_voxels(num):
  com = 'python play_with_voxel.py --division_num {}'.format(num)
  subprocess.call([com], shell=True)

if __name__ == '__main__':

  from joblib import Parallel, delayed
  Parallel(n_jobs=12)(delayed(gen_voxels)(i) for i in range(60))


#  gen_voxels(10)
