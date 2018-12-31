from __future__ import division, print_function
import json
import subprocess
# import bpy

def gen_render_for_division(division_num):
  with open('{:04d}.json'.format(division_num), 'r') as j_file:
    data_list = json.load(j_file)
  for i in range(len(data_list)):
  # for i in range(2):
    # call([' ../blender-2.79b-linux-glibc219-x86_64/blender', '--background', '--verbose',
    #       '0', '--python', 'ycb_demp.py', '--',
    #       '--anno_idx', str(i), '--division_num', str(division_num)], shell=True)

    # command = (
    #   bpy.app.binary_path_python, '/media/Extra/dsl/pix3d/ycb_demo.py',
    #         '--anno_idx', str(i), '--division_num', str(division_num)
    # )
    # proc = subprocess.Popen(command)
    # while proc.poll() is None:
    #   print("Waiting...")
    #   time.sleep(0.1)
    com = ' ../blender-2.79b-linux-glibc219-x86_64/blender --background --verbose 0 --python ycb_demo.py -- --anno_idx ' + \
    str(i) + ' --division_num ' + str(division_num)
    # p = subprocess.Popen([com], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # out, err = p.communicate()
    subprocess.call([com], shell=True)
    # import pdb;pdb.set_trace()

if __name__ == '__main__':
  # gen_render_for_division(0)

  from joblib import Parallel, delayed
  Parallel(n_jobs=6)(delayed(gen_render_for_division)(i) for i in range(60))