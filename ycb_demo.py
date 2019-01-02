from __future__ import print_function, absolute_import, division

from os import makedirs
from os.path import dirname, exists, join, basename
import sys
import json
import argparse
import numpy as np
import bpy
from mathutils import Matrix
import mathutils


def set_cycles(w=None, h=None, n_samples=None):
  scene = bpy.context.scene
  scene.render.engine = 'CYCLES'
  cycles = scene.cycles

  cycles.use_progressive_refine = True
  if n_samples is not None:
    cycles.samples = n_samples
  cycles.max_bounces = 100
  cycles.min_bounces = 10
  cycles.caustics_reflective = False
  cycles.caustics_refractive = False
  cycles.diffuse_bounces = 10
  cycles.glossy_bounces = 4
  cycles.transmission_bounces = 4
  cycles.volume_bounces = 0
  cycles.transparent_min_bounces = 8
  cycles.transparent_max_bounces = 64

  # Avoid grainy renderings (fireflies)
  world = bpy.data.worlds['World']
  world.cycles.sample_as_light = True
  cycles.blur_glossy = 5
  cycles.sample_clamp_indirect = 5

  # Ensure no background node
  world.use_nodes = True
  try:
    world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
  except KeyError:
    pass

  scene.render.tile_x = 16
  scene.render.tile_y = 16
  if w is not None:
    scene.render.resolution_x = w
  if h is not None:
    scene.render.resolution_y = h
  scene.render.resolution_percentage = 100
  scene.render.use_file_extension = True
  scene.render.image_settings.file_format = 'PNG'
  scene.render.image_settings.color_mode = 'RGBA'
  scene.render.image_settings.color_depth = '8'
  cycles.device = "GPU"

  for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

def add_object(model_path, rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
               trans_vec=(0, 0, 0), scale=1, name=None):
  # Import
  if model_path.endswith('.obj'):
    bpy.ops.import_scene.obj(filepath=model_path, axis_forward='-Z', axis_up='Y')
  else:
    raise NotImplementedError("Importing model of this type")

  obj_list = []
  for i, obj in enumerate(bpy.context.selected_objects):
    # Rename
    if name is not None:
      if len(bpy.context.selected_objects) == 1:
        obj.name = name
      else:
        obj.name = name + '_' + str(i)

    # Compute world matrix
    # trans_4x4 = Matrix.Translation(trans_vec)
    trans_4x4 = Matrix.Translation(trans_vec)
    rot_4x4 = Matrix(rot_mat).to_4x4()
    scale_4x4 = Matrix(np.eye(4))  # don't scale here

    #rot_aux=Matrix(rot_aux).to_4x4()
    # intrinsics=1.0e+03 * Matrix([[1.0668,0,0.3130],[0,1.0675,0.2413],[0,0,0.0010]])
    obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4
    rot_aux = ((1.0, 0.0, 0.0),
               (0.0, -1.0, 0.0),
               (0.0, 0.0, -1.0))

    rot_aux = Matrix(rot_aux).to_4x4()
    obj.matrix_world = rot_aux * obj.matrix_world

    obj.scale = (scale, scale, scale)

    obj_list.append(obj)

  if len(obj_list) == 1:
    return obj_list[0]
  else:
    return obj_list


def add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None,
               proj_model='PERSP', f=35, sensor_fit='HORIZONTAL',
               sensor_width=22.118):
  bpy.ops.object.camera_add()
  cam = bpy.context.active_object

  if name is not None:
    cam.name = name

  cam.location = xyz
  cam.rotation_euler = rot_vec_rad

  cam.data.type = proj_model
  cam.data.lens = f
  cam.data.sensor_fit = sensor_fit
  cam.data.sensor_width = sensor_width
  cam.rotation_mode = "YZX"

  return cam


def render_to_file(outpath):
  outdir = dirname(outpath)
  if not exists(outdir):
    makedirs(outdir)

  # Set active camera, just in case
  for o in bpy.data.objects:
    if o.type == 'CAMERA':
      bpy.context.scene.camera = o
      break

  # Render
  bpy.context.scene.render.filepath = outpath
  bpy.ops.render.render(write_still=True)


def render(data, output_path, division_num):
  # w, h = data['img_size']
  w, h = 640, 480
  set_cycles(w=w, h=h, n_samples=50)

  # Remove all default objects
  for obj in bpy.data.objects:
    obj.select = True
  bpy.ops.object.delete()

  # Object
  obj = add_object(data['model'], data['rot_mat'],
                   data['trans_mat'], name='object')

  # Lighting
  world = bpy.data.worlds['World']
  world.light_settings.use_ambient_occlusion = True
  world.light_settings.ao_factor = 0.9
  data['focal_length'] = 35.27039762259147

  # Camera
  camera = add_camera(name='camera', proj_model='PERSP',
                      f=data['focal_length'], sensor_fit='HORIZONTAL')


  camera.data.clip_end = 1e10

  # https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree
  links = tree.links

  # Add passes for additionally dumping albedo and normals.
  bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
  bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True

  render_layers = tree.nodes.new('CompositorNodeRLayers')

  depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
  depth_file_output.label = 'Depth Output'
  # Remap as other types can not represent the full range of depth.
  map = tree.nodes.new(type="CompositorNodeMapValue")
  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
  map.offset = [-0.7]
  map.size = [0.5]
  map.use_min = True
  map.min = [0]
  links.new(render_layers.outputs['Depth'], map.inputs[0])
  links.new(map.outputs[0], depth_file_output.inputs[0])

  scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
  scale_normal.blend_type = 'MULTIPLY'
  # scale_normal.use_alpha = True
  scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
  links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

  bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
  bias_normal.blend_type = 'ADD'
  # bias_normal.use_alpha = True
  bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
  links.new(scale_normal.outputs[0], bias_normal.inputs[1])

  normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
  normal_file_output.label = 'Normal Output'
  links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

  albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
  albedo_file_output.label = 'Albedo Output'
  links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

  for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
    output_node.base_path = ''

  scene = bpy.context.scene
  # scene.render.filepath = "./render/" + output_path + "{:04d}".format(division_num) + ".png"
  render_dir = join(output_path, "render", "{:04d}".format(division_num))
  if not exists(render_dir):
    makedirs(render_dir)

  render_name = basename(data["mask"]).replace("mask", "render")
  scene.render.filepath = join(render_dir, render_name)
  print(output_path)
  # depth_file_output.file_slots[0].path = "./depth/" + output_path + "{:04d}".format(division_num) + "#"
  depth_dir = join(output_path, "depth", "{:04d}".format(division_num))
  if not exists(depth_dir):
    makedirs(depth_dir)

  depth_name = basename(data["mask"]).replace("mask", "depth").split(".")[0]
  depth_file_output.file_slots[0].path = join(depth_dir, depth_name) + "#"



  # normal_file_output.file_slots[0].path = "./normals/" + output_path + "{:04d}".format(division_num) + "#"
  normal_dir = join(output_path, "normal", "{:04d}".format(division_num))
  if not exists(normal_dir):
    makedirs(normal_dir)

  normal_name = basename(data["mask"]).replace("mask", "normal").split(".")[0]
  normal_file_output.file_slots[0].path = join(normal_dir, normal_name) + "#"
  # import pdb;pdb.set_trace()

  # albedo_file_output.file_slots[0].path = "./RGB/" + output_path + "#"
  # albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo.png"

  for o in bpy.data.objects:
    if o.type == 'CAMERA':
      bpy.context.scene.camera = o
      break

  bpy.ops.render.render(write_still=True)  # render still

  # render_to_file(output_path)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--anno_idx', type=int, default=0,
                      help='index of annotion')
  parser.add_argument('--division_num', type=int, default=0,
                      help='division number 0<=division_num<60')
  parser.add_argument('--output_path', type=str, default='/media/hdd/YCBvideo/YCB_Video_Dataset/Generated_YCB_Video_Dataset',
                      help='output image path')
  if '--' not in sys.argv:
    argv = []
  else:
    argv = sys.argv[sys.argv.index('--') + 1:]
  args = parser.parse_args(argv)

  with open('{:04d}.json'.format(args.division_num), "r") as j_file:
    data_list = json.load(j_file)


  render(data_list[args.anno_idx], args.output_path, args.division_num)
  #
  # data_list = json.load(open('mug.json'))
  # render(data_list[args.anno_idx], args.output_path)
  # print('Original Image:', data_list[args.anno_idx]['img'])
  # print('Saved to:', args.output_path)
  # import gen_mask_and_json as gsif
  #
  # obj_pose_dict = gsif.rt_from_label("/media/hdd/YCBvideo/YCB_Video_Dataset/YCB_Video_Dataset/data/0000/000001-label.png")
  # data = dict()
  # data['img_size'] = (640, 480)
  # data['model'] = '/media/hdd/YCBvideo/YCB_Video_Models/models/009_gelatin_box/textured.obj'
  # data['rot_mat'] = obj_pose_dict['8'][:,:-1]
  # data['trans_mat'] = obj_pose_dict['8'][:, -1]
  # data['focal_length'] = 35.27039762259147
  # render(data, 'test')
  # data_list = json.load(open('0000.json'))
  # with open('0000.json', "r") as j_file:
  #   data_list = json.load(j_file)
  # output_path = "./test"
  # render(data_list[0], output_path)