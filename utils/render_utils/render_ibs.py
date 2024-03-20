# blender packages
import logging
# built-in modules
import sys
import re
from itertools import product
from pathlib import Path
from typing import List

import bpy
# third-party packages
import numpy as np
from bpy.types import (
    Scene, Material, Object
)
from mathutils import Vector, Euler


def init_scene():
    """
    Initialize a scene with the basic rendering configurations.
    """
    # the bpy.context module is usually read-only, so we access the current scene through bpy.data
    scene_name: str = bpy.context.scene.name
    scene: Scene = bpy.data.scenes[scene_name]
    # scene.render.engine = 'BLENDER_EEVEE'
    scene.render.engine = 'CYCLES'
    # output image settings
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 2048
    scene.render.film_transparent = True  # transparent background
    # remove the default cube and lights created by blender
    for obj in bpy.data.objects:
        if obj.name != 'Camera':
            logging.info(f'remove object {obj.name} from the scene')
            bpy.data.objects.remove(obj)


def create_materials():
    """
    Create materials for rendering the input point cloud / output mesh
    """
    params = [
        {'name': 'mesh1', 'color': (1, 0.243, 0.243, 1.0), 'transparent': False},
        {'name': 'mesh2', 'color': (0.489, 0.794, 0.243, 1.0), 'transparent': False},
        {'name': 'ibs', 'color': (0.243, 0.489, 0.794, 1.0), 'transparent': True}
    ]
    global protected_material_names
    protected_material_names = [param['name'] for param in params]
    roughness = 0.5
    for param in params:
        # create a new material and enable nodes
        bpy.data.materials.new(name=param['name'])
        material: Material = bpy.data.materials[param['name']]
        material.use_nodes = True

        nodes: bpy_prop_collection = material.node_tree.nodes
        links: bpy_prop_collection = material.node_tree.links
        # remove the default Principle BSDF node in the material's node tree
        for node in nodes:
            if node.type != 'OUTPUT_MATERIAL':
                nodes.remove(node)
        # add a Diffuse BSDF node
        BSDF_node = nodes.new('ShaderNodeBsdfDiffuse')
        BSDF_node.inputs['Color'].default_value = param['color']
        BSDF_node.inputs['Roughness'].default_value = roughness
        output_node: ShaderNodeOutputMaterial = nodes['Material Output']
        if param['transparent']:
            # for a transparent material, create a Mix Shader node and enable color
            # blending
            transparent_node = nodes.new('ShaderNodeBsdfTransparent')
            mix_node = nodes.new('ShaderNodeMixShader')
            mix_node.inputs['Fac'].default_value = 0.3

            # here we have to use index instead of key to access the 'Shader' input
            # of a Mix Shader node, because there are two input slots with the same
            # name 'Shader' and we need to use both of them
            links.new(BSDF_node.outputs['BSDF'], mix_node.inputs[1])
            links.new(transparent_node.outputs['BSDF'], mix_node.inputs[2])
            links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

            material.blend_method = 'BLEND'
            material.shadow_method = 'CLIP'
        else:
            # for a non-transparent material, link the Diffuse BSDF node's output
            # with the output node's input
            links.new(BSDF_node.outputs['BSDF'], output_node.inputs['Surface'])

        logging.info('Diffuse BSDF material {} has been created'.format(param['name']))


def init_camera():
    """
    Set the camera's position
    """
    camera_obj: Object = bpy.data.objects['Camera']


def init_lights(scale_factor: float = 1):
    """
    Set lights for rendering.
    By default, this function will place
      - two sun lights above the object
      - one point light below the object
    The object is assumed to be normalized, i.e. it can be enclosed by a unit cube
    centered at (0, 0, 0).
    To render larger objects, pass the `scale_factor` parameter explicitly to scale
    the locations of lights.
    """
    # all parameters are obtained through blender GUI
    # the unit of angle is radians as blender API default setting
    params = [
        {
            'name': 'sun light 1', 'type': 'SUN',
            'location': np.array([3.638, 1.674, 4.329]), 'energy': 5.0, 'angle': 0.199
        },
        {
            'name': 'sun light 2', 'type': 'SUN',
            'location': np.array([0.449, -3.534, 1.797]), 'energy': 1.83, 'angle': 0.009
        },
        {
            'name': 'point light 1', 'type': 'POINT',
            'location': np.array([-2.163, -0.381, -2.685]), 'energy': 500
        }
    ]
    for param in params:
        light = bpy.data.lights.new(name=param['name'], type=param['type'])
        light.energy = param['energy']
        if param['type'] == 'SUN':
            light.angle = param['angle']
        light_obj = bpy.data.objects.new(name=param['name'], object_data=light)
        light_obj.location = param['location'] * scale_factor
        bpy.context.collection.objects.link(light_obj)


def track_object(obj: Object):
    """
    Let the camera track the specified object's center.
    By setting the tracking constraint, we can easily make the camera orient to
    the target object we want to render. This is less flexible but easier than
    setting the rotation manually.
    """
    camera: Object = bpy.data.objects['Camera']
    # the Track To constraint can keep the up direction of the camera better
    # than the Damp Track constraint, allowing placing the camera in the half-
    # space where x < 0
    camera.constraints.new('TRACK_TO')
    constraint = camera.constraints['Track To']
    constraint.target = obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'


def clear_imported_objects():
    """
    Remove the imported mesh and point cloud from the current scene, together
    with the materials automatically created by Blender when importing a mesh.
    """
    for obj in bpy.data.objects:
        # after application of geometry nodes, the point cloud data will also
        # be mesh
        if obj.type == 'MESH':
            logging.info(f'remove object {obj.name} from the current scene')
            bpy.data.objects.remove(obj)

    for material in bpy.data.materials:
        if material.name not in protected_material_names:
            logging.info(f'remove material {material.name}')
            bpy.data.materials.remove(material)


def launch_render(base_path: str, img_path: str, filename: str):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    init_scene()
    create_materials()
    init_lights()

    camera_obj: Object = bpy.data.objects['Camera']

    category_patten = "scene\\d"
    category = re.match(category_patten, filename).group()
    base_path = Path(base_path)
    img_path = Path(img_path)
    mesh_path = base_path / 'mesh' / category
    ibs_path = base_path / 'IBSMesh' / category

    mesh1_filename = '{}_0'.format(filename)
    mesh2_filename = '{}_1'.format(filename)
    ibs_filename = '{}'.format(filename)
    mesh1_file = mesh_path / '{}.obj'.format(mesh1_filename)
    mesh2_file = mesh_path / '{}.obj'.format(mesh2_filename)
    ibs_file = ibs_path / '{}.obj'.format(ibs_filename)

    bpy.ops.wm.obj_import(filepath=mesh1_file.as_posix())
    bpy.ops.wm.obj_import(filepath=mesh2_file.as_posix())
    bpy.ops.wm.obj_import(filepath=ibs_file.as_posix())

    mesh1 = bpy.data.objects[mesh1_filename]
    mesh2 = bpy.data.objects[mesh2_filename]
    ibs = bpy.data.objects[ibs_filename]

    mesh1.active_material = bpy.data.materials['mesh1']
    mesh2.active_material = bpy.data.materials['mesh2']
    ibs.active_material = bpy.data.materials['ibs']

    camera_obj.location = np.array([-1.22, -0.3, 0.6]) * 1
    track_object(mesh2)
    bpy.context.scene.render.filepath = (img_path / 'IBS' / '{}.png'.format(filename)).as_posix()
    bpy.ops.render.render(write_still=True)


# 使用命令行渲染，通过arg_parser解析参数
# if __name__ == '__main__':
#     import argparse
#     arg_parser = argparse.ArgumentParser(description="render pointcloud")
#     arg_parser.add_argument(
#         "--base_path",
#         "-b",
#         dest="base_path",
#         default="D:\\dataset\\IBPCDC",
#         required=False,
#     )
#     arg_parser.add_argument(
#         "--output_path",
#         "-o",
#         dest="output_path",
#         default="D:\\dataset\\IBPCDC\\render",
#         required=False,
#     )
#     arg_parser.add_argument(
#         "--filename",
#         "-f",
#         dest="filename",
#         required=True,
#     )
#
#     args = arg_parser.parse_args()
#
#     launch_render(args.base_path, args.output_path, args.filename)


# 在blender中渲染，不使用arg_parser
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    init_scene()
    create_materials()
    init_lights()
    init_camera()

    camera_obj: Object = bpy.data.objects['Camera']

    base_path = "D:\\dataset\\IBPCDC"
    img_path = "D:\\dataset\\IBPCDC\\render"
    filename = "scene2.1000"
    category_patten = "scene\\d"
    category = re.match(category_patten, filename).group()
    base_path = Path(base_path)
    img_path = Path(img_path)
    mesh_path = base_path / 'mesh' / category
    ibs_path = base_path / 'IBSMesh' / category

    mesh1_filename = '{}_0'.format(filename)
    mesh2_filename = '{}_1'.format(filename)
    ibs_filename = '{}'.format(filename)
    mesh1_file = mesh_path / '{}.obj'.format(mesh1_filename)
    mesh2_file = mesh_path / '{}.obj'.format(mesh2_filename)
    ibs_file = ibs_path / '{}.obj'.format(ibs_filename)
    #
    # bpy.ops.wm.obj_import(filepath=mesh1_file.as_posix())
    # bpy.ops.wm.obj_import(filepath=mesh2_file.as_posix())
    # bpy.ops.wm.obj_import(filepath=ibs_file.as_posix())
    bpy.ops.mesh.primitive_uv_sphere_add()
    bpy.ops.mesh.primitive_uv_sphere_add()

    sphere = bpy.data.objects["Sphere"]
    sphere1 = bpy.data.objects["Sphere.001"]

    sphere.active_material = bpy.data.materials['mesh1']
    sphere1.active_material = bpy.data.materials['mesh2']

    sphere.location[0] = 1
    sphere1.location[1] = 2

    sphere.scale[0] = 0.5
    sphere.scale[1] = 0.5
    sphere.scale[2] = 0.5

    # mesh1 = bpy.data.objects[mesh1_filename]
    # mesh2 = bpy.data.objects[mesh2_filename]
    # ibs = bpy.data.objects[ibs_filename]
    #
    # mesh1.active_material = bpy.data.materials['mesh1']
    # mesh2.active_material = bpy.data.materials['mesh2']
    # ibs.active_material = bpy.data.materials['ibs']
