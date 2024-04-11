# blender packages
import logging
# built-in modules
import sys
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
    scene.render.resolution_y = 1024
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
        {'name': 'pointcloud1', 'color': (0.165, 0.564, 0.921, 1.0), 'transparent': False},
        {'name': 'pointcloud2', 'color': (0.165, 0.921, 0.564, 1.0), 'transparent': False},
        {'name': 'mesh', 'color': (0.794, 0.489, 0.243, 1.0), 'transparent': True}
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
            mix_node.inputs['Fac'].default_value = 0.5

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
    # the location is obtained through GUI
    camera_obj.location = (0.7359, -0.6926, 0.4958)


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


def create_pointcloud1_modifier():
    """
    Create the geometry nodes as a modifier for point clouds.
    This modifier will expand each point to a ico sphere for rendering.
    """
    # create a node group and enable it as a geometry modifier
    geom_nodes = bpy.data.node_groups.new('pointcloud1 modifier', 'GeometryNodeTree')
    geom_nodes.is_modifier = True
    nodes = geom_nodes.nodes
    links = geom_nodes.links
    interface = geom_nodes.interface
    interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # create all node with their properties set
    input_node = nodes.new('NodeGroupInput')
    output_node = nodes.new('NodeGroupOutput')
    mesh_to_points_node = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.mode = 'VERTICES'
    ico_sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere_node.inputs['Radius'].default_value = 0.005
    ico_sphere_node.inputs['Subdivisions'].default_value = 3  # control the smoothness of the ico sphere
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    material_node = nodes.new('GeometryNodeReplaceMaterial')
    # only set the New slot of the Replace Material node because we actually
    # use it to set the material of output instances (spheres), the Old slot
    # is not used.
    material_node.inputs['New'].default_value = bpy.data.materials['pointcloud1']

    # link the nodes
    links.new(input_node.outputs['Geometry'], mesh_to_points_node.inputs['Mesh'])
    # the PLY file are imported as mesh, so we need to replace each vertex in
    # the mesh with a point, then we will have a real point cloud in Blender
    links.new(mesh_to_points_node.outputs['Points'], instance_node.inputs['Points'])
    # use the pre-defined ico sphere as the template instance. with the
    # Instance On Points node we can instantiate an instance at each point in
    # the point cloud
    links.new(ico_sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], material_node.inputs['Geometry'])
    links.new(material_node.outputs['Geometry'], output_node.inputs['Geometry'])


def create_pointcloud2_modifier():
    """
    Create the geometry nodes as a modifier for point clouds.
    This modifier will expand each point to a ico sphere for rendering.
    """
    # create a node group and enable it as a geometry modifier
    geom_nodes = bpy.data.node_groups.new('pointcloud2 modifier', 'GeometryNodeTree')
    geom_nodes.is_modifier = True
    nodes = geom_nodes.nodes
    links = geom_nodes.links
    interface = geom_nodes.interface
    interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # create all node with their properties set
    input_node = nodes.new('NodeGroupInput')
    output_node = nodes.new('NodeGroupOutput')
    mesh_to_points_node = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.mode = 'VERTICES'
    ico_sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere_node.inputs['Radius'].default_value = 0.005
    ico_sphere_node.inputs['Subdivisions'].default_value = 3  # control the smoothness of the ico sphere
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    material_node = nodes.new('GeometryNodeReplaceMaterial')
    # only set the New slot of the Replace Material node because we actually
    # use it to set the material of output instances (spheres), the Old slot
    # is not used.
    material_node.inputs['New'].default_value = bpy.data.materials['pointcloud2']

    # link the nodes
    links.new(input_node.outputs['Geometry'], mesh_to_points_node.inputs['Mesh'])
    # the PLY file are imported as mesh, so we need to replace each vertex in
    # the mesh with a point, then we will have a real point cloud in Blender
    links.new(mesh_to_points_node.outputs['Points'], instance_node.inputs['Points'])
    # use the pre-defined ico sphere as the template instance. with the
    # Instance On Points node we can instantiate an instance at each point in
    # the point cloud
    links.new(ico_sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], material_node.inputs['Geometry'])
    links.new(material_node.outputs['Geometry'], output_node.inputs['Geometry'])


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    init_scene()
    create_materials()
    init_lights()
    create_pointcloud1_modifier()
    create_pointcloud2_modifier()

    # pcd_path = Path('/home/greyishsong/workspace/osp-output')
    # mesh_path = pcd_path / 'obj'
    # pointcloud_path = Path('/home/greyishsong/workspace/scps-output/ply')
    # img_path = pcd_path / 'misleaded'
    #
    camera_obj: Object = bpy.data.objects['Camera']
    # the location is obtained through GUI
    camera_location = np.array([0.7359, -0.6926, 0.4958]) * 1.2
    #
    #
    # with open(pcd_path / 'problem-misleaded.txt', 'r') as f:
    #     object_indices = f.readlines()
    # object_indices: List[str] = [index.strip() for index in object_indices if index.strip() != '']

    base_path = Path(r"D:\dataset\IBPCDC\pcdPred\PointAttN_INTE_mads05_madi100_ibsa005_lr1e4\scene1")
    img_path = Path(r"D:\dataset\IBPCDC\render\PointAttN_INTE_mads05_madi100_ibsa005_lr1e4\scene1.1004_view0")
    ply_file1 = base_path / "scene1.1004_view0_0.ply"
    ply_file2 = base_path / "scene1.1004_view0_1.ply"

    bpy.ops.wm.ply_import(filepath=ply_file1.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
    bpy.ops.wm.ply_import(filepath=ply_file2.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')

    pointcloud = bpy.data.objects['scene1.1004_view0_0']
    modifier = pointcloud.modifiers.new('modifier', 'NODES')
    modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']

    pointcloud = bpy.data.objects['scene1.1004_view0_1']
    modifier = pointcloud.modifiers.new('modifier', 'NODES')
    modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    for view_index, sign in enumerate(product(np.array([1, -1]), repeat=3)):
        camera_obj.location = camera_location * sign
        track_object(pointcloud)
        bpy.context.scene.render.filepath = (img_path / f'{view_index}.png').as_posix()
        bpy.ops.render.render(write_still=True)

    # for obj_file in data_path.iterdir():
    # object_index = obj_file.stem
    # for object_index in object_indices:
    #     obj_file = mesh_path / f'{object_index}.obj'
    #     ply_file = pointcloud_path / f'{object_index}.ply'
    #     if not obj_file.exists():
    #         continue
    #     logging.info(f'start rendering object {object_index}')
    #     bpy.ops.wm.obj_import(filepath=obj_file.as_posix())
    #     mesh = bpy.data.objects[object_index]
    #     mesh.active_material = bpy.data.materials['mesh']
    #
    #     bpy.ops.wm.ply_import(filepath=ply_file.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
    #     pointcloud = bpy.data.objects[f'{object_index}.001']
    #     modifier = pointcloud.modifiers.new('modifier', 'NODES')
    #     modifier.node_group = bpy.data.node_groups['pointcloud modifier']
    #
    #     for view_index, sign in enumerate(product(np.array([1, -1]), repeat=3)):
    #         camera_obj.location = camera_location * sign
    #         track_object(mesh)
    #         bpy.context.scene.render.filepath = (img_path / f'{object_index}-{view_index}.png').as_posix()
    #         bpy.ops.render.render(write_still=True)
    #     clear_imported_objects()
    #     logging.info('done')
