# blender packages
import logging
import os.path
# built-in modules
import sys
from itertools import product
from pathlib import Path

import bpy
# third-party packages
import numpy as np
from bpy.types import (
    Scene, Material, Object
)


def init_scene():
    """
    Initialize a scene with the basic rendering configurations.
    """
    # the bpy.context module is usually read-only, so we access the current scene through bpy.data
    scene_name: str = bpy.context.scene.name
    scene: Scene = bpy.data.scenes[scene_name]
    scene.render.engine = 'BLENDER_EEVEE'
    # scene.render.engine = 'CYCLES'
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
        {'name': 'pointcloud2', 'color': (1, 0.8, 0.1, 1.0), 'transparent': False},
        {'name': 'ibs', 'color': (0.794, 0.489, 0.8, 1.0), 'transparent': False},
        {'name': 'mesh', 'color': (0.794, 0.489, 0.8, 1.0), 'transparent': False}
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
        # },
        # {
        #     'name': 'point light 1', 'type': 'POINT',
        #     'location': np.array([0, 0, -2.685]), 'energy': 500
        # }
    ]
    for param in params:
        light = bpy.data.lights.new(name=param['name'], type=param['type'])
        light.energy = param['energy']
        if param['type'] == 'SUN':
            light.angle = param['angle']
        light_obj = bpy.data.objects.new(name=param['name'], object_data=light)
        light_obj.location = param['location'] * scale_factor
        bpy.context.collection.objects.link(light_obj)


def creat_pointcloud_modifier(modifier_name: str, material_name: str, sphere_radius: int = 0.005):
    """
    Create the geometry nodes as a modifier for point clouds.
    This modifier will expand each point to a ico sphere for rendering.
    """
    # create a node group and enable it as a geometry modifier
    geom_nodes = bpy.data.node_groups.new('{}'.format(modifier_name), 'GeometryNodeTree')
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
    ico_sphere_node.inputs['Radius'].default_value = sphere_radius
    ico_sphere_node.inputs['Subdivisions'].default_value = 3  # control the smoothness of the ico sphere
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    material_node = nodes.new('GeometryNodeReplaceMaterial')
    # only set the New slot of the Replace Material node because we actually
    # use it to set the material of output instances (spheres), the Old slot
    # is not used.
    material_node.inputs['New'].default_value = bpy.data.materials['{}'.format(material_name)]

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


def launch_render(base_path: str, img_path: str, filename: str, additional_info: str):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    init_scene()
    create_materials()
    init_lights()
    create_pointcloud1_modifier()
    create_pointcloud2_modifier()

    camera_obj: Object = bpy.data.objects['Camera']
    camera_location = np.array([0.7359, -0.6926, 0.4]) * 1.4
    # camera_location = np.array([0, -0.8, 0.258]) * 1.3
    camera_obj.location = camera_location

    base_path = Path(base_path)
    img_path = Path(img_path)
    ply_basename1 = "{}_0".format(filename)
    ply_basename2 = "{}_1".format(filename)
    ply_filename1 = "{}.ply".format(ply_basename1)
    ply_filename2 = "{}.ply".format(ply_basename2)
    ply_file1 = base_path / ply_filename1
    ply_file2 = base_path / ply_filename2

    bpy.ops.wm.ply_import(filepath=ply_file1.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
    pointcloud1 = bpy.data.objects[ply_basename1]
    modifier = pointcloud1.modifiers.new('modifier', 'NODES')
    modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']

    bpy.ops.wm.ply_import(filepath=ply_file2.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
    pointcloud2 = bpy.data.objects[ply_basename2]
    modifier = pointcloud2.modifiers.new('modifier', 'NODES')
    modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    for view_index, sign in enumerate(product(np.array([1, -1]), repeat=3)):
        camera_obj.location = camera_location * sign
        track_object(pointcloud1)
        bpy.context.scene.render.filepath = (img_path / f'{view_index}_{additional_info}.png').as_posix()
        bpy.ops.render.render(write_still=True)


def mat_data():

    # mat sphere
    # name = 'scene1.1000_{}'.format(0)
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_0.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_0']
    # sphere.active_material = bpy.data.materials['mesh']

    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_1.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_1.001']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_2.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_2']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_3.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_3']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_4.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_4']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_5.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_5']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_6.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_6']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_7.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_7']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_8.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_8']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_9.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_9']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_10.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_10']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_11.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_11']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_12.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_12']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_13.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_13']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_14.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_14']
    # sphere.active_material = bpy.data.materials['mesh']
    #
    # mat_sphere_path = r'D:\dataset\IBPCDC\render\mesh\mat_sphere\scene1.1000_15.obj'
    # bpy.ops.wm.obj_import(filepath=mat_sphere_path)
    # sphere = bpy.data.objects['scene1.1000_15']
    # sphere.active_material = bpy.data.materials['mesh']
    pass


if __name__ == '__main__':
    import re

    # init
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    init_scene()
    create_materials()
    init_lights()
    creat_pointcloud_modifier('pointcloud1 modifier', 'pointcloud1', 0.007)
    creat_pointcloud_modifier('pointcloud2 modifier', 'pointcloud2', 0.007)
    creat_pointcloud_modifier('ibs modifier', 'ibs', 0.007)

    camera_obj: Object = bpy.data.objects['Camera']
    radius = 1.4
    camera_location = np.array([0.5, -1, 0.4])
    camera_location /= np.linalg.norm(camera_location)
    camera_location *= radius
    camera_obj.location = camera_location

    mesh_dir = r"D:\dataset\IBSNet\trainData\mesh"
    pcd_gt_dir = r"D:\dataset\IBSNet\trainData\pcdComplete"
    pcd_scan_dir = r"D:\dataset\IBSNet\trainData\pcdScan\diffUDF"
    pcd_pred_dir = r"D:\dataset\IBPCDC\pcdPred\SeedFormer_INTE_lr1e4"
    ibs_gt_dir = r"D:\dataset\IBSNet\evaluateData\IBS_pcd_complete"
    ibs_mesh_gt_dir = r"D:\dataset\IBPCDC\IBSMesh"
    ibs_scan_dir = r"D:\dataset\IBSNet\evaluateData\IBS_pcd_scan"

    category_re = "scene\\d"
    scene_re = "scene\\d.\\d{4}"
    filename_re = "scene\\d.\\d{4}_view\\d+"

    filename = "scene1.1030"
    result_name = "mesh"

    category = re.match(category_re, filename).group()
    scene = re.match(scene_re, filename).group()

    mesh1_filename = os.path.join(mesh_dir, category, "{}_0.obj".format(scene))
    mesh2_filename = os.path.join(mesh_dir, category, "{}_1.obj".format(scene))
    pcd1_gt = os.path.join(pcd_gt_dir, category, "{}_0.ply".format(scene))
    pcd2_gt = os.path.join(pcd_gt_dir, category, "{}_1.ply".format(scene))
    pcd1_scan = os.path.join(pcd_scan_dir, category, "{}_0.ply".format(filename))
    pcd2_scan = os.path.join(pcd_scan_dir, category, "{}_1.ply".format(filename))
    pcd1_pred = os.path.join(pcd_pred_dir, category, "{}_0.ply".format(filename))
    pcd2_pred = os.path.join(pcd_pred_dir, category, "{}_1.ply".format(filename))
    ibs_gt = os.path.join(ibs_gt_dir, category, "{}.ply".format(scene))
    ibs_mesh_gt = os.path.join(ibs_mesh_gt_dir, category, "{}.obj".format(scene))
    ibs_geometric = os.path.join(ibs_scan_dir, category, "{}.ply".format(filename))

    # mesh
    bpy.ops.wm.obj_import(filepath=mesh1_filename)
    mesh1 = bpy.data.objects['{}_0'.format(scene)]
    mesh1.active_material = bpy.data.materials['pointcloud1']

    bpy.ops.wm.obj_import(filepath=mesh2_filename)
    mesh2 = bpy.data.objects['{}_1'.format(scene)]
    mesh2.active_material = bpy.data.materials['pointcloud2']

    # pcd gt
    # bpy.ops.wm.ply_import(filepath=pcd1_gt, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud1 = bpy.data.objects['{}_0'.format(scene)]
    # modifier = pointcloud1.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']
    #
    # bpy.ops.wm.ply_import(filepath=pcd2_gt, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud2 = bpy.data.objects['{}_1'.format(scene)]
    # modifier = pointcloud2.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    # pcd scan
    # bpy.ops.wm.ply_import(filepath=pcd1_scan, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud1 = bpy.data.objects['{}_0'.format(filename)]
    # modifier = pointcloud1.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']
    #
    # bpy.ops.wm.ply_import(filepath=pcd2_scan, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud2 = bpy.data.objects['{}_1'.format(filename)]
    # modifier = pointcloud2.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    # pcd pred
    # bpy.ops.wm.ply_import(filepath=pcd1_pred, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud1 = bpy.data.objects['{}_0'.format(filename)]
    # modifier = pointcloud1.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']
    #
    # bpy.ops.wm.ply_import(filepath=pcd2_pred, forward_axis='NEGATIVE_Z', up_axis='Y')
    # pointcloud2 = bpy.data.objects['{}_1'.format(filename)]
    # modifier = pointcloud2.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    # ibs pcd gt
    # bpy.ops.wm.ply_import(filepath=ibs_gt, forward_axis='NEGATIVE_Z', up_axis='Y')
    # ibs = bpy.data.objects[scene]
    # modifier = ibs.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['ibs modifier']

    # ibs mesh gt
    # bpy.ops.wm.obj_import(filepath=ibs_mesh_gt)
    # ibs_mesh = bpy.data.objects['{}'.format(scene)]
    # ibs_mesh.active_material = bpy.data.materials['ibs']

    # ibs geometric
    # bpy.ops.wm.ply_import(filepath=ibs_geometric, forward_axis='NEGATIVE_Z', up_axis='Y')
    # ibs = bpy.data.objects[filename]
    # modifier = ibs.modifiers.new('modifier', 'NODES')
    # modifier.node_group = bpy.data.node_groups['ibs modifier']

    save_path = r'D:\dataset\IBPCDC\render\temp'
    for view_index, sign in enumerate(product(np.array([1, -1]), repeat=3)):
        camera_obj.location = camera_location * sign
        track_object(mesh1)
        bpy.context.scene.render.filepath = os.path.join(save_path, '{}.png'.format(result_name))
        bpy.ops.render.render(write_still=True)
        break
