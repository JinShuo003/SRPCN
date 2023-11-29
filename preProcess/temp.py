import bpy
import numpy as np
import mathutils
import os
import cv2


def random_rotation():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def cal_pose(angle_x, angle_y, angle_z, radius, random_R):
    angle_x = angle_x / 180.0 * np.pi
    angle_y = angle_y / 180.0 * np.pi
    angle_z = angle_z / 180.0 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    R = np.dot(random_R, R)
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2] * radius, 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose


def setup_blender(width, height, focal_length):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2

    # render layer
    scene = bpy.context.scene
    # scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # remove default cube
    # if bpy.context.object.mode == 'EDIT':
    #     bpy.ops.object.mode_set(mode='OBJECT')
    if 'Cube' in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    return scene, camera, output


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where((depth > 0) & (depth < 5))
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    # Render paras
    width = 1600
    height = 1200
    focal = 200
    scene, camera, output = setup_blender(width, height, focal)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    open('blender.log', 'w+').close()

    # Redirect output to log file
    old_os_out = os.dup(1)
    os.close(1)
    os.open('blender.log', os.O_WRONLY)

    # Import mesh model
    model_path = './scene1.1000_0.obj'
    bpy.ops.import_scene.obj(filepath=model_path)

    # Render
    random_R = random_rotation()
    scene.frame_set(0)
    angle_x = 0
    angle_y = 0
    pose = cal_pose(angle_x, angle_y, 0, 1, random_R)
    camera.matrix_world = mathutils.Matrix(pose)
    scene.render.filepath = r'C:\Users\71008\Desktop\graduationProjects\IBPCDC\preProcess\scene1.1000_0'
    bpy.ops.render.render(write_still=True)

    depth = cv2.imread(r'./scene1.1000_0.png', cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
    points = depth2pcd(depth, intrinsics, pose)
    print(points.shape)
    # np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')
    # np.savetxt(os.path.join(view_point_dir, '%d.xyz' % i), cal_view_point(pose), '%f')
    # i += 1
