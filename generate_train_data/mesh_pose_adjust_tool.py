import os.path
import re

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh.collision

from utils import geometry_utils, path_utils


class App:
    def __init__(self, specs):
        self.specs = specs
        self.filename_tree = None
        self.obj1_path = None
        self.obj2_path = None
        self.mesh1 = None
        self.mesh2 = None

        self.obj_material = rendering.MaterialRecord()
        self.obj_material.shader = 'defaultLit'
        self.material = {
            "obj": self.obj_material
        }

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("layout", 1440, 900)
        self.window.set_on_layout(self.on_layout)

        self.em = self.window.theme.font_size

        # 渲染窗口
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.tool_bar_layout = gui.Vert()
        self.data_dir_editor_layout = None  # 文件路径输入区域
        self.data_format_editor_layout = None  # 文件格式输入区域
        self.geometry_select_area_layout = None  # 数据选择区域
        self.pose_adjust_layout = None  # 位姿调整区域
        self.btn_collision_test_layout = None  # 碰撞检测区域
        self.btn_save_layout = None  # 保存

        self.init_data_dir_editor_area()
        self.init_data_format_editor_area()
        self.init_geometry_select_area()
        self.init_pose_adjust_btn()
        self.init_collision_test_btn()
        self.init_save_btn()

        self.tool_bar_layout.add_child(self.data_dir_editor_layout)
        self.tool_bar_layout.add_child(self.data_format_editor_layout)
        self.tool_bar_layout.add_child(self.geometry_select_area_layout)
        self.tool_bar_layout.add_child(self.pose_adjust_layout)
        self.tool_bar_layout.add_child(self.btn_collision_test_layout)
        self.tool_bar_layout.add_child(self.btn_save_layout)

        self.window.add_child(self.scene)
        self.window.add_child(self.tool_bar_layout)

    def on_data_dir_btn_clicked(self):
        # 根据用户填写的文件路径构建目录树
        dir = self.data_dir_editor.text_value
        if not os.path.exists(dir):
            self.show_message_dialog("warning", "The directory not exist")
            return
        self.filename_tree = path_utils.get_filename_tree(self.specs, dir)

        # 根据目录树构建类别选择器
        self.category_selector.clear_items()
        for category in self.filename_tree.keys():
            self.category_selector.add_item(category)

        # 判断是否添加了可用的类别
        if self.category_selector.number_of_items > 0:
            self.selected_category = self.category_selector.selected_index
        else:
            self.selected_category = -1

        # 如果没有添加可用的类别，说明该目录下无符合目录规则的数据
        if self.selected_category == -1:
            self.show_message_dialog("warning", "No data in this directory")
            return

        # 有可用的类别，构建场景选择器
        for key in self.filename_tree.get(self.category_selector.get_item(self.selected_category)).keys():
            self.scene_selector.add_item(key)

        # 判断是否添加了可用的场景
        if self.scene_selector.number_of_items > 0:
            self.selected_scene = self.scene_selector.selected_index
        else:
            self.selected_scene = -1

    def on_category_selection_changed(self, val, idx):
        if self.selected_category == idx:
            return
        self.scene_selector.clear_items()
        self.selected_category = idx
        for key in self.filename_tree.get(val).keys():
            self.scene_selector.add_item(key)

    def on_scene_selection_changed(self, val, idx):
        if self.selected_scene == idx:
            return
        self.selected_scene = idx

    def on_load_btn_clicked(self):
        if self.data_dir_editor.text_value == "":
            self.show_message_dialog("warning", "Please input the directory of data")
            return
        if self.data_format_editor.text_value == "":
            self.show_message_dialog("warning", "Please input the format of data")
            return
        if self.category_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a category first")
            return
        if self.scene_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a scene first")
            return

        self.obj1_path, self.obj2_path = self.get_geometry_path()

        if not os.path.isfile(self.obj1_path):
            self.show_message_dialog("warning", "{} is not a file")
            return
        if not os.path.isfile(self.obj2_path):
            self.show_message_dialog("warning", "{} is not a file")
            return

        self.mesh1 = geometry_utils.read_mesh(self.obj1_path)
        self.mesh2 = geometry_utils.read_mesh(self.obj2_path)
        self.mesh1.paint_uniform_color((1, 0, 0))
        self.mesh2.paint_uniform_color((0, 1, 0))
        self.mesh1.compute_triangle_normals()
        self.mesh2.compute_triangle_normals()

        self.scene.scene.clear_geometry()
        self.update_object("mesh1", self.mesh1)
        self.update_object("mesh2", self.mesh2)

        self.show_obj1_checkbox.checked = True
        self.show_obj2_checkbox.checked = True

        self.update_camera(self.mesh1)

    def on_show_obj1_checked(self, is_checked):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            self.show_obj1_checkbox.checked = False
            return
        if is_checked:
            self.scene.scene.add_geometry("mesh1", self.mesh1, self.material.get("obj"))
        else:
            self.scene.scene.remove_geometry("mesh1")

    def on_show_obj2_checked(self, is_checked):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj2 first")
            self.show_obj1_checkbox.checked = False
            return
        if is_checked:
            self.scene.scene.add_geometry("mesh2", self.mesh2, self.material.get("obj"))
        else:
            self.scene.scene.remove_geometry("mesh2")

    def on_obj1_x_increate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([float(self.translate_step_editor.text_value), 0, 0]))
        self.update_object("mesh1", self.mesh1)

    def on_obj1_x_decreate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([-float(self.translate_step_editor.text_value), 0, 0]))
        self.update_object("mesh1", self.mesh1)

    def on_obj1_y_increate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([0, float(self.translate_step_editor.text_value), 0]))
        self.update_object("mesh1", self.mesh1)

    def on_obj1_y_decreate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([0, -float(self.translate_step_editor.text_value), 0]))
        self.update_object("mesh1", self.mesh1)

    def on_obj1_z_increate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([0, 0, float(self.translate_step_editor.text_value)]))
        self.update_object("mesh1", self.mesh1)

    def on_obj1_z_decreate_btn_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.translate(np.array([0, 0, -float(self.translate_step_editor.text_value)]))
        self.update_object("mesh1", self.mesh1)

    def on_obj2_x_increate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([float(self.translate_step_editor.text_value), 0, 0]))
        self.update_object("mesh2", self.mesh2)

    def on_obj2_x_decreate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([-float(self.translate_step_editor.text_value), 0, 0]))
        self.update_object("mesh2", self.mesh2)

    def on_obj2_y_increate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([0, float(self.translate_step_editor.text_value), 0]))
        self.update_object("mesh2", self.mesh2)

    def on_obj2_y_decreate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([0, -float(self.translate_step_editor.text_value), 0]))
        self.update_object("mesh2", self.mesh2)

    def on_obj2_z_increate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([0, 0, float(self.translate_step_editor.text_value)]))
        self.update_object("mesh2", self.mesh2)

    def on_obj2_z_decreate_btn_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.translate(np.array([0, 0, -float(self.translate_step_editor.text_value)]))
        self.update_object("mesh2", self.mesh2)

    def on_obj1_scale_increase_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.scale(float(self.scale_ratio_editor.text_value), self.mesh1.get_center())
        self.update_object("mesh1", self.mesh1)

    def on_obj1_scale_decrease_clicked(self):
        if self.mesh1 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj1_checkbox.checked is False:
            self.show_message_dialog("warning", "obj1 is invisible")
            return
        self.mesh1.scale(1/float(self.scale_ratio_editor.text_value), self.mesh1.get_center())
        self.update_object("mesh1", self.mesh1)

    def on_obj2_scale_increase_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.scale(float(self.scale_ratio_editor.text_value), self.mesh2.get_center())
        self.update_object("mesh2", self.mesh2)

    def on_obj2_scale_decrease_clicked(self):
        if self.mesh2 is None:
            self.show_message_dialog("warning", "Please load obj1 first")
            return
        if self.show_obj2_checkbox.checked is False:
            self.show_message_dialog("warning", "obj2 is invisible")
            return
        self.mesh2.scale(1/float(self.scale_ratio_editor.text_value), self.mesh2.get_center())
        self.update_object("mesh2", self.mesh2)

    def on_collision_test_btn_clicked(self):
        if self.mesh1 is None or self.mesh2 is None:
            self.show_message_dialog("warning", "Please load geometries first")
            return
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object("mesh1", geometry_utils.o3d2trimesh(self.mesh1))
        collision_manager.add_object("mesh2", geometry_utils.o3d2trimesh(self.mesh2))
        is_collision, data = collision_manager.in_collision_internal(return_data=True)
        if is_collision:
            self.show_message_dialog("info", "Collision occured")
            self.show_collicion_area(data)
        else:
            self.show_message_dialog("info", "No Collision")
            if self.scene.scene.has_geometry("collision_area"):
                self.scene.scene.remove_geometry("collision_area")

    def show_collicion_area(self, data):
        if self.scene.scene.has_geometry("collision_area"):
            self.scene.scene.remove_geometry("collision_area")
        collision_points = []
        for collision_data in data:
            collision_points.append(collision_data.point)
        aabb = geometry_utils.get_pcd_from_np(np.array(collision_points)).get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        self.scene.scene.add_geometry("collision_area", aabb, self.material.get("obj"))

    def on_save_btn_clicked(self):
        if self.mesh1 is None or self.mesh2 is None:
            self.show_message_dialog("warning", "Please load object first")

        if not os.path.isfile(self.obj1_path):
            self.show_message_dialog("warning", "{} is not a file")
            return
        if not os.path.isfile(self.obj2_path):
            self.show_message_dialog("warning", "{} is not a file")
            return

        o3d.io.write_triangle_mesh(self.obj1_path, self.mesh1)
        o3d.io.write_triangle_mesh(self.obj2_path, self.mesh2)

        self.show_message_dialog("Info", "Saved\nobj1 in {}\nobj2 in {}".format(self.obj1_path, self.obj2_path))

    def update_object(self, name, mesh):
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, mesh, self.material.get("obj"))

    def update_camera(self, mesh):
        bounds = mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r

        pannel_width = 17 * layout_context.theme.font_size
        pannel_height = r.height

        self.tool_bar_layout.frame = gui.Rect(r.get_right() - pannel_width, r.y, pannel_width, pannel_height)

    def on_dialog_ok(self):
        self.window.close_dialog()

    def get_geometry_path(self):
        data_dir = self.data_dir_editor.text_value
        category = self.category_selector.get_item(self.selected_category)
        format = self.data_format_editor.text_value
        filename1 = "{}_0.{}".format(self.scene_selector.get_item(self.selected_scene), format)
        filename2 = "{}_1.{}".format(self.scene_selector.get_item(self.selected_scene), format)
        file1_path = os.path.join(data_dir, category, filename1)
        file2_path = os.path.join(data_dir, category, filename2)
        return file1_path, file2_path

    def show_message_dialog(self, title, message):
        dlg = gui.Dialog(title)
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self.on_dialog_ok)

        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        dlg_layout.add_child(button_layout)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def init_data_dir_editor_area(self):
        self.data_dir_editor_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, 0))
        self.data_dir_editor_text = gui.Label("data directory")
        self.data_dir_editor = gui.TextEdit()
        self.data_dir_editor.text_value = "D:\\dataset\\IBPCDC\\mesh"
        self.data_dir_editor_btn = gui.Button("confirm")
        self.data_dir_editor_btn.set_on_clicked(self.on_data_dir_btn_clicked)
        self.data_dir_editor_layout.add_child(self.data_dir_editor_text)
        self.data_dir_editor_layout.add_child(self.data_dir_editor)
        self.data_dir_editor_layout.add_fixed(self.em/2)
        self.data_dir_editor_layout.add_child(self.data_dir_editor_btn)

    def init_data_format_editor_area(self):
        self.data_format_editor_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, 0))
        self.data_format_editor_text = gui.Label("data format")
        self.data_format_editor = gui.TextEdit()
        self.data_format_editor.text_value = "obj"
        self.data_format_editor_layout.add_child(self.data_format_editor_text)
        self.data_format_editor_layout.add_child(self.data_format_editor)

    def init_geometry_select_area(self):
        self.geometry_select_area_layout = gui.Vert(self.em/2, gui.Margins(self.em, self.em, self.em, 0))

        # 类别
        self.selected_category = -1
        self.category_selector_layout = gui.Vert()
        self.category_selector_text = gui.Label("category selector")
        self.category_selector = gui.Combobox()
        self.category_selector.set_on_selection_changed(self.on_category_selection_changed)
        self.category_selector_layout.add_child(self.category_selector_text)
        self.category_selector_layout.add_child(self.category_selector)

        # 场景
        self.selected_scene = -1
        self.scene_selector_layout = gui.Vert()
        self.scene_selector_text = gui.Label("scene selector")
        self.scene_selector = gui.Combobox()
        self.scene_selector.set_on_selection_changed(self.on_scene_selection_changed)
        self.scene_selector_layout.add_child(self.scene_selector_text)
        self.scene_selector_layout.add_child(self.scene_selector)

        # 确认
        self.btn_load = gui.Button("Load")
        self.btn_load.set_on_clicked(self.on_load_btn_clicked)

        # 可见性
        self.visibility_layout = gui.Horiz()
        self.show_obj1_checkbox_text = gui.Label("mesh1")
        self.show_obj2_checkbox_text = gui.Label("mesh2")
        self.show_obj1_checkbox = gui.Checkbox("")
        self.show_obj2_checkbox = gui.Checkbox("")
        self.show_obj1_checkbox.set_on_checked(self.on_show_obj1_checked)
        self.show_obj2_checkbox.set_on_checked(self.on_show_obj2_checked)
        self.visibility_layout.add_child(self.show_obj1_checkbox_text)
        self.visibility_layout.add_child(self.show_obj1_checkbox)
        self.visibility_layout.add_stretch()
        self.visibility_layout.add_child(self.show_obj2_checkbox_text)
        self.visibility_layout.add_child(self.show_obj2_checkbox)

        self.geometry_select_area_layout.add_child(self.category_selector_layout)
        self.geometry_select_area_layout.add_child(self.scene_selector_layout)
        self.geometry_select_area_layout.add_child(self.btn_load)
        self.geometry_select_area_layout.add_child(self.visibility_layout)

    def init_pose_adjust_btn(self):
        self.pose_adjust_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, 0))

        # 步长
        self.translate_step_editor_layout = gui.Vert(0, gui.Margins(0, 0, 0, self.em/2))
        self.translate_step_editor_text = gui.Label("translate step")
        self.translate_step_editor = gui.TextEdit()
        self.translate_step_editor.text_value = "0.005"
        self.translate_step_editor_layout.add_child(self.translate_step_editor_text)
        self.translate_step_editor_layout.add_child(self.translate_step_editor)

        # 缩放
        self.scale_ratio_editor_layout = gui.Vert(0, gui.Margins(0, 0, 0, self.em/2))
        self.scale_ratio_editor_text = gui.Label("scale ratio")
        self.scale_ratio_editor = gui.TextEdit()
        self.scale_ratio_editor.text_value = "1"
        self.scale_ratio_editor_layout.add_child(self.scale_ratio_editor_text)
        self.scale_ratio_editor_layout.add_child(self.scale_ratio_editor)

        # obj1
        self.translate_area_obj1_layout = gui.Vert(0, gui.Margins(0, 0, 0, self.em/2))

        self.obj1_text = gui.Label("Object1")
        self.translate_btn_box = gui.Horiz()
        self.translate_obj1_layout = gui.VGrid(3)

        self.btn_obj1_x_increase = gui.Button("x+")
        self.btn_obj1_x_decrease = gui.Button("x-")
        self.btn_obj1_y_increase = gui.Button("y+")
        self.btn_obj1_y_decrease = gui.Button("y-")
        self.btn_obj1_z_increase = gui.Button("z+")
        self.btn_obj1_z_decrease = gui.Button("z-")
        self.btn_obj1_x_increase.set_on_clicked(self.on_obj1_x_increate_btn_clicked)
        self.btn_obj1_x_decrease.set_on_clicked(self.on_obj1_x_decreate_btn_clicked)
        self.btn_obj1_y_increase.set_on_clicked(self.on_obj1_y_increate_btn_clicked)
        self.btn_obj1_y_decrease.set_on_clicked(self.on_obj1_y_decreate_btn_clicked)
        self.btn_obj1_z_increase.set_on_clicked(self.on_obj1_z_increate_btn_clicked)
        self.btn_obj1_z_decrease.set_on_clicked(self.on_obj1_z_decreate_btn_clicked)

        self.translate_obj1_layout.add_child(self.btn_obj1_x_increase)
        self.translate_obj1_layout.add_child(self.btn_obj1_y_increase)
        self.translate_obj1_layout.add_child(self.btn_obj1_z_increase)
        self.translate_obj1_layout.add_child(self.btn_obj1_x_decrease)
        self.translate_obj1_layout.add_child(self.btn_obj1_y_decrease)
        self.translate_obj1_layout.add_child(self.btn_obj1_z_decrease)

        self.translate_btn_box.add_stretch()
        self.translate_btn_box.add_child(self.translate_obj1_layout)
        self.translate_btn_box.add_stretch()

        self.scale_obj1_layout = gui.Horiz()
        self.btn_obj1_scale_increase = gui.Button("scale+")
        self.btn_obj1_scale_decrease = gui.Button("scale-")
        self.btn_obj1_scale_increase.set_on_clicked(self.on_obj1_scale_increase_clicked)
        self.btn_obj1_scale_decrease.set_on_clicked(self.on_obj1_scale_decrease_clicked)
        self.scale_obj1_layout.add_stretch()
        self.scale_obj1_layout.add_child(self.btn_obj1_scale_increase)
        self.scale_obj1_layout.add_child(self.btn_obj1_scale_decrease)
        self.scale_obj1_layout.add_stretch()

        self.translate_area_obj1_layout.add_child(self.obj1_text)
        self.translate_area_obj1_layout.add_child(self.translate_btn_box)
        self.translate_area_obj1_layout.add_child(self.scale_obj1_layout)

        # obj2
        self.translate_area_obj2_layout = gui.Vert(0, gui.Margins(0, 0, 0, self.em/2))

        self.obj2_text = gui.Label("Object2")
        self.translate_btn_box = gui.Horiz()
        self.translate_obj2_layout = gui.VGrid(3)

        self.btn_obj2_x_increase = gui.Button("x+")
        self.btn_obj2_x_decrease = gui.Button("x-")
        self.btn_obj2_y_increase = gui.Button("y+")
        self.btn_obj2_y_decrease = gui.Button("y-")
        self.btn_obj2_z_increase = gui.Button("z+")
        self.btn_obj2_z_decrease = gui.Button("z-")
        self.btn_obj2_x_increase.set_on_clicked(self.on_obj2_x_increate_btn_clicked)
        self.btn_obj2_x_decrease.set_on_clicked(self.on_obj2_x_decreate_btn_clicked)
        self.btn_obj2_y_increase.set_on_clicked(self.on_obj2_y_increate_btn_clicked)
        self.btn_obj2_y_decrease.set_on_clicked(self.on_obj2_y_decreate_btn_clicked)
        self.btn_obj2_z_increase.set_on_clicked(self.on_obj2_z_increate_btn_clicked)
        self.btn_obj2_z_decrease.set_on_clicked(self.on_obj2_z_decreate_btn_clicked)

        self.translate_obj2_layout.add_child(self.btn_obj2_x_increase)
        self.translate_obj2_layout.add_child(self.btn_obj2_y_increase)
        self.translate_obj2_layout.add_child(self.btn_obj2_z_increase)
        self.translate_obj2_layout.add_child(self.btn_obj2_x_decrease)
        self.translate_obj2_layout.add_child(self.btn_obj2_y_decrease)
        self.translate_obj2_layout.add_child(self.btn_obj2_z_decrease)

        self.translate_btn_box.add_stretch()
        self.translate_btn_box.add_child(self.translate_obj2_layout)
        self.translate_btn_box.add_stretch()

        self.scale_obj2_layout = gui.Horiz()
        self.btn_obj2_scale_increase = gui.Button("scale+")
        self.btn_obj2_scale_decrease = gui.Button("scale-")
        self.btn_obj2_scale_increase.set_on_clicked(self.on_obj2_scale_increase_clicked)
        self.btn_obj2_scale_decrease.set_on_clicked(self.on_obj2_scale_decrease_clicked)
        self.scale_obj2_layout.add_stretch()
        self.scale_obj2_layout.add_child(self.btn_obj2_scale_increase)
        self.scale_obj2_layout.add_child(self.btn_obj2_scale_decrease)
        self.scale_obj2_layout.add_stretch()

        self.translate_area_obj2_layout.add_child(self.obj2_text)
        self.translate_area_obj2_layout.add_child(self.translate_btn_box)
        self.translate_area_obj2_layout.add_child(self.scale_obj2_layout)

        self.pose_adjust_layout.add_child(self.translate_step_editor_layout)
        self.pose_adjust_layout.add_child(self.scale_ratio_editor_layout)
        self.pose_adjust_layout.add_child(self.translate_area_obj1_layout)
        self.pose_adjust_layout.add_child(self.translate_area_obj2_layout)

    def init_collision_test_btn(self):
        self.btn_collision_test_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, 0))
        self.btn_collision_test = gui.Button("Test collision")
        self.btn_collision_test.set_on_clicked(self.on_collision_test_btn_clicked)
        self.btn_collision_test_layout.add_child(self.btn_collision_test)

    def init_save_btn(self):
        self.btn_save_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        self.btn_save = gui.Button("Save")
        self.btn_save.set_on_clicked(self.on_save_btn_clicked)
        self.btn_save_layout.add_child(self.btn_save)

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    # 获取配置参数
    config_filepath = 'configs/mesh_pose_adjust_tool.json'
    specs = path_utils.read_config(config_filepath)

    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))

    app = App(specs)
    app.run()
