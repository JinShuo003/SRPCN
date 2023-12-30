import os.path
import re

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from utils import path_utils


class App:
    def __init__(self, specs):
        self.specs = specs
        self.filename_tree = None
        self.obj1_path = None
        self.obj2_path = None
        self.mesh1 = None
        self.mesh2 = None

        self.key_pcd_complete_1 = "pcd_complete_1"
        self.key_pcd_complete_2 = "pcd_complete_2"
        self.key_pcd_partial_1 = "pcd_partial_1"
        self.key_pcd_partial_2 = "pcd_partial_2"
        self.key_pcd_pred1_1 = "pcd_pred1_1"
        self.key_pcd_pred1_2 = "pcd_pred1_2"
        self.key_pcd_pred2_1 = "pcd_pred2_1"
        self.key_pcd_pred2_2 = "pcd_pred2_2"
        self.key_IBS = "IBS"

        self.TAG_PCD1 = "pcd1"
        self.TAG_PCD2 = "pcd2"
        self.TAG_IBS = "IBS"

        self.obj_material = rendering.MaterialRecord()
        self.obj_material.shader = 'defaultLit'
        self.material = {
            "obj": self.obj_material
        }

        gui.Application.instance.initialize()

        self.scene_pcd_complete_str = "complete"
        self.scene_pcd_partial_str = "partial"
        self.scene_pcd_pred1_str = specs.get("sub_window_3_name")
        self.scene_pcd_pred2_str = specs.get("sub_window_4_name")

        self.sub_window_width = specs.get("window_size_options").get("sub_window_width")
        self.sub_window_height = specs.get("window_size_options").get("sub_window_height")
        self.tool_bar_width = specs.get("window_size_options").get("tool_bar_width")
        self.window = gui.Application.instance.create_window("layout", self.sub_window_width * 2 + self.tool_bar_width,
                                                             self.sub_window_height * 2)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_key(self.on_key)

        self.em = self.window.theme.font_size

        # 完整点云窗口
        self.scene_pcd_complete = gui.SceneWidget()
        self.scene_pcd_complete.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_pcd_complete_text = gui.Label("complete")

        # 残缺点云窗口
        self.scene_pcd_partial = gui.SceneWidget()
        self.scene_pcd_partial.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_pcd_partial_text = gui.Label("partial")

        # 补全结果1窗口
        self.scene_pcd_pred1 = gui.SceneWidget()
        self.scene_pcd_pred1.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_pcd_pred1_text = gui.Label(self.specs.get("sub_window_3_name"))

        # 补全结果2窗口
        self.scene_pcd_pred2 = gui.SceneWidget()
        self.scene_pcd_pred2.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_pcd_pred2_text = gui.Label(self.specs.get("sub_window_4_name"))

        # 窗口是否显示的标记
        self.flag_pcd_pred1_window_show = False
        self.flag_pcd_pred2_window_show = True

        # 点云
        self.pcd_complete_1 = None
        self.pcd_complete_2 = None
        self.pcd_partial_1 = None
        self.pcd_partial_2 = None
        self.pcd_pred1_1 = None
        self.pcd_pred1_2 = None
        self.pcd_pred2_1 = None
        self.pcd_pred2_2 = None
        self.IBS = None

        # 工具栏
        self.tool_bar_layout = gui.Vert()

        # 文件路径输入区域
        self.data_dir_editor_layout = None
        self.pcd_pred1_dir_editor_text = None
        self.pcd_pred1_dir_editor = None
        self.pcd_pred2_dir_editor_text = None
        self.pcd_pred2_dir_editor = None
        self.data_dir_confirm_btn = None

        # 数据选择区域
        self.geometry_select_layout = None
        self.selected_category = -1
        self.category_selector_layout = None
        self.category_selector_text = None
        self.category_selector = None
        self.selected_scene = -1
        self.scene_selector_layout = None
        self.scene_selector_text = None
        self.scene_selector = None
        self.selected_view = None
        self.view_selector_layout = None
        self.view_selector_text = None
        self.view_selector = None
        self.btn_load = None

        # 数据切换区域
        self.data_switch_layout = None
        self.category_switch_area = None
        self.category_switch_text = None
        self.category_switch_btn_area = None
        self.btn_pre_category = None
        self.btn_next_category = None
        self.scene_switch_area = None
        self.scene_switch_text = None
        self.scene_switch_btn_area = None
        self.btn_pre_scene = None
        self.btn_next_scene = None
        self.view_switch_area = None
        self.view_switch_text = None
        self.view_switch_btn_area = None
        self.btn_pre_view = None
        self.btn_next_view = None

        # 可见性
        self.show_pcd1_checked = False
        self.show_pcd2_checked = False
        self.show_ibs_checked = False
        self.visible_control_layout = None
        self.visible_text = None
        self.visible_control_checkbox_layout = None
        self.show_pcd1_checkbox_text = None
        self.show_pcd2_checkbox_text = None
        self.show_ibs_checkbox_text = None
        self.show_pcd1_checkbox = None
        self.show_pcd2_checkbox = None
        self.show_ibs_checkbox = None

        # 当前信息
        self.data_info_layout = None
        self.category_info = None
        self.scene_info = None
        self.view_info = None
        self.theta = None
        self.phi = None
        self.category_info_patten = "category: {}"
        self.scene_info_patten = "scene: {}"
        self.view_info_patten = "view: {}"
        self.theta_patten = "theta: {}"
        self.phi_patten = "view: {}"

        self.init_data_dir_editor_area()
        self.init_geometry_select_area()
        self.init_data_switch_area()
        self.init_visible_control_area()
        self.init_info_area()

        self.tool_bar_layout.add_child(self.data_dir_editor_layout)
        self.tool_bar_layout.add_child(self.geometry_select_layout)
        self.tool_bar_layout.add_child(self.data_switch_layout)
        self.tool_bar_layout.add_child(self.visible_control_layout)
        self.tool_bar_layout.add_stretch()
        self.tool_bar_layout.add_child(self.data_info_layout)

        self.window.add_child(self.scene_pcd_complete)
        self.window.add_child(self.scene_pcd_partial)
        self.window.add_child(self.scene_pcd_pred1)
        self.window.add_child(self.scene_pcd_pred2)
        self.window.add_child(self.tool_bar_layout)
        self.window.add_child(self.scene_pcd_complete_text)
        self.window.add_child(self.scene_pcd_partial_text)
        self.window.add_child(self.scene_pcd_pred1_text)
        self.window.add_child(self.scene_pcd_pred2_text)

    def on_data_dir_comfirm_btn_clicked(self):
        # 根据用户填写的文件路径构建目录树
        filename_tree_dir = self.specs.get("path_options").get("filename_tree_dir")
        dir = self.specs.get("path_options").get("geometries_dir").get(filename_tree_dir)
        if not os.path.exists(dir):
            self.show_message_dialog("warning", "The directory not exist")
            return
        self.filename_tree = path_utils.get_filename_tree(self.specs, dir)

        # 类别
        self.category_selector.clear_items()
        for category in self.filename_tree.keys():
            self.category_selector.add_item(category)
        if self.category_selector.number_of_items > 0:
            self.selected_category = self.category_selector.selected_index
            print("seleted category: ", self.selected_category)
        else:
            self.selected_category = -1
            print("seleted category: ", self.selected_category)
            self.show_message_dialog("warning", "No category in this directory")
            return

        # 场景
        self.scene_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        for key in self.filename_tree.get(selected_category).keys():
            self.scene_selector.add_item(key)
        if self.scene_selector.number_of_items > 0:
            self.selected_scene = self.scene_selector.selected_index
            print("seleted scene: ", self.selected_scene)
        else:
            self.selected_scene = -1
            print("seleted scene: ", self.selected_scene)
            self.show_message_dialog("warning", "No scene in this directory")
            return

        # 视角
        self.view_selector.clear_items()
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        if self.view_selector.number_of_items > 0:
            self.selected_view = self.view_selector.selected_index
            print("seleted view: ", self.selected_view)
        else:
            self.selected_view = -1
            print("seleted view: ", self.selected_view)
            self.show_message_dialog("warning", "No view in this directory")
            return

    def on_category_selection_changed(self, val, idx):
        if self.selected_category == idx:
            return
        self.selected_category = idx
        print("seleted category: ", self.selected_category)
        self.scene_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        for key in self.filename_tree.get(selected_category).keys():
            self.scene_selector.add_item(key)

        self.selected_scene = 0
        print("seleted scene: ", self.selected_scene)
        self.view_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        self.selected_view = 0
        print("seleted view: ", self.selected_view)

    def on_scene_selection_changed(self, val, idx):
        if self.selected_scene == idx:
            return
        self.selected_scene = idx
        print("seleted scene: ", self.selected_scene)
        self.view_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        self.selected_view = 0
        print("seleted view: ", self.selected_view)

    def on_view_selection_changed(self, val, idx):
        if self.selected_view == idx:
            return
        self.selected_view = idx
        print("seleted view: ", self.selected_view)

    def clear_all_window(self):
        if self.scene_pcd_complete:
            self.scene_pcd_complete.scene.clear_geometry()
        if self.scene_pcd_partial:
            self.scene_pcd_partial.scene.clear_geometry()
        if self.scene_pcd_pred1:
            self.scene_pcd_pred1.scene.clear_geometry()
        if self.scene_pcd_pred2:
            self.scene_pcd_pred2.scene.clear_geometry()

    def paint_color(self, geometry, color):
        if geometry is None:
            return
        geometry.paint_uniform_color(color)

    def load_data(self):
        if self.category_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a category first")
            return
        if self.scene_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a scene first")
            return
        if self.view_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a view first")
            return

        geometry_path_dict = self.get_geometry_path()
        self.pcd_complete_1 = self.read_pcd(geometry_path_dict.get("pcd_complete_1"))
        self.pcd_complete_2 = self.read_pcd(geometry_path_dict.get("pcd_complete_2"))
        self.pcd_partial_1 = self.read_pcd(geometry_path_dict.get("pcd_partial_1"))
        self.pcd_partial_2 = self.read_pcd(geometry_path_dict.get("pcd_partial_2"))
        self.pcd_pred1_1 = self.read_pcd(geometry_path_dict.get("pcd_pred1_1"))
        self.pcd_pred1_2 = self.read_pcd(geometry_path_dict.get("pcd_pred1_2"))
        self.pcd_pred2_1 = self.read_pcd(geometry_path_dict.get("pcd_pred2_1"))
        self.pcd_pred2_2 = self.read_pcd(geometry_path_dict.get("pcd_pred2_2"))
        self.IBS = self.read_mesh(geometry_path_dict.get("IBS"))
        if self.pcd_pred1_1 is None and self.pcd_pred1_2 is None:
            self.flag_pcd_pred1_window_show = False
        else:
            self.flag_pcd_pred1_window_show = True
        if self.pcd_pred2_1 is None and self.pcd_pred2_2 is None:
            self.flag_pcd_pred2_window_show = False
        else:
            self.flag_pcd_pred2_window_show = True

        self.paint_color(self.pcd_complete_1, (0.7, 0.2, 0.2))
        self.paint_color(self.pcd_complete_2, (0.2, 0.7, 0.2))
        self.paint_color(self.pcd_partial_1, (0.7, 0.2, 0.2))
        self.paint_color(self.pcd_partial_2, (0.2, 0.7, 0.2))
        self.paint_color(self.pcd_pred1_1, (0.7, 0.2, 0.2))
        self.paint_color(self.pcd_pred1_2, (0.2, 0.7, 0.2))
        self.paint_color(self.pcd_pred2_1, (0.7, 0.2, 0.2))
        self.paint_color(self.pcd_pred2_2, (0.2, 0.7, 0.2))
        self.paint_color(self.IBS, (0.2, 0.2, 0.7))

        self.clear_all_window()
        if self.show_pcd1_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_PCD1, self.pcd_complete_1)
            self.add_object(self.scene_pcd_partial, self.TAG_PCD1, self.pcd_partial_1)
            self.add_object(self.scene_pcd_pred1, self.TAG_PCD1, self.pcd_pred1_1)
            self.add_object(self.scene_pcd_pred2, self.TAG_PCD1, self.pcd_pred2_1)
        if self.show_pcd2_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_PCD2, self.pcd_complete_2)
            self.add_object(self.scene_pcd_partial, self.TAG_PCD2, self.pcd_partial_2)
            self.add_object(self.scene_pcd_pred1, self.TAG_PCD2, self.pcd_pred1_2)
            self.add_object(self.scene_pcd_pred2, self.TAG_PCD2, self.pcd_pred2_2)
        if self.show_ibs_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_IBS, self.IBS)
            self.add_object(self.scene_pcd_partial, self.TAG_IBS, self.IBS)
            if self.flag_pcd_pred1_window_show:
                self.add_object(self.scene_pcd_pred1, self.TAG_IBS, self.IBS)
            if self.flag_pcd_pred2_window_show:
                self.add_object(self.scene_pcd_pred2, self.TAG_IBS, self.IBS)

    def on_load_btn_clicked(self):
        self.load_data()
        self.update_info_area()
        self.update_camera_all_scene()

    def get_view_info(self):
        view = self.view_selector.get_item(self.selected_view)
        view_idx = re.findall(r'\d+', view)
        theta = int(int(view_idx[-1]) / 8) * 45 + 45
        phi = (int(view_idx[-1]) % 8) * 45
        return theta, phi

    def update_info_area(self):
        self.category_info.text = self.category_info_patten.format(
            self.category_selector.get_item(self.selected_category))
        self.scene_info.text = self.scene_info_patten.format(self.scene_selector.get_item(self.selected_scene))
        self.view_info.text = self.view_info_patten.format(self.view_selector.get_item(self.selected_view))
        theta, phi = self.get_view_info()
        self.theta.text = self.theta_patten.format(theta)
        self.phi.text = self.theta_patten.format(phi)

    def on_pre_category_btn_clicked(self):
        if self.selected_category <= 0:
            return
        self.on_category_selection_changed(self.category_selector.get_item(self.selected_category - 1),
                                           self.selected_category - 1)
        self.load_data()
        self.update_info_area()
        self.update_camera_all_scene()

    def on_next_category_btn_clicked(self):
        if self.selected_category >= self.category_selector.number_of_items - 1:
            return
        self.on_category_selection_changed(self.category_selector.get_item(self.selected_category + 1),
                                           self.selected_category + 1)
        self.load_data()
        self.update_info_area()
        self.update_camera_all_scene()

    def on_pre_scene_btn_clicked(self):
        if self.selected_scene <= 0:
            return
        self.on_scene_selection_changed(self.scene_selector.get_item(self.selected_scene - 1), self.selected_scene - 1)
        self.load_data()
        self.update_info_area()
        self.update_camera_all_scene()

    def on_next_scene_btn_clicked(self):
        if self.selected_scene >= self.scene_selector.number_of_items - 1:
            return
        self.on_scene_selection_changed(self.scene_selector.get_item(self.selected_scene + 1), self.selected_scene + 1)
        self.load_data()
        self.update_info_area()
        self.update_camera_all_scene()

    def on_pre_view_btn_clicked(self):
        if self.selected_view <= 0:
            return
        self.on_view_selection_changed(self.view_selector.get_item(self.selected_view - 1), self.selected_view - 1)
        self.load_data()
        self.update_info_area()

    def on_next_view_btn_clicked(self):
        if self.selected_view >= self.view_selector.number_of_items - 1:
            return
        self.on_view_selection_changed(self.view_selector.get_item(self.selected_view + 1), self.selected_view + 1)
        self.load_data()
        self.update_info_area()

    def on_show_pcd1_checked(self, is_checked):
        print("show pcd1 checked: ", is_checked)
        self.show_pcd1_checked = is_checked
        if is_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_PCD1, self.pcd_complete_1)
            self.add_object(self.scene_pcd_partial, self.TAG_PCD1, self.pcd_partial_1)
            if self.flag_pcd_pred1_window_show:
                self.add_object(self.scene_pcd_pred1, self.TAG_PCD1, self.pcd_pred1_1)
            if self.flag_pcd_pred2_window_show:
                self.add_object(self.scene_pcd_pred2, self.TAG_PCD1, self.pcd_pred2_1)
        else:
            self.remove_object(self.scene_pcd_complete, self.TAG_PCD1)
            self.remove_object(self.scene_pcd_partial, self.TAG_PCD1)
            self.remove_object(self.scene_pcd_pred1, self.TAG_PCD1)
            self.remove_object(self.scene_pcd_pred2, self.TAG_PCD1)

    def on_show_pcd2_checked(self, is_checked):
        print("show pcd2checked: ", is_checked)
        self.show_pcd2_checked = is_checked
        if is_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_PCD2, self.pcd_complete_2)
            self.add_object(self.scene_pcd_partial, self.TAG_PCD2, self.pcd_partial_2)
            if self.flag_pcd_pred1_window_show:
                self.add_object(self.scene_pcd_pred1, self.TAG_PCD2, self.pcd_pred1_2)
            if self.flag_pcd_pred2_window_show:
                self.add_object(self.scene_pcd_pred2, self.TAG_PCD2, self.pcd_pred2_2)
        else:
            self.remove_object(self.scene_pcd_complete, self.TAG_PCD2)
            self.remove_object(self.scene_pcd_partial, self.TAG_PCD2)
            self.remove_object(self.scene_pcd_pred1, self.TAG_PCD2)
            self.remove_object(self.scene_pcd_pred2, self.TAG_PCD2)

    def on_show_ibs_checked(self, is_checked):
        print("show ibs checked: ", is_checked)
        self.show_ibs_checked = is_checked
        if is_checked:
            self.add_object(self.scene_pcd_complete, self.TAG_IBS, self.IBS)
            self.add_object(self.scene_pcd_partial, self.TAG_IBS, self.IBS)
            if self.flag_pcd_pred1_window_show:
                self.add_object(self.scene_pcd_pred1, self.TAG_IBS, self.IBS)
            if self.flag_pcd_pred2_window_show:
                self.add_object(self.scene_pcd_pred2, self.TAG_IBS, self.IBS)
        else:
            self.remove_object(self.scene_pcd_complete, self.TAG_IBS)
            self.remove_object(self.scene_pcd_partial, self.TAG_IBS)
            self.remove_object(self.scene_pcd_pred1, self.TAG_IBS)
            self.remove_object(self.scene_pcd_pred2, self.TAG_IBS)

    def read_mesh(self, mesh_path):
        if not os.path.exists(mesh_path):
            return None
        if not os.path.isfile(mesh_path):
            self.show_message_dialog("warning", "{} is not a file".format(mesh_path))
            return None
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if np.asarray(mesh.vertices).shape[0] == 0:
            self.show_message_dialog("warning", "{} is not a mesh file".format(mesh_path))
            return None
        return mesh

    def read_pcd(self, pcd_path):
        if not os.path.exists(pcd_path):
            return None
        if not os.path.isfile(pcd_path):
            self.show_message_dialog("warning", "{} is not a file".format(pcd_path))
            return None
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.normals = o3d.utility.Vector3dVector(np.array([]).reshape(0, 3))
        if np.asarray(pcd.points).shape[0] == 0:
            self.show_message_dialog("warning", "{} is not a pcd file".format(pcd_path))
            return None
        return pcd

    def remove_object(self, scene, name):
        if scene is None:
            return
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)

    def add_object(self, scene, name, geometry):
        if scene is None or geometry is None:
            return
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, geometry, self.material.get("obj"))

    def update_camera_all_scene(self):
        self.update_camera(self.scene_pcd_complete, self.pcd_complete_1, self.pcd_complete_2)
        self.update_camera(self.scene_pcd_partial, self.pcd_partial_1, self.pcd_partial_2)
        self.update_camera(self.scene_pcd_pred1, self.pcd_pred1_1, self.pcd_pred1_2)
        self.update_camera(self.scene_pcd_pred2, self.pcd_pred2_1, self.pcd_pred2_2)

    def update_camera(self, scene, geometry1, geometry2):
        bounds1 = None
        bounds2 = None
        if geometry1 is None and geometry2 is None:
            return
        if geometry1 is not None:
            bounds1 = geometry1.get_axis_aligned_bounding_box()
        if geometry2 is not None:
            bounds2 = geometry2.get_axis_aligned_bounding_box()
        if bounds1 is None:
            scene.setup_camera(60, bounds2, bounds2.get_center())
            return
        if bounds2 is None:
            scene.setup_camera(60, bounds1, bounds1.get_center())
            return
        bounds = [bounds1.get_max_bound(), bounds1.get_min_bound(), bounds2.get_max_bound(), bounds2.get_min_bound()]
        bounds = o3d.utility.Vector3dVector(bounds)
        bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bounds)
        scene.setup_camera(60, bounds, bounds.get_center())

    def on_layout(self, layout_context):
        r = self.window.content_rect

        if self.scene_pcd_complete:
            self.scene_pcd_complete.frame = gui.Rect(r.x, r.y, self.sub_window_width, self.sub_window_height)
            self.scene_pcd_complete_text.frame = gui.Rect(r.x, r.y,
                                                          len(self.scene_pcd_complete_str)*8, 0)
        if self.scene_pcd_partial:
            self.scene_pcd_partial.frame = gui.Rect(r.x + self.sub_window_width, r.y, self.sub_window_width,
                                                    self.sub_window_height)
            self.scene_pcd_partial_text.frame = gui.Rect(r.x + self.sub_window_width, r.y,
                                                         len(self.scene_pcd_partial_str)*8, 0)
        if self.scene_pcd_pred1:
            self.scene_pcd_pred1.frame = gui.Rect(r.x, r.y + self.sub_window_height, self.sub_window_width,
                                                  self.sub_window_height)
            self.scene_pcd_pred1_text.frame = gui.Rect(r.x, r.y + self.sub_window_height,
                                                       len(self.scene_pcd_pred1_str)*8, 0)
        if self.scene_pcd_pred2:
            self.scene_pcd_pred2.frame = gui.Rect(r.x + self.sub_window_width, r.y + self.sub_window_height,
                                                  self.sub_window_width, self.sub_window_height)
            self.scene_pcd_pred2_text.frame = gui.Rect(r.x + self.sub_window_width, r.y + self.sub_window_height,
                                                       len(self.scene_pcd_pred2_str)*8, 0)

        self.tool_bar_layout.frame = gui.Rect(r.x + self.sub_window_width * 2, r.y, self.tool_bar_width, r.height)

    def on_key(self, key_event):
        if key_event.type == o3d.visualization.gui.KeyEvent.Type.UP:
            return
        # 切换类别
        if key_event.key == o3d.visualization.gui.KeyName.Q:
            self.on_pre_category_btn_clicked()
            return
        if key_event.key == o3d.visualization.gui.KeyName.E:
            self.on_next_category_btn_clicked()
            return

        # 切换场景
        if key_event.key == o3d.visualization.gui.KeyName.UP:
            self.on_pre_scene_btn_clicked()
            return
        if key_event.key == o3d.visualization.gui.KeyName.DOWN:
            self.on_next_scene_btn_clicked()
            return

        # 切换视角
        if key_event.key == o3d.visualization.gui.KeyName.RIGHT:
            self.on_next_view_btn_clicked()
            return
        if key_event.key == o3d.visualization.gui.KeyName.LEFT:
            self.on_pre_view_btn_clicked()
            return

        # 可见性
        if key_event.key == o3d.visualization.gui.KeyName.A:
            self.on_show_pcd1_checked(not self.show_pcd1_checked)
            return
        if key_event.key == o3d.visualization.gui.KeyName.S:
            self.on_show_pcd2_checked(not self.show_pcd2_checked)
            return
        if key_event.key == o3d.visualization.gui.KeyName.D:
            self.on_show_ibs_checked(not self.show_ibs_checked)
            return

    def on_dialog_ok(self):
        self.window.close_dialog()

    def init_data_dir_editor_area(self):
        self.data_dir_editor_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))

        self.pcd_pred1_dir_editor_text = gui.Label("pcd_pred1_dir")
        self.pcd_pred1_dir_editor = gui.TextEdit()
        self.pcd_pred1_dir_editor.text_value = self.specs.get("path_options").get("geometries_dir").get("pcd_pred1_dir")

        self.pcd_pred2_dir_editor_text = gui.Label("pcd_pred2_dir")
        self.pcd_pred2_dir_editor = gui.TextEdit()
        self.pcd_pred2_dir_editor.text_value = self.specs.get("path_options").get("geometries_dir").get("pcd_pred2_dir")

        self.data_dir_confirm_btn = gui.Button("confirm")
        self.data_dir_confirm_btn.set_on_clicked(self.on_data_dir_comfirm_btn_clicked)

        self.data_dir_editor_layout.add_child(self.pcd_pred1_dir_editor_text)
        self.data_dir_editor_layout.add_child(self.pcd_pred1_dir_editor)
        self.data_dir_editor_layout.add_fixed(self.em / 2)
        self.data_dir_editor_layout.add_child(self.pcd_pred2_dir_editor_text)
        self.data_dir_editor_layout.add_child(self.pcd_pred2_dir_editor)
        self.data_dir_editor_layout.add_fixed(self.em / 2)
        self.data_dir_editor_layout.add_child(self.data_dir_confirm_btn)

    def init_geometry_select_area(self):
        self.geometry_select_layout = gui.Vert(self.em / 2, gui.Margins(self.em, self.em, self.em, self.em))

        # 类别
        self.selected_category = -1
        print("seleted category: ", self.selected_category)
        self.category_selector_layout = gui.Vert()
        self.category_selector_text = gui.Label("category")
        self.category_selector = gui.Combobox()
        self.category_selector.set_on_selection_changed(self.on_category_selection_changed)
        self.category_selector_layout.add_child(self.category_selector_text)
        self.category_selector_layout.add_child(self.category_selector)

        # 场景
        self.selected_scene = -1
        print("seleted scene: ", self.selected_scene)
        self.scene_selector_layout = gui.Vert()
        self.scene_selector_text = gui.Label("scene")
        self.scene_selector = gui.Combobox()
        self.scene_selector.set_on_selection_changed(self.on_scene_selection_changed)
        self.scene_selector_layout.add_child(self.scene_selector_text)
        self.scene_selector_layout.add_child(self.scene_selector)

        # 场景
        self.selected_view = -1
        print("seleted view: ", self.selected_view)
        self.view_selector_layout = gui.Vert()
        self.view_selector_text = gui.Label("view")
        self.view_selector = gui.Combobox()
        self.view_selector.set_on_selection_changed(self.on_view_selection_changed)
        self.view_selector_layout.add_child(self.view_selector_text)
        self.view_selector_layout.add_child(self.view_selector)

        # 确认
        self.btn_load = gui.Button("load data")
        self.btn_load.set_on_clicked(self.on_load_btn_clicked)

        self.geometry_select_layout.add_child(self.category_selector_layout)
        self.geometry_select_layout.add_child(self.scene_selector_layout)
        self.geometry_select_layout.add_child(self.view_selector_layout)
        self.geometry_select_layout.add_child(self.btn_load)

    def init_data_switch_area(self):
        self.data_switch_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))

        self.category_switch_area = gui.Vert()
        self.category_switch_text = gui.Label("switch category")
        self.btn_pre_category = gui.Button("previous category")
        self.btn_pre_category.set_on_clicked(self.on_pre_category_btn_clicked)
        self.btn_next_category = gui.Button("next category")
        self.btn_next_category.set_on_clicked(self.on_next_category_btn_clicked)
        self.category_switch_area.add_child(self.category_switch_text)
        self.category_switch_area.add_child(self.btn_pre_category)
        self.category_switch_area.add_child(self.btn_next_category)

        self.scene_switch_area = gui.Vert()
        self.scene_switch_text = gui.Label("switch scene")
        self.btn_pre_scene = gui.Button("previous scene")
        self.btn_pre_scene.set_on_clicked(self.on_pre_scene_btn_clicked)
        self.btn_next_scene = gui.Button("next scene")
        self.btn_next_scene.set_on_clicked(self.on_next_scene_btn_clicked)
        self.scene_switch_area.add_child(self.scene_switch_text)
        self.scene_switch_area.add_child(self.btn_pre_scene)
        self.scene_switch_area.add_child(self.btn_next_scene)

        self.view_switch_area = gui.Vert()
        self.view_switch_text = gui.Label("switch view")
        self.btn_pre_view = gui.Button("previous view")
        self.btn_pre_view.set_on_clicked(self.on_pre_view_btn_clicked)
        self.btn_next_view = gui.Button("next view")
        self.btn_next_view.set_on_clicked(self.on_next_view_btn_clicked)
        self.view_switch_area.add_child(self.view_switch_text)
        self.view_switch_area.add_child(self.btn_pre_view)
        self.view_switch_area.add_child(self.btn_next_view)

        self.data_switch_layout.add_child(self.category_switch_area)
        self.data_switch_layout.add_fixed(self.em)
        self.data_switch_layout.add_child(self.scene_switch_area)
        self.data_switch_layout.add_fixed(self.em)
        self.data_switch_layout.add_child(self.view_switch_area)

    def init_visible_control_area(self):
        self.visible_control_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))

        self.visible_text = gui.Label("visible")

        self.visible_control_checkbox_layout = gui.Horiz()
        self.show_pcd1_checkbox_text = gui.Label("pcd1")
        self.show_pcd2_checkbox_text = gui.Label("pcd2")
        self.show_ibs_checkbox_text = gui.Label("ibs")
        self.show_pcd1_checkbox = gui.Checkbox("")
        self.show_pcd1_checkbox.checked = True
        self.show_pcd1_checked = True
        self.show_pcd2_checkbox = gui.Checkbox("")
        self.show_pcd2_checkbox.checked = True
        self.show_pcd2_checked = True
        self.show_ibs_checkbox = gui.Checkbox("")
        self.show_pcd1_checkbox.set_on_checked(self.on_show_pcd1_checked)
        self.show_pcd2_checkbox.set_on_checked(self.on_show_pcd2_checked)
        self.show_ibs_checkbox.set_on_checked(self.on_show_ibs_checked)

        self.visible_control_checkbox_layout.add_child(self.show_pcd1_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_pcd1_checkbox)
        self.visible_control_checkbox_layout.add_stretch()
        self.visible_control_checkbox_layout.add_child(self.show_pcd2_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_pcd2_checkbox)
        self.visible_control_checkbox_layout.add_stretch()
        self.visible_control_checkbox_layout.add_child(self.show_ibs_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_ibs_checkbox)

        self.visible_control_layout.add_child(self.visible_text)
        self.visible_control_layout.add_child(self.visible_control_checkbox_layout)

    def init_info_area(self):
        self.data_info_layout = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))

        self.category_info = gui.Label("category: {}".format(""))
        self.scene_info = gui.Label("scene: {}".format(""))
        self.view_info = gui.Label("view: {}".format(""))
        self.theta = gui.Label("theta: {}".format(""))
        self.phi = gui.Label("phi: {}".format(""))

        self.data_info_layout.add_child(self.category_info)
        self.data_info_layout.add_child(self.scene_info)
        self.data_info_layout.add_child(self.view_info)
        self.data_info_layout.add_child(self.theta)
        self.data_info_layout.add_child(self.phi)

    def get_geometry_path(self):
        pcd_complete_dir = self.specs.get("path_options").get("geometries_dir").get("pcd_complete_dir")
        pcd_partial_dir = self.specs.get("path_options").get("geometries_dir").get("pcd_partial_dir")
        pcd_pred1_dir = self.pcd_pred1_dir_editor.text_value
        pcd_pred2_dir = self.pcd_pred2_dir_editor.text_value
        IBS_dir = self.specs.get("path_options").get("geometries_dir").get("IBS_dir")

        category = self.category_selector.get_item(self.selected_category)
        scene = self.scene_selector.get_item(self.selected_scene)
        view = self.view_selector.get_item(self.selected_view)

        pcd_complete_1_filename = "{}_0.ply".format(scene)
        pcd_complete_2_filename = "{}_1.ply".format(scene)
        pcd_partial_1_filename = "{}_0.ply".format(view)
        pcd_partial_2_filename = "{}_1.ply".format(view)
        pcd_pred1_1_filename = "{}_0.ply".format(view)
        pcd_pred1_2_filename = "{}_1.ply".format(view)
        pcd_pred2_1_filename = "{}_0.ply".format(view)
        pcd_pred2_2_filename = "{}_1.ply".format(view)
        IBS_filename = "{}.obj".format(scene)

        pcd_complete_1_path = os.path.join(pcd_complete_dir, category, pcd_complete_1_filename)
        pcd_complete_2_path = os.path.join(pcd_complete_dir, category, pcd_complete_2_filename)
        pcd_partial_1_path = os.path.join(pcd_partial_dir, category, pcd_partial_1_filename)
        pcd_partial_2_path = os.path.join(pcd_partial_dir, category, pcd_partial_2_filename)
        pcd_pred1_1_path = os.path.join(pcd_pred1_dir, category, pcd_pred1_1_filename)
        pcd_pred1_2_path = os.path.join(pcd_pred1_dir, category, pcd_pred1_2_filename)
        pcd_pred2_1_path = os.path.join(pcd_pred2_dir, category, pcd_pred2_1_filename)
        pcd_pred2_2_path = os.path.join(pcd_pred2_dir, category, pcd_pred2_2_filename)
        IBS_path = os.path.join(IBS_dir, category, IBS_filename)

        geometry_path_dict = {
            self.key_pcd_complete_1: pcd_complete_1_path,
            self.key_pcd_complete_2: pcd_complete_2_path,
            self.key_pcd_partial_1: pcd_partial_1_path,
            self.key_pcd_partial_2: pcd_partial_2_path,
            self.key_pcd_pred1_1: pcd_pred1_1_path,
            self.key_pcd_pred1_2: pcd_pred1_2_path,
            self.key_pcd_pred2_1: pcd_pred2_1_path,
            self.key_pcd_pred2_2: pcd_pred2_2_path,
            self.key_IBS: IBS_path,
        }

        return geometry_path_dict

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

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    # 获取配置参数
    config_filepath = 'configs/Interaction_visualizer.json'
    specs = path_utils.read_config(config_filepath)

    app = App(specs)
    app.run()
