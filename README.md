# 项目说明

该项目是金朔硕士学位论文第二个创新点的代码部分。该项目中包含的所有内容均围绕一个点云补全方法展开，该点云补全方法能够在补全过程中使用多个物体间的空间
位置关系约束每一个物体的补全（以损失函数的形式）。

# 项目结构

|-- configs                 // 训练和推理所需的配置文件\
|-- dataset                 // 加载数据所需的文件\
|-- evaluation_utils        // 一些用于评估我们方法效果的代码\
|-- generate_train_data     // 获取训练数据的代码
|-- models                  // 网络模型文件\
|-- render                  // 调用blender进行渲染\
|-- train                   // 所有的训练、测试代码\
|-- utils                   // 一些工具类，包含cd、emd、pointnet++库，需要使用cuda编译并安装\
|-- README.md               // 项目说明\
|-- environment.yml         // 运行代码所需的环境信息\

# 如何运行项目

- 根据environment.yml中的信息配置conda虚拟环境，其中cd、emd、pointnet2-ops需要在github上找合适的开源实现（需要编译cuda代码，因此需要找到与本地cuda版本兼容的实现）
- 以我们的方法SRPCN为例，修改./configs/INTE/train/specs_train_SRPCN_INTE.json中的DataSource为实际的数据集地址，TrainSplit、TestSplit为实际的训练、测试数据目录
- 根据实际情况调整NumEpochs、BatchSize等参数
- 运行train/SRPCN_INTE/train.py

# 如何获取训练所需的数据

该网络是一个点云补全网络，输入是残缺点云，且需要完整点云作为监督。此外，训练阶段还需要计算两物体之间的一些几何结构（IBS、PSVF、IS，这些内容需要看论文），以计算空间关系损失函数。所需的训练数据包括以下几个部分：
- 具有真实遮挡关系的单视角扫描点云，可通过./generate_train_data/get_scan_pcd.py获取
- 完整点云，可通过./generate_train_data/get_pcd_from_mesh.py获取
- ibs，可通过./generate_train_data/get_ibs.py获取
- IS、PSVF（即多物体空隙区域的中轴变换），可通过./generate_train_data/get_medial_axis.py获取

此外，计算ibs、IS、PSVF时需要两物体位于世界坐标系下，而训练时需要将每个物体各自归一化到单位球内（连同上述计算出的IS、PSVF），因此需要调用./generate_train_data/normalize_data.py。在展示结果时，需要使用得到的归一化参数再将两物体变换回世界坐标系下。

