# Ditto: Building Digital Twins of Articulated Objects from Interaction

## 基本信息
- **论文编号**: 0073
- **分类**: [Robotics]
- **发表**: ICLR 2022
- **机构**: Stanford University, CMU
- **关键词**: [[数字孪生]], [[关节物体]], [[隐式表示]], [[机器人交互]]

## 核心问题
如何从机器人与物体的交互视频中自动构建关节物体的数字孪生（Digital Twin）？传统方法需要 CAD 模型或人工标注，难以扩展。

## 方法贡献
**Ditto**：从交互前后的点云/RGB-D 重建关节物体模型
1. **双阶段重建**：交互前重建外观，交互后推断关节结构（轴、范围）
2. **隐式几何表示**：用 occupancy networks 表示物体形状
3. **关节参数估计**：从交互诱导的运动推断关节类型（旋转/平移）和参数

## 关键技术
- Occupancy Networks 建模物体几何
- 运动分割：区分静止部件与运动部件
- 关节结构推断：从部件间相对运动推断 joint axis 和 range

## 实验结果
- 在合成关节物体数据集上验证
- 关节参数估计误差（轴向角度、位移范围）显著低于基线
- 支持 drawer、door、laptop 等多类关节物体

## 创新点（一句话）
从机器人与物体的物理交互中自动学习关节结构，无需 CAD 模型，实现数字孪生构建。

## Idea 价值
- **3DGS 迁移**：可用 3DGS 替代 occupancy network 表示关节物体，保留方法的关节推断框架
- 关节物体的 4D（运动+外观）建模是开放问题，3DGS 的显式表示更易操控关节参数
- 与 [[ObjectFolder]] 互补：一个建外观（含多模态），一个专注关节运动

## 相关工作联系
- 隐式表示机器人：[[Neural Descriptor Fields]]、[[NeuralGrasps]]
- 数据集：[[ObjectFolder]]（含关节物体）
