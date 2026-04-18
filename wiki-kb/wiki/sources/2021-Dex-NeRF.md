# Dex-NeRF: Using a Neural Radiance Field to Grasp Transparent Objects

## 基本信息
- **论文编号**: 0078
- **分类**: [Robotics]
- **发表**: CoRL 2021
- **机构**: UC Berkeley
- **关键词**: [[NeRF]], [[透明物体抓取]], [[深度估计]], [[机器人感知]]

## 核心问题
透明物体（杯子、瓶子等）的机器人抓取：RGB-D 相机对透明物体深度失效（折射、反射导致深度噪声大），传统抓取检测方法无法处理。

## 方法贡献
**Dex-NeRF**：用 NeRF 重建透明物体获得可靠深度，用于抓取规划
1. **NeRF 重建**：多视角 RGB 图像训练 NeRF，隐式建模透明物体几何
2. **深度提取**：从 NeRF volume rendering 提取可靠的深度图，绕过 RGB-D 深度失效问题
3. **抓取检测**：将 NeRF 深度输入 GQ-CNN 生成抓取候选姿态

## 关键技术
- NeRF（Vanilla）在可见光谱下的透明物体建模
- 体积渲染深度：从 NeRF 的 alpha-compositing 提取深度期望值
- GQ-CNN：基于深度图的抓取质量评估网络

## 实验结果
- 在真实透明物体（杯、瓶）的抓取实验
- 抓取成功率：Dex-NeRF 88%，RGB-D 直接法 <60%
- 重建深度比 RGB-D 传感器准确 3-5×

## 创新点（一句话）
NeRF 的透明物体重建能力使机器人首次可靠抓取透明物体，绕过 RGB-D 传感器的透明性失效。

## Idea 价值
- **3DGS 替代 NeRF**：3DGS 渲染速度快 100×，可支持实时的透明物体感知
- 透明物体是 3DGS 的挑战（传统 3DGS 假设朗伯表面），透明 3DGS 是开放问题
- 物理属性建模 + 3DGS：折射、反射的显式 Gaussian 表示
- 与 [[EndoGSLAM]] 类似：医疗场景（内窥镜）也有光泽/透明表面挑战

## 相关工作联系
- NeRF 机器人应用：[[NeRF-Supervision]]（描述符）、[[GIGA]]（抓取）
- 3DGS SLAM：[[GS-SLAM]]（可微渲染感知）
- 医疗感知：[[EndoGSLAM]]（光泽表面）
