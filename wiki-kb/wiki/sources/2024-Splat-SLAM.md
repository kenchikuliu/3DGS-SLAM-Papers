# Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians

## 基本信息
- **论文编号**: 0010
- **分类**: [SLAM]
- **发表**: arXiv 2405.16544 | 2024
- **机构**: Google, ETH Zürich, TU München, University of Amsterdam
- **关键词**: [[3D Gaussian Splatting]], [[单目SLAM]], [[全局优化]], [[无深度SLAM]]

## 核心问题
RGB-only 3DGS-SLAM 方法（无深度相机）重建质量显著差于基于点云的方法，因为它们既不用全局地图/位姿优化，也不利用单目深度先验。

## 方法贡献
1. **首个 RGB-only 全局优化 3DGS-SLAM**：同时优化全局地图和关键帧位姿
2. **3DGS 地图动态变形**：根据关键帧位姿和深度更新主动变形 3DGS 地图
3. **单目深度估计辅助**：用单目深度估计器精化不准确区域的深度更新

## 关键技术
- 全局位姿图优化（非仅 frame-to-model）
- 深度更新 + 地图变形：动态调整 Gaussian 位置
- 单目深度估计器（如 MiDaS/Depth Anything）作为几何先验

## 实验结果
- 数据集：Replica, TUM-RGBD, ScanNet
- 在 tracking/mapping/rendering 上优于或持平于现有 RGB-only SLAM
- 地图尺寸小、运行速度快

## 创新点（一句话）
通过全局位姿优化 + 3DGS 地图动态变形 + 单目深度先验，首次让 RGB-only 3DGS-SLAM 达到与 RGB-D 方法可比的重建质量。

## Idea 价值
- **无深度传感器**方向：单目相机是最广泛的输入，解决单目 3DGS-SLAM 质量问题意义重大
- **全局优化 vs 局部追踪**：大多数 3DGS-SLAM 只做 frame-to-model，全局优化是提升精度的关键
- 地图变形思路：Gaussian 位置可动态调整，与 non-rigid 重建有联系

## 相关工作联系
- 前作：[[SplaTAM]]（名字相似，但本文增加了全局优化）
- Google + ETH 合作：工业界参与，工程成熟度高
- 单目方向同类：[[Photo-SLAM]]（也支持单目，但追踪机制不同）
