# SplatPose: Geometry-Aware 6-DoF Pose Estimation from Single RGB Image via 3D Gaussian Splatting

## 基本信息
- **论文编号**: 0143
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2503.05174 | 2025
- **机构**: Harbin Institute of Technology（哈工大 + 郑州研究院 + 嘉宽先进技术中心）
- **关键词**: [[6-DoF位姿估计]], [[3DGS]], [[单目RGB]], [[DARS-Net]]

## 核心问题
单张 RGB 图像的 6-DoF 位姿估计：传统方法依赖深度传感器或多视角，存在部署成本高和旋转模糊问题。如何仅用单 RGB 实现准确的 6-DoF 位姿？

## 方法贡献
**SplatPose**：基于 3DGS 的单 RGB 6-DoF 位姿估计框架
1. **DARS-Net（Dual-Attention Ray Scoring Network）**：将射线评分解耦为位置分数和方向分数，分别处理平移和旋转，消除旋转模糊
2. **粗到细位姿估计**：先用 DARS-Net 得粗位姿，再用 3DGS 渲染合成视图做关键点匹配精化
3. **3DGS 作为渲染器**：在给定粗位姿渲染深度图，将 2D-2D 对应提升到 2D-3D，通过 PnP+RANSAC 得最终位姿

## 关键技术
- DARS-Net：双注意力机制分离位移方向推理，利用 DINOv2 特征
- 粗定位：从 3DGS 射线采样 top-k 高位置/高方向分数射线
- 精化：渲染图与查询图特征点匹配，PnP 求解最终位姿

## 实验结果（Novel View Synthesis Benchmarks）
- 在 3 个公开数据集上 SOTA 6-DoF 位姿精度
- 单目 RGB 性能可与深度/多视角方法媲美
- 对高达 55° 旋转误差的初始位姿具有鲁棒性

## 创新点（一句话）
DARS-Net 将射线评分解耦为位置和方向两个分量，从根本上消除 3DGS 椭球体射线采样的旋转模糊，实现高精度单 RGB 6-DoF 位姿估计。

## Idea 价值
- **3DGS 作为定位地图**：预建 3DGS 地图 + 单图查询定位，是 SLAM 重定位的新范式
- DARS-Net 的位置/方向解耦可迁移到其他 3DGS 位姿估计问题（如 SLAM 跟踪）
- 与 [[GSFeatLoc]] 类比：都用特征对应做 3DGS 定位，SplatPose 多了 DARS-Net 初始化

## 相关工作联系
- 3DGS 定位：[[GSFeatLoc]]（特征对应）
- 位姿估计视角：[[SurgCalib]]（可微渲染优化位姿）
- 多相机 SLAM：[[MCGS-SLAM]]（多视角 3DGS 建图）
