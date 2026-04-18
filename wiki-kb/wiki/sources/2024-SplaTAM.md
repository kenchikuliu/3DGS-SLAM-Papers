# SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

## 基本信息
- **论文编号**: 0003
- **分类**: [SLAM]
- **发表**: arXiv 2312.02126 | CVPR 2024
- **机构**: CMU, MIT
- **关键词**: [[3D Gaussian Splatting]], [[Dense Visual SLAM]], [[RGB-D SLAM]], [[显式体素表示]]

## 核心问题
如何利用显式体积表示（3DGS）实现高保真密集 RGB-D SLAM，同时支持快速渲染和稠密优化？

## 方法贡献
1. **首次将显式 3DGS 体积表示用于密集 RGB-D SLAM**（对比 NeRF 的隐式表示）
2. **Silhouette Mask**：用轮廓掩码优雅捕获场景密度，快速判断区域是否已建图，指导 Gaussian 扩展
3. **结构化地图扩展**：基于 Silhouette 向未建图区域添加新 Gaussian

## 关键技术
- 追踪：在线优化，将渲染损失对相机位姿求梯度
- 建图：优化 Gaussian 参数（位置、颜色、不透明度、协方差）
- 渲染：400 FPS（分辨率 876×584）
- ATE RMSE：0.6 cm

## 实验结果
- 数据集：Replica, ScanNet 等
- 相机位姿估计：ATE RMSE 0.6 cm
- 比 SOTA baseline 提升 2×（map construction）
- 新视角合成 PSNR：27.4 dB（训练视角），24.2 dB（新视角）

## 创新点（一句话）
用 Silhouette Mask 作为场景密度指示器，驱动 3DGS 地图结构化扩展，实现 400 FPS 渲染的密集 RGB-D SLAM。

## Idea 价值
- **Silhouette Mask** 思路简洁高效，可用于任何需要判断"已建图/未建图"的场景
- 显式表示天然支持地图编辑、可视化、几何查询

## 相关工作联系
- 同期首发之一：与 [[GS-SLAM]]、[[Gaussian-SLAM]] 并列为 3DGS-SLAM 开山之作
- 改进者：[[Splat-SLAM]]（RGB-only 全局优化版本）
