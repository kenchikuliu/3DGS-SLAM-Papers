# GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

## 基本信息
- **论文编号**: 0001
- **分类**: [SLAM]
- **发表**: arXiv 2311.11700 | CVPR 2024
- **机构**: Shanghai AI Lab, Fudan, HKUST, Northwestern Poly
- **关键词**: [[3D Gaussian Splatting]], [[Dense Visual SLAM]], [[RGB-D SLAM]], [[实时重建]]

## 核心问题
NeRF-based SLAM 渲染速度慢、无法实时。如何将 3DGS 的快速光栅化优势引入 SLAM 系统，同时保持定位精度？

## 方法贡献
1. **首个 3DGS-based RGB-D 密集 SLAM**：用 3DGS 场景表示替代 NeRF，渲染速度提升 100×
2. **自适应 3D Gaussian 扩展策略**：动态增删 noisy Gaussians，高效重建新观测区域，同时优化历史区域
3. **粗到细相机追踪**：先用低分辨率初始估计位姿，再用高分辨率精化，兼顾速度与精度

## 关键技术
- 场景表示：3D Gaussians $G_i = (X_i, \Sigma_i, \Lambda_i, Y_i)$（位置、协方差、不透明度、球谐系数）
- 渲染：可微光栅化，α-blending
- 协方差参数化：$\Sigma = RSS^TR^T$（避免直接优化非正定矩阵）
- 追踪：解析梯度 + RGB-D 渲染损失反向传播

## 实验结果
- 数据集：Replica, TUM-RGBD
- 速度：8.43 FPS（比 NeRF-based SLAM 快约 100×）
- 渲染：386 FPS（vs ESLAM 3 FPS, Point-SLAM 3 FPS）
- 达到 SOTA tracking + mapping + rendering 三项平衡

## 创新点（一句话）
首次将 3D Gaussian Splatting 引入密集 SLAM，用自适应扩展策略解决动态场景建图问题，实现实时 RGB-D 重建。

## 相关工作联系
- 对比：[[iMAP]], [[NICE-SLAM]], [[ESLAM]], [[Point-SLAM]]（均为 NeRF-based）
- 基础：[[3DGS]]（Kerbl et al. 2023）
- 后续被：[[CG-SLAM]], [[RTG-SLAM]], [[SplaTAM]] 等引用或改进
