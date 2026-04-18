# Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping

## 基本信息
- **论文编号**: 0002
- **分类**: [SLAM]
- **发表**: arXiv 2311.16728 | CVPR 2024
- **机构**: HKUST, Sun Yat-sen University
- **关键词**: [[3D Gaussian Splatting]], [[Dense Visual SLAM]], [[单目SLAM]], [[Hyper Primitives]]

## 核心问题
现有神经渲染 SLAM 完全依赖隐式表示，计算资源消耗大，无法在便携设备（如 Jetson AGX Orin）上实时运行。

## 方法贡献
1. **Hyper Primitives Map**：融合显式几何特征（点云+ORB特征+旋转/缩放/SH系数）和隐式光度特征，两者协同优化
2. **Gaussian-Pyramid 学习**：渐进式多尺度特征学习，提升高保真度 mapping 质量
3. **支持三种相机**：单目、双目、RGB-D，首个统一框架

## 关键技术
- 追踪：Factor graph solver（显式几何）
- 建图：反向传播（隐式光度特征）
- 渲染：3DGS 光栅化（而非光线采样）
- 几何密化：基于几何特征的主动密化策略

## 实验结果
- 数据集：Replica, TUM-RGBD, EuRoC, outdoor datasets
- PSNR 比 SOTA 高 30%（Replica）
- 渲染速度：最高 **1000 FPS**
- 可在嵌入式平台（Jetson AGX Orin）实时运行

## 创新点（一句话）
Hyper Primitives Map 将显式几何追踪与隐式光度建图解耦，首次实现可在嵌入式设备运行的单目/双目/RGB-D 统一光真实感 SLAM。

## Idea 价值
- **嵌入式部署** 是重要方向，显式+隐式混合表示比纯隐式更轻量
- Gaussian-Pyramid 多尺度训练策略可迁移到其他 3DGS 任务
- [[单目SLAM]] 方向：不依赖深度相机仍能高质量建图

## 相关工作联系
- 同期：[[GS-SLAM]], [[SplaTAM]], [[Gaussian-SLAM]]
- 优势：支持单目；可嵌入式运行
