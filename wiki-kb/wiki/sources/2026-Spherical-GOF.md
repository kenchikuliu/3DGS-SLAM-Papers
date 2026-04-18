# Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction

## 基本信息
- **论文编号**: 0190
- **分类**: [General]
- **发表**: arXiv 2603.08503 | 2026
- **机构**: Hunan University
- **关键词**: [[全景3DGS]], [[球面渲染]], [[几何一致性]], [[机器人感知]]

## 核心问题
标准 3DGS 为透视相机设计，直接应用于全景（omnidirectional）相机引入畸变和几何不一致。如何将 3DGS 扩展到全景成像？

## 方法贡献
基于 **Gaussian Opacity Fields（GOF）** 的全向渲染框架：
1. **球面光线投射**：直接在单位球面球极空间中进行 GOF 光线采样，而非近似投影
2. **球面保守包围规则**：用于快速球面光线-Gaussian 剔除
3. **球面滤波方案**：自适应 Gaussian footprint 到畸变变化的全景像素采样
4. **OmniRob 数据集**：真实世界机器人全向数据集（UAV + 四足机器人）

## 实验结果（OmniBlender/OmniPhotos）
- 深度重投影误差减少 **57%**（vs 最强基线）
- 循环内点率提升 **21%**
- 更清晰的深度图和更一致的法向图

## 创新点（一句话）
在单位球面球极空间中进行射线-Gaussian 交互，从根本上解决全景 3DGS 的几何不一致问题。

## Idea 价值
- **全景相机** 是机器人/无人机感知的重要传感器，3DGS 全景扩展有实际需求
- 球面空间推理原理可迁移到其他非透视成像模型（鱼眼镜头等）
- OmniRob 数据集：UAV + 四足机器人全向重建 benchmark

## 相关工作联系
- 机器人感知：[[GST-VLA]]（机器人空间感知）
- 全景成像：与 [[Photo-SLAM]] 的多相机支持形成对比（Photo-SLAM 不含全景）
