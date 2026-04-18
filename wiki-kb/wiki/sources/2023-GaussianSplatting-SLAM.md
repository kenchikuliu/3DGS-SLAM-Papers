# Gaussian Splatting SLAM

## 基本信息
- **论文编号**: arXiv 2312.06741
- **分类**: [Unlabeled→SLAM] / 3DGS SLAM核心论文
- **发表**: arXiv 2312.06741 | Apr 2024
- **机构**: Imperial College London（Dyson Robotics Laboratory + Software Performance Optimisation Group）
- **关键词**: [[单目SLAM]], [[3DGS]], [[在线重建]], [[Lie群追踪]]

## 核心问题
首个基于3DGS的单目视觉SLAM系统：原始3DGS需要离线SfM获取已知相机位姿，无法在线运行。如何让3DGS成为完整SLAM系统（实时追踪+高质量建图）？

## 方法贡献
**Gaussian Splatting SLAM**（Imperial College）：
1. **Lie群解析Jacobian**：推导相机位姿相对于3DGS地图的解析Jacobian，实现直接位姿优化（无需SfM初始化）
2. **各向同性Gaussian正则化**：约束Gaussian形状以保持增量式重建的几何一致性
3. **几何验证**：处理增量式重建中的歧义性，保证几何正确性
4. **Gaussian资源分配与剪枝**：动态管理Gaussian数量，保持精准相机追踪
5. **单目+RGB-D统一**：单目是最困难设置，RGB-D无缝扩展

## 实验结果
- 3fps实时运行（在线，单目）
- 新视角合成和轨迹估计均达到SOTA
- 能精确重建细薄结构（电线）和透明物体（玻璃杯边缘）
- 比基于map-centric的NeRF SLAM更大的相机位姿收敛域

## 创新点（一句话）
首次将3DGS作为单目SLAM系统的唯一3D表示，通过Lie群解析Jacobian实现直接相机位姿优化，统一追踪、建图和高质量渲染。

## Idea 价值
- **里程碑意义**：继SplaTAM（2023.12）之后，Imperial College同期独立完成的3DGS SLAM，标志着3DGS-SLAM进入爆发期
- Lie群+Gaussian：将经典SLAM追踪技术（SE(3)李代数）与3DGS无缝结合
- 单目最难：只用RGB做SLAM是最具挑战的设置，本文验证了3DGS的可行性
- 与其他[SLAM]论文的对比：GS-SLAM/SplaTAM用RGB-D，本文单目，Photo-SLAM追求速度，本文追求质量

## 相关工作联系
- 同类核心论文：[[2024-GS-SLAM]]，[[2024-SplaTAM]]，[[2024-Photo-SLAM]]（[SLAM]分类前10篇）
- 位姿估计：[[2025-SplatPose]]（已知地图定位），[[2025-SmallGS]]（小基线位姿估计）
- 多模态扩展：[[2024-LIV-GaussMap]]（LiDAR-Visual-Inertial 3DGS）
