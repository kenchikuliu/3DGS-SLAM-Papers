# 2DGS-SLAM: Globally Consistent RGB-D SLAM with 2D Gaussian Splatting

## 基本信息
- **论文编号**: 0149
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2506.00970 | 2025
- **机构**: University of Bonn, TU Delft（Stachniss 组）
- **关键词**: [[2D Gaussian Splatting]], [[RGB-D SLAM]], [[深度一致性]], [[回环检测]]

## 核心问题
3DGS-SLAM的几何弱点：3DGS椭球体在不同视角渲染的深度不一致，导致位姿优化中深度信息利用率低，几何重建精度差。如何从根本上解决3DGS的深度不一致问题？

## 方法贡献
**2DGS-SLAM**：用2D Gaussian替代3D Gaussian作为SLAM地图表示
1. **2DGS地图表示**：2D圆盘替代3D椭球，射线-圆盘交点计算精确深度（无视角歧义）
2. **基于渲染的位姿估计**：利用2DGS深度一致性推导Jacobian，实现精确frame-to-map位姿跟踪（CUDA实现）
3. **MASt3R回环检测**：利用3D基础模型估计初始相对位姿，ICP精化，位姿图全局优化
4. **局部活跃地图**：Gaussian在活跃/非活跃状态间转换，防止漂移累积

## 实验结果（Replica / ScanNet）
- 跟踪精度优于或与3DGS-SLAM方法持平
- 表面重建质量比3DGS-based方法更一致
- 在真实场景ScanNet上全局一致重建
- 开源：https://github.com/PRBonn/2DGS-SLAM

## 创新点（一句话）
2DGS的深度一致渲染属性从根本上解决了3DGS-SLAM的几何歧义，配合MASt3R回环实现全局一致建图。

## Idea 价值
- **2DGS是3DGS-SLAM的重要变体**：深度一致性对SLAM非常关键（位姿优化依赖深度）
- MASt3R作为回环检测器：大模型泛化能力强，无需场景特定训练
- CUDA Jacobian推导：说明2DGS的可微性质也适合实时SLAM
- 与 [[PLANING]] 对比：同是解决3DGS几何问题，路线不同（2DGS vs 混合表示）

## 相关工作联系
- 几何对比：[[PLANING]]（混合三角-Gaussian表示）
- 3DGS-SLAM基础：[[CG-SLAM]]（不确定性感知）、[[RTG-SLAM]]
- 回环检测：[[GRAND-SLAM]]（多智能体回环）、[[Splat-SLAM]]
