# CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

## 基本信息
- **论文编号**: 0008
- **分类**: [SLAM]
- **发表**: arXiv 2403.16095 | 2024
- **机构**: Zhejiang University (State Key Lab of CAD&CG), ZJU-UIUC
- **关键词**: [[3D Gaussian Splatting]], [[不确定性建模]], [[Dense Visual SLAM]], [[几何稳定性]]

## 核心问题
现有 3DGS-SLAM 构建的 Gaussian Field 一致性差、几何不稳定，导致追踪精度下降。如何构建高一致性、几何稳定的 Gaussian Field？

## 方法贡献
1. **Uncertainty-aware 3D Gaussian Field**：为每个 Gaussian 建模深度不确定性，指导优化过程
2. **一致性保证技术**：多种技术组合，确保 Gaussian Field 在追踪和建图中保持几何稳定
3. **深度不确定性模型**：在优化中优先选择高价值 Gaussian 基元，提升追踪效率

## 关键技术
- 不确定性建模：每个 Gaussian 附带深度不确定性估计
- 一致性约束：抑制 Gaussian 漂移和几何退化
- 选择性优化：不确定性高的区域优先优化

## 实验结果
- 追踪速度：**15 Hz**
- 数据集：多个 benchmark
- 追踪和建图性能均优于现有方法

## 创新点（一句话）
不确定性感知 3D Gaussian Field 通过深度不确定性建模和选择性优化，系统提升 3DGS-SLAM 的几何一致性和追踪精度。

## Idea 价值
- **不确定性建模** 是提升 SLAM 鲁棒性的重要工具，3DGS 天然可以附加不确定性属性
- 可扩展：不确定性不仅可用于深度，也可用于颜色、语义特征
- 与 [[RTG-SLAM]] 的 stable/unstable 分类有异曲同工之妙

## 相关工作联系
- 同机构：浙大 CAD&CG 也是 [[RTG-SLAM]] 的主要机构
- 互补方向：不确定性 vs 稳定性分类
