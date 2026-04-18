# IG-SLAM: Instant Gaussian SLAM

## 基本信息
- **论文编号**: 0013
- **分类**: [SLAM]
- **发表**: arXiv 2408.01126 | Aug 2024
- **机构**: Center for Image Analysis (OGAM), EEE Department, METU, Turkey
- **关键词**: [[RGB-only SLAM]], [[深度不确定性]], [[即时Gaussian]], [[大规模场景]]

## 核心问题
RGB-only 3DGS-SLAM的规模化问题：现有方法需要深度图监督建图，或缺乏考虑环境规模的训练设计。如何只用RGB输入实现10fps单进程实时密集3DGS SLAM？

## 方法贡献
**IG-SLAM**（Instant Gaussian SLAM）：
1. **密集SLAM追踪**：鲁棒密集位姿估计 + 精炼密集深度图（从追踪生成）
2. **深度不确定性感知建图**：用深度不确定性提高3D重建对噪声的鲁棒性
3. **decay策略**：地图优化中的衰减策略增强收敛，支持高帧率（单进程10fps）
4. **全局Bundle Adjustment**：减少漂移
5. **大规模验证**：在Replica/TUM-RGBD/ScanNet/EuRoC均验证，特别在EuRoC大规模序列

## 实验结果
- 单进程 **10 fps** 运行
- Replica/TUM-RGBD/ScanNet/EuRoC全系列验证
- 与SOTA RGB-only系统（Photo-SLAM等）竞争性性能，EuRoC大规模序列表现优异

## 创新点（一句话）
密集追踪生成深度图 + 深度不确定性感知3DGS建图 + decay收敛策略，实现RGB-only的10fps单进程密集3DGS SLAM。

## Idea 价值
- **深度不确定性→建图质量**：追踪预测的深度存在误差，不确定性权重可减少噪声影响，与 [[CG-SLAM]] 的Gaussian不确定性思路一致
- 单进程架构：10fps单进程意味着部署更简单，硬件要求更低
- EuRoC验证：含快速运动和大场景，验证了实际鲁棒性

## 相关工作联系
- RGB-only SLAM：[[2024-Photo-SLAM]]（1000 FPS Hyper Primitives）、[[2024-Splat-SLAM]]（全局优化）
- 深度不确定性：[[2024-CG-SLAM]]（浙大，Gaussian不确定性场）
- 大规模：[[2024-RTG-SLAM]]（SIGGRAPH，大场景实时）
