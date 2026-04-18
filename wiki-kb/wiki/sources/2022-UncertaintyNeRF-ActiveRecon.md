# Uncertainty Guided Policy for Active Robotic 3D Reconstruction using Neural Radiance Fields

## 基本信息
- **论文编号**: arXiv 2209.08409
- **分类**: [Unlabeled] / 主动重建·NeRF
- **发表**: arXiv 2209.08409 | Sep 2022
- **机构**: ETH Zürich, Oracle Labs Zürich
- **关键词**: [[主动重建]], [[次佳视点选择]], [[NeRF不确定性]], [[机器人视觉]]

## 核心问题
主动机器人3D重建的下一最佳视点（NBV）选择：现有方法依赖显式点云/体素等表示来估计信息增益，NeRF的隐式性使得直接推理3D几何不确定性更困难。

## 方法贡献
**Uncertainty Guided NeRF Policy**：
1. **射线体积不确定性估计器**：计算NeRF颜色样本权重分布的熵，作为底层3D几何不确定性的代理
2. **不确定性引导NBV策略**：用射线体积不确定性选择下一最佳观测视点
3. **隐式表示优势**：NeRF作为表示不依赖点云/深度传感器，泛化能力更强
4. **验证**：合成和真实数据上均验证方法有效

## 实验结果
- 合成基准数据集：超越依赖显式3D几何的基线
- 可泛化到真实场景
- 首次用NeRF隐式表示解决主动机器人3D重建的NBV问题

## 创新点（一句话）
用NeRF权重分布熵作为几何不确定性代理，实现无需显式3D表示的机器人主动重建视点选择策略。

## Idea 价值
- **前3DGS时代基线**：NeRF不确定性→NBV的思路被后来的3DGS主动重建（如 [[GauSS-MI]]）继承和发展
- 从隐式密度估计不确定性的范式：NeRF权重熵 → 3DGS残差损失概率模型
- 与 [[GauSS-MI]] 的演进关系：同一问题（主动重建视点选择），从NeRF到3DGS的技术升级

## 相关工作联系
- 主动重建：[[GauSS-MI]]（3DGS版本），[[2023-ActiveImplicit-NBV]]（隐式场NBV优化）
- NeRF不确定性：与 [[CG-SLAM]] 的不确定性Gaussian理念一脉相承
