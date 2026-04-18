# Active Implicit Object Reconstruction using Uncertainty-guided Next-Best-View Optimization

## 基本信息
- **论文编号**: arXiv 2309.13240
- **分类**: [Unlabeled] / 主动重建·隐式场
- **发表**: IEEE RA-L 2023 | Accepted Jul 2023
- **机构**: Harbin Institute of Technology Shenzhen, Shenzhen University General Hospital
- **关键词**: [[主动重建]], [[隐式占据场]], [[次佳视点优化]], [[连续流形优化]]

## 核心问题
隐式表示下的主动物体重建：传统NBV方法在有限候选集中选择，无法在连续流形上优化；基于学习的不确定性方法需要预定义参考视角。如何无需候选视角集直接优化NBV？

## 方法贡献
**Uncertainty-guided Active Implicit Reconstruction**：
1. **隐式占据场**：用Instant-NGP架构构建占据场，结合包围盒先验和深度监督
2. **采样式熵不确定性**：从占据概率场直接采样，计算遮挡感知熵作为视角信息增益度量
3. **可微NBV连续优化**：利用隐式表示可微性，用梯度下降直接在连续流形上优化NBV位姿
4. **Top-N策略**：兼顾局部细节和全局注意力的重建策略

## 实验结果
- 仿真+真实世界机器人平台验证
- 在重建完整性上显著超越迭代式基线
- 适应性更强（无需预定义候选集）

## 创新点（一句话）
用隐式占据场的可微性直接在连续流形上梯度下降优化NBV，消除候选集预定义限制，实现更自适应的主动重建。

## Idea 价值
- **可微表示→可微规划**：从离散候选集到连续优化的范式升级，在3DGS也可实现
- 与 [[GauSS-MI]] 对比：都是主动重建，GauSS-MI用互信息量化视觉质量，本文用熵量化几何不确定性
- 3DGS可微性：3DGS同样可微，可以直接继承本文的连续NBV优化框架

## 相关工作联系
- 主动重建：[[GauSS-MI]]（3DGS+香农互信息），[[2022-UncertaintyNeRF-ActiveRecon]]（NeRF版NBV）
- 视角规划：[[2024-PRVNet-ViewPlanning]]（预测所需视角数量）
- 占据场：与 [[2022-GIGA]] 的隐式可供性场同属隐式场机器人应用
