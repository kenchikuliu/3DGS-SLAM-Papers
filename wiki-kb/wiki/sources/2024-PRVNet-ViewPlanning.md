# How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF?

## 基本信息
- **论文编号**: arXiv 2310.00684
- **分类**: [Unlabeled] / 主动重建·视角规划
- **发表**: arXiv 2310.00684 | Feb 2024
- **机构**: University of Bonn, Intel Asia-Pacific, Lamarr Institute for ML & AI
- **关键词**: [[NeRF主动重建]], [[所需视角数预测]], [[一次性视角规划]], [[PRVNet]]

## 核心问题
主动NeRF重建的停止准则问题：何时收集足够的视角？迭代式NBV因每次收集新图像都需重训NeRF，效率极低。如何预测重建特定物体所需的视角数量，实现一次性非迭代规划？

## 方法贡献
**PRVNet（Prediction of Required Views）+ 一次性视角规划**：
1. **PRVNet**：从初始多帧RGB图像提取特征，预测重建该物体所需的视角数量（回归问题）
2. **物体复杂度感知**：复杂多彩物体需要更多视角，简单单色物体需要更少（ShapeNet验证）
3. **Tammes球面配置**：在半球面上均匀排布N个视角（Tammes问题最优解）
4. **全局最短路径规划**：在预测的视角集合内规划最短移动路径，一次性执行

## 实验结果
- 仿真+真实机器人验证
- 重建质量与迭代式基线相当或更好
- 显著降低移动代价和规划时间
- 开源：https://github.com/psc0628/NeRF-PRV

## 创新点（一句话）
PRVNet预测物体复杂度自适应所需视角数，配合球面均匀配置和最短路径规划，实现高效非迭代主动NeRF重建。

## Idea 价值
- **重建效率思路**：不是优化"去哪"，而是预测"要去几个地方"，根本性提升效率
- 物体复杂度→视角数：这个映射关系在3DGS重建中同样适用
- 与 [[2024-OSVP-OneShot]] 互补：本文预测视角数，后者预测视角集合

## 相关工作联系
- 一次性视角规划：[[2024-OSVP-OneShot]]（隐式表示一次性视角预测）
- 主动重建：[[GauSS-MI]]，[[2023-ActiveImplicit-NBV]]（主动重建系列）
- NeRF重建质量：与 [[2022-UncertaintyNeRF-ActiveRecon]] 同属NBV规划问题
