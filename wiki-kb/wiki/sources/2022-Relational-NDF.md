# Relational-NDF: SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields

## 基本信息
- **论文编号**: 0074
- **分类**: [Robotics]
- **发表**: CoRL 2022
- **机构**: CMU
- **关键词**: [[SE(3)等变]], [[神经描述符场]], [[物体重排]], [[少样本学习]]

## 核心问题
机器人操作中的物体重排（rearrangement）：如何让机器人从少量示范中学习"将物体 A 放到物体 B 的指定相对位置"的技能？关键挑战是泛化到新物体实例。

## 方法贡献
**Relational-NDF**：将 Neural Descriptor Fields（NDF）扩展到关系级别
1. **关系描述符**：同时编码两个物体局部几何的关系，而非单一物体
2. **SE(3) 等变性**：相对姿态估计对全局旋转不敏感，支持任意朝向泛化
3. **少样本迁移**：5 个示范即可迁移到新类别物体

## 关键技术
- Neural Descriptor Fields (NDF)：在点云空间中学习 SE(3) 等变特征场
- 关系描述符：联合编码 query 点在两物体坐标系中的特征
- Energy-based 优化：最小化关系描述符误差来估计目标位姿

## 实验结果
- mug-on-rack、book-on-shelf、bottle-in-container 三类任务
- 5-shot 条件下成功率 >70%，显著优于非等变基线
- 对物体形状变化（同类不同实例）鲁棒

## 创新点（一句话）
将 NDF 的等变描述符扩展到物体间关系，5-shot 学习多物体相对位姿操作。

## Idea 价值
- **3DGS 视角**：可用 3DGS 的语义特征场替代 NDF 描述符，获得更快的查询速度（光栅化 vs 隐式查询）
- SE(3) 等变性在 3DGS 中的实现：通过球谐函数的旋转等变性可自然支持
- 关系推理 + 几何表示是机器人操作的核心问题，3DGS 可提供可微的几何表示

## 相关工作联系
- 同框架：[[Neural Descriptor Fields]]（基础工作）
- 隐式表示操作：[[NeuralGrasps]]、[[GIGA]]
