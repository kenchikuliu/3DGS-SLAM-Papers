# NEDS-SLAM: A Novel Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting

## 基本信息
- **论文编号**: 0006
- **分类**: [SLAM]
- **发表**: arXiv 2403.11679 | 2024
- **机构**: Harbin Institute of Technology
- **关键词**: [[语义SLAM]], [[3D Gaussian Splatting]], [[特征压缩]], [[异常值过滤]]

## 核心问题
3DGS-based 语义 SLAM 面临两个挑战：1) 预训练分割头的语义空间不一致性导致错误估计；2) 高维语义特征直接嵌入 Gaussian 导致内存爆炸。

## 方法贡献
1. **Spatially Consistent Feature Fusion (SCFF)**：解决预训练分割头在不同帧间语义不一致问题
2. **轻量级 Encoder-Decoder**：将高维语义特征压缩为紧凑 3DGS 表示，大幅降低内存消耗
3. **Virtual Camera View Pruning**：用虚拟相机视角检测并删除异常 Gaussians，提升场景质量

## 关键技术
- SCFF：多帧语义特征融合，减少单帧分割噪声
- 特征蒸馏：轻量编码器将语义特征压缩入 Gaussian 参数
- 异常值剪枝：虚拟相机从多视角检测不合理 Gaussians

## 实验结果
- 数据集：Replica, ScanNet
- 超越现有密集语义 SLAM 方法（mapping + tracking 精度）
- 3D 语义建图质量出色

## 创新点（一句话）
通过空间一致特征融合 + 轻量编码器 + 虚拟视角剪枝，系统性解决语义 3DGS-SLAM 的三大工程难题。

## Idea 价值
- **语义特征压缩** 是语义 3DGS 的实用化关键
- **Virtual Camera Pruning** 思路可迁移：用假设视角检测几何/外观异常值
- 与 [[SGS-SLAM]] 互补：一个强调语义损失设计，一个强调特征一致性和压缩

## 相关工作联系
- 同类：[[SGS-SLAM]]
