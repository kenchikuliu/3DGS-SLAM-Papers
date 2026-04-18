# Leveraging Neural Radiance Fields for Uncertainty-Aware Visual Localization

## 基本信息
- **论文编号**: arXiv 2310.06984
- **分类**: [Unlabeled] / NeRF·视觉定位
- **发表**: arXiv 2310.06984 | Oct 2023
- **机构**: Max Planck Institute for Intelligent Systems, ETH Zurich, Microsoft Mixed Reality & AI Lab
- **关键词**: [[视觉定位]], [[场景坐标回归]], [[NeRF数据增强]], [[不确定性感知]]

## 核心问题
场景坐标回归（SCR）视觉定位的训练数据瓶颈：获取大量标注训练数据（2D像素→3D场景坐标对应）代价高昂；NeRF渲染虽可生成合成视角，但渲染图像含噪声/模糊，冗余度高，影响训练质量。

## 方法贡献
**NeRF增强不确定性感知SCR**：
1. **U-NeRF**：不确定性感知NeRF，分别预测渲染RGB和深度的不确定性，反映数据可靠性
2. **不确定性引导视角选择**：基于U-NeRF的不确定性，从合成视角集中选择信息增益最高的样本
3. **E-SCRNet**：证据深度学习（Evidential DL）建模SCR不确定性，评估场景坐标质量
4. **数据增强pipeline**：少量真实图像 → U-NeRF + 视角选择 → 高质量增强数据集 → E-SCRNet微调

## 实验结果
- 公共视觉定位数据集验证
- 方法只用小比例训练集，可达到与全量训练集相当甚至更好的定位性能
- 数据效率显著提升

## 创新点（一句话）
NeRF分离建模颜色和深度不确定性，指导合成视角筛选，大幅提升SCR视觉定位的数据效率。

## Idea 价值
- **NeRF作数据引擎（定位版）**：与 [[2023-NEO-FOVExtrapolation]] 的"NeRF生成导航训练数据"异曲同工
- 3DGS替代方向：3DGS渲染速度更快，可更高效地生成定位训练数据
- 与 [[GSFeatLoc]] 对比：GSFeatLoc直接用3DGS渲染特征定位（无需SCR网络），更简洁

## 相关工作联系
- 视觉定位：[[2025-GSFeatLoc]]（3DGS特征渲染定位，速度100倍提升）
- NeRF数据增强：[[2023-NEO-FOVExtrapolation]]（NeRF生成FOV扩展训练数据）
- 不确定性：[[CG-SLAM]]（Gaussian不确定性），[[2022-UncertaintyNeRF-ActiveRecon]]（NeRF熵不确定性）
