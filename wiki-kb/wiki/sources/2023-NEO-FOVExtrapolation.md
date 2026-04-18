# NeRF-Enhanced Outpainting for Faithful Field-of-View Extrapolation

## 基本信息
- **论文编号**: arXiv 2303.00304
- **分类**: [Unlabeled] / NeRF·导航感知
- **发表**: arXiv 2303.00304 | Sep 2023
- **机构**: University of Louisville, Penn State University, Manycore Tech Inc.
- **关键词**: [[FOV扩展]], [[NeRF数据增强]], [[图像外绘]], [[机器人导航感知]]

## 核心问题
机器人导航和远程视觉辅助中的忠实FOV外推（Faithful FOV Extrapolation）：与美观性图像外绘（hallucination）不同，导航场景要求外推内容与真实场景保持几何和语义一致性。

## 方法贡献
**NEO（NeRF-Enhanced Outpainting）**：
1. **NeRF场景建模**：用训练图像序列训练场景特定NeRF
2. **密集虚拟视角采样**：在目标FOV内密集采样相机位姿，用NeRF渲染扩展FOV图像
3. **外绘模型训练**：用NeRF生成的扩展FOV图像集训练图像外绘模型
4. **推理时外推**：外绘模型对新捕获图像实时扩展FOV，无需NeRF在线推理

## 实验结果
- 3个光真实感数据集 + 1个真实世界数据集验证
- 在忠实FOV外推质量上持续超过3个基线方法
- 优于纯图像拼接和视频扩展方法

## 创新点（一句话）
NeRF作为场景先验生成忠实几何一致的扩展FOV合成数据，训练专用外绘模型实现实时导航感知增强。

## Idea 价值
- **NeRF作数据引擎**：用神经渲染生成训练数据的范式，可迁移到3DGS数据增强
- 导航感知增强：扩大相机FOV是低成本提升导航性能的手段
- 与3DGS渲染类比：3DGS渲染速度远超NeRF，同样范式在3DGS下可实现实时数据增强

## 相关工作联系
- NeRF导航：[[2023-RNRMap-Navigation]]（可渲染NeRF导航地图）
- 视觉定位：[[2023-NeRF-SCR-Localization]]（NeRF增强视觉定位训练数据）
- 3DGS导航：[[2025-ATLAS-Navigator]]（语言嵌入3DGS层次导航地图）
