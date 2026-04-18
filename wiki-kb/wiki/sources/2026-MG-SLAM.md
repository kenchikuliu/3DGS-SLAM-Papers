# MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis

## 基本信息
- **论文编号**: 0011
- **分类**: [SLAM]
- **发表**: IEEE Transactions on Automation Science and Engineering | arXiv 2405.20031 | Jan 2026
- **机构**: U Tokyo, Columbia U, NTU Singapore, Dalian U of Technology
- **关键词**: [[Manhattan World假设]], [[线段特征]], [[平面表面补全]], [[无纹理室内SLAM]]

## 核心问题
室内无纹理场景中3DGS-SLAM的两大缺陷：（1）无纹理区域追踪失败（缺乏特征点）；（2）遮挡/有限视角导致重建空洞（Gaussian无法插值未观测区域）。

## 方法贡献
**MG-SLAM**：曼哈顿世界假设驱动的结构化Gaussian SLAM（RGB-D）
1. **线段特征追踪**：从结构化场景提取融合线段，用于无纹理区域的重投影追踪和全Bundle Adjustment
2. **平面约束**：平行线和平面表面约束，提高追踪和建图的结构一致性
3. **Gaussian平面补全**：用提取的线段作为边界识别平面区域，插值新Gaussian填充空洞
4. **泊松网格提取**：通过正则化项从Gaussian体积表示直接提取高质量网格

## 实验结果
- ScanNet数据集：SOTA追踪和全面地图重建
- 大规模Apartment数据集：ATE降低 **50%**，PSNR提升 **5dB**
- 高帧率运行

## 创新点（一句话）
Manhattan World假设将室内场景先验（正交线面结构）直接注入3DGS-SLAM的追踪和建图，解决无纹理场景的追踪和空洞问题。

## Idea 价值
- **结构先验+3DGS**：人造室内环境的线/面结构是强约束，可大幅提升3DGS-SLAM鲁棒性
- Gaussian平面插值：主动推断未观测区域的思路，与 [[2024-OSVP-OneShot]] 中隐式表示推断未见区域异曲同工
- 网格提取：从3DGS直接导出网格对下游任务（碰撞检测、物体抓取）有直接价值

## 相关工作联系
- 语义SLAM：[[2024-SGS-SLAM]]（语义3DGS）、[[2025-Hier-SLAM]]（层次语义）
- 结构SLAM：与 [[2024-CG-SLAM]] 的不确定性感知都是提升追踪鲁棒性的路径
- 网格：[[2026-PLANING]]（三角-Gaussian混合，几何更优）
