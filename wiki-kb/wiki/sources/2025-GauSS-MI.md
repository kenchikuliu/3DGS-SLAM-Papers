# GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction

## 基本信息
- **论文编号**: 0152
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2504.21067 | 2025
- **机构**: University of Hong Kong, KTH Royal Institute of Technology
- **关键词**: [[主动重建]], [[视点选择]], [[香农互信息]], [[不确定性量化]]

## 核心问题
主动3D重建的视觉质量优化：现有方法用几何完整性（探索未知体素）选择下一视点，忽略视觉质量。如何实时量化3DGS地图的视觉不确定性并指导下一最佳视点选择？

## 方法贡献
**GauSS-MI**：基于香农互信息的3DGS主动重建
1. **概率Gaussian模型**：将每个3DGS椭球的残差图像损失映射为概率模型，量化视觉不确定性
2. **GauSS-MI度量**：用Shannon Mutual Information量化新视点观测对重建模型的期望信息增益
3. **实时视点选择**：无需先验即可评估任意候选视点的互信息，实时选择下一最佳视点
4. **主动重建系统**：集成视点规划器+运动规划器+在线3DGS重建，完整端到端系统

## 实验结果（仿真+真实场景）
- 视觉保真度（PSNR/SSIM）显著优于仅基于几何完整性的主动重建方法
- 重建效率（用更少视点达到同等质量）提升
- 开源：https://github.com/JohannaXie/GauSS-MI

## 创新点（一句话）
将Shannon互信息引入3DGS主动重建，基于渲染损失的概率Gaussian模型实时量化视觉不确定性，指导高视觉质量的下一最佳视点选择。

## Idea 价值
- **主动感知+3DGS**：机器人自主重建场景时决定"往哪走"的核心问题
- 视觉质量驱动 vs 几何完整性驱动：两种目标在实际任务中各有侧重，可结合
- 概率Gaussian模型：为3DGS引入贝叶斯视角，与 [[CG-SLAM]] 的不确定性Gaussian类似
- 与 [[ATLAS-Navigator]] 互补：导航选择任务目标，GauSS-MI选择重建视点

## 相关工作联系
- 不确定性3DGS：[[CG-SLAM]]（Gaussian不确定性用于SLAM）
- 主动感知：与 [[ATLAS-Navigator]] 的主动规划有共同问题域
- 信息论重建：FisherRF、ActiveGS（同类先驱）
