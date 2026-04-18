# LIV-GaussMap: LiDAR-Inertial-Visual Fusion for Real-time 3D Radiance Field Map Rendering

## 基本信息
- **论文编号**: arXiv 2401.14857
- **分类**: [Unlabeled→SLAM] / 多模态3DGS SLAM
- **发表**: IEEE RA-L 2024（Accepted Apr 2024）| arXiv 2401.14857 | May 2024
- **机构**: The Hong Kong University of Science and Technology (HKUST)
- **关键词**: [[LiDAR惯性视觉SLAM]], [[3DGS]], [[多模态传感器融合]], [[实时建图]]

## 核心问题
多模态传感器融合构建精确光真实感3DGS地图：单相机受光照灵敏度和深度感知限制，单LiDAR无法捕获高质量视觉纹理。如何将LiDAR几何精度与视觉纹理质量在3DGS框架下紧密融合？

## 方法贡献
**LIV-GaussMap**：LiDAR-惯性-视觉紧耦合3DGS建图系统
1. **LiDAR-惯性初始化**：用LiDAR-惯性系统精确估计每帧传感器位姿，提供尺寸自适应体素初始化Gaussian
2. **视觉光度梯度优化**：用视觉测量的光度梯度精化Gaussian的颜色质量和密度
3. **球谐函数系数**：Gaussian用球谐函数编码方向依赖视觉信息
4. **兼容多种LiDAR**：固态和机械式LiDAR（Ouster OS1-128、Livox Avia、Realsense L515）均支持

## 实验结果
- HKU LSK（室内）、HKU主楼（室外）、HKUST Tower C2（室内外）等多场景
- 比纯视觉3DGS（如原始3DGS）在大规模室外场景的几何精度显著更高
- 实时生成光真实感渲染
- 开源：https://github.com/sheng0125/LIV-GaussMap

## 创新点（一句话）
LiDAR提供精确几何结构和位姿，视觉光度梯度精化Gaussian外观质量，首个紧耦合LiDAR-惯性-视觉3DGS建图系统。

## Idea 价值
- **传感器互补**：LiDAR→几何精度，视觉→外观质量，3DGS统一表示同时享受两者优势
- 大规模室外可行：LiDAR克服纯视觉的scale歧义和光照问题，使3DGS适用于自动驾驶场景
- 与 [[GRAND-SLAM]] 的互补：GRAND-SLAM多智能体RGB-D室外，LIV-GaussMap单系统LiDAR室外精度更高

## 相关工作联系
- 大规模SLAM：[[2025-GRAND-SLAM]]（多智能体室外），[[2026-VPGS-SLAM]]（体素渐进式大规模）
- 多相机：[[2026-MCGS-SLAM]]（多相机RGB-only），与本文多传感器融合互补
- 传感器融合：与 [[2024-RTG-SLAM]] 的大场景实时策略异同（RTG-SLAM用RGB-D）
