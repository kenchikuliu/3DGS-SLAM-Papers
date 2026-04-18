# SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos

## 基本信息
- **论文编号**: 0153
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2504.17810 | CVPR 2025
- **机构**: University of Cambridge, Meshcapade
- **关键词**: [[小基线视频]], [[相机位姿估计]], [[3DGS]], [[动态场景]]

## 核心问题
小基线视频（社交媒体短视频：TikTok/Instagram等，相机运动极小）的相机位姿估计：三角化不足、基线太小导致深度不确定性大，MonST3R等基于点图的方法在小基线下不稳定。

## 方法贡献
**SmallGS**：专为小基线视频设计的3DGS位姿估计框架
1. **3DGS稳定参考**：从首帧重建3DGS场景，作为后续帧的稳定渲染参考（小基线时Gaussian不变）
2. **批量位姿联合优化**：在3DGS光栅化框架内批量更新相邻帧位姿（vs逐帧配对）
3. **DINOv2特征集成**：将视觉特征嵌入Gaussian，增强小基线下的渲染鲁棒性
4. **动态掩码**：MonST3R置信度掩码剔除动态区域，聚焦静态结构
5. **轨迹平滑约束**：相机轨迹平滑先验消除抖动

## 实验结果（TUM-Dynamics数据集）
- ATE（绝对轨迹误差）和RPE（相对位姿误差）显著优于MonST3R
- 速度分析：估计位置速度与真值更一致（更平滑）
- 动态场景中鲁棒性更好

## 创新点（一句话）
3DGS的显式稳定渲染特性使其天然适合小基线位姿估计，批量联合优化+DINOv2特征进一步提升小基线动态场景精度。

## Idea 价值
- **社交媒体视频重建**：日常小基线视频是3DGS重建最常见的使用场景，SmallGS解决了关键痛点
- 3DGS渲染作为稳定约束：利用Gaussian的显式表示特性（不像NeRF需要逐视角采样）
- 与 [[SplatPose]] 的视角互补：SplatPose从已知3DGS地图定位，SmallGS边建图边定位
- DINOv2+3DGS特征：特征Gaussian的应用场景验证

## 相关工作联系
- 位姿估计：[[SplatPose]]（单RGB 6-DoF）、[[GSFeatLoc]]（视觉定位）
- 动态场景：MonST3R（基础方法）
- 特征3DGS：[[GST-VLA]]、[[NeRF-Supervision]]（特征学习）
