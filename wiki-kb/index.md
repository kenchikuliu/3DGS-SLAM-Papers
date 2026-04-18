# 知识库索引

> 最后更新：2026-04-09

---

## 概览

- 主题：3DGS & SLAM 论文 idea 库
- 素材总数：50（[SLAM] 10 + [General] 10 + [Robotics] 10 + [SLAM-Supplement] 10 + [Unlabeled] 10）
- Wiki 页面总数：54

---

## 实体页

> 人物、组织、概念、工具等

- [[3D Gaussian Splatting]] — 核心表示方法，所有论文的基础技术
- [[Dense Visual SLAM]] — 密集视觉SLAM综述，发展脉络与挑战
- [[语义SLAM]] — 语义感知的SLAM，挑战与解决思路

---

## 主题页

> 研究主题、知识领域

- [[3DGS-SLAM综述]] — 前10篇SLAM论文的技术对比与Idea提炼

---

## 素材摘要

> 每个消化过的素材都有一篇摘要

### [SLAM] 分类（10/68篇）
- [[2024-GS-SLAM]] — 首个3DGS RGB-D SLAM，自适应扩展策略，CVPR 2024
- [[2024-Photo-SLAM]] — Hyper Primitives，单目/双目/RGB-D统一，1000 FPS，CVPR 2024
- [[2024-SplaTAM]] — Silhouette Mask，CMU/MIT，400 FPS，CVPR 2024
- [[2024-Gaussian-SLAM]] — Sub-map大场景，阿姆斯特丹大学
- [[2024-SGS-SLAM]] — 首个语义3DGS-SLAM，多通道优化
- [[2024-NEDS-SLAM]] — 语义特征压缩+一致性+异常值剪枝
- [[2024-EndoGSLAM]] — 内窥镜医疗SLAM，100+ FPS
- [[2024-CG-SLAM]] — 不确定性感知Gaussian Field，浙大
- [[2024-RTG-SLAM]] — 大场景实时，Stable/Unstable分类，SIGGRAPH 2024
- [[2024-Splat-SLAM]] — RGB-only全局优化，Google/ETH

### [General] 分类（10/501篇）
- [[2026-GSStream]] — 流式3DGS渲染，实时大场景
- [[2026-ProGS]] — 程序化3DGS，结构化场景生成
- [[2026-X-GS]] — 跨模态3DGS，文本/图像驱动
- [[2026-DenoiseSplat]] — 3DGS去噪，噪声场景重建
- [[2026-GST-VLA]] — 3DGS + VLA，机器人操作感知
- [[2026-SkipGS]] — 后密化反向传播跳过，23%训练加速
- [[2026-SurgCalib]] — 3DGS手眼标定，手术机器人
- [[2026-3DGS-Watermarking]] — 可解释3DGS水印，内容安全
- [[2026-ImprovedGS-Plus]] — C++/CUDA重实现，26.8%训练提速
- [[2026-Spherical-GOF]] — 球面GOF，全景相机3DGS

### [SLAM-Supplement] 分类（10/37篇）
- [[2025-SplatPose]] — 单RGB 6-DoF位姿估计，DARS-Net旋转解耦，哈工大
- [[2026-PLANING]] — 三角-Gaussian混合表示，流式重建几何+外观解耦，浙大/上海AI Lab
- [[2025-ATLAS-Navigator]] — 语言嵌入3DGS层次地图，任务驱动导航，宾大GRASP
- [[2026-MCGS-SLAM]] — 首个多相机RGB-only 3DGS SLAM，MCBA，ETH Zurich，ICRA 2026
- [[2025-GRAND-SLAM]] — 多智能体子图+回环，公里级室外3DGS SLAM，MIT，IROS 2025
- [[2025-2DGS-SLAM]] — 2D Gaussian深度一致性，MASt3R回环，波恩大学
- [[2026-VPGS-SLAM]] — 体素渐进式大规模3DGS SLAM，2D-3D融合跟踪，上海交大
- [[2025-GSFeatLoc]] — 单次渲染特征对应视觉定位，100倍提速，RSS 2025
- [[2025-GauSS-MI]] — 香农互信息主动重建视点选择，港大
- [[2025-SmallGS]] — 小基线视频位姿估计，DINOv2特征3DGS，剑桥，CVPR 2025

### [Unlabeled] 分类（10篇，含pre-3DGS主动重建·导航·3DGS-SLAM核心）
- [[2022-UncertaintyNeRF-ActiveRecon]] — NeRF射线熵不确定性主动重建NBV策略，ETH Zürich
- [[2022-OneShotNeRF-Robotics]] — 单视图NeRF潜编码统一渲染/抓取，NVIDIA/KAIST
- [[2023-NEO-FOVExtrapolation]] — NeRF生成扩展FOV训练数据，忠实导航感知增强
- [[2023-RNRMap-Navigation]] — 可渲染NeRF网格地图，图像目标导航，首尔大学
- [[2023-ActiveImplicit-NBV]] — 隐式占据场可微NBV连续优化，哈工大深圳，IEEE RA-L 2023
- [[2024-PRVNet-ViewPlanning]] — PRVNet预测所需视角数量，一次性主动NeRF重建，波恩大学
- [[2024-OSVP-OneShot]] — OSVP一次性视角集合预测，隐式重建高效规划，波恩大学/Intel
- [[2023-NeRF-SCR-Localization]] — NeRF不确定性感知数据增强，场景坐标回归定位，MPI/ETH/Microsoft
- [[2023-GaussianSplatting-SLAM]] — **首个单目3DGS SLAM**，Lie群解析Jacobian追踪，3fps，Imperial College London
- [[2024-LIV-GaussMap]] — LiDAR-惯性-视觉紧耦合3DGS建图，实时光真实感，HKUST，IEEE RA-L 2024

### [Robotics] 分类（10/65篇）
- [[2022-Ditto]] — 关节物体数字孪生，从交互中重建，ICLR 2022
- [[2022-Relational-NDF]] — SE(3)等变关系描述符，多物体重排，CoRL 2022
- [[2022-Neural-Descriptor-Fields]] — SE(3)等变特征场，少样本操作，ICRA 2022
- [[2021-Neural-Motion-Fields]] — 隐式值函数编码抓取轨迹，RSS 2021
- [[2020-Grasping-Field]] — 人手抓取隐式场，多样抓取生成，3DV 2020
- [[2021-Dex-NeRF]] — NeRF抓取透明物体，深度感知，CoRL 2021
- [[2022-NeRF-Supervision]] — NeRF生成密集对应监督，ICRA 2022
- [[2021-GIGA]] — 几何+可供性联合隐式场，6-DoF抓取，RSS 2021
- [[2022-NeuralGrasps]] — 多机器人手统一抓取隐式表示，CoRL 2022
- [[2022-ObjectFolder]] — 视觉/触觉/听觉三模态物体数据集，CVPR 2022

---

## 对比分析

> 对比不同方案、工具、观点

（待补充）

---

## 综合分析

> 跨素材的深度分析

（见主题页 [[3DGS-SLAM综述]]）
