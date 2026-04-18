# LoopSplat: Loop Closure by Registering 3D Gaussian Splats

## 基本信息
- **论文编号**: 0014
- **分类**: [SLAM]
- **发表**: arXiv 2408.10154 | Aug 2024
- **机构**: Stanford University, University of Amsterdam, ETH Zurich
- **关键词**: [[3DGS回环闭合]], [[子图配准]], [[位姿图优化]], [[全局一致性]]

## 核心问题
3DGS-SLAM缺乏全局一致性机制：现有coupled 3DGS SLAM系统无策略实现全局一致性（回环/全局BA），导致位姿误差累积和地图变形。传统点云ICP配准无法直接用于3DGS回环约束提取。

## 方法贡献
**LoopSplat**：3DGS子图+回环配准的RGB-D SLAM
1. **3DGS子图建图**：增量式构建3DGS子图，frame-to-model追踪
2. **3DGS直接配准**：新型配准方法直接在3DGS表示上操作（而非转换为点云），提取相对位姿约束
3. **在线回环检测**：图像匹配+子图重叠度检测，实时触发回环
4. **鲁棒位姿图优化**：检测回环后用位姿图优化实现全局一致性
5. **刚性地图对齐**：子图间刚性对齐保持全局一致

## 实验结果
- Replica/TUM-RGBD/ScanNet/ScanNet++全系列验证
- ScanNet scene0054：PSNR **28.52 dB**（vs GO-SLAM 22.33 dB, Gaussian-SLAM 16.21 dB）
- 追踪、建图、渲染均优于或竞争于现有密集RGB-D SLAM

## 创新点（一句话）
直接在3DGS表示上提取回环约束（无需转为点云），通过在线3DGS配准实现coupled RGB-D SLAM的全局一致性。

## Idea 价值
- **3DGS即地图即配准工具**：3DGS既作建图表示又作回环配准媒介，充分利用表示的连续可微性
- 与 [[2025-2DGS-SLAM]] 对比：后者用MASt3R特征做回环，本文用3DGS几何直接配准
- 回环思路汇总：LoopSplat(3DGS配准) → GLC-SLAM(位姿图+子图) → GRAND-SLAM(ICP+子图) → 2DGS-SLAM(MASt3R)

## 相关工作联系
- 全局一致性：[[2024-Splat-SLAM]]（全局优化RGB-only）、[[2025-2DGS-SLAM]]（MASt3R回环）
- 回环：[[2024-GLC-SLAM]]（层次回环）、[[2025-GRAND-SLAM]]（多智能体ICP回环）
- Gaussian SLAM：[[2024-Gaussian-SLAM]]（Sub-map大场景）
