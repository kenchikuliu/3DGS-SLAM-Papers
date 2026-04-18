# GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM

## 基本信息
- **论文编号**: 0148
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2506.18885 | IROS 2025
- **机构**: MIT Aerospace Controls Laboratory
- **关键词**: [[多智能体SLAM]], [[大规模3DGS]], [[子图局部优化]], [[回环检测]]

## 核心问题
大规模多智能体3DGS SLAM：现有方法受限于小规模室内场景，多智能体协作探索大规模室外环境时存在累积误差和缺乏全局一致性机制。

## 方法贡献
**GRAND-SLAM**：协作式多智能体Gaussian SLAM
1. **子图局部优化**：每个智能体基于局部子图优化，避免全局Gaussian联合优化的高代价
2. **智能体间回环检测**：基于关键帧描述符的粗到细回环，跨智能体ICP精细对齐
3. **位姿图优化**：检测到回环后用位姿图更新地图，支持智能体内+智能体间双向回环
4. **RGB-D输入**：支持多智能体RGB-D流式输入

## 实验结果（Replica / Kimera-Multi）
- Replica室内：PSNR比现有方法高 **28%**，跟踪误差低 **91%**
- Kimera-Multi室外大规模：改进渲染质量+鲁棒跟踪
- 首个支持室外大规模多智能体3DGS SLAM的工作

## 创新点（一句话）
子图局部优化 + 智能体间ICP回环，首次将多智能体3DGS SLAM扩展到公里级室外环境并实现全局一致性。

## Idea 价值
- **多智能体协作建图**是机器人集群探索的核心需求，GRAND-SLAM提供了可落地框架
- 子图+回环的架构与传统SLAM思路一致，但用3DGS替换了地图表示
- 室外大规模验证：比只在Replica验证的方法实用性高得多
- 与 [[MCGS-SLAM]] 互补：一个多智能体，一个多相机

## 相关工作联系
- 多智能体SLAM：MAGiC-SLAM（同类先驱）
- 大规模场景：[[VPGS-SLAM]]（体素进行式）、[[RTG-SLAM]]（大场景实时）
- 回环检测：[[Splat-SLAM]]（全局优化）、[[2DGS-SLAM]]（MASt3R回环）
