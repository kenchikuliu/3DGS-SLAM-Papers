# Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting

## 基本信息
- **论文编号**: 0004
- **分类**: [SLAM]
- **发表**: arXiv 2312.10070 | 2024
- **机构**: University of Amsterdam
- **关键词**: [[3D Gaussian Splatting]], [[Dense Visual SLAM]], [[Sub-map]], [[大场景重建]]

## 核心问题
现有 3DGS-SLAM 方法场景表示不可扩展（需将整个场景保留在内存中），无法处理大规模场景。

## 方法贡献
1. **Sub-map 架构**：将场景组织为独立优化的子地图，无需同时在内存中保留所有内容，天然支持大场景
2. **新 Gaussian 播种策略**：为新探索区域高效初始化 Gaussians
3. **Frame-to-model 相机追踪**：通过最小化输入帧与渲染帧的光度+几何损失追踪位姿

## 关键技术
- Sub-map 独立优化：每个子地图局部优化，不受全局场景大小影响
- 支持真实世界 RGB-D 视频（非合成数据集）
- Photo-realistic 实时渲染

## 实验结果
- 数据集：Replica（合成）+ 真实世界 RGB-D 数据集
- 性能：mapping/tracking/rendering 全面优于或持平于现有神经密集 SLAM

## 创新点（一句话）
Sub-map 架构使 3DGS-SLAM 首次具备扩展到大规模场景的能力，同时保持实时性。

## Idea 价值
- **大场景可扩展性** 是 3DGS-SLAM 的核心挑战之一，Sub-map 是一种经典解法
- 与 [[RTG-SLAM]] 同样关注大场景，但思路不同（RTG 用 stable/unstable Gaussian 分类）

## 相关工作联系
- 同期：[[GS-SLAM]], [[SplaTAM]]
- 作者重叠：Martin R. Oswald 也是 [[Splat-SLAM]] 作者
