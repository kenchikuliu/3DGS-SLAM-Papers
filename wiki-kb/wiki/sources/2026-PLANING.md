# PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction

## 基本信息
- **论文编号**: 0144
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2601.22046 | 2026
- **机构**: Zhejiang University, Shanghai AI Lab, SJTU, Northwestern Polytechnical, Fudan, CUHK, HKU
- **关键词**: [[三角面片-Gaussian混合]], [[流式重建]], [[几何-外观解耦]], [[具身AI]]

## 核心问题
流式3D重建（从单目序列实时建图）：现有方法要么高质量渲染但几何差，要么几何好但渲染质量差。3DGS倾向于外观建模而牺牲几何结构一致性。

## 方法贡献
**PLANING**：松耦合三角面片-Gaussian混合表示
1. **几何-外观解耦**：可学习三角面片建模几何（边缘保持、平面结构），神经Gaussian建模外观（高保真渲染）
2. **松耦合设计**：三角面片作为稳定几何锚点约束Gaussian更新，两者可独立优化
3. **流式重建框架**：前馈模型提供位姿先验，流式初始化+全局地图调整策略
4. **平面导出**：几何清晰的三角面片可导出紧凑平面结构，用于具身AI仿真环境

## 实验结果
- 比PGSR提升 Chamfer-L2 **18.52%**
- 比ARTDECO提升 **1.31 dB PSNR**
- ScanNetV2场景重建 <100秒（比2DGS快 **5×**）
- 兼顾几何精度和渲染质量，两者同时SOTA

## 创新点（一句话）
三角面片建几何 + 神经Gaussian建外观的松耦合混合表示，流式单目重建中同时实现高精度几何和高保真渲染。

## Idea 价值
- **混合表示**是解决3DGS几何退化的根本思路之一（另一路是2DGS）
- 平面导出能力：对机器人导航/具身AI的价值极高（结构化环境理解）
- 与 [[GRAND-SLAM]] 的子图大场景思路正交，可结合
- 三角面片几何约束 vs 2DGS的法向量约束：两种路线的对比

## 相关工作联系
- 几何重建：[[2DGS-SLAM]]（2D Gaussian几何一致性）
- 流式SLAM：[[VPGS-SLAM]]（体素渐进式）、[[MCGS-SLAM]]
- 具身AI场景：[[ATLAS-Navigator]]（语言导航）
