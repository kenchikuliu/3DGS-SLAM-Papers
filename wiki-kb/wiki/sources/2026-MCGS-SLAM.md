# MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping

## 基本信息
- **论文编号**: 0147
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2509.14191 | ICRA 2026
- **机构**: ETH Zurich, University of Zurich, University of Stuttgart, Microsoft Research Asia
- **关键词**: [[多相机SLAM]], [[3DGS]], [[多相机束调整]], [[自动驾驶]]

## 核心问题
多相机3DGS SLAM：现有方法几乎全部针对单目/双目，多相机平台（自动驾驶、机器人）的宽视野观测被大量浪费。如何充分利用多相机的互补视角实现高保真建图？

## 方法贡献
**MCGS-SLAM**：首个纯视觉（RGB-only）多相机3DGS SLAM系统
1. **Multi-Camera Bundle Adjustment（MCBA）**：跨相机联合优化位姿和深度，通过光度+几何一致性约束
2. **Joint Depth-Scale Alignment（JDSA）**：跨相机尺度一致性对齐，使用低秩几何先验
3. **多相机高斯建图**：多相机关键帧融合到统一Gaussian地图，密化+剪枝跨相机协同
4. **Waymo Open Dataset验证**：前+左+右三相机，240°视野覆盖

## 实验结果（Waymo / 合成数据集）
- 轨迹精度比单目基线显著提升
- 高保真渲染：depth+RGB多相机一致重建
- 实时性能维持在大规模场景

## 创新点（一句话）
MCBA联合优化多相机位姿和深度，首次将3DGS SLAM扩展到多相机平台，宽视野消除单目的固有遮挡盲区。

## Idea 价值
- **多相机3DGS SLAM**是自动驾驶/无人机的刚需，单目FoV限制是主要瓶颈
- MCBA思路：多相机联合BA比分别建图后融合更准确，是系统设计关键
- Waymo数据集验证：说明可扩展到真实复杂驾驶场景
- 与 [[Spherical-GOF]] 互补：一个做多相机协同，一个做全景单相机

## 相关工作联系
- 多相机扩展：[[Spherical-GOF]]（全景单相机 3DGS）
- 大场景 SLAM：[[VPGS-SLAM]]、[[GRAND-SLAM]]
- 自动驾驶感知：与 Photo-SLAM 的多模态支持对比
