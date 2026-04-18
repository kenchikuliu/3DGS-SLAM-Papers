# EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting

## 基本信息
- **论文编号**: 0007
- **分类**: [SLAM]
- **发表**: arXiv 2403.15124 | 2024
- **机构**: Shanghai Jiao Tong University, CUHK, East China Normal University
- **关键词**: [[3D Gaussian Splatting]], [[医疗SLAM]], [[内窥镜重建]], [[实时渲染]]

## 核心问题
内窥镜手术场景下，现有 SLAM 方法难以同时实现高质量手术视野重建与高效实时计算，限制了术中应用。

## 方法贡献
1. **内窥镜专用 3DGS-SLAM**：针对组织变形、镜面反射、遮挡等手术场景特点优化
2. **精简 Gaussian 表示**：减少冗余参数，适应手术实时性要求
3. **可微光栅化追踪**：在线相机追踪 + 组织重建，>100 FPS

## 关键技术
- Streamlined Gaussian representation（精简参数）
- 可微光栅化实现实时在线渲染
- 组织纹理与几何联合重建

## 实验结果
- 渲染速度：> **100 FPS**（手术实时要求）
- 比传统和神经 SLAM 更好的术中可用性/重建质量权衡

## 创新点（一句话）
首次将 3DGS-SLAM 应用于内窥镜手术场景，精简 Gaussian 表示实现 100+ FPS 实时组织重建。

## Idea 价值
- **垂直领域应用**：3DGS-SLAM 在医疗、手术导航等垂直场景的迁移潜力
- **精简表示**思路：并非所有场景都需要完整 3DGS 参数，场景特定简化可大幅提速
- 手术场景特有挑战：组织变形（非刚体）、镜面反射、遮挡 → 未来研究方向

## 相关工作联系
- 同分类：其他 [SLAM] 论文均为通用场景，本文是医疗特化
- 非刚体 SLAM 挑战与 [[动态场景SLAM]] 相关
