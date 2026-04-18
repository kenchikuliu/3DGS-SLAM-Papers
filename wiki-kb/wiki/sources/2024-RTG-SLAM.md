# RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting

## 基本信息
- **论文编号**: 0009
- **分类**: [SLAM]
- **发表**: arXiv 2404.19706 | **SIGGRAPH 2024**
- **机构**: Zhejiang University (CAD&CG), Baidu Research, University of Utah
- **关键词**: [[3D Gaussian Splatting]], [[大场景重建]], [[实时SLAM]], [[内存效率]]

## 核心问题
大规模场景下，3DGS-SLAM 面临内存爆炸和速度瓶颈。如何实现大场景实时重建同时控制内存消耗？

## 方法贡献
1. **紧凑 Gaussian 表示**：强制每个 Gaussian 要么完全不透明要么近似透明
   - 不透明 Gaussian：拟合表面颜色
   - 透明 Gaussian：拟合残差颜色
   - 一个不透明 Gaussian 即可拟合局部表面区域，大幅减少 Gaussian 数量
2. **Stable/Unstable Gaussian 分类**：
   - Stable：已充分优化，仅渲染不再优化
   - Unstable：新增或误差大，优先优化
   - 每帧只渲染 Unstable Gaussians，大幅降低计算量
3. **On-the-fly Gaussian 优化**：三类像素触发新增 Gaussian（新观测、大颜色误差、大深度误差）

## 实验结果
- 速度：**16.28 FPS**，内存：**7.3 GB**（vs Co-SLAM 8.77 FPS/17GB，Point-SLAM 0.22 FPS/9.4GB）
- 场景尺度：酒店大房间（约 56m² × 1.7m 高）
- 发表于顶会 SIGGRAPH 2024

## 创新点（一句话）
不透明/透明 Gaussian 二元分类 + Stable/Unstable 优化策略，以约一半内存两倍速度实现大场景实时重建。

## Idea 价值
- **Stable/Unstable 分类** 是重要工程 idea：只优化"不稳定"的部分，大幅降低计算量
- **紧凑 Gaussian 表示** 思路：用约束（不透明/透明二元）换取效率
- 可扩展：Stable/Unstable 机制可用于任何增量式 3DGS 构建场景

## 相关工作联系
- 同机构：浙大 CAD&CG，与 [[CG-SLAM]] 同出一门
- 大场景方向：与 [[Gaussian-SLAM]] 的 Sub-map 思路对比
- SIGGRAPH 发表，工程质量更高
