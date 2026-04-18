# SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM

## 基本信息
- **论文编号**: 0005
- **分类**: [SLAM]
- **发表**: arXiv 2402.03246 | 2024
- **机构**: Dalian University of Technology, University of Tokyo, Columbia, SJTU
- **关键词**: [[语义SLAM]], [[3D Gaussian Splatting]], [[Dense Visual SLAM]], [[3D语义分割]]

## 核心问题
现有 3DGS-SLAM 缺乏语义理解能力；直接将语义标签嵌入 Gaussian 参数存在问题（splatting 中语义通道的累计混合导致类别信息失真）。

## 方法贡献
1. **首个基于 Gaussian Splatting 的语义视觉 SLAM**
2. **多通道优化**：外观、几何、语义三路联合优化，解决神经隐式 SLAM 的过度平滑问题
3. **独特语义特征损失**：有效补偿深度/颜色损失在目标级优化中的缺陷
4. **语义引导关键帧选择**：避免累计误差导致的错误重建

## 关键技术
- 语义特征嵌入：用语义特征向量（而非类别标签）参数化 Gaussians
- 分割头：预训练语义分割网络提供伪标签
- 关键帧策略：语义变化触发关键帧选择

## 实验结果
- 数据集：Replica, ScanNet
- 性能：SOTA 相机位姿估计 + 地图重建 + 精确语义分割 + 目标级几何精度
- 实时渲染能力保留

## 创新点（一句话）
将语义特征（非类别标签）嵌入 3D Gaussians，通过语义引导关键帧选择实现首个实时语义密集 SLAM。

## Idea 价值
- **语义 SLAM** 方向：3DGS 天然支持多属性参数化，适合扩展语义通道
- 语义特征 vs 类别标签：splatting 的 α-blending 对离散标签不友好，连续特征更合适
- 与 [[NEDS-SLAM]] 同为语义方向，但侧重点不同

## 相关工作联系
- 同类：[[NEDS-SLAM]]（也做语义，但更关注特征压缩和异常值过滤）
- 基础：[[GS-SLAM]], [[SplaTAM]]
