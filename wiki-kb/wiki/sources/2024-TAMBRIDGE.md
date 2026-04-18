# TAMBRIDGE: Bridging Frame-Centered Tracking and 3D Gaussian Splatting for Enhanced SLAM

## 基本信息
- **论文编号**: 0012
- **分类**: [SLAM]
- **发表**: arXiv 2405.19614 | May 2024（Under review）
- **机构**: Peking University（National Key Lab of General AI, Shenzhen Graduate School）, ETH Zurich
- **关键词**: [[ORB视觉里程计]], [[融合桥接]], [[在线3DGS]], [[长会话SLAM]]

## 核心问题
3DGS追踪收敛难题：随机传感器噪声和视角视差导致3DGS在线渲染收敛困难，在长会话机器人任务中累积误差严重。现有系统无法在实时性、噪声鲁棒性和映射质量之间同时达到SOTA。

## 方法贡献
**TAMBRIDGE**（Tracking And Mapping BRIDGE）：
1. **Fusion Bridge模块**：将追踪中心的ORB视觉里程计（含回环）与建图中心的在线3DGS无缝集成
2. **精确位姿初始化**：联合优化稀疏重投影误差和密集渲染误差，消除视角重叠问题
3. **战略视点选择**：为在线重建选择适当关键帧，确保视角稀疏性和最小重叠
4. **回环借用**：利用ORB回环检测消除轨迹累积误差

## 实验结果
- 多真实世界数据集验证（含大规模场景）
- 首个3DGS-SLAM系统稳定实现5+ FPS实时性能
- 渲染质量和定位精度均达到SOTA

## 创新点（一句话）
Fusion Bridge作为追踪中心ORB和建图中心3DGS之间的即插即用桥接模块，通过精确位姿初始化和视点选择解决在线3DGS收敛困难。

## Idea 价值
- **解耦追踪+建图**的深度融合：不是简单串联，而是双向信息流（追踪提供位姿，建图提供渲染约束）
- ORB回环+3DGS地图：传统SLAM强项（回环检测）与3DGS强项（高质量渲染）的优势组合
- 工业可用性：5+ FPS实时对机器人部署是关键门槛

## 相关工作联系
- Photo-SLAM：[[2024-Photo-SLAM]]（也用ORB追踪，Hyper Primitives建图）
- LoopSplat：[[2024-LoopSplat]]（3DGS子图+3DGS注册回环）
- GLC-SLAM：[[2024-GLC-SLAM]]（层次回环+不确定性关键帧选择）
