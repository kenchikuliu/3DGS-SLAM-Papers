# ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3DGS

## 基本信息
- **论文编号**: 0189
- **分类**: [General]
- **发表**: arXiv 2603.08661 | 2026
- **机构**: Universidad de Murcia, Spain
- **关键词**: [[3DGS训练加速]], [[CUDA优化]], [[工程实现]], [[LichtFeld-Studio]]

## 核心问题
现有 3DGS 实现（包括 ImprovedGS）以 Python 为主，主机-设备同步开销和训练延迟高，限制了大规模和交互式场景捕捉的实用性。

## 方法贡献
将 ImprovedGS 策略用原生 **C++/CUDA** 重实现，重点优化两个最关键组件：
1. **Long-Axis-Split（LAS）CUDA 核函数**：自定义 Laplacian 滤波器 + NMS 边缘分数
2. **Edge-Score 重要性采样**：高效 Gaussian 密化
3. 附加改进：指数尺度调度、优化位置学习、增强 Laplacian 遮罩

## 实验结果（Mip-NeRF 360）
- 训练时间减少 **26.8%**（节省约17分钟/场景）
- Gaussian 数量减少 **13.3%**
- PSNR 提升 **1.28 dB**
- 参数复杂度降低 **38.4%**

## 创新点（一句话）
将 ImprovedGS 的 Python 逻辑完全用 C++/CUDA 原生实现，在保持质量的同时大幅提升速度和参数效率。

## Idea 价值
- **工程实现质量**：算法不变，C++/CUDA 重写即可获得显著性能提升
- 26.8% 训练加速 + 1.28dB 质量提升 同时实现，说明原始 Python 实现存在大量开销
- 与 [[SkipGS]] 正交：SkipGS 跳过无效反向传播，ImprovedGS+ 优化实现效率，可叠加

## 相关工作联系
- 训练效率同类：[[SkipGS]]（正交方向）
- 集成框架：LichtFeld-Studio 生态
