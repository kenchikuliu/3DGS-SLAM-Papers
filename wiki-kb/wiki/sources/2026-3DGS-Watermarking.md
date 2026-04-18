# Where, What, Why: Toward Explainable 3D-GS Watermarking

## 基本信息
- **论文编号**: 0188
- **分类**: [General]
- **发表**: arXiv 2603.08809 | 2026
- **机构**: Waseda University, Southeast University, NTU
- **关键词**: [[3DGS水印]], [[版权保护]], [[可解释AI]], [[内容安全]]

## 核心问题
3DGS 作为 3D 资产的主流表示，其显式可编辑性带来严重版权风险——攻击者可轻易复制、篡改、重新分发。现有水印方法缺乏可解释性（不知道水印嵌在哪、为什么选那些载体）。

## 方法贡献
1. **Trio-Experts 模块**：直接在 Gaussian 基元上操作，推导载体选择先验（综合多视角可见性、频率域线索、几何/外观稳定性）
2. **SBAG（Safety and Budget Aware Gate）**：将 Gaussians 分配到水印载体或视觉补偿者，兼顾鲁棒性和视觉质量预算
3. **Channel-wise Group Mask**：控制载体和补偿者的梯度传播，限制 Gaussian 参数更新范围，保留高频细节
4. **可解释性**：解耦微调提供 per-Gaussian 归因（where 水印在哪、what 被编码、why 选择该载体）

## 实验结果
- PSNR 提升 **+0.83 dB**
- 比特精度提升 **+1.24%**
- 在压缩/噪声等常见图像失真下保持水印

## 创新点（一句话）
Trio-Experts 从可见性/频率/稳定性三维选择水印载体，SBAG 平衡鲁棒性与质量，首次实现可解释的 3DGS 水印。

## Idea 价值
- **3DGS 内容安全**是随着商业化落地必然出现的需求
- 可解释水印框架：per-Gaussian 归因可用于审计和溯源
- SBAG 思路（预算感知分配）可迁移到 3DGS 的其他属性编辑场景

## 相关工作联系
- 安全方向：3DGS 版权保护是新兴研究领域
- 与 [[ProGS]]、[[GSStream]] 均属于 3DGS 实际部署基础设施方向
