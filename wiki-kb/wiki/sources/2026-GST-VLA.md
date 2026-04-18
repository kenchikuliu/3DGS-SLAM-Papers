# GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models

## 基本信息
- **论文编号**: 0185
- **分类**: [General]
- **发表**: arXiv 2603.09079 | 2026
- **机构**: Yeungnam University, KAIST
- **关键词**: [[3D Gaussian Splatting]], [[VLA机器人]], [[空间推理]], [[具身智能]]

## 核心问题
VLA 模型将视觉观测编码为 2D patch token，缺乏内在几何结构（无深度、无表面法向、无几何置信度），在精度要求高的操作任务（边缘抓取、插销插入）中性能下降。

## 方法贡献
1. **Gaussian Spatial Tokenizer（GST）**：将冻结深度特征 + 语义 patch 特征转化为 128 个结构化 3D Gaussian 基元
   - 每个 Gaussian 参数化为：均值 $\mu \in \mathbb{R}^3$，对数尺度协方差 $\sigma$，不透明度 $\alpha$
   - 协方差特征结构编码局部表面朝向和几何置信度
2. **Depth-Aware Chain-of-Thought（DA-CoT）**：在行动解码前进行结构化中间推理（3D目标定位 → 抓取接触几何 → 度量空间关系 → SE(3)轨迹点）
3. **条件化 ODE 集成**：流匹配行动专家解码 7-DoF delta 动作块

## 实验结果
- LIBERO 基准：**96.4%**（+2.0% vs baseline）
- SimplerEnv：**80.2%**（+5.4%）

## 创新点（一句话）
用 3D Gaussian 基元作为 VLA 的空间 token，配合 DA-CoT 中间推理，将几何感知能力注入 VLA 行动策略。

## Idea 价值
- **Gaussian as Spatial Token** 是重要跨界 idea：3DGS 从渲染工具变为推理载体
- DA-CoT 将 3D 空间推理显式化，对 Embodied AI 有普适价值
- 与 [[X-GS]] 的 X-GS-Thinker 思路相近，但更聚焦机器人行动

## 相关工作联系
- 统一框架视角：[[X-GS]]（也将 Gaussian 对接 VLM）
- 机器人应用：[[SurgCalib]]（也是机器人+3DGS）
