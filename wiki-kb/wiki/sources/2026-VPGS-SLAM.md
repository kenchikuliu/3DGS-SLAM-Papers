# VPGS-SLAM: Voxel-based Progressive 3D Gaussian SLAM in Large-Scale Scenes

## 基本信息
- **论文编号**: 0150
- **分类**: [SLAM-Supplement]
- **发表**: arXiv 2505.18992 | 2026（IEEE Transactions on Intelligent Transportation Systems）
- **机构**: Shanghai Jiao Tong University, Hong Kong University of Science and Technology, Nanyang Technological University, University of Bonn
- **关键词**: [[大规模SLAM]], [[体素渐进式3DGS]], [[2D-3D融合跟踪]], [[自动驾驶]]

## 核心问题
大规模城市场景3DGS SLAM三大挑战：(a) Gaussian数量爆炸导致内存溢出；(b) 大运动/光照变化下渲染损失跟踪失败；(c) 长序列累积位姿漂移。

## 方法贡献
**VPGS-SLAM**：体素渐进式大规模3DGS SLAM
1. **体素渐进式Gaussian表示**：多分辨率体素网格，每个锚点生成带偏移的神经Gaussian，子图机制减少内存
2. **2D-3D融合相机跟踪**：粗阶段用2D光度损失+3D Gaussian ICP，细阶段用3D体素ICP精化
3. **2D-3D Gaussian回环检测**：体素ICP检测回环，位姿图优化+在线子图融合蒸馏
4. **信息丰富度自适应**：评估场景2D信息密度，自适应选择关键参数（室内/室外切换）

## 实验结果（室内+室外数据集）
- 室内和城市级室外场景均验证
- 优于现有3DGS-SLAM方法（跟踪+建图）
- 代码开源：https://github.com/dtc111111/vpgs-slam

## 创新点（一句话）
体素结构约束Gaussian分布，2D-3D融合跟踪+回环，首次将3DGS SLAM有效扩展到大规模城市场景且不内存爆炸。

## Idea 价值
- **大规模3DGS SLAM的系统性解决方案**：从表示、跟踪、回环三个维度全面设计
- 体素锚点+神经Gaussian偏移：比纯自由Gaussian更紧凑，泛化到Scaffold-GS系列
- 自动驾驶场景验证：比只在室内测试的方法更具实用价值
- 2D-3D融合跟踪对比：与 [[2DGS-SLAM]] 的纯渲染跟踪路线形成互补

## 相关工作联系
- 大规模SLAM：[[GRAND-SLAM]]（多智能体）、[[RTG-SLAM]]（大场景实时）
- 体素表示：与 Scaffold-GS 的体素锚点思路一致
- 自动驾驶应用：[[MCGS-SLAM]]（多相机）
