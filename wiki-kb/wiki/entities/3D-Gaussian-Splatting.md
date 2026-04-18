# 3D Gaussian Splatting（3DGS）

## 定义
3D Gaussian Splatting 是一种基于显式点云的实时三维场景表示与渲染方法。场景被表示为一组带属性的 3D 高斯椭球体，通过可微光栅化渲染为 2D 图像。

**核心论文**：Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023

## 核心属性
每个 3D Gaussian 拥有：
- **位置** $X \in \mathbb{R}^3$
- **协方差矩阵** $\Sigma = RSS^TR^T$（旋转 R + 缩放 S）
- **不透明度** $\alpha \in [0,1]$
- **颜色**（球谐系数 SH，支持视角相关外观）

## 渲染原理
1. 将 3D Gaussians 投影到 2D 图像平面
2. 按深度排序，从前到后 α-blending：$\hat{C} = \sum_{i} c_i \alpha_i \prod_{j<i}(1-\alpha_j)$

## 核心优势 vs NeRF
| 特性 | 3DGS | NeRF |
|------|------|------|
| 表示类型 | 显式（点云） | 隐式（MLP） |
| 渲染速度 | 100-1000 FPS | 0.1-10 FPS |
| 训练速度 | 分钟级 | 小时级 |
| 可编辑性 | 强（直接操作点） | 弱 |
| 内存 | 较大（显式存储） | 较小（网络权重） |

## 在 SLAM 中的应用
3DGS 引入 SLAM 的优势：
- **实时建图**：光栅化远快于体素渲染
- **增量更新**：可直接增删 Gaussians
- **可微渲染**：对位姿和场景参数均可求梯度

## 已知局限
- Gaussian 数量随场景增大线性增长（内存压力）
- 对未观测区域无法外推（纯数据驱动）
- 大场景可扩展性挑战（→ [[Sub-map]], [[Stable/Unstable Gaussian 分类]]）
- 不确定性建模默认缺失（→ [[CG-SLAM]] 扩展）

## 在本库论文中的角色
几乎所有 [SLAM] 类论文均以 3DGS 为核心表示：
- [[GS-SLAM]] - 首个应用
- [[SplaTAM]] - 显式体积表示
- [[RTG-SLAM]] - 紧凑 Gaussian 表示
- [[SGS-SLAM]], [[NEDS-SLAM]] - 语义扩展
