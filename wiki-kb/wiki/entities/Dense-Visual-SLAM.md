# Dense Visual SLAM

## 定义
Dense Visual SLAM（密集视觉同步定位与建图）是指在估计相机运动轨迹的同时，构建场景的**密集**三维地图（而非稀疏特征点地图）的技术。

## 传统方法分类
| 类型 | 代表方法 | 地图表示 |
|------|---------|---------|
| 点云/面元 | TSDF-Fusion, KinectFusion | TSDF体素 |
| 体素哈希 | ElasticFusion | Surfel |
| NeRF-based | iMAP, NICE-SLAM, ESLAM, Point-SLAM | MLP/特征网格 |
| **3DGS-based** | GS-SLAM, SplaTAM, RTG-SLAM 等 | 3D Gaussians |

## 核心挑战
1. **实时性**：建图与追踪需实时（≥10 FPS）
2. **大场景扩展**：内存和计算随场景增大
3. **相机追踪精度**：ATE RMSE 亚厘米级
4. **渲染质量**：PSNR、SSIM 等指标
5. **传感器限制**：从 RGB-D → 单目（更难）

## 3DGS-SLAM 的演化趋势
```
2023 Q4: 奠基期
  GS-SLAM / SplaTAM / Gaussian-SLAM（三驾马车同期出现）

2024 Q1: 扩展期
  CG-SLAM（不确定性）
  SGS-SLAM / NEDS-SLAM（语义）
  Photo-SLAM（多相机类型）
  EndoGSLAM（医疗应用）

2024 Q2: 成熟期
  RTG-SLAM（大场景，SIGGRAPH）
  Splat-SLAM（RGB-only 全局优化）
```

## 关键技术维度
- **地图表示**：[[3D Gaussian Splatting]] vs NeRF
- **追踪策略**：frame-to-model vs 全局优化（[[Splat-SLAM]]）
- **场景扩展**：[[Sub-map]] / [[Stable/Unstable 分类]]
- **语义集成**：[[语义SLAM]]
- **传感器**：RGB-D / 单目 / 双目

## 常用评估数据集
- **Replica**：合成室内，高质量 GT
- **TUM-RGBD**：真实 RGB-D，追踪基准
- **ScanNet**：真实室内，大规模
- **EuRoC**：MAV 飞行，双目
