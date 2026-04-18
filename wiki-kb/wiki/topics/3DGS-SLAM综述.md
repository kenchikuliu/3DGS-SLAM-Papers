# 3DGS-SLAM 综述：核心技术与 Idea 全景

> 综合自 [SLAM] 分类前 10 篇论文 | 生成日期：2026-04-08

## 背景
2023 年底至 2024 年，3D Gaussian Splatting 作为 NeRF 的显式替代方案被引入 SLAM 领域，引发了密集视觉 SLAM 的新一轮研究热潮。

## 开山之作（三驾马车）
2023 年 Q4 同期出现三篇奠基性工作：
- [[GS-SLAM]]：上海 AI Lab，首个 3DGS RGB-D SLAM，自适应扩展策略
- [[SplaTAM]]：CMU/MIT，Silhouette Mask，显式体积表示
- [[Gaussian-SLAM]]：阿姆斯特丹大学，Sub-map 大场景扩展

## 关键技术维度对比

### 1. 场景规模应对策略
| 论文 | 策略 | 优劣 |
|------|------|------|
| [[Gaussian-SLAM]] | Sub-map 分块 | 独立优化，内存可控 |
| [[RTG-SLAM]] | Stable/Unstable 分类 + 紧凑表示 | 实时性好，SIGGRAPH 认可 |
| [[CG-SLAM]] | 不确定性感知选择性优化 | 精度优先 |

### 2. 相机类型支持
| 论文 | RGB-D | 单目 | 双目 |
|------|-------|------|------|
| GS-SLAM | ✅ | ❌ | ❌ |
| Photo-SLAM | ✅ | ✅ | ✅ |
| SplaTAM | ✅ | ❌ | ❌ |
| Splat-SLAM | ❌ | ✅（+单目深度） | ❌ |

### 3. 语义扩展
- [[SGS-SLAM]]：语义特征向量 + 多通道优化
- [[NEDS-SLAM]]：特征压缩 + 一致性 + 异常值剪枝

### 4. 特殊应用
- [[EndoGSLAM]]：医疗内窥镜，精简表示，100+ FPS

## 核心 Idea 提炼

### Idea 1：Stable/Unstable Gaussian 分类
来源：[[RTG-SLAM]]
原理：已充分优化的 Gaussian 标记为 Stable，仅渲染不优化；新增或误差大的标记为 Unstable，优先优化
**可迁移性**：高。适用于任何增量式 3DGS 构建（不限于 SLAM）

### Idea 2：Silhouette Mask 作为场景密度指示器
来源：[[SplaTAM]]
原理：用轮廓掩码判断"已建图/未建图"区域，指导 Gaussian 扩展
**可迁移性**：高。适用于需要判断覆盖率的任何 3DGS 场景

### Idea 3：不确定性感知优化
来源：[[CG-SLAM]]
原理：为每个 Gaussian 附带深度不确定性，高不确定性区域优先优化
**可迁移性**：中。可扩展到颜色/语义不确定性

### Idea 4：Sub-map 大场景扩展
来源：[[Gaussian-SLAM]]
原理：场景分为独立优化的子地图，不受整体规模限制
**可迁移性**：高。是处理大场景的经典工程方案

### Idea 5：语义特征嵌入（连续特征 vs 离散标签）
来源：[[SGS-SLAM]], [[NEDS-SLAM]]
原理：splatting α-blending 对离散标签不友好，连续语义特征更适合
**可迁移性**：高。适用于所有 3DGS 语义扩展任务

### Idea 6：全局位姿优化 + 地图动态变形
来源：[[Splat-SLAM]]
原理：不仅做 frame-to-model 追踪，还全局优化关键帧位姿并变形地图
**可迁移性**：中。是提升单目 3DGS-SLAM 精度的关键

## 开放研究方向
1. **回环检测**：现有 3DGS-SLAM 多数缺乏 loop closure（→ [[LoopSplat]] 等后续工作）
2. **动态场景**：非刚体（人体、组织变形）下的 3DGS-SLAM
3. **开放词汇语义**：CLIP/SAM + 3DGS-SLAM
4. **多机协同**：分布式 3DGS 地图构建
5. **长时间运行**：地图压缩与遗忘机制

## 相关实体
- [[3D Gaussian Splatting]]
- [[Dense Visual SLAM]]
- [[语义SLAM]]
