# SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery

## 基本信息
- **论文编号**: 0187
- **分类**: [General]（实际偏 cs.RO）
- **发表**: arXiv 2603.08983 | 2026
- **机构**: UBC, NUS
- **关键词**: [[3D Gaussian Splatting]], [[手眼标定]], [[外科机器人]], [[da Vinci]]

## 核心问题
da Vinci 手术机器人的手眼标定（相机-机器人坐标系变换 AX=XB）：传统方法需要标定板（污染无菌环境）或依赖准确的本体感觉数据（腱驱动机器人本体感觉噪声大）。

## 方法贡献
**SurgCalib**：无标记 3DGS 手眼标定框架
1. 用运动学测量初始化手术器械位姿
2. 在 3DGS 可微渲染流水线内通过 RCM（远程运动中心）约束两阶段优化精化位姿

## 实验结果
- dVRK 基准 SurgPose 验证
- 2D 工具尖端重投影误差：**12.24px（2.06mm）**，**11.33px（1.9mm）**
- 3D 工具尖端欧式距离误差：**5.98mm**，**4.75mm**（左右器械）

## 创新点（一句话）
将 3DGS 可微渲染引入手术机器人手眼标定，无需标定板，通过 RCM 约束在渲染误差中优化位姿变换。

## Idea 价值
- **可微渲染作为标定工具**：3DGS 的梯度可以流向位姿变换参数
- 手术场景 3DGS 与 [[EndoGSLAM]] 互补（一个做 SLAM，一个做标定）
- RCM 约束 + 可微渲染优化 → 可迁移到其他机器人外参标定

## 相关工作联系
- 医疗机器人：[[EndoGSLAM]]（内窥镜 SLAM）
- 机器人应用：[[GST-VLA]]（VLA 行动策略）
