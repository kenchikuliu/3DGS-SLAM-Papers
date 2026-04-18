# Industrial Benchmark Section Draft

## 1. Benchmark Motivation

Existing SLAM and 3DGS-SLAM benchmarks mainly emphasize generic indoor or outdoor reconstruction quality, while the operational conditions encountered in industrial inspection, plant maintenance, warehouse autonomy, and equipment-room navigation are substantially harsher. In particular, three degradation factors appear repeatedly in real deployments: large textureless surfaces, low and spatially non-uniform illumination, and dynamic interference caused by personnel, AGVs, forklifts, doors, or moving machinery. Although these factors have been discussed separately across prior SLAM, dense mapping, low-light reconstruction, and dynamic-scene papers, they are rarely integrated into a single controlled benchmark tailored to industrial deployment.

To address this gap, we design an industrial complex environment benchmark that targets three failure dimensions simultaneously: texture scarcity, photometric degradation, and dynamic disturbance. The benchmark is intended not only for trajectory estimation, but also for dense mapping, reconstruction fidelity, and dynamic-region robustness analysis. The benchmark therefore supports both classical SLAM baselines and recent 3DGS-based reconstruction systems.

## 2. Benchmark Design Principles

The benchmark is built around four design principles.

First, degradations must be explicit and measurable. Instead of loosely labeling sequences as “hard” or “night”, each sequence is tagged with structured condition metadata, including texture level, illumination level, dynamic level, average lux, minimum lux, and dynamic mask coverage.

Second, the benchmark must remain reproducible. Dynamic sequences are not treated as purely incidental activity; repeatable dynamic events such as pedestrian crossing, AGV passing, and partial corridor occlusion are scripted whenever possible.

Third, the benchmark must support both tracking and mapping evaluation. Accordingly, each sequence is associated with trajectory ground truth and, whenever feasible, a reconstruction reference such as a point cloud or surface model.

Fourth, reporting must be condition-aware. Global averages alone are insufficient in complex industrial settings. All benchmark tracks therefore require grouped reporting by texture, light, and dynamic condition.

## 3. Scenario Construction

We organize the dataset around four representative industrial scene types: a metal corridor, a storage aisle under reduced illumination, an equipment-room passage, and an AGV or forklift traffic corridor. These scenes are selected to capture common deployment difficulties in industrial inspection and robotic navigation: repeated structures, weak local texture, narrow free space, hard shadows, reflective surfaces, and intermittent occlusion by moving agents.

For each scene, we define a controlled subset of condition combinations over three axes:

- texture level: high, medium, low, extreme-low
- illumination level: normal, dim, dark, extreme-dark, mixed-hard-shadow
- dynamic level: static, low-dynamic, medium-dynamic, high-dynamic

Rather than exhaustively enumerating all combinations, we prioritize representative difficult subsets, such as low-texture with dark illumination, or low-texture with dark illumination and medium dynamic interference. This keeps the benchmark focused on practically relevant failure modes.

## 4. Sensor and Annotation Protocol

The minimum sensor configuration includes an RGB camera, a depth sensor or LiDAR, an IMU, and synchronized timestamps with calibrated intrinsics and extrinsics. A preferred rig further includes both RGB-D or stereo sensing and a high-accuracy ground-truth source, such as motion capture, total station, or laser-scan-based offline registration.

Each sequence stores:

- synchronized sensor streams
- calibration files
- trajectory ground truth
- dynamic masks where available
- a structured metadata file describing condition severity

This design choice is critical. Without condition metadata and dynamic annotations, later robustness analysis reduces to anecdotal comparisons rather than benchmark evidence.

## 5. Benchmark Tracks

We define three complementary benchmark tracks.

### 5.1 Industrial-Track

This track focuses on localization and tracking robustness. Methods are evaluated using ATE, RPE, track-loss count, lost duration, relocalization success, and loop-closure success. The purpose is to reveal not only final drift, but also recovery behavior and operational stability in long or repetitive industrial routes.

### 5.2 Industrial-Map

This track evaluates dense reconstruction quality. Primary metrics include Chamfer distance, accuracy, and completeness against reference geometry. For methods that support view synthesis, optional photometric metrics such as PSNR, SSIM, and LPIPS are also reported. This track is particularly relevant to 3DGS-SLAM systems, which are often designed to balance geometric consistency and render quality.

### 5.3 Industrial-Robust

This track explicitly targets degradation robustness. It aggregates success rate, drift, and map quality across condition levels, and introduces dynamic contamination analysis to estimate how much moving content is mistakenly fused into the static map. This track is designed to expose failure boundaries under compound degradations rather than idealized operation.

## 6. Reporting Protocol

Results are reported at four levels: per route, per condition, per degradation level, and global average. We emphasize that global averages should be reported last. For industrial deployment, grouped reporting is the primary result: for example, drift under low-texture conditions, or reconstruction quality under dark and dynamic conditions.

A run is considered failed if the method cannot produce a valid trajectory, remains lost for more than half the sequence, cannot align outputs with ground truth, or fails to produce the required mapping artifact for the corresponding benchmark track.

## 7. Expected Benchmark Value

The proposed benchmark is designed to fill the gap between academic reconstruction benchmarks and industrial deployment validation. It offers three concrete benefits.

First, it enables fair comparison between classical SLAM, neural implicit mapping, and 3DGS-SLAM systems under the same controlled degradation axes.

Second, it makes industrial difficulty measurable. Texture scarcity, darkness, and dynamic disturbance become benchmark variables rather than post-hoc explanations for failure.

Third, it supports both method development and deployment readiness assessment. A system that performs well on this benchmark is more likely to transfer to practical industrial navigation, inspection, and reconstruction tasks than one validated only on standard academic datasets.

## 8. Short Version For Methods Section

We construct an industrial complex environment benchmark tailored to three deployment-critical degradations: textureless surfaces, low-light conditions, and dynamic interference. The benchmark contains representative industrial scenes including metal corridors, storage aisles, equipment-room passages, and AGV traffic corridors. Each sequence is annotated with structured condition metadata, including texture level, illumination level, dynamic level, and lux measurements, and is paired with trajectory ground truth and, where available, reference reconstruction geometry. We evaluate methods under three tracks: Industrial-Track for localization robustness, Industrial-Map for dense reconstruction quality, and Industrial-Robust for degradation-aware failure analysis. Metrics include ATE, RPE, track-loss statistics, Chamfer distance, completeness, accuracy, and dynamic contamination ratio. This design enables condition-aware evaluation of both classical SLAM and 3DGS-based mapping systems under realistic industrial complexity.

## 9. Short Version For Introduction

Current SLAM and 3DGS-SLAM benchmarks insufficiently reflect the compound degradations encountered in industrial environments, where textureless surfaces, poor illumination, and dynamic interference co-occur frequently. To better evaluate deployment readiness, we introduce an industrial complex environment benchmark with controlled condition labels, trajectory ground truth, dynamic annotations, and reconstruction references. The benchmark supports both tracking and dense mapping evaluation and is specifically designed to expose failure modes under difficult industrial operating conditions.
