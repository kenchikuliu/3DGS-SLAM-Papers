# Industrial Experiments Section Draft

## 1. Experimental Goals

The experiments are designed to answer four questions.

First, how robust is a method to low-texture industrial structure where appearance cues are weak or repetitive.

Second, how much does performance degrade under low-light and spatially non-uniform illumination.

Third, how reliably can the method separate static structure from moving objects in dynamic industrial scenes.

Fourth, how do classical SLAM baselines and 3DGS-based methods differ when evaluated under the same industrial condition matrix.

These questions motivate a condition-aware evaluation protocol rather than a single global score.

## 2. Dataset and Splits

We evaluate on the proposed industrial complex-environment benchmark, which contains representative industrial scenes such as metal corridors, dark storage aisles, equipment-room passages, and AGV or forklift traffic corridors. Each scene is captured under controlled condition variants along three axes: texture level, illumination level, and dynamic level.

The dataset is split into training, validation, and test partitions at the scene-condition-route level to avoid leakage between nearly identical captures. All reported benchmark numbers are measured on the test split, while validation sequences are used for parameter tuning when required.

For each sequence, the benchmark provides synchronized sensor streams, calibration files, trajectory ground truth, condition metadata, and optional dynamic-region annotations and reference geometry.

## 3. Baselines

We recommend organizing baselines into three groups.

### 3.1 Classical SLAM Baselines

This group includes feature-based, direct, RGB-D, visual-inertial, or LiDAR-assisted systems that primarily target robust tracking and mapping without neural scene representations.

### 3.2 Neural or Implicit Mapping Baselines

This group includes methods based on neural fields, radiance fields, or hybrid dense representations that emphasize scene reconstruction quality.

### 3.3 3DGS-SLAM Baselines

This group includes Gaussian Splatting based SLAM and dense mapping methods that jointly address tracking, reconstruction, and rendering.

The purpose of this grouping is not only to compare individual systems, but also to compare method families under industrial degradation.

## 4. Evaluation Tracks

We report results on three complementary tracks.

### 4.1 Industrial-Track

This track evaluates localization and trajectory estimation quality. Metrics include:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- track-loss count
- lost duration ratio
- relocalization success rate
- loop-closure success rate

This track is intended to reveal whether a method remains operational in long, repetitive, and difficult industrial routes.

### 4.2 Industrial-Map

This track evaluates dense scene reconstruction quality. Metrics include:

- Chamfer distance
- accuracy
- completeness
- optional PSNR
- optional SSIM
- optional LPIPS

The geometric metrics are treated as primary. Photometric metrics are reported when the method supports rendering or view synthesis.

### 4.3 Industrial-Robust

This track evaluates condition robustness and failure boundaries. Metrics include:

- success rate by condition level
- ATE grouped by texture level
- ATE grouped by illumination level
- ATE grouped by dynamic level
- reconstruction quality grouped by condition
- dynamic contamination ratio

This track is critical because industrial deployment failure usually appears first in edge-case conditions rather than in global averages.

## 5. Reporting Protocol

We report results at four granularities:

1. per-route
2. per-condition
3. per-domain
4. global aggregate

The reporting order matters. Per-condition and per-domain results should appear before global aggregate numbers, because they directly expose the sensitivity of each method to texture scarcity, darkness, and dynamic interference.

We also recommend that every table explicitly marks sequence failures instead of silently dropping failed runs from averaged statistics.

## 6. Main Comparison Tables

The experiments section should typically include the following tables.

### Table A: Overall Tracking Comparison

Columns:

- Method
- Sensor Setup
- ATE
- RPE
- Lost Count
- Lost Duration Ratio
- Relocalization Success
- Loop Closure Success

### Table B: Overall Mapping Comparison

Columns:

- Method
- Representation
- Chamfer
- Accuracy
- Completeness
- PSNR
- SSIM
- LPIPS

### Table C: Condition-Aware Robustness

Columns:

- Method
- Low-Texture ATE
- Dark ATE
- Dynamic ATE
- Low-Texture Success Rate
- Dark Success Rate
- Dynamic Success Rate
- Dynamic Contamination Ratio

### Table D: Efficiency

Columns:

- Method
- FPS
- GPU Memory
- CPU Memory
- Runtime Per Meter
- Map Build Time

This table is especially useful for industrial deployment claims.

## 7. Ablation Studies

For a proposed method, the experiments should include at least three ablations.

### 7.1 Texture Robustness Ablation

Analyze how the method behaves when feature density or appearance distinctiveness decreases. This can be tested through progressively harder low-texture sequences or through modules specifically designed for repetitive or texture-poor structure.

### 7.2 Low-Light Ablation

Analyze the effect of illumination-aware preprocessing, exposure adaptation, denoising, or photometric modeling. Report grouped results by lux range or illumination label.

### 7.3 Dynamic-Scene Ablation

Analyze the effect of dynamic masking, motion filtering, temporal consistency modules, or static-background regularization. Dynamic contamination ratio should be included here rather than reported only in the main table.

## 8. Qualitative Evaluation

Qualitative results should not be limited to visually pleasing renderings. The recommended qualitative set includes:

- representative success cases in low-texture corridors
- failure cases under severe darkness
- dynamic-scene examples showing whether moving objects are fused into the map
- side-by-side trajectory plots for long repetitive routes
- reconstruction snapshots in static versus dynamic conditions

These figures should be selected to explain metric behavior rather than to decorate the paper.

## 9. Failure Analysis

A serious industrial benchmark paper should include failure analysis. We recommend grouping failures into at least four categories:

- drift accumulation in repetitive structure
- photometric breakdown in dark regions
- corrupted map fusion under dynamic interference
- relocalization failure after prolonged occlusion

This section is often more valuable than another small average improvement number, because it clarifies deployment boundaries.

## 10. Short Version For Experiments Section

We evaluate all methods on the proposed industrial complex-environment benchmark using three tracks: tracking, mapping, and robustness. Tracking quality is measured by ATE, RPE, track-loss statistics, relocalization success, and loop-closure success. Mapping quality is measured by Chamfer distance, accuracy, completeness, and optional rendering metrics for methods that support view synthesis. Robustness is evaluated by grouping performance over texture, illumination, and dynamic condition levels, and by measuring dynamic contamination in reconstructed maps. We compare classical SLAM baselines, neural or implicit mapping baselines, and 3DGS-SLAM baselines under the same protocol. In addition to overall comparison tables, we report condition-aware results, efficiency metrics, and ablations on low texture, low light, and dynamic-scene handling.

## 11. Short Version For Paper Template

### Experiments

We conduct experiments on the proposed industrial benchmark covering textureless, low-light, and dynamic industrial scenes. All methods are evaluated using the same synchronized sensor inputs, calibration protocol, and condition metadata. We report results on three tracks: Industrial-Track for trajectory robustness, Industrial-Map for dense reconstruction quality, and Industrial-Robust for degradation-aware analysis. For trajectory estimation, we use ATE, RPE, track-loss statistics, relocalization success, and loop-closure success. For mapping, we report Chamfer distance, accuracy, completeness, and optional rendering metrics. To analyze deployment readiness, we further group results by texture level, illumination level, and dynamic level, and report dynamic contamination ratios in non-static scenes.
