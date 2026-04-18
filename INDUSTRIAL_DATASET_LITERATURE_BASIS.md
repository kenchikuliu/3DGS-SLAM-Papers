# Industrial Dataset Literature Basis

## Purpose

This note explains which parts of the industrial complex environment dataset design were informed by the paper collection already organized in this repository.

It is not a full survey.

It is a design-to-literature mapping for:

1. textureless environments
2. low-light environments
3. dynamic environments
4. localization / mapping evaluation
5. robotics and active reconstruction tasks

## Repository Sources Used

Primary internal references used during design:

- [slam_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\slam_index.md)
- [robotics_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\robotics_index.md)
- [general_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_index.md)
- [reconstruction_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\reconstruction_index.md)
- [enhancement_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\enhancement_index.md)
- [evaluation_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\evaluation_index.md)
- [3DGS-SLAM综述.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\topics\3DGS-SLAM综述.md)

Representative source notes used as anchors:

- [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
- [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
- [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
- [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
- [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)
- [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
- [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)
- [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)
- [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
- [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md)
- [2022-NeuralGrasps.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-NeuralGrasps.md)
- [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md)

## Design Basis Matrix

### 1. Textureless Scene Dimension

Dataset design decision:

- explicitly include low-texture and extreme-low-texture industrial corridors
- store quantitative texture metadata instead of only human labels
- evaluate failure boundaries under repeated structure and weak features

Repository basis:

- [slam_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\slam_index.md)
- [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)
- [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)

Borrowed idea:

- texture-poor scenes should be treated as a first-class benchmark axis
- repeated structure and feature scarcity should be tagged, not buried in generic indoor sequences
- tracking robustness must be evaluated beyond average ATE

### 2. Low-Light Scene Dimension

Dataset design decision:

- separate `normal`, `dim`, `dark`, `extreme_dark`
- record `avg_lux` and `min_lux`
- keep low-light as a benchmark dimension independent from dynamics

Repository basis:

- [enhancement_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\enhancement_index.md)
- [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
- [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)
- [2023-NEO-FOVExtrapolation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NEO-FOVExtrapolation.md)

Borrowed idea:

- low-light must be quantified instead of described narratively
- image quality degradation, enhancement difficulty, and reconstruction difficulty should be benchmarked jointly
- photometric robustness needs its own controlled capture protocol

### 3. Dynamic Scene Dimension

Dataset design decision:

- include scripted dynamic agents and dynamic mask annotations
- separate static, low, medium, high dynamic conditions
- report dynamic contamination ratio in map quality analysis

Repository basis:

- [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
- [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)
- [0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md)
- [0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md)
- [0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md)

Borrowed idea:

- dynamic disturbances should be measured as map corruption risk, not only tracking inconvenience
- dynamic benchmark conditions should be repeatable
- masking and dynamic-region analysis should be included in dataset design from the start

### 4. Tracking Benchmark Design

Dataset design decision:

- track ATE, RPE, track-loss count, relocalization count, loop-closure success
- keep trajectory GT mandatory
- define failure conditions explicitly

Repository basis:

- [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
- [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
- [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
- [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
- [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
- [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)

Borrowed idea:

- SLAM evaluation must separate local tracking quality, global consistency, and recovery behavior
- loop closure is an independent robustness axis
- one-number reporting is not enough for difficult industrial environments

### 5. Mapping Benchmark Design

Dataset design decision:

- evaluate geometry with Chamfer / accuracy / completeness
- optionally evaluate rendering with PSNR / SSIM / LPIPS
- keep point cloud or mesh reference in the dataset plan

Repository basis:

- [reconstruction_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\reconstruction_index.md)
- [evaluation_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\general_subindexes\evaluation_index.md)
- [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)
- [2026-Spherical-GOF.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2026-Spherical-GOF.md)

Borrowed idea:

- industrial evaluation should not stop at trajectory quality
- geometry and photometric fidelity should be separated but both supported
- reconstruction GT is necessary if the dataset is meant for 3DGS-SLAM papers rather than pure odometry papers

### 6. Robotics / Active Reconstruction Basis

Dataset design decision:

- include AGV corridors, active view planning routes, and navigation-oriented scenarios
- support both SLAM and active reconstruction / navigation methods

Repository basis:

- [robotics_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\indexes\robotics_index.md)
- [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)
- [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
- [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md)
- [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md)

Borrowed idea:

- the same industrial dataset should serve localization, mapping, and embodied reconstruction tasks
- active perception methods need route diversity and occlusion-rich trajectories
- robotic deployment relevance requires more than standard academic corridor sequences

### 7. Metadata-Heavy Release Design

Dataset design decision:

- create structured `metadata.json`
- store texture level, lux, dynamic level, reflective flags, GT type
- define train / val / test splits at release time

Repository basis:

- the repo-wide graph outputs and category analyses:
  - [graphify_final_reviewed_v3_deduped.json](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\graphify_final_reviewed_v3_deduped.json)
  - [master_index.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\master_reports\master_index.md)
  - [year_trends.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\graphify-out\filtered\recategorized\final_reviewed\final_reviewed_v2\final_reviewed_v3\deduped\year_trends\year_trends.md)

Borrowed idea:

- if the goal is a benchmark rather than a media dump, metadata must be queryable and condition-aware
- later analysis depends on structured labels being present at capture time

## What Was Not Directly Copied

The dataset spec was not copied from any single paper.

It is a synthesis of:

- SLAM benchmark practice
- 3DGS reconstruction evaluation practice
- low-light enhancement evaluation practice
- dynamic-scene robustness concerns
- robotics deployment constraints

In particular, the following parts are engineering additions rather than direct paper transcriptions:

- the exact industrial scene list
- the condition matrix naming
- the dataset folder structure
- the metadata schema fields
- the evaluation output contract

## Recommended Citation Strategy For Future Writing

If you later write a proposal, paper, or benchmark introduction, cite by section:

1. for 3DGS-SLAM benchmark motivation:
   - GS-SLAM
   - SplaTAM
   - Gaussian-SLAM
   - Photo-SLAM

2. for low-light / photometric degradation motivation:
   - LL-GaussianMap
   - LL-Gaussian

3. for dynamic-scene motivation:
   - DynaGSLAM
   - WildGS-SLAM
   - DGS-SLAM
   - ADD-SLAM

4. for robotics / active reconstruction motivation:
   - RNR-Map style navigation / mapping work
   - Ditto
   - ObjectFolder

## Bottom Line

Yes: the industrial dataset design already borrows from the paper collection in this repository.

The borrowing is strongest in:

- benchmark decomposition
- degradation axes
- metric selection
- scene and task selection

The current spec is therefore aligned with the literature structure already extracted from your paper set, rather than being an ad hoc checklist.
