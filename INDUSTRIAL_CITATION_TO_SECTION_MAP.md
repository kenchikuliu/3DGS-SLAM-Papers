# Industrial Citation To Section Map

## Purpose

This file maps paper sections to the minimum repository-scoped references that should support them.

It is designed for direct writing use. The goal is to avoid searching across the whole repository every time a paragraph is drafted.

All references listed here come from:

1. papers already present in this repository
2. source notes derived from those papers
3. references explicitly cited by those papers, only when later expansion is needed

## 1. Abstract

The abstract should cite sparingly. In many venues, explicit citations are not used in the abstract. If your template allows them or if you are drafting a long abstract for a proposal, keep the support logic behind the text tied to this small set:

### Core Support

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
4. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
5. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)

### What These Support

- 3DGS-SLAM as the current system context
- textureless, low-light, and dynamic conditions as the benchmark gap
- the need for an industrial benchmark instead of another generic indoor benchmark

## 2. Introduction

The introduction should answer four things:

1. why 3DGS-SLAM matters
2. why industrial environments are different
3. why current evaluation is insufficient
4. what the paper contributes

### Paragraph 1: 3DGS-SLAM Context

Use:

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)

Use this paragraph to say:

- 3DGS-based methods are pushing dense SLAM toward better rendering and map quality
- tracking and dense mapping are increasingly coupled
- current systems are validated mostly on academic benchmarks

### Paragraph 2: Textureless Industrial Difficulty

Use:

1. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
2. [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)
3. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)

Use this paragraph to say:

- weak texture and repeated structure remain hard
- dense alignment is attractive but benchmark evidence is still limited in industrial settings

### Paragraph 3: Low-Light And Dynamic Difficulty

Use:

1. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
2. [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)
3. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
4. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)

Use this paragraph to say:

- low-light and dynamic interference are already recognized as hard subproblems
- they are usually treated separately rather than under a unified industrial benchmark

### Paragraph 4: Gap Statement

Use:

1. [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
2. [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)
3. [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)

Use this paragraph to say:

- current systems improve one axis at a time
- industrial benchmark design still lags behind system development
- grouped robustness reporting is missing

### Paragraph 5: Contributions

This paragraph usually does not need citations. It should state:

- an industrial complex-environment dataset
- a three-track benchmark
- condition-aware metadata and evaluation
- support for both SLAM and 3DGS-based mapping systems

## 3. Related Work

Split related work into four subsections.

### 3.1 3DGS-SLAM Systems

Use:

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
5. [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)
6. [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
7. [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)

### 3.2 Hard-Condition SLAM

Use:

1. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
2. [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)
3. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
4. [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)
5. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
6. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)

### 3.3 Localization And Robotics-Driven Scene Representation

Use:

1. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)
2. [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)
3. [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
4. [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md)
5. [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md)

### 3.4 Benchmark And Evaluation Practice

Use:

1. [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md)
2. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)
3. [0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md)
4. [0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md)
5. [0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md)

## 4. Benchmark Section

This section should cite papers that justify why each benchmark axis exists.

### Texture Axis

Use:

1. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
2. [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)

### Illumination Axis

Use:

1. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
2. [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)

### Dynamic Axis

Use:

1. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
2. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)
3. [0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md)

### Robotics Relevance

Use:

1. [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)
2. [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
3. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)

## 5. Problem Formulation Section

This section should be conservative. It usually needs fewer citations than the benchmark section.

### Recommended Support

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)
4. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)

### What These Support

- the observation-to-pose-and-map formulation
- the connection between tracking and reconstruction
- dynamic contamination as a meaningful failure type

## 6. Experiments Section

### Baseline Grouping

Use:

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)

### Tracking Metrics

Use:

1. [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md)
2. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)
3. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)

### Mapping Metrics

Use:

1. [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md)
2. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)
3. [0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md)
4. [0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md)
5. [0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md)

### Robustness Metrics

Use:

1. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
2. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
3. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
4. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)

## 7. Smallest Usable First-Draft Citation Pack

If you want the smallest set that still supports a first full paper draft, use these ten:

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
5. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
6. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
7. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
8. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)
9. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)
10. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)
