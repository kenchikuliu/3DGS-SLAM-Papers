# Industrial Core References

## Purpose

This file is a repository-scoped reading and citation shortlist for writing the industrial benchmark paper.

It is not a full survey.

It is a practical citation pool for these sections:

- motivation
- related work
- benchmark design
- problem formulation
- experiments

All entries are selected from papers already present in this repository, or from repository source notes that summarize those papers. If a classical reference outside the repository is needed later, it should be introduced only through the explicit reference lists of the papers below.

## How To Use This File

For each benchmark claim, cite from this order:

1. direct industrial-condition papers in this file
2. representative 3DGS-SLAM system papers in this file
3. robotics or localization papers in this file
4. older references only if one of the above papers explicitly points to them

Do not cite summary dashboards or index files as final paper evidence. Use them only to navigate toward underlying papers.

## A. Textureless And Repetitive Industrial Structure

### Primary Papers

1. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)  
Use for:
- textureless corridor failure motivation
- repeated structure and weak-feature tracking difficulty
- why texture scarcity should be an explicit benchmark axis

2. [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)  
Use for:
- challenging-environment SLAM framing
- motivation for robustness-oriented evaluation
- failure analysis under difficult sensing conditions

3. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)  
Use for:
- dense alignment helping in feature-poor cases
- visual localization under weak sparse-feature support
- robotics-facing localization argument

### Supporting Notes

- [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
- [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)

## B. Low-Light And Photometric Degradation

### Primary Papers

1. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)  
Use for:
- low-light degradation as a first-class problem
- justification that darkness should be quantified rather than narratively described
- bridge between enhancement and reconstruction robustness

2. [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)  
Use for:
- additional low-light or photometric degradation support
- motivation for separating illumination difficulty from geometric difficulty

3. [2023-NEO-FOVExtrapolation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NEO-FOVExtrapolation.md)  
Use for:
- photometric instability, view synthesis, and robustness discussion
- supporting argument that visibility and photometric quality affect downstream scene use

### Writing Guidance

Use this group to support:

- `dim / dark / extreme_dark` condition levels
- `avg_lux / min_lux / brightness / dark_ratio` metadata
- low-light evaluation as a separate benchmark dimension

## C. Dynamic Environments And Map Contamination

### Primary Papers

1. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)  
Use for:
- dynamic-scene 3DGS-SLAM motivation
- online tracking plus rendering in dynamic conditions
- need for dynamic-aware evaluation

2. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)  
Use for:
- dynamic industrial-like scene difficulty
- monocular dynamic-scene robustness
- benchmark rationale for dynamic-level grouping

3. [0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md)  
Use for:
- dynamic-scene baseline framing
- static-map corruption risk under motion

4. [0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md)  
Use for:
- adaptive handling of dynamic content
- condition-aware dynamic-scene evaluation

5. [0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md)  
Use for:
- non-rigid or non-static scene modeling
- stronger justification for dynamic-region annotation

### Writing Guidance

Use this group to support:

- `static / low_dynamic / medium_dynamic / high_dynamic`
- dynamic mask annotation
- dynamic contamination ratio
- scripted dynamic agents and repeatable disturbance design

## D. Core 3DGS-SLAM System Papers

These are the main papers to cite when describing the current 3DGS-SLAM landscape.

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)  
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)  
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)  
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)  
5. [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)  
6. [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)  
7. [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)

Use this group for:

- related work overview of 3DGS-SLAM
- dense tracking and mapping with Gaussian representations
- loop closure limitations and improvements
- system-level benchmark motivation

## E. Tracking And Localization Evaluation

### Primary Papers

1. [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md)  
Use for:
- ATE-centered tracking evaluation
- reconstruction plus localization joint reporting
- typical 3DGS-SLAM comparison style

2. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)  
Use for:
- dense RGB-D SLAM evaluation style
- joint tracking, mapping, and rendering metrics

3. [0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md)  
Use for:
- dense SLAM framing
- relation between tracking, rendering, and loop closure

4. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)  
Use for:
- localization success criterion
- initialization sensitivity
- robotics localization evaluation design

### Metrics Typically Backed By This Group

- ATE
- RPE
- relocalization success
- loop closure behavior
- success rate under difficult initialization or observation conditions

## F. Mapping And Reconstruction Evaluation

### Primary Papers

1. [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md)  
2. [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md)  
3. [0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md)  
4. [0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md)  
5. [0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md)

Use this group for:

- geometric reconstruction evaluation
- PSNR / SSIM / LPIPS reporting convention
- dense SLAM quality comparison
- multi-objective evaluation combining tracking and mapping

### Metrics Typically Backed By This Group

- Chamfer
- accuracy
- completeness
- PSNR
- SSIM
- LPIPS
- Depth-L1 or depth RMSE where relevant

## G. Robotics, Navigation, And Active Reconstruction

### Primary Papers

1. [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)  
Use for:
- navigation-facing motivation
- scene representation serving downstream robotics tasks

2. [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)  
Use for:
- localization with neural scene representations
- bridge between mapping and downstream pose estimation

3. [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md)  
Use for:
- interaction-rich embodied settings
- why geometry and articulation matter in robotic environments

4. [2022-NeuralGrasps.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-NeuralGrasps.md)  
Use for:
- downstream robotic task relevance of scene representations

5. [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md)  
Use for:
- benchmark thinking for robotic perception and manipulation assets

### Writing Guidance

Use this group when you want to justify:

- AGV corridor scenes
- robotic deployment relevance
- active perception or route diversity
- the claim that the dataset should support both SLAM and robotics-facing downstream tasks

## H. Recommended Minimal Citation Set For The First Paper Draft

If you want a small but usable first-pass bibliography, start with these:

1. [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
2. [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
3. [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
4. [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
5. [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
6. [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
7. [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
8. [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
9. [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)
10. [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)

This set is enough to support the first version of:

- motivation
- related work
- benchmark design
- metric design
- robotics relevance

## I. Suggested Next Step

The next useful file is a citation-to-section mapping, for example:

- which 5 to cite in Introduction
- which 8 to cite in Related Work
- which 6 to cite in Benchmark
- which 4 to cite in Experiments

That would convert this literature pool into a direct paper-writing checklist.
