# Industrial Related Work Draft

Citation note: this draft should be supported only by papers already present in this repository and, where necessary, by references explicitly cited by those papers.

## 1. Related Work Draft

### 1.1 3DGS-SLAM Systems

Recent progress in 3D Gaussian Splatting has significantly influenced the design of dense SLAM systems. Representative methods such as GS-SLAM, Photo-SLAM, Gaussian-SLAM, and SplaTAM show that Gaussian-based scene representations can support joint camera tracking, dense mapping, and high-fidelity rendering within a unified system pipeline. Compared with earlier implicit or volumetric formulations, these methods typically benefit from explicit scene parameterization and efficient differentiable rasterization, which improves rendering quality and can reduce optimization overhead. At the same time, later systems such as CG-SLAM, LoopSplat, and 2DGS-SLAM further indicate that the 3DGS-SLAM landscape is rapidly expanding toward stronger loop handling, broader scene coverage, and improved system robustness. However, although these methods differ in tracking formulation, representation detail, and optimization strategy, most evaluations still remain centered on standard academic datasets rather than complex industrial operating conditions.

### 1.2 SLAM Under Hard Visual Conditions

A separate line of work highlights that robust SLAM in challenging environments remains unresolved even before dense Gaussian mapping is considered. Studies on textureless environments show that long corridors, repeated structures, and weak local appearance can severely degrade tracking reliability and map consistency. Related work on challenging RGB-D SLAM environments also emphasizes that average trajectory accuracy alone is insufficient when systems are deployed in hard sensing conditions. In parallel, low-light and photometric degradation oriented papers show that darkness should not be treated as a vague scene description, but rather as a measurable source of image-quality loss and downstream reconstruction difficulty. These studies collectively suggest that industrial evaluation should explicitly model hard conditions such as weak texture and poor illumination instead of assuming that generic indoor benchmarks are representative enough.

### 1.3 Dynamic Gaussian SLAM And Non-Static Scene Modeling

Dynamic environments introduce an additional challenge that is particularly relevant to industrial deployment. In warehouses, plant corridors, and equipment rooms, moving personnel, AGVs, forklifts, doors, and machinery can break the static-scene assumptions used by many SLAM systems. Recent Gaussian-based methods including DynaGSLAM, WildGS-SLAM, DGS-SLAM, ADD-SLAM, and 4DTAM demonstrate growing attention to this problem. These works show that dynamic content affects not only camera tracking, but also the integrity of the resulting dense map, because moving objects may be incorrectly fused into a representation intended to describe static structure. This observation motivates benchmark designs that include repeatable dynamic conditions, dynamic annotations, and robustness measures beyond simple trajectory drift.

### 1.4 Localization, Navigation, And Robotics-Facing Scene Representations

Beyond dense mapping alone, recent work also explores how Gaussian or neural scene representations can support localization and robotics-oriented downstream tasks. GSLoc demonstrates that 3DGS representations can serve as dense maps for visual localization, especially in settings where sparse feature matching becomes unreliable. Related localization and navigation oriented works based on radiance or neural scene representations further suggest that scene models should be evaluated not only by reconstruction fidelity, but also by how well they support downstream pose estimation and robot operation. Additional robotics-facing works on interaction-rich perception and object-centric benchmarks reinforce the broader point that scene representation quality should be judged in terms of deployment utility rather than rendering quality alone. This is particularly relevant for industrial environments, where mapping systems are often used in support of navigation, inspection, and autonomous operation rather than only offline visualization.

### 1.5 Gap Summary

Taken together, the current literature provides strong ingredients for an industrial benchmark, but not yet the benchmark itself. 3DGS-SLAM papers provide system-level advances in joint tracking, mapping, and rendering. Hard-condition SLAM papers reveal the importance of weak texture and challenging sensing conditions. Low-light papers emphasize photometric degradation as a measurable variable, while dynamic-scene Gaussian SLAM papers show that non-static content must be handled as both a tracking and mapping problem. Robotics-facing localization papers further demonstrate the need to evaluate scene representations in terms of downstream operational value. What remains missing is a unified, condition-aware benchmark that combines these difficulty sources under one industrial protocol with explicit condition labels, trajectory ground truth, reconstruction references, and robustness-oriented reporting.

## 2. Short Related Work Variant

Existing 3DGS-SLAM systems such as GS-SLAM, Photo-SLAM, Gaussian-SLAM, and SplaTAM demonstrate that Gaussian-based representations can jointly support tracking, dense mapping, and rendering. At the same time, prior work on textureless SLAM, low-light degradation, and dynamic-scene Gaussian SLAM indicates that weak texture, poor illumination, and non-static content remain major failure sources for localization and mapping. Additional localization and robotics-oriented scene-representation work further suggests that scene models should be evaluated for downstream operational utility, not only for reconstruction fidelity. However, these directions are still evaluated largely in isolation. A unified industrial benchmark that jointly measures texture scarcity, illumination degradation, and dynamic interference remains missing.

## 3. Subsection To Citation Hints

### 3.1 3DGS-SLAM Systems

Use:

- [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md)
- [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md)
- [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md)
- [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md)
- [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md)
- [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md)
- [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md)

### 3.2 Hard Visual Conditions

Use:

- [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md)
- [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md)
- [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md)
- [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md)

### 3.3 Dynamic Gaussian SLAM

Use:

- [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md)
- [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md)
- [0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md)
- [0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md)
- [0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md)

### 3.4 Robotics And Localization

Use:

- [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md)
- [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md)
- [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md)
- [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md)
- [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md)
