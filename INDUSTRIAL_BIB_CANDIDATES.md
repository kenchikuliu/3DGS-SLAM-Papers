# Industrial Bib Candidates

## Purpose

This file is a repository-scoped candidate bibliography list for the industrial benchmark paper.

It is not a final `.bib` file.

It is a staging document for:

- selecting references
- assigning citation keys
- marking which section each citation supports
- later converting the selected entries into BibTeX

All items here come from papers already present in this repository or from repository source notes that point to them.

## Suggested Key Style

Use a consistent key pattern such as:

- `AuthorYearShortTitle`
- `ProjectNameYear`

For example:

- `GSSLAM2024`
- `PhotoSLAM2024`
- `GaussianSLAM2024`
- `SplaTAM2024`
- `LoopSplat2024`
- `LLGaussianMap2024`
- `DynaGSLAM2024`

## A. Core 3DGS-SLAM Systems

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `GSSLAM2024` | [2024-GS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-GS-SLAM.md) | Intro, related work, baseline grouping |
| `PhotoSLAM2024` | [2024-Photo-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Photo-SLAM.md) | Intro, related work, baseline grouping |
| `GaussianSLAM2024` | [2024-Gaussian-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-Gaussian-SLAM.md) | Intro, related work |
| `SplaTAM2024` | [2024-SplaTAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-SplaTAM.md) | Intro, related work, metrics |
| `CGSLAM2024` | [2024-CG-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-CG-SLAM.md) | Related work, gap statement |
| `LoopSplat2024` | [2024-LoopSplat.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2024-LoopSplat.md) | Related work, loop discussion |
| `TwoDGSSLAM2025` | [2025-2DGS-SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2025-2DGS-SLAM.md) | Related work, gap statement |

## B. Hard-Condition SLAM

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `TexturelessORB2024` | [Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Improving_Orb-Slam3_Performance_in_Textureless_Environments_Mapping_and_Localization_in_Hospital_Corridors.md) | Intro, benchmark motivation, robustness |
| `RobustRGBDSLAMChallenging` | [Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\Toward_Accurate_Efficient_and_Robust_RGB-D_Simultaneous_Localization_and_Mapping_in_Challenging_Environments.md) | Intro, related work, robustness |

## C. Low-Light And Photometric Degradation

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `LLGaussianMap2024` | [0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md) | Intro, related work, illumination axis |
| `LowLightGS2025` | [2504.10331v3.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\2504.10331v3.md) | Related work, illumination axis |
| `NEOFOVExtrapolation2023` | [2023-NEO-FOVExtrapolation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NEO-FOVExtrapolation.md) | Related work, photometric robustness |

## D. Dynamic Gaussian SLAM

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `DynaGSLAM2024` | [0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0049_[SLAM]_DynaGSLAM Real-Time Gaussian-Splatting SLAM for Online Rendering, Trac.md) | Intro, related work, dynamic axis |
| `WildGSSLAM2024` | [0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0052_[SLAM]_WildGS-SLAM Monocular Gaussian Splatting SLAM in Dynamic Environments.md) | Intro, related work, dynamic axis |
| `DGSSLAMDynamic2024` | [0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0027_[SLAM]_DGS-SLAM Gaussian Splatting SLAM in Dynamic Environment.md) | Related work, dynamic benchmarks |
| `ADDSLAM2025` | [0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0059_[SLAM]_ADD-SLAM Adaptive Dynamic Dense SLAM with Gaussian Splatting.md) | Related work, dynamic handling |
| `FourDTAM2025` | [0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0061_[SLAM]_4DTAM Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians.md) | Related work, dynamic modeling |

## E. Tracking And Mapping Metrics

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `GSSLAMPaper2024` | [0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0001_[SLAM]_GS-SLAM Dense Visual SLAM with 3D Gaussian Splatting.md) | ATE, mapping metrics, experiments |
| `SplaTAMPaper2024` | [0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0003_[SLAM]_SplaTAM Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.md) | ATE, PSNR, SSIM, LPIPS, experiments |
| `GaussianSLAMPaper2024` | [0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0004_[SLAM]_Gaussian-SLAM Photo-realistic Dense SLAM with Gaussian Splatting.md) | Tracking plus rendering evaluation |
| `SGSSLAM2024` | [0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0005_[SLAM]_SGS-SLAM Semantic Gaussian Splatting For Neural Dense SLAM.md) | Mapping metrics, semantic extension |
| `NEDSSLAM2024` | [0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0006_[SLAM]_NEDS-SLAM A Novel Neural Explicit Dense Semantic SLAM Framework using.md) | Mapping metrics, experiments |
| `EndoGSLAM2024` | [0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0007_[SLAM]_EndoGSLAM Real-Time Dense Reconstruction and Tracking in Endoscopic Su.md) | Tracking plus reconstruction metrics |

## F. Localization And Robotics Relevance

| Candidate Key | Local Source | Intended Use |
|---|---|---|
| `GSLoc2024` | [0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\extracted_markdown\0128_[Robotics]_GSLoc Visual Localization with 3D Gaussian Splatting.md) | Intro, robotics relevance, localization evaluation |
| `RNRMapNavigation2023` | [2023-RNRMap-Navigation.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-RNRMap-Navigation.md) | Robotics motivation |
| `NeRFSCRLocalization2023` | [2023-NeRF-SCR-Localization.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2023-NeRF-SCR-Localization.md) | Localization with scene representations |
| `Ditto2022` | [2022-Ditto.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-Ditto.md) | Robotics and articulated scene relevance |
| `NeuralGrasps2022` | [2022-NeuralGrasps.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-NeuralGrasps.md) | Downstream robotic task relevance |
| `ObjectFolder2022` | [2022-ObjectFolder.md](C:\Users\Administrator\Downloads\3DGS-SLAM-Papers\wiki-kb\wiki\sources\2022-ObjectFolder.md) | Benchmark thinking for robotics assets |

## G. Minimal First-Pass Bib Set

If you want to build a first `.bib` fast, start with these keys:

- `GSSLAM2024`
- `PhotoSLAM2024`
- `GaussianSLAM2024`
- `SplaTAM2024`
- `LoopSplat2024`
- `TexturelessORB2024`
- `LLGaussianMap2024`
- `DynaGSLAM2024`
- `WildGSSLAM2024`
- `GSLoc2024`

## H. BibTeX Completion Checklist

For each selected item, fill:

- final citation key
- authors
- title
- venue
- year
- pages or article number
- DOI if available
- arXiv ID if applicable
- URL if needed by your template

## I. Suggested Next Step

Convert the minimal first-pass set into:

- `industrial_paper_refs.bib`

and then replace plain-text placeholders in the paper draft with actual citation keys.
