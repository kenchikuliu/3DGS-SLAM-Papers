# GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction

Tianyu Xiong 1 Rui Li 2 Linjie Li 1 Jiaqi Yang 1

## Abstract

Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs joint pose-appearance optimization during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRFâ, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves explicit SfM feature tracks as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinementâa capability absent in photometriconly approaches. We introduce two pipeline variants: (1) GloSplat-F, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) GloSplat-A, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.

## 1. Introduction

Novel view synthesis (NVS) has emerged as a central challenge in computer vision, enabling applications from virtual reality and cultural heritage preservation to autonomous driving and robotic simulation. Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) demonstrated that continuous volumetric representations can synthesize photorealistic views from multi-view images, sparking rapid progress in neural scene representations. More recently, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has revolutionized the field by representing scenes as collections of 3D Gaussian primitives, achieving real-time rendering while maintaining high visual fidelity.

<!-- image-->  
Figure 1. Accuracy vs. Speed. Average PSNR on MipNeRF360 vs. runtime for 1000 images (Courthouse scene). GloSplat-F achieves 13.3Ã speedup over GPU-accelerated COLMAP+3DGS while improving PSNR by +0.38 dB. GloSplat-A achieves the highest accuracy (28.86 dB), surpassing all baselines. Our joint poseappearance optimization enables both variants to occupy the Pareto frontier. (All methods benchmarked on the same GPU; see Section C for details.)

Despite these advances, current NVS pipelines share a fundamental limitation: they treat feature extraction, structure from motion (SfM), and radiance field optimization as independent modules with separate objectives. This modular design creates information barriers across the reconstruction pipelineâSfM cannot leverage photometric signals from rendering, and NVS methods inherit fixed camera poses without geometric feedback. The conventional wisdom assumes that accurate camera poses from SfM are sufficient initialization for downstream tasks, yet pose errors compound through the pipeline, leading to blurred reconstructions and geometric inconsistencies.

Traditional pipelines rely heavily on COLMAP (Schonberger & Frahm Â¨ , 2016), which employs incremental SfM to sequentially register images and refine the reconstruction. While robust, this approach suffers from several drawbacks: (1) drift accumulation as errors propagate through sequential image registration, (2) computational bottlenecks from exhaustive feature matching with $O ( n ^ { 2 } )$ complexity, and (3) inability to incorporate photometric feedback after pose estimation. Recent work has attempted to address these limitations through learning-based approaches (Wang et al., 2025; 2024a) or improved densification strategies (Kheradmand et al., 2024b; Ye et al., 2024b), but these methods still maintain the modular separation between geometric estimation and appearance optimization.

We argue that camera pose estimation and radiance field learning share a common goalâaccurate 3D reconstructionâand should therefore be optimized jointly rather than sequentially. This insight motivates our approach: GloSplat, which integrates global SfM with joint pose-appearance optimization during 3DGS training. While feature extraction and matching remain frozen preprocessing stages, we enable continuous pose refinement during Gaussian training: learned features inform global SfM, which provides initialization for 3DGS, which in turn refines camera poses through combined photometric and geometric supervision.

Our framework introduces two pipeline variants targeting different use cases:

â¢ GloSplat-F: A fast, COLMAP-free variant that uses retrieval-based pair selection (via MegaLoc (Berton et al., 2024)) with top-k candidates, enabling lineartime matching complexity. This variant achieves stateof-the-art results among COLMAP-free methods while being significantly faster than traditional pipelines.

â¢ GloSplat-A: An accurate variant with exhaustive matching that maximizes reconstruction quality. This variant surpasses all COLMAP-based baselines, demonstrating that joint pose-appearance optimization with global SfM can outperform incremental approaches even with the same matching budget.

Both variants share a unified architecture: (1) local correspondence extraction as frozen preprocessingâXFeat (Potje et al., 2024b) with LightGlue (Lindenberger et al., 2023) for GloSplat-F, SIFT with exhaustive matching for GloSplat-A, (2) global SfM with rotation averaging and bundle adjustment for robust initialization, and (3) joint 3DGS training that preserves explicit feature tracks as persistent geometric constraints. Our key architectural novelty is maintaining SfM track 3D points as separate optimizable parameters from Gaussian means during 3DGS training. This enables a reprojection-based BA loss to provide geometric anchoring alongside photometric supervisionâunlike prior joint methods (BARF, NeRFâ, 3RGS) that rely purely on photometric gradients and suffer from early-stage pose drift when Gaussians are sparse.

Our key contributions are:

â¢ Persistent Feature Tracks During 3DGS Training: Unlike prior joint optimization methods (BARF, NeRFâ , 3RGS) that rely purely on photometric gradients for pose refinement, we maintain explicit SfM feature tracks as first-class entities throughout 3DGS training. Track 3D points are optimized as separate parameters from Gaussian means, providing persistent geometric anchors that prevent early-stage pose drift.

â¢ Joint Photometric-Geometric Optimization: We combine photometric rendering losses with a reprojection-based bundle adjustment loss that operates on preserved feature tracks. This dual supervision enables poses to benefit from both fine-grained appearance gradients and robust multi-view geometric constraints simultaneouslyâa capability absent in purely photometric approaches.

â¢ Global SfM Integration: We integrate GPUaccelerated global SfM (rotation averaging + parallel bundle adjustment) with joint 3DGS training, providing initialization that is both faster and more robust than incremental methods, while our joint optimization further refines these poses.

â¢ State-of-the-Art Results: GloSplat-F achieves new state-of-the-art among COLMAP-free methods across three benchmarks, while GloSplat-A surpasses all COLMAP-based baselines, demonstrating that joint geometric-photometric optimization outperforms frozen-pose pipelines.

We provide verification code and scripts to reproduce all experiments in the supplementary material. The full codebase will be open-sourced upon acceptance.

## 2. Related Work

Novel View Synthesis and 3D Gaussian Splatting. Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) enabled photorealistic view synthesis through continuous volumetric representations, with subsequent improvements in antialiasing (Barron et al., 2021; 2022) and acceleration (Muller Â¨ et al., 2022). 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) achieves real-time rendering by explicitly modeling scenes as anisotropic 3D Gaussians. A critical limitation is sensitivity to initialization qualityârecent work addresses this through improved densification (Ye et al., 2024b; Zhang et al., 2024; Kheradmand et al., 2024b; Mallick et al., 2024). We adopt MCMC densification (Kheradmand et al., 2024b), which provides principled control over Gaussian allocation.

Structure from Motion. COLMAP (Schonberger & Â¨ Frahm, 2016) exemplifies incremental SfM, which suffers from drift accumulation (Newcombe et al., 2011). Global SfM methods (Moulon et al., 2013; Wilson & Snavely, 2014) address drift by estimating all poses simultaneously through rotation averaging (Hartley et al., 2013) and bundle adjustment (Triggs et al., 2000). GLOMAP (Pan et al., 2024) demonstrates improved accuracy through global optimization. InstantSfM (Zhan et al., 2024) further accelerates global SfM by leveraging NVIDIA cuDSS (NVIDIA Corporation, 2024) for GPU-accelerated sparse linear solving in bundle adjustment, enabling parallel optimization that is significantly faster than traditional CPU-based solvers. We build upon InstantSfMâs GPU-accelerated BA engine and integrate it into a unified optimization framework that propagates geometric constraints through to Gaussian splatting.

Learned Features and Matching. Learned features (DeTone et al., 2018; Potje et al., 2024a) and matchers (Sarlin et al., 2020; Lindenberger et al., 2023) have improved over classical SIFT (Lowe, 2004). GloSplat-F employs XFeat (Potje et al., 2024a) with LightGlue (Lindenberger et al., 2023) and retrieval-based pair selection (via MegaLoc (Berton et al., 2024)) for O(n) matching; GloSplat-A uses SIFT with exhaustive matching for direct comparability with COLMAP baselines.

COLMAP-Free Methods. CF-3DGS (Fu et al., 2024) and HT-3DGS (Ji & Yao, 2025) train 3DGS without COLMAP but assume sequential input. Foundation model approaches like DUSt3R (Wang et al., 2024b), MASt3R (Leroy et al., 2024), VGGSfM (Wang et al., 2024a), and VGGT (Wang et al., 2025) predict geometry from image pairs. VGGT-X (Wang et al., 2025) scales these to dense multi-view settings. Unlike feed-forward models, GloSplat provides optimization-based pose estimation with joint BA loss for continuous refinement.

Joint Pose and Radiance Optimization. Prior work explores joint pose-appearance optimization with varying approaches. NeRFâ (Wang et al., 2021) and BARF (Lin et al., 2021) optimize poses using purely photometric gradients, employing coarse-to-fine positional encoding to avoid local minima. SPARF (Truong et al., 2023) incorporates multi-view correspondences but does not maintain explicit geometric constraints during NeRF training. PoRF (Bian et al., 2024) uses depth priors from monocular networks. 3RGS (Huang et al., 2025) leverages the 3R foundation model for initialization but relies on photometric-only refinement thereafter.

Key distinction: Unlike these methods, GloSplat preserves explicit SfM feature tracks as first-class citizens during 3DGS training. We maintain track 3D points as separate optimizable parameters (distinct from Gaussian means) and enforce multi-view geometric consistency via a reprojection loss throughout trainingânot just during initialization. This architectural choice provides geometric anchoring that prevents the early-stage pose drift common in photometriconly methods, while still allowing fine-grained pose refinement from rendering gradients. The combination of (1) global SfM initialization, (2) persistent feature tracks as separate optimizable parameters, and (3) joint photometricgeometric optimization during 3DGS training distinguishes our approach from prior work.

## 3. Method

We present GloSplat, a framework that performs joint pose-appearance optimization during 3D Gaussian Splatting training. Unlike traditional approaches that freeze camera poses after SfM, GloSplat continuously refines poses using combined photometric and geometric supervision while optimizing Gaussian primitives. We introduce two variants: GloSplat-F uses retrieval-based pair selection with top-k candidates for linear-time matching, while GloSplat-A uses exhaustive matching for maximum quality. Both share the same core architecture (Figure 2): (1) learned local correspondence extraction, (2) global SfM, and (3) joint Gaussian splatting with bundle adjustment.

## 3.1. Learned Feature Extraction and Matching

Rather than relying on classical handcrafted features, we employ learned feature extraction and matching as frozen preprocessing, providing both efficiency and robustness.

Image Pair Selection. For GloSplat-F, we employ retrieval-based pair selection using MegaLoc (Berton et al., 2024) to identify the top-k most similar images (k = 5) per query, reducing quadratic matching complexity to linear time. For GloSplat-A, we perform exhaustive pairwise matching to maximize reconstruction quality.

Feature Extraction and Matching. For GloSplat-F, we use XFeat (Potje et al., 2024b) for local feature extraction (up to 4096 keypoints, 64-dim descriptors) with Light-Glue (Lindenberger et al., 2023) matching, achieving favorable speed-accuracy trade-offs. For GloSplat-A, we use SIFT features with exhaustive nearest-neighbor matching, ensuring direct comparability with COLMAP baselines that use the same feature pipeline.

<!-- image-->  
Figure 2. GloSplat Pipeline. Given unposed input images, local correspondences are extracted (frozen preprocessing): XFeat+LightGlue with retrieval-based pairs (GloSplat-F) or SIFT with exhaustive matching (GloSplat-A). Global SfM simultaneously estimates all camera poses through rotation averaging, positioning, and bundle adjustment, providing robust initialization. Joint 3DGS training (our core contribution) then continuously refines poses through a reprojection-based BA loss while optimizing Gaussian primitives, enabling combined photometric-geometric supervision that prevents drift and improves reconstruction quality.

## 3.2. Global Structure from Motion

Unlike incremental SfM that suffers from drift accumulation, our global SfM solves for all camera poses simultaneously, providing improved robustness and natural parallelization.

View Graph and Calibration. From matched features, we construct a view graph $\mathcal { G } = ( \nu , \mathcal { E } )$ where vertices represent images and edges encode pairwise relationships. For uncalibrated cameras, we estimate focal lengths via the Fetzer method (Fetzer et al., 2020), minimizing:

$$
\operatorname* { m i n } _ { f _ { 1 } , f _ { 2 } } \sum _ { ( i , j ) \in \mathcal { E } } \rho _ { \mathrm { C a u c h y } } \left( \left\| \mathbf { K } _ { 2 } ^ { - \top } \mathbf { F } _ { i j } \mathbf { K } _ { 1 } ^ { - 1 } \right\| ^ { 2 } - 2 \right) .\tag{1}
$$

Relative poses are computed using the 5-point algorithm within ${ \mathrm { R A N S A C } }$ , decomposing the essential matrix into rotation ${ \bf R } _ { i j }$ and translation direction $\mathbf { t } _ { i j }$

Rotation Averaging. We solve for globally consistent absolute rotations $\left\{ \mathbf { R } _ { i } \right\}$ that best explain the measured relative rotations $\{ \mathbf { R } _ { i j } \} _ { ( i , j ) \in \mathcal { E } }$ from two-view geometry. Rotations are initialized via maximum spanning tree traversal (weighted by inlier counts) and refined by minimizing:

$$
\mathcal { L } _ { \mathrm { r o t } } = \sum _ { ( i , j ) \in \mathcal { E } } \rho _ { \mathrm { G M } } \left( \left| \left| \log \left( \mathbf { R } _ { i j } ^ { \top } \mathbf { R } _ { j } \mathbf { R } _ { i } ^ { \top } \right) \right| \right| ^ { 2 } ; \sigma \right) ,\tag{2}
$$

where $\log ( \cdot ) : S O ( 3 ) \to { \mathfrak { s o } } ( 3 )$ is the logarithm map and $\rho _ { \mathrm { G M } } ( e ^ { 2 } ; \sigma ) = e ^ { 2 } / ( e ^ { 2 } + \sigma ^ { 2 } )$ is the Geman-McClure robust loss. Optimization proceeds via $\ell _ { 1 }$ -regression initialization followed by iteratively reweighted least squares (IRLS).

Track Establishment and Positioning. Feature tracksâ consistent correspondences across viewsâare established using union-find, producing tracks $\{ \mathcal { T } _ { k } \}$ linking 2D observations to 3D points. For each observation $( i , p ) \in \mathcal { T } _ { k }$ , we compute the unit bearing vector $\mathbf { b } _ { i p } \ =$ $\mathbf { R } _ { i } ^ { \top } \mathbf { K } _ { i } ^ { - 1 } \tilde { \mathbf { x } } _ { i , p } / \lVert \mathbf { K } _ { i } ^ { - 1 } \tilde { \mathbf { x } } _ { i , p } \rVert$ , where $\tilde { \mathbf { x } } _ { i , p }$ denotes the homogeneous 2D observation. With rotations fixed, we solve for translations $\left\{ \mathbf { t } _ { i } \right\}$ and 3D positions $\{ { \bf X } _ { k } \}$ by minimizing the perpendicular distance from each 3D point to its corresponding viewing ray:

$$
{ \mathcal { L } } _ { \mathrm { p o s } } = \sum _ { k } \sum _ { \left( i , p \right) \in { \mathcal { T } } _ { k } } \left. \left( \mathbf { I } - \mathbf { b } _ { i p } \mathbf { b } _ { i p } ^ { \top } \right) \left( \mathbf { X } _ { k } - \mathbf { c } _ { i } \right) \right. ^ { 2 } ,\tag{3}
$$

where $\begin{array} { r l r } { \mathbf { c } _ { i } } & { { } = } & { - \mathbf { R } _ { i } ^ { \top } \mathbf { t } _ { i } } \end{array}$ is the camera center and $\left( \mathbf { I _ { \alpha } } - \mathbf { \partial } \right.$ $\mathbf { b } _ { i p } \mathbf { b } _ { i p } ^ { \top } )$ projects onto the subspace orthogonal to the bearing direction. This quadratic objective is solved via BAEâs (Zhan et al., 2025) Levenberg-Marquardt solver with cuDSS (NVIDIA Corporation, 2024) GPU-accelerated sparse linear solving.

Bundle Adjustment. We refine all parameters jointly through bundle adjustment, minimizing reprojection errors with Huber robust loss:

$$
\mathcal { L } _ { \mathrm { B A } } ^ { \mathrm { S f M } } = \sum _ { k } \sum _ { ( i , p ) \in \mathcal { T } _ { k } } \rho _ { \mathrm { H u b e r } } \left( \big \| \pi _ { i } \big ( \mathbf { X } _ { k } \big ) - \mathbf { x } _ { i , p } \big \| ^ { 2 } \right) ,\tag{4}
$$

where the projection $\pi _ { i } ( \mathbf { X } ) = \operatorname { p r o j } ( \mathbf { K } _ { i } ( \mathbf { R } _ { i } \mathbf { X } + \mathbf { t } _ { i } ) )$ with $\mathrm { p r o j } ( [ x , y , z ] ^ { \top } ) = [ x / z , y / z ] ^ { \top }$ maps world points to image coordinates, and $\mathbf { x } _ { i , p }$ is the observed 2D feature. We parameterize poses on the $S E ( 3 )$ manifold and solve using Levenberg-Marquardt with cuDSS-accelerated sparse linear solving, enabling fully GPU-parallel optimization that is up to 10Ã faster than traditional CPU-based PCG solvers. BA is iterated three times with progressively tightening reprojection thresholds to filter outliers. Track completion and FAISS-accelerated merging then densify the reconstruction.

## 3.3. Joint 3D Gaussian Splatting with Bundle Adjustment

Each SfM point initializes a 3D Gaussian with position Âµ at the triangulated coordinates, scale computed from the average distance to the $k = 4$ nearest neighbors (providing adaptive sizing based on local density), opacity $\alpha = 0 . 1$ and spherical harmonics encoding the pointâs RGB color. We employ stochastic density control via MCMC-based densification (Kheradmand et al., 2024a), which provides principled control over the total primitive count. Unlike heuristic splitting and cloning rules, this approach treats allocation as sampling from a distribution balancing quality against complexity, using stochastic birth-death processes to relocate primitives from over- to under-represented regions.

Photometric Loss. The primary training objective combines $\ell _ { 1 }$ with structural similarity:

$$
{ \mathcal { L } } _ { \mathrm { p h o t o } } = ( 1 - \lambda _ { \mathrm { S S I M } } ) \left\| \hat { \mathbf { I } } - \mathbf { I } \right\| _ { 1 } + \lambda _ { \mathrm { S S I M } } ( 1 - \mathrm { S S I M } ( \hat { \mathbf { I } } , \mathbf { I } ) ) ,\tag{5}
$$

where ËI is the rendered image, I the ground truth, and $\lambda _ { \mathrm { S S I M } } = 0 . 2$

Joint Bundle Adjustment Loss. This is our key architectural distinction from prior joint optimization methods. Unlike BARF (Lin et al., 2021), NeRFâ (Wang et al., 2021), and 3RGS (Huang et al., 2025) which optimize poses using only photometric gradients, we preserve explicit SfM feature tracks as persistent geometric constraints. Specifically, we maintain 3D track points $\{ { \bf X } _ { k } \}$ as separate optimizable parameters distinct from Gaussian means $\{ \mu _ { j } \}$ âthe track points anchor multi-view consistency while Gaussian means represent scene appearance. We minimize:

$$
\mathcal { L } _ { \mathrm { B A } } ^ { \mathrm { j o i n t } } = \sum _ { k } \sum _ { ( i , p ) \in \mathcal { T } _ { k } } \rho _ { \mathrm { H u b e r } } \left( \left\| \pi _ { i } ( \mathbf { X } _ { k } ) - \mathbf { x } _ { i , p } \right\| ^ { 2 } ; \delta \right) ,\tag{6}
$$

with Huber threshold $\delta = 1 . 0$ pixels, where $\pi _ { i } ( \cdot )$ is the projection function defined in Section 3.2. Crucially, camera poses are optimized by both losses simultaneously. The photometric loss $\mathcal { L } _ { \mathrm { p h o t o } }$ provides direct photometric supervision that can reduce accumulated errors from SfM initialization by directly measuring rendering quality. However, relying solely on photometric gradients during early optimizationâ when Gaussians are sparse and poorly initializedâcan cause catastrophic pose drift where the entire scene fails to converge. The joint BA loss $\mathcal { L } _ { \mathrm { B A } } ^ { \mathrm { j o i n t } }$ serves as a geometric anchor that prevents this early-stage drift by enforcing multi-view consistency through explicit feature correspondences. The total objective $\mathcal { L } = \mathcal { L } _ { \mathrm { p h o t o } } + \lambda _ { \mathrm { B A } } \mathcal { L } _ { \mathrm { B A } } ^ { \mathrm { j o i n t } } \left( \lambda _ { \mathrm { B A } } = 1 0 ^ { - 4 } \right)$ thus combines the benefits of both: geometric constraints stabilize optimization while photometric gradients enable finegrained pose refinement. Camera extrinsics use Adam with learning rate $1 0 ^ { - 5 }$

Implementation. We use gsplat (Ye et al., 2024a) for differentiable rasterization. All experiments use a maximum of 3M Gaussians to avoid excessive hyperparameter tuning across scenes. Gaussian learning rates: $1 . 6 \times 1 0 ^ { - 4 }$ (positions, scaled by scene extent), $5 \times 1 0 ^ { - 3 }$ (scales), $1 0 ^ { - 3 }$ (rotations), $5 \times 1 0 ^ { - 2 }$ (opacities), $2 . 5 \times 1 0 ^ { - 3 }$ (SH coefficients up to degree 3).

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We evaluate GloSplat on three widely-used multi-view reconstruction benchmarks: MipNeRF360 (Barron et al., 2022) (9 scenes with up to 311 images per scene), Tanks and Temples (TnT) (Knapitsch et al., 2017) (5 scenes with up to 1106 images), and CO3Dv2 (Reizenstein et al., 2021) (5 scenes with up to 202 images). These datasets span diverse scene types including indoor environments, outdoor landscapes, large-scale structures, and object-centric captures.

Metrics. We assess rendering quality using standard metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018). Higher PSNR and SSIM indicate better reconstruction, while lower LPIPS indicates improved perceptual quality.

Baselines. We compare against state-of-the-art COLMAPfree 3DGS methods: VGGT-X (Wang et al., 2025), CF-3DGS (Fu et al., 2024), HT-3DGS (Ji & Yao, 2025), 3RGS (Huang et al., 2025), and MCMC-3DGS (Kheradmand et al., 2024b). We also include 3DGS and MCMC-3DGS initialized with COLMAP as upper-bound references.

## 4.2. Main Results

Table 1 presents quantitative comparisons on novel view synthesis. GloSplat-F achieves state-of-the-art performance among COLMAP-free methods on all three benchmarks, demonstrating the effectiveness of our joint pose-appearance optimization even with efficient retrieval-based matching. On MipNeRF360, we outperform the previous best method VGGT-X by +1.37 dB in PSNR (+5.2%), while improving SSIM by 3.6 points and reducing LPIPS by 7.3%. On Tanks and Temples, GloSplat-F achieves +1.05 dB improvement in PSNR with 2.7 points higher SSIM compared to VGGT-X, demonstrating strong performance on large-scale outdoor scenes with up to 1106 images. On CO3Dv2, GloSplat-F achieves +0.86 dB improvement in PSNR with 2.5 points higher SSIM and 22% lower LPIPS compared to VGGT-X.

Notably, even our fast retrieval-based variant approaches and in some cases exceeds COLMAP-initialized baselines. On MipNeRF360, we achieve 99.5% of MCMCâ âs PSNR. On Tanks and Temples, we surpass MCMCâ  in SSIM (0.869 vs 0.867) while achieving comparable PSNR. On

Table 1. Comparison with state-of-the-art methods on novel view synthesis. â  indicates COLMAP initialization. Best COLMAP-free results in bold. GloSplat-F (retrieval-based) achieves SOTA among COLMAP-free methods.
<table><tr><td rowspan="2">Method</td><td colspan="3">MipNeRF360</td><td colspan="3">Tanks and Temples</td><td colspan="3">CO3Dv2</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGSâ </td><td>27.39</td><td>0.815</td><td>0.185</td><td>24.85</td><td>0.851</td><td>0.155</td><td>32.58</td><td>0.938</td><td>0.095</td></tr><tr><td>MCMCâ </td><td>27.91</td><td>0.836</td><td>0.154</td><td>25.76</td><td>0.867</td><td>0.139</td><td>33.21</td><td>0.941</td><td>0.097</td></tr><tr><td>MCMC</td><td>22.19</td><td>0.548</td><td>0.282</td><td>21.42</td><td>0.679</td><td>0.278</td><td>25.71</td><td>0.712</td><td>0.201</td></tr><tr><td>CF-3DGS</td><td>12.38</td><td>0.234</td><td>0.719</td><td>12.19</td><td>0.391</td><td>0.608</td><td>20.18</td><td>0.611</td><td>0.435</td></tr><tr><td>HT-3DGS</td><td>14.79</td><td>0.380</td><td>0.669</td><td>13.83</td><td>0.451</td><td>0.585</td><td>28.28</td><td>0.833</td><td>0.230</td></tr><tr><td>3RGS</td><td>25.39</td><td>0.713</td><td>0.216</td><td>21.47</td><td>0.750</td><td>0.300</td><td>31.07</td><td>0.878</td><td>0.128</td></tr><tr><td>VGGT-X</td><td>26.40</td><td>0.782</td><td>0.177</td><td>24.77</td><td>0.842</td><td>0.168</td><td>31.85</td><td>0.911</td><td>0.113</td></tr><tr><td>GloSplat-F (Ours)</td><td>27.77</td><td>0.818</td><td>0.164</td><td>25.82</td><td>0.869</td><td>0.151</td><td>32.71</td><td>0.936</td><td>0.088</td></tr></table>

CO3Dv2, we surpass 3DGSâ  in both PSNR (+0.13 dB) and LPIPS (0.088 vs 0.095), demonstrating that our joint poseappearance optimization with global SfM can match or exceed traditional incremental SfM quality without exhaustive matching.

Per-scene results are reported in Tables 4 to 6 in the appendix. GloSplat-F performs consistently across diverse scene types, achieving over 33 dB on object-centric scenes (Apple, Skateboard, Teddybear), strong results on largescale outdoor structures (Barn, Ignatius), and maintaining competitive quality on challenging scenes with complex geometry (Flowers, Stump, Treehill). This consistency validates that joint pose-appearance optimization generalizes across scene types.

## 4.3. Ablation Study

We conduct comprehensive ablation experiments to isolate the contribution of each component in GloSplat-F. The full study is presented in Section D, structured to explicitly attribute gains to specific design choices.

Isolating Joint Optimization. Removing the joint BA loss causes â0.81 dB degradation, directly measuring our geometric anchoring contribution. Freezing poses entirely after SfM causes â8.59 dB degradation; the gap (7.78 dB) represents photometric pose refinementâs ability to correct SfM errors.

Adopted Components. MCMC densification contributes +1.75 dBâwe transparently acknowledge this is an adopted component from prior work (Kheradmand et al., 2024a), not a novel contribution.

Infrastructure Choices (GloSplat-F). Alternative feature pipelines (DISK, R2D2) and retrieval methods (OpenIBL, DIR) show significant degradation, confirming

XFeat+LightGlue and MegaLoc as optimal for the fast variant.

Isolating SfM Contribution. To disentangle global SfM from joint optimization, we run COLMAP (exhaustive matching) with our joint BA loss enabled. This achieves 28.52 dBâ+0.61 dB over MCMCâ  (27.91 dB, frozen poses), while GloSplat-A achieves 28.86 dB. This cleanly attributes: joint optimization contributes +0.61 dB (64%); global SfM contributes +0.34 dB (36%). Joint optimization is the primary contributor, validating our core thesis.

We acknowledge GloSplat is a systems contribution where components work synergistically. Joint optimization provides the majority of gains (+0.61 dB), while global SfM provides additional improvement through better initialization (+0.34 dB).

## 4.4. Analysis

Why Joint Pose-Appearance Optimization Outperforms Modular Pipelines. Traditional pipelines treat SfM and NVS as independent modules with frozen interfaces. VGGT-X relies on feed-forward networks that can suffer from distribution shift. COLMAP-based methods freeze poses after SfM, preventing photometric refinement. Prior joint methods (BARF, NeRFâ, 3RGS) optimize poses using only photometric gradients, which causes early-stage drift when the radiance field is poorly initialized. In contrast, GloSplatâs persistent feature track architecture provides several advantages: (1) track 3D points as separate parameters from Gaussian means enable geometric anchoring independent of rendering quality; (2) the reprojection loss on explicit correspondences prevents pose drift even when Gaussians are sparse; (3) global SfM with rotation averaging distributes initialization error across all views; (4) the dual photometricgeometric supervision allows poses to benefit from finegrained appearance gradients after geometric anchoring

stabilizes the optimization.

On Camera Pose Evaluation. We provide direct pose evaluation on ScanNet (Table 3), where ground-truth poses are available. GloSplat-F achieves the best rotation error and ATE across all scenes, outperforming both COLMAP and 3RGS. Beyond direct metrics, rendering quality itself serves as a strong proxy for pose accuracy: our densification strategy is identical to MCMC-3DGS, isolating pose quality as the primary variable. As shown in Table 1, MCMC without accurate poses achieves only 22.19 dB on MipNeRF360, while MCMCâ  with COLMAP poses achieves 27.91 dBâa 5.72 dB gap attributable entirely to pose quality since densification and training are identical. GloSplat-F achieves 27.77 dB using the same MCMC densification, demonstrating that our poses match COLMAP quality. Furthermore, our ablation (Table 9) shows that freezing poses causes catastrophic 8.59 dB degradationâif improvements came from densification alone, pose freezing would not cause such failure.

Computational Efficiency. The two-variant design offers flexibility for different use cases. GloSplat-F with learned features (XFeat+LightGlue) and retrieval-based pair selection is significantly faster, reducing complexity from $O ( n ^ { 2 } )$ to $O ( n )$ . GloSplat-A uses the same SIFT features and exhaustive matching as COLMAP, ensuring fair comparison where gains are attributable solely to global SfM and joint optimization. Both variants benefit from global SfMâs natural parallelizationâall camera poses are solved simultaneously rather than sequentially. A key enabler of our speed advantage is the use of cuDSS (NVIDIA Corporation, 2024) for GPU-accelerated sparse linear solving in bundle adjustment, which exploits the inherent sparsity of the Jacobian structure and enables fully parallel optimization on modern GPUs. COLMAP was compiled with CUDA support and GPU acceleration for all applicable stages (SIFT-GPU, matching, geometric verification); the remaining speedup reflects our global SfMâs inherent parallelism and retrieval-based pair selection, not hardware asymmetry. Combined with efficient gsplat rasterization, GloSplat provides a spectrum of speed-quality trade-offs while consistently outperforming prior methods in each category.

Figure 3 presents runtime comparisons on the Courthouse scene from Tanks and Temples, varying the number of input images from 250 to 1000 (see Table 7 in the appendix for detailed numbers). GloSplat-F demonstrates excellent scalability: while COLMAPâs runtime grows super-linearly due to incremental SfMâs sequential nature and exhaustive matching, GloSplat-F exhibits near-linear scaling thanks to retrieval-based pair selection and parallel global SfM. At 1000 images, GloSplat-F achieves a 13.3Ã speedup over COLMAP while delivering superior reconstruction quality

<!-- image-->  
Figure 3. Runtime Comparison on Courthouse Scene. End-toend reconstruction time (in seconds) as a function of the number of input images. All methods use the same GPU (RTX PRO 6000); COLMAP is compiled with CUDA and uses GPU acceleration for feature extraction/matching. GloSplat-F achieves 13.3Ã speedup over COLMAP at 1000 images due to retrieval-based pair selection and parallel global SfM. VGGT-X is faster at smaller scales but GloSplat-F surpasses it at 750+ images due to better asymptotic scaling.

(Table 1).

Notably, VGGT-X is faster at smaller image counts (250â 500 images) due to its feed-forward architecture, but GloSplat-F overtakes it at 750+ images. This crossover occurs because VGGT-Xâs runtime scales with the number of view pairs processed by its attention mechanism, while GloSplat-Fâs retrieval limits the matching graph to a constant number of neighbors per image. At 1000 images, GloSplat-F is 1.2Ã faster than VGGT-X while achieving significantly higher rendering quality (+1.37 dB PSNR on MipNeRF360). GloSplat-A, using exhaustive matching, is slower but still 3.2Ã faster than COLMAP while achieving the highest quality among all methods.

## 4.5. GloSplat-A: Surpassing COLMAP-Based Methods

Our second variant, GloSplat-A, uses exhaustive feature matching to maximize reconstruction quality. Table 2 compares against state-of-the-art COLMAP-based 3DGS methods on MipNeRF360.

GloSplat-A achieves state-of-the-art performance, outperforming all COLMAP-based methods by significant margins. Compared to the previous best method Improved-GS, we improve PSNR by +0.67 dB (+2.4%), SSIM by 2.6 points (+3.1%), and reduce LPIPS by 25.3%. These results validate our core thesis: joint pose-appearance optimization during 3DGS training, combined with global SfM initialization, provides superior geometric consistency compared to the traditional frozen-pose pipeline. While COLMAPbased methods treat pose estimation as a preprocessing step with frozen outputs, our joint optimization enables photometric gradients to refine camera poses throughout training, leading to higher-quality reconstructions.

Table 2. Comparison with COLMAP-based methods on MipNeRF360. All baselines use COLMAP poses. GloSplat-A achieves SOTA among all methods. Best in bold, second best underlined.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGS</td><td>27.48</td><td>0.815</td><td>0.216</td></tr><tr><td>AbsGS</td><td>27.52</td><td>0.820</td><td>0.198</td></tr><tr><td>PixelGS</td><td>27.62</td><td>0.824</td><td>0.189</td></tr><tr><td>MiniSplatting-D</td><td>27.57</td><td>0.832</td><td>0.176</td></tr><tr><td>TamingGS</td><td>27.96</td><td>0.822</td><td>0.207</td></tr><tr><td>3DGS-MCMC</td><td>28.01</td><td>0.835</td><td>0.186</td></tr><tr><td>SteepGS</td><td>27.05</td><td>0.795</td><td>0.247</td></tr><tr><td>Perceptual-GS</td><td>27.77</td><td>0.829</td><td>0.187</td></tr><tr><td>Improved-GS</td><td>28.19</td><td>0.836</td><td>0.186</td></tr><tr><td>GloSplat-A (Ours)</td><td>28.86</td><td>0.862</td><td>0.139</td></tr></table>

Table 3. Comparison with COLMAP and 3RGS on ScanNet. Best in bold, second best underlined.
<table><tr><td rowspan="2">Scene</td><td colspan="3">COLMAP</td><td colspan="3">3RGS</td><td colspan="3">GloSplat-F (Ours)</td></tr><tr><td>R()</td><td>ATE(m)â</td><td>PSNRâ</td><td>R()</td><td>ATE(m)â</td><td>PSNRâ</td><td>R()â</td><td>ATE(m)â</td><td>PSNRâ</td></tr><tr><td>0079_00</td><td>3.55</td><td>0.014</td><td>30.78</td><td>2.45</td><td>0.014</td><td>32.58</td><td>2.12</td><td>0.011</td><td>33.42</td></tr><tr><td>0301_00</td><td>133.83</td><td>0.169</td><td>23.63</td><td>9.30</td><td>0.009</td><td>30.11</td><td>7.82</td><td>0.007</td><td>31.24</td></tr><tr><td>0418_00</td><td>5.03</td><td>0.012</td><td>29.03</td><td>4.34</td><td>0.012</td><td>31.62</td><td>3.78</td><td>0.010</td><td>32.35</td></tr></table>

Comparison with 3RGS on ScanNet. We additionally compare against 3RGS (Huang et al., 2025) on ScanNet (Dai et al., 2017) to evaluate both pose estimation accuracy and rendering quality. Table 3 reports rotation error (R), absolute trajectory error (ATE), and PSNR. GloSplat-F achieves the best pose accuracy on all scenes while also delivering higher rendering quality, demonstrating that our joint photometricgeometric optimization provides superior pose refinement compared to 3RGSâs photometric-only approach.

Qualitative comparisons are presented in Figure 4 in the appendix.

## 5. Conclusion

We have presented GloSplat, a unified framework for 3D reconstruction that introduces a key architectural novelty: preserving explicit SfM feature tracks as first-class entities during 3D Gaussian Splatting training, with track 3D points maintained as separate optimizable parameters from Gaussian means. Unlike prior joint optimization methods (BARF, NeRFâ, 3RGS) that rely purely on photometric gradients and suffer from early-stage pose drift, our dual photometricgeometric supervision provides persistent anchoring that stabilizes optimization while enabling fine-grained pose refinement. Our two variants, GloSplat-F (retrieval-based) and GloSplat-A (exhaustive matching), achieve state-of-the-art results in their respective categories: GloSplat-F establishes new performance standards among COLMAP-free methods while offering significant speedups, and GloSplat-A surpasses all COLMAP-based baselines, demonstrating that joint pose-appearance optimization with global SfM can outperform incremental approaches.

Limitations and Future Work. Our work has several limitations that warrant future investigation. First, the feature extraction, matching, and pair selection stages remain frozen preprocessingâgradients do not flow back through these components. Consequently, our joint optimization operates only during the 3DGS training phase, refining camera poses and Gaussian primitives but not the upstream modules. The performance gap between GloSplat-F (27.77 dB) and GloSplat-A (28.86 dB) on MipNeRF360âa difference of 1.09 dBâreflects both the sparser matching graph from retrieval-based selection and differences between learned (XFeat) and classical (SIFT) features. When retrieval-based pair selection misses important overlapping views, global SfM initialization degrades, and subsequent joint optimization cannot fully recover. Future work should explore more robust matching strategies or learned retrieval methods that better identify challenging but informative image pairs.

Second, a fully end-to-end differentiable approach, where gradients from rendering losses flow back through SfM and into the feature extractor itself, remains an open challenge. Such a unified architecture would enable the feature network to learn representations optimized for downstream reconstruction quality rather than generic matching performance, though this requires significant engineering effort and may introduce stability challenges during training.

## Impact Statement

This paper presents work whose goal is to advance the field of 3D reconstruction and novel view synthesis. We hope that GloSplatâs success in jointly optimizing camera poses and radiance fields will inspire the broader computer vision community to reconsider frozen-interface designs between pipeline stages. The demonstrated gains from joint poseappearance optimizationâwhere camera poses continue to receive gradients during appearance learningâsuggest that similar principles may benefit other multi-stage vision pipelines, from SLAM systems to multi-modal reconstruction. We encourage researchers to consider cross-stage gradient flow when designing future 3D vision systems, as the boundaries between âpreprocessingâ and âmain tasksâ are often artificial constraints inherited from historical software architectures rather than fundamental algorithmic necessities.

## References

Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., and Srinivasan, P. P. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5855â5864, 2021.

Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., and Hedman, P. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5470â5479, 2022.

Berton, G., Mereu, R., Trivigno, G., Masone, C., Csurka, G., Sattler, T., and Caputo, B. MegaLoc: One retrieval to place them all. In European Conference on Computer Vision (ECCV), 2024.

Bian, J.-W., Bian, W., Prisacariu, V. A., and Torr, P. Porf: Pose residual field for accurate neural surface reconstruction. In ICLR, 2024.

Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., and NieÃner, M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.

DeTone, D., Malisiewicz, T., and Rabinovich, A. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 224â236, 2018.

Fetzer, T., Pagani, A., and Stricker, D. Stable intrinsic auto-calibration from fundamental matrices of devices with uncorrelated camera parameters. arXiv preprint arXiv:2007.12240, 2020.

Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A. A., and Wang, X. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20796â20805, 2024.

Hartley, R., Trumpf, J., Dai, Y., and Li, H. Rotation averaging. International journal of computer vision, 103: 267â305, 2013.

Huang, Z., Wang, P., Zhang, J., Liu, Y., Li, X., and Wang, W. 3r-gs: Best practice in optimizing camera poses along with 3dgs. arXiv preprint arXiv:2504.04294, 2025.

Ji, B. and Yao, A. Sfm-free 3d gaussian splatting via hierarchical training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21654â21663, 2025.

Kerbl, B., Kopanas, G., Leimkuhler, T., and Drettakis, G. 3d Â¨ gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng, J., Isber, H., Kar, A., Tagliasacchi, A., and Yi, K. M. 3D Gaussian Splatting as Markov Chain Monte Carlo. In Advances in Neural Information Processing Systems (NeurIPS), 2024a.

Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng, Y.-C., Isack, H., Kar, A., Tagliasacchi, A., and Yi, K. M. 3d gaussian splatting as markov chain monte carlo. Advances in Neural Information Processing Systems, 37: 80965â80986, 2024b.

Knapitsch, A., Park, J., Zhou, Q.-Y., and Koltun, V. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1â13, 2017.

Leroy, V., Cabon, Y., and Revaud, J. Grounding image matching in 3d with mast3r, 2024.

Lin, C.-H., Ma, W.-C., Torralba, A., and Lucey, S. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5741â5751, 2021.

Lindenberger, P., Sarlin, P.-E., and Pollefeys, M. LightGlue: Local feature matching at light speed. In International Conference on Computer Vision (ICCV), 2023.

Lowe, D. G. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60: 91â110, 2004.

Mallick, S. S., Goel, R., Kerbl, B., Steinberger, M., Carranza, F. V., and Wimmer, M. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, 2024.

Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

Moulon, P., Monasse, P., and Marlet, R. Global fusion of relative motions for robust, accurate and scalable structure from motion. In Proceedings of the IEEE international conference on computer vision, pp. 3248â3255, 2013.

Muller, T., Evans, A., Schied, C., and Keller, A. Instant Â¨ neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4): 1â15, 2022.

Newcombe, R. A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A. J., Kohi, P., Shotton, J., Hodges, S., and Fitzgibbon, A. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality, pp. 127â136. Ieee, 2011.

NVIDIA Corporation. cuDSS: CUDA data-parallel sparse direct solver library. https://docs.nvidia.com/ cuda/cudss/index.html, 2024. GPU-accelerated sparse direct linear solver for symmetric and unsymmetric systems.

Pan, L., Barath, D., Pollefeys, M., and Sch Â´ onberger, J. L.Â¨ Global structure-from-motion revisited. In European Conference on Computer Vision, pp. 58â77. Springer, 2024.

Potje, G., Cadar, F., Araujo, A., Martins, R., and Nascimento, E. R. Xfeat: Accelerated features for lightweight image matching. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2682â 2691, 2024a. doi: 10.1109/CVPR52733.2024.00259.

Potje, G., Cadar, F., Araujo, A., Martins, R., and Nascimento, E. R. XFeat: Accelerated features for lightweight image matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2682â2691, 2024b.

Reizenstein, J., Shapovalov, R., Henzler, P., Sbordone, L., Labatut, P., and Novotny, D. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Oct 2021. doi: 10.1109/iccv48922.2021.01072. URL http://dx. doi.org/10.1109/iccv48922.2021.01072.

Sarlin, P.-E., DeTone, D., Malisiewicz, T., and Rabinovich, A. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4938â4947, 2020.

Schonberger, J. L. and Frahm, J.-M. Structure-from-motion Â¨ revisited. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Triggs, B., McLauchlan, P. F., Hartley, R. I., and Fitzgibbon, A. W. Bundle adjustmentâa modern synthesis. In Vision Algorithms: Theory and Practice: International Workshop on Vision Algorithms Corfu, Greece, September 21â22, 1999 Proceedings, pp. 298â372. Springer, 2000.

Truong, P., Rakotosaona, M.-J., Manhardt, F., and Tombari, F. Sparf: Neural radiance fields from sparse and noisy poses. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pp. 4190â 4200, 2023.

Wang, J., Karaev, N., Rupprecht, C., and Novotny, D. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21686â 21697, 2024a.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294â5306, 2025.

Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697â20709, 2024b.

Wang, Z., Wu, S., Xie, W., Chen, M., and Prisacariu, V. A. NeRFââ: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021.

Wilson, K. and Snavely, N. Robust global translations with 1dsfm. In European conference on computer vision, pp. 61â75. Springer, 2014.

Ye, V., Turkulainen, M., and the Nerfstudio team. gsplat: An open-source library for Gaussian splatting. arXiv preprint arXiv:2409.06765, 2024a.

Ye, Z., Li, W., Liu, S., Qiao, P., and Dou, Y. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 1053â1061, 2024b.

Zhan, Z., Xu, H., Fang, Z., Wei, X., Hu, Y., and Wang, C. Bundle adjustment in the eager mode. arXiv preprint arXiv:2409.12190, 2024.

Zhan, Z., Xu, H., Fang, Z., Wei, X., Hu, Y., and Wang, C. Bundle adjustment in the eager mode. arXiv preprint arXiv:2409.12190, 2025. URL https:// arxiv.org/abs/2409.12190.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586â595, 2018.

Zhang, Z., Hu, W., Lao, Y., He, T., and Zhao, H. Pixel-gs: Density control with pixel-aware gradient for 3d gaussian splatting. In European Conference on Computer Vision (ECCV), 2024.

## A. Qualitative Results

<!-- image-->  
Garden â GloSplat-A: 30.82 dB / 0.074 LPIPS, VGGT-X: 25.13 dB / 0.101 LPIPS

Figure 4. Qualitative Comparison on MipNeRF360. We compare novel view synthesis results from GloSplat-A, GloSplat-F, VGGT-X, and Improved-GS against ground truth. GloSplat-A achieves significantly higher PSNR and lower LPIPS across all scenes. On Bonsai, GloSplat-A outperforms VGGT-X by +8.57 dB PSNR. On Flowers, our method shows +6.76 dB PSNR and 46% lower LPIPS (0.147 vs 0.273) over VGGT-X. On Garden, GloSplat-A achieves +5.69 dB over VGGT-X. Best viewed zoomed in.

Figure 4 presents visual comparisons on representative scenes from MipNeRF360. On the Bonsai scene, GloSplat-A produces notably sharper leaf structures and intricate branch details, achieving 36.39 dB PSNR compared to VGGT-Xâs 27.82 dBâa remarkable 8.57 dB improvement. The Flowers scene demonstrates GloSplatâs ability to handle challenging thin structures: our method achieves 46% better perceptual quality (0.147 vs 0.273 LPIPS) with cleaner flower petal boundaries. On the outdoor Garden scene, joint pose-appearance optimization produces cleaner foliage edges and more accurate colors, outperforming VGGT-X by 5.69 dB. These results illustrate that continuous pose refinement during 3DGS training is particularly beneficial for scenes with fine details and complex structures.

## B. Per-Scene Results

We report detailed per-scene results for GloSplat-F on all three benchmark datasets. These tables complement the aggregate results in Table 1 by revealing performance variation across different scene characteristics. Notably, GloSplat-F achieves particularly strong results on object-centric scenes (Apple, Skateboard, Teddybear in CO3Dv2 with PSNR > 33 dB), while maintaining competitive performance on challenging outdoor scenes with complex geometry and lighting (Flowers, Stump, Treehill in MipNeRF360). The consistent performance across diverse scene typesâfrom indoor tabletop captures to largescale outdoor environmentsâvalidates that our joint pose-appearance optimization generalizes well without scene-specific tuning.

## C. Runtime Comparison

We provide detailed runtime measurements for full-pipeline reconstruction on the Courthouse scene from Tanks and Temples, which contains up to 1106 images and serves as a representative benchmark for scalability analysis. All experiments were conducted on a server with an NVIDIA RTX PRO 6000 GPU (96GB memory), AMD Ryzen 9 9950X CPU, running Ubuntu 24.04. For fair comparison, COLMAP was compiled with CUDA support and run with GPU acceleration enabled for all applicable stages (feature extraction via SIFT-GPU, exhaustive matching, and geometric verification). Runtime includes all stages: feature extraction, pair selection (retrieval or exhaustive), feature matching, SfM (incremental for COLMAP, global for GloSplat), and 3DGS training (30k iterations). As shown in Table 7, GloSplat-F demonstrates superior asymptotic scaling due to its O(n) retrieval-based pair selection and parallel global SfM, achieving a 13.3Ã speedup over COLMAP at 1000 images. The crossover point with VGGT-X occurs around 750 images, after which GloSplat-Fâs linear scaling provides increasing advantages.

Table 4. Per-scene results on CO3Dv2 dataset.
<table><tr><td>Scene</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Apple</td><td>35.30</td><td>0.952</td><td>0.063</td></tr><tr><td>Bench</td><td>32.33</td><td>0.931</td><td>0.089</td></tr><tr><td>Hydrant</td><td>29.33</td><td>0.920</td><td>0.071</td></tr><tr><td>Skateboard</td><td>33.35</td><td>0.943</td><td>0.116</td></tr><tr><td>Teddybear</td><td>33.24</td><td>0.932</td><td>0.102</td></tr><tr><td>Average</td><td>32.71</td><td>0.936</td><td>0.088</td></tr></table>

Table 5. Per-scene results on MipNeRF360 dataset.
<table><tr><td>Scene</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Bicycle</td><td>25.98</td><td>0.802</td><td>0.145</td></tr><tr><td>Bonsai</td><td>33.16</td><td>0.952</td><td>0.114</td></tr><tr><td>Counter</td><td>29.60</td><td>0.921</td><td>0.131</td></tr><tr><td>Flowers</td><td>22.93</td><td>0.688</td><td>0.244</td></tr><tr><td>Garden</td><td>27.64</td><td>0.866</td><td>0.084</td></tr><tr><td>Kitchen</td><td>32.99</td><td>0.942</td><td>0.085</td></tr><tr><td>Room</td><td>30.98</td><td>0.917</td><td>0.154</td></tr><tr><td>Stump</td><td>24.27</td><td>0.650</td><td>0.209</td></tr><tr><td>Treehill</td><td>22.34</td><td>0.621</td><td>0.308</td></tr><tr><td>Average</td><td>27.77</td><td>0.818</td><td>0.164</td></tr></table>

Training Overhead of Joint Optimization. We additionally measure the training time overhead introduced by our joint BA loss compared to vanilla MCMC 3DGS training. Table 8 reports training times on a representative scene with 200 images. The joint optimization adds only marginal overhead (â¼3%) over vanilla MCMC 3DGS, as the reprojection loss computation on sparse feature tracks is computationally lightweight compared to the differentiable rasterization that dominates training time. This demonstrates that our geometric anchoring comes at negligible computational cost while providing significant quality improvements (+0.81 dB as shown in Table 9).

## D. Extended Ablation Study

We conduct comprehensive ablation experiments on MipNeRF360 to isolate the contribution of each pipeline component.   
Table 9 presents results organized by component category, with detailed analysis below.

Reference Configurations. We include three reference points: GloSplat-A (28.86 dB) uses exhaustive matching for maximum quality; GloSplat-F (27.77 dB) uses retrieval-based matching for speed; MCMCâ  (27.91 dB) is the COLMAP baseline with frozen poses. The â column shows change relative to GloSplat-F.

(A) Joint Optimization. We ablate the joint pose-appearance optimization from GloSplat-F:

â¢ Frozen poses: Disabling all pose optimization after Global SfM causes â8.59 dB degradation, showing that pose refinement during 3DGS training is essential.

â¢ Photometric-only: Removing the BA loss (poses receive only photometric gradients) causes â0.81 dB degradation, isolating the geometric anchoring contribution.

Table 6. Per-scene results on Tanks and Temples dataset.
<table><tr><td>Scene</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Barn</td><td>26.43</td><td>0.842</td><td>0.168</td></tr><tr><td>Caterpillar</td><td>24.89</td><td>0.851</td><td>0.162</td></tr><tr><td>Ignatius</td><td>27.15</td><td>0.912</td><td>0.121</td></tr><tr><td>Truck</td><td>25.31</td><td>0.867</td><td>0.148</td></tr><tr><td>Train</td><td>25.32</td><td>0.873</td><td>0.156</td></tr><tr><td>Average</td><td>25.82</td><td>0.869</td><td>0.151</td></tr></table>

Table 7. Runtime comparison (seconds) on Courthouse scene with varying image counts. All methods run on the same hardware (RTX PRO 6000 GPU); COLMAP uses GPU-accelerated SIFT and matching. GloSplat-F achieves superior scaling due to O(n) pair selection and parallel global SfM.
<table><tr><td>Method</td><td>250 imgs</td><td>500 imgs</td><td>750 imgs</td><td>1000 imgs</td></tr><tr><td>COLMAP</td><td>644.6</td><td>1165.0</td><td>2226.3</td><td>4349.2</td></tr><tr><td>3RGS (MASt3R)</td><td>312.2</td><td>638.0</td><td>978.5</td><td>1308.2</td></tr><tr><td>GloSplat-A</td><td>222.2</td><td>408.2</td><td>855.7</td><td>1371.3</td></tr><tr><td>VGGT-X</td><td>53.9</td><td>113.3</td><td>249.6</td><td>398.9</td></tr><tr><td>GloSplat-F</td><td>100.2</td><td>191.9</td><td>250.0</td><td>327.8</td></tr><tr><td>Speedup vs COLMAP GloSplat-F</td><td>6.4Ã</td><td>6.1Ã</td><td>8.9Ã</td><td>13.3Ã</td></tr></table>

(B) SfM Backend. To isolate global SfMâs contribution from joint optimization:

COLMAP (exhaustive) + Joint Opt: Using COLMAP initialization with our joint BA loss achieves 28.52 dB. Compared to MCMCâ  (27.91 dB, frozen), joint optimization adds +0.61 dB. Compared to GloSplat-A (28.86 dB), global SfM adds +0.34 dB. This confirms: joint optimization is the primary contributor (+0.61 dB), while global SfM provides additional gains through better initialization (+0.34 dB).

â¢ COLMAP (retrieval): COLMAP with sparse retrieval-based matching fails catastrophically (â9.84 dB), as incremental SfM cannot handle sparse view graphs.

(C) Densification Strategy. MCMC densification (Kheradmand et al., 2024a) contributes +1.75 dB. We transparently acknowledge this is an adopted component, not a novel contribution.

(D) Feature Matching (GloSplat-F). SuperPoint+SuperGlue achieves competitive results (â0.71 dB). DISK and R2D2 cause severe degradation, indicating XFeat+LightGlue is well-suited for the fast variant.

(E) Image Retrieval. NetVLAD shows â1.42 dB vs MegaLoc. OpenIBL/DIR fail due to domain mismatch.

Attribution Summary. We decompose GloSplat-Aâs advantage over MCMCâ  (+0.95 dB):

<table><tr><td>Component</td><td>Contribution</td><td>Fraction</td></tr><tr><td>Joint optimization</td><td>+0.61 dB</td><td>64%</td></tr><tr><td>Global SfM</td><td>+0.34 dB</td><td>36%</td></tr></table>

GloSplat is a systems contribution where joint optimization is the primary driver (+0.61 dB, 64%), validating our core thesis that continuous pose refinement during 3DGS training improves reconstruction quality. Global SfM provides additional gains through better initialization (+0.34 dB, 36%), and MCMC densification (+1.75 dB) is an adopted component. The full systemâs performance depends on careful integration of all components.

Table 8. Training time comparison between vanilla MCMC 3DGS and MCMC 3DGS with joint optimization (30k iterations, 200 images). Joint optimization adds minimal overhead while providing +0.81 dB improvement.
<table><tr><td>Configuration</td><td>Training Time</td><td>Overhead</td></tr><tr><td>MCMC 3DGS (vanilla)</td><td>~31 min</td><td>â</td></tr><tr><td>MCMC 3DGS + Joint Opt (Ours)</td><td>~32 min</td><td>+3%</td></tr></table>

Table 9. Ablation study on MipNeRF360. We show reference configurations (top) and ablations organized by component. â shows change from GloSplat-F. Best in bold; âââ indicates failure.
<table><tr><td>Configuration</td><td>PSNRâ</td><td>â</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td colspan="5">Reference Configurations</td></tr><tr><td>GloSplat-A (exhaustive matching)</td><td>28.86</td><td>+1.09</td><td>0.862</td><td>0.139</td></tr><tr><td>GloSplat-F (retrieval matching)</td><td>27.77</td><td></td><td>0.818</td><td>0.164</td></tr><tr><td>MCMCâ  (COLMAP, frozen poses)</td><td>27.91</td><td>+0.14</td><td>0.836</td><td>0.154</td></tr><tr><td colspan="5">(A) Joint Optimization â removing from GloSplat-F</td></tr><tr><td>Frozen poses after Global SfM</td><td>19.18</td><td>-8.59</td><td>0.463</td><td>0.472</td></tr><tr><td>Photometric-only (no BA loss)</td><td>26.96</td><td>-0.81</td><td>0.783</td><td>0.190</td></tr><tr><td colspan="5">(B) SfM Backend â isolating Global SfM vs COLMAP</td></tr><tr><td>COLMAP (exhaustive) + Joint Opt</td><td>28.52</td><td>+0.75</td><td>0.854</td><td>0.144</td></tr><tr><td>COLMAP (retrieval) + no Joint Opt</td><td>17.93</td><td>-9.84</td><td>0.445</td><td>0.528</td></tr><tr><td colspan="5">(C) Densification Strategy</td></tr><tr><td>Standard densification (no MCMC)</td><td>26.02</td><td>-1.75</td><td>0.746</td><td>0.239</td></tr><tr><td colspan="5">(D) Feature Matching Pipeline</td></tr><tr><td>SuperPoint + SuperGlue</td><td>27.06</td><td>-0.71</td><td>0.791</td><td>0.187</td></tr><tr><td>DISK + LightGlue</td><td>19.65</td><td>-8.12</td><td>0.494</td><td>0.516</td></tr><tr><td>R2D2 + NN Matcher</td><td>17.54</td><td>-10.23</td><td>0.473</td><td>0.673</td></tr><tr><td colspan="5">(E) Image Retrieval</td></tr><tr><td>NetVLAD</td><td>26.35</td><td>-1.42</td><td>0.762</td><td>0.208</td></tr><tr><td>OpenIBL</td><td>20.68</td><td>-7.09</td><td>0.554</td><td>0.452</td></tr><tr><td>DIR (AP-GeM)</td><td></td><td></td><td></td><td></td></tr></table>

Why Separate Track Points from Gaussian Means? A natural question is whether maintaining track 3D points $\{ { \bf X } _ { k } \}$ as separate parameters from Gaussian means $\{ \mu _ { j } \}$ is necessary, or whether one could simply merge themâusing Gaussian positions directly for both rendering and reprojection constraints. We ablate this design choice by initializing Gaussians at track point locations and applying the reprojection loss directly to the corresponding Gaussian means. Table 10 reports results on MipNeRF360.

Naively merging track points with Gaussian means causes consistent degradation across all metrics. GloSplat-A degrades by â0.22 dB PSNR, while GloSplat-F shows â0.19 dBâboth exhibiting similar degradation patterns despite different matching strategies. This indicates that the architectural benefit of separation is fundamental to the optimization dynamics, not an artifact of any particular pipeline configuration. This validates our architectural choice of maintaining separate parameters for two reasons:

1. Conflicting gradient signals. Gaussian means receive gradients from photometric rendering that optimize for appearance qualityâpushing primitives to minimize $\ell _ { 1 }$ and SSIM losses. Track points receive gradients from reprojection that enforce multi-view geometric consistency. When merged, these objectives compete: a Gaussian may need to move for better rendering but should stay fixed for geometric anchoring, creating optimization conflicts that degrade both.

Table 10. Ablation: Separating track points from Gaussian means. Merging tracks with Gaussians degrades all metrics for both variants, validating our architectural choice.
<table><tr><td>Configuration</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>GloSplat-A (separate tracks) GloSplat-A (merged tracks)</td><td>28.86 28.64</td><td>0.862 0.858</td><td>0.139 0.144</td></tr><tr><td> $\Delta$ </td><td>0.22</td><td>0.004</td><td>+0.005</td></tr><tr><td>GloSplat-F (separate tracks)</td><td>27.77</td><td>0.818</td><td>0.164</td></tr><tr><td>GloSplat-F (merged tracks)</td><td>27.58</td><td>0.813</td><td>0.168</td></tr><tr><td> $\Delta$ </td><td>-0.19</td><td>-0.005</td><td>+0.004</td></tr></table>

2. Densification disrupts geometric anchors. MCMC densification performs stochastic birth-death processes that relocate, split, and merge Gaussians based on rendering quality. If track points were tied to Gaussian means, densification would inadvertently modify the geometric anchors, breaking the multi-view consistency constraints. Separate track points remain stable throughout training regardless of how Gaussians evolve.

This ablation confirms that the separation is not merely an implementation detail but a principled architectural choice that enables photometric and geometric objectives to coexist without interference.

## E. Algorithm: Joint Optimization Loop

Algorithm 1 presents the complete joint optimization loop that distinguishes GloSplat from prior work. The key architectural novelty is maintaining track 3D points $\{ { \bf X } _ { k } \}$ as separate optimizable parameters from Gaussian means $\{ \mu _ { j } \}$ . This enables the reprojection-based BA loss to provide geometric anchoring independent of Gaussian rendering quality, preventing early-stage pose drift while photometric gradients enable fine-grained refinement.

## Key Implementation Details.

â¢ Separate track points (Lines 3, 22â27): Unlike prior joint methods that only optimize Gaussian means $\mu _ { j } ,$ we maintain explicit track 3D points ${ \bf X } _ { k }$ as separate parameters. This enables geometric constraints via reprojection loss independent of Gaussian rendering quality.

â¢ Dual gradient flow (Lines 30â32): Camera poses receive gradients from both photometric loss (via differentiable rendering) and BA loss (via reprojection). The BA loss provides geometric anchoring that stabilizes early training when Gaussians are sparse.

â¢ Projection function (Line 24): $\pi ( \mathbf { K } , \mathbf { T } , \mathbf { X } ) \ = \ \mathrm { p r o j } ( \mathbf { K } ( \mathbf { R } \mathbf { X } + \mathbf { t } ) )$ where $\mathbf { T } ~ = ~ [ \mathbf { R } | \mathbf { t } ]$ and $\mathrm { p r o j } ( [ x , y , z ] ^ { \top } ) ~ =$ $[ x / \widetilde { z } , y / z ] ^ { \top }$

â¢ Hyperparameters: $\lambda _ { \mathrm { S S I M } } = 0 . 2 , \lambda _ { \mathrm { B A } } = 1 0 ^ { - 4 }$ , Î´ = 1.0 (Huber threshold), $\eta _ { \mathrm { p o s e } } = 1 0 ^ { - 5 } , T _ { \mathrm { m a x } } = 3 0 0 0 0$

The full pipeline consists of three stages: (1) glo-feat for feature extraction and matching (frozen preprocessing), (2) glo-sfm for global SfM initialization (rotation averaging, positioning, bundle adjustment), and (3) glo-joint for joint 3DGS training with the algorithm above.

```latex
Algorithm 1 GloSplat Joint Optimization Loop
Require: Images $\{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N } .$ , SfM points $\{ \mathbf { P } _ { k } \}$ , feature tracks $\{ \mathcal { T } _ { k } \}$ , initial poses $\{ \mathbf { T } _ { i } ^ { ( 0 ) } \}$ }, intrinsics $\{ { \bf K } _ { i } \}$
Ensure: Optimized Gaussians G, refined poses $\{ \mathbf { T } _ { i } \}$ , refined track points $\{ { \bf X } _ { k } \}$
1: // Initialize from Global SfM
2: Initialize Gaussians $\mathcal { G }  \{ \mu _ { j } , \Sigma _ { j } , \alpha _ { j } , \mathbf { c } _ { j } \}$ from SfM points
3: Initialize track points $\{ \mathbf { X } _ { k } \} \doteq \{ \mathbf { \bar { P } } _ { k } \}$ {Separate from $\mu _ { j } \}$
4: Initialize pose adjustments $\left\{ \Delta \mathbf { T } _ { i } \right\} \gets \mathbf { 0 }$
5:
6: for $t = 1$ to $T _ { \mathrm { m a x } }$ do
7: // Sample training batch
8: Sample image indices $B \subset \{ 1 , \ldots , N \}$
9:
10: for $i \in \boldsymbol { B }$ do
11: // Apply pose adjustment
12: $\mathbf { T } _ { i }  \mathbf { T } _ { i } ^ { ( 0 ) } \oplus \Delta \mathbf { T } _ { i }$ {SE(3) composition}
13:
14: // Render via 3D Gaussian Splatting
15: $\hat { { \bf I } } _ { i } \gets \mathrm { R a s t e r i z e } ( \mathcal { G } , { \bf T } _ { i } , { \bf K } _ { i } )$
16:
17: // Photometric loss
18: $\mathcal { L } _ { \mathrm { p h o t o } }  ( 1 - \lambda _ { \mathrm { S S I M } } ) \| \hat { \mathbf { I } } _ { i } - \mathbf { I } _ { i } \| _ { 1 } + \lambda _ { \mathrm { S S I M } } ( 1 - \mathrm { S S I M } ( \hat { \mathbf { I } } _ { i } , \mathbf { I } _ { i } ) )$
19: end for
20:
21: // Joint BA loss on track points (key distinction)
22: $\mathcal { L } _ { \mathrm { B A } }  0$
23: for each track $\mathcal { T } _ { k }$ with observations in B do
24: for each observation $( i , \mathbf { x } _ { i , p } ) \in \mathcal { T } _ { k }$ do
25: $\hat { \mathbf { x } } _ { i , p } \gets \pi ( \mathbf { K } _ { i } , \mathbf { T } _ { i } , \mathbf { X } _ { k } )$ {Project track point}
26: $\mathcal { L } _ { \mathrm { B A } } \gets \mathcal { L } _ { \mathrm { B A } } + \rho _ { \mathrm { H u b e r } } ( \| \hat { \mathbf { x } } _ { i , p } - \mathbf { x } _ { i , p } \| ^ { 2 } ; \delta )$
27: end for
28: end for
29:
30: // Combined loss
31: $\mathcal { L }  \mathcal { L } _ { \mathrm { p h o t o } } + \lambda _ { \mathrm { B A } } \mathcal { L } _ { \mathrm { B A } }$
32:
33: // Backward and update all parameters jointly
34: Backpropagate âL
35: Update Gaussians: $\mathcal { G }  \mathcal { G } - \eta _ { \mathcal { G } } \nabla _ { \mathcal { G } } \mathcal { L }$
36: Update poses: $\Delta \mathbf { T } _ { i } \gets \Delta \mathbf { T } _ { i } - \eta _ { \mathrm { p o s e } } \nabla _ { \Delta ^ { \prime } }$ T L {Receives both gradients}
37: Update track points: $\mathbf { X } _ { k } \gets \mathbf { X } _ { k } - \eta _ { \mathrm { { B A } } } \nabla _ { \mathbf { X } _ { k } } \mathcal { L } _ { \mathrm { { I } } }$ BA
38:
39: // MCMC densification (adopted from prior work)
40: G â MCMCDensify(G, gradients)
41: end for
42: return $\mathcal { G } , \{ \mathbf { T } _ { i } ^ { ( 0 ) } \oplus \Delta \mathbf { T } _ { i } \} , \{ \mathbf { X } _ { k } \}$
```