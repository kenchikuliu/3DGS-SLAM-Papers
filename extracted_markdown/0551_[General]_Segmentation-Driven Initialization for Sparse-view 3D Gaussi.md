# Segmentation-Driven Initialization for Sparse-view 3D Gaussian Splatting

Yi-Hsin Li, Thomas Sikora, Senior Member, IEEE, Sebastian Knorr, Senior Member, IEEE,

Marten Sj Ë ostr Â¨ om, Â¨ Senior Member, IEEE,

AbstractâSparse-view synthesis remains a challenging problem due to the difficulty of recovering accurate geometry and appearance from limited observations. While recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time rendering with competitive quality, existing pipelines often rely on Structurefrom-Motion (SfM) for camera pose estimationâan approach that struggles in genuinely sparse-view settings. Moreover, several SfM-free methods replace SfM with multi-view stereo (MVS) models, but generate massive numbers of 3D Gaussians by back-projecting every pixel into 3D space, leading to high memory costs. We propose Segmentation-Driven Initialization for Gaussian Splatting (SDI-GS), a method that mitigates inefficiency by leveraging region-based segmentation to identify and retain only structurally significant regions. This enables selective downsampling of the dense point cloud, preserving scene fidelity while substantially reducing Gaussian count. Experiments across diverse benchmarks show that SDI-GS reduces Gaussian count by up to 50% and achieves comparable or superior rendering quality in PSNR and SSIM, with only marginal degradation in LPIPS. It further enables faster training and lower memory footprint, advancing the practicality of 3DGS for constrainedview scenarios.

Index Termsâsegmentation, gaussian splatting, sparse-view rendering

## I. INTRODUCTION

Sparse-view rendering, the task of synthesizing novel views from a limited number of input images, has become a critical challenge in 3D computer vision. It is especially relevant in real-world applications such as robotics [1], augmented reality [2], and medical imaging [3], [4], where acquiring densely sampled views is either impractical or impossible. This paper investigates how segmentation can be leveraged within Gaussian Splatting to intelligently select a compact yet sufficient set of Gaussians for efficient reconstruction from sparse views.

A wave of progress in view synthesis was sparked by Neural Radiance Fields (NeRF) [5]. It models scenes as continuous volumetric fields via implicit neural networks. While initially reliant on dense inputs, several variants [6]â[9] have sought to extend NeRFâs capabilities to sparse-view settings. These methods introduce auxiliary constraints, such as geometry priors, stereo-inspired similarity, or multi-view correspondence losses, to guide reconstruction from limited views. However, NeRF and its variants are notoriously slow to train and render, making them unsuitable for real-time or large-scale scenarios.

Recently, view synthesis and radiance field modelling has emerged with a powerful alternative in 3D Gaussian Splatting (3DGS) [10]. It represents a scene using a set of spatially distributed 3D Gaussians, each with learnable attributes such as position, opacity, and color. This explicit, point-based representation enables high-quality rendering in real time and has rapidly become a new standard for efficient view synthesis.

3DGS exposes a critical limitation when applied to sparseview scenarios. Typically, it depends on Structure-from-Motion (SfM) [11] to estimate camera poses and initialize 3D point distributions. SfM, which relies on robust feature matching across multiple views, becomes fragile when inputs are sparse, often resulting in erroneous geometry and pose estimates. Several recent pipelines claim to operate under sparse-view conditions, but they assume known camera poses as input. In practice, these poses are not obtained from the sparse views themselves, but are instead computed on densely sampled video sequences using Structure-from-Motion (e.g., COLMAP [12], [13]). Although only a few camera views are selected for point cloud generation, the underlying poses originate from the full set of dense views. This reliance on precomputed dense-view poses undermines the true sparsity assumption and raises concerns about the fairness and generality of such evaluations.

Recent methods [14], [15] have proposed SfM-free pipelines to overcome the limitations of SfM in sparse-view settings. They estimate camera poses directly from the input views using multi-view stereo networks. While these approaches eliminate the dependency on SfM, they introduce a new inefficiency: dense, pixel-wise lifting of every image into 3D, regardless of structural relevance. This uniform initialization inflates the number of Gaussians, especially leading to redundant representations in flat or low-texture regions. The result is excessive memory usage and slower rendering, which undermines the efficiency gains expected from SfM-free

pipelines.

We observe that improved initialization plays a critical role in mitigating the drawbacks of dense, unstructured initial point clouds. Our previous work on 2D Gaussian-based regression [16], [17] demonstrated that region-based segmentation can effectively reduce redundancy while preserving important structures. Motivated by this, we extend the segmentation-driven paradigm from 2D to the 3D Gaussian Splatting domain. Our method leverages 2D region-based cues to identify consistent, structurally meaningful areas across views and guide selective downsampling. This pre-filtering step yields compact, geometry-aware initialization, retaining essential structure while reducing unnecessary overhead. As a result, our method improves memory and runtime efficiency without compromising visual fidelity. Our contributions are as follows:

â¢ We introduce a segmentation-driven strategy for initializing 3D Gaussians, reducing computational and memory demands in sparse-view rendering.

â¢ Our method reduces memory usage by up to 50% while preserving image quality and maintaining the fast training performance characteristic of SfM-free pipelinesâachieving competitive PSNR and SSIM with minimal LPIPS degradation.

â¢ We evaluate against SfM-free and SfM-based baselines, demonstrating improved efficiency and competitive quality across diverse sparse-view scenarios.

## II. RELATED WORK

This section reviews three key domains relevant to our work: 3D Gaussian Splatting for sparse-view rendering, segmentation-guided strategies for learning Gaussian parameters, and learning-based camera pose estimation methods. First, we emphasize 3D Gaussian Splatting (3DGS) for its rendering efficiency and explicit point-based structure, which make it particularly effective in sparse-view scenarios where volumetric or implicit approaches face challenges with scalability and transparency. Second, segmentation has become an important tool for structuring Gaussian representations. Unlike prior works that typically use semantic segmentation as a training regularizer, our method leverages region-based segmentation at initialization. This motivates a broader investigation into how segmentation can guide the learning of Gaussian parameters throughout different stages. Finally, we review learning-based camera pose estimation methods because our approach demands alternatives to traditional SfM for accurate pose initialization. We recognize that SfMâs reliance on feature matching falters when input views are sparse or challenging.

## A. Sparse-view 3D Gaussian Splatting

Within the 3DGS framework, pipelines designed for sparseview rendering can be broadly categorized as SfM-based, Hybrid, and SfM-free methods, depending on whether they rely on Structure-from-Motion for camera pose estimation and scene initialization.

1) SfM-based Methods: Several methods in sparse-view rendering rely heavily on SfM, most commonly using COLMAP for estimating camera poses and reconstructing initial point clouds. Few-shot View Synthesis using Gaussian Splatting (FSGS) [18] uses COLMAP-derived points and poses to initialize Gaussians, then improves scene coverage through proximity-guided unpooling, inserting new Gaussians between visually distinct ones. Similarly, SparseGS [19] leverages COLMAP outputs and refines novel views through depth correlation and floater pruning guided by rendered depth. CoR-GS [20] trains two parallel Gaussian radiance fields using SfM-based initialization. It enforces consistency by pruning unmatched Gaussians and using co-rendered pseudo views as mutual supervision during training.

The problem with SfM-based methods is that the initialization becomes unreliable in sparse-view settings due to limited feature correspondences. A common workaround is to estimate poses from densely sampled sequences and then select a sparse subset for training. This practice raises concerns about the fairness and validity of such evaluations. These limitations motivate the development of SfM-free alternatives for both pose estimation and Gaussian initialization.

2) Hybrid Methods: Hybrid approaches retain SfM for camera pose estimation but eliminate its role in point cloud initialization. DNGaussian [21] replaces SfM-derived geometry with randomly initialized Gaussians and introduces depthbased regularization using locally and globally normalized rendered depth maps. However, it continues to rely on COLMAP for pose estimation. RegSegField [22] relies on SfM-derived camera poses but initializes from randomly distributed 3D Gaussians rather than SfM-generated geometry. It incorporates semantic segmentation during training to guide hierarchical refinement, effectively bypassing traditional SfM-based point cloud initialization. While hybrid methods attempt to reduce reliance on SfM, they still depend on it for critical components like camera pose estimation. This residual dependence limits their applicability in scenarios where SfM is unreliable or infeasible. Our work explicitly targets full removal of SfM, which motivates us to compare against methods that either fully adopt SfM or entirely avoid it, rather than hybrids that inherit its limitations.

3) SfM-free Methods: To overcome the fragility and inefficiency of SfM, several recent methods have explored SfMfree alternatives. COLMAP-free GS [14] lifts 2D pixels into 3D using monocular depth predictions and camera intrinsics, forming local Gaussians for each view. These Gaussians are aligned across views via affine transformations optimized with a rendering loss. To improve coverage, the method progressively densifies regions with high reconstruction error. However, this approach struggles with large camera motions, where affine alignment becomes unreliable and reconstruction quality degrades.

InstantSplat [15] further demonstrates the viability of SfMfree pipelines by jointly estimating camera poses and initializing Gaussians from dense pixel-wise point clouds. Although it incorporates confidence-aware filtering to remove redundant points in co-visible regions, the resulting point cloud remains overly dense, particularly in non-overlapping and low-texture areas. This leads to significant memory overhead and limits scalability in truly sparse-view conditions.

<!-- image-->  
Fig. 1: Overview of our segmentation-driven initialization pipeline for sparse-view 3D Gaussian Splatting. Given sparse input views, we estimate camera poses and lift all image pixels into a dense 3D point cloud. We apply region-based segmentation on each image and propagate these segmentations across views to construct segment-aware 3D labels. These labels guide a structured downsampling process that prunes redundant points while preserving geometric structure. The resulting filtered points initialize 3D Gaussians, which are jointly optimized with camera poses to produce the final radiance field.

Rather than relying solely on confidence in co-visible areas, our method applies region-based segmentation to guide the filtering process. We identify structurally meaningful regions across views and selectively retain representative points. This segmentation-driven filtering reduces redundancy at the source, yielding more compact and spatially coherent Gaussian distributions.

## B. Segmentation-Guided Methods

Segmentation has been incorporated into the family of Gaussian-based models for both training-time supervision and initialization.

1) Training-Based Segmentation: Several methods incorporate segmentation during training to regularize Gaussian supervision. SAM3D [23] lifts 2D masks from Segment Anything (SAM) into 3D and merges them via mesh-based alignment and bidirectional refinement. Gaussian Grouping [24] associates SAM masks across views using identity tracking, then constrains the rendered Gaussians to match these labels. RegSegField [22], as discussed earlier, learns view-invariant segment descriptors from 2D SAM masks to guide hierarchical refinement during optimization.

Previous training-based segmentation approaches operate under dense or densifying conditions and depend on high-level semantic labels. In contrast, our method targets sparse-view settings where no densification is applied, using lightweight, region-based segmentation to inform Gaussian initialization rather than supervision.

2) Initialization-Based Segmentation: Certain prior efforts use segmentation for guiding Gaussian initialization. S-SMoE [16] and AS-SMoE [17] leverage region-based segmentation to reduce kernel redundancy in Steered Mixture of Experts (SMoE), an edge-aware Gaussian-based regression model. These approaches focus on preserving high-frequency structures while simplifying homogeneous regions.

Although limited to 2D, these works demonstrate that region-aware initialization offers meaningful trade-offs between compactness and fidelity. Recognizing the conceptual alignment between SMoE and 3D Gaussian Splatting, we extend this segmentation-driven strategy to the 3D domain. Our method leverages cross-view region consistency to guide 3D Gaussian initialization, enabling efficient, structure-aware modeling from the outset without relying on semantic labels or progressive densification.

## C. Learning-Based Camera Pose Estimation Methods

Structure-from-Motion (SfM) extracts camera poses and sparse 3D points through feature matching and incremental optimization. While reliable, SfM depends on robust feature detection and careful calibration, limiting flexibility and speed.

Some works focus on improving SfMâs components rather than bypassing it entirely. For instance, SuperGlue [25] enhances feature matching via learning-based attention, boosting robustnessâbut it remains tethered to SfM pipelines.

<!-- image-->  
(a) Input RGB image

<!-- image-->  
(b) Segmentation map

<!-- image-->  
(c) Retained mask

<!-- image-->  
(d) Downsampled 3D points  
Fig. 2: Visualization of segmentation-guided downsampling. (a) Input RGB image; (b) region-based segmentation map; (c) retained pixel mask after stratified sampling; (d) final downsampled 3D points projected onto the image plane. Redundant points in flat areas (e.g., sky) are removed, while structural details are preserved by retaining more points in high-frequency regions.

Breaking away, PoseNet [26] introduced a bold alternative: directly regressing camera poses from single images using convolutional neural networks, eliminating reliance on feature matching or multi-view geometry. Building on this, MapNet [27] incorporates geometric constraints and temporal context, refining pose estimation accuracy without SfM.

Despite progress, many learning-based pose methods focus solely on camera poses, often overlooking scene geometry or producing sparse outputs. DUSt3R [28] breaks this mold by combining pose estimation and dense 3D reconstruction in a Transformer-based, end-to-end pipeline. MASt3R [29], built on DUSt3Râs backbone, enhances both accuracy and efficiency by introducing local dense feature proposals and a dedicated matching loss. By jointly regressing dense points and camera poses from uncalibrated images, it delivers a fully learningdriven, SfM-free solution, making it the method of choice for our pipeline.

## III. METHOD

An overview of our SDI-GS pipeline is shown in Fig. 1. The process begins with Dense and Unconstrained Stereo 3D Reconstruction [29] for estimating camera poses and generating dense point clouds from the input views. Subsequently, 2D region-based segmentation is performed on each view, and the dense point clouds are projected across views to construct segment-aware 3D labels. These labels guide a structured downsampling process that reduces redundancy while maintaining the sceneâs structural fidelity. The downsampled points serve as the basis for initializing 3D Gaussians, which are then refined jointly with camera poses through an optimization procedure. The following subsections describe each component in detail.

## A. Camera Pose and Dense Point Estimation

Given a sparse set of RGB views $\{ I _ { 1 } , I _ { 2 } , \ldots , I _ { N } \}$ , we employ MASt3R to estimate both the relative camera poses $\{ ( R _ { i } , t _ { i } ) \} _ { i = 1 } ^ { N }$ and dense pixel-wise 3D point clouds. For each pixel $( u , v )$ in view $I _ { i }$ , MASt3R lifts it into a 3D point $\mathbf { x } _ { i } ( u , v ) \in \mathbb { R } ^ { 3 }$ , forming a 3D point set $X _ { i } = \{ \mathbf { x } _ { i } ( u , v ) \} _ { ( u , v ) \in I _ { i } }$ per view. The full unfiltered point set $\mathcal { P }$ across all views is:

$$
{ \mathcal { P } } = \bigcup _ { i = 1 } ^ { N } X _ { i } .\tag{1}
$$

While comprehensive, these lifted point clouds contain redundancy due to overlapping views and homogeneous regions. Hence, downsampling is critical for efficiency and quality.

## B. 2D Segmentation

To identify structurally meaningful regions in each view, we adopt a modified DBSCAN algorithm (MDBSCAN) [16], which clusters pixels based on color similarity in the RGB space. Given the input image $I _ { i } ,$ MDBSCAN generates a segmentation map, where each pixel $( u , v )$ in view $I _ { i }$ is assigned a region label:

$$
\ell _ { i } ^ { 2 D } ( u , v ) \in \mathbb { N } .\tag{2}
$$

This region-based segmentation approach offers several advantages over semantic segmentation methods. First, it preserves fine-grained structural details by grouping pixels based solely on local color similarity, without imposing semantic constraints that may oversimplify textures or repetitive patterns. This is particularly important in our setting, where we do not employ subsequent densification or pseudo-view regularization. Therefore, we require high-frequency details to be preserved from the initialization stage.

Moreover, unlike many region-growing methods [30]â[34], MDBSCAN imposes no explicit constraint on segment size. This flexibility is crucial: flat regions can be efficiently represented by large Gaussian components, and unnecessarily subdividing them would introduce redundant initializations. Conversely, high-frequency regions are naturally segmented into finer clusters, ensuring that structurally rich areas are wellcaptured without manual tuning.

In addition to its representational benefits, MDBSCAN is computationally efficient. It runs in approximately 4 ms per view, making it well-suited for our pipeline, where runtime overhead must be minimized. In contrast, deep learningâbased segmentation methods such as Segment Anything [35] typically require around 150 ms per view, which is impractical for our intended efficiency. Overall, MDBSCAN strikes a desirable balance between structural fidelity, computational efficiency, and compatibility with sparse-view training objectives.

## C. Segment-Aware 3D Labeling

As defined in Eq. (1), each $X _ { i }$ is a subset of $\mathcal { P }$ containing 3D points lifted from pixels in view $I _ { i } .$ . Each 3D point $\mathbf { x } \in X _ { i }$ is projected onto two adjacent training views, $I _ { i - 1 }$ and $I _ { i + 1 }$ , using projection functions $p _ { i - 1 } , p _ { i + 1 } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ , respectively. These functions map the 3D point x to its corresponding pixel coordinates in the adjacent views, where segmentation labels $\ell _ { i - 1 } ^ { 2 D }$ and $\ell _ { i + 1 } ^ { 2 D }$ are defined.

For each $\textbf { x } \in \ X _ { i }$ , the segment-aware 3D label vector is constructed by aggregating segmentation labels from the source and adjacent views as:

$$
\ell ^ { 3 D } ( { \bf x } ) = \left[ \ell _ { i } ^ { 2 D } , \ell _ { i - 1 } ^ { 2 D } ( p _ { i - 1 } ( { \bf x } ) ) , \ell _ { i + 1 } ^ { 2 D } ( p _ { i + 1 } ( { \bf x } ) ) \right] .\tag{3}
$$

This label vector encodes how a segment from view i is preserved or fragmented across viewpoints. To ensure that downsampling preserves a meaningful structure without excessive fragmentation, we construct each 3D label vector using only the segment labels from a source view and its two adjacent views. Incorporating more views would increase label dimensionality and risk of oversegmentation, as small inconsistencies across projections can unnecessarily split coherent regions. In practice, three views provide sufficient discriminative power to separate structurally distinct areas while maintaining computational efficiency (see Section 4.2 for ablation results on label dimensionality).

We define a structural cluster at the set of all points in the scene that share an identical 3D label vector:

$$
C _ { j } = \{ \mathbf { x } \in \mathcal { P } \mid \ell ^ { 3 D } ( \mathbf { x } ) = \ell _ { j } ^ { 3 D } \} ,\tag{4}
$$

where $\ell _ { j } ^ { 3 D }$ denotes a unique label vector across the entire set $\mathcal { P }$

## D. Downsampling

To reduce redundancy while preserving structural diversity, we perform stratified sampling within each cluster $C _ { j }$ . We retain up to $N _ { \mathrm { m a x } }$ points per cluster, using sampling with replacement. If a cluster contains fewer points than $N _ { \mathrm { m a x } }$ , we sample only as many as available:

$$
\hat { C } _ { j } = \operatorname { R a n d o m S a m p l e } ( C _ { j } , \operatorname* { m i n } ( N _ { \operatorname* { m a x } } , | C _ { j } | ) ) .\tag{5}
$$

where $| C _ { j } |$ denotes the cardinality. The final downsampled point cloud is the union of sampled points across all clusters:

$$
\hat { \mathcal { P } } = \bigcup _ { j } \hat { C } _ { j } .\tag{6}
$$

This sampling strategy assigns a similar number of points to each structurally distinct region, avoiding overrepresentation of large, uninformative areas. The core idea is to enforce cross-view consistency: if a segment remains coherent when projected across views, it likely corresponds to a structurally reliable region. In contrast, projections that scatter into dissimilar or fragmented segments indicate inconsistency and are treated separately. This strategy enables us to group 3D points based on consistent low-level appearance patterns across views before downsampling, preserving geometric structure while reducing redundant points.

As illustrated in Fig. 2, our segmentation-driven strategy effectively prunes redundant points while preserving structural fidelity. Fig. 2(a) shows the original input view, and (b)

visualizes the region-based segmentation result. Based on these segments, Fig. 2(c) indicates the retained mask after sampling representative points within each region. The final retained points, shown in (d), form a significantly reduced yet structurally consistent point cloud.

Regions with low visual complexityâsuch as sky or flat surfacesâare typically grouped into large segments, from which only a few points are retained. In contrast, high-frequency regions generate more segments and retain more samples, ensuring finer detail. This targeted reduction eliminates redundancy at the source and provides a strong initialization prior for Gaussian optimization.

## E. Gaussian Initialization and Joint Optimization

Each retained 3D point $\mathbf { x } \in \hat { \mathcal { P } }$ is used to initialize a 3D Gaussian $G = ( \mu , \Sigma , c , \alpha )$ , where $\mu$ represents the position, Î£ the covariance, c the color, and Î± the opacity. The full set of Gaussians is denoted as $\hat { \mathcal { G } } = \{ G | \mathbf { x } \in \mathcal { \hat { P } } \}$

The initialization and optimization processes adhere to the differentiable rendering framework of 3D Gaussian Splatting (3DGS) [10]. Additionally, we employ the joint optimization strategy introduced in InstantSplat [15], wherein both Gaussian parameters and camera poses are refined simultaneously. The optimization is driven by a photometric loss $L _ { \mathrm { p h o t o } }$ , which is formulated as:

$$
L _ { \mathrm { p h o t o } } = \sum _ { i = 1 } ^ { N } \left| I _ { i } - R ( \hat { \mathcal { G } } , T _ { i } ) \right| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $I _ { i }$ is the ground truth image for view i, $T _ { i }$ is the corresponding camera pose, and $R ( \hat { \mathcal { G } } , T _ { i } )$ denotes the rendered image produced by splatting the Gaussians $\hat { \mathcal G }$ under pose $T _ { i }$ This joint optimization strategy ensures high-fidelity view synthesis, facilitating accurate reconstruction even with sparsely sampled input views.

## IV. EXPERIMENTAL SETUP

We evaluated our method (SDI-GS) under SfM-free and SfM-based protocols, using diverse datasets and consistent benchmarks. This enabled fair comparison against baselines under varying sparse-view conditions.

## A. Datasets

SfM-free: We followed the InstantSplat [15] protocol, evaluating on eight Tanks and Temples [36] scenes, seven diverse MVImgNet [37] scenes, and all nine Mip-NeRF 360 scenes. For Tanks and Temples and MVImgNet, 24 images were uniformly sampled per scene. For Mip-NeRF 360, 24 images were drawn from the first 48 frames, capturing objects from varying elevations along a 360â¦ trajectory. Training was performed on sparse subsets of $N = 3 , 6 , 1 2$ views, with the remaining 12 views (excluding first and last frames) reserved for testing, consistent with InstantSplat [15].

<!-- image-->  
Fig. 3: Qualitative comparison across three datasets under 3-view (top subrow) and 12-view (bottom subrow) settings. Each row corresponds to a different dataset. Within each dataset, we show rendering results for CF-3DGS, InstantSplat, and our method. CF-3DGS exhibits severe artifacts due to unreliable pose estimation. In contrast, both InstantSplat and our method use MASt3R for initialization and refine poses during training, leading to stable and accurate reconstructions. Our segmentationdriven downsampling further reduces memory usage without compromising visual quality.

SfM-based: The SfM-based evaluation shared the Mip-NeRF 360 dataset but adopts a different training and testing split in line with RegNeRF [8] and RegSegField [22] protocols. We selected five Mip-NeRF 360 scenes with clear object boundaries, training on 12 views and testing every 8th image, following community standards. Additionally, we used ten DTU scans with multiple foreground objects, training on 3 views focused on foreground reconstruction.

This distinction in Mip-NeRF 360 splits ensured internal consistency within each protocol and prevents overinterpretation when comparing SfM-free and SfM-based methods. To further explore cross-protocol robustness, we also evaluated InstantSplat under the SfM-based split, highlighting how camera pose quality and view selection influence generalizability.

## B. Baselines

SfM-free: We compared our method against InstantSplat [38] and CF-3DGS [14], using identical datasets and view selections to ensure a fair comparison.

TABLE I: SfM-Free Comparison on Tanks and Temples across 3, 6, and 12 views
<table><tr><td rowspan="2">Method</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td><td colspan="3">Size (MB)â</td><td colspan="3">Training Timeâ</td><td colspan="3">Rendering Speed (FPS)â</td></tr><tr><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td></tr><tr><td>CF-3DGS [14]</td><td>0.406</td><td>0.469</td><td>0.507</td><td>0.452</td><td>0.421</td><td>0.418</td><td>321.1</td><td>590.4</td><td>800.2</td><td>1m19s</td><td>2m55s</td><td>5m6s</td><td>156</td><td>96</td><td>83</td></tr><tr><td>InstantSplat [15]</td><td>0.768</td><td>0.846</td><td>0.866</td><td>0.175</td><td>0.137</td><td>0.139</td><td>98.6</td><td>168.1</td><td>278.4</td><td>7.12s</td><td>10.12s</td><td>12.88s</td><td>152</td><td>131</td><td>105</td></tr><tr><td>SDI-GS (Ours)</td><td>0.754</td><td>0.831</td><td>0.853</td><td>0.245</td><td>0.192</td><td>0.178</td><td>21.8</td><td>46.3</td><td>85.2</td><td>6.50s</td><td>8.62s</td><td>10.00s</td><td>190</td><td>173</td><td>148</td></tr></table>

TABLE II: SfM-free Comparison on Mip-NeRF 360 across 3, 6, and 12 views
<table><tr><td rowspan="2">Method</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td><td colspan="3">Size (MB)â</td><td colspan="3">Training Timeâ</td><td colspan="3">Rendering Speed (FPS)â</td></tr><tr><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td></tr><tr><td>CF-3DGS [14]</td><td>0.217</td><td>0.256</td><td>0.244</td><td>0.594</td><td>0.599</td><td>0.609</td><td>337.27</td><td>715.09</td><td>968.58</td><td>1m46s</td><td>3m57s</td><td>7m7s</td><td>131</td><td>87</td><td>69</td></tr><tr><td>InstantSplat [15]</td><td>0.317</td><td>0.417</td><td>0.467</td><td>0.534</td><td>0.466</td><td>0.441</td><td>118.88</td><td>227.74</td><td>429.53</td><td>16.57s</td><td>19.86s</td><td>25.43s</td><td>115</td><td>95</td><td>70</td></tr><tr><td>SDI-GS (Ours)</td><td>0.336</td><td>0.431</td><td>0.476</td><td>0.569</td><td>0.510</td><td>0.473</td><td>20.48</td><td>37.29</td><td>71.99</td><td>16.14s</td><td>18.29s</td><td>20.86s</td><td>138</td><td>125</td><td>101</td></tr></table>

TABLE III: SfM-free Comparison on MVImgNet across 3, 6, and 12 views
<table><tr><td rowspan="2">Method</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td><td colspan="3">Size (MB)â</td><td colspan="3">Training Timeâ</td><td colspan="3">Rendering Speed (FPS)â</td></tr><tr><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td><td>3view</td><td>6view</td><td>12view</td></tr><tr><td>CF-3DGS [14]</td><td>0.341</td><td>0.383</td><td>0.431</td><td>0.552</td><td>0.539</td><td>0.559</td><td>413.61</td><td>766.60</td><td>708</td><td>3m16s</td><td>7m24s</td><td>13m5s</td><td>90</td><td>63</td><td>70</td></tr><tr><td>InstantSplat [15]</td><td>0.554</td><td>0.694</td><td>0.720</td><td>0.386</td><td>0.283</td><td>0.280</td><td>102.71</td><td>195.88</td><td>347</td><td>8.00s</td><td>13.33s</td><td>19.56s</td><td>130</td><td>112</td><td>76</td></tr><tr><td>SDI-GS (Ours)</td><td>0.550</td><td>0.671</td><td>0.700</td><td>0.438</td><td>0.347</td><td>0.332</td><td>25.53</td><td>55.91</td><td>104</td><td>7.00s</td><td>10.78s</td><td>14.11s</td><td>188</td><td>176</td><td>145</td></tr></table>

TABLE IV: SfM-based Comparison on Mip-NeRF 360 and DTU
<table><tr><td rowspan="2">Method</td><td colspan="6">DTU (3 views)</td><td colspan="6">Mip-NeRF 360 (12 views)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ Size (MB)</td><td></td><td>Train Time</td><td>Render FPS</td><td>PSNRâ</td><td>SSIMâ</td><td></td><td>LPIPSâ Size (MB) Train Time</td><td></td><td>Render FPS</td></tr><tr><td>FSGS [18]</td><td>20.39</td><td>0.827</td><td>0.206</td><td>21</td><td>11m37s</td><td>141</td><td>17.49</td><td>0.582</td><td>0.459</td><td>142</td><td>13m35s</td><td>28</td></tr><tr><td>SparseGS [19]</td><td>19.89</td><td>0.778</td><td>0.205</td><td>16</td><td>11m41s</td><td>125</td><td>16.72</td><td>0.550</td><td>0.463</td><td>134</td><td>44m30s</td><td>36</td></tr><tr><td>CoR-GS [20]</td><td>21.98</td><td>0.827</td><td>0.222</td><td>34</td><td>7m23s</td><td>138</td><td>16.95</td><td>0.559</td><td>0.515</td><td>40</td><td>41m36s</td><td>37</td></tr><tr><td>InstantSplat [15]</td><td>20.92</td><td>0.781</td><td>0.210</td><td>142</td><td>34s</td><td>88</td><td>17.26</td><td>0.606</td><td>0.491</td><td>465</td><td>2m23s</td><td>23</td></tr><tr><td>SDI-GS (Ours)</td><td>21.35</td><td>0.793</td><td>0.224</td><td>24</td><td>25s</td><td>134</td><td>18.50</td><td>0.648</td><td>0.501</td><td>59</td><td>1m16s</td><td>43</td></tr></table>

<!-- image-->  
Fig. 4: Compression-performance trend across increasing view counts (3, 6, 12), averaged over Tanks and Temples, MVImgNet, and Mip-NeRF 360. Each line shows PSNR versus file size for a method. As the number of input views increases, our method achieves greater compression gainsâreducing file size by up to 75% at 12 viewsâwhile maintaining comparable reconstruction quality.

SfM-based: SfM-based comparisons included FSGS [18], SparseGS [19], and CoR-GS [20]. We also included InstantSplat under this protocol to enable direct comparison across SfM paradigms.

## C. Implementation Details

SfM-free: We set training iterations for our method and InstantSplat to 300, aligning with InstantSplatâs protocol, emphasizing rapid convergence from dense initializations. CF-3DGS followed its default two-stage schedule: 300 iterations for local Gaussian optimization, then 300 for global refinement. For evaluation, we used the same resolution as in training to maintain consistency. All SfM-free experiments ran on A100 GPUs.

SfM-based: Training iterations for our method were increased to 1000 on Mip-NeRF 360, reflecting the complexity of outdoor 360Â° scenes and the slower convergence typical of SfM-based methods. On the simpler DTU dataset, we kept 300 iterations. Baselines FSGS, SparseGS, and CoR-GS used their default settings: 30,000 iterations on Mip-NeRF 360, and on DTU, 30,000 for FSGS and SparseGS, and 10,000 for CoR-GS.

Evaluation generally used each methodâs training resolution, except for Mip-NeRF 360 SfM-based runs, where we tested on the original high-resolution images rather than downsampled inputs. This provided a more challenging benchmark for highfidelity view synthesis, albeit with slower rendering speeds compared to DTU and SfM-free settings. All SfM-based experiments used A40 GPUs.

## V. RESULTS

We report results under the SfM-free and SfM-based settings described in Section 4. The following subsections present detailed comparisons with state-of-the-art methods in each category.

<!-- image-->  
Fig. 5: Qualitative comparison with SfM-based methods (FSGS, SparseGS, CoR-GS) and SfM-free InstantSplat. SDI-GS matches the visual quality of SfM-based pipelines while keeping the lightweight memory footprint of SfM-free methods (Table IV) and drastically reducing training time. Unlike InstantSplat, which demands more storage, our segmentation-driven initialization provides a compact, efficient representation without compromising structural fidelity.

## A. SfM-free Comparison

Tables I, II, and III present quantitative results on Tanks and Temples, Mip-NeRF 360, and MVImgNet under varying sparse-view settings (3, 6, and 12 views), respectively. Across all datasets, our method achieves comparable SSIM and only marginally worse LPIPS scores compared to InstantSplat, while significantly reducing the memory footprint. On Tanks and Temples, our file size is reduced to one-third of InstantSplatâs, and we match its rendering quality with faster training and 25% lower rendering latency. On Mip-NeRF 360, our method attains slightly better SSIM while LPIPS remains competitive; notably, we reduce file size by over 80%. On MVImgNet, our rendering quality is slightly lower, but we maintain a 3Ã smaller model and up to 40% faster rendering speed.

Visual comparisons are shown in Fig. 3, which display 3-view and 12-view rendering results. In the 3-view case, CF-3DGS exhibits severe artifacts due to inaccurate camera pose estimation, often leading to completely misaligned views. Notably, its progressive densification strategy does not guarantee improved quality with more views; additional training views may propagate incorrect geometry due to faulty pose initialization, worsening results in some cases. In contrast, InstantSplat and our method both use MASt3R for pose initialization and jointly optimize poses during training, resulting in more accurate and stable reconstructions.

Our results preserve global structure well compared to InstantSplat, though we may miss some high-frequency details. Importantly, our segmentation-guided downsampling smooths the transitions in overlapping regionsâareas where InstantSplat tends to exhibit abrupt visual artifacts. Moreover, our method achieves these results with significantly lower memory usage: requiring only one-third the memory of InstantSplat in 3-view setups and up to 80% less in the 12-view configuration (Table III).

Fig. 4 provides a deeper analysis of how compression efficiency scales with the number of input views. As the number of training views increases from 3 to 12, so does the redundancy in the lifted point clouds. Our segmentationguided downsampling capitalizes on this redundancy, yielding more aggressive compression. Specifically, we observe a 50% reduction in file size for 3-view inputs, and up to 75% for 12 views compared to InstantSplat. Importantly, this is achieved without compromising reconstruction quality, as shown by the maintained PSNR. This trend underscores the scalability of our method: the more complex the input, the greater the efficiency gain, all while preserving structural fidelity in the final representation.

These results demonstrate the practical effectiveness of our method in SfM-free scenarios, especially for compact and realtime applications in sparse-view settings. These results demonstrate the practical efficiency and effectiveness of our method for SfM-free sparse-view synthesis. Our method enables highquality rendering with significantly lower memory and time costs, which is ideal for real-time and resource-constrained scenarios.

## B. SfM-based Comparison

Table IV summarizes the comparison of our method with SfM-based baselines. These baselines begin with sparse point clouds generated by SfM and therefore rely on explicit densification to reach sufficient coverage. This procedure is not required in SfM-free methods, which begin with dense pixel-wise reconstructions. As a result, SfM-based pipelines typically require significantly more trainingâoften 10Ã longerâto progressively grow the Gaussian representation. In contrast, our method achieves fast convergence by directly training on filtered dense point clouds.

Our method achieves comparable quantitative performance to SfM-based baselines on the DTU dataset, and outperforms them in both PSNR and SSIM on Mip-NeRF 360, albeit with a slightly higher LPIPS. In terms of file size, we match or exceed the compactness of SfM-based methods. On DTU, these methods benefit from the sparsity of the foreground-centric scenes, resulting in lower file sizes. However, InstantSplat incurs a large memory overhead due to projecting all pixelsâincluding backgroundâinto 3D, even when such regions are low in structural relevance. Our segmentation-driven downsampling addresses this redundancy, resulting in a much more compact representation while preserving fidelity.

We now compare efficiency across methods in terms of memory footprints and rendering speed. On Mip-NeRF 360, SfM-based methods require longer training to achieve comparable reconstruction quality, which leads to larger Gaussian counts and slower rendering. Only CoR-GS applies a Gaussian pruning strategy to reduce the final memory size. However, its dual-Gaussian design doubles memory consumption during training and maintains a high computational cost. In contrast, our method benefits from initializing with downsampled dense predictions, reducing file size to nearly half that of FSGS and SparseGS, and to one-eighth that of InstantSplat. Our rendering speed is faster than all baselines on Mip-NeRF 360, and competitive with FSGS on DTU.

Fig. 5 presents qualitative comparisons on representative scenes. While maintaining visual quality comparable to SfMbased methods, our approach offers drastically faster training by orders of magnitude and a more compact model.

These results underscore that our method bridges the gap between SfM-free and SfM-based pipelines. By leveraging dense initialization while enforcing compactness through structured downsampling, we combine the efficiency of SfM-free training with the lightweight output of SfM-based pipelines. Our method achieves low memory and time cost without sacrificing rendering quality.

## C. Ablation study

Our method uses a 3D label vector constructed from regionbased segmentations across three input views. This design balances computational efficiency with segmentation consistency. To validate this design choice, we conduct an ablation study comparing against a variant that uses various available labels from the 12-view setup (i.e., a label dimension of 6 and 12). Since higher label dimensions yield more segments and therefore more Gaussians, we increase training to 1000 iterations to ensure convergence across all settings.

TABLE V: Ablation study on label dimension under the 12- view setting.
<table><tr><td>Dimension</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Size (MB)</td><td>Train Time</td><td>Render FPS</td></tr><tr><td>3</td><td>17.83</td><td>0.445</td><td>0.452</td><td>72</td><td>44s</td><td>161</td></tr><tr><td>6</td><td>17.42</td><td>0.434</td><td>0.442</td><td>154</td><td>55s</td><td>116</td></tr><tr><td>12</td><td>17.28</td><td>0.419</td><td>0.438</td><td>247</td><td>67s</td><td>89</td></tr></table>

Table V shows that increasing the label dimension from 3 to 12 leads to a significant growth in file sizeâwith only marginal improvements in PSNR and SSIM. This is because higher label dimensionality introduces many fragmented or noisy segment combinations, breaking coherent structures into smaller, less meaningful clusters. These fine-grained segments inflate the number of retained Gaussians, undermining the intended sparsity without substantial benefit to reconstruction quality.

## VI. CONCLUSION

We have presented a segmentation-driven initialization strategy for 3D Gaussian Splatting in sparse-view settings, addressing the inefficiencies present in existing SfM-free methods. Rather than lifting every pixel into 3D space indiscriminately, our method applies region-based segmentation to guide structured downsampling, significantly reducing redundancy while preserving scene fidelity.

Through extensive experiments on diverse datasets, we have demonstrated that our SDI-GS achieves comparable or superior rendering quality to existing SfM-free and SfM-based methods, with up to 50% fewer Gaussians and substantially lower memory and runtime cost. Notably, SDI-GS preserves the fast training times that make SfM-free pipelines attractive, while producing representations as compact as those of SfMbased methods.

These results highlight the importance of structure-aware initialization in 3DGS. By reducing unnecessary overhead at the source, we improve scalability and practicality under real-world sparse-view constraints. Future directions include exploring adaptive or learned segmentation techniques and extending our framework to unconstrained, dynamic, or lowlight environments.

## REFERENCES

[1] S. Zhu, G. Wang, X. Kong, D. Kong, and H. Wang, â3d gaussian splatting in robotics: A survey,â 2024. [Online]. Available: https://arxiv.org/abs/2410.12262

[2] Y. Jiang, C. Yu, T. Xie, X. Li, Y. Feng, H. Wang, M. Li, H. Lau, F. Gao, Y. Yang, and C. Jiang, âVr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality,â in ACM SIGGRAPH 2024 Conference Papers, ser. SIGGRAPH â24. New York, NY, USA: Association for Computing Machinery, 2024. [Online]. Available: https://doi.org/10.1145/3641519.3657448

[3] L. Zhu, Z. Wang, J. Cui, Z. Jin, G. Lin, and L. Yu, âEndogs: Deformable endoscopic tissues reconstruction with gaussian splatting,â in Medical Image Computing and Computer Assisted Intervention â MICCAI 2024 Workshops, M. E. Celebi, M. Reyes, Z. Chen, and X. Li, Eds. Cham: Springer Nature Switzerland, 2025, pp. 135â145.

[4] Z. Zhou, H. Zhao, J. Fang, D. Xiang, L. Chen, L. Wu, F. Wu, W. Liu, C. Zheng, and X. Wang, âTiavox: Time-aware attenuation voxels for sparse-view 4d dsa reconstruction,â 2023. [Online]. Available: https://arxiv.org/abs/2309.02318

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, p. 99â106, Dec. 2021. [Online]. Available: https://doi.org/10.1145/3503250

[6] J. Chibane, A. Bansal, V. Lazova, and G. Pons-Moll, âStereo radiance fields (srf): Learning view synthesis for sparse views of novel scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 7911â7920.

[7] P. Truong, M.-J. Rakotosaona, F. Manhardt, and F. Tombari, âSparf: Neural radiance fields from sparse and noisy poses,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 4190â4200.

[8] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger, and N. Radwan, âRegnerf: Regularizing neural radiance fields for view synthesis from sparse inputs,â in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

[9] A. Meuleman, Y.-L. Liu, C. Gao, J.-B. Huang, C. Kim, M. H. Kim, and J. Kopf, âProgressively optimized local radiance fields for robust view synthesis,â in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2023, pp. 16 539â16 548. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.01587

[10] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3D Gaussian Splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, pp. 139:1â139:14, Jul. 2023.

[11] M. Westoby, J. Brasington, N. Glasser, M. Hambrey, and J. Reynolds, ââstructure-from-motionâ photogrammetry: A lowcost, effective tool for geoscience applications,â Geomorphology, vol. 179, pp. 300â314, 2012. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0169555X12004217

[12] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,âÂ¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[13] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, âPixel-Â¨ wise view selection for unstructured multi-view stereo,â in European Conference on Computer Vision (ECCV), 2016.

[14] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmapfree 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 20 796â20 805.

[15] Z. Fan, K. Wen, W. Cong, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, âInstantsplat: Sparse-view gaussian splatting in seconds,â 2024.

[16] Y.-H. Li, M. Sjostr Â¨ om, S. Knorr, and T. Sikora, âSegmentation-based Â¨ initialization for steered mixture of experts,â in 2023 IEEE International Conference on Visual Communications and Image Processing (VCIP), Jeju, Korea (South), 2023, pp. 1â5.

[17] Y.-H. Li, S. Knorr, M. Sjostr Â¨ om, and T. Sikora, âAdaptive segmentation- Â¨ based initialization for steered-mixture-of-experts image regression,â IEEE Transaction on Multimedia, 2025.

[18] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, âFsgs: Real-time few-shot view synthesis using gaussian splatting,â 2023.

[19] H. Xiong, S. Muttukuru, R. Upadhyay, P. Chari, and A. Kadambi, âSparsegs: Real-time 360Â° sparse view synthesis using gaussian splatting,â Arxiv, 2023.

[20] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, and X. Bai, âCor-gs: Sparse-view 3d gaussian splatting via co-regularization,â in Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part I. Berlin, Heidelberg: Springer-Verlag, 2024, p. 335â352.

[21] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, â DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization ,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2024, pp. 20 775â 20 785.

[22] K. Gu, T. Maugey, K. Sebastian, and C. Guillemot, âRegsegfield: Mask-regularization and hierarchical segmentation for novel view synthesis from sparse inputs,â in Proceedings of 21st ACM SIGGRAPH Conference on Visual Media Production, ser. CVMP â24. New York,

NY, USA: Association for Computing Machinery, 2024. [Online]. Available: https://doi.org/10.1145/3697294.3697299

[23] D. Zhang, D. Liang, H. Yang, Z. Zou, X. Ye, Z. Liu, and X. Bai, âSam3d: Zero-shot 3d object detection via segment anything model,â Science China Information Sciences, 2023.

[24] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in ECCV, 2024.

[25] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperGlue: Learning feature matching with graph neural networks,â in CVPR, 2020. [Online]. Available: https://arxiv.org/abs/1911.11763

[26] A. Kendall, M. Grimes, and R. Cipolla, â PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization ,â in 2015 IEEE International Conference on Computer Vision (ICCV). Los Alamitos, CA, USA: IEEE Computer Society, Dec. 2015, pp. 2938â2946. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ICCV.2015.336

[27] S. Brahmbhatt, J. Gu, K. Kim, J. Hays, and J. Kautz, âGeometryaware learning of maps for camera localization,â in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[28] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in CVPR, 2024.

[29] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part LXXII. Berlin, Heidelberg: Springer-Verlag, 2024, p. 71â91.

[30] J. Shen, X. Hao, Z. Liang, Y. Liu, W. Wang, and L. Shao, âRealtime superpixel segmentation by dbscan clustering algorithm,â IEEE Transactions on Image Processing, vol. 25, no. 12, pp. 5933â5942, 2016.

[31] Y. Zhang, Y. Zhang, L. Fan, and N. Wang, âFast and accurate superpixel segmentation algorithm with a guidance image,â Image and Vision Computing, vol. 129, p. 104596, Jan. 2023.

[32] F. Yang, Q. Sun, H. Jin, and Z. Zhou, âSuperpixel Segmentation With Fully Convolutional Networks,â in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Seattle, WA, USA: IEEE, Jun. 2020, pp. 13 961â13 970. [Online]. Available: https://ieeexplore.ieee.org/document/9156320/

[33] P. Neubert and P. Protzel, âCompact Watershed and Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation Algorithms,â in 2014 22nd International Conference on Pattern Recognition, Aug. 2014, pp. 996â1001, iSSN: 1051-4651.

[34] Z. Li and J. Chen, âSuperpixel segmentation using Linear Spectral Clustering,â in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2015, pp. 1356â1363.

[35] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick, âSegment anything,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 4015â4026.

[36] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: benchmarking large-scale scene reconstruction,â ACM Trans. Graph., vol. 36, no. 4, Jul. 2017. [Online]. Available: https://doi.org/10.1145/3072959.3073599

[37] X. Yu, M. Xu, Y. Zhang, H. Liu, C. Ye, Y. Wu, Z. Yan, T. Liang, G. Chen, S. Cui, and X. Han, âMvimgnet: A large-scale dataset of multiview images,â in CVPR, 2023.

[38] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural Â¨ graphics primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, no. 4, Jul. 2022. [Online]. Available: https://doi.org/10.1145/3528223.3530127

Yi-Hsin Li She started a double degree PhD program at Technical University Berlin, Germany and seconded to MIUN in November 2021. She in her fouryear PhD journey works on high-dimensional data compression, focusing on gating networks.

Sebastian Knorr Sebastian Knorr, professor of Visual Computing at HTW Berlin. Broad competence in 3D/ 360Â° imaging technologies. Expertise in 1) capture and synthesis of stereo 3D and 360Â° video; 2) visual attention and quality assessment in VR; 3) light field imaging and neural radiance fields.

Marten Sj Ë ostr Â¨ omÂ¨ Marten Sj Ë ostr Â¨ om, professor of signal processing, Mid Â¨ Sweden University. Multidimensional signal processing, imaging and compression; system modelling and identification; inverse problems using machine learning. 120+ scientific articles, two book chapters. Supervisor of 8 PhD students, 8 previously.

Thomas Sikora Thomas Sikora, professor and director of the Communication Systems Lab, Technische Universitat Berlin, Germany. Â¨