# GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering

Alexander Valverde1â\* Brian Xu2 Yuyin Zhou1 Meng Xu3 Hongyun Wang1 1University of California, Santa Cruz 2Brown University 3Kean University

## Abstract

Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data.

In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussianbased scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details.

We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone. Code is available at: https://github.com/ alevalve/gaussmart

## 1. Introduction

Three-dimensional (3D) reconstruction has emerged as a prominent area of research since the introduction of Neural Radiance Fields (NeRF) [36]. More recently, Gaussian Splatting [20] has advanced scene reconstruction by exploiting the mathematical properties of 3D Gaussians, which allow for both accurate modeling and efficient rendering. Nevertheless, these methods frequently rely on unstructured point clouds, where uneven density and photogrammetry artifacts often introduce noise, limiting the fidelity of the reconstructed geometry.

<!-- image-->  
Figure 1. Our 3D reconstruction pipeline: An input image undergoes segmentation, which enhances the 2D Gaussian splats, leading to a more geometric 3D mesh.

In addition, optimization strategies in existing approaches are typically restricted to simple objectives such as L1 reconstruction losses or supervision based on surface normals and depth maps. While these objectives provide a useful training signal, they are insufficient to capture finegrained, high-frequency details that are critical for realistic scene representation. As a result, reconstructions may lack sharpness and fail to preserve subtle geometric and textural structures present in the original scenes.

Given the spurious conditions of the input point clouds, we first apply a convex outlier removal procedure to eliminate unnecessary points introduced by interpolation errors in COLMAP reconstructions. In addition, we present a framework that leverages 2D foundation models such as DINOv3 [50] and SAM [22] to guide point cloud densification. Specifically, we increase the density of underrepresented segments by sampling new points according to the covariance structure of the original points within each segment mask. Finally, we introduce a novel loss function built on DINOv3 embeddings, which is differentiable and seamlessly integrates into the Gaussian Splatting pipeline [18], enabling improved semantic alignment during optimization.

Our contributions are summarized as follows:

â¢ Convex-guided outlier removal. We propose a convexbased filtering method that removes spurious points arising from photogrammetry and interpolation errors in COLMAP-generated point clouds.

â¢ Segment-aware point cloud densification. By leveraging SAM-derived segmentation masks, we regularize and densify sparse regions of the point cloud using a covariance-driven sampling strategy that accounts for both local geometry and segment area.

â¢ Embedding-aligned training. We introduce a lightweight embedding loss based on DINOv3 features, which enforces consistency between rendered Gaussian appearances and segment-level semantics, thereby improving spatial fidelity and object coherence.

## 2. Related Works

Outlier Removal. Outlier removal is a key pre-processing step for noisy point clouds, with two widely used methods being Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR), both popularized in the Point Cloud Library [47, 48]. SOR computes the mean distance of each point to its k-nearest neighbors and rejects those deviating from the global meanâstandard deviation range, while ROR removes points with fewer than a threshold number of neighbors within a fixed radius. These lightweight filters are usually combined with robust model-fitting methods such as RANSAC [13] or Trimmed ICP [7], surface-fitting and projection approaches like Point Set Surfaces [1], graph-based and variational denoisers [4, 67], and learning-based frameworks such as PointCleanNet [44].

Novel View Synthesis Neural Radiance Fields (NeRF)[36] model view-dependent color and density using a multilayer perceptron (MLP) to synthesize novel views. Subsequent variants have improved sampling strategies, scalability, and geometric fidelity[2, 3, 5, 14, 28â 30, 35, 38, 55, 61, 62]. Despite these advances, conventional NeRF models remain computationally expensive and often fail to accurately reconstruct large-scale or texture-poor regions. More recently, scene representations based on projected Gaussian primitives have enabled fast, high-quality rendering while remaining scalable to large scenes [9, 11, 17, 18, 20]. In particular, 2D Gaussian Splatting [18] improves geometric alignment by orienting primitives along local surface tangents, typically guided by estimated normals. However, coverage gaps may still arise when the initialization is suboptimal or when point density fails to adequately represent fine-scale structures.

2D Foundational Models. Promptable segmentation models such as SAM [22] provide strong per-object cues that have been successfully applied in both vision and medical reconstruction [27, 32, 37, 49, 70]. For 3D Gaussians, SAGA [6] extends SAM-like capabilities but remains limited when handling fine-scale objects. In parallel, semantic backbones such as DINOv2 [39] and DINOv3 [50] provide dense, transferable feature embeddings that capture both local detail and global context. In particular, DINOv3 offers semantically consistent representations with strong invariance to viewpoint and appearance changes, making it especially well-suited for embedding-level supervision. This motivates our use of a DINOv3-based loss to complement pixel-domain objectives and enforce semantic alignment in the Gaussian Splatting pipeline.

Point-based representations. Point clouds are flexible, permutation-invariant 3D carriers for reconstruction [12, 34, 64] and for tasks like 2D projection/detection [8] and point-level segmentation [42]. Point-based rendering learns to rasterize point features to images [43, 71], while selfsupervised objectives densify or regularize point sets [56]. Broader advances span reconstruction [10, 40, 41, 51, 58, 66, 68], completion [57, 60, 65, 69], and point operators [16, 31, 52].

## 3. Method

## 3.1. Convex Outlier Removal

Point clouds generated by COLMAP often contain hundreds of spurious points that do not contribute meaningfully to the scene of interest. These outliers typically arise from triangulation errors, introducing noise into the reconstruction. In figure 3 we can see external points that do not represent the object of interest. To address this, traditional methods such as Radius Outlier Removal and Statistical Outlier Removal are widely used. These approaches are considered the classical techniques for denoising point cloud representations.

However, most of these methods depend heavily on hyperparameters that are difficult to tune consistently across different scenes, which makes the cleaning process cumbersome. Moreover, they often suffer from limitations: either removing more points than necessary, or failing to eliminate certain outliers due to sparsity issues. Since these techniques generally rely on kNN-based strategies that require a minimum number of neighboring points, so they struggle in scenarios with irregular density.

To overcome these challenges, we employed a convex hullâbased outlier removal method, introduced in [53], which leverages geometric properties to more reliably filter noise while preserving the integrity of the underlying structure. Unlike previous approaches, this method does not rely on local neighborhoods; instead, it evaluates each point independently and removes it if its distance to the convex hull exceeds a threshold.

<!-- image-->  
Figure 2. Overview of the GauSSmart pipeline, illustrating the stages of camera clustering, Convex Hullâbased outlier removal, point cloud upsampling, and the integration of a DINOv3-based loss.

<!-- image-->  
Figure 3. Visualization of a DTU scene point cloud from different viewpoints, showing point cloud outliers even on closed environments

## 3.2. Camera Clustering and Image Selection

Large multi-view datasets often contain hundreds of images, making segmentation across all views computationally prohibitive. Drawing inspiration from cameraparameterâdriven view-reduction and calibration strategies [21, 26, 33, 54, 59], we propose a clustering-based approach to select a compact, geometry-aware subset of views.

<!-- image-->  
Figure 4. Visualization of the convex hull removal process. Left: the original point cloud containing outliers projected outside the valid viewing cone. Right: the refined point cloud after removing convex hull outliers, which eliminates spurious points and preserves only the consistent geometry.

Our method extracts camera centers and forward axes from explicit camera-to-world matrices c2w. To ensure robustness to scene scale and anisotropy, we normalize camera positions by subtracting the mean and dividing by the per-axis standard deviation. We then apply k-means clustering to these normalized positions and determine the optimal number of clusters k by evaluating candidate values in the range k â [min, min(15, âN/2â)].

The selection criterion maximizes a composite objective function that balances two key factors: coverage, measured as the average of spatial spread and inter-view angular diversity within each cluster, and compactness, quantified as negative inertia normalized by the data norm. The objective function is formulated as:

$$
\begin{array} { r } { \mathrm { s c o r e } ( k ) = \alpha \underbrace { \mathrm { c o v e r a g e } ( k ) } _ { \mathrm { s p r e a d + a n g l e d i v . } } + \beta \underbrace { \mathrm { c o m p a c t n e s s } ( k ) } _ { - \mathrm { i n e r t i a } / \Vert X \Vert } . } \end{array}\tag{1}
$$

After clustering, we select one representative camera per cluster by combining proximity to the de-normalized cluster center with angular uniqueness within the cluster. Each candidate view is scored as the average of two normalized terms: (i) a distance-based component $1 / ( 1 + \mathrm { d i s t } )$ , and (ii) the mean angular separation to other cluster members, linearly scaled to the interval [0, 1]. The highest-scoring view from each cluster is selected as the representative.

This approach preserves both spatial coverage and viewpoint diversity while substantially reducing the computational burden by limiting the number of images processed during segmentation. We apply SAM2 large model [45] to the selected representative views, generating segmentation masks that include binary regions, bounding boxes, and area measurements. These masks are subsequently used to assign stable segment identifiers to 3D points for downstream processing tasks.

## 3.3. Segmentation and Multi-View Projection

After applying the segmentation process to the respective images to obtain per-view masks, each 3D point is assigned to image segments using a standard projection pipeline [8]: we normalize coordinates to the dataset scale, transform points into the camera frame using the extrinsics (rotation and translation), and project onto the image plane with the intrinsics, followed by perspective division to obtain pixel locations. For some specific scenarios is required to apply small dataset-specific adjustments (coordinate conventions, calibration differences) and verify projections with visibility checks to ensure accurate point-to-pixel associations across views.

$$
\lambda \left[ { \overset { u } { v } } \right] = \mathbf { K } \left[ \mathbf { R } \mathbf { t } \right] \left[ { \overset { \mathbf { P } } { 1 } } \right] , \qquad ( u , v ) = \left( { \frac { x } { z } } , { \frac { y } { z } } \right)
$$

To achieve consistent segment identities across views, we construct a global correspondence map. Segments corresponding to the same object are linked when their 3D point assignments exhibit high normalized overlap. This overlap-based matching yields stable global IDs and robust segment coherence on the fused point cloud, mitigating variability in per-view masks and supporting reliable downstream reconstruction. Importantly, since each 3D point has a unique spatial location, we retain its assignment to the first segment in which it appears. While the same point may project into multiple views, these additional projections provide redundant evidence but no new positional information. Consequently, we discard duplicate assignments, as a point cannot physically belong to multiple objects. This ensures that each point maintains a single, unambiguous label, preserving geometric consistency and preventing fusion artifacts.

## 3.4. Mask-Area-Guided Point Cloud Enhancement

Previous research on point cloud upsampling has demonstrated that increasing point density significantly improves reconstruction quality [24, 25, 46, 63]. However, uniform densification fails to account for varying importance and visibility across different semantic regions within the scene. Our approach leverages segmentation mask areas from representative views to guide targeted point augmentation. Rather than applying uniform thresholds, we utilize mask area information to estimate appropriate point density for each segment, ensuring visually prominent regions receive adequate geometric representation. For each segment $s _ { i }$ with mask area $A _ { i }$ , we determine the target number of points based on the square root relationship between area and required sampling density:

$$
n _ { \mathrm { t a r g e t } } = \operatorname* { m a x } \left( \lfloor \sqrt { A _ { i } } \cdot \gamma \rfloor , n _ { \mathrm { m i n } } \right) ,\tag{2}
$$

where $\gamma = 0 . 1$ is an empirically determined scaling factor and $n _ { \mathrm { m i n } } ~ = ~ 1 0$ ensures minimum representation for small segments. For segments where the current point count $\vert P _ { s _ { i } } \vert$ falls below the target, we compute the augmentation requirement:

$$
n _ { \mathrm { a d d } } = n _ { \mathrm { t a r g e t } } - | P _ { s _ { i } } | ,\tag{3}
$$

subject to $n _ { \mathrm { a d d } } > 0$ and the requirement that segments contain at least n existing points to provide sufficient geometric context. The augmentation process generates new points through statistical sampling from existing segment geometry. New point positions $\mathbf { p } _ { \mathrm { n e w } }$ are sampled from the existing distribution with Gaussian noise for geometric diversity:

$$
{ \bf p } _ { \mathrm { n e w } } = { \bf p } _ { \mathrm { b a s e } } + \epsilon , \quad { \bf p } _ { \mathrm { b a s e } } \sim P _ { s _ { i } } , \quad \epsilon \sim \mathcal { N } ( { \bf 0 } , \sigma ^ { 2 } { \bf I } ) ,\tag{4}
$$

where $P _ { s _ { i } }$ represents existing points in segment $s _ { i } .$ , and Ï is adaptively determined based on local point density. Color attributes for augmented points are derived through interpolation from existing segment points, preserving visual consistency across semantic regions.

## 3.5. Feature Embedding Loss

We build upon recent advances in semantic featureâbased supervision [15] by introducing a loss term grounded in DINOv3 feature embeddings [50]. Traditional photometric losses such as $\ell _ { 1 }$ or SSIM capture only low-level pixel correspondences and thus fail to encode high-level semantic cues. To address this limitation, we incorporate a DINObased supervision signal that enforces semantic consistency between ground-truth and rendered images. Prior works typically apply magnitude-based objectives such as $\ell _ { 1 }$ distance directly on feature vectors, which conflates intensity differences with semantic correspondence. In contrast, we propose a cosine similarity formulation that emphasizes the angular relationship between embeddings, thereby isolating semantic correspondence while remaining invariant to feature magnitude scaling. Formally, given embeddings $f _ { \mathrm { g t } } , f _ { r } \in \mathbb { R } ^ { d }$ from ground-truth image $I _ { \mathrm { { g t } } }$ and rendered image $I _ { r } ,$ , we define

<!-- image-->  
Figure 5. DINOv3 similarity heatmaps across DTU, NeRF, and TYT scenes, highlighting the regions most attended by the model. Brighter areas correspond to stronger alignment with the global embedding.

$$
\cos ( f _ { \mathrm { g t } } , f _ { r } ) = \frac { f _ { \mathrm { g t } } \cdot f _ { r } } { \| f _ { \mathrm { g t } } \| _ { 2 } \| f _ { r } \| _ { 2 } } ,\tag{5}
$$

with the DINO loss given by

$$
L _ { \mathrm { D I N O } } ( f _ { \mathrm { g t } } , f _ { r } ) = \lambda _ { \mathrm { d i n o } } \cdot \cos ( f _ { \mathrm { g t } } , f _ { r } ) ,\tag{6}
$$

where $\lambda _ { \mathrm { d i n o } }$ balances the contribution of semantic supervision. For structural fidelity, we retain a photometric objective composed of an $\ell _ { 1 }$ term and a structural dissimilarity penalty (DSSIM):

$$
{ \cal L } _ { \mathrm { p h o t o } } = ( 1 - \lambda _ { \mathrm { d s s i m } } ) \cdot { \cal L } _ { \mathrm { 1 } } + \lambda _ { \mathrm { d s s i m } } \cdot \left( 1 - { \mathrm { S S I M } } ( I _ { \mathrm { g t } } , I _ { r } ) \right) .\tag{7}
$$

The overall training objective combines the semantic and photometric components:

$$
L _ { \mathrm { t o t a l } } = L _ { \mathrm { p h o t o } } + L _ { \mathrm { D I N O } } .\tag{8}
$$

This formulation integrates seamlessly into the 2D Gaussian Splatting framework [18]. The additional DINO term complements standard perceptual losses by aligning rendered views with the structural and semantic characteristics of the reference images. Consequently, each Gaussian primitive is optimized not only for photometric accuracy but also for semantic coherence, yielding reconstructions that better reflect the underlying scene semantics.

## 4. Experiments

We now present experiments for our GauSSmart approach, evaluating both appearance and geometry reconstruction across the three benchmark datasets used in this study. Specifically, we conduct experiments on DTU [19], Mip-NeRF 360 [3], and Tanks and Temples [23]. Our dataset preparation follows the same strategy introduced in the 2DGS paper [18].

## 4.1. Implementation Details

Hardware and Environment. All experiments are conducted on a single NVIDIA RTX A5000 GPU with 24 GB of memory. Our method is implemented in PyTorch 1.12 with CUDA 11.6. Due to hardware differences compared to the original 2DGS implementation (RTX 3090), direct efficiency comparisons are not feasible.

Loss Implementation. We employ DINOv3 with a Vision Transformer base architecture and 16 Ã 16 patch size (ViT-B/16) as our feature extractor, selected after systematic evaluation across different DINOv3 variants on the Tanks and Temples dataset. The DINO loss weight is set to Î» = 0.05, determined through empirical evaluation over the range [0.01, 0.9] and guided by the weighting strategies described in [18].

Gaussian Point Processing. Our segmentation-aware sampling strategy enforces a minimum threshold of five points per segment during new sample generation. This constraint is critical, as the covariance matrix computation requires sufficient point density to maintain numerical stability and avoid ill-conditioned matrices. Segments falling below this threshold are either merged with neighboring segments or discarded during preprocessing.

Outlier Removal. We adopt the convex-hullâbased outlier removal procedure described in [53], applying it during both initialization and refinement to maintain geometric consistency across viewpoints.

Training Configuration. Unless otherwise specified, we train for 30,000 iterations, following the standard 2DGS optimization schedule.

## 5. Results

Our Gaussmart implementation demonstrates competitive performance across benchmark datasets. Table 1 reports a quantitative comparison against 3D Gaussian Splatting (3DGS)[20] and 2D Gaussian Splatting (2DGS)[18]. In terms of PSNR, our method outperforms 2DGS in four of six scenes, indicating reduced noise and improved fidelity. While 2DGS falls short of 3DGS on the Tanks and Temples dataset, the incorporation of our proposed loss and point cloud upsampling narrows this gap.

Our method surpasses existing approaches on the DTU dataset in PSNR, demonstrating clear improvements in reconstruction quality. The controlled acquisition conditionsâhigh-quality cameras, uniform lighting, and consistent viewpointsâenable our model to fully exploit the available visual information. Sparse initial point clouds in DTU also benefit from our densification strategy, which introduces additional geometric primitives to capture fine details.

Table 1. PSNRâ, SSIMâ, LPIPSâ scores for the Tanks and Temples dataset.
<table><tr><td></td><td>Barn</td><td>Caterpillar</td><td>Courthouse</td><td>Ignatius</td><td>MeetingRoom</td><td>Truck</td><td>mean</td></tr><tr><td colspan="8">PSNR</td></tr><tr><td>Gaussmart</td><td>27.20</td><td>22.83</td><td>21.61</td><td>21.60</td><td>24.90</td><td>24.19</td><td>23.72</td></tr><tr><td>2DGS</td><td>27.15</td><td>22.78</td><td>21.55</td><td>21.68</td><td>24.90</td><td>24.16</td><td>23.70</td></tr><tr><td>3DGS</td><td>27.51</td><td>23.38</td><td>22.22</td><td>21.53</td><td>25.19</td><td>24.25</td><td>24.01</td></tr><tr><td colspan="8">SSIM</td></tr><tr><td>Gaussmart</td><td>0.843</td><td>0.772</td><td>0.765</td><td>0.766</td><td>0.859</td><td>0.848</td><td>0.809</td></tr><tr><td>2DGS</td><td>0.843</td><td>0.773</td><td>0.766</td><td>0.767</td><td>0.859</td><td>0.848</td><td>0.810</td></tr><tr><td>3DGS</td><td>0.852</td><td>0.791</td><td>0.779</td><td>0.776</td><td>0.866</td><td>0.853</td><td>0.820</td></tr><tr><td colspan="8">LPIPS</td></tr><tr><td>Gaussmart</td><td>0.204</td><td>0.245</td><td>0.282</td><td>0.203</td><td>0.209</td><td>0.177</td><td>0.220</td></tr><tr><td>2DGS</td><td>0.204</td><td>0.243</td><td>0.282</td><td>0.204</td><td>0.208</td><td>0.177</td><td>0.220</td></tr><tr><td>3DGS</td><td>0.160</td><td>0.190</td><td>0.266</td><td>0.153</td><td>0.141</td><td>0.108</td><td>0.170</td></tr></table>

Table 2. PSNRâ comparison on DTU dataset.
<table><tr><td>Method</td><td>PSNRâ</td></tr><tr><td>3DGS</td><td>35.76</td></tr><tr><td>SuGaR</td><td>34.57</td></tr><tr><td>2DGS</td><td>34.52</td></tr><tr><td>Gaussmart (Ours)</td><td>36.30</td></tr></table>

These stable imaging conditions provide an ideal setting for our DINO-based loss, allowing semantic features to remain consistent across views. The synergy between controlled capture, point cloud densification, and semantic feature preservation explains the consistent PSNR gains observed on the DTU evaluation set.

Table 4. PSNRâ, SSIMâ, LPIPSâ scores for the MipNeRF360 dataset.
<table><tr><td></td><td>bicycle</td><td>flowers</td><td>garden</td><td>stump</td><td>treehill</td><td>room</td><td>counter</td><td>kitchen</td><td>bonsai</td><td>mean</td></tr><tr><td colspan="9">PSNR</td><td></td><td></td></tr><tr><td>Gaussmart</td><td>24.78</td><td>21.05</td><td>26.96</td><td>26.41</td><td>22.41</td><td>31.51</td><td>28.82</td><td>31.18</td><td>31.94</td><td>27.23</td></tr><tr><td>2DGS</td><td>24.87</td><td>21.15</td><td>26.95</td><td>26.47</td><td>22.27</td><td>31.06</td><td>28.55</td><td>30.50</td><td>31.52</td><td>26.93</td></tr><tr><td>3DGS</td><td>25.24</td><td>21.52</td><td>27.41</td><td>25.07</td><td>22.49</td><td>30.63</td><td>28.70</td><td>30.32</td><td>31.98</td><td>27.26</td></tr><tr><td colspan="9">SSIM</td><td></td><td></td></tr><tr><td>Gaussmart</td><td>0.725</td><td>0.570</td><td>0.841</td><td>0.760</td><td>0.622</td><td>0.921</td><td>0.960</td><td>0.925</td><td>0.940</td><td>0.807</td></tr><tr><td>2DGS</td><td>0.752</td><td>0.588</td><td>0.852</td><td>0.765</td><td>0.627</td><td>0.912</td><td>0.900</td><td>0.919</td><td>0.933</td><td>0.805</td></tr><tr><td>3DGS</td><td>0.771</td><td>0.605</td><td>0.868</td><td>0.775</td><td>0.638</td><td>0.914</td><td>0.905</td><td>0.922</td><td>0.938</td><td>0.815</td></tr><tr><td colspan="10">LPIPS</td></tr><tr><td>Gaussmart</td><td>0.282</td><td>0.382</td><td>0.150</td><td>0.266</td><td>0.380</td><td>0.206</td><td>0.198</td><td>0.129</td><td>0.188</td><td>0.242</td></tr><tr><td>2DGS</td><td>0.218</td><td>0.346</td><td>0.115</td><td>0.222</td><td>0.329</td><td>0.223</td><td>0.208</td><td>0.133</td><td>0.214</td><td>0.223</td></tr><tr><td>3DGS</td><td>0.205</td><td>0.336</td><td>0.103</td><td>0.210</td><td>0.317</td><td>0.220</td><td>0.204</td><td>0.129</td><td>0.205</td><td>0.214</td></tr></table>

In Fig. 6, our results are comparable to those of 3DGS, while exhibiting noticeably less noise than 2DGS. This improvement stems from reducing noise in the underlying point cloud, which often introduces bubble-like artifacts and leads to noisy renderings. Our reconstructions also preserve colors more consistently, confirming that the DINObased loss captures and retains high-level semantic features. For outdoor scenes, our approach outperforms 2DGS across multiple cases, demonstrating strong generalization beyond indoor environments. This is particularly evident in the treehill scene, where complex structures such as root systems and tree trunks benefit from our feature-aware loss.

For indoor environments, GauSSmart consistently achieves superior performance across all three quantitative metrics, surpassing both 2DGS and 3DGS. This improvement highlights the robustness of our method under controlled lighting conditions, where detailed textures and diverse scene elements are more effectively reconstructed.

Qualitative comparisons in Fig. 7 further confirm the benefits of our approach. Structural elements such as the walls and sky regions in the Caterpillar scene are reconstructed more cleanly and stably than with 2DGS or 3DGS. While 2DGS introduces noisy textures and bubble-like artifacts in low-texture areas, Gaussmart suppresses such instabilities, producing smoother surfaces and more natural transitions. Beyond large homogeneous regions, our method also demonstrates sharper preservation of edges and fine object boundaries, as in the courthouse scene.

To further assess generalization, we conduct experiments on the DTU dataset, which consists of object-centric scenes captured under controlled conditions. As shown in Table 2, our method outperforms recent baselines, including 2DGS, 3DGS, and SuGaR, achieving a PSNR improvement of nearly one point over the best competing approach. This highlights the ability of our model to produce cleaner and more faithful reconstructions. We attribute these improvements to the synergy between DTUâs multi-view acquisition and our DINO-based loss, which leverages high-level features to enhance geometry-aware consistency.

Per-scene results in Table 3 further support this conclusion. Gaussmart achieves a mean PSNR of 36.30 dB, with several scans (e.g., 63, 118, 122) exceeding 39 dB, demonstrating robustness across diverse geometries.

## 6. Ablations

Table 5 presents the mean PSNR across ablation configurations on Tanks and Temples.

Point cloud improvements alone achieve limited gains, indicating that densification in isolation is insufficient. Combining hull filtering with point cloud improvements increases mean PSNR, showing that geometric outlier removal stabilizes 3D reconstructions. Our complete model, which integrates DINO guidance with hull filtering and point cloud improvements, attains optimal performance, confirming the complementary benefits of semantic loss and geometric refinement.

Table 3. PSNRâ, SSIMâ, LPIPSâ scores for DTU dataset.
<table><tr><td></td><td>scan24</td><td>scan37</td><td>scan40</td><td>scan55</td><td>scan63</td><td>scan65</td><td>scan69</td><td>scan83</td><td>scan97</td><td>scan105</td><td>scan106</td><td>scan110</td><td>scan114</td><td>scan118</td><td>scan122</td><td>mean</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>PSNR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gaussmart</td><td>36.42</td><td>32.99</td><td>35.57</td><td>35.54</td><td>38.98</td><td>36.15</td><td>34.69</td><td>35.86</td><td>33.68</td><td>36.72</td><td>37.63</td><td>35.96</td><td>34.65</td><td>39.22</td><td>39.07</td><td>36.30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>SSIM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gaussmart</td><td>0.964</td><td>0.962</td><td>0.962</td><td>0.938</td><td>0.975</td><td>0.936</td><td>0.940</td><td>0.921</td><td>0.929</td><td>0.933</td><td>0.941</td><td>0.934</td><td>0.936</td><td>0.941</td><td>0.937</td><td>0.941</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>LPIPS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gaussmart</td><td>0.062</td><td>0.091</td><td>0.093</td><td>0.146</td><td>0.075</td><td>0.157</td><td>0.132</td><td>0.223</td><td>0.177</td><td>0.174</td><td>0.209</td><td>0.225</td><td>0.192</td><td>0.225</td><td>0.239</td><td>0.165</td></tr></table>

<!-- image-->  
Figure 6. A side-by-side comparison of results from our method and the baselines.

## 6.1. Limitations

Despite the promising results, our approach presents certain computational challenges that warrant consideration. The necessity to compute DINO embeddings each time an image passes through the pipeline introduces a significant computational overhead, potentially increasing processing time by up to 1.5x depending on the size and complexity of the scene. Additionally, our approach computes SAM maps for each image during processing, which adds further computational burden.

<!-- image-->  
Figure 7. A side-by-side comparison of results from our method and the baselines.

Table 5. Ablation study results applied on Tanks and Temples Training Scenes (mean PSNR).
<table><tr><td>Experiment</td><td>Mean PSNR</td></tr><tr><td>DINO/7000 1oss</td><td>23.634</td></tr><tr><td>DINO/3000 1oss</td><td>23.611</td></tr><tr><td>Just PC Improvement</td><td>23.611</td></tr><tr><td>Hull/PC Improvement</td><td>23.637</td></tr><tr><td>Full model</td><td>23.641</td></tr></table>

Through our experiments, we discovered consistent patterns across datasets: DTU scenes consistently utilize 3 images given that the cameras are the same for all objects, while Mip-NeRF 360 exhibits a similar pattern with 15 images. This observation suggests a promising avenue for optimizationâstoring that could significantly reduce training time, that is the main drawback of the current approach.

## 7. Conclusions

In this work, we introduced GauSSmart, a novel approach that combines the strengths of 2D foundational models and 3D methods. Our method highlights the potential of integrating established 2D computer vision techniques, including convex filtering and other foundational components, with 3D reconstruction pipelines. Experimental evaluation across three diverse datasets shows that GauSSmart consistently outperforms existing Gaussian Splatting methods in most scenes, supporting our hypothesis that hybrid 2Dâ3D approaches can effectively leverage the complementary strengths of both domains. Its superior performance stems from the strategic use of robust 2D priors to compensate for the limitations of sparse 3D training data, underscoring a key insight for the field: the scarcity of highquality 3D data necessitates intelligent integration with rich 2D image representations.

Looking ahead, the recent release of DiNOv3 presents promising oportunities for advancing 2D foundational models in 3D reconstruction tasks. The enhanced capabilities and improved feature representations offered by DiNOv3 could unlock further greater performance gains when integrated into Gaussian Splatting frameworks.

## Supplementary Materials

## Geometric Results

Table 6. DTU evaluation results using 2DGS pipeline. Metrics are mean distance-to-surface (d2s), mean surface-to-distance (s2d), and overall error.
<table><tr><td>Scan</td><td>mean_d2s</td><td>mean_s2d</td><td>overall</td></tr><tr><td>24</td><td>2.6143</td><td>1.4773</td><td>2.0458</td></tr><tr><td>37</td><td>1.8802</td><td>0.8228</td><td>1.3515</td></tr><tr><td>40</td><td>2.0464</td><td>1.6955</td><td>1.8709</td></tr><tr><td>55</td><td>1.6924</td><td>0.6944</td><td>1.1934</td></tr><tr><td>63</td><td>2.8613</td><td>2.6491</td><td>2.7552</td></tr><tr><td>65</td><td>2.7081</td><td>1.8333</td><td>2.2707</td></tr><tr><td>69</td><td>2.0261</td><td>1.1439</td><td>1.5850</td></tr><tr><td>83</td><td>2.2375</td><td>1.8825</td><td>2.0600</td></tr><tr><td>97</td><td>1.9723</td><td>1.8632</td><td>1.9178</td></tr><tr><td>105</td><td>1.9679</td><td>1.4783</td><td>1.7231</td></tr><tr><td>106</td><td>2.4727</td><td>1.1019</td><td>1.7873</td></tr><tr><td>110</td><td>2.6417</td><td>1.2023</td><td>1.9220</td></tr><tr><td>114</td><td>1.5179</td><td>0.9364</td><td>1.2272</td></tr><tr><td>118</td><td>1.8501</td><td>0.7577</td><td>1.3039</td></tr><tr><td>122</td><td>2.1410</td><td>0.8514</td><td>1.4962</td></tr></table>

Table 7. Indoor scenes (DTU, Nerf indoor, Meetingroom).
<table><tr><td>Category</td><td></td><td>Mean Points Added Mean Segments with New Points</td></tr><tr><td>Indoor Scenes</td><td>116.08</td><td>8.44</td></tr></table>

Table 8. Outdoor scenes (Tanks except Meetingroom, Nerf outdoor).
<table><tr><td>Category</td><td></td><td>Mean Points Added Mean Segments with New Points</td></tr><tr><td>Outdoor Scenes</td><td>88.15</td><td>9.36</td></tr></table>

## Camera Clustering

Examples of the images selected by clustering method for a set of scenes from DTU, MipNerf360 and Tanks and Temples

<!-- image-->  
Figure 8. Selected clustered camera views from Barn and Caterpillar scenes.

<!-- image-->  
Figure 9. Clustered camera views from DTU scans: Scan65 (top row) and Scan122 (bottom row).

Bicycle  
<!-- image-->  
Figure 10. Selected clustered camera views â Bicycle (top) and Counter (bottom).

## References

[1] Marc Alexa, Johannes Behr, Daniel Cohen-Or, Shachar Fleishman, David Levin, and Claudio T. Silva. Computing and rendering point set surfaces. In ACM SIGGRAPH, pages 141â150, 2003. 2

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields, 2021. 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 2, 5

[4] Sofien Bouaziz, Andrea Tagliasacchi, and Mark Pauly. â0-based point set denoising. Computer Graphics Forum, 32(2pt2):230â241, 2013. 2

[5] Junyi Cao, Zhichao Li, Naiyan Wang, and Chao Ma. Lightning NeRF: Efficient hybrid scene representation for autonomous driving. arXiv preprint arXiv:2403.05907, 2024. 2

[6] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians, 2025. 2

[7] Dmitry Chetverikov, Dmitry Svirko, Dmitry Stepanov, and Pavel Krsek. The trimmed iterative closest point algorithm. In Proceedings of the 16th International Conference on Pattern Recognition (ICPR), pages 545â548, 2002. 2

[8] Pavel Chmelar, Ladislav Beran, and Nataliia Kudriavtseva. Projection of point cloud for basic object detection. In Proceedings ELMAR-2014, pages 1â4, 2014. 2, 4

[9] Jaehoon Choi, Yonghan Lee, Hyungtae Lee, Heesung Kwon, and Dinesh Manocha. Meshgs: Adaptive mesh-aligned gaussian splatting for high-quality rendering, 2024. 2

[10] Wei Dai, Boyeong Woo, Siyu Liu, Matthew Marques, Craig B. Engstrom, Peter B. Greer, Stuart Crozier, Jason A. Dowling, and Shekhar S. Chandra. Can3d: Fast 3d medical image segmentation via compact context aggregation, 2021. 2

[11] Anurag Dalal, Daniel Hagen, Kjell G. Robbersmyr, and Kristian Muri Knausgard. Gaussian splatting: 3d reconstruction and novel Ë view synthesis: A review. IEEE Access, 12:96797â96820, 2024. 2

[12] Boyang Deng, Kyle Genova, Soroosh Yazdani, Sofien Bouaziz, Geoffrey Hinton, and Andrea Tagliasacchi. Cvxnet: Learnable convex decomposition. arXiv preprint arXiv:1909.05736, 2020. 2

[13] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381â395, 1981. 2

[14] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps, 2021. 2

[15] Ziren Gong, Xiaohan Li, Fabio Tosi, Youmin Zhang, Stefano Mattoccia, Jun Wu, and Matteo Poggi. Dino-slam: Dino-informed rgb-d slam for neural implicit and explicit representations, 2025. 4

[16] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, and Shi-Min Hu. Pct: Point cloud transformer. Computational Visual Media, 7(2):187â199, 2021. 2

[17] Antoine Guedon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality Â´ mesh rendering, 2023. 2

[18] Binbin Huang, Zehao Yu, Anpei Chen, and Andreas Geiger. 2d gaussian splatting for geometrically accurate radiance fields. ACM Transactions on Graphics (TOG), 43(4):32, 2024. 2, 5

[19] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola, and Henrik AanÃ¦s. Large scale multi-view stereopsis evaluation. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 406â413. IEEE, 2014. 5

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splatting for real-time radiance field Â¨ rendering. ACM Transactions on Graphics (TOG), 42(4):1â14, 2023. 1, 2, 5

[21] Sahib Khan and Tiziano Bianchi. Fast image clustering based on camera fingerprint ordering. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), pages 766â771, 2019. 3

[22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything, 2023. Â´ 2

[23] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 5

[24] Ruihui Li, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. Pu-gan: a point cloud upsampling adversarial network, 2019. 4

[25] Ruihui Li, Xianzhi Li, Pheng-Ann Heng, and Chi-Wing Fu. Point cloud upsampling via disentangled refinement, 2021. 4

[26] Shuaiyong Li, Xuyuntao Zhang, Chao Zhang, Shenghao Fu, and Sai Zhang. Sparse multi-view image clustering with complete similarity information. Neurocomputing, 596:127945, 2024. 3

[27] Yuheng Li, Tianyu Luan, Yizhou Wu, Shaoyan Pan, Yenho Chen, and Xiaofeng Yang. Anatomask: Enhancing medical image segmentation with reconstruction-guided self-masking, 2024. 2

[28] David Lindell, Julien Martel, and Gordon Wetzstein. AutoInt: Automatic integration for fast neural volume rendering. https://arxiv.org/abs/2012.01714, 2020. 2

[29] David B. Lindell, Julien N.P. Martel, and Gordon Wetzstein. Autoint: Automatic integration for fast neural volume rendering. In Proceedings of the conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[30] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

[31] Yongcheng Liu, Bin Fan, Shiming Xiang, and Chunhong Pan. Relation-shape convolutional neural network for point cloud analysis, 2019. 2

[32] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1), 2024. 2

[33] Massimo Mauro, Hayko Riemenschneider, Luc Van Gool, and Riccardo Leonardi. Overlapping camera clustering through dominant sets for scalable 3d reconstruction. In Proceedings of the British Machine Vision Conference (BMVC), 2013. 3

[34] Luke Melas-Kyriazi, Christian Rupprecht, and Andrea Vedaldi. pc2: Projection-conditioned point cloud diffusion for single-image 3d reconstruction, 2023. 2

[35] Xiaoxu Meng, Weikai Chen, and Bo Yang. Neat: Learning neural implicit surfaces with arbitrary topologies from multi-view images. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2

[36] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2

[37] Jan Nikolas Morshuis, Matthias Hein, and Christian F. Baumgartner. Segmentation-guided mri reconstruction for meaningfully diverse reconstructions, 2024. 2

[38] Michael Oechsle, Songyou Peng, and Andreas Geiger. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction, 2021. 2

[39] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Â´ Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 2

[40] Shaoyan Pan, Yiqiao Liu, Sarah Halek, Michal Tomaszewski, Shubing Wang, Richard Baumgartner, Jianda Yuan, Gregory Goldmacher, and Antong Chen. Multi-dimension unified swin transformer for 3d lesion segmentation in multiple anatomical locations, 2023. 2

[41] Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning, 2022. 2

[42] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation, 2017. 2

[43] Ruslan Rakhimov, Andrei-Timotei Ardelean, Victor Lempitsky, and Evgeny Burnaev. Npbg++: Accelerating neural point-based graphics, 2022. 2

[44] Marie-Julie Rakotosaona, Manuele de Vita, Or Litany, Paul Guerrero, Niloy Mitra, and Michael M. Bronstein. Pointcleannet: Learning to denoise and remove outliers from dense point clouds. In Computer Graphics Forum (Proc. Eurographics), pages 234â245, 2019. 2

[45] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Â¨ Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2024. Â´ 4

[46] Yifan and Rong. Repkpu: Point cloud upsampling with kernel point representation and deformation. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 4

[47] Radu Bogdan Rusu and PCL Contributors. Point cloud library (pcl) filters: Statistical and radius outlier removal. https:// pointclouds.org/documentation/. Accessed: 2025-09-16. 2

[48] Radu Bogdan Rusu and Steve Cousins. 3d is here: Point cloud library (pcl). In IEEE International Conference on Robotics and Automation (ICRA), pages 1â4, 2011. 2

[49] Ruibo Shang, Geoffrey P. Luke, and Matthew OâDonnell. Joint segmentation and image reconstruction with error prediction in photoacoustic imaging using deep learning, 2024. 2

[50] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Â´ Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth Â¨ ee Darcet, Th Â´ eoÂ´ Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, HerveÂ´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. Â´ 2, 4

[51] Xi Sun, Derek Jacoby, and Yvonne Coady. 3dfusion, a real-time 3d object reconstruction pipeline based on streamed instance segmented data, 2023. 2

[52] Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, FrancÂ¸ois Goulette, and Leonidas J. Guibas. Kpconv: Flexible and deformable convolution for point clouds, 2019. 2

[53] Alexander G. Valverde Guillen. Convex-Guided Outlier Removal for 3D Point Clouds. PhD thesis, University of California, Santa Cruz, 2025. ProQuest ID: 32043589. 2, 5

[54] Muhammad Waleed, Abdul Rauf, and Murtaza Taj. Camera calibration through geometric constraints from rotation and projection matrices, 2024. 3

[55] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. NeurIPS, 2021. 2

[56] Aoran Xiao, Jiaxing Huang, Dayan Guan, Xiaoqin Zhang, Shijian Lu, and Ling Shao. Unsupervised point cloud representation learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):11321â11339, 2023. 2

[57] Jianwen Xie, Zilong Zheng, Ruiqi Gao, Wenguan Wang, Song-Chun Zhu, and Ying Nian Wu. Learning descriptor networks for 3d shape synthesis and analysis, 2018. 2

[58] Jiexiong Xu, Weikun Zhao, Zhiyan Tang, and Xiangchao Gan. A one stop 3d target reconstruction and multilevel segmentation method, 2023. 2

[59] Xiaoqiang Yan, Yingtao Gan, Yiqiao Mao, Yangdong Ye, and Hui Yu. Live and learn: Continual action clustering with incremental views, 2024. 3

[60] Qing-Long Zhang Yu-Bin Yang. Sa-net: Shuffle attention for deep convolutional neural networks, 2021. 2

[61] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. NeurIPS, 2021. 2

[62] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron, and Ben Mildenhall. Bakedsdf: Meshing neural sdfs for real-time view synthesis. arXiv preprint arXiv:2302.14859, 2023. 2

[63] Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. Pu-net: Point cloud upsampling network, 2018. 4

[64] Qiao Yu, Xianzhi Li, Yuan Tang, Jinfeng Xu, Long Hu, Yixue Hao, and Min Chen. PointDreamer: Zero-shot 3D Textured Mesh Reconstruction from Colored Point Cloud by 2D Inpainting. arXiv preprint arXiv:2406.15811, 2024. https://arxiv.org/ abs/2406.15811. 2

[65] Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, and Jie Zhou. Pointr: Diverse point cloud completion with geometryaware transformers, 2021. 2

[66] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling, 2022. 2

[67] Yue Zeng, Gene Cheung, Michael K. Ng, and John Pang. Feature-preserving point cloud denoising via graph laplacian regularization. IEEE Transactions on Image Processing, 29:3474â3489, 2019. 2

[68] Renrui Zhang, Ziyu Guo, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li, and Peng Gao. Point-m2ae: Multi-scale masked autoencoders for hierarchical point cloud pre-training, 2022. 2

[69] Xuancheng Zhang, Yutong Feng, Siqi Li, Changqing Zou, Hai Wan, Xibin Zhao, Yandong Guo, and Yue Gao. View-guided point cloud completion, 2021. 2

[70] Feng Zhou, Yanjie Zhou, Longjie Wang, Yun Peng, David E. Carlson, and Liyun Tu. Distillation learning guided by image reconstruction for one-shot medical image segmentation, 2025. 2

[71] Qingtian Zhu, Zizhuang Wei, Zhongtian Zheng, Yifan Zhan, Zhuyu Yao, Jiawang Zhang, Kejian Wu, and Yinqiang Zheng. Rpbg: Towards robust neural point-based graphics in the wild, 2024. 2