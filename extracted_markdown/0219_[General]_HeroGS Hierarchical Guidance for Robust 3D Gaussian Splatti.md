# HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views

Jiashu Liâ  Xumeng Hanâ  Zhaoyang Weiâ  Zipeng Wang Kuiran Wang Guorong Li

Zhenjun Hanâ Jianbin Jiao

University of the Chinese Academy of Sciences

lijiashu24@mails.ucas.ac.cn

## Abstract

3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributionsâcharacterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGSâHierarchical Guidance for Robust 3D Gaussian Splattingâa unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and copruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.

## 1. Introduction

Reconstructing high-fidelity 3D scenes for photorealistic novel view synthesis remains a core challenge in computer vision. While Neural Radiance Fields (NeRF) [20] and its variants [2, 3, 5] achieve remarkable visual quality, their implicit representation incurs slow rendering. Recently, 3D Gaussian Splatting (3DGS) [13] introduces an explicit and efficient alternative, delivering NeRF-level fidelity with real-time performance, and a series of its extensions [25, 27, 34] rapidly establishing a new state-of-the-art for high-quality, efficient scene reconstruction.

With the success of these methods in dense-view scenarios, the research frontier is now shifting toward the more challenging sparse-input setting. In this regime, the scarcity of viewpoints often leads to irregular Gaussian distributions, resulting in geometric ambiguities and severe rendering artifacts. FSGS [36] accelerates early-stage densification to compensate for the insufficient initial Gaussians, while DropGaussian [23] adopts a dropout-based strategy to regularize the Gaussian distributions in occluded regions. However, the absence of comprehensive guidance still leaves the Gaussian field imperfectly optimized, resulting in inaccurate and uneven distributions â with insufficient Gaussians in background regions leading to blurriness, and inadequate supervision over high-frequency details causing misaligned or misplaced Gaussians. This motivates us to introduce a multi-level framework that globallyto-locally coherent guidance to more comprehensively direct the formation of accurate Gaussian distributions.

Sparse-view settings often cause Gaussians to fall outside the field of view, receiving limited gradient feedback and overfitting to a few training images [23]. As shown in Fig. 1, adding more views improves gradient coverage and alleviates this issue. Inspired by this, we synthesize pseudo-labels that, together with real views, provide dense image-level supervision to regularize Gaussian distributions. This enhanced supervision enriches gradient propagation across the scene, leading to more uniform Gaussian distributions and improved reconstruction fidelity under sparse inputs, while providing richer spatial structural cues for fine-grained guidance at the feature level.

To further refine and regularize the Gaussian distributions in high-frequency regions, we introduce the Feature-

<!-- image-->  
Figure 1. Motivation of HeroGS. (Left) Image-level pseudolabels bridge the gap between sparse and dense supervision, yielding a more complete Gaussian distributions. (Right) FADP and CPG refine inaccurate Gaussians by enhancing distributions closer to ground-truth geometry and pruning inconsistencies.

Adaptive Densification and Pruning (FADP) at the feature level. FADP increases Gaussian density along edgeaware regions to capture high-frequency details, prunes redundant splats within homogeneous patches to avoid oversaturation, and adds new Gaussians in sparse background areas to improve coverage. The refined Gaussian distributions from FADP provide a stable foundation for the parameter-level, enabling more reliable co-pruning based on geometric consistency.

At the parameter level, we introduce the Co-Pruned Geometry Consistency (CPG), which eliminates abnormal or inconsistent Gaussian distributions via a co-pruning mechanism combined with a post-freeze behavior strategy. Co-Pruning leverages the self-consistency of the Gaussian field to jointly evaluate and filter Gaussian parameters, retaining only those that exhibit stable and geometrically consistent distributions. This process effectively preserves robust Gaussians while eliminating unstable or redundant ones, leading to a cleaner and more reliable scene representation, as illustrated in the right part of Fig. 1. The whole pipeline constitutes a hierarchical guidance strategy that collaboratively optimizes the Gaussian distributions across levels, enhancing its global consistency and fidelity. Our contributions can be summarized as follows:

â¢ We propose HeroGS, a hierarchical guidance framework that enables compact and high-fidelity 3D reconstructions under sparse-view conditions.

â¢ At image level, pseudo-labels are generated to promote more accurate Gaussian distributions. At feature and parameter levels, the proposed FADP and CPG further refine Gaussian field by enhancing feature-aware density and enforcing geometric consistency, respectively.

â¢ Extensive qualitative and quantitative experiments on real-world datasets, including large-scale scenes and various training views, demonstrate that our method significantly outperforms state-of-the-art baselines.

## 2. Related Work

Novel View Synthesis. Neural Radiance Fields (NeRF) [1, 4, 20] and 3D Gaussian Splatting (3DGS) [13] have emerged as two prominent paradigms for high-fidelity novel view synthesis. NeRF-based methods have achieved impressive results in producing photorealistic novel views, especially when abundant views (often hundreds) are available. Meanwhile, 3DGS represents scenes explicitly as a set of 3D Gaussian primitives, which enables faster convergence and real-time rendering performance through GPUfriendly splatting operations. Recent works such as MiniSplatting [12] focus on optimizing 3DGS by introducing memory-efficient representation and lightweight pipeline, while Stop-the-Pop [24] improves rendering realism and robustness by a novel hierarchical rasterization approach. Beyond these, Deformable 3DGaussians [32] extends 3DGS to efficiently reconstruct and render dynamic scenes by learning a canonical static scene and a time-dependent deformation field for Gaussians. However, both categories are fundamentally data-hungry and show performance degradation under sparse input settings, where their reliance on dense photometric supervision becomes a limiting factor.

Novel View Synthesis with sparse views. To address the challenges of sparse views, existing methods based on NeRF are mainly divided into two categories: (1) regularization-based methods, which apply techniques such as depth smoothness [21], occlusion regularization [31], or frequency control to existing sparse data to prevent overfitting; (2) methods incorporating external priors like using pre-trained models to generate depth maps [6, 8, 10, 33] or feature extractors [9, 15] to enhance geometric and visual consistency. For 3DGS specifically, some methods have been proposed, such as DNGaussian [16] introducing depth regularization, SparseGS [30] combining depth and diffusion regularization, FSGS [36] increasing point cloud density through Gaussian unpooling, and CoR-GS [35] and CoherentGS [22] utilizing multi-view consistency or monocular depth for optimization. However, existing sparse-view 3D reconstruction methods generally still suffer from severe overfitting, indicating a need for further research to improve their robustness in practical applications.

## 3. Method

Existing Gaussian Splatting methods often suffer from overfitting caused by irregular Gaussian distributions under sparse-view conditions. To mitigate this limitation, we propose HeroGS, a unified multi-level (image, feature and parameter) guidance framework that provides diverse and complementary supervisory signals to refine and regularize the Gaussian field throughout training. At the image level, HeroGS introduces pseudo supervision, offering comprehensive guidance for Gaussians across different regions. This enhanced supervision enriches gradient propagation, leading to more globally consistent and wellstructured Gaussian field, with camera extrinsics initialized through interpolation and jointly optimized during training. Building upon this, at the feature level, Feature-Adaptive Densification and Pruning (FADP) leverages edge-aware and patch-based features extracted from training views to reinforce high-frequency structures while removing redundant Gaussians, resulting in a more accurate and compact representation. Finally, at the parameter level, Co-Pruned Geometry Consistency (CPG) prunes Gaussians with large parameter discrepancies, effectively suppressing geometric distortions and enhancing global consistency across the scene. This three-level supervision forms a coherent framework (illustrated in Fig. 2) that not only enhances supervision quality but also preserves structural fidelity under sparse-view conditions.

<!-- image-->  
Figure 2. HeroGS Overview. Initialized from SfM, our framework operates across three levels. (1) Image-level Guidance: A set of intermediate RGB frames is synthesized from sparse inputs, offering pseudo-dense guidance that globally regularizes the Gaussian distributions and, at the feature level, manifests as patch-based Gaussian numbers C. (2) Feature-level Refinement: The Feature-Adaptive Densification and Pruning (FADP) leverages edge- and patch-aware features from training views to enhance high-frequency and background regions, while suppressing redundant Gaussians and dilivering finer details for next level. (3) Parameter-level Consistency: The Co-Pruned Geometry Consistency (CPG) employs auxiliary Gaussian fields with partially frozen parameters to perform co-pruning, eliminating geometrically inconsistent splats. These levels form a hierarchical guidance with interconnections (dashed lines) that jointly constrains and optimizes the Gaussian field for improved structural fidelity.

## 3.1. Image Level

Motivated by the observation that denser inputs yield superior reconstructions, we propose to generate intermediate viewpoints between adjacent training views and synthesize their corresponding RGB images as pseudo-labels using a frame interpolation model. Two consecutive training images are defined as ${ \mathbf I } _ { n }$ and ${ \mathbf { I } } _ { n + 1 }$ , with corresponding camera poses $\mathbf { P } _ { n } = ( \mathbf { R } _ { n } , \mathbf { T } _ { n } )$ and $\mathbf P _ { n + 1 } = ( \mathbf R _ { n + 1 } , \mathbf T _ { n + 1 } )$ To simplify the notation, we assume a single intermediate frame, denoted as $\mathbf { I } _ { n } ^ { ( \alpha ) }$ , is generated between consecutive training views ${ \mathbf I } _ { n }$ and ${ \mathbf { I } } _ { n + 1 }$ , where $\alpha ~ \in ~ ( 0 , 1 )$ is the interpolation weight. In our experiments, however, multiple intermediate frames are generated (see Sec. 4.4 for details). In this paper, VFI [26], a state-of-the-art frame interpolation model, is utilized to generate the intermediate images, i.e.,

<!-- image-->  
Figure 3. Removing inconsistent Gaussians. Columns 3 and 4: original training views and frame-interpolated pseudo-labels, respectively. Although pseudo-labels provide overall supervisory signals, they may lack accuracy in fine details and fail to offer precise guidance for high-frequency geometric structures. Columns 1 and 2: novel-view renderings produced by full pipeline versus those without CPG. It suppresses drifts, yielding sharper edges, coherent surfaces and markedly improved spatial fidelity.

$$
\begin{array} { r } { { \bf I } _ { n } ^ { ( \alpha ) } = \mathrm { V F I } \big ( { \bf I } _ { n } , { \bf I } _ { n + 1 } , \alpha \big ) . } \end{array}\tag{1}
$$

Given that our pseudo-labeled images lack ground-truth extrinsic parameters, we initialize the camera pose for each generated image via interpolation. Specifically, spherical linear interpolation (slerp) is employed for rotation and linear interpolation for translation:

$$
\begin{array} { r l } & { \mathbf { R } _ { n } ^ { ( \alpha ) } = \operatorname { s l e r p } ( \mathbf { R } _ { n } , \mathbf { R } _ { n + 1 } , \alpha ) , } \\ & { \mathbf { T } _ { n } ^ { ( \alpha ) } = ( 1 - \alpha ) \mathbf { T } _ { n } + \alpha \mathbf { T } _ { n + 1 } . } \end{array}\tag{2}
$$

In the training phase, the scene is rendered from the generated viewpoints, and supervision is applied by comparing the results with the pseudo-labeled images. Let $\hat { \mathbf { I } } _ { n } ^ { ( \alpha ) }$ be the rendered image from the n-th generated viewpoint. The estimated depth of the generated image ${ \bf { I } } _ { n } ^ { ( \alpha ) }$ is denoted as $\mathbf { D } _ { n } ^ { ( \alpha ) }$ , and the rendered depth map from the same viewpoint is $\hat { \mathbf { D } } _ { n } ^ { ( \alpha ) }$ , which is calculated via volumetric rendering:

$$
\hat { \mathbf { D } } _ { n } ^ { ( \alpha ) } = \sum _ { i = 1 } ^ { L } d _ { i } o _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } ) .\tag{3}
$$

The sum is over the L ordered 3D Gaussians that overlap a pixel. For the i-th Gaussian, $d _ { i }$ is its depth and $o _ { i }$ is its learned opacity. Similar to [36], our training objective integrates photometric supervision on the generated images, measured by L1 and D-SSIM losses, with geometric supervision on the rendered depth maps, assessed using a Pearson correlation coefficient loss, $i . e .$ ,

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { g } = \sum _ { n = 1 } ^ { N - 1 } \Big ( \lambda _ { 1 } \| \mathbf { I } _ { n } ^ { ( \alpha ) } - \hat { \mathbf { I } } _ { n } ^ { ( \alpha ) } \| _ { 1 } + \lambda _ { 2 } { \mathcal { L } } _ { \mathrm { D - S S I M } } ( \mathbf { I } _ { n } ^ { ( \alpha ) } , \hat { \mathbf { I } } _ { n } ^ { ( \alpha ) } ) } } \\ { { \displaystyle ~ + \lambda _ { 3 } \big ( 1 - \mathrm { C o r r } ( \mathbf { D } _ { n } ^ { ( \alpha ) } , \hat { \mathbf { D } } _ { n } ^ { ( \alpha ) } ) \big ) \Big ) . } \quad \quad \quad ( } \end{array}\tag{4}
$$

To ensure reliable supervision, low-quality pseudolabels are filtered out by a selection module, and only the high-quality ones are retained for training. Further details are discussed in appendix.

Since the frame interpolation model lacks 3D awareness, a slight mismatch can occur between generated images and their camera poses. To mitigate this, these interpolated camera poses are parameterized as learnable variables, optimizing them jointly with the Gaussian field throughout training.

## 3.2. Feature Level

While pseudo-labels offer valuable global supervision, their limited precision in fine details makes it difficult to accurately constrain high-frequency geometric structures (Fig. 3), motivating further refinement at feature level.

To resolve this concern and enhance geometric detail, Feature-Adaptive Densification and Pruning (FADP) is introduced. FADP refines Gaussian distributions in highfrequency regions by improving texture-aware geometry through two complementary strategies: edge-aware densification and patch-based density control.

For each training image ${ \mathbf { I } } _ { n } ,$ we first extract its corresponding edge map $\mathbf { E } _ { n }$ using an edge detection model [18]. 2D points are then sampled along the detected edges and back-projected into 3D space to define the centers of new Gaussians. The attributes of each new Gaussian $\hat { \mathbf { G } } ,$ , namely color, opacity, and shape, are initialized based on its neighbors. Specifically, we identify its K-nearest neighbors $( K = 3$ by default), $\{ \hat { \mathbf { G } } _ { 1 } , \hat { \mathbf { G } } _ { 2 } , \hdots , \hat { \mathbf { G } } _ { K } \}$ , from the existing Gaussians. Each attribute AË of the new Gaussian $\hat { \mathbf { G } }$ is computed via inverse-distance weighted interpolation:

$$
{ \hat { \mathbf { A } } } = { \frac { \sum _ { k = 1 } ^ { K } w _ { k } \cdot \mathbf { A } _ { k } } { \sum _ { k = 1 } ^ { K } w _ { k } } } , \quad w _ { k } = { \frac { 1 } { d _ { k } + \epsilon } } ,\tag{5}
$$

where $\mathbf { A } _ { k }$ is the attribute of the k-th nearest inherent $\mathbf { G } _ { k }$ $d _ { k }$ is its Euclidean distance to $\hat { \mathbf { G } } ,$ , and Ïµ is a small constant.

To complement edge-aware densification, a patch-based density controlling strategy is proposed to prevent oversampling and promote balanced Gaussian distributions. Specifically, each training image is divided into an $m \times m$ grid of patches $( m = 8$ by default) and Gaussians are projected onto image plane. Let $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { m ^ { 2 } } \} \in \mathbb { R } ^ { m ^ { 2 } }$ denote the number of projected Gaussians in each of the m Ã m patches. We reweight these counts to obtain ${ \mathcal { C } } ^ { \prime } =$ $\{ c _ { 1 } ^ { \prime } , c _ { 2 } ^ { \prime } , \ldots , c _ { m ^ { 2 } } ^ { \prime } \}$ using the following piecewise function:

$$
c _ { i } ^ { \prime } = \left\{ \begin{array} { l l } { c _ { \mathrm { m i n } } , } & { \mathrm { i f } c \leq \tau _ { \mathrm { s p a r s e } } } \\ { c _ { i } \cdot \lambda _ { \mathrm { l o w } } , } & { \mathrm { i f } \tau _ { \mathrm { s p a r s e } } < c < \tau _ { \mathrm { l o w } } } \\ { c _ { i } , } & { \mathrm { i f } \tau _ { \mathrm { l o w } } \leq c \leq \tau _ { \mathrm { h i g h } } } \\ { c _ { i } \cdot \lambda _ { \mathrm { h i g h } } , } & { \mathrm { i f } c > \tau _ { \mathrm { h i g h } } } \end{array} \right.\tag{6}
$$

where Ïsparse, Ïlow, $\tau _ { \mathrm { h i g h } }$ are density thresholds, and $\lambda _ { \mathrm { l o w } } >$ 1, $\lambda _ { \mathrm { h i g h } } < 1$ are scaling factors that increase sampling in underrepresented regions and suppress sampling in overdense areas. A minimum count $c _ { \mathrm { m i n } }$ is enforced in sparse patches to guarantee coverage.

To maintain the global number of Gaussians, we apply normalization:

$$
{ \mathcal { C } } ^ { \prime } \gets \mathrm { r o u n d } \left( { \mathcal { C } } ^ { \prime } \cdot \frac { \sum _ { i } c _ { i } } { \sum _ { i } c _ { i } ^ { \prime } } \right) .\tag{7}
$$

A residual correction step is applied to ensure $\textstyle \sum _ { i } c _ { i } ^ { \prime } =$ $\sum _ { i } c _ { i }$ by adjusting the entries in $\mathbf { c } ^ { \prime }$

In essence, the globally refined Gaussian distributions obtained from the previous stage provide a more complete structural foundation, bringing $\mathcal { C }$ closer to the distributions required for realistic rendering even before optimization. This enables patch-based density controlling strategy to achieve more effective and reliable refinement.

By jointly leveraging edge-based guidance and patchbased controlling, FADP achieves an optimal balance between texture-sensitive densification and globally consistent Gaussian distributions. The two strategies are tightly coupled: while edge-based densification focuses on adding high-frequency details near object boundaries, patch-based controlling ensures that such additions do not result in local over-concentration or sparsity elsewhere. This synergy enables effective geometry refinement without disrupting the global spatial coherence of the scene.

## 3.3. Parameter Level

Motivated by CoR-GS [35] and aimed at eliminating erroneous Gaussians, we introduce the Co-Pruned Geometry Consistency (CPG) at parameter level. CPG incorporates two auxiliary Gaussian fields, which are trained jointly with the primary field. To facilitate more effective co-pruning, the parameters of the auxiliary Gaussian fields are frozen after a predefined training iteration. By comparing the consistency of corresponding Gaussians across all three fields, we identify and remove inconsistent points through a comprehensive co-pruning strategy. This filtering process effectively suppresses geometric artifacts, such as blurriness and shape distortion, leading to significantly improved spatial coherence, as illustrated in Fig. 3.

Training Strategy. We adopt a two-stage strategy. Before a predefined iteration number $N _ { \mathrm { i t e r } }$ , all three Gaussian Splatting (GS) fields perform mutual co-pruning. Subsequently, the two auxiliary fields are partially frozen (fixing scale and rotation), and only the primary GS field continues to be updated. The pruning becomes unilateral: the primary field is pruned based on geometric agreement with the two frozen auxiliary fields.

Co-Pruning Criterion. Given a source GS field $\mathcal { G } _ { s } ~ =$ $\big \{ \mathbf { G } _ { 1 } ^ { s } , \dots , \mathbf { G } _ { Y } ^ { s } \big \}$ and a target field $\mathcal { G } _ { t }$ , let $\mathbf { p } _ { y } ^ { s } , \mathbf { p } _ { z } ^ { t } \in \mathbb { R } ^ { 3 }$ denote the 3D position of $\mathbf { G } _ { y } ^ { s }$ and $\mathbf { G } _ { z } ^ { t }$ , respectively. For each Gaussian in $\mathcal { G } _ { s } .$ , its nearest neighbor in the target field is obtained as follows:

$$
z ^ { * } = \arg \operatorname* { m i n } _ { z } \| \mathbf { p } _ { y } ^ { s } - \mathbf { p } _ { z } ^ { t } \| _ { 2 } ,\tag{8}
$$

$$
w _ { y } = \| \mathbf { p } _ { y } ^ { s } - \mathbf { p } _ { z ^ { * } } ^ { t } \| _ { 2 } .\tag{9}
$$

$\mathbf { G } _ { y } ^ { s }$ is pruned from source field if the distance exceeds a threshold $w _ { y } \ > \ \delta ,$ where we set the threshold $\delta \ : = \ : 5$ Guided by parameter level, Gaussian fields contain finer details (as highlighted by the orange points in Fig. 2), making it easier to accurately identify erroneous Gaussian $\mathbf { G } _ { y } ^ { s } .$

Post-Freeze Behavior. After $N _ { \mathrm { i t e r } }$ , pruning is applied solely to the primary field, leveraging both auxiliary fields as geometric references. This strategy facilitates progressive refinement through early-stage alignment via multi-field redundancy and late-stage geometry stabilization. Throughout training, each GS field, i.e., the primary and the two auxiliary fieldsâis independently supervised using a combination of training view and generated view losses:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { g } \mathcal { L } _ { g } + \mathcal { L } _ { r } , } \end{array}\tag{10}
$$

where $\mathcal { L } _ { g }$ and $\mathcal { L } _ { r }$ follow the same formulation as in Eq. 4.

## 4. Experiments

## 4.1. Setup

HeroGS is implemented based upon FSGS [36], and is evaluated on the LLFF [19] and Tanks&Temples [14] datasets. On LLFF, we use 2, 3, and 6 training views, with resolutions downsampled by $8 \times$ . Settings for 3 and 6 training views follow previous work [11]. For the 2-view case, as FSGS [36] requires COLMAP-based multi-view stereo which fails on 3 scenes, we evaluate on the remaining 5 scenes. Here, two views are randomly chosen for training, and the rest for testing. On Tanks&Temples, 3 and 6 training views are used without downsampling. In our experiments, a subset of training views is uniformly sampled, with remaining views used for testing. Following prior works [35, 36], PSNR, SSIM, and LPIPS are employed as the evaluation metrics.

## 4.2. Comparison

LLFF. Quantitative results are presented in Tab. 1, and qualitative comparisons are shown in Fig. 4. HeroGS achieves superior performance across multiple metrics. Notably, under the challenging 2-view training setting, HeroGS demonstrates a significant performance advantage over existing baselines. This substantial gain is primarily driven by the strategic introduction of hierarchical guidance, effectively compensating for the scarcity of reliable supervision in extremely sparse input views.

Moreover, as Fig. 4 illustrates, 3DGS struggles to recover the structure of some objects, while DRGS tends to synthesize smooth views lacking high-frequency details, compared to FSGS and HeroGS. Although FSGS recovers most structural details, it often fails to produce accurate local texture patterns. In contrast, HeroGS renders more accurate and detailed textures, visible in the intricate patterns of fortress and trex, and produces clearer background regions, such as the distant structures in fern. This visual fidelity largely stems from our paradigmâs ability to mitigate overfitting by guiding Gaussian distributions at multiple levels.

Tanks&Temples. Extensive evaluations on the Tanks&Temples dataset are conducted to assess the effectiveness of HeroGS in complex and large-scale environments. As shown in Fig. ?? (Appendix) and Tab. 3, HeroGS is compared against several 3DGS-based baselines. The basic 3DGS struggles to preserve geometric and textural fidelity under limited supervision, while prior approaches [35, 36] introduce sparse or noisy regularization, often resulting in oversmoothed geometry and noticeable artifacts. DropGaussian [23] further exhibits ghosting effects due to inaccurate Gaussian distributions in high-frequency regions. In contrast, the proposed HeroGS framework injects pseudo dense supervision at the image level, effectively bridging the sparse-to-dense gap. Furthermore, FADP and CPG collaboratively refine Gaussian distributions, yielding more accurate reconstructions of finegrained structures and high-frequency textures. Notably, HeroGS consistently achieves superior PSNR and SSIM scores, demonstrating robust generalization and stability under sparse-input conditions.

<table><tr><td rowspan="2">Methods</td><td colspan="3">2 views</td><td colspan="3">3 views</td><td colspan="3">6 views</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Mip-NeRF [2]</td><td>-</td><td>-</td><td>-</td><td>16.11</td><td>0.401</td><td>0.460</td><td>22.91</td><td>0.756</td><td>0.213</td></tr><tr><td>SparseNeRF [28]</td><td>17.80</td><td>0.532</td><td>0.372</td><td>19.86</td><td>0.624</td><td>0.328</td><td>23.26</td><td>0.741</td><td>0.235</td></tr><tr><td>FrugalNeRF [17]</td><td>18.29</td><td>0.557</td><td>0.342</td><td>19.92</td><td>0.634</td><td>0.297</td><td>-</td><td>-</td><td>-</td></tr><tr><td>3DGS [13]</td><td>16.19</td><td>0.437</td><td>0.388</td><td>19.24</td><td>0.649</td><td>0.237</td><td>23.63</td><td>0.809</td><td>0.129</td></tr><tr><td>DRGS [7]</td><td>17.04</td><td>0.513</td><td>0.318</td><td>16.73</td><td>0.487</td><td>0.310</td><td>18.60</td><td>0.560</td><td>0.239</td></tr><tr><td>DNGaussian [16]</td><td>17.03</td><td>0.519</td><td>0.362</td><td>19.12</td><td>0.591</td><td>0.294</td><td>22.18</td><td>0.755</td><td>0.198</td></tr><tr><td>FSGS [36]</td><td>15.65</td><td>0.460</td><td>0.412</td><td>20.43</td><td>0.682</td><td>0.248</td><td>24.15</td><td>0.823</td><td>0.128</td></tr><tr><td>CoR-GS [35]</td><td>17.38</td><td>0.539</td><td>0.350</td><td>20.45</td><td>0.712</td><td>0.196</td><td>24.29</td><td>0.824</td><td>0.135</td></tr><tr><td>DropGaussian [23]</td><td>17.32</td><td>0.509</td><td>0.343</td><td>20.55</td><td>0.710</td><td>0.200</td><td>24.55</td><td>0.835</td><td>0.117</td></tr><tr><td>HeroGS (Ours)</td><td>18.78</td><td>0.595</td><td>0.317</td><td>21.30</td><td>0.739</td><td>0.189</td><td>24.59</td><td>0.837</td><td>0.135</td></tr></table>

Table 1. Quantitative results on LLFF with 2, 3, 6 training views. The best and second-best entries are denoted using bold and underline respectively. 3DGS-based methods require multi-view stereo estimation from COLMAP, which fails on 3 scenes for 2 training views. We therefore report the averaged metrics of the remaining scenes.
<table><tr><td rowspan="2">Methods</td><td colspan="3">2 views</td><td colspan="3">3 views</td><td colspan="3">6 views</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGS</td><td>16.19</td><td>0.437</td><td>0.388</td><td>19.24</td><td>0.649</td><td>0.237</td><td>23.63</td><td>0.809</td><td>0.129</td></tr><tr><td>FSGS</td><td>15.65</td><td>0.460</td><td>0.412</td><td>20.43</td><td>0.682</td><td>0.248</td><td>24.15</td><td>0.823</td><td>0.128</td></tr><tr><td>+VFI</td><td>16.91</td><td>0.489</td><td>0.384</td><td>20.68</td><td>0.704</td><td>0.204</td><td>24.18</td><td>0.822</td><td>0.135</td></tr><tr><td>+FADP</td><td>17.28</td><td>0.503</td><td>0.370</td><td>20.99</td><td>0.720</td><td>0.192</td><td>24.25</td><td>0.824</td><td>0.136</td></tr><tr><td>+GSField  $( \mathrm { f r e e z e } _ { s c a l e } )$ </td><td>17.93</td><td>0.564</td><td>0.345</td><td>21.08</td><td>0.732</td><td>0.191</td><td>24.40</td><td>0.832</td><td>0.136</td></tr><tr><td>HeroGS (Ours)</td><td>18.78</td><td>0.595</td><td>0.317</td><td>21.30</td><td>0.739</td><td>0.189</td><td>24.59</td><td>0.837</td><td>0.135</td></tr></table>

Table 2. Ablation study on LLFF dataset with 2, 3, 6 training views. The best and second-best entries are denoted using bold and underline respectively. Under the 2-view setting, the baseline method (FSGS [36]) performs poorly, while our approach achieves a significant performance improvement.

<table><tr><td>Methods</td><td colspan="4">3 Views 6 Views PSNRâ SSIMâLPIPSâ PSNRâ SSIMâ LPIPSâ</td></tr><tr><td>3DGS</td><td>16.39 0.442</td><td>0.432 22.49</td><td>0.725</td><td>0.279</td></tr><tr><td>FSGS</td><td>16.88 0.474</td><td>0.434</td><td>23.63 0.742</td><td>0.281</td></tr><tr><td>CoR-GS</td><td>17.06 0.491</td><td>0.443</td><td>23.59</td><td>0.742 0.275</td></tr><tr><td>DropGaussian</td><td>16.81 0.480</td><td>0.438</td><td>24.15</td><td>0.772 0.215</td></tr><tr><td>HeroGS</td><td>17.51 0.512</td><td>0.422</td><td>24.70</td><td>0.781 0.272</td></tr></table>

Table 3. Quantitative results on Tanks&Temples with 3, 6 training views without downsampling.

## 4.3. Ablation study

We conduct comprehensive ablation studies on the LLFF dataset, encompassing both qualitative and quantitative comparisons. As shown in Tab. 2, simply incorporating image-level guidance significantly improves performance over the baseline (FSGS), especially in extremely sparse view settings. This highlights the effectiveness of using interpolated frames as pseudo-supervision to guide the model. Additionally, Fig. 5 demonstrates that integrating the FADP into the proposed framework results in clearer reconstructions and more reasonable Gaussian distributions. Notably, while the total number of Gaussians is reduced compared to the baseline, their density increases near object boundaries. This contributes to improved geometric fidelity and the preservation of high-frequency details. To validate the effectiveness of multiple GS fields, an ablation study is conducted on their number and configuration. HeroGS employs two additional GS fields, with scale and rotation frozen respectively. This setup is compared against: (1) using a single additional GS field with its scale frozen, and (2) the baseline with only one GS field. As Tab. 2 (in the Appendix) indicates, HeroGS achieves superior performance, demonstrating that disentangled geometric representation improves reconstruction accuracy and detail preservation. Fig. 3 provides further qualitative comparison. We observe that the interpolation network alone fails to accurately recover fine details and textures on interpolated images (Column 4) when compared with the training view (Column 3). When CPG is disabled (Column 2), the rendering suffers from slight geometric misalignment of textures. In contrast, enabling CPG (Column 1) in our paradigm effectively ameliorates both geometric drift and texture degradation, as it prunes Gaussians with spatial deviations and preserves those aligned with consistent geometry.

<!-- image-->

Figure 4. Qualitative Comparison on LLFF (3 training views). Under extreme view sparsity, 3DGS [13] collapses into severe artifacts and blurred geometry. DRGS and FSGS recover coarse structure yet still exhibit over-smoothed textures and noisy backgrounds. In contrast, HeroGS guides the model from complementary levels to deliver markedly sharper object boundaries, richer high-frequency textures, and distinctly clearer distant regionsâdemonstrating the efficacy of our full pipeline.  
<!-- image-->  
Figure 5. Comparison with and without image and feature level guidance. Columns 1 and 2: novel-view renderings produced by baseline versus baseline with pseudo-labels and FADP. Columns 3 and 4: their corresponding Gaussians.

<table><tr><td>Settings</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>w/o Post-Freeze</td><td>21.16</td><td>0.735</td><td>0.190</td></tr><tr><td>All</td><td>21.30</td><td>0.739</td><td>0.189</td></tr></table>

Table 4. Ablation study on Post-Freeze Behavior on LLFF dataset for 3 training views.

## 4.4. Analysis

Impact of Post-Freeze Behavior. To illustrate the effect of the parameter-freezing strategy, the Post-Freeze Behavior is ablated in Tab. 4. The frozen parameters compel the two auxiliary Gaussian fields to learn distinct local optima. This, in turn, facilitates the primary Gaussian field in effectively pruning misplaced or ill-shaped Gaussians, thereby yielding more authentic texture details.

Analysis of Training Dynamics. We further analyze the training dynamics by monitoring the variation of Gaussian counts and PSNR values throughout the optimization process on the LLFF dataset, as illustrated in Fig. 6. As training progresses, our method exhibits a steady improvement in PSNR, indicating consistent convergence and stable optimization, whereas the baseline shows noticeable fluctuations and even performance degradation in later stages. Notably, at around 5K iterations, our approach already surpasses the baseline by a clear margin, while maintaining a significantly lower number of Gaussians. This reduction demonstrates that HeroGS achieves more compact scene representation, leading to improved memory efficiency and faster rendering speed.

Interpolation Factor. The interpolation factor, denoted as S, represents the number of frames generated for supervision between two consecutive training views. Specifically, for Eq. 2, S â 1 views are generated between the n-th and $( n + 1 )$ -th frames, in which case Î± takes values from the set $\textstyle { \Bigl \{ } { \frac { 1 } { S } } , { \frac { 2 } { S } } , \dots , { \frac { S - 1 } { S } } { \Bigr \} }$ , leading to S â 1 intermediate camera extrinsics evenly distributed along the motion path. As Fig. 8 (in the Appendix) shows, performance metrics generally exhibit lower values at an interpolation factor of S = 2. While performance improves significantly from S = 4 onwards, including at S = 8 and $S = 1 6 .$ , all metrics demonstrate a tendency to stabilize beyond $S = 4$ . An interpolation factor of S = 4 is selected in the experiments to strike an optimal balance between visual quality and computational cost.

Frame Interpolation Model Comparisons. Tab. 5 presents a comparative analysis of the effects of employing different frame interpolation models to generate pseudolabels. The first row showcases the performance when using alternative frame interpolation model [29]. The second row demonstrates the results achieved with our chosen model. The third row provides the outcome when the interpolated images are replaced by uniformly sampled ground truth images, while keeping all other weights constant. To clearly demonstrate the efficacy of parameter level, CPG is excluded in all experiments above. As shown in the table, utilizing the interpolated results from our chosen frame interpolation model at image level yields superior performance compared to other models. However, a minor gap still persists when compared to using ground truth images. This discrepancy arises from the presence of irregularly distributed Gaussians, which leads to local geometric misalignment and texture inconsistencies. CPG effectively mitigates the impact of these inconsistencies, thereby enhancing model performance and bridging this gap.

<table><tr><td>Settings</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>PerVFI [29]</td><td>20.78</td><td>0.711</td><td>0.204</td></tr><tr><td>BIMVFI</td><td>20.99</td><td>0.720</td><td>0.192</td></tr><tr><td>Ground Truth</td><td>21.09</td><td>0.720</td><td>0.195</td></tr><tr><td>Ours</td><td>21.30</td><td>0.739</td><td>0.189</td></tr></table>

Table 5. Comparison between different Interpolation Model on LLFF dataset for 3 training views.

Level Interdependency. The numerical trend in Tab. 6 reveals clear hierarchical synergy among the three levels.

<!-- image-->

<!-- image-->  
Figure 6. Analysis of training dynamics. Comparison of PSNR (left) and the number of Gaussians (right) between our HeroGS and baseline during training.

<table><tr><td>PL</td><td>FADP</td><td>CPG</td><td>PSNR â</td><td>SSIM â LPIPS</td></tr><tr><td rowspan="3">One Level</td><td rowspan="3">â â</td><td></td><td>16.91 0.489</td><td>0.384</td></tr><tr><td>16.64</td><td>0.483</td><td>0.390</td></tr><tr><td>V 16.78</td><td>16.78 0.535 0.502</td><td>0.405 0.393</td></tr><tr><td rowspan="4">Two Levels</td><td>â</td><td>average</td><td>17.28</td><td>0.503</td></tr><tr><td rowspan="3">&gt;&gt;</td><td>J&gt;</td><td>18.33 0.577</td><td>0.370 0.336</td></tr><tr><td>â</td><td>17.43</td><td>0.372</td></tr><tr><td>average</td><td>0.540 17.68 0.540</td><td>0.359</td></tr><tr><td>â</td><td>â</td><td>â</td><td>18.78 0.595</td><td>0.317</td></tr></table>

Table 6. Ablation study on level interdependencies on LLFF dataset for 2 training views.

On average, adding a second level yields gains of +0.9 PSNR, +0.038 SSIM, and â0.034 LPIPS, indicating complementary interaction. Integrating all three levels further improves performance by +1.1 PSNR, +0.055 SSIM, and â0.042 LPIPS, surpassing the previous stage. The difference in incremental gains further indicates the presence of intrinsic coupling across levels, where improvements at one level reinforce and amplify others. This interdependency implies that the modules do not function in isolation but collaboratively contribute to a unified optimization process.

## 5. Conclusion

In this work, we present HeroGS, a tightly coupled hierarchical guidance framework that tames the ill-posed challenge of sparse-view 3D reconstruction. At the image level, pseudo-labels transform sparse supervision into pseudo-dense guidance, providing global regularization and rich structural cues for feature level. Building upon this, FADP at the feature level refines local geometry by injecting texture-aware high-frequency feature cues. The CPG at the parameter level further enforces geometric consistency via self-supervised co-pruning. These three levels form a hierarchical guidance framework that optimizes the overall Gaussian distributions. Extensive experiments on diverse benchmarks demonstrate that HeroGS sets a new state of the art for sparse-view reconstruction.

## Acknowledgments

This work was supported in part by the Key Deployment Program of the Chinese Academy of Sciences, China under Grant KGFZD-145-25-39, the National Natural Science Foundation of China under Grants 62272438, and Beijing Natural Science Foundation L25700.

## References

[1] Yanqi Bao, Yuxin Li, Jing Huo, Tianyu Ding, Xinyue Liang, Wenbin Li, and Yang Gao. Where and how: Mitigating confusion in neural radiance fields from sparse inputs. In Proceedings of the 31st ACM International Conference on Multimedia, pages 2180â2188, 2023. 2

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields, 2021. 1, 6

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5470â5479, 2022. 1

[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022. 2

[5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. ICCV, 2023. 1

[6] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14124â14133, 2021. 2

[7] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. Depth-regularized optimization for 3d gaussian splatting in few-shot images. arXiv preprint arXiv:2311.13398, 2023. 6

[8] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2(3):4, 2024. 2

[9] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5885â5894, 2021. 2

[10] Wonbong Jang and Lourdes Agapito. Codenerf: Disentangled neural radiance fields for object categories. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12949â12958, 2021. 2

[11] Youngkyoon Jang and Eduardo Perez-Pellitero. Comapgs: Â´ Covisibility map-based gaussian splatting for sparse novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 5

[12] Jorg Kaiser, Georgios Kopanas, Gernot Riegler, Margret Keuper, and Andreas Geiger. Minisplatting: Memory-efficient 3d gaussian splatting for real-time novel view synthesis on mobile devices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 6, 7

[14] Alexander Knapitsch, Jae-Hyun Park, Michael Zollhoefer, Timo Stich, and Christian Theobalt. Tanks and temples: Benchmarking large-scale reconstructions. In ACM Transactions on Graphics (TOG), pages 1â15. ACM, 2017. 5

[15] Min-Seop Kwak, Jiuhn Song, and Seungryong Kim. Geconerf: Few-shot neural radiance fields via geometric consistency. arXiv preprint arXiv:2301.10941, 2023. 2

[16] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20775â20785, 2024. 2, 6

[17] Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, and Yu-Lun Liu. Frugalnerf: Fast convergence for extreme few-shot novel view synthesis without learned priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11227â11238, 2025. 6

[18] Xing Liufu, Chaolei Tan, Xiaotong Lin, Yonggang Qi, Jinxuan Li, and Jian-Fang Hu. Sauge: Taming sam for uncertainty-aligned multi-granularity edge detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5766â5774, 2025. 4, 1

[19] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 2019. 5

[20] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[21] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480â5490, 2022. 2

[22] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. Coherentgs: Sparse novel view synthesis with coherent 3d gaussians. In European Conference on Computer Vision, pages 19â37. Springer, 2024. 2

[23] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian

splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21600â21609, 2025. 1, 6

[24] Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering. ACM Transactions on Graphics, 43 (4), 2024. 2

[25] Sara Sabour, Lily Goli, George Kopanas, Mark Matthews, Dmitry Lagun, Leonidas Guibas, Alec Jacobson, David J. Fleet, and Andrea Tagliasacchi. SpotLessSplats: Ignoring distractors in 3d gaussian splatting. arXiv:2406.20055, 2024. 1

[26] Wonyong Seo, Jihyong Oh, and Munchurl Kim. Bimvfi: Bidirectional motion field-guided frame interpolation for video with non-uniform motions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7244â7253, 2025. 4

[27] Yunji Seo, Young Sun Choi, Hyun Seung Son, and Youngjung Uh. Flod: Integrating flexible level of detail into 3d gaussian splatting for customizable rendering. ACM Transactions on Graphics (Proceedings of SIGGRAPH), 44 (4), 2025. 1

[28] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9065â9076, 2023. 6

[29] Guangyang Wu, Xin Tao, Changlin Li, Wenyi Wang, Xiaohong Liu, and Qingqing Zheng. Perception-oriented video frame interpolation via asymmetric blending. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2753â2762, 2024. 8

[30] Haolin Xiong. SparseGS: Real-time 360 sparse view synthesis using Gaussian splatting. University of California, Los Angeles, 2024. 2

[31] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8254â8263, 2023. 2

[32] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023. 2

[33] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578â4587, 2021. 2

[34] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1

[35] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision, pages 335â352. Springer, 2024. 2, 5, 6, 1

[36] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145â163. Springer, 2024. 1, 2, 4, 5, 6

# HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views

Supplementary Material

## 6. Implementation Details

We conduct training for 20k iterations and set $N _ { \mathrm { i t e r } } = 1 0 k$ for both datasets. Feature Adaptive Densification and Prune is applied at 10k iterations and patch-based density controlling strategy at 8k iterations for stability. Further more, we set $\lambda _ { \mathrm { l o w } } ~ = ~ 2 . 0$ and $\lambda _ { \mathrm { h i g h } } ~ = ~ 0 . 8$ . In addition, the parameter $\tau _ { \mathrm { s p a r s e } }$ is set to be the number of Gaussians in the top 90% of patches sorted by point density in descending order, while $\tau _ { \mathrm { h i g h } }$ to be the number of top 10%. Besides, the number of generated views is set to 4Ã, meaning that 3 generated views are inserted between every two training views. Frame interpolation is introduced after 2k iterations, with its loss evaluated at an interval of every 10 iterations. Starting from iteration 2000, it is applied for the first 100 iterations of every subsequent 200-iteration cycle, and disabled for the remaining 100 iterations. We first set $\lambda _ { g } = 0 . 0 7 5$ , which increases as the training iterations grows. SAUGE [18], a model based on SAM, is used to extract the edge of each training image. The co-pruning parameters are set following the configuration used in CoR-GS [35]. HeroGS is initialized with point clouds and precomputed camera poses from COLMAP.

Selection Module. The Gaussian fields are first trained for 2000 iterations without image-level guidance. After this stage, pseudo-label images are filtered based on their quality, and the high-quality ones are used for subsequent supervision. The selection metric is computed as:

$$
M = \lambda _ { 1 } \| I ^ { \alpha } - \hat { I } ^ { \alpha } \| + \lambda _ { 2 } \mathcal { L } _ { D \cdot S S I M } \operatorname { C o r } ( \hat { I } ^ { \alpha } , I ^ { \alpha } ) ,\tag{11}
$$

where $I ^ { \alpha }$ denotes a pseudo-label, and $\hat { I } ^ { \alpha }$ represents the corresponding rendered image. To avoid the influence of model instability during the early training phase, we progressively re-evaluate and re-select pseudo-labels according to the rendered outputs as training proceeds. If there are N pseudo-labeled images in total, $N / 2$ images with the smallest values of M (i.e., those closest to the rendered results) are selected as high-quality supervision for the subsequent training phase.

## 7. More Comparison Results

Ablation on Selection Module. Table 8 demonstrates that incorporating the Selection module yields consistent improvements across all evaluation metrics, this improvement verifies the effectiveness of filtering out low-quality pseudolabels, which stabilizes supervision and prevents the propagation of noise during optimization. In essence, the Selection module ensures that only reliable pseudo-labels contribute to training, leading to cleaner gradients and more robust convergence under sparse-view settings.

<!-- image-->  
Figure 7. Qualitative Comparison on Tanks for 3 training views. In large-scale dataset, 3DGS and DropGaussian struggles to maintain geometry and texture fidelity, exhibiting significant artifacts. FSGS and CoR-GS recover coarse structure yet still exhibit over-smoothed geometry and artifacts. In contrast, HeroGS reconstruct fine structures and high-frequency textures.

<table><tr><td>Methods</td><td>12 views 24 views PSNRâSSIMâLPIPS PSNRâSSIMâLPIPSâ</td></tr><tr><td>3DGS</td><td>18.44 0.521 0.385 23.22 0.730 0.234</td></tr><tr><td>FSGS</td><td>18.93 0.539 0.380 23.46 0.738 0.237</td></tr><tr><td>CoR-GS</td><td>19.59 0.578 0.374 23.39 0.727 0.272</td></tr><tr><td>DropGaussian</td><td>19.49 0.573 0.366 24.03 0.762 0.225</td></tr><tr><td>HeroGS</td><td>19.99 0.591 0.373 24.18 0.766 0.229</td></tr></table>

Table 7. Quantitative results on Mip-NeRF360 [3] with 12, 24 training views.
<table><tr><td>Settings</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>w/o Selection</td><td>21.19</td><td>0.736</td><td>0.190</td></tr><tr><td>All</td><td>21.30</td><td>0.739</td><td>0.189</td></tr></table>

Table 8. Ablation study on Selection module on LLFF dataset for 3 training views.

Depth Visualization. Fig. 9 presents qualitative comparisons of depth maps rendered from Gaussian fields reconstructed by 3DGS, DRGS, FSGS, and our proposed HeroGS framework. 3DGS suffers from severe artifacts and structural inconsistencies, particularly near object boundaries and occlusions. DRGS improves upon this with depth supervision, yet still exhibits oversmoothing and background bleeding, compromising geometric fidelity.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 8. Ablation study on LLFF dataset showing the impact of varying interpolation factor S on PSNR, SSIM, and LPIPS.

<!-- image-->  
Figure 9. Qualitative comparison of depth rendering quality. We visualize the depth maps rendered from the reconstructed 3D Gaussian fields of different methods and our proposed HeroGS framework. The visualizations show that our method produces more consistent and artifact-free depth results with finer geometry structure preservation.

FSGS enhances robustness via Proximity-guided Gaussian Unpooling, capturing global shapes more reliably but still fails to preserve finer geometric details, leading to blurry depth transitions. In contrast, HeroGS framework yields significantly sharper and cleaner depth maps. It better preserves thin structuresâsuch as plant stems and insect limbsâand maintains clear depth discontinuities. The consistent performance gains can be ascribed to the multi-level hierarchical guidance and the synergistic coupling across different supervision levels, which jointly refine Gaussian distributions for more robust reconstruction.

Mip-NeRF360. To further demonstrate the generalizability of HeroGS in unbounded real-world scenes, it is evaluated on the Mip-NeRF360 dataset using 12 and 24 input views with resolutions downsampled by 8Ã. As shown in Tab. 7, HeroGS consistently achieves the best performance across all metrics. In particular, it surpasses the second-best method by a notable margin of +0.5 dB in PSNR and +0.02 in SSIM on average, while maintaining competitive perceptual quality in terms of LPIPS. These results highlight that HeroGS can effectively handle complex illumination and large-scale geometry, showing superior robustness under sparse-input conditions and strong generalization to unbounded scene reconstruction.

## 8. Discussion

Another View of the Overall Framework. In our framework, a dense set of RGB images is synthesized as pseudolabels, which, together with the training views, jointly constrain the optimization of the Gaussian Splatting field. To mitigate the potential 3D geometric inconsistencies introduced by the primary pseudo-labels, we design a refinement pipeline that incorporates two synergistic submodules. First, the Feature-Adaptive Densification and Pruning (FADP) enhances features discriminability using training views through adaptive densification controlling, while stochastically pruning redundant textures to prevent overfitting to label noise. This process encourages finer representation of geometry and textures in high-frequency regions. Second, the Co-Pruned Geometry Consistency (CPG) adopts a freeze-and-co-prune strategy to suppress erroneous structures arising from pseudo-label supervision, thereby mitigating distortion artifacts and improving global consistency. Pseudo-labels generation and two refinement submodules form a coherent framework that not only enhances supervision quality but also preserves structural fidelity under sparse-view conditions.

Limitation and Future Work. Although our framework forms a hierarchical guidance across multiple levels, the current implementation only applies the Feature-Adaptive Densification and Pruning (FADP) once during the training phase. This design is motivated by the observation that a single round of FADP is sufficient to recover most highfrequency geometric details, whereas additional iterations only bring marginal gains in performance. In future work, we plan to extend this mechanism into a multi-stage adaptive refinement process, where density control can be repeatedly guided by the pseudo-labels from the image level. Such a recurrent optimization loop would enable dynamic feedback across levels, thereby further enhancing geometric precision and strengthening global consistency in sparseview reconstruction tasks.