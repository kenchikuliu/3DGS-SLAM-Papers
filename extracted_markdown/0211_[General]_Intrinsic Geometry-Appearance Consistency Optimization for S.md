# Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting

Kaiqiang Xiong 1,2 Rui Peng 4 Jiahao Wu 1,2 Zhanke Wang 1

Jie Liang 1,2 Xiaoyun Zheng 2 Feng Gao 5 Ronggang Wang 1,2,3

1Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology,

2Peng Cheng Laboratory 3Migu Culture Technology Co., Ltd 4 Alibaba Group 5School of Arts, Peking University

xiongkaiqiang@stu.pku.edu.cn rgwang@pkusz.edu.cn https://KaiqiangXiong.github.io/ICO-GS/

<!-- image-->  
Figure 1. Novel view synthesis quality under sparse inputs. Our method produces superior photometric quality with faithful texture recovery in challenging regions (e.g., leaf gaps in zoomed insets and weakly-textured surfaces), enabled by more accurate geometry with sharper structural boundaries, outperforming state-of-the-art sparse-view 3DGS approaches [14, 19, 31, 45, 47].

## Abstract

3D Gaussian Splatting (3DGS) represents scenes through primitives with coupled intrinsic properties: geometric attributes (position, covariance, opacity) and appearance attributes (view-dependent color). Faithful reconstruction requires intrinsic geometry-appearance consistency, where geometry accurately captures 3D structure while appearance reflects photometry. However, sparse observations lead to appearance overfitting and underconstrained geometry, causing severe novel-view artifacts. We present

ICO-GS (Intrinsic Geometry-Appearance Consistency Optimization for 3DGS), a principled framework that enforces this consistency through tightly coupled geometric regularization and appearance learning. Our approach first regularizes geometry via feature-based multi-view photometric constraints by employing pixel-wise top-k selection to handle occlusions and edge-aware smoothness to preserve sharp structures. Then appearance is coupled with geometry through cycle-consistency depth filtering, which identifies reliable regions to synthesize virtual views that propagate geometric correctness into appearance optimization.

Experiments on LLFF, DTU, and Blender show ICO-GS substantially improves geometry and photometry, consistently outperforming existing sparse-view baselines, particularly in challenging weakly-textured regions.

## 1. Introduction

Novel view synthesis (NVS) has witnessed remarkable progress with 3DGS [19], which represents scenes as collections of anisotropic 3D Gaussians to achieve photorealistic rendering at real-time speeds. While 3DGS achieves real-time rendering and high visual quality on densely captured scenes, its performance degrades dramatically under sparse-view settings commonly encountered in practical scenarios.

The degradation stems from a fundamental issue in 3DGS optimization: lack of intrinsic consistency between geometry and appearance. In 3DGS, each primitive is parameterized by coupled intrinsic properties, including geometric attributes (position Âµ, covariance Î£, opacity Î±) defining its spatial structure, and appearance attributes (view-dependent color c(d)) determining its photometric contribution. For faithful scene reconstruction, these properties must satisfy intrinsic consistency: geometry should accurately capture the underlying 3D structure, while appearance should coherently reflect surface photometry across viewpoints. However, sparse-view 3DGS violates this consistency. The standard 3DGS optimization relies on per-view photometric supervision that independently minimizes rendering loss for each training view. With limited observations, this independent supervision allows appearance to overfit individual views by compensating for geometric errors, while 3D geometry remains severely underconstrained due to lack of explicit multi-view regularization. This leads to internally inconsistent Gaussians that produce plausible renderings on training views but severe artifacts (floaters or blurriness) on novel views. These challenges are particularly pronounced in weakly-textured regions, where the absence of distinctive appearance cues further exacerbates geometric ambiguity.

Addressing this requires jointly solving two coupled challenges. First, how to effectively constrain geometry under sparse observations? Recent studies have employed pretrained depth estimation models [21, 37, 38, 47] to regularize 3DGS geometry, but such approaches suffer from scale ambiguity and noise introduced by imperfect pretrained models. Other methods [46, 47] rely on dense initialization, yet the initial geometric cues tend to be gradually forgotten during subsequent optimization. Second, how to couple accurate geometry with reliable appearance optimization to prevent overfitting? BinocularGS [14] builds virtual binocular pairs from rendered depth to enforce disparity consistency, but this approach relies on rendered depth whose reliability is not guaranteed, potentially propagating depth errors into appearance optimization and limiting rendering quality.

To this end, we propose ICO-GS (Intrinsic Geometry-Appearance Consistency Optimization for Sparse-view Gaussian Splatting), a principled framework that restores intrinsic consistency in sparse-view 3DGS through synergistic geometry-appearance optimization. Our key insight is that faithful geometry and appearance emerge from their mutual reinforcement: well-constrained geometry guides appearance to learn view-consistent photometry, while reliable appearance supervision in turn refines geometry. We realize this synergy through two coupled components.

Geometric regularization via multi-view photometric consistency. Under ideal conditions, a 3D point observed from multiple viewpoints should exhibit photometric consistency. However, illumination variations, occlusions, and monocularly visible regions violate this assumption. We therefore adopt feature-based multi-view consistency to regularize geometry, mitigating the impact of lighting and imaging variations. To handle occlusions prevalent in sparse-view settings, we employ pixel-wise top-k selection: for each pixel, we compute photometric errors across all source views and retain only the k most consistent ones, robustly filtering occluded or unreliable observations. For monocularly visible regions, we further incorporate an edge-aware depth smoothness term that enforces local coherence while preserving sharp geometric boundaries. These complementary constraints yield geometry that is both multi-view consistent and structurally sound, proving particularly effective in weakly-textured regions where photometric cues are scarce.

Geometry-guided appearance optimization via virtual view consistency. To couple geometry with appearance, we leverage the regularized depth to synthesize virtual views, propagating geometric constraints to appearance. However, naively using all depth estimates introduces noise from uncertain regions. We address this via cycleconsistency depth filtering: we project each pixelâs depth to source views and back-project to the original view, retaining only pixels with consistent spatial locations. The filtered depth then guides virtual view synthesis through depth-conditioned warping, providing geometric supervision that encourages appearance to capture view-consistent photometry rather than overfit individual observations.

Our contributions are summarized as follows:

â¢ We identify intrinsic consistency, defined as the coupled correctness of geometry and appearance, as a fundamental principle for sparse-view 3DGS, and reveal how its violation causes severe degradation in novel views.

â¢ We address geometry underconstraint through featurebased multi-view photometric regularization with pixelwise top-k consistency and edge-preserving smoothness.

â¢ We prevent appearance overfitting through geometryguided optimization that synthesizes virtual views from cycle-filtered depth, coupling geometric accuracy with photometric quality.

â¢ Extensive experiments validate that ICO-GS achieves state-of-the-art sparse-view novel view synthesis across diverse scenes, particularly on weakly-textured data, with 1.1 dB PSNR gains over prior arts on 3-view DTU scenarios.

## 2. Related Work

## 2.1. Novel View Synthesis

Novel View Synthesis (NVS) seeks to generate novel perspectives of a scene given a collection of input images. The introduction of Neural Radiance Fields (NeRF) [27] revolutionized this domain by representing scenes as volumetric radiance fields parameterized by multilayer perceptrons. Following this seminal work, substantial research efforts have been devoted to enhancing various aspects of NeRF, including photorealistic quality [1â3], rendering efficiency [6, 10, 12, 28], and generalization to complex scenarios such as dynamic environments [32, 33] and unconstrained captures [24]. Despite achieving impressive visual fidelity, NeRF-based approaches inherently suffer from prohibitive computational demands.To overcome these computational bottlenecks, 3D Gaussian Splatting (3DGS) [19] emerged as a promising alternative by representing scenes with explicit 3D Gaussian primitives rather than implicit neural functions, enabling real-time novel view synthesis with comparable quality. This breakthrough has inspired numerous follow-up works that extend 3DGS to various challenging scenarios [11, 20, 23, 25, 41, 43], including anti-aliasing [43], surface reconstruction [15, 22, 44], dynamic scene modeling [36], and memory-efficient representations [8]. Despite achieving impressive results on densely captured scenes, both NeRF and 3DGS exhibit severe degradation under sparse views, as their standard optimization relies on dense multi-view supervision to disambiguate geometry and appearance.

## 2.2. Sparse-view Novel View Synthesis

Neural rendering methods like NeRF and 3DGS optimize appearance independently per view through photometric losses, allowing appearance to overfit training observations while geometry remains underconstrained, violating the multi-view consistency principle essential for sparseview novel view synthesis. Early sparse-view NeRF methods introduced various regularization strategies, including semantic consistency via CLIP embeddings [16], patchbased geometric regularization [29], and frequency regularization [42] to constrain this ill-posed optimization.

For 3DGS, several works analyzed the overfitting issue in sparse-view 3DGS and include dual-model regularization [45] or dropout [7, 31, 40] to mitigate it. However, these methods lack analysis of the intrinsic geometry of the Gaussian primitives. Alternative approaches [9, 21, 30, 37, 47] integrate external monocular depth priors from pretrained estimators [5, 34], such priors are subject to scale ambiguity and can introduce erroneous or misleading noise. Some methods propose dense initialization [46] and binocular warping consistency [14], but they cannot guarantee that the geometry of the Gaussian primitives remains accurate throughout iterative optimization, which limits their effectiveness. Unlike prior methods that do not adequately enforce intrinsic consistency between accurate geometry and reliable appearance, we propose a framework grounded in intrinsic consistency optimization, based on the principle that geometry and appearance should be mutually correct and reinforcing.

## 3. Method

In this section, we introduce ICO-GS, a sparse-view reconstruction approach that enforces intrinsic consistency by coupling geometric regularization with appearance optimization. The overall framework is illustrated in Fig. 2. We first analyze the geometry-appearance discrepancy phenomenon under sparse-view settings (Sec. 3.1), then elaborate our robust geometric regularization (Sec. 3.2) and geometry-guided appearance optimization (Sec. 3.3). The complete optimization pipeline is described in Sec. 3.4.

## 3.1. Motivation

Despite success in dense-view reconstruction, 3DGS [19] severely overfits under sparse observations. We reveal a critical geometry-appearance discrepancy: as view count decreases, appearance quality remains superficially high on training views while geometric accuracy collapses, causing severe artifacts on novel views (Fig. 3).

We identify two underlying deficiencies:

Insufficient Geometric Constraints. In dense-view settings, each 3D point is observed by multiple cameras (typically 10+), providing strong multi-view constraints on Gaussian positions. However, with sparse views, each point is visible in only 2â3 views, leading to severe geometric ambiguity. Mathematically, the photometric loss in [19] only constrains the projected appearance of Gaussians, not their depth. Consequently, Gaussians can be placed at any position along the camera ray while maintaining zero photometric error. This depth ambiguity explains the noisy geometry in Fig. 3: without sufficient multi-view overlap, optimization lacks constraints to determine correct 3D positions.

Unreliable Appearance-Geometry Coupling. Moreover, the 3DGS representation does not inherently ensure geometry-appearance consistency. In principle, when a Gaussian is misplaced, the photometric loss should drive the optimizer to correct its position. However, we observe a problematic shortcut: instead of moving the Gaussian, the optimizer adjusts its color and opacity to compensate for geometric errors, which we term appearance compensation. As shown in Fig. 3, training views achieve high PSNR despite severely incorrect geometry (noisy depth), demonstrating that appearance parameters mask geometric mistakes. This becomes particularly severe under sparse views, where weak geometric constraints enable the model to overfit through appearance manipulation rather than learning correct 3D structure.

<!-- image-->

Figure 2. Framework of ICO-GS. Given sparse input views, we initialize 3D Gaussians and extract deep features. Our method enforces intrinsic geometry-appearance consistency through two synergistic components: (1) Robust Geometric Regularization. Source features are warped to reference views via rendered depth, establishing occlusion-aware multi-view constraints through: (a) Robust Multi-view Photometric Consistency that employs pixel-wise top-k selection for occlusion handling, and (b) Edge-aware Depth Smoothness that preserves sharp geometric structures. (2) Geometry-Guided Appearance Optimization. We leverage geometrically reliable regions identified by (c) Cycle Consistency Depth Filtering to synthesize virtual views, then apply (d) Virtual-view Photometric Consistency between synthesized and rendered images to propagate geometric correctness into appearance learning.  
<!-- image-->  
Figure 3. Geometry-appearance discrepancy under sparseview settings. From top to bottom: RGB on training views, depth on training views, and RGB on test views, rendered by 3D Gaussian Splatting [19] with varying training view densities. With decreasing views, training-view appearance (top) remains well-fitted, but depth quality (middle) collapses with noise and floaters due to insufficient multi-view constraints. This geometryappearance discrepancy leads to severe artifacts in novel-view rendering (bottom).

To address these deficiencies, we propose ICO-GS, which restores intrinsic consistency by coupling robust geometric regularization (Sec. 3.2) with geometry-guided appearance optimization (Sec. 3.3). These components work synergistically to promote both geometry and appearance.

## 3.2. Robust Geometric Regularization

As identified in Sec. 3.1, sparse-view 3DGS suffers from insufficient geometric constraints. BinocularGS [14] attempts to address this by enforcing stereo consistency via depthwarped virtual views. However, this approach suffers from a fundamental circular dependency: unreliable depth produces misaligned virtual views, which in turn provide corrupted supervision that further degrades geometry.

We break this loop via robust geometry regularization, which warps pixels between training views according to rendered depth and penalizes inconsistencies. To handle illumination variations and occlusions that undermine naÂ¨Ä±ve photometric matching, we introduce robust multi-view photometric consistency (Sec. 3.2.1) enhanced with edge-aware depth smoothness (Sec. 3.2.2).

## 3.2.1. Multi-view Photometric Consistency

Given n sparse training views $\{ I _ { i } \} _ { i = 0 } ^ { n - 1 }$ , our goal is to regularize the rendered Gaussian depths $\{ D _ { i } \} _ { i = 0 } ^ { n - 1 }$ via multiview photometric consistency. Take one reference view $I _ { 0 }$ and its corresponding source views $\{ I _ { j } \} _ { j = 1 } ^ { n - 1 }$ for example, we first render the depth map $D _ { 0 }$ via alpha-blending rendering. Then for each pixel $p$ in the reference image $I _ { 0 } ,$ its corresponding pixel $p _ { j } ^ { \prime }$ in the source images $\{ I _ { j } \} _ { j = 1 } ^ { n - 1 }$ can be computed via:

$$
p _ { j } ^ { \prime } = K T _ { 0  j } ( D _ { 0 } ( p ) \cdot K ^ { - 1 } p ) ,\tag{1}
$$

where $K , T$ denote the associated intrinsic and the relative transformation. We enforce inverse warp from the source views to reference views to acquire the reconstructed reference images $\{ I _ { j \to 0 } \} _ { j = 1 } ^ { n - 1 }$ with a binary validity mask $\{ M _ { j } \} _ { j = 1 } ^ { n - 1 }$ indicating valid projected pixels during warping. For ideal unoccluded Lambertian surfaces, pixels in $\{ I _ { 0 } , \{ I _ { j  0 } \} _ { j = 1 } ^ { n - 1 } \}$ should be photometrically consistent. The multi-view photometric loss enforces this:

$$
L = \frac { 1 } { n - 1 } \sum _ { j = 1 } ^ { n - 1 } \frac { \lVert ( I _ { j  0 } - I _ { 0 } ) \odot M _ { j } \rVert _ { 1 } } { \lVert M _ { j } \rVert _ { 1 } } .\tag{2}
$$

Illumination-robust Feature Matching. Equation 2 relies on RGB consistency, which is fragile to lighting variations, shadows, and specular reflections common in real scenes. To achieve robust geometric supervision, we replace it with feature-based matching using a frozen pre-trained feature network in [13]:

$$
L = \frac { 1 } { n - 1 } \sum _ { j = 1 } ^ { n - 1 } \frac { \lVert \frac { 1 } { 2 } ( 1 - \cos ( \mathcal { F } _ { 0 } , \mathcal { F } _ { j  0 } ) ) \odot M _ { j } \rVert _ { 1 } } { \lVert M _ { j } \rVert _ { 1 } } ,\tag{3}
$$

where $\mathcal { F } _ { 0 }$ and $\mathcal { F } _ { j  0 }$ are features extracted from the reference view and features warped from source views $j$ to the reference view 0. Since features are computed once during preprocessing and remain frozen during training, this incurs negligible computational overhead while significantly improving robustness to illumination changes.

Occlusion-aware Photometric Consistency. Occlusions cause photometric consistency to fail even with accurate depth. We address this via pixel-wise top-k selection, which adaptively chooses the most reliable correspondences from visible source views. For each reference pixel p, we identify the top-k most consistent correspondences across all warped source features $\{ \mathcal { F } _ { j \to 0 } \} _ { j = 1 } ^ { n - 1 }$ :

$$
\mathcal { T } _ { k } ( \boldsymbol { p } ) = \underset { S \subset \{ 1 , \ldots , n - 1 \} } { \arg \operatorname* { m i n } } \sum _ { j \in S } \| \frac { 1 } { 2 } ( 1 - \cos ( \mathcal { F } _ { 0 } ( \boldsymbol { p } ) , \mathcal { F } _ { j  0 } ( \boldsymbol { p } ) ) ) \| _ { 1 }\tag{4}
$$

This reformulates Eq. (3) as an adaptive aggregation over $\tau _ { k } ( p )$

$$
\mathcal { L } _ { \mathrm { m p c } } ^ { \mathrm { F e a } } ( p ) = \frac { 1 } { k } \sum _ { j \in \mathcal { T } _ { k } ( p ) } \| \frac { 1 } { 2 } ( 1 - \cos ( \mathcal { F } _ { 0 } ( p ) , \mathcal { F } _ { j  0 } ( p ) ) ) \| _ { 1 } .\tag{5}
$$

This pixel-wise selection naturally handles spatiallyvarying occlusions: for pixels occluded in half the views, the remaining visible views still provide valid supervision. Setting $k = \lceil ( n - 1 ) / 2 \rceil$ balances coverage and outlier rejection.

## 3.2.2. Edge-aware Depth Smoothness.

In regions visible from only one views, multi-view photometric consistency fails to provide sufficient geometric constraints. We therefore regularize these under-constrained areas with edge-aware depth smoothness:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { p } \| \nabla D _ { 0 } ( p ) \| _ { 1 } \cdot \exp \left( - \alpha \left\| \nabla I _ { 0 } ( p ) \right\| _ { 1 } \right) ,\tag{6}
$$

where $\nabla D$ and $\nabla I$ denote the depth and image gradients, respectively, and $\alpha = 1$ controls edge sensitivity. This encourages smooth depth in textureless regions while preserving discontinuities at object boundaries.

By enforcing geometric consistency through this robust regularization, our method significantly improves novel view synthesis quality in weakly-textured regionsâ areas where 3D Gaussian Splatting typically struggles due to insufficient photometric constraints. The geometric regularization provides reliable supervision even when appearance cues are ambiguous.

## 3.3. Geometry-guided Appearance Optimization

We propose to leverage regularized geometry for appearance optimization through virtual-view sampling. Existing methods [14, 21, 39, 47] render virtual novel views to mitigate texture under-constraint in sparse settings, enforcing regularization via monocular or binocular depth consistency. Yet they are limited by: scale ambiguity in monocular depth [21, 47], noise in MVS priors [39], and restricted diversity with inaccurate rendered depth [14].

We instead utilize our regularized Gaussian geometry to enable flexible, reliable virtual-view supervision. Our approach comprises: cycle-consistency filtering to identify valid depth regions (Sec. 3.3.1), and appearance supervision via virtual-view photometric consistency over these validated regions (Sec. 3.3.2).

## 3.3.1. Cycle Consistency Depth Filtering

To ensure geometry reliability, we validate rendered depth through cycle-consistency filtering before synthesizing virtual views. Given a reference view $I _ { 0 } ,$ source views $\{ I _ { j } \} _ { j = 1 } ^ { n - 1 }$ , camera intrinsic K, relative transformations $\{ T _ { 0 \to j } \} _ { j = 1 } ^ { n - 1 }$ , and rendered depth maps $\{ D _ { i } \} _ { i = 0 } ^ { n - 1 }$ from Gaussian splatting, we perform forward-backward warping. For each pixel $p$ in $I _ { 0 } ,$ , we first forward warp to source view $I _ { j }$ using depth $D _ { 0 } ( p )$ to obtain projected pixel $p _ { j } ^ { \prime }$ (see Eq. (1)). We then backward warp $p _ { j } ^ { \prime }$ to $I _ { 0 }$ using source depth $D _ { j } ( p _ { j } ^ { \prime } )$ to obtain reprojected pixel $p _ { j } ^ { \prime \prime }$ and depth $\tilde { D } _ { j } ( p )$

Table 1. Quantitative comparisons on LLFF [26] dataset under sparse view settings.
<table><tr><td rowspan="2">Methods</td><td colspan="3">PSNRâ</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td></tr><tr><td>3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td></tr><tr><td>DietNeRF [16]</td><td>14.94</td><td>21.75</td><td>24.28</td><td>0.370</td><td>0.717</td><td>0.801</td><td>0.496</td><td>0.248</td><td>0.183</td></tr><tr><td>RegNeRF [29]</td><td>19.08</td><td>23.10</td><td>24.86</td><td>0.587</td><td>0.760</td><td>0.820</td><td>0.336</td><td>0.206</td><td>0.161</td></tr><tr><td>FreeNeRF [42]</td><td>19.63</td><td>23.73</td><td>25.13</td><td>0.612</td><td>0.779</td><td>0.827</td><td>0.308</td><td>0.195</td><td>0.160</td></tr><tr><td>SparseNeRF [35]</td><td>19.86</td><td>23.26</td><td>24.27</td><td>0.714</td><td>0.741</td><td>0.781</td><td>0.243</td><td>0.235</td><td>0.228</td></tr><tr><td>3DGS [19]</td><td>15.52</td><td>19.45</td><td>21.13</td><td>0.405</td><td>0.627</td><td>0.715</td><td>0.408</td><td>0.268</td><td>0.214</td></tr><tr><td> SS [47]</td><td>20.31</td><td>24.20</td><td>25.32</td><td>0.652</td><td>0.811</td><td>0.856</td><td>0.288</td><td>0.173</td><td>0.136</td></tr><tr><td>DNGaussian [21]</td><td>19.12</td><td>22.18</td><td>23.17</td><td>0.591</td><td>0.755</td><td>0.788</td><td>0.294</td><td>0.198</td><td>0.180</td></tr><tr><td>CoR-GS [45]</td><td>20.45</td><td>24.49</td><td>26.06</td><td>0.712</td><td>0.837</td><td>0.874</td><td>0.196</td><td>0.115</td><td>0.089</td></tr><tr><td>BinocularGS [14]</td><td>21.44</td><td>24.87</td><td>26.17</td><td>0.751</td><td>0.845</td><td>0.877</td><td>0.168</td><td>0.106</td><td>0.090</td></tr><tr><td>DropGaussians [31]</td><td>20.76</td><td>24.74</td><td>26.21</td><td>0.713</td><td>0.837</td><td>0.874</td><td>0.200</td><td>0.117</td><td>0.088</td></tr><tr><td>NexusGS [46]</td><td>21.07</td><td>-</td><td>-</td><td>0.738</td><td>-</td><td>-</td><td>0.177</td><td>-</td><td>-</td></tr><tr><td>ComapGS [17]</td><td>21.11</td><td>25.20</td><td>26.73</td><td>0.747</td><td>0.854</td><td>0.886</td><td>0.182</td><td>0.108</td><td>0.082</td></tr><tr><td>Ours</td><td>22.20</td><td>25.37</td><td>26.45</td><td>0.778</td><td>0.856</td><td>0.881</td><td>0.157</td><td>0.109</td><td>0.096</td></tr></table>

$$
p _ { j } ^ { \prime \prime } = K T _ { 0 \to j } ^ { - 1 } D _ { j } ( p _ { j } ^ { \prime } ) K ^ { - 1 } p _ { j } ^ { \prime } .\tag{7}
$$

The depth error between original and reprojected depth measures geometric consistency:

$$
e _ { j } ( p ) = \Big | D _ { 0 } ( p ) - \tilde { D } _ { j } ( p ) \Big | .\tag{8}
$$

A pixel is reliable if its depth error falls below threshold $\tau _ { d }$ for at least m of the n â 1 source views:

$$
\mathcal { M } _ { \mathrm { r e l i a b l e } } ( p ) = \mathbb { I } \left[ \sum _ { j = 1 } ^ { n - 1 } \mathbb { I } [ e _ { j } ( p ) < \tau _ { d } ] \geq m \right] ,\tag{9}
$$

where $m = \lceil ( n - 1 ) / 2 \rceil$ ensures consistency with at least half the sources, and $\tau _ { d } = 0 . 0 1 \cdot \operatorname* { m a x } ( D _ { 0 } )$ . This binary mask $\mathcal { M } _ { \mathrm { r e l i a b l e } }$ identifies regions where rendered depth $D _ { 0 }$ is validated by cycle consistency, ensuring subsequent warping produces views aligned with true scene structure.

## 3.3.2. Virtual-view Photometric Consistency

With reliable depth identified, we propagate accurate geometry to unseen appearances via virtual-view photometric consistency. Unlike prior stereo-pair approaches [14], we sample virtual poses $\{ \mathcal { P } _ { v } \} _ { v = 1 } ^ { N _ { v } }$ across a wider range: for each reference position x, we randomly sample within a sphere of radius r, providing sufficient viewpoint diversity.

For each virtual view, we forward warp (Eq. (1)) pixels from all training images $\{ I _ { i } \} _ { i = 0 } ^ { n - 1 }$ using masked depths $\{ \mathcal { M } _ { \mathrm { r e l i a b l e } } ^ { i } \odot D _ { i } \} _ { i = 0 } ^ { n - 1 }$ to synthesize virtual image $\mathcal { T } _ { v }$ with validity mask $M _ { v }$ , excluding unreliable regions to prevent geometric errors from contaminating supervision.

Vitural-view Photometric Consistency Loss. The synthesized virtual images are incorporated into training for appearance optimization. For each virtual view, we render the Gaussian to acquire the rendered virtual image $\mathcal { I } _ { v } ^ { R }$ from the virtual pose $\mathcal { P } _ { v }$ and enforce photometric consistency on valid pixels:

Table 2. Quantitative comparisons on DTU dataset under sparse view settings.
<table><tr><td rowspan="2">Methods</td><td colspan="3">PSNRâ</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td></tr><tr><td>| 3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td></tr><tr><td>DietNeRF [16]</td><td>11.85</td><td>20.63</td><td>23.83</td><td>0.633</td><td>0.778</td><td>0.823</td><td>0.214</td><td>0.201</td><td>0.173</td></tr><tr><td>RegNeRF [29]</td><td>18.89</td><td>22.20</td><td>24.93</td><td>0.745</td><td>0.841</td><td>0.884</td><td>0.190</td><td>0.117</td><td>0.089</td></tr><tr><td>FreeNeRF [42]</td><td>19.52</td><td>23.25</td><td>25.38</td><td>0.787</td><td>0.844</td><td>0.888</td><td>0.173</td><td>0.131</td><td>0.102</td></tr><tr><td>SparseNeRF [35]</td><td>19.47</td><td>-</td><td>-</td><td>0.829</td><td>-</td><td>-</td><td>0.183</td><td>-</td><td>-</td></tr><tr><td>3DGS [19]</td><td>10.99</td><td>20.33</td><td>22.90</td><td>0.585</td><td>0.776</td><td>0.816</td><td>0.313</td><td>0.223</td><td>0.173</td></tr><tr><td>FSG S [47]</td><td>17.34</td><td>21.55</td><td>24.33</td><td>0.818</td><td>0.880</td><td>0.911</td><td>0.169</td><td>0.127</td><td>0.106</td></tr><tr><td>DNGaussian [21]</td><td>18.91</td><td>22.10</td><td>23.94</td><td>0.790</td><td>0.851</td><td>0.887</td><td>0.176</td><td>0.148</td><td>0.131</td></tr><tr><td>CoR-GS [45]</td><td>19.21</td><td>24.51</td><td>27.18</td><td>0.853</td><td>0.917</td><td>0.947</td><td>0.119</td><td>0.068</td><td>0.045</td></tr><tr><td>BinocularGS [14]</td><td>20.71</td><td>24.31</td><td>26.70</td><td>0.862</td><td>0.917</td><td>0.947</td><td>0.111</td><td>0.073</td><td>0.052</td></tr><tr><td>NexusGS [46]</td><td>20.21</td><td>-</td><td>-</td><td>0.869</td><td>.</td><td>.</td><td>0.102</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>21.77</td><td>25.09</td><td>27.19</td><td>0.888</td><td>0.928</td><td>0.953</td><td>0.092</td><td>0.064</td><td>0.045</td></tr></table>

$$
L _ { a p p } = \sum _ { p \in \mathcal { M } _ { \mathrm { v } } } \left. \mathcal { T } _ { v } ( p ) - \mathcal { T } _ { v } ^ { R } ( p ) \right. _ { 1 } .\tag{10}
$$

This serves two purposes: it provides additional observations to optimize appearance in unseen views and prevent overfitting, and conversely, it constrains geometry through supervision from novel viewpoints. Critically, because virtual-view images are synthesized from reliability-filtered depth, they provide clean supervision without the geometric distortions introduced by prior methods relying on unreliable depth predictions.

## 3.4. Overall Pipeline

We integrate geometric regularization and geometry-guided appearance optimization via curriculum learning [4].

Training Objective. The complete loss combines four terms:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { 3 D G S } } + { \mathcal { L } } _ { \mathrm { c o n s i s } } + \lambda _ { \mathrm { m p c } } { \mathcal { L } } _ { \mathrm { m p c } } ^ { \mathrm { F e a } } } \\ & { ~ + \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } } + \lambda _ { \mathrm { a p p } } { \mathcal { L } } _ { \mathrm { a p p } } , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { 3 D G S } }$ is the base photometric loss, $\mathcal { L } _ { \mathrm { c o n s i s } }$ enforces binocular consistency inherited from the baseline [14], ${ \mathcal { L } } _ { \mathrm { m p c } } ^ { \mathrm { F e a } } ~ + ~ \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } }$ enforces geometry regularization (Sec. 3.2), and $\mathcal { L } _ { \mathrm { a p p } }$ optimizes appearance in unseen views (Sec. 3.3).

Optimization. We employ a three-stage curriculum: Stage 1 optimizes $\mathcal { L } _ { \mathrm { 3 D G S } }$ to establish coarse geometry; Stage 2 activates geometric regularization $\lambda _ { \mathrm { m p c } } \mathcal { L } _ { \mathrm { m p c } } ^ { \mathrm { F e a } } + \lambda _ { \mathrm { s m o o t h } } \mathcal { L }$ smooth; Stage 3 adds appearance supervision $\dot { \lambda _ { \mathrm { a p p } } } \mathcal { L } _ { \mathrm { a p p } }$ from virtual views. This staged approach ensures stable convergence.

## 4. Experiments

## 4.1. Datasets

We conduct experiments on three standard benchmarks: LLFF [26] for forward-facing scenes, DTU [18], a challenging dataset with extensive weakly-textured regions, for object-centric captures, and Blender [27] for 360Â° objectcentric scenes. Following [14, 17, 45], we use 3, 6, 9 training views for LLFF/DTU and 8 views for Blender scenes. Input images are downsampled by 8Ã (LLFF), 4Ã (DTU), and 2Ã (Blender) to balance quality and efficiency, consistent with prior work [14, 17, 45].

<!-- image-->  
Figure 4. Visual comparison on LLFF [26] dataset.

<!-- image-->  
Figure 5. Visual comparison on DTU [18] dataset.

## 4.2. Implementation Details

We build upon the BinocularGS framework [14] with dense point cloud initialization for LLFF and DTU, and random initialization for Blender. We train for 30k iterations on LLFF and DTU datasets, and 7k iterations on Blender, consistent with the baseline. Geometric regularization begins at iteration 20k for LLFF/DTU and 4k for Blender, while geometry-guided appearance optimization starts at iteration 25k and 5k respectively. Both operate at every iteration once activated. All experiments run on the NVIDIA L40s

Table 3. Quantitative comparison on Blender [27] for 8 views.
<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>DietNeRF [16]</td><td>22.50</td><td>0.823</td><td>0.124</td></tr><tr><td>RegNeRF [29]</td><td>23.86</td><td>0.852</td><td>0.105</td></tr><tr><td>FreeNeRF [42]</td><td>24.26</td><td>0.883</td><td>0.098</td></tr><tr><td>SparseNeRF [35]</td><td>22.41</td><td>0.861</td><td>0.199</td></tr><tr><td>3DGS [19]</td><td>22.23</td><td>0.858</td><td>0.114</td></tr><tr><td>FSGS [47]</td><td>22.76</td><td>0.829</td><td>0.157</td></tr><tr><td>DNGaussian [21]</td><td>24.31</td><td>0.886</td><td>0.088</td></tr><tr><td>CoR-GS [45]</td><td>23.98</td><td>0.891</td><td>0.094</td></tr><tr><td>BinocularGS [14]</td><td>24.71</td><td>0.872</td><td>0.101</td></tr><tr><td>NexusGS [46]</td><td>24.37</td><td>0.893</td><td>0.087</td></tr><tr><td>DropGaussians [31]</td><td>25.42</td><td>0.888</td><td>0.089</td></tr><tr><td>Ours</td><td>25.56</td><td>0.884</td><td>0.100</td></tr></table>

GPU. Loss weights are set to $\lambda _ { \mathrm { m p c } } = 0 . 1 , \lambda _ { \mathrm { s m o o t h } } = 0 . 0 1$ and $\lambda _ { \mathrm { a p p } } = 1 . 0$ . All warping operations are accelerated using batched parallel processing in PyTorch. We report average results over three independent runs with different seeds.

<!-- image-->  
Figure 6. Visual comparison on Blender [27] dataset.

## 4.3. Baselines

We compare against state-of-the-art methods from both NeRF and 3D Gaussian Splatting (3DGS) paradigms. For NeRF-based methods, we include DietNeRF [16], Reg-NeRF [29], FreeNeRF [42], and SparseNeRF [35]. For 3DGS-based methods, we evaluate vanilla 3DGS [19] and sparse-view variants: FSGS [47], DNGaussian [21], CoR-GS [45], BinocularGS [14], DropGaussians [31], NexusGS [46], and ComapGS [17].

## 4.4. Comparisons

Results on LLFF [26] . Tab. 1 presents quantitative results on the LLFF dataset. Our method achieves state-of-the-art performance across all view settings: +0.76 dB improvement at 3 views, +0.17 dB over ComapGS [17] at 6 views, and competitive performance at 9 views (+0.24 dB over the third-best). Fig. 4 demonstrates fewer artifacts in fine details and sharper object boundaries. Notably, our depth maps exhibit significantly clearer geometric edges in both background textures and foreground structures.

Results on DTU [18]. Tab. 2 shows consistent improvements across all settings: +1.06 dB, +0.58 dB, and +0.01 dB over baselines at 3, 6, and 9 views, respectively. Fig. 5 shows our method renders clearer textures in regions (red boxes) via occlusion-aware photometric consistency (Sec. 3.2), while depth maps reveal sharper boundaries and finer details (white boxes).

Results on Blender [27]. For 360Â° object-centric evaluation, Tab. 3 reports results on the Blender dataset [27] under 8-view setting. Our method achieves the best PSNR. While SSIM and LPIPS are slightly lower than some methods, this trade-off reflects our prioritization of geometric accuracy over perceptual optimization. Fig. 6 shows fine-grained textures and geometrically accurate structures, validating that our geometric regularization (Sec. 3.2) and geometryguided appearance optimization (Sec. 3.3) effectively preserve structural fidelity critical for 3D reconstruction.

Table 4. Ablation studies on LLFF [26] and DTU [18].
<table><tr><td rowspan="2">LFepa</td><td rowspan="2"> ${ \mathcal { L } } _ { \mathrm { { s m o o t h } } }$ </td><td rowspan="2">CCDF</td><td rowspan="2"> $\mathcal { L } _ { \mathrm { a p p } }$ </td><td colspan="3">LLFF(3-view)</td><td colspan="3">DTU(3-view)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã &gt;</td><td>21.44</td><td>0.751</td><td>0.168</td><td>20.71</td><td>0.862</td><td>0.111</td></tr><tr><td>Ã</td><td>â</td><td>â</td><td></td><td>21.82</td><td>0.761</td><td>0.165</td><td>21.31</td><td>0.883</td><td>0.099</td></tr><tr><td>â</td><td>Ã</td><td>â</td><td>â</td><td>22.16</td><td>0.775</td><td>0.159</td><td>21.67</td><td>0.879</td><td>0.093</td></tr><tr><td>â</td><td>â</td><td>Ã</td><td>â</td><td>21.86</td><td>0.767</td><td>0.162</td><td>21.25</td><td>0.875</td><td>0.104</td></tr><tr><td>â</td><td>â</td><td>â</td><td>Ã</td><td>21.79</td><td>0.763</td><td>0.164</td><td>21.20</td><td>0.870</td><td>0.105</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>22.20</td><td>0.778</td><td>0.157</td><td>21.77</td><td>0.888</td><td>0.092</td></tr></table>

## 4.5. Analysis

Ablation Study. Tab. 4 validates each componentâs contribution on LLFF (3 views) and DTU (3 views) by removing: robust multi-view photometric consistency Loss $\mathcal { L } _ { \mathrm { m p c } } ^ { \mathrm { F e a } }$ (introduced in Eq. (5)), depth smoothness Loss $\mathcal { L } _ { \mathrm { s m o o t h } }$ (introduced in Eq. (6)), cycle consistency depth filtering (CCDF, introduced in Eq. (9)), and virtual-view photometric consistency Loss $\mathcal { L } _ { \mathbf { a p p } }$ (introduced in Eq. (10)).All experiments use BinocularGS [14] as the baseline model.

Removing $\mathcal { L } _ { \mathrm { m p c } } ^ { \mathrm { F e a } }$ causes severe degradation (LLFF: -0.38 dB, DTU: -0.46 dB). As shown in Fig. 7, this leads to noticeable blur and noise in both RGB and depth, confirming that robust geometric regularization is essential. Lsmooth contributes -0.10 dB on DTU and prevents loss of structural information in depth maps (Fig. 7), effectively mitigating artifacts from depth discontinuities. CCDF improves virtual view synthesis (DTU: -0.52 dB, LLFF: -0.34 dB without it). Without CCDF, visible rendering artifacts appear (Fig. 7), indicating insufficient appearance-geometry consistency. $\mathcal { L } _ { \mathbf { a p p } }$ contributes -0.57 dB on DTU and maintains image sharpness (Fig. 7), demonstrating that appearance optimization benefits from explicit geometric guidance. Our full model integrates all components to achieve superior geometric fidelity with clear structures and crisp boundaries.

Limitations. Our method assumes view-independent appearance during virtual view synthesis. In regions with strong view-dependent effects (e.g., specular highlights and reflections), the warped appearance may provide incorrect supervision. However, due to view sparsity, prior methods also struggle in these regions. We refer readers to the supplementary material for visual examples. Despite our accelerated implementation, it requires additional computation, resulting in a training time approximately 1.5Ã that of the baseline. The memory overhead is modest ( 0.3 GB on 3- view LLFF scenes). We believe these costs are acceptable given the improvements in rendering quality and geometric accuracy under sparse views.

## 5. Conclusion

We have presented a novel framework for sparse-view novel view synthesis that effectively addresses the geometryappearance discrepancy problem in 3D Gaussian Splatting.

<!-- image-->  
Figure 7. Visualization of ablation study results using 3-views.

Through robust geometric regularization, geometry-guided appearance optimization, our method achieves joint optimization of geometry and appearance without relying on external depth priors. Extensive experiments demonstrate state-of-the-art performance with reduced floater artifacts and improved geometric accuracy.

## References

[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 3

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022.

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697â19705, 2023. 3

[4] Yoshua Bengio, JerÂ´ ome Louradour, Ronan Collobert, and Ja- Ë son Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41â48, 2009. 6

[5] Chenjie Cao, Xinlin Ren, and Yanwei Fu. Mvsformer: Multi-view stereo by learning robust image features and temperature-based depth. arXiv preprint arXiv:2208.02541, 2022. 3

[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pages 333â350. Springer, 2022. 3

[7] Kangjie Chen, Yingji Zhong, Zhihao Li, Jiaqi Lin, Youyu Chen, Minghan Qin, and Haoqian Wang. Quantifying and alleviating co-adaptation in sparse-view 3d gaussian splatting. arXiv preprint arXiv:2508.12720, 2025. 3

[8] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, pages 422â438. Springer, 2024. 3

[9] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. Depth-regularized optimization for 3d gaussian splatting in few-shot images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 811â820, 2024. 3

[10] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501â5510, 2022. 3

[11] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796â20805, 2024. 3

[12] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14346â 14355, 2021. 3

[13] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2495â2504, 2020. 5

[14] Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. Binocular-guided 3d gaussian splatting with view consistency for sparse view synthesis. Advances in Neural Information Processing Systems, 37:68595â68621, 2024. 1, 2, 3, 4, 5, 6, 7, 8

[15] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024. 3

[16] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5885â5894, 2021. 3, 6, 7, 8

[17] Youngkyoon Jang and Eduardo Perez-Pellitero. Comapgs: Â´ Covisibility map-based gaussian splatting for sparse novel view synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26779â26788, 2025. 6, 7, 8

[18] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik AanÃ¦s. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 406â413, 2014. 6, 7, 8

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3, 4, 6, 7, 8

[20] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 505â515, 2024. 3

[21] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d

gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20775â20785, 2024. 2, 3, 5, 6, 7, 8

[22] Kunyi Li, Michael Niemeyer, Zeyu Chen, Nassir Navab, and Federico Tombari. Monogsdf: Exploring monocular geometric cues for gaussian splatting-guided implicit surface reconstruction. arXiv preprint arXiv:2411.16898, 2024. 3

[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 3

[24] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210â7219, 2021. 3

[25] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024. 3

[26] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1â14, 2019. 6, 7, 8

[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 3, 7, 8

[28] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 3

[29] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480â5490, 2022. 3, 6, 7, 8

[30] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. Coherentgs: Sparse novel view synthesis with coherent 3d gaussians. In European Conference on Computer Vision, pages 19â37. Springer, 2024. 3

[31] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21600â21609, 2025. 1, 3, 6, 7, 8

[32] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5865â5874, 2021. 3

[33] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10318â10327, 2021. 3

[34] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-Â´ sion transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179â12188, 2021. 3

[35] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9065â9076, 2023. 6, 7, 8

[36] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024. 3

[37] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360 {\deg} sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206, 2023. 2, 3

[38] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453â16463, 2025. 2

[39] Wangze Xu, Huachen Gao, Shihe Shen, Rui Peng, Jianbo Jiao, and Ronggang Wang. Mvpgs: Excavating multi-view priors for gaussian splatting from sparse input views. In European Conference on Computer Vision, pages 203â220. Springer, 2024. 5

[40] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao, Li Li, and Yulan Guo. Dropoutgs: Dropping out gaussians for better sparse-view rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 701â710, 2025. 3

[41] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pages 156â173. Springer, 2024. 3

[42] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8254â8263, 2023. 3, 6, 7, 8

[43] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 3

[44] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (ToG), 43(6):1â13, 2024. 3

[45] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision, pages 335â352. Springer, 2024. 1, 3, 6, 7, 8

[46] Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun, Junyu Dong, Huaidong Zhang, and Yong Du. Nexusgs: Sparse view synthesis with epipolar depth priors in 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26800â26809, 2025. 2, 3, 6, 7, 8

[47] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145â163. Springer, 2024. 1, 2, 3, 5, 6, 7, 8