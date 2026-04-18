# SplitGaussian: Reconstructing Dynamic Scenes via Visual Geometry Decomposition

Jiahui Li1, Shengeng Tang2, Jingxuan He2,

Gang Huang1, Zhangye Wang1, Yantao Pan3, Lechao Cheng2,3\*

1Zhejiang University 2Hefei University of Technology 3KAIYANG Laboratory, Chery

## Abstract

Reconstructing dynamic 3D scenes from monocular video remains fundamentally challenging due to the need to jointly infer motion, structure, and appearance from limited observations. Existing dynamic scene reconstruction methods based on Gaussian Splatting often entangle static and dynamic elements in a shared representation, leading to motion leakage, geometric distortions, and temporal flickering. We identify that the root cause lies in the coupled modeling of geometry and appearance across time, which hampers both stability and interpretability. To address this, we propose SplitGaussian, a novel framework that explicitly decomposes scene representations into static and dynamic components. By decoupling motion modeling from background geometry and allowing only the dynamic branch to deform over time, our method prevents motion artifacts in static regions while supporting view- and time-dependent appearance refinement. This disentangled design not only enhances temporal consistency and reconstruction fidelity but also accelerates convergence. Extensive experiments demonstrate that SplitGaussian outperforms prior state-of-the-art methods in rendering quality, geometric stability, and motion separation.

## Introduction

Reconstructing dynamic 3D scenes from monocular video remains a core challenge in computer vision, with farreaching applications in virtual reality, free-viewpoint rendering, and autonomous perception. These scenes present non-rigid motion, occlusions, and appearance variation, demanding joint inference of structure, motion, and camera pose from limited visual cues. Traditional multi-view stereo and depth sensors offer stronger constraints but restrict flexibility. Methods like NR-NeRF (Tretschk et al. 2021) introduce a canonical volume plus deformation field to enable dynamic reconstruction from monocular video, but they require expensive per-scene ray-based optimization and converge slowly.

Compared to implicit volumetric fields, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) offers explicit and compact representations, enabling faster optimization and realtime rendering. Early attempts at dynamic reconstruction with 3DGS treat each frame independently by reconstructing a separate set of Gaussians per frame (Luiten et al. 2024a), which fails to establish temporal consistency and leads to redundant or unstable representations. To this end, existing cutting-edge dynamic extensionsâsuch as Deformable 3DGS (Yang et al. 2023b)âuse a unified deformation network across static and dynamic regions, often disturbing static structures and introducing temporal artifacts. When a single deformation field is applied uniformly across both dynamic and static regions, it tends to propagate motion artifacts into static areas, leading to geometric distortions (e.g., rigid structures slightly shifted or warped) and appearance inconsistencies (e.g., temporal texture flicker or color drift). This issue also persists in prior works (Wu et al. 2024; Lu et al. 2025; Kwak et al. 2025; Katsumata, Vo, and Nakayama 2024), where static scene elements exhibit lingering artifacts despite motion modeling.

Let us revisit the Gaussian Splatting process, where each primitive jointly encodes visual appearance (e.g., color defined by spherical harmonic coefficients, opacity) and geometry (e.g., position, rotation, and scaling). We argue that such joint modeling underlies many common artifacts in dynamic reconstruction, such as motion leakage into static regions and temporal inconsistencies. Existing methods like DeGauss (Wang et al. 2025) (depth-aware compositing), DynaSplat (Deng et al. 2025) (optical flow guidance), and GauFre (Liang et al. 2025) (multi-stream processing) attempt to address dynamics, but all suffer from effectively disentangling dynamic and static regions from the underlying meta-representations.

Motivated by this, we propose to decompose the visual geometry within the gaussian representation to explicitly model dynamic and static regions. Specifically, the static part maintains fixed geometry but allows appearance to vary over time, while the dynamic part models time-varying geometry and appearance through a deformation network conditioned on shared spatiotemporal encodings. This design effectively decouples motion modeling from background representation, yielding more robust and interpretable reconstructions. We further improve reconstruction by applying visibility-driven pruning to remove low-contribution static Gaussians and introducing a depth-aware pretraining phase for better geometric initialization and depth consistency. Our approach yields more stable optimization, temporally consistent reconstructions, and superior visual quality compared to existing methods that use a single deformation field for the entire scene. We summarize the key contributions as follows:

â¢ We introduce an explicit decomposition of Gaussian primitives into static and dynamic components, enabling disentangled modeling of geometry and appearance to improve reconstruction stability.

â¢ We orchestrate a unified framework with shared spatiotemporal encoding, dedicated deformation network, and visibility-driven pruning for efficient and coherent dynamic scene reconstruction from monocular video.

â¢ Extensive experiments demonstrate that our method achieves superior performance in reducing geometric distortions and appearance flickering, outperforming existing state-of-the-art baselines.

## Related Work

## Dynamic Scene Reconstruction

Dynamic scene reconstruction seeks to recover geometry and appearance under challenging conditions such as occlusions and illumination changes. Traditional methods, including multi-view stereo (Newcombe, Fox, and Seitz 2015) and scene flow (Vogel, Schindler, and Roth 2013), require dense depth and struggle with significant deformations or fast motion. Learning-based approaches (Ma et al. 2019) jointly predict geometry and motion from monocular inputs but often lack temporal coherence. Recent neural implicit methods, such as NeRF (Mildenhall et al. 2020), utilize continuous volumetric representations with deformation fields (D-NeRF (Pumarola et al. 2020), NSFF (Li et al. 2021)) or higher-dimensional embeddings (HyperNeRF (Park et al. 2021)) for dynamic reconstruction. Despite their high visual fidelity, these methods involve costly per-scene optimization and slow inference, limiting real-time application. In contrast, 3DGS (Kerbl et al. 2023) employs rasterization-based anisotropic Gaussians, enabling efficient optimization and real-time rendering. Recent dynamic extensions (Yang et al. 2023b; Wu et al. 2024; Lu et al. 2025; Kwak et al. 2025; Deng et al. 2025) incorporate time-varying transformations into Gaussian to improve spatiotemporal consistency.

## Decomposition of Dynamic and Static Region

Decomposing scenes into static and dynamic components simplifies motion modeling and enhances reconstruction quality. Early methods such as DeGauss (Wang et al. 2025) and DynaSplat (Deng et al. 2025) employ external motion cues or optical flow-based masks, often using separate branches or losses. GauFre (Liang et al. 2025) further introduces dual-branch architectures with occlusion reasoning to improve temporal coherence. CoGS (Yu et al. 2024) uses compositional modeling with learned masks for flexible blending, but at the cost of increased complexity. BARD-GS (Lu et al. 2025) introduces deformation modeling into static regions, addressing motion blur but risking static geometry distortion. These methods commonly face limitations such as dependency on external priors, architectural complexity, or unintended static-region deformation. In contrast, our method explicitly decomposes geometry and appearance within Gaussian primitives through unified spatiotemporal encoding. We avoid separate encoders and complex fusion modules, maintain fixed geometry for static Gaussians with residual temporal appearance modeling, and employ visibility-based pruning and depth-aware pretraining, significantly enhancing stability, temporal consistency, and realism.

## Method

We propose a dynamic scene reconstruction framework that explicitly decomposes geometry into static and dynamic components, modeled via a unified spatiotemporal encoding. Static Gaussians maintain fixed positions with time-varying appearance, while dynamic Gaussians undergo learned motion-based deformation. Reconstruction is supervised through region-specific losses guided by visibility masks, ensuring temporal consistency and disentanglement. A visibility-driven pruning strategy improves static reliability and efficiency, and depth-aware pretraining further refines geometry alignment. We will detail this later.

## Preliminary: 3D Gaussian Splatting

3D Gaussian Splatting (Kerbl et al. 2023) represents a scene using a set of anisotropic Gaussians, each defined by a 3D center $\mu ,$ covariance matrix Î£, spherical harmonic (SH) color coefficients C, and opacity Î±. The Gaussian density at a point X is given by:

$$
G ( X ) = \exp \left( - { \frac { 1 } { 2 } } X ^ { \top } \Sigma ^ { - 1 } X \right) .\tag{1}
$$

For efficient optimization and interpretation, the covariance matrix is typically decomposed as:

$$
\Sigma = R S S ^ { \top } R ^ { \top } ,\tag{2}
$$

where R is a rotation matrix and S is a scaling matrix.

During rendering, the Gaussian is projected into screen space by applying the viewing transformation matrix W and the Jacobian matrix J of the affine approximation of the camera projection:

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { \top } J ^ { \top } .\tag{3}
$$

The final pixel color is computed via alpha compositing in front-to-back order as:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

where $c _ { i }$ and $\alpha _ { i }$ denote the color and opacity of the i-th Gaussian. This representation enables real-time rendering with high visual fidelity, but it inherently assumes static scene geometry. When extended to dynamic scenes, naively optimizing a shared set of Gaussians often leads to motion artifacts and temporal inconsistencies, due to the entanglement of geometry and appearance modeling. These challenges motivate our decomposition-based formulation, which explicitly separates static and dynamic components to ensure more stable and interpretable reconstructions.

<!-- image-->  
Figure 1: Framework Overview. We adopt a two-stage training pipeline: Stage I disentangles static and dynamic Gaussians via region-specific supervision and visibility-driven pruning to enhance geometric stability; Stage II jointly optimizes both components, where static appearance is modeled without deformation and dynamic motion is learned via a spatiotemporallyconditioned deformation network, enabling mutual refinement and improved reconstruction fidelity.

## Visual Geometry Decomposition

A core challenge in dynamic scene reconstruction is simultaneously modeling time-varying geometry and appearance. To address this, we explicitly decompose the scene at time t into two sets of Gaussian primitives:

$$
G ( t ) : = \{ G _ { \mathrm { s } } ( \mu _ { s } , \Sigma _ { s } , w _ { s } ( t ) ) \} \cup \{ G _ { \mathrm { d } } ( \mu _ { d } ( t ) , \Sigma _ { d } ( t ) , w _ { d } ( t ) ) \} ,\tag{5}
$$

where each Gaussian primitive $G ( \mu , \Sigma , w )$ consists of:

â¢ Appearance: represented by attributes w, including spherical harmonic coefficients and opacity.

â¢ Geometry: represented by center position Âµ and covariance matrix Î£.

In our formulation:

â¢ The static component $G _ { \mathrm { s } }$ maintains fixed geometry $( \mu _ { s } , \Sigma _ { s } )$ and only allows temporal variation in appearance $w _ { s } ( t )$ â¢

â¢ The dynamic component $G _ { \mathrm { d } }$ models both geometry deformation $( \mu _ { d } ( t ) , \Sigma _ { d } ( t ) )$ and appearance variation $( w _ { d } ( t ) )$ over time.

This explicit separation of geometry and appearance preserves static background integrity and accurately isolates dynamic regions.

Unified Spatiotemporal Encoding. We adopt a unified sinusoidal positional encoding $\gamma ( \cdot )$ to ensure consistent parameterization across both the geometry deformation and appearance modeling modules. Specifically, given the 3D Gaussian center position $\boldsymbol { \mu } \in \mathbb { R } ^ { 3 }$ and a scalar time t, the encoding is defined as:

$$
\begin{array} { r } { \gamma ( p ) = \left( \sin ( 2 ^ { k } \pi p ) , \cos ( 2 ^ { k } \pi p ) \right) _ { k = 0 } ^ { L - 1 } , } \end{array}\tag{6}
$$

where p represents either a spatial coordinate $( \mu _ { x } , \mu _ { y } , \mu _ { z } )$ or the temporal scalar $t ,$ and $\dot { L }$ controls the number of frequency bands. The combined input feature for subsequent modules is thus constructed as:

$$
[ \gamma ( \mu ) , \gamma ( t ) ] .\tag{7}
$$

This encoding is consistently shared across both the deformation MLP and the residual appearance MLPs. Empirically, we set $L = 1 0$ for spatial coordinates and $L = 6$ for temporal encoding in synthetic scenes, while using $L = 1 0$ for both dimensions in real-world scenarios.

Static Component. We represent static regions using Gaussian primitives $\mathcal { N } _ { s } ( \mu _ { s } , \bar { \Sigma _ { s } } )$ , whose geometry $( \mu _ { s } , \Sigma _ { s } )$ is fixed over time. To model temporal variations in appearance, such as illumination changes, we predict a residual to the initial (frozen) appearance parameters:

$$
w _ { s , i } ( t ) = w _ { s , i } ^ { ( 0 ) } + \Delta w _ { s , i } ( t )\tag{8a}
$$

$$
\begin{array} { r } { \Delta w _ { s , i } ( t ) = \mathrm { M L P } _ { \mathrm { a p p } } ^ { ( s ) } \big ( \left[ \gamma ( \mu _ { s , i } ) , \gamma ( t ) \right] \big ) , } \end{array}\tag{8b}
$$

where ${ w _ { s , i } ^ { ( 0 ) } }$ denotes the initial spherical harmonic (SH) coefficients and opacity, which remain fixed, and $\Delta w _ { s , i } ( t )$ is the residual predicted by the appearance MLP. Thus, each static Gaussian can temporally adapt its appearance without altering its geometry. We supervise the static reconstruction using a combination of L1 and Structural Similarity (SSIM) losses, computed within a binary static-region mask M:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { s t a t i c } } = { \mathcal { L } } _ { 1 } ( \hat { I } _ { s } \odot \mathbf { M } , I _ { \mathrm { g t } } \odot \mathbf { M } ) } \\ & { \qquad + \lambda _ { \mathrm { s s i m } } \cdot \mathrm { S S I M } ( \hat { I } _ { s } \odot \mathbf { M } , I _ { \mathrm { g t } } \odot \mathbf { M } ) , } \end{array}\tag{9}
$$

where $\hat { I } _ { s }$ is the rendered static image, and $I _ { \mathrm { g t } }$ is the groundtruth reference.

Dynamic Component. A straightforward baseline for modeling dynamic content is to optimize separate Gaussian primitives per timestamp and interpolate post-hoc (Luiten et al. 2024b). However, such decoupled modeling lacks temporal coherence and cannot effectively represent continuous motion patterns. Motivated by recent advances (Yang et al. 2023b), we employ a deformation-based formulation. Specifically, we introduce a deformation network, parameterized by Î¸, which predicts time-dependent offsets to canonical Gaussian parameters. Given a canonical position $\mu _ { d } ( 0 )$ and a timestamp t, the deformation network outputs offsets for position, scale, and rotation:

$$
( \delta \mu , \delta r , \delta s ) = \mathcal { F } _ { \theta } \big ( [ \gamma ( \mu _ { d } ( 0 ) ) , \gamma ( t ) ] \big ) .\tag{10}
$$

These offsets update the Gaussianâs geometry at time t as:

$$
\mu _ { d } ( t ) = \mu _ { d } ( 0 ) + \delta \mu ,\tag{11a}
$$

$$
\Sigma _ { d } ( t ) = \mathbf { A } _ { d } ( t ) \Sigma _ { d } ( 0 ) \mathbf { A } _ { d } ^ { \top } ( t )\tag{11b}
$$

where ${ \bf A } _ { d } ( t )$ is derived from $\delta r ,$ Î´s and governs the anisotropic scaling and orientation. This formulation enables temporally smooth and flexible deformation modeling through shared spatiotemporal encoding.

The dynamic component is supervised via region-specific losses computed over the dynamic mask region 1 â M:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d y n a m i c } } = \mathcal { L } _ { 1 } ( \hat { I } _ { d } , I _ { \mathrm { g t } } \odot ( \mathbf { 1 } - \mathbf { M } ) ) } \\ & { \qquad + \lambda _ { \mathrm { s s i m } } \cdot \mathrm { S S I M } ( \hat { I } _ { d } , I _ { \mathrm { g t } } \odot ( \mathbf { 1 } - \mathbf { M } ) ) } \end{array}\tag{12}
$$

where $\hat { I } _ { d }$ is the dynamically rendered image. This maskbased supervision ensures each Gaussian module receives gradients exclusively from its corresponding visible regions, promoting stable training and effective disentanglement between static and dynamic components.

Remark I. Beyond Prior Decomposition Schemes. Recent works (Lu et al. 2025; Liang et al. 2025; Yu et al. 2024; Wang et al. 2025; Deng et al. 2025)share the similar sprit of decomposing scenes into dynamic and static parts via auxiliary cues like masks, optical flow, or multibranch networks. While effective, these methods often suffer from entangled representations, causing motion leakage and unstable training. In contrast, our method performs explicit geometry-level decomposition within the 3D Gaussian representation. Static Gaussians maintain fixed geometry with time-varying appearance, while dynamic Gaussians deform via a shared spatiotemporal encoding. This unified design eliminates the need for dual-stream architectures or occlusion modeling, improving both efficiency and stability. Combined with visibility-driven pruning and depth-aware initialization, our framework achieves disentangled, temporally coherent reconstructions across diverse scenes.

Remark II. Mask-Guided Disentangled Optimization. To segment dynamic regions, we employ explicit per-frame binary masks $\mathbf { \bar { M } } \in \{ 0 , \bar { 1 } \}$ generated by the open-vocabulary tracker Track Anything (Yang et al. 2023a). These masks supervise static and dynamic Gaussians separatelyâusing M to ensure robust structural disentanglement. Unlike prior methods (Wang et al. 2025; Deng et al. 2025; Liang et al.

2025) that rely on optical flow or learned soft masks for implicit separation, our approach, akin to BARD-GS (Lu et al. 2025), benefits from externally provided masks for more accurate and temporally consistent guidance. To further enhance disentanglement, we introduce an asymmetric masking strategy: for static regions (Eq. (9)), both prediction and ground truth are masked by M, while for dynamic regions (Eq. (12)), only the ground truth is masked. This preserves occluded static geometry while enabling complete reconstruction of dynamic content for more stable training and improved reconstruction quality.

## Visibility-Driven Pruning

In dynamic scene reconstruction, static Gaussians located near view boundaries or occluded regions are often weakly supervised due to limited visibility, leading to unstable optimization and redundancy. To address this, we propose a visibility-driven pruning strategy that quantifies each static Gaussianâs long-term contribution over the video sequence.

Let $G _ { \mathrm { s } } = \{ G _ { s , i } ( \mu _ { s , i } , \Sigma _ { s , i } , w _ { s , i } ( t ) ) \} _ { i = 1 } ^ { N _ { s } }$ denote the set of static Gaussians. For each $G _ { s , i } \in G _ { \mathrm { s } } ,$ we define its integrated visibility score as:

$$
\bar { V } _ { s , i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { 1 } _ { \{ G _ { s , i } \mathrm { ~ r e n d e r e d ~ a t ~ } t \} } \cdot ( 1 - \alpha _ { s , i } ( t ) )\tag{15}
$$

where $T$ is the total number of training frames, $\mathcal { V } _ { t } ^ { ( s ) } \subseteq G _ { \mathrm { s } }$ denotes the subset of visible static Gaussians at time t, $\mathbb { 1 } _ { \{ G _ { s , i } \mathrm { ~ r e n d e r e d ~ a t ~ } t \} }$ is the indicator function (1 if $G _ { s , i }$ is rendered at time $t ) , \alpha _ { s , i } ( t )$ is the opacity of $G _ { s , i }$ at time t.

This formulation unifies notation with earlier definitions and reflects both temporal visibility and opacity modulation. We prune low-contribution Gaussians based on $\bar { V } _ { s , i }$ thresholds to improve training stability and reduce redundancy.

## Depth-Aware Pretraining

3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) demonstrates that initializing Gaussian primitives using Structurefrom-Motion (SfM) (Schonberger and Frahm 2016) point clouds can facilitate effective training. However, we find that such initializations are often suboptimal for depth-guided reconstruction, frequently leading to geometric inconsistencies. This issue stems from the requirement to align the SfM reconstruction with available depth images by estimating a global scale and offset. In practice, inaccuracies or sparsity in the SfM point cloud can impair this calibration, thereby weakening the efficacy of depth-based regularization. To address this, we introduce a short pretraining stage before the main optimization. This stage refines the initial geometry and improves alignment between the scene structure and depth supervision, resulting in a more reliable point cloud for subsequent learning. Specifically, we regularize the static Gaussian geometry using monocular depth maps through the following loss:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \lambda _ { \mathrm { d e p t h } } ( t ) \cdot \left\| ( \hat { D } - D _ { \mathrm { g t } } ) \odot \mathbf { M } \right\| _ { 1 }\tag{14}
$$

where $\hat { D }$ denotes the rendered depth map from the static Gaussians, $D _ { \mathrm { { g t } } }$ is the ground-truth monocular depth map,

Deformable 3DGS

4DGS

<!-- image-->  
GauFre  
MoDec-GS  
Ours  
Ground Truth

Figure 2: Qualitative comparison on the NeRF-DS (Yan, Li, and Lee 2023) monocular video dataset. Red and blue boxes highlight regions where our method notably improves visual quality compared to prior approaches.

and M is the binary mask identifying static regions. The time-dependent decay factor Î»depth(t) gradually reduces the influence of depth supervision as training progresses. This pretraining stage enhances geometric consistency and stabilizes later optimization.

## Training Protocol.

Our training pipeline follows a two-stage design with an initial Depth-Aware Pretraining (DAP) phase to refine SfMinitialized geometry.

Stage I (S1) focuses on disentangling static and dynamic components via region-specific supervision using binary masks M. Static Gaussians are optimized with pixels where M = 1, while dynamic Gaussians use the complementary region. To further improve static geometry quality, we introduce a Visibility-Driven Pruning (VDP) strategy that removes low-visibility static Gaussians (e.g., near view frustums or edges), thereby mitigating supervision noise and improving geometric stability.

Stage II (S2) performs joint optimization of both components. Static Gaussians retain fixed geometry and are refined using an Appearance Model (APP), which predicts view- and time-dependent appearance (e.g., color and opacity) without invoking deformation. Dynamic Gaussians are updated through a learned deformation network conditioned on shared spatiotemporal encodings. During this stage, gradients propagate across both static and dynamic components, allowing mutual refinement. This two-stage design significantly improves reconstruction fidelity and training stability, as validated in our ablation studies (Table 3), more details can be seen in Figure 1.

## Experiment

## Experimental Setup

We implement our method in PyTorch (Paszke et al. 2019), building upon the official 3D Gaussian Splatting (Kerbl et al. 2023) and Deformable 3DGS (Yang et al. 2023b) frameworks. Training is conducted in two stages: first, we separately optimize static and dynamic components for 30k iterations without the static appearance model; second, we jointly train both components with appearance modeling for 40k iterations. A single Adam optimizer (Adam et al. 2014) with $\beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9$ , and a learning rate decaying exponentially from $8 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 . 6 \times 1 0 ^ { - 6 }$ is used. Both deformation and appearance MLPs share architecture and schedule. Visibility-driven pruning removes rarely rendered Gaussians to improve training efficiency. For depth-supervised scenes, a short pretraining stage aligns SfM point clouds with depth maps before applying depth regularization.

We evaluate our method on three datasets: (1) iPhone (Gao et al. 2022), featuring 14 real-world scenes (4180 frames at 720 Ã 480) with handheld motion and diverse dynamics; (2) NeRF-DS (Yan, Li, and Lee 2023), a monocular dataset containing specular objects and challenging motion; and (3) HyperNeRF (Park et al. 2021), which includes complex dynamic scenes captured in real-world environments. Following standard protocols, we report PSNR, SSIM, and LPIPS (Zhang et al. 2018) to evaluate novel view synthesis quality.

## Compared with SOTA Results

Quantitative Comparison. We compare our approach with Deformable 3DGS (Yang et al. 2023b), 4DGS (Wu et al. 2024), HyperNeRF (Park et al. 2021), NeRF-DS (Yan,

<table><tr><td></td><td colspan="3">as</td><td colspan="3">basin</td><td colspan="3">bell</td><td colspan="3">cup</td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS </td></tr><tr><td>HyperNeRFSIGGRAPH Asia 2021</td><td>25.59</td><td>0.8567</td><td>0.1754</td><td>20.41</td><td>0.8099</td><td>0.1889</td><td>23.06</td><td>0.7698</td><td>0.2479</td><td>23.98</td><td>0.8531</td><td>0.1988</td></tr><tr><td>NeRF-DScvPR 2023</td><td>25.34</td><td>0.8679</td><td>0.1515</td><td>20.23</td><td>0.8032</td><td>0.2008</td><td>22.57</td><td>0.7821</td><td>0.2489</td><td>24.51</td><td>0.8659</td><td>0.1668</td></tr><tr><td>Deformable 3DGScvPR 2024</td><td>26.03</td><td>0.8836</td><td>0.1351</td><td>19.67</td><td>0.7867</td><td>0.1498</td><td>24.48</td><td>0.7997</td><td>0.1822</td><td>24.50</td><td>0.8763</td><td>0.1472</td></tr><tr><td>4DGScvPR 2024</td><td>24.77</td><td>0.8642</td><td>0.1521</td><td>19.36</td><td>0.7677</td><td>0.1678</td><td>23.16</td><td>0.8015</td><td>0.1571</td><td>23.88</td><td>0.8691</td><td>0.1532</td></tr><tr><td>GauFrewacCV 2025</td><td>26.05</td><td>0.8790</td><td>0.1244</td><td>19.54</td><td>0.7780</td><td>0.1222</td><td>25.24</td><td>0.8130</td><td>0.1351</td><td>24.04</td><td>0.8191</td><td>0.2054</td></tr><tr><td>MoDec-GScvPR 2025</td><td>24.65</td><td>0.8538</td><td>0.1460</td><td>19.57</td><td>0.7787</td><td>0.1805</td><td>22.19</td><td>0.7562</td><td>0.2312</td><td>24.18</td><td>0.8798</td><td>0.2643</td></tr><tr><td>Ours</td><td>26.01</td><td>0.8806</td><td>0.1031</td><td>19.78</td><td>0.7885</td><td>0.1278</td><td>25.55</td><td>0.8484</td><td>0.1425</td><td>24.63</td><td>0.8829</td><td>0.1112</td></tr><tr><td></td><td></td><td>plate</td><td></td><td></td><td>press</td><td></td><td></td><td>sieve</td><td></td><td></td><td>Average</td><td></td></tr><tr><td>Method</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>HyperNeRFSIGGRAPH Asia 2021</td><td>21.10</td><td>0.7979</td><td>0.2614</td><td>24.59</td><td>0.8263</td><td>0.2385</td><td>25.41</td><td>0.8593</td><td>0.2142</td><td>23.44</td><td>0.8247</td><td>0.2178</td></tr><tr><td>NeRF-DScvPR 2023</td><td>19.70</td><td>0.7813</td><td>0.2467</td><td>25.34</td><td>0.8711</td><td>0.2032</td><td>24.99</td><td>0.8705</td><td>0.2067</td><td>23.24</td><td>0.8345</td><td>0.2035</td></tr><tr><td>Deformable 3DGScvPR 2024</td><td>19.88</td><td>0.8293</td><td>0.1914</td><td>25.32</td><td>0.8752</td><td>0.1378</td><td>25.62</td><td>0.8627</td><td>0.1206</td><td>23.64</td><td>0.8447</td><td>0.1520</td></tr><tr><td>4DGScvPR 2024</td><td>18.77</td><td>0.7891</td><td>0.1857</td><td>24.81</td><td>0.8311</td><td>0.1598</td><td>25.16</td><td>0.8611</td><td>0.1234</td><td>22.84</td><td>0.8262</td><td>0.1570</td></tr><tr><td>GauFrewacv 2025</td><td>20.00</td><td>0.8051</td><td>0.2323</td><td>25.05</td><td>0.8545</td><td>0.1763</td><td>24.88</td><td>0.8568</td><td>0.1623</td><td>23.54</td><td>0.8293</td><td>0.1654</td></tr><tr><td>MoDec-GScvPR 2025</td><td>18.87</td><td>0.7306</td><td>0.2547</td><td>22.87</td><td>0.7296</td><td>0.2111</td><td>23.48</td><td>0.7982</td><td>0.2001</td><td>22.25</td><td>0.7895</td><td>0.2125</td></tr><tr><td>Ours</td><td>20.34</td><td>0.8116</td><td>0.1413</td><td>25.43</td><td>0.8701</td><td>0.1498</td><td>26.49</td><td>0.8753</td><td>0.1137</td><td>24.03</td><td>0.8510</td><td>0.1270</td></tr></table>

Table 1: Quantitative results comparison on the NeRF-DS (Yan, Li, and Lee 2023) dataset. Red and orange cells denote the best and second-best results, respectively.  
Deformable 3DGS GauFre

Ours  
Ground Truth  
<!-- image-->  
Figure 3: Qualitative results comparison on HyperNeRF (Yan, Li, and Lee 2023) monocular video dataset.

Li, and Lee 2023), GauFre (Liang et al. 2025), MoDec-GS (Kwak et al. 2025). As summarized in Table 1, our method consistently achieves the best or second-best performance across most sequences on the NeRF-DS dataset, demonstrating robustness under challenging dynamics and lighting. Averaged over all metrics, it surpasses competing baselines, indicating superior reconstruction capability. Table 2 further reports evaluations on the HyperNeRF and iPhone datasets, where our method remains highly competitive against Deformable 3DGS and GauFre. These results collectively validate the generalizability of our approach across diverse real-world settings involving non-rigid motion and handheld camera trajectories.

Qualitative Comparison. We qualitatively evaluate our method on NeRF-DS (Yan, Li, and Lee 2023) and Hyper-NeRF (Park et al. 2021), as shown in Figure 2 and Figure 3. These datasets cover dynamic motions, complex lighting, and varying camera paths. Our method achieves highfidelity results with sharp details, temporally coherent motion, and clean static regions. In particular, it maintains geometric stability and appearance consistency even where dynamic and static elements interact, outperforming existing baselines. This highlights the effectiveness of our decomposition and unified spatiotemporal modeling for robust and perceptually plausible dynamic scene reconstruction.

<table><tr><td colspan="4">HyperNeRF</td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>Deformable 3DGScvPR 2024</td><td>24.57</td><td>0.7641</td><td>0.2439</td></tr><tr><td>GauFrewACV 2025</td><td>23.59</td><td>0.7486</td><td>0.2416</td></tr><tr><td>Ours</td><td>24.61</td><td>0.7626</td><td>0.2398</td></tr><tr><td colspan="4">iPhone</td></tr><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Deformable 3DGScvPR 2024</td><td>12.56</td><td>0.2902</td><td>0.5896</td></tr><tr><td>GauFrewAcV 2025</td><td>13.27</td><td>0.3382</td><td>0.6206</td></tr><tr><td>Ours</td><td>13.53</td><td>0.3391</td><td>0.6205</td></tr></table>

Table 2: Quantitative results on HyperNeRF (Park et al. 2021) and iPhone (Gao et al. 2022) datasets. Red and orange indicate best and second-best.

<table><tr><td>Variant</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>(a) S1(baseline)</td><td>22.41</td><td>0.8268</td><td>0.1843</td></tr><tr><td>(b) S1+S2</td><td>23.35</td><td>0.8413</td><td>0.1732</td></tr><tr><td>(c) S1+S2+APP</td><td>23.36</td><td>0.8419</td><td>0.1505</td></tr><tr><td>(d) S1+S2+APP+DAP</td><td>23.78</td><td>0.8457</td><td>0.1342</td></tr><tr><td>(e) (d)+VDP(full)</td><td>24.03</td><td>0.8505</td><td>0.1274</td></tr></table>

Table 3: Ablation study on individual components of Split-Gaussian on the NeRF-DS dataset.

## Ablation Study

Visual Geometry Decomposition Matters. We conduct an ablation study on the NeRF-DS dataset to quantify the contribution of each component in the SplitGaussian framework, as summarized in Tab. 3. Starting from the baseline (a) S1, which separately trains static and dynamic branches, we observe modest reconstruction quality. Adding S2 in (b) introduces joint optimization between both branches, leading to consistent gains of +0.94 dB in PSNR, +0.0145 in SSIM, and a LPIPS drop of 0.0111, indicating the benefit of mutual spatiotemporal supervision. The introduction of the appearance modeling module in (c) further reduces perceptual distortion (LPIPS â 0.0227), suggesting improved handling of time-varying appearance. In (d), depth-aware pretraining facilitates geometric alignment, contributing an additional +0.42 dB in PSNR and a 0.0163 reduction in LPIPS. Finally, incorporating visibility-driven pruning in (e) yields the best overall performance, achieving a PSNR of 24.03, SSIM of 0.8505, and LPIPS of 0.1274. These results demonstrate that each module brings measurable improvement, and the full model provides the most stable and perceptually accurate reconstruction.

<!-- image-->  
Ground Truth

<!-- image-->  
Ours

<!-- image-->  
w/o APP

Figure 4: Ablation on Appearance Modeling (APP). Comparison of reconstruction results with and without the appearance module. Ours exhibits better alignment and fewer artifacts in static regions.  
<!-- image-->  
Ground Truth

<!-- image-->  
Ours

<!-- image-->  
w/o VDP  
Figure 5: Removing VDP leads to bright blotches near view boundaries due to undertrained Gaussians accumulated at sparsely visible regions.

Appearance Modeling Improves Static Reconstruction. Figure 4 illustrates reconstructions w/ and w/o APP. Removing the appearance module results in noticeable degradation in static regions, especially under subtle lighting changes, leading to artifacts and reduced photorealism. A closer examination reveals that the model with APP can accurately reconstruct fine-grained illumination effectsâsuch as the soft shadow on the static paper surfaceâwhile the version without APP fails to capture this detail, producing oversmoothed or inconsistent shading. This highlights the necessity of temporally adaptive appearance modeling for static Gaussians to ensure consistent visual quality across time.

VDP Improves Peripheral Realism. Figure 5 illustrates the impact of removing the Visibility-Driven Pruning (VDP) module. Without VDP, static Gaussians located near the periphery of training viewsâwhere visibility is sparseâreceive inadequate optimization. As a result, these insufficiently supervised Gaussians accumulate during densification and exhibit artificially elevated opacities. This leads to prominent residual artifacts, particularly along image boundaries, which degrade rendering quality and visual coherence. In contrast, our full method incorporates visibility-aware filtering to suppress low-visibility Gaussians early in training, effectively reducing peripheral noise and producing cleaner, artifact-free reconstructions.

<!-- image-->  
(a)

<!-- image-->

<!-- image-->  
(b)

<!-- image-->  
(d)

(c)  
<!-- image-->  
(e)

<!-- image-->  
(f)  
Figure 6: (a) Ground truth. (b) Our dynamic reconstruction. (c) Our static reconstruction. (d) Learned dynamic mask shared by both methods. (e) Dynamic result from GauFRe (Liang et al. 2025). (f) Static result from GauFRe (Liang et al. 2025).

Beyond Mask-Guided Approaches. Figure 6 presents a visual comparison between our method and GauFRe (Liang et al. 2025). GauFRe effectively integrates occlusion reasoning and mask-based decomposition, enabling basic separation of dynamic and static regions. However, as shown in (e)(f), it occasionally produces blending and structural artifacts under complex motion. In contrast, our approach (b)(c), guided by a learned dynamic mask (d), achieves clearer decomposition in both regions, suggesting enhanced robustness in dynamic-static separation.

## Limitations and Conclusions

While SplitGaussian achieves state-of-the-art performance in dynamic scene reconstruction, its two-stage training pipelineâseparately optimizing static and dynamic components before joint appearance modelingâintroduces additional computational overhead compared to end-to-end methods. Nevertheless, the proposed decomposition of geometry and appearance, combined with unified spatiotemporal encoding, visibility-driven pruning, and depth-aware regularization, enables temporally coherent and photorealistic reconstruction. Extensive experiments confirm its robustness and generalization across diverse dynamic scenes.

## References

Adam, K. D. B. J.; et al. 2014. A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 1412(6).

Deng, J.; Shi, P.; Li, Q.; and Guo, J. 2025. DynaSplat: Dynamic-Static Gaussian Splatting with Hierarchical Motion Decomposition for Scene Reconstruction. arXiv:2506.09836.

Gao, H.; Li, R.; Tulsiani, S.; Russell, B.; and Kanazawa, A. 2022. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems, 35: 33768â33780.

Katsumata, K.; Vo, D. M.; and Nakayama, H. 2024. A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis. In Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part LXXXVI, 394â412. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3-031- 73015-3.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Kwak, S.; Kim, J.; Jeong, J. Y.; Cheong, W.-S.; Oh, J.; and Kim, M. 2025. MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Li, Z.; Niklaus, S.; Snavely, N.; and Wang, O. 2021. Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Liang, Y.; Khan, N.; Li, Z.; Nguyen-Phuoc, T.; Lanman, D.; Tompkin, J.; and Xiao, L. 2025. GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis. In WACV.

Lu, Y.; Zhou, Y.; Liu, D.; Liang, T.; and Yin, Y. 2025. BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting. arXiv preprint arXiv:2503.15835.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024a. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024b. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV.

Ma, W.-C.; Wang, S.; Hu, R.; Xiong, Y.; and Urtasun, R. 2019. Deep rigid instance scene flow. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3614â3622.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Newcombe, R. A.; Fox, D.; and Seitz, S. M. 2015. DynamicFusion: Reconstruction and Tracking of Non-Rigid Scenes in Real-Time.

Park, K.; Sinha, U.; Hedman, P.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Martin-Brualla, R.; and Seitz, S. M. 2021. HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields. ACM Trans. Graph., 40(6).

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Pumarola, A.; Corona, E.; Pons-Moll, G.; and Moreno-Noguer, F. 2020. D-NeRF: Neural Radiance Fields for Dynamic Scenes. arXiv preprint arXiv:2011.13961.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4104â4113.

Tretschk, E.; Tewari, A.; Golyanik, V.; Zollhofer, M.; Lass- Â¨ ner, C.; and Theobalt, C. 2021. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF international conference on computer vision, 12959â12970.

Vogel, C.; Schindler, K.; and Roth, S. 2013. Piecewise rigid scene flow. In Proceedings of the IEEE International Conference on Computer Vision, 1377â1384.

Wang, R.; Lohmeyer, Q.; Meboldt, M.; and Tang, S. 2025. DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction. arXiv:2503.13176.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2024. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20310â20320.

Yan, Z.; Li, C.; and Lee, G. H. 2023. NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8285â8295.

Yang, J.; Gao, M.; Li, Z.; Gao, S.; Wang, F.; and Zheng, F. 2023a. Track Anything: Segment Anything Meets Videos. arXiv:2304.11968.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2023b. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. arXiv preprint arXiv:2309.13101.

Yu, H.; Julin, J.; Milacski, Z. A.; Niinuma, K.; and Jeni, Â´ L. A. 2024. Cogs: Controllable gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21624â21633.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.