<!-- page 1 -->
1
DET-GS: Depth- and Edge-Aware Regularization for
High-Fidelity 3D Gaussian Splatting
Zexu Huang , Min Xu , Member, IEEE, Stuart Perry , Senior Member, IEEE
Abstract—3D Gaussian Splatting (3DGS) represents a significant
advancement in the field of efficient and high-fidelity novel
view synthesis. Despite recent progress, achieving accurate
geometric reconstruction under sparse-view conditions remains a
fundamental challenge. Existing methods often rely on non-local
depth regularization, which fails to capture fine-grained structures
and is highly sensitive to depth estimation noise. Furthermore,
traditional smoothing methods neglect semantic boundaries and
indiscriminately degrade essential edges and textures, consequently
limiting the overall quality of reconstruction. In this work, we
propose DET-GS, a unified depth and edge-aware regularization
framework for 3D Gaussian Splatting. DET-GS introduces a hier-
archical geometric depth supervision framework that adaptively
enforces multi-level geometric consistency, significantly enhancing
structural fidelity and robustness against depth estimation noise.
To preserve scene boundaries, we design an edge-aware depth
regularization guided by semantic masks derived from Canny
edge detection. Furthermore, we introduce an RGB-guided
edge-preserving Total Variation loss that selectively smooths
homogeneous regions while rigorously retaining high-frequency
details and textures. Extensive experiments demonstrate that DET-
GS achieves substantial improvements in both geometric accuracy
and visual fidelity, outperforming state-of-the-art (SOTA) methods
on sparse-view novel view synthesis benchmarks. (The link to the
code will be made available after publication)
Index Terms—3D Gaussian Splatting, Neural Rendering, 3D
Reconstruction, Novel View Synthesis.
I. INTRODUCTION
T
HE field of 3D reconstruction has experienced significant
advancements, notably enhancing innovative view synthe-
sis and facilitating the display of photorealistic volumetric
scenes. Neural radiance fields (NeRF) [1] have achieved
significant advancements in reconstructing photorealistic per-
formances and precise 3D structures from sparse view sets [2]–
[4]. However, numerous sparse-input NeRF frameworks are
constrained by slow computation speeds and high memory
consumption. The identified limitations lead to substantial
computational and temporal expense, thereby considerably
hindering their application in practical applications. To address
efficiency issues, some approaches integrate grid-based struc-
tures [5], [6] to accelerate inference. However, these methods
present inherent trade-offs: although they enhance rendering
speed, they often lead to a substantial increase in training
overhead or a reduction in the fidelity of the rendered images.
Zexu Huang, Min Xu and Stuart Perry are with the Perceptual Imaging
Laboratory (PILab), School of Electrical and Data Engineering, Faculty
of Engineering and Information Technology, University of Technology
Sydney, Ultimo, NSW 2007, Australia (email: zexu.huang@student.uts.edu.au;
min.xu@uts.edu.au; stuart.perry@uts.edu.au)
Manuscript received August 6, 2025;
In recent work, 3D Gaussian Splatting (3DGS) [7] proposed
an unstructured radiance field representation utilizing 3D
Gaussian primitives, demonstrating exceptional performance in
fast, high-quality, and cost-effective novel view synthesis when
trained with densely sampled color images. Even under sparse
view conditions, it successfully preserves detailed and well-
defined regional characteristics. Nonetheless, 3DGS frequently
encounters visual distortions when subjected to camera angles
that were not included in its training or during proximity
observations, which can be attributed to its restricted ability
to reconstruct fine details. While several improvements [8]–
[10] have been proposed to enhance 3DGS through advanced
smoothing filters [8], better initialization strategies [9], and
more expressive appearance models [10], these approaches
primarily focus on appearance preservation and do not directly
address the underlying geometric accuracy. Depth information
has recently emerged as a powerful geometric prior for Gaus-
sian splatting frameworks. However, existing depth-regularized
methods [11]–[13] typically impose supervision by enforcing
pixel-to-pixel consistency between the predicted and reference
depth maps over the entire image. Such supervision overlooks
local geometric context, making it difficult to capture fine-
grained structures and increasing sensitivity to monocular depth
noise. Moreover, current regularization techniques [11]–[14]
apply uniform smoothing across the scene without considering
semantic or geometric boundaries, resulting in blurred object
contours and loss of structural details. While 3DGS models [7]–
[10], [13], [14] achieve high-quality color reconstructions, they
lack image-space regularization that distinguishes flat regions
from high-frequency areas.
To address these challenges, we propose DET-GS, a depth
and edge-aware regularization framework for 3D Gaussian
Splatting that fundamentally refocuses geometric supervision
and structural preservation. DET-GS introduces three key
innovations designed to enhance both geometric accuracy and
visual fidelity under sparse-view conditions. First, our method
introduces a hierarchical geometric depth supervision strategy
that integrates an error-tolerant mechanism, delivering robust
geometric guidance while effectively mitigating the impact of
depth estimation noise and addressing the inherent limitations
of conventional non-local regularization schemes. To preserve
important geometric structures, we design an edge-aware depth
smoothing approach based on Canny edge detection [15], which
selectively regularizes non-boundary regions while maintaining
sharp object contours. Finally, we develop an RGB-guided edge-
preserving Total Variation loss, a structure-aware image-space
regularization technique that selectively enforces smoothness in
homogeneous areas while rigorously protecting high-frequency
arXiv:2508.04099v1  [cs.CV]  6 Aug 2025

<!-- page 2 -->
2
Multi-view Inputs
Training
Outputs
Depth Estimator
Depth Maps
Edge-Aware 
Regularization
Fig. 1. DET-GS: Our method reconstructs 3D scenes from multi-view RGB inputs using a representation of 3D Gaussian primitives. To enhance geometric
coherence, we introduce hierarchical depth supervision leveraging depth maps predicted by a pretrained estimator. We further propose an edge-aware regularization
strategy based on edge masks extracted via Canny edge detection, preserving essential geometric boundaries. Additionally, we develop an RGB-guided total
variation loss to refine appearance details effectively. These novel components collectively enforce geometric consistency, yielding photorealistic renderings that
faithfully retain structural detail.
details and texture (see Fig. 1).
Through rigorous experimentation, DET-GS demonstrates
substantial improvements over existing state-of-the-art (SOTA)
3DGS-based methods in both geometric accuracy and visual
fidelity. Ablation studies further validate the effectiveness of
each proposed component, highlighting the importance of
hierarchical geometric depth supervision, edge-aware regu-
larization, and structure-preserving image-space smoothing.
These results collectively establish DET-GS as a robust and
principled solution for high-quality 3D scene reconstruction
under sparse-view constraints.
Contributions
In summary, the contributions of this work are as follows:
1) We propose a hierarchical geometric depth supervision
strategy that captures geometric structures across multi-
level geometric details and introduces an error-tolerant
mechanism to improve robustness against noisy monoc-
ular depth priors.
2) We design an edge-aware depth regularization mechanism
guided by semantic masks derived from Canny edge
detection, enabling selective smoothing that preserves
critical scene boundaries and enhances geometric fidelity.
3) We develop an RGB-guided edge-preserving Total Varia-
tion loss that selectively smooths homogeneous regions
while rigorously protecting high-frequency textures, sig-
nificantly improving the perceptual quality of novel view
synthesis.
II. RELATED WORK
MLP-based Radiance Fields
Early neural field methods
primarily relied on multi-layer perceptrons (MLPs) to approxi-
mate both the geometric structure and visual features of three-
dimensional scenes. These models process spatial coordinates
and viewing angles to derive scene-specific properties. Typical
outputs include the signed distance function (SDF) of the
surface geometry [16]–[18], or density and color values at
particular spatial points [1], [19], [20]. Neural Radiance Field
(NeRF) methods [1] have significantly advanced novel view
synthesis by producing highly realistic imagery and effectively
modeling view-dependent visual effects. Utilizing established
volume rendering formulas, NeRF methods train a coordinate-
centric MLP to simultaneously encode geometry and radiance,
mapping these directly from spatial coordinates and viewing
directions enhanced by positional encoding. Despite delivering
high synthesis quality through volumetric modeling, MLP-
based approaches confront notable limitations. The primary
constraint is the necessity for extensive sampling along each
camera ray, necessitating that the MLP assess several sites.
This expensive procedure significantly reduces rendering speed
and limits performance, particularly for extensive and complex
scenes. To mitigate the computational challenges associated
with dense sampling and extensive MLP evaluations, various
hybrid feature-grid methods have been introduced. These
approaches [5], [21]–[25] essentially “cache” intermediate
feature representations to streamline the rendering pipeline.
Among these techniques, multi-resolution hash encoding [5]
has gained popularity, offering a versatile foundation that accel-
erates rendering by capturing detailed scene information across
multiple scales and facilitating efficient level-of-detail (LOD)
renderings [26], [27]. Although these solutions significantly
improve rendering speed and visual quality, they still fail
to eliminate the fundamental computational burden posed by
extensive ray casting operations. Consequently, there remains a
substantial performance gap preventing true real-time rendering.
Point-based Radiance Fields
Point-based radiance field
rendering benefits significantly from using explicit point clouds
as proxies for scene representation. These point sets can be
quickly and accurately acquired using LiDAR sensors [28]
or through reconstruction techniques such as Structure-from-

<!-- page 3 -->
3
Motion (SfM) and Multi-View Stereo (MVS) [29]. Recent
advancements incorporate neural-based descriptors [30] or
specially designed attributes [31], enabling high-quality visual
synthesis via differentiable point rendering pipelines [32],
[33]. However, discrete rasterization can introduce visual
imperfections, notably aliasing and overdrawing, especially
when multiple points overlap in a single pixel. Recently, 3D
Gaussian Splatting (3DGS) [7] has significantly enhanced novel
view synthesis, achieving real-time rendering at high-definition
resolutions. Unlike continuous density-based fields and ray-
based methods, 3DGS leverages a collection of anisotropic 3D
Gaussians. Rendering occurs through rasterization, projecting
these Gaussians onto a 2D image plane. Pixels are then
determined by sorting the projected Gaussians based on depth
and blending them using alpha composition. Concurrent studies
have rapidly expanded the capabilities of the 3DGS framework,
focusing on improvements in visual fidelity [10], [34]–[39],
avatar reconstruction [40], [41], anti-aliasing techniques [8],
[42], [43], and structured grid representations [9]. Nevertheless,
these enhancements primarily emphasize visual appearance
improvements without explicitly tackling underlying geometric
precision.
Depth Supervision in Radiance Fields
Depth has previously
functioned as a crucial indicator in several 3D computer vision
applications [44]–[46], and has lately been prominent in the
supervision of sparse-view radiance fields. Current depth-
supervised methodologies may be generally classified into
two categories. One category [47], [48] obtains accurate but
sparse depth information from reliable point cloud sources,
whereas another category [49]–[51] generates dense depth
cues from dependable monocular depth estimators [48], [52]–
[54]. Monocular depth forecasts provide enhanced reliability
and density in instances when point clouds are sparse or
absent. Monocular depth assessments are intrinsically plagued
by scale ambiguity and possible inaccuracy. To tackle these
challenges, previous studies and new sparse-view 3D Gaus-
sian Splatting (3DGS) methods employ scale-invariant loss
functions [11]–[14], [49], [55], [56], such as depth ranking
loss [50]. Although they are popular, these strategies possess
limits within our context. The highly adaptable character of
Gaussian representations primarily heightens susceptibility
to erroneous depth signals, necessitating more regularization.
Moreover, these losses generally impose alignment on a non-
local scale, disregarding local depth discrepancies. This neglect
can result in noisy Gaussian distributions, particularly in regions
with intricate textures. Furthermore, existing regularization
methods [11]–[14] apply uniform smoothing over the entire
scene, ignoring semantic and geometric boundaries. This
often leads to blurred edges and a loss of fine structural
details. To address the limitations discussed above, we pro-
pose a hierarchical geometric regularization framework that
effectively mitigates noise sensitivity inherent in monocular
depth estimation and avoids the shortcomings associated with
non-local smoothing. We also propose a depth smoothing
strategy guided by Canny edge detection [15], enabling targeted
regularization within homogeneous regions and simultaneously
preserving sharp boundaries crucial for maintaining geometric
accuracy. Moreover, we devise an RGB-driven, structure-
sensitive Total Variation (TV) loss in image space, selectively
imposing smoothness only on uniform regions while rigorously
safeguarding intricate textures and high-frequency details.
III. METHODOLOGY
A. Preliminary
3D Gaussian splatting [7] leverages multiple 3D Gaussian
elements to encode spatial information. Specifically, this
technique generates the color value C at each pixel using
Gaussian primitives denoted as θ, camera orientation P, and
intrinsic camera parameters including the center position o.
Each Gaussian primitive is characterized by three distinct
parameters: a central position µ ∈R3, a scaling vector
s ∈R3, and a rotational quaternion q ∈R4. Formally, the
n-th primitive’s Gaussian basis function Gn is mathematically
given by:
Gn(x) = e−1
2 (x−µn)T Σ−1
n (x−µn),
(1)
where the covariance Σn is determined using scale sn and
rotation qn. Each Gaussian primitive further contains an opacity
scalar α ∈R and a color descriptor feature f
∈RK.
Consequently, the full parameterization for Gaussian primitive
i is denoted as θn = µn, sn, qn, αn, fn. Rendering with
3D Gaussian splatting relies on point-based accumulation.
Specifically, pixel color C(xi) results from compositing the
contributions of N overlapping Gaussians:
C(xi) =
X
n∈N
cneαn
n−1
Y
j=1
(1 −eαj), where eαn = αnG2D
n (xi),
(2)
with cn representing the decoded color from feature fn. Distinct
from conventional radiance field [1], which performs ray-wise
sampling, Gaussian primitives for each pixel are selected
using an optimized rasterization method, guided by pixel
coordinate xi, camera intrinsics, pose P, and certain predefined
criteria. The rendered opacity eαn of each Gaussian primitive is
computed from its inherent opacity αn and its 2D projection
G2D
n
on the image plane.
Depth value D(xi) at each pixel location is calculated
analogously, defined by the distance from the primitive center
µn to the camera center o:
D(xi) =
X
n∈N
||µn −o||2 × eαn
n−1
Y
j=1
(1 −eαj).
(3)
Parameter optimization in 3D Gaussian splatting is performed
through gradient-based methods guided by color information.
During the training process, the system duplicates the primitives
exhibiting the highest activation to better capture complex
visual features, simultaneously pruning unnecessary elements.
This study adopts these optimization strategies directly for color
guidance. Traditionally, methods initialize Gaussians with point
clouds from COLMAP [29], [57] or similar structure-from-
motion approaches.
B. Hierarchical Geometric Depth Supervision
Existing 3DGS-based methods [11]–[13] typically enforce
depth consistency at broad spatial scales, overlooking fine-
grained geometric variations within local regions. This oversight

<!-- page 4 -->
4
Inputs
SFM Points
Monocular Depth 
Estimator
Depth Maps
Depth Map Patches
Rendered Depth Patches
Rendered Image
Ground Truth
ℒ!"#"$, ℒ%&
ℒ'()%*
Edge-Aware 
Regularization
Render
Render
ℒ('+(
Splatting
Rendered Images
Fig. 2. Overview of DET-GS: Given a set of RGB images, a pre-trained monocular depth estimator provides monocular depth priors to guide the optimization.
The scene is represented by a set of 3D Gaussian primitives, which can render both color images and depth maps. During training, we decompose both the
predicted and pseudo-ground-truth depth maps into non-overlapping patches and apply Hierarchical Geometric Depth Supervision, enabling dual-scale
geometric consistency through patch-wise normalization. To preserve geometric boundaries, we incorporate an Edge-Aware Depth Regularization mechanism
based on Canny edge detection from ground-truth RGB images, enforcing structure-aware smoothing selectively in non-edge regions. Additionally, we propose
a RGB-Guided Edge-Preserving Total Variation Regularization, which adaptively penalizes image-space gradients based on RGB-derived structural
information. These loss components (Ldepth, Ledge, Ltv, Lcolor) collectively guide the optimization of Gaussian parameters for high-fidelity 3D reconstructions.
leads to suboptimal Gaussian primitive arrangements, especially
in areas containing complex surface details where precise depth
alignment is crucial. Such non-local-only supervision fails to
address localized depth inconsistencies, resulting in scattered
Gaussian distributions that compromise reconstruction quality
in texture-rich regions.
We leverage Depth Anything V2 [54] as a prior to estimate
depth predictions eD for training images. It is a state-of-the-art
(SOTA) pre-trained monocular depth estimation model. This
provides pseudo-ground-truth depth information that guides the
optimization of Gaussian primitive positions during training.
During the training process, we compute the predicted depth
D from our 3D Gaussian representation by rendering depth
values along rays cast from the camera center o through
each pixel xi. To effectively utilize depth information across
different spatial scales, we decompose both rendered images
and corresponding depth maps into non-overlapping patches,
as shown in Fig. 2. This approach enables fine-grained depth
supervision at multiple hierarchical levels, allowing the model
to capture both local geometric details and global structural
coherence. For spatial reshaping of Gaussian fields, we compute
a patch-wise depth that emphasizes the contribution of nearest
Gaussians by applying enhanced opacity values ω to all
primitives:
D(xi) =
X
n∈N
ω exp(−ι(n −1))G2D
n (xi)||µn −o||2,
where ι = −ln(1 −ω),
(4)
where µi denotes the center position of the n-th Gaussian
primitive. The term −ln(1 −ω) makes the nearest Gaussian
receive the highest weight, and farther Gaussians contribute
less proportionally.
Our approach establishes a dual-scale geometric supervision
paradigm that jointly optimizes hierarchical spatial relations
and enhances structural fidelity across scales. For each patch P,
we focus on capturing relative depth variations within localized
regions to enhance fine-grained geometric structure learning.:
Dpatch(x) = D(x) −µP
σP + δ
, where µP =
1
|P|
X
x∈P
D(x)
and σP =
s
1
|P|
X
x∈P
(D(x) −µP)2 ,
(5)
where x ∈P and δ ensures numerical stability. The proposed
fine-scale supervision strategy accentuates subtle depth varia-
tions that are typically diminished in coarse-grained processing,
ensuring accurate reconstruction of geometric structures. To
maintain overall scene coherence, we apply an image-based
normalization using image-wide statistics:
Dimage(x) = D(x) −µP
σI
,
where σI =
s
1
|I|
X
x∈I
(D(x) −µI)2 and µI = 1
|I|
X
x∈I
D(x).
(6)
Here, while maintaining patch-wise mean subtraction, we utilize
the global standard deviation from the entire image I. This
preserves the global depth distribution while allowing local
adaptivity. The depth supervision directly targets the mean
positions µ of 3D Gaussian primitives. By computing the
loss between rendered depth maps and Depth Anything V2
predictions eD at both scales, we create optimization gradients
that drive Gaussian centers toward geometrically consistent
positions.
The final depth supervision loss combines both hierarchical
scales through similarity losses computed on target areas:
Ldepth = γ∥Dpatch −eDpatch∥2
2 +η∥Dimage −eDimage∥2
2, (7)
where eDpatch and eDimage represent the correspondingly nor-
malized depth predictions from monocular depth estimator [54],

<!-- page 5 -->
5
and γ, η are weighting factors balancing local and global
supervision contributions. This hierarchical loss formulation
enables the model to learn accurate geometry at multiple scales
simultaneously, resulting in improved 3D scene reconstruction
quality.
C. Edge-Aware Depth Regularization
Accurately reconstructing fine geometric structures requires
preserving sharp depth discontinuities at object boundaries,
where conventional uniform regularization often leads to
undesirable blurring and structural degradation. To overcome
this fundamental limitation, we propose a novel edge-aware
depth regularization strategy that integrates structural priors
from the RGB image, enabling geometry-adaptive smoothness
control. Specifically, we leverage the inherent correlation
between image edges and depth discontinuities. A structural
edge map E was extracted using the Canny edge detector [15]
from the ground-truth RGB image I:
E = Canny(I),
(8)
where the lower and upper thresholds for Canny detection are
empirically set to 20 and 200, ensuring a robust yet sensitive
edge response that captures critical structural boundaries. To
focus smoothing on homogeneous regions, we construct a non-
edge mask by inverting E. We define an edge-aware binary
mask m(xi) as:
m(xi) =
(
1,
if E(xi) = 0,
0,
otherwise.
(9)
This binary mask ensures that regularization is applied only to
homogeneous regions, while structural boundaries are explicitly
preserved. Unlike traditional depth smoothness terms that
uniformly enforce local continuity, our method selectively
constrains smoothness only within non-edge regions. Let D(x)
denote the rendered depth at pixel x. For each pixel xi, we
define its local neighborhood N(xi) and compute the masked
local mean depth as:
D(xi) =
P
xj∈N (xi) D(xj) · m(xj)
P
xj∈N (xi) m(xj) + ϵ
,
(10)
where D(xi) represents the masked local mean depth within
a neighborhood centered at xi, and ϵ = 10−8 is a small
constant to ensure numerical stability. We adopt a cross-shaped
3 × 3 convolutional kernel (consisting of the center and four-
connected neighbors) to balance fine-grained detail preservation
and smoothing efficiency while reducing the risk of over-
smoothing near object boundaries. The proposed edge-aware
depth regularization loss is formulated as:
Ledge = 1
P
X
xi
m(xi) ·
D(xi) −D(xi)
2 ,
(11)
where P denotes the number of valid pixels considered
for regularization. This strategy establishes an explicit syn-
ergy between RGB structures and 3D depth geometry. By
incorporating structural priors into the depth regularization
process, we enable a content-aware smoothing mechanism
that dynamically adapts to the underlying scene geometry.
Unlike traditional uniform smoothing, our edge-aware approach
employs a structure-guided, masked, cross-shaped local mean
constraint that preserves boundary sharpness and enhances
geometric fidelity under sparse-view constraints.
D. RGB-Guided Edge-Preserving Total Variation Regulariza-
tion
While depth-based supervision focuses on recovering ac-
curate 3D structures, ensuring smoothness and preserving
visual details in the rendered images remains crucial for
achieving high-fidelity reconstructions. Conventional Total
Variation (TV) regularization promotes spatial smoothness by
penalizing large intensity differences between adjacent pixels.
However, standard TV loss tends to over-smooth edges and
fine structures, resulting in the loss of critical details in the
reconstructed images. To address this limitation, we propose
an RGB-guided edge-preserving Total Variation regularization
that selectively enforces smoothness constraints based on the
local gradient information from the ground-truth RGB image.
Let Ipred denote the rendered image and Igt the ground-truth
image, both defined over pixel grid Ω. Ωdenotes the set of all
pixel coordinates in the image domain, formally defined as:
Ω= {(a, b) | 1 ≤a ≤H, 1 ≤b ≤W} ,
(12)
where H and W are the height and width of the image,
respectively. For each pixel xi ∈Ω, we define the horizontal
and vertical gradients for the predicted and ground-truth images
as:
∇hIpred(xi) = Ipred(xi + 1h) −Ipred(xi),
∇hIgt(xi) = Igt(xi + 1h) −Igt(xi),
∇vIpred(xi) = Ipred(xi + 1v) −Ipred(xi),
∇vIgt(xi) = Igt(xi + 1v) −Igt(xi),
(13)
where 1h and 1v denote one-pixel shifts in the horizontal and
vertical directions, respectively. To avoid penalizing genuine
edges and sharp transitions, we introduce an edge-aware
modulation function based on the RGB gradients. Specifically,
for each direction, we construct binary masks Mh(xi) and
Mv(xi) as:
Mh(xi) = I (|∇hIgt(xi)| < τedge) ,
Mv(xi) = I (|∇vIgt(xi)| < τedge) ,
(14)
where τedge is a threshold controlling edge sensitivity, and I(·)
denotes the indicator function, returning 1 if the condition
holds and 0 otherwise. The edge-preserving Total Variation
loss is then defined as:
Ltv = 1
|Ω|
X
x∈Ω

Mh(x) · max (|∇hIpred(x)| −τsmooth, 0)
+ Mv(x) · max (|∇vIpred(x)| −τsmooth, 0)

,
(15)
where τsmooth is a small margin to tolerate minor gradient
fluctuations. We set τedge = 10−2 to filter out strong edges
and τsmooth = 10−4 to suppress small gradients. The final TV
regularization term is scaled and integrated into the overall
optimization objective. This formulation promotes smoothness

<!-- page 6 -->
6
TABLE I
QUANTITATIVE COMPARISON RESULTS ON THE MIP-NERF 360 [20], TANKS&TEMPLES [58] AND DEEPBLENDING [59] DATASETS. OUR METHOD
OUTPERFORMS BASELINES AND SOTA METHODS [7], [8], [10], [14]. SOME COMPETING METRICS ARE SOURCED FROM THE RESPECTIVE PAPERS.
Mip-NeRF 360
Tanks&Temples
Deep Blending
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Instant-NGP [5]
25.59
0.699
0.331
21.72
0.723
0.330
23.62
0.797
0.423
Plenoxels [23]
23.08
0.626
0.463
21.08
0.719
0.379
23.06
0.795
0.510
Mip-NeRF 360 [20]
27.69
0.792
0.237
22.22
0.759
0.257
29.40
0.901
0.245
3DGS [7]
27.79
0.826
0.202
23.13
0.840
0.182
29.42
0.905
0.241
DNGaussian [14]
27.87
0.828
0.195
23.75
0.846
0.178
29.65
0.904
0.240
Mip-Splatting [8]
27.79
0.827
0.203
23.86
0.853
0.175
29.71
0.904
0.242
Scaffold-GS [9]
27.98
0.824
0.207
23.93
0.854
0.176
30.20
0.905
0.253
Spec-Gaussian [10]
28.12
0.834
0.177
23.79
0.858
0.166
29.75
0.906
0.241
Ours
28.29
0.840
0.175
23.98
0.861
0.165
30.05
0.908
0.239
TABLE II
RESULTS ON NERF SYNTHETIC DATASET [1]. SOME COMPETING
METRICS ARE SOURCED FROM THE RESPECTIVE PAPERS.
NeRF Synthetic
PSNR ↑
SSIM ↑
LPIPS ↓
Instant-NGP [5]
33.18
0.963
0.045
Mip-NeRF [19]
33.09
0.961
0.043
Tri-MipRF [60]
33.65
0.963
0.042
3D-GS [7]
33.78
0.969
0.031
DNGaussian [14]
33.81
0.968
0.030
Mip-Splatting [8]
33.88
0.970
0.032
Scaffold-GS [9]
33.46
0.967
0.035
Spec-Gaussian [10]
34.05
0.970
0.029
Ours
34.13
0.972
0.028
TABLE III
RESULTS ON NSVF SYNTHETIC [22] DATASET. SOME COMPETING
METRICS ARE SOURCED FROM THE RESPECTIVE PAPERS.
NSVF Synthetic
PSNR ↑
SSIM ↑
LPIPS ↓
TensoRF [24]
36.52
0.982
0.026
Tri-MipRF [60]
34.58
0.973
0.030
NeuRBF [61]
37.80
0.986
0.019
3D-GS [7]
37.01
0.987
0.016
DNGaussian [14]
37.25
0.985
0.015
Mip-Splatting [8]
37.78
0.986
0.017
Scaffold-GS [9]
36.41
0.983
0.019
Spec-Gaussian [10]
38.30
0.987
0.013
Ours
38.39
0.988
0.012
in homogeneous regions while preserving edges and fine
details aligned with the ground-truth structure. By adapting the
regularization strength to local image gradients, our method
achieves a better balance between noise suppression and
detail preservation. In contrast to conventional TV loss that
uniformly penalizes intensity differences, our RGB-guided
strategy leverages structural cues to selectively smooth flat
areas without degrading important textures and boundaries,
resulting in more photorealistic rendering.
E. Training Losses
The overall training objective of our method consists of four
components: a color reconstruction loss Lcolor, a hierarchical
geometric depth supervision loss Ldepth, an edge-aware depth
regularization loss Ledge, and an RGB-guided edge-preserving
total variation loss Ltv. Following 3D Gaussian Splatting [7],
the color reconstruction loss combines an L1 reconstruction
loss and a D-SSIM term between the rendered image Ipred and
the ground-truth image Igt:
Lcolor = L1(Ipred, Igt) + λLD−SSIM(Ipred, Igt).
(16)
The final training loss function is defined as:
L = Lcolor + Ldepth + βLedge + ϕLtv.
(17)
IV. EXPERIMENTS
A. Dataset and Metrics
To comprehensively evaluate the effectiveness of our method,
we conduct experiments on five publicly available datasets,
including three real-world datasets and two synthetic ones.
These benchmarks are widely adopted in the field of novel view
synthesis and are used to assess the performance of baseline
methods such as 3DGS [7], Scaffold-GS [9], Mip-Splatting [8],
DNGaussian [14], and Spec-Gaussian [10], which represent
the current state of the art. Our evaluation includes nine scenes
from the Mip-NeRF 360 dataset [20], two scenes each from
the Tanks & Temples [58] and DeepBlending [59] datasets, as
well as eight scenes from both the NeRF Synthetic [1] and
NSVF Synthetic [22] datasets. These scenes were selected
to cover a broad spectrum of geometric and photometric
complexity, enabling us to assess the model’s robustness under
varying conditions. The dataset collection spans both indoor and
outdoor environments and includes scenes with diverse levels
of detail, providing a rigorous test of rendering adaptability
and scalability. Moreover, the benchmark suite consists of both
real-world captures and synthetic data.
Performance is evaluated using standard metrics for image-
based rendering: PSNR (Peak Signal-to-Noise Ratio), SSIM
(Structural Similarity Index) [62], and LPIPS (Learned Percep-
tual Image Patch Similarity) [63]. These metrics respectively
measure pixel-level fidelity, structural coherence, and perceptual
quality between the rendered and ground-truth images.
B. Baselines and Implementation
We evaluate our method against several state-of-the-art base-
lines, including 3DGS [7], Scaffold-GS [9], Mip-Splatting [8],
DNGaussian [14], and Spec-Gaussian [10], with the latter
showing top-tier performance in recent benchmarks. To further
broaden the comparison, we also include results from additional

<!-- page 7 -->
7
Ours
Spec-Gaussian
GT
Mip-Splatting
Scaffold-GS
Fig. 3. Qualitative Comparison Results Across Different Datasets [20], [58]. Zoomed-in regions showcase fine-grained rendering differences. Red arrows
mark visual artifacts, such as local blurriness, present in current state-of-the-art [8]–[10] approaches. Compared to these methods, our model achieves more
faithful detail preservation and delivers more realistic and high-fidelity renderings.
representative methods: Instant-NGP [5], Plenoxels [23], Mip-
NeRF [19], Mip-NeRF 360 [20], Tri-MipRF [60], TensoRF [24],
and NeuRBF [61]. All models are trained for 30k iterations to
ensure consistency across evaluations.
Our approach is implemented on top of the official PyTorch-
based 3D Gaussian Splatting framework. The proposed Hi-
erarchical Geometric Depth Supervision is invoked every 5
iterations. To mitigate over-constraining the learning process,
we introduce an error tolerance mechanism into the depth
L2 loss. The loss function parameters are fixed across all
experiments: γ = 0.1, η = 1, β = 0.1, and ϕ = 0.8. For
experiments on the Mip-NeRF 360 dataset [20], input images
are downsampled by a factor of 4 for outdoor scenes and by a
factor of 2 for indoor ones.
C. Results Comparison
a) Real-world Unbounded Datasets: To evaluate the
effectiveness of our method in real-world scenarios, we conduct
experiments on the Mip-NeRF 360 [20], Tanks and Tem-
ples [58], and DeepBlending [59] datasets. All baseline models
are trained using their official default configurations to ensure a
fair comparison. As shown in Tab. I, our approach consistently
outperforms previous state-of-the-art methods [7]–[10], [14]
across all datasets. Qualitative results shown in Fig. 3 further
demonstrate the visual benefits of our method. Compared to
existing models, our approach produces sharper details and
higher rendering quality results. This performance gain can be
attributed to the integration of depth-aware supervision, which
refines the spatial positioning of Gaussians in the early training
stages. With improved positional accuracy, the Gaussians can be
further optimized in subsequent steps to capture scene-specific
features better. This results in more accurate rendering with
fewer floating artifacts and improved structural consistency,
particularly in complex indoor and outdoor scenes. These
results highlight that our approach demonstrates a strong
ability to preserve fine-grained details and achieve high-fidelity
reconstructions across diverse real-world environments.
b) Synthetic Bounded Datasets: To validate the effective-
ness of our method under controlled conditions, we evaluate it
on two widely used synthetic datasets: NeRF Synthetic [1] and
NSVF [22]. The datasets exhibit clean geometry, consistent
lighting, and dense ground-truth views, making them appropri-
ate for evaluating fine-grained reconstruction performance. Our
comparisons were made with the most relevant state-of-the-art

<!-- page 8 -->
8
Ours
Spec-Gaussian
GT
DNGaussian
Mip-Splatting
Fig. 4. Qualitative Comparison Results of NeRF Synthetic Dataset [1]. The red boxes highlight detailed regions of the rendered images. With the aid of
depth and edge-aware supervision, as well as our proposed TV loss, our approach better preserves intricate visual structures and delivers more accurate and
detailed renderings than leading contemporary methods.
Ours
Ours w/o ℒ!"
GT
Ours w/o ℒ#$%!&
Ours w/o ℒ$#'$
26.09/0.179
25.85/0.194
25.29/0.221
26.54/0.151
Fig. 5. Ablation of Our Model Components. We conduct experiments that remove each of the three core innovations individually to assess their effectiveness.
We present the PSNR and LPIPS metrics, with the best scores highlighted in red.
methods, including point-based renderers such as 3DGS [7],
Scaffold-GS [9], Mip-Splatting [8], DNGaussian [14] and Spec-
Gaussian [10], as well as several NeRF-based approaches such
as TensoRF [24], NeuRBF [61], and Tri-MipRF [60]. As shown
in Tab. II and Tab. III, our method consistently outperforms all
baselines in terms of PSNR, SSIM, and LPIPS. This indicates
that our depth and edge aware regularization not only enhances
visual fidelity but also improves generalization in synthetic
environments. Visual results in Fig. 4 demonstrate clearer object
contours and sharper edges, particularly in complex scenes.
These improvements are attributed to better Gaussian alignment
with geometry, driven by hierarchical depth supervision and
edge-preserving regularization. In addition to per-view image
quality, we also observe improved multi-view consistency. Since
our approach enforces coherent structure during optimization,
it leads to fewer view-dependent artifacts and more stable
reconstructions across adjacent viewpoints. This is especially
evident in specular and thin-structure regions, where baseline
methods often struggle with geometry misalignment or color
bleeding. Together, these results highlight the strength of our
approach in maintaining both photometric accuracy and geo-
metric consistency across viewpoints, validating its robustness
in synthetic bounded scenarios.
c) Depth Map Comparison: Fig. 6 presents a comparison
of rendered depth maps between our method and existing state-
of-the-art approaches. Notably, the depth maps produced by
Spec-Gaussian [10] contain visible floaters and exhibit reduced
accuracy in object boundaries and fine geometry. In contrast,

<!-- page 9 -->
9
TABLE IV
ABLATION STUDY OF COMPONENTS IN OUR MODEL. WE PRESENT QUANTITATIVE RESULTS FOR TWO REAL-WORLD DATASETS [20], [58] AND ONE
SYNTHETIC DATASET [1] IN OUR ABLATION STUDY.
Mip-NeRF 360
Tanks&Temples
NeRF Synthetic
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
None
27.69
0.823
0.201
23.07
0.837
0.185
33.79
0.968
0.033
w/ Ldepth
28.20
0.834
0.183
23.88
0.853
0.173
34.07
0.971
0.029
w/ Ledge
28.09
0.828
0.188
23.69
0.846
0.178
33.99
0.970
0.031
w/ Ltv
27.89
0.825
0.195
23.57
0.841
0.181
33.90
0.970
0.032
Ours
28.29
0.840
0.175
23.98
0.861
0.165
34.13
0.972
0.028
Ours
Spec-Gaussian
Monocular Depth
Fig. 6. Depth Map Comparisons Between Our Method and State-of-the-art
models [10]. While Spec-Gaussian exhibits blurry floaters and less accurate
geometric details, our model generates sharper and structurally consistent
depth maps. This improvement is attributed to our Hierarchical Geometric
Depth Supervision and Edge-Aware Depth Regularization, which guide depth
refinement through local-global constraints and edge-preserving smoothing.
our method delivers clearer and more structurally faithful
depth maps. This is largely due to the proposed Hierarchical
Geometric Depth Supervision, which enforces both local and
global consistency during training. Additionally, the Edge-
Aware Depth Regularization, guided by Canny edge detection,
restricts smoothing to homogeneous regions while preserving
sharp transitions at semantic boundaries. Unlike conventional
depth regularization that uniformly enforces continuity, our
approach selectively encourages smoothness only where it is
structurally appropriate.
D. Ablation Study
To better understand the contributions of each component
in our framework, we conduct an ablation study across three
datasets: Mip-NeRF 360 [20], Tanks & Temples [58], and
NeRF Synthetic [1]. We evaluate four variants of our model:
without Hierarchical Geometric Depth Supervision (Ldepth),
without Edge-Aware Depth Regularization (Ledge), without
RGB-Guided Edge-Preserving TV Regularization (Ltv), and
the full model with all components enabled. The results are
summarized in Tab. IV.
a) Hierarchical Geometric Depth Supervision: Among
all components, removing Ldepth leads to the most significant
performance drop across all datasets. This validates the critical
role of hierarchical depth cues in guiding the optimization
of Gaussian positions. By leveraging both local patch-wise
and global image-level depth normalization, this supervision
strategy effectively improves geometric alignment, enabling
the model to better capture scene structure. As shown in
Fig. 5 (bicycle scene, Mip-NeRF 360), the absence of depth
supervision causes substantial degradation: floating artifacts
become prominent, and scene geometry appears misaligned.
The PSNR and LPIPS scores also drop markedly, indicating a
loss in both numerical and perceptual fidelity.
b) Edge-Aware Depth Regularization: When Ledge is
removed, the rendering quality moderately degrades. Without
the guidance of image-space edges, depth supervision becomes
overly aggressive in smooth regions and fails to preserve
structural boundaries. This leads to excessive smoothing near
object edges, as depth values are inaccurately propagated across
semantic boundaries. In Fig. 5, this is reflected in the blurred
contours of the bicycle and reduced sharpness around object
boundaries. The regularization provided by Canny-based edge
priors proves essential in preserving high-frequency geometric
features while avoiding depth overshoot in ambiguous regions.
c) RGB-Guided Edge-Preserving Total Variation Regu-
larization: The absence of Ltv results in subtle but noticeable
quality degradation. Without this regularization, small texture-
level inconsistencies emerge, especially in homogeneous sur-
faces. These inconsistencies are often manifested as noise or
minor ringing artifacts in smooth regions of the rendered images.
While the overall structural alignment remains mostly intact,
the visual cleanliness and perceptual smoothness decrease. As
seen in Fig. 5, the variant without Ltv introduces slight visual
clutter in flat background areas, leading to lower LPIPS and
marginal PSNR drops.
TABLE V
QUANTITATIVE RESULTS OF OUR MODEL ACROSS DIFFERENT Patch Size
SETTINGS.
Dataset
Patch Size
PSNR ↑
SSIM ↑
LPIPS ↓
0
27.71
0.823
0.205
Mip-NeRF 360
5
28.21
0.833
0.181
10
28.18
0.830
0.183
20
28.20
0.832
0.185
0
23.23
0.843
0.181
Tanks&Temples
5
23.84
0.852
0.172
10
23.86
0.854
0.175
20
23.88
0.856
0.174

<!-- page 10 -->
10
E. Discussion
Monocular Depth Estimator. For monocular depth predic-
tion, we employ the pre-trained Depth Anything V2 model [54],
which represents the current state-of-the-art in monocular
depth estimation. While DPT [52] has been widely used in
NeRF-based pipelines, we choose Depth Anything V2 due
to its superior depth prediction quality and generalization
performance. While its best-performing “giant” variant is not
publicly released, we utilize the publicly available “large”
version (vitl), which offers a favorable trade-off between
performance and accessibility. This estimator is used to generate
initial depth maps that guide our regularization modules during
training.
Patch Size Selection. In our experiments, we investigate
the impact of different patch sizes on model performance. As
shown in Tab V, varying the patch size yields comparable
results, indicating that the model is not highly sensitive to this
parameter. This robustness arises from the relative formulation
of our patch-based normalization, as well as the localized
nature of our loss functions across multiple spatial scales. To
further enhance generalization and avoid overfitting to a fixed
patch resolution, we adopt a randomized patch size sampling
strategy during training. Specifically, patch sizes are uniformly
sampled from a predefined range ([5, 20]), which introduces
diversity in spatial context and encourages stability across
varying granularities.
F. Limitation and Future Work
While our method improves rendering fidelity, certain
limitations remain. First, the effectiveness of depth-based
supervision relies on the quality of the estimated depth maps,
which may be inaccurate in texture-less or occluded regions.
This can introduce biases into Gaussian placement, particularly
in complex scenes. Second, although our edge-aware and TV-
based regularizations enhance structural detail, they are still
hand-crafted components and may not generalize optimally
across all scene types or lighting conditions. In future work,
we plan to explore adaptive or learned regularization strategies
that can dynamically adjust to scene content. Incorporating
confidence-aware depth supervision or leveraging multi-modal
inputs (e.g., surface normals or segmentation cues) may further
improve robustness in challenging scenarios such as reflections,
transparency, or thin structures.
V. CONCLUSION
In this paper, we proposed a novel depth- and edge-guided
framework for 3D Gaussian Splatting, designed to improve
rendering fidelity and geometric consistency in novel view
synthesis tasks. Our method introduces three key components:
Hierarchical Geometric Depth Supervision for multi-scale
structural alignment, Edge-Aware Depth Regularization to
preserve semantic boundaries, and an RGB-Guided Edge-
Preserving TV loss to suppress visual artifacts while main-
taining texture details. Extensive experiments on both real-
world and synthetic datasets demonstrate that our approach
consistently outperforms existing state-of-the-art methods in
terms of image quality and structural accuracy. Ablation studies
further validate the individual contributions of each component
to the overall performance. Our method effectively enhances the
spatial precision of Gaussians, reduces floaters, and achieves
superior multi-view consistency. This study emphasises the
advantages of integrating geometric guidance into point-based
rendering, representing a significant advancement towards
enhanced accuracy and reliability in 3D reconstruction.
APPENDIX
MORE EXPERIMENTS RESULT
In the appendix, we provide detailed per-scene evaluation re-
sults on the Mip-NeRF 360 dataset [20], comparing our method
with baseline models. As shown, our approach consistently
outperforms existing 3DGS-based methods [7]–[10], [14] in
terms of PSNR, SSIM [62] and LPIPS [63] across most scenes.
TABLE VI
PSNR COMPARISON ON THE MIP-NERF 360 DATASET.
bicycle
flowers
garden
stump
treehill
room
counter
kitchen
bonsai
Instant-NGP
22.17
20.65
25.07
23.47
22.37
29.69
26.69
29.48
30.69
Plenoxels
21.91
20.10
23.49
20.66
22.25
27.59
23.62
23.42
24.67
Mip-NeRF360
24.37
21.73
26.98
26.40
22.87
31.63
29.55
32.23
33.46
3D-GS
25.63
21.94
27.73
27.02
22.79
31.80
29.12
31.61
32.48
DNGaussian
25.69
21.89
27.75
27.03
22.90
31.79
29.15
31.77
32.88
Mip-Splatting
25.72
21.93
27.76
26.94
22.98
31.74
29.16
31.55
32.31
Scaffold-GS
25.61
21.74
27.82
26.79
23.38
32.14
29.62
31.81
32.87
Spec-Gaussian
25.89
21.85
28.07
27.14
22.57
32.03
29.92
32.25
33.34
Ours
25.95
21.98
28.09
27.15
23.41
32.09
30.07
32.46
33.40
TABLE VII
SSIM COMPARISON ON THE MIP-NERF 360 DATASET.
bicycle
flowers
garden
stump
treehill
room
counter
kitchen
bonsai
Instant-NGP
0.512
0.486
0.701
0.594
0.542
0.871
0.817
0.858
0.906
Plenoxels
0.496
0.431
0.606
0.523
0.509
0.842
0.759
0.648
0.814
Mip-NeRF360
0.685
0.583
0.813
0.744
0.632
0.913
0.894
0.920
0.941
3D-GS
0.778
0.623
0.874
0.784
0.651
0.928
0.916
0.933
0.948
DNGaussian
0.779
0.635
0.877
0.790
0.654
0.927
0.917
0.932
0.944
Mip-Splatting
0.780
0.623
0.875
0.786
0.655
0.928
0.916
0.933
0.948
Scaffold-GS
0.773
0.609
0.867
0.774
0.657
0.931
0.919
0.931
0.950
Spec-Gaussian
0.796
0.648
0.881
0.795
0.645
0.934
0.922
0.937
0.952
Ours
0.801
0.655
0.882
0.805
0.661
0.939
0.924
0.939
0.953
TABLE VIII
LPIPS COMPARISON ON THE MIP-NERF 360 DATASET.
bicycle
flowers
garden
stump
treehill
room
counter
kitchen
bonsai
Instant-NGP
0.446
0.441
0.257
0.421
0.450
0.261
0.306
0.195
0.205
Plenoxels
0.506
0.521
0.386
0.503
0.540
0.419
0.441
0.447
0.398
Mip-NeRF360
0.301
0.344
0.170
0.261
0.339
0.211
0.204
0.127
0.176
3D-GS
0.204
0.328
0.103
0.207
0.318
0.191
0.178
0.113
0.173
DNGaussian
0.187
0.321
0.101
0.206
0.310
0.186
0.169
0.110
0.169
Mip-Splatting
0.206
0.331
0.103
0.209
0.320
0.192
0.179
0.113
0.173
Scaffold-GS
0.224
0.339
0.112
0.228
0.315
0.182
0.177
0.114
0.174
Spec-Gaussian
0.166
0.264
0.092
0.186
0.271
0.177
0.166
0.108
0.162
Ours
0.165
0.263
0.093
0.181
0.269
0.173
0.164
0.105
0.159
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[2] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, “Mvsnerf:
Fast generalizable radiance field reconstruction from multi-view stereo,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2021, pp. 14 124–14 133.
[3] J. Yang, M. Pavone, and Y. Wang, “Freenerf: Improving few-shot neural
rendering with free frequency regularization,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2023,
pp. 8254–8263.
[4] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance
fields from one or few images,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2021, pp. 4578–
4587.

<!-- page 11 -->
11
[5] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[6] C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Super-
fast convergence for radiance fields reconstruction,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5459–5469.
[7] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[8] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-splatting: Alias-
free 3d gaussian splatting,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2024, pp. 19 447–19 456.
[9] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[10] Z. Yang, X. Gao, Y.-T. Sun, Y. Huang, X. Lyu, W. Zhou, S. Jiao, X. Qi,
and X. Jin, “Spec-gaussian: Anisotropic view-dependent appearance
for 3d gaussian splatting,” Advances in Neural Information Processing
Systems, vol. 37, pp. 61 192–61 216, 2024.
[11] J. Chung, J. Oh, and K. M. Lee, “Depth-regularized optimization for 3d
gaussian splatting in few-shot images,” in 2024 IEEE/CVF Conference
on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024,
pp. 811–820.
[12] H. Yu, X. Long, and P. Tan, “Lm-gaussian: Boost sparse-view 3d gaussian
splatting with large model priors,” arXiv preprint arXiv:2409.03456,
2024.
[13] L. Shen, T. Liu, H. Sun, J. Li, Z. Cao, W. Li, and C. C. Loy, “Dof-gaussian:
Controllable depth-of-field for 3d gaussian splatting,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
26 462–26 471.
[14] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
“Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with
global-local depth normalization,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2024, pp. 20 775–
20 785.
[15] J. Canny, “A computational approach to edge detection,” IEEE Transac-
tions on pattern analysis and machine intelligence, no. 6, pp. 679–698,
1986.
[16] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove,
“Deepsdf: Learning continuous signed distance functions for shape
representation,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2019, pp. 165–174.
[17] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus:
Learning neural implicit surfaces by volume rendering for multi-view
reconstruction,” NeurIPS, 2021.
[18] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and
L. Liu, “Neus2: Fast learning of neural implicit surfaces for multi-
view reconstruction,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2023, pp. 3295–3306.
[19] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and
P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-aliasing
neural radiance fields,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2021, pp. 5855–5864.
[20] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in
Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[21] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, “Plenoctrees
for real-time rendering of neural radiance fields,” in Proceedings of
the IEEE/CVF international conference on computer vision, 2021, pp.
5752–5761.
[22] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt, “Neural sparse
voxel fields,” Advances in Neural Information Processing Systems, vol. 33,
pp. 15 651–15 663, 2020.
[23] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa,
“Plenoxels: Radiance fields without neural networks,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5501–5510.
[24] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in European conference on computer vision.
Springer, 2022,
pp. 333–350.
[25] Z. Huang, S. M. Erfani, S. Lu, and M. Gong, “Efficient neural implicit
representation for 3d human reconstruction,” Pattern Recognition, vol.
156, p. 110758, 2024.
[26] L. Xu, V. Agrawal, W. Laney, T. Garcia, A. Bansal, C. Kim, S. Rota Bul`o,
L. Porzi, P. Kontschieder, A. Boˇziˇc et al., “Vr-nerf: High-fidelity
virtualized walkable spaces,” in SIGGRAPH Asia 2023 Conference Papers,
2023, pp. 1–12.
[27] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Zip-nerf: Anti-aliased grid-based neural radiance fields,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 19 697–19 705.
[28] Y. Liao, J. Xie, and A. Geiger, “Kitti-360: A novel dataset and benchmarks
for urban scene understanding in 2d and 3d,” IEEE Transactions on
Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3292–3310,
2022.
[29] J. L. Sch¨onberger, E. Zheng, J.-M. Frahm, and M. Pollefeys, “Pixelwise
view selection for unstructured multi-view stereo,” in Computer Vision–
ECCV 2016: 14th European Conference, Amsterdam, The Netherlands,
October 11-14, 2016, Proceedings, Part III 14.
Springer, 2016, pp.
501–518.
[30] D. R¨uckert, L. Franke, and M. Stamminger, “Adop: Approximate
differentiable one-pixel point rendering,” ACM Transactions on Graphics
(ToG), vol. 41, no. 4, pp. 1–14, 2022.
[31] G. Kopanas, J. Philip, T. Leimk¨uhler, and G. Drettakis, “Point-based
neural rendering with per-view optimization,” in Computer Graphics
Forum, vol. 40, no. 4.
Wiley Online Library, 2021, pp. 29–43.
[32] O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, “Synsin: End-to-end
view synthesis from a single image,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp. 7467–
7477.
[33] G. Arvanitis, E. I. Zacharaki, L. V´aˆsa, and K. Moustakas, “Broad-to-
narrow registration and identification of 3d objects in partially scanned
and cluttered point clouds,” IEEE Transactions on Multimedia, vol. 24,
pp. 2230–2245, 2021.
[34] Z. Huang, M. Xu, and S. Perry, “Structgs: Adaptive spherical harmonics
and rendering enhancements for superior 3d gaussian splatting,” arXiv
preprint arXiv:2503.06462, 2025.
[35] ——, “Gaussianfocus: Constrained attention focus for 3d gaussian
splatting,” arXiv preprint arXiv:2503.17798, 2025.
[36] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3d gaussian
splatting for efficient dynamic scene reconstruction,” IEEE Transactions
on Circuits and Systems for Video Technology, 2024.
[37] T. Zhou, S. Chen, S. Wan, H. Lv, Z. Luo, and J. Wu, “Gedr: Gaussian-
enhanced detail reconstruction for real-time high-fidelity 3d scene
reconstruction,” IEEE Transactions on Circuits and Systems for Video
Technology, 2025.
[38] W. Li, X. Pan, J. Lin, P. Lu, D. Feng, and W. Shi, “Frpgs: Fast, robust, and
photorealistic monocular dynamic scene reconstruction with deformable
3d gaussians,” IEEE Transactions on Circuits and Systems for Video
Technology, 2025.
[39] H. Yu, W. Gong, J. Chen, and H. Ma, “Get3dgs: Generate 3d gaussians
based on points deformation fields,” IEEE Transactions on Circuits and
Systems for Video Technology, 2024.
[40] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu,
“Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-
time human novel view synthesis,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
[41] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, and S. Tang, “3dgs-avatar:
Animatable avatars via deformable 3d gaussian splatting,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
2024, pp. 5020–5030.
[42] Z. Yan, W. F. Low, Y. Chen, and G. H. Lee, “Multi-scale 3d gaussian
splatting for anti-aliased rendering,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
20 923–20 931.
[43] J. Liu, L. Kong, J. Yan, and G. Chen, “Mesh-aligned 3d gaussian
splatting for multi-resolution anti-aliasing rendering,” IEEE Transactions
on Circuits and Systems for Video Technology, 2025.
[44] C. Wang, X. Wang, J. Zhang, L. Zhang, X. Bai, X. Ning, J. Zhou,
and E. Hancock, “Uncertainty estimation for stereo matching based on
evidential deep learning,” pattern recognition, vol. 124, p. 108498, 2022.
[45] X. Wang, H. Luo, Z. Wang, J. Zheng, and X. Bai, “Robust training
for multi-view stereo networks with noisy labels,” Displays, vol. 81, p.
102604, 2024.
[46] Z. Wang, H. Luo, X. Wang, J. Zheng, X. Ning, and X. Bai, “A contrastive
learning based unsupervised multi-view stereo with multi-stage self-
training strategy,” Displays, vol. 83, p. 102672, 2024.
[47] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised nerf:
Fewer views and faster training for free,” in Proceedings of the IEEE/CVF

<!-- page 12 -->
12
conference on computer vision and pattern recognition, 2022, pp. 12 882–
12 891.
[48] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, and M. Nießner,
“Dense depth priors for neural radiance fields from sparse input views,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 892–12 901.
[49] J. Song, S. Park, H. An, S. Cho, M.-S. Kwak, S. Cho, and S. Kim,
“D¨arf: Boosting radiance fields from sparse input views with monocular
depth adaptation,” Advances in Neural Information Processing Systems,
vol. 36, pp. 68 458–68 470, 2023.
[50] G. Wang, Z. Chen, C. C. Loy, and Z. Liu, “Sparsenerf: Distilling
depth ranking for few-shot novel view synthesis,” in Proceedings of
the IEEE/CVF international conference on computer vision, 2023, pp.
9065–9076.
[51] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, “Monosdf:
Exploring monocular geometric cues for neural implicit surface recon-
struction,” Advances in neural information processing systems, vol. 35,
pp. 25 018–25 032, 2022.
[52] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense
prediction,” in Proceedings of the IEEE/CVF international conference
on computer vision, 2021, pp. 12 179–12 188.
[53] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth
anything: Unleashing the power of large-scale unlabeled data,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 10 371–10 381.
[54] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao,
“Depth anything v2,” Advances in Neural Information Processing Systems,
vol. 37, pp. 21 875–21 911, 2024.
[55] H. Xiong, SparseGS: Real-time 360° sparse view synthesis using
Gaussian splatting.
University of California, Los Angeles, 2024.
[56] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, “Fsgs: Real-time few-shot view
synthesis using gaussian splatting,” in European conference on computer
vision.
Springer, 2024, pp. 145–163.
[57] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[58] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
[59] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Bros-
tow, “Deep blending for free-viewpoint image-based rendering,” ACM
Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1–15, 2018.
[60] W. Hu, Y. Wang, L. Ma, B. Yang, L. Gao, X. Liu, and Y. Ma, “Tri-miprf:
Tri-mip representation for efficient anti-aliasing neural radiance fields,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 19 774–19 783.
[61] Z. Chen, Z. Li, L. Song, L. Chen, J. Yu, J. Yuan, and Y. Xu, “Neurbf:
A neural fields representation with adaptive radial basis functions,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 4182–4194.
[62] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[63] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
