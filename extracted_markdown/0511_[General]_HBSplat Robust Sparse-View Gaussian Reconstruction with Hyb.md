<!-- page 1 -->
1
HBSplat: Robust Sparse-View Gaussian Reconstruction with
Hybrid-Loss Guided Depth and Bidirectional Warping
Yu Ma
, Guoliang Wei
, Haihong Xiao
, Yue Cheng
,
Abstract—Novel View Synthesis (NVS) from sparse views
presents a formidable challenge in 3D reconstruction, where lim-
ited multi-view constraints lead to severe overfitting, geometric
distortion, and fragmented scenes. While 3D Gaussian Splatting
(3DGS) delivers real-time, high-fidelity rendering, its perfor-
mance drastically deteriorates under sparse inputs, plagued
by floating artifacts and structural failures. To address these
challenges, we introduce HBSplat, a unified framework that
elevates 3DGS by seamlessly integrating robust structural cues,
virtual view constraints, and occluded region completion. Our
core contributions are threefold: a Hybrid-Loss Depth Estimation
module that ensures multi-view consistency by leveraging dense
matching priors and integrating reprojection, point propagation,
and smoothness constraints; a Bidirectional Warping Virtual
View Synthesis method that enforces substantially stronger con-
straints by creating high-fidelity virtual views through bidi-
rectional depth-image warping and multi-view fusion; and an
Occlusion-Aware Reconstruction component that recovers oc-
cluded areas using a depth-difference mask and a learning-
based inpainting model. Extensive evaluations on LLFF, Blender,
and DTU benchmarks validate that HBSplat sets a new state-
of-the-art, achieving up to 21.13 dB PSNR and 0.189 LPIPS,
while maintaining real-time inference. Code is available at:
https://github.com/eternalland/HBSplat.
Index Terms—Novel View Synthesis, Sparse Views, 3D Gaus-
sian Splatting, Depth Estimation, Bidirectional Warping
I. INTRODUCTION
N
OVEL View Synthesis (NVS) aims to generate realistic
3D scenes from 2D images, serving as a cornerstone for
applications in virtual reality [1], robotics [2], and autonomous
driving [3]. While Neural Radiance Fields (NeRF) [4] achieve
photorealistic quality through implicit neural modeling, they
require dozens of input views and suffer from high compu-
tational costs, hindering real-time use. 3D Gaussian Splatting
(3DGS) [5] emerged as an efficient explicit alternative, lever-
aging optimizable Gaussians and differentiable rasterization
for high-fidelity, real-time rendering. However, under sparse-
view conditions (e.g., 3–5 images), insufficient multi-view
constraints cause 3DGS to overfit, leading to geometric distor-
tions, floating artifacts, background collapse, and incomplete
occlusion handling, especially in complex 360° scenes.
NVS under sparse-view conditions primarily grapples with
three fundamental challenges: (1) Inadequate Geometry Ini-
tialization: Vanilla 3DGS relies on COLMAP-generated sparse
point clouds for initializing Gaussian primitives. Under sparse
inputs, these points become insufficient and noisy, failing to
Yu Ma, Guoliang Wei and Yue Cheng are with the University of Shang-
hai for Science and Technology, Shanghai 200093, China, (e-mail: yum-
mychina@163.com; guoliang.wei@usst.edu.cn; chengyue916@163.com).
Haihong Xiao is with the Hefei University of Technology, Hefei 230009,
China, (e-mail: hhxiao@hfut.edu.cn).
(a) Under Sparse-View Reconstruction: Rendered images and depth maps
comparison between MCGS (left) and HBSplat (right).
(b) Training Efficiency vs. Rendering Quality.
Fig. 1. (a) MCGS outputs (left) vs. HBSplat outputs (right) for both rendered
image and depth map. (b) presents an efficiency-quality scatter plot comparing
HBSplat with various baseline methods.
support meaningful Gaussian reconstruction. This often causes
primitives to over-scale in an attempt to cover surrounding
regions, rather than densify appropriately through cloning or
splitting under sparse conditions, resulting in blurry and un-
stable geometry. (2) Under-Constrained Optimization: Limited
training views provide weak supervisory signals, leading to
overfitting and poor generalization. The model often produces
floaters and inconsistent geometry when rendered from novel
poses. (3) Occlusion and Unseen Regions: Areas unobserved
in all training views cannot be recovered without strong priors
or explicit reasoning, resulting in holes or incorrect content in
disoccluded regions.
To address the challenges of sparse-view NVS, recent
arXiv:2509.24893v3  [cs.CV]  8 Oct 2025

<!-- page 2 -->
2
studies have proposed various approaches [6]–[12]. NeRF-
based methods mitigate overfitting through depth regulariza-
tion, semantic consistency, and pre-trained priors, but they
remain constrained by long training times and slow infer-
ence. 3DGS-based methods exploit the flexibility of explicit
representations, exploring depth-guided optimization, multi-
view consistency constraints, and diffusion model priors,
significantly improving efficiency and quality. For instance,
depth regularization methods [11] provide geometric priors
via monocular depth estimation or Multi-View Stereo (MVS)
with virtual view generation and co-regularization enhance
multi-view consistency. Nevertheless, existing methods still
face limitations in geometric consistency, occlusion handling,
and detail recovery under sparse views, particularly in complex
scenes, where scale inconsistencies in monocular depth priors
and low overlap in MVS can lead to inaccurate geometry.
To tackle these challenges, we propose HBSplat, a novel
3DGS-based framework designed for high-quality and efficient
novel view synthesis from sparse input views. The framework
introduces three key innovations: First, a Hybrid-Loss Depth
Estimation module utilizes a dense matcher to extract suf-
ficient initialization points, followed by a novel composite
loss integrating reprojection, point propagation, and Total
Variation (TV) smoothness constraints to obtain robust depth
estimates, significantly enhancing multi-view consistency. The
point propagation constraint specifically strengthens geomet-
ric consistency by exploiting indirect correspondences across
multiple views (e.g., deriving constraints between points a
and c via their common matches to point b in overlapping
images). Second, a Bidirectional Warping Virtual View Syn-
thesis strategy is employed to generate photometrically and
geometrically consistent virtual views, providing additional su-
pervision signals to improve robustness. Third, an Occlusion-
Aware Reconstruction component addresses occlusion chal-
lenges through depth-difference-based foreground mask and a
dedicated inpainting model, effectively reconstructing missing
regions by prioritizing background content. Figure 1 presents
a comparison between HBSplat and other baseline methods.
The main contributions of this work are the following:
• We introduce a comprehensive framework that seamlessly
integrates depth estimation, virtual view synthesis, and
occlusion-aware processing into 3DGS to achieve high-
quality NVS from sparse inputs.
• We develop a robust Hybrid-Loss Depth Estimation
module that leverages dense matching priors through a
composite loss function, enforcing multi-view geometric
consistency by combining reprojection, point propaga-
tion, and Total Variation constraints.
• We propose a Bidirectional Warping strategy to gener-
ate photometrically and geometrically consistent virtual
training views, thereby significantly enhancing recon-
struction quality.
• We design an innovative Occlusion-Aware Reconstruction
mechanism that addresses severe occlusions through a
depth-difference-based foreground mask and a learning-
based inpainting model to effectively recover background
content in heavily occluded regions.
Extensive experiments on LLFF, IBRNet, Blender, DTU,
and Tanks&Temples show that HBSplat achieves new state-
of-the-art performance across metrics including PSNR (up to
21.13dB) and LPIPS (as low as 0.189), while maintaining real-
time rendering speeds. The framework demonstrates remark-
able robustness across a spectrum of settings, from forward-
facing to 360° scenes.
II. RELATED WORKS
1) NeRF-Based NVS Methods: NeRF [4] implicitly repre-
sent scenes via Multi-Layer Perceptrons (MLPs), achieving
photorealistic NVS through volume rendering and becoming
a dominant approach in recent years. Subsequent improve-
ments include Mip-NeRF [13], which mitigates aliasing via
conical frustum rendering; Mip-NeRF360 [14], extending the
framework to 360° unbounded scenes; and InstantNGP [15],
which reduces training time to seconds using multi-resolution
hash encoding. However, these methods generally require
dense view inputs (often dozens of images) and suffer from
slow training and inference. Moreover, in sparse-view settings,
NeRF-based approaches are prone to overfitting, resulting in
blurry or inconsistent novel views.
To address sparse-view challenges, various regularization
strategies have been proposed. RegNeRF [16] mitigates over-
fitting with color and depth consistency regularization for
unseen views, DietNeRF [6] enforces semantic consistency
using CLIP [17] embeddings, and SparseNeRF [7] enhances
geometry with scale-invariant losses from monocular depth
estimation. ViP-NeRF [18] and FreeNeRF [8] improve training
stability via visibility regularization and frequency control,
respectively. ReconFusion [19] integrates diffusion models to
generate additional views while struggling with view con-
sistency. Pose-free reconstruction methods like iNeRF [20],
NeRFmm [21], and BARF [22] jointly optimize camera poses
and scene representations but are limited under complex
camera trajectories (e.g., 360° scenes). These methods rely on
volume rendering, resulting in low efficiency unsuitable for
real-time applications.
2) 3DGS-Based NVS Methods: 3DGS represents scenes
with explicit Gaussian primitives, achieving real-time ren-
dering via differentiable rasterization, surpassing NeRF in
efficiency and quality. 3DGS excels with dense views, support-
ing tasks like text-to-3D generation [23] and dynamic scene
modeling [24], [25]. However, in sparse views, 3DGS suffers
from insufficient initial point clouds and limited multi-view
constraints, leading to artifacts and geometry degradation.
Recent studies optimize sparse-view 3DGS with various
strategies. FSGS [26] enhances geometry via unpooling and
monocular depth priors but suffers from long training times,
limiting its applicability in time-sensitive scenarios. Gaus-
sianObject [10] proposes structure-prior-aided initialization,
needing only 4 views, and uses diffusion models to repair oc-
cluded regions. DNGaussian [27] optimizes Gaussian positions
with global-local depth normalization, addressing scale incon-
sistencies. CoherentGS [28] enhances Gaussian consistency
using an implicit decoder and smoothness loss, filling occluded
regions. CoR-GS [29] suppresses Gaussian field disagreements

<!-- page 3 -->
3
Fig. 2. HBSplat pipeline. First, sparse input images are processed by dense matching, structure from motion (SfM), and monocular depth estimation to obtain
correspondences, camera poses, and depth maps. The Hybrid-Loss module fuses these inputs to produce robust point-wise depths. Subsequent least-squares
optimization aligns the point cloud with monocular depths, recovering the metric scale. The Bidirectional Warping module leverages these aligned depths
and images to synthesize novel virtual training views through depth-image warping and interpolation. Simultaneously, the Occlusion-Aware Reconstruction
component restores missing background content in occluded regions using learning-based inpainting model (Simple-LAMA) guided by local foreground mask.
Finally, the framework reconstructs the 3D Gaussian scene by optimizing a joint loss that combines color and depth supervision.
via co-regularization, improving geometric consistency. LM-
Gaussian [30] integrates stereo priors [31] and diffusion mod-
els [32] for iterative detail refinement. MVPGS [33] leverages
MVS [34] and Vision Transformers (ViTs) [35] to excavate
multi-view cues, optimizing initialization and appearance.
MCGS [36] introduces a sparse matcher and progressive prun-
ing for multi-view consistency. SCGaussian [37]introduces a
hybrid Gaussian representation with structure-consistent op-
timization and matching priors to enhance 3D consistency.
Binocular-Guided 3DGS [38] constrains rendered depths with
binocular stereo consistency, improving view consistency. SID
[39], as an enhanced version of FSGS, incorporating semantic
regularization from DINO-ViT [40] features and local depth
constraints. While existing methods have made progress in
specific areas, fundamental challenges including monocular
depth scale inconsistencies, robust occlusion handling, and
high-fidelity detail recovery in complex scenes have yet to
be adequately resolved.
Our HBSplat method presents a unified approach for high-
quality sparse-view synthesis, integrating three core technical
contributions that advance beyond previous methods. The
proposed Hybrid-Loss Depth Estimation incorporates multi-
view geometric constraints including point propagation, offer-
ing stronger robustness than SCGaussian [37], which relies
only on reprojection error. The Bidirectional Warping method
produces more complete virtual views compared to the forward
warping used in MVPGS [33], avoiding hole-filling and post-
inpainting. Furthermore, the Occlusion-Aware Reconstruction
module performs consistent occlusion handling within a sin-
gle training stage, unlike the complex two-stage diffusion-
based refinement in LM-Gaussian [30]. The unified framework
jointly addresses depth, consistency, and occlusion challenges
under extreme sparsity, demonstrating clear improvements in
rendering quality and operational efficiency.
III. METHODS
This section details the HBSplat framework, which is
structured as follows: III-A. Preliminaries outlines the 3DGS
fundamentals. III-B. Dense matching for sufficient initial-
ization, ray-based 3DGS Optimization reduces degrees of
freedom to prevent overfitting. III-C. Hybrid-Loss Depth
Estimation improves geometric consistency via hybrid loss
fusion. III-D. Bidirectional Warping generates virtual views
to enhance photometric and geometric constraints. III-E.
Occlusion-Aware Reconstruction handles occlusions using
depth-difference mask and inpainting model. Figure 2 illus-
trates the pipeline of the proposed HBSplat framework.
A. Preliminary for 3D Gaussian Splatting
HBSplat leverages 3DGS [5] to represent 3D scenes with
explicit Gaussian primitives, enabling real-time NVS through
differentiable rasterization. Each Gaussian primitive is defined
by a set of differentiable parameters: position vector µ, rota-
tion quaternion q, scaling matrix s, opacity α, and spherical
harmonic coefficients sh for view-dependent color, denoted
as G = {Gi : µi, qi, si, αi, shi}P
i=1. The position and shape of
each Gaussian follow a Gaussian distribution, formulated as:
G(x) = e−1
2 (x−µ)T Σ−1(x−µ).
(1)
To ensure the covariance matrix Σ is positive semi-definite,
it is decomposed into a scaling matrix s and a rotation
matrix R represented by quaternion q, reducing optimization
complexity:
Σ = RSST RT .
(2)
For projection into the image plane, the view transformation
matrix W and the projective transformation Jacobian J are
applied to map the 3D Gaussian as follows:
Σ′ = JWΣW T JT .
(3)

<!-- page 4 -->
4
Pixel color C is computed via volume rendering, blending
N ordered overlapping Gaussians:
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −α′
j),
(4)
where ci is the color of the i-th Gaussian. α′
j = αjG(P) is
derived from the projected 2D Gaussian Σ′. Pixel depth D is
similarly computed:
D =
X
i∈N
diαi
i−1
Y
j=1
(1 −α′
j),
(5)
where di is the depth of the i-th Gaussian.
B. Dense Matching and Ray-Based Gaussian Splatting
Vanilla 3DGS relies on sparse 3D points from COLMAP,
which become critically insufficient under sparse-view condi-
tions, leading to initialization failure and overfitting. Although
some methods employ MVSNet [41] or VGGT [42] to gener-
ate dense point clouds for initialization, they still suffer from
multi-stage cascading errors or scale inaccuracies. Here, we
customize two simple yet effective core prior enhancements.
Dense Matching for Sufficient Initialization: The standard
sparse feature matching in COLMAP is replaced with a
dense matching network. By leveraging inter-image correla-
tions rather than independently extracted feature points, this
approach produces denser and more reliable correspondences,
significantly improving the quality and coverage of the initial
3D point cloud for Gaussian reconstruction.
Ray-Constrained Optimization: Gaussian centers are con-
strained to lie along camera rays, reducing 3D position opti-
mization to a one-dimensional depth adjustment. Each point
is parameterized as r = o+z ·d, where o is the optical center,
d is the viewing direction, and z is the depth. This constraint
enhances optimization stability, suppresses floating artifacts,
and improves convergence, particularly under limited views.
C. Hybrid-Loss Depth Estimation
Accurate depth estimation is crucial for high-quality scene
reconstruction, as it provides the essential geometric infor-
mation lacking in 2D imagery. However, under sparse-view
conditions, the lack of strong multi-view constraints makes
high-precision depth estimation difficult through reprojection
error minimization alone. To improve depth reliability and
multi-view consistency, a hybrid-loss module is proposed that
integrates multiple geometric constraints. Figure 3 illustrates
the pipeline of the Hybrid-Loss Depth Estimation module.
1) Initial Depth Estimation Stage: The depth estimation
incorporates two geometric constraints and an outlier filtering
mechanism as follows:
Reprojection Constraint (RC): Using dense image match-
ing correspondences, such as point pair a-b between view pair
i-j, point a from view i is back-projected into 3D space to
obtain point A, which is then projected into view j to yield
a′. The reprojection loss minimizes the Euclidean distance
between the reprojected and original points a′-b:
LRC
init =
X
i,j
∥π(K, Rij, Ray(pa
i , za
i )) −pb
j∥2
2,
(6)
Fig. 3. Hybrid-Loss Depth Estimation pipeline, which first estimates initial
point depth from densely matched points using reprojection and point propa-
gation constraints, then filters outliers. During scene training, rendered depth is
refined under reprojection, point propagation, and TV smoothness constraints.
Point propagation constraint are computed from the common points. Nearest-
neighbor view and secondary filtering reducing redundant computation.
where pi and pj are matching image points, K is the camera
intrinsic, Rij denotes the relative pose from view i to view j,
Ray(·) denotes the ray function that generates spatial points,
and π(·) denotes the projection function.
Point Propagation Constraint (PPC): Unlike sparse
matcher that independently extract feature points, dense
matcher generate matching points based on image pairs,
making reprojection error constraints alone insufficient for
ensuring depth consistency and robustness. To address this,
the Point Propagation Constraint is introduced: for matching
point pair a-b between view pair i-j, and b′-c between view
pair i-k, we use point b and b′ in view j as a bridge to establish
indirect geometric consistency between a and c across views.
The common point set C is defined as:
C(pa
i , pc
k) =







(pa
i , pb
j) ∈Mi,j,
(pb′
j , pc
k) ∈Mj,k,
|pb
j −pb′
j | < dNN,
(7)
where Mi,j denotes matched point pairs between views i and
j, and dNN is a mutual nearest-neighbor distance threshold.
Since identical physical points may not align exactly in pixel
coordinates, a mutual nearest-neighbor search is performed in
a KD-Tree: for each pb
j, its nearest neighbor pb′
j is identified,
and vice versa. If the distance between them is less than dNN,
they are considered the same point. The point propagation loss,
similar to reprojection loss is formulated as:
LP P C
init
=
X
i,k
∥π(K, Rik, Ray(pa
i , za
i )) −pc
k∥2
2,
(8)
where Rik denotes the relative pose from view i to view k.
Algorithm 1 describes the process of constructing the common
point set C.

<!-- page 5 -->
5
Algorithm 1 Common Point Set C
Input: Matching point pairs Mi,j = (pa
i , pb
j) between views i and
j, Mj,k = (pb′
j , pc
k) between views j and k, pixel distance
threshold dNN;
Output: Common point set C = (pa
i , pc
k) for indirect matching
between views i and k;
1: Construct KD-Trees for View j:
2:
Build KD-Tree Tj using all pb
j from Mi,j
3:
Build KD-Tree T ′
j using all pb′
j from Mj,k
4: Search for Mutual Nearest Neighbors:
5:
Initialize empty set C
6:
for each pb
j ∈Mi,j do
7:
Find its nearest neighbor pb′
j in T ′
j
8:
Find the nearest neighbor pb′′
j
of that pb′
j in Tj
9:
if pb
j = pb′′
j
and |pb
j −pb′
j | < dNN then
10:
Extract corresponding pa
i from Mi,j and pc
k from
Mj,k
11:
Add pair (pa
i , pc
k) to C
12:
end if
13:
end for
14: return C
Outlier Filtering Mechanism: Mismatched pairs on object
edges can have low reprojection error yet high depth differ-
ences, whereas those in untextured areas may have low depth
differences but high reprojection error. Hence, the Outlier
Filtering Mechanism jointly considers both metrics for the
Reprojection Constraint:















τdy = τbase + α · Sigmoid(2ˆz −1),
M depth
i,j
←D(zij, zji) =
|zij−zji|
min(zij,zji) ≤τdy,
M error
ij
= ERC
ij
≤τRC,
M RC-dep
i,j
= M error
ij
⊙M error
ji
⊙M depth
i,j
,
(9)
where the dynamic threshold τdy is implemented using a
Sigmoid function to enforce stricter depth consistency con-
straints for nearby points and more relaxed tolerances for
distant points; τbase denotes the base threshold, α is a scaling
factor, and ˆz represents the normalized depth value; zij denotes
the ray depth of view i in the view pair i-j; M depth
i,j
is the
depth consistency mask, obtained by comparing the relative
depth difference with τdy; ERC
ij
denotes the reprojection error
between views i and j; τRC is a fixed reprojection error
threshold; M error
ij
is the reprojection error mask; M RC-dep
i,j
is the final valid matching mask that combines bidirectional
reprojection errors and depth consistency. Based on this mech-
anism, matched point pairs are filtered if either the reprojection
error exceeds the predefined threshold or the relative depth
difference is too large.
The same filtering principle is extended to multi-view chains
for the Point Propagation Constraint. The co-visible point set
through the bridge view j is filtered as:



M P P C
jik
←C(pa
i , pc
k), M depth
i,k
←D(zij, zkj),
M P P C-dep
jik
= M P P C
jik
⊙M depth
i,k
,
(10)
where M P P C
jik
denotes the initial matching mask for the
propagation path i →jbridge →k, obtained from Eq. 7, and
(a) Matching pair images (left, middle) and depth map(right) of SCGaussian.
(b) Matching pair images (left, middle) and depth map (right) of HBSplat.
Fig. 4. Visual comparison of matching pair images and depth maps. (a) and (b)
respectively show the matching pair images (left, middle) and corresponding
depth maps (right) for SCGaussian and HBSplat. SCGaussian relies on the
RANSAC algorithm for outlier filtering, whereas HBSplat employs Outlier
Filtering Mechanism. The comparison demonstrates that HBSplat more effec-
tively removes outliers while preserving a greater number of valid points.
M depth
i,k
is obtained from Eq. 9. M P P C-dep
jik
represents the point
propagation mask after applying depth-based filtering.
Additionally, two enhancement operations are applied: point
propagation relations with a common point count below a
predefined threshold are filtered out. In 360° unbounded
scenes, point propagation relations are constructed based on
the nearest-neighbor distances computed between different
views. These two operations reduce unnecessary computations
and significantly accelerate processing.
2) Scene Training Stage: In the scene training stage, we
employ refined depth constraints, converting filtered matching
points into spatial points and then into Gaussian primitives.
To ensure robustness, the training stage includes three loss
functions to constrain the rendered depth as follows:
Reprojection & Point Propagation Constraint: Inspired
by SCGaussian [38], we apply Bilinear Sampling to the
rendered depth map along rays, computing reprojection and
point propagation errors from training views to other views to
enforce spatial structure consistency.
˙p = M ⊙p,
(11)
LRC
train =
X
i,j
∥π(K, Rij, Ray( ˙pa
i , Bil(Di))) −˙pb
j∥2
2,
(12)
LP P C
train =
X
i,k
∥π(K, Rik, Ray( ˙pa
i , Bil(Di)) −˙pc
k∥2
2,
(13)
where ˙p denotes the pixel point p after filtering, which is
performed using either the mask M RC-dep
i,j
or M P P C-dep
jik
. Di
is the rendered depth map of view i, Ray(·) denotes the
ray function that generates spatial points, π(·) denotes the
projection function, Bil(·) denotes bilinear sampling.
Smoothness Constraint: The Total Variation (TV) loss is
applied to promote spatial smoothness in the rendered depth
map:
LT V =
X
u,v
(|∇xDi(u, v)| + |∇yDi(u, v)|) ,
(14)

<!-- page 6 -->
6
Fig. 5.
Bidirectional Warping Virtual View Synthesis Pipeline. The monocular depth maps from the real views are aligned with sparse depth points via
least-squares optimization. Depth Warping generates virtual depth maps, filling holes. Image Warping samples real views to create virtual views. Distance
scores between real and virtual views are computed, warping multiple optimal real views to a single virtual view. The nearest-neighbor virtual view serves as
the base for Multi-View Fusion. The 3D Gaussian scene is reconstructed using gradient-domain loss and Pearson Correlation Coefficient (PCC) loss constraints.
where u, v are pixel coordinates, Di is the rendered depth map
of view i, and ∇x and ∇y denote the horizontal and vertical
gradient operators, respectively.
The final depth loss functions for the Initial Depth Estima-
tion Stage and the Scene Training Stage are defined as follows:
Linit = LRC
init + LP P C
init ,
(15)
Ltrain = LRC
train + LP P C
train + LT V .
(16)
Figure 4 presents a comparative visualization between the
results without and with the Outlier Filtering Mechanism,
along with a comparison of the depth maps generated after
applying the Hybrid-Loss.
D. Bidirectional Warping Virtual View Synthesis
1) Virtual View Generation Strategy: In sparse-view sce-
narios, particularly with as few as 3 images, 3DGS rendering
often suffers from overfitting and severe geometric distortions
due to insufficient multi-view constraints, especially when
input views are spatially clustered. To mitigate this, we
propose generating multiple virtual views via Depth-Image-
Based Rendering (DIBR) to impose stronger photometric and
geometric constraints during 3DGS optimization. DIBR, also
known as 3D Image Warping, is a core computer graphics
technique that projects reference images into 3D space using
depth maps and reprojects 3D points onto a virtual cam-
era’s image plane. Two primary warping strategies exist: (1)
Forward Warping, which is computationally simple but often
produces holes and disocclusions, leading to less effective
synthesis; and (2) Backward Warping, which provides higher-
quality results but requires the depth map of the target view.
Our approach integrates both strategies through Bidirectional
Warping, effectively combining their advantages. Quantitative
comparison between Forward and Bidirectional Warping is
provided in Table I.
The Bidirectional Warping process operates in two sequen-
tial stages—Depth Warping followed by Image Warping—to
effectively overcome the limitations of pure Forward Warping.
Specifically, the method first warps and interpolates depth
values to generate a complete virtual depth map, then per-
forms reverse warping for photometrically consistent color
interpolation. This approach yields virtual views with signifi-
cantly reduced artifacts such as graininess and blurring while
maintaining computational efficiency. The overall pipeline is
illustrated in Figure 5, and the warping process is formulated
as follows:
(
pvir = Kvir · Rvir · R−1
src · (Dsrc · K−1
src · psrc),
Dvir(pvir) = Fill(Bil(Dsrc(psrc))),
(17)
(
psrc = Ksrc · Rsrc · R−1
vir · (Dvir · K−1
vir · pvir),
Ivir(pvir) = Bil(Isrc(psrc)),
(18)
where psrc and pvir denote pixels in the real view and
the virtual view, respectively, Dvir is the virtual depth map
generated through Depth Warping, and Ivir is the synthesized
virtual image obtained via Image Warping. The source depth
map Dsrc is estimated using monocular depth estimator, with
scale consistency enforced via least-squares optimization. The
Fill(·) operator handles hole completion in the warped depth
map, Bil(·) denotes bilinear sampling.
2) Scale-Agnostic Monocular Depth Recovery:
Virtual
view synthesis relies on accurate depth maps from the real

<!-- page 7 -->
7
TABLE I
COMPARISON BETWEEN FORWARD WARPING AND BIDIRECTIONAL
WARPING METRICS.
Scene
Forward Warping
Bidirectional Warping
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Fern
22.58
0.751
0.175
22.67
0.752
0.175
Leaves
18.07
0.658
0.238
18.14
0.659
0.237
Room
22.43
0.868
0.142
22.53
0.868
0.141
views. The scale and offset of scale-agnostic monocular depth
maps are recovered via an iterative least-squares optimization:
s′, b′ = arg min
s,b
X
z∈DSP
∥MRC ⊙DSP (z) −D′(s, b)∥2
2, (19)
where D′ is obtained through the linear transformation D′ =
s · Dmono + b, DSP is sparse depth points generated from
robust camera-space depths z obtained through Hybrid-Loss
Depth Estimation, and MRC is the reprojection error mask.
The training view’s depth map with the recovered scale s′ and
offset b′ is given by:
Dsrc = s′ · Dmono + b′.
(20)
3) Virtual View Selection & Fusion: Virtual Pose Selection
via Pose Scoring: To generate high-quality virtual views, a
scoring function is proposed to prioritize source real views
that are geometrically close to the target virtual pose. The
Nearest Neighbor Score for each candidate pose is computed
based on both translational distance and rotational difference,
with higher scores assigned to closer poses with more similar
orientations.
Multi-View Fusion: Since a virtual view generated from
a single source view may contain occlusions or missing
regions, a fusion strategy aggregates information from multiple
sources. Specifically, the top-k virtual views rendered at the
same target pose are fused; these views are generated from
source views selected based on the highest pose similarity
scores. The view from the closest source serves as a base,
while others are blended to expand visible regions. The
complete fusion process is formulated as follows:
I′
vir = Fusek(Ivir|Order(Score(Rsrc, Rvir))),
(21)
where Score(·) denotes the Nearest Neighbor Score quantify-
ing pose similarity, Order(·) sorts source poses in descending
order based on their Score values, indicating their proximity
to the target virtual pose, Fusek(·) represents the operation
of fusing the top-k virtual views, and I′
vir is the final fused
virtual view. Figure 6 illustrates the fusion result.
The virtual depth map and occlusion mask corresponding
to the virtual view are also generated through Bidirectional
Warping and the Selection & Fusion operations, as illustrated
in Figure 5.
4) Loss function: The loss function for virtual view super-
vision is defined as follows:
(
Lgrad = L1(G(M) ⊙G(Iren), G(M) ⊙G(I′
vir)),
Lvir
color = αLgrad + βLssim(M ⊙Iren, M ⊙I′
vir),
(22)
Lvir
depth = PCC(M ⊙Dren, M ⊙Dvir),
(23)
Fig. 6. Multi-View Fusion. Fusing the left, middle virtual views produces the
result shown on the right.
where M is the virtual view mask, G(·) computes the image
gradient domain, α and β are weighting coefficients satisfying
α + β = 1, and PCC(·) denotes the Pearson Correlation
Coefficient operation. The gradient-domain loss Lgrad enhances
sensitivity to edges and structural details by emphasizing high-
frequency information, while maintaining robustness to abso-
lute intensity variations. The Pearson Correlation Coefficient
for depth loss measures linear dependence between rendered
and virtual depths, effectively capturing structural consistency
while being invariant to global scale shifts.
E. Occlusion-Aware Reconstruction
In NVS, models are typically optimized to reconstruct only
the visible regions present in the training views. However, syn-
thesizing views beyond the original training set often reveals
occluded areas. This issue is exacerbated under sparse-view
conditions, where limited coverage increases the probability
of exposing unseen regions in novel views.
Regions exhibiting substantial depth differences between
foreground and background are particularly prone to occlu-
sions. Based on this, an Occlusion-Aware Reconstruction
method grounded in depth differences is proposed, as illus-
trated in Figure 7. This component is suited for reconstruct-
ing smaller occluded regions. First, edge gradients of the
depth map are computed using the Sobel operator to extract
candidate regions a with large foreground-background depth
disparities. Since such edge regions often fail to form contin-
uous occlusion boundaries, the monocular depth map Dmono
is partitioned into n discrete depth layers. The depth slice b
where pixels from region a are predominantly distributed is
identified to generate an initial foreground mask. To refine
the mask and exclude foreground regions with insignificant
depth variations, the minimum bounding rectangle c of region
a is extracted. This rectangle is used to crop content from
depth slice b, producing a local foreground mask emphasizing
regions with pronounced depth differences:
Mlocal_F G = F(Soble(Dmono), Slice(Dmono)),
(24)
where Dmono denotes the monocular depth map, Sobel(·)
represents the Sobel edge detection operator, Slice(·) refers
to the depth slicing operation, and F denotes the function that
extracts the local foreground mask Mlocal_F G.

<!-- page 8 -->
8
Fig. 7.
Occlusion-Aware Reconstruction Pipeline. Local foreground mask,
generated by utilizing the sobel map and depth slices, identifies the inpainting
regions and guides the Simple-LAMA model to perform the inpainting.
The Simple-LAMA [43] inpainting model is subsequently
employed to generate plausible completion results. This model
uses the Mlocal_F G along with the training views as input,
where the mask specifically identifies regions that require
completion:
Iinp = Lama(Mlocal_F G, I),
(25)
where Iinp denotes the image after inpainting. To enforce geo-
metric consistency and refine the reconstruction, the inpainting
image Iinp is used to supervise the background Gaussian
splatting process. The background point set is first isolated
by filtering out foreground regions using the occlusion-aware
mask:
PBG = P ⊙¬(Mlocal_F G ⊙MD_slice),
(26)
S(t) =
(
Train(Iinp, PBG),
if t ≤tBG,
Train(I, P),
if t > tBG,
(27)
where MD_slice is a depth slice mask that identifies regions
with depth values less than the current slice threshold, P
denotes the original 3D point cloud, and PBG represents the
point cloud after excluding points in the occluded foreground
regions. The background Gaussians are optimized exclusively
using Iinp as supervision for the first tBG iterations, after
which the entire set of Gaussians is trained to reconstruct the
complete scene. Ablation will demonstrate the effectiveness of
the proposed Occlusion-Aware Reconstruction.
F. Total Loss Function
The overall optimization objective combines photometric,
geometric, and virtual view consistency terms, defined as:
Lcol(I) = λL1(Iren, I) + (1 −λ)Lssim(Iren, I),
(28)
Ltotal = Lcol(I)+Ldep(D)+Lcol(Ivir)+Ldep(Dvir), (29)
where Ldep(D) denotes the depth loss as defined in Eq. 16,
Lcol(Ivir) denotes the virtual image color loss as defined
in Eq. 22, and Ldep(Dvir) denotes the virtual depth loss
as defined in Eq. 23). The weight λ is set to 0.8 in all
experiments.
IV. EXPERIMENT
A. Setup
1) Datasets:
HBSplat
is
evaluated
on
five
widely
adopted public datasets: LLFF, IBRNet, Blender, DTU, and
Tanks&Temples. These datasets cover forward-facing, syn-
thetic, object-centric, and 360° unbounded scenes, ensuring
a comprehensive assessment of robustness across diverse
settings. For LLFF and IBRNet (forward-facing scenes), we
follow the protocol of DNGaussian [27]: one every 8 images
is held out for testing, while 3 training views are uniformly
sampled from the remainder. Evaluation is performed over
LLFF scenes with 8×, 4× downsampling rate and IBRNet
scenes with 2× downsampling rate. For Blender (synthetic
objects), models are trained on 8 views and evaluated on 25
uniformly sampled test images with 2× downsampling rate.
For DTU (object-centric scenes), we adopt the setup from SC-
Gaussian [37], using provided object masks during evaluation
to exclude background regions and focus on the target object.
4× downsampling rate is applied. For Tanks&Temples (360°
scenes), 6 scenes are selected. Training uses 24 views, training
and testing split strategy follows the same protocol as used in
LLFF (one every 8 images) with 4× downsampling rate.
2) Evaluation Metrics: Rendering quality is evaluated us-
ing PSNR, SSIM, and LPIPS. In addition, following DNGaus-
sian [27], the Average (AVG) is computed as a composite
metric, defined as the geometric mean of MSE (10−PSNR/10),
√
1 −SSIM, and LPIPS. Efficiency is assessed via frames per
second (FPS) and average training time. Final dataset-level
metrics are reported as the average across all scenes.
3) Baselines: HBSplat is compared against several excel-
lent sparse-view NVS jobs, including NeRF-based and 3DGS-
based approaches. NeRF-based baselines include FreeNeRF
[8] and SparseNeRF [7]. 3DGS-based baselines include vanilla
3DGS [5], FSGS [26], SID [39], DNGaussian [27], MCGS
[36], and SCGaussian [37]. For NeRF-based methods, we
report their best quantitative results from the respective papers.
To ensure a fair comparison, we use public code with consis-
tent training-testing splits and a unified base environment with
core dependencies fixed, including PyTorch and CUDA.
4) Implementation Details:
HBSplat is built upon the
3DGS framework [5]. Camera poses are estimated using
COLMAP across all views for robustness. To ensure consistent
comparison, similar to SCGaussian, we employed the GIM-
DKM [44] model for dense matching to extract matching
points per training view. Besides, we use ML-depth-pro [45]
for monocular depth estimation, which excels in capturing fine
details. Both depth estimation and scene training are iterated
2,000 times. Initialize and generate Gaussian primitives after
filtering outliers. Bidirectional Warping generates 100 virtual
views; one virtual view iteration is performed for every 5 real
view iterations. The Occlusion-Aware Reconstruction applies
a depth difference mask to prioritize rendering of occluded
region Gaussians in the first 1,000 iterations, followed by
training all Gaussians. Gaussian cloning and pruning follow
the vanilla 3DGS settings. All experiments are conducted on
a single NVIDIA RTX 3090 GPU. For detailed configuration,
please refer to the publicly available code.

<!-- page 9 -->
9
3DGS
SID
MCGS
SCGaussian
HBSplat(Ours)
GT
Fig. 8. Qualitative comparison on LLFF dataset under 3 training views. The rendering results of our HBSplat are more accurate and display finer details.
3DGS
FSGS
DNGaussian
SCGaussian
HBSplat(Ours)
GT
Fig. 9. Qualitative comparison on IBRNet dataset with 3 training views. The rendering results of HBSplat method are more accurate and display finer details.
B. Comparison
1) LLFF & IBRNet: Quantitative results on the LLFF and
IBRNet datasets under extremely sparse settings (3 training
views) are summarized in Table II. HBSplat achieves state-of-
the-art performance, reaching a PSNR of 21.13 dB and LPIPS
of 0.189 on LLFF, and 22.19 dB PSNR with 0.249 LPIPS on
IBRNet, outperforming all baseline methods across metrics.
Detailed per-scene results on LLFF (III) further validate the
advantage of HBSplat in forward-facing scene reconstruction.
Qualitative results are shown in Figures 8 and 9. The GT
depth map is obtained from monocular depth estimator. In the
Fortress and Horns scenes of LLFF dataset, SID, MCGS, and
SCGaussian exhibit various artifacts and detail loss. Although
SID preserves depth details better, this comes at the cost of
generating more spatial points and longer training time. In
the Signboard and Table scenes of IBRNet dataset, methods
such as FSGS, DNGaussian, and SCGaussian also suffer from
artifacts and a lack of fine details. SCGaussian is particularly
prone to artifacts due to insufficient outlier filtering. Moreover,
SID, MCGS, FSGS, and DNGaussian all show a common
limitation: the loss of fine details in close-range areas and
the introduction of blur in distant regions.
In contrast, HBSplat produces renderings that closely match
the GT across all forward-facing scenes, with enhanced edge
sharpness, improved texture preservation, and minimal arti-
facts. These results demonstrate HBSplat’s robustness and su-
periority in challenging sparse-view reconstruction of forward-
facing scenes.
2) Blender:
Quantitative comparisons on the Blender
dataset with 8 training views are presented in Table IV, where
HBSplat achieves competitive results across all metrics.
Qualitative results are shown in Figure 10. In the Chair
object, MCGS produces noticeable floaters, while DNGaussian
suffers from blurred details. For the Lego object, both MCGS
and DNGaussian fail to reconstruct fine structures, leading

<!-- page 10 -->
10
TABLE II
QUANTITATIVE COMPARISONS ON THE LLFF (1/8, 1/4 RESOLUTION), IBRNET (1/2 RESOLUTION) DATASETS WITH 3 TRAINING VIEWS. TOP-3 ENTRIES
ARE HIGHLIGHTED: RED (1ST), ORANGE (2ND), YELLOW (3RD).
Method
1/8 LLFF
1/4 LLFF
1/2 IBRNet
PSNR↑
SSIM↑
LPIPS↓
AVG↓
PSNR↑
SSIM↑
LPIPS↓
AVG↓
PSNR↑
SSIM↑
LPIPS↓
AVG↓
FreeNeRF[CVPR23] [8]
19.63
0.612
0.308
0.134
18.73
0.562
0.384
0.169
19.76
0.588
0.333
0.135
SparseNeRF[ICCV23] [7]
19.86
0.624
0.328
0.127
19.07
0.564
0.401
0.168
19.90
0.593
0.364
0.137
3DGS[SIGGRAPH23] [5]
15.42
0.383
0.463
0.218
13.28
0.350
0.486
0.264
17.20
0.556
0.355
0.165
FSGS[ECCV24] [26]
20.27
0.697
0.206
0.102
19.70
0.667
0.265
0.117
19.67
0.605
0.306
0.127
SID[ICASSP25] [39]
20.40
0.701
0.215
0.102
19.08
0.635
0.336
0.135
19.44
0.599
0.360
0.137
DNGaussian[CVPR24] [27]
19.12
0.591
0.294
0.132
18.23
0.575
0.386
0.155
18.14
0.554
0.415
0.161
MCGS[TPAMI25] [36]
20.32
0.700
0.219
0.103
19.63
0.663
0.292
0.122
21.02
0.674
0.282
0.109
SCGaussian[NeurIPS24] [37]
20.73
0.725
0.196
0.095
20.03
0.683
0.266
0.114
21.47
0.689
0.275
0.103
HBSplat(Ours)
21.13
0.735
0.189
0.090
20.30
0.693
0.256
0.109
22.19
0.708
0.249
0.093
TABLE III
QUANTITATIVE COMPARISONS ON THE NERF-LLFF DATASET WITH 2,3,5 TRAINING VIEWS. THE BEST AND SECOND-BEST ENTRIES ARE MARKED IN
RED AND ORANGE, RESPECTIVELY.
Scene
Method
2-view
3-view
5-view
PSNR ↑
SSIM ↑
LPIPS ↓
AVG ↓
PSNR ↑
SSIM ↑
LPIPS ↓
AVG ↓
PSNR ↑
SSIM ↑
LPIPS ↓
AVG ↓
Fern
3DGS[SIGGRAPH23]
14.07
0.306
0.509
0.255
15.80
0.348
0.502
0.220
25.02
0.647
0.255
0.078
SCG[NeurIPS24]
19.38
0.627
0.317
0.133
22.15
0.741
0.175
0.084
24.20
0.824
0.120
0.057
HBSplat(Ours)
20.09
0.658
0.258
0.115
22.73
0.753
0.173
0.078
24.52
0.832
0.110
0.054
Flower
3DGS[SIGGRAPH23]
16.15
0.370
0.437
0.203
16.37
0.445
0.437
0.195
26.10
0.782
0.146
0.055
SCG[NeurIPS24]
19.84
0.608
0.307
0.134
21.66
0.690
0.217
0.103
25.86
0.858
0.097
0.045
HBSplat(Ours)
20.10
0.630
0.274
0.126
21.70
0.693
0.221
0.104
25.96
0.861
0.090
0.044
Fortress
3DGS[SIGGRAPH23]
15.70
0.325
0.379
0.203
18.80
0.437
0.367
0.153
23.74
0.613
0.229
0.084
SCG[NeurIPS24]
21.34
0.558
0.294
0.114
25.00
0.823
0.122
0.056
27.48
0.867
0.094
0.040
HBSplat(Ours)
22.17
0.646
0.274
0.097
25.68
0.836
0.117
0.051
27.60
0.871
0.093
0.039
Horns
3DGS[SIGGRAPH23]
14.35
0.291
0.511
0.250
15.65
0.391
0.463
0.214
19.07
0.528
0.359
0.145
SCG[NeurIPS24]
16.84
0.554
0.372
0.176
19.86
0.737
0.221
0.109
22.71
0.821
0.158
0.077
HBSplat(Ours)
17.76
0.624
0.319
0.152
20.33
0.746
0.219
0.105
23.38
0.832
0.145
0.070
Leaves
3DGS[SIGGRAPH23]
13.18
0.249
0.452
0.266
14.94
0.342
0.410
0.220
18.20
0.528
0.300
0.145
SCG[NeurIPS24]
15.30
0.419
0.454
0.216
17.97
0.651
0.240
0.131
19.46
0.720
0.203
0.106
HBSplat(Ours)
15.86
0.482
0.396
0.192
18.14
0.659
0.237
0.128
19.70
0.734
0.194
0.102
Orchids
3DGS[SIGGRAPH23]
14.88
0.214
0.492
0.242
15.95
0.313
0.460
0.213
20.21
0.472
0.340
0.132
SCG[NeurIPS24]
14.40
0.384
0.363
0.218
16.43
0.530
0.248
0.157
17.93
0.619
0.203
0.126
HBSplat(Ours)
14.63
0.414
0.328
0.205
16.67
0.543
0.234
0.150
17.83
0.617
0.202
0.127
Room
3DGS[SIGGRAPH23]
12.09
0.326
0.581
0.308
13.51
0.451
0.546
0.262
16.23
0.451
0.508
0.207
SCG[NeurIPS24]
19.87
0.793
0.228
0.104
22.12
0.858
0.150
0.071
26.20
0.910
0.107
0.045
HBSplat(Ours)
20.75
0.823
0.190
0.089
22.53
0.868
0.141
0.066
26.64
0.915
0.103
0.043
Trex
3DGS[SIGGRAPH23]
11.20
0.225
0.552
0.332
12.35
0.344
0.518
0.290
14.67
0.436
0.436
0.223
SCG[NeurIPS24]
18.40
0.665
0.291
0.138
20.62
0.774
0.193
0.093
23.24
0.851
0.132
0.063
HBSplat(Ours)
19.35
0.702
0.236
0.115
21.27
0.787
0.171
0.084
24.01
0.864
0.120
0.057
Mean
3DGS[SIGGRAPH23]
13.95
0.288
0.489
0.255
15.42
0.383
0.463
0.218
20.41
0.557
0.322
0.124
SCG[NeurIPS24]
18.17
0.576
0.328
0.148
20.73
0.725
0.196
0.095
23.38
0.809
0.139
0.065
HBSplat(Ours)
18.84
0.622
0.280
0.131
21.13
0.735
0.189
0.090
23.71
0.816
0.133
0.062
to a loss of detail. In the Drums and Ship objects, 3DGS
and SCGaussian exhibit floters, with 3DGS showing the most
severe floters as well as a clear color discrepancy from the GT.
In contrast, HBSplat produces superior reconstruction results
and more accurate depth maps, with sharper geometry and
better-preserved textures.
3) DTU: Quantitative comparisons on the DTU dataset are
summarized in Table V, where HBSplat achieves the highest
scores across all metrics, followed by SCGaussian and MCGS.
Figure 11 presents the rendering results. Both 3DGS and
DNGaussian produce a considerable number of floaters. FSGS
and SID fail to reconstruct textual details and suffer from
overall blurriness. While MCGS and SCGaussian also exhibit
floaters, MCGS shows the most severe cases, which further
lead to inaccurate depth estimations. Benefiting from the Out-
lier Filtering Mechanism and virtual view constraints, HBSplat
significantly reduces artifacts and improves texture fidelity,
particularly around text and fine structures.
4) Tanks&Temples: The Tanks&Temples dataset is em-
ployed to evaluate the performance of HBSplat in large-scale
360° unbounded scenes. Quantitative results are presented in
Table VI. Note that evaluations on this dataset utilize only the
Hybrid-Loss Depth Estimation component of our framework.
DNGaussian and MCGS are excluded from comparison as they
lack configurations suitable for 360° unbounded scenes.
Qualitative results are shown in Figure 12. As SID is
an enhanced version of FSGS, only SID is presented for
clarity. In the Family and Horse scenes, both vanilla 3DGS

<!-- page 11 -->
11
MCGS
DNGaussian
HBSplat(Ours)
GT
3DGS
SCGaussian
HBSplat(Ours)
GT
Fig. 10.
Qualitative comparison on Blender dataset with 8 training views.
HBSplat method excels both in geometry and rendering qualities.
TABLE IV
QUANTITATIVE COMPARISONS ON THE BLENDER (1/2 RESOLUTION)
DATASETS WITH 8 TRAINING VIEWS. TOP-3 ENTRIES ARE HIGHLIGHTED:
RED (1ST), ORANGE (2ND), YELLOW (3RD).
Method
Blender
PSNR↑
SSIM↑
LPIPS↓
AVG↓
FreeNeRF[CVPR23] [8]
24.26
0.883
0.098
0.050
SparseNeRF[ICCV23] [7]
22.41
0.861
0.119
0.063
3DGS[SIGGRAPH23] [5]
22.85
0.836
0.141
0.066
FSGS[ECCV24] [26]
16.35
0.601
0.369
0.175
SID[ICASSP25] [39]
12.65
0.729
0.321
0.208
DNGaussian[CVPR24] [27]
22.79
0.870
0.106
0.058
MCGS[TPAMI25] [36]
24.06
0.887
0.089
0.048
SCGaussian[NeurIPS24] [37]
23.33
0.883
0.097
0.053
HBSplat(Ours)
24.18
0.894
0.086
0.047
and SID exhibit significant artifacts, with the latter also
showing noticeable color rendering errors. SCGaussian suffers
from partial object omission. In contrast, HBSplat produces
more complete and geometrically stable reconstructions with
reduced artifacts, demonstrating its robustness in challenging
large-scale unbounded scenarios.
5) Efficiency: Efficiency evaluation of different methods
on the LLFF dataset in Table VII. The proposed HBSplat
method matches the efficiency of other advanced approaches,
achieving an inference speed of 250 FPS, which is com-
parable to methods such as SCGaussian and MCGaussian,
while maintaining a low average training time of 2.5 minutes.
For GPU memory, HBSplat requires 3 GB for training and
3DGS
DNGaussian
HBSplat(Ours)
GT
FSGS
SID
HBSplat(Ours)
GT
MCGS
SCGaussian
HBSplat(Ours)
GT
Fig. 11. Qualitative comparison on DTU dataset with 3 training views.
TABLE V
QUANTITATIVE COMPARISONS ON THE DTU (1/4 RESOLUTION) DATASETS
WITH 3 TRAINING VIEWS. TOP-3 ENTRIES ARE HIGHLIGHTED: RED (1ST),
ORANGE (2ND), YELLOW (3RD).
Method
DTU
PSNR↑
SSIM↑
LPIPS↓
AVG↓
FreeNeRF[CVPR23] [8]
19.92
0.787
0.182
0.098
SparseNeRF[ICCV23] [7]
19.55
0.769
0.201
0.102
3DGS[SIGGRAPH23] [5]
12.76
0.595
0.376
0.233
FSGS[ECCV24] [26]
17.41
0.728
0.247
0.132
SID[ICASSP25] [26]
16.35
0.654
0.334
0.165
DNGaussian[CVPR24] [27]
18.46
0.807
0.168
0.101
MCGS[TPAMI25] [36]
19.02
0.810
0.154
0.094
SCGaussian[NeurIPS24] [37]
19.11
0.857
0.123
0.082
HBSplat(Ours)
20.22
0.872
0.110
0.072
an additional 5 GB for the Virtual View Synthesis (VVS)
component to generate 100 virtual views, outperforming meth-
ods like FreeNeRF (4×48 GB) and SparseNeRF (32 GB).
Compared to approaches such as 3DGS (7.5 minutes, 2 GB)
and FSGS (17 minutes, 3 GB), HBSplat offers a substantial
reduction in training time while preserving real-time inference
capabilities, making it a highly efficient solution for practical
3D reconstruction tasks.
C. Ablation Study
Ablation study is conducted on the LLFF dataset under
a 3-view setting to evaluate the individual contributions of
the core components in HBSplat: Hybrid-Loss Depth Estima-
tion (HLDE), Bidirectional Warping Virtual View Synthesis
(VVS), and Occlusion-Aware Reconstruction (OAR).
Quantitative Component Analysis: Quantitative results
are summarized in Table VIII. The full HBSplat model
(HLDE+VVS+OAR) yields the best overall performance,

<!-- page 12 -->
12
3DGS
SID
SCGaussian
HBSplat(Ours)
GT
Fig. 12. Qualitative comparison on Tanks&Temples dataset with 24 training views. HBSplat method excels both in geometry and rendering qualities.
TABLE VI
QUANTITATIVE COMPARISONS ON THE TANKS&TEMPLES (1/4
RESOLUTION) DATASETS WITH 24 TRAINING VIEWS. TOP-3 ENTRIES ARE
HIGHLIGHTED: RED (1ST), ORANGE (2ND), YELLOW (3RD).
Method
Tanks&Temples
PSNR↑
SSIM↑
LPIPS↓
AVG↓
3DGS[SIGGRAPH23] [5]
18.77
0.706
0.267
0.124
FSGS[ECCV24] [26]
19.39
0.720
0.273
0.118
SID[ICASSP25] [26]
19.51
0.723
0.273
0.117
SCGaussian[NeurIPS24] [37]
18.96
0.722
0.298
0.125
HBSplat(Ours)
20.00
0.749
0.268
0.110
TABLE VII
EFFICIENCY COMPARISON OF DIFFERENT METHODS ON LLFF DATASET.
Method
Inference
(FPS)
Average
Training
Time
GPU Mem
FreeNeRF[CVPR23] [8]
9 × 10−2
2.3 h
4×48 GB
SparseNeRF[ICCV23] [7]
9 × 10−2
1.5 h
32 GB
3DGS[SIGGRAPH23] [5]
400
7.5 min
2 GB
FSGS[ECCV24] [26]
300
17 min
3 GB
SID[ICASSP25] [26]
300
28 min
3.5 GB
DNGaussian[CVPR24] [27]
500
3.5 min
2 GB
MCGS[TPAMI25] [36]
250
2.5 min
2 GB
SCGaussian[NeurIPS24] [37]
250
1.5 min
3 GB
HBSplat(Ours)
250
2.5 min
3 GB
5 GB (VVS)
achieving a PSNR of 21.13 dB and an LPIPS of 0.189. Among
the components, VVS contributes the highest PSNR due to
the strong photometric constraints imposed by virtual views.
Although HLDE yields a lower PSNR than VVS, it infers more
accurate geometry, resulting in the lowest LPIPS. Notably,
each individual component of HBSplat outperforms all other
baselines. Furthermore, the performance gain from combining
TABLE VIII
ABLATION STUDY ON THE COMPONENTS OF HBSPLAT.
Method
HLDE
VVS
OAR
PSNR↑
SSIM↑
LPIPS↓
AVG↓
3DGS
✗
✗
✗
15.42
0.383
0.463
0.221
SCGaussian
✗
✗
✗
20.73
0.725
0.196
0.100
HBSplat
(Ours)
✓
✗
✗
20.91
0.732
0.187
0.097
✗
✓
✗
20.93
0.731
0.191
0.098
✗
✗
✓
20.85
0.727
0.191
0.099
✓
✓
✗
21.11
0.734
0.191
0.096
✓
✓
✓
21.13
0.735
0.189
0.096
components is non-linear, integrating HLDE with VVS alone
is sufficient to achieve substantial improvement.
Qualitative Component Analysis: Figure 13 provides a
qualitative comparison on the Fern scene. The following
observations are made in the boxed regions: +HLDE: Produces
more robust depth estimates, effectively resolving structural
distortions present in 3DGS. +VVS: Introduces virtual view
constraints, significantly reducing artifacts and improving con-
sistency. +OAR: Generates more plausible background content
and alleviates occlusion-related artifacts.
In the second row of the comparison, MCGS reconstructs
leaves as sparse, needle-like structures and suffers from blurred
edges. SCGaussian exhibits black artifacts and fails to re-
construct the background completely. In contrast, HBSplat
produces renderings that are visually closer to the GT, demon-
strating the effectiveness of the proposed components.
PPC with different nearest neighbor distance (dNN):
Figure 14 shows the line graph of PSNR values over different
dNN. It can be observed that as dNN increases, both the
number of common points and the PSNR value gradually rise.
However, when the dNN exceeds 3, the PSNR for both the
Fern scene and the LLFF dataset begins to decline. The point
(22.38, 99) represents the PSNR value and the number of

<!-- page 13 -->
13
+HLDE
+HLDE+VVS
+HLDE+VVS+OAR
MCGS
SCGaussian
GT
Fig. 13.
Ablation study on component contributions. Compared to MCGS
and SCGaussian, the rendering results of full HBSplat model are closest to
GT.
Fig. 14.
PSNR vs. dNN. Both the number of common points and PSNR
initially increase with dNN but decline after the dNN exceeds 3.
common points for the Fern scene, respectively. Notably, if
the number of common points is less than 100 following a
secondary filtering step, the PPC operation is skipped to avoid
redundant processing.
V. CONCLUSION
This paper presents HBSplat, an effective framework for
sparse-view NVS based on 3DGS. The core of our approach
includes three innovations: a Hybrid-Loss Depth Estimation
module that enforces multi-view consistency through repro-
jection, point propagation, and TV smoothness constraints;
a Bidirectional Warping Virtual View Synthesis method that
enhances the coverage of unobserved regions, thereby mitigat-
ing overfitting to the limited input views; and an Occlusion-
Aware Reconstruction component that improves background
rendering through depth-difference priors. Extensive experi-
ments across standard benchmarks demonstrate that HBSplat
achieves state-of-the-art performance under extreme sparsity.
We believe our work offers a valuable step toward practical
and high-quality 3D reconstruction from very few images.
Limitations and Future Work: HBSplat has several limi-
tations. It relies on COLMAP for pose estimation and uses
fixed propagation thresholds, potentially limiting adaptabil-
ity. The occlusion module, optimized for foreground objects,
may struggle with large homogeneous regions such as sky.
Additionally, it only supports static scenes. Future work will
explore self-supervised pose estimation, adaptive thresholds,
and dynamic scene modeling to address these constraints.
REFERENCES
[1] D.-Y. Nam and J.-K. Han, “An efficient algorithm for generating
harmonized stereoscopic 360° vr images,” IEEE Transactions on Circuits
and Systems for Video Technology, vol. 31, no. 12, pp. 4864–4882, 2021.
[2] S. Zhang, W. Zhao, Z. Guan, W. Zhao, J. Peng, and J. Fan, “Learning
cross-view consistent 3d keypoints for object 6d pose estimation,” IEEE
Transactions on Circuits and Systems for Video Technology, vol. 35,
no. 7, pp. 6816–6831, 2025.
[3] Y. Wen, Y. Zhao, Y. Liu, B. Huang, F. Jia, Y. Wang, C. Zhang, T. Wang,
X. Sun, and X. Zhang, “Panacea+: Panoramic and controllable video
generation for autonomous driving,” IEEE Transactions on Circuits and
Systems for Video Technology, pp. 1–1, 2025.
[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, p. 99–106, Dec.
2021.
[5] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, p. 1–14, Jul. 2023.
[6] A. Jain, M. Tancik, and P. Abbeel, “Putting nerf on a diet: Semanti-
cally consistent few-shot view synthesis,” in 2021 IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV).
IEEE, Oct. 2021, p.
5865–5874.
[7] G. Wang, Z. Chen, C. C. Loy, and Z. Liu, “Sparsenerf: Distilling
depth ranking for few-shot novel view synthesis,” in 2023 IEEE/CVF
International Conference on Computer Vision (ICCV). IEEE, Oct. 2023.
[8] J. Yang, M. Pavone, and Y. Wang, “Freenerf: Improving few-shot neural
rendering with free frequency regularization,” in 2023 IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR).
IEEE,
Jun. 2023, p. 8254–8263.
[9] Z. Liu, J. Su, G. Cai, Y. Chen, B. Zeng, and Z. Wang, “Georgs:
Geometric regularization for real-time novel view synthesis from sparse
inputs,” IEEE Transactions on Circuits and Systems for Video Technol-
ogy, vol. 34, no. 12, pp. 13 113–13 126, 2024.
[10] C. Yang, S. Li, J. Fang, R. Liang, L. Xie, X. Zhang, W. Shen, and
Q. Tian, “Gaussianobject: High-quality 3d object reconstruction from
four views with gaussian splatting,” ACM Transactions on Graphics,
vol. 43, no. 6, p. 1–13, Nov. 2024.
[11] J. Chung, J. Oh, and K. M. Lee, “Depth-regularized optimization for 3d
gaussian splatting in few-shot images,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp. 811–
820.
[12] S. Guo, Q. Wang, Y. Gao, R. Xie, L. Li, F. Zhu, and L. Song, “Depth-
guided robust point cloud fusion nerf for sparse input views,” IEEE
Transactions on Circuits and Systems for Video Technology, vol. 34,
no. 9, pp. 8093–8106, 2024.
[13] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in 2021 IEEE/CVF International Con-
ference on Computer Vision (ICCV).
IEEE, Oct. 2021.
[14] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in 2022
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR).
IEEE, Jun. 2022.
[15] T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions on
Graphics, vol. 41, no. 4, p. 1–15, Jul. 2022.
[16] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger,
and N. Radwan, “Regnerf: Regularizing neural radiance fields for
view synthesis from sparse inputs,” in 2022 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR).
IEEE, Jun. 2022.
[17] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable
visual models from natural language supervision,” in International
conference on machine learning.
PmLR, 2021, pp. 8748–8763.
[18] N. Somraj and R. Soundararajan, “Vip-nerf: Visibility prior for sparse
input neural radiance fields,” in Special Interest Group on Computer
Graphics and Interactive Techniques Conference Conference Proceed-
ings, ser. SIGGRAPH ’23.
ACM, Jul. 2023, p. 1–11.
[19] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srini-
vasan, D. Verbin, J. T. Barron, B. Poole, and A. Hoły´nski, “Reconfusion:
3d reconstruction with diffusion priors,” in 2024 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2024,
p. 21551–21561.

<!-- page 14 -->
14
[20] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y.
Lin, “inerf: Inverting neural radiance fields for pose estimation,” in 2021
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS).
IEEE, Sep. 2021.
[21] Z. Wang, S. Wu, W. Xie, M. Chen, and V. A. Prisacariu, “Nerf–: Neural
radiance fields without known camera parameters,” 2021.
[22] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting
neural radiance fields,” in 2021 IEEE/CVF International Conference on
Computer Vision (ICCV).
IEEE, Oct. 2021, p. 5721–5731.
[23] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “Dreamgaussian:
Generative gaussian splatting for efficient 3d content creation,” arXiv
preprint arXiv:2309.16653, 2023.
[24] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d
gaussians: Tracking by persistent dynamic view synthesis,” in 2024
International Conference on 3D Vision (3DV).
IEEE, Mar. 2024, p.
800–809.
[25] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian,
and X. Wang, “4d gaussian splatting for real-time dynamic scene
rendering,” in 2024 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
IEEE, Jun. 2024, p. 20310–20320.
[26] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, “Fsgs: Real-time few-shot view
synthesis using gaussian splatting,” in European conference on computer
vision.
Springer, 2024, pp. 145–163.
[27] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
“Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with
global-local depth normalization,” in 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR).
IEEE, Jun. 2024,
p. 20775–20785.
[28] A. Paliwal, W. Ye, J. Xiong, D. Kotovenko, R. Ranjan, V. Chandra, and
N. K. Kalantari, “Coherentgs: Sparse novel view synthesis with coherent
3d gaussians,” in European Conference on Computer Vision.
Springer,
2024, pp. 19–37.
[29] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, and X. Bai, CoR-GS:
Sparse-View 3D Gaussian Splatting via Co-regularization.
Springer
Nature Switzerland, Sep. 2024, p. 335–352.
[30] H. Yu, X. Long, and P. Tan, “Lm-gaussian: Boost sparse-view 3d gaus-
sian splatting with large model priors,” arXiv preprint arXiv:2409.03456,
2024.
[31] B. Ke, K. Qu, T. Wang, N. Metzger, S. Huang, B. Li, A. Obukhov, and
K. Schindler, “Marigold: Affordable adaptation of diffusion-based image
generators for image analysis,” IEEE Transactions on Pattern Analysis
and Machine Intelligence, p. 1–18, 2025.
[32] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control
to text-to-image diffusion models,” in Proceedings of the IEEE/CVF
international conference on computer vision, 2023, pp. 3836–3847.
[33] W. Xu, H. Gao, S. Shen, R. Peng, J. Jiao, and R. Wang, MVPGS:
Excavating Multi-view Priors for Gaussian Splatting from Sparse Input
Views.
Springer Nature Switzerland, Nov. 2024, p. 203–220.
[34] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, MVSNet: Depth Inference
for Unstructured Multi-view Stereo.
Springer International Publishing,
2018, p. 785–801.
[35] A. Beyer and L. Alexy, “An image equals 16x16 words: Scaling image
recogni:on with transformers,” 2025.
[36] Y. Xiao, D. Zhai, W. Zhao, K. Jiang, J. Jiang, and X. Liu, “Mcgs: Mul-
tiview consistency enhancement for sparse-view 3d gaussian radiance
fields,” arXiv preprint arXiv:2410.11394, 2024.
[37] R. Peng, W. Xu, L. Tang, J. Jiao, R. Wang et al., “Structure consistent
gaussian splatting with matching prior for few-shot novel view synthe-
sis,” Advances in Neural Information Processing Systems, vol. 37, pp.
97 328–97 352, 2024.
[38] L. Han, J. Zhou, Y.-S. Liu, and Z. Han, “Binocular-guided 3d gaussian
splatting with view consistency for sparse view synthesis,” Advances
in Neural Information Processing Systems, vol. 37, pp. 68 595–68 621,
2024.
[39] Z. He, Z. Xiao, K.-C. Chan, Y. Zuo, J. Xiao, and K.-M. Lam, “See in
detail: Enhancing sparse-view 3d gaussian splatting with local depth and
semantic regularization,” arXiv preprint arXiv:2501.11508, 2025.
[40] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and
A. Joulin, “Emerging properties in self-supervised vision transformers,”
in Proceedings of the International Conference on Computer Vision
(ICCV), 2021.
[41] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth inference
for unstructured multi-view stereo,” European Conference on Computer
Vision (ECCV), 2018.
[42] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny,
“Vggt: Visual geometry grounded transformer,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2025.
[43] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha,
A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky,
“Resolution-robust large mask inpainting with fourier convolutions,”
arXiv preprint arXiv:2109.07161, 2021.
[44] X. Shen, Z. Cai, W. Yin, M. Müller, Z. Li, K. Wang, X. Chen, and
C. Wang, “Gim: Learning generalizable image matcher from internet
videos,” in The Twelfth International Conference on Learning Repre-
sentations, 2024.
[45] A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R.
Richter, and V. Koltun, “Depth pro: Sharp monocular metric depth in
less than a second,” in International Conference on Learning Represen-
tations, 2025.
