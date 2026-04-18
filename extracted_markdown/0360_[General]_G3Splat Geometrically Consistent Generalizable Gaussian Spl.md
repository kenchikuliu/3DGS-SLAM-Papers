<!-- page 1 -->
G3Splat: Geometrically Consistent Generalizable Gaussian Splatting
Mehdi Hosseinzadeh1*†
Shin-Fang Chng2*
Yi Xu2
Simon Lucey1
Ian Reid1,3
Ravi Garg1
1 Australian Institute for Machine Learning
2 Goertek Alpha Labs
3 MBZUAI
m80hz.github.io/g3splat
w/o priors
I1
I2
I1
I75
I150
...
(150 views, Sora)
DUSt3R Adaptation
VGGT Adaptation
with our priors
w/o priors
with our priors
(2 views, RE10K)
...
Figure 1. G3Splat enables geometrically consistent, pose-free generalizable Gaussian splatting across backbones. Left: our VGGT-
based [56] adaptation without / with the proposed priors. Right: our DUSt3R-based [58] adaptation without / with the proposed priors.
We visualize reconstructions on a Sora-generated video (150 input views) and RealEstate10K [75] (2 input views). Our priors encourage
geometrically consistent Gaussians and markedly reduce floating artifacts. Sora prompt: “Generate a video inside the Louvre Museum,
including the paintings.”
Abstract
3D Gaussians have recently emerged as an effective scene
representation for real-time splatting and accurate novel-
view synthesis, motivating several works to adapt multi-
view structure prediction networks to regress per-pixel 3D
Gaussians from images. However, most prior work extends
these networks to predict additional Gaussian parameters—
orientation, scale, opacity, and appearance—while relying
almost exclusively on view-synthesis supervision. We show
that a view-synthesis loss alone is insufficient to recover
geometrically meaningful splats in this setting. We analyze
and address the ambiguities of learning 3D Gaussian splats
under self-supervision for pose-free generalizable splatting,
and introduce G3Splat, which enforces geometric priors to
obtain geometrically consistent 3D scene representations.
Trained on RE10K, our approach achieves state-of-the-art
performance in (i) geometrically consistent reconstruction,
(ii) relative pose estimation, and (iii) novel-view synthesis.
We further demonstrate strong zero-shot generalization on
ScanNet, substantially outperforming prior work in both
geometry recovery and relative pose estimation. Code and
pretrained models are released on our project page.
*Equal contribution.
†Corresponding author.
1. Introduction
3D Gaussian splatting (3DGS) [31] has recently revolution-
ized 3D structure and appearance modeling from multi-view
images. Departing from traditional depth or point cloud
representations of the scene structure, 3D Gaussians implic-
itly model surface reflections and environment lighting to
encode view-dependent scene appearance. They are memory-
efficient compared to explicit volumetric alternatives and fa-
cilitate rendering of the scene from arbitrary viewpoints in a
fraction of a second. Due to these capabilities, 3D Gaussians
have become a prevalent choice for scene representation.
Generalizable Gaussian splatting extends 3DGS from
per-scene optimization to feedforward reconstruction of
novel views. Instead of fitting Gaussians separately for
each scene, recent methods train neural networks that pre-
dict 3D Gaussians directly from one or a few input im-
ages [3, 5, 28, 29, 48, 51, 59, 63, 68, 72]. Given these
sparse views, the feedforward networks output a set of Gaus-
sians (means, orientations, scales, and colors), achieving
photorealistic novel-view synthesis.
Most existing generalizable splatting frameworks are
built by adapting well-established geometry networks orig-
inally designed for dense depth [56, 62] or pixelwise 3D
point prediction [35, 58]. These networks typically use
image encoders to process one or multiple images, fol-
1
arXiv:2512.17547v1  [cs.CV]  19 Dec 2025

<!-- page 2 -->
lowed by decoders to predict Gaussian means as per-pixel
depthmaps [62] or 3D points in a canonical frame [35, 56,
58] for each frame. Additional decoders are appended to the
geometry predictors to predict Gaussian properties such as
orientation, scale, opacity and view-dependent color – typ-
ically without much foresight. These networks are usually
trained predominantly by minimizing view-synthesis loss on
a few target views, closely following existing self-supervised
depth estimators, but with a different image-formation mech-
anism due to the underlying change in scene representation.
However, this prevalent setup overlooks several key issues
inherited from the underlying 3DGS optimization:
• Overparameterization: 3D Gaussians are overparameter-
ized compared to depth maps or point clouds. Successful
estimation of 3D Gaussians typically requires a large num-
ber of densely sampled viewpoints. Few-shot 3DGS is
known to require priors [7, 77] even in optimization-based
settings; yet generalizable methods typically inherit the
3DGS parameterization without introducing correspond-
ing geometry constraints during training.
• Geometric ambiguity: Unlike per-pixel depths or 3D
point locations – which are uniquely defined (up to scale) –
multiple 3D Gaussian configurations can produce equally
valid renderings. This ambiguity means that purely image-
based loss can let the network converge to geometrically
incorrect solutions that still explain the views.
• Lack of heuristic: Successful per-scene Gaussian splat-
ting methods typically rely on non-differentiable heuris-
tics (i.e., splitting, duplication and pruning of Gaus-
sians).
However, existing generalizable methods are
trained purely via view-synthesis gradient loss and ne-
glect these heuristics; i.e. all Gaussians remain perpetually
alive throughout training.
As a result, existing generalizable Gaussian splatting
methods often converge to geometrically degenerate Gaus-
sians: while the predicted locations (means) remain reason-
ably accurate — benefiting from well-established depth or
pointcloud estimators – orientations, scales, and opacities
are often incorrect. As shown in Figure 2, existing generaliz-
able splatting approaches struggle to learn meaningful opaci-
ties, orientations, or scales when trained with view-synthesis
loss alone: Gaussian orientations (shown as normal maps)
are misaligned with the underlying surface, and 3D Gaus-
sians become unjustifiably elongated or collapsed (shown in
scales). We observe these degeneracies both in splat predic-
tors that anchor Gaussian centers via depth maps [5] and in
those that use per-pixel 3D point maps aligned to a common
reference frame [68]. Although we highlight these two repre-
sentative approaches in Figure 2, similar issues appear across
all existing generalizable splatting methods. We attribute
these geometry artifacts to the inherent overparameterization
of 3D splats combined with purely photometric supervision:
without additional structural priors, self-supervised learning
of Gaussian parameter is ill-posed.
In this work, we aim to systematically characterize what
constitutes a geometrically consistent Gaussian in the gen-
eralizable setting and propose priors that encourage such
configurations. We build on a DUSt3R-style [58] backbone
that predicts per-pixel 3D points for pairs of input images
defined in a single canonical frame. On top of this backbone,
we explore both standard 3DGS [31] and "surfel-like" 2DGS
representations, and study how our regularization strategies
improve the geometric quality of generalizable splatting.
Our main findings are:
• Aligning Gaussian’s orientations with dominant local sur-
face normals is crucial for resolving structural ambiguities.
This constraint provides the essential supervision for ori-
entation that existing frameworks currently lack.
• Standard depth-normal consistency losses [4, 26] used to
enforce this constraint are not straightforward to deploy
when learning splat predictors, see supplementary material
for details. Instead, directly enforcing consistency between
Gaussian orientations and their means by leveraging the
local image neighborhood of pixel-aligned 3D Gaussians
promotes stable training.
• Ensuring that predicted Gaussians are pixel-aligned is es-
sential for self-supervision. In particular, for methods that
predicts point cloud, violating pixel alignment leads to
structure-pose ambiguity.
• Splat prediction frameworks can be trained with both full-
rank 3D Gaussians and rank-deficient "surfel-like" Gaus-
sians. While the added flexibility of full-rank Gaussians
benefits accurate pose and structure estimation, when pro-
posed priors are applied. These differences start to dimin-
ish when the proposed priors are applied.
Our method outperforms prior work in novel-view syn-
thesis and, crucially, produces plausible scene geometry that
enables direct depth rendering from arbitrary viewpoints –
something existing generalizable methods cannot achieve.
These consistent virtual depths can be fused using a Trun-
cated Signed Distance Function (TSDF) [73]; and the re-
constructed meshes provide geometric comparison against
prior art, as shown in Figure 4. Beyond view synthesis,
our approach establishes state-of-the-art relative pose esti-
mation from image pairs, surpassing task-specific methods
such as RoMa [11], geometry-supervised approaches such
as DUSt3R [35, 58], and pose-free generalizable splatting
method [68] – despite using less data and weaker supervision
in some cases. Our method achieves state-of-the-art zero-
shot 3D reconstruction on ScanNet [9], outperforming both
pose-free and pose-required generalizable splatting methods.
Note that we choose DUSt3R [58] as the geometric back-
bone due to its relative simplicity (compared to [56]), and
its capability to infer 3D structure without requiring relative
pose between input images (unlike epipolar attention [3] or
2

<!-- page 3 -->
MVSplat
NoPoSplat
Ours
Gaussian Normals
Gaussian Normals
Surface 
Normal
Legend
Gaussian Isotropy s3/s1
Gaussian Isotropy s2/s1
Gaussian Isotropy s3/s1
Gaussian Isotropy s2/s1
Source
Image I
Source
Image I
Isotropy 
Legend
Isotropic 
Anisotropic
(a)
(d)
(g)
(b)
(e)
(h)
(c)
(f)
(i)
(a)
(d)
(g)
(b)
(e)
(h)
(c)
(f)
(i)
Figure 2. Qualitative comparison of predicted Gaussian parameters. For visualization, we denote by (s1, s2, s3) the sorted eigen-scales
of each Gaussian covariance in Equation (7), such that s1 ≥s2 ≥s3; the smallest scale s3 characterizes uncertainty along the surface normal
direction. Row 1 (ours) shows: (a) the source image to which Gaussians are aligned, (b) skewness of the estimated Gaussians within their
defining plane, and (c) predicted Gaussian orientations visualized as surface-normal maps. Rows 2 and 3 show results for NoPoSplat [68]
and MVSplat [5], respectively: (d/g) Gaussians’ elongation perpendicular to the dominant plane defined by it, (e/h) Gaussians’ skewness
within the dominant plane, and (f/i) normals to the dominant plane. Existing methods yield Gaussian orientations without clear geometric
meaning: MVSplat Gaussians (i) align mostly fronto-parallel to the source image plane, and NoPoSplat Gaussians orientations (f) strongly
depend on texture, spanning a few dominant directions inconsistent with scene geometry. Our method produces plausible, near-Manhattan
structured surface orientations. Baseline Gaussians exhibit significant elongation perpendicular to their dominant surfaces (visible as non-red
colors in d/g). Notably, our Gaussians remain relatively circular (blue color in b) on planar, textureless surfaces and become skewed ellipses
(red color in b) near sharp geometric edges such as shelves or wall corners.
cost-volume-based architectures [5]). However, our contribu-
tions are not tied to a specific backbone. To our knowledge,
this is the first work in generalizable Gaussian splatting that
systematically analyze and address the degeneracies in the
predicted Gaussian orientation and scale, which outperforms
the prior arts on various geometric measures. The proposed
regularizers can be seamlessly integrated into any other geo-
metric estimator – such as VGGT [56] – to construct corre-
sponding splatting frameworks; in Sec. 4.2 we demonstrate
the benefit of splattifying VGGT using our priors. We be-
lieve that the presented contributions lay the groundwork
for future research on training neural networks to predict
Gaussians form images, in both depth- and self-supervised
setups.
2. Related Work
Owing to the state-of-the-art real-time view synthesis per-
formance of 3D Gaussian splitting [31], significant effort
has been put into improving 3DGS for scenarios such as
few-view reconstruction [7, 24, 36, 54, 61, 77], dynamically
moving objects [39, 60, 65, 66], surface extraction [22, 26],
and incorporating object semantics into 3D reconstructions
[37]. Real-time simultaneous localization and mapping ap-
proaches have also adapted Gaussian splats as an inherent
scene representation [30, 41]. Additionally, Gaussian splats
have been used for generating geometrically consistent im-
ages and video sequences [57, 60].
The deep learning revolution of the last decade has sig-
nificantly influenced geometric inference from one or more
images. Earlier works focused on training neural networks
to map a single image to depth map obtained from range
sensors [13, 14, 33, 34, 45]. Multi-view extensions for these
supervised learning algorithms are well explored as well
[2, 21, 27, 53, 55, 67]. More recently, methods have explored
reconstructing registered sets of per-pixel point clouds from
multiple images, providing state-of-the-art relative pose and
scene structure [35, 58].
Additionally, it has been demonstrated that these feed-
forward geometry predictors can be trained without depth
sensors in a self-supervised manner by minimizing view
3

<!-- page 4 -->
synthesis losses [16, 18, 19, 71, 74]. Structure prediction
from single or few images has also been utilized as an
optimization-free building block in high-fidelity tracking and
mapping systems [76, 78]. Generalizable Gaussian Splat-
ting methods have evolved recently to learn neural networks
that predict 3D Gaussians explaining a scene directly from a
few images. We broadly categorize these methods into two
categories:
Pose-Dependent Generalizable 3DGS: Several works
assume input images come with known or precomputed
poses (e.g., via SfM) and focus on designing architectures
to infer 3D Gaussians from these posed views [3, 5, 15,
43, 52, 59, 63, 72]. A prominent example is pixelSplat [3],
which introduced a two-view feed-forward network that uti-
lizes epipolar cross attention transformer architecture to fuse
multi-view information and predict per-pixel depth distri-
bution for input images. This distributions are sampled
to create a set of 3D Gaussian centers along the viewing
rays. MVSplat [5] uses cost volume based fusion of multi-
view information, adapting the Unimatch [62] architecture
to regress for depth instead. Both methods use additional de-
coder heads to estimate rest of the 3D Gaussian parameters.
Pose-Free Generalizable 3DGS: An emerging frontier
involves dispensing with known camera poses—allowing
the network to infer scene geometry and camera registration
jointly from images alone [28, 29, 48, 68]. Early efforts in
this direction often build upon learned stereo matching. For
example, [48] tackles uncalibrated stereo pairs by extend-
ing a foundation model (MASt3R [35]) that predicts dense
point clouds from two images. It then outputs 3D Gaussians
directly in a canonical frame, augmenting each point in the
MASt3R reconstruction with color and covariance attributes.
This process is supervised using the geometry of the 3D
point cloud and followed by a novel-view synthesis stage to
fine-tune appearance. NoPoSplat [68] adopts a more self-
supervised, multi-view approach by anchoring one view’s
coordinate system as canonical and training a network to
predict all Gaussians directly in that space, using only a
photometric loss for training.
To the best of our knowledge, all aforementioned general-
izable splitting methods struggle to learn geometrically faith-
ful orientations and scales for 3D Gaussians. The proposed
approach alleviates this issue from generalizable splatting
using appropriate geometric priors.
3. Method
In this section, we present our generalizable Gaussian splat-
ting framework and loss functions we propose to address
the ill-posed nature of self-supervised learning in predicting
geometrically consistent Gaussians. For the architectural
details, we refer the reader to the supplementary material.
Problem Definition. Assuming that we are given a set of
sparse images I = {It ∈RH×W ×3}T
t=1 (which is also
known as context images in [3, 5, 29, 48, 63, 68]), each
with known camera intrinsics that form the set K = {Kt ∈
R3×3}T
t=1 capturing a rigid scene, our aim is to learn a
feedforward neural network fΘ that maps these images and
intrinsics (I, K) to a set of pixel-aligned Gaussians as:
fΘ(I, K) =
n
Gj
t :=

µj
t, αj
t, qj
t , sj
t, cj
t
oj=1:H×W
t=1:T
, (1)
where Gj
t is the 3D Gaussian defined in the 3D space corre-
sponding to a pixel j in image t. Each Gj
t is characterized by
its center µ ∈R3; orientation represented by a unit quater-
nion vector q ∈R4; three scale parameters s ∈R3 defining
the elongation of the 3D Gaussians; opacity α ∈R; and
color encoded as spherical harmonics c ∈Rd. In addition to
the prevalent 3DGS representation adopted by generalizable
Gaussian splatting frameworks [3, 5, 63, 68], we also explore
the 2DGS representation introduced by [26], which models
the scene with 2D surfels instead of standard 3D Gaussians.
We present extensive evaluations and ablations quantifying
the impact of this representation on generalizable Gaussian
splatting in Section 4 and in the supplementary material.
Note that both Gaussian centers µj
t and orientations qj
t
are defined in the image coordinates of the first image I1.
Given these M × N × T Gaussians predictions, we render
novel views of the scene {ˆIf ∈RH×W ×3}F
f=1 ⊂I from F
different viewpoints defined by its projection matrix Pf =
(Rf, Tf) ∈SE(3) to be matched with its observed images
Ifs during training.
We minimize a view-synthesis loss [3, 5, 68] between the
ground-truth image If and the rendered image ˆIf:
Lsynthesis =
F
X
f=1
h
Lrgb(If,ˆIf) + Llpips(If,ˆIf)
i
.
(2)
For a pixel (u, v) in view f, the rendered color ˆIf(u, v) is
obtained by alpha-blending K depth-sorted projected Gaus-
sians G′
k:
ˆIf(u, v) =
K
X
k=1
ck wk(u, v),
(3)
wk(u, v) = Tk(u, v) αk G′
k(u, v),
(4)
Tk(u, v) =
Y
i<k
 1 −αi G′
i(u, v)

,
(5)
where ck and αk denote the color and opacity of Gaussian
k, and G′
k(u, v) is its 2D footprint in the image plane of If
(see supplementary material for details).
As shown in Section 4, solely relying on view synthesis
loss is proven to be insufficient for learning geometrically
meaningful Gaussians. In this work, we propose to mini-
mize two additional regularization losses: (i) a depth-surface
normal consistency term Lorient to align the orientations of
4

<!-- page 5 -->
the Gaussians with dominant surface normal of the scene;
(ii) a grid alignment loss Lalign to ensure that the estimated
Gaussians are aligned with the pixels of the provided images.
Combining these two regularization with the view synthesis
loss, we define our training objective function Ltotal as
Ltotal = Lsynthesis + λoLorient + λaLalign,
(6)
where λo and λa are weighting factors balancing the influ-
ence of each regularization (see Section 8). We discuss the
motivation, formulation and impact of the regularization
term in the following sections.
3.1. Learning Gaussian’s Orientations
Recall that existing pose-free and pose-aware generalizable
Gaussian splatting approaches struggle to learn meaningful
Gaussian orientations; see Figure 2. To endow the orien-
tations with geometric meaning, we align them with the
dominant surface normals of the underlying scene.
In both the 3DGS and 2DGS parameterizations, each
Gaussian Gj
t is associated with a covariance matrix
Σj
t = R(qj
t ) diag
 [sj,1
t , sj,2
t , sj,3
t ]⊤
R(qj
t )⊤,
(7)
where R(qj
t ) ∈SO(3) is the rotation induced by the quater-
nion qj
t and sj,1
t , sj,2
t , sj,3
t
≥0 denote the axis-aligned scales
in the local Gaussian frame. We define the Gaussian normal
N j
t as the unit eigenvector of Σj
t corresponding to its small-
est eigenvalue (equivalently, the column of R(qj
t ) associated
with arg mink sj,k
t ).
In the 2DGS variant [26], each Gaussian is surfel-like
and is parameterized by only two in-plane scales. For nota-
tional consistency with Equation (7), we represent this by
appending a third scale fixed to zero, i.e., we set sj,3
t
= 0
in the diagonal of Σj
t. This yields a rank-deficient covari-
ance whose null-space direction defines N j
t as in [26] (and
arg mink sj,k
t
is attained by the appended zero scale). While
this 2DGS specialization reduces over-parameterization, the
view-synthesis loss in Eq. (2) still provides only weak super-
vision for learning orientations, for both 3DGS and 2DGS.
A natural alternative is the rendered normal–depth consis-
tency regularizer from [26]; however, naively deploying this
rasterization-based regularizer in our generalizable setting
does not yield stable training. We analyze this behavior and
discuss remedies in the supplementary material.
Orientation supervision from local geometry. Instead, we
directly supervise orientations using local surface normals
estimated from neighboring Gaussian means. Assuming
each Gaussian Gj
t is aligned with pixel j = (u, v) in frame
t, we define central differences
∆xµ(u,v)
t
= µ(u+1,v)
t
−µ(u−1,v)
t
,
∆yµ(u,v)
t
= µ(u,v+1)
t
−µ(u,v−1)
t
,
(8)
and compute a local surface normal from the corresponding
3D points as
ˆ
N j
t =
∆yµ(u,v)
t
× ∆xµ(u,v)
t

∗,
(9)
where ∥· ∥∗denotes vector normalization and µ(u,v)
t
is the
3D mean of the Gaussian aligned with pixel (u, v).
Edge-aware orientation loss. Normals derived from local
finite differences are least reliable at depth discontinuities.
To reduce the influence of such pixels, we use an edge-aware
weight based on the local 3D variation:
dj
t =
∆xµ(u,v)
t

2 +
∆yµ(u,v)
t

2,
η = Quantileq

{dj
t}

,
wj
t = w0 exp

−κ dj
t/(η + ϵ)

,
(10)
with fixed constants (w0, κ), q-quantile normalization, and a
small ϵ for stability (see Section 8). We then encourage the
predicted Gaussian normal N j
t (smallest-eigenvalue direc-
tion for 3DGS, null-space direction for 2DGS) to agree with
ˆ
N j
t using a Huber penalty in cosine space:
Lorient = 1
|Ω|
X
(t,j)∈Ω
wj
t Hδ

1 −⟨N j
t , ˆ
N j
t ⟩

,
(11)
where ⟨·, ·⟩denotes the dot product, Hδ(·) is the Huber loss
with threshold δ, and Ωdenotes the set of valid interior
pixels over all frames (excluding a one-pixel boundary where
central differences are undefined), i.e., |Ω| = T(H−2)(W −
2).
Unlike rasterization-based priors, Lorient depends only on
the predicted Gaussian means and covariances, providing
direct supervision of orientation given µj
t. This formulation
is representation-agnostic and applies equally to 3DGS and
2DGS. Conceptually, it mirrors supervised surface-normal
training [12], where reference normals are derived from
depth via Equation (9). In our experiments, this simple
formulation consistently outperforms alternative orientation
regularizers and can also be used in depth-supervised training
of generalizable Gaussian splats.
Scale regularization (3DGS only). In our 3DGS variant,
we additionally apply an anisotropy bias on the Gaussian
scales to discourage near-isotropic covariances. Concretely,
we add a penalty on the minimum per-Gaussian scale,
Lflat = 1
P
X
t,u,v
min
 s(u,v),1
t
, s(u,v),2
t
, s(u,v),3
t

,
Ltotal ←Ltotal + λflat Lflat.
(12)
where the scales are those used in Equation (7), λflat is the
weight, and P = TWH is the total number of pixel-aligned
Gaussians in the scene. Unless stated otherwise, for 3DGS
we apply this term together with the orientation loss (i.e.,
whenever Lorient is enabled). This term is disabled for 2DGS,
where s(u,v),3
t
= 0 by construction.
5

<!-- page 6 -->
3.2. Pixel-aligned Gaussians
Although the first generalizable splatting approach [3] op-
erates in a pose-aware setup and adapts a two-view depth
prediction network, it by construction constrains every Gaus-
sian to lie on its corresponding viewing ray. Pose-free vari-
ants [68] drop the camera-pose assumption by directly esti-
mating Gaussian locations in a canonical space using a DPT
decoder. While this removes the need to warp Gaussians with
known cameras, the resulting parameterization renders struc-
ture estimation ill-posed, especially in the self-supervised
regime. In contrast to depth-supervised frameworks such
as DUSt3R [58], which learn an implicit structural prior by
enforcing the reconstructed 3D point cloud to project onto a
regular image grid, the pure view-synthesis loss in Eq. (2)
does not impose such a constraint. Gaussians can therefore
move freely into geometrically degenerate configurations,
degrading both structure and relative pose estimation, while
still explaining the input appearance.
We therefore explicitly align each Gaussian with its
pixel’s viewing ray. Concretely, for each pixel (u, v) in
frame t, the Gaussian center µ(u,v)
t
should reproject to (u, v)
under the corresponding camera intrinsics Kt and extrin-
sics (Rt, Tt). We enforce this constraint with the alignment
loss
Lalign =
1
P
t,u,v M(u,v)
t
T
X
t=1
W
X
u=1
H
X
v=1
M(u,v)
t
ℓalign(u, v, t),
(13)
ℓalign(u, v, t) =
 u
v⊤−Π
 Kt[Rt | Tt]˜µ(u,v)
t
2
2,
(14)
where ˜µ(u,v)
t
=
h
(µ(u,v)
t
)⊤
1
i⊤
, M(u,v)
t
=1 if the pro-
jected mean lies within the image bounds and has pos-
itive depth (and 0 otherwise), and Π([X, Y, Z]⊤)
=
[X/Z, Y/Z]⊤denotes the perspective projection.
As shown in Section 4, Lalign plays an important role in
PnP-based relative pose estimation (Tables 1 and 2) as well
as accurate depth/structure estimation (Table 3).
4. Experiments
Datasets and implementation details. Following [3, 5, 68],
we train our models on the large-scale RealEstate10K [75]
(RE10K) dataset, with the train-test splits used by [68].
RE10K comprises predominantly indoor real-estate videos
from YouTube, containing 67,477 training and 7,289 testing
scenes, with camera poses computed using COLMAP [46].
For evaluating generalization, we further test on two addi-
tional datasets: ACID [40], containing aerial nature scenes
captured by drones (with COLMAP-computed poses), and
ScanNet [9], an RGB-D indoor scene dataset with distinct
camera motion and characteristics. Specifically, we evaluate
relative pose and geometry (depth and mesh) estimation on
the ScanNet. Our training broadly follows recent generaliz-
able splatting methods; see supplementary material for full
details.
4.1. Relative Pose Evaluation
Relative pose is evaluated by computing the AUC of the
cumulative pose error curve at three thresholds. We report
results deploying a PnP + RANSAC algorithm to align the
Gaussian means with image grid as proposed in DUSt3R[58].
NoPoSplat [68] proposes a gradient-descent relative pose
refinement in which the predicted Gaussians are rendered
to generate optimal input image pairs for refining the pose
obtained by PnP+RANSAC. Pose Jacobians from [41] are
used for this refinement over a fixed number of iterations.
Table 1 compares relative pose estimation performance
across methods that do not require pose at inference (pose-
free). CoPoNeRF [25] is trained on RE10K and ACID with
explicit pose supervision. DUSt3R [58] leverages indoor
RGB-D and Internet SfM datasets (e.g., ScanNet++ [69],
MegaDepth [38]) with a 3D regression objective supervis-
ing both depth and camera pose. MASt3R [35] follows a
similar regime but additionally includes large-scale outdoor
sequences from the Waymo Open Dataset [50]. RoMA [11]
is trained on MegaDepth and ScanNet with joint depth-and-
pose supervision. In contrast, our models use no explicit
depth supervision and are trained solely on RE10K. Despite
this, we outperform all these methods by a large margin, both
on the in-domain RE10K test set and in zero-shot evaluations
on ACID and ScanNet. The sole exception is RoMA [11] on
ScanNet, the dataset it was explicitly trained on for relative
pose estimation.
Compared to NoPoSplat, our method yields substantial
relative-pose gains using only PnP+RANSAC. Incorporating
the alignment loss Lalign produces marked improvements
in both in-domain and cross-domain zero-shot tests, while
the orientation loss Lorient provides a further pose-estimation
boost, with the combined full loss achieving the best perfor-
mance. As in NoPoSplat, minimizing the input-image syn-
thesis loss also benefits our pose estimation. Although gains
on RE10K and ACID were modest, we observe proportion-
ally larger improvements on ScanNet with this optimization.
We refer the reader to the supplementary material for full
implementation details and additional ablations of our losses
under different pose estimation settings.
To further assess the quality of the predicted Gaussians
structure under our proposed losses, we use pose estimation
via PnP-only (least-squares optimization, without RANSAC)
as a proxy for structure evaluation. Specifically, we estimate
the relative pose between input views for both 3DGS and
2DGS representations on RE10K and ScanNet, as reported
in Table 2. We deliberately use PnP-only in this comparison
to more clearly expose structural differences: RANSAC’s
outlier rejection can otherwise mask the impact of our priors.
6

<!-- page 7 -->
Table 1. Pose estimation (AUC) at multiple error thresholds on RE10K [75] (in-domain) and on ScanNet [9] and ACID [40]
(cross-domain). The overall best results are shown in bold. and the best result in each category is underlined. Methods marked with † are
trained on additional data (e.g., ScanNet, ScanNet++, ACID), and those marked with ‡ use extra supervision (e.g., ground-truth depth). ∧
indicates evaluation with PnP+RANSAC only (no photometric pose refinement).
Pose
Estimation
Method
RE10K
ScanNet
ACID
Method
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
CoPoNeRF† [25]
0.161
0.362
0.575
-
-
-
0.078
0.216
0.398
DUSt3R†‡ [58]
0.301
0.495
0.657
0.085
0.210
0.398
0.166
0.304
0.437
MASt3R†‡ [35]
0.372
0.561
0.709
0.083
0.200
0.381
0.234
0.396
0.541
RoMa†‡ [11]
0.546
0.698
0.797
0.168
0.361
0.575
0.463
0.588
0.689
PnP+RANSAC
Splatt3R† [48]
0.158
0.325
0.504
0.011
0.042
0.119
0.044
0.121
0.260
SelfSplat [29]
0.030
0.083
0.180
0.030
0.098
0.254
0.064
0.153
0.283
NoPoSplat∧[68]
0.572
0.728
0.833
0.078
0.198
0.394
0.337
0.497
0.646
Ours (3DGS+Align+Orient)
0.629
0.770
0.858
0.124
0.282
0.493
0.404
0.560
0.689
w/ Refinement
(same loss as train)
SelfSplat [29]
0.031
0.086
0.182
0.033
0.112
0.274
0.069
0.156
0.285
NoPoSplat [68]
0.672
0.791
0.868
0.109
0.256
0.463
0.456
0.593
0.705
Ours (3DGS+Align+Orient)
0.684
0.801
0.875
0.148
0.326
0.540
0.466
0.598
0.713
Inputs
pixelSplat
MVSplat
DepthSplat
NoPoSplat
Ours
GT RGB/Depth
Figure 3. Qualitative comparison of rendered novel-view depth on RE10K (first row), ACID (second row), and ScanNet (last row).
We report average translation and rotation errors (in de-
grees), where the (scale-invariant) translation error is defined
as the angular difference between normalized predicted and
ground-truth translation vectors. The results show that our
priors yield significantly lower translation and rotation er-
rors, indicating more accurate Gaussian mean predictions
compared to the baseline without priors. All comparisons
are conducted under identical hyper-parameter settings.
4.2. Geometry Evaluation
Geometric veracity of the estimated 2D/3D Gaussian splats
is the key focus of this work. Traditionally, the geometry
predicted by neural networks is evaluated by measuring the
depth errors for the input views. However, input depths do
not capture the interpolation capability of predicted Gaus-
sians and are insensitive to the opacity, orientation, and scale.
7

<!-- page 8 -->
No Prior (VGGT)
With Our Prior (VGGT)
Ground Truth
Input Views
No Prior (DUSt3R)
With Our Prior (DUSt3R)
Figure 4. Qualitative ablation of reconstructed meshes on ScanNet [9] (2 input views) using VGGT [56] and DUSt3R [58] backbones.
Our proposed priors consistently yield sharper, more complete, and less noisy geometry across both backbones.
Table 2. Ablation of pose-estimation translation and rotation
errors using PnP (least squares).
Estimation
Method
Method
RE10K
ScanNet
Trans.↓
Rot.↓
Trans.↓
Rot.↓
PnP (LS)
3DGS (No Prior)
27.423
37.560
56.839
144.950
Ours (3DGS+Align)
11.504
18.861
31.170
58.516
Ours (3DGS+Orient)
10.689
18.626
31.108
55.403
Ours (3DGS+Align+Orient)
10.594
18.587
30.848
55.294
2DGS (No Prior)
29.693
47.225
54.057
82.707
Ours (2DGS+Align)
15.667
23.619
23.271
11.729
Ours (2DGS+Orient)
34.414
116.029
67.849
60.254
Ours (2DGS+Align+Orient)
8.495
17.420
31.474
60.480
We propose a more holistic evaluation of the predicted scene
structure by rendering multiple virtual depth maps from the
reconstructed Gaussians and reporting Absolute Relative Er-
ror and depth accuracy for two different thresholds. As we
do not aim to extrapolate beyond the given view frustum,
we use the same view-synthesis test set for depth evalua-
Table 3. Depth estimation on ScanNet [9]: novel vs. source
views. We report novel- and source-view depth accuracy side by
side for each metric. A dash (–) indicates a metric is not applicable.
Method
Abs Rel ↓
δ1 < 1.10 ↑
δ1 < 1.25 ↑
Novel Source Novel Source Novel Source
Supervised DUSt3R [58]
–
0.059
–
0.886
–
0.967
Pose-
required
pixelSplat [3]
0.299
0.288
0.552
0.553
0.818
0.820
MVSplat [5]
0.189
0.132
0.412
0.641
0.745
0.891
FreeSplat [59]
0.126
0.124
0.556
0.556
0.831
0.833
DepthSplat [63]
0.135
0.105
0.578
0.722
0.864
0.914
Pose-
free
Splatt3R [48]
0.148
–
0.546
–
0.806
–
SelfSplat [29]
0.160
0.155
0.502
0.460
0.810
0.801
NoPoSplat [68]
0.131
0.121
0.554
0.662
0.851
0.869
Ours (3DGS+Align+Orient) 0.090
0.082
0.713
0.740
0.916
0.928
Table 4. Depth estimation ablation on ScanNet [9]: novel views.
Gray columns use the test-time pose refinement scheme (same loss
as training) prescribed in [68].
Abs Rel ↓
δ1<1.10 ↑
δ1<1.25 ↑
Model
w/o
w/
w/o
w/
w/o
w/
3DGS representation
3DGS (No Prior)
0.106
0.102
0.688
0.715
0.897
0.901
Ours (3DGS+Align)
0.097
0.089
0.701
0.729
0.907
0.920
Ours (3DGS+Orient)
0.093
0.085
0.707
0.733
0.913
0.925
Ours (3DGS+Align+Orient)
0.090
0.083
0.713
0.738
0.916
0.928
2DGS representation
2DGS (No Prior)
0.121
0.114
0.668
0.692
0.879
0.884
Ours (2DGS+Align)
0.107
0.099
0.684
0.713
0.894
0.908
Ours (2DGS+Orient)
0.097
0.090
0.704
0.726
0.909
0.920
Ours (2DGS+Align+Orient)
0.094
0.082
0.715
0.743
0.916
0.931
tion. Virtual depth maps are rendered using the ground-truth
relative pose w.r.t the first input frame, assuming perfectly
aligned multi-view Gaussians. This puts pose-free meth-
ods at a severe disadvantage – small pose alignment errors
amplify depth errors – yet they outperform pose-aware coun-
terparts by a large margin, as shown in Table 3. Our approach
substantially outperforms the baselines. As shown by the
ablations in Table 4, the alignment and orientation losses
provide markedly larger gains. In particular, we observe that
Lalign alone is insufficient to learn stable Gaussian orienta-
tions, whereas adding Lorient enforces geometry-consistent
orientations and yields sharper novel-view renderings (color,
depth, and surface normals), see supp. for details.
We compare novel-view depth estimation of our method
against pixelSplat [3], MVSplat [5], NoPoSplat [68] and
DepthSplat [63] in Figure 3. MVSplat, DepthSplat and No-
PoSplat depths are hypersensitive to texture, while pixelSplat
produces notably noisier depths in textureless regions. In
8

<!-- page 9 -->
contrast, our method yields more plausible depths despite
not requiring relative poses.
While our primary goal is to predict pixel-aligned, geo-
metrically consistent Gaussians for novel-view depth render-
ing, we also benchmark source-view depth estimation accu-
racy of all baseline methods in Table 3. For each method,
we report their best depth—whether rendered from Gaus-
sians or predicted by their depth-estimation head—under
its best-performing configuration. For example, pixelSplat
attains its highest accuracy using rendered depth, whereas
MVSplat and DepthSplat perform best with their network-
predicted depths. Our method achieves the lowest AbsRel
error and performs competitively in thresholded accuracy.
More detailed results for one- and two-view depth prediction
are provided in the supplementary material.
Mesh Evaluation.
To further assess geometric consistency,
we reconstruct scene meshes using only virtual novel views.
For each scene, we first estimate a Gaussian-splatting rep-
resentation from two source images; we then synthesize an
interpolated camera trajectory between the sources, render
per-view depth maps along this path, and fuse them with
TSDF-Fusion [8] to obtain a surface mesh. We compare the
reconstructed meshes (with and without our proposed priors)
against the ground-truth meshes on ScanNet [9] using stan-
dard metrics [23, 76]: accuracy, completeness, and Chamfer
distance. As shown in Table 5, our priors outperform the
baselines across all metrics, indicating a consistent and ac-
curate Gaussian-splatting representation of the scenes. Note
that our model is not trained on ScanNet; all evaluations
are zero-shot. Qualitative comparisons in Figure 4 further
show that our priors yield more accurate and more complete
reconstructions than the baselines for both DUSt3R [58]
and VGGT [56] backbone architectures. Additional details
on the reconstruction protocol and metric computation are
provided in the supplementary material.
Adaptation of VGGT. For our VGGT-based variant, we
keep the pose and point-cloud branches of VGGT [56] intact
and append an additional decoder that predicts Gaussian
parameters. We train this splat predictor purely with a view-
synthesis loss, without using the depth branch. Keeping
spirit of self supervised structure learning, we remove the
pseudo-depth supervision as used in [28]. We found that
accounting for the different intrinsic and extrinsic parameter
conventions used by VGGT is critical for stable training. To
that end following [28] we include a pseudo–pose supervi-
sion term to keep the predicted camera parameters consistent.
We evaluate VGGT-based splatting both with and without
our proposed priors in Table 5. Figure 4 shows the recon-
structed meshes obtained with and without deployment of
prior on handpicked ScanNet scenes. From these results,
it is clear that our quick adaptation of VGGT for learning
3D splat gives better results than the DUSt3R-based adapta-
Table 5. Mesh reconstruction ablation on ScanNet [9] (2 source
views). Under the TSDF-Fusion [8] protocol, our method signifi-
cantly outperforms all baselines across all metrics.
Acc↓
Comp↓
Chamfer Dist.↓
No Prior (DUSt3R)
0.266
0.514
0.390
With Our Priors (DUSt3R)
0.255
0.498
0.377
No Prior (VGGT)
0.150
0.362
0.256
With Our Priors (VGGT)
0.139
0.349
0.244
tions presented before. Additionally, proposed priors indeed
improve the VGGT based splat predictors both in terms of
accuracy and completeness.
4.3. Novel View Synthesis Evaluation
We noted that the proposed method outperforms prior work
in novel-view synthesis on the RE10K dataset, largely thanks
to its warping-free formulation. Small improvements were
observed due to priors (see supp. for detailed results).
5. Conclusion
We propose a novel self-supervised, generalizable splatting
network that mitigates geometric inconsistencies in Gaus-
sian splat recovery previously overlooked by the community.
Our model produces state-of-the-art, geometrically consis-
tent Gaussian splats from just two unposed images. While
we train on RE10K using an asymmetric transformer archi-
tecture under self-supervision, our core contributions are
invariant to these design choices. The priors introduced
here will help future work on generalizable splatting and
learning-based 3D scene recovery.
9

<!-- page 10 -->
G3Splat: Geometrically Consistent Generalizable Gaussian Splatting
Supplementary Material
In this supplementary material, we first present additional
quantitative and qualitative evaluations for both our DUSt3R-
(Section 6.2) and VGGT-based (Section 6.1) generalizable
splatting variants. We then describe the underlying architec-
tures for these backbones (Section 7), followed by implemen-
tation details (Section 8) and baseline protocols. Next, we
discuss two key design choices: (i) rendered normal–depth
consistency (Section 9) for the 2DGS variant and its relation
to our proposed orientation prior, and (ii) the depth render-
ing strategy (Section 10) adopted in our model. We also
discuss our pose refinement strategy in Section 11. Finally,
we outline our mesh reconstruction and evaluation protocol
(Section 12).
6. Additional Evaluations
6.1. Multi-view Results on VGGT-based Adaptation
Architecture of VGGT [56] offers the flexibility for easy
adaptation of the trained splat predictor to be seamlessly
used with single as well as large baseline multiple frames
typically used in multi-view structure-from-motion (SfM)
frameworks.
We train our VGGT-based splat predictor using only two
context (source) views, without employing additional views
for the view-synthesis loss (see Sections 7 and 8). Never-
theless, the resulting model can be applied as is in multi-
view setups. We also find that the priors proposed in this
work remain beneficial in this regime—and in fact become
even more critical in this multi-view testing setup. To show-
case the multi-view capabilities of the trained model, we
select 24 test frames with non-degenerate baselines from the
RE10K [75] test set, feed them jointly as input, and directly
visualize the predicted Gaussians in Figure 6.
It can be seen that, similar to VGGT, our splat predic-
tors generalize seamlessly to multi-view inputs. While the
prior-free baseline trained only with a view-synthesis loss
works out of the box, the visualizations reveal clear mis-
alignment of Gaussian means across frames, indicating poor
relative pose estimation in the vanilla VGGT adaptation. As
a result, the predicted Gaussians are difficult to consolidate
into a coherent, holistic 3D representation. Moreover, as
the number of input views and the baseline increase, these
misalignments accumulate and the baseline degrades further.
In contrast, when our priors are imposed, Gaussians from
different frames are well aligned, indicating high out-of-the-
box pose estimation accuracy. The predicted Gaussians can
therefore be used directly as proxy 3D maps, without any
post-processing heuristics such as bundle-adjustment–based
pose correction. Even without pose refinement to fix mis-
alignments, our prior-augmented model produces Gaussians
that yield high-fidelity renderings, whereas the baseline adap-
tation struggles and often produces distorted or smoothed-out
novel views.
Figure 5 illustrates the effectiveness of our priors on a 150-
view virtual museum sequence generated by Sora [44]. The
baseline result exhibits noticeable trails from misaligned per-
frame Gaussians, particularly near scene boundaries where
the camera undergoes large motions. Even in regions with
smoother camera motion, gradual drift in the Gaussians
leads to smeared reconstructions (left). In contrast, the prior-
assisted model is largely free from both large misalignments
and small drifts, yielding cleaner geometry and sharper tex-
tures across most observed areas. Figure 7 further shows
that these priors generalize to the out-of-domain Tanks and
Temples dataset, despite being trained only on RE10K [75].
6.2. Additional Evaluations of the DUSt3R-based
Adaptation
6.2.1. Source View Depth from Single Image
Using Gaussian splats as a scene representation naturally sup-
ports both novel-view synthesis and holistic 3D reconstruc-
tion. While most generalizable splatting methods emphasize
image interpolation between input views, our work places
stronger focus on geometric consistency and full 3D recon-
struction, with interpolation capabilities. On ScanNet [9],
our Gaussian-splat–based approach achieves the best source-
view depth accuracy among self-supervised splatting base-
lines (Table 3), particularly in terms of AbsRel, while remain-
ing competitive with stronger supervised baselines. This mo-
tivates us to further examine whether replacing depth-map
or point-cloud representations with Gaussian splats brings
tangible benefits for source-view depth prediction.
As our method is fully self-supervised, the fairest compar-
ison would be against other self-supervised approaches that
take two uncalibrated views and are trained on RE10K [75].
However, such two-view self-supervised structure estima-
tors are scarce. Instead, we evaluate on the well-studied
single-view depth estimation setting, where a range of
strong monocular baselines is available. We follow the
DUSt3R [58] single-view depth evaluation protocol by du-
plicating each input image into both views. As in Table 3,
for each splatting-based baseline we report its best depth
estimate in Table 6, choosing between depth rendered from
Gaussians and depth read directly from pixel-aligned 3D
Gaussian means.
In addition to splatting-based baselines, we include three
state-of-the-art self-supervised single-view depth estima-
1

<!-- page 11 -->
VGGT Adaptation (With Our Priors)
VGGT Adaptation (No Prior)
(Input Images, 24 views, Sora)
...
...
Figure 5. Qualitative ablation of reconstructed Gaussians on a Sora-generated video (VGGT backbone, 24 input views). Prompt used
to generate the video: “A single unbroken orbital camera move through a vast, empty gothic library, with static architecture, medium-wide
framing, warm steady lighting, and crisp sharp geometric details”.
tors: MonoDepthV2 [20], SC-SfM-Learners [1], and SC-
DepthV3 [49]. On this NYUD-v2 [47] single-view bench-
mark, methods specifically designed for monocular depth
generally perform strongest, as expected.
Among the
splatting-based approaches, DepthSplat [63] attains the best
overall scores; this can be attributed in part to its use of a
strong depth encoder (DepthAnything-V2 [64]) pretrained
with dense depth supervision on a large corpus of depth
datasets, giving it a significant advantage in this setting.
Within the family of self-supervised, two-view gener-
alizable splatting methods, pixelSplat [3] struggles in the
duplicated-view (zero-baseline) setting, while MVSplat [5]
and NoPoSplat [68] provide much stronger baselines. Our
best variant, Ours (3DGS+Align+Orient), matches or slightly
improves upon NoPoSplat and the 3DGS no-prior baseline
across the reported metrics, while remaining competitive
with classical self-supervised single-view methods. This
experiment indicates that our geometric priors do not com-
promise, and in fact modestly improve, single-view depth
quality within the pose-free splatting regime, even though
our model is primarily optimized for multi-view reconstruc-
tion and pose estimation rather than specialized monocular
depth prediction.
6.2.2. Novel-View Synthesis Evaluation
While novel-view synthesis is not the primary focus of this
work, we evaluate both in-domain and zero-shot out-of-
domain performance against relevant baselines (Table 7 and
Table 8). For pose-free methods, novel views are rendered
directly from all reconstructed Gaussian splats at a fixed
2

<!-- page 12 -->
VGGT Adaptation Without Priors
VGGT Adaptation With Our Priors
Figure 6. Qualitative ablation of reconstructed Gaussians on RE10K [75] (VGGT backbone, 24 input views).
target pose relative to the first input image. In contrast, meth-
ods that predict Gaussian means as depth maps require an
additional warping step, using the ground-truth relative pose,
to align Gaussians to the first-view coordinate frame. Al-
though some works (e.g., [48]) report novel-view results
for DUSt3R [58] and MASt3R [35], these models are not
designed for view synthesis, so we omit them from our
depth and image-synthesis comparisons. Importantly, we do
not optimize camera poses for target-view image synthesis.
While pose refinement is common in some NeRF and 3DGS
pipelines, it can mask geometric inconsistencies and effec-
tively “peek” at the ground truth during synthesis [6, 17],
which we explicitly avoid here.
Our warping-free approach is consistently competitive
with, and typically slightly outperforms, prior art in novel-
view synthesis. Incorporating our proposed priors yields
3

<!-- page 13 -->
VGGT Adaptation Without Priors
VGGT Adaptation With Our Priors
Figure 7. Qualitative ablation of reconstructed Gaussians on Tanks and Temples [32] (VGGT backbone, 20 input views).
Table 6. Single-view depth estimation results on NYUD-v2 [47].
Each scene is reconstructed from a single image, which is dupli-
cated to form the two-view input required by our splatting-based
estimators. For every splatting-based method, we report its “best
depth estimate”, which may come either from rendered depth or
from an intermediate depth prediction. We compare these against
several state-of-the-art self-supervised single-view depth estimation
methods, alongside the splatting-based baselines.
Training
Scheme
Best Source-View Depths
Method
Abs Rel↓δ1 < 1.10 ↑δ1 < 1.25 ↑
Two-view
Supervised
DUSt3R [58]
0.065
-
0.941
Single-view
Self-Supervised
MonoDepthV2 [20]
0.162
-
0.745
SC-SfM-Learners [1]
0.138
-
0.796
SC-DepthV3 [49]
0.123
-
0.848
Two-view
Self-Supervised
Pose Req.
pixelSplat [3]
0.746
0.138
0.314
MVSplat [5]
0.281
0.277
0.574
DepthSplat [63]
0.112
0.619
0.880
Two-view
Self-Supervised
Pose Free
NoPoSplat [68]
0.172
0.423
0.749
3DGS (No Prior)
0.172
0.425
0.750
Ours (3DGS+Align)
0.190
0.396
0.720
Ours (3DGS+Orient)
0.203
0.356
0.687
Ours (3DGS+Align+Orient)
0.166
0.434
0.766
small but systematic improvements over pose-free meth-
ods on both RE10K [75] and ScanNet [9], with the largest
quantitative gains observed on the out-of-domain ACID [40]
benchmark. This behavior is consistent with our design
goal: the priors are primarily intended to improve geo-
metric consistency (depth, pose, and mesh quality), and
they achieve this without sacrificing—indeed, while slightly
improving—novel-view image quality.
6.2.3. Additional Geometry Evaluations
In addition to the results in the main paper, we provide
further qualitative comparisons of novel-view depth maps
rendered by our method on RE10K [75] and ScanNet [9] in
Figure 10.
We also report more detailed ablations of our proposed
priors for both pose and depth estimation. For pose, Ta-
ble 9 evaluates our models on RE10K [75], ScanNet [9],
and ACID [40] under three schemes: PnP (least squares),
PnP+RANSAC, and test-time photometric refinement (see
Section 11 for details). For depth, Table 10 reports novel-
and source-view depth error and accuracy on ScanNet [9].
Overall, the proposed priors yield substantial gains in ge-
ometric quality. In particular, under the PnP (least-squares)
setting they almost double the pose AUC on RE10K (in-
domain) and ACID (cross-domain), with even larger relative
improvements on ScanNet, indicating fewer outliers and
more reliable Gaussian means. Similarly, on ScanNet we
observe more than a 20% reduction in absolute relative depth
error for both novel and source views when both priors are
enabled. These geometric improvements are also apparent
in the reconstructed Gaussians on RE10K (Figure 8), which
exhibit cleaner geometry and markedly fewer floating arti-
facts.
As discussed in the main paper, Lalign alone is insufficient
to learn stable Gaussian orientations. Adding Lorient enforces
4

<!-- page 14 -->
Table 7. Novel-view synthesis performance on RE10K [75]. We compare pose-required and pose-free methods against our model across
different source-view overlap thresholds used in [68]. The best pose-free results (without target-pose optimization) are shown in bold, and
the top pose-required method is underlined.
Small
Medium
Large
Average
Method
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Pose-
Required
pixelNeRF [70]
18.417
0.601
0.526
19.930
0.632
0.480
20.869
0.639
0.458
19.824
0.626
0.485
AttnRend [10]
19.151
0.663
0.368
22.532
0.763
0.269
25.897
0.845
0.186
22.664
0.762
0.269
pixelSplat [3]
20.263
0.717
0.266
23.711
0.809
0.181
27.151
0.879
0.122
23.848
0.806
0.185
MVSplat [5]
20.353
0.724
0.250
23.778
0.812
0.173
27.408
0.884
0.116
23.977
0.811
0.176
FreeSplat [59]
19.411
0.691
0.277
22.839
0.790
0.192
26.433
0.869
0.130
23.026
0.788
0.196
DepthSplat [63]
22.820
0.798
0.193
25.383
0.851
0.145
28.317
0.900
0.104
25.595
0.852
0.145
Pose-
Free
Splatt3R [48]
14.352
0.475
0.472
15.529
0.502
0.425
15.817
0.483
0.421
15.318
0.490
0.436
CoPoNeRF [25]
17.393
0.585
0.462
18.813
0.616
0.392
20.464
0.652
0.318
18.938
0.619
0.388
SelfSplat [29]
15.557
0.572
0.435
19.648
0.703
0.301
24.142
0.817
0.191
19.931
0.704
0.303
NoPoSplat [68]
21.097
0.723
0.237
23.191
0.779
0.187
25.107
0.817
0.144
23.244
0.778
0.187
Ours (3DGS+Align+Orient) 21.221
0.731
0.235
23.347
0.785
0.185
25.418
0.826
0.141
23.417
0.783
0.185
Ours (2DGS+Align+Orient) 21.377
0.739
0.234
23.426
0.787
0.184
25.459
0.827
0.141
23.504
0.787
0.184
Input View 1
Input View 2
Input View 1
Input View 2
No Prior
Ours (3DGS+Align+Orient)
No Prior
Ours (3DGS+Align+Orient)
Figure 8. Qualitative ablation of reconstructed Gaussians on RE10K [75] (DUSt3R backbone, 2 input views).
Table 8. Zero-shot out-of-distribution novel-view performance
on ACID [40] and ScanNet [9]. We compare our models trained
with the proposed priors for both 2DGS and 3DGS representations.
All models are trained exclusively on the RE10K [75] dataset. Note
that these results are obtained without optimizing camera poses for
target views.
ACID
ScanNet
Method
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
NoPoSplat [68]
23.379
0.684
0.237
21.069
0.646
0.269
Ours (3DGS+Align+Orient) 23.763
0.700
0.236
21.137
0.648
0.269
Ours (2DGS+Align+Orient) 23.827
0.701
0.235
21.168
0.650
0.266
geometry-consistent normals and yields sharper novel-view
renderings, as illustrated in Figure 9.
Finally, Figure 11 compares mesh reconstructions from
two input views on ScanNet [9]. We evaluate pose-required
baselines MVSplat [5] and DepthSplat [63], the pose-free
NoPoSplat [68], and our method. Meshes are reconstructed
by fusing virtual (novel-view) rendered depth maps via
TSDF-Fusion [8] (see Section 12 for details). For each
method, we visualize Gaussian orientations (surface nor-
mals) for the first input view alongside the rendered depth for
a novel view and its ground-truth depth. Our approach con-
sistently produces geometrically coherent meshes, whereas
competing methods’ inconsistent novel-view depths often
lead to deformed planar regions (rows one and three) and
large holes (row two), losing fine scene detail.
7. Architectures
We instantiate our framework with two multi-view trans-
former backbones: a DUSt3R-style [58] encoder (similar in
spirit to [68]) and a VGGT-style generalist geometry trans-
former [56]. In both cases, the backbone produces per-image
feature maps and multi-view aggregated features, and our
Gaussian decoders predict per-pixel splat parameters (cen-
ters, scales, orientations, opacities, and colors) for 2DGS
and 3DGS parameterizations. All variants are trained with
5

<!-- page 15 -->
Table 9. Pose estimation ablation of our method. We report AUC at thresholds of 5◦, 10◦, and 20◦on RE10K [75] (in-domain) and
ScanNet [9] / ACID [40] (cross-domain). We evaluate both 2DGS and 3DGS models trained with the proposed priors under three pose
estimation schemes: (i) PnP (least squares), (ii) PnP+RANSAC, and (iii) photometric test-time optimization (same loss as training). The best
results within each pose estimation scheme are shown in bold.
Pose method
Model
RE10K
ScanNet
ACID
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
PnP (LS)
3DGS (No Prior)
0.296
0.437
0.570
0.019
0.050
0.104
0.189
0.279
0.378
Ours (3DGS+Align)
0.388
0.558
0.692
0.029
0.089
0.199
0.250
0.366
0.476
Ours (3DGS+Orient)
0.477
0.642
0.753
0.068
0.141
0.221
0.296
0.414
0.508
Ours (3DGS+Align+Orient)
0.482
0.645
0.753
0.052
0.121
0.209
0.301
0.416
0.511
2DGS (No Prior)
0.223
0.344
0.487
0.008
0.026
0.076
0.154
0.237
0.339
Ours (2DGS+Align)
0.338
0.519
0.668
0.031
0.083
0.176
0.229
0.341
0.453
Ours (2DGS+Orient)
0.194
0.252
0.314
0.015
0.035
0.055
0.049
0.063
0.093
Ours (2DGS+Align+Orient)
0.526
0.679
0.774
0.051
0.126
0.210
0.327
0.442
0.529
PnP+RANSAC
3DGS (No Prior)
0.572
0.728
0.833
0.078
0.198
0.394
0.337
0.497
0.646
Ours (3DGS+Align)
0.594
0.742
0.840
0.090
0.222
0.431
0.344
0.506
0.650
Ours (3DGS+Orient)
0.600
0.746
0.845
0.132
0.298
0.491
0.367
0.533
0.674
Ours (3DGS+Align+Orient)
0.629
0.770
0.858
0.124
0.282
0.493
0.404
0.560
0.689
2DGS (No Prior)
0.588
0.737
0.832
0.085
0.223
0.432
0.344
0.513
0.659
Ours (2DGS+Align)
0.619
0.759
0.849
0.120
0.279
0.471
0.382
0.540
0.674
Ours (2DGS+Orient)
0.592
0.743
0.836
0.099
0.241
0.448
0.374
0.535
0.672
Ours (2DGS+Align+Orient)
0.629
0.768
0.856
0.128
0.281
0.477
0.387
0.546
0.682
w/ Refinement
(same loss as train)
3DGS (No Prior)
0.672
0.791
0.868
0.109
0.256
0.463
0.456
0.593
0.705
Ours (3DGS+Align)
0.680
0.797
0.871
0.129
0.284
0.513
0.460
0.596
0.709
Ours (3DGS+Orient)
0.684
0.801
0.874
0.144
0.318
0.527
0.469
0.604
0.718
Ours (3DGS+Align+Orient)
0.684
0.801
0.875
0.148
0.326
0.540
0.466
0.598
0.713
2DGS (No Prior)
0.672
0.788
0.859
0.129
0.298
0.494
0.460
0.599
0.713
Ours (2DGS+Align)
0.681
0.799
0.870
0.136
0.311
0.512
0.474
0.607
0.718
Ours (2DGS+Orient)
0.675
0.793
0.869
0.130
0.301
0.503
0.466
0.601
0.714
Ours (2DGS+Align+Orient)
0.686
0.802
0.875
0.153
0.334
0.541
0.478
0.609
0.723
Table 10. Depth estimation ablation on ScanNet [9]: novel and source views. Gray columns use the test-time pose refinement scheme
(same loss as training) prescribed in [68].
Novel view
Source view
Abs Rel ↓
δ1<1.10 ↑
δ1<1.25 ↑
Abs Rel ↓
δ1<1.10 ↑
δ1<1.25 ↑
Model
w/o
w/
w/o
w/
w/o
w/
w/o
w/
w/o
w/
w/o
w/
3DGS representation
3DGS (No Prior)
0.106
0.102
0.688
0.715
0.897
0.901
0.105
0.097
0.689
0.707
0.897
0.905
Ours (3DGS+Align)
0.097
0.089
0.701
0.729
0.907
0.920
0.089
0.086
0.729
0.740
0.918
0.923
Ours (3DGS+Orient)
0.093
0.085
0.707
0.733
0.913
0.925
0.085
0.083
0.733
0.742
0.925
0.928
Ours (3DGS+Align+Orient)
0.090
0.083
0.713
0.738
0.916
0.928
0.082
0.080
0.740
0.747
0.928
0.930
2DGS representation
2DGS (No Prior)
0.121
0.114
0.668
0.692
0.879
0.884
0.118
0.105
0.665
0.705
0.875
0.894
Ours (2DGS+Align)
0.107
0.099
0.684
0.713
0.894
0.908
0.100
0.096
0.712
0.723
0.904
0.911
Ours (2DGS+Orient)
0.097
0.090
0.704
0.726
0.909
0.920
0.090
0.090
0.727
0.735
0.920
0.922
Ours (2DGS+Align+Orient)
0.094
0.082
0.715
0.743
0.916
0.931
0.086
0.079
0.736
0.752
0.925
0.934
the same view-synthesis objective and our proposed geo-
metric priors; only the choice of backbone and aggregation
mechanism differs.
DUSt3R-based variant. Our first instantiation builds on
6

<!-- page 16 -->
Input View 1
Input View 2
Gaussians Orientation View 1 
Rendered Depth (Novel View)
Gaussians Orientation View 1 
Gaussians Orientation View 1 
Rendered RGB (Novel View)
Rendered Depth (Novel View)
Rendered Depth (Novel View)
Rendered Normal (Novel View)
Surface 
Normal
Legend
Depth
Legend
Far
Close
Gaussians Orientation View 1 
Rendered Depth (Novel View)
No Priors
Gaussians Orientation View 1 
Gaussians Orientation View 1 
Gaussians Orientation View 1 
Rendered RGB (Novel View)
Gaussians Orientation View 1 
Rendered Depth (Novel View)
Rendered Depth (Novel View)
Rendered Depth (Novel View)
Rendered Normal (Novel View)
Rendered Depth (Novel View)
Input View 1
Input View 2
Figure 9. Qualitative ablation of the losses. We visualize the learned Gaussian orientations (from the first input view) along with rendered
novel-view depth, color, and surface normals, on RE10K [75]. As shown, using Lalign alone is insufficient to learn reliable Gaussian
orientations; adding Lorient encourages geometry-consistent orientations and, together, they produce more accurate rendered depth.
the pose-free, N-view transformer design pioneered by
DUSt3R [58], which was also adopted in [68] for gener-
alizable Gaussian splatting, and adapts it to predict Gaussian
splats with our 2DGS/3DGS parameterizations and priors.
The architecture comprises three main components: (i) a
transformer-based image encoder, (ii) two asymmetric multi-
view feature aggregators, and (iii) dense prediction decoders
for Gaussian parameters.
The image encoder maps each RGB frame and its intrinsic
parameters to a sequence of tokens (patch embeddings plus
camera embeddings), which are processed independently by
a ViT encoder. The resulting per-view features are then fused
by two sets of cross-attention-based aggregators: one pro-
duces features for the reference (first) image, and the other
produces features for the remaining images, conditioned
on the reference. This asymmetric aggregation follows the
spirit of DUSt3R, aligning all frames to a common canonical
coordinate frame without requiring ground-truth poses.
Given the aggregated features, two DPT-style decoders
predict Gaussian parameters. The first decoder regresses
the Gaussian centers (i.e., 3D positions associated with each
input pixel), while the second decoder predicts the remain-
ing parameters (scales, orientations, opacities, and colors),
optionally combining higher-resolution image features for
appearance. Because all predicted Gaussians are expressed
in a shared canonical frame, their union can be directly
rendered from arbitrary viewpoints using the differentiable
splatting pipeline, without explicit warping or known camera
poses. Both the image tokenizer and the feature aggregators
are built entirely from standard Vision Transformer blocks,
7

<!-- page 17 -->
Inputs
pixelSplat
MVSplat
DepthSplat
NoPoSplat
Ours
GT RGB/Depth
Figure 10. More qualitative comparison of novel-view rendered depth on RE10K [75] and ScanNet [9]. pixelSplat depths are relatively
geometrically consistent but noisy when the baseline is small. Large errors can be observed in the pixelSplat depths when the image overlap
is small. Other baselines provide depth maps, which are hypersensitive to image texture. While some potentially meaningful fine structural
edges are visible in these depth maps (see chair handles in row 4), the depth maps have many non-geometric “fake edges” (paintings in row
1, sofa in row 2, cabinet in row 3 – to name a few). Our method provides geometrically consistent renderings in all these scenarios.
without epipolar-specific attention or explicit multi-view
cost volumes, keeping the architecture geometry-free and
compatible with our priors.
VGGT-based variant. To demonstrate that our priors are not
tied to a specific backbone, we also instantiate the framework
with a VGGT-style architecture [56], which jointly predicts
camera poses and 3D structure from multiple views. We
retain the original pose and point-cloud branches of VGGT
8

<!-- page 18 -->
Input View 1
Input View 2
Gaussians Orientation
View 1 
Rendered Depth
Novel View
MVSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
DepthSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
NoPoSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
Ours
Groundtruth 
Depth
Input View 1
Input View 2
Gaussians Orientation
View 1 
Rendered Depth
Novel View
MVSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
DepthSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
NoPoSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
Ours
Groundtruth 
Depth
Input View 1
Input View 2
Gaussians Orientation
View 1 
Rendered Depth
Novel View
MVSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
DepthSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
NoPoSplat
Gaussians Orientation
View 1 
Rendered Depth
Novel View
Ours
Groundtruth 
Depth
Figure 11. Qualitative comparison of mesh reconstruction on ScanNet [9] (DUSt3R backbone, 2 input views). For each scene, we show
the two input context views, the textured mesh reconstructed by fusing virtual rendered depth maps via TSDF-Fusion [8], the Gaussian
normals for the first input view, and the ground-truth and rendered depth from a novel (virtual) viewpoint. Baselines exhibit inaccurate
rendered depth and normals; when fused, these inconsistencies lead to holes and deformed regions in the reconstructed meshes.
9

<!-- page 19 -->
and append an additional Gaussian-splat decoder that con-
sumes the multi-view features to predict per-pixel Gaussian
parameters. In this variant, the network is trained purely with
a view-synthesis loss on the rendered Gaussians. The Gaus-
sian decoder attached to VGGT mirrors the DUSt3R-based
variant: it predicts centers, scales, orientations, opacities,
and colors for 3DGS splats, and is trained jointly with the
backbone using our view-synthesis loss and geometric pri-
ors.
8. Implementation Details
Common training setup. All models are trained in a gen-
eralizable splatting regime on the RealEstate10K (RE10K)
training split [75]. We use two input images per sample
and render three virtual novel views to minimize the view-
synthesis loss described in the main paper. The same loss
weights are used across all architectures and ablations: the
alignment and orientation priors are weighted by λa = 0.1
and λo = 0.05.
For the edge-aware weights in Equation (10), we
set the robust scale to the q=0.95 quantile, i.e., η =
Quantile0.95({dj
t}).
We use fixed constants (w0, κ) =
(10, 4) and a small ϵ for numerical stability (we use ϵ = 10−8
in all experiments). For the cosine-space penalty in Equa-
tion (11), we use the Huber (SmoothL1) threshold δ = 0.1.
We apply the scale regularization Lflat only for the 3DGS
variant, with a fixed weight λflat = 1000. All variants are
evaluated on the same test splits and protocols described be-
low. We will release the source code and pretrained models
to facilitate reproducibility.
DUSt3R-based variant. For the DUSt3R-based backbone,
training is performed on a cluster of 24 NVIDIA A100
(40 GB) GPUs with a batch size of 6 per GPU (144 total),
while all evaluations are run on a single NVIDIA A6000
GPU. We train for 18,751 iterations on RE10K using the
setup above, with input images resized to 256×256. The
Gaussian decoders are optimized with a base learning rate of
2×10−4, and the DUSt3R layers are updated with a reduced
rate of 2×10−5. In line with observations from [68], we
found that training this architecture from scratch on RE10K
is unstable in the fully self-supervised setting; instead, we ini-
tialize the backbone with MASt3R-pretrained weights [35]
and fine-tune it jointly with our Gaussian decoders. Com-
peting DUSt3R-style baselines are allowed to use the same
supervised backbone initialization for fairness. Under this
setup, training the DUSt3R-based variant on RE10K takes
approximately 6 hours.
VGGT-based variant. For the VGGT-based backbone [56],
we adopt the same dataset, number of source views, and
number of rendered novel views as in the DUSt3R-based
setup. During training, we cap the longer image side at 448
pixels and randomly vary the aspect ratio between 0.5 and
1.0. The Gaussian decoder attached to VGGT is optimized
with a base learning rate of 2×10−4, while the VGGT back-
bone is updated with a lower rate of 2×10−5 to preserve its
pre-trained multi-view geometry priors. We initialize the
backbone from VGGT pretrained weight and train the added
Gaussian head end-to-end with our view-synthesis loss and
geometric priors. The corresponding wall-clock training
time for the VGGT-based variant is approximately 8 hours
on a single GPU with a batch size of 36.
Evaluation protocols. To evaluate zero-shot generalization
in pose estimation, geometry reconstruction, and novel-view
synthesis, we use the same test sets and splits for all architec-
tures and baselines. Specifically, we evaluate on the ACID
split from [68] and on the ScanNet test set [9], which com-
prises 2000 indoor RGB-D image pairs. For ScanNet, novel
views are obtained by uniformly sampling up to four interme-
diate viewpoints along the camera trajectory between each
pair of source views used for pose evaluation, resulting in
1592 novel-view samples (out of 2000). These novel views
are used consistently for both depth, mesh, and novel-view
synthesis evaluations.
Baselines and retraining protocol. For all baselines, we use
publicly released pretrained checkpoints whenever available.
When multiple versions exist, we select models trained on
RE10K [75] and at least at our input resolution of 256×256;
if a model is trained on RE10K plus additional data or at
higher resolution, we still use that checkpoint to give the
baseline a slight advantage. If no RE10K-pretrained model
is available (e.g., for FreeSplat [59]), we retrain the method
following the authors’ original training protocol and hyper-
parameters to ensure a fair comparison.
For NoPoSplat [68] specifically, we additionally retrain
their model under our setup (RE10K dataset, iteration count,
input resolution, and view-synthesis protocol). Minor devia-
tions from the originally reported numbers are summarized
in Table 11.
9. Rendered Normal for 2DGS
In this subsection, we revisit the rendered normal–depth
consistency loss introduced by 2DGS [26] and compare it
against our proposed orientation prior Lorient. While Lorient
is defined directly on Gaussian normals and can be applied
to both 2DGS and 3DGS parameterizations, the 2DGS loss
operates on rendered surface normals and thus can only be
instantiated and ablated for our 2DGS variant, where we
explicitly render per-pixel normals.
For a 3D Gaussian corresponding to pixel j in image t,
Gj
t =
 µj
t, αj
t, Σj
t, cj
t

,
3DGS [31] first projects the mean and covariance to the
image plane of a novel view. Let
Pf = Kf

Rf | Tf

∈R3×4
10

<!-- page 20 -->
Table 11. Comparison of our retrained NoPoSplat against the public checkpoint (NoPoSplat∗). (a) Pose evaluation (with test-time
photometric pose refinement) on RE10K [75], ScanNet [9] and ACID [40]. (b) Depth estimation for novel views (with pose refinement) on
ScanNet. (c) Novel-view synthesis on RE10K.
(a) Pose evaluation
RE10K
ScanNet-V1
ACID
Method
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
NoPoSplat∗
0.672
0.792
0.869
0.111
0.254
0.465
0.454
0.591
0.709
NoPoSplat
0.672
0.791
0.868
0.109
0.256
0.463
0.456
0.593
0.705
(b) Depth estimation
Rendered Depth (Novel Views)
Method
Abs Rel↓
δ1 < 1.10 ↑
δ1 < 1.25 ↑
NoPoSplat∗
0.127
0.564
0.859
NoPoSplat
0.126
0.567
0.861
Our baseline (No Prior 3DGS)
0.102
0.715
0.901
Our baseline (No Prior 2DGS)
0.114
0.692
0.884
(c) Novel-view synthesis
Small
Medium
Large
Average
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
NoPoSplat∗
21.086
0.721
0.237
23.134
0.776
0.185
25.086
0.818
0.141
23.189
0.775
0.185
NoPoSplat
21.097
0.723
0.237
23.191
0.779
0.187
25.107
0.817
0.144
23.244
0.778
0.187
be the projection matrix of view f. Dropping (t, j, f) for
clarity, the homogeneous image of the mean is
¯µ = P [µ⊤1]⊤,
µ′ =
¯µx/¯µz
¯µy/¯µz

.
Denoting by
J = ∂(µ′ ¯µz)
∂µ
the Jacobian of the local affine approximation of the per-
spective map, the (unnormalized) screen–space covariance
is
Σ′ = J P Σ P⊤J⊤,
and we keep only its upper–left 2 × 2 block, Σ′
uv =
(Σ′)1:2,1:2. The projected 2D Gaussian footprint is then
G′(u, v) = exp
h
−1
2
 (u, v)⊤−µ′⊤Σ′−1
uv
 (u, v)⊤−µ′i
.
For a pixel (u, v) in view f, novel-view RGB is rendered
via front-to-back α-blending of K depth-sorted Gaussians:
ˆIf(u, v) =
K
X
k=1
ck wk(u, v),
(15)
wk(u, v) = Tk(u, v) αk G′
k(u, v),
(16)
with transmittance
Tk(u, v) =
Y
j<k
 1 −αj G′
j(u, v)

.
The same weights wk(u, v) can be reused to render depth
and surface normals for that view.
Rendered normal–depth consistency loss from 2DGS.
2DGS [26] proposes a rendered normal–depth consistency
loss that enforces agreement between rendered surface nor-
mals and normals estimated from the rendered depth map.
Let x = (u, v)⊤and let Dr(x) denote the rendered depth
obtained by combining per-Gaussian depths dk(x) with the
same weights as color:
Dr(x) =
X
k
wk(x) dk(x).
Let nk be the (unit) normal associated with Gaussian Gk.
The rendered normal is then
Nr(x) =

X
k
wk(x) nk

∗
,
where ∥· ∥∗denotes vector normalization. A correspond-
ing normal c
N(x) can be estimated from Dr(x) via finite
differences and normalization. The rendered normal–depth
consistency loss introduced in [26] penalizes the angular
discrepancy between these two normals:
LRNC = 1
|Ω|
X
x∈Ω
ω(x)

1 −

Nr(x), c
N(x)

,
(17)
where Ωis the set of valid pixels and
ω(x) =
X
k
wk(x)
acts as an opacity-based confidence weight.
11

<!-- page 21 -->
Comparison to our orientation prior.
In our work, LRNC
is not part of the final model; we implement it in the 2DGS
variant purely as a baseline to compare against our proposed
orientation prior Lorient, which operates directly on Gaus-
sian normals and is defined consistently for both 2DGS and
3DGS parameterizations. As discussed in the main paper,
LRNC acts after rasterization and thus enforces coherence be-
tween rendered depth and normals, whereas Lorient provides
direct supervision on the predicted Gaussian orientations,
independent of the rasterizer.
In practice, naively applying LRNC in our generalizable
splatting setup often causes the optimization of Gaussian
means and orientations to converge to a near-planar local
minimum (see Figure 12). Detaching the rendered depth
from the computation graph alleviates this by treating the
depth-derived normals as pseudo labels for Gaussian orienta-
tions, but this configuration still requires the alignment loss
Lalign to be effective. Table 12 compares models trained with
Lalign + LRNC against our full model using Lalign + Lorient,
showing that our orientation prior is more stable and yields
better geometry. Unless otherwise stated, all 2DGS and
3DGS results reported in the main paper and supplementary
use only Lalign and Lorient; LRNC is used solely as an ablation
baseline in Table 12.
10. Depth Rendering for Gaussian Splatting
To evaluate geometry (depth and meshes), we must con-
vert the Gaussian representation into a per-pixel depth map.
Along each camera ray, multiple Gaussians may contribute,
so there is no uniquely defined “depth.” In this work we
considered two choices, both consistent with standard alpha
compositing.
Recall that for a pixel (u, v) in view f, the rendered
color is obtained by alpha-blending depth-sorted projected
Gaussians G′
k as
ˆIf(u, v) =
K
X
k=1
ck wk(u, v),
(18)
wk(u, v) = Tk(u, v) αk G′
k(u, v),
(19)
Tk(u, v) =
Y
i<k
 1 −αi G′
i(u, v)

,
(20)
where ck and αk denote the color and opacity of Gaussian
Gk, and G′
k(u, v) is its 2D footprint in the image plane of If.
For depth rendering, we reuse the same weights wk(u, v)
but replace the color ck by the scalar depth dk(u, v) of Gaus-
sian Gk in the camera coordinate system. For brevity, we
write x = (u, v) and index the Gaussians along the ray by i.
Accumulated depth.
A common practice is to treat depth
as an additional “channel” and apply the same alpha-
blending rule as for RGB. This yields an accumulated depth
Dacc(x) =
X
i
wi(x) di(x),
(21)
where wi(x) is defined as in (18). This definition aggregates
contributions from all Gaussians along the ray and implicitly
couples depth with the overall opacity.
Expected depth.
Alternatively, we can interpret the
weights wi(x) as defining a discrete distribution along the
ray and compute the expected depth:
Dexp(x) =
P
i wi(x) di(x)
P
i wi(x)
,
(22)
Compared to Dacc, this normalizes out the accumulated opac-
ity and is less sensitive to residual transmittance or brightness
variations.
Choice of baseline depth renderer.
We implemented both
Dacc and Dexp and compared them quantitatively for our
model. In contrast to existing generalizable Gaussian splat-
ting works that effectively rely on accumulated depth, we
found that the expected depth Dexp consistently yields lower
depth errors and more stable TSDF fusion [8], resulting
in higher-quality mesh reconstructions. In particular, Ta-
ble 11 compares a re-trained NoPoSplat checkpoint (under
our protocol) with its public checkpoint, and reports depth
evaluation for our “no prior” baselines with both 2DGS and
3DGS representations.
Consequently, we adopt Dexp in (22) as our default depth
rendering and use it for all “no prior” baselines. All proposed
priors (alignment and orientation) are built on top of this
expected-depth renderer and ablated accordingly.
11. Test-time Pose Refinement
For two-view reconstruction, many different 3D configura-
tions can explain the same image pair, so even if a model
produces a self-consistent scene, its recovered camera poses
may not coincide exactly with the ground-truth poses in
the evaluation datasets. Following the spirit of prior pose-
free works, including [68], we therefore allow an optional
test-time pose refinement step for fair comparison against
pose-aware baselines.
Given an input pair, we first run the network once to
predict the Gaussian splats from the source views. During
evaluation, these Gaussian parameters are frozen, and we
optimize the given camera pose by minimizing the same
photometric objectives used at train time. Concretely, for a
given view f with ground-truth image If and rendered image
ˆIf(Rf, Tf), we solve a small gradient-based optimization
problem
min
Rf ,Tf Lsynthesis(If,ˆIf) + λaLalign + λoLorient,
12

<!-- page 22 -->
Table 12. Ablation of Gaussian orientation losses. We compare a model trained with Lalign + LRNC against our full model trained with
Lalign + Lorient. (a) Pose evaluation with test-time refinement (same loss as training) on RE10K [75], ScanNet [9], and ACID [40]. (b)
Novel-view depth estimation with pose refinement on ScanNet. (c) Novel-view synthesis on RE10K.
(a) Pose evaluation
RE10K
ScanNet-V1
ACID
Method
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
5◦↑
10◦↑
20◦↑
Ours (2DGS+Align+RNC)
0.681
0.799
0.870
0.137
0.313
0.521
0.476
0.609
0.720
Ours (2DGS+Align+Orient)
0.686
0.802
0.875
0.153
0.334
0.541
0.478
0.609
0.723
(b) Depth evaluation
Rendered Depth (Novel Views)
Method
Abs Rel↓
δ1 < 1.10 ↑
δ1 < 1.25 ↑
Ours (2DGS+Align+RNC)
0.099
0.714
0.910
Ours (2DGS+Align+Orient)
0.082
0.743
0.931
(c) Novel-view synthesis
Small
Medium
Large
Average
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Ours (2DGS+Align+RNC)
21.344
0.736
0.236
23.423
0.786
0.185
25.432
0.825
0.141
23.501
0.785
0.185
Ours (2DGS+Align+Orient)
21.377
0.739
0.234
23.426
0.787
0.184
25.459
0.827
0.141
23.504
0.787
0.184
LRNC and Lalign
LRNC Only
Figure 12. Failure modes of rendered normal–depth consistency compared to our orientation loss. We replace the proposed orientation
prior Lorient with the rendered normal–depth consistency loss LRNC [26] and visualize the learned scene during training on RE10K [75], by
projecting the reconstructed 3D Gaussians onto three axis-aligned planes. Using LRNC without the alignment loss Lalign yields severely
degenerate reconstructions. Adding Lalign recovers some structure but remains clearly inferior to the results obtained with Lalign + Lorient
reported in the main paper.
where only the pose parameters (Rf, Tf) are updated and
all Gaussian parameters remain fixed. For each model vari-
ant, we include exactly the same loss terms as used during
training (e.g., Lalign and/or Lorient), so that the pose refine-
ment is consistent with the learned priors.
We use this test-time pose refinement in two contexts:
(i) for pose evaluation, where the refined pose is compared
against ground truth using rotation and translation error met-
rics; and (ii) for depth evaluation, where we render depth
maps from the refined camera pose to disentangle errors due
to misaligned camera poses from errors in the underlying
scene structure. All relevant pose and depth evaluations re-
sults are reported both with and without this pose refinement
scheme, and whenever refinement is used it is explicitly
stated in the corresponding table or figure caption.
12. Mesh Reconstruction and Evaluation
Predicted mesh reconstruction. For each ScanNet [9] test
scene, we first predict a Gaussian-splatting representation
from two source views using our model (either DUSt3R- or
VGGT-based). The predicted Gaussians are then used to
render depth maps from a virtual camera trajectory interpo-
lated between the two source poses. Concretely, we sample
a fixed number of intermediate viewpoints (20 in our im-
plementation) by smoothly interpolating extrinsics between
the two input cameras, and use our expected-depth renderer
(Sec. Section 10) to obtain per-view rendered depth maps
along this path. These depth maps, together with the corre-
sponding camera intrinsics and extrinsics, are fused into a
volumetric TSDF using TSDF-Fusion [8]. We use a scene-
adaptive voxel size (proportional to the scene radius) and a
standard truncation distance (a small multiple of the voxel
13

<!-- page 23 -->
size), and extract a watertight surface mesh via Marching
Cubes. Finally, we keep the largest connected component
and remove small isolated clusters and degenerate faces,
yielding the predicted mesh for that scene.
Ground-truth meshes and visibility cropping. ScanNet
provides a metric-scale mesh for each scene. We rigidly
transform this mesh into the coordinate frame where the first
source camera is placed at the origin.
To avoid penalizing geometry that is never observed in the
source views, we crop the ground-truth mesh to the region
seen by the cameras. Specifically, we construct the union of
viewing frusta of the source frames and intersect the mesh
with this union. The resulting cropped mesh is used as the
ground-truth surface for evaluation.
Global Sim(3) alignment. Because our pose-free model is
trained without metric depth or absolute pose supervision,
the recovered scene geometry and cameras are only defined
up to a global Sim(3) transform (rotation, translation, and
uniform scale). Before computing metrics, we therefore
align each predicted mesh to its ground-truth counterpart
using a single global Sim(3) transformation. Concretely,
we sample points from the surfaces of both the predicted
mesh and the cropped ground-truth mesh, and run a point-to-
point ICP procedure with scaling to estimate the best-fitting
similarity transform between them. This transform is then
applied to the predicted mesh, and all reconstruction metrics
are computed in the resulting aligned, metric coordinate
frame.
Evaluation metrics. Let P = {pi}Np
i=1 and G = {gj}Ng
j=1
be point sets sampled uniformly from the aligned predicted
mesh and the cropped ground-truth mesh, respectively. For
each point pi ∈P, we compute the distance to its nearest
neighbor in G,
dpred→gt(pi) = min
g∈G ∥pi −g∥2,
and similarly for each gj ∈G we compute
dgt→pred(gj) = min
p∈P ∥gj −p∥2.
We report three standard reconstruction metrics [23, 42]:
• Accuracy (lower is better):
Acc = 1
Np
Np
X
i=1
dpred→gt(pi),
measuring how close the predicted surface lies to the near-
est ground-truth surface.
• Completeness (lower is better):
Comp = 1
Ng
Ng
X
j=1
dgt→pred(gj),
measuring how well the predicted mesh covers the ground-
truth surface.
• Chamfer distance (lower is better):
CD = 1
2
 Acc + Comp

,
the symmetric Chamfer distance between P and G.
Per-scene scores are averaged over all ScanNet test scenes
where both predicted and ground-truth meshes are non-
empty after reconstruction and cropping. These aggregated
metrics are reported in Table 5.
14

<!-- page 24 -->
References
[1] Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun Chin,
Chunhua Shen, and Ian Reid. Auto-rectify network for un-
supervised indoor depth estimation. IEEE transactions on
pattern analysis and machine intelligence, 44(12):9802–9813,
2021. 2, 4
[2] Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo match-
ing network. In CVPR, 2018. 3
[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In CVPR,
2024. 1, 2, 4, 6, 8, 5
[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. TVCG,
2024. 2
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In ECCV, 2024. 1, 2, 3, 4, 6, 8, 5
[6] Shin-Fang Chng, Ravi Garg, Hemanth Saratchandran, and
Simon Lucey. Invertible neural warp for nerf. In ECCV, 2024.
3
[7] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. Depth-
regularized optimization for 3d gaussian splatting in few-shot
images. In CVPR, 2024. 2, 3
[8] Brian Curless and Marc Levoy. A volumetric method for
building complex models from range images. In Proceedings
of the 23rd annual conference on Computer graphics and
interactive techniques, pages 303–312, 1996. 9, 5, 12, 13
[9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber,
Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-
annotated 3d reconstructions of indoor scenes. In CVPR,
2017. 2, 6, 7, 8, 9, 1, 4, 5, 10, 11, 13
[10] Yilun Du, Cameron Smith, Ayush Tewari, and Vincent Sitz-
mann. Learning to render novel views from wide-baseline
stereo pairs. In CVPR, 2023. 5
[11] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck,
and Michael Felsberg. Roma: Robust dense feature matching.
In CVPR, 2024. 2, 6, 7
[12] David Eigen and Rob Fergus. Predicting depth, surface nor-
mals and semantic labels with a common multi-scale convolu-
tional architecture. In Proceedings of the IEEE international
conference on computer vision, pages 2650–2658, 2015. 5
[13] David Eigen, Christian Puhrsch, and Rob Fergus. Depth
map prediction from a single image using a multi-scale deep
network. In NeurIPS, 2014. 3
[14] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Bat-
manghelich, and Dacheng Tao.
Deep ordinal regression
network for monocular depth estimation. In CVPR, 2018.
3
[15] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A
Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting.
In CVPR, 2024. 4
[16] Ravi Garg, Vijay Kumar Bg, Gustavo Carneiro, and Ian Reid.
Unsupervised cnn for single view depth estimation: Geometry
to the rescue. In ECCV, 2016. 4
[17] Ravi Garg, Shin-Fang Chng, and Simon Lucey. Direct align-
ment for robust nerf learning. In ACCV, 2024. 3
[18] Clément Godard, Oisin Mac Aodha, and Gabriel J Brostow.
Unsupervised monocular depth estimation with left-right con-
sistency. In CVPR, 2017. 4
[19] Clement Godard, Oisin Mac Aodha, Michael Firman, and
Gabriel J. Brostow. Digging into self-supervised monocular
depth estimation. In ICCV, 2019. 4
[20] Clément Godard, Oisin Mac Aodha, Michael Firman, and
Gabriel J Brostow. Digging into self-supervised monocular
depth estimation. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 3828–3838,
2019. 2, 4
[21] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong
Tan, and Ping Tan. Cascade cost volume for high-resolution
multi-view stereo and stereo matching. In CVPR, 2020. 3
[22] Antoine Guédon and Vincent Lepetit. Sugar: Surface-aligned
gaussian splatting for efficient 3d mesh reconstruction and
high-quality mesh rendering. CVPR, 2024. 3
[23] Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang,
Guofeng Zhang, Hujun Bao, and Xiaowei Zhou. Neural 3d
scene reconstruction with the manhattan-world assumption.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 5511–5520, 2022. 9, 14
[24] Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han.
Binocular-guided 3d gaussian splatting with view consistency
for sparse view synthesis. NeurIPS, 2024. 3
[25] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong
Yang, Seungryong Kim, and Chong Luo. Unifying corre-
spondence pose and nerf for generalized pose-free novel view
synthesis. In CVPR, 2024. 6, 7, 5
[26] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In ACM SIGGRAPH 2024, 2024. 2, 3, 4,
5, 10, 11, 13
[27] Jia-Bin Huang, Iain Matthews, and Wolf Kienzle. Deepmvs:
Learning multi-view stereopsis. In CVPR, 2018. 3
[28] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 1, 4, 9
[29] Seunghyun Kang, Hyunwoo Lee, and Hyeongju Chae. Self-
splat: Pose-free and 3d-prior-free generalizable 3d gaussian
splatting. CVPR, 2025. 1, 4, 7, 8, 5
[30] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat, track & map 3d gaussians
for dense rgb-d slam. In CVPR, 2024. 3
[31] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM TOG, 2023. 1, 2, 3, 10
[32] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM TOG, 2017. 4
15

<!-- page 25 -->
[33] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Fed-
erico Tombari, and Nassir Navab. Deeper depth prediction
with fully convolutional residual networks. In 3DV, 2016. 3
[34] Jin Han Lee, Youngbok Bae, and In So Kweon Han. Bts:
Depth estimation via local planar guidance. arXiv preprint
arXiv:1907.10326, 2019. 3
[35] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Ground-
ing image matching in 3d with mast3r.
arXiv preprint
arXiv:2406.09756, 2024. 1, 2, 3, 4, 6, 7, 10
[36] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normalization.
In CVPR, 2024. 3
[37] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Se-
mantic gaussian splatting for neural dense slam. In ECCV,
2024. 3
[38] Zhengqi Li and Noah Snavely. Megadepth: Learning single-
view depth prediction from internet photos. In CVPR, 2018.
6
[39] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In CVPR, 2024. 3
[40] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Maka-
dia, Noah Snavely, and Angjoo Kanazawa. Infinite nature:
Perpetual view generation of natural scenes from a single
image. In ICCV, 2021. 6, 7, 4, 5, 11, 13
[41] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J
Davison. Gaussian splatting slam. In CVPR, 2024. 3, 6
[42] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Se-
bastian Nowozin, and Andreas Geiger. Occupancy networks:
Learning 3d reconstruction in function space. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 4460–4470, 2019. 14
[43] Zhiyuan Min, Yawei Luo, Jianwen Sun, and Yi Yang.
Epipolar-free 3d gaussian splatting for generalizable novel
view synthesis. NeurIPS, 2024. 4
[44] OpenAI. Creating video from text, 2024. 1
[45] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In ICCV, 2021. 3
[46] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In CVPR, 2016. 6
[47] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob
Fergus. Indoor segmentation and support inference from rgbd
images. In Computer Vision–ECCV 2012: 12th European
Conference on Computer Vision, Florence, Italy, October 7-
13, 2012, Proceedings, Part V 12, pages 746–760. Springer,
2012. 2, 4
[48] Brandon Smart, Chuanxia Zheng, Iro Laina, and Vic-
tor Adrian Prisacariu.
Splatt3r:
Zero-shot gaussian
splatting from uncalibrated image pairs.
arXiv preprint
arXiv:2408.13912, 2024. 1, 4, 7, 8, 3, 5
[49] Libo Sun, Jia-Wang Bian, Huangying Zhan, Wei Yin, Ian
Reid, and Chunhua Shen. Sc-depthv3: Robust self-supervised
monocular depth estimation for dynamic scenes. IEEE trans-
actions on pattern analysis and machine intelligence, 46(1):
497–508, 2023. 2, 4
[50] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al. Scalability in perception
for autonomous driving: Waymo open dataset. In CVPR,
2020. 6
[51] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng,
Dylan Campbell, Joao F Henriques, Christian Rupprecht, and
Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d
scene reconstruction from a single image. In 2025 Interna-
tional Conference on 3D Vision (3DV), pages 670–681. IEEE,
2025. 1
[52] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang. Hisplat: Hierarchical 3d gaus-
sian splatting for generalizable sparse-view reconstruction.
ICLR, 2025. 4
[53] Benjamin Ummenhofer, Hao Zhou, Jonas Uhrig, Nikolaus
Mayer, Eddy Ilg, Alexey Dosovitskiy, and Thomas Brox.
Demon: Depth and motion network for learning monocular
stereo. In CVPR, 2017. 3
[54] Ziyu Wan, Hao Gao, Rui Xiong, and Fang Du. S2gaussian:
Sparse-view super-resolution 3d gaussian splatting. In CVPR,
2025. 3
[55] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo
Speciale, and Marc Pollefeys. Patchmatchnet: Learned multi-
view patchmatch stereo. In CVPR, 2021. 3
[56] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt:
Visual geometry grounded transformer. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025. 1, 2, 3, 8, 9, 5, 10
[57] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A.
Efros, and Angjoo Kanazawa. Continuous 3d perception
model with persistent state. In CVPR, 2025. 3
[58] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud.
Dust3r: Geometric 3d
vision made easy. In CVPR, 2024. 1, 2, 3, 6, 7, 8, 9, 4, 5
[59] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee. Freesplat: Generalizable 3d gaussian splatting towards
free view synthesis of indoor scenes. NeurIPS, 2024. 1, 4, 8,
5, 10
[60] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In CVPR, 2024. 3
[61] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay,
Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-
time 360° sparse view synthesis using gaussian splatting. In
3DV, 2025. 3
[62] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi,
Fisher Yu, Dacheng Tao, and Andreas Geiger.
Unifying
flow, stereo and depth estimation. IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 45(11):13941–13958,
2023. 1, 2, 4
[63] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum,
Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depth-
splat: Connecting gaussian splatting and depth. In CVPR,
2025. 1, 4, 8, 2, 5
16

<!-- page 26 -->
[64] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything
v2. arXiv:2406.09414, 2024. 2
[65] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction. In CVPR,
2024. 3
[66] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time
photorealistic dynamic scene representation and rendering
with 4d gaussian splatting. In ICLR, 2024. 3
[67] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan.
Mvsnet: Depth inference for unstructured multi-view stereo.
In ECCV, 2018. 3
[68] Botao Ye, Sifei Liu, Haofei Xu, Li Xueting, Marc Pollefeys,
Ming-Hsuan Yang, and Peng Songyou. No pose, no problem:
Surprisingly simple 3d gaussian splats from sparse unposed
images. In ICLR, 2025. 1, 2, 3, 4, 6, 7, 8, 5, 10, 12
[69] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and
Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor
scenes. In ICCV, 2023. 6
[70] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images. In
CVPR, 2021. 5
[71] Huangying Zhan, Ravi Garg, Chamara Saroj Weerasekera,
Kejie Li, Harsh Agarwal, and Ian Reid. Unsupervised learning
of monocular depth estimation and visual odometry with deep
feature reconstruction. In CVPR, 2018. 4
[72] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu,
Shengping Zhang, Liqiang Nie, and Yebin Liu. Gps-gaussian:
Generalizable pixel-wise 3d gaussian splatting for real-time
human novel view synthesis. In CVPR, 2024. 1, 4
[73] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3d:
A modern library for 3d data processing. arXiv preprint
arXiv:1801.09847, 2018. 2
[74] Tinghui Zhou, Matthew Brown, Noah Snavely, and David G
Lowe. Unsupervised learning of depth and ego-motion from
video. In CVPR, 2017. 4
[75] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magnification: learning view syn-
thesis using multiplane images. ACM TOG, 2018. 1, 6, 7, 3,
4, 5, 8, 10, 11, 13
[76] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun
Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys.
Nice-slam: Neural implicit scalable encoding for slam. In
CVPR, 2022. 4, 9
[77] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang.
Fsgs: Real-time few-shot view synthesis using gaussian splat-
ting. In ECCV, 2024. 2, 3
[78] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-
slam: Neural implicit scene encoding for rgb slam. In 3DV,
2024. 4
17
