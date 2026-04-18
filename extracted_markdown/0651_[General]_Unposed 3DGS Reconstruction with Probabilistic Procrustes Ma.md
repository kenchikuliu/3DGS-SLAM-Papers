<!-- page 1 -->
Unposed 3DGS Reconstruction with Probabilistic
Procrustes Mapping
Chong Cheng∗
Zijian Wang∗
Sicheng Yu
Yu Hu
Nanjie Yao
Hao Wang†
The Hong Kong University of Science and Technology (Guangzhou)
{ccheng735, zwang886, yhu847}@connect.hkust-gz.edu.cn
yusch@mail2.sysu.edu.cn
nanjiey@uci.edu
haowang@hkust-gz.edu.cn
Abstract
3D Gaussian Splatting (3DGS) has emerged as a core technique for 3D representa-
tion. Its effectiveness largely depends on precise camera poses and accurate point
cloud initialization, which are often derived from pretrained Multi-View Stereo
(MVS) models. However, in unposed reconstruction task from hundreds of outdoor
images, existing MVS models may struggle with memory limits and lose accuracy
as the number of input images grows. To address this limitation, we propose a
novel unposed 3DGS reconstruction framework that integrates pretrained MVS
priors with the probabilistic Procrustes mapping strategy. The method partitions
input images into subsets, maps submaps into a global space, and jointly opti-
mizes geometry and poses with 3DGS. Technically, we formulate the mapping of
tens of millions of point clouds as a probabilistic Procrustes problem and solve
a closed-form alignment. By employing probabilistic coupling along with a soft
dustbin mechanism to reject uncertain correspondences, our method globally aligns
point clouds and poses within minutes across hundreds of images. Moreover, we
propose a joint optimization framework for 3DGS and camera poses. It constructs
Gaussians from confidence-aware anchor points and integrates 3DGS differentiable
rendering with an analytical Jacobian to jointly refine scene and poses, enabling
accurate reconstruction and pose estimation. Experiments on Waymo and KITTI
datasets show that our method achieves accurate reconstruction from unposed
image sequences, setting a new state of the art for unposed 3DGS reconstruction.
1
Introduction
3D Gaussian Splatting (3DGS) has emerged as a revolutionary technique for 3D representation and
novel view synthesis, owing to its superior rendering quality and real-time performance [Kerbl et al.,
2023, Lu et al., 2024]. By optimizing a set of 3D Gaussian parameters to represent the scene, 3DGS
achieves high-fidelity and efficient visual effects, and has quickly become a focal point of research.
However, applying 3DGS to real-world scenarios, particularly for reconstruction from hundreds
of uncalibrated outdoor images, remains highly challenging. Traditional 3DGS pipelines heavily
rely on accurate precomputed camera poses and an initial point cloud [Schonberger and Frahm,
2016, Schönberger et al., 2016]. These prerequisites are often difficult to obtain in complex outdoor
environments, which significantly limits the broader applicability of 3DGS.
Several studies [Fu et al., 2024, Jiang et al., 2024, Dong et al., 2025, Shi et al., 2025] attempt
to jointly optimize camera poses and Gaussian parameters from images in an end-to-end manner,
thereby enabling unposed 3DGS reconstruction. However, they struggle on outdoor scenes due to
∗Equal contribution.
†Corresponding author.
Preprint. Under review.
arXiv:2507.18541v1  [cs.CV]  24 Jul 2025

<!-- page 2 -->
scale ambiguity, sparse supervision, and sensitivity to noisy initialization, often resulting in limited
accuracy [Fan et al., 2024]. Another common strategy combines Structure-from-Motion (SfM) with
3DGS [Schonberger and Frahm, 2016], but the SfM phase is typically computationally expensive,
often requiring hours of processing and being prone to failure in challenging outdoor conditions.
Pretrained Multi-View Stereo (MVS) models [Wang et al., 2024b, Leroy et al., 2024] have long served
as a structured approach for inferring dense point clouds and camera poses directly from images,
and remain a promising foundation for unposed 3D reconstruction. In contrast, feed-forward 3DGS
methods predict Gaussians directly from images with improved efficiency, but typically support only
a dozen views and are prone to out-of-memory (OOM) issues Ye et al. [2024], Xu et al. [2024], Chen
et al. [2024b], Zhang et al. [2025]. While modern MVS models can handle larger input batches, they
still face accuracy degradation and memory bottlenecks as the number of views increases, especially
in outdoor scenes [Wang et al., 2025a, Yang et al., 2025].
These challenges inspire a divide-and-conquer strategy: decomposing a large image collection
into smaller subsets, processing them individually, and merging them into a globally consistent
reconstruction. However, since each submap is inferred in its own local frame, the results often suffer
from scale ambiguity and geometric inconsistency. Existing registration methods [Yang et al., 2020,
Besl and McKay, 1992, Lawrence et al., 2019, Chen et al., 2024a] typically fail under scale ambiguity,
geometric deviations, and the computational challenge of aligning tens of millions of points. A key
challenge, therefore, is how to efficiently align these submaps into a unified coordinate system to
enable high-quality 3DGS reconstruction.
To address these challenges, we propose a collaborative framework for unposed 3DGS reconstruction
that integrates pretrained MVS with a divide-and-conquer strategy. Using feed-forward priors and
overlapping views across image groups, we progressively recover globally consistent point clouds
and camera poses from local submaps, leading to high-quality 3DGS reconstruction.
Specifically, we reformulate the original alignment of tens of millions of points as a probabilistic
Procrustes problem by designing overlapping-frame correspondences at the pixel level. We first
obtain a closed-form Sim(3) solution using the Kabsch-Umeyama algorithm, and then refine it via a
probabilistic coupling with a soft dustbin mechanism that rejects uncertain matches. This approach
effectively resolves scale ambiguity and local geometric discrepancies between submaps, achieving
robust global alignment within minutes
Further, we propose a joint optimization framework for 3DGS and camera poses, where Gaussians are
initialized from downsampled anchor points obtained via confidence-aware correspondence filtering.
Camera poses are optimized through differentiable 3DGS rendering, with gradients propagated via
an analytical quaternion Jacobian, leading to improved pose accuracy and view synthesis quality.
Our main contributions are as follows:
1. We propose an alignment method that casts submap mapping as a probabilistic Procrustes
problem. It combines closed-form Sim(3) estimation with probabilistic and outlier rejection,
enabling global pose and point-cloud recovery from hundreds of images within minutes.
2. We propose a 3DGS and pose joint optimization module that constructs Gaussians from
confidence-guided anchor points and refines scene and poses via 3DGS differentiable
rendering with an analytical Jacobian, improving pose accuracy and reconstruction quality.
3. Experiments on the Waymo and KITTI datasets demonstrate that our method achieves highly
efficient and accurate global reconstruction from unposed images, setting a new state of the
art for unposed 3DGS reconstruction.
2
Related Work
2.1
Unposed 3D Gaussian splatting
Traditional 3D Gaussian Splatting (3DGS) [Kerbl et al., 2023] relies on accurate camera poses
and sparse point clouds typically provided by COLMAP [Schonberger and Frahm, 2016]. Due to
COLMAP’s high computational cost and limited robustness in challenging conditions, recent works
aim to recover camera parameters and reconstruct Gaussian scenes directly from multi-view images.
2

<!-- page 3 -->
CF-3DGS [Fu et al., 2024] initializes the Gaussian field using monocular depth and progressively
refines both camera parameters and Gaussians to support unposed reconstruction. COGS [Jiang et al.,
2024] incrementally reconstructs the scene by registering cameras through 2D correspondences, while
Rob-GS [Dong et al., 2025] introduces a robust pairwise pose estimation strategy. NoParameters
[Shi et al., 2025] jointly optimizes intrinsics, extrinsics, and Gaussians, removing the need for prior
camera calibration. InstantSplat [Fan et al., 2024] leverages the pre-trained pointmap model DUSt3R
[Wang et al., 2024b] for initialization and accelerates optimization via parallel grid partitioning, but
remains limited to sparse-view scenarios with relatively few images.
Another line of work Smart et al. [2024], Ye et al. [2024], Charatan et al. [2024] leverages pre-training
to enable feed-forward networks that directly predict high-quality Gaussian scenes from paired
images. Recent extensions Xu et al. [2024], Chen et al. [2024b], Zhang et al. [2025] support more
inputs and improve quality, but typically scale only to a dozen views. As scene size and view count
grow, these methods face significant memory and runtime demands or degraded robustness.
To enable scalable unposed 3DGS on outdoor scenes with hundreds of images, we introduce pretrained
MVS models and adopt a divide-and-conquer strategy to 3DGS reconstruction.
2.2
Multi-view 3D Reconstruction
Traditional multi-view reconstruction pipelines [Hartley and Zisserman, 2003] consist of handcrafted
stages including feature matching, triangulation, and bundle adjustment. Systems like COLMAP
[Schonberger and Frahm, 2016, Mur-Artal and Tardós, 2017, Schönberger et al., 2016] perform
well in static scenes, but suffer from accumulated errors, high computational cost, and failure in
challenging scenarios.
Learning-based approaches [Yao et al., 2018, 2019, Zhang et al., 2023, Ma et al., 2022] leverage end-
to-end networks to recover high-quality geometry from calibrated images. More recently, end-to-end
differentiable SfM frameworks [Wei et al., 2020, Wang et al., 2021, 2024a, Smith et al., 2025] aim to
jointly estimate camera parameters and scene structure directly from image collections.
DUSt3R [Wang et al., 2024b] and MASt3R [Leroy et al., 2024] regress dense point clouds and camera
parameters from paired images, replacing handcrafted components with pre-trained backbones. This
feedforward paradigm has been extended to multi-image settings using memory encoders [Wang
and Agapito, 2024, Wang et al., 2025b, Cabon et al., 2025] and subgraph fusion networks [Liu et al.,
2024]. VGGT [Wang et al., 2025a] and Fast3R [Yang et al., 2025] further adopt global attention
mechanisms to reason across multiple views. MV-DUSt3R+ [Tang et al., 2024] and FLARE [Zhang
et al., 2025] enable end-to-end 3D Gaussian Splatting reconstruction from sparse-view inputs, and
similar strategies have been applied to dynamic scene modeling [Zhang et al., 2024, Chen et al.,
2025]. However, these methods struggle with increasing view counts, facing memory bottlenecks
and degraded reconstruction robustness. Inconsistent in structure and scale across independently
processed submaps complicates global alignment, limiting fidelity in large-scale outdoor scenes.
To address these challenges, we adopt a divide-and-conquer strategy that partitions images into local
submaps and reconstructs a globally consistent 3DGS scene through alignment and joint optimization.
3
Method
We aim to reconstruct high-quality 3D Gaussian scenes from hundreds of unposed outdoor images. As
illustrated in Fig. 1, the image set is partitioned into overlapping subsets, each independently processed
by a pretrained MVS model to estimate local point clouds and camera poses. These submaps are then
globally aligned via probabilistic Procrustes mapping, followed by joint optimization of the 3DGS
and camera poses, resulting in high-fidelity and globally consistent reconstructions.
3.1
Problem Formulation
Given K images {Ik}K
k=1 split into G fixed-size subsets{Sg}G
g=1, each subset is fed to a pretrained
MVS network to produce a local submap Mg = (Pg, {T (g)
i
}i∈Sg), containing a dense point cloud
and its camera poses. Our goal is to fuse these into a globally consistent scene (Pglobal, {Ti}K
i=1).
3

<!-- page 4 -->
Feed-Forward
MVS Model
Submap
Image Subset 
with Overlap
...
...
...
Differentiable
Rendering 
3DGS Scene
Probabilistic Procrustes Mapping
3DGS-Pose Joint Optimization
Correspondence
𝜃∗= (s, R, t) ∈ Sim(3)
Mapping
Iterative 
Refinement
Closed form 
Solution
Anchor 
Sampling
Confidence 
Pruning
Point Cloud
{𝑇!}!"#
$ ⊂ 𝑆𝑒(3)
Camera Poses
𝑃%&'()&
Optimization
Similarity 
Transformation
GT
Analytical Jacobian
Gradient flow
Update 𝜃
Overlapping 
Frame
Update 𝛾
Initialization
Probabilistic 𝛾!
Weights 𝑤"#  = $
%
Figure 1: We begin by partitioning the unposed image sequence into multiple subsets, and apply a
pretrained MVS model to infer local point clouds and camera poses. Overlapping-frame correspon-
dences are constructed to reformulate large-scale submap alignment as a probabilistic Procrustes
problem. This is solved via a closed-form Sim(3) estimator, followed by probabilistic refinement
and soft outlier rejection. The final 3DGS and poses are jointly optimized through anchor-based
initialization and differentiable rendering, with gradients propagated via analytical Jacobians.
To achieve globally consistent alignment across submaps, we aim to estimate the optimal similarity
transformation θ = (s, R, t) ∈Sim(3) between submaps, where s ∈R+ is the scale factor,
R ∈SO(3) is the rotation matrix, and t ∈R3 is the translation vector. However, feed-forward MVS
submaps suffer from scale ambiguity and geometric distortions. The structural complexity of outdoor
scenes further leads to the failure of standard registration methods. Moreover, each submap typically
contains tens of millions of points, making global alignment a high-dimensional and computationally
intensive task that challenges both accuracy and efficiency.
To address these challenges, we define k overlapping frames between each pair of adjacent subsets,
denoted as Oab. This allows us to reformulate multi-submap alignment as a classical Procrustes
problem. We then extract per-pixel 3D correspondences between submaps a and b within Oab:
Cab =

(pa
i , qb
j) | π
 T a
i pa
i

= π
 T b
j qb
j
	
,
(1)
where π : R3 →R2 is the standard camera projection model. We then solve the classical Procrustes
problem as an optimization that minimizes the distances between transformed point pairs:
θ∗=
arg min
s>0,R∈SO(3),t∈R3
X
(i,j)∈Cab
s R pa
i + t −qb
j
2.
(2)
3.2
Probabilistic Procrustes Mapping
3.2.1
Procrustes Closed-form Solution
To efficiently estimate the optimal similarity transformation in Eq. (2), we adopt the Kabsch-Umeyama
algorithm [Umeyama, 1991, Lawrence et al., 2019] to compute a closed-form solution based on the
correspondence set Cab. First, we compute the centroids of each point set:
¯p = 1
N
X
(i,j)∈Cab p a
i ,
¯q = 1
N
X
(i,j)∈Cab q b
j ,
(3)
where N = |Cab| denotes the number of point pairs. These centroids reflect the global offsets of
the two point clouds and will be used to compute the translation vector. Next, we construct the
cross-covariance matrix Σ between the two point clouds to capture their spatial correlation structure:
Σ = 1
N
X
(i,j)∈Cab
(p a
i −¯p)(q b
j −¯q)⊤.
(4)
By performing singular value decomposition (SVD) Σ = UΛV ⊤, we obtain the principal directions
of the two point sets. This allows us to compute the closed-form solution of θ∗:
4

<!-- page 5 -->
R0 = U diag
 1, 1, det(UV ⊤)

V ⊤,
s0 = tr(Λ)
tr(Σp),
t0 = ¯q −s0R0¯p,
(5)
where R0 is a proper rotation ensuring a right-handed coordinate system, s0 is given by the ratio
between the singular values and the variance of the source point cloud, and t0 aligns centroids. This
closed-form step provides an efficient initialization for refining submap alignment.
Although the closed-form solution is theoretically optimal, it relies on two critical assumptions:
1. The spatial distributions of the two point sets must be identical.
2. The correspondences must be noise-free, i.e.,
s0 R0 pa
i + t0 −qb
j
 = 0, ∀(i, j) ∈Cab.
These conditions hold in ideal cases where point clouds are perfectly accurate and geometrically
consistent. However, in practical monocular reconstruction, even with overlapping frames providing
pixel-level correspondences, the spatial distributions of the 3D points may vary significantly. This
violates the assumptions of the closed-form solution and often leads to systematic bias when used
directly, making the mapping results less robust.
3.2.2
Probabilistic Mapping
We observe that point clouds predicted by feed-forward MVS models exhibit structural bias due
to learned priors, which leads to systematic errors in closed-form alignment. To address this, we
formulate point cloud registration as a probabilistic Procrustes problem augmented with a dustbin
mechanism. Specifically, we associate each candidate correspondence (pl, ql) between submaps
with a probabilistic matching weight γl ∈[0, 1].
To handle outliers, we introduce a probability-based dustbin mechanism: a control parameter η ∈
[0, 1] specifies the maximum allowable fraction of correspondences that can be excluded from
alignment. To implement this, we augment the target set with a virtual dustbin point qdustbin, and
assign it a fixed marginal weight bdustbin = δ.
Our objective is to jointly optimize the similarity transformation θ = (s, R, t) ∈Sim(3) and the
correspondence probabilities {γl}N
l=1. Each weight γl encodes the soft association strength between
a source point pl and its target ql under the current transformation. The objective is formulated as:
min
s,R,t,γ
X
l
γl∥sRpl + t −ql∥2 + ϵ
X
l
γl ln γl,
subject to
X
l
γl = 1,
(6)
where γl ∈[0, 1] denotes the soft matching probability between source point pl and target point ql.
Probability Weights Update.
We initialize the transformation parameters θ(0) = (s(0), R(0), t(0))
using the closed-form Kabsch–Umeyama method. Given a fixed transformation, the correspondence
weights γ are updated via entropy-regularized optimization:
γl ∝exp

−∥sRpl + t −ql∥2
ϵ

,
(7)
where the proportionality is followed by a normalization step to satisfy the marginal constraints using
the step-wise iteration optimization.
Transformation Update.
Fixing γ, we compute the gradients of the objective Lθ with respect to
the transformation parameters θ = (s, R, t). Let p′
l = Rpl, Then:
∇tLθ = 2
N
X
l=1
γ(k)
l
(sp′
l + t −ql) .
(8)
∇sLθ = 2
N
X
l=1
γ(k)
l
(sp′
l + t −ql)⊤p′
l.
(9)
5

<!-- page 6 -->
To update the rotation R, we parameterize it using a unit quaternion q = (w, v⊤)⊤, where v =
(x, y, z)⊤, and use the chain rule to compute:
∇qLθ = 2
N
X
l=1
γ(k)
l
(sR(q)pl + t −ql)⊤s∂(R(q)pl)
∂q
.
(10)
The transformation parameters are updated using gradient descent:
θ(k+1) = θ(k) −ηθ∇θLθ(θ(k)),
(11)
where ηθ is the learning rate. The optimization terminates when either the pose converges or
a maximum number of iterations is reached. In practice, the accurate closed-form initialization
typically leads to convergence within a few iterations.
The resulting optimal transformation θ⋆= (sg, Rg, tg) is then applied to all 3D points and camera
poses in submap Sg, transforming them into the global coordinate frame and updating the correspond-
ing poses. Iteratively applying this procedure across all submaps yields a globally consistent point
cloud and a unified camera trajectory.
3.3
3DGS and Pose Joint Optimization
After submap alignment, we obtain initial camera poses and a dense point cloud in a unified global
coordinate system. However, due to the inherent scale uncertainty, depth noise, and residual pose
drift in monocular reconstruction, further refinement is necessary. To this end, we jointly optimize 3D
Gaussian parameters and camera poses using a differentiable 3DGS rendering pipeline, improving
both pose accuracy and reconstruction quality.
3D Gaussian Splatting.
We model the scene as a set of 3D Gaussians:
G
=
{Gi
:
(µi, Σi, ci, Λi)|i = 1, ..., N}. Each Gaussian point is defined by position µi, 3D covariance matrix
Σi ∈R3×3, opacity Λi and color ci obtained by spherical harmonics.
For a specific view, given camera pose T = (R, t) and camera intrinsic K ∈R3×3, we can the render
RGB image ˆI via rasterization pipeline. First, project our 3D Gaussians to 2D image plane:
µ′ = π(T · µ),
Σ′ = JWΣW ⊤J⊤,
(12)
where π is the projection operation, W is the rotational component of T, and J is the Jacobian of the
affine approximation of the projective transformation. Then, the color of pixel can be formulated as
the alpha-blending of N ordered points that overlap the pixel:
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(13)
where αi is the density given by evaluating a 2D Gaussian with covariance Σ′.
Joint Optimization.
We first extract a high-confidence subset from the global point cloud and apply
downsampling to obtain the initial anchor set A = {xi}, which is used to initialize 3D Gaussian
Gi = (µi, Σi, ci, Λi). These Gaussians G = {Gi} together form the initial global scene.
Based on this, we define a closed-loop optimization framework that jointly optimizes the camera
poses T = (R, t) and Gaussian parameters {µi, Σi, ci, Λi}N
i=1 through the following objective:
Ltotal = α ∥ˆIk −Ik∥1 + (1 −α)SSIM(ˆIk, Ik),
(14)
where ˆIk denotes the rendered image under the current view k, Ik is the ground truth, and α is the
weight balancing the L1 and SSIM terms.
From the 3DGS rendering pipeline, it follows that the gradient of the camera pose T depends on two
intermediate quantities: Σ′, and the projected coordinates µ′
i of each Gaussian Gi. By applying the
chain rule, we can derive a fully analytic expression for ∂L
∂T , thereby avoiding the runtime overhead
6

<!-- page 7 -->
DUSt3R*
Ours
MASt3R*
InstantSplat
COLMAP*
CF-3DGS
GT
Figure 2: Qualitative Comparison on Waymo (top three rows) and KITTI (bottom three rows).
Methods marked with an asterisk (*) are reconstructed using 3DGS. InstantSplat is trained on only
80 images due to memory constraints. Our method achieves high-fidelity image reconstruction with
clearer textures and finer details.
of automatic differentiation and ensuring numerical stability during quaternion normalization. The
resulting analytic gradient takes the following form:
∂L
∂T = ∂L
∂ˆIk
∂ˆIk
∂T = ∂L
∂ˆIk
∂ˆIk
∂αi
∂αi
∂Σ′
∂Σ′
∂T + ∂αi
∂µ′
∂µ′
∂T

,
(15)
∂Σ′
∂T = ∂Σ′
∂J
∂J
∂µc
∂µc
∂T + ∂Σ′
∂W
∂W
∂T ,
(16)
∂µ′
∂T = ∂µ′
∂µc
∂µc
∂T ,
(17)
where µc denotes the point µ in world coordinates transformed into the camera frame by the pose T.
We parameterize the camera pose T = (R, t) by a unit quaternion q ∈R4 and a translation vector
t ∈R3, and we provide the analytic gradients with respect to q and t in Appendix 5.
To keep q unit-length, we apply the projected gradient updates:
q ←
q −η∇qL
∥q −η∇qL∥.
(18)
By jointly optimizing the camera poses, 3D Gaussian parameters, and image reprojection, we obtain
a globally consistent 3D Gaussian scene with accurate pose estimation and high-fidelity rendering.
4
Experiments
4.1
Experimental Setup
Implementation details.
Our experiments are implemented using the PyTorch framework and conducted on a single NVIDIA
RTX A6000 GPU with an AMD EPYC 7542 CPU. All results are reported using the best-performing
pretrained MVS model, VGGT [Wang et al., 2025a].
We set the group size to 60 and the inter-group overlap to K = 1, which empirically yielded the best
performance. Camera poses are optimized with an initial learning rate of 10−5, decayed to 10−7 until
7

<!-- page 8 -->
Table 1: Quantitative results on Waymo and KITTI datasets. Tm denotes matching time and Tt
denotes training time. Methods marked with an asterisk (*) indicate methods reconstructed using
3DGS. Flare fails due to out-of-memory (OOM). ATE measures pose accuracy; PSNR, SSIM, and
LPIPS evaluate image reconstruction quality. Best results are in bold. Our method achieves the best
accuracy and reconstruction fidelity.
Method
Waymo [Sun et al., 2020]
KITTI Geiger et al. [2013]
ATE↓PSNR↑SSIM↑LPIPS↓
Tm
Tt
GPU
ATE↓PSNR↑SSIM↑LPIPS↓
Tm
Tt
GPU
COLMAP+SPSG*
3.68
30.17
0.893
0.314
45min 58min 13GB
12.1
19.52
0.647
0.438
41 min 35 min 10GB
CF-3DGS
5.46
22.69
0.736
0.316
-
67min 12GB
5.99
15.91
0.490
0.486
-
195min 11GB
DUSt3R*
5.59
29.39
0.871
0.310
38min 42min 46GB
3.10
23.87
0.767
0.311
32min
51min
45GB
MASt3R*
6.12
27.49
0.867
0.308
82min 46min 40GB
5.71
23.63
0.778
0.274
94min
57min
37GB
Fast3R*
43.9
20.66
0.764
0.468
30s
46min 14GB
47.3
16.73
0.533
0.581
30s
28min
10GB
Flare
-
-
-
-
-
-
-
-
-
-
-
-
-
-
InstantSplat
2.55
19.22
0.739
0.515
58min
8min
41GB
2.23
13.09
0.414
0.680
97min
7min
40GB
Ours
1.41
31.53
0.915
0.245
1min
63min 14GB
1.64
24.83
0.780
0.272
8min
31min
12GB
MASt3R
Fast3R
Ours+ICP
Ours
Figure 3: Qualitative comparison of reconstructed point clouds. The bottom row shows the estimated
camera trajectories. Fast3R exhibits significant drift, while Ours+ICP still suffers from misalignment.
Our method achieves accurate submap fusion and globally consistent pose estimation.
convergence. The dustbin capacity is set to 20%. To enable efficient optimization, we first prune the
lowest-confidence 3% of points and then apply voxel-based downsampling to retain 0.05% of points
as anchors. Additional implementation details are provided in the supplementary material.
Dataset. We conduct evaluations on two outdoor datasets, selecting 9 scene groups from Waymo Sun
et al. [2020] and 8 from KITTI Geiger et al. [2013], each consisting of 200 front-view images captured
under diverse conditions. All images are used for evaluation. We assess both the image reconstruction
quality and the accuracy of the estimated camera poses across entire sequences.
Metrics. We evaluate camera pose estimation and scene reconstruction (in terms of image rendering
quality). For pose, we report translation error via Absolute Trajectory Error (ATE), measured in
meters (m). For reconstruction, we use PSNR, SSIM, and LPIPS. We also log training time and peak
memory to assess efficiency and scalability.
Baselines. We compare our method with seven baselines, including COLMAP+SPSG [Schonberger
and Frahm, 2016], CF-3DGS [Fu et al., 2024], DUSt3R [Wang et al., 2024b], MASt3R [Leroy et al.,
2024], Fast3R [Yang et al., 2025], Flare [Zhang et al., 2025], and InstantSplat [Fan et al., 2024].
Since COLMAP+SPSG, DUSt3R, MASt3R, and Fast3R only estimate point clouds and camera poses
from images without directly producing Gaussians, we use the original 3DGS [Kerbl et al., 2023]
training pipeline for scene reconstruction, indicated with an asterisk (*).
4.2
Analysis of Experimental Results.
We evaluate our method on the Waymo and KITTI datasets, with results summarized in Tab. 1.
InstantSplat, designed for sparse-view settings, fails to scale to large inputs due to memory limitations
and performs poorly even when restricted to 80 views. Fast3R is efficient but suffers from severely
8

<!-- page 9 -->
DUSt3R
Ours
MASt3R
InstantSplat
Fast3R
CF-3DGS
Figure 4: Qualitative comparison of camera pose estimation. Red denotes estimated poses while
gray denotes ground truth. Our method achieves superior pose accuracy compared to other methods.
inaccurate pose estimation. Flare supports only a limited number of input view sizes. COLMAP
underperforms due to divergence in several scenes, where large errors skew the average ATE.
In contrast, our method combines the pretrained VGGT model, the PPM mapping module, and
joint pose optimization to achieve superior reconstruction quality and trajectory accuracy. Fig. 2
shows rendering results comparison in 6 scenes, including road layouts, building structures, vehicles,
and vegetation. Fig. 3 highlights the consistency of point clouds at submap boundaries. Compared
to Fast3R, Mast3R, and ICP-based registration, our approach achieves seamless alignment across
groups, with minimal drift in overlapping regions. The corresponding trajectory plots confirm the
effectiveness of our pose refinement. Additionally, our method produces globally consistent and
accurate point clouds and camera poses within just a few minutes.
Fig. 4 further demonstrates that our estimated trajectories are significantly more accurate and stable
than those of competing methods. These results demonstrate the effectiveness of our framework for
unposed reconstruction from hundreds of outdoor images.
4.3
Ablation Study
Table 2: Ablation results on the Waymo dataset.
Top: comparison of different submap alignment
strategies based on our method. “Ours + ICP” and
“Ours + COLMAP” denote using relative poses
from ICP or COLMAP for submap mapping. Bot-
tom: ablations of our Probabilistic Procrustes Map-
ping (PPM) and 3DGS pose joint optimization
(Joint Opt.) modules.
Method
ATE
PSNR
SSIM
LPIPS
Ours w/ ICP
3.24
27.63
0.852
0.341
Ours w/ COLMAP
3.77
28.13
0.835
0.336
w/o PPM
5.97
27.54
0.824
0.363
w/o Joint Opt.
0.68
30.47
0.903
0.242
Ours
0.56
32.72
0.935
0.211
We conduct ablation experiments on the Waymo
dataset to evaluate the effectiveness of the pro-
posed probabilistic Procrustes mapping (PPM)
and the joint 3DGS optimization module. As
shown in Table 2, we first compare different
alignment strategies. Using ICP or COLMAP-
predicted relative poses for submap registration
leads to notable pose errors and visible misalign-
ments in the final reconstructions. In contrast,
our probabilistic Procrustes mapping module
achieves significantly higher registration accu-
racy and reconstruction fidelity, demonstrating
the advantage of combining closed-form align-
ment with probabilistic refinement.
We further ablate each core module to assess its
individual impact. Disabling the PPM module
and replacing it with VGGT relative pose estimation results in degraded global consistency and
lower-quality novel view synthesis. Similarly, fixing camera poses during the 3DGS training stage
leads to performance drops in both image quality and geometric coherence. These results confirm
that jointly optimizing camera poses and scene representation is essential for accurate and robust
reconstruction. Overall, both the PPM mapping module and joint pose optimization play critical roles
in ensuring accurate global alignment and high-quality 3DGS reconstruction.
4.4
Limitations
Our approach relies on the quality of the pretrained MVS predictions for initial poses and geometry.
While the joint optimization stage can correct moderate errors, severe inaccuracies in initialization
may degrade the final reconstruction quality. As the number of input frames increases, accumulated
drift and higher optimization costs can limit scalability to large-scale or long sequences. Moreover,
in highly dynamic scenes with frequent motion or occlusions, the lack of consistent correspondences
across views may hinder stable optimization and reduce reconstruction fidelity.
9

<!-- page 10 -->
5
Conclusion
We presented a scalable and robust framework for unposed 3D Gaussian Splatting reconstruction.
By integrating pretrained MVS models with a divide-and-conquer strategy, our method efficiently
handles outdoor scenes with hundreds of uncalibrated views. We introduce a Probabilistic Procrustes
Mapping module for global registration, followed by a 3DGS and poses joint optimization module for
jointly refining camera poses and 3D Gaussians. Our method achieves state-of-the-art performance
and offers practical value for unposed 3D reconstruction in real-world scenarios.
10

<!-- page 11 -->
References
Paul J Besl and Neil D McKay. Method for registration of 3-d shapes. In Sensor fusion IV: control
paradigms and data structures, volume 1611, pages 586–606. Spie, 1992.
Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud,
and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. arXiv preprint
arXiv:2503.01661, 2025.
David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian
splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 19457–19467, 2024.
Suyi Chen, Hao Xu, Haipeng Li, Kunming Luo, Guanghui Liu, Chi-Wing Fu, Ping Tan, and
Shuaicheng Liu. Pointreggpt: Boosting 3d point cloud registration using generative point-cloud
pairs for training, 2024a. URL https://arxiv.org/abs/2407.14054.
Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating
disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391, 2025.
Zequn Chen, Jiezhi Yang, and Heng Yang. Pref3r: Pose-free feed-forward 3d gaussian splatting from
variable-length image sequence. arXiv preprint arXiv:2411.16877, 2024b.
Zhen-Hui Dong, Sheng Ye, Yu-Hui Wen, Nannan Li, and Yong-Jin Liu. Towards better robustness:
Progressively joint pose-3dgs learning for arbitrarily long videos. arXiv preprint arXiv:2501.15096,
2025.
Zhiwen Fan, Kairun Wen, Wenyan Cong, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris
Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Sparse-view sfm-free gaussian
splatting in seconds. arXiv preprint arXiv:2403.20309, 2024.
Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d
gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20796–20805, 2024.
Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti
dataset. The international journal of robotics research, 32(11):1231–1237, 2013.
Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge
university press, 2003.
Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xiaolong Wang, Hao Su, and Ravi Ra-
mamoorthi. A construct-optimize approach to sparse view synthesis without camera pose. In ACM
SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting
for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
Jim Lawrence, Javier Bernal, and Christoph Witzgall. A purely algebraic justification of the kabsch-
umeyama algorithm. Journal of Research of the National Institute of Standards and Technology,
124, October 2019. ISSN 2165-7254. doi: 10.6028/jres.124.028. URL http://dx.doi.org/10.
6028/jres.124.028.
Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r.
In European Conference on Computer Vision, pages 71–91. Springer, 2024.
Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan
Chen. Slam3r: Real-time dense scene reconstruction from monocular rgb videos. arXiv preprint
arXiv:2412.09401, 2024.
Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20654–20664, 2024.
11

<!-- page 12 -->
Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo with cascaded epipolar raft. In European
Conference on Computer Vision, pages 734–750. Springer, 2022.
Raul Mur-Artal and Juan D Tardós. Orb-slam2: An open-source slam system for monocular, stereo,
and rgb-d cameras. IEEE transactions on robotics, 33(5):1255–1262, 2017.
Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of
the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.
Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view
selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV),
2016.
Dongbo Shi, Shen Cao, Lubin Fan, Bojian Wu, Jinhui Guo, Renjie Chen, Ligang Liu, and Jieping Ye.
No parameters, no problem: 3d gaussian splatting without camera intrinsics and extrinsics. arXiv
e-prints, pages arXiv–2502, 2025.
Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot
gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024.
Cameron Smith, David Charatan, Ayush Tewari, and Vincent Sitzmann. Flowmap: High-quality
camera poses, intrinsics, and depth via gradient descent. In 3DV, 2025.
Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui,
James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam,
Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang,
Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous
driving: Waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), June 2020.
Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and
Zhicheng Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds.
arXiv preprint arXiv:2412.06974, 2024.
S. Umeyama. Least-squares estimation of transformation parameters between two point patterns.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991. doi:
10.1109/34.88573.
Hengyi Wang and Lourdes Agapito.
3d reconstruction with spatial memory.
arXiv preprint
arXiv:2408.16061, 2024.
Jianyuan Wang, Yiran Zhong, Yuchao Dai, Stan Birchfield, Kaihao Zhang, Nikolai Smolyanskiy, and
Hongdong Li. Deep two-view structure-from-motion revisited. In Proceedings of the IEEE/CVF
conference on Computer Vision and Pattern Recognition, pages 8953–8962, 2021.
Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry
grounded deep structure from motion. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 21686–21697, 2024a.
Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David
Novotny.
Vggt: Visual geometry grounded transformer.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2025a.
Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continu-
ous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025b.
Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r:
Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 20697–20709, 2024b.
Xingkui Wei, Yinda Zhang, Zhuwen Li, Yanwei Fu, and Xiangyang Xue. Deepsfm: Structure from
motion via deep bundle adjustment. In Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 230–247. Springer, 2020.
12

<!-- page 13 -->
Jiale Xu, Shenghua Gao, and Ying Shan. Freesplatter: Pose-free gaussian splatting for sparse-view
3d reconstruction. arXiv preprint arXiv:2412.09573, 2024.
Heng Yang, Jingnan Shi, and Luca Carlone. Teaser: Fast and certifiable point cloud registration,
2020. URL https://arxiv.org/abs/2001.07715.
Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai,
Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one
forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), June 2025.
Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured
multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages
767–783, 2018.
Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. Recurrent mvsnet for
high-resolution multi-view stereo depth inference. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 5525–5534, 2019.
Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng.
No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv
preprint arXiv:2410.24207, 2024.
Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing
Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence
of motion. arXiv preprint arXiv:2410.03825, 2024.
Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun
Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation
from uncalibrated sparse views. arXiv preprint arXiv:2502.12138, 2025.
Zhe Zhang, Rui Peng, Yuxi Hu, and Ronggang Wang. Geomvsnet: Learning multi-view stereo with
geometry perception. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 21508–21518, 2023.
13

<!-- page 14 -->
Appendix: Quaternion–Point Jacobian
In this appendix, we derive the Jacobian of the rotated point R(q) p with respect to the unit quaternion
q = [w, v⊤]⊤, where v = (x, y, z)⊤.
We start from the equivalent expression for the rotation:
R(q) p =
 w2 −∥v∥2
p + 2 v (v⊤p) + 2 w (v × p).
(19)
Define:
∆(q) = R(q) p = A + B + C,
(20)
with:
A = (w2 −v⊤v) p,
B = 2 v (v⊤p),
C = 2 w (v × p).
(21)
1. Derivative with respect to w
Only A and C depend on w. We have:
∂A
∂w = 2w p,
∂C
∂w = 2 (v × p).
(22)
Therefore:
∂(R(q) p)
∂w
= 2 w p + 2 (v × p).
(23)
2. Derivative with respect to v
Let [p]× denote the skew-symmetric matrix such that [p]×u = p × u. We compute:
∂A
∂v = −2 (v⊤p) I3 = −2 p v⊤,
(24)
∂B
∂v = 2 (v⊤p) I3 + 2 v p⊤,
(25)
∂C
∂v = 2 w [p]×.
(26)
Combining these terms yields:
∂(R(q) p)
∂v
= −2 p v⊤+ 2 (v⊤p) I3 + 2 v p⊤+ 2 w [p]×,
(27)
3. Assembling the 3 × 4 Jacobian
Stacking the partial derivatives with respect to w and v produces the full Jacobian:
∂(R(q) p)
∂q
=

2w p + 2 (v × p)
|
{z
}
3×1
 −2 p v⊤+ 2 (v⊤p) I3 + 2 v p⊤+ 2 w [p]×
|
{z
}
3×3

3×4
.
(28)
where the first column corresponds to ∂/∂w and the remaining three columns correspond to
∂/∂x, ∂/∂y, ∂/∂z. This Jacobian can be directly used in gradient-based optimization.
14

<!-- page 15 -->
Appendix: Analytic Gradients of µ′ and Σ′ w.r.t. Pose T
In this appendix, we derive the analytic gradients of the projected coordinate µ′ = π(µc) and the
projected covariance Σ′ = J R(q) ΣR(q)⊤J⊤with respect to the camera pose T = (R(q), t), where
µc = R(q) µ + t,
q = [qr, qi, qj, qk]⊤,
t ∈R3,
µ ∈R3 is a 3D point, and J = ∂π(µc)
∂µc
is the 2 × 3 projection Jacobian.
1. Gradients of the projected coordinate µ′
By chain rule, the derivative w.r.t. translation is
∂µ′
∂t = J ∂µc
∂t = J =


fx
zc
0
−fx xc
z2c
0
fy
zc
−fy yc
z2c

.
(29)
The derivative w.r.t. the quaternion is
∂µ′
∂q = J ∂µc
∂q = J
h
∂µc
∂qr
∂µc
∂qi
∂µc
∂qj
∂µc
∂qk
i
.
(30)
Here the 3 × 1 blocks ∂µc/∂qα are:
∂µc
∂qr
= 2
" 0
−qk
qj
qk
0
−qi
−qj
qi
0
#
µ,
(31)
∂µc
∂qi
= 2
" 0
qj
qk
qj
−2qi
−qr
qk
qr
−2qi
#
µ,
(32)
∂µc
∂qj
= 2
"−2qj
qi
qr
qi
0
qk
qr
qk
−2qj
#
µ,
(33)
∂µc
∂qk
= 2
"−2qk
−qr
qi
qr
−2qk
qj
qi
qj
0
#
µ.
(34)
2. Gradients of the projected covariance Σ′
Since translation does not affect covariance:
∂Σ′
∂t = 0.
(35)
For the quaternion:
∂Σ′
∂q = J ∂
 R Σw R⊤
∂q
J⊤= J
∂R
∂q Σw R⊤+ R Σw
∂R⊤
∂q

J⊤,
(36)
where ∂R/∂q is the classic gradient of the rotation matrix with respect to the quaternion, and ∂R⊤/∂q
is its transpose.
These closed-form derivatives enable efficient back-propagation of both µ′ and Σ′ through the
differentiable 3DGS rendering pipeline.
15

<!-- page 16 -->
NeurIPS Paper Checklist
1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract and introduction accurately summarize the contributions of the
paper, including the proposed method, key technical insights, and empirical improvements.
The claims are well-supported by theoretical analysis and experimental results.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The paper includes a dedicated Limitations section discussing assumptions,
potential failure cases, and generalizability issues.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
16

<!-- page 17 -->
Answer: [NA]
Justification: The paper does not contain formal theoretical results or proofs.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The paper provides complete details about the experimental setup, dataset
usage, model architecture, training procedures, and evaluation protocols. Sufficient informa-
tion is included to enable reproduction of all key results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code
17

<!-- page 18 -->
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: The code and data will be released upon acceptance, with complete instructions
for reproducing the main results.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: We report experimental details in our main paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: The paper reports PSNR, SSIM, ATE, LPIPS, which are commonly used as a
measure of performance in image processing experiments. This approach is standard in the
field and is sufficient to convey the performance of the methods under investigation.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
18

<!-- page 19 -->
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: We describe the computing environment used in our experiments, including
GPU types, memory size, number of training hours.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We do not foresee any ethical concerns related to data usage, environmental
impact, or fairness.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: The paper includes a Broader Impact section discussing potential societal
applications of our 3D scene reconstruction framework.
Guidelines:
19

<!-- page 20 -->
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper does not release models or data with high misuse risk
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: All third-party datasets and tools used in the paper are properly cited with
licenses stated where applicable.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
20

<!-- page 21 -->
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not introduce new datasets or models requiring documentation.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The research does not involve crowdsourcing or experiments with human
subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15. Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: No human subjects were involved in the research, so IRB approval is not
applicable.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
21

<!-- page 22 -->
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16. Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA]
Justification: LLMs were not used in the design or implementation of the core methods in
the paper. They were only used for minor editing support.
Guidelines:
• The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
22
