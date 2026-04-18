<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
JOGS: Joint Optimization of Pose Estimation and 3D Gaussian
Splatting
Xianben Yang∗, Yuxuan Li∗, Tao Wang, Yi Jin, Yidong Li and Haibin Ling
Abstract—Traditional novel view synthesis methods heavily
rely on external camera pose estimation tools such as COLMAP,
which often introduce computational bottlenecks and propagate
errors. To address these challenges, we propose a unified frame-
work that jointly optimizes 3D Gaussian points and camera poses
without requiring pre-calibrated inputs. Our approach iteratively
refines 3D Gaussian parameters and updates camera poses
through a novel co-optimization strategy, ensuring simultaneous
improvements in scene reconstruction fidelity and pose estimation
accuracy. The key innovation lies in decoupling the joint opti-
mization into two interleaved phases: first, updating 3D Gaussian
parameters via differentiable rendering with fixed poses, and
second, refining camera poses using a customized 3D optical
flow algorithm that incorporates geometric and photometric
constraints. This formulation progressively reduces projection
errors, particularly in challenging scenarios with large viewpoint
variations and sparse feature distributions, where traditional
methods struggle. Extensive evaluations on multiple datasets
demonstrate that our approach significantly outperforms existing
COLMAP-free techniques in reconstruction quality, and also
surpasses the standard COLMAP-based baseline in general.
Index Terms—3D Gaussian splatting, camera pose estimation,
gradient computation, novel view synthesis.
I. INTRODUCTION
R
ECENT advancements in the field of computer vision
have led to significant progress in 3D scene reconstruc-
tion and rendering. In particular, the introduction of 3D Gaus-
sian Splatting (3DGS) [1] technology has provided an efficient
and realistic technique for scene representation and rendering.
3DGS explicitly models the scene using a group of Gaussian
ellipsoids. This provides rapid and accurate rendering, clearly
exhibiting its benefits in real-time situations. Due to its explicit
representation and efficient rendering capabilities, 3DGS has
been widely applied in various fields [2]–[7], especially in sce-
narios requiring efficient processing and realistic rendering [8].
Accurate pose estimation is extremely important [9] for
most novel view synthesis methods including 3DGS. Most
existing 3DGS methods do not include a pose estimation com-
ponent, but rely on external inputs (e.g., COLMAP [10], [11]).
The separation of pose estimation and 3DGS optimization may
lead to suboptimal solutions. On the other hand, the reliance on
external input may limit its application to certain scenarios [9],
[12]. To solve these issues, recent research suggests many
∗Xianben Yang and Yuxuan Li contributed equally to this work.
Xianben Yang, Yuxuan Li, Tao Wang, Yi Jin and Yidong Li are with the Key
Laboratory of Big Data & Artificial Intelligence in Transportation (Beijing
Jiaotong University), Ministry of Education, China, and also with the School
of Computer Science and Technology, Beijing Jiaotong University, Beijing
100044, China.
Haibin Ling is with the Department of Artificial Intelligence, Westlake
University, Hangzhou 310030, China.
Corresponding author: Tao Wang(e-mail: twang@bjtu.edu.cn).
3DGS solutions that do not require inputs of camera poses. For
example, CFGS [9] uses a combined optimization of camera
parameters and Gaussian points. This method transforms the
camera pose registration problem into an image optimization
task between two consecutive frames. It achieves excellent
reconstruction results in continuous and dense image streams,
but its performance degradations when the constraints are
violated. ZeroGS [13] and InstantSplat [14] do not require
pre-supplied camera poses, but they need to load a pre-trained
model in advance for pose estimation, and usually work in
very sparse views. These methods either require the integration
of additional information or pre-trained models, or can only
operate with strong constraints on the input images, which
substantially limits their applicability.
In this paper, to address these challenges, we propose JOGS
that Jointly Optimizes the camera poses and the 3D Gaussian
Splatting representations without requiring pre-calibrated in-
puts. In addition to the losses used in the standard 3DGS
algorithm [1], our framework introduces a reprojection loss to
penalize the inconsistencies between different views. After an
initial coarse pipeline setup, the camera poses are subsequently
optimized in conjunction with the 3DGS parameters using
the alternating direction method (ADM) algorithm [15]. In
each iteration, the 3DGS parameters are updated following the
standard 3DGS algorithm. For the refinement of camera poses,
we propose a lucas-kanade 3D optical flow (LK3D) algorithm,
which leverages Gaussian points and image reprojection errors
by integrating image gradients with transformation-based pro-
jection error relationships. This alternating optimization strat-
egy significantly improves pose accuracy and achieves stable
convergence even under large camera viewpoint movements
or sparse feature distributions.
For validation, we compare the proposed method with sev-
eral state-of-the-art methods on three public datasets, including
Tanks and Temples [16], LLFF-NeRF [17] and Shiny [18].
The experimental results show that our method outperforms
the baselines in novel view synthesis, and achieves high
reconstruction quality in different scenarios. The source code
of our method will be released upon paper publication.
In summary, we make the following contributions:
(1) we propose a unified framework for joint optimization
of 3DGS parameters and camera poses, which does not rely
on external tools such as COLMAP;
(2) we propose an LK3D algorithm to optimize camera
poses based on the reprojection errors between 3D Gaussian
points and image pixels, which is independent of the sequential
relationship between images and is able to effectively fine-tune
the camera pose; and
(3) we validate the effectiveness of our method in different
arXiv:2510.26117v2  [cs.CV]  15 Jan 2026

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
datasets, which exhibits robust reconstruction quality across
all scenarios.
II. RELATED WORK
A. Novel View Synthesis
The task aims to generate photorealistic renderings of target
scenes from unknown viewpoints using a limited set of input
images. Recent advancements in neural rendering [19]–[22]
have significantly improved NVS in terms of reconstruction
quality and efficiency. The seminal work on Neural Radiance
Fields (NeRF) [19] introduced a paradigm shift in NVS by
representing scenes as continuous implicit neural radiance
fields, encoded via multilayer perceptrons (MLPs). Subsequent
studies extended NeRF along several directions. For example,
Mip-NeRF 360 [23] improve core rendering fidelity and scene
representation. Some works [24], [25] enhanced dynamic
scene modeling. Some methods [26], [27] optimized com-
putational efficiency to accelerate training. However, NeRF
continues to face challenges, including prolonged training
times, high hardware demands and limited editability. Re-
cently, the emergence of 3DGS [1] has achieved breakthroughs
by utilizing explicit differentiable representations, striking a
balance between rendering quality and efficiency. Extensive
research has been conducted on 3DGS, covering areas such
as scene rendering quality and realism [28], [29], 3DGS
acceleration [30], geometry reconstruction [31], [32], dynamic
scenes [33] and few-shot reconstruction [34], [35]. Neverthe-
less, most existing methods still rely on camera poses and
sparse point clouds precomputed by COLMAP [10], [11].
B. NVS without Pose Input
Eliminating the dependence of input pose has become a
main topic in recent research of NVS, for both NeRF and
3DGS methods. I-NeRF [36] introduced inverse rendering to
estimate camera poses through keypoint alignment using pre-
trained NeRF. BARF [37] proposed a coarse-to-fine coordinate
encoding strategy, with further improvement in GARF [38],
[39]. Nope-NeRF [12] trained NeRF by incorporating undis-
torted depth priors. For 3DGS-based methods, CFGS [9] is
the most closely related to our work. It builds the entire 3D
Gaussian in a continuous fashion, ”growing” some Gaussian
points with each new view added. It optimizes the camera
pose by minimizing the photometric loss between the ren-
dered image and the next frame image. While it achieves
3DGS scene representation without relying on COLMAP, its
optimization depends on the temporal relationship between
adjacent images, and the change of view angles between con-
secutive frames needs to be small. ZeroGS [13] relies on a pre-
trained DUSt3R-based [40] model called Spann3R [41]. In-
stantSplat [14] implements camera-free pose reconstruction in
sparse views. While GSHT [42] achieves quality enhancement
over CFGS, this improvement is constrained by the inherent
reliance of the method on the temporal ordering within image
sequences. In summary, current mainstream methods either
require the integration of additional information or pre-trained
models [13], [40], hence limited to working with only a small
number of images due to high computational resource [12],
[14], [37], or assume minimal camera motion [9], [12], [42].
To overcome these limitations, we design a new framework
that jointly optimizes 3D Gaussian and camera pose.
III. METHOD
A. Problem Definition
Let I = {I1, . . . , In} be a set of n images from different
viewpoints, G = {g1, . . . , gk} be 3D Gaussian points con-
sisting of k points, and P = {P1, . . . , Pn} denote the pose
information of the n images. Each Pi is represented by a
rotation matrix Ri and a shift vector si, which describe the
rotation and translation relative to the world coordinate system
(with P1 being the reference frame)
The objective of 3D reconstruction is to recover optimal 3D
structures G, as well as camera poses P, which minimizes the
differences between the training images and the projection of
the 3D Gaussian points onto the current image views as:
L = min
G,P fd(I, fr(G, P)),
(1)
where fr(·) and fd(·) are the render function and the distance
function respectively.
In traditional 3DGS methods, P is treated as known param-
eters, and the problem is reduced to:
L = min
G fd(I, fr(G)).
(2)
Specifically, the distance function fd is defined as the combi-
nation of the L1 loss and D-SSIM terms:
L = (1 −λ)L1 + λLD−SSIM.
(3)
Detailed definition of L1 and LD−SSIM can be found in [1].
In this paper, we treat both G and P as learnable parameters,
and optimize them jointly in the training step. To this end, we
introduce a 3D optical flow loss to penalize the difference
between the projections of two different views. The defini-
tion and the optimization method are described in detail in
Section III-D.
B. Joint Optimization Framework
Our joint optimization framework establishes a dual-phase
alternating minimization scheme to solve the coupled problem
in Section III-A. Let G(t) and P(t) = {R(t)
i , s(t)
i }n
i=1 denote
the 3D Gaussian parameters and camera poses at iteration t.
As shown in Algorithms 1, these two parts of parameters are
optimized by an alternating direction method, which contains
two phases as follows:
Phase 1: Gaussian Parameter Update. With fixed cam-
era poses P(t), we optimize G(t) using the standard 3DGS
pipeline. This involves minimizing the photometric reprojec-
tion error between rendered views and observed images:
G(t+1) = arg min
G fd(I, fr(G, P(t))),
(4)
where fr(·) denotes the differentiable rendering function of
3DGS. The optimization employs adaptive density control,
spherical harmonic coefficients and opacity modulation as the
original 3DGS formulation.

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
Optimized Poses
∆��
∆��
∆��
∆��
...
Input
...
Initialized Poses
Camera
 Pose ��
... 
Sparse Gaussian Points
Optimized Gaussian Points
Novel View
Pose 
Initialization
JOGS Pipeline
3
( ,
)
DGS
JOGS

G P
G
3
( ,
)
LK D
JOGS

G
G P
,
min
(
,
( ,
))
d
GT
r
f
f

G P
L
I
G P
G
P
∆��
∆��
∆��
∆��
∆�
∆�
Fig. 1. Method Overview. Our JOGS framework jointly optimizes Pose Estimation and 3D Gaussian Splatting. It starts with a simple SfM initialization, then
iteratively updates 3D Gaussian splatting parameters G and refines camera poses P, ensuring simultaneous improvements in scene reconstruction fidelity and
pose estimation accuracy. The updating of Gaussian points follows a standard 3DGS pipeline, while the refinement of camera poses is done by the proposed
LK3D algorithm.
Phase 2: Camera Pose Update. With frozen Gaussian
parameters G(t+1), we refine camera poses by solving:
P(t+1) = arg min
P fd(I, fr(G(t+1), P)).
(5)
Specifically, the incremental pose adjustment is computed
using the LK3D algorithm described in Algorithms 2, of which
the detailed explanation is described in Section III-D.
The two phases alternate at a fixed number of iterations.
The differentiable nature of 3DGS rendering enables gradient
flow through both phases. This alternating scheme progres-
sively reduces the joint loss to convergence, with each phase
benefiting from increasingly accurate estimates of the other.
C. Initialize Camera Poses and Gaussian Points
Our initialization pipeline adopts a Structure-from-Motion
(SfM) strategy similar in spirit to standard frameworks [10],
but is independently implemented in a lightweight and mod-
ular manner tailored for downstream joint optimization. We
extract Scale-Invariant Feature Transform (SIFT) [43] de-
scriptors and perform multi-view matching using Random
Sample Consensus (RANSAC) [44] to estimate fundamental
matrices. An initial camera pair with the highest number of
correspondences is selected, and its relative pose is recovered
via essential matrix decomposition. The 3D structure is then
progressively expanded using Perspective-n-Point (PnP) [45]
pose estimation and multi-view triangulation. All camera poses
and 3D points are jointly refined through global Bundle
Adjustment (BA) [46] with robust cost functions to minimize
reprojection errors.
Unlike the standard 3DGS, our method reconstructs both
the initial sparse Gaussian points and camera poses entirely
from scratch, without relying on external tools or pose priors.
This design ensures compatibility with our joint optimization
pipeline and provides greater control over reconstruction qual-
ity, sparsity, and initialization behavior.
D. Camera Pose Refinement
We propose a method of optimizing camera pose based
on 3D Gaussian points and image reprojection error. As
illustrated in Algorithms 1, our method interleaves camera
pose refinement with 3DGS training during the initial m
iterations. Specifically, we freeze the Gaussian parameters
when performing pose optimization at regular intervals, while
updating the 3DGS parameters in the remaining iterations.
This alternating strategy ensures stable gradient propagation
for subsequent scene reconstruction. During the training pro-
cess, the camera pose was optimized with the 3DGS model,
and the camera rotation matrix and shift vector were calculated
using the projection of a 3D Gaussian points from multiple
viewpoints.
The objective of this method is to reduce the photometric
discrepancy between the source image and the reprojected
appearance of 3D Gaussian points by adjusting the camera
pose. Specifically, given an initial estimate of the camera pose
P = [R | s], where R(θ) is a rotation matrix parameterized

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Algorithm 1 Joint Optimization Framework
% I: input images
% G: Gaussian points
% P: camera poses
% TG: max iterations
% k: pose refinement interval
% m: last iteration for pose refinement
1: Initialize (G, P)
2: for t ←1 to TG do
3:
if t mod k = 0 and t ≤m then
4:
P ←LK3D(G, P)
5:
else
6:
G ←3DGS(G, P)
7:
end if
8: end for
9: return G, P
Algorithm 2 Pose Optimization via LK3D
% I: input images
% G = {g1, . . . , gk}: Gaussian points
% P = {P1, . . . , Pn}: camera poses
% TL: max iterations
1: for all camera poses P ∈P do
2:
for t ←1 to TL do
3:
for all Gaussians g ∈G do
4:
lg ←c(g) −I(W(x(g); P))
5:
dg ←∇I ∂W
∂P
6:
end for
7:
H ←P
g(dg)⊤dg
8:
∆P ←H−1 P
g(dg)⊤lg
9:
P ←P + ∆P
10:
end for
11: end for
12: return optimized poses P
by Euler angles θ = (θx, θy, θz), and s is a shift vector, the
goal of the optimization is to refine θ and s such that the
projection W(x(g); P) of each Gaussian point x(g) onto the
image plane better aligns with its corresponding appearance in
the source image. This alignment is achieved by minimizing
the pixel-wise color difference , thereby enabling accurate and
robust camera pose estimation.
Lucas-Kanade 3D Optical Flow Algorithm. Let p = [R|s] ∈
P denote the camera pose of a certain target image. Given a
Gaussian point g ∈G, we denote c(g) the color value of g
and x(g) = (xg, yg, zg)⊤the 3D position coordinates of g in
the world coordinate system. We then define a transformation
function W(x(g); p) that maps the 3D Gaussian coordinates
x(g) from the world coordinate system to the target image
plane following the standard projective geometry.
The goal of optimization is to minimize the discrepancy
between the transformed image and the target image, which
is defined as follows:
p∗= arg min
p
X
g∈G
 c(g) −I(W(x(g); p))
2.
(6)
By minimizing the differences in pixels between the source
and transformed images, the optimal pose parameters p∗can
be determined.
It is difficult to directly compute the optimal camera pose
P, since no close-form solution is available. Our method uses
a gradient-based update approach by extending the standard
LK algorithm [47] to 3-dimensional space, which iteratively
revises the transformation matrix p with an increment ∆p as:
∆p∗= arg min
∆p
X
g∈G
 I(W(x(g); p + ∆p)) −c(g)
2.
(7)
For computation efficiency, we further approximate this using
a first-order Taylor expansion:
∆p∗≈arg min
∆P
X
g∈G
 I(W(x(g); p)) + ∇I ∂W
∂p ∆p −c(g)
2,
(8)
where ∇I represents the image gradient and
∂W
∂p
is the
Jacobian matrix of the transformation function W with respect
to the transformation parameters p. According to the principle
that the derivative at the extreme value is zero, the pose
increment ∆p is computed via Gauss-Newton approximation:
∆p ≈H−1 X
g∈G

∇I ∂W
∂p
⊤ c(g) −I(W(x(g); p))

,
(9)
where the Hessian H is computed as:
H =
X
g∈G

∇I ∂W
∂p
⊤
∇I ∂W
∂p

.
(10)
This method significantly reduces the error between the
source and target photos, resulting in accurate camera poses.
Note that the refinement of each camera pose P ∈P can be
performed independently following the same pipeline, which
facilitates the parallel implementation of the LK3D algorithm.
E. Optimization Strategy of Rotation Matrix.
In the pose optimization based on reprojection error, the
rotation part of the camera transformation matrices must
strictly remain as valid rotation matrices with orthonormal
columns and determinant equal to one. Direct gradient updates
on rotation matrices may violate their orthogonality, leading
to numerical instability. Here we adopt an Euler angle pa-
rameterization strategy that decomposes rotation matrices into
independent Euler angles (pitch α, yaw β, roll γ) around x,
y and z-axes, and utilizes their analytic derivatives for stable
iteration while preserving orthogonality.
Specifically, the rotation matrix R is parameterized as:
R = Rz(γ)Ry(β)Rx(α),
(11)
where the axial rotation matrices Rx(α), Ry(β), and Rz(γ)
inherently satisfy orthogonality. During optimization, the Ja-

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
TABLE I
QUANTITATIVE COMPARISON ON TANKS AND TEMPLES. THE BEST RESULTS ARE HIGHLIGHTED IN BOLD, AND THE SECOND IN UNDERLINE, AND THE
SAME STYLES ARE ADOPTED IN THE SUBSEQUENT TABLES.
Scene
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS CFGS [9] GSHT [42] Ours 3DGS CFGS GSHT Ours 3DGS CFGS GSHT Ours
Ballroom 32.13
16.83
16.56
30.96
0.86
0.45
0.45
0.94
0.12
0.40
0.25
0.05
Barn
28.30
17.28
21.16
26.88
0.92
0.51
0.63
0.88
0.09
0.42
0.27
0.12
Church
29.01
20.51
20.55
25.88
0.92
0.64
0.75
0.86
0.09
0.33
0.15
0.16
Family
25.67
14.37
29.02
25.44
0.90
0.45
0.91
0.88
0.12
0.47
0.09
0.15
Francis
24.97
20.45
28.89
27.92
0.83
0.62
0.84
0.87
0.23
0.37
0.20
0.19
Horse
20.22
17.49
27.94
26.53
0.80
0.61
0.90
0.90
0.22
0.35
0.09
0.11
Ignatius
26.87
17.16
20.95
25.13
0.85
0.37
0.61
0.81
0.12
0.41
0.21
0.15
Museum 27.25
16.36
12.44
26.54
0.88
0.52
0.30
0.87
0.09
0.47
0.59
0.10
Mean
26.80
17.55
22.57
26.91
0.87
0.52
0.67
0.88
0.13
0.40
0.23
0.13
TABLE II
QUANTITATIVE COMPARISON ON LLFF-NERF. FOR THE FORTRESS AND LEAVES SCENES (MARKED WITH *), WE DIRECTLY CITE THE RESULTS OF
CFGS AND GSHT FROM ZEROGS [13], BECAUSE OUR EXPERIMENTAL ENVIRONMENT COULD NOT MEET THE RUNNING REQUIREMENTS OF THEIR
CODES.
Scene
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS CFGS GSHT
Ours
3DGS CFGS GSHT Ours 3DGS CFGS GSHT Ours
Fern
23.55
16.65
18.09
22.93
0.80
0.50
0.56
0.77
0.23
0.46
0.44
0.22
Flower
25.56
21.16
19.20
27.50
0.82
0.67
0.67
0.85
0.24
0.41
0.46
0.20
Fortress∗
29.50
14.73
16.26
29.13
0.87
0.40
0.48
0.86
0.18
0.46
0.46
0.19
Horns
26.98
16.13
17.62
26.71
0.88
0.49
0.56
0.87
0.19
0.52
0.54
0.20
Leaves∗
17.91
15.38
15.69
18.25
0.59
0.42
0.42
0.60
0.21
0.40
0.33
0.27
Orchids
19.45
13.65
13.73
19.07
0.65
0.29
0.29
0.64
0.25
0.55
0.56
0.25
Room
31.85
19.25
19.76
32.14
0.95
0.77
0.80
0.95
0.13
0.36
0.35
0.13
Trex
26.00
18.16
18.30
27.41
0.90
0.61
0.64
0.91
0.20
0.44
0.47
0.18
Mean
25.10
16.89
17.33
25.39
0.81
0.52
0.55
0.80
0.20
0.45
0.45
0.21
cobians of the rotation matrix with respect to Euler angles are
computed via chain rule:
∂R
∂α = Rz(γ)Ry(β)∂Rx(α)
∂α
,
(12)
∂R
∂β = Rz(γ)∂Ry(β)
∂β
Rx(α),
(13)
∂R
∂γ = ∂Rz(γ)
∂γ
Ry(β)Rx(α).
(14)
This parameterization decouples the rotation matrix de-
grees of freedom into unconstrained Euler angle increments
∆α, ∆β, ∆γ. Using gradient descent with learning rate η, the
angles are updated as:
α ←α + η∆α,
β ←β + η∆β,
γ ←γ + η∆γ.
(15)
The strategy offers two main advantages for minimizing
the reprojection error. First, the local linearization of Euler
angle updates preserves the orthogonality of R ∈SO(3),
preventing manifold deviations that can arise from direct
matrix optimization. Second, compared to global matrix pa-
rameterization, the angle-based decomposition significantly
reduces the complexity of Jacobian computations, enhancing
both optimization efficiency and numerical stability.
IV. EXPERIMENTS
A. Datasets
We conducted extensive experiments on various datasets,
including LLFF-NeRF [17], Tanks and Temples [16] and
Shiny [18]. LLFF-NeRF: This dataset contains real-world
multi-view images captured by various devices, comprising
eight scenes. The number of images varies across scenes, with
the scene fern having the fewest (twenty images) and horns
the most (sixty-two images). Tanks and Temples: This dataset
comprises eight scenes, encompassing both indoor and outdoor
environments. In line with the configurations of CFGS [9] and
GSHT [42], we further enhanced the complexity of the dataset.
Given the limited variation in camera poses in the original
dataset, we uniformly sampled one-fifth of the images from
each scene to amplify the pose variation between consecutive
frames. Shiny: The dataset consists of a number of challenging
scenes with significant reflected or refracted lighting changes.
Since the data volume per scene varies across datasets, we
adopted proportional data splitting rather than using fixed
quantities. For each scene, seven-eighths of the data were
allocated for training, with the remaining one-eighth reserved
for testing.

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
LLFF-Fern
LLFF-Flower
LLFF-Trex
Shiny-Lab
Tanks-Horse
(a) GT
(b) 3DGS
(c) CFGS
(d) GSHT
(e) Ours
Fig. 2.
Qualitative results of several representative samples from LLFF-NeRF, Tanks and Temples, Shiny. Our method achieves consistently high
rendering quality across all scenes.
TABLE III
QUANTITATIVE COMPARISON ON SHINY. THE ORIGINAL DATASET CONTAINS EIGHT SCENES, BUT BOTH CFGS AND GSHT SUFFER FROM RUNNING
ERRORS AND FAIL TO REPORT THE FINAL RESULTS IN SOME SCENES. THUS WE ONLY REPORT THE EXPERIMENTAL RESULTS FOR FOUR SCENES.
Scene
PSNR ↑
SSIM ↑
LPIPS ↓
3DGS CFGS GSHT
Ours
3DGS CFGS GSHT Ours 3DGS CFGS GSHT Ours
Cd
28.29
26.60
26.44
28.18
0.94
0.87
0.90
0.94
0.12
0.17
0.16
0.12
Giants
21.69
14.37
16.01
20.52
0.72
0.45
0.36
0.68
0.28
0.47
0.62
0.25
Lab
28.54
26.26
27.99
29.28
0.93
0.82
0.91
0.94
0.15
0.18
0.15
0.15
Tools
24.59
12.44
11.67
24.33
0.84
0.50
0.47
0.83
0.35
0.59
0.60
0.32
Mean
25.77
18.51
19.05
25.58
0.86
0.58
0.59
0.85
0.23
0.39
0.43
0.21
B. Evaluation Metrics
We employed the same evaluation metrics as CFGS [9]
and GSHT [42]. For novel view synthesis, we used standard
metrics: peak signal-to-noise ratio (PSNR), structural similar-
ity index measure (SSIM) [48] and learned perceptual image
patch similarity (LPIPS) [49]. For pose evaluation, we treated
COLMAP-estimated poses as ground truth and measured ab-
solute trajectory error (ATE), which includes relative rotation
error (RPEr) and relative translation error (RPEt), along with
relative pose error (RPE). ATE quantifies the discrepancy
between estimated camera positions and ground truth, while
RPE measures relative pose errors between image pairs.
C. Implementation Details
We initialized camera poses and sparse Gaussian points
using only scene images and camera intrinsics. During 3D
Gaussian reconstruction, we alternately optimized 3D Gaus-
sian points and camera poses. Global pose optimization was
performed every 100 iterations, limited to the first 15,000
iterations. This restricted optimization strategy prevents error
accumulation, as pose estimation errors could degrade recon-
struction quality, which might further corrupt pose estimation
accuracy. Thus, optimizing poses only during the initial quarter
of training iterations is empirically justified. All experiments
were conducted on a single RTX 3090 GPU. Unless otherwise
stated, our experiments follow the same 3DGS parameter
settings.
D. Comparing with Baseline
Our experimental framework is built upon the original
3DGS architecture [1]. While the proposed modules are
theoretically compatible with advanced 3DGS variants, our

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
(a) GT
(b) 3DGS
(c) CFGS
(d) GSHT
(e) Ours
(f) GT
(g) 3DGS
(h) CFGS
(i) GSHT
(j) Ours
Fig. 3. Comparison of fine-grained scene details. Our method preserves sharper textures in novel-view synthesis, benefiting from camera pose optimization
during training.
current implementation specifically adheres to the canonical
formulation due to two methodological considerations: (1)
make sure that the comparison with the original 3DGS can be
made directly so that future generations can easily reproduce
our work; and (2) isolating the performance impact of our
contributions from other confounding factors. To maintain
consistency, all architectural parameters strictly follow the
original 3DGS configuration. This design choice facilitates
direct comparability with COLMAP-based 3DGS baselines
under identical experimental protocols.
For the COLMAP-free methods, NeRF-based approaches
exhibit significantly longer training times and performance
gaps compared to 3DGS variants, so we exclude them from
comparison. Our quantitative and qualitative comparisons
emphasize Ground Truth, the proposed JOGS, 3DGS [1],
CFGS [9] and GSHT [42], of which the last two are also
COLMAP-free methods.
E. Novel View Synthesis Evaluation
As shown in Tab. I, II and III, both CFGS and GSHT
suffer degraded reconstruction quality, primarily due to their
reliance on temporal continuity—when pose changes become
large (e.g., under frame-subsampling on Tanks and Temples),
their performance deteriorates sharply, as shown by the sharp
drop in PSNR. In contrast, our method is sequence-agnostic
and remains robust even under aggressive subsampling. As
illustrated in Fig. 2, our method generates sharper geometric
features and more coherent textures, in contrast to the blurred
reconstructions of CFGS and the fragmented surfaces of
GSHT. Beyond holistic visual assessment, Fig. 3 presents
fine-grained comparisons of structural details. Our method
achieves superior fidelity in geometric preservation and texture
reconstruction compared to baseline approaches.
In addition, it is noteworthy that in Fig. 2, the CFGS method
produces noticeably blurred novel view synthesis images due
to its inaccurate pose estimation. This issue becomes par-
ticularly pronounced in the detailed regions as illustrated in
Fig. 3. As demonstrated in the figure, both CFGS and GSHT
exhibitscale drift and spatial misalignment of the display units.
As shown in Fig. 3 (the first row), 3DGS shows obvious
blurring around high-frequency structures such as the edge of
the display. This is primarily due to the lack of joint camera
pose optimization during training, where even slight pose
inaccuracies can be amplified during dense rendering, leading

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
CFGS
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
CF3DGS(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.075
0.050
0.025
0.000
0.025
0.050
0.075
Ground Truth
CF3DGS(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.2
0.1
0.0
0.1
0.2
Z
0.04
0.02
0.00
0.02
0.04
0.06
0.08
Ground Truth
CF3DGS(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.20
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.05
0.04
0.03
0.02
0.01
0.00
0.01
0.02
0.03
Ground Truth
CF3DGS(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.06
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
CF3DGS(aligned)
X
0.15
0.10
0.05
0.00
0.05
0.10
0.15
0.20
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
0.20
Z
0.075
0.050
0.025
0.000
0.025
0.050
0.075
0.100
Ground Truth
CF3DGS(aligned)
GSHT
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.1
0.0
0.1
0.2
Z
0.050
0.025
0.000
0.025
0.050
0.075
0.100
Ground Truth
GSHT(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.08
0.06
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
GSHT(aligned)
X
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.2
0.1
0.0
0.1
0.2
Z
0.04
0.02
0.00
0.02
0.04
0.06
0.08
Ground Truth
GSHT(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.06
0.04
0.02
0.00
0.02
0.04
0.06
0.08
Ground Truth
GSHT(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.06
0.04
0.02
0.00
0.02
0.04
Ground Truth
GSHT(aligned)
X
0.15
0.10
0.05
0.00
0.05
0.10
0.15
0.20
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.06
0.04
0.02
0.00
0.02
0.04
0.06
0.08
Ground Truth
GSHT(aligned)
Ours
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.10
0.05
0.00
0.05
0.10
Z
0.02
0.00
0.02
0.04
0.06
Ground Truth
Ours(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.10
0.05
0.00
0.05
0.10
Z
0.08
0.06
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
Ours(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.2
0.1
0.0
0.1
0.2
Z
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
Ours(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.05
0.04
0.03
0.02
0.01
0.00
0.01
0.02
Ground Truth
Ours(aligned)
X
0.2
0.1
0.0
0.1
0.2
Y
0.10
0.05
0.00
0.05
0.10
0.15
Z
0.06
0.04
0.02
0.00
0.02
0.04
Ground Truth
Ours(aligned)
X
0.20
0.15
0.10
0.05
0.00
0.05
0.10
0.15
0.20
Y
0.10
0.05
0.00
0.05
0.10
Z
0.06
0.04
0.02
0.00
0.02
0.04
0.06
Ground Truth
Ours(aligned)
Fern
Flower
Orchids
Room
Trex
Horns
Fig. 4. Trajectory comparison on the LLFF dataset. Each row represents a method (CFGS, GSHT and Ours), and each column represents a scene.
TABLE IV
POSE ESTIMATION PERFORMANCE COMPARISON ON LLFF DATASET.
Scene
RPEtrans ↓
RPErot ↓
ATE ↓
CFGS
GSHT
Ours
CFGS
GSHT
Ours
CFGS
GSHT
Ours
Fern
8.908
6.656
0.146
2.830
2.349
0.039
0.161
0.129
0.014
Flowers
2.615
3.534
0.100
0.148
0.229
0.052
0.064
0.073
0.005
Horns
3.395
2.428
0.051
1.573
1.310
0.027
0.088
0.072
0.019
Orchids
3.586
4.170
0.135
1.992
2.059
0.117
0.074
0.098
0.018
Room
5.290
2.898
0.039
1.792
1.675
0.030
0.117
0.082
0.005
Trex
5.065
5.849
0.084
1.901
2.112
0.026
0.120
0.127
0.006
Mean
4.810
4.256
0.093
1.706
1.622
0.049
0.104
0.097
0.011
to structural blur and color artifacts. We evaluate on the Shiny
dataset, which features strong reflections and refractions. As
shown in Tab. III, JOGS matches COLMAP+3DGS in overall
metrics and significantly outperforms both COLMAP-free
baselines.
F. Camera Pose Estimation
In Tab. IV and Tab. V, we provide a quantitative com-
parison of camera pose estimation performance on the LLFF
and Tanks and Temples datasets, respectively. Following the
alignment strategy proposed in GSHT, the estimated camera
trajectories are first aligned with the ground truth in scale
before evaluation. We report the results in terms of ATE
and RPE. As shown in these results, our method achieves
the lowest errors on most scenes. Qualitative results are
visualized in Fig. 4 and Fig. 5. Our method produces estimated
trajectories that are significantly more consistent with the
ground truth than the baselines, demonstrating its robustness
and effectiveness in handling complex camera motions and
large-scale environments.
Furthermore, we compare our method with VGGT [50], a
feed-forward neural network that reports strong performance
across multiple 3D tasks, including camera parameter estima-
tion. As shown in Table VI, on the LLFF dataset, our method
achieves lower ATE and RPE on most scenes, with a lower
mean error overall.
G. Ablation Study
To validate the necessity of our joint optimization frame-
work, we conduct an ablation study comparing two variants:
(1) Initialization-only (using initialized poses without iterative
refinement during training) and (2) Full method (with alter-
nating Gaussian and pose optimization).
As shown in Tab. VII, the full method outperforms the
reduced initialization-only variant across all datasets. Init
means working with only pose initialization without iterative
refinement, while Full means working with the full version of
our method containing joint optimization of Gaussian points
and camera poses. The ablation study demonstrates that our
combined optimization framework effectively mitigates error
accumulation and enhances the synthesis accuracy of novel
view scenes by alternately updating Gaussian points and
refining camera poses using LK3D.

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
CFGS
X
0.2
0.1
0.0
0.1
0.2
Y
0.015
0.010
0.005
0.000
0.005
0.010
0.015
Z
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Ground Truth
CF3DGS(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.020
0.015
0.010
0.005
0.000
0.005
0.010
0.015
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
CF3DGS(aligned)
X
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Y
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
Z
0.04
0.02
0.00
0.02
0.04
0.06
0.08
0.10
Ground Truth
CF3DGS(aligned)
X
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.008
0.006
0.004
0.002
0.000
0.002
0.004
0.006
Z
0.02
0.01
0.00
0.01
0.02
0.03
Ground Truth
CF3DGS(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.02
0.01
0.00
0.01
0.02
0.03
Z
0.02
0.00
0.02
0.04
0.06
Ground Truth
CF3DGS(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.02
0.01
0.00
0.01
0.02
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
CF3DGS(aligned)
GSHT
X
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.015
0.010
0.005
0.000
0.005
0.010
0.015
0.020
Z
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Ground Truth
GSHT(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.010
0.005
0.000
0.005
0.010
0.015
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
GSHT(aligned)
X
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Y
0.0100
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
Z
0.04
0.02
0.00
0.02
0.04
0.06
0.08
0.10
Ground Truth
GSHT(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
0.4
Y
0.0100
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
Z
0.02
0.01
0.00
0.01
0.02
0.03
0.04
Ground Truth
GSHT(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.02
0.01
0.00
0.01
0.02
0.03
Z
0.02
0.00
0.02
0.04
0.06
Ground Truth
GSHT(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
0.0125
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
GSHT(aligned)
Ours
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.015
0.010
0.005
0.000
0.005
0.010
0.015
0.020
Z
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Ground Truth
Ours(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.005
0.000
0.005
0.010
0.015
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
Ours(aligned)
X
0.15
0.10
0.05
0.00
0.05
0.10
0.15
Y
0.0100
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
Z
0.050
0.025
0.000
0.025
0.050
0.075
0.100
0.125
Ground Truth
Ours(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
0.4
Y
0.015
0.010
0.005
0.000
0.005
0.010
0.015
Z
0.03
0.02
0.01
0.00
0.01
0.02
0.03
0.04
Ground Truth
Ours(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.02
0.01
0.00
0.01
0.02
0.03
Z
0.04
0.02
0.00
0.02
0.04
0.06
0.08
Ground Truth
Ours(aligned)
X
0.3
0.2
0.1
0.0
0.1
0.2
0.3
Y
0.0075
0.0050
0.0025
0.0000
0.0025
0.0050
0.0075
0.0100
0.0125
Z
0.05
0.00
0.05
0.10
0.15
Ground Truth
Ours(aligned)
Ballroom
Barn
Church
Family
Francis
Horse
Fig. 5. Trajectory comparison on the Tanks and Temples dataset. Each row represents a method (CFGS, GSHT and Ours), and each column represents a
different scene.
TABLE V
POSE ESTIMATION PERFORMANCE COMPARISON ON TANKS AND TEMPLES DATASET.
Scene
RPEtrans ↓
RPErot ↓
ATE ↓
CFGS
GSHT
Ours
CFGS
GSHT
Ours
CFGS
GSHT
Ours
Ballroom
2.759
0.306
0.126
3.374
0.076
0.035
0.196
0.004
0.040
Barn
6.915
1.007
0.153
7.216
0.202
0.063
0.190
0.025
0.005
Church
1.892
0.070
0.110
12.061
0.065
0.049
0.119
0.006
0.018
Family
1.838
0.484
0.127
7.192
0.126
0.030
0.169
0.007
0.033
Francis
4.141
0.276
0.102
6.112
0.566
0.057
0.194
0.011
0.016
Horse
8.963
0.789
0.192
7.140
0.159
0.026
0.205
0.009
0.005
Ignatius
8.785
0.345
0.174
7.381
0.059
0.052
0.206
0.011
0.034
Museum
8.224
3.418
0.232
4.835
2.912
0.039
0.227
0.057
0.168
Mean
5.440
0.837
0.152
6.914
0.521
0.044
0.188
0.016
0.040
TABLE VI
POSE ESTIMATION PERFORMANCE COMPARISON BETWEEN OUR
METHOD AND VGGT ON THE LLFF DATASET.
Scene
RPEtrans ↓
RPErot ↓
ATE ↓
VGGT
Ours
VGGT
Ours
VGGT
Ours
Fern
0.285
0.146
0.071
0.039
0.038
0.014
Flowers
0.160
0.100
0.036
0.052
0.004
0.005
Horns
0.056
0.051
0.009
0.027
0.010
0.019
Orchids
0.312
0.135
0.087
0.117
0.025
0.018
Room
0.104
0.039
0.035
0.030
0.017
0.005
Trex
0.138
0.084
0.032
0.026
0.011
0.006
Mean
0.203
0.093
0.059
0.049
0.017
0.011
V. CONCLUSION
In this paper, we introduce a novel view synthesis frame-
work that jointly optimize pose estimation and 3DGS, without
TABLE VII
ABLATION STUDY OF JOINT OPTIMIZATION ACROSS THREE BENCHMARK
DATASETS.
Dataset
PSNR↑
SSIM↑
LPIPS↓
Init
Full
Init
Full
Init
Full
LLFF-NeRF
25.25
25.39
0.81
0.81
0.20
0.20
Tanks and Temples
25.94
26.91
0.86
0.88
0.14
0.13
Shiny
24.98
25.58
0.80
0.85
0.23
0.21
Mean
25.39
25.96
0.82
0.85
0.19
0.18
requiring camera poses as inputs. This framework outperforms
state-of-the-art methods in both pose estimation accuracy and
rendering quality, particularly under challenging conditions,
by leveraging an alternating optimization strategy for 3D
Gaussian representations and camera poses.
Limitations. Despite the effectiveness in both camera pose

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
estimation and rendering quality, our method requires an
increased training time due to the increased pose refinement
operation. We plan to address this issue by exploring parallel
optimization strategy in the future work.
ACKNOWLEDGMENTS
This work is supported by the National Natural Science
Foundation of China (No. 62376020).
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Trans. Graph.,
vol. 42, no. 4, 2023.
[2] H. Shuai, Y. Shi, Y. Sun, and Q. Liu, “Adversarial pruning networks for
compact 3d gaussian splatting,” IEEE Transactions on Multimedia, pp.
1–10, 2025.
[3] Z. Huang, M. Xu, and S. Perry, “Structgs: Adaptive spherical harmonics
and rendering enhancements for superior 3d gaussian splatting,” IEEE
Transactions on Multimedia, pp. 1–12, 2025.
[4] C. Luo, D. Di, X. Yang, Y. Ma, Z. Xue, W. Chen, X. Gou, and
Y. Liu, “Trame: Trajectory-anchored multi-view editing for text-guided
3d gaussian manipulation,” IEEE Transactions on Multimedia, vol. 27,
pp. 2886–2898, 2025.
[5] Y. Jiang, J. Li, H. Qin, Y. Dai, J. Liu, G. Zhang, C. Zhang, and T. Yang,
“Gs-sfs: Joint gaussian splatting and shape-from-silhouette for multiple
human reconstruction in large-scale sports scenes,” IEEE Transactions
on Multimedia, vol. 26, pp. 11 095–11 110, 2024.
[6] D. R. Freitas, I. Tabus, and C. Guillemot, “Visibility-based geometry
pruning of neural plenoptic scene representations,” IEEE Transactions
on Multimedia, vol. 28, pp. 1–16, 2026.
[7] C. Niu, M. Tao, B.-K. Bao, and C. Xu, “Colview: Consistent text-
guided grayscale scene colorization from multi-view images,” IEEE
Transactions on Multimedia, pp. 1–11, 2026.
[8] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and
L. Gao, “Recent advances in 3d gaussian splatting,” 2024.
[9] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, “Colmap-
free 3d gaussian splatting,” in CVPR, 2024, pp. 20 796–20 805.
[10] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,”
in CVPR, 2016.
[11] J. L. Sch¨onberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixelwise
view selection for unstructured multi-view stereo,” in ECCV, 2016.
[12] W. Bian, Z. Wang, K. Li, J. Bian, and V. A. Prisacariu, “Nope-nerf:
Optimising neural radiance field with no pose prior,” in CVPR, 2023.
[13] Y. Chen, R. A. Potamias, E. Ververas, J. Song, J. Deng, and G. H. Lee,
“Zerogs: Training 3d gaussian splatting from unposed images,” 2024.
[14] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu,
B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, “In-
stantsplat: Unbounded sparse-view pose-free gaussian splatting in 40
seconds,” 2024.
[15] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed
optimization and statistical learning via the alternating direction method
of multipliers,” Found. Trends Mach. Learn., vol. 3, no. 1, p. 1–122,
Jan. 2011.
[16] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM TOG, vol. 36,
no. 4, 2017.
[17] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari,
R. Ramamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical
view synthesis with prescriptive sampling guidelines,” ACM TOG, 2019.
[18] S. Wizadwongsa, P. Phongthawee, J. Yenphraphai, and S. Suwajanakorn,
“Nex: Real-time view synthesis with neural basis expansion,” in CVPR,
2021.
[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: representing scenes as neural radiance fields for view
synthesis,” Commun. ACM, vol. 65, no. 1, p. 99–106, 2021.
[20] S. Shen, W. Li, X. Huang, Z. Zhu, J. Zhou, and J. Lu, “Sd-nerf: Towards
lifelike talking head animation via spatially-adaptive dual-driven nerfs,”
IEEE Transactions on Multimedia, vol. 26, pp. 3221–3234, 2024.
[21] Y. Chen, L. Zhang, S. Zhao, and Y. Zhou, “Atm-nerf: Accelerating train-
ing for nerf rendering on mobile devices via geometric regularization,”
IEEE Transactions on Multimedia, vol. 27, pp. 3279–3293, 2025.
[22] W. Guo, B. Wang, and L. Chen, “Neuv-slam: Fast neural multireso-
lution voxel optimization for rgbd dense slam,” IEEE Transactions on
Multimedia, vol. 27, pp. 7546–7556, 2025.
[23] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in CVPR,
2022, pp. 5460–5469.
[24] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz,
and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,” in
ICCV, 2021, pp. 5845–5854.
[25] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields
for space-time view synthesis of dynamic scenes,” in CVPR, 2021.
[26] S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, and J. Valentin,
“Fastnerf: High-fidelity neural rendering at 200fps,” arXiv preprint
arXiv:2103.10380, 2021.
[27] C. Sun, M. Sun, and H. Chen, “Direct voxel grid optimization: Super-
fast convergence for radiance fields reconstruction,” in CVPR, 2022.
[28] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-splatting:
Alias-free 3d gaussian splatting,” in CVPR, 2024, pp. 19 447–19 456.
[29] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-
gs: Structured 3d gaussians for view-adaptive rendering,” in CVPR,
2024, pp. 20 654–20 664.
[30] L. Radl, M. Steiner, M. Parger, A. Weinrauch, B. Kerbl, and M. Stein-
berger, “StopThePop: Sorted Gaussian Splatting for View-Consistent
Real-time Rendering,” ACM TOG, vol. 4, no. 43, 2024.
[31] R. Liu, R. Xu, Y. Hu, M. Chen, and A. Feng, “Atomgs: Atomizing
gaussian splatting for high-fidelity radiance field,” 2024.
[32] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in SIGGRAPH, 2024.
[33] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, “Sc-
gs: Sparse-controlled gaussian splatting for editable dynamic scenes,” in
CVPR, 2024, pp. 4220–4230.
[34] Q. Shen, X. Yang, M. B. Mi, and X. Wang, “Vista3d: Unravel the 3d
darkside of a single image,” in ECCV, 2024, p. 405–421.
[35] R. Peng, W. Xu, L. Tang, L. Liao, J. Jiao, and R. Wang, “Structure
consistent gaussian splatting with matching prior for few-shot novel view
synthesis,” in NeurIPS, 2024.
[36] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-
Y. Lin, “inerf: Inverting neural radiance fields for pose estimation,” in
IROS, 2021, pp. 1323–1330.
[37] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting
neural radiance fields,” in ICCV, 2021.
[38] S.-F. Chng, S. Ramasinghe, J. Sherrah, and S. Lucey, “Gaussian ac-
tivated neural radiance fields for high fidelity reconstruction and pose
estimation,” in ECCV, 2022.
[39] S. Ramasinghe and S. Lucey, “Beyond periodicity: towards a unifying
framework for activations in coordinate-mlps,” in ECCV, 2022.
[40] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r:
Geometric 3d vision made easy,” in CVPR, 2024.
[41] H. Wang and L. Agapito, “3d reconstruction with spatial memory,” arXiv
preprint arXiv:2408.16061, 2024.
[42] B. Ji and A. Yao, “Sfm-free 3d gaussian splatting via hierarchical
training,” 2024.
[43] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,”
IJCV, vol. 60, no. 2, pp. 91–110, 2004.
[44] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm
for model fitting with applications to image analysis and automated
cartography,” Commun. ACM, vol. 24, no. 6, p. 381–395, 1981.
[45] V. Lepetit, F. Moreno-Noguer, and P. Fua, “Epnp: An accurate o(n)
solution to the pnp problem,” IJCV, vol. 81, 2009.
[46] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon,
“Bundle adjustment - a modern synthesis,” in ICCV Workshop, 1999,
p. 298–372.
[47] B. D. Lucas and T. Kanade, “An iterative image registration technique
with an application to stereo vision,” in IJCAI, 1981, p. 674–679.
[48] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality
assessment: from error visibility to structural similarity,” IEEE TIP,
vol. 13, no. 4, pp. 600–612, 2004.
[49] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,”
2018.
[50] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and
D. Novotny, “Vggt: Visual geometry grounded transformer,” 2025.
[Online]. Available: https://arxiv.org/abs/2503.11651
