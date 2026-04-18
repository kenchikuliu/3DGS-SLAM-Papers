<!-- page 1 -->
PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D
Reconstruction
Changjian Jiang1,2*
Kerui Ren3,2*
Xudong Li4,2
Kaiwen Song5,2
Guanghao Li6
Linning Xu7,2
Tao Lu2
Junting Dong2
Yu Zhang1†
Bo Dai8
Mulin Yu2†
1Zhejiang University
2Shanghai Artificial Intelligence Laboratory
3Shanghai Jiao Tong University
4Northwestern Polytechnical University
5University of Science and Technology of China
6Fudan University
7The Chinese University of Hong Kong
8The University of Hong Kong
Neural Gaussians
Triangles
Planes
SelfCaptured
(INNOVATION)
…
Monocular Captures (126 views) 
TIME: 63.9s  PSNR: 35.30dB
Figure 1. PLANING introduces a loosely coupled triangle-Gaussian representation for streaming 3D reconstruction, balancing geometric
accuracy, high-fidelity rendering, and computational efficiency. Building upon this hybrid representation, we further adapt it to an efficient
streaming reconstruction framework for monocular image sequences, enabling effective modeling of both scene geometry and appearance in
a streaming setting. Leveraging the inherent edge-preserving property of triangle primitives, our method allows for the explicit extraction of
compact planar structures, which can serve as a high-performance simulation environment for locomotion training in embodied AI.
Abstract
Streaming reconstruction from monocular image sequences
remains challenging, as existing methods typically favor ei-
ther high-quality rendering or accurate geometry, but rarely
both. We present PLANING, an efficient on-the-fly recon-
∗Equal contribution. †Corresponding author.
struction framework built on a hybrid representation that
loosely couples explicit geometric primitives with neural
Gaussians, enabling geometry and appearance to be mod-
eled in a decoupled manner. This decoupling supports an
online initialization and optimization strategy that separates
geometry and appearance updates, yielding stable stream-
ing reconstruction with substantially reduced structural re-
1
arXiv:2601.22046v4  [cs.CV]  11 Mar 2026

<!-- page 2 -->
dundancy. PLANING improves dense mesh Chamfer-L2 by
18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR,
and reconstructs ScanNetV2 scenes in under 100 seconds,
over 5× faster than 2D Gaussian Splatting, while matching
the quality of offline per-scene optimization. Beyond recon-
struction quality, the structural clarity and computational ef-
ficiency of PLANING make it well suited for a broad range of
downstream applications, such as enabling large-scale scene
modeling and simulation-ready environments for embodied
AI. Project page: https://city-super.github.io/PLANING/.
1. Introduction
3D scene reconstruction is a core capability for embodied
intelligence, autonomous driving, and AR/VR, providing
the spatial understanding required for perception and in-
teraction [30, 43]. While offline reconstruction methods
following a capture-then-process paradigm have reached
a high level of maturity, their reliance on time-intensive
post-processing limits scalability and responsiveness in time-
critical scenarios. This has driven a growing demand for
real-time, on-the-fly reconstruction frames.
A central challenge in on-the-fly 3D reconstruction is a
scene representation that jointly achieves high geometric
accuracy and real-time efficiency. Recently, 3D Gaussian
Splatting (3DGS) [18] has emerged as a compelling explicit
representation, offering high visual fidelity with efficient ren-
dering, and has therefore been widely adopted in streaming
reconstruction methods [5, 17, 20, 25, 26, 50]. Despite their
success, existing streaming 3DGS-based methods share a
fundamental limitation: the absence of explicit, compact,
and stable geometry. While Gaussian primitives are effective
for appearance modeling, they lack well-defined structural
boundaries, making it difficult to recover coherent and ed-
itable surface geometry without sacrificing rendering quality.
Moreover, optimizing Gaussians to reproduce input views
inherently biases learning toward appearance, often at the
expense of geometric consistency especially under sparse
observations or novel viewpoints. To compensate, these
methods rely on a large number of primitives, leading to
significant redundancy, increased computational cost, and
limited scalability in streaming settings.
To address this challenge, (1) we propose a hybrid repre-
sentation that decouples geometry from appearance, enabling
both efficient geometric reconstruction and high-fidelity
rendering. For geometry, we introduce learnable triangle
primitives. Triangles provide well-defined edges and ex-
plicitly model surface structures, making them particularly
effective for capturing the planar layouts prevalent in in-
door environments. For appearance modeling, we adopt a
neural-Gaussian formulation inspired by Scaffold-GS [23],
in which Gaussian attributes are decoded from a fused fea-
ture representation that combines triangle features with per-
Appearance
Planar Acc.
#Prim.
Time
Geometry
Ours
ARTDECO
2DGS
PGSR
MeshSplatting
Figure 2. PLANING consistently outperforms existing streaming
and per-scene reconstruction methods across geometry accuracy,
rendering quality, computational efficiency, and memory usage,
while maintaining clear and well-structured planar geometry.
Gaussian features, encouraging local rendering consistency
and smoothness. This design establishes a synergistic cou-
pling: triangles act as stable structural anchors that mitigate
drift and redundancy, while rendering gradients propagated
through neural Gaussians refine the underlying geometry in
a controlled way, allowing appearance cues to guide surface
optimization without conflicting against the structural con-
straints. Building upon this hybrid representation, (2) we
introduce PLANING, a framework for efficient monocular
3D reconstruction in a streaming setting. Our framework
leverages feed-forward models as learned priors to enable
robust camera pose estimation and to provide stable geo-
metric guidance for scene modeling. To achieve both high
efficiency and global consistency, we adopt a tailored initial-
ization strategy that applies photometric and spatial filtering
to reduce redundant primitives, and perform global map ad-
justment to keep the reconstructed 3D model aligned with
continually optimized camera poses.
Extensive experiments across diverse indoor and outdoor
benchmarks demonstrate that our method outperforms state-
of-the-art approaches in geometric accuracy, rendering qual-
ity, training efficiency, and primitive count, as illustrated
in Fig. 2. By preserving salient structures while removing
redundant geometry, our representation enables the export of
compact and consistent 3D planes. This highly compressed
geometric output, characterized by a significantly reduced tri-
angle count, shows strong potential for enhancing large-scale
scene reconstruction and improving the global consistency
of pose estimation. Additionally, the structural clarity and
computational efficiency of our model make it well suited for
simulation-ready scene modeling, such as supporting local
motion policy training in embodied AI.
Our main contributions can be summarized as follows:
• Decoupled Geometry and Appearance Modeling. We in-
troduce a hybrid scene representation that loosely couples
explicit, learnable triangle primitives for geometry with
2

<!-- page 3 -->
neural Gaussians for appearance, enabling compact, sta-
ble, and editable structure while preserving high-fidelity
rendering.
• Efficient Streaming Reconstruction Framework. We de-
velop an efficient on-the-fly monocular reconstruction
framework that leverages the proposed representation to-
gether with streaming-aware initialization and global map
adjustment.
• State-of-the-Art Results and Broad Applicability.
We
demonstrate state-of-the-art performance in both geomet-
ric accuracy and rendering quality across diverse indoor
and outdoor benchmarks, and showcase the versatility
of our approach for downstream tasks including plane-
guided pose refinement, large-scale scene reconstruction,
and simulation-ready environments for embodied AI.
2. Related Work
3D Reconstruction.
Reconstructing 3D geometry from
multi-view images is a long-standing and fundamental prob-
lem in computer graphics. Traditional methods [34] trans-
form calibrated images into point clouds and optimize them
into implicit fields, followed by mesh extraction using
Marching Cubes [22]. More Recently, Neural Radiance
Fields (NeRF) [27] established a neural rendering milestone
by using MLPs for ray-based synthesis. However, NeRF-
based methods are limited by their implicit nature and costly
per-ray sampling, which hinders scalability and geometric
control. To address these limitations, 3D Gaussian Splatting
(3DGS) [18] employs explicit anisotropic Gaussian primi-
tives, leveraging efficient rasterization to enable real-time re-
construction [16, 32]. Nevertheless, the emphasis on render-
ing efficiency in 3DGS-based methods often compromises
geometric consistency, making it difficult to recover intricate
structural details without robust geometric constraints.
3DGS Variants.
Various extensions have explored al-
ternative primitives to better align with scene geometry.
2DGS [14], GSS [7], and Quadratic Gaussian Splatting [52]
replace anisotropic Gaussians with ellipsoidal or quadric
forms for superior surface alignment. Other works incor-
porate explicit geometric elements, such as the 3D con-
vexes [12] and triangles [1, 11, 15], to compactly model
hard-edged scenes. Similarly, PlanarSplatting [37] utilizes
rectangular primitives to achieve structured and efficient in-
door planar reconstructions. Despite these advances, single-
representation methods often struggle to balance geometric
precision with rendering fidelity. To bridge this gap, recent
dual-branch approaches such as GSDF [47] and 3DGSR [24]
integrate neural signed distance fields (SDFs) with 3DGS.
While this enables partial geometry–appearance decoupling,
it introduces significant computational overhead and opti-
mization complexity. Alternatively, 3D-GES [44] adopts a
bi-scale formulation using 2D surfels for coarse structure and
3D Gaussians for fine detail. However, this design primar-
ily targets appearance enhancement rather than achieving a
principled, explicit decoupling of geometry and appearance.
Streaming Reconstruction.
Classical visual SLAM
frameworks provide robust online tracking and mapping
but often lack the fidelity required for high-quality render-
ing [3, 28, 31]. To address this, recent works have integrated
volumetric rendering into SLAM pipelines to enable online
novel view synthesis [2, 49, 51, 54]. While NeRF-based
SLAM achieves photorealistic results, the high computa-
tional cost of per-ray volumetric rendering limits its suitabil-
ity for real-time applications.
In contrast, 3DGS has attracted increasing attention for
SLAM integration due to its explicit representation and
efficient rendering, with some methods directly propagat-
ing gradients from rendering losses to optimize camera
poses [10, 17, 25, 50]. However, monocular frameworks
often struggle to simultaneously balance robustness, recon-
struction accuracy, and efficiency. Recent on-the-fly NVS
approaches [26] show that GPU-friendly mini-bundle ad-
justment combined with incremental 3DGS updates can
enable interactive reconstruction, yet they remain fragile
on casual, unposed sequences. Meanwhile, feed-forward
models [21, 29, 38, 39] pretrained on large-scale datasets
have emerged as an alternative paradigm, reconstructing 3D
scenes directly without per-scene optimization. These meth-
ods fall into two categories: pose-aware approaches, which
leverage camera poses for rapid reconstruction, and pose-free
approaches, which perform end-to-end reconstruction from
raw images using point maps or 3DGS. While these meth-
ods offer strong robustness and fast inference across diverse
scenarios, they generally underperform optimization-based
approaches in accuracy and struggle with global consistency,
high-resolution inputs, and long-sequence scalability.
3. Method
In this section, we first introduce our dual scene representa-
tion that combines learnable triangles with neural Gaussians
(Sec. 3.1). We then describe how we adapt this representation
into an on-the-fly reconstruction framework, achieving both
efficiency and high-quality 3D reconstruction (Sec. 3.2).
3.1. Loosely-coupled Triangle-Gaussian Represen-
tation
We first detail the triangle primitives and our differentiable
rasterizer. Subsequently, we explain the interaction between
neural Gaussians and their corresponding triangles, followed
by the integrated rendering process.
3.1.1. Learnable Triangles for Geometry
We propose learnable triangle primitives based on a vertex-
based formulation and a differentiable triangle rasterizer.
3

<!-- page 4 -->
Key
Frames
Common 
Frames
Input Frames
Tracking & Bundle Adjustment
Lgeo, Lrgb
Laplace Norm
RGB
 Triangle Voxelization
Gaussian Initialization
Primitive  Initialization
Neural GS Rasterization
Pose from Backend 
Triangle Rasterization
Global Map 
Update 
Planes 
Planar Abstraction
⋯
Figure 3. Pipeline of PLANING. PLANING adopts a hybrid representation in which triangles explicitly model scene geometry, while
neural Gaussians decoded from these triangles render appearance. Built upon this representation, we develop a streaming reconstruction
framework that takes unposed monocular image sequences as input and comprises a frontend for camera tracking, a backend for global pose
optimization, and a mapper for scene reconstruction. Specifically, the mapper incorporates an efficient primitive initialization strategy to
reduce redundancy. The recontructed triangle soup further enables efficient planar abstraction, facilitating a range of downstream tasks.
p0
p1
p2
μ
tv
n
tu
(a) Definition of the Local Frame 
Ours Rendered Normal
GT Normal
w/o Subdivision
w/o Our Visibility 
Criterion
(b) Results of Forward Rendering 
Figure 4. Definition of the local frame and results of forward
rendering. Our triangle rasterizer enables correct and reliable
forward rendering of triangles.
Vertix-based Primitive Definition.
As illustrated in
Fig. 4(a), we parameterize each triangle primitive by its
three learnable vertices {p0, p1, p2}. To facilitate efficient
and differentiable rendering, we define a local coordinate
frame for each triangle:
tu =
p0 −µ
∥p0 −µ∥2
,
tv = n × tu,
su = ∥p0 −µ∥2 ,
sv = |tv · (p1 −µ)| ,
n =
(p1 −p0) × (p2 −p0)
∥(p1 −p0) × (p2 −p0)∥2
,
(1)
where the barycenter µ is set as the origin of the local
frame. Under this construction, the three vertices can be
expressed in the local tangent plane as {p′
0, p′
1, p′
2} =
{(0, 1)T , (a, 1)T , (−1 −a, −1)T }, where a = tu · (p1 −µ)
is the only degree of freedom in the local frame.
Following 3D Convex Splatting (3DCS) [12], we further
introduce two learnable triangle-wise parameters, δ > 0 and
σ > 0, to control edge sharpness and boundary smoothness.
Each triangle is also associated with a learnable opacity
parameter α, analogous to 3DGS [18].
Differentiable Triangle Rasterizer.
We implement an ef-
ficient differentiable triangle rasterizer that enables direct
supervision of triangles using prior normals and depths. To
obtain unbiased depth rendering, we adopt an explicit ray-
triangle intersection strategy, similar in spirit to 2DGS [14].
We further introduce the edge-preserving contribution func-
tion as:
w(ˆx) =
Sigmoid

−σ log


2
X
j=0
exp (δ dist(ˆx, ej))



α,
(2)
where dist(ˆx, ej) denotes the distance from the intersection
point ˆx to the j-th triangle edge in the local tangent plane.
Thanks to the local frame parameterization, these distances
can be computed analytically as:





dist(ˆx, e0) = u + (1 −a)v −1,
dist(ˆx, e1) = −2u + (2a + 1)v −1,
dist(ˆx, e2) = u + (−2 −a)v −1,
(3)
where ˆx = (u, v)T . This closed-form formulation signifi-
cantly simplifies both forward evaluation and gradient prop-
agation. Notably, our contribution computation differs from
3DCS, where contributions are computed directly on the
image plane rather than in the local surface domain.
Finally, triangles are rendered into depth and normal maps
4

<!-- page 5 -->
using front-to-back alpha compositing:
N(x) =
N
X
i=1
niw (ˆxi)
i−1
Y
j=1
(1 −w (ˆxj)) ,
D(x) =
N
X
i=1
diw (ˆxi)
i−1
Y
j=1
(1 −w (ˆxj)) ,
(4)
where di denotes the distance from the i-th intersection point
to the pixel. The N ordered intersection points {ˆxi} between
the triangles and pixel x are computed using our custom
CUDA-based rasterizer. To enable accurate differentiable
rendering of triangles, we define a new criterion for visibil-
ity determination and design a triangle-subdivision-based
primitive depth sorting algorithm in the rasterizer to address
rendering issues introduced by the edge-preserving contribu-
tion function, as illustrated in Fig. 4(b). The detailed forward
rendering pipeline is described in the Appendix A.1.
3.1.2. Neural Gaussians for Appearance Modeling
To achieve a decoupled yet consistent representation of ge-
ometry and appearance, we introduce neural Gaussians to
flexibly encode view-dependent appearance. Inspired by
Scaffold-GS [23], neural Gaussians are anchored to the tri-
angles and used for appearance. Specifically, each learnable
triangle is associated with a context feature ft ∈R24. Each
Gaussian is parameterized by a learnable position offset
og ∈R3, spherical harmonics (SH) coefficients, opacity
αg ∈R, a base scale sg ∈R3, a base quaternion qg ∈R4,
and an individual feature fg ∈R8. In addition, each Gaus-
sian maintains the index it of its corresponding triangle as
the geometric association.
During rendering, the position of each Gaussian µg =
og+µt, where µt denotes the barycenter of the associated tri-
angle. Then we predict the final scale s = sg⊙MLPs(ft⊕fg)
and rotation q = ϕ(qg ⊙MLPq(ft ⊕fg)), where ⊙denotes
element-wise multiplication, ⊕denotes feature concatena-
tion and ϕ(·) denotes ℓ2 normalization to ensure valid rota-
tion quaternions. Through this design, geometry and appear-
ance are represented in a consistent and coherent manner.
Notably, each triangle hosts a flexible number of Gaussians,
enabling the representation to adapt to local scene details.
3.2. Streaming Reconstruction Framework
3.2.1. Overview
As shown in Fig. 3, we design a streaming reconstruction
framework built upon our hybrid representation, leveraging
its capacity for high-fidelity modeling. Following [20], our
framework takes unposed monocular image sequences as
input and comprises three main components: a frontend for
camera tracking, a backend for global pose optimization, and
a mapper for scene reconstruction.
The frontend processes incoming frames in a stream-
ing manner to estimate camera motion, select keyframes,
and predict per-frame dense point maps using feed-forward
models [19]. The backend subsequently performs loop clo-
sure detection [39] and global bundle adjustment [29] over
keyframes to improve global pose consistency, which is criti-
cal for accurate geometry reconstruction. The mapper recon-
structs scene geometry and appearance by integrating posed
images and dense point maps provided by the backend.
Unlike previous streaming methods that rely on a single
representation [20, 26], our mapper utilizes a loosely coupled
triangle–Gaussian representation to decouple geometry from
appearance modeling, thus mitigating mutual interference.
Guided by geometric priors from the backend, we introduce
a novel primitive initialization and optimization strategy. To
maintain global geometric consistency, we perform a global
map adjustment whenever the backend updates the global
camera poses.
Following streaming reconstruction, planar structures can
be directly extracted from the triangle soup via a coarse-
to-fine plane extraction algorithm. Furthermore, our frame-
work supports dense mesh reconstruction through depth fu-
sion. Additional implementation details are provided in
Appendix A.3.
3.2.2. Primitive Initialization
Upon the arrival of a keyframe from the backend, the frame-
work determines the optimal locations for instantiating new
primitives. To maintain a compact global map and miti-
gate structural redundancy, triangle insertion is restricted to
regions exhibiting insufficient geometric coverage or high
reconstruction error, guided by image-level priors. Specifi-
cally, we first apply photometric filter, which prioritizes high-
frequency regions and poorly reconstructed areas by com-
puting an insertion probability Pa(u, v) at each pixel (u, v)
using the Laplacian of Gaussian (LoG) operator Φ(·)[26]
to measure the discrepancy between the ground truth and
rendered images:
Pa(u, v) = max

Φ(I) −Φ(˜I), 0

,
(5)
where Φ(I) = min(∥∇2(Gσg) ∗I(u, v)∥, 1), I and ˜I rep-
resent the ground-truth and rendered images, respectively,
and Gσg denotes a Gaussian smoothing kernel. A new geo-
metric primitive is considered only when Pa(u, v) exceeds a
predefined threshold τa.
To further suppress structural redundancy, we apply a
spatial filter to candidates passing the photometric filter. For
each candidate pixel, we compute its back-projected 3D
center ci and prune it if any existing triangles fall within its
local vicinity of size V (di):
V (di) = Vmin + (Vmax −Vmin) ·
 di −dmin
dmax −dmin
p
, (6)
5

<!-- page 6 -->
where
di
denotes
the
observation
depth,
and
{Vmin, Vmax, dmin, dmax, p}
are
hyperparameters
that
modulate the vicinity scale. This depth-adaptive spatial filter
ensures map compactness by preventing redundant primitive
growth in already-reconstructed regions.
Once a candidate pixel (u, v) is selected, a triangle is
initialized. Each triangle is parameterized by its vertices
pt, opacity αt, sharpness δt, smoothness σt, and a feature
vector ft. Following geometric scaling principles, the world-
space scale st = 3di

2f
p
Φ(I), where f is the focal length.
The triangle orientation is determined by the normal prior
at (u, v). Specifically, three unit vectors vt,k are sampled
on the local tangent plane, and the vertex positions pt =
st vt. The opacity is initialized as αt = 0.2 C(u, v) to down-
weight low-confidence regions, where C(u, v) is the backend
confidence score.
Then, neural Gaussians are initializeded at triangle
barycenters for appearance modeling. We adaptively set
the number of Gaussians per triangle to Kmax if Φ(I) > 0.4,
and Kmin otherwise. Here, the hyperparameters Kmax and
Kmin define the bounds of the representational capacity
based on scene detail. For primitive attributes, we initialize
offsets og, rotation qg, and features fg to zero, while Gaus-
sian opacity αg is synchronized with αt. The base scale is
defined as sg = di

2f
p
Φ(I) · 1 to align with local geome-
try. Crucially, the zero-order spherical harmonic coefficient
SH0 is extracted from the pixel color at (u, v), with higher
coefficients zero-initialized.
3.2.3. Training
We supervise the triangles and Gaussians with separate geo-
metric and appearance losses for decoupled optimization:
L = Lgeo + Lrgb.
(7)
For geometry, we leverage multi-view depth Dp and nor-
mal Np priors from MASt3R [19] to supervise triangles,
penalizing deviations from the rendered depth Dt and nor-
mals Nt:
Lgeo = λd∥Dt −Dp∥1 + λn∥Nt −Np∥1 + λoLo,
(8)
where λd and λn are user-prescribed weights. Lo is an en-
tropy loss on triangle opacity α, following [9]. We regularly
prune triangle primitives with α < 0.5, which removes re-
dundant geometry and maintains a compact representation.
The appearance loss supervises neural Gaussians via:
Lrgb =
(1 −λc)∥Cgt −Cgs∥1 + λcSSIM
 Cgt, Cgs

+ λsLs, (9)
where Ls is a volume regularization term adopted from
Scaffold-GS [23]. Notably, appearance gradients from Gaus-
sians are back-propagated to the triangles, enabling implicit
refinement of the underlying geometry. More details are
provided in Appendix A.2.
3.2.4. Global Map Update
In our streaming framework, camera poses are continuously
refined within the backend, while primitives in the map-
per are initialized and optimized using the poses available
at that timestamp. This asynchronous update can lead to
pose–model misalignment. To maintain consistency between
the refined poses and the 3D model, we explicitly transform
the primitives after the pose optimization. Specifically, we
record the source keyframe for each primitive and apply a
relative transformation ∆T = TnT−1
o
to its attributes when
the corresponding keyframe pose changes from To to Tn:
p′
t = ∆Tpt,
o′
g = ∆T(og + µt) −µ′
t,
q′
g = R−1(∆RR(qg)),
(10)
where ∆R is the rotation component of ∆T, and R(·) maps
quaternions to rotation matrices. Here, {p′
t, µ′
t, o′
g, q′
g} de-
note the updated triangle and Gaussian parameters.
4. Experiments
4.1. Experimental Setup
Datasets.
We evaluate PLANING on 56 real-world scenes
from diverse benchmarks: 20 from ScanNet++ [45], 10
from ScanNetV2 [6], 6 from VR-NeRF [42], 4 from FAST-
LIVO2 [53], 8 from KITTI [8], and 8 from Waymo [36],
covering a wide range of indoor and outdoor environments.
Baselines.
We compare PLANING with state-of-the-art
methods across three categories.
For per-scene recon-
struction, we evaluate 2DGS [14], PGSR [4], and Mesh-
Splatting [11].
For streaming reconstruction, we select
ARTDECO [20], OnTheFly-NVS [26], S3PO-GS [5], and
MonoGS [25]. For planar reconstruction, we include Planar-
Splatting [37] and AirPlanes [41]. To ensure fair comparison,
all per-scene reconstruction baselines are augmented with
the same MASt3R geometric priors used in ours. For meth-
ods requiring poses, we provide our estimated poses for fair
comparison.
Metrics.
We conduct a comprehensive evaluation of our
framework across three tasks. For planar reconstruction,
following PlanarSplatting [37], we evaluate plane geometry
using Chamfer Distance and F-score. For datasets with
ground-truth plane annotations, we further assess the top-20
largest planes using Planar Fidelity, Planar Accuracy, and
Planar Chamfer metrics. For dense mesh reconstruction, we
report Chamfer Distance and F-score, while for novel view
synthesis (NVS), we use standard metrics including PSNR,
SSIM [40], and LPIPS [48]. In addition, we report training
time and the number of primitives to quantify computational
efficiency.
6

<!-- page 7 -->
Table 1. Quantitative comparison of planar reconstruction. We evaluate the geometric and planar metrics on the ScanNet++, ScanNetV2,
and FAST-LIVO2 datasets. Ours achieves top-tier performance in most categories while significantly reducing primitive count and runtime
(reported in minutes).
Method
ScanNet++
ScanNetV2
FAST-LIVO2
Geometry
Planar
Time #Prim.
Geometry
Planar
Time #Prim.
Geometry
Time #Prim.
Ch-L2↓F-score↑Fidelity↓Acc↓Ch-L2↓
Ch-L2↓F-score↑Fidelity↓Acc↓Ch-L2↓
Acc↓Comp↓Ch-L2↓F-score↑
2DGS†
3.89
81.64
8.16
7.19
7.67
16.1 415.3k
6.48
53.73
15.56
8.12
11.84
10.9 1196.8k 14.11 48.17
53.45
60.47
35.8 3197.0k
PGSR†
3.87
81.98
7.44
7.23
7.33
31.2 353.4k
6.59
54.28
15.88
8.46
12.17
21.3 629.1k 13.95 49.16
54.13
60.75
25.5 1065.8k
MeshSplatting†
9.13
47.19
37.87
10.71 24.29
38.5 1825k 11.15
30.73
40.16
11.45 25.81
9.7
291.3k 14.52 66.68
62.97
47.31
26.3 2505.1k
AirPlanes
25.19
19.21
47.10
25.97 36.53
3.7
/
6.34
55.33
9.68
8.90
9.29
3.5
/
-
-
-
-
-
-
PlanarSplatting
7.27
49.78
9.64
13.35 11.50
8.8
1.0k
6.54
51.67
9.72
10.77 10.24
3.1
1.76k
-
-
-
-
-
-
ARTDECO
3.82
83.08
15.84
7.92
11.88
5.6 478.3k
6.05
57.58
19.62
8.73
14.18
2.2
621.5k 14.17 63.63
57.19
54.23
6.5
501.6k
Ours
3.53
86.88
7.24
6.95
7.09
5.5
61.6k
5.68
62.15
10.55
7.58
9.07
2.1
56.1k 12.58 30.60
36.77
65.89
3.6
101.4k
/: w/o explicit geometric primitives, –: beyond the scope (indoor scenes) of the method, †: leveraging geometric priors.
Table 2. Quantitative comparison of appearance rendering. We evaluate the rendering quality metrics across six diverse indoor and
outdoor datasets. Our method achieves state-of-the-art performance in most categories while significantly reducing the runtime (reported in
minutes).
Method
ScanNetV2
VR-NeRF
ScanNet++
Waymo
FAST-LIVO2
KITTI
Time
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
2DGS†
27.74
0.873
0.234
29.61
0.905
0.200
32.00
0.937
0.129
26.97
0.854
0.324
29.24
0.867
0.287
22.29
0.751
0.338
31.9
PGSR†
27.73
0.880
0.233
29.37
0.903
0.201
31.38
0.937
0.133
27.42
0.865
0.306
29.22
0.870
0.280
22.88
0.785
0.284
39.9
MeshSplatting†
25.64
0.830
0.351
25.23
0.819
0.352
27.71
0.876
0.294
23.10
0.781
0.424
25.78
0.789
0.397
17.35
0.551
0.499
24.6
MonoGS
22.17
0.806
0.542
15.30
0.583
0.655
17.08
0.708
0.632
19.06
0.744
0.639
19.80
0.694
0.649
14.56
0.489
0.767
8.3
S3PO-GS
24.37
0.829
0.476
24.00
0.810
0.371
23.34
0.820
0.444
25.33
0.821
0.395
24.99
0.776
0.419
19.23
0.622
0.430
24.5
OnTheFly-NVS 23.33
0.823
0.376
29.10
0.895
0.237
21.54
0.794
0.357
27.22
0.848
0.300
21.92
0.735
0.443
17.17
0.584
0.427
1.3
ARTDECO
28.44
0.877
0.232
30.02
0.911
0.230
31.64
0.941
0.140
26.59
0.869
0.305
32.86
0.926
0.210
22.99
0.777
0.282
6.9
Ours
28.83
0.882
0.222
32.59
0.933
0.168
31.91
0.941
0.133
29.24
0.887
0.278
33.97
0.938
0.180
23.82
0.793
0.253
7.4
†: leveraging geometric priors.
Table 3. Quantitative comparison of dense mesh reconstruction.
Method
ScanNet++
ScanNetV2
FAST-LIVO2
Ch-L2↓F-score↑Ch-L2↓F-score↑Ch-L2↓F-score↑
2DGS†
3.95
80.90
6.45
53.11
52.83
61.06
PGSR†
3.92
81.47
6.55
53.89
53.56
60.99
MeshSplatting†
9.24
46.30
11.05
31.22
61.03
51.61
ARTDECO
3.87
82.34
6.00
57.61
36.99
61.41
Ours
3.76
84.81
5.87
59.93
38.44
64.36
†: leveraging geometric priors.
Implementation Details.
Following standard novel view
synthesis practice, every eighth frame is held out for eval-
uation, which are excluded from the mapper while their
poses are optimized for evaluation. Following [20, 26], our
method, ARTDECO [20], and OnTheFly-NVS [26] perform
a 15k-iteration global optimization after the streaming stage,
whereas per-scene baselines are trained for 30k iterations.
More implementation details are provided in Appendix A.4.
4.2. Results Analysis
Geometry Results.
We first evaluate our method on pla-
nar reconstruction, comparing it with six baselines spanning
a diverse set of learnable scene representations, including
triangles, 3D Gaussians, surfels, rectangles, and implicit
embedding-based planar representations. Quantitative re-
sults in Tab. 1 show that our method consistently achieves
superior geometric accuracy, attaining the lowest Chamfer
Distance and highest F-score, while maintaining a compact
primitive count and the shortest training time. As shown
in Fig. 5, our hybrid representation preserves planar regu-
larity and sharp geometric features by explicitly modeling
planar structures with triangles. In contrast, rectangle-based
representations, despite their compactness, lack the flexibil-
ity to capture fine-grained geometry, limiting their ability
to model complex scene structures. Surfel-based methods,
which tightly couple geometry and appearance, often suffer
from appearance-induced distortions, resulting in uneven or
erroneous surfaces even when geometric priors are applied.
We also evaluate our method on dense mesh reconstruction,
with all meshes extracted via depth fusion for fair compar-
ison. As reported in Tab. 3, our method achieves higher
geometric accuracy while requiring less than 20% of the
training time compared to per-scene optimization methods.
7

<!-- page 8 -->
Triangles
ScanNetV2
ScanNet++
(top-down view)
PlanarSplatting
2DGS 
+ Geometric Priors
GT
Ours 
Rectangles
Mesh
Plane
PlanarSplatting
2DGS 
+ Geometric Priors
Ours 
Plane
Plane
Figure 5. Qualitative comparison of geometric reconstruction. We visualize planar reconstruction and geometric modeling across
different primitives, with 2DGS shown as dense mesh for comparison. Overall, our method preserves planar structures while capturing fine
geometric details.
Rendering Results.
Our method achieves state-of-the-art
rendering performance, outperforming both per-scene opti-
mization and streaming reconstruction baselines, as shown
in Tab. 2. In particular, it demonstrates clear advantages in
texture-less and low-light regions (Fig. 6). In these chal-
lenging scenes, per-scene optimization models are prone
to overfitting or Gaussian instability due to poor initializa-
tion, while streaming approaches frequently suffer from pose
drift that manifests as rendering artifacts. By contrast, our
approach mitigates these issues through a precise and con-
sistent geometric model. Furthermore, the integration of a
feed-forward model ensures robust pose estimation, further
driving the improvement in rendering fidelity.
4.3. Applications
Plane-Guided Camera Pose Optimization.
Most stream-
ing reconstruction frameworks decouple pose estimation
from mapping, preventing effective use of the global scene
map and often resulting in drift. We instead feed back the
reconstructed planar map to the frontend and refine cam-
era poses via online plane extraction and a point-to-plane
alignment loss, improving global consistency (Fig. 9). Due
to the geometric regularity and structural sparsity of planar
primitives, these constraints provide strong and stable geo-
metric supervision for pose estimation. Details are provided
in Appendix B.1.
Large Scale Scene Reconstruction.
Although our hybrid
representation is compact, large-scale reconstruction remains
challenging under limited GPU memory. We therefore adopt
a dynamic loading strategy that swaps primitive parame-
ters between the GPU and CPU, enabling our framework to
scale to large environments (Fig. 8). Additional details are
provided in the Appendix B.2.
Efficient Locomotion Strategy Training.
Our method
produces compact, simulation-ready scenes composed of
planar primitives. By preserving the geometric correctness
and consistency of large-scale structures, the reconstructed
environments provide reliable contact geometry for phys-
8

<!-- page 9 -->
Ours
ARTDECO
OntheFly NVS
S3POGS
GT
PGSR
2DGS
Figure 6. Qualitative comparison of appearance rendering. We evaluate our method against state-of-the-art approaches. White wireframes
highlight regions where our method excels, faithfully reconstructing fine structures and complete surface.
(a) walking in a room
(b) climbing the stairs
Figure 7. Locomotion. To demonstrate the utility of our geometric output as a robust simulation environment, we trained two motion
policies using Proximal Policy Optimization (PPO) within the Isaac Lab framework: (a) indoor walking with a Unitree H1 humanoid, and (b)
stair climbing with a Unitree A1 quadruped. These experiments validate that our reconstructed geometry provides a high-fidelity foundation
for reinforcement learning.
ical simulation. The resulting scenes are lightweight, en-
abling fast asset conversion and scalable training pipelines,
as shown in Fig. 7. Additional details are provided in the
Appendix B.3.
4.4. Ablation Studies
We conduct ablation studies to systematically evaluate the
contributions of our representation and framework design.
Representation Design.
We replace triangles with 2D
Gaussians to ablate their contribution. As shown in Fig. 10,
triangles offer two advantages: (i) higher-quality geometry
with sharp boundaries; and (ii) improved rendering, since
their clear boundaries cause them to be influenced by fewer
pixels than 2D Gaussians, which stabilizes parameter op-
timization. We further ablate the hybrid representation by
replacing it with unanchored neural Gaussians. As shown in
Tab. 4, the hybrid representation improves both geometric
accuracy and rendering quality. Moreover, the proposed rep-
resentation reduces redundancy and encourages Gaussians
to concentrate around the underlying surface, as shown in
Fig. 11.
Framework Design.
We conduct ablation studies on the
mapping module of our on-the-fly reconstruction framework.
Disabling spatial filtering substantially increases the num-
ber of primitives (+200% on ScanNetV2 and +245% on Scan-
9

<!-- page 10 -->
Triangles
Dense Mesh
# Images: 2200
PSNR: 33.60 dB
# Triangles: 100k
Time: 15min
60m
Figure 8. Large-scale indoor reconstruction. We captured over 2000 monocular images of an indoor corridor using a mobile phone.
Leveraging our dynamic loading strategy, our method achieves high-quality dense mesh reconstruction and rendering.
w/o Plane-Guided Camera 
Pose Optimization
w/ Plane-Guided Camera 
Pose Optimization
Figure 9. Effect of plane-guided camera pose optimization.
Feeding back planar map constraints into pose estimation effec-
tively reduces drift.
Triangles + Gaussians
Rendered Normal using Triangles/Surfels
Surfels + Gaussians
Opacity: 0.93
Opacity: 0.95
Opacity: 0.66
Opacity: 0.62
Rendered RGB using Gaussians
PSNR: 32.35
PSNR: 31.14
PSNR: 31.19
PSNR: 30.77
Triangles + Gaussians
Surfels + Gaussians
Figure 10. Ablation on triangle representation. Compared to
surfels, our representation produces clearer, opaque surfaces and
enables finer rendering details.
Net++), confirming its effectiveness in reducing redundancy.
As shown in Fig. 12, disabling the global map update im-
proves geometric consistency and, consequently, rendering
quality. More ablation results are provided in Appendix C.2.
(a) w/o Hybrid Structure (#GS: 630.6k)
0 cm
30 cm
(b) w/ Hybrid Structure (#GS: 126.5k)
Figure 11. Ablation on hybrid representation. Our design ef-
fectively reduces representation redundancy and mitigates the geo-
metric inconsistencies commonly observed in depth predicted by
feed-forward methods. The point clouds visualize the centers of
Gaussians.
(b) w/ Global Map Update
(a) w/o Global Map Update
Figure 12. Ablation on global map update. Our framework
effectively improves the global consistency.
5. Limitations
PLANING is a modular framework whose components can
benefit from future advances in scene representation and
rendering. Our current formulation inherits limitations from
the chosen primitives and scene assumptions. In particular,
neural Gaussian primitives are not well suited for modeling
semi-transparent or transparent objects, where unreliable ap-
pearance gradients may adversely affect geometry optimiza-
tion. Moreover, the framework focuses on surface modeling
10

<!-- page 11 -->
Table 4. Ablation studies on the ScanNetV2 dataset. We conduct
ablation studies on the hybrid representation and framework design,
evaluating performance across both geometric and appearance met-
rics.
Setting
Geometry
Rendering
# Primitives
Ch-L2↓F-score↑PSNR↑SSIM↑LPIPS↓(#Geo/#GS)
Ours
5.68
62.15
28.83
0.882
0.222
56.1k/222.2k
w/o triangles
5.90
59.85
28.44
0.876
0.232
52.8k/157.3k
w/o hybrid
6.06
57.54
28.48
0.877
0.231
-/621.5k
w/o spatial filtering
6.01
58.86
28.66
0.880
0.213 211.5k/625.7k
w/o global map update
6.20
56.00
28.33
0.877
0.229
55.3k/166.6k
−: w/o geometric primitives.
and does not explicitly handle sky or distant background
regions in outdoor scenes, which can lead to inconsistent
initialization and degraded appearance quality. Addressing
these limitations is a significant bonus in practice and left as
future work.
6. Conclusion
PLANING addresses a fundamental limitation of existing
streaming Gaussian-based reconstruction frameworks: the
absence of a robust and compact anchoring geometry that
does not compromise appearance modeling. By introducing
a loosely coupled triangle–Gaussian representation together
with a streaming-aware optimization framework, PLANING
decouples geometry from appearance while preserving high-
fidelity rendering. This design help resolve long-standing
issues of geometric drift, redundancy, and instability in on-
the-fly reconstruction that arise from conflicts between accu-
rate geometry and appearance modeling. PLANING enables
efficient, structurally robust streaming reconstruction, and
further showcases its potential for simulation-ready 3D scene
assets suitable for a wide range of downstream applications.
7. Acknowledgments
The authors gratefully acknowledge Guanghao Li, Kerui
Ren, and collaborators for their work on ARTDECO [20],
whose framework design served as a foundation for parts of
this work.
References
[1] Nathaniel Burgdorfer and Philippos Mordohai. Radiant trian-
gle soup with soft connectivity forces for 3d reconstruction
and novel view synthesis. arXiv preprint arXiv:2505.23642,
2025. 3
[2] Roberto Caldara and Sébastien Miellet.
i map: A novel
method for statistical fixation mapping of eye movement data.
Behavior research methods, 43(3):864–878, 2011. 3
[3] Carlos Campos, Richard Elvira, Juan J Gómez Rodríguez,
José MM Montiel, and Juan D Tardós. Orb-slam3: An accu-
rate open-source library for visual, visual–inertial, and mul-
timap slam. IEEE transactions on robotics, 37(6):1874–1890,
2021. 3
[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. IEEE
Transactions on Visualization and Computer Graphics, 2024.
6, 15
[5] Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and Hao
Wang. Outdoor monocular slam with global scale-consistent
3d gaussian pointmaps. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 26035–
26044, 2025. 2, 6
[6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5828–5839, 2017. 6, 16
[7] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin
Wang, and Weiwei Xu. High-quality surface reconstruction
using gaussian surfels. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 3
[8] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite.
In 2012 IEEE conference on computer vision and
pattern recognition, pages 3354–3361. IEEE, 2012. 6, 16
[9] Antoine Guédon and Vincent Lepetit. Sugar: Surface-aligned
gaussian splatting for efficient 3d mesh reconstruction and
high-quality mesh rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 5354–5363, 2024. 6
[10] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam. In European Conference on Computer Vision, pages
180–197. Springer, 2024. 3
[11] Jan Held, Sanghyun Son, Renaud Vandeghen, Daniel Rebain,
Matheus Gadelha, Yi Zhou, Anthony Cioppa, Ming C Lin,
Marc Van Droogenbroeck, and Andrea Tagliasacchi. Mesh-
splatting: Differentiable rendering with opaque meshes. arXiv
preprint arXiv:2512.06818, 2025. 3, 6, 15
[12] Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien
Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi,
Bernard Ghanem, and Marc Van Droogenbroeck. 3d convex
splatting: Radiance field rendering with 3d smooth convexes.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 21360–21369, 2025. 3, 4
[13] SA Hojjatoleslami and Josef Kittler. Region growing: a new
approach. IEEE Transactions on Image processing, 7(7):
1079–1084, 1998. 15
[14] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 3, 4, 6, 14, 15
[15] Changjian Jiang, Kerui Ren, Linning Xu, Jiong Chen, Jiang-
miao Pang, Yu Zhang, Bo Dai, and Mulin Yu. Halogs: Loose
coupling of compact geometry and gaussian splats for 3d
scenes. arXiv preprint arXiv:2505.20267, 2025. 3
[16] Lihan Jiang, Kerui Ren, Mulin Yu, Linning Xu, Junting Dong,
Tao Lu, Feng Zhao, Dahua Lin, and Bo Dai. Horizon-gs:
11

<!-- page 12 -->
Unified 3d gaussian splatting for large-scale aerial-to-ground
scenes. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 26789–26799, 2025. 3
[17] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 2, 3
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 2, 3,
4
[19] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 5, 6,
15
[20] Guanghao Li, Kerui Ren, Linning Xu, Zhewen Zheng,
Changjian Jiang, Xin Gao, Bo Dai, Jian Pu, Mulin Yu, and
Jiangmiao Pang. Artdeco: Towards efficient and high-fidelity
on-the-fly 3d reconstruction with structured scene representa-
tion. arXiv preprint arXiv:2510.08551, 2025. 2, 5, 6, 7, 11,
14, 15
[21] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu
Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything
3: Recovering the visual space from any views. arXiv preprint
arXiv:2511.10647, 2025. 3
[22] William E Lorensen and Harvey E Cline. Marching cubes: A
high resolution 3d surface construction algorithm. In Seminal
graphics: pioneering efforts that shaped the field, pages 347–
353. 1998. 3
[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang,
Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians
for view-adaptive rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 20654–20664, 2024. 2, 5, 6
[24] Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu,
Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi.
3dgsr: Implicit surface reconstruction with 3d gaussian splat-
ting. ACM Transactions on Graphics (TOG), 43(6):1–12,
2024. 3
[25] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18039–18048, 2024. 2, 3, 6
[26] Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bern-
hard Kerbl, and George Drettakis. On-the-fly reconstruction
for large-scale novel view synthesis from unposed images.
ACM Transactions on Graphics (TOG), 44(4):1–14, 2025. 2,
3, 5, 6, 7
[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
3
[28] Raul Mur-Artal and Juan D Tardós. Orb-slam2: An open-
source slam system for monocular, stereo, and rgb-d cameras.
IEEE transactions on robotics, 33(5):1255–1262, 2017. 3
[29] Riku Murai, Eric Dexheimer, and Andrew J Davison. Mast3r-
slam: Real-time dense slam with 3d reconstruction priors. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16695–16705, 2025. 3, 5
[30] Zhenghao Qi, Shenghai Yuan, Fen Liu, Haozhi Cao, Tianchen
Deng, Jianfei Yang, and Lihua Xie. Air-embodied: An effi-
cient active 3dgs-based interaction and reconstruction frame-
work with embodied large language model. arXiv preprint
arXiv:2409.16019, 2024. 2
[31] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A
robust and versatile monocular visual-inertial state estimator.
IEEE transactions on robotics, 34(4):1004–1020, 2018. 3
[32] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 3
[33] Xuqian Ren, Matias Turkulainen, Jiepeng Wang, Otto
Seiskari, Iaroslav Melekhov, Juho Kannala, and Esa Rahtu.
Ags-mesh: Adaptive gaussian splatting and meshing with
geometric priors for indoor room reconstruction using smart-
phones. In International Conference on 3D Vision (3DV),
2025. 15
[34] Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm,
and Marc Pollefeys. Pixelwise view selection for unstruc-
tured multi-view stereo. In European conference on computer
vision, pages 501–518. Springer, 2016. 3
[35] Christian Sigg, Tim Weyrich, Mario Botsch, and Markus H
Gross. Gpu-based ray-casting of quadratic surfaces. In PBG@
SIGGRAPH, pages 59–65, 2006. 14
[36] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, et al. Scalability in perception
for autonomous driving: Waymo open dataset. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 2446–2454, 2020. 6
[37] Bin Tan, Rui Yu, Yujun Shen, and Nan Xue. Planarsplat-
ting: Accurate planar surface reconstruction in 3 minutes. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 1190–1199, 2025. 3, 6
[38] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt:
Visual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 3
[39] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua
Shen, and Tong He. π3: Permutation-equivariant visual ge-
ometry learning. arXiv preprint arXiv:2507.13347, 2025. 3,
5
[40] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[41] Jamie Watson, Filippo Aleotti, Mohamed Sayed, Zawar
Qureshi, Oisin Mac Aodha, Gabriel Brostow, Michael Firman,
and Sara Vicente. Airplanes: Accurate plane estimation via
3d-consistent embeddings. In Proceedings of the IEEE/CVF
12

<!-- page 13 -->
Conference on Computer Vision and Pattern Recognition,
pages 5270–5280, 2024. 6
[42] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia,
Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo
Porzi, Peter Kontschieder, Aljaž Božiˇc, et al. Vr-nerf: High-
fidelity virtualized walkable spaces. In SIGGRAPH Asia 2023
Conference Papers, pages 1–12, 2023. 6, 16
[43] Yandan Yang, Baoxiong Jia, Peiyuan Zhi, and Siyuan Huang.
Physcene: Physically interactable 3d scene synthesis for em-
bodied ai.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 16262–
16272, 2024. 2
[44] Keyang Ye, Tianjia Shao, and Kun Zhou. When gaussian
meets surfel: Ultra-fast high-fidelity radiance field rendering.
ACM Transactions on Graphics (TOG), 44(4):1–15, 2025. 3
[45] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d
indoor scenes. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 12–22, 2023. 6, 16
[46] Mulin Yu and Florent Lafarge. Finding good configurations of
planar primitives in unorganized point clouds. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 6367–6376, 2022. 14
[47] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli,
and Bo Dai. Gsdf: 3dgs meets sdf for improved neural ren-
dering and reconstruction. Advances in Neural Information
Processing Systems, 37:129507–129530, 2024. 3
[48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages
586–595, 2018. 6
[49] Wei Zhang, Tiecheng Sun, Sen Wang, Qing Cheng, and Nor-
bert Haala. Hi-slam: Monocular real-time dense mapping
with hybrid implicit fields. IEEE Robotics and Automation
Letters, 9(2):1548–1555, 2023. 3
[50] Wei Zhang, Qing Cheng, David Skuddis, Niclas Zeller, Daniel
Cremers, and Norbert Haala. Hi-slam2: Geometry-aware
gaussian slam for fast monocular scene reconstruction. IEEE
Transactions on Robotics, 41:6478–6493, 2025. 2, 3
[51] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d instant
reconstruction. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 3727–3737, 2023. 3
[52] Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xi-
aojun Xiang, and Shuhan Shen. Quadratic gaussian splatting:
High quality surface reconstruction with second-order geo-
metric primitives. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 28260–28270,
2025. 3
[53] Chunran Zheng, Wei Xu, Zuhao Zou, Tong Hua, Chongjian
Yuan, Dongjiao He, Bingyang Zhou, Zheng Liu, Jiarong Lin,
Fangcheng Zhu, et al. Fast-livo2: Fast, direct lidar-inertial-
visual odometry. IEEE Transactions on Robotics, 2024. 6
[54] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun
Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys.
Nice-slam: Neural implicit scalable encoding for slam. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 12786–12796, 2022. 3
13

<!-- page 14 -->
The following appendices provide additional technical
details and experimental results that support the main find-
ings of this work. They include descriptions of the technical
details of our method (Sec. A), application details and im-
plementation (Sec. B), and additional experimental results
(Sec. C).
A. Technical Details
A.1. Differentiable Triangle Rasterizer
To enable unbiased depth and normal rendering with triangle
primitives, we adopt an explicit ray–primitive intersection
strategy [35], following 2DGS [14]. We define the transfor-
mation from a triangle’s local coordinate system to world
space as
H =
sutu
svtv
0
µ
0
0
0
1

,
(11)
where µ, su, tu, sv, and tv follow the definition of the local
triangle frame in Eq. 1.
When combined with the edge-preserving contribution
function (Eq. 2), this formulation leads to two practical chal-
lenges: (i) inaccurate depth sorting for large triangles
whose barycenters deviate from true ray–triangle intersec-
tions; and (ii) incorrect visibility estimation when triangle
barycenters are occluded while portions of the triangle re-
main visible.
To address these issues, we propose a subdivision-aware
forward rendering pipeline that integrates adaptive trian-
gle subdivision for robust depth sorting and a vertex-based
visibility criterion for accurate occlusion handling. The com-
plete procedure is summarized in Algorithm 1.
A.2. Training Strategy
In our streaming reconstruction system, we adopt a staged
training strategy to balance efficiency and reconstruction
quality, following [20]. Specifically, when a keyframe is
encountered, new primitives are initialized and the scene is
optimized for M iterations (set to 20 in our implementation),
while common frames are optimized for only M/2 iterations
without adding new Gaussians. Training frames are sampled
with a probability of 0.2 from the current frame and 0.8 from
past frames to mitigate local overfitting. After processing
the sequence in a streaming fashion, a global optimization
is performed over all frames, prioritizing those with fewer
prior updates.
A.3. Planar Primitive Extraction
Planar primitives provide an efficient structural abstraction of
the scene and can be directly leveraged in downstream tasks,
such as robot local motion training. To extract these planes,
we adopt a coarse-to-fine strategy based on GoCoPP [46],
where the method is applied iteratively with progressively
ALGORITHM 1: Subdivision-aware Forward Ren-
dering
Input: Triangle soup T , camera pose W, screen
resolution
Output: Rendered depth and normal maps
Triangle Preprocessing:
Initialize visible triangle set Tv ←∅
foreach triangle t ∈T do
if at least one vertex of t is visible then
Construct local triangle frame and
transformation H ;
Subdivide t recursively until all edges are
shorter than threshold ϵ ;
Assign parent triangle ID to all subdivision
triangles ;
Add subdivision triangles to Tv ;
end
end
Subdivision Processing:
foreach subdivision triangle ts ∈Tv do
if at least one vertex of ts is visible then
Project vertices to image plane ;
Determine overlapped tiles ;
Compute view-space depth using barycenter
of ts ;
Generate sorting key (depth, tile ID) ;
end
end
Depth Sorting:
Perform GPU-based radix sort on all subdivision
triangles using sorting keys ;
Rendering:
foreach pixel x = (x, y)T do
Define the camera ray using two orthogonal
homogeneous planes;
Transform rays into local triangle coordinates
using (WH)T ;
Compute ray–triangle intersection ˆx on the
original triangle ;
Evaluate rendering contribution using Eq. 2 ;
end
Render depth and normal images following Eq. 4 in
the main text ;
finer parameters to detect smaller planes from the residual
points remaining after coarser planes are extracted.
A.4. More Implementation Details
For our method, we set Kmin = 4 and Kmax = 8, with loss
weights λd = 10.0, λn = 3.0, λo = 0.2, λc = 0.2, and λs =
0.01. For dense mesh extraction, our method fuses triangle-
14

<!-- page 15 -->
100
150
200
250
300
350
400
28.5
28.6
28.7
28.8
Number of Gaussians (k)
PSNR ↑
PSNR
SSIM
0.8800
0.8805
0.8810
0.8815
0.8820
SSIM ↑
Figure 13. Effect of the number of Gaussians on rendering
quality. PSNR and SSIM improve initially and then saturate as
Gaussian count increases.
rendered depth maps into meshes using TSDF, following the
procedure in 2DGS [14]. For per-scene methods, geometric
priors are incorporated according to the parameterization in
AGS-Mesh [33], which provides a comprehensive study of
geometric prior integration. For planar primitive extraction,
since baseline methods, including 2DGS [14], PGSR [4],
MeshSplatting [11], and ARTDECO [20], typically output
dense meshes, we extract multi-level planar shapes from their
results using the same strategy and parameters applied to
our method to ensure a fair comparison. All experiments are
performed on an Intel Core i9-14900K CPU and an NVIDIA
RTX 4090 GPU.
B. Application Details
B.1. Plane-Guided Camera Pose Optimization
In our streaming reconstruction system, we optionally feed
back the reconstructed planar map to the frontend to refine
camera poses via a point-to-plane alignment loss, improving
global consistency. Specifically, in the mapper, we main-
tain a voxel map using a spatial hash to manage triangle
primitives. During training, planar primitives are regularly
extracted via region growing [13]. In our implementation,
the voxel size is set to 3 cm, and plane extraction is per-
formed every 10 frames. The extracted plane parameters and
associated voxel keys are then shared with the frontend.
In the frontend, high-confidence points predicted by
MASt3R [19] are associated with the planar map via the
voxel grid. For each point p and its corresponding plane, we
adopt a simple yet effective point-to-plane alignment loss:
Lp = ∥(p −c) · n∥1,
(12)
where n and c denote the plane’s normal and center, respec-
tively.
Table 5. Comparison of scene import and conversion time under
non-headless (GUI-based) and headless Isaac Sim pipelines. Both
settings perform the same sequence of operations, including mesh
import, collision geometry construction, and USD packaging, but
differ in execution mode.
Setting
Non-headless (s) Headless (s) # Primitives
Ours (Plane)
89.73
5.27
17k
2DGS
–
657
277k
2DGS∗
120.00
37.21
17k
−: impractical runtime (> 30,min), *: 16× mesh simplification.
B.2. Large Scale Scene Reconstruction
To enable large-scale scene reconstruction and alleviate GPU
memory limitations, we introduce a dynamic loading strategy
that swaps primitive parameters between the GPU and CPU.
Specifically, we periodically evaluate the projected scale of
each neural Gaussian on the most recent image plane. A
neural Gaussian is marked as invisible if its projected scale
is smaller than a pixel, and a triangle is marked as invisible
when all its associated neural Gaussians are invisible. In-
visible triangles and their corresponding Gaussians are then
offloaded from the GPU to the CPU. Upon detecting a loop
closure in the Backend module, we reload the primitives ini-
tialized from the associated images from the CPU back to the
GPU. This dynamic loading strategy allows our framework
to efficiently scale to large environments.
B.3. Locomotion Strategy Training
Beyond visual fidelity, our hybrid representation facilitates
downstream embodied tasks by providing the geometric
consistency essential for stable contact dynamics in physics-
based locomotion. Unlike appearance-driven methods that
generate redundant primitives, our approach prioritizes large-
scale, load-bearing structures such as floors and walls. This
results in a highly compact representation that significantly
reduces triangle counts while preserving the structural in-
tegrity required for high-fidelity simulation and reinforce-
ment learning.
While 2DGS [14] serves as a strong baseline, it produces
highly complex meshes that incur significant preprocessing
overhead in Isaac Sim. In practice, a 2DGS scene (277k
faces) requires over 30 minutes for standard import and
conversion. By contrast, our lower mesh complexity cir-
cumvents these bottlenecks, consistently leading to faster
processing in both standard and headless (convert_mesh)
pipelines. Quantitative comparisons are reported in Tab. 5.
To evaluate the impact of geometric reconstruction qual-
ity on policy learning, we conduct locomotion experiments
in Isaac Lab using the Unitree H1 humanoid and the Uni-
tree A1 quadruped. We first consider a setting without a
height scanner to enforce reliance on the physical correct-
15

<!-- page 16 -->
Triangles
ScanNetV2
ScanNet++
(top-down view)
PlanarSplatting
2DGS 
+ Geometric Priors
GT
Ours 
Rectangles
Mesh
Plane
PlanarSplatting
2DGS 
+ Geometric Priors
Ours 
Plane
Plane
Figure 14. More geometric comparison results. We visualize planar reconstruction and geometric modeling across different primitives,
with 2DGS shown as dense mesh for comparison. Overall, our method preserves planar structures while capturing fine geometric details.
ness of the simulated geometry. Under this configuration,
policies trained in 2DGS scenes after aggressive mesh sim-
plification fail to converge due to degraded planar geometry,
whereas policies trained in our reconstructed scenes consis-
tently achieve stable locomotion under identical observation
settings. Overall, our method enables the construction of
simulation environments that are both geometrically accurate
and compact. By reducing the real-to-sim gap, our approach
provides a practical foundation for efficient downstream lo-
comotion policy training and deployment.
C. Supplementary Experiments
C.1. Supplementary Comparison Experiments
In Fig. 14, we provide additional reconstruction comparisons
on the ScanNet++ [45] and ScanNetV2 [6] datasets, showing
Table 6. Ablation studies on the ScanNet++ dataset. Our abla-
tions are divided into two categories: representation design and
framework design.
Setting
Geometry
Rendering
# Primitives
Chamfer↓F-score↑PSNR↑SSIM↑LPIPS↓(#Geo/#GS)
Ours
3.53
86.88
31.91
0.941
0.133
61.6k/291.0k
w/o triangles
3.63
84.91
31.05
0.932
0.150
39.4k/201.9k
w/o hybrid
3.81
83.01
31.60
0.940
0.140
-/478.3k
w/o spatial filtering
3.71
84.94
31.95
0.942
0.126 233.6k/981.6k
w/o global map update
3.59
85.07
31.78
0.941
0.131
61.5k/291.4k
−: w/o geometric primitives.
that our method more faithfully preserves the scene’s geomet-
ric structures. Fig. 15 presents rendering quality comparisons
on the KITTI [8] and VR-NeRF [42] datasets, demonstrat-
16

<!-- page 17 -->
OntheFly NVS
S3POGS
PGSR
2DGS
Ours
ARTDECO
Mesh Splatting
GT
MonoGS
Ours
ARTDECO
OntheFlyNVS
S3POGS
GT
PGSR
2DGS
Figure 15. More rendering comparison results. White boxes highlight artifacts and fine-grained details from baseline methods. Our
approach yields significantly sharper results on intricate structures, such as text, while achieving superior overall rendering quality.
ing our method’s robustness and applicability across diverse
scenarios, including both indoor and outdoor environments.
C.2. Supplementary Ablation Studies
We also conduct ablation studies on ScanNet++, using the
same experimental setup as in the main text. As shown in
Tab. 6, our representation design achieves the best perfor-
mance in both rendering quality and geometric accuracy.
Regarding system design, by enabling spatial filtering, our
method can achieve higher geometric accuracy and compa-
rable rendering quality with less than one-third of the primi-
tives, highlighting the efficiency of our framework. Further-
more, we investigate the effect of the number of Gaussians
on rendering quality. As shown in Fig. 13, PSNR and SSIM
initially improve with increasing Gaussian count and then
saturate.
17
