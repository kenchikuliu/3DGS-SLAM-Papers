<!-- page 1 -->
Graphical Abstract
Time-Archival Camera Virtualization for Sports and Visual Perfor-
mances
Yunxiao Zhang, William Stone, Suryansh Kumar†
(a) Football match broadcast from virtual 
camera from a distant viewpoint
(b) Scene rendering corresponding to (a) data 
from a different viewpoint after a certain time
(c) Football match broadcast from virtual 
camera from a near viewpoint
(d) Football match broadcast from virtual 
camera from a top viewpoint
Figure 1:
(a)-(d) Camera virtualization for football sports showing the image
rendering from camera placed at different distances from the subject(s), i.e., (a)
far-distance viewpoint, (c) near-distance viewpoint (d) top viewpoint.
Camera virtualization enables photorealistic rendering of dynamic scenes
†Corresponding Author
arXiv:2602.15181v1  [cs.CV]  16 Feb 2026

<!-- page 2 -->
from novel viewpoints using limited static camera setups, significantly benefit-
ing visual entertainment, live performances, and sports broadcasting. Current
dynamic-scene rendering methods, particularly those based on 3D Gaussian
Splatting (3DGS), enable real-time image synthesis but often depend on high-
quality initial 3D point clouds, making them unsuitable for time-archival for
our targeted applications. To overcome these limitations, we revisit the neural
implicit scene representation and propose a neural volume rendering frame-
work grounded in multiview projective geometry. We argue that a dynamic
scene observed under a well-constrained synchronized multiview setup, typical
in sports and visual performance scenarios, is already strongly constrained by
geometry, and we may not need a temporally coupled constraint or 3d point
cloud initialization. Our method exploits such a default design choice and
represents dynamic scenes using synchronized neural representations across
multiple camera views, naturally supporting efficient temporal archival and
retrospective novel-view synthesis. Experiments on standard benchmarks
and newly introduced dynamic-scene datasets demonstrate that our method
achieves superior rendering quality compared to state-of-the-art 4DGS ap-
proaches, establishing a new benchmark for camera virtualization in dynamic
visual media. This work also advances compact plenoptic scene modeling,
enabling comprehensive archival and replay of dynamic events.
2

<!-- page 3 -->
Highlights
Time-Archival Camera Virtualization for Sports and Visual Perfor-
mances
Yunxiao Zhang, William Stone, Suryansh Kumar†
• Implicit neural scene representation learning to compactly store tempo-
ral instances, allowing users to “rewind" and synthesize novel views of
past and current moments.
• An approach that represents a dynamic scene using simple neural
network model over discrete time step with excellent image-rendering
quality compared to the state-of-the-art approaches.
• While alternative representations (such as 3DGS) could, in principle,
support similar functionality, we show that, under the deliberate design
choice of a synchronized multiview camera setup typical of sports and vi-
sual performance capture, the problem is geometrically well-constrained.
To this end, the proposed approach offers a better alternative for time-
archival and retrospective novel-view synthesis.
• Presents a new synthetic dataset targeted at visual performance and
sports applications and benchmarks the related state-of-the-art methods
on this dataset.
†Corresponding Author

<!-- page 4 -->
Time-Archival Camera Virtualization for Sports and
Visual Performances
Yunxiao Zhanga,b,e, William Stoned,e, Suryansh Kumar†
a,b,c,d,e
aVisual and Spatial AI Lab, Visual Computing & Computational Media (VCCM) Section,
bCollege of Performance, Visualization, and Fine Arts (PVFA),
cDepartment of Electrical and Computer Engineering (ECEN),
dDepartment of Computer Science and Engineering (CSCE),
eTexas A&M University, College Station, Texas, USA
Abstract
Camera virtualization—an emerging solution to novel view synthesis—holds
transformative potential for visual entertainment, live performances, and
sports broadcasting by enabling the generation of photorealistic images from
novel viewpoints using images from a limited set of calibrated multiple static
physical cameras. Despite recent advances, achieving spatially and temporally
coherent and photorealistic rendering of dynamic scenes with efficient time-
archival capabilities, particularly in fast-paced sports and stage performances,
remains challenging for existing approaches. Recent methods based on 3D
Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-
synthesis results. Yet, they are hindered by their dependence on accurate 3D
point clouds from the structure-from-motion method and their inability to
handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps,
articulations, sudden player-to-player transitions). Moreover, independent
motions of multiple subjects can break the Gaussian-tracking assumptions
†Corresponding Author

<!-- page 5 -->
commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This
paper advocates reconsidering a neural volume rendering formulation for
camera virtualization and efficient time-archival capabilities, making it useful
for sports broadcasting and related applications. By modeling a dynamic
scene as rigid transformations across multiple synchronized camera views at
a given time, our method performs neural representation learning, providing
enhanced visual rendering quality at test time. A key contribution of our
approach is its support for time-archival, i.e., users can revisit any past
temporal instance of a dynamic scene and can perform novel view synthesis,
enabling retrospective rendering for replay, analysis, and archival of live
events—a functionality absent in existing neural rendering approaches and
novel view synthesis methods for dynamic scenes. While, in principle, dynamic
3DGS approaches can also perform time-archival, however, it will require
either a multi-view structure-from-motion (SfM) point cloud to be stored at
every time step or some form of additional multi-body temporal modeling
constraint—both of which are complex, computationally expensive, and could
be memory-intensive. We argue that a dynamic scene observed under a
well-constrained synchronized multiview setup—typical in sports and visual
performance scenarios, is already strongly constrained by geometry, and we
may not need a temporally coupled constraint or 3d point cloud initialization.
Extensive experiment and ablations on established benchmarks and our newly
proposed dynamic scene datasets demonstrate that our method surpasses
4DGS-based baselines in rendered image quality and other performance metric
for time-archival view-synthesis for a dynamic scene, thus setting a new
standard for virtual camera systems in dynamic visual media. Furthermore,
2

<!-- page 6 -->
our approach could be an encouraging step towards compactly modeling the
plenoptic function, allowing for time-archival of a long video sequence.
Keywords:
Camera Virtualization, Time-Archival Representation, Neural
Representation Learning, Multiview Geometry, Dynamic Scenes, Neural
Image Based Rendering, Multi-layer Perceptron.
1. Introduction
Sporting events and visual performances are inherently dynamic in na-
ture. More importantly, the dynamic subjects are of significant interest here
compared to other parts of the scene. Traditional methods of capturing such
events typically involve physical multi-view camera setups constrained by
fixed viewpoints, limited spatial coverage, and logistical complexities. These
constraints restrict the freedom to fully exploit the rich visual dynamics
intrinsic to sports and visual performance such as dance events, and oth-
ers. In such applications, we can greatly enhance viewer engagement by
allowing the user to watch the event from novel viewpoints. Furthermore,
such an application must support time-archival, i.e., the user can revisit the
scene in the past and can observe its dynamics from a novel viewpoint. We
term this problem here as “camera virtualization”. Although this problem
closely aligns with dynamic scene novel view synthesis, where the goal is
to render a photorealistic image of dynamic subject(s) from a novel camera
viewpoint at test time, camera virtualization must allow for time-archival.
By synthesizing novel visual perspectives of a dynamic scene from limited
sets of physical cameras, camera virtualization enables unprecedented flex-
3

<!-- page 7 -->
ibility in visual content generation. To this end, recent advances in neural
rendering techniques such as Neural Radiance Fields (NeRF) [1] and 3D
Gaussian splatting (GS) [2] and its extensions to dynamic scenes [3, 4] could
provide a promising solution. Yet, significant technical challenges remain
unresolved despite excellent progress, particularly regarding dynamic scene
rendering quality, spatial-temporal coherence, and time-archival (compact
and memory efficient representation of the dynamic scene for replay from
novel viewpoints). This paper investigates these challenges and introduces an
approach to camera virtualization, explicitly focused on visual performance
and sports applications.
0.1
0.2
0.3
0.4
0.5
LPIPS
10
15
20
25
30
PSNR
Ours
D-NeRF
Dynamic Gaussian
4D Gaussian
Spacetime Gaussian
Ours
Figure 2: Quantitative comparison of
the image rendering quality on the
proposed dynamic scene dataset with
state.
Since time-archival is one of the prime
motivation of this paper, we advocate
that despite the 3D-GS [2] and its exten-
sion to the dynamic scene such as 4D-GS
[3] and Spacetime-GS [4] being encourag-
ing in providing real-time rendering per-
formance, novel view synthesis solution
specifically for dynamic scenes should not
wholly rely on GS based approaches. In-
stead, exploiting physical camera setup
configuration (such as synchronization,
view angles, etc.), multiview rigidity con-
straint, and benefits of neural scene representation learning in image rendering
must not be ignored. Not that the rendering speed will not suffer a bit due to
such a take, yet the reason for such research exploration lies on the fact the
4

<!-- page 8 -->
performance of 3DGS based approaches observed to rely heavily on the accu-
racy of initial 3D points recovered from structure from motion pipeline such
as COLMAP [5] or active sensing modality such as LiDAR [6] or RGB-D [7].
On the contrary, popular NeRF-based methods [1, 8] do not rely on explicit
3D points for image-based rendering at the same time could help memorize
the scene representation compactly over time while also providing accurate
image rendering quality. Let’s understand how 3d point cloud dependent
representation such as 3DGS and related approaches leads to several practical
challenges in dynamic scene time-archival for applications such as sports.
• Storage Efficiency: A 3DGS model if used for time-archival in our sce-
narios will typically requires 1-5 million Gaussians per scene, with each
Gaussian storing position, covariance, opacity, and dozens of spherical
harmonics coefficients. This results in 200-300MB per time step. Thus,
for long sequences, per-time-step 3DGS requires tens of gigabytes for
storage (e.g., 100 frames: 20-30 GB). In contrast, our proposition for
using implicit per-frame radiance field needs 12.7M parameters (ap-
prox. 25-50MB). Therefore, the memory storage usage per time step is
10–20x smaller, allowing time-archival over long sequences. This makes
the proposed formulation significantly more scalable while keeping the
memory footprint predictable and independent of scene complexity.
• For a dynamic scene, if 3DGS is applied for time-archival, it would, in
principle, require either a multi-view structure-from-motion (SfM) point
cloud at every time step or some form of additional multi-body tempo-
ral modeling constraint—both of which are complex, computationally
expensive, and, frankly, memory-intensive.
5

<!-- page 9 -->
• Exact vs. approximated state retrieval: 3DGS extensions to dynamic
scene such as 4DGS and others typically rely on tracking or deforming
a shared set of Gaussians. Even with per–time-step Gaussian storage,
parameter reuse and propagation can lead to drift or approximation
errors, especially at time steps distant from keyframes. Our method on
the other hand stores a complete radiance function Ft independently for
each time step, enabling exact, drift-free reconstruction for each time
instance.
• Independent per-frame optimization avoids compounding errors, i.e.,
Dynamic 3DGS updates Gaussians sequentially, meaning errors may
accumulate across time. On the contrary, our method avoids this entirely
by reinitializing the optimization per timestep. Yet, our method stores
dense radiance fields that are temporally independent—an advantage
in multi-subject, extreme-motion sports environments.
This brings us to the point of justifying the above notions and why it makes
sense to revisit neural rendering approaches for sports and visual performance
applications, which is the paper’s primary focus. Firstly, in such applications,
we are often provided with a multiview synchronized static camera setup.
Therefore, any common dynamic subject(s) between viewpoints at a given
time will be rigidly related. This allows us to model a dynamic scene at a
given time using an implicit neural representation without the requirement
of 3D points as input. Such an approach is easy, reliable, and favorably fast
with inherent time-archival capability, i.e., time indexing of neural implicit
scene representation.
6

<!-- page 10 -->
By unifying geometric ideas with neural rendering, our approach can trans-
form applications like sports broadcasting, enabling users to analyze scenes
from optimal angles, or theatrical archives preserving performances as 4D
experiences. This work sets a new standard for dynamic scene virtualization
with inherent time-archival capability (see Figure 2 for quantitative result
comparison with current state-of-the-art methods). So, conceptually, our
approach can be thought of as a way to compactly model plenoptic function
Φ [9, 10]
Φ(x, Ωθ, Ωϕ, λ, t),
where, x = (x, y, z)T ∈R3, Ωθ ∈[0, π], Ωϕ ∈[0, 2π). (1)
Here, λ is the wavelength which we assume as a constant, and t for time.
Contrary to [11, 12], we are interested in modeling the notion of Φ tailored
for a dynamic scene, where subject(s) moves through time. Here, x is a ray
position, (Ωθ, Ωϕ) is the ray direction. So, if we could model Φ, we could
visually reconstruct every possible view, at every moment, from every position.
Fig. 3 shows a couple of examples use of our approach for sports. In this
paper, we make the following contributions:
1. The paper proposes the notion and solution to camera virtualization
with time-archival capabilities. By revisiting the implicit neural scene
representation learning to compactly store temporal instances of a dynamic
scene, this paper allows users to “rewind” and synthesize novel views of
past and current moments. Consequently, enabling a key step in compact
modeling of the plenoptic function [9, 10].
2. An approach to represent dynamic scene using simple neural network model
over discrete time step with better image-rendering quality compared to
7

<!-- page 11 -->
the state-of-the-art approaches such as [3, 4], is proposed. Furthermore,
the approach does not need explicit 3d points to model dynamic subjects’
large, non-rigid, rapid motions across subjects in synchronized views (e.g.,
flips, jumps, articulations, sudden player-to-player transitions).
3. While alternative representations (such as 3DGS) could, in principle,
support similar functionality, we show that, under the deliberate design
choice of a synchronized multiview camera setup typical of sports and visual
performance capture, the problem is geometrically well-constrained. To
this end, the proposed approach offers a better alternative for time-archival
and retrospective novel-view synthesis.
4. The paper introduces a new synthetic dataset targeted at visual perfor-
mance and sports applications and benchmarks the related state-of-the-art
methods on this dataset.
2. Related Work
Image-based rendering (IBR), now popularly re-branded as novel view
synthesis, has long been a central problem in computer graphics and computer
vision. This research area has a rich body of classical literature [13, 14]. For
this paper, we limit our discussion to recent papers that are most relevant to
our work. For broad literature survey, we refer the readers to [15, 16, 17].
(i) Neural Scene Representations. Neural representations for a 3D scene
have significantly advanced novel view synthesis by learning radiance fields
from sparse multi-view observations. Early methods map learnable features
to volumes [18], texture maps [19, 20], or point clouds [21]. Not long ago,
NeRF [1] introduced differentiable volume rendering via MLPs without the
8

<!-- page 12 -->
a) A typical football match dynamic scene
VC1
VC2
VC3
VC4
Virtual Camera1 (VC1)
Virtual Camera2 (VC2)
Virtual Camera3 (VC3)
Virtual Camera4 (VC4)
b) Corresponding virtual cameras images 
c) A typical tennis match dynamic scene
VC1
VC3
VC2
VC4
Virtual Camera1 (VC1)
Virtual Camera2 (VC2)
Virtual Camera3 (VC3)
Virtual Camera4 (VC4)
d) Corresponding virtual camera images
Figure 3: Overall setup for our application. A typical multiview synchronized camera
setup installed in a sports scene for broadcasting (shown in black). The virtual cameras to
inspect the scene are shown in red color for time-archival or broadcast from users interest
view points. a)-b)Top-row: A typical football sports scene and image-rendering from
virtual cameras. c)-d)Bottom-row: A typical tennis sports match scene and respective
image-rendering from virtual camera view points.
need for explicit scene geometry. This marks NeRF [22] as a foundational
milestone in neural scene representation. Extending NeRF, numerous works
were proposed to improve computational efficiency [8] and expressiveness
[23, 24]: some optimize ray sampling to reduce point queries [25, 26], while
others adopt light field-based representations [27, 28, 29, 30, 31, 32, 33] and
multi-scale representation [34, 35, 36, 37]. Another direction extending NeRF
introduced explicit, localized encoding for time efficiency [38, 39, 40, 41, 42,
43, 8, 44, 45, 46, 47, 48], including Instant-NGP [8] with multiresolution hash
9

<!-- page 13 -->
grids and TensoRF [39] using tensor decomposition for compactness.
(ii) 3D Gaussian Splatting. A recent alternative to neural implicit models
is 3D Gaussian Splatting (3DGS) [2]. It represents scenes via anisotropic
3D Gaussians defined by position, covariance, opacity, and view-dependent
color. These are rendered directly through a forward rasterization pipeline,
achieving real-time synthesis with competitive image quality. Yet, this real-
time rendering speed comes with trade-offs, i.e., 3DGS lacks a learnable
memory structure and temporal indexing, making it unsuitable for time-
archival tasks where revisiting previous moments is critical. Moreover, it
heavily depends on accurate 3D point initialization from SfM [5], which has
well-known limitations dealing with complex dynamic scenes [49].
(iii) Novel View Synthesis for Dynamic Scenes. Extending static repre-
sentations to dynamic settings has attracted considerable interest. Broxton et
al. [50] introduce multi-sphere images bootstrapped into layered meshes. [51]
decompose static and dynamic regions for screen-space video manipulation.
Other methods incorporate view, time, or lighting parameters into learned
2D encodings [52], or leverage multi-sphere models for depth and occlusion
resolution in 360-degree video [53]. Lin et al. [54, 55] introduce 3D mask vol-
umes to enforce temporal coherence. Neural Volumes [56] and its extensions
decode 3D fields using encoder-decoder architectures.
Dynamic NeRF-style models extend implicit neural representations to
time-varying scenes [57, 58, 59, 60, 61, 62, 63, 64, 65, 25, 66, 67, 68, 69, 70].
DyNeRF [57] uses time-conditioned latent codes, StreamRF [59] models inter-
frame differences, and NeRFPlayer [60] decomposes fields into static and
deformable components. Tensor4D [63], HexPlane [65], and others factorize
10

<!-- page 14 -->
space-time volumes for efficient modeling.
Complementary to these are
monocular dynamic view synthesis methods [71, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 83] or multi-view neural view synthesis, which rely on priors over
motion [37], scene flow [71], or depth [23, 84] to compensate for sparse camera
views while maintaining image rendering results. In contrast, our framework
exploits synchronized multi-view imagery to better constrain dynamic scene
for novel view synthesis.
More recently, 3DGS has been extended to dynamic settings [85, 86, 87,
88, 3, 89, 90, 91, 92]. 4K4D [85] combines 4D point clouds with K-Planes
[64] and depth peeling. [86] and [87] employ 4D Gaussians with temporally
discrete keyframes and linear motion. Despite their favorable performance,
these methods suffer from flickering, temporal inconsistency [93], limited
expressiveness [93, 94], despite lighter model. Furthermore, most, if not all,
require SfM-based 3d point initialization, which is error-prone in dynamic
scenes. Not long ago, MCMC-GS [95] and structure from motion 3d point-
free approaches to 3DGS [96] are proposed that could work with inaccurate
random or sparse point cloud initialization. Yet, these approaches becomes
unstable when motion lacks smooth temporal continuity, leading to unreliable
propagation of Gaussians across frames. For instance, precise per-frame
3D point recovery and tracking become significantly more error-prone in
dynamic scene [97, 98] such as sports, where limb articulation, occlusion, and
pose discontinuity are common. Therefore, after testing on several dynamic
sports scene with multi subject sports motions, where subjects undergo
high acceleration and non-rigid deformation using different approaches 3D
priors [99, 100], requirement of accurate explicit 3D point module becomes
11

<!-- page 15 -->
an indispensable choice for 3DGS based pipeline to maintain excellent novel
view synthesis results. This raises the primary concern that maintaining
image-synthesis quality requires storing accurate 3D points per instance,
which results in either a high memory footprint or degraded performance
when 3DGS-based approaches are used for our dynamic scene time-archival
objective. A recent work QUEEN [101] compresses 3DGS dynamic scenes
yet relies on Gaussian identity consistency, tracked primitives, canonical
motion fields, which degrades under large motions—a common case in sports
applications.
Our approach departs from prior work in several ways. First, we propose a
fully implicit, temporally indexed MLP representation learned from synchro-
nized multi-view inputs, completely eliminating the reliance on SfM-derived
3D points. Second, our method supports compact time-archival for reliable
view synthesis, i.e., novel view synthesis at past moments, allowing efficient
modeling plenoptic function for dynamic subjects moving through space-time
contrary to the static scene papers [11, 12]. This capability is essential for
sports replay and visual performance applications and is absent in current
3DGS and NeRF-based approaches. Additionally, unlike [102], our approach
do not need pre-reconstructed accurate T-pose priors for each player for novel
view synthesis of a sports scene.
Lastly, we note that because each radiance field is modeled independently,
training our approach is trivially parallelizable across GPUs, enabling a
distributed computing framework that provides exact, drift-free reconstruction
for every moment. While 3DGS approaches bring many benefits—without
taking anything away from their tremendous value to novel view synthesis
12

<!-- page 16 -->
research—in the context of sports applications for time-archival, we observe
that per-time-step optimization combined with time-archival management is
more practical with the proposed implicit formulation than with sequential
dynamic 3DGS pipelines, which rely on temporal dependencies that hinder
full parallelization and complicate time-archival in dynamic settings.
3. Methodology
Central to our work is the goal to support the time-archival of a dy-
namic event, which leads to this notion of modeling the plenoptic function
Φ(x, Ωθ, Ωϕ, λ, t) (refer to Eq.(1)). In our formulation, we assume constant
λ to focus on RGB color synthesis that maintains a temporally indexed
functional scene representation Ft for each time instance. This design allows
for time-archival of dynamic scenes, enabling novel-view synthesis for any
prior or current moment. Before, we delve into formulation, let’s understand
our scene acquisition setup.
(i) Multi-View Acquisition Setup.
We begin with N synchronized
multi-view static cameras arranged strategically around the scene to ensure
comprehensive spatial coverage and minimize occlusion, which is a typical
setup for sports and visual performance scene. At each discrete time instance
t, the setup captures N synchronized multi-view images:
It = {I(1)
t , I(2)
t , . . . , I(N)
t
},
(2)
where I(i)
t
∈RH×W×3 represents the RGB image captured by the ith camera
at time t, with image dimensions H × W.
Additionally, each camera’s intrinsic calibration matrix Ki ∈R3×3 and
13

<!-- page 17 -->
extrinsic parameters (rotation Ri ∈SO(3) and translation ti ∈R3) are known
or estimated using classical SfM method [5].
(ii) Neural Implicit Representation. We put-forward a neural implicit
representation approach to model the dynamic 3D scene. This representation
models the scene implicitly via a multilayer perceptron (MLP) augmented with
a spatial hashing-based encoding to facilitate efficient scene representation and
real-time rendering. Similar to [8], we use an input encoding mechanism based
on a multi-resolution hash grid. These mappings are efficiently implemented
via a hash function that indexes a fixed-size table, allowing a compact yet
expressive encoding of spatial detail. The features of each resolution level are
concatenated and serve as the input to a lightweight multilayer perceptron
(MLP). The MLP itself is intentionally kept shallow and narrow, usually
consisting of two to three hidden layers with modest width, making it highly
efficient to train and evaluate (refer to Appendix A.3 for technical details
on software implementation). Formally, the scene at each time instance t is
encoded as a continuous volumetric function:
Ft : (x, d) →(c, σ),
(3)
where x ∈R3 denotes a 3D spatial location, d ∈S2 is the viewing direction,
c ∈R3 is the predicted RGB color at point x viewed from direction d, and
σ ∈R+ represents the volume density.
Unlike NeRF method and its extensions, which construct a single static
scene representation, our approach maintains a separate set of parameters
Θt for each time step t. This enables temporal indexing, thus preserving
the capacity to revisit and render any past state of the dynamic scene. It
is not an outlandish assumption that in application like sports and visual
14

<!-- page 18 -->
performance, we have multiview synchronized static camera setup. Therefore,
any dynamic subject observed by multiple synchronized cameras only varies
by a rigid transformation. Consequently, the function Ft is approximated by
a compact neural network:
Ft(x, d; Θt) = MLP(γ(x), γ(d); Θt),
(4)
where γ(·) is the spatial hashing-based positional encoding and Θt are the
learnable network parameters at time t. By distributing the overall scene
representation burden across multiple MLP over time, we achieve efficient
dynamic scene modeling enabling time-archival capabilities. This design
highlights a new direction for modular implicit neural representation, open-
ing opportunities for scalability, compositionality, and retrospective neural
rendering. Mathematically,
F(x, c, t) = Ft(γ(x), γ(d)); where,γ(.) denotes positional encoding.
(5)
So, overall dynamic event can be written as a collection of Ft or {F1, F2, ..., FT}
(iii) Volume Rendering for View Synthesis. To synthesize novel views,
we perform volumetric rendering along camera rays.
Given a nth novel
viewpoint parameterized by camera intrinsics Kn and extrinsics (Rn, tn), each
pixel color is computed via numerical integration along rays cast into the
volume:
ˆC(r, t) =
Z sf
sn
T t(s)σt(r(s))ct(r(s), d)ds, with T t(s) = exp

−
Z s
sn
σt(r(u))du

,
(6)
where T t(s) is the transmittance at time t, r(s) = o + sd defines the ray
originating from camera center o in direction d, parameterized by the distance
15

<!-- page 19 -->
s, with sn and sf indicating near and far clipping planes, respectively. Similar
to [8], we discretize this integral into M samples for computational efficiency:
ˆC(r, t) ≈
M
X
j=1
T t
j(1 −e−σjδj)cj,
T t
j = e−Pj−1
k=1 σkδk,
(7)
where δj is the distance between consecutive sampled points along the ray,
and (cj, σj) represent the color and density predictions at sample point j.
(iv) Training and Optimization. At each time instance t, the neural
implicit function Ft is trained using the captured multi-view images by
minimizing the following photometric reconstruction loss:
L(Θt) =
N
X
i=1
X
r∈Ri
 ˆC(r; Θt) −C(r)

2
2 + κ∥Θt+1 −Θt∥2
2,
(8)
where ˆC(r; Θt) is the rendered pixel color from viewpoint i along ray r, C(r)
is the observed pixel color, and Ri denotes the set of sampled rays from
camera i, and κ is a constant scalar. To facilitate temporal consistency
across time instances, we optionally include temporal regularization terms or
impose consistency constraints between consecutive neural implicit functions
(Ft−1, Ft). This encourages consecutive time MLPs to have similar weights.
Yet, for maintaining the time efficiency while keeping the memory footprint as
minimum as possible, in our experiment, we treats the dynamic event at each
time step independently. Our take on this stems from the application this
paper targets, i.e., sports and visual entertainment, where several dynamic
subjects are present and their 3D positions are observed to change drastically
between frames. Even empirically, independent time modeling has been
observed to work well, and therefore, in this work we adhere to it.
16

<!-- page 20 -->
(v) Inference and Novel View Generation. At inference time, given a
desired viewpoint not included in the original capture setup, we render novel
views in real-time using the trained neural implicit functions. This design
supports not only rendered image visualization but also retrospective rendering
by querying archived scene representations Ft. As such, our approach enables
time-aware visualizations that are particularly well-suited for sports replay,
performance reenactment, and others.
4. Experiment and Results
Implementation Details. We implemented our approach using the Py-
Torch 2.5.1 and tested on NVIDIA GPUs with CUDA version 11.8. All
comparative experiments were performed on the NVIDIA A40 GPU (50 GB
RAM), and additionally tested on the NVIDIA H100 GPU (251 GB RAM).
To enable efficient training and inference across temporally indexed neural
scene representations, we adopted a modular design that supports time-wise
synchronized optimization while preserving model parameters. We used syn-
chronized calibrated camera intrinsic and extrinsic to evaluate our approach
result. For benchmarking, we compared our approach against different state-
of-the-art dynamic scene rendering methods, including D-NeRF [72], D-3DGS
[86], 4DGS [3], and ST-GS [4]. Quantitative and qualitative evaluations
were performed on the publicly available CMU Panoptic Studio dataset [103],
which offers richly annotated multi-view video sequences suitable for dynamic
scene analysis for complex human motion analysis. In addition, we evaluated
performance on a newly introduced synthetic multiview dynamic scene dataset
curated by us to reflect complex motion characteristics of visual performances
17

<!-- page 21 -->
and sports scenarios. This dual evaluation protocol ensures the robustness
and generalizability of our method across diverse dynamic environments. We
used the popular PSNR as well as the LPIPS metric to compare the image
rendering quality with other methods, while the rendering speed is quantified
using the FPS metric. More details on the model train time settings for each
datasets—detailed next, are provided in Appendix A.3. Code and dataset
will be available at link.
4.1. Dataset for Evaluation
In this section, we first introduce our newly proposed dataset and experi-
ment setup simulating the visual performance and sports events, which is the
main focus of this paper. Next, we detail on our method’s performance on
CMU Panoptic dataset [103] and its experimental setup.
Synthetic Multiview Dynamic Scene Dataset. We acquired synthetic
datasets under three settings using Blender 4.0 for experimental evaluation:
(i) Dancing-Walking-Standing (ii) Soccer Penalty Kick, and (iii) Soccer
Multiplayer. The 3D models used in creation of these datasets were taken
from publicly available Internet assets.
In our acquisition setup, we sample N camera poses on a hemisphere
of radius R using Fibonacci sphere sampling. Given N viewpoints (i =
0, 1, . . . , N −1), the i-th camera position pi ∈R3 is computed using the
following relations between ϕ, θi and zi (refer to Appendix A.1 for visual
illustration):
ϕ = π (3 −
√
5), (the golden angle);
zi = R

1 −i
N

; θi = (ϕ×i) mod 2π; ri =
q
R2 −z2
i ; pi =
 ri cos θi, ri sin θi, zi

.
18

<!-- page 22 -->
Here, R is the radius of the sphere. After obtaining the camera positions,
we oriented each camera to face the scene center at (0, 0, 0) and configured
the intrinsics of the camera as follows:
fx = fy = 2666.67,
(cx, cy) = (960.0, 540.0),
resolution = 1920 × 1080.
All radial distortion coefficients k1 = k2 = 0 and tangential distortion co-
efficients p1 = p2 = 0, implying an undistorted image. The resulting horizontal
and vertical field of view angles are camera_anglex = 0.6911 rad, camera_angley =
0.4711 rad. Adopting synchronized multiview camera setup, at each discrete
time instance, we acquire It (refer to Eq.(2))
(i) Dancing-Walking-Standing dataset. This dataset features three distinct
dynamic subjects exhibiting complex motion patterns such as dancing, walking,
and standing. A total of 65 time instances were captured, each rendered from
100 calibrated camera viewpoints. We indexed 0th, 30th, 60th and 90th camera
as our test cameras, while 1st camera is used for validation. This leads to 95,
1, and 4 camera viewpoints for training, validation, and testing, respectively.
This partitioning results in 6,175 training images (65 × 95), 65 validation
images (65 × 1), and 260 test images (65 × 4). All images are temporally
synchronized, and the camera split is consistent across all time steps to ensure
uniform view coverage and benchmarking reproducibility.
(ii) Soccer Penalty Kick dataset. This dataset consists of two dynamic subjects
simulating a penalty kick scenario. We rendered 109 distinct time instances,
each from 60 camera viewpoints. We indexed 21st, 37th, 40th, and 56th camera
as our test camera, while the 0th camera is used for validation. This leads to
55 camera viewpoints for training, 1 for validation, and 4 for testing, providing
us with 5,995 training images (109 × 55), 109 validation images (109 × 1),
19

<!-- page 23 -->
Figure 4: Visual Performance Qualitative Comparison Results with 4DGS approach
[3] on our synthetic multiview dataset. Left: The four camera frustum highlighted
in red shows the virtual cameras that will be used for dynamic scene broadcasting.
Right: Our rendered image results from those virtual cameras at a given time as
compared to 4D-GS [3] approach. We also provide the PSNR and LPIPS values for
quantitative comparison. Here, VC denotes corresponding virtual camera.
and 436 test images (109 × 4). The cameras are aptly synchronized in time
to capture diverse motion with consistent spatial sampling.
(iii) Soccer Multiplayer dataset. It captures three dynamic subjects in coor-
dinated soccer actions. A total of 83 time instances were rendered using 60
camera viewpoints. Of these, 55 viewpoints were used for generating training
images, 1 for validation, and 4 for testing. We indexed 21st, 37th, 40th and
56th camera number as our test camera, while 0th camera is for validation.
This results in 4,565 training images (83 × 55), 83 validation images (83 × 1),
and 332 test images (83 × 4). Similar to the previous dataset, the cameras
are synchronized for proper evaluation.
Real World Multiview Dynamic Scene Dataset.
To evaluate our
method on real-world dynamic scenes, we utilize the CMU Panoptic Studio
20

<!-- page 24 -->
Method→
D-NeRF [72]
D-3DGS [86]
4DGS [3]
ST-GS [4]
Ours
Dancing-Walking-Standing
PSNR↑
6.44
18.45
28.17
20.03
34.28
LPIPS↓
0.572
0.139
0.08
0.112
0.027
Soccer Penalty Kick
PSNR↑
10.64
26.45
26.25
25.99
33.81
LPIPS↓
0.407
0.071
0.045
0.077
0.028
Soccer Multiplayer
PSNR↑
6.15
26.43
26.2
25.92
31.85
LPIPS↓
0.533
0.087
0.061
0.104
0.039
Table 1: Quantitative comparison of our approach with state-of-the-art approaches
on synthetic dataset. The best results are shaded with green and blue for PSNR
and LPIPS metric, respectively
dataset [103]. It offers densely captured multi-view human activity sequences.
For controlled testing of dynamic subject rendering, we isolate the primary
foreground subjects using high-precision semantic segmentation. Specifically,
we apply the Segment Anything Model (SAM) [104] and its high-quality
variant SAM-HQ [105] to extract clean subject masks for each frame. Refer
to Appendix A.2 for details on coordinate conversion used for model training.
The preprocessing pipeline begins by detecting human subjects and as-
sociated props (e.g., baseball bats) in each frame using a YOLOv8 object
detector. Detected bounding boxes are then passed to SAM-HQ to generate
high-fidelity binary segmentation masks. These masks are aggregated into a
single foreground mask per frame and used to convert the RGB images into
RGBA format by embedding the merged mask into the alpha channel. This
results in background-removed imagery that isolates dynamic subjects for
21

<!-- page 25 -->
Baseball Bat
Hand Gesture
PSNR: 31.59
LPIPS: 0.0529
PSNR: 29.70
LPIPS: 0.0526
VC1
VC2
PSNR: 30.21
LPIPS: 0.0689 
PSNR: 30.79
LPIPS:  0.0713
VC1
VC2
a) CMU Panoptic Camera Setup
b) Our method results
Figure 5: Results on CMU Panoptic dataset [103]. Left: Multiview Camera setup.
Actual cameras are shown in black, where as virtual cameras are highlighted with
red. Right: Results using our approach on a couple of challenging sports sequence.
Here, VC denotes virtual camera.
controlled evaluation of dynamic subject rendering and evaluation.
We experiment on two challenging categories within the CMU Panoptic
dataset. For the Baseball Bat sequence (1 subject), we extract 100 consecutive
frames from the Sports1 subset, using 29 of the 31 high-definition cameras
for training and the remaining 2 for testing. We used 10th and 15th camera
as our test cameras. This configuration yields 2,900 training images and 200
test images. For the Hand Gesture sequence (1 subject), we extracted 201
consecutive frames from the Hands2 subset, following the same camera split,
resulting in 5,829 training images and 402 test images. Such a controlled and
high-quality pre-processing enables robust evaluation for real-world dynamic
novel view synthesis methods.
22

<!-- page 26 -->
Method→
D-NeRF [72]
D-3DGS [86]
4DGS [3]
ST-GS [4]
Ours
Baseball Bat (CMU Panoptic [103, 86] + SAM [104])
PSNR↑
6.35
♣
♣
♣
29.43
LPIPS↓
0.605
♣
♣
♣
0.066
HandGesture (CMU Panoptic [103, 86] + SAM [104])
PSNR↑
12.99
♣
♣
♣
29.19
LPIPS↓
0.135
♣
♣
♣
0.050
Table 2: Quantitative comparison of our approach with state-of-the-art approaches
on CMU Panoptic dataset. ♣symbolizes that the corresponding method fail to
give results. Note that D-3DGS [86] uses active sensor depth data, which is not
present in our experimental setting.
4.2. Evaluation and Result
We primarily assess our experimental results using various metrics. These
include the peak-signal-to-noise ratio (PSNR) and the perceptual quality
measure LPIPS [106], which measures the quality of rendered images. We
further analyze the rendering speed using the popular FPS metric.
(i) Results on Synthetic Multiview Dynamic Scene Dataset. We
compared our approach results with several state-of-the-art methods, such
as D-NeRF [72], D-3DGS [86], 4DGS [3], and SpaceTime GS [4], to examine
our performance on dynamic scene novel view synthesis. Table 1 provides the
quantitative results obtained on the proposed synthetic dynamic scene dataset.
While the state-of-the-art method shows some promise, it generally fails to
render convincing results on the proposed dynamic scene. Moreover, the
notion of time archival is absent in all the previous methods, hence unsuitable
for sports and visual performance applications. Figure 4 shows a qualitative
comparison of our approach with 4DGS [3], clearly demonstrating the benefit
of our approach.
23

<!-- page 27 -->
(ii) Results on CMU Panoptic Dataset. Table 2 presents the results
obtained from real-world datasets. D-3DGS need depth data, which is not an
input for the experimental setup, thus shown with ♣symbol. Furthermore, it
is apparent that 4DGS [3] and ST-GS [4] will fail due to over-reliance on 3D
points priors from the structure-from-motion framework. On the contrary, our
method successfully provides dynamic scene rendering results at the current
time. It also provides the flexibility to analyze the same scene in the past
and allows the user to place the new virtual camera and examine the scene
retrospectively. Surprisingly D-NeRF [72] provides unsatisfactory results
and this maybe due to the linear deformation field approximation in their
approach. Figure 5 provides qualitative results obtained on this dataset using
our method.
Method→
4DGS [3]
ST-GS [4]
Ours
PSNR↑
17.08
20.03
34.28
LPIPS↓
0.123
0.112
0.027
Table 3: The proposed approach faithfully
represent a dynamic scene compared to
3DGS based extensions. The above re-
sults for other methods are obtained on
our synthetic Dancing-Walking-Standing
sequence with random 3d point initializa-
tion.
0
20
40
60
80
100
120
140
FPS
10
15
20
25
30
PSNR
Ours
D-NeRF
Dynamic Gaussian
4D Gaussian
Spacetime Gaussian
Ours
Figure 6: Quantitative comparison of
the FPS and PSNR on the proposed
dynamic scene dataset.
4.3. Ablations
(i) Random 3D points initialization for 3DGS based approaches.
As shown in Table 2 that 3DGS [2] based methods for dynamic scene novel
24

<!-- page 28 -->
view synthesis, such as 4DGS [3] and ST-GS [4], fails on real world scene.
This is due to over-reliance on accurate 3D points from SfM to anchor the
spatial distribution of Gaussian primitives. By allowing flexibility to these
methods assuming a less ideal condition, we initialize Gaussian centers by
randomly sampling 3D positions within the bounding volume of the scene.
The hope is that such random initialization could serve as a warm start for
3DGS based approaches to initiate the optimization process for favorable
image rasterization.
The objective of this ablation was to assess whether current 3DGS frame-
works possess sufficient representational and optimization flexibility for a
complex dynamic scene. Empirical results when tested on synthetic Dancing-
Walking-Standing dataset (see Table 3) clearly show their limitation for
dynamic scene applications compared to ours that leverages rigidity across
synchronous camera views for neural scene representation learning without
any requirement of 3d points from SfM pipelines.
(ii) Trade-off Between Rendering Speed and Quality in Dynamic
Novel View Synthesis. The objective is to analyze the metric between
frames-per-second (FPS) and image rendering quality (measured in PSNR).
Figure 6 provides the quantitative results for the same. While 3DGS-based
approaches, such as 4D-GS [3] and ST-GS [4], deliver high-speed novel view
synthesis results via forward rasterization of explicit primitives, our work
observed a notable limitation in their applicability to dynamic scenes. Specif-
ically, despite 3DGS methods achieve high FPS, their rendering quality
degrades in the presence of complex dynamic motion, often yielding unsatis-
factory PSNR scores.
25

<!-- page 29 -->
Approach/Evaluation Metric
PSNR
LPIPS
Model (per time step)
Input PointCloud (per time step)
3DGS (GT point cloud)
36.47
0.0916
77 MB
∼6.2 GB
3DGS (random point cloud)
16.33
0.3761
91 MB
∼6.2 GB
Ours (no point cloud)
34.28
0.0255
48.8 MB
0.0
Table 4: Quantitative comparison of our approach with 3DGS for single time instance.
Without any ground truth 3d point cloud priors, our approach provides excellent
results with significantly low-memory requirement. On the contrary, if we use 3DGS
approach here instead of the proposed implicit approach for each time step, we need
to maintain a persistent dense folder with 3d data by possibly running COLMAP
for each time step (due to dynamic scene setup). For the Dance-Walking-Standing
dataset, this folder occupies nearly 6.2 GB to achieve the results mentioned above.
In contrast, our proposed implicit neural scene representation framework
provides a more favorable balance between quality and efficiency. By learning
temporally indexed radiance fields through a compact yet expressive neural
architecture, our method achieves consistently high PSNR across challenging
dynamic scenes while maintaining real-time or near-real-time FPS on modern
GPU hardware. These results highlight the limitations of relying solely on
speed-centric splatting methods for dynamic environments. We position
our approach as a more acceptable solution for the targeted applications
requiring temporal consistency and photorealistic rendering quality. Yet, our
current implementation provides 4-5 FPS, i.e., near real-time performance, a
limitation nonetheless2.
(iii) 3DGS vs Our MLP based representation on Single Time In-
stance. To further understand the suitability of implicit versus explicit scene
2refer to Appendix for more experimental analysis and video
26

<!-- page 30 -->
representations for time-archival in dynamic sports and visual-performance
settings, we conducted a controlled single-time-instance ablation comparing
our per-time-step MLP-based radiance field with 3D Gaussian Splatting under
two initialization regimes on Dance-Walking-Standing dataset: (i) using an
accurate ground-truth (GT) point cloud, and (ii) using a randomly initialized
point cloud. The results in Table 4 reveal several important insights. First,
3DGS achieves strong performance only when provided with a high-quality
GT point cloud (PSNR 36.47, LPIPS 0.0916), confirming its reliance on
precise geometric priors for stable reconstruction. However, in the absence of
such priors—which is the realistic setting for fast-paced sports scenes with
severe articulation, occlusion, or motion blur—the performance of 3DGS
degrades drastically (PSNR 16.33, LPIPS 0.3761). In contrast, our implicit
representation, which requires no point-cloud initialization, produces con-
sistently high-quality renderings (PSNR 34.28, LPIPS 0.0255) comparable
to 3DGS with GT points but without any geometric supervision. Moreover,
the memory footprint per time step for our compact radiance field (48.8
MB) is significantly smaller than that of 3DGS (77–91 MB), highlighting
a critical advantage for long-horizon time-archival. Furthermore, our
approach overcomes the limitation of maintaining the persistent 3d data for
3DGS to work, saving nearly 6.2 GB of memory on the test case presented
in Table 4. Sports broadcasts and performance recordings often span hun-
dreds or thousands of frames, and storing explicit Gaussian fields per time
step becomes prohibitively expensive and brittle due to their dependence on
accurate per-frame geometry. This ablation demonstrates that our implicit
formulation is not only more robust in the absence of high-quality 3D geome-
27

<!-- page 31 -->
Method
Train Time
Parallelizable
(in Time)
PSNR
LPIPS
4DGS [3]
∼0.30 −3.0 hours
No (sequential)
∼26.2 - 28.1
∼0.045 - 0.061
ST-GS [4]
∼0.46 −0.71 hours
No (sequential)
∼20.0 - 25.9
∼0.078 - 0.112
D-3DGS [86]
∼1.40 hours
No (Limited)
∼18.5 - 26.5
∼0.071 - 0.139
Ours
∼5.65 - 8.90 hours
Yes (Fully)
∼31.9 - 34.3
∼0.027 - 0.039
Table 5: Our per-time-step radiance fields achieve the highest reconstruction quality
(PSNR, LPIPS) while offering fully parallelizable training across time, unlike
sequential 4DGS, ST-GS, and D-3DGS based approaches. Despite a higher per-
sequence training cost on a single GPU, full parallelization makes our method
significantly more scalable for long-horizon time-archival in sports and visual-
performance applications.
try but also far more storage-efficient, making it particularly well suited for
scalable time-archival of dynamic events where extreme motion, rapid pose
changes, and multi-actor interactions undermine the assumptions required by
Gaussian-based representations.
(iv) Full pipeline train-time and result comparison compared to
4DGS and other similar baselines. To further assess the practical feasi-
bility of our per-time-step neural representation for time-archival applications
in sports and visual performance, we compare the end-to-end training time
of our method against state-of-the-art dynamic Gaussian splatting pipelines
and dynamic NeRF variants. As summarized in Table 5, methods such as
4DGS [3], ST-GS [4], and D-3DGS [86] are inherently sequential, i.e., their
optimization relies on propagating Gaussian primitives, motion fields, or
deformation information across frames, making them unsuitable for fully
28

<!-- page 32 -->
parallel processing of long dynamic sequences. Their per-sequence train-time
consequently ranges from approximately 0.3 to 3.0 hours for 4DGS and up
to 1.4 hours for D-3DGS. In contrast, although our radiance-field models
require 5.65–8.90 hours of total training time for an entire sequence on a single
NVIDIA A 6000 GPU, each time step is independent, making our approach
trivially parallelizable across tens or hundreds of GPUs. This parallelism
capability—absent in Gaussian-based dynamic methods—effectively reduces
wall-clock training time by an order of magnitude in multi-GPU settings,
which is critical for practical deployment in broadcast or replay pipelines
where rapid turnaround is essential. Importantly, our fully parallelizable de-
sign does not sacrifice reconstruction quality and achieves the highest PSNR
(31.9–34.3) and lowest LPIPS (0.027–0.039) across all baselines, substantially
outperforming Gaussian-based methods for dynamic scene. These results
demonstrate that independent, per-time-step implicit radiance fields provide
a compelling trade-off between compute, accuracy, and scalability, particu-
larly in the context of time-archival for dynamic sports scenes, where long
sequences, rapid motion, and high-fidelity reconstruction requirements make
sequential Gaussian-based pipelines prohibitive.
(v) Comparison with both implicit and explicit representation base-
lines for dynamic scene time-archival view synthesis. To assess the
suitability of implicit versus explicit scene representations for time-archival
view synthesis in dynamic environments, we conducted a comprehensive com-
parison against both 3D Gaussian Splatting (3DGS)–based dynamic methods
and recent implicit spatio-temporal radiance-field models (Table 6). Across
all three synthetic datasets, namely, Dance-Walking-Standing (DWS), Soccer
29

<!-- page 33 -->
Method→
D-3DGS [86]
4DGS [3]
ST-GS [4]
D-NeRF [72]
T-4D [63]
HP [65]
KP [64]
S-RF [59]
Ours
DWS
PSNR
18.45
28.17
20.03
6.44
16.55
15.82
16.40
18.87
34.28
LPIPS
0.1396
0.0800
0.1120
0.5726
0.2165
0.2470
0.2327
0.2054
0.0275
Memory
2.0MB
21.0MB
3.2MB
512k
280MB
68MB
419MB
2.2GB
3.1GB
Train Time
1.41h
0.83h
0.46h
4.25h
10.02h
0.96h
1.02h
0.65h
5.25h
Iterations
130K
17K
30K
20K
200K
25K
30K
80K
19K
S-PK
PSNR
26.45
26.25
25.99
10.64
25.86
22.45
21.30
22.28
33.81
LPIPS
0.0719
0.0450
0.0778
0.4070
0.0588
0.1567
0.1866
0.1792
0.0282
Memory
2.0MB
13.9MB
0.9MB
197K
280MB
68MB
419MB
1.8 GB
5.2GB
Train Time
1.48h
3.02h
0.47h
7.74h
9.12h
0.75h
0.94h
0.55h
8.90h
Iterations
130K
17K
30K
20K
200K
25K
30K
80K
16.5K
S-MP
PSNR
26.43
26.20
25.92
6.15
25.82
20.42
19.34
24.98
31.85
LPIPS
0.0872
0.061
0.104
0.5330
0.0881
0.2104
0.2115
0.0906
0.0392
Memory
2.0MB
19MB
0.9MB
228K
280MB
68M
419MB
1.8GB
4.0GB
Train Time
1.46h
2.94h
0.71h
4.21h
10.1h
0.83h
1.03h
0.58h
6.28h
Iterations
130K
17K
30K
20K
200K
25K
30K
80K
16.5K
Table 6: Comparison of implicit and explicit dynamic-scene representations on
three synthetic datasets. Our per-time-step implicit radiance fields achieve the
highest PSNR/LPIPS performance while maintaining a compact memory footprint,
outperforming 3DGS- and NeRF-based methods that rely on explicit geometry or
deformation tracking. This makes our approach substantially more scalable and
robust for long-horizon time-archival in dynamic sports and performance scenes.
Note: here we have not included the initial 3d point cloud size that is used by 3DGS
based approaches as input. DWS, S-PK, and S-MP stand for Dance-Walking-
Standing, Soccer Penalty-Kick, and Soccer Multi-Player dataset.
30

<!-- page 34 -->
GT
Ours
Tensor4D
HexPlane
Kplanes
Stream-RF
0
90
60
30
PSNR: 34.28
LPIPS: 0.0275 
PSNR: 16.55
LPIPS: 0.2165 
PSNR: 15.82
LPIPS: 0.2470 
PSNR: 16.40
LPIPS: 0.2327 
PSNR: 18.87
LPIPS: 0.2054 
Dome Capture Setup
(Top View)
Dome Capture Setup
(Side View)
Figure 7: Qualitative results demonstrating dynamic scene view synthesis from novel
viewpoints on our proposed Dancing-Walking-Standing dataset. Left: shows the
acquisition setup from top view and side view, while the novel views are shown
in red. From Top row to Bottom row: Image-based rendering result with the
state-of-the-art neural implicit methods for dynamic scenes. The red numbers in
each row indicate the camera-id used by the methods to render the scene. Our
approach generates images closer to the ground truth than others.
Penalty-Kick (S-PK), and Soccer Multi-Player (S-MP) our approach con-
sistently achieves the highest reconstruction quality, outperforming 3DGS
variants and dynamic NeRF baselines by a substantial margin in both PSNR
and LPIPS. Notably, dynamic splatting methods such as D-3DGS, 4DGS,
and ST-GS exhibit a strong dependence on initialization quality and temporal
regularization, which leads to performance degradation in scenes with rapid
articulation, multi-subject interactions, and occlusion patterns common in
sports. Implicit factorization approaches (Tensor4D, HexPlane, K-Planes,
and StreamRF) provide smoother reconstructions but struggle to maintain
31

<!-- page 35 -->
fidelity under abrupt motion discontinuities—refer to Figure 7 and Figure
8 for visual comparison on Dancing-Walking-Standing dataset. In contrast,
our per-time-step implicit radiance fields remain stable even under extreme
motion, owing to their independence from deformation graphs and Gaussian
tracking. Crucially, while 3DGS-based approaches often require large explicit
point or Gaussian sets, our method stores only a compact MLP per time
step, yielding a significantly more scalable representation for long-horizon
temporal archival. This computational efficiency is especially important for
sports broadcasting and performance capture, where hundreds or thousands
of frames must be retained and queried retrospectively. Taken together, these
results demonstrate that our implicit, temporally indexed formulation pro-
vides a more robust, memory-efficient, and high-fidelity solution for dynamic
scene time-archival compared to both explicit splatting-based and implicit
factorized alternatives.
(vi) Warm-Start Chaining in Time-Archival (Per-Timestep) 3D
Gaussian Splatting. By Warm-Start, we mean a good point cloud ini-
tialization to start dynamic Gaussian Splatting3 approach. To evaluate the
robustness of warm-start chaining in dynamic Gaussian Splatting, where
the optimized model at time step t is used to initialize time step t + 1, we
conducted a controlled ablation on a 15-frame soccer sequence under two
settings. All experimental settings were kept identical across runs, including
warm-start chaining, a fixed Gaussian count (densification disabled), 8000
optimization iterations per frame, and identical rendering parameters (r = 1).
3Note however that the term dynamic Gaussian Splatting here implies the per-timestep
(time-archival) 3DGS.
32

<!-- page 36 -->
Stream-RF
Kplanes
HexPlane
Tensor4D
Ours
Ground Truth
PSNR: 16.55
LPIPS: 0.2165 
PSNR: 15.82
LPIPS: 0.2470 
PSNR: 16.40
LPIPS: 0.2327 
PSNR: 18.87
LPIPS: 0.2054 
PSNR: 34.28
LPIPS: 0.0275 
Figure 8: Additional qualitative results showing results using different approaches
on Dancing-Walking-Standing dataset.
The only difference between the two runs was the quality of the point-cloud
initialization at the first frame.
In the GT-initialization (GT-init) setting, the first frame was initialized
using a Blender-exported ground-truth point cloud containing approximately
100k points. In the Noisy-initialization (Noise-init) setting, we constructed a
deliberately degraded initialization by retaining only 1% of the original points
and replacing the remaining 99% with uniformly sampled points within an
expanded bounding box (bbox_scale = 4.0), with randomized RGB values,
while keeping the total point count unchanged. Despite identical warm-start
chaining thereafter, the GT-initializtion run achieved PSNR 28.94/SSIM
0.851/LPIPS 0.390, whereas the Noisy-initialization run dropped to PSNR
26.20/SSIM 0.812/LPIPS 0.430. Notably, the resulting ∼2.74dB PSNR gap
persisted consistently across frames 1–15, indicating that inaccuracies in the
33

<!-- page 37 -->
first-frame initialization propagate through the entire warm-start chain and
are not substantially corrected by subsequent optimization.
Frame Number
2
4
6
8
10
12
14
PSNR Value
25.5
26
26.5
27
27.5
28
28.5
29
29.5
Warm Start: GT-init
Warm Start: Noisy-init
Figure 9:
Per-frame PSNR vari-
ation, illustrating the sensitivity
of 3DGS-based approaches to the
quality of the reference 3D point
cloud (warm start).
These results demonstrate that, while
warm-start chaining can improve efficiency
when accurate geometry is available, its
performance—contrary to ours—remains
strongly dependent on the accuracy of the
reference-frame point cloud.
In dynamic
sports scenarios, where occlusion, motion
blur, and rapid articulation frequently de-
grade point-cloud quality thus posing lim-
itation and motivates use of self-contained
per-time-step implicit representations for re-
liable dynamic scene time-archival.
5. Discussion
Independence Vs Temporal Cou-
pling. An important design decision in this work is to model each time
step with an independent implicit radiance field, rather than enforcing ex-
plicit information sharing across time steps. While many dynamic-scene
methods, both implicit and Gaussian-based, successfully exploit temporal
coupling to improve efficiency and coherence, such coupling implicitly assumes
smooth motion, stable correspondences, or persistent primitives across frames.
In the sports and visual-performance scenarios we target, these assumptions
are frequently violated due to abrupt motion, strong articulation, multi-person
34

<!-- page 38 -->
interactions, and rapid occlusion changes. In such regimes, temporal coupling
can become brittle, leading to drift or bias accumulation. By contrast, treat-
ing each time step as an independent, geometry-constrained reconstruction
problem enables exact temporal indexing and drift-free archival, which is
central to retrospective analysis and replay. Our decision to train independent
models per time step is therefore not meant to suggest that Gaussian-based
or temporally coupled methods are inherently inferior. Instead, it reflects a
deliberate design choice aligned with the specific problem setting we target,
i.e., time-archival camera virtualization under synchronized multi-view cap-
ture, where each time step is already strongly constrained by geometry and
where extreme, non-smooth motion and multi-actor interactions can make
temporal coupling brittle. In these regimes, enforcing information exchange
across time may introduce drift or bias, whereas independent optimization
provides exact, temporally indexed reconstructions.
6. Limitations
While the proposed time-archival camera virtualization framework demon-
strates strong performance on dynamic sports and visual-performance scenes
captured with synchronized multi-view capture setups, the proposed frame-
work has limitations. Firstly, our per-time-step formulation enables straight-
forward parallelization across time, yet full parallelism of our approach could
require additional GPU resources which may not always be available in
practice. On the other hand, while some components of dynamic Gaussian
splatting can also be parallelized, many dynamic Gaussian splatting pipelines
remain partially sequential in practice c.f., Table 5. This is due to tempo-
35

<!-- page 39 -->
ral dependencies such as Gaussian propagation, deformation tracking, or
canonical-space optimization, which limit the extent of parallelization without
introducing approximations. Secondly, small fine grained low-lying object(s)
on ground or stages (such as grass, bottles) [107, 108] and challenging illumina-
tion conditions (such as mixed color temperatures, specular highlights, rapid
exposure changes, or complex stage lighting) [109, 110] remains boundary case.
Under these conditions, we occasionally observe artifacts such as faint color
leakage or reduced contrast in the rendered results. In practice, incorporating
an explicit foreground–background separation stage (e.g., SAM-HQ [104, 105]
or similar high-quality segmentation) could help mitigate these effects, while
more explicit illumination modeling is an important direction for future work.
7. Conclusion
In this paper, we introduced an approach for time-archival camera virtual-
ization in dynamic scenes, unlike [11, 12], targeting applications in sports and
visual performance. Our approach learns temporally indexed neural implicit
representations from synchronized multi-view inputs, enabling photorealistic
novel view synthesis across space and time. Unlike 3DGS-based extensions
to dynamic scenes that rely on explicit geometry and fail under complex
motion, our approach offers superior temporal consistency and fidelity. A
key contribution is our ability to implicitly model the plenoptic function
without requiring explicit 3D points as input as the subject(s) in the scene
move through time. This enables re-rendering from arbitrary viewpoints
at any past moment, supporting archival and interactive replay. Extensive
evaluations demonstrate high-quality image rendering, outperforming current
36

<!-- page 40 -->
state-of-the-art baselines in targeted areas of application, such as sports and
visual performance.
References
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, R. Ng, Nerf: Representing scenes as neural radiance fields for view
synthesis, Communications of the ACM 65 (1) (2021) 99–106.
[2] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3d gaussian splat-
ting for real-time radiance field rendering., ACM Trans. Graph. 42 (4)
(2023) 139–1.
[3] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian,
X. Wang, 4d gaussian splatting for real-time dynamic scene rendering,
in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 20310–20320.
[4] Z. Li, Z. Chen, Z. Li, Y. Xu, Spacetime gaussian feature splatting for
real-time dynamic view synthesis, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
8508–8520.
[5] J. L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[6] J. Shen, H. Yu, J. Wu, W. Yang, G.-S. Xia, Lidar-enhanced 3d gaussian
37

<!-- page 41 -->
splatting mapping, in: IEEE International Conference on Robotics and
Automation (ICRA), 2025.
[7] J. Wei, S. Leutenegger, Gsfusion: Online rgb-d mapping where gaussian
splatting meets tsdf fusion, IEEE Robotics and Automation Letters
(2024).
[8] T. Müller, A. Evans, C. Schied, A. Keller, Instant neural graphics
primitives with a multiresolution hash encoding, ACM transactions on
graphics (TOG) 41 (4) (2022) 1–15.
[9] J. R. Bergen, E. H. Adelson, The plenoptic function and the elements
of early vision, Computational models of visual processing 1 (8) (1991)
3.
[10] A. Lippman, Movie-maps: An application of the optical videodisc to
computer graphics, Acm Siggraph Computer Graphics 14 (3) (1980)
32–42.
[11] Z. Li, W. Xian, A. Davis, N. Snavely, Crowdsampling the plenoptic
function, in: Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, Springer,
2020, pp. 178–196.
[12] A. Liu, S. Ginosar, T. Zhou, A. A. Efros, N. Snavely, Learning to
factorize and relight a city, in: Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020, Proceedings,
Part IV 16, Springer, 2020, pp. 544–561.
38

<!-- page 42 -->
[13] S. M. Seitz, C. R. Dyer, View morphing, in: Proceedings of the 23rd
annual conference on Computer graphics and interactive techniques,
1996, pp. 21–30.
[14] C. Buehler, M. Bosse, L. McMillan, S. Gortler, M. Cohen, Unstructured
lumigraph rendering, in: Proceedings of the 28th annual conference on
Computer graphics and interactive techniques, 2001, pp. 425–432.
[15] W. Xiao, R. Chierchia, R. S. Cruz, X. Li, D. Ahmedt-Aristizabal,
O. Salvado, C. Fookes, L. Lebrat, Neural radiance fields for the real
world: A survey, arXiv preprint arXiv:2501.13104 (2025).
[16] Y. Xie, T. Takikawa, S. Saito, O. Litany, S. Yan, N. Khan, F. Tombari,
J. Tompkin, V. Sitzmann, S. Sridhar, Neural fields in visual computing
and beyond, in: Computer graphics forum, Vol. 41, Wiley Online
Library, 2022, pp. 641–676.
[17] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, Y. He, 3d gaussian splatting
as new era: A survey, IEEE Transactions on Visualization and Computer
Graphics (2024).
[18] S. Lombardi, T. Simon, J. Saragih, G. Schwartz, A. Lehrmann,
Y. Sheikh, Neural volumes: learning dynamic renderable volumes from
images, ACM Transactions on Graphics (TOG) 38 (4) (2019) 1–14.
[19] Z. Chen, A. Chen, G. Zhang, C. Wang, Y. Ji, K. N. Kutulakos, J. Yu,
A neural rendering framework for free-viewpoint relighting, in: Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2020, pp. 5599–5610.
39

<!-- page 43 -->
[20] J. Thies, M. Zollhöfer, M. Nießner, Deferred neural rendering: Image
synthesis using neural textures, Acm Transactions on Graphics (TOG)
38 (4) (2019) 1–12.
[21] K.-A. Aliev, A. Sevastopolsky, M. Kolos, D. Ulyanov, V. Lempitsky,
Neural point-based graphics, in: Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020, Proceedings,
Part XXII 16, Springer, 2020, pp. 696–712.
[22] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, R. Ng, Nerf: Representing scenes as neural radiance fields for view
synthesis, Communications of the ACM 65 (1) (2021) 99–106.
[23] N. Jain, S. Kumar, L. Van Gool, Enhanced stable view synthesis, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 13208–13217.
[24] G. Riegler, V. Koltun, Stable view synthesis, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 12216–12225.
[25] B. Attal, J.-B. Huang, C. Richardt, M. Zollhoefer, J. Kopf, M. O’Toole,
C. Kim, Hyperreel: High-fidelity 6-dof video with ray-conditioned
sampling, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 16610–16620.
[26] T. Neff, P. Stadlbauer, M. Parger, A. Kurz, J. H. Mueller, C. R. A.
Chaitanya, A. Kaplanyan, M. Steinberger, Donerf: Towards real-time
rendering of compact neural radiance fields using depth oracle networks,
40

<!-- page 44 -->
in: Computer Graphics Forum, Vol. 40, Wiley Online Library, 2021, pp.
45–59.
[27] B. Attal, J.-B. Huang, M. Zollhöfer, J. Kopf, C. Kim, Learning neural
light fields with ray-space embedding, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
19819–19829.
[28] B. Y. Feng, A. Varshney, Signet: Efficient neural representation for
light fields, in: Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2021, pp. 14224–14233.
[29] Z. Li, L. Song, C. Liu, J. Yuan, Y. Xu, Neulf: Efficient novel view
synthesis with neural 4d light field, in: EGSR (ST), 2022, pp. 59–69.
[30] M. Suhail, C. Esteves, L. Sigal, A. Makadia, Light field neural rendering,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 8269–8279.
[31] H. Wang, J. Ren, Z. Huang, K. Olszewski, M. Chai, Y. Fu, S. Tulyakov,
R2l: Distilling neural radiance field to neural light field for efficient
novel view synthesis, in: European Conference on Computer Vision,
Springer, 2022, pp. 612–629.
[32] B. Kaya, S. Kumar, C. Oliveira, V. Ferrari, L. Van Gool, Uncalibrated
neural inverse rendering for photometric stereo of general surfaces, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2021, pp. 3804–3814.
41

<!-- page 45 -->
[33] B. Kaya, S. Kumar, F. Sarno, V. Ferrari, L. Van Gool, Neural radiance
fields approach to deep multi-view photometric stereo, in: Proceedings
of the IEEE/CVF winter conference on applications of computer vision,
2022, pp. 1965–1977.
[34] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
P. P. Srinivasan, Mip-nerf: A multiscale representation for anti-aliasing
neural radiance fields, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 5855–5864.
[35] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman,
Mip-nerf 360: Unbounded anti-aliased neural radiance fields, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 5470–5479.
[36] N. Jain, S. Kumar, L. Van Gool, Robustifying the multi-scale repre-
sentation of neural radiance fields, in: 33rd British Machine Vision
Conference Proceedings, BMVA Press, 2022, p. 578.
[37] N. Jain, S. Kumar, L. Van Gool, Learning robust multi-scale represen-
tation for neural radiance fields from unposed images, International
Journal of Computer Vision 132 (4) (2024) 1310–1335.
[38] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman,
Zip-nerf: Anti-aliased grid-based neural radiance fields, in: Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 19697–19705.
42

<!-- page 46 -->
[39] A. Chen, Z. Xu, A. Geiger, J. Yu, H. Su, Tensorf: Tensorial radiance
fields, in: European conference on computer vision, Springer, 2022, pp.
333–350.
[40] Z. Chen, Y. Zhang, K. Genova, S. Fanello, S. Bouaziz, C. Häne, R. Du,
C. Keskin, T. Funkhouser, D. Tang, Multiresolution deep implicit
functions for 3d shape representation, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 13087–13096.
[41] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, A. Kanazawa,
Plenoxels: Radiance fields without neural networks, in: Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5501–5510.
[42] W. Hu, Y. Wang, L. Ma, B. Yang, L. Gao, X. Liu, Y. Ma, Tri-miprf:
Tri-mip representation for efficient anti-aliasing neural radiance fields, in:
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 19774–19783.
[43] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, C. Theobalt, Neural sparse voxel
fields, Advances in Neural Information Processing Systems 33 (2020)
15651–15663.
[44] C. Reiser, S. Peng, Y. Liao, A. Geiger, Kilonerf: Speeding up neural
radiance fields with thousands of tiny mlps, in: Proceedings of the
IEEE/CVF international conference on computer vision, 2021, pp.
14335–14345.
43

<!-- page 47 -->
[45] C. Sun, M. Sun, H.-T. Chen, Direct voxel grid optimization: Super-fast
convergence for radiance fields reconstruction, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5459–5469.
[46] T. Takikawa, J. Litalien, K. Yin, K. Kreis, C. Loop, D. Nowrouzezahrai,
A. Jacobson, M. McGuire, S. Fidler, Neural geometric level of detail:
Real-time rendering with implicit 3d shapes, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition,
2021, pp. 11358–11367.
[47] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, U. Neumann,
Point-nerf: Point-based neural radiance fields, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5438–5448.
[48] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, A. Kanazawa, Plenoctrees
for real-time rendering of neural radiance fields, in: Proceedings of
the IEEE/CVF international conference on computer vision, 2021, pp.
5752–5761.
[49] R. Hartley, Multiple view geometry in computer vision, Vol. 665, Cam-
bridge university press, 2003.
[50] M. Broxton, J. Flynn, R. Overbeck, D. Erickson, P. Hedman, M. Duvall,
J. Dourgarian, J. Busch, M. Whalen, P. Debevec, Immersive light
field video with a layered mesh representation, ACM Transactions on
Graphics (TOG) 39 (4) (2020) 86–1.
44

<!-- page 48 -->
[51] A. Bansal, M. Vo, Y. Sheikh, D. Ramanan, S. Narasimhan, 4d visu-
alization of dynamic events from unconstrained multi-view videos, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2020, pp. 5366–5375.
[52] M. Bemana, K. Myszkowski, H.-P. Seidel, T. Ritschel, X-fields: Implicit
neural view-, light-and time-image interpolation, ACM Transactions on
Graphics (TOG) 39 (6) (2020) 1–15.
[53] B. Attal, S. Ling, A. Gokaslan, C. Richardt, J. Tompkin, Matryodshka:
Real-time 6dof video view synthesis using multi-sphere images, in:
European Conference on Computer Vision, Springer, 2020, pp. 441–459.
[54] K.-E. Lin, L. Xiao, F. Liu, G. Yang, R. Ramamoorthi, Deep 3d mask
volume for view synthesis of dynamic scenes, in: Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2021, pp.
1749–1758.
[55] K.-E. Lin, G. Yang, L. Xiao, F. Liu, R. Ramamoorthi, View synthesis of
dynamic scenes based on deep 3d mask volume, IEEE Transactions on
Pattern Analysis and Machine Intelligence 45 (11) (2023) 13250–13264.
[56] S. Lombardi, T. Simon, G. Schwartz, M. Zollhoefer, Y. Sheikh,
J. Saragih, Mixture of volumetric primitives for efficient neural rendering,
ACM Transactions on Graphics (ToG) 40 (4) (2021) 1–13.
[57] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim,
T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe, et al., Neural 3d
video synthesis from multi-view video, in: Proceedings of the IEEE/CVF
45

<!-- page 49 -->
Conference on Computer Vision and Pattern Recognition, 2022, pp.
5521–5531.
[58] L. Wang, J. Zhang, X. Liu, F. Zhao, Y. Zhang, Y. Zhang, M. Wu,
J. Yu, L. Xu, Fourier plenoctrees for dynamic radiance field rendering in
real-time, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2022, pp. 13524–13534.
[59] L. Li, Z. Shen, zhongshu wang, L. Shen, P. Tan, Streaming radiance
fields for 3d video synthesis, in: A. H. Oh, A. Agarwal, D. Belgrave,
K. Cho (Eds.), Advances in Neural Information Processing Systems,
2022.
[60] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, A. Geiger,
Nerfplayer: A streamable dynamic scene representation with decom-
posed neural radiance fields, IEEE Transactions on Visualization and
Computer Graphics 29 (5) (2023) 2732–2742.
[61] S. Peng, Y. Yan, Q. Shuai, H. Bao, X. Zhou, Representing volumet-
ric videos as dynamic mlp maps, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp.
4252–4262.
[62] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, M. Wu,
Neural residual radiance fields for streamably free-viewpoint videos, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 76–87.
46

<!-- page 50 -->
[63] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, Y. Liu, Tensor4d: Efficient
neural 4d decomposition for high-fidelity dynamic reconstruction and
rendering, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 16632–16642.
[64] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, A. Kanazawa,
K-planes: Explicit radiance fields in space, time, and appearance, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 12479–12488.
[65] A. Cao, J. Johnson, Hexplane: A fast representation for dynamic scenes,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 130–141.
[66] F. Wang, S. Tan, X. Li, Z. Tian, Y. Song, H. Liu, Mixed neural voxels
for fast multi-view video synthesis, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), 2023, pp. 19706–
19716.
[67] M. Işık, M. Rünz, M. Georgopoulos, T. Khakhulin, J. Starck, L. Agapito,
M. Nießner, Humanrf: High-fidelity neural radiance fields for humans
in motion, ACM Transactions on Graphics (TOG) 42 (4) (2023) 1–12.
[68] H. Lin, S. Peng, Z. Xu, T. Xie, X. He, H. Bao, X. Zhou, High-fidelity
and real-time novel view synthesis for dynamic scenes, in: SIGGRAPH
Asia 2023 Conference Papers, 2023, pp. 1–9.
[69] F. Wang, Z. Chen, G. Wang, Y. Song, H. Liu, Masked space-time hash
47

<!-- page 51 -->
encoding for efficient dynamic scene reconstruction, Advances in Neural
Information Processing Systems 36 (2024).
[70] S. Kim, J. Bae, Y. Yun, H. Lee, G. Bang, Y. Uh, Sync-nerf: Generalizing
dynamic nerfs to unsynchronized videos, in: AAAI Conference on
Artificial Intelligence, 2024.
[71] Z. Li, S. Niklaus, N. Snavely, O. Wang, Neural scene flow fields for
space-time view synthesis of dynamic scenes, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 6498–6508.
[72] A. Pumarola, E. Corona, G. Pons-Moll, F. Moreno-Noguer, D-nerf: Neu-
ral radiance fields for dynamic scenes, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp.
10318–10327.
[73] C. Gao, A. Saraf, J. Kopf, J.-B. Huang, Dynamic view synthesis from dy-
namic monocular video, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 5712–5721.
[74] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M.
Seitz, R. Martin-Brualla, Nerfies: Deformable neural radiance fields, in:
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 5865–5874.
[75] E. Tretschk, A. Tewari, V. Golyanik, M. Zollhöfer, C. Lassner,
C. Theobalt, Non-rigid neural radiance fields: Reconstruction and
novel view synthesis of a dynamic scene from monocular video, in:
48

<!-- page 52 -->
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 12959–12970.
[76] C. Wang, B. Eckart, S. Lucey, O. Gallo, Neural trajectory fields for
dynamic novel view synthesis, arXiv preprint arXiv:2105.05994 (2021).
[77] W. Xian, J.-B. Huang, J. Kopf, C. Kim, Space-time neural irradi-
ance fields for free-viewpoint video, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp.
9421–9431.
[78] Y. Du, Y. Zhang, H.-X. Yu, J. B. Tenenbaum, J. Wu, Neural radiance
flow for 4d view synthesis and video processing, in: 2021 IEEE/CVF
International Conference on Computer Vision (ICCV), IEEE Computer
Society, 2021, pp. 14304–14314.
[79] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Gold-
man, R. Martin-Brualla, S. M. Seitz, Hypernerf: A higher-dimensional
representation for topologically varying neural radiance fields, ACM
Transactions on Graphics (TOG) 40 (2021) 1 – 12.
[80] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner,
Q. Tian, Fast dynamic radiance fields with time-aware neural voxels,
in: SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.
[81] H. Gao, R. Li, S. Tulsiani, B. Russell, A. Kanazawa, Monocular dy-
namic view synthesis: A reality check, Advances in Neural Information
Processing Systems 35 (2022) 33768–33780.
49

<!-- page 53 -->
[82] Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.-Y.
Chuang, J. Kopf, J.-B. Huang, Robust dynamic radiance fields, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 13–23.
[83] Z. Li, Q. Wang, F. Cole, R. Tucker, N. Snavely, Dynibar: Neural
dynamic image-based rendering, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR),
2023, pp. 4273–4284.
[84] C. Liu, S. Kumar, S. Gu, R. Timofte, L. Van Gool, Single image depth
prediction made better: A multivariate gaussian take, in: Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 17346–17356.
[85] Z. Xu, S. Peng, H. Lin, G. He, J. Sun, Y. Shen, H. Bao, X. Zhou,
4k4d: Real-time 4d view synthesis at 4k resolution, in: Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2024, pp. 20029–20040.
[86] J. Luiten, G. Kopanas, B. Leibe, D. Ramanan, Dynamic 3d gaussians:
Tracking by persistent dynamic view synthesis, in: 2024 International
Conference on 3D Vision (3DV), IEEE, 2024, pp. 800–809.
[87] Z. Yang, H. Yang, Z. Pan, L. Zhang, Real-time photorealistic dynamic
scene representation and rendering with 4d gaussian splatting, in: In-
ternational Conference on Learning Representations (ICLR), 2024.
50

<!-- page 54 -->
[88] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, X. Jin, Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,
in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 20331–20341.
[89] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, C. Jiang, Phys-
gaussian: Physics-integrated 3d gaussians for generative dynamics, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 4389–4398.
[90] A. Kratimenos, J. Lei, K. Daniilidis, Dynmf: Neural motion factorization
for real-time dynamic view synthesis with 3d gaussian splatting, in:
European Conference on Computer Vision, Springer, 2024, pp. 252–269.
[91] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, X. Qi, Sc-gs:
Sparse-controlled gaussian splatting for editable dynamic scenes, in:
Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2024, pp. 4220–4230.
[92] Y. Lin, Z. Dai, S. Zhu, Y. Yao, Gaussian-flow: 4d reconstruction
with dynamic 3d gaussian particle, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21136–21145.
[93] S. Kumar, Y. Dai, H. Li, Multi-body non-rigid structure-from-motion,
in: 2016 Fourth International Conference on 3D Vision (3DV), IEEE,
2016, pp. 148–156.
51

<!-- page 55 -->
[94] S. Kumar, Y. Dai, H. Li, Spatio-temporal union of subspaces for multi-
body non-rigid structure-from-motion, Pattern Recognition 71 (2017)
428–443.
[95] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack,
A. Kar, A. Tagliasacchi, K. M. Yi, 3d gaussian splatting as markov
chain monte carlo, Advances in Neural Information Processing Systems
37 (2024) 80965–80986.
[96] Y. Foroutan, D. Rebain, K. M. Yi, A. Tagliasacchi, Evaluating alter-
natives to sfm point cloud initialization for gaussian splatting, arXiv
preprint arXiv:2404.12547 (2024).
[97] S. Kumar, Y. Dai, H. Li, Monocular dense 3d reconstruction of a
complex dynamic scene from two perspective frames, in: Proceedings
of the IEEE international conference on computer vision, 2017, pp.
4649–4657.
[98] S. Kumar, Y. Dai, H. Li, Superpixel soup: Monocular dense 3d recon-
struction of a complex dynamic scene, IEEE transactions on pattern
analysis and machine intelligence 43 (5) (2019) 1705–1717.
[99] S. Kumar, L. Van Gool, Organic priors in non-rigid structure from
motion, in: European Conference on Computer Vision, Springer, 2022,
pp. 71–88.
[100] S. Kumar, Non-rigid structure from motion: Prior-free factorization
method revisited, in: Proceedings of the IEEE/CVF Winter Conference
on Applications of Computer Vision, 2020, pp. 51–60.
52

<!-- page 56 -->
[101] S. Girish, T. Li, A. Mazumdar, A. Shrivastava, S. De Mello, et al., Queen:
Quantized efficient encoding of dynamic gaussians for streaming free-
viewpoint videos, Advances in Neural Information Processing Systems
37 (2024) 43435–43467.
[102] J. Huang, S. Subhajyoti Mallick, A. Amat, M. Ruiz Olle, A. Mosella-
Montoro, B. Kerbl, F. Vicente Carrasco, F. De la Torre, Echoes of the
coliseum: Towards 3d live streaming of sports events, ACM Transactions
on Graphics (TOG) 44 (4) (2025) 1–17.
[103] H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade,
S. Nobuhara, Y. Sheikh, Panoptic studio: A massively multiview system
for social motion capture, in: Proceedings of the IEEE international
conference on computer vision, 2015, pp. 3334–3342.
[104] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., Segment anything,
in: Proceedings of the IEEE/CVF international conference on computer
vision, 2023, pp. 4015–4026.
[105] L. Ke, M. Ye, M. Danelljan, Y.-W. Tai, C.-K. Tang, F. Yu, et al.,
Segment anything in high quality, Advances in Neural Information
Processing Systems 36 (2023) 29914–29934.
[106] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The un-
reasonable effectiveness of deep features as a perceptual metric, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
53

<!-- page 57 -->
[107] S. Kumar, A. Dewan, K. M. Krishna, A bayes filter based adaptive floor
segmentation with homography and appearance cues, in: Proceedings
of the Eighth Indian Conference on Computer Vision, Graphics and
Image Processing, 2012, pp. 1–8.
[108] S. Mittal, M. S. Karthik, S. Kumar, K. M. Krishna, Small object
discovery and recognition using actively guided robot, in: 2014 22nd
International Conference on Pattern Recognition, IEEE, 2014, pp. 4334–
4339.
[109] Y. Yao, Z. Zeng, C. Gu, X. Zhu, L. Zhang, Reflective gaussian splatting,
in: The Thirteenth International Conference on Learning Representa-
tions (ICLR), 2025.
[110] Y. Jiang, J. Tu, Y. Liu, X. Gao, X. Long, W. Wang, Y. Ma, Gaus-
sianshader: 3d gaussian splatting with shading functions for reflective
surfaces, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 5322–5332.
Appendix A. Technical Appendices
Appendix A.1. Synthetic Multiview Dataset Camera Pose Acquisition via
Fibonacci Sphere Sampling
For our synthetic multiview dataset, we assumed uniformly distributed
camera setup. The camera positions are assigned uniformly over the surface
of a hemisphere using Fibonacci sphere sampling. Our approach ensures
comprehensive coverage and approximately equal angular spacing between
54

<!-- page 58 -->
camera viewpoints. For a given hemisphere of radius R, the method calculates
positions for N camera viewpoints indexed by i, ∀i ∈{0, 1, . . . , N −1}.
The camera position pi for the i-th viewpoint is computed via following
steps—refer to Figure A.10 for visual understanding of the camera position
computation in our experimental setup.
1. Golden Angle: We utilize the golden angle ϕ ≈2.39996 radians,
defined mathematically as:
ϕ = π

3 −
√
5

2. Z-coordinate: The vertical position zi is evenly spaced along the
hemisphere’s vertical axis, defined by:
zi = R

1 −i
N

where zi ranges from R (when i = 0) down to R/N (when i = N −1),
covering the upper hemisphere without reaching the equatorial plane.
3. Azimuthal Angle (θi): For each camera index i, the azimuthal angle
θi around the vertical axis (measured from the positive X-axis in the
horizontal XY -plane) is given by:
θi = (ϕ × i) mod 2π; where, i varies from{0, 1, . . . , N −1}
For instance, since we index the cameras starting from 0, the fifth
camera corresponds to index 5. We can compute θ5 that corresponds to
the fifth camera as π
 3 −
√
5

× 5 mod 2π ≈1.81966π
55

<!-- page 59 -->
Figure A.10: Fibonacci sampling on a hemisphere of radius R for camera placement. The
i-th camera center pi (dark-blue solid marker) has vertical coordinate zi, in-plane radius
ri =
p
R2 −z2
i , and azimuth θi. The next center pi+1 (green solid marker) is rotated by
the golden angle ϕ about the vertical axis and lies at height zi+1 ≈zi.
4. Radius on the XY-plane: The horizontal radial distance ri at height
zi is:
ri =
q
R2 −z2
i
5. Cartesian Coordinates: With the radial and angular coordinates
established, each camera position is:
pi =
 ri cos θi, ri sin θi, zi

.
The resulting set of camera positions pi are uniformly distributed across a
hemisphere of radius R, providing balanced spatial coverage. This mathemat-
ical formulation aligns precisely with the geometric representation provided
in Figure A.10, clearly illustrating the relationships between ϕ, zi, ri, θi, and
56

<!-- page 60 -->
the resultant camera positions pi.
Appendix A.2. Technical Details on CMU Multiview Dataset Calibration and
Coordinate Conversion
To make the coordinate system used in CMU Panoptic dataset [103]
compatible with our coordinate system setting, we process each high-definition
(HD) camera entry from their provided JSON dataset file by introducing the
following steps:
1. Intrinsic Parameters Extraction: We first extract the image resolution
parameters w (width), h (height), and the intrinsic camera matrix K,
defined as:
K =


fx
0
cx
0
fy
cy
0
0
1

,
alongside radial distortion coefficients (k1, k2, k3) and tangential distortion
coefficients (p1, p2). We assign zero values when distortion parameters, i.e.,
(k1, k2, k3, p1, p2) are absent.
2. Field of View Computation: Next, we calculate the horizontal and
vertical fields of view from the intrinsic parameters as:
camera_anglex = 2 arctan
w/2
fx

,
camera_angley = 2 arctan
h/2
fy

.
3. Extrinsic Matrix Formation: The provided rotation R and translation
t vectors define a world-to-camera transformation matrix:
Mw→c =

R
t
0
1

.
57

<!-- page 61 -->
We invert this matrix to obtain the camera-to-world transformation:
Mc→w = M −1
w→c.
4. Coordinate System Alignment: Since the original CMU Panoptic
dataset [103] adopts a Y-up coordinate convention, we convert it to a
standard Z-up, Y-forward coordinate system.
This transformation is
performed using two sequential rotations:
Ry→z =


1
0
0
0
0
0
1
0
0
−1
0
0
0
0
0
1


,
Rx,180 =


1
0
0
0
0
−1
0
0
0
0
−1
0
0
0
0
1


.
The final aligned camera-to-world transformation matrix is then computed
as:
M ′
c→w = Rx,180 Ry→z Mc→w.
By following the outlined steps, we generate camera configurations fully
compatible with our corrdinate system. The resulting dataset entries ensure
consistency in intrinsic calibration, distortion correction, image alignment,
and world-coordinate system compatibility for effective training of ours as
well as other approaches that uses similar coordinate system.
Appendix A.3. Model Training Technical Details
Spatial bounding box.. To eliminate empty-space sampling and stabilise opti-
misation, we constrain every scene to the axis-aligned cube
B = [−B, B]3,
B = 2.
58

<!-- page 62 -->
All rays are clipped to their entry/exit points tin/out with respect to B, so
subsequent integration is restricted to the interval [tin, tout] ⊆[tmin, tmax].
Volume rendering.. For a camera ray r(t) = o + td, we follow the standard
formulation
C(r) =
Z tout
tin
T t(t)σ(r(t))ct(r(t))dt
T t(t) = exp

−
Z t
tin
σt(r(s))ds

,
which we approximate with a Riemann sum of step size δ:
C(r) ≈
NB
X
i=1
Tiσiciδ,
NB =

(tout −tin)/δ

.
Camera scaling.. Because the original camera centres p may fall outside the
bounding box B, we apply a global scale factor scale such that the rescaled
positions
p′ = scale p
satisfy p′ ∈B.
Network architecture.. Our model comprises three sequential modules that
operate on learned positional embeddings and an encoded view direction:
• Positional encoder. For each sample point x = (x, y, z)T ∈B, we
first normalise its coordinates to [0, 1]3 and then feed them into a multi-
resolution hash grid with L = 16 levels and C = 2 channels per entry.
Let Rmin = 16 and Rmax = 2048 × B be the grid resolutions at levels 0
and L−1, respectively. The geometric scale factor is
α = (Rmax/Rmin)1/(L−1),
Rl = Rmin αl.
The resulting positional embedding has dimension L × C = 32.
59

<!-- page 63 -->
• Density MLP. A two-layer network
Linear(32, 64)
ReLU
−−−→Linear(64, 1+15)
yields the volume density σ and a 15-D geometric feature f.
• Color MLP. The view direction is expanded to a Ddir=16-D spherical-
harmonics vector, concatenated with f, and processed by
Linear(Ddir+15, 64)
ReLU
−−−→Linear(64, 64)
ReLU
−−−→Linear(64, 3)
Sigmoid
−−−−→
to predict RGB.
Parameter count.. Each model has a single hash table of 222 = 4,194,304
entries (2 channels each) and, together with the two MLPs, amounts to
approximately 12.7 million learnable parameters.
Training Configurations for Each Dataset .. Detailed training configurations
for each dataset are:
• Dancing-Walking-Standing Dataset: Each MLP model was trained
for 19, 000 iterations, with scale = 0.3.
• Soccer Penalty Kick Dataset: Each MLP model was trained for
16, 500 iterations, with scale = 0.1.
• Soccer Multiplayer Dataset: Each MLP model was trained for
16, 500 iterations, with scale = 0.1.
• Baseball Bat from CMU Panoptic Dataset: Each MLP model
was trained for 14, 500 iterations, with scale = 0.006.
• Hand Gesture from CMU Panoptic Dataset: Each MLP model
was trained for 14, 500 iterations, with scale = 0.006.
60
