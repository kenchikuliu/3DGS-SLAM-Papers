<!-- page 1 -->
GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate
Surface Reconstruction
Mai Su1, Qihan Yu1, Zhongtao Wang1, Yilong Li1, Chengwei Pan2, Yisong Chen1, Guoping Wang1,∗Fei Zhu1,∗
1School of Computer Science, Peking University
2Institute of Artificial Intelligence, Beihang University
2DGS Zoom in
✗over-smoothed geometry
PGSR Zoom in
✗depth inconsistency artifacts
Ours Zoom in
✓complete & consistent geometry
2DGS
PGSR
Ours
Depth/flow-based visibility is unreliable, while Gaussian-level visibility enables consistent geometry.
Flow: Unstable
Depth: Fragmented
Ours: Coherent
Figure 1: Modeling visibility is critical for accurate surface reconstruction. Existing methods suffer from unreliable
visibility estimation: depth-based approaches produce fragmented supervision [3], while flow-based methods introduce noisy
correspondence [24]. As shown in the top comparisons, these issues lead to over-smoothed geometry (2DGS [13]) and depth
inconsistency artifacts (PGSR [3]). In contrast, our Gaussian-level visibility modeling aggregates cross-view contributions of
shared Gaussians, producing a coherent visibility signal that enables consistent geometry reconstruction. For visualization,
warmer colors indicate stronger visibility intensity.
Abstract
3D Gaussian Splatting (3DGS) enables efficient rendering, yet
accurate surface reconstruction remains challenging due to
unreliable geometric supervision. Existing approaches pre-
dominantly rely on depth-based reprojection to infer visibility
and enforce multi-view consistency, leading to a fundamental
circular dependency: visibility estimation requires accurate
depth, while depth supervision itself is conditioned on visibil-
ity. In this work, we revisit multi-view geometric supervision
from the perspective of visibility modeling. Instead of infer-
ring visibility from pixel-wise depth consistency, we explicitly
model visibility at the level of Gaussian primitives. We in-
troduce a Gaussian visibility-aware multi-view geometric con-
sistency (GVMV) formulation, which aggregates cross-view
visibility of shared Gaussians to construct reliable supervi-
sion over co-visible regions. To further incorporate monocu-
lar priors, we propose a progressive quadtree-calibrated depth
1
arXiv:2601.20331v2  [cs.CV]  30 Mar 2026

<!-- page 2 -->
alignment (QDC) strategy that performs block-wise affine cali-
bration under visibility-aware guidance, effectively mitigating
scale ambiguity while preserving local geometric structures.
Extensive experiments on DTU and Tanks and Temples demon-
strate that our method consistently improves reconstruction
accuracy over prior Gaussian-based approaches. Our code is
fully open-sourced and available at an anonymous repository:
https://github.com/GVGScode/GVGS.
1
Introduction
3D Gaussian Splatting (3DGS) [16] has recently emerged as a
powerful and efficient representation for novel view synthesis,
offering high-quality rendering with significantly reduced train-
ing and rendering cost compared to neural implicit radiance
fields [1, 23, 25]. By directly optimizing geometric and appear-
ance parameters with rasterization-based rendering, 3DGS has
quickly become a strong foundation for real-time and large-
scale scene representation. However, despite its success in
appearance modeling, accurately recovering surface geometry
from 3DGS remains challenging, as the representation is in-
herently optimized for rendering rather than precise geometric
reconstruction.
A core challenge lies in the reliability of geometric supervi-
sion under multi-view settings. Under purely photometric super-
vision, the volumetric and unstructured nature of Gaussian prim-
itives allows them to drift away from true surfaces while still
explaining image observations, leading to geometric ambiguity,
thickness artifacts, and degraded multi-view consistency [32].
To address this issue, recent works introduce geometry-aware
regularization into the 3DGS framework, such as depth and nor-
mal priors or local smoothness constraints [4, 30]. Meanwhile,
other approaches reformulate Gaussian primitives as surface-
aligned elements, including planar-based representations [3],
surfel-based or 2D Gaussian formulations [12, 13, 35], and
hybrid methods combining Gaussian splatting with implicit
geometry fields [14, 19, 37]. These advances highlight the
importance of incorporating geometric structure into Gaussian
representations for accurate surface reconstruction.
Despite these efforts, existing methods rely heavily on depth-
based supervision, assuming that visibility can be inferred from
depth reprojection. This leads to a fundamental circular depen-
dency: visibility estimation depends on accurate depth, while
depth supervision itself is conditioned on visibility. Conse-
quently, when depth becomes unreliable—e.g., under occlusion,
wide baselines, or weak textures—both visibility estimation
and geometric supervision degrade, resulting in unstable opti-
mization and inferior reconstruction quality.
To alleviate this issue, monocular depth and normal priors
are often introduced to guide Gaussian optimization [21, 30].
However, these priors suffer from scale ambiguity and local
inconsistency, and without proper alignment may introduce
artifacts or oversmooth fine structures. As a result, simply
combining multi-view depth consistency with monocular priors
does not fully resolve the underlying limitation.
In this paper, we revisit multi-view geometric supervision
from the perspective of visibility modeling. We argue that
the fundamental limitation of existing methods does not lie
solely in inaccurate depth estimation, but in the underlying
assumption that visibility can be reliably inferred from depth
reprojection. This assumption introduces an inherent circular
dependency: visibility estimation requires accurate depth, while
geometric supervision itself is conditioned on visibility. As a
result, once depth becomes unreliable, both visibility estimation
and geometric constraints degrade simultaneously.
Based on this insight, we introduce a Gaussian visibility-
aware multi-view geometric consistency (GVMV) formulation.
Our method estimates per-Gaussian visibility in a neighboring
view by aggregating its rendering contributions, and uses this
information to construct a visibility-aware supervision signal
in the reference view. This formulation allows geometric con-
sistency to be enforced over a broader set of co-visible regions,
rather than being restricted to areas where depth reprojection is
already reliable.
To further incorporate monocular priors, we propose a pro-
gressive quadtree-calibrated depth alignment strategy, which
aligns monocular depth with Gaussian-rendered geometry un-
der visibility-aware guidance. By performing coarse-to-fine,
block-wise affine calibration, our method mitigates scale ambi-
guity while preserving fine-grained local structures, enabling
monocular depth to serve as an effective geometric prior.
In summary, our contributions are threefold:
• Rethinking visibility modeling in multi-view supervi-
sion. We identify a fundamental circular dependency in
existing depth-based formulations, which leads to unreli-
able geometric supervision.
• Gaussian visibility-aware multi-view geometry. We
propose a Gaussian-level visibility modeling framework
that explicitly captures cross-view co-visibility and en-
ables robust geometric consistency beyond depth-reliable
regions.
• Visibility-guided monocular depth alignment. We in-
troduce a quadtree-calibrated depth alignment strategy
that integrates monocular priors under visibility-aware
guidance, improving both global consistency and local
geometric fidelity.
2
Related Works
Novel View Synthesis
Neural radiance field (NeRF) based
methods model scenes as continuous volumetric fields opti-
mized via differentiable rendering and are widely used for
novel view synthesis [11, 23], but incur high computational
cost due to dense ray marching and neural networks [1]. As
an efficient alternative, 3D Gaussian Splatting (3DGS) repre-
sents scenes with anisotropic Gaussian primitives and renders
via rasterization [16], enabling fast optimization and real-time
or near-real-time performance. Building on this representa-
tion, subsequent works have further improved rendering quality
and efficiency [7, 38], and extended it to large-scale [17, 29],
dynamic [6, 22], and sparse-view scenarios [5, 40].
2

<!-- page 3 -->
Gaussian Splatting for Surface Reconstruction
To explic-
itly recover surface geometry from Gaussian splatting, prior
work reformulates Gaussian primitives into surface-oriented
representations. SuGaR [12] introduces a surface-alignment
regularizer that encourages Gaussians to form locally surface-
tangent configurations, and derives an approximate distance
function for efficient mesh extraction via Poisson reconstruction
[15]. Planar-based formulations reinterpret anisotropic Gaus-
sians as locally planar primitives, enabling unbiased depth and
normal rendering and facilitating geometry-aware constraints
[3]. More generally, subsequent works constrain Gaussians to
planar or disk-like configurations to better adhere to underlying
surfaces [13, 35, 41]. In parallel, recent approaches address
transparent surface reconstruction by learning transparency at-
tributes to handle view-dependent appearance and geometric
ambiguities [18].
Another line of work combines Gaussian splatting with im-
plicit geometry fields to leverage the complementary strengths
of explicit and continuous representations. Methods such as
GSDF and GaussianUDF jointly optimize Gaussian primitives
with signed or unsigned distance functions to guide surface
reconstruction while retaining efficient rendering [19, 37]. Ge-
ometry Field Splatting further unifies this paradigm by repre-
senting geometry fields with Gaussian surfels and deriving an
efficient differentiable rendering formulation, establishing a
more principled connection between Gaussian splatting and
implicit geometry [14]. GOF models surfaces via a compact
continuous opacity field over Gaussian primitives, enabling effi-
cient and memory-compact reconstruction in unbounded scenes
without explicit distance fields or dense volumetric sampling
[39].
Geometric Constraints for Gaussian-based Surface Recon-
struction
Beyond representation-level designs, recent work
improves Gaussian-based surface reconstruction by incorporat-
ing geometric constraints during optimization. 2DGS [13] intro-
duces a normal consistency constraint to explicitly model per-
Gaussian surface normals and enforce alignment with depth-
derived normals, stabilizing disk-like primitives. DN-Splatter
[30] aligns Gaussian orientations with monocular normal priors
and further imposes local smoothness to enforce consistency
of depth, normal, and scale among neighboring Gaussians.
PGSR [3] proposes a multi-view geometric consistency loss
that jointly enforces depth and photometric consistency across
views by constraining shared planar Gaussian structures, draw-
ing inspiration from classical multi-view stereo formulations
[2]. These methods highlight the importance of geometric reg-
ularization for accurate surface reconstruction, but typically
assume reliable depth estimation or consistent cross-view scale.
Despite their effectiveness, existing multi-view constraints
still rely on accurate Gaussian depth and are sensitive to depth
bias in challenging regions, while monocular supervision does
not explicitly address cross-view scale ambiguity. Our method
addresses these limitations by explicitly modeling cross-view
visibility at the Gaussian level and introducing a quadtree-
calibrated monocular depth constraint, enabling more robust
multi-view supervision and reliable single-view depth guidance
under imperfect depth priors.
3
Method
Given a set of calibrated multi-view RGB images, we recon-
struct scene geometry using Gaussian splatting by jointly lever-
aging multi-view geometric cues and monocular depth priors.
An overview of the proposed framework is illustrated in Fig. 2.
Our method consists of two key components: (1) Gaussian
visibility-aware multi-view geometric consistency (GVMV),
which models cross-view visibility at the Gaussian level to
provide reliable geometric supervision; and (2) a quadtree-
calibrated depth constraint (QDC), which refines monocular
depth to offer coarse structural guidance during training. We
first review the formulation of 3D Gaussian Splatting, followed
by our two components, GVMV and QDC, and finally present
the overall training objective.
3.1
Preliminary: 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) [16] represents a scene as a
collection of learnable 3D Gaussian primitives distributed in
space. Each Gaussian is parameterized by a center µ ∈R3 and
an anisotropic covariance matrix Σ ∈R3×3, which together
define its spatial extent and density contribution.
G(x) = exp
 −1
2(x −µ)⊤Σ−1(x −µ)

,
(1)
where x ∈R3 denotes a 3D query location. In addition to
its geometric parameters, each Gaussian is associated with an
opacity value α ∈[0, 1] and appearance features that encode
view-dependent color.
During rendering, the 3D Gaussian primitives are trans-
formed into the image plane via the camera projection, giving
rise to corresponding 2D Gaussian footprints:
G′(x′) = exp
 −1
2(x′ −µ′)⊤Σ′−1(x′ −µ′)

,
(2)
where x′ ∈R2 denotes a pixel location. The final pixel color
is obtained by compositing the projected Gaussians in a depth-
sorted order via alpha blending:
C(x′) =
X
i
Ti αi(x) G′
i(x′) ci,
Ti =
i−1
Y
j=1
 1 −αj(x) G′
j(x′)

,
(3)
where ci denotes the color of the i-th Gaussian. As the entire
splatting and compositing process is fully differentiable, all
Gaussian parameters can be optimized end-to-end by minimiz-
ing photometric reconstruction errors.
3.2
Gaussian Visibility-Aware Multi-View Geo-
metric Consistency
Previous Gaussian-based surface reconstruction methods en-
force multi-view geometric consistency through pixel-wise
depth constraints.
Specifically, these approaches reproject
3

<!-- page 4 -->
3D Gaussians
Depth Map 𝑣𝑟
Depth Map 𝑣𝑛
Monocular depth
Quadtree-calibrated depth 
Rendered Image 𝑣𝑛
Training Image 𝑣𝑟
Rendered Image 𝑣𝑟
Depth-based mask
(✗Missing regions)
𝑊𝑖
Eq.(4)
δᵢ
Eq.(5)
Gaussian visibility-aware opacity 𝑂𝑟
Eq.(6)
Eq.(6)
Eq.(7)
>𝜏
Depth-based weight
Gaussian visibility-aware mask
(✓More complete)
Depth Anything V2
Gaussian-level Visibility Modeling (GVMV)
Visibility-guided Depth Alignment (QDC)
Gaussian visibility-aware weight
Figure 2: Overview of GVGS. Given 3D Gaussians, we render RGB images and depth maps from a reference view vr and a
neighboring view vn. Unlike conventional depth-based supervision that infers visibility implicitly and yields incomplete masks
under occlusion, we explicitly model cross-view visibility at the Gaussian level. We compute per-Gaussian visibility weights Wi
in vn (Eq. (4)) and obtain binary indicators δi, which are projected back to vr to construct a visibility-aware opacity Or (Eq. (5)).
This produces a more complete supervision mask over co-visible regions and enables the Gaussian visibility-aware geometric
consistency loss Lgvmvgeom (Eq. (6)). In parallel, we introduce a visibility-guided quadtree-calibrated depth alignment (QDC),
where monocular depth (Depth-Anything V2) is progressively aligned to Gaussian-rendered depth to define Lqdc (Eq. (7)). Both
depth-based and visibility-aware weights share the same color bar for visualization.
Gaussian-derived depth maps into neighboring views and pe-
nalize discrepancies; we refer readers to PGSR [3] for repre-
sentative implementations. However, such formulations rely
on the assumption that depth estimates are sufficiently accurate
to infer visibility, which often breaks down under occlusion,
large baselines, or geometric inconsistencies, as illustrated by
the “Depth-based” subfigure in Fig. 1, where reprojection con-
straints fail on the left building facade.
More fundamentally, existing methods implicitly assume that
visibility can be reliably inferred from depth reprojection. How-
ever, this introduces a circular dependency: visibility estimation
depends on accurate depth, while geometric supervision itself
is conditioned on visibility. As a result, once depth estimates
become unreliable, both visibility estimation and geometric
supervision degrade simultaneously.
To address this issue, we introduce the following principle:
multi-view geometric consistency should be enforced over
all pixels that are co-visible across neighboring views, rather
than being restricted to regions where depth reprojection is
already consistent. Unlike prior methods that infer visibility
implicitly from depth or patch consistency [8, 28], we explicitly
model visibility at the level of Gaussian primitives.
We therefore introduce a Gaussian-based visibility estima-
tion to explicitly capture cross-view co-visibility, and further
develop a Gaussian visibility-aware multi-view geometric
consistency formulation to enforce geometric supervision over
co-visible regions. An overview is shown in Fig. 2.
Gaussian-based Visibility Estimation.
Given a reference
view vr and a neighboring view vn, our goal is to determine
which Gaussian primitives are visible in vn, and therefore
should contribute to cross-view supervision.
To this end, we explicitly estimate the rendering contribution
of each Gaussian primitive during the differentiable rasteriza-
tion of the neighboring view vn. We define the visibility weight
of each Gaussian based on its rendering influence on the image,
resulting in Wi ∈R+:
Wi =
X
x∈Ωn
αi(x) · Ti(x),
(4)
where αi(x) denotes the opacity contribution of gi at pixel
x, and Ti(x) is the accumulated transmittance under standard
alpha compositing (Eq. (3)).
Intuitively, Wi measures how much Gaussian gi contributes
to the rendering of view vn. We then define a binary visibility
indicator δi = I(Wi > τ), where τ is a small threshold to
suppress negligible contributions.
Visibility Projection to the Reference View.
We next trans-
fer the estimated visibility back to the reference view vr. Specif-
ically, we construct a selectively accumulated opacity map:
Or(x) =
X
i
δi αi(x)
Y
j<i
(1 −αj(x)) ,
(5)
4

<!-- page 5 -->
where δi acts as a visibility gate that activates only those Gaus-
sians that are visible in vn.
As a result, Or(x) aggregates the contributions of Gaus-
sians that are co-visible across the two views, while preserving
the standard depth-ordered alpha compositing structure. Im-
portantly, Or(x) is not a simple opacity accumulation, but a
Gaussian-level visibility-aware weighting term that encodes
cross-view co-visibility. This design enables reliable super-
vision even in regions where depth-based reprojection fails,
as illustrated in Fig. 1, where facade regions with large depth
discrepancies are still correctly constrained.
Gaussian Visibility-Aware Geometric Consistency.
We
build upon the multi-view geometric consistency loss Lmvgeom
introduced in PGSR [3]. Each reference pixel x is associated
with a forward–backward reprojection error ϕ(x), which is
mapped to a confidence weight via a monotonic function such
as exp(−ϕ(x)). Pixels with large reprojection errors (e.g.,
ϕ(x) > 1) are typically excluded from supervision.
We extend this formulation by incorporating Gaussian-level
visibility information through Or(x). This allows geometric su-
pervision to be applied over a broader set of co-visible regions,
instead of being limited to areas where depth reprojection is
already reliable. As shown in Fig. 2, regions that were pre-
viously unsupervised (e.g., facade areas) become effectively
constrained.
The resulting Gaussian visibility-aware multi-view geomet-
ric consistency loss is defined as
Lgvmvgeom = 1
|V|
X
x∈V
 exp
 −ϕ(x)

+ λ Or(x)

ϕ(x), (6)
where V is constructed as the union of: (i) pixels satisfying
conventional depth-based consistency, and (ii) pixels identified
as co-visible by our Gaussian visibility-aware opacity. The
parameter λ controls the relative contribution of the visibility
term.
3.3
Quadtree-calibrated Monocular Depth Con-
straint
Recent advances in large-scale vision models have enabled
highly detailed monocular depth prediction from a single im-
age [33, 34]. However, when integrated into multi-view re-
construction pipelines, monocular depth exhibits two inherent
limitations: scale ambiguity and view-dependent bias. These
issues often lead to misalignment between monocular depth
and the geometry represented by Gaussians, resulting in biased
gradients and unstable optimization.
Existing approaches typically mitigate this problem by apply-
ing a global scale–and–shift calibration using sparse COLMAP
SfM points, or by restricting supervision to pixels that satisfy
multi-view reprojection consistency [9, 17]. However, such
strategies fail to address spatially varying depth bias, and can-
not ensure consistent alignment between monocular depth and
Gaussian-rendered geometry across complex scenes.
(a)
(b)
(c)
(d)
(e)
Figure 3: Progressive quadtree depth calibration. (a) Raw
monocular depth and (e) Gaussian-rendered depth show clear
bias. (b–d) Coarse-to-fine block-wise affine calibration (Lv1–
Lv3) progressively aligns monocular and rendered depth.
To address this limitation, we introduce a quadtree-
calibrated depth alignment strategy. Instead of applying
a single global calibration, we progressively align monocular
depth with Gaussian-rendered depth at multiple spatial scales,
under the guidance of reliable geometric regions identified by
our visibility-aware formulation.
Coarse-to-fine Quadtree Alignment.
During training, we
adopt a coarse-to-fine quadtree schedule. At iteration t, we
select a quadtree level L(t)∈{0, . . . , Lmax} and partition the
image into 2L(t) ×2L(t) blocks. As training proceeds, L(t)
is gradually increased, allowing depth alignment to transition
from global coarse calibration to fine-grained local refinement.
For each quadtree block Bk, we align the monocular depth
Dm(x) to the Gaussian-rendered depth Dg(x) using a block-
wise affine model [27]:
D′
m(x) = ak Dm(x) + bk,
x ∈Bk ∩V,
ak = σy∈Bk
 Dg(y)

σy∈Bk
 Dm(y)
,
bk = µy∈Bk

Dg(y) −ak Dm(y)

,
(7)
where (ak, bk) denote the affine calibration parameters associ-
ated with block Bk. Here, µ(·) and σ(·) denote robust estima-
tors of location and scale, implemented as the median and the
mean absolute deviation about the median.
Intuitively, this formulation aligns monocular depth to
Gaussian-rendered geometry in a locally adaptive manner.
Coarse quadtree levels correct global scale mismatch, while
finer levels capture spatially varying bias. Moreover, calibra-
tion is restricted to pixels within V, i.e., regions that are either
validated by multi-view reprojection or identified as co-visible
through Gaussian visibility (Eq. (6)), ensuring that alignment
is guided by reliable geometric regions.
The affine parameters are recomputed only when the
quadtree level L(t) increases and reused across subsequent
iterations, resulting in negligible computational overhead. As
illustrated in Fig. 3, this progressive strategy yields increas-
ingly accurate alignment, particularly in regions with large
initial depth bias (e.g., rooftop areas).
Quadtree-calibrated Monocular Depth Constraint.
After
alignment, the calibrated monocular depth D′
m(x) is supervised
against the Gaussian-rendered depth Dg(x) using an ℓ1 loss:
Lqdc =
X
x∈V
∥D′
m(x) −Dg(x)∥1 ,
(8)
5

<!-- page 6 -->
Table 1: Chamfer Distance on DTU (lower is better). For each scan, the best, second, and third results are highlighted with
red, orange, and yellow backgrounds, respectively.
CD (mm)↓
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
Time
NeuS [31]
1.00
1.37
0.93
0.43
1.10
0.65
0.57
1.48
1.09
0.83
0.52
1.20
0.35
0.49
0.54
0.84
>12h
VolSDF [36]
1.14
1.26
0.81
0.49
1.25
0.70
0.72
1.29
1.18
0.70
0.66
1.08
0.42
0.61
0.55
0.86
>12h
Neuralangelo [20]
0.37
0.72
0.35
0.35
0.87
0.54
0.53
1.29
0.97
0.73
0.47
0.74
0.32
0.41
0.43
0.61
>128h
3DGS [16]
2.14
1.53
2.08
1.68
3.49
2.21
1.43
2.07
2.22
1.75
1.79
2.55
1.53
1.52
1.50
1.96
11.2min
SuGaR [12]
1.47
1.33
1.13
0.61
2.25
1.71
1.15
1.63
1.62
1.07
0.79
2.45
0.98
0.88
0.79
1.33
1h
2DGS [13]
0.48
0.91
0.39
0.39
1.01
0.83
0.81
1.36
1.27
0.76
0.70
1.40
0.40
0.76
0.52
0.80
19.2min
GOF [39]
0.50
0.82
0.37
0.37
1.12
0.74
0.73
1.18
1.29
0.68
0.77
0.90
0.42
0.66
0.49
0.74
1h
QGS [41]
0.42
0.65
0.36
0.37
0.85
0.65
0.50
1.14
0.97
0.61
0.48
0.67
0.34
0.41
0.37
0.59
48min
PGSR [3]
0.39
0.54
0.39
0.36
0.78
0.57
0.49
1.07
0.64
0.59
0.47
0.54
0.30
0.37
0.34
0.52
40min
Ours
0.32
0.53
0.33
0.33
0.79
0.52
0.47
1.05
0.63
0.58
0.37
0.53
0.30
0.35
0.32
0.49
43min
Table 2: Quantitative F1-score comparison on the TNT
dataset (higher is better). The best/second/third results are
highlighted in red, orange, and yellow.
F1-Score↑
N-angelo 2DGS
GOF
PGSR
QGS
Ours(30k) Ours(60k)
Barn
0.70
0.41
0.51
0.56
0.55
0.54
0.58
Caterpillar
0.36
0.24
0.41
0.41
0.40
0.44
0.47
Courthouse
0.28
0.16
0.28
0.26
0.28
0.22
0.24
Ignatius
0.89
0.52
0.68
0.79
0.81
0.80
0.81
Meetingroom
0.32
0.17
0.28
0.34
0.31
0.36
0.39
Truck
0.48
0.45
0.58
0.65
0.64
0.68
0.68
Mean
0.50
0.33
0.46
0.50
0.50
0.51
0.53
Time
>127h
34min 114min 66min 75min
69min
117min
where V denotes the trusted region defined in Eq. (6).
This monocular depth is only used in early training to provide
coarse geometric guidance, rather than serving as final super-
vision. By restricting supervision to visibility-aware regions
and performing coarse-to-fine local alignment, our formula-
tion enables monocular depth to serve as a stable and accurate
geometric prior. This improves both global consistency and
local structural detail, while remaining robust to noisy or biased
depth predictions.
3.4
Training Objective
The overall training objective is formulated as a weighted sum
of multiple loss terms:
L = Lrgb+λ1Ls+λ2Lmvrgb+λ3Lgvmvgeom+λ4Lqdc, (9)
where λ1–λ4 are scalar weights that balance the contributions
of the individual loss terms.
Here, Lrgb denotes the standard photometric reconstruction
loss in 3D Gaussian Splatting, which combines ℓ1 and SSIM
terms. Following PGSR, we directly adopt single-view depth
and normal supervision via Ls, together with a multi-view
photometric consistency loss Lmvrgb [3]. Building upon these
components, we introduce our proposed Gaussian visibility-
aware multi-view geometric consistency loss Lgvmvgeom and
the quadtree-calibrated monocular depth loss Lqdc, which to-
gether enable robust geometric supervision and adaptive cali-
bration of monocular depth priors throughout training.
4
Experiments
4.1
Datasets
We evaluate our method on two standard multi-view surface
reconstruction benchmarks: DTU and Tanks and Temples
(TNT). DTU consists of calibrated indoor scenes with high-
quality structured-light ground-truth geometry. Following prior
work [3], we report results on the standard 15 scans using sym-
metric Chamfer Distance, without explicit alignment between
reconstructed geometry and ground-truth point clouds. We use
the DTU data preprocessed by 2DGS [13].
The Tanks and Temples (TNT) benchmark contains large-
scale indoor and outdoor scenes. We evaluate on the Intermedi-
ate set of six scenes (Barn, Caterpillar, Courthouse, Ignatius,
Meetingroom, and Truck), reporting per-scene F-score (F1).
We adopt the TNT data from GOF [39] and follow the evalua-
tion protocol of QGS [41].
4.2
Implementation Details
Our implementation builds upon PGSR [3], inheriting most
default hyperparameters, and uses monocular depth priors from
Depth Anything v2 [34], which are further calibrated via the
proposed quadtree-calibrated depth constraint. The influence
computation follows EAGLES [10]. For DTU, each scene
is trained for 30k iterations, during which the quadtree split
level L(t) is progressively increased at 10k, 15k, and 20k iter-
ations. The quadtree-calibrated monocular depth loss Lqdc is
activated from 7k to 25k iterations to ensure stable optimiza-
tion. Negligible Gaussian contributions are filtered using a
threshold τ = 0.0001, with Gaussians below this threshold
considered non-visible in the neighboring view. We set the visi-
bility weighting coefficient λ in Eq. (6) to 0.5. For Tanks and
Temples (TNT), each scene is trained for 60k iterations with cor-
responding hyperparameters adjusted accordingly. Meshes are
extracted via TSDF fusion [26]. All experiments are conducted
on an Ubuntu server equipped with four NVIDIA A6000 GPUs
(48 GB memory each). Additional implementation details are
provided in our code.
6

<!-- page 7 -->
scan55
scan65
scan106
scan110
Caterpillar
Reference
2DGS
QGS
PGSR
Ours
Courthouse
Truck
Figure 4: Qualitative Comparison of Reconstructed Geometry with Related Works on DTU and Tanks and Temples. Red boxes
highlight regions with noticeable geometric differences.
4.3
Comparison
For geometry evaluation, we follow the standard protocol of
recent Gaussian-based methods and conduct all experiments on
DTU and TNT at half resolution. Chamfer Distance and F-score
are reported as the primary geometry metrics. We quantitatively
compare our method with implicit surface methods [20, 31, 36]
and Gaussian-based surface reconstruction approaches [3, 12,
13, 16, 39, 41]. We reproduce the results of PGSR [3] and
QGS [41] using their official implementations, and adopt the
remaining baselines from QGS [41].
For DTU dataset, as shown in Table 1, our method achieves
the best Chamfer Distance on 14 out of the 15 evaluated scans.
In terms of overall accuracy, our approach attains a mean Cham-
fer Distance of 0.49 mm (0.4933 mm before rounding), improv-
ing upon the best prior result by approximately 5%. As illus-
trated in Fig. 4, our method produces more complete rabbit ears,
smoother skull forehead surfaces, more faithful reconstruction
of dental cavities, and a clearer separation between the bird’s
feet and the supporting base. We attribute these improvements
to the stronger geometric constraints introduced by our method,
which provide more reliable supervision in regions with un-
even illumination and sparse viewpoints. Despite additional
geometric supervision, our method incurs only minimal over-
head and maintains comparable training efficiency to existing
7

<!-- page 8 -->
scan55
scan40
Courthouse
Barn
Reference
Dptflow
PGSR
Ours
Figure 5: Qualitative comparison of visibility masks across different methods.
Gaussian-based approaches.
For the Tanks and Temples benchmark, Table 2 reports
quantitative F1-score results. Our method achieves the highest
average F1-score of 0.53 (0.5302 before rounding), outper-
forming all compared approaches. Specifically, it attains the
best performance on 3 of the 6 scenes and ranks second on
two, indicating strong and consistent reconstruction quality. As
shown in Fig. 4, only our method reconstructs the Caterpil-
lar bucket without holes and avoids depth artifacts on the left
side. Moreover, it is the only method that correctly recovers
the pillar beneath the staircase and the associated wall details
in the Courthouse scene, and also achieves the most faithful
reconstruction of the hollow wheel hub structures in the Truck
scene.
Visibility Comparison. Although no multi-view dataset pro-
vides ground-truth visibility masks, we qualitatively compare
the predicted visibility across methods. We compare visibility
masks produced by a flow-based method, DPFlow [24], and
a depth-based method, PGSR [3], as shown in Fig. 5. Flow-
based visibility is often noisy and unstable due to unreliable
motion estimation, while depth-based visibility is sensitive to
depth inaccuracies and leads to fragmented masks. In contrast,
our Gaussian-level visibility modeling produces cleaner and
more coherent masks, providing more reliable supervision over
co-visible regions.
Table 3: Ablation study on DTU and TNT.
Setting
CD (DTU) ↓
F1 (TNT) ↑
Full
0.493
0.530
w/o QDC
0.505
0.520
w/o QDC + MonoD
0.512
0.513
w/o QDC + MonoD + GVMV
0.519
0.503
w/o GVMV
0.511
0.512
(a) Full 
(b) w/o QDC + MonoD
(d) w/o QDC + MonoD + Gvmv
(c) w/o Gvmv
Figure 6: Qualitative ablation of the proposed components on
Meetingroom. Red boxes highlight failure cases.
4.4
Ablation Studies
We evaluate the contribution of each component via ablation.
Starting from the full model (Full), we progressively remove:
(1) the quadtree-calibrated depth constraint (w/o QDC), where
Gaussian depth is directly supervised by monocular depth [34];
(2) monocular depth supervision (w/o QDC + MonoD); (3)
the Gaussian visibility-aware multi-view geometric consistency
(w/o QDC + MonoD + GVMV); and (4) only GVMV (w/o
GVMV) to isolate its effect.
As shown in Table 3, ablation results on DTU and Tanks
and Temples (TNT) show that removing any component consis-
tently degrades reconstruction performance in terms of Chamfer
Distance and F1. We further provide qualitative comparisons
on the Meetingroom scene from TNT in Fig. 6, where the
progressive degradation in chair structures highlights the com-
plementary role of each component. Together, these results
provide consistent quantitative and visual evidence of our de-
sign effectiveness.
We also evaluate the sensitivity of the visibility threshold τ.
8

<!-- page 9 -->
Table 4: Ablation on the visibility threshold τ. Performance is
stable for small τ, with the best result at τ = 1e−4.
τ
1e−5
1e−4
1e−3
1e−2
1e−1
1
10
Soft
CD↓(DTU)
0.493 0.493 0.497 0.497 0.497 0.499 0.503 0.497
F1↑(TNT)
0.529 0.530 0.529 0.529 0.529 0.526 0.524 0.529
As shown in Table 4, the performance remains stable across a
wide range of small τ values, indicating that our method is not
sensitive to the exact threshold choice. When τ becomes large,
performance gradually degrades due to overly strict visibility
filtering. We also evaluate a soft visibility weighting formu-
lation, which achieves comparable results but does not show
clear advantages.
5
Conclusion
We present a Gaussian visibility-aware multi-view geometric
consistency formulation together with a quadtree-calibrated
monocular depth constraint, which provide richer geometric
supervision and lead to consistent improvements in surface
reconstruction quality.
Our method also produces high-quality multi-view visibility
masks as a byproduct, which can benefit a wide range of tasks,
including multi-view semantic segmentation, scene understand-
ing, and geometry-aware rendering and editing. We hope this
capability inspires future exploration of Gaussian-based visibil-
ity reasoning as a general geometric prior for multi-view vision
and graphics.
A current limitation of our method is the lack of dedicated
modeling for highly specular or transparent surfaces, which we
leave for future work.
References
[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Pe-
ter Hedman, Ricardo Martin-Brualla, and Pratul P Srini-
vasan. Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields. In Proceedings of the
IEEE/CVF international conference on computer vision,
pages 5855–5864, 2021.
[2] Neill DF Campbell, George Vogiatzis, Carlos Hern´andez,
and Roberto Cipolla. Using multiple hypotheses to im-
prove depth-maps for multi-view stereo. In European
conference on computer vision, pages 766–779. Springer,
2008.
[3] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Wei-
jian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun
Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian
splatting for efficient and high-fidelity surface reconstruc-
tion. IEEE Transactions on Visualization and Computer
Graphics, 2024.
[4] Hanlin Chen, Fangyin Wei, Chen Li, Tianxin Huang, Yun-
song Wang, and Gim Hee Lee. Vcr-gaus: View consistent
depth-normal regularizer for gaussian surface reconstruc-
tion. Advances in Neural Information Processing Systems,
37:139725–139750, 2024.
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan
Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham,
and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting
from sparse multi-view images. In European Conference
on Computer Vision, pages 370–386. Springer, 2024.
[6] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He,
Wenzheng Chen, and Baoquan Chen. 4d-rotor gaussian
splatting: towards efficient novel view synthesis for dy-
namic scenes. In ACM SIGGRAPH 2024 Conference
Papers, pages 1–11, 2024.
[7] Guangchi Fang and Bing Wang. Mini-splatting: Repre-
senting scenes with a constrained number of gaussians.
In European Conference on Computer Vision, pages 165–
181. Springer, 2024.
[8] Yasutaka Furukawa and Jean Ponce. Accurate, dense,
and robust multiview stereopsis. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 32(8):1362–
1376, 2010.
[9] Yuanyuan Gao, Hao Li, Jiaqi Chen, Zhengyu Zou, Zhi-
hang Zhong, Dingwen Zhang, Xiao Sun, and Junwei Han.
Citygs-x: A scalable architecture for efficient and geomet-
rically accurate large-scale scene reconstruction. arXiv
preprint arXiv:2503.23044, 2025.
[10] Sharath Girish, Kamal Gupta, and Abhinav Shrivas-
tava. Eagles: Efficient accelerated 3d gaussians with
lightweight encodings. In European Conference on Com-
puter Vision, pages 54–71. Springer, 2024.
[11] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai,
Feitong Tan, and Ping Tan. Cascade cost volume for
high-resolution multi-view stereo and stereo matching. In
Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 2495–2504, 2020.
[12] Antoine Gu´edon and Vincent Lepetit. Sugar: Surface-
aligned gaussian splatting for efficient 3d mesh recon-
struction and high-quality mesh rendering. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 5354–5363, 2024.
[13] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger,
and Shenghua Gao. 2d gaussian splatting for geometri-
cally accurate radiance fields. In ACM SIGGRAPH 2024
conference papers, pages 1–11, 2024.
[14] Kaiwen Jiang, Venkataram Sivaram, Cheng Peng, and
Ravi Ramamoorthi. Geometry field splatting with gaus-
sian surfels. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 5752–5762, 2025.
9

<!-- page 10 -->
[15] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe.
Poisson surface reconstruction. In Proceedings of the
fourth Eurographics symposium on Geometry processing,
volume 7, 2006.
[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–
1, 2023.
[17] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas,
Michael Wimmer, Alexandre Lanvin, and George Dret-
takis. A hierarchical 3d gaussian representation for real-
time rendering of very large datasets. ACM Transactions
on Graphics (TOG), 43(4):1–15, 2024.
[18] Mingwei Li, Pu Pang, Hehe Fan, Hua Huang, and Yi Yang.
Tsgs: Improving gaussian splatting for transparent sur-
face reconstruction via normal and de-lighting priors. In
Proceedings of the 33rd ACM International Conference
on Multimedia, pages 7220–7229, 2025.
[19] Shujuan Li, Yu-Shen Liu, and Zhizhong Han.
Gaus-
sianudf: Inferring unsigned distance functions through 3d
gaussian splatting. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 27113–27123,
2025.
[20] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Tay-
lor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan
Lin. Neuralangelo: High-fidelity neural surface recon-
struction. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8456–
8465, 2023.
[21] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and
Kui Jia. Gs-ir: 3d gaussian splatting for inverse render-
ing. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 21644–
21653, 2024.
[22] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by
persistent dynamic view synthesis. In 2024 International
Conference on 3D Vision (3DV), pages 800–809. IEEE,
2024.
[23] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view
synthesis. Communications of the ACM, 65(1):99–106,
2021.
[24] Henrique Morimitsu, Xiaobin Zhu, Roberto M Cesar, Xi-
angyang Ji, and Xu-Cheng Yin. Dpflow: Adaptive optical
flow estimation with a dual-pyramid framework. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pages 17810–17820, 2025.
[25] Thomas M¨uller, Alex Evans, Christoph Schied, and
Alexander Keller. Instant neural graphics primitives with
a multiresolution hash encoding. ACM transactions on
graphics (TOG), 41(4):1–15, 2022.
[26] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Push-
meet Kohi, Jamie Shotton, Steve Hodges, and Andrew
Fitzgibbon. Kinectfusion: Real-time dense surface map-
ping and tracking. In 2011 10th IEEE international sym-
posium on mixed and augmented reality, pages 127–136.
Ieee, 2011.
[27] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad
Schindler, and Vladlen Koltun. Towards robust monocu-
lar depth estimation: Mixing datasets for zero-shot cross-
dataset transfer. IEEE transactions on pattern analysis
and machine intelligence, 44(3):1623–1637, 2020.
[28] S.M. Seitz, B. Curless, J. Diebel, D. Scharstein, and
R. Szeliski. A comparison and evaluation of multi-view
stereo reconstruction algorithms. In 2006 IEEE Com-
puter Society Conference on Computer Vision and Pattern
Recognition (CVPR’06), volume 1, pages 519–528, 2006.
[29] Mai Su, Zhongtao Wang, Huishan Au, Yilong Li, Xizhe
Cao, Chengwei Pan, Yisong Chen, and Guoping Wang.
Hug: Hierarchical urban gaussian splatting with block-
based reconstruction for large-scale aerial scenes. In Pro-
ceedings of the IEEE/CVF International Conference on
Computer Vision, pages 28839–28848, 2025.
[30] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto
Seiskari, Esa Rahtu, and Juho Kannala.
Dn-splatter:
Depth and normal priors for gaussian splatting and mesh-
ing. In 2025 IEEE/CVF Winter Conference on Appli-
cations of Computer Vision (WACV), pages 2421–2431.
IEEE, 2025.
[31] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt,
Taku Komura, and Wenping Wang. Neus: Learning neu-
ral implicit surfaces by volume rendering for multi-view
reconstruction. arXiv preprint arXiv:2106.10689, 2021.
[32] Yuru Xiao, Deming Zhai, Wenbo Zhao, Kui Jiang, Junjun
Jiang, and Xianming Liu. Mcgs: Multiview consistency
enhancement for sparse-view 3d gaussian radiance fields.
IEEE Transactions on Pattern Analysis and Machine In-
telligence, 2025.
[33] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi
Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng
Chi, Haiyang Sun, Bing Wang, et al. Pixel-perfect depth
with semantics-prompted diffusion transformers. arXiv
preprint arXiv:2510.07316, 2025.
[34] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xi-
aogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth
anything v2. Advances in Neural Information Processing
Systems, 37:21875–21911, 2024.
10

<!-- page 11 -->
[35] Yixin Yang, Yang Zhou, and Hui Huang. Introducing un-
biased depth into 2d gaussian splatting for high-accuracy
surface reconstruction. In Computer Graphics Forum,
volume 44, page e70252. Wiley Online Library, 2025.
[36] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman.
Volume rendering of neural implicit surfaces. Advances
in neural information processing systems, 34:4805–4815,
2021.
[37] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xi-
angli, and Bo Dai. Gsdf: 3dgs meets sdf for improved
neural rendering and reconstruction. Advances in Neu-
ral Information Processing Systems, 37:129507–129530,
2024.
[38] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler,
and Andreas Geiger. Mip-splatting: Alias-free 3d gaus-
sian splatting. In Proceedings of the IEEE/CVF confer-
ence on computer vision and pattern recognition, pages
19447–19456, 2024.
[39] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction
in unbounded scenes. ACM Transactions on Graphics
(ToG), 43(6):1–13, 2024.
[40] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Min-
min Yi, and Haoqian Wang. Transplat: Generalizable 3d
gaussian splatting from sparse multi-view images with
transformers. In Proceedings of the AAAI Conference
on Artificial Intelligence, volume 39, pages 9869–9877,
2025.
[41] Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou,
Xiaojun Xiang, and Shuhan Shen.
Quadratic gaus-
sian splatting: High quality surface reconstruction with
second-order geometric primitives. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 28260–28270, 2025.
11
