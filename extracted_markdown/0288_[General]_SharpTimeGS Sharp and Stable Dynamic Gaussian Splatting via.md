<!-- page 1 -->
SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting
via Lifespan Modulation
Zhanfeng Liao1, Jiajun Zhang2, Hanzhang Tu1, Zhixi Wang1, Yunqi Gao4,
Hongwen Zhang3, Yebin Liu∗,1
1Tsinghua University 2Beijing University of Posts and Telecommunications
3Beijing Normal University 4Central China Normal University
Abstract
Novel view synthesis of dynamic scenes is fundamental
to achieving photorealistic 4D reconstruction and immer-
sive visual experiences.
Recent progress in Gaussian-
based representations has significantly improved real-time
rendering quality, yet existing methods still struggle to
maintain a balance between long-term static and short-
term dynamic regions in both representation and optimiza-
tion. To address this, we present SharpTimeGS, a lifespan-
aware 4D Gaussian framework that achieves temporally
adaptive modeling of both static and dynamic regions un-
der a unified representation.
Specifically, we introduce
a learnable lifespan parameter that reformulates tempo-
ral visibility from a Gaussian-shaped decay into a flat-
top profile, allowing primitives to remain consistently ac-
tive over their intended duration and avoiding redundant
densification. In addition, the learned lifespan modulates
each primitive’s motion, reducing drift in long-lived static
points while retaining unrestricted motion for short-lived
dynamic ones.
This effectively decouples motion mag-
nitude from temporal duration, improving long-term sta-
bility without compromising dynamic fidelity.
Moreover,
we design a lifespan-velocity-aware densification strategy
that mitigates optimization imbalance between static and
dynamic regions by allocating more capacity to regions
with pronounced motion while keeping static areas com-
pact and stable. Extensive experiments on multiple bench-
marks demonstrate that our method achieves state-of-the-
art performance while supporting real-time rendering up to
4K resolution at 100 FPS on one RTX 4090. Project page:
https://liaozhanfeng.github.io/SharpTimeGS.
1. Introduction
Dynamic novel view synthesis (NVS) aims to generate
photorealistic free-viewpoint videos of dynamic scenes,
enabling immersive applications in VR/AR, telepresence,
∗Corresponding author.
and digital content creation.
Traditional geometry-based
pipelines, such as multi-view stereo [10, 34], mesh-based
capture [4, 6], and volumetric fusion [7], explicitly recover
scene geometry but require dense camera rigs and con-
trolled environments. Image-based view interpolation [23]
sidesteps explicit reconstruction, yet struggles to main-
tain geometric and temporal consistency.
Neural Radi-
ance Fields [3, 9, 19, 26–28, 30] alleviate these limitations
through continuous implicit representations that achieve
high rendering fidelity. However, their heavy computation
and slow rendering speed limit practical deployment for dy-
namic scenes.
Recently, 3D Gaussian Splatting (3DGS) [14] has
emerged as a highly efficient explicit representation, en-
abling real-time photorealistic rendering. Extending 3DGS
to dynamic scenes has led to two major paradigms.
Canonical-space deformation methods [21, 43, 47] learn
per-frame deformation fields that warp a static canoni-
cal representation to each timestamp. Although conceptu-
ally appealing, these approaches struggle with complex or
large-scale motions because of the difficulty of optimizing
high-dimensional deformation fields and maintaining tem-
poral coherence. On the other hand, motion-based meth-
ods like 4DGS [48], 4DRotorGS [8], STGS [20], and Free-
TimeGS [41] model the time-varying motion of Gaussian
primitives in 3D space.
However, the temporal visibility profile and motion for-
mulation for Gaussian primitives used in these methods
overlook the fundamental differences between static and
dynamic points. Specifically, for temporal visibility, ex-
isting methods use a Gaussian curve to model opacity
(Fig.1(a)).
However, its bell-shaped profile causes long-
lived primitives to decay gradually. As a result, represent-
ing a flat, time-invariant visibility requires multiple over-
lapping Gaussians to approximate the true curve, leading
optimization to repeatedly insert new primitives (Fig.1(a),
other primitives). For motion modeling, existing methods
neglect to fully model the relationship between velocity and
lifespan. As a consequence, a static primitive must learn an
arXiv:2602.02989v2  [cs.CV]  5 Feb 2026

<!-- page 2 -->
Figure 1. (a) Temporal visibility in existing motion-based meth-
ods. A step-like lifespan (blue line) requires multiple Gaussian
primitives for approximation. (b) With a learnable radius r, our
visibility function allows a single Gaussian primitive to represent
a step-like lifespan (blue line). (c) In existing motion-based meth-
ods (e.g., FreeTimeGS [41]), residual velocities accumulate over
time, causing drift in static regions. (d) With our lifepan modula-
tion term f(σt, r), where σt is the lifespan variance and r is the
lifespan radius, static primitive remain static without drift.
extremely small velocity to remain stable. However, it is
practically impossible for optimization to converge to an ab-
solute zero velocity, so even tiny residual motion inevitably
accumulates over long time periods and leads to noticeable
spatial drift and instability (Fig.1(c)). In essence, while a
behavior-agnostic formulation brings simplicity and consis-
tent optimization, it inevitably entangles static and dynamic
behaviors, making it difficult to represent both faithfully
within a single representation.
To address these challenges, we propose SharpTimeGS,
a novel 4D Gaussian representation that makes temporal be-
havior adaptive through a learnable lifespan, enabling bal-
anced modeling of long-term static structures and short-
lived dynamic motions without breaking the unified repre-
sentation. Our key insight is that the characteristics of a
primitive’s motion and temporal visibility are strongly tied
to its lifespan. We therefore introduce lifespan as a learn-
able per-primitive attribute and incorporate it directly into
the opacity and motion formulations. First, we reformu-
late temporal visibility by replacing the conventional Gaus-
sian decay with a flat-top profile that controls the lifespan.
This design eliminates the smooth, gradual decay charac-
teristic of Gaussian kernels, allowing primitives to main-
tain stable opacity within their active lifespan and to drop
off sharply once that lifespan ends (Fig. 1(b)), thereby re-
moving motion dragging and producing clearer temporal
boundaries. Second, by using lifespan as a modulation term
in motion formulation, long-lived primitives naturally sup-
press displacement and remain stable (Fig. 1(d)) because
the lifespan-induced scaling attenuates their effective mo-
tion, whereas short-lived primitives preserve expressive mo-
tions. This enables a unified motion formulation that bal-
ances static stability and dynamic flexibility.
Based on the lifespan modulation representation, we fur-
ther perform dynamic–static separation at initialization, as-
signing short lifespans and initial velocities to dynamic
primitives while keeping static ones long-lived and station-
ary, which significantly stabilizes optimization. Finally, to
balance optimization between static and dynamic regions,
we introduce a lifespan–velocity–aware densification strat-
egy. Each primitive is assigned a score reflecting its motion
magnitude relative to its lifespan, allowing short-lived, fast-
moving primitives to be cloned more frequently while long-
lived static ones remain compact. This adaptive allocation
assigns more representational capacity to dynamic regions
while keeping static areas stable and lightweight. Together,
these components yield a unified 4D Gaussian representa-
tion that remains stable over time and faithfully captures
dynamic motion.
Extensive experiments on Neural3DV[19], ENeRF-
Outdoor[22], and SelfCap [46] datasets demonstrate that
SharpTimeGS achieves state-of-the-art dynamic scene ren-
dering quality compared with existing methods. The contri-
butions of this work are summarized as follows:
• A lifespan-controlled flat-top visibility that avoids the
gradual Gaussian decay, thereby removing motion drag-
ging and producing sharper temporal boundaries.
• A lifespan-modulated motion formulation that balances
static stability and dynamic expressiveness.
• A velocity-aware initialization that separates dynamic
and static points and assigns lifespan–velocity priors ac-
cordingly, stabilizing the optimization process.
• A lifespan–velocity–aware densification strategy that pri-
oritizes dynamic regions while keeping static areas com-
pact and stable.
2. Related Work
NeRF-based
dynamic
NVS.
With
the
advent
of
NeRF [26] and differentiable rendering, neural scene
representations have become the mainstream paradigm for
dynamic scene reconstruction.
Extending static NeRF-
related representations to dynamic scenes are also being
actively explored [1, 3, 9, 13, 16, 19, 27–30, 35, 37–39].
Methods such as NeRFies [27], HyperNeRF [28], and
D-NeRF [30] construct a canonical NeRF and learn
per-frame deformation fields via multi-layer perceptrons
(MLPs). NeRFies [27] linked the observation space to a
canonical space via deformation fields. Neural3DV [19]
employs time-conditioned neural fields to directly represent
dynamics in 4D space, offering strong expressiveness but
incurring heavy computational and memory costs.
To
improve efficiency and scalability, hybrid representations
like K-Planes [9], HEX-Plane [3], and Tensor4D [35] com-
bine voxel grids with neural fields, significantly reducing
training and inference time. HEX-Plane [3] decomposes
the 4D domain into six feature planes, where point features
are sampled via interpolation and concatenated to predict

<!-- page 3 -->
density and color. Similarly, K-planes [9] offers a unified
approach for both static and dynamic scenes. Despite these
advances, NeRF-based methods still face challenges in
dynamic scene reconstruction, including slow rendering,
limited quality, and high storage overhead. In contrast, our
approach achieves superior rendering efficiency, quality,
and scalability, making it a practical solution for large-scale
dynamic scene reconstruction.
Gaussian-based dynamic NVS.
3D Gaussian Splat-
ting (3DGS) [14] has emerged as a competitive alter-
native for dynamic scenes, offering real-time render-
ing with sharp detail.
Approaches to modeling motion
with Gaussian Splatting can be broadly categorized into
two types: deformation-based and motion-based methods.
Deformation-based methods [2, 12, 15, 17, 21, 24, 31, 36,
44, 45, 47, 51] employ MLPs or low-rank K-planes to dy-
namically adjust the parameters of Gaussians over time.
While expressive, the continuous-deformation formulation
introduces overhead that hampers training speed and ren-
dering efficiency. Deformable-3DGS [47] applies forward
deformation, warping each Gaussian primitive from canon-
ical space to observation space before rendering. More re-
cent methods [8, 11, 18, 20, 25, 41, 46, 48] eschew de-
formations and directly model dynamics with 4D Gaus-
sian primitives.
In 4DGS [48] and 4DRotorGS [8], ge-
ometry and velocity are entangled, complicating joint op-
timization. The spatial and temporal scales are tightly cou-
pled, yielding complex convergence behavior. STGS [20]
employs polynomial motion with angular velocity, but the
high-order, high-dimensional parameterization is difficult
to optimize in complex regimes and prone to overfitting.
FreeTimeGS [41] adopts linear velocities but lacks a uni-
fied treatment of static versus dynamic regions, leading to
static jitter and loss of detail in fast motions. 7DGS [11]
introduces a unified representation over position, time, and
view direction by slicing 7D Gaussians into 3D subspaces,
similar to 4DGS. Nevertheless, it still struggles to model
motion-related dynamics.
3. Method
Given multi-view videos of a dynamic scene, our goal
is to reconstruct a temporally continuous 4D representa-
tion for novel view synthesis over time.
Fig. 2 demon-
strates the overall pipeline of SharpTimeGS. The frame-
work consists of a velocity-aware initialization that provides
motion priors (Sec. 3.3), a lifespan-modulated 4D Gaus-
sian representation that adjust temporal visibility and mo-
tion (Sec. 3.1), and a velocity–lifespan–aware densification
strategy (Sec. 3.2).
3.1. 4D Gaussians with Lifespan Modulation
The vanilla 3DGS [14] formulation models static scenes us-
ing a set of spatial Gaussian primitives, each parameterized
by its position X, scale S, rotation R, opacity O and view-
dependent color coefficients Y . To extend this represen-
tation to dynamics scenes, recent works [18, 20, 41, 48]
introduce a temporal dimension by assigning each primi-
tive a time anchor T and modeling its motion and opacity
as functions of the time offset ∆t = t −T. These for-
mulations currently omit explicit lifespan modeling, even
though both velocity and temporal opacity are inherently
lifespan-dependent. To address this limitation, we incorpo-
rate a lifespan parameter to enhance both the efficiency of
expression and optimization.
Lifespan-Modulated Motion Dynamics.
We observe
that a primitive’s motion behavior is inherently tied to its
temporal lifespan. Based on this, we formulate a lifespan-
modulated motion function that adaptively scales motion
amplitude by each primitive’s temporal persistence, defined
as:
Xt = X +
v
f(σt, r)(t −T),
f(σt, r) = 1.0 + max

1.0, (σt + r)2	
,
(1)
Here, f(σt, r) denotes the coupled lifespan that modulates
the effective motion strength. Specifically, σt represents
the lifespan variance, controlling how gradually a primitive
fades over time, and r defines its temporal radius, within
which it remains fully active. For static regions, σt + r is
large, making f(σt, r) →∞and thus v/f(σt, r) →0,
which effectively freezes the primitive’s position over time.
For dynamic regions with short lifespans, f(σt, r) becomes
small, allowing larger motion amplitudes that quickly adapt
to rapid changes. Both σt and r are learnable parameters
for each Gaussian primitive, allowing the model to adap-
tively adjust temporal behavior during optimization. This
proposed formulation naturally balances static stability and
dynamic flexibility, enabling both to be represented within
the same unified 4D Gaussian space.
Lifespan-Modulated Temporal Visibility.
We further
differentiate static and dynamic behaviors at the rendering
level by modulating opacity with the same lifespan parame-
ters r used in motion modeling. The key observation is that
most existing approaches still describe the lifespan function
using a Gaussian-shaped temporal profile. While this bell-
shaped formulation effectively models short-lived dynamic
events, it is suboptimal for long-term regions. Ideally, static
primitives should maintain a flat, time-invariant visibility
rather than a peaked Gaussian decay. To this end, we refor-
mulate the lifespan function into a flat-top profile that better
reflects long-term visibility, defined as:
Ot = O · l(t),
l(t) =





exp
 
−
|t −T| −r
σt
2!
,
|t −T| > r,
1,
|t −T| ≤r,
(2)

<!-- page 4 -->
…
…
…
Frame 1
Frame 2
Frame 3
Frame 0
…
…
Input View 1
Input View N
…
Pointcloud initialization
SfM for dynamic
SfM for dynamic
SfM for dynamic
SfM for dynamic
SfM for dynamic
SfM for dynamic
SfM for static
SfM for static
Dynamic Mask
Dynamic Mask
Supervision
Dynamic Mask
Dynamic Mask
Dynamic Mask
Dynamic Mask
MSE  𝓛୫ୱୣ
SSIM  𝓛𝒔𝒔𝒊𝒎
LPIPS  𝓛𝒍𝒑𝒊𝒑𝒔
Ground Truth
Rendering RGB
Depth-Normal 
Consistency
Rendering 
Depth
Rendering 
Normal
static
static
dynamic
dynamic
𝑿𝒕
𝒕
𝒕
𝑿𝒕
𝑿𝒕= 𝑿
𝑿𝒕= 𝑿+ 𝒗(𝒕−𝑻)
opacity
opacity
𝒓
𝑶𝒕
𝒕
Figure 2. The pipeline of our method. We represent a dynamic scene using Gaussian primitives whose temporal visibility adapts to
the actual lifespan of each point. To achieve this, we introduce a lifespan-dependent parameter r that modulates the temporal Gaussian,
allowing a single primitive to accurately model its full lifespan. Moreover, through the modulation terms f(σt, r) related to σt and r, the
static part can be completely static and still able to express dynamic parts (the static and fast dynamic regions will be transformed into
equations of motion in red and blue boxes, respectively). Note that the formulas in the boxes are only approximations. During optimization,
all Gaussian representations remain identical.
where Ot denotes the time-dependent opacity, and l(t) rep-
resents the lifespan modulation function. This design allows
static primitives to maintain a stable, time-invariant opac-
ity (large r), while dynamic primitives fade in and out over
shorter temporal spans (small r and σt), thus unifying both
behaviors within a single continuous formulation.
Based on the moved Gaussian primitive, we calculate its
color at position Xt through the spherical harmonics model:
Ct =
L
X
l=0
l
X
m=−l
Clm Ylm(d(Xt)) ,
(3)
where Ct is the color of Gaussian primitive at time t. L,
Clm, d(Xt), and Ylm(·) are the degree of spherical harmon-
ics, the spherical harmonics coefficients, the view direction
from the center of the camera to the position Xt, and the
spherical harmonics basis function, respectively.
Finally, we convert the 4D representation to 3D Gaussian
primitives at time t and render them following 3DGS [14].
3.2. Velocity-lifespan-aware densification.
Fast, complex regions are represented by short-lived, high-
velocity Gaussians that receive far fewer effective up-
dates during training than long-lived primitives, lead-
ing to blurred or under-detailed reconstructions in dy-
namic regions.
To alleviate this, we introduce a veloc-
ity–lifespan–aware densification strategy that adaptively ad-
justs Gaussian densification based on motion speed and
temporal persistence, achieving a balanced optimization be-
tween static and dynamic regions.
We adopt a two-stage training scheme to progressively
refine the 4D representation. In the initial stage (the first
1/3 of training iterations), we follow the densification pro-
cedure of AbsGS [49], which selects Gaussian primitives
based on both average and absolute average image gradients
and clones them following [33]. The first stage expands the
number of Gaussians to sufficiently cover the scene con-
tent. When this stage ends, the number of primitives at that
point is recorded as N, and kept fixed thereafter, while the
subsequent stage focuses on refining their spatial-temporal
distribution.
In the second stage, we remove primitives with low opac-
ity and clone new ones based on their velocity–lifespan be-
havior. To this end, we introduce a scoring metric s for each
Gaussian primitive, defined as:
s = λeE + λoO + λl

1 −exp

−∥v∥+ 1
f(σt, r)

.
(4)
This score integrates multiple factors: E denotes the ac-
cumulated reconstruction error from rendered and ground-
truth images, reflecting how well a primitive explains ob-
served data. The detailed computation of E is provided
in the supplementary material.
O represents the opacity
of the Gaussian primitive, encouraging preservation of vi-
sually significant regions. The last term prioritizes short-
lived, fast-moving primitives by assigning higher scores to
Gaussians with large motion magnitude and short lifespan,
effectively emphasizing fast and transient motions. λe, λo,
and λl are weights for these three components.
At each densification step, primitives with opacity be-
low a small threshold are removed. The same number of
new Gaussians are cloned from the top-ranked primitives
according to their scores s. This replacement strategy adap-

<!-- page 5 -->
tively allocates more representational capacity to transient
motions to reconstruct details.
3.3. Velocity-aware initialization.
A well-designed initialization is crucial for stabilizing 4D
Gaussian optimization, especially in dynamic scenes. To
provide physically meaningful priors for both spatial and
temporal attributes, we design a velocity-aware initializa-
tion that separately handles dynamic and static regions.
For the dynamic region, we first identify moving objects
by computing optical flow using RAFT [40]. The detected
motion points are served as prompts for SAM2 [32] to ob-
tain complete object masks. Given these masks and cam-
era parameters, we reconstruct per-frame 3D point clouds
of moving objects using COLMAP [34]. Then, the corre-
sponding points across adjacent frames are matched via K-
nearest neighbors (KNN), and their 3D displacements de-
fine the initial velocity vinit of Gaussian primitives. Each
extracted dynamic point cloud then initializes standard 3D
Gaussian attributes, including position X, scale S, rotation
R, opacity O, and SH coefficients Y , following 3DGS [14].
The temporal parameters are assigned as follows: the time
T corresponds to the current frame, the velocity v is initial-
ized as the estimated motion vinit, the lifespan variance σt is
set to cover three frames, and the temporal radius r is ini-
tialized to 1e −6 to keep the initial temporal visibility close
to the Gaussian distribution.
For the static region, we use COLMAP [34] with full im-
ages to reconstruct the first frame, which includes both dy-
namic and static points (the wrong dynamic points will be
removed during the optimization), to initialize long-lived
Gaussian primitives. These Gaussians are initialized with
zero velocity, a time T corresponding to the middle of the
sequence, an extended lifespan variance σt lasting three
times the total number of frames, and a lifespan radius ini-
tialize to 1e −6.
3.4. Training
The dynamic scene is trained under a reconstruction loss
Lrecon, a regularization Lreg, and an auxiliary loss Le for
densification (Le is related to E, which will be introduced
in the supplementary material):
L = Lrecon + Lreg + Le.
(5)
Reconstruction loss. The L1 loss L1, SSIM loss [42] Ls
and perceptual loss [50] Lp are applied to the rendered im-
ages ˜I to measure the difference between the ground-truth
images Igt:
Lrecon = λ1L1(˜I, Igt) + λsLs(˜I, Igt) + λpLp(˜I, Igt), (6)
in which we set λ1 = 0.8, λs = 0.2 and λp = 0.01, respec-
tively.
Regularization. In order to further improve the quality of
reconstruction, similar to PGSR [5], we introduce Lscale to
make Gaussian primitives flatten as much as possible, while
enhancing single view normal and depth consistency con-
straints with Ln. All items of regularization are as follows:
Lreg = λscaleLscale + λopacityLopacity + λnLn + λtLt.
(7)
Lt = 1
N
X
1
p
−2 log(oth)σ2
t + r
,
Lopacity = 1
N
X
O · /∇[l(t)] ,
(8)
where Lt extends the lifespan of Gaussian primitives, en-
couraging reuse of the same primitive rather than fragment-
ing it into multiples. N is the number of Gaussian primi-
tives, and /∇[·] is the stop-gradient operation. oth denotes the
truncation threshold, and Gaussian primitives with opacity
Ot < oth at time t are excluded from rendering. We add the
opacity loss Lopacity at the second densification stage and
stop resetting opacities [14] to stabilize convergence.
4. Experiments
4.1. Experimental Settings
Datasets.
We evaluate our method on three widely used
dynamic-scene benchmarks:
Neural3DV [19], ENeRF-
Outdoor [22], and SelfCap [41]. Neural3DV contains six
indoor scenes captured by 19–21 cameras at a resolution
of 2704 × 2028 and 30 FPS. Following [41], we use the
first 300 frames of each scene and downsample images by
a factor of 0.5. ENeRF-Outdoor [22] provides three out-
door sequences recorded with 18 synchronized cameras at
1920 × 1080 and 60 FPS. We use the first 300 frames with-
out resizing. SelfCap [41] contains fast-motion sequences;
we use six scenes (60 frames each), resizing images by 0.5
except for the bike scene, which is kept at full resolution.
Metrics.
We evaluate rendering quality using three
widely adopted image-based metrics: Peak Signal-to-Noise
Ratio (PSNR), Structural Similarity Index (SSIM) [42], and
Learned Perceptual Image Patch Similarity (LPIPS) [50].
PSNR and SSIM measure pixel-level fidelity and structural
consistency, while LPIPS assesses perceptual similarity us-
ing deep features, providing a more human-aligned evalua-
tion of visual quality.
Baselines.
We compare our method with well-established
dynamic 4D Gaussian baselines, including the canonical-
deformation
based
method
Deformable-3DGS
[47]
and motion-based approaches such as Ex4DGS [18],
4DGS [48], STGS [20], and FreeTimeGS [41].
For
FreeTimeGS, we reproduce its pipeline for qualitative
comparison, while quantitative results are taken directly
from the original paper.

<!-- page 6 -->
GT
Ours
FreeTimeGS
STGS
4DGS
GT
Ours
FreeTimeGS
STGS
4DGS
Figure 3. Qualitative comparison on the SelfCap Dataset [46]. Our method achieves the rendering quality compared with baseline methods,
especially for distant static regions (e.g., books and wall) and fast-moving dynamic regions (e.g., hairs and ball).
4.2. Results and Comparisons
Quantitative results on Neural3DV, ENeRF-Outdoor and
SelfCap datasets are reported in Tab. 1. Our method con-
sistently achieves the best performance across all metrics
on all three benchmarks. Qualitative comparisons (Fig. 4
and Fig. 3, and the supplementary material) further demon-
strate high-fidelity reconstructions with sharper dynamics
and better-preserved static details.
Across datasets, baseline methods face a recurring trade-
off in optimization, making it difficult to balance static
background fidelity and dynamic reconstruction quality.

<!-- page 7 -->
Figure 4. Qualitative comparison on the ENeRF-Outdoor Dataset [22]. Our method achieves the best rendering quality compared with
baseline methods, especially for distant static regions and fast-moving dynamic regions.
Figure 5. Ablation study on the SelfCap Dataset [46]. Our full model achieves the best rendering quality, especially for distant static
regions and fast-moving dynamic regions.
Due to its fully coupled representation, 4DGS [48] of-
ten struggles to converge for rapidly moving content (e.g.,
ball in SelfCap and watermelon in ENeRF-Outdoor) lead-
ing to artifacts and blurred details.
Similarly, the com-
plex high-order formulation of STGS [20] complicates op-
timization and prevents full convergence (e.g., face and toy
in ENeRF-Outdoor). FreeTimeGS [41] overlooks the de-
pendency between velocity and lifespan, yielding an overly
unconstrained parameterization that degrades static fidelity
(e.g., wall in ENeRF-Outdoor and books in SelfCap) and in-
troduces an optimization imbalance between static and dy-
namic regions, leaving dynamic elements under-converged
(e.g., toy in ENeRF-Outdoor and hair/skin/ball in SelfCap).
In contrast, our method employs lifespan modulation to re-
tain the strong static modeling capability of 3DGS [14],
thereby preserving high-fidelity backgrounds. Meanwhile,
velocity-aware initialization together with velocity-lifespan
aware densification improves convergence and allocates
sufficient capacity to dynamic regions, enabling robust re-
construction of fast and complex motions across all bench-
marks. Moreover, the extended duration of the static seg-
ment maintains a stable background without noticeable
flicker. The specific temporal results and the free-viewpoint
video can be found in the supplementary video.

<!-- page 8 -->
Table 1. Quantitative comparison on Neural3DV [19] Dataset, ENeRF-Outdoor [22] Dataset, and SelfCap [46] Dataset. We report PSNR,
SSIM2 [42], and LPIPS [50] to evaluate the rendering quality. Values in boldface denote the best result in the corresponding column.
Method
Neural3DV
ENeRF-Outdoor
SelfCap
PSNR ↑
SSIM2 ↑
LPIPS ↓
PSNR ↑
SSIM2 ↑
LPIPS ↓
PSNR ↑
SSIM2 ↑
LPIPS ↓
Deformable-3DGS [47]
31.15
0.970
0.049
24.26
0.801
0.318
25.85
0.920
0.312
Ex4DGS [18]
32.11
0.970
0.048
24.89
0.817
0.305
24.96
0.920
0.299
4DGS [48]
32.01
0.972
0.055
24.82
0.822
0.317
25.86
0.923
0.245
STGS [20]
32.05
0.972
0.044
24.93
0.818
0.297
24.77
0.894
0.291
FreeTimeGS [41]
33.19
0.974
0.036
25.36
0.846
0.244
27.50
0.951
0.201
Ours
33.57
0.977
0.031
25.82
0.872
0.233
28.14
0.960
0.192
Table 2. Ablation study on SelfCap [46] Dataset (Partial). We
report PSNR, SSIM2, and LPIPS to evaluate the rendering quality.
Method
PSNR ↑
SSIM2 ↑
LPIPS ↓
w/o our representation
25.96
0.907
0.299
w/o lifespan r
26.76
0.927
0.321
w/o our densification
26.82
0.919
0.317
w/o our initialization
26.83
0.927
0.297
full model
27.36
0.947
0.244
4.3. Ablation Studies
To verify the effectiveness of our 4D representation and our
densification strategy, we conduct independent experiments
for each component while keeping the others unchanged.
We then evaluate the quantitative metrics and present the
qualitative results. As shown in Tab. 2, the performance
decreases when removing any of the modules we proposed.
Effectiveness of our 4D representation. To evaluate our
4D representation, we replace it with the coupled represen-
tation used in 4DGS [48] and report the results in Fig. 5
and Tab. 2. The coupled baseline exhibits artifacts on fast-
moving thin structures as well as in background regions.
Since motion and appearance are entangled in the cou-
pled parameterization, optimization becomes unstable un-
der rapid motion, which leads to local non-convergence and
produces artifacts (e.g., hairs and bicycle spokes). With de-
coupled 4D representation, artifacts are largely suppressed,
yielding sharper dynamic details and cleaner static regions.
Effectiveness of our temporal visibility representation.
To evaluate our temporal visibility representation, we use
the original temporal visibility representation and compare
the results in Fig. 5 and Tab. 2. Using the baseline visibility
leads to temporal blur on moving details (e.g., hairs) and
oversmoothing on long-lived structures (e.g., potted plant).
Compared to the smooth Gaussian-shaped visibility in the
baseline, our flat-top visibility profile with steep falloff at
the lifespan boundaries reduces temporal mixing and yields
sharper dynamic details and quasi-static regions.
Effectiveness of our densification. To verify the effective-
ness of velocity-lifespan-aware densification, we design an
experiment in which we use the 4DGS [48] densification
and compare results in Fig. 5 and Tab. 2. Because the orig-
inal densification in 4DGS [48] does not account for lifes-
pan and velocity, Gaussian primitives undergo an uneven
number of updates across fast/slow and short-/long-lived re-
gions, leading to imbalanced optimization. Fast, short-lived
components are prone to non-convergence, whereas slow,
long-lived components tend to overfit. Consequently, arti-
facts arise across all cases. In contrast, our densification
adapts the update frequency based on velocity and lifespan,
ensuring that fast, short-lived regions are sufficiently opti-
mized while slow, long-lived regions avoid overfitting.
Effectiveness of velocity-aware initialization. To verify
the effectiveness of velocity-aware initialization, we design
an experiment in which we do not separate the static and
dynamic region and only use point cloud to initialize the
Gaussian primitive parameter without velocity (i.e., the ve-
locity is set to 0). We compare the results in Fig. 5 and
Tab. 2.
It can be seen that both dynamic and static re-
gions have artifacts (e.g., hairs and potted plant). Therefore,
our velocity-aware initialization can improve the ability to
model the static region.
5. Discussion
We introduced SharpTimeGS, a unified 4D Gaussian rep-
resentation driven by a learnable lifespan that modulates
motion and temporal visibility. This design stabilizes long-
lived primitives, preserves high-frequency dynamic motion,
and eliminates motion dragging through a flat-top tempo-
ral kernel.
Combined with lifespan–velocity–aware den-
sification and velocity-guided initialization, SharpTimeGS
achieves sharper reconstructions and superior temporal con-
sistency.
Experiments on diverse dynamic-scene bench-
marks confirm clear improvements over prior 4D Gaussian
methods, while retaining real-time rendering efficiency.
Limitation. Our reconstruction system is currently not real-
time, requiring several hours to convert multi-view videos
into the proposed 4D representation. Future improvements
could focus on accelerating training through stronger geo-
metric priors or regularizing the spatial distribution of Gaus-
sian primitives. Moreover, the current representation targets
novel view synthesis and does not yet support relighting.
However, since our method provides a strong geometric rep-
resentation, future work can readily enable relighting by in-
corporating additional material and reflectance properties.

<!-- page 9 -->
References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim.
Hyperreel:
High-fidelity 6-dof video with ray-
conditioned sampling. In CVPR, pages 16610–16620, 2023.
2
[2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun
Bang, and Youngjung Uh. Per-gaussian embedding-based
deformation for deformable 3d gaussian splatting. In ECCV,
pages 321–335. Springer, 2024. 3
[3] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In CVPR, pages 130–141, 2023. 1,
2
[4] Dan Casas, Marco Volino, John Collomosse, and Adrian
Hilton. 4d video textures for interactive character appear-
ance. In Computer Graphics Forum, pages 371–380. Wiley
Online Library, 2014. 1
[5] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. IEEE
TVCG, 2024. 5
[6] Alvaro Collet, Ming Chuang, Pat Sweeney, Don Gillett, Den-
nis Evseev, David Calabrese, Hugues Hoppe, Adam Kirk,
and Steve Sullivan. High-quality streamable free-viewpoint
video. ACM TOG, 34(4):1–13, 2015. 1
[7] Mingsong Dou, Sameh Khamis, Yury Degtyarev, Philip
Davidson, Sean Ryan Fanello, Adarsh Kowdle, Sergio Orts
Escolano, Christoph Rhemann, David Kim, Jonathan Taylor,
et al. Fusion4d: Real-time performance capture of challeng-
ing scenes. ACM TOG, 35(4):1–13, 2016. 1
[8] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
towards efficient novel view synthesis for dynamic scenes.
In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,
2024. 1, 3
[9] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
CVPR, pages 12479–12488, 2023. 1, 2, 3
[10] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and ro-
bust multiview stereopsis. IEEE TPAMI, 32(8):1362–1376,
2009. 1
[11] Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa
Choudhuri, Terrence Chen, and Ziyan Wu. 7dgs: Unified
spatial-temporal-angular gaussian splatting. arXiv preprint
arXiv:2503.07946, 2025. 3
[12] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for effi-
cient dynamic scene reconstruction. IEEE TCSVT, 2024. 3
[13] Mustafa Is¸ık, Martin R¨unz, Markos Georgopoulos, Taras
Khakhulin, Jonathan Starck, Lourdes Agapito, and Matthias
Nießner. Humanrf: High-fidelity neural radiance fields for
humans in motion. ACM TOG, 42(4):1–12, 2023. 2
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM TOG, 2023. 1, 3, 4, 5, 7
[15] Mijeong Kim, Jongwoo Lim, and Bohyung Han. 4d gaus-
sian splatting in the wild with uncertainty-aware regulariza-
tion. Advances in Neural Information Processing Systems,
37:129209–129226, 2024. 3
[16] Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee, Gun
Bang, and Youngjung Uh. Sync-nerf: Generalizing dynamic
nerfs to unsynchronized videos. In AAAI, pages 2777–2785,
2024. 2
[17] Isaac Labe, Noam Issachar, Itai Lang, and Sagie Benaim.
Dgd: Dynamic 3d gaussians distillation. In European Con-
ference on Computer Vision, pages 361–378. Springer, 2024.
3
[18] Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae,
and Hae-Gon Jeon. Fully explicit dynamic gaussian splat-
ting. Advances in Neural Information Processing Systems,
37:5384–5409, 2024. 3, 5, 8
[19] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
CVPR, pages 5521–5531, 2022. 1, 2, 5, 8
[20] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In CVPR, pages 8508–8520, 2024. 1, 3, 5, 7, 8
[21] Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-
Phuoc, Douglas Lanman, James Tompkin, and Lei Xiao.
Gaufre: Gaussian deformation fields for real-time dynamic
novel view synthesis. pages 2642–2652. IEEE, 2025. 1, 3
[22] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai,
Hujun Bao, and Xiaowei Zhou.
Efficient neural radiance
fields for interactive free-viewpoint video. In SIGGRAPH
Asia 2022 Conference Papers, pages 1–9, 2022. 2, 5, 7, 8
[23] Christian Lipski, Christian Linz, Kai Berger, and Marcus
Magnor. Virtual video camera: Image-based viewpoint nav-
igation through space and time. In SIGGRAPH’09: Posters,
pages 1–1, 2009. 1
[24] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Min Yang,
Xiao Tang, Feng Zhu, and Yuchao Dai. 3d geometry-aware
deformable gaussian splatting for dynamic view synthesis.
In CVPR, pages 8900–8910, 2024. 3
[25] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In International Conference
on 3D Vision (3DV), 2024. 3
[26] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2
[27] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In ICCV, pages 5845–5854, 2021. 2
[28] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T.
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M. Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. ACM TOG, 40(6), 2021. 1, 2

<!-- page 10 -->
[29] Sida Peng, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xi-
aowei Zhou.
Representing volumetric videos as dynamic
mlp maps. In CVPR, pages 4252–4262, 2023.
[30] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance fields for
dynamic scenes. In CVPR, pages 10318–10327, 2021. 1, 2
[31] LIU Qingming, Yuan Liu, Jiepeng Wang, Xianqiang Lyu,
Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dy-
namic gaussian splatting from casually-captured monocular
videos with depth priors. In ICLR, 2025. 3
[32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, et al.
Sam 2:
Segment anything in images and videos.
arXiv preprint
arXiv:2408.00714, 2024. 5
[33] Samuel Rota Bul`o, Lorenzo Porzi, and Peter Kontschieder.
Revising densification in gaussian splatting. In ECCV, pages
347–362. Springer, 2024. 4
[34] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In CVPR, pages 4104–4113, 2016. 1,
5
[35] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural
4d decomposition for high-fidelity dynamic reconstruction
and rendering. In CVPR, pages 16632–16642, 2023. 2
[36] Richard Shaw,
Michal Nazarczuk,
Jifei Song,
Arthur
Moreau, Sibi Catley-Chandar, Helisa Dhamo, and Eduardo
P´erez-Pellitero. Swings: sliding windows for dynamic 3d
gaussian splatting. In ECCV, pages 37–54. Springer, 2024.
3
[37] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger.
Nerf-
player: A streamable dynamic scene representation with
decomposed neural radiance fields.
IEEE TVCG, 29(5):
2732–2742, 2023. 2
[38] Cong Wang, Di Kang, Yanpei Cao, Linchao Bao, Ying Shan,
and Song-Hai Zhang. Neural point-based volumetric avatar:
Surface-guided neural points for efficient and photorealistic
volumetric head avatar. In ACM SIGGRAPH Asia 2023 Con-
ference Proceedings, 2023.
[39] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei
Song, and Huaping Liu. Mixed neural voxels for fast multi-
view video synthesis. In CVPR, pages 19706–19716, 2023.
2
[40] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple,
efficient, accurate raft for optical flow. In ECCV, pages 36–
54. Springer, 2024. 5
[41] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhan-
hua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei
Zhou. Freetimegs: Free gaussian primitives at anytime any-
where for dynamic scene reconstruction. In CVPR, pages
21750–21760, 2025. 1, 2, 3, 5, 7, 8
[42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. IEEE TIP, 13(4):600–612, 2004. 5,
8
[43] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In CVPR, pages 20310–20320, 2024. 1
[44] Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d
decomposed hash encoding for high-fidelity dynamic gaus-
sian splatting. Advances in Neural Information Processing
Systems, 37:123787–123811, 2024. 3
[45] Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming
Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou.
4k4d:
Real-time 4d view synthesis at 4k resolution. In CVPR, pages
20029–20040, 2024. 3
[46] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun,
Hujun Bao, and Xiaowei Zhou. Representing long volumet-
ric video with temporal gaussian hierarchy. ACM TOG, 43
(6):1–18, 2024. 2, 3, 6, 7, 8
[47] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction. In CVPR,
2024. 1, 3, 5, 8
[48] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting. In ICLR, 2024. 1, 3, 5, 7, 8
[49] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong
Dou. Absgs: Recovering fine details in 3d gaussian splat-
ting. In ACM MM, pages 1053–1061, 2024. 4
[50] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In CVPR, pages 586–
595, 2018. 5, 8
[51] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng,
Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong
Zhang.
Motiongs:
Exploring explicit motion guidance
for deformable 3d gaussian splatting.
In NeurIPS, pages
101790–101817, 2024. 3
