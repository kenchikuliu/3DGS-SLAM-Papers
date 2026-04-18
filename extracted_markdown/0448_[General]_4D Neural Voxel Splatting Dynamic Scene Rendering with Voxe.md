<!-- page 1 -->
4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian
Splatting
Chun-Tin Wu1, Jun-Cheng Chen2
1National Taiwan University, 2Academia Sinica
Ours
Training Time: 12 Mins
FPS:45
Memory Usage: 3050 MiB
TiNeuVox
Training Time: 18 Mins
FPS:1
Memory Usage: 21,558 MiB
4DGS
Training Time: 23 Mins
FPS:34
Memory Usage: 4500 MiB
Figure 1. Our approach demonstrates remarkable memory efficiency and training speed, while achieving superior image quality
Abstract
Although 3D Gaussian Splatting (3D-GS) achieves efficient
rendering for novel view synthesis, extending it to dynamic
scenes still results in substantial memory overhead from
replicating Gaussians across frames. To address this chal-
lenge, we propose 4D Neural Voxel Splatting (4D-NVS),
which combines voxel-based representations with neural
Gaussian splatting for efficient dynamic scene modeling.
Instead of generating separate Gaussian sets per times-
tamp, our method employs a compact set of neural voxels
with learned deformation fields to model temporal dynam-
ics. The design greatly reduces memory consumption and
accelerates training while preserving high image quality.
We further introduce a novel view refinement stage that se-
lectively improves challenging viewpoints through targeted
optimization, maintaining global efficiency while enhanc-
ing rendering quality for difficult viewing angles. Exper-
iments demonstrate that our method outperforms state-of-
the-art approaches with significant memory reduction and
faster training, enabling real-time rendering with superior
visual fidelity.
1. Introduction
Dynamic 3D scene reconstruction and novel view synthe-
sis are recent research hotspots in computer vision with a
widespread applications in virtual reality (VR), augmented
reality (AR), and digital entertainment. Capturing and ren-
dering temporal phenomena—such as human motion or de-
formable objects—is essential for immersive experiences.
While Neural Radiance Fields (NeRF) [12] achieve pho-
torealistic view synthesis in static scenes, extending them
to dynamics faces key bottlenecks: long training times,
slow inference, and poor scalability with temporal com-
plexity. Acceleration methods like DirectVoxGO [20] and
Instant-NGP [13] improve efficiency but remain limited by
volumetric rendering. 3D Gaussian Splatting (3D-GS) [4]
shifts toward explicit point-based rendering, enabling real-
time performance, yet naive extensions to dynamics suf-
fer from memory growth proportional to sequence length.
Canonical-space methods like 4D-GS [24] reduce this but
rely on heavy deformation networks and costly backward
mapping. Recent advances attempt to balance efficiency
and expressiveness: Ex4DGS [5] improves temporal con-
sistency via explicit motion modeling, FreeTimeGS [23]
decouples space and time for flexible representation, and
LongVolCap [25] compresses long sequences effectively.
However, the practical deployment of these methods re-
mains constrained by memory and training overhead, espe-
cially in the scenarios of embodied AI which has a critical
demand for the resource consumption. For instance, Mo-
bile robots on edge devices (e.g., NVIDIA Jetson with 4–
8GB memory) must rapidly reconstruct and track dynamic
scenes for navigation and interaction. Field robots likewise
require scene adaptation within minutes, not hours, to oper-
ate effectively. Current methods often exceed available re-
sources on edge or mid-range GPUs, forcing compromises
arXiv:2511.00560v1  [cs.CV]  1 Nov 2025

<!-- page 2 -->
in scene complexity or hardware cost. The core challenge
is thus: how to capture rich temporal dynamics while main-
taining memory and computational efficiency for real-time,
resource-constrained applications.
To address these limitations, we introduce 4D Neural
Voxel Splatting (4D-NVS), which combines voxel-based ef-
ficiency with Gaussian splatting quality through learned
temporal deformation. Our key insight is decoupling spa-
tial structure from temporal dynamics: we generate neural
Gaussians on-demand from persistent voxel anchors and ap-
ply unified deformations. This achieves O(fV + F) mem-
ory complexity compared to traditional O(N · T) scaling,
where V is the number of voxels, f is the feature dimension
per voxel, F represents the deformation network parame-
ters, N is the number of Gaussians, and T is the number
of timestamps.s While our method builds upon established
techniques (voxel-based generation from Scaffold-GS [10]
and HexPlane decomposition [2]), we introduce several key
innovations that distinguish our approach:
• Unified 4D Voxel Architecture: We extend 3D voxel
grids to 4D by treating time as an additional dimension in
the voxel feature space, enabling temporal-aware Gaus-
sian generation that adapts both spatially and temporally.
Unlike Scaffold-GS which generates static Gaussians, our
voxels produce time-varying Gaussians through learned
temporal features.
• Selective Deformation Strategy: Through extensive ex-
perimentation, we identified that deforming all Gaussian
properties leads to training instability. We introduce a
selective approach that only deforms geometric proper-
ties (position, scale, rotation) while keeping appearance
properties (color, opacity) fixed, significantly improving
convergence and quality.
• View-Adaptive Refinement: We propose a novel refine-
ment mechanism that identifies and selectively improves
underperforming viewpoints through adaptive densifica-
tion, addressing temporal inconsistencies without global
overhead.
• Memory-Efficient Design:
Our framework achieves
O(fV + F) memory complexity instead of O(N · T).
This makes dynamic scene rendering feasible on con-
sumer GPUs.
To sum up, our 4D Neural Voxel Splatting successfully
bridges computational efficiency and visual quality, en-
abling new possibilities for real-time dynamic visualization
and interactive scene manipulation.
2. Related Works
In this section, we briefly review the relevant works of
NeRF and Rasterization based methods for scene rendering.
2.1. NeRF-based Methods
Neural Radiance Fields (NeRF) [12] revolutionized novel
view synthesis by representing scenes with MLPs that out-
put view-dependent color and density. While achieving im-
pressive visual fidelity, early NeRF models suffered from
slow rendering due to costly volumetric ray marching.
To address this, researchers introduced explicit representa-
tions like Plenoxels [16], which replaced MLPs with sparse
voxel grids, and Instant-NGP [13], which employed multi-
resolution hashed grids for faster training and inference.
For dynamic scenes, methods like D-NeRF input time
directly into the MLP, while more sophisticated approaches
such as Nerfies [14] and NSFF [7] use explicit deformation
networks to map points from canonical space to each frame.
K-Planes [17] factorizes radiance fields into low-rank grids
across space and time dimensions.
Despite these ad-
vances, NeRF-based dynamic methods often face optimiza-
tion challenges and increased memory requirements, strug-
gling to capture complex non-rigid motions efficiently.
2.2. Rasterization Methods
3D Gaussian Splatting (3D-GS) [4] emerged as an efficient
alternative, representing scenes with colored 3D Gaussian
primitives rendered via projective splatting rather than vol-
umetric rendering. This approach achieves real-time per-
formance while maintaining high visual quality. Grid-based
extensions like Scaffold-GS [10] generate neural Gaussians
from anchor points, while SVRaster [21] uses sparse voxel-
based rendering.
For dynamic scenes, 4D Gaussian Splatting (4D-GS)
[24] extends 3D-GS with learned deformation fields to
transform canonical Gaussians over time, avoiding the
memory overhead of per-frame Gaussian sets. Other ap-
proaches include GaGS [11], which uses geometry-aware
architectures for deformation modeling, and DynMF [8],
which factorizes motion into trajectory bases.
However,
these methods either suffer from prolonged optimization
times or demand excessive memory.
Recent advances include Ex4DGS [5] with explicit mo-
tion modeling, but requires complex trajectory computa-
tion increasing training overhead. FreeTimeGS [23] intro-
duces temporal flexibility but struggles with memory scal-
ing for long sequences. Long Volcap [25] addresses du-
ration through compression yet sacrifices real-time perfor-
mance due to decompression costs.
While these meth-
ods target specific challenges, they introduce new compu-
tational bottlenecks that limit practical deployment.
In contrast to prior approaches, our method balances
training efficiency and memory usage by integrating com-
pact 4D neural voxel representations with selective property
deformation, enabling accelerated convergence and real-
time dynamic scene rendering with lower memory foot-
print.

<!-- page 3 -->
HexPlane Encoder
Forward
Rendering
Backward
Neural Gaussians
O1
O2
O3
F
F
F
F
α
c
s
r
Visible Neural Voxels at time t
Deformed Gaussians
φ
φ
φ
x
r
s
Δs
Δr
Δx, Δy, Δz
View densification
of visible crude
voxels at time t
Rendered Image
Neural Gaussians
Deformed Gaussians
GT
HexPlane Encoder
Rendered
Image
4D Neural Voxel Splatting
Visible Voxel
Invisible Voxel
GT
View Refinement
Time t
Crude View
Voxel
Densification
Figure 2. Pipeline overview: (1) Initialize with voxel-based Gaussian splatting, (2) Generate neural Gaussians with temporal information,
(3) Apply HexPlane temporal corrections, (4) Optimize with color loss, total variation loss, and scaling regularization, (5) View refinement
stage for underperforming viewpoints through adaptive densification.
3. Preliminaries
3.1. 3D Gaussian Splatting
3D Gaussian Splatting (3D-GS) [4] represents scenes using
anisotropic 3D Gaussians, each defined by a mean position
µ ∈R3 and covariance matrix Σ:
G(x) = e−1
2 (x−µ)⊤Σ−1(x−µ),
(1)
where the covariance matrix is constructed as Σ
=
RSST RT using scaling matrix S and rotation matrix R.
Each Gaussian has associated color c and opacity α (mod-
eled with spherical harmonics) .
Unlike volumetric rendering, 3D-GS projects Gaussians
onto the image plane as 2D Gaussians and applies α-
blending:
C(x′) =
N
X
i=1
ciαi
i−1
Y
j=1
(1 −αj),
(2)
where x′ is the pixel position, α depends on the pixel posi-
tion x′ and N is the number of overlapping Gaussians. This
tile-based rasterization enables real-time rendering with dif-
ferentiable optimization.
3.2. Scaffold-GS: Grid-Based Gaussian Generation
Scaffold-GS [10] generates Gaussians from structured an-
chor points rather than optimizing individual primitives.
Starting from a sparse point cloud P ∈RM×3, it creates
voxel centers:
V =
P
ϵ

· ϵ,
(3)
where ϵ is the voxel size. Each voxel center v ∈V serves
as an anchor with local features fv, scaling factor lv ∈R3,
and k learnable offsets Ov ∈Rk×3.
Neural Gaussians are generated on-demand within the
viewing frustum. For each visible anchor, k Gaussians are
spawned with positions µi = xv + Oi · lv, while attributes
(opacity, color, scale, rotation) are decoded from anchor
features using MLPs conditioned on viewing direction and
distance. This approach significantly reduces memory re-
quirements compared to storing explicit Gaussians for the
entire scene.
4. Method
4.1. Overview
Our 4D Neural Voxel Splatting (4D-NVS) framework ad-
dresses the challenge of efficiently rendering dynamic
scenes by extending traditional 3D voxel representations
into the temporal domain. The core insight is that while 3D
voxels effectively capture spatial structure, dynamic scenes
require an additional temporal dimension. We achieve this
through a three-stage training approach:
1. Coarse Initialization.
We initialize the spatial voxel
structure using mixed timestamps to establish a founda-
tional geometric representation. This stage focuses on
learning the basic scene structure without temporal de-
formation.
2. Fine Temporal Training.
We activate the HexPlane
deformation module and train the full temporal model.
During this stage, we also identify poorly reconstructed
viewpoints for subsequent refinement.
3. View Refinement. We focus training exclusively on the
identified crude viewpoints using more aggressive den-
sification parameters, improving quality in challenging
regions without affecting well-performing areas.
The key insight behind our method is the decoupling of

<!-- page 4 -->
spatial representation from temporal dynamics. Rather than
storing explicit primitives for each frame, we generate neu-
ral Gaussians on-demand from a persistent voxel grid and
apply learned temporal deformations.
4.2. Initialization and Voxel Setup
We initialize neural voxels using the sparse point cloud from
Structure-from-Motion (SfM) [18], distributing voxel cen-
ters to ensure comprehensive spatial coverage. The scene is
first trained using Scaffold-GS [10] for 3K iterations with
mixed timestamp samples to establish a foundational geo-
metric representation.
4.3. Neural Gaussian Generation
On-Demand Generation.
Unlike existing methods that
pre-compute and store Gaussians, our approach generates
neural Gaussians dynamically based on viewing conditions
and temporal context. This represents a fundamental shift
from static storage to dynamic computation, enabling both
memory efficiency and view-dependent optimization.
At each timestamp t, we perform visibility culling to
identify voxels within the camera frustum, significantly re-
ducing computational overhead from O(N) to O(Vvisible)
where Vvisible ≪N. For each visible voxel v, we generate
k neural Gaussians with positions:
µi = xv + Oi · lv,
i = 0, . . . , k −1,
(4)
where xv ∈R3 is the voxel center position, Oi ∈R3 are
learnable offset vectors for each of the k Gaussians, and
lv ∈R3 is the voxel’s spatial extent (scale factor).
Gaussian attributes are decoded using dedicated MLPs
that take anchor features fv, viewing distance δvc, and di-
rectional vector ⃗dvc as input:
{α0, . . . , αk−1} = Fα(fv, δvc, ⃗dvc),
(5)
{c0, . . . , ck−1} = Fc(fv, δvc, ⃗dvc),
(6)
{s0, . . . , sk−1} = Fs(fv, δvc, ⃗dvc),
(7)
{r0, . . . , rk−1} = Fr(fv, δvc, ⃗dvc),
(8)
where Fα, Fc, Fs, and Fr are MLPs for opacity, color, scale,
and rotation respectively.
4.4. Dynamic Gaussians Deformation
Unified 4D Representation.
Our method introduces a
novel temporal modeling approach that treats space and
time as a unified 4D manifold while maintaining compu-
tational efficiency.
Unlike previous methods that either
use separate deformation networks for each Gaussian or
employ expensive per-frame optimization, we leverage a
shared HexPlane decomposition that captures temporal cor-
relations across the entire scene.
To capture dynamic motion, we employ HexPlane [17]
decomposition, encoding 4D space into six planes with
multi-resolution temporal features fh. Specifically, Hex-
Plane factorizes the 4D space-time volume into six 2D
planes: three spatial planes (XY, XZ, YZ) and three space-
time planes (XT, YT, ZT). For a 4D point (x, y, z, t), we
extract features via bilinear interpolation from each plane
and aggregate them:
fh =
X
p∈{XY,XZ,YZ,XT,YT,ZT}
Interpp(x, y, z, t).
(9)
This decomposition provides several advantages over di-
rect 4D parameterization: (1) significantly reduced memory
footprint (2) natural handling of temporal correlations, and
(3) efficient gradient flow during optimization. A compact
MLP ϕd integrates these features:
fd = ϕd(fh),
(10)
where ϕd is a shallow network for computational efficiency.
Separate deformation decoders predict position, rotation,
and scale changes:
∆µ = φx(fd),
(11)
∆r = φr(fd),
(12)
∆s = φs(fd),
(13)
where φx, φr, and φs are single-layer MLPs that output 3D
position offsets, quaternion rotations, and 3D scale factors,
respectively.
The final deformed Gaussian values are (µ′, r′, s′) =
(µ + ∆µ, r + ∆r, s + ∆s), while color and opacity remain
unchanged to prevent error propagation, a design choice
that emerged from our analysis of training stability in dy-
namic scenarios.
Selective Deformation Strategy. Through extensive exper-
imentation, we discovered that deforming all five Gaussian
properties (position, opacity, color, scale, rotation) leads to
training instability and error accumulation. Specifically, we
found that appearance properties (color and opacity) are
highly sensitive to deformation errors, causing cascading
failures during backpropagation. By keeping these prop-
erties fixed and only deforming geometric attributes, we
maintain stable gradients while still capturing complex mo-
tion patterns.
4.5. Optimization
Our loss function combines multiple terms for robust opti-
mization:
L = Lcolor + λtvLtv + λvolLvol,
(14)
where Lcolor = L1+λSSIMLSSIM ensures pixel accuracy
and structural coherence, Ltv enforces spatial smoothness

<!-- page 5 -->
in the HexPlane, and Lvol = PNng
i=1 sxi · syi · szi prevents
oversized Gaussians.
For density control, we grow anchors in high-gradient
regions and prune those producing consistently low-opacity
Gaussians.
4.6. View Refinement
Motivation.
During training, we observed that certain
viewpoints consistently under-perform due to large defor-
mations, complex occlusions, or rapid temporal changes.
Rather than applying uniform densification across all
views—which wastes computational resources on already
well-reconstructed areas—we developed an adaptive refine-
ment strategy that identifies and specifically improves prob-
lematic viewpoints while maintaining global efficiency.
Crude View Selection. We implement two complementary
strategies for identifying crude viewpoints that require re-
finement:
1. PSNR-based Detection. We track the PSNR of each
rendered view and compare it against an exponentially
weighted moving average (EMA) to identify statistical out-
liers:
PSNRi < (1 + γ) · EMAPSNR,
(15)
where γ starts at 0.05 and gradually decays to 0.02 dur-
ing training to become more selective as optimization pro-
gresses. The EMA is updated with momentum 0.4 to bal-
ance responsiveness and stability:
EMAPSNR = 0.4 · PSNRcurrent + 0.6 · EMAPSNR.
(16)
2. Gradient-based Detection. For scenarios where gra-
dient information reveals optimization difficulties not cap-
tured by quality metrics, we track the gradient magnitude of
viewspace points and identify views with abnormally high
gradients indicating convergence issues:
∥∇Lview∥> (1 + γ) · EMAgrad.
(17)
This dual-criteria approach ensures robust detection across
different failure modes—quality-based detection captures
rendering artifacts while gradient-based detection identifies
optimization instabilities.
Identified viewpoints are added to a refinement stack
with associated metadata including failure type (quality vs.
gradient), severity score, and temporal consistency flags.
Views appearing consecutively in multiple frames receive
higher priority for refinement.
Adaptive Quality Enhancement. For views identified as
crude, we apply specialized training in a dedicated third
stage (14k iterations) that focuses computational resources
exclusively on problematic areas:
1.
Focused Training.
Only crude viewpoints from the
adaptive camera list are used for training, with sampling
probability weighted by severity scores. This concentrates
gradient updates on areas that need improvement most,
avoiding dilution of learning signals from well-performing
regions.
2. Adjusted Thresholds. We use more aggressive densifi-
cation parameters specifically tuned for challenging views:
These lower thresholds encourage more frequent Gaussian
splitting and pruning in problematic regions.
3. Enhanced Gaussian Generation. The reduced thresh-
olds facilitate creation of additional Gaussians in regions
with complex motion patterns, fine-grained occlusions, or
temporal discontinuities. New Gaussians inherit temporal
features from nearby anchors and undergo immediate de-
formation field training to capture local dynamics.
This targeted approach enhances overall quality and tem-
poral consistency without computational overhead on well-
performing areas, achieving 0.5-1.2 dB PSNR improve-
ments on challenging viewpoints while maintaining global
rendering efficiency.
5. Experiments
5.1. Setup
Setup. We set k = 10 Gaussians per Neural Voxel and
feature dimension fv ∈R32. HexPlane grids use resolution
[64, 64, 64, 150] for DyNeRF and [64, 64, 64, 25] for Hyper-
NeRF with scale multipliers [1, 2]. Training consists of 3k
initialization iterations, 14k main training, and 14k view re-
finement iterations.
Implementation Details.
We use Adam optimizer with
learning rates: Gaussian offsets (0.01), color MLP (0.008),
opacity MLP (0.002), rotation/scaling MLPs (0.004), Hex-
Plane grids (0.0016), deformation MLPs (0.00016).
All
rates decay to 1/10 initial values by iteration 14,000.
Loss weights: λvol = 0.015 for volume regularization
and λtv = 0.0002 for total variation loss. Densification
threshold: τg = 0.0002 (coarse) →0.0001 (refinement);
opacity threshold: τα = 0.05 →0.03.
Dataset.
HyperNeRF [15]: 1-2 cameras with straight-
forward motion.
Neu3D [6]: 15-20 static cameras with
complex motion. COLMAP initialization from first frame
(Neu3D) or 200 random frames (HyperNeRF), yielding
5,000-20,000 initial points for voxel grid initialization.
5.2. Results
Quantitative Results.
To assess the quality of novel
view synthesis, we conducted benchmarking against several
state-of-the-art methods in the field. The results are summa-
rized in Table 1 and Table 2. On HyperNeRF, our method
achieves 28.5 PSNR while using the least memory (3,050
MiB) and fastest training time (13 minutes). On Neu3D,
we maintain competitive quality (33.12 PSNR) with real-
time rendering at 43 FPS—an order of magnitude faster
than Im4D (5 FPS) and MSTH (2 FPS). This combination

<!-- page 6 -->
Model
PSNR(dB) ↑
MS-SSIM↑
Times↓
FPS↑
Memory Usage(MiB)↓
Nerfies [14]
22.2
0.803
∼hours
< 1
-
3D-GS* [4]
19.7
0.68
40 mins
55
-
Scaffold-GS* [10]
20.7
0.688
35 mins
55
-
HyperNeRF [15]
22.4
0.814
32 hours
< 1
-
TiNeuVox-B [3]
24.3
0.836
18 mins**
1
21,558**
4D-GS [24]
25.2
0.845
25 mins**
34
4,500**
GAGS [11]
24.26
0.83
120 mins**
11
6875**
Ours w/o. Refining
25.8
0.846
10 mins†
45
3,050
Ours
28.5
0.872
13 mins†
44
3,050
* : 3D-GS, Scaffold-GS are trained on randomly sampled timestamps from all frames, without temporal modeling.
**: The training time and memory is re-evaluated by us on a single RTX 4090 GPU.
†: Training time includes 3k initialization iterations (approximately 2 minutes).
Table 1. Quantitative results on the HyperNeRF [15] VRIG dataset, rendered at a resolution of 960×540. The best and second best
results are highlighted in red and yellow, respectively. Our method performed the best compared to other methods, while using the least
memory and training time.
Model
PSNR(dB) ↑
D-SSIM↓
Time↓
FPS↑
NeRFPlayer [19]
30.69
0.034
6 hours
0.045
HyperReel [1]
31.10
0.036
9 hours
2.0
HexPlane [2]
31.70
0.014
12 hours
0.2
KPlanes [17]
31.63
-
1.8 hours
0.3
Im4D [9]
32.58
-
28 mins
5
MSTH [22]
32.37
0.015
20 mins
2
4D-GS(CVPR’24) [24]
31.15
0.016
32 mins
34
4D-GS(ICLR’24) [26]
31.91
0.015
105 mins
114
Ex4DGS[5]*
32.11
0.015
36 mins
25
Ours**
33.12
0.021
25 mins
43
* : The training time and FPS is evaluated by us on a single RTX-4090.
** : We introduced gradient-selection selection in view refinement stage
which improves the metrics
Table 2. Quantitative results on the Neu3D [6] dataset. Note: ren-
dering resolution is set to 1352×1014.
of high quality, minimal memory footprint, and real-time
performance makes our approach ideal for practical deploy-
ment.
Qualitative Results. The qualitative results of our study,
compared with several other methods, are shown in Figure
3, 4. These results demonstrate the effectiveness of our ap-
proach. With reduced training time, rendering time, and
memory consumption, our method still produces competi-
tive qualitative results compared to previous methods. In
some cases, especially those with smaller movements, we
manage to produce more detailed images.
5.3. Ablation Study
Loss Design. We present the ablation studies in Table 3
to highlight the differences in quantitative results regard-
ing Volume regularization and Total Variation Loss. We
also rendered images specifically focused on Volume reg-
ularization, as it had a significant impact on the qualitative
results. As shown in Figure 6 without the volume regular-
ization term Lvol, we observe large Gaussians floating in
front of the camera, causing the blurry effect on the image.
Loss
Metrics
Lvol
Ltv
PSNR
MS-SSIM
-
-
22.73
0.67
✓
-
25.58
0.76
-
✓
24.89
0.78
✓
✓
29.51
0.92
Table 3. Comparison of PSNR, SSIM for ablation studies with
Volume regularization and Total Variation Loss on HyperNeRF-
Chicken without gradient-based detection.
Metrics
PSNR
MS-SSIM
Ours w/o. HexPlane
22.31
0.71
Ours w/o. Refinement
28.61
0.88
Ours w/o. View dependent sorting
29.31
0.91
Ours w/o. Gradient-Based Detection
29.51
0.92
Ours
31.28
0.94
Table 4.
Comparison of PSNR and SSIM for ablations on
HyperNeRF-Chicken.
Selective Deformation Strategy. A key design choice in
our method is the selective deformation of Gaussian prop-
erties.
Table 5 demonstrates the impact of this strategy.
When deforming all properties (including color and opac-
ity), training becomes unstable and quality degrades. Our
selective approach, which only deforms geometric prop-
erties (position, scale, rotation), maintains stable training
while achieving better results.
View Refinement. We observed that some viewpoints were
not reconstructed accurately, which we attribute to incon-
sistencies in the Gaussians. As shown in Table 4, the view
refinement effectively enhances image quality by refining
consistency across viewpoints, as illustrated in Figure 7, re-

<!-- page 7 -->
GT
HyperNeRF
TiNeuVox
3D-GS
4D-GS
Ours
Figure 3. Visual comparisons of the proposed method on the HyperNeRF dataset with other methods. The proposed method achieves better
rendering results.
GT
Scaffold GS
GAGS
4D-GS
Ours
Figure 4. Visualization of the Neu3D dataset compared with other methods. From the visual illustration shown in the top and bottom left,
the proposed method strikes a balance while the others either perform worse on the hand or the spinach in the pan. More rendering videos
can be found in the supplementary materials.
Ours
4DGS
Figure 5. Continuous Frames on HyperNeRF Dataset compared with 4DGS. Top: Ours, Bottom: 4DGS. The proposed method deliver a
better rendering results with more details than 4DGS, which can be seen in the top right.

<!-- page 8 -->
Deformation Strategy
PSNR
MS-SSIM
Training Stability
All Properties
26.42
0.81
Unstable
Geometric Only (Ours)
29.51
0.92
Stable
Appearance Only
24.18
0.73
Stable
No Deformation
22.31
0.71
Stable
Table 5. Ablation study on selective deformation strategy. De-
forming only geometric properties achieves the best balance be-
tween quality and training stability.
Figure 6. Left: with Lvol Right: w/o. Lvol.
Figure 7. Left: w/o. view refinement. Right:with view refinement.
sulting in more visually appealing outcomes.
5.4. Memory and Speed Analysis
Memory Usage.
Our method requires only 3,050 MiB,
achieving 86% reduction vs. TiNeuVox-B (21,558 MiB),
32% vs. 4D-GS (4,500 MiB), and 56% vs. GaGS (6,875
MiB). This enables deployment on consumer hardware with
limited GPU resources.
Training Time. Our model completes training in 17 min-
utes with refinement (13 minutes without), achieving 94%
speedup vs. HyperNeRF (32 hours), 32% vs. 4D-GS (25
minutes), and 89% vs. GaGS (120 minutes). This enables
rapid deployment in real-time interactive environments.
5.5. Limitations
Though our 4D Neural Voxel Splatting achieves rapid con-
vergence and yields real-time rendering in many scenarios,
a few key challenges remain.
Large Motions. The absence of initial points and inaccu-
racies in camera poses make it challenging to optimize the
Gaussians effectively. Although incorporating view refine-
ment has improved this issue, further enhancements in mo-
tion estimation are necessary for a complete solution.
Popping Artifacts. Although introducing view-dependent
sorting solves the “static” part of the popping artifact, it
is not entirely eliminated. Further enhancements in voxel
representation and 3D Gaussians’ rasterizing process are
needed to completely remove the artifacts.
6. Conclusion
We introduced 4D Neural Voxel Splatting (4D-NVS),
a method for real-time dynamic scene rendering that
extends voxel representations to the temporal domain.
By decoupling spatial structure from temporal dynam-
ics through neural Gaussians and deformation fields, our
approach achieves superior rendering quality with mem-
ory reduction and computational efficiency.
4D-NVS
opens new avenues for real-time dynamic visualization
and interactive scene manipulation across various applica-
tions.
References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollh¨ofer, Johannes Kopf, Matthew O’Toole, and Changil
Kim.
Hyperreel:
High-fidelity 6-dof video with ray-
conditioned sampling.
arXiv preprint arXiv:2301.02238,
2023. 6
[2] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 130–141, 2023. 2, 6
[3] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural voxels.
In SIGGRAPH Asia 2022 Conference Papers, 2022. 6

<!-- page 9 -->
[4] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(ToG), 42(4):1–14, 2023. 1, 2, 3, 6
[5] Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae,
and Hae-Gon Jeon. Fully explicit dynamic guassian splat-
ting. In Proceedings of the Neural Information Processing
Systems, 2024. 1, 2, 6
[6] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al.
Neural 3d video synthesis from multi-view video.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 5521–5531,
2022. 5, 6
[7] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang.
Neural scene flow fields for space-time view synthesis of dy-
namic scenes. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2021.
2
[8] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
arXiv preprint arXiv:2312.16812, 2023. 2
[9] Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hu-
jun Bao, and Xiaowei Zhou.
High-fidelity and real-time
novel view synthesis for dynamic scenes.
In SIGGRAPH
Asia Conference Proceedings, 2023. 6
[10] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 20654–20664, 2024. 2, 3, 4, 6
[11] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Ming Yang,
Xiao Tang, Feng Zhu, and Yuchao Dai. 3d geometry-aware
deformable gaussian splatting for dynamic view synthesis.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024. 2, 6
[12] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2
[13] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Trans. Graph., 41(4):102:1–
102:15, 2022. 1, 2
[14] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien
Bouaziz, Dan B. Goldman, Steven M. Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pages 5865–5874, 2021. 2, 6
[15] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T.
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M. Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. ACM Trans. Graph., 40(6), 2021. 5, 6
[16] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In CVPR, 2022. 2
[17] Sara Fridovich-Keil and Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
CVPR, 2023. 2, 4, 6
[18] Johannes
Lutz
Sch¨onberger
and
Jan-Michael
Frahm.
Structure-from-motion revisited. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 4104–4113, 2016. 4
[19] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger.
Nerf-
player: A streamable dynamic scene representation with de-
composed neural radiance fields. IEEE Transactions on Visu-
alization and Computer Graphics, 29(5):2732–2742, 2023.
6
[20] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. arXiv preprint arXiv:2306.01496, 2023. 1
[21] Cheng Sun, Jaesung Choe, Charles Loop, Wei-Chiu Ma,
and Yu-Chiang Frank Wang.
Sparse voxels rasterization:
Real-time high-fidelity radiance field rendering.
ArXiv,
abs/2412.04459, 2024. 2
[22] Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, and
Huaping Liu. Masked space-time hash encoding for efficient
dynamic scene reconstruction. In Advances in Neural Infor-
mation Processing Systems (NeurIPS), 2023. 6
[23] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhan-
hua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei
Zhou. Freetimegs: Free gaussian primitives at anytime any-
where for dynamic scene reconstruction. In CVPR, 2025. 1,
2
[24] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene render-
ing. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 20310–
20320, 2024. 1, 2, 6
[25] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun,
Hujun Bao, and Xiaowei Zhou. Representing long volumet-
ric video with temporal gaussian hierarchy. ACM Transac-
tions on Graphics, 43(6), 2024. 1, 2
[26] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting. In International Conference
on Learning Representations (ICLR), 2024. 6

<!-- page 10 -->
4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian
Splatting
Supplementary Material
7. Supplementary Material
7.1. Introduction
In the supplementary material, we provide additional de-
tails and videos on our hyperparameter settings in 8. More
qualitative results are presented in 9, further ablation study
results are discussed in 9.1, and additional discussions are
included in 10.
8. Hyperparameters
8.1. Gaussian Generation
The following learning rates are configured for the Gaussian
generation process.
Offset. The learning rate for the offset vector starts at
1 × 10−2 and decays to 1 × 10−5.
Opacity. The learning rate for MLP with opacity starts
at 2 × 10−3 and decreases to 2 × 10−6.
Covariance. This includes rotation and scaling. The
learning rate for these MLPs starts at 4 × 10−3 and decays
to 4 × 10−6.
Color. The learning rate for color MLP starts at 8×10−3
and decays to 5 × 10−7.
8.2. HexPlane Module
The following learning rates are configured for the Hex-
Plane module:
HexPlane Feature. The learning rate for the HexPlane
feature starts at 1.6 × 10−3 and decays to 1.6 × 10−4.
Deformation Decoder. The learning rate for the decoder
starts at 1.6 × 10−4 and decays to 1.6 × 10−5.
8.3. Iterations
For initializing the Neural Voxels, we train for 3000 iter-
ations on static scenes. Next, we train for 14,000 itera-
tions with temporal information included and HexPlane ac-
tivated. Finally, we perfperformed another 14,000 iterations
for view refinement on underperforming views.
8.4. Pruning and Growing
Anchor points are dynamically pruned and grown during
training:
Growing. Growing starts at iteration 500 and is activated
every 100 iterations until iteration 12,000.
Pruning. Pruning starts at iteration 500 and is activated
every 100 iterations throughout the training.
9. Results
Since we are unable to render videos in the main paper, this
section includes several videos comparing our method to
4D-GS as well as additional videos demonstrating our per-
formance in photorealistic rendering.
Appendix 1: Comparison of our method with 4D-GS on
the HyperNeRF-Interp dataset.
Appendix 2 Rendered scenes in HyperNeRF showcas-
ing our qualitative results.
Appendix 3 We achieved PSNR of 22.15 Rendered a
scene in a monocular dataset Ub4D without any geometry
priors to showcase our ability to reconstruct a monocular
scene.
9.1. Ablation Studies
This section demonstrates the effectiveness of our method
by rendering with and without view refinement. The results
are provided in Appendix 4.
10. Discussions
10.1. Limitations of the Current Approach
Although our 4D Neural Voxel Splatting method achieves
significant improvements in memory efficiency, training
speed, and rendering quality, there are still limitations. For
example, dynamic scenes with large motions or signifi-
cant occlusions present challenges for Gaussian generation
and deformation, requiring further enhancements in motion
modeling and temporal consistency.
10.2. Potential Applications and Impacts
Our method holds promise for various applications, includ-
ing virtual reality, augmented reality, robotics, and the cre-
ation of digital content.
By enabling efficient and high-
quality rendering of dynamic scenes, it can facilitate ad-
vancements in immersive environments, simulation train-
ing, and real-time visual effects.
10.3. Future Work Directions
There are several avenues for future research. One direc-
tion involves improving the robustness of Gaussian defor-
mation fields to better handle large motions and noisy cam-
era poses. Another promising area is optimizing the voxel
representation to entirely eliminate the Gaussian ”popping
artifact”.
Furthermore, exploring hybrid approaches that
combine explicit and implicit representations could enhance
scalability without sacrificing efficiency.
