<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
PocketGS: On-Device Training of 3D Gaussian
Splatting for High Perceptual Modeling
Wenzhi Guo, Guangchi Fang, Shu Yang, Bing Wang∗
3DGS-MVS  on Workstation 
Time: 247s, LPIPS: 0.241
3DGS-SFM on Workstation 
Time: 224s, LPIPS: 0.360 
Fig. 1: Our PocketGS enables high-quality end-to-end 3DGS reconstruction on commodity smartphones. Compared to standard
3DGS workstation baselines, PocketGS achieves superior visual fidelity (LPIPS: 0.108) within a tight training budget ( 500
iterations, ∼4 minutes on an iPhone 15).
Abstract—Efficient and high-fidelity 3D scene modeling is a
long-standing pursuit in computer graphics. While recent 3D
Gaussian Splatting (3DGS) methods achieve impressive real-
time modeling performance, they rely on resource-unconstrained
training assumptions that fail on mobile devices, which are
limited by minute-scale training budgets and hardware-available
peak-memory. We present PocketGS, a mobile scene modeling
paradigm that enables on-device 3DGS training under these
tightly coupled constraints while preserving high perceptual
fidelity. Our method resolves the fundamental contradictions of
standard 3DGS through three co-designed operators: G builds
geometry-faithful point-cloud priors; I injects local surface
statistics to seed anisotropic Gaussians, thereby reducing early
conditioning gaps; and T unrolls alpha compositing with cached
intermediates and index-mapped gradient scattering for stable
mobile backpropagation. Collectively, these operators satisfy the
competing requirements of training efficiency, memory compact-
ness, and modeling fidelity. Extensive experiments demonstrate
that PocketGS is able to outperform the powerful mainstream
workstation 3DGS baseline to deliver high-quality reconstruc-
tions, enabling a fully on-device, practical capture-to-rendering
workflow.
Index Terms—3D Gaussian Splatting, Rendering, on-devide,
modeling system.
I. INTRODUCTION
3D Gaussian Splatting (3DGS) [1] is a promising paradigm
for high-quality scene modeling, advancing mixed reality [2],
[3], digital twins [4], [5], and robotic simulation [6], [7]. By
replacing implicit neural representations [8] with rasterization-
friendly 3D Gaussians, it achieves real-time high-fidelity
modeling, but still suffers from high computation, memory
overhead, and training time, hindering on-device training for
mobile 3D content creation.
Training 3DGS on device introduces substantial challenges,
as it must deliver high perceptual quality under the limited
computational resources and strict thermal constraints of mo-
bile platforms. Consequently, the training process must meet
on-device constraints: 1) completion within a few minutes
(e.g., < 5 min), 2) respecting the hardware-available memory
budget (e.g., < 3 GB).
These constraints require rethinking 3DGS under resource-
limited regimes beyond conventional resource-unconstrained
assumptions. Gaussian-based modeling typically includes: (1)
recovering camera parameters and a point cloud from images,
(2) initializing 3D Gaussians from this geometry, and (3)
optimizing the Gaussians via differentiable rendering for scene
recovery. Existing designs assume ample time and memory
across all stages, but on-device settings break these assump-
tions, yielding three tightly coupled contradictions. 21
Input-Recovery Contradiction. A primary limitation of
on-device 3DGS arises from the contradiction between un-
reliable geometric input and accurate scene recovery. Mobile
scans provide coarse geometry as imprecise poses and sparse
points, causing noisy initialization. Existing methods com-
pensate by densifying Gaussians, but this raises computation
arXiv:2601.17354v4  [cs.CV]  28 Mar 2026

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
and memory overhead, leading to long training times and
memory overflow. Consequently, on-device 3DGS requires
reliable geometric input to minimize reliance on excessive
densification for scene recovery.
Initialization-Convergence Contradiction. Another key
limitation lies in the contradiction between heuristic initial-
ization and stable convergence. Existing methods disregard
geometric priors and initialize input points as isotropic 3D
Gaussians, relying on prolonged optimization to compensate
for the resulting deficiencies. Under on-device constraints,
extra iterations are costly and violate time, energy, and thermal
budgets. Thus, on-device 3DGS requires geometry-conditioned
initialization to reduce optimization burden, thereby facilitat-
ing stable convergence.
Hardware–Differentiability Mismatch. On-device 3DGS
conflicts with mobile execution: while differentiable splatting
needs per-step alpha-compositing states for backpropagation,
mobile fixed-function pipelines expose only the final blended
color and conceal the depth-ordered accumulation needed to
route gradients. Recovering these states via framebuffer read-
backs, full buffer clears, or backward-time re-enumeration of
per-pixel splat sequences is bandwidth-prohibitive and poorly
aligned with mobile GPU state. Thus, on-device 3DGS calls
for a mobile-oriented differentiable rendering framework that
explicitly unrolls compositing and replays only the minimal
cached intermediates required for Gaussian optimization.
In this work, we present PocketGS, a 3DGS training
paradigm that achieves high-perceptual modeling under on-
device constraints via three co-designed operators (Fig. 1,2).
To address the Input-Recovery Contradiction, the geometry-
prior construction operator G builds a geometry-faithful prior,
providing the required geometric guidance without costly in-
loop Gaussian densification beyond the on-device envelope.
To resolve the Initialization-Convergence Contradiction, the
prior-conditioned Gaussian parameterization operator I uses
this prior to shape initial Gaussians, improving conditioning
for stable, iteration-limited optimization and reducing degra-
dation from noise-sensitive initialization. Finally, to overcome
the Hardware-Differentiability Contradiction, the hardware-
aligned splatting optimization operator T makes differentiable
rendering practical on mobile GPUs, preserving correct gradi-
ents and stable updates without prohibitive memory traffic. Ex-
tensive experiments show that PocketGS achieves perceptual
quality superior to mainstream workstation 3DGS baselines,
while operating within on-device constraints.
Our main contributions are:
• We introduce a geometry-prior construction operator G
that provides compact yet reliable geometric inputs for
on-device 3DGS, reducing the dependence on heavy-
weight reconstruction and costly training-time densifica-
tion.
• We propose a prior-conditioned Gaussian parameteriza-
tion operator I that improves initialization quality and
optimization conditioning, enabling stable convergence
under a strict on-device iteration budget.
• We develop a hardware-aligned splatting optimization
operator T that renders differentiable splatting practi-
cal on mobile GPUs, supporting accurate gradients and
stable updates within hardware memory and bandwidth
constraints.
We empirically show that PocketGS satisfies strict on-device
runtime and peak-memory budgets while achieving competi-
tive perceptual quality compared with standard workstation
3DGS pipelines, on both publicly available benchmarks and
our self-collected MobileScan dataset.
II. RELATED WORK
A. Novel View Synthesis
Novel view synthesis targets photorealistic rendering from
unseen viewpoints [9], [10]. The field has progressed from
implicit NeRF-style radiance fields [8], [11], [12] to more
efficient explicit voxel/hash-grid representations [13]–[17],
and further to 3DGS with rasterization-friendly Gaussians for
fast training and rendering [1], [18], [19]. Despite advances
[4], [20]–[26], most 3DGS pipelines still rely on offline SfM
[27], [28] for pose and sparse geometry, whose high cost
and incomplete coverage hinder practical on-device training
in resource-constrained settings.
B. Mobile 3D Scanning
Mobile 3D scanning captures images on mobile devices
and reconstructs real-world 3D structure, typically via visual-
inertial odometry for pose estimation and multi-view stereo
for geometry [29]–[32]. Systems such as ARKit and ARCore
follow this pipeline [33]–[35] but are limited by point/mesh
representations, whose sparse and noisy reconstructions lack
the capacity for high-fidelity rendering [9], [10]. Meanwhile,
novel view synthesis has shown superior modeling and render-
ing quality, opening new opportunities for mobile 3D scanning
[1], [9]; however, existing efforts largely emphasize mobile
rendering efficiency [36], [37], leaving on-device training
underexplored.
III. THE POCKETGS PARADIGM
PocketGS enables on-device 3D Gaussian Splatting (3DGS)
training under strict runtime and peak-memory constraints.
Given a sequence of video frames {It}n
t=1 and their corre-
sponding coarse camera poses { ˆTt}n
t=1 obtained from mobile
tracking systems (ARKit or ARCore), PocketGS comprises
three tightly coupled operators. Specifically, the geometry op-
erator G refines the coarse poses { ˆTt} into accurate ones {Tt},
and extracts a dense point cloud P, which serves as a reliable
geometric prior. The initialization operator I maps P to a
well-conditioned 3D Gaussian initialization Θ0. Finally, the
training operator T performs hardware-aligned differentiable
rendering to optimize Θ0 into the final Gaussian representation
Θ∗. Formally, these operators are defined as: G : ({It, ˆTt}) 7→
({Tt}, P), I : P 7→Θ0, and T : (Θ0, {It, Tt}) 7→Θ∗.
A. Geometry Prior Construction under Resource Budgets
Prior 3DGS approaches rely on prolonged optimization
with training-time densification to recover scenes from noisy
and sparse point inputs. On mobile devices, this paradigm
often yields unstable geometry recovery and substantial com-
putational overhead, making it impractical under on-device

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
Census 
Matching
d1
dn
d2
Reference View
Source view
Sharpness
Displacement
Fig. 2: Overview of the PocketGS framework. PocketGS tackles on-device 3DGS training through three coupled operators:
(G) Geometry Prior Construction employs an information-gated gate for frame selection, followed by GPU-native Schur-
complement BA and single-reference MVS. The MVS module constructs a 3D cost volume by sampling depth hypotheses
d1, . . . , dn via census matching to produce a dense geometric scaffold. (I) Prior-Conditioned Parameterization seeds anisotropic
Gaussians by estimating local surface statistics (normals sn and scales) to front-load structure discovery via disc-like covariance
seeding. (T ) Hardware-Aligned Splatting implements a mobile-native differentiable renderer using unrolled alpha-compositing
(S = id, Cin, α) and index-mapped gradient scattering to ensure stable backpropagation within tight mobile memory bounds
of the canonical parameter buffer (µ, Σ, c, α).
constraints. We instead leverage the mobile capture {It, ˆTt} to
build a dense, low-noise, and low-redundancy geometry prior
by applying a geometry operator G, which maps ({It, ˆTt}) to
refined poses {Tt} and a compact geometric representation
P. Conditioning the subsequent optimization on ({Tt}, P)
removes the need for training-time densification, resulting in
more stable convergence and improved computational effi-
ciency.
1) Information-Gated Frame Subsampling: Video frames
captured during mobile scanning are often affected by motion
blur, which degrades geometric reconstruction and undermines
the photometric optimization of 3DGS. To mitigate this issue,
we subsample the input sequence using an information-gated
keyframe selection strategy based on viewpoint change and
distance variation, retaining only sharp and high-utility frames
while filtering out blurred and redundant ones.
Displacement Gate. We ensure sufficient parallax for ro-
bust triangulation by computing the camera displacement d
between the current frame tcurr and the last selected keyframe
tlast, d = ∥tcurr −tlast∥2. A frame is considered only if
d ≥τd, where τd = 0.05m, suppressing redundant viewpoints
while maintaining tracking stability.
Sharpness Heuristic. We reject frames degraded by motion
or defocus blur using an efficient approximation of gradient
energy. We estimate the sharpness score S by summing
absolute luma differences over a sparsely sampled grid Ω:
S =
1
|Ω|
X
(x,y)∈Ω
(|I(x + ∆, y) −I(x, y)| + |I(x, y + ∆) −I(x, y)|)
(1)
where ∆is the step size. This score correlates with high-
frequency content and favors sharp frames.
Candidate Windowing. We select the locally optimal frame
from a short temporal window (8 frames). A candidate replaces
the current best only if Snew > (1 + r)Sbest, with r = 0.05.
This commits the sharpest representative per viewpoint and
bounds downstream optimization workload.
2) GPU-Native Global BA as Mobile MAP Refinement:
Initial camera poses provided by mobile tracking systems (e.g.,
ARKit and ARCore) are often noisy, which degrades both
geometric reconstruction and the photometric optimization of
3DGS. To address this issue, we perform GPU-native global
BA on-device to refine poses, thereby providing accurate pose
estimates for subsequent dense point recovery and 3DGS
optimization.
The robustified reprojection objective for PocketGS:
min
{ˆTi},{Pj}
X
i,j
ρ

∥π(ˆTi, Pj) −pij∥2
Σij

(2)
where π is the projection function, ˆTi ∈SE(3) is the pose
of camera i, Pj ∈R3 is the 3D position of point j, pij ∈R2
is the observed 2D projection, and ρ is the robust Huber loss.
Scale-Aware Gauge Fixing. To resolve the 7-DoF ambi-
guity of monocular reconstruction, we fix the 6-DoF pose of
the first keyframe and fix scale by constraining the transla-
tional component of the second keyframe along the dominant
baseline axis, yielding a consistent metric frame.
GPU-Native Schur Complement. The BA bottleneck is
solving the normal equations H∆= b, where H = JT J.
We exploit the sparse block structure via a full GPU Schur
complement. Partitioning parameters into camera pose (t) and
point (p) blocks yields the reduced camera system:
(Htt −HtpH−1
pp Hpt)∆c = bt −HtpH−1
pp bp
(3)

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Since Hpp is block-diagonal with independent 3×3 blocks per
point, H−1
pp is computed in parallel across points on the GPU.
After solving for ∆t, we recover ∆p by back-substitution.
Iterative Geometric Refinement. BA is embedded in an
iterative refinement loop. After each global step, we re-
triangulate features using updated poses, filter points with
high reprojection error or insufficient triangulation angles,
and remove observations behind the camera. This purification
reduces outliers and stabilizes dense reconstruction.
3) Single-Reference Cost-Volume MVS: Point clouds pro-
vided by mobile tracking systems are typically sparse, which
necessitates costly densification during 3DGS training. To ad-
dress this limitation, our lightweight MVS reconstructs a dense
and geometry-faithful point cloud for Gaussian initialization,
thereby reducing over-densification under mobile budgets.
Probabilistic Depth Range Estimation. Plane-sweep ef-
ficiency depends on depth range. For each target frame, we
project sparse BA points and compute the 5% and 95%
depth quantiles q.05 and q.95. We define the search range as
[q.05·0.6, q.95·1.6], focusing computation on the most probable
volume.
Memory-Efficient Cost Volume. To minimize memory
footprint, we construct the cost volume using only a single,
optimally chosen reference frame. The reference maximizes
Sref = exp

−(b −btarget)2
2σ2
b

· max

α
αmin
, 1

(4)
balancing baseline length b against a target baseline btarget
and enforcing sufficient viewing angle α. The matching cost
is computed using the Census Transform [38] for robustness to
illumination changes. Depth is inferred via plane sweep [39]
with Semi-Global Matching aggregation. We filter depth maps
using a confidence threshold of 0.4 and fuse them into a dense
point cloud, which serves as the geometric scaffold for I.
B. Prior-Conditioned Gaussian Parameterization
Prior methods initialize points as isotropic Gaussians based
solely on inter-point distances, resulting in arbitrary parameter
initialization that requires unnecessary updates and conse-
quently slow convergence. In contrast, we exploit point-cloud
surface geometry P to estimate normals and initialize surface-
aligned anisotropic Gaussians via an initialization operator I,
which maps P to the initial Gaussian parameters Θ0. This
structure-consistent starting state Θ0 = I(P) improves con-
ditioning and accelerates optimization under tight on-device
budgets.
1) Local Surface Statistics Estimation: To extract geomet-
ric priors from the mobile capture, we estimate local surface
statistics for each point pi in the dense MVS point cloud
using a fixed-size KNN neighborhood (K=16). Let {pi,k}K
k=1
denote the K nearest neighbors of pi, with centroid
¯pi = 1
K
K
X
k=1
pi,k.
(5)
We compute the local covariance as
Ci = 1
K
K
X
k=1
(pi,k −¯pi)(pi,k −¯pi)T.
(6)
The eigenvector of Ci associated with its smallest eigenvalue
provides an estimate of the local surface normal ni. This
per-point, fixed-K procedure is embarrassingly parallel with
bounded memory, making it well-suited for GPU execution
under PocketGS’s mobile constraints.
2) Disc-Like Covariance Seeding: To inject geometric prior
knowledge into the initialization process, we initialize each
Gaussian as a thin, disc-like ellipsoid tangent to the esti-
mated surface. This approach directly addresses the limita-
tions of naive isotropic initialization by introducing geometry-
conditioned anisotropy. The tangential scale st is derived from
local point density, estimated as the average distance to the
K=3 nearest neighbors:
st = 1
3
3
X
k=1
∥p −pk∥
(7)
We set the normal-direction scale sn as a fraction of tangential
scale:
sn = st × rnormal
(8)
where rnormal = 0.3. Scale parameters are optimized in
log-space, slog = log(s), ensuring positivity and numerical
stability. The initial rotation q aligns the Gaussian local z-axis
with the estimated surface normal ni. Opacity is initialized to
a fixed low value (logit of 0.1). This geometry-conditioned
anisotropy supplies strong priors on surface orientation and
extent, accelerating convergence under limited iterations.
C. Hardware-Aligned Differentiable Splatting Optimization
On mobile GPUs, on-device 3D Gaussian Splatting (3DGS)
training is bandwidth-bound: fixed-function blending reveals
only the final color, hiding the intermediate compositing states
needed for backpropagation, while auxiliary-buffer clears incur
heavy memory traffic. We therefore align optimization with
the mobile pipeline by unrolling front-to-back compositing
into an explicit differentiable operator and caching a com-
pact forward replay trace. This defines a GPU-resident map
T : (Θ0, {It, Tt}) 7→Θ∗that delivers correct, stable gradients
without framebuffer readbacks or backward-time reconstruc-
tion of per-pixel splat sequences.
1) Unrolled
Alpha-Compositing
with
Forward
Replay
Cache: Mobile fixed-function blending returns only the fi-
nal composited color, while 3DGS backpropagation requires
the intermediate accumulator state at each compositing step.
Moreover, the set and order of Gaussians contributing to a
pixel are not accessible from the rendering pipeline, making
a backward pass that replays the original per-pixel splat
list infeasible or prohibitively expensive. We address this by
(1) unrolling the front-to-back compositing into an explicit
computation graph and (2) caching a compact forward replay
trace per pixel for backward traversal.
a) Explicit Computation Graph.: For depth-sorted visi-
ble Gaussians, we explicitly perform front-to-back composit-
ing:
Cout = Cin(1 −α) + αc,
(9)

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
LLFF
NeRF-Synthetic
MobileScan
Method
PSNR↑SSIM↑LPIPS↓Time↓(s) Count PSNR↑SSIM↑LPIPS↓Time↓(s) Count PSNR↑SSIM↑LPIPS↓Time↓(s) Count
3DGS-SFM-WK
21.01
0.641
0.405
108.0
18k
21.75
0.800
0.243
83.7
12k
21.16
0.687
0.398
112.8
23k
3DGS-MVS-WK
19.53
0.637
0.387
313.1
40k
24.47
0.887
0.128
532.1
50k
20.85
0.781
0.281
534.5
165k
PocketGS (Ours) 23.54
0.791
0.222
105.4
33k
24.32
0.858
0.144
101.4
47k
23.67
0.791
0.225
255.2
168k
TABLE I: Average Metrics Across Datasets. Dataset-level averages on LLFF, NeRF Synthetic, and our MobileScan mobile-
native dataset. Higher is better for PSNR/SSIM, and lower is better for LPIPS/Time. Best results are in bold.
where Cin is the accumulator input and (α, c) are the opacity
and color of the current Gaussian fragment. The backward
pass follows the same unrolled chain:
∂L
∂Cin
=
∂L
∂Cout
(1 −α),
∂L
∂α =
 ∂L
∂Cout
, (c −Cin)

,
(10)
where ⟨·, ·⟩denotes the per-channel inner product.
b) Bandwidth-Efficient Forward Replay Cache.: To avoid
backward-time reconstruction of per-pixel splat sequences, we
record only the intermediates necessary to replay gradients:
S = {id, Cin, α},
(11)
where id is the (sorted) splat identifier used to scatter gradients
back to canonical parameter buffers. A per-pixel counter
count(u) stores the number of valid cached entries for pixel u.
Each iteration resets only the O(WH) counters, rather than
clearing the full O(WHKmax) cache. During backpropaga-
tion, we traverse only
k ∈[0, count(u) −1],
treating stale cache entries beyond count(u) as invalid, thereby
eliminating bandwidth-heavy full-cache clears while preserv-
ing gradient correctness.
2) Index-Mapped Gradient Scattering: Maintaining opti-
mizer consistency during depth-sorted rendering typically re-
quires physically reordering parameter buffers, which incurs
heavy data movement and can misalign optimizer states. We
resolve this by decoupling the sorted view used for render-
ing/backprop from the canonical layout used for parameter
storage and optimizer moments. A GPU-based sort produces
a permutation π that maps each sorted index i to its canonical
index π(i). The renderer and backward kernels operate on the
sorted view, while gradients are scattered back to canonical
buffers:
∇θπ(i) += gi,
(12)
ensuring that Adam optimizer states (stored in canonical order)
remain aligned with their respective parameters, without CPU
intervention or redundant memory copies.
IV. EXPERIMENTS
A. Experimental Setup
1) Implementation Details and Experimental Settings: We
implement PocketGS as a fully on-device system and evaluate
it on an iPhone 15 (Apple A16). The entire process was written
in Swift and accelerated by using the Apple Metal API to
take advantage of the mobile GPU execution stack. Unless
otherwise stated, all methods are evaluated under identical
image resolution, color space, and metric implementations.
To reflect realistic mobile constraints, we use a fixed training
budget for all methods. In particular, we optimize each model
for a constant number of iterations (500 iterations) at a fixed
view rendering resolution.
2) Baselines: We compare against two workstation base-
lines that follow the standard two-stage paradigm: (i) recon-
struct camera poses and a point-cloud prior, and (ii) train
a standard 3D Gaussian Splatting (3DGS) model initialized
from the reconstructed prior. All workstation experiments are
conducted on a machine equipped with 2× NVIDIA RTX
3090 GPUs and an Intel Core i9-class CPU. For both baselines,
point-cloud reconstruction is performed using COLMAP [28].
a) 3DGS-SFM-WK: We adopt the standard COLMAP-
based Structure-from-Motion (SfM) pipeline on a workstation
to estimate camera extrinsic/intrinsic parameters and recon-
struct a sparse point cloud. This sparse geometric prior is then
used to initialize and train a vanilla 3D Gaussian Splatting
(3DGS) model, adhering to the original 3DGS training proto-
col.
b) 3DGS-MVS-WK: For this workstation-based baseline,
we leverage the standard COLMAP Multi-View Stereo (MVS)
workflow to generate a dense point cloud as the geometric
prior. Subsequently, a vanilla 3DGS model is initialized with
this dense point cloud and trained following the default 3DGS
optimization workflow.
We choose these two baselines because they (1) repre-
sent the standard reconstruction-first 3DGS pipeline, (2) span
sparse SfM versus dense but costly MVS priors to separate
prior quality from reconstruction overhead under matched
budgets, and (3) provide a workstation reference that reflects
PocketGS’s goal: competitive quality under strict on-device
constraints without heavy offline reconstruction.
3) Datasets and Metrics:
a) Datasets.:
We
evaluate
on
three
representative
datasets spanning synthetic and real captures: NeRF-Synthetic
[8], LLFF [40], and MobileScan. NeRF-Synthetic provides
object-centric synthetic scenes with accurate poses; LLFF
consists of real forward-facing captures exhibiting real-world
degradations; and MobileScan is our Phone-captured dataset
designed to reflect mobile imaging characteristics, including
motion blur, defocus, and exposure variation. For MobileScan,
we use ARKit-provided intrinsics, poses, and sparse structure
to emulate practical on-device inputs, enabling evaluation
under noisy geometric priors and limited compute budgets.
More details are in the supplement.
b) Metrics.: We report standard novel-view synthesis
metrics, including PSNR (dB, ↑), SSIM (↑), and LPIPS (↓).
We additionally report end-to-end runtime to capture practical
deployability. Specifically, we define the total runtime as
Ttotal = Tgeom + Ttrain,
(13)

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
Fig. 3: Qualitative comparison on our MobileScan dataset. PocketGS (Ours) consistently recovers sharper textures and finer
details compared to baselines. Notably, 3DGS-MVS-WK suffers from artifacts despite its high-density prior, as the over-
concentrated initial points impede effective Gaussian redistribution within a limited iteration budget. Our method avoids
this bottleneck through prior-conditioned initialization, achieving superior structural fidelity (e.g., bicycle spokes) that closely
matches the ground truth (GT).
where Tgeom includes geometry and pose acquisition as well
as associated optimization, and Ttrain denotes the 3DGS opti-
mization time. For fair comparison, all methods are evaluated
under the same training budget and at the same rendering
resolution. Finally, we report Count (the number of Gaussians)
to characterize the trade-off between representation capacity
and computational efficiency.
B. Main Results: Quality and Efficiency Comparison
Table I summarizes averages under a strict, matched op-
timization budget. Across all datasets, PocketGS consistently
delivers a strong quality–efficiency trade-off while maintaining
compact or comparable Gaussian counts, indicating better
representation efficiency.
a) LLFF Dataset.: LLFF stresses forward-facing cap-
tures and view extrapolation, where imperfect geometry can
destabilize optimization. PocketGS achieves the best over-
all quality while remaining fast. Notably, 3DGS-MVS-WK
performs poorly despite a larger Count, suggesting that a
generic dense prior does not ensure good convergence in
this regime. In contrast, PocketGS attains higher quality with
fewer Gaussians, indicating better representation efficiency
and conditioning under the same iteration budget.
Fig. 4 shows PocketGS better preserves fine, high-frequency
geometry structures (e.g., leaves in Fern, petals in Flower),
with fewer floating artifacts and cleaner boundaries. This is
consistent with our refined geometry prior (G) and robust
anisotropic initialization (I), which improve pose/structure
consistency and reduce Gaussian drift during optimization.
b) NeRF-Synthetic Dataset.:
On the geometry-clean
NeRF Synthetic dataset, 3DGS-MVS-WK achieves the best
metrics, as its dense prior matches the ground-truth geometry.
PocketGS remains competitive (PSNR 24.32 vs. 24.47) while
being ∼5.2× faster in end-to-end runtime Ttotal (101.4s vs.
532.1s), mainly by avoiding the costly offline dense recon-
struction in 3DGS-MVS-WK. With a similar Count (47k vs.
50k), this speedup reflects a more streamlined pipeline and
hardware-aligned splatting rather than reduced capacity.
Fig. 5 shows PocketGS preserves high-fidelity appearance
and clean edges under the same budget, closely matching
3DGS-MVS-WK, demonstrating that PocketGS can substan-
tially reduce capture-to-reconstruction latency without sacri-
ficing quality when geometry is well-conditioned.
c) MobileScan Dataset.: On the challenging MobileScan
dataset, PocketGS consistently outperforms 3DGS-SFM-WK,
indicating that our mobile-native prior construction mitigates
instability from sparse/noisy mobile inputs. Compared to

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
Fig. 4: Qualitative comparison on the LLFF dataset. Our method produces much sharper textures and more accurate thin
structures (e.g., the leaves in Fern and the petals in Flower), closely matching the ground truth (GT).
3DGS-MVS-WK, PocketGS achieves better perceptual quality
(LPIPS 0.225 vs. 0.281) with much lower end-to-end time
(255.2s vs. 534.5s), suggesting that heavyweight offline dense
reconstruction is inefficient for mobile deployment. Notably,
although PocketGS and 3DGS-MVS-WK converge to similar
primitives, PocketGS renders more faithfully, implying im-
proved placement rather than merely the quantity of Gaussians.
Fig. 3 shows that 3DGS-SFM-WK often yields blurry
textures and structural collapse, consistent with its low Count
(23k) and unreliable geometry. While 3DGS-MVS-WK starts
from a dense prior, it still exhibits artifacts when Gaussians
are overly concentrated; under a limited optimization budget,
it cannot sufficiently re-balance spatial coverage, leading to
ghosting and local inconsistency. PocketGS avoids this by
surface-aligned, spatially balanced anisotropic initialization,
producing sharper textures and higher structural fidelity closer
to GT.
d) On-device Rendering and Deployment Perspective.:
Beyond metrics, Fig. 6 demonstrates real-time rendering
screenshots produced on an iPhone 15 across MobileScan,
LLFF, and NeRF Synthetic scenes. The consistently sharp
details across diverse scales and capture conditions indi-
cate robust generalization in practical deployment. PocketGS
maintains high perceptual quality under mobile memory and
bandwidth constraints by avoiding unnecessary densification
while preserving critical structures.
C. Memory Footprint
We profile peak memory usage on MobileScan to quan-
tify the practicality of our on-device pipeline. Following our
end-to-end definition, we report memory for two stages: (i)
geometry prior construction and (ii) full 3DGS training. In
Variant
PSNR↑
SSIM↑
LPIPS↓
Time↓(s)
PocketGS (Full)
23.67
0.791
0.225
255.2
w/o Initialization (I)
22.49
0.7696
0.253
319.5
w/o Global BA
23.45
0.7517
0.232
251.1
w/o MVS
21.07
0.6461
0.414
124.8
TABLE II: Ablation Summary on MobileScan Dataset
(Averages).
all measurements, peak memory is recorded as the maximum
resident memory observed during execution, and the statistics
are aggregated over all MobileScan scenes.
Full 3DGS training reaches an average peak of 2.21 GB
(range: 1.82–2.65 GB), reflecting the combined cost of storing
the Gaussian parameters, intermediate buffers for rasterization,
and optimizer states. Geometry prior construction peaks at
1.53 GB on average (range: 1.19–2.22 GB), dominated by
temporary buffers used for feature processing, correspondence
aggregation, and incremental updates of the prior represen-
tation. Importantly, across all scenes the peak memory re-
mains below 3 GB, which fits within the on-device budget
and leaves practical headroom for commodity smartphones
to accommodate system overhead and concurrent background
processes. We provide per-scene peak memory breakdowns for
both stages in the supplemental material.
D. Ablation Study
We conduct an ablation study on the MobileScan dataset
to quantify the contribution of each core component of Pock-
etGS. The study follows the same experiment setup, and we
summarize dataset averages in Table II.
Information-Gated
Frame
Subsampling.
Removing
information-gated
subsampling
admits
motion/defocus-

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
Fig. 5: Qualitative comparison on the NeRF Synthetic dataset. In these object-centric synthetic scenes, our method produces
high-fidelity results.
degraded frames, corrupting the geometry prior and cascading
into poorer initialization and blurrier reconstructions; the full
pipeline better preserves sharp details (fig. 7).
GPU-native Global BA. Disabling Global BA yields a clear
SSIM drop (from 0.791 to 0.7517) with minor PSNR/LPIPS
changes, indicating reduced global coherence and pose con-
sistency, while incurring negligible runtime overhead (251.1s
vs. 255.2s).
Single-Reference Cost-Volume MVS. Removing lightweight
MVS causes the largest quality degradation (PSNR 21.07,
SSIM 0.6461, LPIPS 0.4137), showing dense geometric sup-
port largely sets the quality ceiling; although runtime drops to
124.8s, quality falls to the level of 3DGS-SFM-WK.
Prior-Conditioned Gaussian Parameterization. Removing I
degrades all metrics (PSNR 23.67→22.49) and increases end-
to-end time (255.2s→319.5s), suggesting prior-conditioned
initialization is critical for convergence efficiency under a
limited iteration budget.
V. CONCLUSION
In this paper, we present PocketGS, a fully on-device
training framework for 3D Gaussian Splatting that targets the
practical constraints of mobile devices. PocketGS is built upon
three co-designed operators: (G) geometry-prior construction
to provide compact and reliable geometric guidance, (I) prior-
conditioned Gaussian parameterization to improve initializa-
tion and optimization under iteration-limited budgets, and (T)
hardware-aligned splatting optimization to ensure efficient and
stable differentiable splatting on mobile GPUs. Extensive ex-
periments on public benchmarks and our MobileScan dataset
show that PocketGS achieves strong perceptual quality while
meeting smartphone budgets, e.g., approximately 4-minute
end-to-end training with < 3 GB peak memory on iPhone
15. These results indicate that high-fidelity 3D reconstruction
and mobile 3D content creation can be made practical on
commodity devices.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Trans. Graph.,
vol. 42, no. 4, 2023.
[2] J. Gao, C. Gu, Y. Lin, Z. Li, H. Zhu, X. Cao, L. Zhang, and Y. Yao,
“Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF
Decomposition and Ray Tracing,” in ECCV, 2024.
[3] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “GS-IR: 3D Gaussian
Splatting for Inverse Rendering,” in CVPR, 2024.
[4] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian
splatting for geometrically accurate radiance fields,” ACM SIGGRAPH
2024 Conference Proceedings, 2024.
[5] Z. Yu, T. Sattler, and A. Geiger, “Gaussian Opacity Fields: Efficient
Adaptive Surface Reconstruction in Unbounded Scenes,” ACM TOG,
vol. 43, no. 6, pp. 1–13, 2024.
[6] A. Escontrela, J. Kerr, A. Allshire, J. Frey, R. Duan, C. Sferrazza, and
P. Abbeel, “GaussGym: An Open-Source Real-to-Sim Framework for
Learning Locomotion from Pixels,” arXiv preprint arXiv:2510.15352,
2025.
[7] M. Zhang, K. Zhang, and Y. Li, “Dynamic 3D Gaussian Tracking for
Graph-Based Neural Dynamics Modeling,” in CoRL, 2025.
[8] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[9] A. Tewari, J. Thies, B. Mildenhall, P. Srinivasan, E. Tretschk, W. Yi-
fan, C. Lassner, V. Sitzmann, R. Martin-Brualla, S. Lombardi et al.,
“Advances in neural rendering,” in Computer Graphics Forum, vol. 41,
no. 2.
Wiley Online Library, 2022, pp. 703–735.

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
Fig. 6: Qualitative results on a mobile device across diverse datasets. We showcase real-time rendering screenshots from an
iPhone 15, covering our self-collected scenes (rows 1-2), LLFF (row 3), and NeRF Synthetic (row 4). PocketGS consistently
achieves high-fidelity reconstruction and sharp details across varying scene scales and capture conditions. The red labels indicate
optimized point counts, demonstrating our method’s representation efficiency and robust generalization for practical on-device
deployment.
Fig. 7: Ablation of information-gated frame subsampling. Our
method can obtain better details information.
[10] K. Gao, Y. Gao, H. He, D. Lu, L. Xu, and J. Li, “Nerf: Neural
radiance field in 3d vision, a comprehensive review,” arXiv preprint
arXiv:2210.00379, 2022.
[11] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-NeRF: A Multiscale Representation for Anti-
Aliasing Neural Radiance Fields,” in ICCV, 2021.
[12] J. T. Barron, B. Mildenhall, M. Tancik, P. P. Srinivasan, X. Han,
and R. Martin-Brualla, “Mipnerf 360: Unbounded anti-aliased neural
radiance fields,” in IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2022.
[13] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5501–5510.
[14] C. Sun, M. Sun, and H.-T. Chen, “Direct Voxel Grid Optimization:
Super-Fast Convergence for Radiance Fields Reconstruction,” in CVPR,
2022.
[15] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph.,
vol. 41, no. 4, 2022.
[16] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields,” in ICCV,
2023.
[17] W. Guo, B. Wang, and L. Chen, “Neuv-slam: Fast neural multireso-
lution voxel optimization for rgbd dense slam,” IEEE Transactions on
Multimedia, 2025.
[18] G. Fang and B. Wang, “Mini-splatting: Representing scenes with a
constrained number of gaussians,” in European Conference on Computer
Vision.
Springer, 2024, pp. 165–181.
[19] ——, “Mini-splatting2: Building 360 scenes within minutes via aggres-
sive gaussian densification,” arXiv preprint arXiv:2411.12788, 2024.
[20] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, “Rtg-
slam: Real-time 3d reconstruction at scale using gaussian splatting,”
ACM SIGGRAPH 2024 Conference Proceedings, 2024.
[21] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “Gs-
slam: Dense visual slam with 3d gaussian splatting,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[22] G. Wenzhi, B. Haiyang, M. Yuanqu, L. Jia, and C. Lijun, “Fvloc-
nerf: Fast vision-only localization within neural radiation field,” in 2023
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS).
IEEE, 2023, pp. 3329–3334.
[23] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu,
H. Bao, and G. Zhang, “PGSR: Planar-based Gaussian Splatting for
Efficient and High-Fidelity Surface Reconstruction,” IEEE TVCG, 2024.

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
[24] L. H¨ollein, A. Boˇziˇc, M. Zollh¨ofer, and M. Nießner, “3dgs-lm: Faster
gaussian-splatting optimization with levenberg-marquardt,” in Proceed-
ings of the IEEE/CVF International Conference on Computer Vision,
2025, pp. 26 740–26 750.
[25] Y. Chen, J. Jiang, K. Jiang, X. Tang, Z. Li, X. Liu, and Y. Nie,
“Dashgaussian: Optimizing 3d gaussian splatting in 200 seconds,” in
Proceedings of the Computer Vision and Pattern Recognition Confer-
ence, 2025, pp. 11 146–11 155.
[26] G. Fang and B. Wang, “Efficient Scene Modeling Via Structure-Aware
and Region-Prioritized 3D Gaussians,” IEEE TPAMI, 2025.
[27] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon,
“Bundle adjustment — a modern synthesis,” in Vision Algorithms:
Theory and Practice.
Springer, 1999.
[28] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2016.
[29] Q. Picard, S. Chevobbe, M. Darouich, and J.-Y. Didier, “A survey
on real-time 3d scene reconstruction with slam methods in embedded
systems,” arXiv preprint arXiv:2309.05349, 2023.
[30] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monocular
visual-inertial state estimator,” IEEE Transactions on Robotics, vol. 34,
no. 4, pp. 1004–1020, 2018.
[31] Y. Furukawa and J. Ponce, “Accurate, dense, and robust multiview stere-
opsis,” IEEE transactions on pattern analysis and machine intelligence,
vol. 32, no. 8, pp. 1362–1376, 2009.
[32] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth infer-
ence for unstructured multi-view stereo,” in European Conference on
Computer Vision (ECCV), 2018.
[33] P. Kim, J. Kim, M. Song, Y. Lee, M. Jung, and H.-G. Kim, “A benchmark
comparison of four off-the-shelf proprietary visual–inertial odometry
systems,” Sensors, vol. 22, no. 24, p. 9873, 2022.
[34] T. Feigl, A. Porada, S. Steiner, C. L¨offler, C. Mutschler, and
M. Philippsen, “Localization limitations of arcore, arkit, and hololens
in dynamic large-scale industry environments.” in VISIGRAPP (1:
GRAPP), 2020, pp. 307–318.
[35] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras,” IEEE transactions
on robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
[36] Z. Chen, T. Funkhouser, P. Hedman, and A. Tagliasacchi, “Mobilenerf:
Exploiting the polygon rasterization pipeline for efficient neural field
rendering on mobile architectures,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp.
16 569–16 578.
[37] S. Rojas, J. Zarzar, J. C. P´erez, J. Sanchez-Riera, A. Rodr´ıguez, F. Segu,
and F. Moreno-Noguer, “Re-rend: Real-time rendering of nerfs across
devices,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2023.
[38] W. S. Fife and J. K. Archibald, “Improved census transforms for
resource-optimized stereo vision,” IEEE Transactions on Circuits and
Systems for Video Technology, vol. 23, no. 1, pp. 60–73, 2012.
[39] X. Yang, L. Zhou, H. Jiang, Z. Tang, Y. Wang, H. Bao, and G. Zhang,
“Mobile3drecon: Real-time monocular 3d reconstruction on a mobile
phone,” IEEE Transactions on Visualization and Computer Graphics,
vol. 26, no. 12, pp. 3446–3456, 2020.
[40] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ra-
mamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view
synthesis with prescriptive sampling guidelines,” ACM Transactions on
Graphics (ToG), vol. 38, no. 4, pp. 1–14, 2019.
APPENDIX
In this section, we provide additional details regarding
the three core operators of PocketGS: Geometry Prior Con-
struction (G), Prior-Conditioned Parameterization (I), and
Hardware-Aligned Splatting (T ).
A. Geometry Prior Construction (G)
a) Information-Gated Frame Selection: To ensure high-
quality input under a limited time budget, we implement a
real-time frame selection mechanism during the capture phase.
We utilize a displacement gate τ = 0.05m to filter redundant
frames. For frames passing the displacement gate, we employ a
candidate window mechanism (maximum 8 frames or 0.25s) to
select the sharpest frame. Sharpness is computed by calculat-
ing the gradient energy on a sparse 160×160 grid of the luma
plane, prioritizing frames with higher high-frequency content
to mitigate motion blur.
b) GPU-Native Global Bundle Adjustment: Our global
BA optimizes both camera poses and 3D point coordinates.
We establish geometric constraints by extracting ORB/FAST
features with BRIEF descriptors, accelerated via Metal. Fea-
ture matching is performed using Hamming distance with ratio
tests and cross-checks. We implement an iterative refinement
process (typically 3 rounds) that includes re-triangulation and
outlier filtering based on reprojection error and triangulation
angles. The optimization uses a Levenberg-Marquardt (LM)
solver with Schur complement, fully resident on the GPU. To
fix the gauge, we anchor the first frame’s 6DoF and the scale
of the baseline between the first two frames.
c) Lightweight Single-Reference MVS: For dense recon-
struction, we estimate per-frame depth maps using a plane-
sweep algorithm. To optimize memory, we select the top-
3 reference frames based on baseline distance and viewing
angle but only utilize the best reference frame for cost volume
construction. The depth search range is dynamically estimated
using the 5%–95% quantiles of sparse point depths. We
use Census transform for robust matching and Semi-Global
Matching (SGM) for cost aggregation. The final dense point
cloud is generated by back-projecting pixels with confidence
≥0.4 and depth within 0.05m to 5.0m.
B. Prior-Conditioned Parameterization (I)
We initialize Gaussian primitives by leveraging the local
surface statistics derived from the MVS point cloud. For each
point, we estimate the local surface normal using PCA on its
k = 16 nearest neighbors. The scale parameters are initialized
based on the average distance to the k = 3 nearest neighbors.
To seed anisotropic Gaussians, we set the scale along the
normal direction to 0.3× the tangential scales, effectively
creating disc-like primitives that better align with the scene
geometry.
C. Hardware-Aligned Splatting (T )
Our differentiable renderer is tailored for Tile-Based De-
ferred Rendering (TBDR) GPUs. We implement a multi-stage
pipeline where alpha compositing is manually unrolled within
the fragment shader. This allows us to cache intermediate
states in tile memory, avoiding expensive global memory
traffic. For backpropagation, we use index-mapped gradient
scattering to ensure that gradients are correctly accumulated
into the canonical parameter buffer despite the depth-sorted
rendering order.
To demonstrate the feasibility and practicality of our end-
to-end on-device training system, we developed a dedicated
mobile application, PocketGS, which integrates the entire
pipeline from data capture to 3DGS training. The application
is built on the Metal framework for high-performance GPU
computation.

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
(a) Main Menu
(b) Splat Capture
(c) Process Capture
(d) Splat Training
Fig. 8: Screenshots of the PocketGS mobile application, illustrating the end-to-end workflow.
The application workflow, as illustrated in Figure 8, is
designed to be intuitive and efficient:
• Main Menu (Fig. 8a): Provides access to the core
functionalities: Splat Capture (data acquisition), View
Point Cloud and View 3DGS (visualization), Splat Train-
ing (optimization), Import Dataset, and Process Capture
(geometry prior construction).
• Splat Capture (Fig. 8b): A guided capture interface that
uses the information-gated frame selection mechanism
to ensure high-quality input images (e.g., 50 frames)
are collected in real-time, minimizing motion blur and
redundancy.
• Process Capture (Fig. 8c): This module manages cap-
tured scenes and executes the Geometry Prior Construc-
tion (G) operator, including the GPU-native BA and
lightweight MVS, to generate the initial PLY point cloud.
Users can manage captured data and trigger the process-
ing step.
• Splat Training (Fig. 8d): This interface allows users to
configure and initiate the on-device 3DGS training. Key
features include selecting the input PLY file, choosing
between Original Pose (ARKit pose) and Optimized Pose
(BA-refined pose), and setting core hyperparameters (e.g.,
learning rates, iterations). The training process, which
implements the I and T operators, runs entirely on the
mobile GPU, with progress displayed in real-time.
The PocketGS app serves as a concrete proof-of-concept,
demonstrating that high-quality 3DGS training can be achieved
end-to-end on a commodity mobile device within minutes.
The MobileScan dataset consists of 16 diverse indoor and
outdoor scenes captured using iPhone 15. The statistics of
the collected scenes are summarized in Table III. Each scene
includes 50 high-resolution images with corresponding ARKit
poses and sparse point clouds as baselines, as shown on 9.
Fig. 9: The scenes on MobileScan Dataset.
The dataset covers various challenging scenarios, including:
• Textureless Surfaces: Large uniform areas like the
”Sofa” and ”Desk” scenes.
• Complex Geometry: Intricate structures in the ”Trans-
former” and ”Shrub” scenes.
• Varying Lighting: Outdoor scenes like ”Pier” with nat-
ural lighting transitions.
We provide per-scene metrics for our MobileScan dataset,
NeRF Synthetic dataset, and LLFF dataset. These tables serve
as the full, unaggregated data supporting the main paper’s
claims on performance metrics (PSNR, SSIM, LPIPS) and
efficiency (Time, Count).
D. MobileScan Dataset Comparison
The detailed per-scene quantitative results on our d Mo-
bileScan dataset are presented in Table IV. PocketGS con-
sistently outperforms 3DGS-SFM-WK and 3DGS-MVS-WK

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
Category
Scenes
Environment
Scenes
Sofa,Table, Desk, Shelf
Indoor
Objects
Robot, Audio, Printer, Bag, Elxbox, audio.
Indoor
Outdoor Pier, Shrub, Bike, Transformer, Chair, Flowerpot
Outdoor
TABLE III: Statistics of the MobileScan Dataset.
Scene
3DGS-SFM-WK
3DGS-MVS-WK
PocketGS (Ours)
PSNR ↑SSIM ↑LPIPS ↓Time (s) ↓CountPSNR ↑SSIM ↑LPIPS ↓Time (s) ↓Count PSNR ↑SSIM ↑LPIPS ↓Time (s) ↓Count
Pier
20.23
0.498
0.495
156.0
35800 20.54
0.695
0.312
519.6
200936 20.29
0.625
0.329
212.8
176804
Chair
22.13
0.542
0.421
174.0
39420 21.34
0.699
0.270
555.1
180664 22.39
0.670
0.256
240.7
176970
Flowerpot
20.93
0.595
0.424
168.0
32415 21.00
0.730
0.281
486.9
242616 21.97
0.725
0.225
265.7
241976
Shrub
18.88
0.507
0.481
204.0
37476 18.53
0.582
0.427
546.7
98616
18.94
0.609
0.331
230.0
174278
Transformer 20.49
0.542
0.465
162.0
27678 19.66
0.636
0.360
546.7
122117 21.05
0.636
0.315
209.7
126222
Sofa
14.26
0.611
0.558
28.6
38
25.32
0.870
0.222
525.4
312949 24.90
0.873
0.192
389.4
289848
Table
19.71
0.754
0.382
78.0
18873 15.01
0.727
0.365
401.9
113738 22.85
0.852
0.194
184.6
179940
Robot
22.90
0.807
0.285
114.0
20795 22.21
0.862
0.207
581.7
209226 24.64
0.850
0.167
291.9
194881
Desk
20.31
0.754
0.313
72.0
20735 17.42
0.750
0.279
491.9
111202 22.03
0.838
0.172
192.8
108018
Elebox
26.17
0.922
0.197
22.4
4410
21.11
0.899
0.181
575.8
52654
27.75
0.936
0.136
255.1
37182
audio
26.05
0.815
0.330
150.0
28628 23.97
0.852
0.263
520.0
147733 27.69
0.857
0.166
270.2
155278
bag
23.71
0.855
0.313
108.0
22584 21.35
0.865
0.260
533.0
170320 24.79
0.859
0.210
320.4
179101
fireExtin
23.94
0.879
0.274
96.0
18162 22.09
0.886
0.239
592.0
146224 26.40
0.896
0.170
279.8
123325
printer
22.56
0.804
0.376
59.8
10368 20.08
0.834
0.277
542.8
203770 26.30
0.882
0.202
282.6
196896
shelf
20.39
0.761
0.418
24.8
4860
21.17
0.811
0.313
518.8
77940
25.40
0.872
0.226
224.4
75872
umbrella
24.29
0.813
0.341
47.4
9736
21.98
0.845
0.273
581.8
179388 23.30
0.826
0.235
261.9
173587
Average
21.16
0.687
0.398
112.8
23486 20.85
0.781
0.281
534.5
165367 23.67
0.791
0.225
255.2
168009
TABLE IV: MobileScan dataset comparison. Best results are highlighted in bold
Scene
3DGS-SFM-WK
3DGS-MVS-WK
PocketGS (Ours)
PSNR↑SSIM↑LPIPS↓Time (s)↓Count PSNR↑SSIM↑LPIPS↓Time (s)↓Count PSNR↑SSIM↑LPIPS↓Time (s)↓Count
Chair
22.11
0.869
0.154
90.0
14433
25.37
0.957
0.054
692.4
60859
23.70
0.870
0.089
85.7
49857
Drums
18.78
0.762
0.268
53.8
12842
21.27
0.861
0.154
570.1
60640
20.97
0.775
0.183
130.1
58511
Ficus
19.99
0.812
0.212
44.2
1511
20.36
0.804
0.233
144.3
5674
22.94
0.879
0.134
33.5
5638
Hotdog
23.13
0.829
0.236
84.0
9703
29.09
0.966
0.050
571.3
40686
28.07
0.940
0.094
78.8
38583
Lego
22.02
0.795
0.228
132.0
29883
25.47
0.930
0.092
770.3
73821
24.92
0.872
0.124
130.5
74572
Materials
21.21
0.779
0.254
120.0
6934
22.45
0.843
0.170
414.8
51456
22.85
0.842
0.158
117.4
52137
Mic
23.62
0.859
0.205
31.7
5879
27.11
0.936
0.066
519.5
39553
26.24
0.901
0.120
81.0
36082
Ship
23.14
0.696
0.386
114.0
15006
24.60
0.798
0.202
574.5
69482
24.87
0.781
0.248
153.9
64208
Average
21.75
0.800
0.243
83.7
12024
24.47
0.887
0.128
532.1
50271
24.32
0.858
0.144
101.4
47449
TABLE V: NeRF Synthetic Dataset comparison.
in terms of perceptual quality (LPIPS) while maintaining a
significantly lower training time than the dense baseline.
E. NeRF Synthetic and LLFF Datasets
We also evaluate PocketGS on standard benchmarks to
demonstrate its generalization capability. The per-scene results
for the NeRF Synthetic dataset are shown in Table V, and the
LLFF dataset results are in Table VI.
We analyze the peak memory usage of PocketGS during
the MVS and training phases. The detailed per-scene memory
usage is provided in Table VII, demonstrating that our system
operates well within the memory budget of commodity mobile
devices.
We conduct ablation studies to evaluate the impact of our
core components. The following tables provide the detailed
per-scene metrics for each ablation configuration, serving as
the full data to support the aggregated results presented in the
main paper.
F. Ablation: w/o Initialization (I)
This
configuration
removes
the
prior-conditioned
anisotropic seeding in I, reverting to isotropic initialization.
The
detailed
per-scene
results
in
Table
VIII
show
a
significant drop in performance, highlighting the importance
of geometry-aware initialization for fast convergence.
G. Ablation: w/o Global BA
This configuration removes the GPU-native global Bundle
Adjustment (BA) in G, relying solely on the initial ARKit
poses. The detailed per-scene results in Table IX indicate that
the global BA is crucial for refining the coarse mobile poses,
leading to better reconstruction quality.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
Scene
3DGS-SFM-WK
3DGS-MVS-WK
PocketGS (Ours)
PSNR↑SSIM↑LPIPS↓Time↓(s) Count PSNR↑SSIM↑LPIPS↓Time↓(s) Count PSNR↑SSIM↑LPIPS↓Time↓(s) Count
Fern
21.11
0.649
0.415
21.0
11436
21.36
0.707
0.335
118.2
38816
25.75
0.849
0.182
107.4
29220
Flower
22.97
0.701
0.328
114.0
22440
20.53
0.632
0.381
316.4
34914
25.06
0.818
0.187
92.0
28523
Fortress
26.51
0.744
0.321
180.0
29365
23.59
0.762
0.277
551.9
69762
27.15
0.901
0.129
91.6
67120
Horns
21.19
0.674
0.412
126.0
22335
17.42
0.626
0.432
530.9
38403
23.39
0.806
0.233
148.6
35996
Leaves
15.36
0.352
0.600
246.0
9358
15.28
0.327
0.588
190.4
20315
15.77
0.456
0.501
60.8
9358
Orchids
18.85
0.562
0.401
66.0
21968
18.11
0.593
0.353
141.1
66975
21.73
0.772
0.183
138.9
55320
Room
21.70
0.764
0.400
15.3
4737
21.34
0.787
0.358
203.4
27610
26.18
0.897
0.168
75.6
19273
Trex
20.42
0.684
0.360
96.0
21645
18.62
0.664
0.376
452.4
30104
23.33
0.830
0.194
128.6
22349
Average
21.01
0.641
0.405
108.0
17910
19.53
0.637
0.387
313.1
40862
23.54
0.791
0.222
105.4
33395
TABLE VI: LLFF dataset comparison.
Metric
audio
printer
fireExtin
flowerpot
Elebox
desk
transformer
shrub
bench
sofa
Keybroad
bike
Pier
bag
robot
umbrella
chair
table
shelf
Avg.
MVS Peak Mem. (GB)
1.55
1.50
1.55
1.48
1.38
1.19
1.48
1.64
2.22
1.39
1.67
1.50
1.73
1.42
1.57
1.56
1.49
1.24
1.47
1.53
Training Peak Mem. (GB)
2.65
2.62
2.50
2.49
2.44
2.41
2.29
2.27
2.20
2.15
2.15
2.12
2.03
2.02
1.98
1.95
1.94
1.91
1.82
2.21
TABLE VII: Per-scene peak memory.
Metric
Elebox
Pier
audio
bag
bench
bike
chair
desk
fireExtin
flowerpot
printer
robot
shelf
shrub
sofa
table
transformer
umbrella
Avg.
PSNR (dB)
27.03
20.16
20.93
22.72
21.97
23.02
22.09
22.07
26.46
20.98
24.83
21.35
24.99
18.80
22.05
22.89
20.91
21.55
22.49
SSIM
0.9335
0.6048
0.8138
0.8303
0.6859
0.7185
0.6514
0.8385
0.8961
0.6836
0.8555
0.7934
0.8647
0.5889
0.8203
0.8522
0.6255
0.7955
0.7696
LPIPS
0.1400
0.3572
0.1859
0.2628
0.3207
0.2553
0.2760
0.1726
0.1781
0.2735
0.2390
0.2259
0.2481
0.3588
0.2706
0.1935
0.3316
0.2699
0.2533
Time (s)
369.1
325.6
363.0
335.8
350.4
327.7
306.8
186.7
429.2
386.9
352.7
283.0
318.3
331.5
329.3
172.6
297.7
284.5
319.5
TABLE VIII: Per-scene metrics w/o initialization (removing prior-conditioned anisotropic seeding in I).
Metric
Elebox
Pier
audio
bag
bench
bike
chair
desk
fireExtin
flowerpot
printer
robot
shelf
shrub
sofa
table
transformer
umbrella
Avg.
PSNR (dB)
27.61
20.21
26.95
24.47
22.40
23.03
22.30
22.02
26.45
21.67
26.10
24.38
24.85
18.82
24.49
22.52
20.80
23.01
23.45
SSIM
0.8975
0.5910
0.8156
0.8201
0.6869
0.6832
0.6318
0.8014
0.8548
0.6843
0.8401
0.8048
0.8282
0.5774
0.8283
0.8107
0.5915
0.7825
0.7517
LPIPS
0.1350
0.3253
0.1743
0.2172
0.2857
0.2595
0.2639
0.1758
0.1821
0.2326
0.2093
0.1869
0.2367
0.3311
0.1928
0.2026
0.3285
0.2437
0.2324
Time (s)
230.7
212.2
280.5
311.5
259.9
231.9
233.5
181.3
285.0
254.2
282.5
263.2
232.0
230.6
352.7
176.7
238.5
263.0
251.1
TABLE IX: Per-scene metrics w/o global BA (removing GPU-native global BA in G).
Metric
Elebox
Pier
audio
bag
bench
bike
chair
desk
fireExtin
flowerpot
printer
robot
shelf
shrub
sofa
table
transformer
umbrella
Avg.
PSNR (dB)
23.10
18.25
24.43
23.67
19.84
21.03
19.46
20.12
23.35
19.56
23.38
22.66
21.09
16.98
21.96
19.61
18.91
21.92
21.07
SSIM
0.9029
0.3181
0.7425
0.8163
0.4339
0.5333
0.3467
0.7574
0.8413
0.5089
0.8195
0.7925
0.7845
0.2820
0.7690
0.7533
0.4489
0.7795
0.6461
LPIPS
0.2113
0.6392
0.3905
0.3201
0.5653
0.4897
0.5609
0.2855
0.3064
0.4894
0.3259
0.2600
0.3583
0.6229
0.3898
0.3478
0.5480
0.3366
0.4137
Time (s)
122.2
93.8
186.7
142.1
101.4
110.7
129.0
95.9
178.5
118.9
149.3
112.2
142.3
88.2
136.2
89.9
111.0
137.5
124.8
TABLE X: Per-scene metrics w/o MVS (removing lightweight single-reference MVS in G).
H. Ablation: w/o MVS
This configuration removes the lightweight single-reference
MVS in G, relying only on the sparse point cloud from ARKit
for initialization. The significant drop in all metrics, as shown
in Table X, demonstrates the necessity of a dense, geometry-
faithful prior for achieving high perceptual quality within a
limited training budget.
