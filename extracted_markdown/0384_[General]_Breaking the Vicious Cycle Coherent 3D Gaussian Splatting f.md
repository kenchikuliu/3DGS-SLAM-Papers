<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from
Sparse and Motion-Blurred Views
Zhankuo Xu, Chaoran Feng, Yingtao Li, Jianbin Zhao, Jiashu Yang, Wangbo Yu,
Li Yuan†, and Yonghong Tian†, Fellow, IEEE
Fig. 1: We present CoherentGS, a novel 3D Gaussian Splatting (3DGS) framework for high-fidelity reconstruction from sparse
and motion-blurred inputs. CoherentGS alternates between pixel-level artifact restoration and geometry completion guided by
a dual prior: a deblurring prior that recovers high-frequency photometric detail and a diffusion prior conditioned on both 2D
frames and 3D renderings to ensure cross-view consistency and 3D reconstruction quality.
Abstract—3D Gaussian Splatting (3DGS) has emerged as a
state-of-the-art method for novel view synthesis. However, its
performance heavily relies on dense, high-quality input imagery,
an assumption that is often violated in real-world applications,
where data is typically sparse and motion-blurred. These two
issues create a vicious cycle: sparse views ignore the multi-view
constraints necessary to resolve motion blur, while motion blur
erases high-frequency details crucial for aligning the limited
views. Thus, reconstruction often fails catastrophically, with
fragmented views and a low-frequency bias. To break this cycle,
we introduce CoherentGS, a novel framework for high-fidelity 3D
reconstruction from sparse and blurry images. Our key insight
is to address these compound degradations using a dual-prior
strategy. Specifically, we combine two pre-trained generative
models: a specialized deblurring network for restoring sharp
details and providing photometric guidance, and a diffusion
model that offers geometric priors to fill in unobserved regions of
the scene. This dual-prior strategy is supported by several key
techniques, including a consistency-guided camera exploration
module that adaptively guides the generative process, and a
depth regularization loss that ensures geometric plausibility. We
evaluate CoherentGS through both quantitative and qualitative
experiments on synthetic and real-world scenes, using as few as 3,
6, and 9 input views. Our results demonstrate that CoherentGS
significantly outperforms existing methods, setting a new state-
of-the-art for this challenging task. The code and video demos
are available at https://potatobigroom.github.io/CoherentGS/.
Index Terms—3D Gaussian Splatting, Novel View Synthesis,
Motion Deblurring, Sparse Viewpoints, Generative Model.
Z. Xu and C. Feng contributed equally to this work.
C. Feng, W. Yu, L. Yuan, and Y. Tian are with Peking University. Z. Xu,
J. Zhao, J. Yang, and Y. Li are research interns at the School of Electronics
and Computer Engineering, Peking University.
L. Yuan and Y. Tian are the corresponding authors (†). E-mail: {yuanli-
ece@pku.edu.cn, yhtian@pku.edu.cn}.
I. INTRODUCTION
Recently, 3D Gaussian Splatting (3DGS) [1] has emerged as
an efficient and expressive scene representation, substantially
advancing novel view synthesis (NVS). With dense multi-
view coverage and sharp observations, 3DGS achieves high-
fidelity reconstructions and has been rapidly adopted in a
variety of applications [2]–[15]. However, such successes
rely on input conditions that rarely hold in practice. In
real-world scenarios, particularly in quick and unconstrained
handheld/robot-mounted capture, image sequences are often
sparse in viewpoints and degraded by motion blur. Under these
conditions, the assumptions underpinning 3DGS break down:
sparse views lead to fragmented local representations that fail
to consolidate into a coherent global scene [16]–[19], while
motion blur suppresses high-frequency details that are crucial
for reliable geometry and texture recovery [20], [21]. When
combined, these factors severely underconstrain optimization,
making blur disentanglement particularly ill-posed and leading
to severe degradation in reconstruction quality.
These challenges have two main consequences for 3D
reconstruction: view fragmentation and low-frequency bias.
Under sparse viewpoints, limited overlap weakens cross-
view constraints and drives each camera to fit a localized
distribution of Gaussian primitives rather than contributing
to a coherent global representation; empirically, the learned
distributions form fragmented clusters with weak cross-view
overlap, leading to geometric drift and visible inconsistencies
across views as shown in Fig. 2. Motion blur suppresses
high-frequency cues such as textures and edges; instead of
recovering these details, optimization typically reduces repro-
arXiv:2512.10369v2  [cs.CV]  27 Dec 2025

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
BAD-Gaussians
CoherentGS (Ours)
t-sne
distribution
 Frequency
Spetrum
Deblurring Failed
Local Optimization
Global Optimization
Deblurring Successfully
Novel View
Synthesis
Less Artifacts
More Artifacts
Fig. 2: Visualizing view fragmentation. We compare NVS
and t-SNE distributions using only 3 input views. BAD-
Gaussians [22] degenerates into fragmented local clusters,
whereas CoherentGS maintains global geometric coherence.
jection error by inflating Gaussian covariances or blending
colors, thereby fitting low-frequency signals at the expense of
structural fidelity [23]–[25]. Fundamentally, fragmentation and
low-frequency bias jointly exacerbate the ill-posedness: degen-
erate view-specific solutions explain blur via over-smoothed
parameterizations, and the loss of high-frequency cues weak-
ens geometric alignment across views. This vicious coupling
leaves the problem severely underconstrained, preventing high-
quality novel-view synthesis and stable scene-level coherence,
consequently photorealistic supervision [1] and exposure mod-
eling [5], [22], [26] alone are inadequate in this regime.
To address these challenges, we introduce a novel 3DGS-
based reconstruction framework, CoherentGS. We posit that
in the challenging real-world setting of sparse and blurry
inputs, existing paradigms are insufficient. On one hand, state-
of-the-art 3D deblurring methods [22], [27] excel at modeling
camera motion but fundamentally rely on dense multi-view
geometric constraints to decouple the trajectory from the
scene. Under sparse views, these constraints vanish, causing
the motion estimation to become ill-posed and the deblurring
process to fail. On the other hand, while generative models can
fill in missing content for sparse views, they are not specialized
for physically-grounded deblurring and can hallucinate details
inconsistent with the underlying motion. Therefore, our core
insight is that a robust solution must synergistically integrate
two distinct, specialized priors: one for photometric deblurring
and another for geometric completion.
Our CoherentGS framework employs this dual-prior strat-
egy through a systematic and iterative process. The foun-
dation of our method is a camera exposure motion model
that explicitly simulates the blur formation by integrating
multiple sharp 3DGS renderings along an optimized camera
trajectory. To provide robust deblurring guidance, we distill
knowledge from a pre-trained image deblurring model into
our 3DGS representation using a perceptual distillation loss,
which enforces the recovery of high-frequency details.
For scene-level geometric completion, we first propose a
consistency-guided trajectory planner that identifies under-
observed areas. A powerful diffusion model is then used to
provide a supervisory signal for these novel views via a score
distillation-inspired loss, ensuring contextual and geometric
coherence. Finally, all components are optimized jointly within
a unified framework: a composite objective combines a blurry
reconstruction loss on real inputs with the two distillation
losses from our generative priors, simultaneously refining the
3DGS scene and the camera motion parameters. This iterative
loop of guided synthesis and joint optimization progressively
enhances the scene’s completeness and high-frequency detail.
We evaluate our method on the challenging Deblur-NeRF
dataset [2] and proposed outdoor motion-blurred dataset. The
experimental results demonstrate significant improvements
over existing approaches and our contributions are summarized
as follows:
• We introduce the first 3D Gaussian Splatting framework
to systematically address the compound challenge of
high-quality reconstruction from inputs that are simul-
taneously sparse and degraded by severe motion blur.
• We propose a novel dual-prior guidance strategy that
integrates a generative diffusion model to resolve sparse-
view ambiguities and a dedicated deblurring model to
restore high-frequency details corrupted by motion.
• We develop a consistency-guided camera exploration for
holistic and unseen view modeling, an exposure pho-
tometric bundle adjustment tailored for motion-blurred
images, and a training scheme that jointly optimizes
3DGS with generated sequences.
II. RELATED WORK
A. 3D Representation from Degraded Images
Neural Radiance Fields (NeRF) [28] and 3D Gaussian
Splatting (3DGS) [1] have emerged as powerful and novel
3D representations. However, their success is predicated on
stringent input requirements, demanding both dense multi-
view coverage and high-fidelity, blur-free imagery. In practical
applications, these ideal conditions are seldom met, as cap-
tured images are often degraded. Consequently, reconstructing
3D scenes from such degraded inputs has become a critical
and burgeoning area of research. Among various degradations,
motion and defocus blur are particularly common in multi-
view captures. Pioneering work in this domain focused on
adapting NeRF to handle such artifacts. DeblurNeRF [2] was
the first to jointly deblur and reconstruct a scene, modeling
the blur kernel as a per-pixel ray transformation. Building on

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
this, subsequent methods sought to impose greater physical
consistency; for instance, DP-NeRF [29] modeled the blur as a
function of 3D rigid body motion, more closely approximating
the physical camera capture process. Other notable contribu-
tions in this area include ExBluRF [30], PDRF [31], and their
variants [26], [32]. With the advent of 3DGS, which offers
efficient rendering and an explicit representation, the focus has
shifted towards deblurring this new paradigm. Methods such as
BAD-Gaussians [22], DeblurGS [33], Deblur3DGS [34], and
others [35]–[38] have all proposed effective solutions, typically
by explicitly modeling the camera’s motion trajectory during
the exposure time to deconvolve the blur.
While previous methods have shown significant success,
they often rely on the assumption that a relatively dense set of
blurry images is available. However, this setup does not fully
reflect the challenges encountered in real-world scenarios,
where data can be both severely degraded and extremely
limited. A more realistic and challenging situation arises when
only a small number of images are available for a given scene,
each suffering from motion blur [39]. We address this gap
by tackling the combined problem of 3D reconstruction from
inputs that are both sparse and blurry. We propose a framework
that integrates deblurring techniques with strategies designed
for sparse-view settings, thus improving the applicability of
3D reconstruction in real-world conditions.
B. Generative Priors for Novel View Synthesis.
Leveraging powerful and pre-trained generative models to
overcome the limitations of 3D reconstruction from sparse or
degraded imagery is a growing research area. This trend has
largely branched into two specialized sub-domains targeting
different types of degradation. The first and most prominent
sub-domain addresses sparsity, where diffusion models are
employed to rectify geometric and photometric deficiencies.
Recent works including Difix3D+ [40], GSFixer [41], 3D-
Enhancer [42] and [43]–[52] typically operate by rendering
a flawed view, enhancing it with the diffusion prior, and
distilling the refined details back into the 3D representation.
On the other hand, some methods such as Sparse-DeRF [39]
to tackle image blur, integrate specialized restoration networks
to explicitly reverse the degradation process.
Despite their individual successes, these two paradigms have
remained isolated. This separation presents a critical challenge:
directly applying sparse-view generative methods to blurry
inputs leads to hallucinated blur where the model mistakes
artifacts for intrinsic texture, while applying deblurring net-
works to sparse data results in severe overfitting. Drawing
inspiration from 2D knowledge distillation [53], our work
is the first to bridge this gap by synergistically integrating
distinct types of generative priors within a single, coherent
optimization framework.
III. PRELIMINARY
A. 3D Gaussian Splatting
3DGS [1] explicitly represents the scene using a collection
of sparse 3D Gaussians. Each primitive is parameterized by a
3D covariance matrix Σ ∈R3×3, a mean position µ ∈R3, an
opacity o ∈[0, 1], and spherical harmonics (SH) coefficients
for view dependent color. The spatial influence of a Gaussian
is defined as:
G(x) = exp

−1
2(x −µ)⊤Σ−1(x −µ)

,
(1)
where Σ is factorized into a scaling matrix S ∈R3 and a
rotation matrix R ∈SO(3) to ensure positive semi definite-
ness, formulated as Σ = RSS⊤R⊤. To render the scene from
a specific viewpoint, the 3D covariance is projected into 2D
image space as Σ′ = JWΣW⊤J⊤, where J is the Jacobian
of the affine projective approximation and W is the viewing
transformation matrix. The final pixel color C is computed by
alpha blending N ordered primitives overlapping the pixel:
C =
N
X
i=1
Tiαici,
with
Ti =
i−1
Y
j=1
(1 −αj),
(2)
where ci is the color derived from SH coefficients, and αi
denotes the final alpha contribution computed by multiplying
the learned opacity oi with the 2D Gaussian evaluation.
B. Image Blur Formation
The physical image formation process involves accumulat-
ing photons over a finite exposure period. Mathematically,
this is modeled as the temporal integration of instantaneous
sharp irradiance. Specifically, the motion blurred image B(u)
at pixel coordinate u is formulated as:
B(u) = 1
τ
Z τ
0
I(t, u) dt,
(3)
where τ denotes the camera exposure duration, and I(t, u) rep-
resents the instantaneous latent sharp image at time t ∈[0, τ].
The normalization factor 1/τ ensures energy conservation
during the integration window. For computational feasibility,
we approximate this continuous integral using a discrete
summation of n virtual sharp samples:
B(u) ≈1
n
n−1
X
k=0
Ik(u),
(4)
where Ik(u) denotes the kth virtual sharp instance sampled
along the camera trajectory within the exposure interval.
C. Diffusion Model
Diffusion Models (DMs) are generative frameworks de-
signed to approximate a data distribution pdata(x) by reversing
a gradual noise addition process. In the forward pass, an input
sample x0 is progressively perturbed across discrete timesteps
t ∈[0, T] by injecting Gaussian noise. This yields a noisy
latent xt = αtx0+σtϵ, where ϵ ∼N(0, I) is standard normal
noise, and αt, σt define the noise schedule. The generative
reverse process utilizes a neural network Fθ, termed the
denoiser, to estimate the added noise. The parameters θ are
optimized via a denoising score matching objective:
LDM = Ex0,t,ϵ,c

∥ϵ −Fθ(xt; t, c)∥2
2

.
(5)

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Sparse Blurry Inputs
Pose Estimation
Point Cloud & Gaussian
Initilization
 
 
 
 
Photometric Restoration via Deblurring Priors
Initialized / Virtual / Explored Camera
Generative Repair
Camera
Exploration
Rasterize
Corrupted Rendering 
in novel view
Repaired image
in novel view
Lgeo
Geometric Guidance via Repair Diffusion Priors
Pose Initilization
Lreg
Exposure time
Tstart
Tend
…
…
Rasterize
Average
Virtual sharp images
Simulated blurred image
Lreg
Lblurry
Deblurring
Image Deblur Model
Pre-deblurred image
Lpr
Fig. 3: Overview of CoherentGS. Our framework synergizes two generative priors to resolve sparse view ambiguity and motion
blur. (Left) We initialize the Gaussian primitives using poses estimated by COLMAP [54]. (Top) Photometric Restoration via
Deblurring Priors: This branch models the physical exposure trajectory to supervise blurry synthesis via Lblurry and distills
sharp high frequency details from a pretrained deblurring network using a perceptual loss Lpr. (Bottom) Geometric Guidance
via Repair Diffusion Priors: This branch utilizes a diffusion model to repair structural defects in explorative viewpoints. A score
distillation loss Lgeo and a depth regularization loss Lreg are applied to guide the geometry completion and ensure consistency.
Here, the network Fθ takes the noisy latent xt, the timestep
t, and an optional condition vector c (e.g., text embeddings or
reference images) as input to reconstruct the signal. We em-
ploy this conditional generation capability to guide geometric
completion in unobserved regions.
IV. METHOD
A. Problem Formulation and Overall Framework
Our CoherentGS framework employs a dual prior strategy
to address the compound degradation of view fragmentation
and low frequency bias inherent in motion blurred sparse
view modeling, as depicted in Fig. 3. Given only a set of K
sparse and blurry images {Bi}K
i=1, we first establish an initial
geometric backbone by estimating camera poses {Ti}K
i=1 by
COLMAP [55] and initializing a 3DGS model Gθ. The core
of our method is an iterative loop designed to progressively
enhance this initial reconstruction by synergistically applying
two distinct generative priors. First, to address motion blur,
we perform photometric restoration guided by a deblurring
prior (Sec. IV-B). This module explicitly models the physical
exposure trajectory to disentangle camera motion from scene
appearance, thereby recovering high frequency details. Second,
to resolve sparse view ambiguity, we introduce geometric
completion guided by a conditional diffusion prior (Sec. IV-C).
This process is steered by a consistency-guided camera ex-
ploration strategy (Sec. IV-D), which targets unobserved re-
gions to steer the diffusion model in generating plausible
geometric and contextual content. Finally, these two guidance
mechanisms are integrated within a joint optimization scheme
(Sec. IV-E), where the 3DGS model and camera parameters
are updated using a composite loss that fuses supervision from
the original blurry inputs with the signals from both generative
Algorithm 1 Photometric Restoration via a Deblurring Prior
1: Function ComputePerceptualLoss(Gθ, Bi, Tstart, Tend)
2: Input: Current 3DGS model Gθ, a blurry input image Bi,
and its corresponding start and end poses.
3: Given: Pre-trained deblurring model DΦ, pre-trained fea-
ture extractor Θvgg.
4: ¯Ii ←DΦ(Bi)
▷Eq. (9).
5: Ftarget ←Θvgg(¯Ii)
6: Tmid ←InterpolatePose(Tstart, Tend, 0.5)
▷Eq. (6).
7: ˆIi ←Render(Gθ, Tmid)
8: Frender ←Θvgg(ˆIi)
9: Lpr ←||Frender −Ftarget||2
2
▷Eq. (10).
10: Return Lpr
priors. This cyclical generation and reconstruction process is
iterated to progressively refine the scene fidelity.
B. Photometric Restoration via Deblurring Prior
Our approach begins by explicitly modeling the physical
process of motion blur as shown in Sec. III-B. The blur in
a captured image B is the result of integrating light from a
dynamic scene along a camera’s trajectory over its exposure
time τ. We model this by optimizing a start pose Tstart and
an end pose Tend for each capture. The latent camera pose Tt
at any time t ∈[0, τ] is derived through linear interpolation in
the SE(3) space:
Tt = Tstart · exp
 t
τ · log
 T−1
start · Tend

.
(6)
By discretizing the exposure time into n samples, we can
render a set of virtual sharp images {ˆIi(Ti, θ)}n−1
i=0 from the

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
3DGS model Gθ. Averaging these sharp renderings allows us
to synthesize a blurry image ˆB:
ˆB(θ, Tstart, Tend) = 1
n
n−1
X
i=0
ˆIi(Ti, θ).
(7)
This formulation allows us to supervise the 3DGS and the
camera motion parameters using a photorealistic loss, Lblurry,
which measures the difference between the synthesized blurry
image ˆB and the real captured image B:
Lblurry = λ1L1( ˆB, B) + λSSIMLSSIM( ˆB, B).
(8)
However, under sparse-view conditions, relying solely on
this blurry reconstruction loss is insufficient. The joint op-
timization of deblurring and 3D geometry becomes highly
ill-posed, as there are not enough multi-view constraints to
robustly decouple the camera motion from the scene’s appear-
ance. A straightforward alternative would be to pre-deblur the
input images and use them for direct pixel-wise supervision.
This approach is prone to failure, as single-image deblurring
is itself an ill-posed problem and applying it independently
to multiple views often generates results that are not 3D-
consistent, leading to severe geometric artifacts.
Inspired by [39], we address this by distilling knowledge
from a powerful, pre-trained image deblurring model [56] in
a robust feature space instead of the pixel space. First, we
generate a set of target sharp images {¯Ii}K
i=1 by applying a
pre-trained deblurring model DΦ to blurry inputs {Bi}K
i=1:
¯Ii = DΦ(Bi).
(9)
Instead of enforcing strict pixel alignment with these
pseudo-targets which may contain artifacts, we leverage them
for high-level semantic guidance. We define a perceptual
restoration loss Lpr, that encourages the features of our
rendered sharp image ˆIi to match the features of the target
deblurred image ¯Ii. To extract robust features, we employ a
shared and pre-trained VGG [57] encoder Θvgg:
Lpr = ||Θvgg(ˆIi) −Θvgg(¯Ii)||2
2.
(10)
For this formulation, the rendered sharp image ˆIi is generated
using the camera pose at the temporal midpoint Tt=τ/2, which
serves as the canonical sharp state of the frame. This distil-
lation mechanism provides strong priors for high-frequency
details without enforcing strict pixel-level correspondence.
C. Geometric Guidance via Diffusion Prior
Recently, leveraging diffusion models to enhance 3DGS
from limited inputs has gained significant attention. Recent
works [40], [42], [43], [47] have shown that these models not
only provide additional 3D priors for sparse views but also
improve reconstruction quality by introducing high-frequency
details. In this work, we integrate Difix3D+ [40] as our founda-
tional generative prior to rectify structural defects. Specifically,
it predicts a clean image ˆIfix
i
by conditioning on a reference
view Iref and the current noisy rendering ˆIi:
ˆIfix
i
= Dψ(ˆIi(Ti, θ); Iref, t0),
(11)
Algorithm 2 Geometric Guidance via Diffusion Prior
1: Function 3D-Consistency Distillation
2: Input: Current 3DGS model Gθ, camera pose Ti, refer-
ence image Iref from vitual viewpoint.
3: Given: Pre-trained diffusion denoiser Dψ, noise schedule
¯αt, fixed timestep t0, feature extractor Θvgg.
4: ˆIi ←Gθ(Ti), Ti ∈{Ti}n−1
i=0
▷Eq. 6
5: t ←t0
6: ϵ ∼N(0, I)
7: ˆIi,t ←√¯αt ˆIi + √1 −¯αtϵ
8: ˆIclean
i
←Dψ(ˆIi,t; Iref, t)
9: ˆIfix
i
←sg(ˆIclean)
10: Lgeo ←||Θvgg(sg(ˆIfix
i
)) −Θvgg(ˆIi)||2
2
▷Eq. (12)
11: Return Lgeo
Here, we sample a pre-deblurred image ¯Ij from Eq. 9
as reference image Iref. However, directly utilizing these
generative models presents a critical challenge in sparse-
view scenarios where observations are severely degraded. A
common approach is to reintroduce the generated images into
the 3DGS training pipeline and apply a photorealistic loss
for supervision. While this naive feedback loop improves
sharpness, it introduces a diffusion bias by overemphasizing
pixel-wise similarity, leading to over-smoothing and multi-
view inconsistencies, especially in sparse regions.
To overcome this limitation, we propose a paradigm shift
from using the diffusion model as an offline post-processor to
leveraging it as an online supervisory signal via distillation.
Drawing inspiration from Score Distillation Sampling
(SDS) [58], our method utilizes the pre-trained 2D diffu-
sion model to provide informative gradients that guide the
3D optimization directly. Unlike the original noise-residual
formulation in SDS, we adopt a more stable image-residual
formulation.
The core objective is to guide the 3DGS to render images
that the diffusion model perceives as structurally valid while
maintaining semantic consistency. Specifically, we treat the
single-step denoised output of the diffusion model as a pseudo-
target and minimize its distance to the original 3DGS render-
ing. For each generated view, we formulate this geometric
guidance loss as:
Lgeo = λgeo||Θvgg(sg(ˆIfix
i
)) −Θvgg(ˆIi)||2
2
(12)
where ˆIi is the image rendered from the current state of our
3DGS. ˆIfix
i
= Dψ(ˆIi; Iref, t0) is the predicted clean image
after a single denoising step by the pre-trained Difix3D+
model, given the rendered image ˆIi and reference image Iref
at a fixed timestep t0. The stop-gradient operator, sg(·), is
crucial as it treats the diffusion model’s output as a fixed
target, ensuring that gradients flow exclusively back to the
3DGS parameters θ. Following the official implementation
of Difix3D+ configuration, we use a high-noise timestep by
setting t0 = 199. The detail is outlined in the Alg. 2

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
Available Exploration
rendered
fixed
Minor Exploration
s < smin
Excessive Exploration
s > smax
rendered
fixed
rendered
fixed
Training view
Explored view
Camera Trajectory
Scored-base Interpolation
 
Camera Trajectory
Linear Interpolation
smin < s < smax
Fig. 4: Comparison of different trajectories.
(Left) Linear interpolation between training views provides only limited
angular diversity, often leading to ambiguous geometry. (Right) Our scored-based interpolation samples candidate poses along
the SE(3) geodesic and evaluates each with the diffusion-consistency score ˜s(T). Valid viewpoints (smin ≤˜s(T) ≤smax)
balance geometric diversity and diffusion stability, while excessive or minor exploration causes instability or redundancy. The
proposed consistency-guided sampling adaptively expands recoverable viewpoints and enhances novel-view fidelity.
D. Consistency-guided Camera Exploration
The efficacy of the diffusion prior (Sec. IV-C) critically
depends on the selection of viewpoints, where the rendered
inputs must be informative enough to guide the geometry while
remaining stable enough to prevent generative hallucinations.
The primary challenge lies in exploring novel camera poses
that maximize geometric supervision while maintaining struc-
tural plausibility. As illustrated in Fig. 4, naive linear inter-
polation between training views limits angular diversity and
preserves geometric ambiguities. In contrast, large deviations
from the image plane frequently result in loss of visual context
and significant occlusions. To balance these conflicting factors,
we propose a consistency-guided exploration strategy. This
consistent-aware mechanism identifies and selects viewpoints
that jointly provide high information gains and remain recov-
erable under the diffusion prior.
Scene Adaptive Consistency Normalization (SACN). The
absolute scale of the generative consistency score is inherently
scene dependent, fluctuating with exposure conditions and
blur severity. To normalize the metric, we compute a per-
scene baseline by averaging diffusion-aided consistency over
all training views:
¯s = 1
K
K
X
i=1
Θeval
 ˆIfix(Ti), ˆI(Ti)

,
(13)
where Θeval is a pixel-wise evaluator, Ti ∈{Ti}K
i=1 are the
training poses, ˆIfix(Ti) is the denoised rendering, and ˆI(Ti)
denotes the pose-specific reference.
Diffusion Reliability Metric. For each candidate pose Tnew,
we quantify its diffusion consistency as
s(Tnew) = Θeval
 ˆIfix(Tnew), ˆI(Tnew)

,
(14)
Here, a high score suggests the rendering deviates significantly
from the diffusion prior, indicating potential artifacts or severe
occlusions that the model cannot reliably repair. Conversely,
an excessively low score implies the view is too similar to
the training set or dominated by noise, offering negligible
Algorithm 3 Consistency-guided Camera Exploration
1: Input: Training poses {Ti}, Reference images {¯Ii}
▷
Eq. (9)
2: Models: 3DGS Gθ, Diffusion Denoiser Dψ, Score func-
tion S(·)
3: Params: Thresholds smin, smax, Interpolation interp(·)
4: Initialize: Buffer Bgen ←{Ti}
5: for each adjacent pair (Ti, Ti+1) in Bgen do
6:
Tnew ←interp(Ti, Ti+1)
7:
Inew ←Gθ(Tnew)
8:
Ifix ←Dψ(Inew, ¯Ii, t0)
▷Eq. (11)
9:
scurr ←S(Ifix)
▷Eq. (15)
10:
if smin ≤scurr ≤smax then
11:
Add Ifix to training set
12:
Add Tnew to Bgen
13:
end if
14: end for
geometric gradients. To derive a scene agnostic measure, we
normalize it by the training baseline ¯s:
˜s(Tnew) = s(Tnew) −¯s.
(15)
The normalized deviation ˜s(T) serves as a signal for recover-
ability, indicating how effectively the diffusion prior can refine
the rendering without breaking geometric consistency.
Moderate deviations provide informative geometric cues
with high reliability. In contrast, negligible deviations lead to
redundant coverage, while excessive deviations risk inducing
the unreliable hallucinations.
Band Pass View Selection. Candidate poses are generated by
densely interpolating on the SE(3) manifold between adjacent
pairs (Ti, Ti+1), supplemented by minor extrapolation. This
strategy introduces controlled viewpoint perturbations while
ensuring the poses remain within vicinity of the observed tra-
jectory. We retain candidates satisfying the band-pass criterion:
smin ≤˜s(T) ≤smax.
(16)

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
The selected viewpoints are incorporated into the optimization
loop, adapting the sampling budget toward under-constrained
but recoverable regions. This targeted exploration progres-
sively enhances novel-view fidelity and stabilizes diffusion-
guided supervision with negligible computational overhead.
E. Joint Optimization with Reconstruction and Generation
The final stage of our framework involves jointly optimizing
the 3D scene representation and the camera motion parameters
as shown in Alg. 4. To achieve this, we formulate a composite
objective function that integrates supervisory signals derived
from both the raw blurry observations and the diffusion
rectified novel views. This joint optimization allows the model
to leverage the ground-truth constraints from the captured data
while simultaneously benefiting from the strong generative
prior to deblur and complete unseen regions.
To mitigate the inherent geometric ambiguity in sparse
view reconstruction, we incorporate a depth regularization loss
Lreg inspired by RegNeRF [59]. This loss enforces a local
smoothness prior on the rendered depth maps, which is critical
for regularizing geometry in under-constrained and textureless
regions. It effectively prevents degenerate solutions such as
surface fragmentation and floating artifacts:
Lreg =
n−1
X
i=0

||∇x ˆDi||1 + ||∇y ˆDi||1

(17)
By integrating these complementary supervisory signals, the
total objective function Ltotal is formulated as a weighted sum
of the reconstruction loss, the perceptual restoration loss, the
geometric guidance loss, and the regularization term:
Ltotal = Lblurry + λprLpr + Lgeo + λregLreg
(18)
where λpr, λgeo and λreg are scalar hyperparameters that
balance the contribution of each term.
V. EXPERIMENTS
A. Implementation details
Dataset and preprocessing. We evaluate our method on the
standard Deblur-NeRF benchmark [2], which comprises 5 syn-
thetic and 5 real-world scenes characterized by severe motion
blur. To further assess the generalization capability in complex,
unconstrained outdoor environments, we establish a bench-
mark DL3DV-BLUR based on the DL3DV-10K dataset [60],
simulating realistic motion blur across 5 diverse scenes. For
experimental settings, we adopt sparse-view configurations
with K ∈{3, 6, 9} views. We follow the train/test splitting
protocol of BAD-Gaussians [22]: for the Deblur-NeRF dataset,
we adopt the same train/test split as in their setting, while for
the DL3DV-Blur dataset, we hold out every 7th image as the
test set of novel views. More details of proposed dataset are
in the appendix.
Evaluation metrics. We utilize several metrics to assess the
quality of reconstruction and reenactment. The peak signal-to-
noise ratio (PSNR), Structural Similarity (SSIM), and Learned
Perceptual Image Patch Similarity (LPIPS) [61] are employed
Algorithm 4 Joint Optimization of CoherentGS
1: Input: Sparse, blurry images {Bi}K
i=1.
2: Given: Training iterations Niters, generation interval Ngen,
warmup iterations Nwarmup.
3: Initialize:
4:
Estimate initial poses {Pi}K
i=1.
5:
Initialize 3DGS and motion θ, {Tstart,i, Tend,i}K
i=1.
6:
Initialize a buffer for generated views Bgen = ∅.
7: for iter = 1, . . . , Niters do
8:
// Phase 1: Generative Prior Expansion
9:
if iter % Ngen == 0 and iter ≥Nwarmup then
10:
Φ ←PlanTrajectory(Gθ, {Pi})
▷Sec. IV-D
11:
Iref ←Render from a training virtual pose.
12:
Snew ←EnhanceViews(Gθ, Φ, Iref, Dψ) ▷Sec. IV-C
13:
Append new views Snew to buffer Bgen.
14:
end if
15:
// Phase 2: Reconstruction Optimization
16:
Sample a blurry image Bi and poses {Tstart,i, Tend,i}.
17:
ˆBi ←RenderBlurImage(Gθ, Tstart,i, Tend,i)
▷Eq. (7)
18:
Lblurry ←ComputeLoss( ˆBi, Bi)
▷Eq. (8)
19:
Lpr ←ComputeLoss(Gθ, Bi, Tstart,i, Tend,i) ▷Eq. (10)
20:
if Bgen is not empty then
21:
Sample a generated view {ˆIj, Tj, ˆDj} from Bgen.
22:
Lgeo ←ComputeLoss(Gθ, ˆIj, Iref)
▷Eq. (12)
23:
Lreg ←ComputeLoss( ˆDj)
▷Eq. (17)
24:
else
25:
Lgeo ←0, Lreg ←0
26:
end if
27:
Ltotal ←Lblurry + λprLpr + λgeoLgeo + λregLreg
28:
Update θ, {Tstart,i}, and {Tend,i}.
29: end for
for evaluating image synthesis quality in novel view synthesis
and deblurring view synthesis.
Training Details. Our framework is implemented based on the
official implementation of BAD-Gaussians [22], incorporating
the pre-trained prior from Difix3D+ [40]. We employ the
Adam optimizer for all learnable parameters. The learning
rate schedules and densification strategies for 3D Gaussian
primitives strictly follow the default configuration in [1], [22].
For the camera trajectory modeling (Eq. (6)), the learning
rates for translation and rotation (Tstart, Tend) are initialized at
5×10−3 and exponentially decayed to 5×10−5.The number of
virtual camera poses n in (Eq. (7)) is set to 10. Regarding the
loss terms, we set the blurry weight to λ1 = 0.8, λD-SSIM = 0.2
and depth regularization weight λreg = 0.1. The weights for
the deblurring (λpr) and geometric (λgeo) priors are set to
0.01. To adapt to different scene distributions, the confidence
thresholds for the score model are calibrated per dataset: we
set {smax, smin} to {14.5, 4.5} for the outdoor DL3DV-Blur
dataset, and {8.5, 2.5} for the Deblur-NeRF dataset.The Θeval
in (Eq. (13) and Eq. (14)) are defined as psnr(·). The training
initiates with a warm-up phase of 1500 iterations dedicated to

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
TABLE I: Quantitative comparison of novel view synthesis on the synthetic dataset [2]. We compare the rendering quality
with baselines given 3, 6, and 9 views. Each column is colored as: best and second best.
Method
PSNR↑
SSIM↑
LPIPS↓
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
BAD-Gaussians [22]
19.58
22.67
25.42
22.55
0.555
0.727
0.801
0.694
0.274
0.173
0.104
0.183
Sparse-DeRF∗[39]
19.70
22.25
23.25
21.73
0.561
0.722
0.746
0.676
0.272
0.169
0.143
0.195
Difix3D+ [40]
18.43
19.14
19.74
19.11
0.504
0.551
0.584
0.547
0.307
0.287
0.275
0.289
GenFusion [47]
16.84
18.49
19.47
18.26
0.509
0.556
0.600
0.555
0.507
0.475
0.452
0.478
Ours
21.12
23.87
26.36
23.78
0.671
0.783
0.851
0.768
0.195
0.122
0.08
0.132
Fig. 5: Qualitative results of novel view synthesis on Deblur-NeRF dataset. We evaluate novel view synthesis performance
against baseline methods under 3, 6, and 9 input view settings. Our approach consistently produces higher-fidelity renderings,
recovering significantly more fine-grained details than competing methods.
the deblurring model. Subsequently, we incorporate diffusion
prior and optimize under the guidance of the camera trajectory
model, with the interpolation interval set to 200 steps. In total,
the training spans a total of 7,000 iterations. All experiments
are performed on a NVIDIA A6000 with 48GB memory. More
training details are in the appendix.
Baselines. We benchmark our method against a comprehen-
sive set of state-of-the-art approaches spanning two relevant
domains. Specifically, to validate the reconstruction capability
under sparse and blurry inputs, we compare our approach with
BAD-Gaussians [22] and Sparse-DeRF1 [39]. Furthermore, to
1This method is not open-source, and we reproduct it based on DP-
NeRF [29] and MPR-Net [56], denoted as Sparse-DeRF∗.
assess the performance of generative priors, we compare our
method with recent 3D-aware generative models, including
Difix3D+ [40] and GenFusion [47].
B. Experiment Results
Evaluation of Novel View Synthesis. We benchmark our
method against state-of-the-art approaches, including GenFu-
sion [47], Difix3D+ [40], and Sparse-DeRF [39]. As shown
in Table I and Table III, CoherentGS achieves a great per-
formance lead on the Deblur-NeRF dataset across all sparsity
settings. Notably, in the most challenging 3-view scenario, our
method outperforms GenFusion achieving a 4.28 dB gain in
PSNR. This significant improvement underscores the efficacy

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
TABLE II: Quantitative comparison of novel view synthesis on proposed DL3DV-BLUR. We compare the rendering quality
with baselines given 3, 6, and 9 views. Each column is colored as: best and second best.
Method
PSNR↑
SSIM↑
LPIPS↓
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
BAD-Gaussians [22]
15.08
18.62
19.82
17.84
0.501
0.635
0.671
0.602
0.466
0.313
0.267
0.349
Sparse-DeRF∗[39]
14.84
18.61
19.72
17.72
0.483
0.618
0.674
0.592
0.501
0.325
0.277
0.368
Difix3D+ [40]
14.36
15.53
16.16
15.35
0.492
0.538
0.594
0.541
0.404
0.416
0.336
0.385
GenFusion [47]
12.82
16.42
17.83
15.69
0.502
0.609
0.646
0.586
0.541
0.478
0.469
0.496
Ours
17.48
19.87
21.64
19.67
0.639
0.678
0.731
0.683
0.368
0.267
0.233
0.289
Fig. 6: Qualitative results of novel view synthesis on proposed DL3DV-BLUR Dataset. Compared the novel view with
other methods with baselines rendering quality using 3, 6, and 9 input views, our approach produces more realistic rendering
results with fine-grained details.
of our approach in handling the ill-posed problem of sparse
deblurring, where traditional constraints typically collapse.
Qualitative results provided in Fig. 5 highlight the superior
fidelity and structural coherence of CoherentGS. While our
method reconstructs photorealistic textures with sharp edges,
competing approaches suffer from distinct failure modes inher-
ent to their paradigms. Specifically, Sparse-DeRF struggles to
resolve fine-grained geometry due to the ambiguity caused by
sparse, motion-blurred inputs. More critically, while diffusion-
guided methods like Difix3D+ and GenFusion employ strong
generative priors to enhance visual quality, they rely heavily
on fixed camera poses estimated from degraded observa-
tions. This dependency creates a critical bottleneck: when
initial trajectories are inaccurate, the resulting reprojection
misalignments are erroneously baked into the geometry by
the generative prior, causing structural distortions rather than
correcting them. Furthermore, the severe blur corrupts the
semantic evidence required by these priors, making diffusion
models prone to hallucinating content that contradicts the
underlying scene. By jointly refining camera poses and scene
representation, CoherentGS breaks this error propagation loop,
yielding results that are not only visually pleasing but also
geometrically faithful to the true scene.
Evaluation of Generalization in large-scale scenes. To eval-
uate the robustness of CoherentGS in unstructured environ-
ments, we use the proposed DL3DV-Blur which is simulated
motion blur on five outdoor scenes and conduct assessments
under sparse-input protocols (3, 6, and 9 views). As reported

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
TABLE III: Quantitative comparison of novel view synthesis on the real-scene dataset [2]. We compare the rendering quality
with baselines given 3, 6, and 9 views. Each column is colored as: best and second best.
Method
PSNR↑
SSIM↑
LPIPS↓
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
3-view
6-view
9-view
Average
BAD-Gaussians [22]
18.57
20.61
21.99
20.39
0.495
0.619
0.659
0.591
0.337
0.257
0.216
0.270
Sparse-DeRF∗[39]
17.78
20.01
21.16
19.65
0.463
0.598
0.618
0.560
0.379
0.281
0.242
0.300
Difix3D+ [40]
18.06
19.50
19.79
19.12
0.533
0.574
0.585
0.564
0.415
0.390
0.368
0.391
GenFusion [47]
13.42
15.95
16.19
15.18
0.346
0.473
0.472
0.430
0.588
0.533
0.540
0.554
Ours
19.57
22.01
23.44
21.67
0.575
0.681
0.747
0.668
0.297
0.201
0.167
0.221
Fig. 7: Frequency Spectrum Analysis. Compared to BAD-Gaussians and GenFusion, our approach produces a frequency
distribution that closely aligns with the Ground Truth. This confirms that our method effectively recovers realistic textures
while suppressing the directional artifacts common in generative priors.
TABLE IV: Ablation study of our method’s components.
CoherentGS
PSNR↑
SSIM↑
LPIPS↓
baseline
19.58
0.555
0.274
+Geometric Priors
20.59
0.615
0.218
+Deblurring Priors
20.83
0.647
0.207
+Depth Loss
21.12
0.671
0.195
in Table I, CoherentGS consistently outperforms competing
methods across all sparsity levels, demonstrating strong gen-
eralization capabilities in geometrically complex, unseen sce-
narios. Beyond numerical improvements, our method delivers
superior visual fidelity. In contrast to generative 3DGS-based
baselines like Difix3D+ and GenFusion, which tend to over-
smooth fine details, CoherentGS effectively preserves high-
frequency textures and intricate geometric structures. This
results in sharper object boundaries and a more faithful re-
construction of thin or cluttered elements. Qualitative results
in Fig. 6 further highlight these advantages, confirming that
CoherentGS produces geometrically clear and view-consistent
renderings, even when optimized from sparsely sampled and
severely blurred observations.
Spectral Analysis via Fast Fourier Transform. To validate
structural fidelity beyond the spatial domain, we compare the
spectra of BAD-Gaussians, GenFusion, Sparse-Derf,Difix3D+
and our synthesized views against the Ground Truth (GT)
using 2D Fast Fourier Transform (FFT). As shown in Fig. 7,
the GT spectrum exhibits a compact low-frequency center
that decays naturally into rich, slightly asymmetrical high-
frequency structures, reflecting the scene’s diverse and natural
geometric details. Our method produces a spectral profile
highly congruent with the GT, preserving the scale of cen-
tral energy concentration and the irregular high-frequency
distribution with minimal amplitude loss. This indicates that
CoherentGS effectively suppresses blur while recovering re-
alistic textures without introducing structural artifacts. In
contrast, the spectrum of BAD-Gaussians displays a rapid
radial energy decay, indicating a severe attenuation of high-
frequency components. This confirms that the optimization
process over-smooths the reconstruction, failing to recover
fine edges and details. Conversely, GenFusion and Difix3D+
manifests prominent horizontal and vertical spectral spikes
and distinct sidelobes. This anisotropic energy concentra-
tion corresponds to directional grid-like artifacts rather than
naturally distributed textures. Although GenFusion exhibits
strong high-frequency amplitude, its deviation from the GT
pattern implies that the generated details are largely spectral
hallucinations or structural noise. Overall, our method achieves
the closest frequency-domain alignment with GT, corroborat-
ing that our diffusion-guided prior maintains semantic and
structural fidelity while avoiding both over-smoothing and
artifact induction.
Efficiency Analysis of Training and Inference. To verify the
computational efficiency of the proposed method, we compare
the Storage consumption as well as the training and inference

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
Fig. 8: Visual ablation of the Supervised Signals. We com-
pare our model against variants lacking specific components
to validate their contributions.
time of different approaches on the DL3DV-Blur dataset. As
shown in Table V, under a unified setting, the number of
training iterations for Gaussian-based methods is fixed to 7000
steps, while Sparse-DeRF is trained for 200000 steps. After
training, we perform a single rendering of a video sequence
with 30 fps and 240 frames in total. Compared with Bad-
Gaussians, which relies solely on photometric reprojection,
our CoherentGS even after introducing a diffusion prior and a
deblur prior as additional supervision, only increases storage
consumption to 16.8G, and the training time rises by merely
3.1 min. The inference time is 0.6 min, accounting for less than
5% of the overall runtime. This indicates that the extra com-
putational cost remains modest while achieving significantly
improved reconstruction quality. Compared with Difix3D+ and
GenFusion, which depend on complex inference pipelines,
CoherentGS further demonstrates an advantage in overall
efficiency. By explicitly integrating priors during training,
CoherentGS substantially simplifies the inference procedure.
In contrast to Sparse-DeRF, which requires 25.6G memory
and up to 132h of training, CoherentGS is one order of
magnitude more efficient in terms of both memory and time,
while still delivering high-quality reconstructions and being
more friendly to practical computational budgets.
Method
Storage
Training Time
Inference Time
(a) Bad-Gaussians
9.63G
8.6min
0.5min
(b) Difix3D+
14.9G
3.7min
6.9min
(c) Genfusion
20.2G
3.9min
9.19min
(d) Sparse-DeRF
25.6G
132h
1.8min
(e) Ours
16.8G
11.7min
0.6min
TABLE V: Comparison of GPU storage usage, training time,
and inference time of different methods on the DL3DV-
Blur dataset. CoherentGS maintains competitive storage and
training cost while significantly reducing inference overhead.
C. Ablation Study
Effectiveness of Supervised Signals. To dissect the contri-
bution of each component, we conduct an ablation study on
the synthetic dataset under the 3-view setting, sequentially
incorporating the diffusion prior, deblur prior, and depth regu-
larization.We take Bad-gaussians as our baseline. Quantitative
TABLE VI: Ablation study on different camera trajectories.
Trajectory
PSNR↑
SSIM↑
LPIPS↓
Linear interpolation
17.24
0.575
0.372
Ellipse sampling
15.64
0.550
0.428
SACN
17.48
0.639
0.368
results in Table IV show that our full model consistently
yields the best performance across all metrics. The qualitative
comparisons in Fig. 8 further reveal the underlying mechanics
of these components. Specifically, removing the deblur prior
significantly degrades 3D consistency. Under sparse and blurry
conditions, the geometry is inherently ambiguous. Without the
deblur prior acting as a semantic anchor, the diffusion model
struggles to distinguish between high-frequency textures and
blur artifacts, leading to inconsistent hallucinations across
views. By explicitly decoupling texture from motion blur,
the deblur prior provides a cleaner guidance signal for gen-
erative refinement. Complementing this, depth regularization
proves essential for suppressing geometric noise. Omitting
this term results in severe floating artifacts and near-camera
noise in novel views. A critical insight here is that since
the diffusion prior primarily optimizes 2D appearance, it may
satisfy visual constraints by projecting realistic textures onto
erroneous geometry (e.g., floaters). The depth regularization
enforces geometric smoothness, ensuring that the perceptual
improvements translate into correct underlying 3D structures
rather than superficial texture mapping.
Effectiveness of SACN. To validate the effectiveness of
our proposed Scene Adaptive Consistency Normalization,
we compare it against two standard interpolation strategies:
linear interpolation and elliptical trajectory interpolation. As
visualized in Fig. 9, standard strategies suffer from inherent
limitations: Linear interpolation constrains novel views to the
baseline between input frames. This strategy offers minimal
information gain, as the rendered views provide limited angu-
lar variation and fail to expose occluded or under-optimized
regions to the diffusion prior. Consequently, the optimization
becomes inefficient due to semantic redundancies. Conversely,
elliptical interpolation maximizes angular coverage but often
ventures into unobserved regions outside the visual hull of
the input views. Since the diffusion prior lacks contextual
image evidence in these blind spots, it tends to hallucinate
content that is inconsistent with the true scene. Forcing the
model to incorporate these erroneous priors will lead to artifact
propagation and geometric distortion.
In contrast, our strategy achieves reliability-aware explo-
ration. It actively guides the camera to explore regions that
are geometrically uncertain yet semantically recoverable. By
maximizing the potential of the diffusion prior within a reliable
trust region, our method effectively repairs artifacts while
avoiding the semantic inconsistencies caused by aggressive
sampling. Quantitative results in Table VI confirm that this
reference-guided strategy significantly improves reconstruc-
tion quality compared to topology-agnostic sampling methods.
Analysis of Warm-up Strategy. To determine the optimal

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
Fig. 9: Effectiveness of the SACN Strategy. We compare with
widely used linear interpolation and ellipse sampling. The red
boxes show the most prominent differences.
Fig. 10: Analysis of the Warm-up Strategy. The model
achieves the best balance between geometric stability and
semantic refinement at N = 1500, yielding the highest scores.
timing for introducing the generative prior, we evaluate per-
formance across different warm-up iterations. As shown in
Fig. 10, the reconstruction quality exhibits a clear unimodal
trend. Premature Injection: Introducing the diffusion prior
too early (e.g., before the deblurring module converges) is
detrimental. At this stage, the scene representation has not
yet disentangled motion blur from geometry. Consequently,
the generative prior risks interpreting motion blur as intrinsic
texture, erroneously solidifying low-frequency artifacts into
the model. Delayed Injection: Conversely, introducing the
prior too late leaves an insufficient optimization window. With
limited iterations remaining, the gradient guidance from the
diffusion prior fails to fully propagate to unseen regions,
resulting in sub-optimal refinement and limiting the quality of
synthesized views. Our empirical results suggest that waiting
for the explicit deblurring module to stabilize before injecting
the prior achieves the best balance.
VI. CONCLUSION AND LIMITATIONS
Conclusion. In this paper, we present CoherentGS, a novel
3D Gaussian Splatting framework designed to achieve high-
fidelity 3D reconstruction from sparse and motion-blurred
inputs. We identify that these prevalent real-world degrada-
tions create a vicious cycle where sparse views impede blur
resolution, and blur degrades high-frequency details essential
for view alignment, leading to fragmented reconstructions and
low-frequency bias. To effectively break this cycle, our core
contribution is a synergistic dual-prior strategy. This strategy
intelligently integrates a specialized deblurring network for
robust photometric guidance and a powerful diffusion model
providing geometric priors for scene completion. These are
further supported by a consistency-guided camera exploration
module for adaptive viewpoint planning and a depth regular-
ization loss for geometric plausibility. Extensive quantitative
and qualitative experiments on both synthetic and real-world
scenes, utilizing as few as 3, 6, and 9 input views, un-
equivocally demonstrate CoherentGS’s superior performance.
Our method consistently outperforms existing state-of-the-art
approaches, delivering significantly more coherent, detailed,
and visually realistic novel view syntheses under challenging
conditions. CoherentGS thus establishes a new benchmark for
robust 3D reconstruction from degraded inputs, substantially
expanding the practical applicability of 3DGS.
Limitations. While our proposed CoherentGS significantly
advances 3D reconstruction from sparse and motion-blurred
inputs, its current design focuses primarily on these specific
degradations. Our framework is not yet optimized to robustly
handle other common real-world challenges, such as defocus
blur, overexposed or dark images, and complex degraded
effects. Addressing these additional types of degraded inputs,
potentially through the integration of more generalized priors
or multi-degradation modeling, presents a promising avenue
for future work to further broaden CoherentGS’s applicability
in diverse and challenging scenarios.
REFERENCES
[1] B.
Kerbl,
G.
Kopanas,
T.
Leimk¨uhler,
and
G.
Drettakis,
“3d
gaussian
splatting
for
real-time
radiance
field
rendering,”
ACM
Transactions on Graphics (TOG), 2023. [Online]. Available: https:
//repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[2] L. Ma, X. Li, J. Liao, Q. Zhang, X. Wang, J. Wang, and P. V.
Sander, “Deblur-NeRF: Neural Radiance Fields from Blurry Images,”
in Computer Vision and Pattern Recognition (CVPR), 2022. [Online].
Available: https://limacv.github.io/deblurnerf/
[3] J. Oh, J. Chung, D. Lee, and K. M. Lee, “Deblurgs: Gaussian splatting
for camera motion blur,” arXiv preprint arXiv:2404.11358, 2024.
[4] H. Li, H. Cao, B. Feng, Y. Shao, X. Tang, Z. Yan, L. Yuan, Y. Tian, and
Y. Li, “Beyond chemical qa: Evaluating llm’s chemical reasoning with
modular chemical operations,” arXiv preprint arXiv:2505.21318, 2025.
[5] W. Chen and L. Liu, “Deblur-gs: 3d gaussian splatting from camera
motion blurred images,” Proceedings of the ACM on Computer Graphics
and Interactive Techniques, vol. 7, no. 1, pp. 1–15, 2024.
[6] B. Lee, H. Lee, X. Sun, U. Ali, and E. Park, “Deblurring 3d gaussian
splatting,” in European Conference on Computer Vision. Springer, 2024,
pp. 127–143.
[7] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
[8] Z. Tang, J. Zhang, X. Cheng, W. Yu, C. Feng, Y. Pang, B. Lin, and
L. Yuan, “Cycle3d: High-quality and consistent image-to-3d generation
via generation-reconstruction cycle,” arXiv preprint arXiv:2407.19548,
2024.
[9] H. Li, D. Long, L. Yuan, Y. Wang, Y. Tian, X. Wang, and F. Mo, “De-
coupled peak property learning for efficient and interpretable electronic
circular dichroism spectrum prediction,” Nature Computational Science,
vol. 5, no. 3, pp. 234–244, 2025.
[10] J. Wang, Y. Ma, J. Guo, Y. Xiao, G. Huang, and X. Li, “Cove:
Unleashing the diffusion feature correspondence for consistent video
editing,” arXiv preprint arXiv:2406.08850, 2024.
[11] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu,
B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, “In-
stantsplat: Unbounded sparse-view pose-free gaussian splatting in 40
seconds,” 2024.
[12] K. Feng, Y. Ma, B. Wang, C. Qi, H. Chen, Q. Chen, and Z. Wang,
“Dit4edit: Diffusion transformer for image editing,” arXiv preprint
arXiv:2411.03286, 2024.
[13] J. Zhang, Z. Tang, Y. Pang, X. Cheng, P. Jin, Y. Wei, W. Yu, M. Ning,
and L. Yuan, “Repaint123: Fast and high-quality one image to 3d
generation with progressive controllable 2d repainting,” arXiv preprint
arXiv:2312.13271, 2023.
[14] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3d
gaussian splatting: Survey, technologies, challenges, and opportunities,”
IEEE Transactions on Circuits and Systems for Video Technology, 2025.
[15] Y. Zhao, R. Ye, R. Zheng, Z. Cheng, C. Feng, J. Yang, P. Qiao,
C. Liu, and J. Chen, “Tune-your-style: Intensity-tunable 3d style transfer
with gaussian splatting,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2025, pp. 19 032–19 042.
[16] C. Cheng, Y. Hu, S. Yu, B. Zhao, Z. Wang, and H. Wang, “Reggs:
Unposed sparse views gaussian splatting with 3dgs registration,” arXiv
preprint arXiv:2507.08136, 2025.
[17] Z. Liu, J. Su, G. Cai, Y. Chen, B. Zeng, and Z. Wang, “Georgs: Geomet-
ric regularization for real-time novel view synthesis from sparse inputs,”
IEEE Transactions on Circuits and Systems for Video Technology, 2024.
[18] W. Yu, C. Feng, J. Tang, X. Jia, L. Yuan, and Y. Tian, “Evagaussians:
Event stream assisted gaussian splatting from blurry images,” arXiv
preprint arXiv:2405.20224, 2024.
[19] X. Lin, S. Luo, X. Shan, X. Zhou, C. Ren, L. Qi, M.-H. Yang,
and N. Vasconcelos, “Hqgs: High-quality novel view synthesis with
gaussian splatting in degraded scenes,” in The Thirteenth International
Conference on Learning Representations.
[20] J. Zhang, F. Zhan, M. Xu, S. Lu, and E. Xing, “Fregs: 3d gaussian
splatting with progressive frequency regularization,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 21 424–21 433.
[21] Z. Tang, C. Feng, X. Cheng, W. Yu, J. Zhang, Y. Liu, X. Long, W. Wang,
and L. Yuan, “Neuralgs: Bridging neural fields and 3d gaussian splatting
for compact 3d representations,” arXiv preprint arXiv:2503.23162, 2025.
[22] L. Zhao, P. Wang, and P. Liu, “Bad-gaussians: Bundle adjusted deblur
gaussian splatting,” arXiv preprint arXiv:2403.11831, 2024.
[23] S. Dai and Y. Wu, “Motion from blur,” in 2008 IEEE conference on
computer vision and pattern recognition.
IEEE, 2008, pp. 1–8.
[24] H. Son, J. Lee, S. Cho, and S. Lee, “Real-time video deblurring
via lightweight motion compensation,” in Computer Graphics Forum,
vol. 41, no. 7.
Wiley Online Library, 2022, pp. 177–188.
[25] K. Nie, X. Shi, S. Cheng, Z. Gao, and J. Xu, “High frame rate video
reconstruction and deblurring based on dynamic and active pixel vision
image sensor,” IEEE Transactions on Circuits and Systems for Video
Technology, vol. 31, no. 8, pp. 2938–2952, 2020.
[26] P.
Wang,
L.
Zhao,
R.
Ma,
and
P.
Liu,
“BAD-NeRF:
Bundle
Adjusted
Deblur
Neural
Radiance
Fields,”
in
Computer
Vision
and Pattern Recognition (CVPR), 2023. [Online]. Available: https:
//wangpeng000.github.io/BAD-NeRF/
[27] L. Ma, X. Li, J. Liao, Q. Zhang, X. Wang, J. Wang, and P. V.
Sander, “Deblur-nerf: Neural radiance fields from blurry images,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 861–12 870.
[28] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for
View Synthesis,” in European Conference on Computer Vision (ECCV),
2020. [Online]. Available: https://www.matthewtancik.com/nerf
[29] D. Lee, M. Lee, C. Shin, and S. Lee, “Dp-nerf: Deblurred neural
radiance field with physical scene priors,” in Computer Vision and
Pattern Recognition (CVPR), 2023.
[30] D. Lee, J. Oh, J. Rim, S. Cho, and K. M. Lee, “Exblurf: Efficient
radiance fields for extreme motion blurred images,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
17 639–17 648.
[31] C. Peng and R. Chellappa, “Pdrf: progressively deblurring radiance field
for fast scene reconstruction from blurry images,” in Proceedings of the
AAAI Conference on Artificial Intelligence, vol. 37, no. 2, 2023, pp.
2029–2037.
[32] C. Feng, W. Yu, X. Cheng, Z. Tang, J. Zhang, L. Yuan, and Y. Tian,
“Ae-nerf: Augmenting event-based neural radiance fields for non-ideal
conditions and larger scenes,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 2924–2932.
[33] W. Chen and L. Liu, “Deblur-gs: 3d gaussian splatting from camera
motion blurred images,” Proceedings of the ACM on Computer Graphics
and Interactive Techniques, vol. 7, no. 1, pp. 1–15, 2024.
[34] B. Lee, H. Lee, X. Sun, U. Ali, and E. Park, “Deblurring 3d gaussian
splatting,” in European Conference on Computer Vision. Springer, 2024,
pp. 127–143.
[35] Y. Lu, Y. Zhou, D. Liu, T. Liang, and Y. Yin, “Bard-gs: Blur-aware
reconstruction of dynamic scenes via gaussian splatting,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
16 532–16 542.
[36] M.-Q. V. Bui, J. Park, J. L. G. Bello, J. Moon, J. Oh, and M. Kim,
“Mobgs: Motion deblurring dynamic 3d gaussian splatting for blurry
monocular video,” arXiv preprint arXiv:2504.15122, 2025.
[37] C. Feng, Z. Tang, W. Yu, Y. Pang, Y. Zhao, J. Zhao, L. Yuan, and
Y. Tian, “E-4dgs: High-fidelity dynamic reconstruction from the multi-
view event cameras,” in Proceedings of the 33rd ACM International
Conference on Multimedia, 2025, pp. 7356–7365.
[38] R. Wu, Z. Zhang, M. Chen, X. Fan, Z. Yan, and W. Zuo, “Deblur4dgs:
4d gaussian splatting from blurry monocular video,” arXiv preprint
arXiv:2412.06424, 2024.
[39] D. Lee, D. Kim, J. Lee, M. Lee, S. Lee, and S. Lee, “Sparse-derf:
Deblurred neural radiance fields from sparse view,” IEEE Transactions
on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 6842–
6858, 2025.
[40] J. Z. Wu, Y. Zhang, H. Turki, X. Ren, J. Gao, M. Z. Shou, S. Fidler,
Z. Gojcic, and H. Ling, “Difix3d+: Improving 3d reconstructions with
single-step diffusion models,” in Proceedings of the Computer Vision
and Pattern Recognition Conference, 2025, pp. 26 024–26 035.
[41] X. Yin, Q. Zhang, J. Chang, Y. Feng, Q. Fan, X. Yang, C.-M.
Pun, H. Zhang, and X. Cun, “Gsfixer: Improving 3d gaussian splat-
ting with reference-guided video diffusion priors,” arXiv preprint
arXiv:2508.09667, 2025.
[42] Y. Luo, S. Zhou, Y. Lan, X. Pan, and C. C. Loy, “3denhancer:
Consistent multi-view diffusion for 3d enhancement,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
16 430–16 440.
[43] Y. Zhong, Z. Li, D. Z. Chen, L. Hong, and D. Xu, “Taming video
diffusion prior with scene-grounding guidance for 3d gaussian splatting
from sparse inputs,” in Proceedings of the Computer Vision and Pattern
Recognition Conference, 2025, pp. 6133–6143.
[44] W. Yu, J. Xing, L. Yuan, W. Hu, X. Li, Z. Huang, X. Gao, T.-T. Wong,
Y. Shan, and Y. Tian, “Viewcrafter: Taming video diffusion models for
high-fidelity novel view synthesis,” arXiv preprint arXiv:2409.02048,
2024.
[45] F. Liu, W. Sun, H. Wang, Y. Wang, H. Sun, J. Ye, J. Zhang, and Y. Duan,
“Reconx: Reconstruct any scene from sparse views with video diffusion
model,” arXiv preprint arXiv:2408.16767, 2024.
[46] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P.
Srinivasan, D. Verbin, J. T. Barron, B. Poole et al., “Reconfusion: 3d
reconstruction with diffusion priors,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2024, pp.
21 551–21 561.
[47] S. Wu, C. Xu, B. Huang, A. Geiger, and A. Chen, “Genfusion: Closing
the loop between reconstruction and generation via videos,” in Proceed-
ings of the Computer Vision and Pattern Recognition Conference, 2025,
pp. 6078–6088.
[48] A. Paliwal, X. Zhou, W. Ye, J. Xiong, R. Ranjan, and N. K. Kalantari,
“Ri3d: Few-shot gaussian splatting with repair and inpainting diffusion
priors,” arXiv preprint arXiv:2503.10860, 2025.
[49] J.
Wei,
S.
Leutenegger,
and
S.
Schaefer,
“Gsfix3d:
Diffusion-
guided repair of novel views in gaussian splatting,” arXiv preprint
arXiv:2508.14717, 2025.
[50] E. Weber, A. Holynski, V. Jampani, S. Saxena, N. Snavely, A. Kar,
and A. Kanazawa, “Nerfiller: Completing scenes via generative 3d
inpainting,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 20 731–20 741.

<!-- page 14 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
14
[51] Y. Li, X. Wang, J. Wu, Y. Ma, and Z. Jin, “Sparsegs-w: Sparse-view
3d gaussian splatting in the wild with generative priors,” arXiv preprint
arXiv:2503.19452, 2025.
[52] M. YU, W. Hu, J. Xing, and Y. Shan, “Trajectorycrafter: Redirecting
camera trajectory for monocular videos via diffusion models,” arXiv
preprint arXiv:2503.05638, 2025.
[53] Y. Zhang and D. Yan, “Knowledge distillation for image restoration:
Simultaneous learning from degraded and clean images,” in ICASSP
2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP).
IEEE, 2025, pp. 1–5.
[54] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion Revisited,”
in Computer Vision and Pattern Recognition (CVPR), 2016. [Online].
Available: https://github.com/colmap/colmap
[55] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 719–21 728.
[56] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and
L. Shao, “Multi-stage progressive image restoration,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2021, pp. 14 821–14 831.
[57] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.
[58] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-
to-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.
[59] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. Sajjadi, A. Geiger,
and N. Radwan, “RegNeRF: Regularizing Neural Radiance Fields
for
View
Synthesis
from
Sparse
Inputs,”
in
Computer
Vision
and Pattern Recognition (CVPR), 2022. [Online]. Available: https:
//m-niemeyer.github.io/regnerf/
[60] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo,
Z. Yu, Y. Lu et al., “Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 22 160–22 169.
[61] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
[62] S. Zhou, J. Zhang, W. Zuo, H. Xie, J. Pan, and J. S. Ren, “Da-
vanet: Stereo deblurring with view aggregation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2019, pp. 10 996–11 005.
[63] S. Niklaus, L. Mai, and F. Liu, “Video frame interpolation via adaptive
separable convolution,” in IEEE International Conference on Computer
Vision, 2017.
[64] L. Kong, J. Dong, J. Tang, M.-H. Yang, and J. Pan, “Efficient visual
state space model for image deblurring,” in CVPR, 2025.
[65] A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach, “Adversarial
diffusion distillation,” in European Conference on Computer Vision.
Springer, 2025, pp. 87–103.

<!-- page 15 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
15
A. Dataset Details
DL3DV-BLUR Dataset. To rigorously assess the general-
ization capability of CoherentGS in complex, unconstrained
outdoor environments, we establish a new benchmark named
DL3DV-BLUR, derived from five diverse scenes within the
DL3DV-10K dataset [60]. In constructing this dataset, we
strictly adhere to the high-fidelity blur generation protocol
proposed in DAVANet [62]. As analyzed in [62], synthesiz-
ing motion blur by simply averaging frames from low-fps
video sequences is insufficient to approximate realistic long-
exposure photography. Such low-temporal-resolution accumu-
lation inevitably introduces discontinuous ghosting artifacts
and temporal aliasing, which differ significantly from the
continuous integral of light on a camera sensor.
To address this and simulate physically realistic blur, we
adopt the following pipeline:
1) Preprocessing for Diffusion Alignment: First, to en-
sure compatibility with the spatial alignment require-
ments of the latent diffusion model, we resize the
original video resolution from 960 × 540 to 960 × 536.
2) High-Frame-Rate Interpolation: Following the strat-
egy in [62], we employ the high-quality frame in-
terpolation method SepConv-Slomo [63] to temporally
upsample the source footage to 480 fps. This dense
temporal sampling is crucial for filling the gaps between
frames, thereby suppressing discrete artifacts.
3) Blur Generation via Accumulation: We generate the
final motion-blurred images by mathematically approxi-
mating the exposure integration process. This is achieved
by averaging a sliding window of varying sizes (specif-
ically 6 and 10 frames from the interpolated sequence)
centered on the corresponding sharp ground truth frame.
This rigorous process ensures that DL3DV-BLUR exhibits
smooth, and realistic motion trails, providing a challenging
and reliable benchmark for evaluating sparse-view deblurring
performance. The dataset is available at huggingface.
DEBLUR-NERF Dataset. To comprehensively evaluate our
method, we conduct experiments on the standard DEBLUR-
NERF benchmark [2], which comprises both synthetic and
real-world subsets tailored for varying degrees of motion blur.
• Synthetic Scenes: This subset includes five diverse 3D
scenes: Cozyroom, Factory, Pool, Tanabata, and Trophy.
These scenes are rendered using Blender, where pho-
torealistic motion blur is physically simulated by accu-
mulating multiple sub-frames along a continuous cam-
era trajectory during the exposure interval. This process
ensures that the synthesized blur strictly adheres to the
physical image formation model, providing high-quality
blurry inputs paired with corresponding sharp Ground
Truth for quantitative evaluation.
• Real-world Scenes: To assess generalization in un-
constrained environments, the dataset provides 10 real-
world sequences captured with a handheld camera. These
sequences feature complex, non-uniform motion blur
caused by natural camera shake and varying lighting con-
ditions. Since pixel-aligned ground truth is unavailable for
TABLE VII: Image indices in training settings.
View selection
3views
6views
9views
DL3DV-BLUR
5,15,25
2,5,10,15,17,25
1,2,5,10,15,17,22,25
Deblur-NeRF
5,15,25
2,5,10,15,17,25
1,2,5,10,15,17,22,25
these in-the-wild captures, we utilize them primarily for
qualitative analysis to verify the robustness of our method
against real-world degradation.
B. Training Details
Our framework is implemented in PyTorch, building upon
the official codebase of BAD-Gaussians [22] and integrating
the pre-trained prior from Difix3D+ [40].
Optimization. We employ the Adam optimizer for all learn-
able parameters. The learning rate schedules and densifica-
tion strategies for 3D Gaussian primitives strictly follow the
default configuration in [1], [22]. For the camera trajectory
modeling (Eq. (6)), the learning rates for translation and
rotation (Tstart, Tend) are initialized at 5 × 10−3 and expo-
nentially decayed to 5 × 10−5. Regarding the loss terms,
we set the blurry weight and depth regularization weight to
λ1 = 0.8, λD-SSIM = 0.2, λreg = 0.1.The weights for the
deblurring (λpr) and geometric (λgeo) priors are set to 0.01.
Reliability Thresholds. To adapt to different scene distri-
butions, the confidence thresholds for the score model are
calibrated per dataset: we set {smax, smin} to {14.5, 4.5}
for the outdoor DL3DV-Blur dataset, and {8.5, 2.5} for the
synthetic Deblur-NeRF dataset.
Training Protocol. The training spans a total of 7,000 iter-
ations. The process initiates with a warm-up phase of 1500
iterations, focusing on stabilizing the deblurring model and
camera poses. Subsequently, we activate the generative branch:
the Difix prior is queried every 200 iterations to guide the
optimization of under-reconstructed regions. All experiments
are conducted on a single NVIDIA A6000 with 48GB memory,
taking approximately 12.3 minutes per scene.
Training Scene Indices. In our training pipeline, we adopt
fixed viewpoint selection settings for both the Deblur-NeRF
dataset and our proposed dataset. The specific image indices
used to train CoherentGS are detailed in Table VII. Notably,
the indices for the 3-view and 6-view settings are constructed
as subsets of the 9-view configuration.
C. Network Architectures of Pre-trained Priors
To facilitate robust reconstruction under sparse and de-
graded observations, our framework integrates two specialized
pre-trained models. Here, we detail their architectural designs
and specific configurations used in our pipeline.
1) Deblurring Prior: EVSSM: For the task of recovering
sharp semantic cues from motion-blurred inputs, we adopt
the Efficient Visual State Space Model (EVSSM) [64] as our
deblurring backbone. Unlike traditional CNN-based methods,
EVSSM leverages the linear complexity of State Space Models
(SSMs) to efficiently model long-range dependencies, which
is critical for restoring large-scale motion blur patterns.

<!-- page 16 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
16
EVSS
Conv3
EVSS
UP
EVSS
UP
Conv3
EVSS
DOWN
DOWN
EVSS
PConv
DConv
PConv
Patch 
Unfolding
IFFT
W
·
FFT
Patch 
folding
(d) EDFFN
(b) EVSS module
Element-wise
Addition
Depth-wise
Convolution
DConv
Element-wise
Production
·
Quantization
Matrix
W
GeoT
Geometric 
Transformation
S6
Selective 
Scan
Linear
DConv
Norm
·
Linear
(c) EVS block
GeoT
Norm
EDFFN
Norm
EVS
Linear
S6
B
C
∆
DConv1D
B
C
∆
A D
X
Point-wise
Convolution
PConv
GELU
·
Fig. 11: The pipeline of ESSVM [64].
As shown in Fig.11, the architecture comprises two core
components designed to maximize information restoration
while minimizing computational overhead:
• Visual Scanning & Geometric Transformation: To
capture non-local features across different orientations,
the input features Fin undergo geometric transforma-
tions before being processed by SSM core. This multi-
directional scanning strategy enhances receptive field
without increasing parameter count.
• Frequency-Aware Filtering: Recognizing that blur pre-
dominantly affects high-frequency components, EVSSM
incorporates a Frequency EDFFN module. Features are
transformed via Fourier Transform F(·), allowing the net-
work to explicitly filter and restore spectral components:
Ffreq = F(Fin),
Fout = Q(Ffreq),
(19)
where Q(·) denotes the learnable frequency domain filter.
The core temporal modeling is governed by the recursive SSM
equations, allowing effective propagation of sharp details:
h(t) = Ah(t −1) + Bx(t),
y(t) = Ch(t) + Dx(t), (20)
where A, B, C, D are learnable parameters. The final de-
blurred output is obtained via a residual connection: Ideblur =
Iblur + R(Iblur). We utilize the pre-trained weights provided
by [64], which have demonstrated superior performance on
dynamic scene deblurring.
2) Generative Prior: Difix: To rectify geometric artifacts
and hallucinate missing textures in sparse-view settings, we
employ the DIFIX model from Difix3D+ [40] as our generative
prior. Built upon the computationally efficient SD-Turbo [65],
DIFIX is specifically fine-tuned to balance artifact removal
with structural fidelity. As shown in Fig.12 ,the key adaptations
in DIFIX for 3D-consistent enhancement include:
• Reference-Conditioned Attention: To ensure that the
generated content remains consistent with the existing
3D scene, DIFIX injects information from clean refer-
ence views. This is achieved by concatenating the latent
representations of target view ˜I and reference views Iref:
zjoint = E(˜I ⊕Iref) ∈RV ×C×H×W ,
(21)
where E is the encoder and ⊕denotes concatenation.
A self-attention mechanism is then applied to zjoint to
capture cross-view dependencies, enabling the model to
borrow sharp details from reference viewpoints.
• Conservative Noise Level: Instead of full stochastic
generation, DIFIX operates at a reduced noise level (τ =
200). This design choice is crucial for our framework: it
allows the model to function as a soft refiner, removing
floaters and high-frequency artifacts—without deviating
excessively from the original scene layout or introducing
uncontrolled hallucinations.
D. Additional Experiment Results
Per-scene Quantitative Breakdown In this section, we pro-
vide a comprehensive breakdown of the quantitative evaluation
presented in the main manuscript. While the main text reports
aggregated metrics averaged across the DEBLUR-NERF and
DL3DV-BLUR datasets in Table I, Table II, and Table III,
this supplementary section details the scene-wise performance.
Table VIII through Table XVI present individual results for
each scene under sparse input settings . As evidenced by these
detailed comparisons, CoherentGS achieves the best quanti-
tative performance in the majority of scenes. It is worth noting
that while generative baselines like Difix3D+ occasionally
yield competitive scores in specific perceptual metrics, they
often suffer from severe view-inconsistency artifacts. Standard

<!-- page 17 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
17
Input
Reference View
skip connection
zero conv
ResBlock
𝑧∈ℝ! " # $ %
( B V ) H W C  ---> B ( V H W ) C   
Reference Mixing Layer
B ( V H W ) C  ---> ( B V ) H W C 
𝑧′ ∈ℝ! " # $ %
U-Net
Output
Reference View
VAE
Encoder
(Freeze)
VAE
Decoder
(Lora Finetune)
+ Lora Weights
Fig. 12: The pipeline of Difix3D+ [40].
2D metrics of novel view synthesis applied to individual
frames may not fully penalize such geometric discrepancies.
In contrast, our method maintains superior 3D consistency
and structural fidelity while effectively removing motion blur,
demonstrating robust performance across diverse scenarios.
Geometry Analysis via Rendered Depthmap. To validate
the structural consistency of our method, we compare the
depth maps of novel views generated by BAD-Gaussians,
Difix3D+, and our approach. As shown in Fig.13, our method
produces a more continuous and monotonic depth distribu-
tion, with a clear separation between foreground and back-
ground, and smooth depth gradients along object surfaces.
This indicates that the camera trajectory and 3D geometry
are jointly estimated in a more stable manner within a unified
framework. In contrast, the depth maps of BAD-Gaussians
exhibit globally low contrast, with large regions collapsing to
nearly a single depth plane and distant structures being overly
smoothed, reflecting that under severe motion blur, relying
solely on photometric reprojection leads the geometric solution
to degenerate into a low-frequency, blurry structure. Although
Difix3D+ is able to preserve part of the foreground contours,
its depth maps show pronounced speckle-like noise and dis-
continuous local fluctuations. This arises from performing de-
artifacting independently in the 2D image domain without
explicit multi-view geometric consistency constraints, making
it difficult for the diffusion model’s per-view textures to remain
structurally and geometrically consistent when lifted into 3D
space. Overall, this depth comparison further confirms that our
diffusion-guided prior effectively maintains global geometric
consistency and stability while avoiding both over-smoothing
and artifact injection.
Geometric Consistency of the Deblurring Prior Fig.14
shows the visualization of the deblurring prior on the
Deblur-NeRF-Synthetic, Deblur-NeRF-Real, and DL3DV-Blur
datasets. For each dataset, the first four images in each row
are deblurred views produced by EVSSM, and the last image
is the corresponding sharp ground-truth view. We observe that
the deblurring prior mainly enhances texture and edge details,
while the object contours, occlusion relationships, and overall
scene layout across different views remain consistent with the
ground truth, without artifacts. Overall,the results on these
three datasets indicate that the deblurring prior significantly
improves image sharpness without breaking multi-view geo-
metric consistency.

<!-- page 18 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
18
Fig. 13: Depth and Geometry Analysis. Compared to BAD-Gaussians and Difix3D+, our approach produces reconstructions
with significantly improved multi-view geometric consistency, recovering coherent 3D structure while suppressing the directional
artifacts commonly introduced by generative priors.
Fig. 14: Visualization of the deblurring prior on Deblur-NeRF-Synthetic, Deblur-NeRF-Real, and DL3DV-Blur. The deblurring
prior sharpens textures and edges while maintaining consistent multi-view geometry.
Methods
blurcozy2room
blurfactory
blurpool
blurtanabata
blurwine
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 22.21
0.673
0.233
19.90
0.652
0.208
22.67
0.545
0.241
16.42
0.428
0.373
16.70
0.477
0.318
19.58
0.555
0.274
Sparse-DeRF∗[39]
21.67
0.642
0.253
20.48
0.701
0.185
23.11
0.566
0.218
17.12
0.447
0.381
16.15
0.451
0.323
19.70
0.561
0.272
Difix3D+ [40]
20.57
0.653
0.246
18.31
0.479
0.335
21.11
0.505
0.299
15.89
0.426
0.339
16.24
0.458
0.317
18.46
0.504
0.307
GenFusion [47]
19.75
0.626
0.436
16.65
0.545
0.472
16.42
0.438
0.571
15.21
0.449
0.556
16.17
0.492
0.505
16.84
0.510
0.508
Ours
24.50
0.786
0.131
20.94
0.724
0.174
23.24
0.638
0.210
18.58
0.611
0.231
18.37
0.598
0.230
21.13
0.671
0.195
TABLE VIII: Quantitative results of novel view synthesis on the synthetic dataset with 3 views. The per-scene results
sum to the average values provided.Each column is colored as: best and second best .

<!-- page 19 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
19
Methods
blurcozy2room
blurfactory
blurpool
blurtanabata
blurwine
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 25.61
0.795
0.138
22.62
0.795
0.121
25.64
0.691
0.161
19.09
0.587
0.261
20.43
0.668
0.184
22.68
0.707
0.173
Sparse-DeRF∗[39]
23.77
0.754
0.168
22.98
0.815
0.109
25.32
0.694
0.153
19.83
0.708
0.221
19.36
0.638
0.196
22.25
0.722
0.169
Difix3D+ [40]
21.43
0.695
0.210
19.05
0.528
0.314
21.23
0.566
0.275
16.62
0.466
0.335
17.41
0.500
0.299
19.15
0.551
0.287
GenFusion [47]
20.97
0.683
0.386
19.63
0.605
0.447
17.52
0.513
0.513
17.27
0.476
0.535
17.07
0.507
0.498
18.49
0.557
0.476
Ours
26.88
0.843
0.101
23.03
0.842
0.090
26.23
0.735
0.150
21.25
0.736
0.151
21.99
0.762
0.121
23.88
0.784
0.123
TABLE IX: Quantitative results of novel view synthesis on the synthetic dataset with 6 views. The per-scene results sum
to the average values provided.Each column is colored as: best and second best .
Methods
blurcozy2room
blurfactory
blurpool
blurtanabata
blurwine
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 28.45
0.801
0.092
24.38
0.866
0.080
27.05
0.745
0.139
23.34
0.794
0.119
23.92
0.800
0.090
25.43
0.801
0.104
Sparse-DeRF∗[39]
25.71
0.768
0.147
23.69
0.833
0.095
25.66
0.714
0.127
20.66
0.737
0.193
20.55
0.679
0.153
23.25
0.746
0.143
Difix3D+ [40]
22.66
0.749
0.183
19.54
0.538
0.317
21.25
0.604
0.254
17.51
0.512
0.318
17.75
0.517
0.301
19.74
0.584
0.275
GenFusion [47]
21.89
0.744
0.341
19.96
0.589
0.474
18.51
0.602
0.439
18.42
0.521
0.525
18.58
0.548
0.481
19.47
0.601
0.452
Ours
29.37
0.899
0.050
25.46
0.891
0.070
27.78
0.789
0.122
24.64
0.837
0.090
24.56
0.841
0.070
26.36
0.851
0.080
TABLE X: Quantitative results of novel view synthesis on the synthetic dataset with 9 views. The per-scene results sum
to the average values provided. Each column is colored as: best and second best .
Methods
blurball
blurbuick
blurcoffee
blurdecoration
blurheron
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 18.59
0.447
0.401
15.88
0.412
0.368
25.70
0.847
0.176
16.75
0.435
0.390
15.96
0.336
0.353
18.58
0.495
0.338
Sparse-DeRF∗[39]
18.36
0.374
0.422
15.96
0.398
0.371
23.16
0.708
0.267
16.13
0.477
0.382
15.28
0.357
0.452
17.78
0.463
0.379
Difix3D+ [40]
20.98
0.556
0.363
15.23
0.458
0.471
22.48
0.797
0.233
16.22
0.526
0.523
15.38
0.330
0.485
18.06
0.533
0.415
GenFusion [47]
13.49
0.276
0.642
13.02
0.375
0.543
13.79
0.268
0.610
12.65
0.438
0.573
14.19
0.377
0.575
13.43
0.347
0.589
Ours
19.57
0.525
0.358
16.23
0.493
0.317
26.95
0.868
0.152
17.82
0.535
0.378
17.32
0.457
0.281
19.58
0.576
0.297
TABLE XI: Quantitative results of novel view synthesis on the real-scene dataset with 3 views. The per-scene results sum
to the average values provided. Each column is colored as: best and second best .
Methods
blurball
blurbuick
blurcoffee
blurdecoration
blurheron
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 20.94
0.562
0.311
18.62
0.569
0.271
27.24
0.872
0.143
18.43
0.581
0.299
17.84
0.514
0.264
20.61
0.620
0.258
Sparse-DeRF∗[39]
20.62
0.523
0.317
18.16
0.525
0.291
24.32
0.768
0.274
19.16
0.626
0.277
17.75
0.549
0.247
20.01
0.598
0.281
Difix3D+ [40]
21.12
0.565
0.360
17.20
0.510
0.420
24.90
0.844
0.205
17.48
0.553
0.516
16.82
0.397
0.446
19.50
0.574
0.389
GenFusion [47]
18.26
0.488
0.583
17.66
0.566
0.431
15.95
0.385
0.563
12.56
0.489
0.535
15.32
0.438
0.554
15.95
0.473
0.533
Ours
22.31
0.634
0.259
19.98
0.662
0.189
29.77
0.919
0.090
19.26
0.631
0.252
18.69
0.560
0.218
22.00
0.681
0.202
TABLE XII: Quantitative results of novel view synthesis on the real-scene dataset with 6 views. The per-scene results
sum to the average values provided. Each column is colored as: best and second best .
Methods
blurball
blurbuick
blurcoffee
blurdecoration
blurheron
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 22.65
0.648
0.257
19.75
0.567
0.241
28.38
0.888
0.133
20.12
0.666
0.242
19.08
0.529
0.209
22.00
0.660
0.216
Sparse-DeRF∗[39]
21.79
0.645
0.297
18.72
0.521
0.290
26.37
0.748
0.166
20.39
0.692
0.227
18.55
0.483
0.231
21.16
0.618
0.242
Difix3D+ [40]
20.57
0.539
0.328
17.97
0.548
0.402
24.92
0.848
0.201
18.39
0.578
0.462
17.08
0.410
0.446
19.79
0.585
0.368
GenFusion [47]
16.92
0.436
0.610
17.57
0.551
0.455
17.08
0.401
0.564
12.97
0.490
0.548
16.41
0.486
0.524
16.19
0.473
0.540
Ours
24.90
0.743
0.220
21.71
0.758
0.142
30.43
0.921
0.086
20.64
0.705
0.194
19.52
0.612
0.193
23.44
0.748
0.167
TABLE XIII: Quantitative results of novel view synthesis on the real-scene dataset with 9 views. The per-scene results
sum to the average values provided. Each column is colored as: best and second best .

<!-- page 20 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
20
Methods
0001
0002
0003
0004
0005
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 15.60
0.513
0.431
15.08
0.484
0.536
14.21
0.389
0.517
12.55
0.517
0.501
17.95
0.602
0.346
15.08
0.501
0.466
Sparse-DeRF∗[39]
15.68
0.565
0.412
15.01
0.425
0.560
13.32
0.328
0.511
12.79
0.508
0.497
17.39
0.589
0.527
14.84
0.483
0.501
Difix3D+ [40]
13.90
0.475
0.430
14.71
0.439
0.415
14.66
0.422
0.423
12.88
0.528
0.396
15.66
0.596
0.358
14.36
0.492
0.404
GenFusion [47]
12.11
0.452
0.563
13.24
0.521
0.582
13.68
0.421
0.585
12.34
0.551
0.459
12.73
0.568
0.519
12.82
0.503
0.542
Ours
18.26
0.657
0.322
16.80
0.607
0.434
15.67
0.535
0.489
14.73
0.597
0.378
21.95
0.801
0.217
17.48
0.639
0.368
TABLE XIV: Quantitative results of novel view synthesis on the DL3DV-Blur dataset with 3 views. The per-scene results
sum to the average values provided. Each column is colored as: best and second best .
Methods
0001
0002
0003
0004
0005
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 19.49
0.643
0.289
18.35
0.586
0.359
16.07
0.471
0.432
16.77
0.647
0.309
22.43
0.826
0.176
18.62
0.635
0.313
Sparse-DeRF∗[39]
19.35
0.618
0.301
18.61
0.611
0.364
16.11
0.508
0.445
17.11
0.625
0.289
21.90
0.729
0.227
18.61
0.618
0.325
Difix3D+ [40]
16.31
0.552
0.638
16.27
0.519
0.363
14.69
0.421
0.413
14.46
0.604
0.304
15.93
0.597
0.362
15.53
0.538
0.416
GenFusion [47]
15.83
0.605
0.486
17.36
0.621
0.506
14.75
0.463
0.536
16.13
0.616
0.448
18.04
0.740
0.415
16.42
0.609
0.478
Ours
21.89
0.704
0.234
19.06
0.636
0.314
17.32
0.512
0.393
17.46
0.695
0.254
23.77
0.847
0.144
19.90
0.679
0.268
TABLE XV: Quantitative results of novel view synthesis on the DL3DV-Blur dataset with 6 views. The per-scene results
sum to the average values provided. Each column is colored as: best and second best .
Methods
0001
0002
0003
0004
0005
Average
PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓PSNR↑
SSIM↑LPIPS↓
BAD-Gaussians [22] 22.56
0.733
0.213
18.81
0.617
0.321
17.68
0.525
0.369
17.24
0.633
0.270
22.80
0.845
0.161
19.82
0.671
0.267
Sparse-DeRF∗[39]
22.76
0.745
0.235
19.21
0.693
0.291
17.25
0.508
0.386
17.39
0.612
0.278
21.95
0.816
0.196
19.72
0.674
0.277
Difix3D+ [40]
16.95
0.628
0.310
14.86
0.553
0.369
17.30
0.505
0.351
12.99
0.577
0.398
18.69
0.706
0.251
16.16
0.594
0.336
GenFusion [47]
18.87
0.656
0.437
18.11
0.654
0.585
15.70
0.476
0.520
17.83
0.682
0.416
18.67
0.765
0.391
17.84
0.647
0.470
Ours
24.32
0.785
0.177
20.99
0.701
0.266
18.81
0.582
0.332
19.60
0.760
0.211
24.51
0.871
0.129
21.65
0.740
0.223
TABLE XVI: Quantitative results of novel view synthesis on the DL3DV-Blur dataset with 9 views. The per-scene results
sum to the average values provided. Each column is colored as: best and second best .
