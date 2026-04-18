<!-- page 1 -->
JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for
In-the-Wild Monocular Reconstruction
Zihan Lou * 1 Jinlong Fan * 2 Sihan Ma 3 Yuxiang Yang 2 Jing Zhang 1 †
Abstract
Reconstructing high-fidelity animatable 3D hu-
man avatars from monocular RGB videos re-
mains challenging, particularly in unconstrained
in-the-wild scenarios where camera parameters
and human poses from off-the-shelf methods (e.g.,
COLMAP, HMR2.0) are often inaccurate. Splat-
ting (3DGS) advances demonstrate impressive
rendering quality and real-time performance, they
critically depend on precise camera calibration
and pose annotations, limiting their applicabil-
ity in real-world settings. We present JOintGS,
a unified framework that jointly optimizes cam-
era extrinsics, human poses, and 3D Gaussian
representations from coarse initialization through
a synergistic refinement mechanism. Our key in-
sight is that explicit foreground-background disen-
tanglement enables mutual reinforcement: static
background Gaussians anchor camera estimation
via multi-view consistency; refined cameras im-
prove human body alignment through accurate
temporal correspondence; optimized human poses
enhance scene reconstruction by removing dy-
namic artifacts from static constraints. We further
introduce a temporal dynamics module to cap-
ture fine-grained pose-dependent deformations
and a residual color field to model illumination
variations. Extensive experiments on NeuMan
and EMDB datasets demonstrate that JOintGS
achieves superior reconstruction quality, with
2.1 dB PSNR improvement over state-of-the-art
methods on NeuMan dataset, while maintaining
real-time rendering. Notably, our method shows
significantly enhanced robustness to noisy initial-
ization compared to the baseline. Our source
code is available at https://github.com/
MiliLab/JOintGS.
*Equal contribution † Corresponding author.
1School of Com-
puter Science, Wuhan University, China 2Hangzhou Dianzi Univer-
sity, China 3Nanyang Technological University, Singapore. Corre-
spondence to: Jing Zhang <jingzhang.cv@gmail.com>.
Preprint. February 5, 2026.
Figure 1. Comparison with Previous Methods. Unlike existing
approaches that assume fixed camera poses and SMPL parameters
as inputs, our JOintGS performs unified joint optimization through
a synergistic refinement mechanism.
1. Introduction
Reconstructing high-fidelity, animatable 3D human avatars
from monocular videos has emerged as a fundamental chal-
lenge in computer vision with broad applications in vir-
tual reality, telepresence, digital entertainment, and human-
computer interaction (Wang et al., 2024). Recent advances
in neural rendering, particularly 3D Gaussian Splatting
(3DGS) (Kerbl et al., 2023), have demonstrated unprece-
dented rendering quality and efficiency, enabling real-time
photorealistic synthesis. Building upon this success, sev-
eral methods (Qian et al., 2024b; Moon et al., 2024; Guo
et al., 2025; Hu et al., 2024b; Guo et al., 2023; Kocabas
et al., 2024; Zhang et al., 2025; Li et al., 2024a; Shao et al.,
2024) have extended 3DGS to dynamic human reconstruc-
tion, achieving impressive results on controlled datasets
with multi-view captures or precisely calibrated cameras.
However, these methods face a fundamental limitation when
applied to in-the-wild monocular videos: they critically de-
pend on highly accurate camera parameters and human pose
annotations to maintain consistent spatial-temporal align-
ment. This dependency severely limits practical applicabil-
ity, as obtaining such precise estimates remains notoriously
challenging in unconstrained settings. Traditional Structure-
from-Motion (SfM) pipelines like COLMAP (Sch¨onberger
1
arXiv:2602.04317v1  [cs.CV]  4 Feb 2026

<!-- page 2 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
& Frahm, 2016; Sch¨onberger et al., 2016) often struggle
with dynamic scenes, producing noisy camera trajectories
due to insufficient static feature correspondences. Similarly,
monocular human pose estimators (Shan et al., 2022; Xu
et al., 2023; Zhang et al., 2023; Cai et al., 2023; Goel et al.,
2023), while achieving impressive 2D keypoint detection,
frequently yield inaccurate 3D SMPL parameters due to
depth ambiguity and occlusions. Even modest errors in
these prerequisites cascade into severe artifacts such as inac-
curate alignment, temporal inconsistencies, and unrealistic
human-scene interpenetration.
The key insight is that while obtaining precise camera and
pose parameters is difficult, coarse estimates are readily
derived. Rather than treating these initial estimates as fixed
ground truth, we pose a central question: Can we leverage
the rich geometric and photometric constraints inherent in
3DGS-based reconstruction to jointly refine camera trajec-
tories, human poses, and 3D representations? This ques-
tion motivates our proposed framework, JOintGS, which
formulates dynamic human reconstruction as a unified opti-
mization problem.
JOintGS introduces a synergistic refinement mechanism
through explicit foreground-background disentanglement,
where different components mutually reinforce each other
through three complementary pathways: (1) Static back-
ground Gaussians, remaining consistent across frames, nat-
urally provide multi-view geometric constraints for camera
pose estimation. By exploiting photometric consistency
on static regions, we progressively refine camera trajecto-
ries without being affected by dynamic human motion; (2)
With refined cameras establishing accurate spatial-temporal
correspondences, we optimize human poses to minimize re-
projection errors of human silhouettes and appearance, cor-
recting initialization errors from monocular pose estimators;
(3) Improved camera and poses enhance human-scene disen-
tanglement by providing accurate foreground-background
separation. Clean background constraints, in turn, stabilize
camera estimation by removing dynamic artifacts that vi-
olate static scene assumptions. Unlike previous methods
that either treat camera and pose as fixed inputs or opti-
mize them separately (Jiang et al., 2023; Hu et al., 2024a)
(Figure 1), our approach forms a closed-loop system where
each component progressively corrects errors in the others.
This synergistic design enables robust reconstruction from
coarse initialization without requiring pre-calibrated inputs
or expensive preprocessing.
Furthermore, to effectively model the complex dynamics
of human motion, we introduce two complementary com-
ponents: a temporal offset module that learns per-frame
non-rigid geometric deformations to capture fine-grained
changes like clothing wrinkles, and a residual color field that
models appearance variations caused by lighting changes
and view-dependent effects. These modules enable our
method to faithfully reconstruct both geometric and photo-
metric details that are challenging to capture with canonical
representations alone (Weng et al., 2022; Hu et al., 2024a;
Qian et al., 2024b).
We perform comprehensive evaluations on two challeng-
ing in-the-wild datasets, NeuMan (Jiang et al., 2022) and
EMDB (Kaufmann et al., 2023). Experimental results show
that JOintGS delivers superior reconstruction quality, achiev-
ing a 2.2 dB PSNR improvement over SOTA approaches
on the NeuMan dataset, while maintaining real-time ren-
dering performance. Furthermore, JOintGS demonstrates
stronger robustness to noisy initialization, exhibiting only a
0.9 dB PSNR drop at σ=0.01, in contrast to the 3.7 dB drop
observed with the strong baseline HUGS (Kocabas et al.,
2024). Comprehensive ablation studies further confirm the
necessity of the joint optimization strategy and the effec-
tiveness of each component in our synergistic refinement
mechanism.
In summary, our main contributions are:
• We propose JOintGS, a unified framework jointly op-
timizing camera trajectories, human poses, and 3D
Gaussians from coarse initialization, enabling robust,
calibration-free reconstruction and achieving SOTA per-
formance.
• We introduce a synergistic refinement mechanism
through explicit foreground-background disentanglement,
where static backgrounds anchor camera optimization,
refined cameras improve human body alignment, and op-
timized poses enhance scene reconstruction, forming a
closed-loop of mutual reinforcement.
• We design efficient temporal offset and residual mod-
ules capturing fine-grained deformations and appearance
variations while maintaining real-time rendering.
2. Related Work
NeRF-based Human Reconstruction. Neural radiance
fields have enabled photorealistic human avatar reconstruc-
tion from monocular videos (Weng et al., 2022; Jiang et al.,
2022; Li et al., 2024b; Guo et al., 2023; Peng et al., 2023;
Feng et al., 2022; Mihajlovic et al., 2022; Liu et al., 2021).
These methods achieve impressive quality by mapping
posed observations to canonical space via SMPL-guided
deformations (Loper et al., 2023), but suffer from slow ren-
dering due to expensive volume rendering. NeuMan (Jiang
et al., 2022) jointly models humans and static scenes, while
HumanNeRF (Weng et al., 2022) focuses on human-only
reconstruction. However, both assume fixed camera poses,
limiting applicability to in-the-wild scenarios where pre-
calibrated cameras are unavailable.
2

<!-- page 3 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
3DGS-based Human Reconstruction. Recent advances
in 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) of-
fer fast alternatives through explicit point-based representa-
tions. GaussianAvatar (Hu et al., 2024a) binds 3D Gaussians
to SMPL vertices with pose-dependent appearance model-
ing via hash-encoded MLPs, achieving real-time rendering.
3DGS-Avatar (Qian et al., 2024b) explicitly models non-
rigid deformations and pose-dependent color changes with
MLPs, achieving high-fidelity reconstruction at the cost of
increased training time. HUGS (Kocabas et al., 2024) jointly
models humans and scenes using 3D Gaussians but lacks
mechanisms for correcting initialization errors. Most ex-
isting methods critically depend on pre-calibrated cameras
and accurate SMPL parameters as fixed inputs, which are
difficult to obtain in real-world scenarios.
Holistic Human-Scene Reconstruction. Several methods
attempt holistic reconstruction of both humans and scenes.
Vid2Avatar (Guo et al., 2023) employs a human-centric
scene model but struggles with proper multi-view geometry
learning. HSR (Xue et al., 2024) extends Vid2Avatar with
scene fields and holistic representation but shows degraded
performance in outdoor scenes. ODHSR (Zhang et al., 2025)
proposes an online dense reconstruction framework, achiev-
ing impressive results through monocular geometric priors.
However, their online updating strategy processes frames
sequentially, potentially missing global optimization oppor-
tunities available when the full sequence is accessible.
Joint Camera and Pose Optimization. Traditional bundle
adjustment (Triggs et al., 1999) jointly optimizes camera
poses and 3D structure but struggles with dynamic scenes.
Recent learning-based methods incorporate human priors
into structure-from-motion. HSfM (M¨uller et al., 2025)
integrates human reconstruction into classic SfM, demon-
strating that modeling humans improves camera pose accu-
racy. PoseDiffusion (Wang et al., 2023) uses diffusion-aided
bundle adjustment for pose estimation. Unlike these meth-
ods that operate on sparse features or separate optimization
stages, our approach performs synergistic refinement within
a unified differentiable rendering framework, where dense
photometric constraints from 3DGS enable tighter coupling
between camera, pose, and appearance optimization.
3. Method
3.1. Overview
Given a monocular RGB video {It}T
t=1 with coarse cam-
era poses { ˆTt = [ ˆR|ˆt]} from COLMAP (Sch¨onberger &
Frahm, 2016; Sch¨onberger et al., 2016) and initial SMPL
parameters {ˆξt = (ˆθt, ˆβ)} from HMR2.0 (Zhang et al.,
2023; Cai et al., 2023; Goel et al., 2023), our goal is to
reconstruct a high-fidelity animatable human avatar and
scene while simultaneously refining the camera pose and
SMPL parameters.
As illustrated in Figure 2, our joint
optimization framework consists of four key components:
(1) Foreground human representation (§3.2) that models
the avatar in canonical space with pose-driven deformation
and temporal dynamics; (2) Background scene represen-
tation (§3.3) using static 3D Gaussians; (3) Synergistic
refinement mechanism (§3.4) that simultaneously refines
cameras, SMPL parameters, and Gaussian fields through
unified differentiable rendering supervision.
3.2. 3D Human Representation
To model the dynamic human body with temporal variations,
we represent the avatar as a collection of 3D Gaussians
GH = {gH
i }NH
i=1 defined in a canonical space and deformed
to arbitrary poses via learned skinning.
3.2.1. CANONICAL GAUSSIAN FIELD
Following recent advances in Gaussian-based avatars (Ko-
cabas et al., 2024; Hu et al., 2024b), we establish a canonical
space corresponding to SMPL rest pose (e.g., A-pose). Each
human Gaussian gH
i is parameterized by its canonical at-
tributes: center position µc
i ∈R3, rotation Rc
i ∈SO(3),
scale Sc
i ∈R3
+, opacity αi ∈[0, 1], SH coefficients ci, and
learned LBS weights wi ∈RK. These canonical attributes
are decoded from features sampled on a Triplane represen-
tation. We initialize GH by uniformly sampling NH=110k
points on the SMPL mesh in the canonical pose. Each Gaus-
sian inherits LBS weights wi ∈RK from its nearest SMPL
vertex via barycentric interpolation, where K=24 denotes
the number of joints.
3.2.2. POSE-DRIVEN DEFORMATION
To render canonical Gaussians under pose θt at frame t, we
apply standard LBS deformation (Jung et al., 2023; Li et al.,
2024b; Pang et al., 2024). Given SMPL parameters (θt, β)
and per-joint transformation matrices {T t
k ∈SE(3)}K
k=1
computed via forward kinematics, we transform each Gaus-
sian’s attributes as:
µt
i =
K
X
k=1
wi,k Tk(θt)
µc
i
1

,
(1)
Rt
i =
 K
X
k=1
wi,k Rk(θt)
!
Rc
i,
(2)
where
µc
i; 1
denotes homogeneous coordinates and
Rk(θt) = Tk(θt):3,:3 extracts the rotation component. We
blend only rotational components for orientation to maintain
shape consistency (Qian et al., 2024a). The covariance is
updated as Σt
i = Rt
i Sc
i(Sc
i)⊤(Rt
i)⊤.
3

<!-- page 4 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Figure 2. JOintGS Framework Overview. Given a monocular RGB video with coarse camera poses T = [R|t] from COLMAP
and initial SMPL parameters ξ = (β, θ) from HMR2.0, we initialize scene Gaussians GB (from COLMAP point cloud) and human
Gaussians GH (from SMPL vertices) in canonical space. Our synergistic refinement mechanism (highlighted by orange gradient flow)
jointly optimizes camera pose corrections ∆T , SMPL parameter refinements ∆ξ, and Gaussian attributes {GH, GB} through unified
differentiable rendering supervision. The optimization operates through three complementary pathways: (1) Background-anchored
camera refinement: static scene Gaussians provide multi-view geometric constraints via photometric loss LB on background regions; (2)
Camera-guided human optimization: refined cameras enable accurate temporal correspondence for SMPL parameter optimization
via human rendering loss LH; (3) Pose-aware Gaussian optimization: improved camera and SMPL parameters enhance foreground-
background disentanglement, facilitating Gaussian field optimization with photometric losses Lrender. This closed-loop mutual refinement
enables robust reconstruction from noisy initialization without requiring pre-calibrated inputs.
3.2.3. TEMPORAL DYNAMICS MODELING
While LBS-based deformation handles skeletal articulation,
it cannot capture non-rigid dynamics such as clothing wrin-
kles and appearance variations caused by motions and light-
ing. To address this limitation, we introduce two lightweight
modules that model residual deformations and appearance
changes beyond the skeletal motion:
Temporal Offset Module. We model per-frame non-rigid
geometric deformations via a shallow MLP Foffset. It pre-
dicts positional and rotational offsets conditioned on the
canonical position µc
i and the encoded frame index t:
[∆µt
i, ∆Rt
i] = Foffset(E(µc
i), γ(t)),
(3)
where E(·) denotes multi-resolution hash encoding (M¨uller
et al., 2022) and γ(·) denotes positional encoding (Vaswani
et al., 2017). The final posed position and rotation incorpo-
rate these offsets by:
µt
i ←µt
i + ∆µt
i,
Rt
i ←Rt
i · ∆Rt
i.
(4)
Temporal Color Module. To model temporal appearance
variations (e.g., shadows, lighting changes), we predict a
per-Gaussian color residual via another MLP Fcolor, which
is added to the base spherical harmonics color during ren-
dering:
ct
i(µc
i) = cSH
i (µc
i) + Fcolor(E(µc
i), γ(t)).
(5)
These modules are regularized to remain small (see §3.5),
ensuring they capture only the residual dynamics beyond
skeletal motion.
3.3. Background Scene Representation
We represent the static scene as a set of 3D Gaussians, de-
noted as GB = {gB
i }NB
i=1, where each Gaussian is character-
ized by a center position µi ∈R3, a 3D covariance matrix
Σi, an opacity αi ∈[0, 1], and a view-dependent color pa-
rameterized by SH coefficients. We initialize GB from the
sparse point cloud reconstructed by COLMAP, which typi-
cally contains between 10k and 50k points, depending on
the scene complexity. The initial Gaussian scales are set
proportionally to the local point density, while their colors
are inherited from the nearest image observations.
3.4. Synergistic Refinement Mechanism
The core innovation of our method lies in the synergistic re-
finement of three interdependent components, camera poses
{Tt}, SMPL parameters {(θt, β)}, and Gaussian fields
{GH, GB}, within a unified differentiable rendering frame-
work. Unlike prior works that optimize these components
separately (Qian et al., 2024a) or assume pre-calibrated
inputs (Jiang et al., 2023; Xu et al., 2024), our approach
exploits their mutual dependencies through a closed-loop
refinement process where each component progressively
corrects errors in the others. To establish a unified world
coordinate system, we employ RANSAC (Fischler & Bolles,
1981) to fit scale-shift parameters, aligning SMPL depths to
COLMAP’s metric scale following.
This synergy operates through three complementary path-
ways that form a closed-loop of mutual reinforcement:
4

<!-- page 5 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Background-Anchored Camera Refinement. Static back-
ground Gaussians GB provide reliable multi-view geometric
constraints across frames. Given initial camera pose ˆTt
from COLMAP, we parameterize a learnable correction:
Tt = ∆Tt ◦ˆTt,
(6)
where ∆Tt ∈SE(3) is represented via axis-angle and trans-
lation. We optimize ∆Tt by minimizing photometric errors
on static regions identified using human masks Mt from
SAM (Kirillov et al., 2023; Ravi et al., 2024):
LB = ∥(1 −Mt) ⊙(It −ˆIB
t )∥1,
(7)
where ˆIB
t is rendered from GB and ⊙denotes element-wise
multiplication. By isolating camera optimization from dy-
namic human motion, this strategy prevents artifacts from
violating temporal consistency constraints.
Camera-Guided Human Pose Optimization. With refined
cameras establishing accurate spatial-temporal correspon-
dences, we optimize SMPL parameters {(θt, β)} through
photometric and silhouette supervision:
LH = ∥Mt ⊙(It −ˆIH
t )∥1 + ∥Mt −ˆMH
t ∥1,
(8)
where ˆIH
t and ˆMH
t are rendered from GH. Improved cam-
era alignment enables gradients from these losses to flow
directly to SMPL parameters via the differentiable LBS
transformation (Eq. 1–2), correcting initialization errors
from monocular pose estimators.
Pose-Aware Gaussian Optimization. Refined camera and
SMPL parameters enhance foreground-background disen-
tanglement by providing accurate skeletal priors.
This
reduces foreground leakage into GB and eliminates back-
ground artifacts from GH. We optimize Gaussian parameters
using photometric losses:
Lrender = λrgbLrgb + λssimLssim + λlpipsLlpips,
(9)
where Lrgb = ∥It −ˆIt∥1 is ℓ1 loss, Lssim (Wang et al., 2004)
measures structural similarity, and Llpips (Zhang et al., 2018)
captures perceptual quality. Cleaner scene separation further
stabilizes camera estimation by removing dynamic motion
from static constraints, completing the feedback loop.
3.5. Training Strategy and Objectives
We employ a structured three-stage optimization schedule
to prevent degenerate solutions and ensure stable conver-
gence. (i) Warm-up: We optimize only Gaussian param-
eters {GH, GB} with fixed camera and SMPL parameters,
establishing a reliable geometry prior before moving to
pose-related updates. (ii) Independent Optimization: We
simultaneously enable the optimization of camera poses
and SMPL parameters {(θt, β)}. During this stage, these
components are updated independently to avoid gradient
interference: cameras are primarily anchored by static back-
ground cues for multi-view consistency, while SMPL param-
eters are refined to correct pose initialization errors based
on human-centric gradients. (iii) Joint Optimization: We
perform full optimization with complete losses, allowing
for synergistic refinement where camera trajectories, body
poses, and Gaussian attributes mutually correct each other
to achieve global consistency. More details are provided in
the supplementary material.
To prevent overfitting and maintain generalization, we in-
troduce extra complementary regularizations. LBS weight
regularization constrains learned weights to remain close to
SMPL initialization:
Llbs =
NH
X
i=1
∥wi −wSMPL
i
∥2
2,
(10)
preventing skinning weights from overfitting to training
poses. Offset regularization penalizes large deformations to
ensure temporal modules capture only residual dynamics:
Loffset =
NH
X
i=1
T
X
t=1
(∥∆µt
i∥2
2+∥∆Rt
i−I∥2
F +∥∆ct
i∥2
2), (11)
where I is the identity rotation. Canonical regularization
softly anchors human Gaussians near SMPL mesh surface:
Lcanonical =
NH
X
i=1
min
j∈{1,...,Nv} ∥µc
i −¯vj∥2
2,
(12)
where {¯vj} are canonical SMPL vertices.
4. Experiments
We conduct comprehensive experiments to evaluate the
effectiveness of our proposed method. We first describe
the datasets and evaluation metrics (§4.1), followed by im-
plementation details (§4.2). We then present quantitative
(§4.3) and qualitative (§4.4) comparisons with state-of-the-
art methods. Finally, ablation studies validate the design
choices of our key components (§4.5).
4.1. Dataset
NeuMan Dataset (Jiang et al., 2022) comprises six in-the-
wild sequences (Seattle, Citron, Parking, Bike, Jogging,
Lab), each capturing a single person performing various
activities over 10–20 seconds. The videos are recorded with
a handheld mobile phone exhibiting natural camera motion,
which provides sufficient viewpoint diversity for multi-view
reconstruction. We follow the original split protocol (Jiang
et al., 2022), allocating 80% of frames for training, 10% for
validation, and 10% for testing. Notably, we do not use the
5

<!-- page 6 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Human-only
Full-image
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Vid2Avatar (Guo et al., 2023)
30.96
0.981
0.018
15.64
0.551
0.572
HUGS (Kocabas et al., 2024)
30.13
0.977
0.017
26.66
0.851
0.126
HSR (Xue et al., 2024)
29.03
0.971
0.026
21.67
0.669
0.526
ODHSR (Zhang et al., 2025)
32.07
0.981
0.016
27.78
0.870
0.153
ExAvatar (Moon et al., 2024)
31.39
0.981
0.016
-
-
-
Vid2Avatar-Pro (Guo et al., 2025)
32.71
0.983
0.019
-
-
-
JOintGS (Ours)
34.84
0.984
0.010
30.23
0.913
0.072
Table 1. Quantitative evaluation on NeuMan dataset (Jiang et al., 2022). We report performance on both human-only regions and entire
frames (Full-image). For the human-only setting, we render the avatar on a white background for all baselines and compute metrics over
the whole image.
Human-only
Full-image
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
GauHuman (Hu et al., 2024b)
25.31
0.943
0.057
-
-
-
3DGS-Avatar (Qian et al., 2024b)
27.95
0.967
0.035
-
-
-
Vid2Avatar (Guo et al., 2023)
24.25
0.948
0.061
16.65
0.413
0.599
HUGS (Kocabas et al., 2024)
26.16
0.947
0.033
21.60
0.659
0.181
HSR (Xue et al., 2024)
25.12
0.920
0.054
18.67
0.463
0.632
ODHSR (Zhang et al., 2025)
28.95
0.966
0.031
23.79
0.767
0.197
JOintGS (Ours)
30.99
0.972
0.027
23.40
0.785
0.173
Table 2. Quantitative evaluation on EMDB dataset (Kaufmann et al., 2023). We report metrics on both human-only regions and entire
frames (Full-image) to provide a multi-faceted assessment. Our method outperforms both offline and online baselines.
keyframe selection strategy on this dataset to enable direct
comparison with prior offline methods. This dataset serves
as our primary benchmark for quantitative evaluation.
EMDB Dataset (Kaufmann et al., 2023) is a large-scale
in-the-wild dataset consisting of 81 video sequences from
10 subjects, totaling 58 minutes of motion data. The dataset
is captured using Wearable trackers and a handheld iPhone,
providing ground-truth global camera poses and body root
trajectories via wireless motion capture sensors. We se-
lect ten representative sequences that present diverse chal-
lenges: extended trajectories, occlusions, complex lighting
(shadows), and unconventional poses (e.g., cartwheels). We
use the first 200 frames from each sequence and adopt an
80%/10%/10% train/val/test split.
Baselines. We compare against state-of-the-art methods
across different categories: (1) NeRF-based human re-
construction: HumanNeRF (Weng et al., 2022), InstantA-
vatar (Jiang et al., 2023); (2) 3DGS-based human reconstruc-
tion: 3DGS-Avatar (Qian et al., 2024b), GaussianAvatar (Hu
et al., 2024a), ExAvatar (Moon et al., 2024); (3) Video-
based human reconstruction: Vid2Avatar (Guo et al., 2023),
Vid2Avatar-Pro (Guo et al., 2025); (4) Holistic human-scene
reconstruction: NeuMan (Jiang et al., 2022), HUGS (Ko-
cabas et al., 2024), HSR (Xue et al., 2024), ODHSR (Zhang
et al., 2025).
Evaluation Metrics. We evaluate reconstruction quality
using standard photometric metrics: PSNR, SSIM (Wang
et al., 2004), and LPIPS (Zhang et al., 2018). For human-
only rendering, we composite the reconstructed avatar onto
a white background and compute metrics over the entire
image region. We also report training time (hours) and
rendering speed (FPS) to assess computational efficiency.
4.2. Implementation Details
All experiments were conducted on a single NVIDIA RTX
5090 GPU (32GB). The model was optimized using the
Adam (Kinga et al., 2015) optimizer, with a stage-dependent
learning rate scheduling strategy employed throughout the
training process. Training converged in approximately 25
minutes over 15,000 iterations. Detailed loss weights, learn-
ing rate schedules, initialization methods, and network ar-
chitectures are provided in the supplementary material.
4.3. Quantitative Results
Novel View Synthesis on NeuMan. Table 1 presents quan-
titative evaluation results on the NeuMan dataset.
Our
method achieves superior performance across all metrics,
an average PSNR of 34.84 dB, surpassing the previous best
Vid2Avatar-Pro (Guo et al., 2025) (32.71 dB) by 2.13 dB.
Notably, 3DGS-based methods (3DGS-Avatar, GaussianA-
vatar, HUGS, Ours) consistently outperform NeRF-based
approaches (HumanNeRF, InstantAvatar) in both quality
and speed, demonstrating the advantages of explicit 3D
6

<!-- page 7 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Figure 3. Qualitative comparison on NeuMan dataset. For each scene, we show full-body rendering (left) and zoomed-in details (right).
Figure 4. Novel Pose Synthesis And Environment Transfer. The
reconstructed avatar can be animated with arbitrary poses while
maintaining photorealistic appearance and seamless integration
with other environments.
Figure 5. Robustness Comparison on noisy initialization.
Gaussian representation for human modeling. Our method
also achieves highest SSIM (0.984) and LPIPS (0.010), indi-
cating excellent structural similarity and perceptual quality.
Novel View Synthesis on EMDB. Table 2 shows results
on the more challenging EMDB dataset featuring complex
poses and occlusions. Our method achieves 30.99 dB PSNR,
surpassing the recent online method ODHSR (Zhang et al.,
2025) (28.95 dB) by 2.04 dB. Despite ODHSR’s online
advantage enabling real-time tracking, our offline global
optimization leverages full sequence information for more
robust reconstruction. The consistent improvements across
both datasets validate the robustness and generalizability of
our approach.
Efficiency Analysis. JointGS requires approximately 23
minutes for training on average, slightly faster than HUGS
(25 minutes), while achieving significantly better quality
(2.13 dB PSNR). Our method maintains real-time rendering
at 27.3 FPS versus HUGS’s 27.5 FPS. This efficiency is
attributed to our lightweight temporal offset module and
residual color field.
Robustness to Initialization Errors. Figure 5 analyzes
the robustness of our method to noisy initialization of cam-
era poses and SMPL parameters. We incrementally add
Gaussian noise to the initialization with standard deviations
ranging from 0 to 0.02 (normalized scale). Our method
exhibits significantly slower performance degradation com-
pared to HUGS: at σ=0.01, our PSNR drops by only 0.9 dB
versus HUGS’s 3.7 dB drop, demonstrating a 2.8 dB ro-
bustness advantage. This robustness is achieved through
the synergistic refinement mechanism, which iteratively re-
fines both camera and SMPL parameters within the joint
optimization framework.
Background Reconstruction Results. While our back-
ground model primarily serves to provide correct context
(e.g., occlusion and depth cues) and anchor camera refine-
ment, it also yields high-quality scene reconstruction as a
byproduct of joint optimization. As shown in Tables 1 and
2, our method surpasses the recent ODHSR by 2.45 dB
in PSNR on the NeuMan dataset and achieves comparable
performance on EMDB. Per-scene breakdowns and visual
comparisons are provided in the supplementary material.
7

<!-- page 8 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Configuration
PSNR↑
SSIM↑
LPIPS↓
w/o Synergistic
31.38
0.976
0.017
w/o Dynamics
34.23
0.983
0.011
Full Model (Ours)
34.84
0.984
0.010
Table 3. Ablation study on NeuMan dataset (average across six
sequences). Each component contributes to the final performance,
with synergistic refinement providing the largest gain.
4.4. Qualitative Results
Human Reconstruction Quality. Figure 3 presents qual-
itative comparisons of human reconstruction against Neu-
Man (Jiang et al., 2022) and HUGS (Kocabas et al., 2024).
For each example, we show the full-body rendering (left)
alongside two zoomed-in views highlighting fine-grained
details. Our method demonstrates superior reconstruction
quality across three key aspects:
Body Alignment.
JointGS accurately aligns the recon-
structed body with the ground-truth pose, avoiding the mis-
alignment artifacts visible in baseline methods. This is
achieved through our synergistic optimization of camera
poses, SMPL parameters, and Gaussian representations.
Detail Preservation. The zoomed-in regions reveal that our
method preserves fine-grained details such as facial features,
clothing wrinkles, and hand gestures more faithfully than
baselines. This benefit stems from the temporal offset mod-
ule, which captures high-frequency deformations beyond
SMPL’s rigid skeletal deformations.
Color Fidelity. Our reconstructions exhibit more accurate
color reproduction compared to HUGS, which suffers from
color bleeding between human and scene. The clean sepa-
ration is enabled by our synergistic refinement mechanism,
which refines SMPL parameters to reduce ambiguity in
human-scene decomposition.
Novel Pose and Environment Transfer. As shown in Fig-
ure 4, our method enables decoupling of human and back-
ground, facilitating novel view synthesis and avatar manip-
ulation. We demonstrate this capability by transferring a
reconstructed human avatar from the Lab sequence to the
Parking sequence environment, and rendering it under a
novel pose extracted from a different subject. The results
exhibit realistic appearance and consistent geometry, vali-
dating that JointGS effectively separates human from the
background while maintaining animatable canonical repre-
sentations.
4.5. Ablation Experiments
We conduct ablation studies to validate the contribution
of each proposed component on the NeuMan dataset (Ta-
ble 3 and Figure 6). Removing the temporal dynamics
Figure 6. Qualitative visualization of the details captured in the hu-
man body reconstruction under different ablations of our method.
module (w/o Dynamics) results in a 0.6 dB PSNR drop
with visible degradation in high-frequency details such as
clothing wrinkles and facial features, validating its neces-
sity for capturing pose-dependent non-rigid deformations
beyond SMPL’s skeletal articulation. More critically, re-
moving synergistic refinement (w/o Synergistic)—which
jointly optimizes cameras, SMPL parameters, and Gaus-
sians—leads to a substantial 3.5 dB degradation with severe
artifacts including misaligned limbs and blurred textures,
demonstrating that correcting initialization errors through
mutual reinforcement is essential for accurate reconstruc-
tion. Our full model achieves 34.84 dB PSNR on average,
confirming that geometric refinement (temporal offsets), ap-
pearance modeling (residual colors), and holistic optimiza-
tion (synergistic mechanism) work synergistically to enable
high-fidelity reconstruction from coarse initialization.
5. Conclusion
We present JointGS, a unified framework for holistic recon-
struction of dynamic humans and static scenes from monoc-
ular RGB videos through synergistic refinement of camera
poses, SMPL parameters, and 3D Gaussian fields. By ex-
ploiting their mutual dependencies within a differentiable
rendering pipeline, our method enables robust reconstruc-
tion from coarse initialization without demanding precise
pre-calibration. Extensive experiments on diverse in-the-
wild datasets demonstrate that JointGS achieves state-of-
the-art performance, while maintaining real-time rendering.
Comprehensive ablation studies validate the effectiveness
of each proposed component, particularly the synergistic
refinement mechanism.
Limitations. Our approach is fundamentally constrained by
the inherent capacity of the SMPL body model, leading to
reduced fidelity in fine-grained regions such as hands and
faces. Although our temporal dynamics module exhibits a
tendency to capture residual detail in these areas, a promis-
ing future direction is to improve hand and face modeling
and enhance the controllability of expressions and gestures
by integrating more expressive parametric models (Shen
et al., 2023), such as SMPL-X (Pavlakos et al., 2019).
8

<!-- page 9 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none of which we feel must be
specifically highlighted here.
References
Cai, Z., Yin, W., Zeng, A., Wei, C., Sun, Q., Yanjun, W.,
Pang, H. E., Mei, H., Zhang, M., Zhang, L., et al. Smpler-
x: Scaling up expressive human pose and shape estima-
tion. Advances in Neural Information Processing Systems,
36:11454–11468, 2023.
Feng, Y., Yang, J., Pollefeys, M., Black, M. J., and Bolkart,
T. Capturing and animation of body and clothing from
monocular video. In SIGGRAPH Asia 2022 Conference
Papers, pp. 1–9, 2022.
Fischler, M. A. and Bolles, R. C. Random sample consensus:
a paradigm for model fitting with applications to image
analysis and automated cartography. Communications of
the ACM, 24(6):381–395, 1981.
Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., and
Malik, J. Humans in 4d: Reconstructing and tracking hu-
mans with transformers. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 14783–
14794, 2023.
Guo, C., Jiang, T., Chen, X., Song, J., and Hilliges, O.
Vid2avatar: 3d avatar reconstruction from videos in the
wild via self-supervised scene decomposition. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 12858–12868, 2023.
Guo, C., Li, J., Kant, Y., Sheikh, Y., Saito, S., and Cao,
C. Vid2avatar-pro: Authentic avatar from videos in the
wild via universal prior. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pp. 5559–
5570, 2025.
Hu, L., Zhang, H., Zhang, Y., Zhou, B., Liu, B., Zhang,
S., and Nie, L. Gaussianavatar: Towards realistic human
avatar modeling from a single video via animatable 3d
gaussians. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 634–644,
2024a.
Hu, S., Hu, T., and Liu, Z. Gauhuman: Articulated gaussian
splatting from monocular human videos. In Proceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 20418–20431, 2024b.
Jiang, T., Chen, X., Song, J., and Hilliges, O. Instantavatar:
Learning avatars from monocular video in 60 seconds. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 16922–16932, 2023.
Jiang, W., Yi, K. M., Samei, G., Tuzel, O., and Ranjan,
A. Neuman: Neural human radiance field from a single
video. In European Conference on Computer Vision, pp.
402–418. Springer, 2022.
Jung, H., Brasch, N., Song, J., Perez-Pellitero, E., Zhou, Y.,
Li, Z., Navab, N., and Busam, B. Deformable 3d gaussian
splatting for animatable human avatars. arXiv preprint
arXiv:2312.15059, 2023.
Kaufmann, M., Song, J., Guo, C., Shen, K., Jiang, T., Tang,
C., Z´arate, J. J., and Hilliges, O. Emdb: The electro-
magnetic database of global 3d human pose and shape in
the wild. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 14632–14643, 2023.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3d
gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4):139–1, 2023.
Kinga, D., Adam, J. B., et al. A method for stochastic
optimization. In International conference on learning
representations (ICLR), volume 5. California;, 2015.
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C.,
Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo,
W.-Y., et al. Segment anything. In Proceedings of the
IEEE/CVF international conference on computer vision,
pp. 4015–4026, 2023.
Kocabas, M., Chang, J.-H. R., Gabriel, J., Tuzel, O., and
Ranjan, A. Hugs: Human gaussian splats. In Proceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 505–515, 2024.
Li, M., Yao, S., Xie, Z., and Chen, K.
Gaussianbody:
Clothed human reconstruction via 3d gaussian splatting.
arXiv preprint arXiv:2401.09720, 2024a.
Li, Z., Zheng, Z., Wang, L., and Liu, Y. Animatable gaus-
sians: Learning pose-dependent gaussian maps for high-
fidelity human avatar modeling. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 19711–19722, 2024b.
Liu, L., Habermann, M., Rudnev, V., Sarkar, K., Gu, J., and
Theobalt, C. Neural actor: Neural free-view synthesis of
human actors with pose control. ACM transactions on
graphics (TOG), 40(6):1–16, 2021.
Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., and
Black, M. J. Smpl: A skinned multi-person linear model.
In Seminal Graphics Papers: Pushing the Boundaries,
Volume 2, pp. 851–866. 2023.
9

<!-- page 10 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Mihajlovic, M., Bansal, A., Zollhoefer, M., Tang, S., and
Saito, S. Keypointnerf: Generalizing image-based volu-
metric avatars using relative spatial encoding of keypoints.
In European conference on computer vision, pp. 179–197.
Springer, 2022.
Moon, G., Shiratori, T., and Saito, S. Expressive whole-
body 3d gaussian avatar. In European Conference on
Computer Vision, pp. 19–35. Springer, 2024.
M¨uller, L., Choi, H., Zhang, A., Yi, B., Malik, J., and
Kanazawa, A. Reconstructing people, places, and cam-
eras. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 21948–21958, 2025.
M¨uller, T., Evans, A., Schied, C., and Keller, A. Instant
neural graphics primitives with a multiresolution hash
encoding. ACM transactions on graphics (TOG), 41(4):
1–15, 2022.
Pang, H., Zhu, H., Kortylewski, A., Theobalt, C., and Haber-
mann, M. Ash: Animatable gaussian splats for efficient
and photoreal human rendering. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 1165–1175, 2024.
Park, K., Sinha, U., Hedman, P., Barron, J. T., Bouaziz,
S., Goldman, D. B., Martin-Brualla, R., and Seitz, S. M.
Hypernerf: A higher-dimensional representation for topo-
logically varying neural radiance fields. arXiv preprint
arXiv:2106.13228, 2021.
Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman,
A. A., Tzionas, D., and Black, M. J. Expressive body
capture: 3d hands, face, and body from a single image.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 10975–10985, 2019.
Peng, S., Geng, C., Zhang, Y., Xu, Y., Wang, Q., Shuai, Q.,
Zhou, X., and Bao, H. Implicit neural representations
with structured latent codes for human body modeling.
IEEE Transactions on Pattern Analysis and Machine In-
telligence, 45(8):9895–9907, 2023.
Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Gieben-
hain, S., and Nießner, M. Gaussianavatars: Photorealistic
head avatars with rigged 3d gaussians. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 20299–20309, 2024a.
Qian, Z., Wang, S., Mihajlovic, M., Geiger, A., and Tang,
S. 3dgs-avatar: Animatable avatars via deformable 3d
gaussian splatting. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pp.
5020–5030, 2024b.
Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T.,
Khedr, H., R¨adle, R., Rolland, C., Gustafson, L., et al.
Sam 2: Segment anything in images and videos. arXiv
preprint arXiv:2408.00714, 2024.
Sch¨onberger, J. L. and Frahm, J.-M. Structure-from-motion
revisited. In Conference on Computer Vision and Pattern
Recognition (CVPR), 2016.
Sch¨onberger, J. L., Zheng, E., Pollefeys, M., and Frahm,
J.-M. Pixelwise view selection for unstructured multi-
view stereo. In European Conference on Computer Vision
(ECCV), 2016.
Shan, W., Liu, Z., Zhang, X., Wang, S., Ma, S., and Gao, W.
P-stmo: Pre-trained spatial temporal many-to-one model
for 3d human pose estimation. In European Conference
on Computer Vision, pp. 461–478. Springer, 2022.
Shao, Z., Wang, Z., Li, Z., Wang, D., Lin, X., Zhang, Y., Fan,
M., and Wang, Z. Splattingavatar: Realistic real-time hu-
man avatars with mesh-embedded gaussian splatting. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 1606–1616, 2024.
Shen, K., Guo, C., Kaufmann, M., Zarate, J. J., Valentin, J.,
Song, J., and Hilliges, O. X-avatar: Expressive human
avatars. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 16911–
16921, 2023.
Triggs, B., McLauchlan, P. F., Hartley, R. I., and Fitzgibbon,
A. W. Bundle adjustment—a modern synthesis. In In-
ternational workshop on vision algorithms, pp. 298–372.
Springer, 1999.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At-
tention is all you need. Advances in neural information
processing systems, 30, 2017.
Wang, J., Rupprecht, C., and Novotny, D. Posediffusion:
Solving pose estimation via diffusion-aided bundle ad-
justment. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 9773–9783, 2023.
Wang, R., Cao, Y., Han, K., and Wong, K.-Y. K. A survey
on 3d human avatar modeling–from reconstruction to
generation. arXiv preprint arXiv:2406.04253, 2024.
Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P.
Image quality assessment: from error visibility to struc-
tural similarity. IEEE transactions on image processing,
13(4):600–612, 2004.
Weng, C.-Y., Curless, B., Srinivasan, P. P., Barron, J. T., and
Kemelmacher-Shlizerman, I. Humannerf: Free-viewpoint
rendering of moving people from monocular video. In
10

<!-- page 11 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Proceedings of the IEEE/CVF conference on computer
vision and pattern Recognition, pp. 16210–16220, 2022.
Xu, C., Tan, R. T., Tan, Y., Chen, S., Wang, X., and Wang, Y.
Auxiliary tasks benefit 3d skeleton-based human motion
prediction. In Proceedings of the IEEE/CVF international
conference on computer vision, pp. 9509–9520, 2023.
Xu, Y., Chen, B., Li, Z., Zhang, H., Wang, L., Zheng, Z.,
and Liu, Y. Gaussian head avatar: Ultra high-fidelity
head avatar via dynamic gaussians. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2024.
Xue, L., Guo, C., Zheng, C., Wang, F., Jiang, T., Ho, H.-I.,
Kaufmann, M., Song, J., and Hilliges, O. Hsr: holistic 3d
human-scene reconstruction from monocular videos. In
European Conference on Computer Vision, pp. 429–448.
Springer, 2024.
Zhang, H., Tian, Y., Zhang, Y., Li, M., An, L., Sun, Z., and
Liu, Y. Pymaf-x: Towards well-aligned full-body model
regression from monocular images. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 45(10):
12287–12303, 2023.
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang,
O. The unreasonable effectiveness of deep features as a
perceptual metric. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pp. 586–595,
2018.
Zhang, Z., Kaufmann, M., Xue, L., Song, J., and Oswald,
M. R. Odhsr: Online dense 3d reconstruction of humans
and scenes from monocular videos. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pp.
21824–21835, 2025.
11

<!-- page 12 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
A. Overview
This supplementary material provides further details regarding our proposed model, JOintGS. These details were omitted
from the main paper due to space constraints. Furthermore, we include additional experimental results demonstrating that
our model achieves State-of-the-Art (SOTA) performance in background reconstruction, even though the background was
not the primary target of our optimization.
The supplementary material is organized as follows:
• SectionB: Detailed network architectures, advanced training schemes, and the loss function formulation.
• SectionC: Additional Experimental Results and Additional Demonstration Video.
B. Implementation Details
B.1. Network Architectures
We present a lightweight Multi-Layer Perceptron (MLP) network for predicting individual Gaussian properties, as illustrated
in Figure S2.
Following HUGS (Kocabas et al., 2024), we employ a Tri-Plane encoding to extract spatial features for each Gaussian,
which serve as input to our MLP-based decoder networks. Specifically, for a Gaussian positioned at p ∈R3, we first
project it onto three orthogonal planes (XY, YZ, XZ), obtaining corresponding 2D coordinates: pxy, pyz, pxz. Utilizing
bilinear interpolation, we sample the triplane feature maps at these coordinates, yielding three feature vectors: fxy, fyz, fxz.
These vectors are concatenated to form the final sampling strategyspatial feature vector fspatial = [fxy; fyz; fxz]. To capture
the temporal dynamics of human motion, we further augment the spatial feature with a time embedding, obtained by
applying a positional encoding function to the time step t. Consequently, the final input feature for each Gaussian is
finput = [fspatial; ftime].
We design separate Appearance and Geometry Decoders to predict distinct Gaussian attributes. The Geometry Decoder
is responsible for estimating spatial properties, including mean position (µ), rotation (r), and scale (s). Conversely,
the Appearance Decoder focuses on predicting visual attributes, namely color (c) and opacity (o). This architectural
disentanglement enables more specialized learning and enhanced representation of geometric and appearance features.
Figure S1. Ablation study on the phased optimization schedule. From left to right: the Ground Truth reference, results of our full
three-stage optimization schedule, and results of performing joint optimization only (without warm-up and independent stages).
B.2. Training Details
As introduced in the main paper (Sec. 3.5), we employ a carefully designed four-stage optimization schedule. During the
Warm-up Stage, the initial learning rate is strategically set to a relatively high value. Specifically, the Gaussian attributes
of the background component (µ, o, s, c, a) are assigned learning rates of 1.6 × 10−4, 0.05, 0.005, 0.001, and 0.0025,
respectively, while the foreground position (µ) and the triplane network receive learning rates of 1.6 × 10−4 and 0.001.
This stage proceeds for 5,000 iterations. For computational efficiency, the Camera Optimization and Human Optimization
proceed concurrently. In this parallel process, the camera pose only receives gradients originating from the background
Gaussians, and the human parameters exclusively receive gradients from the foreground Gaussians. Both the camera pose
1

<!-- page 13 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Triplane
Embedding
xyz
time
Triplane
Feature
Time
Embedding
a) Position/Time Encode
b) Appearance/Geometry Decoder
MLP
GELU
MLP
GELU
Stat. Head
Dyn. Head
96
32
128
128
MLP
GELU
128
Canonical Property
(𝜇𝜇, 𝑠𝑠, 𝑟𝑟, 𝑐𝑐, 𝛼𝛼)
Dynamic Property
(∆𝜇𝜇, ∆𝑟𝑟, ∆𝑐𝑐)
Figure S2. JOintGS Model Architecture. Our model architecture is composed of one Encoder module and two Decoder modules. In the
Encoder module, the position attributes and the global temporal attributes of the Gaussian functions are encoded into positional features
and temporal features, respectively. The Decoder module receives the positional features as input and utilizes a two-layer MLP with
GELU activation to output either appearance or geometry features. These features are then fed into corresponding prediction heads to
derive the specific attribute values. For certain dynamic attributes, we opt to inject the temporal features into the second layer of the MLP
and use the same prediction head to output the corresponding residual values.
and human parameters are optimized with a learning rate of 0.001 for 5,000 iterations. To verify the necessity of this
phased design, we provide an additional ablation study and visualize the results in Figure S1. The results demonstrate that
skipping the initial stages and proceeding directly to joint optimization leads to severe gradient interference; specifically, a
significant amount of foreground-related information is incorrectly captured by the background Gaussians. Finally, in the
Joint Optimization Stage, we remove all constraints on the gradient graph, allowing the camera pose and human parameters
to participate in synergistic refinement with both foreground and background Gaussians, thereby achieving the mutual
correction of each component. In this stage, learning rates follow a cosine annealing schedule, decaying to 0.1 times their
initial value after 10,000 iterations.
B.3. Loss Function Formulation
The overall loss is a weighted sum of three major components: rendering loss (Lrender), prior loss (Lprior), and regularization
loss (Lregular):
L = Lrender + Lprior + Lregular
(13)
B.3.1. RENDERING LOSS
The rendering loss, Lrender, represents the pixel-level fidelity between the rendered image and the ground truth. It is a
combination of three metrics: Lrgb (pixel-wise L1 loss), Lssim (Structural Similarity Index), and Llpips (Learned Perceptual
Image Patch Similarity):
Lrender = λrgbLrgb + λssimLssim + λlpipsLlpips
(14)
We set the corresponding weights as λrgb = 1, λssim = 0.4, and λlpips = 0.2.
B.3.2. PRIOR LOSS
The prior loss, Lprior, incorporates prior knowledge into the optimization process, specifically ensuring the rendered human
avatar adheres to a segmentation mask:
Lprior = λmaskLmask
(15)
Here, Lmask is the Mean Squared Error (MSE) loss between the rendered human body segmentation and the provided prior
mask. The weight for this term is λmask = 0.01.
2

<!-- page 14 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Figure S3. Qualitative comparison on NeuMan dataset. For each scene, we present the complete rendered image (first column) and a
zoomed-in view of a densely textured region (second column), along with the error map of different methods (last three columns).
B.3.3. REGULARIZATION LOSS
The regularization term, Lregular, is introduced to maintain model stability and geometric plausibility, encompassing Linear
Blend Skinning (LBS) constraints, geometric constraints with the SMPL model, and dynamic attribute regularization:
Lregular = λlbsLlbs + λsmplLsmpl + λdynLdyn
(16)
The term Lsmpl is the L1 loss measuring the deviation of foreground Gaussian points from the reconstructed SMPL surface.
Ldyn is the L2 loss applied to all dynamic attributes predicted by the deformation network. We use the following weights:
λlbs = 20, λsmpl = 0.005, and λdyn = 0.01.
Seattle
Citron
Parking
Bike
Jogging
Lab
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Vid2Avatar (Guo et al., 2023)
16.90
0.51
0.27
15.96
0.59
0.28
18.51
0.65
0.26
12.44
0.39
0.54
16.36
0.46
0.30
15.99
0.62
0.34
NeuMan (Jiang et al., 2022)
18.42
0.58
0.20
18.39
0.64
0.19
17.66
0.66
0.24
19.05
0.66
0.21
17.57
0.54
0.29
18.76
0.73
0.23
HUGS (Kocabas et al., 2024)
19.06
0.67
0.15
19.16
0.71
0.16
19.44
0.73
0.17
19.48
0.67
0.18
17.45
0.59
0.27
18.79
0.76
0.18
JOintGS (Ours)
25.13
0.88
0.08
25.39
0.87
0.10
24.70
0.86
0.17
25.21
0.85
0.16
21.77
0.77
0.16
25.30
0.87
0.14
Table S1. Human reconstruction quality on NeuMan dataset. Quantitative comparison on human-only regions cropped using tight
bounding boxes. Our method achieves state-of-the-art performance across all six sequences
Seattle
Citron
Parking
Bike
Jogging
Lab
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
HyperNeRF (Park et al., 2021)
16.43
0.43
0.40
16.81
0.41
0.56
16.04
0.38
0.62
17.64
0.42
0.43
18.52
0.39
0.52
16.75
0.51
0.23
Vid2Avatar (Guo et al., 2023)
17.41
0.56
0.60
14.32
0.62
0.65
21.56
0.69
0.50
14.86
0.51
0.69
15.04
0.41
0.70
13.96
0.60
0.68
NeuMan (Jiang et al., 2022)
23.99
0.78
0.26
24.63
0.81
0.26
25.43
0.80
0.31
25.55
0.83
0.23
22.70
0.68
0.32
24.96
0.86
0.21
HUGS (Kocabas et al., 2024)
25.94
0.85
0.13
25.54
0.86
0.15
26.86
0.85
0.22
25.46
0.84
0.13
23.75
0.78
0.22
26.00
0.92
0.09
JOintGS (Ours)
32.52
0.95
0.04
26.27
0.84
0.09
32.77
0.92
0.09
31.19
0.94
0.04
28.29
0.89
0.11
30.35
0.94
0.06
Table S2. Full-image reconstruction quality assessment on the NeuMan dataset. Evaluation is conducted on the entire frames without
any region-specific cropping or masking.
C. Supplementary Experiments
C.1. Per-Scene Experimental Results
We provide a comprehensive scene-by-scene performance comparison on the NeuMan dataset, as detailed in Table S1 and
Table S2. To offer a multi-faceted evaluation, we report metrics under two distinct settings. First, Table S1 presents the
results calculated exclusively within the human-only regions, which are cropped using tight bounding boxes for each
sequence to directly reflect the fidelity of our human reconstruction. Second, Table S2 provides the evaluation on the entire
images to assess the overall scene reconstruction quality, including the background.
3

<!-- page 15 -->
JOintGS: Joint Optimization for In-the-Wild Monocular Reconstruction
Figure S4. Qualitative comparison of SMPL pose refinement. We overlay the estimated 3D human meshes onto the ground truth
images to evaluate spatial alignment. From left to right: (Left) The reference image with ground truth pose; (Middle) Results after our
JOintGS optimization; (Right) Results without optimization (initial estimates).
Both quantitative evaluations employ standard photometric metrics (PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang
et al., 2018)) as defined in §4.1. As shown in these results, our approach consistently outperforms existing SOTA methods
across various complex scenes. Visual results in Figure S3 further confirm that our model achieves superior spatial alignment
and texture detail for both the foreground human and the static background.
C.2. Visualization of Human Pose Optimization
Beyond pixel-level photometric evaluations, we further assess the effectiveness of our joint optimization by visualizing the
refinement of human pose parameters. In Figure S4, we overlay the estimated 3D human meshes onto the ground truth
images to qualitatively evaluate the spatial alignment.
It is evident that our method successfully corrects significant misalignments present in the initial poses. Quantitatively, our
joint optimization reduces the Mean Per-Joint Position Error (MPJPE) by 4mm compared to the initial estimates. As shown
in the middle column of Figure S4, our optimized results demonstrate precise alignment with the image evidence, whereas
the unoptimized initial estimates (right column) exhibit noticeable deviations from the actual human contours. By iteratively
refining the body pose {θt, β} in tandem with camera trajectories, our model achieves superior geometric precision. This
accurately aligned pose provides a stable and consistent anchor for the Gaussian attributes, thereby ensuring the temporal
coherence of the reconstructed human avatars.
C.3. Demonstration Video
We provide a demonstration video on our GitHub project page to visually showcase the high fidelity of our reconstruction.
The video includes extensive results for both novel-view and novel-pose synthesis, confirming the effectiveness of our
approach in modeling dynamic human bodies within the scene.
4
