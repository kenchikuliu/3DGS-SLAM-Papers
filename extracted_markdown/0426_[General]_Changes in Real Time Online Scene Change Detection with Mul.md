<!-- page 1 -->
Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
Chamuditha Jayanga Galappaththige1,2
Jason Lai3
Lloyd Windrim2,4
Donald Dansereau2,3
Niko S¨underhauf1,2
Dimity Miller1,2
1QUT Centre for Robotics
2ARIAM
3ACFR, University of Sydney
4Abyss Solutions
{chamuditha.galappaththige, d24.miller}@.qut.edu.au
Abstract
Online Scene Change Detection (SCD) is an extremely chal-
lenging problem that requires an agent to detect relevant
changes on the fly while observing the scene from uncon-
strained viewpoints. Existing online SCD methods are sig-
nificantly less accurate than offline approaches. We present
the first online SCD approach that is pose-agnostic, label-
free, and ensures multi-view consistency, while operating
at over 10 FPS and achieving new state-of-the-art perfor-
mance, surpassing even the best offline approaches. Our
method introduces a new self-supervised fusion loss to infer
scene changes from multiple cues and observations, PnP-
based fast pose estimation against the reference scene, and
a fast change-guided update strategy for the 3D Gaussian
Splatting scene representation. Extensive experiments on
complex real-world datasets demonstrate that our approach
outperforms both online and offline baselines.
Code is
available at https://chumsy0725.github.io/O-SCD/.
1. Introduction
Detecting changes in a scene is an essential task in scene
understanding, with numerous applications in environmen-
tal monitoring [46], infrastructure inspection [14], and dam-
age assessment [40]. Scene change detection (SCD) is es-
pecially challenging in the context of robotics, where an
agent observes the scene from unconstrained and indepen-
dent viewpoints when re-visiting it after some time, while
having to discern relevant (e.g. object movement) and irrel-
evant changes (e.g. caused by shadows, or reflections).
To address these challenges, recent approaches [13, 17,
22, 30, 52] leverage photorealistic 3D scene representations
like Neural Radiance Fields [32] (NeRF) and 3D Gaussian
Splatting [19] (3DGS) to enable pose-agnostic SCD from
unconstrained viewpoints. Complementary efforts [8, 13,
20] explore label-free SCD to remove the reliance on costly
and labor-intensive human-labeled changes, improving ro-
bustness under domain and data distribution shifts.
Despite recent advances, state-of-the-art (SOTA) SCD
2
4
6
8
10
12
0
0.2
0.4
0.6
MV3DCD [13] (Offline)
GeSCD [20] (Offline)
ChangeSim (CS)[34]
CS+CYWS2D[38]
CS+GeSCD[20]
OmniPoseAD[52]
SplatPose[22]
SplatPose+[29]
Ours
Frames per Second (FPS)
Change Detection Performance
(F1 Score)
Figure 1. Our online scene change detection method establishes a
new state of the art, detecting changes more reliably than all prior
methods, including the strongest offline baselines. It operates at a
runtime comparable to the fastest online approaches while achiev-
ing substantially higher F1 scores.
methods [13, 20, 30] are confined to an offline setting, where
both pre- and post-change observations are available prior
to inference. In contrast, online change inference—referred
to as online SCD—detects changes on the fly as new images
are acquired during a scene revisit, without access to future
observations. This setting is critical for real-time decision-
making and intervention in embodied and robotic systems.
As in prior online approaches [22, 29, 34, 34], we assume
that the scene remains static within each revisit. As shown
in Fig. 1, existing online SCD methods exhibit a substantial
accuracy gap compared to SOTA offline approaches. More-
over, many fail to sustain real-time performance, limiting
their practical applicability.
We introduce a novel SCD approach that, for the first
time, unifies the strengths of online, pose-agnostic, and
label-free methods, while additionally enforcing multi-view
consistency [13] during inference. Our method surpasses
existing online and offline methods in detection perfor-
mance while operating at over 10 FPS. This leap in both
speed and accuracy is enabled by two key algorithmic and
system-level innovations: a novel self-supervised loss en-
forcing change information consistency across viewpoints,
arXiv:2511.12370v3  [cs.CV]  24 Feb 2026

<!-- page 2 -->
Figure 2. Qualitative comparison with MV3DCD [13]. MV3DCD’s hard thresholding and intersection heuristic lead to missed or spurious
detections, especially for subtle appearance changes in semantically similar objects (red-to-blue T-shaped object in Meeting Room, blue-to-
black bench in Porch). Hard thresholding risks discarding subtle but important changes, while the intersection fails to capture true changes
unless present in both masks. Our method jointly learns complementary change information in pixel- and feature-level cues via our novel
self-supervised loss, capturing fine-grained changes and achieving state-of-the-art performance in both online and offline settings.
addressing the limitations of hard-thresholded intersection
fusion [13] (see Fig. 2), and an ultra-light PnP-based pose
estimation module.
Our third innovation is a change-guided update strat-
egy for the 3DGS-based scene representation. Maintain-
ing an up-to-date representation of an evolving scene with-
out naively reconstructing it from scratch is challenging [1,
5, 7, 51], but essential for long-term monitoring.
Na¨ıve
reconstruction after each inspection round is computation-
ally expensive and discards well-reconstructed information
from unchanged regions. We address this by leveraging the
predicted change masks to guide selective updates: only
changed regions are newly reconstructed, fused with exist-
ing primitives, and refined through a lightweight global ad-
justment while preserving the geometry and the appearance
of unchanged areas. Our selective update approach enables
scene representation updates in seconds while reusing the
high-fidelity representations of unchanged regions.
In summary, we make three contributions validated by
our extensive experiments across real-world environments:
• We present an online approach for pose-agnostic SCD
from unposed monocular images, operating in real time.
Our approach is label-free and multi-view.
• We propose a novel self-supervised loss that jointly inte-
grates feature- and pixel-level cues without heuristic fu-
sion or hard-thresholding, achieving state-of-the-art per-
formance in both online and offline settings.
• We introduce a change-guided selective reconstruction
and fusion strategy that enables efficient, repeatable scene
representation updates within seconds.
2. Related Work
2.1. Scene Change Detection
SCD has traditionally been studied as a bitemporal pair-
wise problem, where a model detects changes between
two images captured at two time instances from identi-
cal [2, 6, 9, 25, 40, 47] or closely-aligned [27, 38, 39]
viewpoints. Most approaches formulate this as a segmen-
tation task [2, 6, 9, 27, 40], while few explore bounding
box prediction [38, 39], relying fully [2, 6, 9, 25, 27, 38–
40, 47] or partially [24, 41] on costly human annotations
to compensate for lighting variations, seasonal changes,
or viewpoint inconsistencies. However, this paradigm has
clear limitations: performance degrades under distribution
shifts, annotation is tedious, and the range of possible
changes in complex scenes is virtually unbounded.
Re-
cent work [3, 8, 13, 18, 20] has shifted toward label-free
or zero-shot approaches, driven by the emergence of pow-
erful visual foundation models [21, 33]. However, these
methods [3, 8, 18, 20] assume image pairs with identical
viewpoints are available—a condition rarely satisfied in au-
tonomous systems operating along independent trajectories,
and a condition we do not assume.
More recent methods [13, 17, 30] exploit high-fidelity
3D scene representations [19, 48] to model the pre-change
scene and render novel views from post-change viewpoints,
motivating pose-agnostic SCD. MV3DCD [13] showed that
learning change information across multiple viewpoints
with a scene representation significantly outperforms pair-
wise predictions.
However, these approaches require a
complete set of pre- and post-change captures, and use
Structure-from-Motion [45] (SfM) to register poses to a
common reference frame, confining them to an offline set-
ting. In contrast, we study SCD in an online and incremen-
tal regime, where changes are inferred on-the-fly.
Pose-agnostic anomaly detection [22, 29, 52] also builds
a 3D representation of a pre-change object.
To detect
anomalies, the object is rendered from the post-change
viewpoint, then scored using feature comparisons. How-
ever, these works focus on single objects rather than large-
scale scenes. Approaches such as [22, 52] optimize camera

<!-- page 3 -->
poses directly against the representation, leading to slower
pose estimation and, as shown by MV3DCD [13], frequent
convergence failures in large complex scenes with multiple
changes and view-dependent inconsistencies. Liu et al. [29]
improved efficiency by replacing this step with HLoc [42].
While also employing a self-supervised objective, the
approach of Furukawa et al. [12] differs fundamentally
from our formulation of the self-supervised loss. Their loss
is designed to exclude high-error regions to facilitate 2D
alignment, whereas ours instead integrates complementary
yet potentially noisy change cues across modalities into a
persistent 3D representation.
ChangeSim [34] also investigates online SCD. However,
ChangeSim depends on an off-the-shelf RGB-D SLAM sys-
tem [23] for pose estimation and assumes that pre- and post-
change trajectories are closely aligned. This reduces the
task to image retrieval, where the nearest pre-change view
(by L1 distance between camera poses) is selected. In con-
trast, we make no assumptions about incoming RGB-only
frames or the trajectories; instead, we estimate poses di-
rectly in the pre-change coordinate frame and infer change
masks by jointly leveraging all viewpoints observed so
far. Our approach operates fully label-free, pose-agnostic,
multi-view, online, and at real-time rates.
2.2. Efficient Representation Update
NeRFs [32] and 3DGS [19] are widely adopted photo-
realistic scene representations, capturing fine geometry
and appearance.
NeRFs regress a 5D plenoptic func-
tion [4] using an MLP network to parameterize density and
view-dependent radiance, while 3DGS employs anisotropic
Gaussian primitives for real-time novel-view-synthesis.
Recently, there has been growing interest in real-time
reconstruction [28, 31].
However, these methods gener-
ally underperform compared to offline counterparts and re-
quire substantial view overlap between frames.
In con-
trast, approaches that address the long-term evolution of
scenes remain less explored, focusing on updating repre-
sentations from sparse and intermittent captures [1, 49].
Closely related is continual learning for photorealistic scene
representations [1, 5, 49, 51]. NeRF-based continual learn-
ing approaches [5, 49] utilize distillation or generative re-
play but inherit slow inference and longer optimization
times. GaussianUpdate [51] proposes a three-stage opti-
mization pipeline requiring substantial training iterations.
CL-Splats [1] introduces a local optimization kernel to cal-
culate gradients only for changed primitives, yet cannot ro-
bustly handle global appearance variations, particularly the
illumination shifts often present between real-world inspec-
tion scenarios.
Long-term scene representations are also well-explored
in robotics for autonomous navigation. To maintain spatial
consistency in dynamic environments, these systems uti-
lize volumetric representations [11, 43], object-aware track-
ing [36, 53], unified metric-semantic frameworks [44], and
recently, 3DGS [50].
Continual learning approaches [1, 5, 49, 51] focus on up-
dating representations while facilitating history recovery. In
contrast, we focus on updating the representation with min-
imal training overhead, enabling frequent repeated inspec-
tions. To this end, we propose a simple selective modeling
strategy that only reconstructs changed regions, guided by
our change masks and the change representation, followed
by fusion with existing primitives. A lightweight global op-
timization step ensures consistency, enabling updates within
seconds while robustly handling both geometric and appear-
ance changes, including global illumination variations.
3. Methodology
Our approach is illustrated in Fig. 3. We begin by con-
structing a 3DGS [19] representation of the pre-change (ref-
erence) scene offline (Sec. 3.1). Incoming images of the
post-change (inference scene) are processed online. We es-
timate its pose relative to the reference scene (Sec. 3.2),
then render the corresponding viewpoint to extract change
cues (Sec. 3.3).
These cues are used to infer a change
mask (Sec. 3.4), leveraging current and previous observa-
tions. After processing all observations, the representation
is updated (Sec. 3.5) to the current state of the environment.
Problem Setup: The reference scene Rref is captured with
a set of nref images, Iref = {Ik
ref}nref
k=1. Over time, the scene
undergoes changes in structure (e.g., additions, removals, or
object movement), or object appearance (e.g., variations in
color or texture), forming the inference scene Rinf. In addi-
tion, there may be ‘distractors’ such as reflections, shadows
and global illumination changes.
Our objective is to generate a binary change segmen-
tation mask M k for each incoming inference frame Ik
inf
that localizes all relevant changes between Rref and Rinf
while suppressing distractors. After processing all infer-
ence frames Iinf = {Ik
inf}ninf
k=1, we obtain a set of refined
masks M = {M k
refined}ninf
k=1. Using the change masks M
together with Iinf, we selectively update the reference rep-
resentation Rref to reflect the scene’s current state Rinf.
3.1. Building Reference Scene Representation
Following the standard 3DGS pipeline [19], we first esti-
mate the camera poses Pref = {P k
ref}nref
k=1 for all images in
Iref using SfM [45]. Using Iref, Pref and a sparse point cloud
from SfM, we construct the Rref with Speedy-Splat [15].
We assume that the viewpoints, scene coverage, and image
quality in Iref are sufficient to produce a high-fidelity Rref.
3.2. Inferring Pose of an Incoming Frame
For each reference image Ik
ref ∈Iref, we extract keypoints
and descriptors using XFeat [35] as a fast, lightweight de-

<!-- page 4 -->
ref
ref
ren
inf
inf
feature
pixel
ref
change
inf
ref
SSF
inf
inf
change
inf
Figure 3. Proposed approach with this paper’s contributions highlighted. We register an incoming inference image Ik
inf to an existing
reference representation Rref with a lightweight PnP-based pose estimator. Using the estimated pose P k
inf and Rref to render an aligned
image Ik
ren, we extract change cues Ck as a combination of pixel- and feature-level cues. Our novel self-supervised fusion loss LSSF guides
the fusion of all observed change cues to build a change representation Rchange that collectively learns change information from multiple
viewpoints and infer change masks M k. Finally, we selectively reconstruct changed regions to update the representation to Rinf.
tector, followed by exhaustive matching across all reference
images. Using the known camera poses Pref, these corre-
spondences are triangulated to form a consistent 3D point
set associated with each Ik
ref.
This point set serves as a
geometric anchor for estimating the pose of incoming in-
ference frames by establishing 2D–3D correspondences be-
tween their detected keypoints and the reference points.
Given an incoming frame Ik
inf, we extract its descriptors
and select the top-n reference frames with the highest num-
ber of matches (n = 4 in our experiments). These ref-
erence frames provide candidate 2D–3D correspondences,
which are then used to estimate the pose P k
inf of Ik
inf via PnP
with RANSAC [10, 26]. Finally, we refine P k
inf with inliers
through a GPU-parallel miniBA [31].
Since pose estimation for Ik
inf relies exclusively on the
retrieved reference frames, the system operates without drift
accumulation. Moreover, by restricting inference to a fixed-
size set of reference frames, we achieve constant-time O(1)
pose estimation. We discuss some limitations of the XFeat
features in the supplementary material.
3.3. Extracting Change Cues
With P k
inf expressed in the coordinate frame of Pref, we
query Rref to render the corresponding pre-change view
Ik
ren, matching the viewpoint of the incoming frame Ik
inf. For
the image pair (Ik
inf, Ik
ren), we extract change cues by com-
puting differences at both the pixel and feature levels, cap-
turing both appearance and structural changes.
Pixel-level change cues: We quantify the differences be-
tween (Ik
inf, Ik
ren) at the pixel level using a combination of
L1 and D-SSIM terms, following the photometric error for-
mulation in 3DGS [19] (Eq. 1) with λ = 0.2 following
3DGS [19]. We normalize Ck
pixel to [0, 1].
Ck
pixel = (1 −λ)L1 + λLD-SSIM.
(1)
Feature-level change cues: To capture high-level seman-
tic differences, we leverage the visual foundation model
SAM2-Tiny [37] to extract dense feature maps (f k
inf, f k
ren)
for (Ik
inf, Ik
ren).
Each feature map is represented as f ∈
R
h
s × w
s ×d, where h and w denote the image height and
width, s is the patch size, and d is the feature dimension-
ality. The feature-level change cues Ck
feature are computed as
the absolute difference between the two feature maps:
Ck
feature =
d
X
i=1
f k,i
inf −f k,i
ren
 ∈R
h
s × w
s ,
(2)
followed by bilinear interpolation to the original image res-
olution (h, w) and normalization to the range [0, 1].
Combined change cues: The final change cue map Ck
for each Ik
inf combines pixel- and feature-level cues through
simple addition (Ck = Ck
pixel + Ck
feature) balancing low-level
appearance differences with high-level semantic variations.
This formulation leverages the complementary strengths
of pixel- and feature-level cues while avoiding the loss of
change information.
Pixel-level cues effectively capture
fine-grained appearance differences, such as color varia-
tions between semantically similar objects, but tend to be
more sensitive to distractor changes caused by shadows,
reflections, or illumination shifts.
Feature-level cues are
more robust to these distractors yet may struggle to de-
tect subtle differences within semantically similar regions.
MV3DCD [13] relies on hard thresholding, which can dis-
regard subtle but relevant changes that fall below prede-
fined thresholds. Moreover, since MV3DCD fuses its struc-
ture and feature-aware masks through intersection, it may
further lose valid change information not simultaneously
captured by both masks. When combined with our novel
self-supervised fusion loss (Sec. 3.4), the proposed for-
mulation jointly integrates information across all observed

<!-- page 5 -->
viewpoints, effectively suppressing inconsistent distractors
while maintaining sensitivity to meaningful changes.
3.4. Inferring Change Masks
MV3DCD [13] first enforced multi-view consistency for
SCD using 3DGS [19]. We depart from its hard-thresholded
heuristic fusion and introduce a novel self-supervised loss
that jointly infers multi-view consistent change masks from
all observed cues at test time.
After the reference scene Rref is constructed (Sec. 3.1),
we initialize the change representation Rchange from Rref by
discarding all color parameters and introducing a learnable
change parameter c [13] for each primitive.
Rchange serves two purposes: (1) it enables fusing change
cues Ck from any viewpoint into a single, multi-view con-
sistent change representation, and (2) it acts as a persis-
tent memory that carries change information over observing
viewpoints. As a result, when a new frame arrives, the in-
coming change cues are fused with all previously observed
cues in Rchange. Rendering Rchange at the pose P k
inf yields
the predicted change mask M k for that viewpoint.
For an incoming Ik
inf, before inferring the change mask
M k, we update Rchange for n iterations (n = 16 in our ex-
periments) using our self-supervised fusion loss:
LSSF = Ci ⊙(1 −˜
M i) + log
 1 + mean( ˜
M i)2
,
(3)
where ⊙denotes Hadamard (i.e. element-wise) multiplica-
tion and ˜
M i is the sigmoid-activated rendered change mask
σ(M i
ren) from the viewpoint of the i-th frame. At each iter-
ation, we randomly sample i from all past inference frame
IDs i ∈[0, k], but biased towards the most recent frame
k with 1/3 probability. Ci contains the combined change
cues of the i-th frame (Sec. 3.3). We infer the change mask
M k for the kth frame after this optimization.
Intuitively, minimizing LSSF encourages the change pa-
rameters ˜c in Rchange to change so that the rendered ˜
M i
has values close to 1 in regions where change cues are
strong via the term Ci ⊙(1 −˜
M i). To prevent the triv-
ial solution of ˜
M i = 1 everywhere, the regularization term
log
 1 + mean( ˜
M i)2
is included.
This formulation allows us to infer M k for Ik
inf jointly
from all past and current change cues. By accumulating
change information from all observed frames in Rchange,
we enforce multi-view consistency and mitigate view-
dependent distractors from irrelevant changes.
3.5. Scene Representation Update
After completing online change detection for all observa-
tions, we perform a post-refinement of Rchange using all Ck,
and render refined change masks M k
refined for k ∈[0, ninf].
We then discard c from each primitive in Rchange and in-
troduce view-dependent appearance modeled via spherical
harmonics [19]. To only reconstruct changed regions, we
mask the inference images using the refined change masks
as ˆIk
inf = Ik
inf ⊙M k
refined. ˆIk
inf guides the reconstruction of
changed regions R∗
change following the standard 3DGS [19]
optimization pipeline. This disentangled reconstruction is
highly efficient and requires a fraction of the primitives
compared to modeling the entire scene, thereby acceler-
ating rendering and avoiding redundant computations in
unchanged regions. Notably, this selective reconstruction
achieves rendering speeds exceeding 400 FPS, substantially
reducing overall optimization time.
Next, we fuse [R∗
ref, R∗
change] to form the inference scene
Rinf, where R∗
ref denotes Rref excluding primitives that con-
tribute to changed pixels. A fast global optimization is then
performed, guided by Iinf. We restrict the adaptive density
control [19] only to the primitives contributing to changed
pixels in at least one view to avoid unnecessary densifica-
tion in unchanged regions.
This restricted global refinement serves multiple pur-
poses: (1) it accounts for global illumination differences,
(2) it mitigates boundary artifacts that may arise around the
changed regions after fusion, and (3) it corrects residual
errors due to imperfect change masks. Our design reuses
primitives from Rref wherever possible, while R∗
change ef-
ficiently models new structures.
Together, these design
choices significantly speed up optimization, enabling com-
plete scene updates within seconds.
4. Experiments
Datasets: We evaluate our method on PASLCD [13] for
SCD. PASLCD comprises 10 room-scale (i.e., a cantina)
indoor and outdoor scenes captured under similar and vary-
ing lighting conditions. It features both surface-level ap-
pearance and object-level geometric changes, along with
numerous distractors such as shadows, reflections, and illu-
mination shifts, making it a highly challenging multi-view
dataset for SCD. Importantly, PASLCD captures scenes
from unconstrained, independently traversed camera tra-
jectories, closely reflecting real-world autonomous opera-
tion. For the scene representation update, we evaluate on
PASLCD [13] and CL-Splats [1]. CL-Splats consists of five
small-scale (i.e., tabletop) scenes, each featuring a single
object-level change.
Baselines and Metrics: We conduct a comprehensive eval-
uation using the best-performing baselines [8, 13, 20, 22,
27, 29, 30, 34, 38, 52]. For pairwise methods [8, 20, 27,
30, 38] evaluated in the offline setting, we render identi-
cal viewpoints using vanilla 3DGS [19] for a fair compar-
ison, although this substantially simplifies the task by re-
moving viewpoint inconsistencies. For the online setting,
we construct two additional baselines by integrating Chan-
geSim’s [34] frame matching with the best-performing pair-
wise methods [20, 38]. We provide ground-truth poses for
ChangeSim’s frame retrieval module to ensure a fair eval-

<!-- page 6 -->
Table 1. Quantitative results for SCD on PASLCD [13] averaged
over all 20 instances. LF: Label-Free, PA: Pose-Agnostic, MV:
Multi-View consistency for change detection, ON: Online. We
additionally report the total runtime, including pose estimation and
reference reconstruction, for offline methods, and the operating
frame rate (FPS) for online methods. Our method achieves the
best performance in both settings, even outperforming all existing
offline methods while operating online.
Method
LF PA MV ON mIoU
F1
Runtime
/ FPS
R-SCD [27]
–
–
–
–
0.118 0.199
194s
CYWS2D [38]
–
–
–
–
0.273 0.398
189s
GeSCD [20]
✓
–
–
–
0.477 0.611
298s
ZeroSCD [8]
✓
–
–
–
0.306 0.414
409s
3DGS-CD [30]
✓
✓
✓
–
0.209 0.339
824s
MV3DCD [13]
✓
✓
✓
–
0.478 0.628
479s
Ours (Offline)
✓
✓
✓
–
0.552 0.694
156s
ChangeSim (CS) [34] –
–
–
✓
0.018 0.034
11.5
CS+CYWS2D [38]
–
–
–
✓
0.243 0.360
8.2
CS+GeSCD [20]
✓
–
–
✓
0.181 0.270
<1
OmniposeAD [52]
✓
✓
–
✓
0.168 0.262
<1
SplatPose [22]
✓
✓
–
✓
0.173 0.281
<1
SplatPose+ [29]
✓
✓
–
✓
0.237 0.358
<1
Ours
✓
✓
✓
✓
0.486 0.638
11.2
uation, as PASLCD lacks depths for ChangeSim’s off-the-
shelf RGB-D SLAM system. We use model checkpoints
provided by the authors for the supervised methods [27, 38].
Following standard practice in SCD [2, 13, 27, 41], we
report mean intersection over union (mIoU) and F1 score
computed for change pixels in the ground-truth mask.
For efficient scene representation update, we adopt
3DGS and fast variants [15, 16, 19] as our baselines. We
also evaluate CLNeRF [5] among publicly available contin-
ual learning methods. Following standard evaluation proto-
cols [1, 5, 19, 49, 51], we report PSNR, SSIM, and LPIPS
for novel views after scene update, along with runtimes.
4.1. Experiments on Scene Change Detection
Offline SCD Results: Table 1 presents an extensive com-
parison against SOTA methods across all SCD settings on
PASLCD [13]. In the offline setting, we follow the proto-
col of MV3DCD [13] by optimizing Rchange using our LSSF
with access to all inference views jointly for 3k iterations.
Generally, label-free approaches yield better perfor-
mance. Among these, our method achieves the highest over-
all performance, improving mIoU by approximately 15%
over the strongest offline competitor, MV3DCD, while run-
ning nearly 3× faster.
The SCD performance gain pri-
marily stems from our proposed self-supervised fusion loss
LSSF, which eliminates the hard thresholding and intersec-
tion heuristics used in MV3DCD, enabling more robust and
Table 2. Runtime analysis of each module in our online SCD
pipeline, measured in milliseconds per frame on PASLCD [13].
Most of the computation time is spent on multi-view change infor-
mation fusion, while other modules are lightweight.
Module
ms/Frame
Percentage (%)
Extracting Descriptors
1.28
1.4
Reference Image Retrieval
11.50
12.8
Pose Estimation
16.47
18.4
Change Cue Generation
1.69
1.9
Multi-View Change Cue Fusion
58.17
64.9
Change Mask Inference
0.49
0.6
Total
89.60
100
fine-grained change localization.
Online SCD Results: We compare our approach with exist-
ing online SCD methods (Table 1). In the online setting, our
approach achieves 2× higher mIoU than the strongest com-
petitor, CS+CYWS2D [38], while maintaining real-time
performance at 11 FPS. Notably, SOTA offline methods
such as GeSCD [20] experience severe degradation when
exposed to viewpoint discrepancies. CYWS2D exhibits a
smaller performance drop under such conditions, likely due
to its pretraining on COCO-Inpainted [38], which includes
image pairs with viewpoint variations. The upper bounds
of these two methods under identical viewpoints are shown
in the offline setting.
Remarkably, our online approach
not only establishes SOTA results among online methods
but also surpasses the best offline models, demonstrating
strong robustness and efficiency under real-world condi-
tions. We conduct all experiments at 1008 × 560 resolution
on a GeForce RTX4090. FPS was measured as empirical
wall-clock time over the number of frames processed.
Runtime Analysis: Table 2 summarizes the runtime break-
down (asynchronous overhead) of our online SCD pipeline
for a single inference frame Ik
inf. Approximately 33% of the
total time is spent on pose estimation, aligning Ik
inf to the
2 4
8
16
32
4
8
12
16
20
Number of Optimization Iterations
FPS
FPS
F1 Score
0.61
0.62
0.63
0.64
F1 Score
Figure 4. Speed–accuracy trade-off of our online method. Our
method can operate between 11–20 FPS with a relative perfor-
mance drop of 3.6% in F1 Score.

<!-- page 7 -->
Figure 5. Additional qualitative comparison with MV3DCD [13]. Our rendered change masks align more closely with the ground truth,
capturing subtle structural and appearance changes that MV3DCD often misses. In contrast to MV3DCD, our method produces fewer
spurious detections and demonstrates strong robustness to distractor variations across both indoor and outdoor environments.
coordinate system of Rref. SCD accounts for 67% of the
overall runtime, of which the majority (94%) is attributed to
multi-view change cue fusion spent on optimizing Rchange.
Speed Vs. Accuracy Trade-off:
We vary the number of
iterations used for multi-view change cue fusion—the dom-
inant contributor to runtime (Fig. 4). Our approach can op-
erate between 11–20 FPS, with only a modest 3.6% drop in
F1 score. This efficiency stems from Rchange, which serves
as a persistent memory of previously learned changes, min-
imizing the iterations needed for subsequent frames.
Qualitative Results: We present qualitative comparisons
against our closest competitor, MV3DCD [13], in Figs. 2
and 5. Our method demonstrates superior change localiza-
tion and robustness across diverse real-world scenes. Unlike
MV3DCD, our approach produces cleaner and more spa-
tially coherent change masks with significantly fewer false
negatives. It effectively captures subtle appearance and ge-
ometric changes that MV3DCD often overlooks, while sup-
pressing false positives caused by distractors.
These re-
sults highlight that our self-supervised fusion of pixel- and
feature-level cues combined with multi-view consistency
enables accurate, fine-grained change detection.
Ablation Analysis:
We conduct an ablation study on
PASLCD [13] (Table 3). Removing either the L1 or LD-SSIM
term in Cpixel noticeably degrades performance, with the
former providing a relatively stronger supervisory signal for
LSSF. When using either Cpixel or Cfeature, the model fails to
converge, indicating that neither modality alone sufficiently
guides LSSF. Removing the regularization term collapses
Table 3. Ablation study of our SCD approach on PASLCD [13].
Performance benefits from every component.
Variant
mIoU ↑
F1 ↑
Ours (Full)
0.486
0.638
– L1
0.320
0.464
– LD-SSIM
0.447
0.620
– Cpixel (using only Cfeature)
✗
✗
– Cfeature (using only Cpixel)
✗
✗
– Regularization term
✗
✗
Ours (With [13]’s Thresholding & Heuristic Fusion)
0.350
0.495
training into the trivial solution ˜
M k = 1 (disscussed in
sec. 3.4). Using MV3DCD’s [13] hard thresholding and in-
tersection heuristic instead of LSSF degrades performance.
4.2. Experiments on 3DGS Representation Update
Quantitative Results: We compare our change-guided rep-
resentation update strategy against reconstructing the scene
from scratch [15, 16, 19] and updating [5] in Table 4.
Our method achieves comparable or slightly superior per-
formance while substantially reducing the training over-
head. For example, total optimization time is 8× faster
than 3DGS-LM [16] and 13× faster than 3DGS [19] on
PASLCD [13].
The slight performance gain arises from
reusing the well-reconstructed, unchanged regions of the
reference scene, which may not be well captured by the lim-
ited set of inference views.

<!-- page 8 -->
Table 4. Quantitative comparison of scene representation update on PASLCD [13] and CL-Splats [1]. Our method achieves comparable or
higher reconstruction quality than approaches that fully re-optimize the evolved scene from scratch, while providing updated representa-
tions within seconds (< 60s), achieving up to 8–9× faster runtimes. Results are averaged over all instances and scenes.
Method
PASLCD
CL-Splats
PSNR (dB) ↑
SSIM ↑
LPIPS ↓
Runtime (s) ↓
PSNR (dB) ↑
SSIM ↑
LPIPS ↓
Runtime (s) ↓
3DGS [19]
22.21
0.7558
0.2426
550
30.31
0.9319
0.1178
364
3DGS-LM [16]
22.26
0.7562
0.2422
340
29.95
0.9322
0.1177
275
SpeedySplats [15]
22.25
0.7603
0.2618
399
29.89
0.9349
0.1290
312
CLNeRF [5]
22.27
0.6239
0.3907
451
26.29
0.7867
0.2235
301
Ours
23.70
0.7868
0.2491
42
30.54
0.9356
0.1256
39
Ours
CLNeRF
3DGS 𝓡ref
3DGS (from scratch)
Figure 6. Qualitative comparison of rendered views from the updated representation with CLNeRF [5] and 3DGS [19] (from scratch). Our
method more accurately reconstructs changed regions (red boxes) while reusing primitives from Rref to preserve high fidelity in unchanged
areas (yellow boxes), compared to na¨ıve reconstruction at each time.
Table 5. Analysis on scene update component on PASLCD [13].
Runtime for change detection and refinement is excluded. GO:
Global Optimization, SR: Selective Reconstruction.
Variant
PSNR (dB) ↑SSIM ↑LPIPS ↓Runtime (s) ↓
GO Only (3DGS)
22.64
0.7611
0.2550
145
GO Only (Ours)
23.01
0.7751
0.2553
79
SR Only
19.89
0.6814
0.3084
28
Ours (Full)
23.70
0.7868
0.2491
36
Qualitative Results: We present qualitative comparisons
with 3DGS [19] and CLNeRF [5] in Fig. 6. Our method
more accurately reconstructs the changed regions compared
to CLNeRF, while also achieving higher visual fidelity in
unchanged areas than 3DGS (built from scratch), owing to
the effective reuse of primitives from the reference scene.
Ablation Analysis: To evaluate runtime efficiency (Ta-
ble 5), all experiments are conducted for 10k iterations. We
begin with standard global optimization following 3DGS’s
adaptive density control [19]. Restricting this process to
primitives associated with changed pixels accelerates train-
ing by avoiding densification in unchanged regions. Se-
lective reconstruction alone runs approximately 5× faster
than standard global optimization but introduces local arti-
facts when used in isolation (discussed in Sec. 3.5). Com-
bining selective reconstruction with our global optimization
achieves the best of both approaches, where the former effi-
ciently models new geometry while the latter corrects resid-
ual artifacts locally and illumination differences globally.
5. Conclusion
We proposed a novel approach to pose-agnostic SCD that
detects change in an online manner with SOTA perfor-
mance, outperforming even the best offline methods. We
introduce two key algorithmic and system-level innovations
to achieve this: an ultra-light PnP-based pose estimator and
a self-supervised fusion loss for learning a multi-view con-
sistent change representation. Additionally, we introduced a
change-guided update strategy for 3DGS, reducing training
overhead to seconds while retaining reconstruction fidelity.
Future work may focus on developing richer complemen-
tary change cues, potentially improving both online and of-
fline SCD performance.

<!-- page 9 -->
6. Acknowledgment
This work was supported by the Australian Research Coun-
cil Research Hub in Intelligent Robotic Systems for Real-
Time Asset Management (ARIAM) (IH210100030) and
Abyss Solutions. C.J., N.S., and D.M. also acknowledge
ongoing support from the QUT Centre for Robotics.
References
[1] Jan Ackermann, Jonas Kulhanek, Shengqu Cai, Xu Haofei,
Marc Pollefeys, Gordon Wetzstein, Leonidas Guibas, and
Songyou Peng.
CL-Splats: Continual learning of Gaus-
sian splatting with local optimization.
In Proceedings of
the IEEE/CVF International Conference on Computer Vision
(ICCV), 2025. 2, 3, 5, 6, 8
[2] Pablo F Alcantarilla, Simon Stent, German Ros, Roberto Ar-
royo, and Riccardo Gherardi. Street-view change detection
with deconvolutional networks. Autonomous Robots, 42(7):
1301–1322, 2018. 2, 6
[3] Tim Alpherts, Sennay Ghebreab, and Nanne van Noord. EM-
PLACE: Self-supervised urban scene change detection. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence, pages 1737–1745, 2025. 2
[4] James R Bergen and Edward H Adelson. The plenoptic func-
tion and the elements of early vision. Computational models
of visual processing, 1(8):3, 1991. 3
[5] Zhipeng Cai and Matthias M¨uller. ClNeRF: Continual learn-
ing meets NeRF. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 23185–23194,
2023. 2, 3, 6, 7, 8
[6] Shuo Chen, Kailun Yang, and Rainer Stiefelhagen.
DR-
TAnet: dynamic receptive temporal attention network for
street scene change detection. In 2021 IEEE Intelligent Ve-
hicles Symposium (IV), pages 502–509. IEEE, 2021. 2
[7] Luqi Cheng, Zhangshuo Qi, Zijie Zhou, Chao Lu, and
Guangming Xiong. LT-Gaussian: Long-term map update us-
ing 3D Gaussian splatting for autonomous driving. In 2025
IEEE Intelligent Vehicles Symposium (IV), pages 1427–1433.
IEEE, 2025. 2
[8] Kyusik Cho, Dong Yeop Kim, and Euntai Kim. Zero-shot
scene change detection. In Proceedings of the AAAI Confer-
ence on Artificial Intelligence, pages 2509–2517, 2025. 1, 2,
5, 6
[9] Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch.
Fully convolutional siamese networks for change detection.
In 2018 25th IEEE international conference on image pro-
cessing (ICIP), pages 4063–4067. IEEE, 2018. 2
[10] Martin A Fischler and Robert C Bolles.
Random sample
consensus: A paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 4
[11] Jiahui Fu, Chengyuan Lin, Yuichi Taguchi, Andrea Co-
hen, Yifu Zhang, Stephen Mylabathula, and John J Leonard.
Planesdf-based change detection for long-term dense map-
ping.
IEEE Robotics and Automation Letters, 7(4):9667–
9674, 2022. 3
[12] Yukuko Furukawa, Kumiko Suzuki, Ryuhei Hamaguchi,
Masaki Onishi, and Ken Sakurada. Self-supervised simul-
taneous alignment and change detection.
2020 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS), pages 6025–6031, 2020. 3
[13] Chamuditha Jayanga Galappaththige, Jason Lai, Lloyd Win-
drim, Donald Dansereau, Niko Sunderhauf, and Dimity
Miller. Multi-view pose-agnostic change localization with
zero labels. In Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pages 11600–11610, 2025. 1,
2, 3, 4, 5, 6, 7, 8
[14] Dongyeob Han, Suk Bae Lee, Mihwa Song, and Jun Sang
Cho. Change detection in unmanned aerial vehicle images
for progress monitoring of road construction. Buildings, 11
(4):150, 2021. 1
[15] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias
Zwicker, and Tom Goldstein. Speedy-Splat: Fast 3D Gaus-
sian splatting with sparse pixels and sparse primitives. In
Proceedings of the Computer Vision and Pattern Recogni-
tion Conference (CVPR), pages 21537–21546, 2025. 3, 6, 7,
8
[16] Lukas H¨ollein, Aljaˇz Boˇziˇc, Michael Zollh¨ofer, and Matthias
Nießner. 3DGS-LM: Faster Gaussian-splatting optimization
with Levenberg-Marquardt. In Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), 2025.
6, 7, 8
[17] Binbin Jiang, Rui Huang, Qingyi Zhao, and Yuxiang Zhang.
Gaussian difference: Find any change instance in 3D scenes.
In ICASSP 2025-2025 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pages
1–5. IEEE, 2025. 1, 2
[18] Shyam Sundar Kannan and Byung-Cheol Min. ZeroSCD:
Zero-shot street scene change detection. In 2025 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 4665–4671, 2025. 2
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3D Gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 2, 3, 4, 5, 6, 7, 8
[20] Jae-Woo Kim and Ue-Hwan Kim.
Towards generalizable
scene change detection.
In Proceedings of the Computer
Vision and Pattern Recognition Conference, pages 24463–
24473, 2025. 1, 2, 5, 6
[21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 2
[22] Mathis Kruse, Marco Rudolph, Dominik Woiwode, and
Bodo Rosenhahn.
Splatpose & detect: Pose-agnostic 3D
anomaly detection. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
3950–3960, 2024. 1, 2, 5, 6
[23] Mathieu Labb´e and Franc¸ois Michaud.
RTAB-Map as an
open-source lidar and visual simultaneous localization and
mapping library for large-scale and long-term online opera-
tion. Journal of field robotics, 36(2):416–446, 2019. 3

<!-- page 10 -->
[24] Seonhoon Lee and Jong-Hwan Kim. Semi-supervised scene
change detection by distillation from feature-metric align-
ment. 2024 ieee. In CVF Winter Conference on Applications
of Computer Vision (WACV), pages 1215–1224, 2024. 2
[25] Yinjie Lei, Duo Peng, Pingping Zhang, Qiuhong Ke, and
Haifeng Li. Hierarchical paired channel fusion network for
street scene change detection. IEEE Transactions on Image
Processing, 30:55–67, 2020. 2
[26] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua.
EPnP: An accurate O(n) solution to the PnP problem. Inter-
national Journal of Computer Vision, 81(2):155–166, 2009.
4
[27] Chun-Jung Lin, Sourav Garg, Tat-Jun Chin, and Feras Day-
oub. Robust scene change detection using visual foundation
models and cross-attention mechanisms. In 2025 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 8337–8343. IEEE, 2025. 2, 5, 6
[28] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen,
Yen-Yu Lin, and Yu-Lun Liu. LongSplat: Robust unposed
3D Gaussian splatting for casual long videos. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion (ICCV), 2025. 3
[29] Yizhe Liu, Yan Song Hu, Yuhao Chen, and John Zelek.
SplatPose+:
Real-time image-based pose-agnostic 3D
anomaly detection. In European Conference on Computer
Vision Workshops, pages 378–391. Springer, 2024. 1, 2, 3, 5,
6
[30] Ziqi Lu, Jianbo Ye, and John Leonard. 3DGS-CD: 3D Gaus-
sian splatting-based change detection for physical object re-
arrangement. IEEE Robotics and Automation Letters, 2025.
1, 2, 5, 6
[31] Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bern-
hard Kerbl, and George Drettakis. On-the-fly reconstruction
for large-scale novel view synthesis from unposed images.
ACM Transactions on Graphics (TOG), 44(4):1–14, 2025.
3, 4
[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
3
[33] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
DINOv2: Learning robust visual features without supervi-
sion. arXiv preprint arXiv:2304.07193, 2023. 2
[34] Jin-Man Park, Jae-Hyuk Jang, Sahng-Min Yoo, Sun-Kyung
Lee, Ue-Hwan Kim, and Jong-Hwan Kim. ChangeSim: To-
wards end-to-end online scene change detection in indus-
trial indoor environments. In 2021 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS), pages
8578–8585. IEEE, 2021. 1, 3, 5, 6
[35] Guilherme Potje, Felipe Cadar, Andr´e Araujo, Renato Mar-
tins, and Erickson R Nascimento. Xfeat: Accelerated fea-
tures for lightweight image matching.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2682–2691, 2024. 3
[36] Jingxing Qian, Veronica Chatrath, James Servos, Aaron
Mavrinac, Wolfram Burgard, Steven L Waslander, and An-
gela P Schoellig. Pov-slam: probabilistic object-aware vari-
ational slam in semi-static environments.
arXiv preprint
arXiv:2307.00488, 2023. 3
[37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, et al.
SAM 2:
Segment anything in images and videos.
arXiv preprint
arXiv:2408.00714, 2024. 4
[38] Ragav Sachdeva and Andrew Zisserman. The change you
want to see. In Proceedings of the IEEE/CVF Winter Confer-
ence on Applications of Computer Vision, pages 3993–4002,
2023. 1, 2, 5, 6
[39] Ragav Sachdeva and Andrew Zisserman. The change you
want to see (now in 3D). In Proceedings of the IEEE/CVF
International Conference on Computer Vision Workshops,
pages 2060–2069, 2023. 2
[40] Ken Sakurada and Takayuki Okatani. Change detection from
a street image pair using CNN features and superpixel seg-
mentation. In Procedings of the British Machine Vision Con-
ference 2015, pages 61.1–61.12, Swansea, 2015. British Ma-
chine Vision Association. 1, 2
[41] Ken Sakurada, Mikiya Shibuya, and Weimin Wang. Weakly
supervised silhouette-based semantic scene change detec-
tion. In 2020 IEEE International conference on robotics and
automation (ICRA), pages 6861–6867. IEEE, 2020. 2, 6
[42] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and
Marcin Dymczyk. From coarse to fine: Robust hierarchical
localization at large scale. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 12716–12725, 2019. 3
[43] Lukas Schmid, Jeffrey Delmerico, Johannes L Sch¨onberger,
Juan Nieto, Marc Pollefeys, Roland Siegwart, and Cesar
Cadena. Panoptic multi-tsdfs: a flexible representation for
online multi-resolution volumetric mapping and long-term
dynamic scene consistency. In 2022 International Confer-
ence on Robotics and Automation (ICRA), pages 8018–8024.
IEEE, 2022. 3
[44] Lukas Schmid, Marcus Abate, Yun Chang, and Luca Car-
lone.
Khronos: A unified approach for spatio-temporal
metric-semantic slam in dynamic environments. In Proc. of
Robotics: Science and Systems, 2024. 3
[45] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 2, 3
[46] Aparna Taneja, Luca Ballan, and Marc Pollefeys.
Image
based detection of geometric changes in urban environments.
In 2011 international conference on computer vision, pages
2336–2343. IEEE, 2011. 1
[47] Ashley
Varghese,
Jayavardhana
Gubbi,
Akshaya
Ra-
maswamy, and P Balamuralidhar. ChangeNet: A deep learn-
ing architecture for visual change detection.
In Proceed-
ings of the European conference on computer vision (ECCV)
workshops, pages 0–0, 2018. 2
[48] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.

<!-- page 11 -->
4D Gaussian splatting for real-time dynamic scene render-
ing. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 20310–20320,
2024. 2
[49] Xiuzhe Wu, Peng Dai, Weipeng Deng, Handi Chen, Yang
Wu, Yan-Pei Cao, Ying Shan, and Xiaojuan Qi. CL-NeRF:
Continual learning of neural radiance fields for evolving
scene representation. Advances in Neural Information Pro-
cessing Systems, 36:34426–34438, 2023. 3, 6
[50] Vladimir Yugay, Thies Kersten, Luca Carlone, Theo Gevers,
Martin R Oswald, and Lukas Schmid. Gaussian mapping for
evolving scenes. arXiv preprint arXiv:2506.06909, 2025. 3
[51] Lin Zeng, Boming Zhao, Jiarui Hu, Xujie Shen, Ziqiang
Dang, Hujun Bao, and Zhaopeng Cui.
GaussianUpdate:
Continual 3D Gaussian splatting update for changing envi-
ronments.
In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), 2025. 2, 3, 6
[52] Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue
Zhou, Shanghang Zhang, and Hao Zhao. PAD: A dataset and
benchmark for pose-agnostic anomaly detection. Advances
in Neural Information Processing Systems, 36:44558–44571,
2023. 1, 2, 5, 6
[53] Liyuan Zhu, Shengyu Huang, Konrad Schindler, and Iro Ar-
meni. Living scenes: Multi-object relocalization and recon-
struction in changing 3d environments. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 28014–28024, 2024. 3
