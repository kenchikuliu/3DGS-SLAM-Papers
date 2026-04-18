<!-- page 1 -->
CuriGS: Curriculum-Guided Gaussian Splatting for Sparse View Synthesis
Zijian Wu
Zhejiang Sci-Tech University
2024110602006@mails.zstu.edu.cn
Mingfeng Jiang
Zhejiang Sci-Tech University
m.jiang@zstu.edu.cn
Zidian Lin
Zhejiang Sci-Tech University
2025010602002@mails.zstu.edu.cn
Ying Song
Zhejiang Sci-Tech University
ysong@zstu.edu.cn
Hanjie Ma
Zhejiang Sci-Tech University
mahanjie@zstu.edu.cn
Qun Wu
Zhejiang Sci-Tech University
wuq@zstu.edu.cn
Dongping Zhang
China Jiliang University
06a0303103@cjlu.edu.cn
Guiyang Pu
Zhejiang University
puguiyanghz@cmhi.chinamobile.com
Abstract
3D Gaussian Splatting (3DGS) has recently emerged as an
efficient, high-fidelity representation for real-time scene re-
construction and rendering. However, extending 3DGS to
sparse-view settings remains challenging because of super-
vision scarcity and overfitting caused by limited viewpoint
coverage. In this paper, we present CuriGS, a curriculum-
guided framework for sparse-view 3D reconstruction using
3DGS. CuriGS addresses the core challenge of sparse-view
synthesis by introducing student views: pseudo-views sam-
pled around ground-truth poses (teacher). For each teacher,
we generate multiple groups of student views with different
perturbation levels. During training, we follow a curricu-
lum schedule that gradually unlocks higher perturbation
level, randomly sampling candidate students from the active
level to assist training. Each sampled student is regularized
via depth-correlation and co-regularization, and evaluated
using a multi-signal metric that combines SSIM, LPIPS, and
an image-quality measure. For every teacher and perturba-
tion level, we periodically retain the best-performing stu-
dents and promote those that satisfy a predefined quality
threshold to the training set, resulting in a stable augmen-
tation of sparse training views. Experimental results show
that CuriGS outperforms state-of-the-art baselines in both
rendering fidelity and geometric consistency across vari-
ous synthetic and real sparse-view scenes. Project page:
https://zijian1026.github.io/CuriGS/
1. Introduction
In recent years, 3D reconstruction has advanced rapidly,
becoming a key enabling technology in fields such as vir-
tual reality, digital twins, cultural heritage preservation, and
intelligent manufacturing [8, 29, 34, 45]. Among various
approaches [25, 28, 38], Neural Radiance Fields (NeRF)
[20] and its variants [3, 7, 24, 30] have achieved impres-
sive results in high-fidelity reconstruction and photoreal-
istic view synthesis using implicit volumetric representa-
tions, albeit with slow optimization and heavy computa-
tional costs. Recently, 3D Gaussian Splatting (3DGS) [13]
has emerged as a more efficient and stable explicit represen-
tation, enabling real-time rendering by modeling scenes as
point-based Gaussians with covariance, color, and opacity
attributes [16, 18, 26, 32, 37, 39, 40, 46]. Beyond repre-
sentation efficiency, sparse view synthesis [5, 6, 10, 17, 42]
has become a key research focus, aiming to reconstruct ac-
curate geometry and realistic appearance from only a few
input images. This setting is both practical and highly chal-
lenging, as limited viewpoints lead to supervision scarcity
and severe overfitting.
In response to these challenges, recent studies have ex-
plored various strategies for enhancing 3D reconstruction
under sparse-input conditions.
Within the NeRF family,
SparseNeRF [36] introduces weak supervision by distilling
depth-ordering priors, thereby constraining volumetric ge-
ometry and alleviating degradation under extremely limited
viewpoints. FreeNeRF [43] approaches the problem from a
frequency and occlusion regularization perspective, impos-
1
arXiv:2511.16030v2  [cs.CV]  24 Feb 2026

<!-- page 2 -->
ing simple yet effective constraints on NeRF’s frequency
spectrum and near-camera density distribution, which im-
proves stability and generalization for few-shot rendering
without external supervision.
In the context of 3DGS, FSGS [50] adapts 3DGS to
sparse view synthesis through an efficient Gaussian expan-
sion strategy, incorporating pretrained monocular depth es-
timation as a geometric prior to guide reconstruction. Sim-
ilarly, DNGaussian [14] exploits depth information by em-
ploying a hard and soft depth-regularization scheme along-
side a global-local depth-normalization strategy to reshape
Gaussian primitives for more accurate geometry. Moreover,
LoopSparseGS [2] proposes a loop-based framework that
progressively densifies Gaussian initialization, aligns both
scale-consistent and inconsistent depths for reliable geome-
try, and adopts sparse-aware sampling to mitigate oversized
ellipsoid artifacts.
These studies address the challenge of sparse view re-
construction from different perspectives, proposing a va-
riety of strategies to enhance reconstruction quality.
In
summary, their methods mainly rely on incorporating ad-
ditional cues such as depth priors or introducing regular-
ization mechanisms to alleviate overfitting. However, the
fundamental limitation of sparse view reconstruction lies in
the inherent lack of supervisory signals caused by extreme
data sparsity, which restricts cross-view generalization and
compromises geometric consistency. Consequently, while
these methods offer partial improvements in visual quality,
they do not fundamentally resolve the core issue. This per-
sistent challenge drives our curriculum-based regularization
framework, which aims to tackle the root cause of supervi-
sion deficiency rather than merely mitigating its symptoms.
Motivated by the above observations,
we propose
CuriGS, a curriculum-guided 3DGS framework designed
for sparse view 3D reconstruction. The key idea of CuriGS
is the introduction of student views—pseudo-views gener-
ated around real cameras (teacher) with controllable pertur-
bation magnitudes. During training, a curriculum sched-
ule progressively unlocks student groups with larger per-
turbations, allowing the model to gradually adapt from lo-
cally consistent to more diverse viewpoints. At each itera-
tion, a subset of student views is randomly sampled from
the currently active group and optimized through depth-
correlation and co-regularization constraints, which enforce
geometric consistency while mitigating reconstruction er-
rors.
Meanwhile, CuriGS maintains the best-performing
student for each teacher and perturbation group, evaluated
via a multi-signal metric combining structural similarity
(SSIM), perceptual similarity (LPIPS), and a no-reference
image-quality score [1]. Students that exceed a predefined
quality threshold are periodically promoted into the train-
ing set, effectively augmenting sparse supervision with re-
liable pseudo-views. As illustrated in Fig. 1, this design ef-
fectively enriches the supervision signal under sparse-view
conditions, improving both geometric fidelity and rendering
realism while enhancing the model’s generalization to un-
seen viewpoints. The main contributions of this work are as
follows:
1. Curriculum-guided sparse view expansion.
We intro-
duce the first curriculum-guided 3DGS framework that
dynamically generates and promotes student views, ex-
panding supervision directly from sparse inputs while
mitigating overfitting and geometric inconsistency.
2. Unified pseudo-view learning mechanism. CuriGS pro-
vides a principled framework for generating, evaluating,
and integrating pseudo-views into scene optimization,
opening a new direction for virtual-view learning in re-
construction and synthesis tasks.
3. Superior performance and generalization. Extensive ex-
periments across multiple benchmarks demonstrate that
CuriGS consistently surpasses state-of-the-art baselines
in rendering fidelity, perceptual quality, and geometric
consistency.
2. Related Work
2.1. NeRF-based Sparse View Synthesis
While NeRF achieves photorealistic results under dense
multi-view supervision, its optimization becomes severely
ill-posed in sparse-view settings, often leading to geome-
try collapse, texture ambiguity, and overfitting to the ob-
served views. To alleviate these issues, several methods
[9, 11, 21, 30, 31, 36, 43] have been proposed to adapt
NeRF-style representations for few-shot reconstruction.
SparseNeRF [36] introduces weak geometric supervi-
sion by distilling depth-ordering priors from pretrained
monocular depth estimators or coarse sensor depth. By in-
jecting such constraints into the volumetric optimization,
SparseNeRF reduces geometric ambiguity and mitigates
depth collapse when only a handful of views are available.
However, its performance heavily depends on the quality
and domain alignment of the external depth priors, which
may mislead optimization when the monocular estimator
is biased or out-of-domain.
FreeNeRF [43] approaches
the problem from a spectral and occlusion-regularization
perspective, imposing frequency and near-camera density
constraints on the learned radiance field to suppress un-
supported high-frequency components and spurious near-
field densities that commonly arise under sparse supervi-
sion. This family of regularizers is appealing for its sim-
plicity and independence from explicit external priors, but
it requires careful hyperparameter tuning and may atten-
uate genuine high-frequency surface details in richly tex-
tured scenes. PixelNeRF [44] extends NeRF into a con-
ditional formulation by leveraging learned multi-view fea-
ture fusion. Through large-scale multi-scene pretraining,
2

<!-- page 3 -->
Figure 1. CuriGS is a curriculum-guided Gaussian Splatting framework that enhances sparse-view 3D reconstruction by progressively
introducing pseudo-views with increasing perturbations, yielding stable geometry and photorealistic rendering from extremely limited
input views.
it improves few-shot generalization and cross-scene adapt-
ability, yet still struggles with limited fidelity and unstable
geometry when applied to novel domains that diverge from
the training distribution. Despite these advances, sparse-
view NeRFs are fundamentally constrained by their im-
plicit volumetric representation. Consequently, achieving
stable geometry and consistent appearance reconstruction
under extremely sparse inputs remains a significant chal-
lenge, motivating the development of alternative represen-
tations and training paradigms that can inherently cope with
data scarcity.
2.2. 3DGS-based Sparse View Synthesis
Compared with NeRF, 3DGS offers superior runtime effi-
ciency, interpretable geometric primitives, and direct ma-
nipulability, making it particularly appealing for interac-
tive visualization and efficient optimization.
Motivated
by these strengths, several studies [14, 15, 22, 23, 47–50]
have explored its potential for sparse view synthesis tasks
with most achieving notably better performance than NeRF-
based methods.
CoR-GS [47] introduces an ensemble-based strategy in
which multiple 3DGS instances are trained jointly, using
mutual inconsistencies in both point-level geometry and
rendered appearance to identify and suppress unreliable
splats through co-pruning.
This co-regularization effec-
tively reduces reconstruction artifacts without relying on
explicit ground-truth geometry. However, the method in-
creases training complexity and computational overhead,
and its effectiveness diminishes in highly symmetric or ex-
tremely sparse scenes where ensemble disagreement be-
comes ambiguous. NexusGS [49] introduces an epipolar-
depth–guided framework that integrates geometric priors
to guide Gaussian densification, pruning, and blending.
However, it relies on additional optical flow information
and performs poorly when training views have low over-
lap, which limits its applicability. DropGaussian [23] pro-
poses a lightweight structural regularization strategy by ran-
domly deactivating subsets of Gaussians during training.
This dropout-like mechanism redistributes gradients toward
under-optimized splats, mitigating overfitting and improv-
ing generalization to unseen views. Unlike methods that
rely on external priors or ensemble models, DropGaussian
is simple and computationally efficient, though its stochas-
tic nature may lead to unstable convergence or suboptimal
results in extremely sparse or unstructured scenes.
Beyond the representative works discussed above, other
3DGS-based sparse-view methods [2, 33, 35, 41] have
made progress by incorporating semantic guidance, Gaus-
sian super-resolution, or confidence-weighted sparse en-
hancements to stabilize optimization. Nevertheless, these
approaches fundamentally build upon stronger regulariza-
tion or auxiliary supervision and do not address the key
challenge of sparse view reconstruction, namely the intrin-
sic scarcity of training viewpoints, thus remaining method-
ologically constrained. In contrast, we directly tackle the
fundamental limitation of sparse training supervision by
expanding the available viewpoints through a curriculum-
guided pseudo-view learning strategy.
3. Method
In this section, we present CuriGS, a curriculum-guided,
data-centric sparse-view 3DGS framework that progres-
sively expands the effective training view distribution
through the generation, evaluation, and selection of pseudo-
views. The overall pipeline of the proposed method is illus-
trated in Fig. 2.
3.1. Preliminaries: Gaussian Splatting
3DGS represents a scene as a set of anisotropic Gaussian
primitives, each parameterized by position µ ∈R3, covari-
ance Σ ∈R3×3, opacity α ∈[0, 1], and spherical harmonic
(SH) coefficients for view-dependent color. Formally, each
Gaussian defines a density function in 3D space:
G(x) = exp
 −1
2(x −µ)⊤Σ−1(x −µ)

(1)
The rendering process follows a rasterization pipeline, in
which each Gaussian is splatted into screen space, compos-
ited according to depth order, and accumulated using alpha
3

<!-- page 4 -->
Figure 2. The overall architecture of the CuriGS framework is shown in (A). The pipeline consists of three key stages: (1) student view
generation, where pseudo-camera poses are sampled around teacher views with multiple perturbation magnitudes, as detailed in (B); (2)
curriculum scheduling, which gradually unlocks perturbation levels during training to progressively expand viewpoint diversity; and (3)
student view evaluation and promotion, where each candidate is scored using perceptual (LPIPS), structural (SSIM), and no-reference
quality metrics. Only the best student at each perturbation level that passes the evaluation criteria is promoted to the training set, as
illustrated in (C). This curriculum-guided process enhances geometric consistency and rendering fidelity under sparse supervision.
blending. Specifically, the pixel color C(p) at screen loca-
tion p is obtained by front-to-back alpha compositing:
C(p) =
N
X
i=1
Ti αi ci(p),
Ti =
i−1
Y
j=1
(1 −αj),
(2)
where αi denotes the opacity of the i-th Gaussian, ci(p) is
its color contribution, and Ti is the transmittance term ac-
counting for accumulated transparency from closer Gaus-
sians.
3.2. Curriculum:
Student View Generation and
Scheduling
Student view generation.
The key idea of our framework
is the introduction of student views with the goal of aug-
menting supervision by generating novel viewpoints that
remain geometrically consistent with the original camera
distribution, while progressively increasing diversity during
training. Specifically, we generate student views by perturb-
ing the extrinsics of each teacher camera within a controlled
range. Given a teacher camera with rotation R and trans-
lation T, we first calculate its optical center C = −R⊤T.
Random angular perturbations in yaw and pitch are then ap-
plied, sampled from a zero-mean Gaussian distribution with
a standard deviation σ (in degrees), together with a mild ra-
dial perturbation σr along the viewing direction, both of
which collectively simulate small camera displacements.:
R′ = R R∆(σ),
C′ = C(1 + ϵr),
ϵr ∼N(0, σ2
r),
T ′ = −R′C′.
(3)
The perturbed parameters (R′, T ′) define a new pseudo-
view centered near the teacher pose and preserving identical
intrinsic calibration.
For each teacher view Ct, we initialize multiple groups
of student views P(σi) = {Cσi,j
s
}Ni
j=1 with varying angular
4

<!-- page 5 -->
Figure 3. Visualization of student view generation. Examples of pseudo-camera poses with different perturbation magnitudes around
teacher view.
perturbation levels σi drawn from a predefined range de-
pending on the scene scale and sparsity. This hierarchical
sampling naturally forms the basis of our curriculum strat-
egy, where views with smaller perturbations are initially
emphasized for stability, and those with larger perturbations
are progressively unlocked as training proceeds. Fig. 3 il-
lustrates examples of student views generated at different
perturbation levels.
Curriculum scheduling.
Instead of introducing all stu-
dent views simultaneously, we adopt a staged curriculum
strategy. Training begins with students at small σ, ensuring
stability by augmenting the dataset with near-teacher views
that preserve local geometry. After a fixed number of iter-
ations, the curriculum unlocks the next perturbation level,
and this process continues iteratively until the largest σ is
reached, progressively exposing the model to increasingly
diverse viewpoints.
We formalize the active perturbation level at iteration t
as:
σactive(t) = min
 σmax, σmin + k · ⌊t/Ts⌋

,
(4)
where σmin and σmax denote the minimum and maximum
angular perturbations, respectively. Ts is the interval iter-
ations after which the next perturbation level is unlocked,
and k is the increment step between successive stages.
By
controlling
the
perturbation
schedule
through
σactive(t), the model first consolidates local consistency
around ground-truth views before gradually adapting to
larger viewpoint variations. This design alleviates overfit-
ting under sparse supervision and ensures a smooth training
trajectory from local refinement to global generalization.
3.3. Student View Evaluation and Promotion
After generating student views, we design an evaluation-
and-promotion mechanism that assesses their quality during
training and selectively integrates the most reliable candi-
dates into the training set. The purpose of this mechanism
is to fully exploit the potential of student views, ensuring
that only informative and geometrically consistent samples
contribute to model learning.
Evaluation during training.
At each training iteration,
for each teacher view Ct, one student view Cσactive,j
s
∈
P(σactive) is randomly sampled from the pool of student
views corresponding to the currently active perturbation
level σactive. The corresponding rendered image is then com-
pared against the teacher’s reference image through a com-
posite multi-signal metric. This metric is designed to pro-
vide an objective evaluation of student views in the absence
of ground-truth references, jointly accounting for structural
similarity (SSIM) to assess spatial fidelity, perceptual dis-
tance (LPIPS) to measure feature-level realism, and a no-
reference image-quality measure [1] to capture global per-
ceptual quality. For each (Ct, σi) pair, we maintain the best-
performing student, defined as the candidate with the lowest
evaluation loss up to the current iteration.
In addition to this scoring mechanism, student views are
optimized using auxiliary pseudo-losses based on depth-
correlation and co-regularization. Unlike the visual scoring,
which serves to select among candidate views, these reg-
ularizers operate directly on the selected student views to
enhance geometric alignment and multi-view consistency.
Further details of these regularization terms are provided in
5

<!-- page 6 -->
Sec. 3.4.
Promotion to training views.
As the curriculum ad-
vances to a new perturbation level σnext, the best-performing
student view retained from the previous level σprev for each
teacher is subsequently evaluated. If its visual quality score
exceeds a predefined threshold, the corresponding student
view is then promoted to the official training set as a valid
camera pose. In this manner, only high-quality, geomet-
rically consistent pseudo-views are incorporated, gradually
expanding the training coverage.
3.4. Optimization Objective
To ensure robust reconstruction that benefits from dataset
expansion while maintaining fidelity to the ground truth, we
formulate our objective function with three distinct compo-
nents: (1) a dynamic reconstruction loss computed over the
current training set (comprising both original and promoted
student views) to drive the primary optimization; (2) an an-
chor loss specifically enforced on the fixed original ground-
truth views to prevent semantic drift; and (3) student-
view geometric regularizers including depth-correlation and
dual-model consistency to constrain the structure. The total
loss Ltotal is defined as:
Ltotal = Ltrain + λdrift Lanchor + λreg Lreg
(5)
Dynamic Training Loss.
This term drives the primary
optimization of the 3D Gaussian field. It is computed over
the current training set Vtrain, which is dynamically updated
throughout the curriculum. Initially, Vtrain contains only the
sparse input views. As training progresses, high-confidence
student views (pseudo-views that pass our quality curricu-
lum) are added to Vtrain.
Ltrain =
1
|Btrain|
X
I∈Btrain
Lphoto(I, ˆI)
(6)
where Btrain is a batch sampled from Vtrain, and Lphoto com-
bines L1 and D-SSIM loss. This allows the model to con-
tinuously learn from the densified view coverage.
Anchor Loss.
As the number of synthetic views in Vtrain
grows, there is a risk that the optimization may drift or over-
fit to artifacts in the generated data. To prevent this, we
enforce a strict consistency constraint on the fixed set of
original input views Vt (the teacher views). This set remains
static throughout training and acts as a geometric anchor. To
maintain efficiency, we randomly sample one anchor view
Ianc ∼Vt at each iteration:
Lanchor = Lphoto(Ianc, ˆIanc)
(7)
By explicitly re-visiting the ground truth at every step, we
ensure that the geometric optimization remains anchored to
the reliable observations, preventing semantic drift without
incurring the computational cost of rendering all original
views.
Student Regularization Loss.
Since student views are
synthetic observations sampled from the curriculum, they
lack ground-truth pixel data for direct supervision.
Op-
timizing these views without constraints can lead to hal-
lucinated geometry or floaters.
Therefore, we introduce
two pseudo-supervision schemes to regularize the struc-
ture of student views. The first scheme is based on depth-
correlation. Specifically, we leverage a pretrained monoc-
ular depth estimation model [27] to extract a proxy depth
map Dproxy from the rendered student image. This proxy
depth is then compared with the metric depth Drender di-
rectly output by the 3DGS rasterization via the Pearson cor-
relation coefficient, ensuring that the 3DGS geometry struc-
turally aligns with the visual cues without enforcing an er-
roneous absolute scale:
Ldepth = 1 −Cov(Drender, Dproxy)
σrenderσproxy
(8)
where Cov denotes covariance and σ denotes standard de-
viation. The second scheme employs a dual-model con-
sistency constraint.
Inspired by recent co-regularization
strategies [47], we leverage the insight that valid geometry
tends to be consistent across independently trained models,
whereas artifacts are typically stochastic and variable. Con-
sequently, we maintain two independently initialized mod-
els, MA and MB, and enforce photometric consistency be-
tween their renderings of the same student view, encourag-
ing the models to reach a consensus on the underlying scene
structure:
Lco = ∥IA
render −IB
render∥2
2
(9)
The final regularization term is the weighted sum:
Lreg = λdLdepth + λcLco
(10)
.
4. Experiments
4.1. Datasets
To comprehensively evaluate the robustness and generaliza-
tion ability of the proposed CuriGS framework, we con-
duct extensive experiments on three diverse and widely
adopted benchmarks: LLFF [19], MipNeRF-360 [4], and
DTU [12]. These datasets encompass a variety of challeng-
ing scenarios, ranging from controlled object-centric cap-
tures to complex unbounded real-world environments. To
ensure a strictly fair comparison, all data splits, image res-
olutions, and preprocessing steps are kept consistent with
prior sparse-view studies [14, 23, 47, 49, 50].
6

<!-- page 7 -->
Table 1. Quantitative Comparison With State-of-the-Art Methods on the LLFF, MipNeRF-360, and DTU Datasets. CuriGS Achieves the
Best or Comparable Results Across PSNR, SSIM, and LPIPS Metrics Under Sparse-View Conditions.
Method
LLFF (3 Views)
MipNeRF-360 (24 Views)
DTU (3 Views)
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
RegNeRF[21]
19.08
0.587
0.336
22.19
0.643
0.335
18.89
0.745
0.190
SparseNeRF[36]
19.86
0.624
0.328
22.85
0.693
0.315
19.55
0.769
0.201
FreeNeRF[43]
19.63
0.612
0.308
22.78
0.689
0.323
19.92
0.787
0.182
3DGS[13]
18.54
0.588
0.272
21.71
0.672
0.248
17.65
0.816
0.146
DNGaussian[14]
19.12
0.591
0.294
18.06
0.423
0.584
18.91
0.790
0.176
FSGS[50]
20.43
0.682
0.248
23.40
0.733
0.238
17.14
0.818
0.162
CoR-GS[47]
20.45
0.712
0.196
23.55
0.727
0.226
19.21
0.853
0.119
DropGaussian[23]
20.76
0.713
0.200
23.66
0.747
0.233
19.31
0.857
0.142
NexusGS[49]
21.00
0.730
0.179
23.86
0.753
0.206
20.21
0.869
0.102
LoopSparseGS[2]
20.85
0.717
0.205
24.09
0.755
0.226
20.68
0.856
0.125
Ours
21.10
0.732
0.193
24.21
0.761
0.202
20.45
0.873
0.129
Figure 4. Qualitative comparison on the LLFF dataset. CuriGS achieves sharper details and reduced texture drift compared with other
baselines.
LLFF.
The LLFF dataset [19] consists of complex,
forward-facing real-world scenes captured by handheld
cameras. Following the standard evaluation protocol, we
designate every 8th image as the test set and utilize the re-
mainder as the training pool. To simulate extreme sparse-
view conditions, we uniformly subsample a highly re-
stricted number of views from the training pool to serve
as our input. All images are downsampled by a factor of
8 to maintain computational efficiency while aligning with
baseline configurations.
MipNeRF-360.
The MipNeRF-360 dataset [4] presents
highly challenging unbounded indoor and outdoor scenes
with intricate background details and 360-degree camera
trajectories. This dataset thoroughly tests the model’s ca-
pacity to handle scale variations and unbounded back-
grounds under sparse supervision. Similar to the LLFF pro-
tocol, we reserve every 8th image for testing and uniformly
sample a sparse subset from the remaining images for train-
ing. The image resolution is also downsampled by a factor
of 8.
DTU.
The DTU dataset [12] features object-centric
scenes captured under strictly controlled laboratory condi-
tions with precise camera poses. For the DTU dataset, we
focus our evaluation on 15 representative scenes, specifi-
cally scans 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82,
103, 110, and 114. For the extreme sparse-view setting,
we specifically utilize views 25, 22, and 28 to construct a
3-view training set, reserving the remaining 25 views for
novel view evaluation. All images are downsampled by a
factor of 4. Furthermore, to strictly focus the optimization
and evaluation on the geometric and photometric fidelity of
the target objects, binary masks are applied to isolate and
remove the background regions.
4.2. Implementation Details
To handle the varying complexity and scene types across
different datasets, we adopt dataset-specific training config-
7

<!-- page 8 -->
Figure 5. Qualitative comparison on the MipNeRF-360 dataset. CuriGS demonstrates improved perceptual fidelity and geometric stability
on large-scale, unbounded scenes.
Figure 6. Qualitative comparison on the DTU dataset. CuriGS bet-
ter preserves fine geometric structures and thin details compared
to baseline methods under 3-views sparse supervision.
urations as detailed below.
LLFF.
For the LLFF scenes, the model is trained for
30,000 iterations. The curriculum-based student sampling
phase is active from iteration 3,000 to 24,000. In this phase,
we generate pseudo-student views using perturbation levels
σ ∈{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, with 5 students per level.
The quality threshold for selecting reliable student views is
set to 0.4.
MipNeRF-360.
In the MipNeRF-360 dataset, training
generally runs for 30,000 iterations, with the curriculum ac-
tivation scheduled between iterations 7,000 and 27,000. For
more complex scenes such as bicycle and stump, we extend
the training to 33,000 iterations and the curriculum phase
from 10,000 to 30,000. Student views are generated using
perturbation levels σ ∈{2, 4, 6, 8, 10}, with 3 students per
level and a quality threshold of 0.45.
DTU.
For the DTU dataset, the model is optimized for
13,000 iterations, with curriculum guidance applied be-
tween 2,000 and 12,000 iterations.
We utilize perturba-
tion levels σ ∈{1, 2, 3, 4, 5} and sample 10 students per
level. The quality threshold is set between 0.45 and 0.55.
To maintain object-centric consistency, we apply an adap-
tive masking strategy for the sampled student views. For a
specific student view Cs, its background mask Ms is deter-
mined by the color statistics of the background regions Bt
from its corresponding teacher view Ct. Let µs and σs be
the mean and standard deviation of pixel colors in Cs at co-
ordinates corresponding to Bt. A pixel p in the student view
is classified as background if:
Ms(p) =
(
1,
if ∥Cs(p) −µs∥< τ σs,
0,
otherwise,
(11)
where τ is a tolerance threshold. This mechanism ensures
that the optimization remains focused on the target object by
robustly suppressing background noise, even when geomet-
ric perturbations are introduced during the student camera
sampling process.
4.3. Baselines
We compared our approach with a representative set of
state-of-the-art methods from two categories, including
8

<!-- page 9 -->
NeRF-based approaches [21, 36, 43] and recent 3DGS-
based methods [14, 23, 47, 49, 50]. For quantitative evalu-
ation, we reported the standard image space metrics PSNR,
SSIM, and LPIPS computed on held-out test views.
4.4. Quantitative and Qualitative Results
LLFF.
On forward-facing real-world scenes from the
LLFF, CuriGS achieves notable gains in both pixel and
perceptual level metrics, as shown in Tab. 1. In particu-
lar, CuriGS attains the highest PSNR (21.10 dB) and SSIM
(0.732) among all compared 3DGS variants, while achiev-
ing LPIPS performance competitive with the top baselines.
These quantitative improvements correspond to a consistent
reduction in cross-view texture drift and localized blurring
in visual comparisons (Fig. 4). CuriGS recovers sharper
high-frequency appearance around edges and small struc-
tures where other methods typically smear or misalign tex-
ture. The improvement indicates that selective promotion of
high-quality pseudo-views effectively supplements super-
vision around original camera viewpoints, stabilizing both
photometric and geometric reconstruction on challenging
forward-facing scenes.
Mip-NeRF360.
For large-scale, unbounded scenes from
the MipNeRF-360, CuriGS demonstrates robust percep-
tual quality and strong generalization.
Quantitatively, it
achieves the best results among the tested methods, with
a PSNR of 24.21 dB, SSIM of 0.761, and LPIPS of 0.202
(Tab. 1). The corresponding visual comparisons are shown
in Fig. 5. Compared to other baselines, CuriGS exhibits
reduced unnatural color shifts and more accurately recon-
structs fine details, such as grass and ground textures.
These improvements stem from the curriculum strategy,
which progressively exposes the model to larger perturba-
tions, while the promotion mechanism filters out inconsis-
tent pseudo-views, ultimately yielding more coherent long-
range geometry and perceptually plausible renderings in
large-scale scenes.
DTU.
On the object-centric DTU benchmark under a 3-
views sparse setup, CuriGS shows clear advantages in
geometry-sensitive metrics, achieving the highest SSIM,
along with competitive PSNR and LPIPS, as shown in
Tab. 1. Qualitative comparisons in Fig. 6 further highlight
that CuriGS better preserves the overall structural integrity
of objects while more effectively maintaining thin structures
and fine geometric details, such as small protrusions and
edges, which are often lost or distorted by other methods.
4.5. Ablation Study
To better quantify and validate the effectiveness of the pro-
posed CuriGS framework in sparse-view reconstruction, we
conduct a comprehensive ablation study.
Table 2. Ablation study on the effect of curriculum guidance.
Dataset
#Views / Setting
PSNR
SSIM
LPIPS
LLFF
2 / Full
20.94
0.768
0.194
2 / w/o Cur.
18.30
0.601
0.231
3 / Full
22.51
0.834
0.150
3 / w/o Cur.
21.03
0.772
0.202
5 / Full
23.95
0.861
0.140
5 / w/o Cur.
22.97
0.841
0.143
MipNeRF-360
16 / Full
21.23
0.781
0.189
16 / w/o Cur.
20.32
0.754
0.202
24 / Full
23.43
0.851
0.131
24 / w/o Cur.
22.91
0.830
0.142
DTU
2 / Full
18.66
0.922
0.077
2 / w/o Cur.
15.04
0.896
0.092
3 / Full
22.65
0.947
0.050
3 / w/o Cur.
18.46
0.922
0.065
6 / Full
24.45
0.947
0.043
6 / w/o Cur.
20.51
0.946
0.048
Effectiveness of Curriculum Guidance.
The overall im-
pact of curriculum guidance across varying datasets and
view numbers is demonstrated through consistent and sig-
nificant improvements in PSNR, SSIM, and LPIPS under
diverse sparse-view configurations, as reported in Tab. 2.
On the LLFF dataset, curriculum guidance consistently
improves performance under all sparse-view conditions.
When trained with only two input views, the full strategy
achieved a PSNR of 20.94 dB compared to 18.30 dB with-
out guidance, alongside notable increases in SSIM (0.768
vs. 0.601) and decreases in LPIPS (0.194 vs. 0.231). As the
number of views increased, overall performance improved,
but the model without curriculum guidance still lagged be-
hind.
A similar pattern was observed on the MipNeRF-360
dataset. With 16 views, the inclusion of curriculum guid-
ance improved PSNR from 20.32 dB to 21.23 dB and SSIM
from 0.754 to 0.781, while LPIPS decreased from 0.202
to 0.189. At 24 views, the improvements remained con-
sistent, indicating that the proposed mechanism maintains
strong generalization capability even as the complexity of
the scene increases.
The improvements were even more pronounced on the
DTU dataset, which contains structured indoor objects with
rich geometric details. The full strategy yielded 3.5–4 dB
higher PSNR compared with the baseline lacking curricu-
lum guidance. For example, at three training views, PSNR
increased from 18.46 to 22.65, SSIM improved from 0.922
to 0.947, and LPIPS dropped from 0.065 to 0.050.
Effectiveness of Loss Components.
To specifically ad-
dress the structural integrity of our optimization process, we
ablate the core loss components formulated in our method:
9

<!-- page 10 -->
Table 3. Ablation study on the contribution of each loss compo-
nent across the LLFF, MipNeRF-360, and DTU datasets.
Dataset
Setting (N)
PSNR↑
SSIM↑
LPIPS↓
LLFF
3 / Full
22.51
0.834
0.150
3 / w/o Anchor Loss
21.84
0.795
0.178
3 / w/o Student Reg.
21.48
0.782
0.190
M-360
16 / Full
21.23
0.781
0.189
16 / w/o Anchor Loss
20.88
0.768
0.194
16 / w/o Student Reg.
20.65
0.760
0.198
DTU
3 / Full
22.65
0.947
0.050
3 / w/o Anchor Loss
21.97
0.935
0.058
3 / w/o Student Reg.
20.82
0.929
0.061
the dynamic reconstruction loss, the anchor loss, and the
student-view regularization. First, removing the anchor loss
on original views leads to noticeable geometric drift and a
drop in overall PSNR, demonstrating its critical role in an-
choring the foundational 3D structure against perturbations.
Second, excluding the student-view regularization results
in the model severely overfitting to the sparse inputs, as
it fails to penalize inconsistent geometric priors introduced
by unconstrained pseudo-views. Finally, the full triple-loss
structure ensures that the dynamic reconstruction leverages
the progressive perturbations effectively while maintain-
ing strict fidelity to the ground-truth observations, yielding
the most photorealistic and geometrically stable novel-view
synthesis.
4.6. Analysis of Pseudo-View Selection Mechanism
In extreme sparse-view scenarios, the fundamental bot-
tleneck lies in the severe lack of multi-view supervision.
While expanding the training set via curriculum-guided
pseudo-views can mitigate this issue, the unchecked inclu-
sion of poorly rendered views inevitably degrades the un-
derlying geometric fidelity. Therefore, a critical challenge
is establishing a robust and dynamic criterion to efficiently
evaluate and select high-quality student candidates.
To bridge the supervision gap caused by the absence
of explicit ground truth for pseudo-views, we formulate a
composite evaluation metric during the continuous tracking
phase. This metric elegantly integrates full-reference met-
rics (LPIPS and SSIM) with a no-reference image-quality
score. Specifically, LPIPS and SSIM measure the structural
and perceptual deviations from the corresponding teacher
view, ensuring that the student maintains strict semantic
consistency with the known observations. Simultaneously,
the no-reference score penalizes synthetic rendering arti-
facts, guaranteeing overall image plausibility. Empirically,
this tripartite criterion serves as a stable proxy for supervi-
sion, effectively screening out degraded viewpoints during
the early optimization stages.
Crucially, our strategy adaptively shifts when the train-
ing reaches a perturbation-switching stage. During the final
evaluation to promote a student into the permanent training
pool, we deliberately decouple the criteria by relying solely
on the no-reference quality score. The rationale behind this
design is that heavily weighting LPIPS and SSIM at the
promotion stage inherently biases the selection toward stu-
dents spatially closest to the teacher, thereby stifling view-
point diversity. By tolerating moderate geometric or appear-
ance deviations—provided the standalone rendering quality
remains high—the model is compelled to explore broader
scene variations. As visualized in Fig. 7, this dynamic se-
lection mechanism not only progressively refines the stu-
dent renders but also successfully identifies the optimal can-
didate that balances structural coherence with informative
viewpoint expansion, ultimately enhancing the generaliza-
tion and robustness of the 3D representation.
4.7. Analysis of Training Dynamics
To
further
investigate
the
dynamic
impact
of
our
curriculum-guided expansion, we monitor the evolution of
PSNR, SSIM, and LPIPS metrics throughout the training
process on a representative DTU scene.
As depicted in
Fig. 8, the baseline without curriculum guidance exhibits
a distinctive performance decay in the later stages of opti-
mization—a hallmark of overfitting where the model mem-
orizes sparse training viewpoints at the expense of general
scene structure. In contrast, CuriGS maintains a steady and
upward trajectory across all evaluation metrics. This sus-
tained improvement demonstrates that the progressive in-
tegration of validated student views provides a continuous
and reliable supervisory signal, effectively anchoring the
optimization process and ensuring consistent convergence
toward a high-fidelity representation.
5. Conclusion
In this work, we present CuriGS, a curriculum-guided
framework for sparse view 3D reconstruction using 3DGS.
Our method tackles the core challenge of sparse supervision
by introducing a hierarchical student-view learning mech-
anism, in which pseudo-views are progressively sampled
and regularized according to a curriculum schedule.
By
leveraging multi-signal evaluation metrics that combine
perceptual, structural, and image-quality measures, CuriGS
selectively promotes high-quality student views, thereby
enhancing geometric consistency and photometric fidelity.
Extensive
experiments
on
the
LLFF,
MipNeRF-360,
and DTU datasets demonstrate that CuriGS consistently
outperforms existing NeRF- and 3DGS-based baselines
under various sparse-view settings. The curriculum-guided
strategy
substantially
mitigates
overfitting,
improves
generalization
to
unseen
viewpoints,
and
preserves
fine
geometric
details
even
under
extreme
sparsity.
10

<!-- page 11 -->
Figure 7. Visualization of a teacher view and its associated student-view group under a specific perturbation magnitude. From left to right,
the rendered results of the students are shown across different training stages, illustrating the progressive enhancement of visual fidelity
and structural coherence. The composite evaluation scores (where lower indicates better quality) are utilized to track performance, with
the optimal student candidate highlighted by a blue box.
Figure 8. Evolution of PSNR, SSIM, and LPIPS during training on a DTU scene with and without curriculum guidance.
References
[1] Lorenzo Agnolucci, Leonardo Galteri, and Marco Bertini.
Quality-aware image-text alignment for opinion-unaware
image quality assessment. arXiv preprint arXiv:2403.11176,
2024. 2, 5
[2] Zhenyu Bao, Guibiao Liao, Kaichen Zhou, Kanglin Liu,
Qing Li, and Guoping Qiu.
Loopsparsegs: Loop based
sparse-view friendly gaussian splatting. IEEE Transactions
on Image Processing, 2025. 2, 3, 7
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 5855–5864,
2021. 1
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
11

<!-- page 12 -->
recognition, pages 5470–5479, 2022. 6, 7
[5] Yuanhao Cai, Jiahao Wang, Alan Yuille, Zongwei Zhou, and
Angtian Wang.
Structure-aware sparse-view x-ray 3d re-
construction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 11174–
11183, 2024. 1
[6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19457–19467, 2024. 1
[7] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang,
Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast general-
izable radiance field reconstruction from multi-view stereo.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 14124–14133, 2021. 1
[8] Zhien Dai,
Zhaohui Tang,
Hu Zhang,
and Yongfang
Xie. Mgs-stereo: Multi-scale geometric-structure-enhanced
stereo matching for complex real-world scenes. IEEE Trans-
actions on Image Processing, 34:6246–6258, 2025. 1
[9] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ra-
manan. Depth-supervised nerf: Fewer views and faster train-
ing for free. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 12882–
12891, 2022. 2
[10] Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong
Han. Binocular-guided 3d gaussian splatting with view con-
sistency for sparse view synthesis. Advances in Neural In-
formation Processing Systems, 37:68595–68621, 2024. 1
[11] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf
on a diet: Semantically consistent few-shot view synthesis.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 5885–5894, 2021. 2
[12] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola,
and Henrik Aanæs. Large scale multi-view stereopsis eval-
uation. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 406–413, 2014. 6, 7
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 7
[14] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normaliza-
tion. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 20775–20785,
2024. 2, 3, 6, 7, 9
[15] Ming Liu, Yuxuan Liang, Siwei Chen, Junjie Wang, and
Yang Na. Sv-2dgs: Optimization of sparse view 3d recon-
struction based on 2dgs models. Expert Systems with Appli-
cations, 297:129334, 2026. 3
[16] Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Jun-
ran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time
high-quality large-scale scene rendering with gaussians. In
European Conference on Computer Vision, pages 265–282.
Springer, 2024. 1
[17] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and
Wenping Wang. Sparseneus: Fast generalizable neural sur-
face reconstruction from sparse views. In European Confer-
ence on Computer Vision, pages 210–227. Springer, 2022.
1
[18] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre.
Taming 3dgs: High-quality radiance
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers, pages 1–11, 2024. 1
[19] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Transac-
tions on Graphics (ToG), 38(4):1–14, 2019. 6, 7
[20] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
1
[21] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall,
Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Reg-
nerf: Regularizing neural radiance fields for view synthesis
from sparse inputs. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
5480–5490, 2022. 2, 7, 9
[22] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko,
Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalan-
tari. Coherentgs: Sparse novel view synthesis with coherent
3d gaussians. In European Conference on Computer Vision,
pages 19–37. Springer, 2024. 3
[23] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian:
Structural regularization for sparse-view gaussian splatting.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 21600–21609, 2025. 3, 6, 7, 9
[24] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
10318–10327, 2021. 1
[25] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.
Pointnet: Deep learning on point sets for 3d classification
and segmentation. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 652–660,
2017. 1
[26] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas
Geiger, and Siyu Tang.
3dgs-avatar: Animatable avatars
via deformable 3d gaussian splatting.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5020–5030, 2024. 1
[27] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 6
[28] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 1
12

<!-- page 13 -->
[29] Maria Skublewska-Paszkowska,
Marek Milosz,
Pawel
Powroznik, and Edyta Lukasik. 3d technologies for intan-
gible cultural heritage preservation—literature review for se-
lected databases. Heritage Science, 10(1):3, 2022. 1
[30] Nagabhushan Somraj and Rajiv Soundararajan.
Vip-nerf:
Visibility prior for sparse input neural radiance fields.
In
ACM SIGGRAPH 2023 conference proceedings, pages 1–11,
2023. 1, 2
[31] Nagabhushan Somraj, Sai Harsha Mupparaju, Adithyan
Karanayil, and Rajiv Soundararajan. Simple-rf: Regulariz-
ing sparse input radiance fields with simpler solutions. arXiv
preprint arXiv:2404.19015, 2024. 2
[32] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea
Vedaldi.
Splatter image: Ultra-fast single-view 3d recon-
struction.
In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 10208–
10217, 2024. 1
[33] Yutao Tang, Yuxiang Guo, Deming Li, and Cheng Peng.
Spars3r: Semantic prior alignment and regularization for
sparse 3d reconstruction. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pages 26810–
26821, 2025. 3
[34] Fei Tao, Bin Xiao, Qinglin Qi, Jiangfeng Cheng, and Ping Ji.
Digital twin modeling. Journal of Manufacturing Systems,
64:372–389, 2022. 1
[35] Yecong Wan, Mingwen Shao, Yuanshuo Cheng, and Wang-
meng Zuo.
S2gaussian: Sparse-view super-resolution 3d
gaussian splatting. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 711–721, 2025.
3
[36] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Zi-
wei Liu. Sparsenerf: Distilling depth ranking for few-shot
novel view synthesis. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 9065–9076,
2023. 1, 2, 7, 9
[37] Kangkan Wang, Chong Wang, Jian Yang, and Guofeng
Zhang. Clocap-gs: Clothed human performance capture with
3d gaussian splatting. IEEE Transactions on Image Process-
ing, 34:5200–5214, 2025. 1
[38] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei
Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh
models from single rgb images. In Proceedings of the Euro-
pean conference on computer vision (ECCV), pages 52–67,
2018. 1
[39] Yuxuan Wang, Xuanyu Yi, Zike Wu, Na Zhao, Long Chen,
and Hanwang Zhang. View-consistent 3d editing with gaus-
sian splatting. In European conference on computer vision,
pages 404–420. Springer, 2024. 1
[40] Yu Wang, Xiaobao Wei, Ming Lu, and Guoliang Kang. Plgs:
Robust panoptic lifting with 3d gaussian splatting.
IEEE
Transactions on Image Processing, 34:3377–3388, 2025. 1
[41] Haolin Xiong, Sairisheek Muttukuru, Hanyuan Xiao, Rishi
Upadhyay, Pradyumna Chari, Yajie Zhao, and Achuta
Kadambi. Sparsegs: Sparse view synthesis using 3d gaus-
sian splatting. In 2025 International Conference on 3D Vi-
sion (3DV), pages 1032–1041. IEEE, 2025. 3
[42] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen,
Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wet-
zstein. Grm: Large gaussian reconstruction model for ef-
ficient 3d reconstruction and generation. In European Con-
ference on Computer Vision, pages 1–20. Springer, 2024. 1
[43] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Im-
proving few-shot neural rendering with free frequency reg-
ularization. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 8254–8263,
2023. 1, 2, 7, 9
[44] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 4578–4587, 2021. 2
[45] Shuwan Yu, Xiaoang Liu, Qianqiu Tan, Zitong Wang, and
Baohua Zhang. Sensors, systems and algorithms of 3d re-
construction for smart agriculture and precision farming: A
review.
Computers and Electronics in Agriculture, 224:
109229, 2024. 1
[46] Daiwei Zhang, Joaquin Gajardo, Tomislav Medic, Isinsu
Katircioglu, Mike Boss, Norbert Kirchgessner, Achim Wal-
ter, and Lukas Roth.
Wheat3dgs: In-field 3d reconstruc-
tion, instance segmentation and phenotyping of wheat heads
with gaussian splatting. In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 5360–5370,
2025. 1
[47] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu,
Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian
splatting via co-regularization. In European Conference on
Computer Vision, pages 335–352. Springer, 2024. 3, 6, 7, 9
[48] Yao Zhang, Jiangshu Wei, Yuchao Wang, and Jiajun Liu.
Usgs: Enhancing sparse view synthesis with unseen view-
point regularization in 3d gaussian splatting. Pattern Recog-
nition, 170:112087, 2026.
[49] Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun,
Junyu Dong, Huaidong Zhang, and Yong Du.
Nexusgs:
Sparse view synthesis with epipolar depth priors in 3d gaus-
sian splatting. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 26800–26809, 2025.
3, 6, 7, 9
[50] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang.
Fsgs:
Real-time few-shot view synthesis using gaussian
splatting. In European conference on computer vision, pages
145–163. Springer, 2024. 2, 3, 6, 7, 9
13
