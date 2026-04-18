<!-- page 1 -->
GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian
Splatting from Sparse Views
Tianyu Chen1
Wei Xiang1*
Kang Han1
Yu Lu1
Di Wu1
Gaowen Liu2
Ramana Rao Kompella2
1La Trobe University, Melbourne, VIC 3086, Australia
2Cisco Research, San Jose, CA, USA
{t.chen, w.xiang, k.han, y.lu, d.wu}@latrobe.edu.au, {gaoliu, rkompell}@cisco.com
Abstract
Feed-forward 3D reconstruction offers substantial runtime
advantages over per-scene optimization, which remains
slow at inference and often fragile under sparse views.
However, existing feed-forward methods still have poten-
tial for further performance gains, especially for out-of-
domain data, and struggle to retain second-level inference
time once a generative prior is introduced. These limita-
tions stem from the one-shot prediction paradigm in ex-
isting feed-forward pipeline: models are strictly bounded
by capacity, lack inference-time refinement, and are ill-
suited for continuously injecting generative priors. We in-
troduce GIFSplat, a purely feed-forward iterative refine-
ment framework for 3D Gaussian Splatting from sparse un-
posed views. A small number of forward-only residual up-
dates progressively refine current 3D scene using render-
ing evidence, achieve favorable balance between efficiency
and quality. Furthermore, we distill a frozen diffusion prior
into Gaussian-level cues from enhanced novel renderings
without gradient backpropagation or ever-increasing view-
set expansion, thereby enabling per-scene adaptation with
generative prior while preserving feed-forward efficiency.
Across DL3DV, RealEstate10K, and DTU, GIFSplat consis-
tently outperforms state-of-the-art feed-forward baselines,
improving PSNR by up to +2.1 dB, and it maintains second-
scale inference time without requiring camera poses or any
test-time gradient optimization.
1. Introduction
Reconstructing 3D scenes from multi-view images is a fun-
damental task in computer vision, perception, and computer
graphics, underpinning applications in robotic perception,
AR/VR, and high-fidelity 3D content creation. 3D recon-
struction has rapidly progressed along two paradigms: per-
scene optimization and feed-forward. Per-scene optimiza-
*Corresponding author.
Figure 1. Conceptual comparison of reconstruction paradigms.
Gradient optimization performs thousands updates, incurring
heavy test-time cost, often achieving high quality in dense-view
scenarios but struggling in sparse-view scenarios; One-shot feed-
forward [7] is efficient but leaves noticeable artifacts; Our iterative
residual feed-forward scheme keeps feed-forward efficiency and
achieves higher reconstruction quality without test-time gradient
backpropagation.
tion methods [9, 13, 30] can achieve high-fidelity recon-
structions by iteratively minimizing photometric objectives
on a single scene, but their inference requires long test-time
gradient optimization, which limits scalability in real ap-
plications. In addition, while such pipelines benefit from
iterative gradient feedback, they often struggle under sparse
view inputs and do not effectively incorporate external data
priors beyond inductive bias and observed samples, leaving
the rich knowledge contained in large-scale 3D data under-
utilized.
In contrast, recent feed-forward approaches
[10, 20,
21, 26] estimate 3D attributes from multi-view images
1
arXiv:2602.22571v1  [cs.CV]  26 Feb 2026

<!-- page 2 -->
in a single forward pass, incorporating multi-view in-
teractions through an attention mechanism within Vision
Transformers and employing dedicated attribute prediction
heads, achieving millisecond- to second-scale inference
time. However, such one-shot prediction introduces two key
limitations: (1) its performance is constrained by model ca-
pacity, leading to lower fidelity in complex scenes; and (2) it
lacks scene-specific refinement, leaving residual errors un-
corrected even in large-scale models [20]. Consequently,
the core challenge remains: how to achieve feed-forward
efficiency while enabling adaptation and continuous refine-
ment to each scene.
However, generating artifact-free novel views remains
challenging in complex scenes, particularly when view cov-
erage is sparse [9]. Recent studies [22, 23] tend to address
this limitation by integrating generative priors but require
long-time per-scene optimization. They alternatively opti-
mize 3D Gaussians and refine rendered views with diffu-
sion models, and these refined views are then fed back into
the training set for further optimization, forming a feed-
back loop that gradually improves reconstruction quality.
Yet this iterative loop is ill-suited for existing feed-forward
pipelines. As the continuously growing set of synthesized
views increases both time and memory complexity, which
are further exacerbated by the expensive self-attention op-
erations in ViT-based backbones [16] adopted by recent
feed-forward methods [1, 2, 7, 25, 27]. Moreover, these
methods reconstruct 3D scene directly from multi-view im-
ages rather than updating an existing scene representation,
making iterative enhancement feedback infeasible, let alone
step-wise fusion of generative priors. This raises an essen-
tial question: how to refine a 3D state in a purely feed-
forward manner, leveraging both observations and gener-
ative priors while keeping second-scale inference runtime?
To tackle this challenge, we propose GIFSplat, an it-
erative feed-forward 3D Gaussian Splatting framework
that leverages the strengths of both feed-forward and
optimization-based methods.
The first stage performs a
fast feed-forward pass to produce a reliable initialization,
and the second stage refines an initial prediction through
multi-step forward-only residual updates guided by scene
evidence and diffusion-enhanced cues. GIFSplat maintains
the speed of feed-forward pipeline while enabling scene-
adaptive refinement and reliable prior injection, bridg-
ing the gap between one-shot feed-forward methods and
optimization-based pipelines. At each step, GIFSplat uses
the current Gaussian state and discrepancy cues to pre-
dict residual updates. This allows continuous improvement
without test-time gradient backpropagation, with memory
and runtime scaling linearly in the number of refinement
steps. A conceptual comparison of paradigms is shown in
Fig. 1. Rather than continuously extending supervised im-
age sets with enhancing rendered images then re-optimizing
the scene, we apply a diffusion enhancer [22] to novel ren-
dered images, extract enhancement deltas, and convert them
into prior guidance cue. This approach brings reliable priors
without view explosion and followed heavy computation.
The primary contributions of this work are as follows:
• We present an iterative feed-forward update mechanism
for 3D Gaussian Splatting that refines a fixed set of Gaus-
sians via multi-step, forward-only residual updates, en-
abling scene-specific refinement without any test-time
gradient descent.
• We introduce a generative prior fusion mechanism that
distills a frozen diffusion prior into lightweight Gaussian-
level discrepancy cues from enhanced novel renderings,
injecting generative information into the iterative updates
without backpropagation or continuously expanding the
set of reference views.
• We demonstrate consistent performance gains across
DL3DV, RealEstate10K, and DTU datasets, under vary-
ing view-overlap settings, showing that our iterative
feed-forward model (IFSplat) and generative-prior vari-
ant (GIFSplat) both outperform recent feed-forward re-
construction methods in sparse-view and cross-domain
scenarios.
2. Related Work
2.1. 3D Reconstruction from Multi-view Images
Classical
per-scene
optimization
pipelines,
such
as
NeRF [13], 3D Gaussian Splatting (3DGS) [9] and its
variants [9, 14, 30], achieve high visual fidelity but require
thousands of test-time gradient steps and meet substantial
performance drop in sparse-view setting, limiting practical-
ity in real applications. To bypass slow SfM/MVS pipelines
and strong pose dependence while improving quality in
sparse view settings, recent feed-forward approaches infer
3D structure directly from images.
DUSt3R [21] predicts dense pointmaps from uncali-
brated pairs, removing explicit matching and bundle adjust-
ment, though it still requires pairwise fusion for larger im-
age sets [10]. VGGT [20] treats multi-view geometry as
a single large-scale model task, jointly predicting cameras,
depth, and pointmaps in one pass. For the 3DGS prediction,
NoPoSplat and FLARE [27, 32] explore pose-free sparse-
view lifting to produce 3D Gaussians, and AnySplat [7]
jointly estimates cameras and Gaussians with redundancy
consolidation for unconstrained inputs. Concurrently, con-
current work iLRM [8] explicitly unrolls the iterative re-
finement into multiple transformer layers with non-shared
parameters, where each layer performs cross-attention with
the reference-view images to inject features and apply one
update. As a result, increasing the number of refinement
steps makes the parameter count grow linearly with the
depth, and the number of refinement steps is fixed at in-
2

<!-- page 3 -->
Figure 2. Overview of GIFSplat. Our framework consists of a Gaussian initializer, an iterative Gaussian head, and a generative prior
fusion module. The initializer takes sparse input views and predicts camera parameters and initial 3DGS g0. The iterative head then refines
the Gaussians over several forward-only steps by updating per-Gaussian parameters gi using residual corrections ∆g predicted from the
concatenated state and cues {gi, oi, pi}. At each step, we render reference and novel views, compute observation evidence oi from feature
differences between input and rendered views, and derive generative prior cues pi by enhancing the renderings with a frozen diffusion
model DIFIX and taking feature-space residuals. These Gaussian-level signals are fed back into the iterative head to progressively improve
the 3DGS, particularly in under-constrained regions.
ference time for a given model. In contrast, we adopt a
single weight-shared refinement module that keeps the pa-
rameter count constant while allowing the number of itera-
tions to be flexibly adjusted during inference. While con-
current work ReSplat [24] performs iterative feed-forward
refinement based on observational errors, it still relies on
known camera poses. In contrast, our pose-free framework
uniquely addresses under-constrained sparse views by in-
jecting on-the-fly generative diffusion priors directly into
the update loop.
Overall, this line shifts reconstruction from slow opti-
mization to fast forward inference. However, most remain
one-shot predictors and lack reliable scene-adaptive refine-
ment mechanisms, precisely the gap addressed by our iter-
ative feed-forward refinement framework.
2.2. Generative Priors for 3D Reconstruction
The aforementioned methods perform well when novel
viewpoints are close to the input views, but their per-
formance drops markedly under sparse or low-coverage
inputs, often producing artifacts in under-constrained re-
gions. Generative models provide strong data-driven pri-
ors for under-constrained regions. Early approaches such
as DreamFusion [15] distill 2D diffusion into 3D but rely
on lengthy per-scene optimization. Later work seeks tighter
integration: multi-view diffusion model [18], and rectified-
flow formulations [19] enable efficient 3DGS generation
rather than 3D reconstruction, while limited to simple
scene.
Diffusion-enhanced 3D reconstruction pipelines [4, 12,
23, 29] render provisional views, enhance them via gen-
erative priors, and feed them back for improved coarse
3D model, both of them need multi-step generation. Di-
fix3D+ [22] compresses diffusion into a single-step en-
hancer for faster enhancement of rendered view.
While
above techniques significantly boost perceptual quality,
they still depend on optimization-based regimes rather than
updating in a feed-forward manner, meaning they require
time-consuming optimization runs.
These developments highlight the importance of gener-
ative priors for sparse-view and out-of-distribution scenes,
but also reveal the need for a framework that retains feed-
forward efficiency while enabling stable, scene-adaptive re-
finement with generative prior. Our work pursues this di-
rection through iterative forward-only residual refinement
with prior injection.
3

<!-- page 4 -->
3. Method
3.1. Problem Definition and Framework Overview
Problem definition
Given a set of uncalibrated multi-
view image observations V = {(Im)}M
m=1, we represent
a scene with 3D Gaussian Splatting (3DGS) G = {gi}N
i=1,
where each Gaussian gi = (xi, si, ri, ci, αi) contains po-
sition, scale, orientation, color, and opacity.
Our goal
is to recover G that is photometrically consistent with V,
while enabling fast inference and robustness under domain
shift. A differentiable rasterizer R produces a rendering
Rm = R(G; Πm) for camera Πm.
Framework overview
To combine feed-forward effi-
ciency with scene-specific refinement and efficient use of
generative prior, our framework consists of three compo-
nents (in Fig. 2): We remove voxelization module in [7]
and fine-tune it as initializer Fϕ of our framework, which
first predicts camera poses and an initial 3DGS G(0) from
V. On top of this, an iterative residual Gaussian head Uθ ap-
plies T forward-only updates to refine the 3D Gaussians (in
Sec. 3.2). Meanwhile, a generative prior fusion mechanism
converts diffusion-enhanced renderings into Gaussian-level
cues that guide these updates (in Sec. 3.3). This decomposi-
tion allows us to retain the speed of feed-forward inference
while still with scene-adaptive refinement.
3.2. Iterative Gaussian Head
Instead of one-shot prediction, we adopt an iterative feed-
forward refinement: starting from an initialization G(0), we
apply T forward update steps to obtain G(T ), without test-
time gradient backpropagation or long per-scene optimiza-
tion. In short, existing paradigms satisfy only partial prop-
erty at a time: feed-forward efficiency, refinement ability,
and prior utilization. We seek a formulation satisfying all
three. Fig. 3 illustrates the process of iterative refinement.
Given the formulation above, we now detail how the
iterative gaussian head enables feed-forward scene refine-
ment without test-time gradient backpropagation. Existing
optimization-based methods refine 3D Gaussians through
long-time gradient descent, whereas one-shot feed-forward
models lack the ability to absorb new evidence. Our goal
is to combine their strengths:
efficient inference while
still allowing scene-adaptive refinement.
Under no test-
time gradients, the iterative head transforms the initial
3DGS G(0) into G(T ) by predicting per-Gaussian resid-
ual {gi}(T ) across T forward passes, reducing render-
ing–observation discrepancies and optionally incorporating
prior cues. Equivalently, it forward-approximates minimiz-
ing ∥ψ(Im) −ψ(R(G; Πm))∥, where ψ is a frozen image
feature extractor and R is a differentiable renderer, more
details are stated in Appendix.
At step t, the rendered views are processed by ψ(·) to
Figure 3.
Visualizing iterative residual refinement.
Starting
from the initial 3DGS prediction (left column), our iterative Gaus-
sian head progressively refines geometry and appearance over
three forward-only steps (S1–S3).
The zoomed regions high-
lighted by red dashed boxes show reduced blur, sharper edges, and
fewer artifacts as the iteration proceeds, illustrating how the pro-
posed updates gradually improve the scene representation without
test-time gradient backpropagation.
compute feature differences O(t)
m = ψ(Im)−ψ(R(t)
m ). Pixel
cues are pooled to Gaussians via soft assignment weights
wi(u), producing o(t)
i .
o(t)
i
←
P
m∈S(t)
P
u wi(u) O(t)
m (u)
P
m∈S(t)
P
u wi(u) + ε
.
(1)
The iterative gaussian head predicts residuals ∆G(t) =
∆{g(t)
i }N
i=1 = (∆x(t)
i , ∆s(t)
i , ∆c(t)
i , ∆α(t)
i ) and update the
current 3DGS as
∆G(t+1) ←Uθ
 [G(t) ∥{oi}(t)]

,
(2)
G(t+1) ←G(t) + ∆G(t),
t = 0, . . . , T −1.
(3)
This design preserves feed-forward efficiency while pro-
viding a lightweight refinement signal on top of the one-shot
initialization.
Previous feed-forward approaches [1, 2, 27] adopt pixel-
aligned Gaussians, coupling splats to image grids. Such
alignments are computationally heavy, misaligned with true
3D neighborhoods, and prone to over-densifying smooth
4

<!-- page 5 -->
Figure 4. Generative prior fusion. Given sparse input views
(left), our feed-forward 3DGS first produces a rendered view (mid-
dle). A frozen diffusion-based enhancer then refines this render-
ing into an enhanced image (right) with sharper textures and richer
details, which is converted into Gaussian-level prior cues for sub-
sequent residual updates.
areas while under-representing detailed or occluded re-
gions. To ensure refinement respects 3D geometry, we con-
vert pixel-aligned Gaussians to point-based Gaussians and
project correspondent pretrained features by camera. In ad-
dition, we use window attention [33] to model local rela-
tionships of 3D gaussians efficiently.
For training, we adopt the architecture from [7] as ini-
tializer but remove voxelization. It is fine-tuned partially
with ground-truth poses, and we compute losses on novel
views. The iterative head is unrolled for T=3 steps with
|S(t)| matching inference. We supervise with MSE and per-
ceptual loss [31].
3.3. Generative Prior Fusion
When observation-only cues are weak (e.g., sparse views
or domain shift), generative prior can provide complemen-
tary high-frequency appearance. Beyond observation-only
refinement, we inject a frozen diffusion-based generative
prior.
To avoid test-time optimization cost, we adopt a
conservative design: the diffusion enhancer is frozen and
without using of gradient when inference. Rather than re-
optimizing the scene by extending the supervised image set
from enhanced views, we enhance intermediate renderings
and distill the enhancement into Gaussian-level cues for the
next residual update, improving detail fidelity while pre-
serving feed-forward efficiency, as illustrated in 4.
For each m ∈S(t), we enhance the current rendering
using a diffusion-based enhancer Eϕ and pretrained model
Difix [22]: ˜R(t)
m = Eϕ
 R(t)
m

. We compute a prior difference
in feature space: P (t)
m
= ψ
  ˜R(t)
m

−ψ
 R(t)
m

. Then pool-
ing to Gaussians mirrors the evidence pooling to get p(t)
i .
These cues regularize forward updates of Uθ under domain
shift and provide high-frequency hints in under-constrained
regions. The generative prior cues are similar to observed
cues.
p(t)
i
=
P
m∈S(t)
P
u wi(u) P (t)
m (u)
P
m∈S(t)
P
u wi(u) + ε
.
(4)
We fuse observation-driven evidence and prior cues by
concatenation at the gaussian level.
∆G(t)
i
= Uθ
 [g(t)
i
∥{oi}(t) ∥{pi}(t)]

,
(5)
G(t+1) = G(t) + ∆G(t).,
t = 0, . . . , T −1,
(6)
where g(t)
i
denotes the Gaussian-state including current per-
gaussian parameters and corresponding features, {oi}(t) the
observation cues are distilled from rendered–image fea-
ture differences, and {pi}(t) the prior cues distilled from
diffusion-enhanced renderings. We do not implement gra-
dient backpropagation through single-step diffusion model
Eϕ, the prior serves only as a forward cue.
Our framework enables feed-forward efficiency, scene-
adaptive refinement, and use of generative priors without
ever-increasing view synthesis or test-time gradient up-
dates, scales roughly linearly with T and the per-iteration
view budget, and improves robustness under domain shift
via the refinement by observed cue and generative prior.
3.4. Loss Functions
We train in two stages, applying geometric distillation Ldist
as [7] and photometric loss in novel view during Stage I to
bootstrap a stable initialization.
In stage 1 initialization (t=0), we supervise the first for-
ward prediction with a pixel reconstruction loss Lrec
Lrec = λrgb∥Im −Rm∥2.
(7)
Lstage1 =
X
m∈M
 Lrec + Ldist

.
(8)
In stage 2 refinement (t>0), we unroll T forward-only
refinement steps and supervise each rendering using the
multi-steps render loss Lstage2. More details are stated in
Appendix.
Lstage2 =
T
X
t=1
ωt
X
m∈M
∥Im −R(t)
m ∥2.
(9)
4. Experiments
4.1. Experimental Settings
Datasets
We train our model on datasets DL3DV [11]
and RealEstate10K [34], which contains in-door and out-
door scene. We test in the remained set from training of
DL3DV and RealEstate10K, and test the out-of-domain per-
formance on DTU [6].
5

<!-- page 6 -->
Table 1. Novel view synthesis performance comparison on the RealEstate10K with 2 views as input. Our method largely outperforms
pose-free and pose-required methods across all overlap settings, especially for small overlap setting. The best and second-best results are
highlighted.
Method
Small
Medium
Large
Average
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Pose-Required
pixelNeRF [28]
18.417
0.601
0.526
19.930
0.632
0.480
20.869
0.639
0.458
19.824
0.626
0.485
AttnRend [3]
19.151
0.663
0.368
22.532
0.763
0.269
25.897
0.845
0.186
22.664
0.762
0.269
pixelSplat [1]
20.263
0.717
0.266
23.711
0.809
0.181
27.151
0.879
0.122
23.848
0.806
0.185
MVSplat [2]
20.353
0.724
0.250
23.778
0.812
0.173
27.408
0.884
0.116
23.977
0.811
0.176
Pose-Free
DUST3R [21]
14.101
0.432
0.468
15.419
0.451
0.432
16.427
0.453
0.402
15.382
0.447
0.432
MAST3R [10]
13.534
0.407
0.494
14.945
0.436
0.451
16.028
0.444
0.418
14.907
0.431
0.452
Splatt3r [17]
14.352
0.475
0.472
15.529
0.502
0.425
15.817
0.483
0.421
15.318
0.490
0.436
CoPoNeRF [5]
17.393
0.585
0.462
18.813
0.616
0.392
20.464
0.652
0.318
18.938
0.619
0.388
NopoSplat [27]
22.514
0.784
0.210
24.899
0.839
0.160
27.411
0.883
0.119
25.033
0.838
0.160
FLARE [32]
22.468
0.763
0.198
24.872
0.836
0.162
27.012
0.881
0.123
24.891
0.831
0.161
AnySplat [7]
22.703
0.781
0.221
25.107
0.844
0.156
27.414
0.883
0.118
25.176
0.839
0.161
IFSplat(Ours)
24.207
0.813
0.183
26.421
0.853
0.145
27.895
0.891
0.114
26.291
0.854
0.145
GIFSplat(Ours)
24.617
0.827
0.181
26.714
0.866
0.131
27.987
0.902
0.112
26.559
0.867
0.138
Figure 5. Qualitative comparisons on representative indoor scenes from RealEstate10K. Columns: sparse input views, FLARE,
AnySplat, our GIFSplat with generative prior, and ground truth (GT). Red dashed boxes highlight that GIFSplat recovers sharper boundaries
(e.g., door frames, wall corners), more faithful textures, and fewer artifacts such as texture sticking.
6

<!-- page 7 -->
Baselines and metrics
To compare with previous meth-
ods, we mainly consider three evaluation setups.
First,
novel view synthesis from 8 input views at 448×448 res-
olution on DL3DV, where we adopt PixelSplat, MVSplat,
FLARE, AnySplat as baselines. Second, we also evaluate
on the commonly used 2-view at 256×256 resolution setup
on RealEstate10K [34] with different overlap settings, ex-
cept for aforementioned baseline, we compare with bench-
mark in [27]. In addition, we test the generalization abil-
ity on DTU using model trained on RealEstate10K. We use
PSNR, SSIM, LPIPS as our metric to evaluate the perfor-
mance of novel view synthesis. More information is pro-
vided in the Appendix.
Experimental settings
We use gsplat [30] as differential
rasterizer and AdamW as optimizer . We train and evaluate
our model on 4 NVIDIA DGX H200 141GB GPU for all ex-
periments. For experiments on DL3DV, we fine-tuned the
initializer for 30K steps and trained the iterative gaussians
head for 300K steps. For experiments on RealEstate10K,
we fine-tuned the initializer for 50K steps and trained the
iterative gaussians head for 100K steps. More information
is provided in the Appendix. For DL3DV, we follow the
FLARE test split and evaluate in 112 unseen scenes (8 views
for input, 9 views for evaluation). For RealEstate10K (2
views for input, 3 views for evaluation) and generalization
on DTU (2 views for input, 4 views for evaluation) experi-
ments, we all follow the NopoSplat evaluation setting.
4.2. Comparison with Feed-forward baseline
2-view evaluation on RealEstate10K
Tab. 1 and Fig. 5
show the quantitative and qualitative results, respectively.
We follow the settings of NopoSplat [27], splitting the scene
to three parts according to the overlap ratio. Across all set-
ting, our iterative residual model IFSplat get the better re-
sults than existing baselines. Moreover, GIFSplat gets the
best results with the help of generative prior, preserving
fine textures, with fewer blurs than all other models. No-
tably, some implausible deformation such as the door and
wardrobe are rectified by generative prior.
8-view evaluation on DL3DV
We evaluate on dataset
DL3DV with 8 input views, results are summarized in
Tab. 2.
To reduce storage, when training the model for
8 views, we downsample 2x the depth and project to 3D
space as location of 3D gaussians, and predict other param-
eters of 3d gaussians. Across both baselines, GIFSplat con-
sistently outperforms strong feed-forward baselines on all
metrics in Tab. 2, aligning with the sharper, artifact-free
visuals in Fig. 6. Notably, our method attains these im-
provements without relying on camera pose, demonstrates
the robustness to sparse views.
Table 2.
Quantitative comparison on DL3DV with 8 input
views.
The pose-needed column indicates whether a method
requires camera parameters.
Metrics are PSNR↑, SSIM↑, and
LPIPS↓.
Among feed-forward approaches, Ours requires nei-
ther intrinsics nor extrinsics and achieves the highest PSNR/SSIM
and the lowest LPIPS. The best and second-best results are high-
lighted.
Method
Pose?
Metrics @ 8 views
PSNR ↑
SSIM ↑
LPIPS ↓
PixelSplat [1]
✓
22.55
0.727
0.192
MVSplat [2]
✓
22.08
0.717
0.189
FLARE [32]
✗
23.33
0.746
0.237
AnySplat [7]
✗
23.76
0.762
0.187
IFSplat(Ours)
✗
24.69
0.809
0.171
GIFSplat(Ours)
✗
24.91
0.824
0.164
Figure 6.
Qualitative comparisons on DL3DV. Columns:
FLARE, AnySplat, GIFSplat (ours), and GT. Across representa-
tive scenes, GIFSplat preserves sharper edges and textures while
suppressing blur and texture sticking compared with feed-forward
baselines.
Generalization to out-of-domain datasets
Tab. 3 results
of cross-dataset evaluation of baselines, indicate that our
iterative residual feed-forward design generalizes well to
unconstrained captures, complementing the out-of-domain
evaluation on DTU dataset, with over 2 dB performance
gain. In addition, we further show the generalization ability
of our model in Appendix, demonstrating that the robust-
ness of our model even for in-the-wild images.
7

<!-- page 8 -->
Table 3.
Out-of-distribution performance comparison.
Our
method shows superior performance when cross-dataset evalua-
tion on DTU using the model solely trained on RealEstate10K.
The best and second-best results are highlighted.
Methods
Pose?
DTU @ 2 views
PSNR ↑
SSIM ↑
LPIPS ↓
pixelSplat [1]
✓
11.551
0.321
0.633
MVSplat [2]
✓
13.929
0.474
0.385
NopoSplat [27]
✗
17.899
0.629
0.279
FLARE [32]
✗
17.528
0.596
0.283
AnySplat [7]
✗
18.122
0.632
0.276
IFSplat(ours)
✗
19.921
0.701
0.274
GIFSplat(ours)
✗
20.214
0.716
0.251
4.3. Ablation Study
Tab. 5 validates the contribution of each component. Re-
moving the iterative refinement (w/o Stage 2) yields the
largest degradation across all metrics, confirming the neces-
sity of iterative residual updates. Disabling the generative
prior (w/o G. prior) also reduces performance, particularly
in LPIPS, showing that the prior-aware cues effectively sup-
press artifacts and improve perceptual fidelity. Disabling
the window attention (w/o window) also make substantial
performance drop, which indicates the function of iteraction
in 3D point space. The full configuration achieves the best
performance, demonstrating the complementarity between
all components.
Table 4. Ablation study. We ablate the iterative refinement mod-
ule (w/o Refinement), the window attention (w/o window att.), and
the generative prior (w/o Gen. prior) from our full model, respec-
tively. The full configuration achieves the best performance across
all metrics. The best and second-best results are highlighted.
Components
Metrics
PSNR ↑
SSIM ↑
LPIPS ↓
w/o Refinement
24.781
0.826
0.169
w/o window att.
25.327
0.837
0.152
w/o Gen. Prior
26.291
0.854
0.145
Full
26.559
0.867
0.138
4.4. Additional Experiments
Hyperparameter analysis for iterative steps
We vary
the number of forward refinements N while keeping all
other settings fixed. All experiments are implemented on
RealEstate10K dataset in a two-view setting. Accuracy im-
proves monotonically from N=0 and exhibits clear dimin-
ishing returns after a few steps. In practice we use a small
N = 3 for all main experiments as a favorable trade-off
between accuracy and latency .
Table 5. Analysis of iterative steps. We report metrics as the
number of refinement steps T increases from the initial prediction,
for both IFSplat and our GIFSplat. Performance improves steadily
and saturates around 3 iterations.
Steps
Metrics@IFSplat
Metrics@GIFSplat
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
initial
24.901
0.831
0.164
24.901
0.831
0.164
1
25.512
0.843
0.155
25.774
0.845
0.151
2
25.873
0.847
0.150
26.107
0.849
0.147
3
26.291
0.854
0.145
26.559
0.867
0.138
4
26.311
0.855
0.146
26.561
0.865
0.137
Additional experiments
Due to space constraints, we de-
fer three sets of experiments to the Appendix: (1) a time-
complexity analysis of our model, showing that our method
retains second-scale runtime even with scene-specific re-
finement and fusion of generative priors, with Fig. 7 illus-
trating one representative part of this analysis; (2) compar-
isons to 3D reconstruction baselines that incorporate gener-
ative priors, demonstrating that our method is consistently
superior in both quality and efficiency in sparse-view set-
tings; and (3) experiments on in-the-wild images demon-
strating robustness under domain shift. Additional experi-
mental settings, quantitative results, and qualitative visual-
izations are provided in the Appendix.
Initial
1
2
3
#Iterations
0
1
2
3
4
Inference Time (s)
IFSplat (Real10K)
IFSplat (DL3DV)
GIFSplat (Real10K)
GIFSplat (DL3DV)
Figure 7. Time complexity analysis. Inference time of our meth-
ods IFSplat and GIFSplat on two datasets as a function of the
number of refinement steps. Both methods scale approximately
linearly with T.
5. Conclusion
We presented GIFSplat, an iterative feed-forward 3D Gaus-
sian Splatting framework that refines an initial prediction
through multi-step forward-only residual updates guided by
both observations and diffusion-enhanced cues. Our IFS-
plat baseline already brings scene-specific refinement with-
out test-time backpropagation, and the full GIFSplat vari-
ant further injects a frozen generative prior via lightweight
8

<!-- page 9 -->
Gaussian-level discrepancy cues. Experiments on DL3DV,
RealEstate10K, and DTU show consistent gains over re-
cent feed-forward 3D reconstruction methods, especially in
cross-domain settings.
Limitations.
Our refinement head still focus on static
scenes and a small, fixed set of input modalities. Extending
GIFSplat to dynamic content or to flexible geometric priors
such as known depth maps or normal maps is an interesting
direction for future work.
Acknowledgments
This research was supported by the Australian Research
Council Discovery Project (Grant Number DP240103334).
References
[1] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19457–19467, 2024. 2, 4, 6,
7, 8
[2] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision, pages 370–386. Springer, 2024. 2, 4, 6, 7, 8
[3] Yilun Du, Cameron Smith, Ayush Tewari, and Vincent Sitz-
mann. Learning to render novel views from wide-baseline
stereo pairs. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4970–
4980, 2023. 6
[4] Tobias Fischer, Samuel Rota Bul`o, Yung-Hsu Yang, Nikhil
Keetha, Lorenzo Porzi, Norman M¨uller, Katja Schwarz,
Jonathon Luiten, Marc Pollefeys, and Peter Kontschieder.
Flowr: Flowing from sparse to dense 3d reconstructions. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 27702–27712, 2025. 3
[5] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong
Yang, Seungryong Kim, and Chong Luo. Unifying corre-
spondence, pose and nerf for pose-free novel view synthesis
from stereo pairs, 2023. 6
[6] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola,
and Henrik Aanæs. Large scale multi-view stereopsis eval-
uation. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 406–413, 2014. 5
[7] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 1, 2, 4, 5, 6, 7, 8
[8] Gyeongjin Kang, Seungtae Nam, Xiangyu Sun, Sameh
Khamis, Abdelrahman Mohamed, and Eunbyung Park. ilrm:
An iterative large 3d reconstruction model. arXiv preprint
arXiv:2507.23277, 2025. 2
[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):1–14, 2023. 1, 2
[10] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 1, 2,
6
[11] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al.
Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22160–22169, 2024. 5
[12] Xi Liu, Chaoyi Zhou, and Siyu Huang.
3dgs-enhancer:
Enhancing unbounded 3d gaussian splatting with view-
consistent 2d diffusion priors. Advances in Neural Informa-
tion Processing Systems, 37:133305–133327, 2024. 3
[13] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[14] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 2
[15] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall.
Dreamfusion: Text-to-3d using 2d diffusion.
arXiv
preprint arXiv:2209.14988, 2022. 3
[16] Gilad Sharir, Asaf Noy, and Lihi Zelnik-Manor. An image is
worth 16x16 words, what is a video worth? arXiv preprint
arXiv:2103.13915, 2021. 2
[17] Brandon Smart, Chuanxia Zheng, Iro Laina, and Vic-
tor Adrian Prisacariu.
Splatt3r:
Zero-shot gaussian
splatting from uncalibrated image pairs.
arXiv preprint
arXiv:2408.13912, 2024. 6
[18] Stanislaw Szymanowicz, Jason Y Zhang, Pratul Srinivasan,
Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo
Martin-Brualla, Jonathan T Barron, and Philipp Henzler.
Bolt3d: Generating 3d scenes in seconds. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 24846–24857, 2025. 3
[19] Bo Wang, Yifan Zhang, Jian Li, Yang Yu, Zhenping Sun, Li
Liu, and Dewen Hu. Splatflow: Learning multi-frame optical
flow via splatting. International Journal of Computer Vision,
132(8):3023–3045, 2024. 3
[20] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 1, 2
[21] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20697–
20709, 2024. 1, 2, 6
9

<!-- page 10 -->
[22] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Goj-
cic, and Huan Ling.
Difix3d+: Improving 3d reconstruc-
tions with single-step diffusion models. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 26024–26035, 2025. 2, 3, 5
[23] Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger, and
Anpei Chen. Genfusion: Closing the loop between recon-
struction and generation via videos. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
6078–6088, 2025. 2, 3
[24] Haofei Xu, Daniel Barath, Andreas Geiger, and Marc Polle-
feys.
Resplat: Learning recurrent gaussian splats.
arXiv
preprint arXiv:2510.08575, 2025. 3
[25] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16453–16463, 2025. 2
[26] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff,
Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt
Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images
in one forward pass. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 21924–21935,
2025. 1
[27] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys,
Ming-Hsuan Yang, and Songyou Peng. No pose, no problem:
Surprisingly simple 3d gaussian splats from sparse unposed
images. arXiv preprint arXiv:2410.24207, 2024. 2, 4, 6, 7, 8
[28] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 4578–4587, 2021. 6
[29] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li,
Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan,
and Yonghong Tian. Viewcrafter: Taming video diffusion
models for high-fidelity novel view synthesis. arXiv preprint
arXiv:2409.02048, 2024. 3
[30] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 1, 2, 7
[31] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 5
[32] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
don Wetzstein. Flare: Feed-forward geometry, appearance
and camera estimation from uncalibrated sparse views. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 21936–21947, 2025. 2, 6, 7, 8
[33] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and
Vladlen Koltun.
Point transformer.
In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 16259–16268, 2021. 5
[34] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely.
Stereo magnification:
Learning
view synthesis using multiplane images.
arXiv preprint
arXiv:1805.09817, 2018. 5, 7
10
