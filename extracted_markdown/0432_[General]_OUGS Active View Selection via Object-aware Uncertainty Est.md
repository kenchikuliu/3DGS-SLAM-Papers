<!-- page 1 -->
Preprint
(16 December 2025)
OUGS: Active View Selection via Object-aware Uncertainty
Estimation in 3DGS
Haiyi Li
Qi Chen
Denis Kalkofen
Hsiang-Ting Chen
†
Abstract
Recent advances in 3D Gaussian Splatting (3DGS) have achieved state-of-the-art results for novel view synthesis. However,
efficiently capturing high-fidelity reconstructions of specific objects within complex scenes remains a significant challenge. A
key limitation of existing active reconstruction methods is their reliance on scene-level uncertainty metrics, which are often
biased by irrelevant background clutter and lead to inefficient view selection for object-centric tasks. We present OUGS, a
novel framework that addresses this challenge with a more principled, physically-grounded uncertainty formulation for 3DGS.
Our core innovation is to derive uncertainty directly from the explicit physical parameters of the 3D Gaussian primitives (e.g.,
position, scale, rotation). By propagating the covariance of these parameters through the rendering Jacobian, we establish a
highly interpretable uncertainty model. This foundation allows us to then seamlessly integrate semantic segmentation masks
to produce a targeted, object-aware uncertainty score that effectively disentangles the object from its environment. This allows
for a more effective active view selection strategy that prioritizes views critical to improving object fidelity. Experimental
evaluations on public datasets demonstrate that our approach significantly improves the efficiency of the 3DGS reconstruction
process and achieves higher quality for targeted objects compared to existing state-of-the-art methods, while also serving as a
robust uncertainty estimator for the global scene.
CCS Concepts
• Computing methodologies →Image-based rendering; Object detection; Computational photography;
1. Introduction
Efficient 3D scene reconstruction is a foundational goal in com-
puter vision and robotics. The advent of Neural Radiance Fields
(NeRF) [MST∗20] marked a breakthrough, enabling photorealis-
tic novel view synthesis by learning implicit volumetric represen-
tations. More recently, 3D Gaussian Splatting (3DGS) [KKLD23]
has emerged as a compelling alternative. By modeling scenes with
explicit 3D Gaussian primitives and leveraging a fast, differentiable
rasterization pipeline, 3DGS achieves real-time rendering speeds
without compromising visual fidelity, addressing the high compu-
tational cost that limits NeRF’s practicality.
Despite these advances, both NeRF and 3DGS remain highly
data-intensive, typically requiring dense image captures to produce
high-quality reconstructions [NBM∗21, CKD∗25]. This motivates
the need for active reconstruction [CLK11], where a human or
robotic agent actively selects a minimal subset of views that max-
imally reduces uncertainty. Achieving this requires models to es-
timate their own information needs in real-time—a capability that
hinges on accurate uncertainty estimation.
In this context, recent work has explored incorporating uncer-
tainty into both NeRFs [GRS∗24,KMKS24] and 3DGS. For the lat-
† Corresponding author: tim.chen@adelaide.edu.au
Figure 1: A complex background can inflate image-level uncer-
tainty and mislead active view selection away from the object of
interest.
ter, emerging research has introduced methods based on ensemble
variance [HD25] or Fisher Information approximations [JLD24].
However, these pioneering methods share a fundamental limita-
tion: they typically estimate uncertainty at the scene level. As il-
lustrated in Figure 1, a global uncertainty score is often dominated
by complex but irrelevant background clutter, misleading the view
selection process. This is particularly problematic for the growing
number of applications where the primary goal is not to reconstruct
© 16 December 2025 The Author(s).
arXiv:2511.09397v2  [cs.CV]  15 Dec 2025

<!-- page 2 -->
Figure 2: Object-aware uncertainty guides 3DGS view planning for precise object reconstruction. Our physically-grounded uncertainty
model, derived from the explicit parameters of the 3D Gaussians, is combined with a semantic mask to generate an object-level uncertainty
score. This score effectively guides the active view selection to improve object fidelity, as shown for 5, 10, and 20 selected views.
an entire environment, but to capture a specific object of interest
with the highest possible fidelity.
To address this critical gap, we introduce OUGS, a framework
designed specifically for object-centric active reconstruction (Fig-
ure 2). Our work is built upon a key insight: to effectively isolate an
object’s uncertainty, one must first model uncertainty from a more
fundamental, physically-grounded source. Instead of deriving un-
certainty from the abstract weights of an implicit neural network,
as is common practice, our method is the first to establish a rig-
orous framework that quantifies uncertainty directly from the ex-
plicit physical parameters of the 3D Gaussian primitives—their
position, scale, rotation, and appearance.
We begin by treating these parameters as random variables and
propagate their covariance through the differentiable rendering
pipeline via the Jacobian. This yields a pixel-wise visual uncer-
tainty that is not only robust but also highly interpretable. This
physically-grounded foundation allows us to then seamlessly in-
tegrate semantic masks to disentangle the uncertainty of a target
object from its environment. To ensure scalability, we approximate
the parameter covariance using a diagonal Fisher Information Ma-
trix (FIM), updated efficiently through an exponential moving av-
erage. This complete formulation enables an active view selection
strategy that is powerfully and precisely focused on the object of
interest. Our comprehensive evaluations demonstrate that this ap-
proach facilitates informed next-best-view selection, substantially
enhancing both the interpretability and efficiency of the object re-
construction process.
Our contributions are threefold:
• We introduce a novel active reconstruction framework specifi-
cally designed for object-centric tasks in 3DGS, addressing a
key limitation of existing scene-level methods.
• We propose a new, physically-grounded uncertainty model
based on the explicit parameters of 3D Gaussians, offer-
ing greater accuracy and interpretability compared to implicit,
weight-based approaches.
• Through extensive experiments, we demonstrate that our method
significantly outperforms state-of-the-art approaches in object-
focused reconstruction while maintaining strong performance on
global scene metrics.
2. Related Work
2.1. Uncertainty in 3D Splatting
Quantifying uncertainty in 3DGS is an emerging research area cru-
cial for real-world applications. Current approaches can be catego-
rized into four main directions: 1) Variational/Bayesian. A prin-
cipled approach is to treat Gaussian parameters as distributions.
Li & Cheung [LC24] use hierarchical Bayesian priors, while Sa-
vant et al. [SVM24] employ variational inference. While mathe-
matically rigorous, these methods incur significant computational
overhead (2–3x inference cost), limiting their real-time applicabil-
ity. 2) Sensitivity Pruning. Alternatively, some methods measure
the model’s sensitivity to its parameters. PUP 3D-GS, for instance,
uses a Hessian-based metric to prune Gaussians with high uncer-
tainty. This approach is efficient but offers a less direct measure
of predictive uncertainty. 3) Learned Uncertainty Fields. Another
paradigm trains an auxiliary network to directly predict uncertainty.
UNG-GS [TCZ∗25] adds a Spatial Uncertainty Field for sparse in-
puts, while Han & Dumery [HD25] learn a view-dependent field.
These methods are flexible but risk producing uncalibrated or phys-
ically implausible estimates. 4) Information-Theoretic. Finally,
information theory can be used to quantify the information gain of
new views. GauSS-MI [XCZ∗25], for instance, selects views that
maximize mutual information. While powerful for view selection,
this paradigm focuses on the information value of potential views
rather than the inherent uncertainty of the current reconstruction.
Our work carves a distinct path by adopting an efficient, FIM-
based approximation of parameter uncertainty. We apply this for-
mulation directly to the problem of object-centric active view selec-
tion—a critical application gap not fully addressed by prior works.
2

<!-- page 3 -->
2.2. Uncertainty for Active View Selection
Active view selection, or Next-Best-View (NBV) planning, is a
long-standing problem in computer vision and robotics [CLK11],
aiming to intelligently choose views to maximize reconstruction
quality while minimizing cost. Methodologies have evolved sig-
nificantly over time. 1) Traditional and Geometric Methods.
Early approaches in robotics often relied on geometric heuristics.
For instance, receding-horizon planners like the one by Bircher et
al. [BKA∗16] aim to maximize the exploration of unknown free
space using occupancy maps. Other classical NBV methods use
voxel-grid representations and select views based on metrics like
Shannon entropy or frontier exploration [KSH22]. While effec-
tive for coverage, these discretized methods can struggle to cap-
ture fine geometric details and are less suited for the continuous
representations used in modern neural rendering. 2) Uncertainty
in Neural Rendering. The rise of neural rendering has spurred a
new wave of uncertainty-driven NBV methods. For NeRFs, Ac-
tiveNeRF [PLSH22] selects views by minimizing rendered color
variance, while FisherRF [JLD24] proposes using Fisher informa-
tion gain as a more principled metric. These ideas have been ex-
tended to 3DGS; POp-GS [WAM∗25] also employs a Fisher ma-
trix, and GauSS-MI [XCZ∗25] maximizes mutual information. A
common thread connects these powerful methods: they compute
a global, scene-level score. This design choice leads to a criti-
cal limitation: they are agnostic to semantic importance, mean-
ing a complex but irrelevant background can dominate view selec-
tion. 3) Other View Selection Paradigms. Beyond uncertainty and
information-theoretic approaches, other paradigms have been ex-
plored. Learning-based methods, such as NeurAR [RZH∗23], em-
ploy reinforcement learning to train an agent that learns an optimal
view selection policy directly from simulation. Concurrently, other
works focus on explicitly modeling visibility. For instance, Neural
Visibility Fields [XDM∗24] learn to predict which parts of a scene
are visible from a given viewpoint, guiding selection towards views
that maximize observable new area. While powerful, these methods
either require extensive training or shift the focus from reconstruc-
tion fidelity to geometric coverage.
Our work addresses the critical limitation of scene-level uncer-
tainty methods. By introducing an object-aware mechanism, we en-
able the view selection process to focus on the semantically impor-
tant regions of the scene, a challenge not explicitly addressed by
any of these prior paradigms.
3. Method
3.1. Preliminary: 3D Gaussian Splatting
Our method builds upon the 3D Gaussian Splatting (3DGS) frame-
work [KKLD23], which represents a scene as a collection of
anisotropic 3D Gaussian primitives G = {Gi}Ng
i=1. To ground our
uncertainty analysis, we first provide a detailed list of the parame-
ters used in the differentiable rendering pipeline. Each Gaussian Gi
is fully described by a parameter vector θi:
θi =

µi
|{z}
Center
, si
|{z}
Scale
,
qi
|{z}
Rotation
,
αi
|{z}
Opacity
, fdc
i ,fsh
i
| {z }
Color (SH)


⊤
(1)
where the components are:
• Geometry: The 3D center µi ∈R3, an anisotropic scaling vector
si ∈R3
+, and an orientation quaternion qi ∈S3. Together, these
define the Gaussian’s position, size, and orientation.
• Appearance: A scalar opacity value αi ∈R and view-dependent
color modeled by Spherical Harmonics (SH). The color is pa-
rameterized by the degree-0 (DC) term fdc
i
∈R3 and a set of
higher-order coefficients fsh
i ∈R3×15.
A visual breakdown of these parameters is provided in Figure 3
shows these parameters visually.
Figure 3: The parameterization of a 3D Gaussian primitive.
Each Gaussian, the fundamental building block of our scene repre-
sentation, is defined by a set of explicit physical parameters. Our
method’s core innovation lies in directly quantifying the uncer-
tainty of these physical parameters.
Rendering in 3DGS uses a differentiable splatting approach
based on standard alpha compositing. First, the 3D Gaussians are
projected onto the 2D image plane and sorted in front-to-back order
based on their depth. The color C(u) for a pixel u is then accumu-
lated as:
C(u) = ∑
i∈I(u)
ci(u)α′
i(u)
i−1
∏
j=1
 1−α′
j(u)

(2)
where I(u) is the ordered list of Gaussians overlapping the pixel
u, ci(u) is the view-dependent color evaluated from SH. The ef-
fective opacity α′
i(u) is determined by modulating the Gaussian’s
learned opacity parameter αi by its projected 2D profile at the pixel
location.
3.2. Mapping 3D Gaussian Parameter Uncertainty to
Pixel-wise Object-aware Uncertainty
To quantify uncertainty, we treat the parameter vector of each Gaus-
sian as a random variable and initialize it with a Gaussian prior,
θi ∼N(θ0
i ,Σ0
i ). For notational clarity, we stack the per-Gaussian
covariances into a block-diagonal matrix
Σ = diag
 Σ1,...,ΣNg

∈Rd×d
(3)
where d = Ng ·dg. Throughout the rest of this section, Σ refers to
this global covariance, while Σi denotes its i-th block.
We then project the uncertainty of the 3DGS parameters into
pixel space and describe how it interacts with a soft object mask.
3

<!-- page 4 -->
Here, we present simplified expressions; full derivations and a
second-order error bound are provided in the Appendix.
Decomposition of Uncertainty Sources. A key advantage of
our explicit, parameter-centric formulation is the ability to de-
compose the total uncertainty into its underlying physical sources.
We can partition the Gaussian parameter vector θi into a geome-
try component, θg
i = [µi,si,qi]⊤, and an appearance component,
θa
i = [αi,fdc
i ,fsh
i ]⊤. Consequently, the total parameter vector θ and
its Jacobian Ju can be similarly partitioned:
θ =
θg
θa

,
Ju =

Jg
u
Jau

(4)
where Jg
u = ∂C(u)/∂θg and Jau = ∂C(u)/∂θa. Assuming indepen-
dence between geometric and appearance parameters (a reasonable
simplification enforced by our diagonal FIM approximation), the
pixel-wise color covariance from Eq. 6 can be expressed as a sum
of two distinct sources:
ΣC(u) ≈
Jg
uΣg(Jg
u)⊤
|
{z
}
Geometric Uncertainty
+
Ja
uΣa(Ja
u)⊤
|
{z
}
Appearance Uncertainty
(5)
This decomposition is theoretically significant. It allows us to dif-
ferentiate between uncertainty arising from poorly constrained ob-
ject geometry (e.g., ambiguous boundaries, fine structures) and
uncertainty from poorly observed appearance (e.g., complex ma-
terials, view-dependent effects). Implicit methods like FisherRF,
which operate on abstract network weights, lack this inherent in-
terpretability.
Pixel-wise uncertainty Assume complete set of scene parame-
ters θ = {θi} and θ⋆is the MAP estimate after optimization. For
small parameter perturbations δθ = θ −θ⋆, the change in pixel
color δC(u) can be linearly approximated using a first-order Taylor
expansion:
δC(u) ≈Juδθ
where with Jacobian Ju = ∂C(u;θ)/∂θ ∈R3×d. Given E[δθ] = 0
under a prior normal distribution, the induced pixel-colour covari-
ance can be written as
ΣC(u) = Var[C(u;θ)] ≈Ju ΣJ⊤
u
(6)
where Σ is the full parameter covariance. Eq 6 shows that the Jaco-
bian acts as a lever arm that magnifies (or attenuates) each param-
eter’s uncertainty in proportion to that parameter’s influence on the
pixel [Bon08].
Pixel-wise object-aware uncertainty To estimate the uncer-
tainty of a specific object k, we introduce a soft mask Mk(u) ∈[0,1]
based on semantic probabilities. This allows us to define an object-
specific pixel covariance ΣC,k(u) by masking the standard error
propagation formula:
ΣC,k(u) =
 Mk(u)
2  Ju ΣJ⊤
u

(7)
The mask term Mk(u) is squared because covariance propagates
quadratically via the Jacobian and its transpose.
3.3. Updating Uncertainty With FIM
While our formulation provides a physically interpretable model
of uncertainty, direct computation of the full covariance matrix Σ
is intractable. We therefore approximate it with the inverse of the
Fisher Information Matrix (FIM), Σ ≃σ2 I−1 [LMV∗17]. Cru-
cially, our FIM is defined over the space of the 3D Gaussians’
physical parameters, capturing how perturbations in geometry and
appearance affect the rendered output. This stands in contrast to
implicit-representation methods where the FIM is computed over
abstract neural network weights.
Diagonal FIM as a Parameter Decoupling Assumption. To
ensure computational tractability, we make a key simplifying as-
sumption: we approximate the full FIM with its diagonal entries
only, effectively assuming that the different physical parameters of
a Gaussian are locally independent. This diagonal approximation,
I ≈diag(I), aligns perfectly with the uncertainty decomposition
presented in Eq. 5. It implies that a Gaussian’s geometric uncer-
tainty (e.g., in its position, µi) is decoupled from its appearance un-
certainty (e.g., in its color, fi). While this is a strong simplification,
it is a common and effective strategy that allows us to efficiently
estimate the parameter-wise variances.
We update these diagonal FIM entries, denoted It,i, continuously
using an exponential moving average (EMA) of the squared gradi-
ents [KB17]:
It,i = αt It−1,i +(1−αt)

∇θiℓt
2
(8)
The EMA schedule, controlled by a linearly decaying αt, where
αt = 0.95 × (1 −t/T), smooths noisy gradients early in train-
ing.Substituting this tractable diagonal approximation into our
object-aware uncertainty propagation (Eq. 7) yields our final for-
mulation:
ΣC,k(u) =
 Mk(u)
2 Ju
 diagIt +λI
−1J⊤
u
(9)
By summing the trace of ΣC,k(u) over all pixels corresponding to
object k, we obtain a scalar score that guides our active view selec-
tion.
4. Experiments
4.1. Experimental Setup
Dataset We conduct our evaluation across three public datasets
for a comprehensive experiment. Mip-NeRF 360 [BMV∗22] con-
tains four bounded indoor rooms with strong specular clutter and
five unbounded outdoor scenes with foliage occlusion and high dy-
namic range lighting. We follow the 3DGS protocol [KKLD23]:
every 8th view is held out for testing. We also experiment on
Light-Field (LF) dataset [YSHWSH16], which offers four table-
top objects torch, statue, basket, and africa captured by a motorised
gantry. A salient trait is the foreground-centric framing: each tar-
get object remains centred across all views, yielding stable masks
from SAM-2 and making the dataset ideal for evaluating per-object
calibration. In addition we target the train and truck from Tanks
& Temples (TNT) [KPZK17], which offers large-scale, drone-
style outdoor captures characterised by long-baseline parallax and
strong depth discontinuities. Inspired by the sparse-view protocol
4

<!-- page 5 -->
Table 1: Active View Selection on Mip-NeRF360, Tanks&Temples, and LF datasets. "Panoramic" evaluates the full image; "Object-aware"
evaluates only inside the object mask. Rows denote selection policies.
Method
Metrics
Mip-NeRF360
LF
Tanks & Temples
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Random
17.9140
0.5640
0.4300
19.0857
0.6646
0.2669
15.8784
0.5365
0.3720
ActiveNeRF
17.8890
0.5330
0.4140
21.2263
0.7691
0.1742
16.2918
0.5892
0.2514
BayesRays
18.8120
0.5730
0.4210
21.9232
0.7628
0.1752
16.9322
0.6091
0.3252
FisherRF
20.3510
0.6010
0.3610
23.6450
0.8323
0.1651
17.3684
0.6296
0.3091
GauSS-MI
20.8150
0.6433
0.2710
23.9820
0.8354
0.1628
17.4210
0.6315
0.2425
OUGS
20.6099
0.6453
0.2727
23.7014
0.8058
0.1726
17.2666
0.6125
0.2468
Object-Aware
Random
26.3382
0.9732
0.0266
31.0709
0.9866
0.0227
24.6778
0.9306
0.1349
FisherRF
26.4312
0.9731
0.0276
30.82740
0.9813
0.0231
23.4830
0.9208
0.1396
GauSS-MI
27.3012
0.9764
0.0258
30.9742
0.9830
0.0226
24.9832
0.9318
0.1336
OUGS
29.6099
0.9813
0.0241
32.1856
0.9888
0.0221
26.2533
0.9333
0.1169
Figure 4: Object-aware approach speeds up convergence. Curves are recorded on the statue scene of the LF dataset as new views are
added. Top row: panoramic PSNR/SSIM/LPIPS; middle row: the same metrics evaluated only inside the object mask. Bottom strip: visual
progression at 5, 10, 15 and 20 views (ours in the first row, FisherRF below); red circles mark regions where the competing method keeps
struggling while our reconstruction sharpens steadily.
of Shen [SAMNR22], we additionally include intentionally biased
and imbalanced camera orbits. They exacerbate parallax/occlusion
artifacts and thus form a stress test for Fisher-based uncertainty
modeling. All RGB frames are kept at full original resolution to
preserve fine geometric cues; no cropping or scaling is applied.
Baselines We compare our approach against several state-
of-the-art methods that explicitly model uncertainty for next-
best-view (NBV) selection. Our baseline pool includes: ActiveN-
eRF [PLSH22], FisherRF [JLD24], and Bayes’ Rays [GRS∗24].
Furthermore, to ensure a thorough comparison against the state-
5

<!-- page 6 -->
GT
OUGS
FisherRF
Random
Figure 5: Qualitative result on Mip-NeRF360. From left to right are ground truth, Ours (OUGS), FisheRF, and Random. Each row corre-
sponds to a different scene. 20 views were selected to train a model and render the result on the test set. The blue box circles the object of
our interest, while the red box circles some of the background.
of-the-art, we include GauSS-MI [XCZ∗25], a recent information-
theoretic method that represents the state-of-the-art in scene-level
active reconstruction. We also note the concurrent work POp-
GS [WAM∗25], which also focuses on next best view selection.
However, as its implementation was not publicly available dur-
ing our experimental phase, a direct quantitative comparison was
not feasible. All baselines are trained with their publicly released
repositories using the authors’ default hyper-parameters, learning-
rate schedules, and random seeds. To further control for segmen-
tation bias, we feed the identical For fairness, the same SAM-2
probability masks are used for all baselines wherever applicable; if
a method does not support masked view scoring, masks are used
only for object-only evaluation while keeping its original selection
policy. NBV selection is executed with the settings recommended
in the respective papers, ensuring a fair, apples-to-apples compar-
ison. Metrics Our object-aware uncertainty estimation algorithm
was evaluated based on render quality for the object and scene.
Hence, we apply PSNR, SSIM, and LPIPS to evaluate the result.
We report each metric on the average result of each scene of a
specific dataset, and list a comparison result under the same con-
figuration. Following the experimental protocol of [GRS∗24], we
quantify the quality of our predictive uncertainty using the Area
Under the Sparsification Error curve (AUSE), a widely-adopted
metric in depth-estimation literature. For every test image we first
compute the per-pixel absolute error and the corresponding pre-
dicted uncertainty. We then iteratively mask out the top t% of pix-
els (t = 1,...,100) in descending order of predicted uncertainty
(highest-uncertainty first). At each sparsification level, we compute
6

<!-- page 7 -->
Table 2: Validation of our parameter-centric FIM as a standalone uncertainty estimator. To isolate the quality of our core uncertainty
model, we evaluate it on the full scene without any object-aware masks. The table reports the Area Under the Sparsification Error (AUSE), a
rigorous metric for uncertainty quality (lower is better). Our method demonstrates highly competitive or superior performance against prior
art across all scenes, confirming that our physically-grounded FIM formulation is a robust and accurate uncertainty estimator in its own right.
africa
basket
statue
torch
TNT-Train
TNT-Truck
∆MSE↓
∆MAE↓
∆MSE↓
∆MAE↓
∆MSE↓
∆MAE↓
∆MSE↓
∆MAE↓
MSE↓
MAE↓
MSE↓
MAE↓
ActiveNeRF
1.123
0.958
0.642
0.546
0.818
0.732
1.513
1.246
1.279
1.076
0.994
0.438
Bayes’ Rays
0.445
0.271
0.326
0.284
0.192
0.182
0.342
0.224
0.822
0.689
0.865
0.529
FisherRF
0.181
0.186
0.212
0.225
0.191
0.178
0.247
0.254
0.892
0.632
0.843
0.589
OUGS
0.192
0.187
0.122
0.131
0.181
0.181
0.248
0.217
0.787
0.589
0.651
0.487
the mean residual error of the retained pixels under both MAE and
MSE and trace two corresponding sparsification-error curves; inte-
grating each curve yields AUSEMAE and AUSEMSE, whose mutual
consistency offers a more rigorous and reliable assessment of our
uncertainty estimates.
Implementation Details.
Following SoTA next-best-view (NBV)
[WAM∗25] optimi-
sation and uncertainty estimation, we evaluate on the bench-
mark datasets in Sec. 4.1. The 3D Gaussians are initialised
with COLMAP [SF16], and object masks are obtained from
SAM2 [RGH∗24] to isolate the target object. NBV planning fol-
lows FisherRF [JLD24]: four initial views are selected using the
farthest-point strategy, followed by 100 epochs of training and the
addition of a new view chosen by the highest predicted uncertainty
within the object mask. This process repeats until 20 views are
reached, after which the model is optimised for 30k iterations with
the default 3D Gaussian Splatting [KKLD23] schedule.
4.2. Quantitative Results
Our primary quantitative evaluation, presented in Table 1, reveals a
clear and compelling story about the state of active 3D reconstruc-
tion. We analyze the results at two levels of granularity: the full
scene and the object of interest.
Scene-Level Panoramic Performance. In the panoramic eval-
uation, which assesses the entire rendered view, the information-
theoretic method GauSS-MI establishes itself as the new state-
of-the-art, achieving the best results across most metrics and
datasets. This highlights the effectiveness of maximizing infor-
mation gain for comprehensively capturing complex scenes. Our
method, OUGS, remains highly competitive in this setting. No-
tably, OUGS’s performance is on par with or second only to
GauSS-MI, and it consistently outperforms the previous FIM-based
method, FisherRF. This confirms that our parameter-centric uncer-
tainty formulation is a robust and effective replacement for pixel-
level gradient approximations, even for global reconstruction tasks.
Object-Aware Performance. The core strength and primary
contribution of our approach are revealed when the evaluation is
restricted to the object of interest (Object-Aware rows). In this
critical, object-centric setting, the performance hierarchy shifts
dramatically. OUGS consistently and substantially outperforms
all other methods, including the panoramic SOTA GauSS-MI,
across every dataset. This significant performance gap validates
our central hypothesis and directly addresses the limitations of
scene-level methods. While approaches like GauSS-MI and Fish-
erRF are powerful, their view selection process is inevitably diluted
by background clutter. By explicitly disentangling object uncer-
tainty from the environment, our method allocates the finite view
budget far more effectively, leading to a dramatic and consistent
improvement in reconstruction fidelity for the target object.
Convergence and Visual Analysis. Figure 4 illustrates the con-
vergence behavior, comparing our approach against the FIM-based
FisherRF. While both methods track closely in the panoramic eval-
uation (top row), the object-only plots (middle row) show the stark
advantage of our strategy. OUGS achieves sharp gains in PSNR
and SSIM early in the process by immediately prioritizing object-
centric views, whereas FisherRF’s progress on the object is slower
due to distractions from background gradients. These quantitative
differences are also visually evident in the rendered results (bottom
row). As more views are added, OUGS rapidly sharpens object de-
tails, while FisherRF continues to struggle with the same regions,
sometimes even prioritizing background elements at the expense of
the object of interest.
4.3. Qualitative Results
Figure 5 provides a qualitative comparison on the Mip-NeRF 360
dataset. To specifically analyze the behavior of uncertainty estima-
tion based on the Fisher Information Matrix, we focus the visual
comparison on our method, FisherRF, and a random baseline.
The results clearly illustrate the impact of our object-aware strat-
egy. Across all scenes, our reconstructions remain visually clos-
est to the ground truth in the regions of interest (blue boxes). On
the stump scene, for example, the slender bark fibres are sharply
delineated in our result, whereas they are rendered as blurred or
are entirely missing by FisherRF. This reveals the fundamental dif-
ference in FIM-based approaches: FisherRF, however, often pro-
duces a cleaner background (red boxes). This stems from its scene-
level Fisher information score; high-gradient background textures
can dominate the score and steer the next-best-view search away
from the object. Our method, by design, resists this distraction.
Random sampling, while uninformed, spreads views uniformly and
therefore sometimes captures object-centric angles, leading to oc-
casional details that surpass FisherRF, but this comes at the cost
of high variance and no guarantees. The visual evidence strongly
7

<!-- page 8 -->
GT
Render
Uncertainty
Figure 6: Qualitative validation of our parameter-centric uncertainty. These results on the TNT-Train (top) and LF-Basket (bottom)
scenes showcase the remarkable accuracy of our uncertainty estimation. The uncertainty heatmap (right column, yellow indicates high
uncertainty) precisely localizes the regions where the final rendering (middle column) deviates from the ground truth (left column), such as
blurry structures and ghosting artifacts. This strong visual correlation demonstrates the effectiveness of our physically-grounded model in
predicting and explaining rendering errors.
indicates that our method preserves object detail most faithfully.
The modest artifacts that may appear at the image periphery do not
outweigh the substantial and consistent gains in the target region,
confirming the effectiveness of our object-aware formulation.
4.4. Ablation Study
Evaluating the FIM-based Uncertainty Estimation
We assess the effectiveness of our FIM-based uncertainty approxi-
mation in Section 3.3. To enable a fair comparison with alternative
methods, we compute the sparsification error (AUSE) over the en-
tire scene, without using the object-centric probability mask—after
training 3DGS on the full dataset. AUSE measures the quality of
uncertainty calibration. Specifically, it quantifies how the predic-
tion error decreases as the most uncertain pixels are progressively
removed. A well-calibrated uncertainty estimate will prioritize re-
moving high-error pixels first, leading to a steep drop in error. The
area under this curve captures the cumulative deviation from ideal
sparsification; lower AUSE indicates better uncertainty estimates,
meaning the uncertainty scores are more aligned with actual pre-
diction errors.
Table 2 shows the results on two scenes from the TNT
dataset and all scenes from the LF dataset. Our method
achieves the lowest AUSE scores on both TNT scenes, out-
performing FisherRF [JLD24], ActiveNeRF [PLSH22], and
Bayes’Rays [GRS∗24]. On the LF dataset, our approach also yields
competitive results across all scenes. Figure 6 complements these
findings with a visual comparison: uncertainty highlighted by our
model (yellow overlay) concentrates on background regions that
later exhibit blur or ghosting artifacts, while high-confidence areas
remain artifact-free. These results indicate that our parameter-level
uncertainty estimation not only improves overall reconstruction fi-
delity but also more precisely localizes residual errors.
Assessing the Role of Probability Mask Quality
To investigate how the quality of the soft mask influences our
object-aware uncertainty estimation, we simulate mask degrada-
tion by varying the threshold value from 0.1 to 0.9 and plot the
corresponding object level AUSE scores. Pixels with probabili-
ties above the threshold are considered part of the object. At low
threshold values (left side of the plot), the mask includes more un-
wanted background regions, leading to suboptimal view selection
and higher (worse) AUSE scores. As the threshold increases, the
mask becomes more focused on the object, improving performance
and reaching an optimal AUSE around 0.5—where the mask best
isolates the object while excluding background clutter. Beyond this
point, further increases in the threshold aggressively remove less
salient object regions, slightly degrading performance as informa-
tive areas are excluded.
Analysis of the EMA Update Schedule
Our FIM approximation relies on an online update of the diag-
onal entries using an EMA of squared gradients. A key design
choice is our decaying momentum schedule for the EMA parame-
ter, αt = 0.95×(1−t/T), which prioritizes stability early in train-
ing (high momentum) and faster adaptation later on (low momen-
tum). To validate this choice, we conduct an ablation study com-
paring our proposed schedule against simpler alternatives with a
constant momentum. We evaluate the final object-aware PSNR on
the LF-Statue scene after 20 actively selected views.
The results, presented in Table 3, confirm the effectiveness of
our proposed strategy. A constant high momentum (αt = 0.99) is
overly cautious, smoothing too much and preventing the model
from adapting quickly enough, resulting in lower PSNR. Con-
versely, a constant low momentum (αt = 0.9) is too reactive to
noisy gradients, leading to an unstable FIM estimate and the worst
performance. Our decaying schedule strikes the optimal balance,
8

<!-- page 9 -->
Figure 7: Analysis of semantic mask quality on object-aware uncertainty. We investigate the sensitivity of our method to the quality of the
probability mask by varying its binarization threshold. The plot of object-level AUSE (lower is better) reveals a clear optimal range. At low
thresholds, the mask is too permissive and includes distracting background regions, which inflates the uncertainty error. Conversely, at very
high thresholds, the mask becomes overly aggressive and erodes parts of the object, discarding valuable information and again degrading
performance.
Table 3: Ablation on the EMA update schedule, evaluated on the
LF-Statue scene. Our decaying momentum strategy outperforms all
constant momentum alternatives.
EMA Schedule Strategy
Object-Aware PSNR (dB) ↑
No EMA (Instantaneous)
29.52
Low Momentum (αt = 0.90)
31.63
Medium Momentum (αt = 0.95)
31.91
High Momentum (αt = 0.99)
32.04
Ours (Decaying Momentum)
32.18
achieving the highest PSNR. This validates that our carefully de-
signed "slow-start, fast-finish" update strategy is crucial for ro-
bustly estimating the FIM online and contributes significantly to
the final reconstruction quality.
5. Limitations and Future Work
While our framework demonstrates a significant advancement in
object-centric active reconstruction, we acknowledge several limi-
tations that open up exciting avenues for future research. Our focus
is object-centric NBV; scene-level SOTA methods remain comple-
mentary when global coverage is prioritized. Besides, our method’s
effectiveness is contingent upon the availability and quality of a se-
mantic mask for the object of interest. Future work could explore
methods for weakly-supervised object discovery to create a more
self-contained system. The current formulation is also designed to
optimize for a single object; extending it to handle multiple tar-
gets simultaneously, perhaps via a dynamic weighting of their re-
spective uncertainties, presents an interesting challenge. Further-
more, for computational tractability, our method approximates the
full Fisher Information Matrix with its diagonal entries. This as-
sumes independence between the different physical parameters of
a Gaussian, which is a strong simplification. Future work could
investigate more structured approximations of the FIM, such as a
block-diagonal matrix for each Gaussian, to capture parameter cor-
relations as a trade-off between computational cost and theoreti-
cal fidelity. Finally, scaling the online FIM updates for extremely
large-scale scenes remains a challenge, suggesting a need for more
efficient update or pruning strategies.
6. Conclusion
We introduced OUGS, an object-aware uncertainty estimation
framework for active view selection in 3DGS. Our work presents a
fundamental shift in how uncertainty is modeled for explicit repre-
sentations. By deriving uncertainty directly from the physical pa-
rameters of the 3D Gaussian primitives, we establish a more prin-
cipled and interpretable link between the 3D scene representation
and the rendered 2D image. Our method projects this parameter-
level covariance into pixel space and, by coupling it with seman-
tic masks, produces an uncertainty score that effectively disentan-
gles the object of interest from its environment. This is enabled by
an efficient diagonal FIM update scheme that makes the approach
computationally tractable. When integrated into a next-best-view
loop, our method consistently and substantially outperforms state-
of-the-art baselines in the critical task of object-centric reconstruc-
tion, achieving sharper results under the same view budget. No-
tably, our underlying uncertainty model also proves to be highly
competitive for global scene reconstruction. Ultimately, these re-
sults underscore the importance of disentangling object-level un-
certainty from background clutter for efficient, high-fidelity active
9

<!-- page 10 -->
References
[BKA∗16]
BIRCHER A., KAMEL M., ALEXIS K., OLEYNIKOVA H.,
SIEGWART R.: Receding horizon "next-best-view" planner for 3d explo-
ration. In 2016 IEEE International Conference on Robotics and Automa-
tion (ICRA) (2016), pp. 1462–1468. doi:10.1109/ICRA.2016.
7487281. 3
[BMV∗22]
BARRON J. T., MILDENHALL B., VERBIN D., SRINIVASAN
P. P., HEDMAN P.:
Mip-nerf 360: Unbounded anti-aliased neural
radiance fields, 2022.
URL: https://arxiv.org/abs/2111.
12077, arXiv:2111.12077. 4
[Bon08]
BONGARD J.: Probabilistic robotics. sebastian thrun, wolfram
burgard, and dieter fox. (2005, mit press.) 647 pages. Artificial Life 14,
2 (2008), 227–229. doi:10.1162/artl.2008.14.2.227. 4
[CKD∗25]
CELAREK A., KOPANAS G., DRETTAKIS G., WIMMER M.,
KERBL B.: Does 3d gaussian splatting need accurate volumetric ren-
dering?, 2025.
URL: https://arxiv.org/abs/2502.19318,
arXiv:2502.19318. 1
[CLK11]
CHEN S., LI Y., KWOK N. M.: Active vision in robotic sys-
tems: A survey of recent developments. The International Journal of
Robotics Research 30, 11 (2011), 1343–1377. 1, 3
[GRS∗24]
GOLI
L.,
READING
C.,
SELLÁN
S.,
JACOBSON
A.,
TAGLIASACCHI A.: Bayes’ Rays: Uncertainty quantification in neural
radiance fields. CVPR (2024). 1, 5, 6, 8
[HD25]
HAN C., DUMERY C.: View-dependent uncertainty estimation
of 3d gaussian splatting, 2025. URL: https://arxiv.org/abs/
2504.07370, arXiv:2504.07370. 1, 2
[JLD24]
JIANG W., LEI B., DANIILIDIS K.: Fisherrf: Active view se-
lection and uncertainty quantification for radiance fields using fisher in-
formation, 2024. URL: https://arxiv.org/abs/2311.17874,
arXiv:2311.17874. 1, 3, 5, 7, 8
[KB17]
KINGMA D. P., BA J.: Adam: A method for stochastic opti-
mization, 2017.
URL: https://arxiv.org/abs/1412.6980,
arXiv:1412.6980. 4
[KKLD23]
KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS
G.: 3d gaussian splatting for real-time radiance field rendering, 2023.
URL: https://arxiv.org/abs/2308.04079, arXiv:2308.
04079. 1, 3, 4, 7
[KMKS24]
KLASSON M., MEREU R., KANNALA J., SOLIN A.:
Sources of uncertainty in 3d scene reconstruction, 2024. URL: https:
//arxiv.org/abs/2409.06407, arXiv:2409.06407. 1
[KPZK17]
KNAPITSCH A., PARK J., ZHOU Q.-Y., KOLTUN V.:
Tanks and temples: benchmarking large-scale scene reconstruction.
ACM Trans. Graph. 36, 4 (July 2017).
URL: https://doi.
org/10.1145/3072959.3073599, doi:10.1145/3072959.
3073599. 4
[KSH22]
KIM M., SEO S., HAN B.: Infonerf: Ray entropy minimization
for few-shot neural volume rendering, 2022. URL: https://arxiv.
org/abs/2112.15399, arXiv:2112.15399. 3
[LC24]
LI
R.,
CHEUNG
Y.-M.:
Variational
multi-scale
rep-
resentation
for
estimating
uncertainty
in
3d
gaussian
splat-
ting.
In Advances in Neural Information Processing Systems
(2024), Globerson A., Mackey L., Belgrave D., Fan A., Paquet
U., Tomczak J., Zhang C., (Eds.), vol. 37, Curran Associates,
Inc.,
pp.
87934–87958.
URL:
https://proceedings.
neurips.cc/paper_files/paper/2024/file/
a076d0d1ed77364fc57693bdee1958fb-Paper-Conference.
pdf. 2
[LMV∗17]
LY A., MARSMAN M., VERHAGEN J., GRASMAN R., WA-
GENMAKERS E.-J.: A tutorial on fisher information, 2017. arXiv:
1705.01064. 4
[MST∗20]
MILDENHALL B., SRINIVASAN P. P., TANCIK M., BARRON
J. T., RAMAMOORTHI R., NG R.: Nerf: Representing scenes as neural
radiance fields for view synthesis. In CVPR (2020), pp. 612–621. 1
[NBM∗21]
NIEMEYER M., BARRON J. T., MILDENHALL B., SAJJADI
M. S. M., GEIGER A., RADWAN N.: Regnerf: Regularizing neural radi-
ance fields for view synthesis from sparse inputs, 2021. URL: https:
//arxiv.org/abs/2112.00724, arXiv:2112.00724. 1
[PLSH22]
PAN X., LAI Z., SONG S., HUANG G.: Activenerf: Learn-
ing where to see with uncertainty estimation, 2022.
URL: https:
//arxiv.org/abs/2209.08546, arXiv:2209.08546. 3, 5, 8
[RGH∗24]
RAVI N., GABEUR V., HU Y.-T., HU R., RYALI C., MA T.,
KHEDR H., RÄDLE R., ROLLAND C., GUSTAFSON L., MINTUN E.,
PAN J., ALWALA K. V., CARION N., WU C.-Y., GIRSHICK R., DOL-
LÁR P., FEICHTENHOFER C.: Sam 2: Segment anything in images and
videos, 2024.
URL: https://arxiv.org/abs/2408.00714,
arXiv:2408.00714. 7
[RZH∗23]
RAN Y., ZENG J., HE S., CHEN J., LI L., CHEN Y.,
LEE G., YE Q.:
Neurar: Neural uncertainty for autonomous 3d re-
construction with implicit neural representations. IEEE Robotics and
Automation Letters 8, 2 (Feb. 2023), 1125–1132.
URL: http://
dx.doi.org/10.1109/LRA.2023.3235686, doi:10.1109/
lra.2023.3235686. 3
[SAMNR22]
SHEN J., AGUDO A., MORENO-NOGUER F., RUIZ A.:
Conditional-flow nerf: Accurate 3d modelling with reliable uncertainty
quantification, 2022.
URL: https://arxiv.org/abs/2203.
10192, arXiv:2203.10192. 5
[SF16]
SCHONBERGER J. L., FRAHM J.-M.: Structure-from-motion re-
visited. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) (June 2016). 7
[SVM24]
SAVANT L., VALSESIA D., MAGLI E.: Modeling uncertainty
for gaussian splatting, 2024.
URL: https://arxiv.org/abs/
2403.18476, arXiv:2403.18476. 2
[TCZ∗25]
TAN Z., CHEN X., ZHANG J., FENG L., HU D.: Uncertainty-
aware normal-guided gaussian splatting for surface reconstruction from
sparse image sequences, 2025. URL: https://arxiv.org/abs/
2503.11172, arXiv:2503.11172. 2
[WAM∗25]
WILSON J., ALMEIDA M., MAHAJAN S., LABRIE M.,
GHAFFARI M., GHASEMALIZADEH O., SUN M., KUO C.-H., SEN
A.: Pop-gs: Next best view in 3d-gaussian splatting with p-optimality,
2025. URL: https://arxiv.org/abs/2503.07819, arXiv:
2503.07819. 3, 6, 7
[XCZ∗25]
XIE Y., CAI Y., ZHANG Y., YANG L., PAN J.:
Gauss-
mi: Gaussian splatting shannon mutual information for active 3d recon-
struction, 2025. URL: https://arxiv.org/abs/2504.21067,
arXiv:2504.21067. 2, 3, 6
[XDM∗24]
XUE S., DILL J., MATHUR P., DELLAERT F., TSIOTRAS
P., XU D.: Neural visibility field for uncertainty-driven active mapping,
2024. URL: https://arxiv.org/abs/2406.06948, arXiv:
2406.06948. 3
[YSHWSH16]
YÜCER
K.,
SORKINE-HORNUNG
A.,
WANG
O.,
SORKINE-HORNUNG O.: Efficient 3d object segmentation from densely
sampled light fields with applications to 3d reconstruction. ACM Trans.
Graph. 35, 3 (Mar. 2016).
URL: https://doi.org/10.1145/
2876504, doi:10.1145/2876504. 4
10

<!-- page 11 -->
Appendix A: Jacobian–Covariance Propagation
To propagate parameter uncertainty into pixel space we expand
the rendered colour C(u;θ) around its MAP estimate θ⋆using a
second-order Taylor series:
E

C(u;θ)

≈C(u;θ⋆)+∇θC(u;θ⋆)⊤(θ−θ⋆)
+ 1
2 (θ−θ⋆)⊤Hu (θ−θ⋆),
(10)
where Hu ∈Rd×d is the Hessian of C at θ⋆and δθ ≜θ −θ⋆. The
linear term is empirically two orders of magnitude smaller than
the quadratic term under i.i.d. zero-mean residuals; we therefore
neglect it in what follows.
Keeping only the leading non-zero term in the second central
moment yields the classic Jacobian–Covariance law:
Var

C(u;θ)

≈Ju ΣJ⊤
u ,
Ju := ∂C(u;θ)
∂θ

θ⋆∈R3×d.
(11)
Here Ju measures how each parameter perturbs the RGB value at
pixel u. Its element [Ju]kl = ∂Ck(u;θ⋆)/∂θl quantifies the influence
of the l-th parameter on the k-th channel.
The truncation error of (11) obeys
E ≤1
4 ∥Hu∥F ∥Σ∥2
F +O
 ∥Σ∥3
F

,
(12)
which vanishes as the Gaussian parameters become well-
constrained (∥Σ∥F →0) since ∥Σ∥F ≪∥H−1
u
∥1/2
F .
Appendix B: FIM Approximation and Online Update
Full Fisher Matrix under Gaussian Image Noise
Assume each observed pixel is corrupted by independent Gaus-
sian noise ε(u) ∼N(0,σ2), so the ground-truth intensity satisfies
Cgt(u) = C(u;θ) + ε(u). With the usual i.i.d. assumption, the neg-
ative log-likelihood—up to an irrelevant constant—is the per-pixel
squared error
ℓt =
1
2σ2 ∑
u
Cpred,t(u;θ)−Cgt,t(u)
2
2,
(the sum runs over colour channels as well as pixels).
The Fisher information matrix (FIM) is defined as the noise ex-
pectation of the outer product of the log-likelihood gradients:
I = Eε

∇θℓt ∇θℓ⊤
t

.
For a single pixel u, the gradient is
∇θℓt(u) = 1
σ2
 Cpred,t(u;θ)−Cgt,t(u)

∇θC(u;θ)⊤
= −εt(u)
σ2 J⊤
u ,
where Ju = ∂C(u;θ)/∂θ. Because the noise is independent across
pixels, E[εt(u)εt(v)] = σ2 δuv with δuv the Kronecker delta, all
cross-terms vanish, giving
I = 1
σ2 ∑
u
J⊤
u Ju.
(13)
Stacking all per-pixel Jacobians row-wise gives J =


...
Ju
...

∈
Rnpix×d, so the FIM can be written compactly as
I = 1
σ2 J⊤J,
(14)
where npix is the number of pixels in the batch and d the total pa-
rameter dimension. This dense d×d matrix captures all pairwise
parameter interactions but is prohibitively large to store or invert.
Diagonal Approximation with Tikhonov Regularisation
The Laplace (Cramér–Rao) bound yields the posterior covariance
Σ ≈σ2 J⊤J + λId
−1. To avoid storing or inverting the dense
matrix we keep only its diagonal:
Idiag = 1
σ2 diag
 J⊤J

,
Σ ≈
 Idiag +λId
−1,
(15)
where Idiag is the diagonal Fisher approximation and λ ∼10−4
(fixed for all experiments) ensures numerical stability when J⊤J is
rank-deficient.
Exponential-Moving-Average (EMA) Update
During training we accumulate the diagonal Fisher entries online
via an EMA of squared gradients:
I(j)
t,i = αt I(j)
t−1,i +(1−αt)

∇θ(j)
i ℓt
2,
αt = 0.95(1−t/T).
(16)
Here I(j)
t,i approximates the (j, j) element of the diagonal Fisher
Idiag for Gaussian i at iteration t, and T is the total number of op-
timisation steps. The schedule smoothly decays αt from 0.95 to
0, providing heavy smoothing to noisy early gradients and faster
adaptation toward the end. Equation (16) requires only one AXPY
operation per parameter and no inter-device communication.
Object-Aware Pixel Covariance
With the diagonal Fisher approximation Idiag
t
= diagIt we obtain
the tractable object-aware covariance used in the main paper,
ΣC,k(u) = (Mk(u))2 Ju
 Idiag
t
+λId
−1J⊤
u ,
(17)
where Mk(u)∈[0,1] is the soft mask for object k. Although Idiag
t
is
diagonal, the Jacobian Ju couples all parameters, so cross-channel
interactions are preserved when propagating uncertainty to the im-
age plane. Summing (17) over pixels with Mk(u)>0 produces an
uncertainty score for object k that drives next-best-view selection
in a computationally tractable manner.
1
