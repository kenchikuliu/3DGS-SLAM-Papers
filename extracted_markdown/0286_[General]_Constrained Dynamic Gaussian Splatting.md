<!-- page 1 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
1
Constrained Dynamic Gaussian Splatting
Zihan Zheng, Zhenlong Wu, Xuanxuan Wang, Houqiang Zhong, Xiaoyun Zhang, Member, IEEE,
Qiang Hu, Member, IEEE, Guangtao Zhai, Fellow, IEEE, Wenjun Zhang, Fellow, IEEE
Abstract—While Dynamic Gaussian Splatting enables high-
fidelity 4D reconstruction, its deployment is severely hindered
by a fundamental dilemma: unconstrained densification leads to
excessive memory consumption incompatible with edge devices,
whereas heuristic pruning fails to achieve optimal rendering
quality under preset Gaussian budgets. In this work, we pro-
pose Constrained Dynamic Gaussian Splatting (CDGS), a novel
framework that formulates dynamic scene reconstruction as a
budget-constrained optimization problem to enforce a strict,
user-defined Gaussian budget during training. Our key insight
is to introduce a differentiable budget controller as the core
optimization driver. Guided by a multi-modal unified importance
score, this controller fuses geometric, motion, and perceptual
cues for precise capacity regulation. To maximize the utility
of this fixed budget, we further decouple the optimization of
static and dynamic elements, employing an adaptive allocation
mechanism that dynamically distributes capacity based on motion
complexity. Furthermore, we implement a three-phase training
strategy to seamlessly integrate these constraints, ensuring pre-
cise adherence to the target count. Coupled with a dual-mode
hybrid compression scheme, CDGS not only strictly adheres to
hardware constraints (error <2%) but also pushes the Pareto
frontier of rate-distortion performance. Extensive experiments
demonstrate that CDGS delivers optimal rendering quality under
varying capacity limits, achieving over 3× compression compared
to state-of-the-art methods.
Index Terms—Dynamic Gaussian Splatting, Neural Rendering,
Resource-Constrained Rendering, Immersive Media.
I. INTRODUCTION
Free-viewpoint Video (FVV) enables users to freely and
dynamically explore 3D scenes from arbitrary viewpoints, rev-
olutionizing immersive media experiences in VR/AR, sports
broadcasting, and telepresence. While offering unprecedented
interactivity, the mass deployment of FVV is severely bot-
tlenecked by the heterogeneity of end-user hardware. The
strict memory and bandwidth limitations of edge devices stand
in sharp contrast to the massive data requirements of high-
fidelity volumetric content, making it crucial to maximize
reconstruction quality under strict resource constraints.
Early methods for FVV reconstruction typically employed
dynamic meshes [1], [2], point clouds [3]–[5], depth maps [6],
[7] or image-based view interpolation [8], frequently resulting
in compromised visual quality, especially in complex dynamic
scenes. Neural Radiance Fields (NeRF) [9] and its variants
[10]–[15] marked a breakthrough in novel view synthesis.
Subsequent work on dynamic NeRF [16]–[20] extensions in-
corporating temporal modeling further expanded FVV’s poten-
tial. However, practical deployment remains challenging due
Zihan Zheng, Zhenglong Wu, Xuanxuan Wang, Houqiang Zhong, Xi-
aoyun Zhang, Qiang Hu, Guangtao Zhai and Wenjun Zhang are with
the Shanghai Jiao Tong University, Shanghai, 200240, China. (email:
{1364406834, 1821863716, wangxuanxuan, zhonghouqiang, xiaoyun.zhang,
qiang.hu, zhaiguangtao, zhangwenjun}@sjtu.edu.cn).
to slow rendering, inconsistent output quality, and substantial
computational requirements.
Recently, 3D Gaussian Splatting (3DGS) [24] has revo-
lutionized real-time rendering. However, dynamic extensions
[21]–[23], [25] typically struggle with explosive growth in
Gaussian counts, where unconstrained densification defies
hardware limits. While recent advancements attempt to address
efficiency, they fail to simultaneously achieve controllability
and optimal allocation. For instance, Ex4DGS [23] reduces
redundancy via decomposition but lacks explicit capacity
control, relying on heuristics that yield unpredictable model
complexity. Conversely, methods like Taming 3DGS [26] in-
troduce budget control but are restricted to static scenes. More
critically, they enforce limits via rigid pruning rather than
integrating the budget constraint directly into the training loop
as a differentiable objective. Consequently, the optimization
trajectory remains unaware of the capacity limit, leading to
suboptimal convergence. Furthermore, applying such static
strategies to dynamic settings proves inadequate: relying solely
on static geometric metrics leads to an incomplete assessment
of primitive importance, as it fails to account for kinematic
significance and thus cannot effectively preserve transient,
high-frequency motion details. Our key insight is that opti-
mal constrained reconstruction requires integrating the budget
directly into the training loop while dynamically balancing
resources between static and dynamic components.
In this paper, we propose Constrained Dynamic Gaussian
Splatting (CDGS), a framework that reformulates dynamic
scene reconstruction as a budget-constrained optimization
problem, as illustrated in Fig. 1. Departing from the con-
ventional paradigm of uncontrolled growth, CDGS treats the
Gaussian count as a strict budget. By optimizing the spatio-
temporal distribution within this user-defined limit, it not
only ensures precise controllability but also pushes the Pareto
frontier of rate-distortion performance. Our approach is built
upon three key innovations. First, to enforce the target ca-
pacity, we introduce a differentiable budget controller. This
mechanism is driven by a differentiable budget loss guided by
a multi-modal unified importance score, which fuses geometric
stability, kinematic significance, and perceptual impact. This
ensures that visually critical dynamic details are preserved
even under tight budgets.
Second, to ensure robust adaptability across varying scene
dynamics, we introduce an adaptive dynamic-static allocation
method. Instead of relying on heuristic ratios, we leverage
distribution analysis to autonomously identify the natural
boundary between static and dynamic components, ensuring
that the limited Gaussian budget is invested where it con-
tributes most to the rendering quality. Third, we implement
a three-phase training strategy to seamlessly integrate these
arXiv:2602.03538v1  [cs.CV]  3 Feb 2026

<!-- page 2 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
2
Fig. 1: Left: Our CDGS leverages differentiable budget control for precise Gaussian number regulation, achieving adaptive
static-dynamic allocation across varying target numbers and optimal rendering quality. Middle: Visual comparison with state-
of-the-art methods, highlighting advantages in visual quality, model size, and Gaussian count controllability (second row:
actual/target counts). Right: Superior rate-distortion performance and precise Gaussian number control of our approach,
outperforming all prior works (e.g. 4DGS [21], STGS [22], Ex4DGS [23]).
constraints, ensuring precise adherence to the target count
(error<2%). Finally, to minimize storage footprint, a dual-
mode hybrid compression strategy is tailored specifically for
the decomposed static and dynamic streams.
Collectively, these innovations enable CDGS to deliver
controllable, compact, and high-quality FVV representations
tailored to arbitrary hardware specifications. Experimental
results demonstrate a 3× model size reduction compared to
state-of-the-art methods while maintaining comparable quality.
In summary, our contributions are as follows:
• We reinterpret dynamic Gaussian splatting as a budget-
constrained optimization problem, enabling controllable
model complexity and predictable capacity.
• We introduce a differentiable budget controller guided by
a unified importance score, together with an autonomous
adaptive static-dynamic allocation strategy that optimizes
Gaussian distribution under a fixed budget.
• We
design
a
budget-consistent
three-phase
training
scheme and a dual-mode hybrid compression pipeline that
jointly enforce strict budget adherence while minimizing
spatio-temporal redundancy.
• Extensive experiments across multiple datasets show that
CDGS consistently outperforms existing dynamic scene
reconstruction methods, achieving superior rate-distortion
performance and precise Gaussian count control.
II. RELATED WORK
A. Novel View Synthesis for Static Scenes
NeRF [9] revolutionized novel view synthesis by parameter-
izing continuous radiance fields with Multi-Layer Perceptrons,
yet its reliance on expensive ray marching imposes prohibitive
computational costs. To mitigate this, subsequent works [10]–
[15], [27]–[31] adopted structured representations such as
voxel grids, octrees, and hash tables to accelerate the query
process. While these hybrid approaches significantly improve
training and rendering efficiency, they fundamentally remain
bound by the ray-marching paradigm, where the requisite
point-wise sampling along each ray continues to limit the
optimal balance between high-frequency detail capture and
computational throughput.
Marking a paradigm shift from implicit ray marching, 3DGS
[24] and its variants [32]–[41] leverage explicit anisotropic
Gaussians and tile-based GPU rasterization to achieve supe-
rior visual quality in real-time. However, this explicit nature
introduces new challenges: the adaptive density control strat-
egy allows the number of primitives to grow unboundedly,
resulting in high and variable memory usage that often exceeds
consumer-grade hardware capacities. This uncontrolled model
complexity and unstable convergence pattern severely hinder
the deployment of 3DGS under strict computational or mem-
ory budgets, necessitating more robust control mechanisms.
B. Novel View Synthesis for Dynamic Scenes
Extending view synthesis to dynamic environments requires
addressing complex motion and temporal redundancy. Early
approaches adapted the NeRF framework, either by intro-
ducing deformation fields [16]–[20], [42]–[45] to map time-
variant observations to a canonical space, or by adopting 4D
spatio-temporal representations [46]–[54] like 4D grids. While
deformation-based methods struggle with large topological
changes, 4D spatio-temporal representations often incur sig-
nificant memory costs and inherit the slow rendering speeds
of implicit methods.
With the advent of 3DGS, research has shifted towards
explicit dynamic modeling, generally falling into two streams.
The first stream [21]–[23], [25], [55]–[62] utilizes unified 4D
primitives or time-dependent attributes for continuous inter-
polation, while the second [63]–[68] employs frame-by-frame
tracking or streaming updates to handle topological changes.
Notably, methods like Ex4DGS [23] and DeGauss [69] further
decompose scenes into static and dynamic components. De-
spite these advances, current dynamic 3DGS methods typically

<!-- page 3 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
3
Fig. 2: Overview of the proposed CDGS framework. (Top) Three-Phase Pipeline: The training progresses from a Warm-up
phase to establish foundational priors, through a Budget Enforcement phase where constraints are actively applied, to a final
Fine-tuning phase that maximizes quality under the fixed count Ntarget. (Bottom Left) Adaptive Dynamic-Static Allocation: This
module analyzes the distribution of motion magnitudes to identify a natural separation threshold τmotion via peak identification. It
decomposes the scene into a Static set S and a Dynamic set D(t) . (Bottom Right) Differentiable Budget Control: This module
regulates capacity by computing a Unified Importance Score Mi. It fuses Perceptual cues Fperceptual and Geometric/Motion cues
Fgeom/motion. The resulting score guides the differentiable budget loss Lbudget to precisely prune or densify Gaussians towards
Ntarget.
rely on heuristic-driven densification, allowing the Gaussian
count to grow arbitrarily. This unconstrained model complexity
results in millions of redundant primitives, burdening storage
and rendering efficiency without proportional visual gains.
C. Gaussian Number Management
Deploying 3DGS on resource-constrained devices faces a
fundamental conflict between limited hardware capacities and
the standard optimization strategy, which minimizes error
via unbounded densification. To alleviate storage pressure,
recent compression-oriented methods [37], [70]–[72] propose
techniques such as vector quantization or importance-based
pruning. For instance, LightGaussian [37] prunes redundant
Gaussians via significance scores, while C3DGS [71] utilizes
codebook-based compression. However, these approaches pre-
dominantly operate in a train-then-prune or post-processing
manner. They lack the mechanism to precisely align the
model size with a predefined hardware budget during training.
Consequently, removing primitives from a converged model
often leads to suboptimal solutions and unpredictable quality
degradation, as the remaining Gaussians are not jointly opti-
mized to compensate for the information loss.
The most relevant work to ours is Taming 3DGS [26],
which introduces a controlled growth schedule to maintain
the Gaussian count near a target level. While effective for
static objects, Taming 3DGS relies solely on geometric density
and fails to address the complexities of dynamic scenarios,
where temporal redundancy and motion blurring require more
sophisticated importance assessment. To bridge this gap, we
formulate dynamic reconstruction as a budget-constrained op-
timization problem. Distinct from passive compression, our
method integrates the target count directly into the training
objective. By employing an adaptive static-dynamic allocation
strategy and a differentiable budget controller, we ensure the
model actively seeks the optimal configuration within the strict
Gaussian budget, maximizing rendering fidelity while strictly
adhering to the specified capacity limits.
III. METHOD
We propose CDGS, a unified framework designed for high-
fidelity dynamic 3D scene reconstruction under explicit re-
source constraints. Unlike prior methods prone to heuristic
pruning or uncontrolled Gaussian growth, CDGS reformulates
the reconstruction task as a budget-constrained optimization
problem, enabling precise control over model capacity. As
illustrated in Fig. 2, given multi-view video inputs and a
target Gaussian count Ntarget, our pipeline is driven by a
Differentiable Budget Controller (Sec. III-B). This module
supervises densification and pruning based on a unified impor-
tance score that integrates geometric, motion, and perceptual
cues. To maximize the utility of the fixed budget, an Adaptive
Static-Dynamic Allocation module (Sec. III-C) redistributes
the Gaussian budget across static and dynamic regions. The
overall training follows a three-phase pipeline (Sec. III-D) to
stabilize the representation, and employs a dual-mode hybrid
compression scheme (Sec. III-E) to minimize storage and
transmission overhead for efficient deployment.
A. Problem Formulation
Given calibrated multi-view dynamic frames {Iv,t}, where v
indexes views V and t indexes time, our goal is to reconstruct
a compact spatio-temporal Gaussian scene representation G
that enables real-time rendering under a fixed capacity budget

<!-- page 4 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
4
Ntarget. Following 3D Gaussian Splatting [24], the scene is
modeled as a set of anisotropic Gaussians G = {gi}:
gi = {µi, Ri, si, αi, fi},
(1)
where µi denotes the 3D center position. The geometric shape
of each Gaussian is determined by a 3D covariance matrix
Σi, which is decomposed into a rotation matrix Ri and a
scaling matrix si to ensure positive semi-definiteness during
optimization:
Σi = RisisT
i RT
i .
(2)
In terms of appearance, each Gaussian carries an opacity
scalar αi ∈[0, 1] and a set of spherical harmonics (SH)
coefficients fi which encode view-dependent radiance. To
render novel views, the 3D Gaussians are projected onto the
2D image plane using splatting techniques. The final color ˆIv,t
for a specific view v at time t is computed via differentiable
α-blending of the sorted Gaussians overlapping a pixel:
ˆIv,t =
X
i∈|G|
coloriα′
i
i−1
Y
j=1
(1 −α′
j),
(3)
where colori is the color decoded from fi based on the
viewing direction, and α′
i represents the effective opacity in
2D projection space. To drive the optimization of the Gaus-
sian parameters, we minimize the discrepancy between the
rendered image ˆIv,t and the corresponding ground truth Iv,t.
We supervise the reconstruction with a perceptual appearance
loss:
Lrender = (1−λssim)∥ˆIv,t−Iv,t∥1+λssim LSSIM(ˆIv,t, Iv,t), (4)
Unlike prior dynamic Gaussian approaches that freely grow
and prune Gaussians post-training, we explicitly constrain the
representational capacity during optimization:
min
G
Lrender(G)
s.t.
|G| ≤Ntarget.
(5)
This constraint directly governs runtime memory, rendering
cost, and even streaming bitrate. Our objective is therefore not
the unconstrained best reconstruction, but the best reconstruc-
tion achievable under a fixed Gaussian-number budget, which
forms the foundation for our differentiable population control
and adaptive allocation strategies described next.
B. Differentiable Budget Control
Enforcing the hard constraint |G| ≤Ntarget in Eq. 5 is
challenging, since the Gaussian count is discrete and non-
differentiable. We therefore design a differentiable population
controller, as illustrated in Fig. 3, that (1) guides the contribu-
tion of Gaussians in rendering, (2) ranks them by importance
for adaptive densification and pruning, and (3) penalizes
deviations from the target capacity through a differentiable
budget loss.
Differentiable Counting. Each Gaussian gi is assigned a
continuous activation variable ci ∈[0, 1], implemented via
a temperature-controlled hard-sigmoid gate with a learnable
Gaussian importance score Mi. During rendering, ci directly
regulates the participation of the Gaussian, so Gaussians with
ci ≈0 contribute negligibly. The effective active count is
estimated as a differentiable proxy:
Np =
X
i
ci.
(6)
To match the target capacity, we introduce a quadratic budget
loss:
Lbudget = (Np −Ntarget)2,
(7)
which drives Np toward Ntarget. To enable differentiable op-
timization, the binary existence of each Gaussian is relaxed
into a continuous activation ci ∈[0, 1]. This is achieved via a
temperature-controlled Hard-Sigmoid function applied to the
Gaussian’s importance score Mi:
ci = clamp
Mi −0.5
τc
+ 0.5, 0, 1

,
(8)
where τc is the temperature parameter controlling the steepness
of the gate, and Mi is the unified importance score detailed
in the following part.
To physically enforce this selection within the rendering
pipeline, we modulate the opacity of each Gaussian using
this activation variable. The effective opacity ˆαi used in the
splatting equation (Eq. 3) is redefined as:
ˆαi = ci · αi.
(9)
This modulation serves as a differentiable bridge: when
ci →0, the Gaussian becomes transparent and ceases to
contribute to the image, physically aligning the soft deletion
with the visual output. Crucially, this ensures that gradients
from the rendering loss Lrender (Eq. 4) are back-propagated to
the importance score Mi via ci, establishing a cooperative op-
timization where perceptually critical Gaussians are protected
from pruning.
The gate temperature is gradually annealed so that ci
approaches a binary mask. Specifically, the temperature τc
decays exponentially from an initial value τinit to a final value
τend during the budget enforcement phase. The decay schedule
is formulated as:
τc(k) = τinit ·
τend
τinit

k−kstart
kend−kstart
,
(10)
where k denotes the current iteration, and [kstart, kend] rep-
resents the duration of the enforcement phase. In our ex-
periments, we set τinit = 1.0 and τend = 0.01 to ensure a
smooth transition from soft gating to binary selection.Upon
convergence, thresholding ci yields an explicit active set with
size ≈Ntarget. This converts the discrete constraint into a dif-
ferentiable, Lagrangian-style penalty that is fully compatible
with gradient-based optimization.
Unified Importance Score. To determine which Gaussians
should survive under a strictly constrained budget, we require
a metric that evaluates the contribution of each primitive to
the final reconstruction. While previous pruning strategies
like Taming 3DGS [26] focus solely on static geometric
attributes, dynamic scenes introduce temporal redundancies
and kinematic complexities that static metrics fail to capture.
To address this, we define a unified importance score Mi

<!-- page 5 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
5
Fig. 3: Illustration of the Differentiable Budget Controller. The
controller aggregates geometric, motion, and perceptual cues
to compute a Unified Importance Score Mi. This score passes
through a hard-sigmoid gate to estimate the effective Gaussian
count Np, which is strictly regulated towards the target Ntarget
via a quadratic budget loss Lbudget. Guided by this budget
constraint, the closed-loop policy performs densification on
high-importance Gaussians and pruning on low-importance
ones to dynamically optimize capacity.
that fuses geometric stability, kinematic significance, and
perceptual impact:
Mi = N (λgm · Fgeom/motion(gi) + Fperceptual(gi)) ,
(11)
where N(·) denotes Min-Max normalization scaling values to
[0, 1], ensuring a balanced aggregation of multi-modal cues.
Geometric and Motion Cues (Fgeom/motion). This module
captures the structural and kinematic necessity of a Gaussian.
To handle the complexity of dynamic scenes, we explicitly
decompose the score into five key components:
Fgeom/motion(gi) = wT
1 · N
h
∇µ
i , αmax
i
, d−1
i ,
λmax(Σi), Moi
i
,
∇µ
i = ∥∇µLi∥2,
αmax
i
= max
t∈Ti αi(t),
d−1
i
=
1
|V |
X
v
d−1
i,v ,
Moi =
(
∥Ti∥2,
if gi ∈S
∥f µ
i ∥2 + ∥f R
i ∥2,
if gi ∈D(t)
(12)
Here, w1 serves as a weighting vector to balance the
contribution of each term, where each component targets a
specific geometric or kinematic property critical for high-
fidelity reconstruction. Specifically, the Positional Gradient ∇µ
i
quantifies the sensitivity of the reconstruction loss with respect
to the Gaussian’s position, effectively identifying primitives
located in structure-critical regions where spatial precision
is paramount. To capture temporal transients, Peak Opacity
αmax
i
utilizes the maximum opacity over the temporal sequence
rather than the average, ensuring that fleeting structures are
preserved rather than pruned due to low average visibility.
We further incorporate Proximity (d−1
i ) via the inverse mean
depth to explicitly prioritize foreground elements, as they
occupy larger screen areas and demand higher rendering
fidelity compared to distant background objects. To maintain
background continuity, Spatial Extent λmax(Σi), derived from
the largest eigenvalue of the covariance matrix, safeguards
large-scale Gaussians that cover extensive regions, prevent-
ing the formation of geometric holes. Finally, the Motion
Magnitude Moi employs a piece-wise formulation adapted to
our decomposition: for static Gaussians , it retains primitives
necessary for correcting global translation, while for dynamic
Gaussians, it prioritizes those exhibiting large displacements
and rotations to protect high-frequency motion details that are
otherwise difficult to recover.
Perceptual Cues (Fperceptual). To ensure visual fidelity, this
component quantifies the rendering degradation caused by
removing a Gaussian, aggregated across training views V :
Fperceptual(gi) = wT
2 · N
h
Ii, Areai, Var−1
i
i
,
Ii =
X
v∈V
∥Iv −I\{i}
v
∥1,
Areai =
X
v∈V
Areai,v,
Var−1
i
= Varv∈V (ci,v)−1.
(13)
Here, w2 denotes the weighting vector that balances the
contribution of each perceptual metric. Specifically, the Pho-
tometric Residual Ii directly quantifies the rendering degra-
dation by aggregating the pixel-wise L1 error introduced to
the rendered image when the Gaussian gi is excluded. To
evaluate visual impact, Pixel Coverage Areai accumulates the
projected screen area of the Gaussian across views, where
larger footprints imply a greater contribution to the final
appearance. Finally, Multi-view Consistency Var−1
i
utilizes the
inverse variance of the Gaussian’s contribution across views
to prioritize view-consistent primitives, effectively acting as a
filter to suppress view-dependent noise or artifacts that would
otherwise be visually distracting.
Densify-prune Policy Under Budget. Guided by the im-
portance score Mi and regulated by a predefined budget, our
method performs densification and pruning in a unified frame-
work. The global target Ntarget is dynamically decomposed into
a per-iteration sub-target via a quadratic schedule, as proposed
in [26], throughout the training process. This decomposition
guides a closed-loop controller for population evolution: (i)
densification: Gaussians with high Mi are cloned or split
with higher probability, ensuring computational resources are
allocated to perceptually salient regions by injecting detail
where needed; (ii) pruning: Gaussians with low Mi are tar-
geted for removal via a random sampling strategy, gradually
eliminating redundant or visually insignificant ones while
preserving perceptually critical structures.
This closed-loop controller dynamically reallocates the
Gaussian budget to the most informative spatial and temporal

<!-- page 6 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
6
regions while keeping the overall count near Ntarget. Unlike
heuristic grow-and-prune pipelines [23], [25], our differen-
tiable control embeds capacity regulation directly into opti-
mization, enabling stable training and precise runtime control
over model complexity.
C. Adaptive Dynamic-Static Allocation
Dynamic scenes are rarely uniformly active: while the
majority of regions often remain largely static or undergo
rigid, regular motion, significant non-rigid deformations are
typically confined to specific objects within limited temporal
spans. Consequently, uniformly distributing Gaussians across
space and time results in a substantial waste of representational
capacity. To address this and fully utilize the fixed Gaussian
budget Ntarget, we introduce an Adaptive Static-Dynamic Al-
location strategy designed to maximize efficiency under strict
constraints.
This mechanism decomposes the scene into a static field
and a dynamic field, parameterizing their temporal behaviors
separately. We represent the full Gaussian set G as the union
of two disjoint subsets,
G = S ∪D(t),
S ∩D(t) = ∅.
(14)
Here, S comprises time-invariant Gaussians responsible for
the static objects, while D(t) contains time-varying Gaussians
dedicated to handling motion and deformation. Instead of
relying on pre-labeled masks, heuristic priors, or fixed static-
dynamic ratios, we propose to infer this decomposition auto-
matically in a fully data-driven manner.
Our core insight is to frame the separation task as iden-
tifying a natural boundary within the distribution of motion
magnitudes. We first assign each Gaussian gi a translational
attribute Ti to characterize its motion intensity. After an initial
warm-up phase, we analyze the intrinsic distribution properties
of {Ti} to determine the optimal separation. Formally, we
construct a distribution histogram of motion magnitudes using
B bins and apply smoothing to suppress noise. Let HT denote
the smoothed histogram and PT be the set of identified peaks:
HT = smooth(hist(Ti, B)),
PT = {p ∈B | HT (p) > HT (p ± 1)}.
(15)
This smoothed histogram typically exhibits a bimodal distri-
bution, corresponding to the distinct populations of static and
dynamic elements. From the set of peaks PT , we identify the
two most significant modes, denoted as ps and pd, representing
the centroids of the static and dynamic clusters, respectively.
To achieve a clean separation, we locate the valley between
these two peaks, which signifies the boundary where the
probability density of mixed states is lowest:
{ps, pd} =
argmax
{p1,p2}⊆PT ,p1̸=p2
(HT (p1) + HT (p2)) ,
τs = argmin
x∈{ps,pd}
HT (x).
(16)
Here, τs represents the initial separation threshold in the
histogram domain. To ensure robustness against binning ar-
tifacts and outliers, we map this threshold back to the data
domain by calculating an adaptive percentile αT :
αT = |Ti | Ti < τs|
|Ti|
,
τmotion = quantile({Ti}, αT ).
(17)
The resulting τmotion serves as the final, adaptive threshold.
This formulation enables our method to automatically adapt to
varying scene dynamics, providing an optimal separation that
emerges naturally from the data distribution without manual
parameter tuning.
Based on this classification, we apply distinct parameter-
ization strategies to ensure model compactness. The static
Gaussians S maintain time-invariant attributes (position, ro-
tation, scale, color) throughout the sequence, incorporating
lightweight, per-frame global transforms Ti to accommodate
minor camera misalignments or lighting changes without re-
dundant per-Gaussian updates. Conversely, the dynamic Gaus-
sians D(t) are equipped with explicit temporal parameters to
capture complex motions. Each gi ∈D(t) carries a position
vector f µ
i
and a rotation vector f R
i
(interpolated at time t
to determine the instantaneous state), alongside a scale si,
spherical harmonics coefficients fi, and a learnable activation
window [ts
i, te
i]. To enforce temporal sparsity, the opacity αi(t)
is designed to decay smoothly outside this interval, ensur-
ing that a dynamic Gaussian only consumes computational
resources when it effectively contributes to the rendering.
Ultimately,
this
derived
static-dynamic
decomposition
serves as a critical prior for our resource management. We
allocate the Gaussian budget proportionally: regions identified
as D(t) or frames exhibiting higher dynamic activity receive
a denser representation budget, while static regions S retain
compact but stable coverage. This decomposition directly
guides the Differentiable Controller (Sec. III-B), instructing it
on where Gaussian capacity should be invested across space
and time. Together, this joint mechanism effectively balances
reconstruction fidelity and storage efficiency, achieving high-
quality dynamic rendering under strict model capacity con-
straints.
D. Training Strategy
Training our constrained dynamic Gaussian model involves
jointly optimizing scene appearance, differentiable budget
control, and adaptive allocation in a stable and progressive
manner. We adopt a three-phase training strategy designed to
(i) obtain a reliable initialization, (ii) introduce differentiable
population control, and (iii) stabilize optimization while en-
forcing the capacity constraint.
Phase I: Warm-up and Initialization. We begin with a
short warm-up stage that initializes the representation without
budget constraints. We utilize static Gaussians equipped with
per-frame global transforms, allowing the model to roughly
cover the scene geometry and rigid motion using only the
rendering loss Lrender (Eq. 4). These pre-trained Gaussians
serve as a foundational prior, providing reliable geometric and
kinematic guidance for the subsequent modules.
Phase II: Differentiable Budget Enforcement. After
warm-up, we activate the population controller (Sec. III-B)

<!-- page 7 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
7
Fig. 4: Illustration of our dual-mode hybrid compression
strategy, which separates outliers from both static and dynamic
Gaussian data, then applies distinct compression approaches.
and introduce the budget loss Lbudget and regularization Lreg.
The total loss is formulated as:
L = Lrender + λbLbudget + λrLreg
(18)
where λb and λr control the strength of the budget penalty and
regularization, respectively. Crucially, the Adaptive Dynamic-
Static Allocation (Sec. III-C) is performed periodically along-
side the densification and pruning operations. At each interval,
the budget is dynamically redistributed between static and
dynamic sets based on the updated motion priors. Simulta-
neously, discrete pruning is triggered for Gaussians with low
opacity or near-zero mask values (ci ≈0), while the gate
temperature is annealed to gradually enforce a binary selection.
Phase III: Stabilization and Fine-tuning. Once the ef-
fective count Np converges near Ntarget, we enter a stabiliza-
tion phase. The soft masks ci are explicitly binarized, and
redundant Gaussians (where ci = 0) are permanently culled
to strictly satisfy the budget constraint. The remaining compact
set is then fine-tuned solely with Lrender and Lreg to recover
any detail loss caused by the masking process and to ensure
temporal smoothness.
E. Dual-Mode Hybrid Compression
To further reduce storage and transmission cost under the
fixed Gaussian budget, we adopt a dual-mode hybrid compres-
sion strategy tailored to the static and dynamic components
introduced in Sec. III-C.
Static Compression. For static Gaussians, we observe that
outliers far from the scene center significantly degrade the
post-compression quality. This is because these distant points
excessively expand the coordinate bounding box. Under uni-
form quantization, an expanded range forces a larger quantiza-
tion step size, which drastically reduces the precision allocated
to the densely populated foreground regions. To mitigate this,
we first compute the mean µd and standard deviation σd
of all Gaussian distances to the scene center and separate
background Gaussians using a threshold of µd + 3σd. For the
remaining foreground, after attribute quantization, we employ
a KD-tree to spatially reorder points, ensuring local continuity.
Leveraging these spatial neighborhoods, we apply predictive
encoding to residuals to reduce entropy. Finally, geometric data
and attribute residuals undergo entropy encoding to complete
the compression.
Dynamic Compression. For dynamic Gaussians, we first
conduct a statistical analysis of the data distribution across
channels, observing that: (1) distributions within the same
attribute are highly similar across channels; and (2) each
attribute is generally concentrated but contains significant
outliers. Based on these findings, we design a dedicated
compression pipeline. First, we separate top 5% of outlier
data to narrow the value range and improve concentration.
Next, we reshape each quantized attribute of the dynamic
Gaussians into a 2D image format and rearrange them by
grouping identical attributes together. These organized video
sequences are then encoded using an H.264 encoder (based
on the x264 library). To ensure high fidelity and temporal
consistency, the encoder is configured with the YUV 4:4:4
color space, the “medium” preset, and a constant Quantization
Parameter of 20. Furthermore, we restrict the encoding to I-
and P-frames (disabling B-frames) with 3 reference frames to
optimize for decoding efficiency.
IV. EXPERIMENTS
A. Configurations
Datasets. We evaluate our method on three challenging real-
world datasets representing different dynamic characteristics.
First, the N3DV dataset [50] provides 6 scenes with com-
plex non-rigid deformations, captured by 18-21 cameras at
2704 × 2028 resolution and 30 FPS. Second, the Technicolor
dataset [74] offers studio-quality scenes recorded by a 16-
camera rig at 2048×1088, featuring rich textures and lighting
effects. Third, the MeetRoom dataset [75] captures long-
duration human activities in an indoor setting using 14 cameras
at 1280 × 720 and 30 FPS. Consistent with prior works [23],
[64], we employ a standard leave-one-out strategy. For the
N3DV and Meetroom datasets, the first camera view is held
out for testing. In contrast, for the Technicolor dataset, we
reserve the camera positioned at the intersection of the second
row and second column for evaluation. All remaining views
serve as the training set.
Implementation. The experiments were conducted on hard-
ware with an Intel(R) Xeon(R) W-2245 CPU @ 3.90 GHz
and an RTX 3090 graphics card. For each sequence, our
three-phase training strategy consists of 500, 29500, and
10000 iterations, respectively, with Gaussian densification and
pruning performed every 500 steps during the second phase.
λssim = 0.2, λb = 1 × 10−7, λr = 1 × 10−4 and λgm = 2
were set for all sequences. In the compression phase, 16-bit
quantization was used for µ, T and fµ, while 8-bit quantization
was applied to all other attributes.
Metrics. To assess the modeling and control capabilities
of our method across experimental datasets, we adopt Peak
Signal-to-Noise Ratio (PSNR) and Structural Similarity Index
(SSIM) [76] as quality metrics, along with model size mea-
sured in megabytes for the entire sequence. For comprehensive
rate-distortion performance analysis, we apply Bjontegaard
Delta PSNR (BD-PSNR) [77]. Rendering efficiency is gauged
by frames per second (FPS). To validate our method’s capacity

<!-- page 8 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
8
Fig. 5: Rate-distortion curves on different datasets, illustrating the superiority of our method over ReRF [17], TeTriRF [73],
4DGC [65], 4DGCPro [66], RD4DGS [60] and Ex4DGS [23].
TABLE I: Quantitative comparison on the N3DV [50] dataset.
The PSNR, SSIM, and rendering speed are averaged over all
300 frames for each scene. The reported model size is the total
for the entire sequence. Ours-l and Ours-s are obtained by
setting the target Gaussian numbers to 300,000 and 100,000,
respectively. Best and second best results are highlighted.
Method
PSNR↑
(dB)
SSIM↑
Size↓
(MB)
Render↑
(FPS)
Controll-
able
K-Planes [48]
29.91
0.920
300
0.15
✕
ReRF [17]
29.71
0.918
231
2.0
✕
TeTriRF [73]
30.65
0.931
227
2.7
✕
StreamRF [75]
30.61
0.930
2280
8.3
✕
3DGStream [64]
31.54
0.942
2430
215
✕
4DGC [65]
31.58
0.943
150
168
✕
4DGCPro [66]
31.64
0.943
192
170
✕
4DGaussians [25]
31.15
0.939
34
147
✕
4DGS [21]
32.01
0.944
6270
72
✕
STGS [22]
32.05
0.944
200
107
✕
RD4DGS [60]
29.66
0.917
11.1
100.9
✕
Swift4D [62]
31.79
0.944
30
128
✕
GIFStream [61]
31.75
0.938
10
95
✕
Ex4DGS [23]
32.11
0.945
115
128
✕
Ours-l
32.14
0.946
31.5
149
✓
Ours-s
31.83
0.944
6.8
186
✓
TABLE II: Validation of our precise control capability over the
total number of Gaussians. We report the average PSNR on the
coffee martini sequence, the number of static/dynamic/overall
Gaussians, and the error ratio of Gaussian number.
Method
Target
PSNR
Static
Dynamic
Overall
Ratio
Ex4DGS [23]
-
28.79
292.2k
47.7k
339.9k
-
Ours
100k
28.53
72.4k
27.6k
99.9k
0.1%
200k
28.68
156.8k
41.0k
197.8k
1.1%
300k
28.81
244.4k
51.4k
295.8k
1.4%
400k
28.95
316.4k
81.2k
397.6k
0.6%
for Gaussian count control, we further report the number of
Gaussians and the error ratio relative to the target.
B. Comparison
Quantitative Comparisons. To validate the effectiveness
of our method, we compare it against several state-of-the-
art approaches, in particular those based on 3DGS, including
TABLE III: Quantitative comparison on the MeetRoom dataset
[75] and Technicolor dataset [74].
Dataset
Method
PSNR↑
(dB)
SSIM↑
Size↓
(MB)
Render↑
(FPS)
MeetRoom
Dataset [75]
ReRF [17]
26.43
0.911
189
2.9
TeTriRF [73]
27.37
0.917
183
3.8
StreamRF [75]
26.71
0.913
2469
10
3DGStream [64]
28.03
0.921
2430
288
4DGC [65]
28.08
0.922
126
213
4DGCPro [66]
28.02
0.921
123
222
STGS [22]
29.01
0.929
15.2
159
Ex4DGS [23]
29.12
0.930
40.2
148
Ours-l
29.18
0.931
10.4
165
Ours-s
28.60
0.927
4.5
215
Technicolor [74]
4DGC [65]
31.56
0.939
310
139
4DGCPro [66]
31.53
0.939
453
133
STGS [22]
31.96
0.941
58.7
113
RD4DGS [60]
31.33
0.938
33.6
123
Ex4DGS [23]
32.38
0.942
170.3
100
Ours-l
32.41
0.942
48.0
125
Ours-s
31.90
0.940
17.1
144
3DGStream [64], 4DGC [65], 4DGCPro [66], 4DGaussians
[25], 4DGS [21], STGS [22], RD4DGS [60], Swift4D [62],
GIFStream [61] and Ex4DGS [23]. We present two variants of
our method, Ours-l and Ours-s, obtained by setting different
target Gaussian numbers to demonstrate its scalability. Tab. I
shows the detailed quantitative results on the N3DV dataset.
As observed from the table, our method achieves the optimal
rate-distortion performance. Specifically, 4DGS, STGS and
Ex4DGS achieve reconstruction quality comparable to that of
our method, but they typically require hundreds to thousands
of MB in model size to enable dynamic scene reconstruction,
whereas ours only requires 31.5 MB. 4DGaussians, RD4DGS,
Swift4D and GIFStream achieve a model size similar to that
of our method, but results in varying degrees of PSNR degra-
dation ranging from 0.08 dB to 2.17 dB. More importantly,
we can achieve precise control over the total number of
Gaussians. As shown in Tab. II, under different target total
numbers of Gaussians, we maintain the error rate within 2%
and simultaneously achieve the optimal allocation of static and
dynamic Gaussians.
We further validate the generality of our method on the

<!-- page 9 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
9
Fig. 6: Qualitative comparison of our CDGS against STGS [22] and Ex4DGS [23] on the N3DV [50], MeetRoom [75] and
Technicolor [74] datasets, demonstrating the performance of CDGS under different target Gaussian numbers.
MeetRoom [75] and Technicolor [74] datasets. As summarized
in Tab. III, our approach consistently outperforms a wide range
of methods in PSNR, SSIM, and model size, corroborating its
robustness across diverse scenes. What’s more, we evaluate
the rate-distortion performance against contemporary works,
including ReRF [17], TeTriRF [73], 4DGC [65], 4DGCPro
[66], and RD4DGS [60]. As visualized in the RD curves of
Fig. 5, our approach consistently pushes the Pareto frontier,
outperforming variable-bitrate baselines across the spectrum.
This superiority is also quantitatively corroborated by the BD-
PSNR results in Tab. IV. Our method demonstrates the highest
gains, with improvements over 4DGC of 1.90 dB and 1.72 dB
on the N3DV and MeetRoom datasets, which is a substantial
margin above all other comparisons.
Tab. V presents a comparison of computational efficiency
between our CDGS and several dynamic scene reconstruction
and compression methods. Our CDGS exhibits significantly
improved computational efficiency: its training time is 1.0
hour, compared to 1.2 hours for Ex4DGS [23], owing to
the increased interval of Gaussian expansion. Additionally,
experimental results on the N3DV dataset demonstrate that
our approach enables high-speed rendering, achieving a rate of
5.4 ms per frame. For encoding and decoding, CDGS achieves
times of 16 s and 0.5 s for 300 frames in total, respectively,
outperforming all other methods by a significant margin. These
results fully demonstrate that CDGS exhibits comprehensively
fast performance across reconstruction, rendering, and encod-
ing/decoding, confirming it as a highly efficient solution for
TABLE IV: The BD-PSNR(dB) results of our CDGS, ReRF
[17], TeTriRF [73], 4DGCPro [66] and RD4DGS [60] when
compared with 4DGC [65] on different datasets.
Dataset
ReRF [17] TeTriRF [73] 4DGCPro [66] RD4DGS [60] Ours
N3DV [50]
-1.99
-1.12
0.08
1.33
1.90
MeetRoom [75]
-1.84
-0.86
-0.02
-
1.72
TABLE V: Complexity comparison of our method with dy-
namic scene reconstruction and compression methods.
Time
ReRF [17] TeTriRF [73] 4DGC [65] Ex4DGS [23] Ours
Encode(s)
246
219
810
-
16
Decode(s)
18.3
16.8
28.2
-
0.5
Train(h)
>100
5.2
4.2
1.2
1.0
Render(ms)
497
372
5.6
7.8
5.4
constrained dynamic Gaussian splatting.
Qualitative Comparisons. As shown in Fig. 6, we qual-
itatively compare our method at different Gaussian num-
ber targets against STGS [22] and Ex4DGS [23] on the
flame salmon sequence, the trimming sequence, and the birth-
day sequence. Our method delivers reconstruction quality
comparable to these approaches while operating at a signif-
icantly lower model size. Compared to STGS, CDGS better
preserves fine-grained details, such as the head, window, and
desk in flame salmon, the fast-moving hands and intricate
plants in trimming, as well as balloons, fans, and candles
in birthday. Furthermore, our approach enables precise Gaus-

<!-- page 10 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
10
Fig. 7: Evolution of Gaussian counts under varying target budgets. The subplots correspond to target limits of 1 × 105 (Left),
2 × 105 (Middle), and 3 × 105 (Right). The curves illustrate the growth trajectories of Static, Dynamic, and Total Gaussians.
Under our constrained optimization framework, the total number steadily increases and converges precisely to the predefined
Target Number.
Fig. 8: Visualization of densification strategies at Iter. 7k.
Unlike the gradient-based baseline (c) which generates exces-
sive redundant primitives in the background, our importance-
driven method (d) effectively suppresses irrelevant growth,
concentrating Gaussians strictly on geometric and motion
boundaries.
sian count control, consistently yielding optimal performance
across varying total Gaussian numbers. For instance, in our
lightweight variant (ours-s), reconstructing the flame salmon
sequence requires only 10k Gaussians, which is far fewer than
the 35k used by Ex4DGS. This underscores that CDGS not
only accurately captures dynamic scene content and retains
high-fidelity details in complex objects but also achieves a
highly compact model size, all while realizing precise regula-
tion of the Gaussian count.
C. Evaluations
Gaussian Number Constraint. We conduct an ablation
study to quantify the contribution of our proposed importance
score and budget loss to Gaussian count control. As presented
in Tab. VI, the unified importance score, which evaluates each
Gaussian’s actual rendering contribution, is more effective
than relying on traditional training gradients. This superiority
is visually explicated in Fig. 8. As shown in Fig. 8 (c),
TABLE VI: Evaluation results of our differentiable budget
control and adaptive dynamic-static allocation.
PSNR(dB)↑
Size(MB)↓
Ratio↓
w/o Importance score
31.91
31.2
1.6%
w/o Fgeom/motion
31.97
31.5
1.3%
w/o Fperceptual
31.99
31.3
1.4%
λgm = 1.5
32.08
31.4
1.3%
λgm = 2.5
32.11
31.5
1.4%
w/o Budget loss
32.15
31.7
4.8%
w/o Adaptive allocation
31.96
33.2
1.3%
Ours full
32.14
31.5
1.3%
the gradient-based baseline struggles to distinguish between
structural details and background noise, leading to blind den-
sification in empty regions. Conversely, our method (Fig. 8
(d)) acts as a precise filter, concentrating the Gaussian budget
solely on perceptually salient object boundaries. Ablating the
entire importance score causes a significant performance drop
of 0.23 dB. The individual components of the score are also
validated: removing the geometric/motion term (Fgeom/motion)
leads to a 0.17 dB loss, while omitting the perceptual term
(Fperceptual) results in a 0.15 dB degradation. We further
conduct a sensitivity analysis on the weighting coefficient
λgm. Our experiments indicate that deviating from the default
setting leads to varying degrees of quality degradation. This
confirms that a balanced trade-off between geometric/motion
stability and perceptual fidelity is essential for maximizing the
utility of the limited Gaussian budget. Furthermore, the budget
loss is proven critical for regulating the score distribution
during training, ensuring the Gaussian count aligns with the
target. Without it, the Gaussian count error increases by 3.5%.
This precise controllability is visually corroborated in Fig. 7.
As observed, the total number of Gaussians monotonically
increases and converges exactly to the preset limits without
overshooting, demonstrating the stability and strictness of our
budget enforcement strategy.
Dynamic-Static Allocation. Our adaptive allocation ac-
curately determines the optimal static-dynamic ratio. As il-
lustrated in Fig. 9, this approach achieves a more rational

<!-- page 11 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
11
Fig. 9: Visualization of our adaptive dynamic-static allocation
strategy (bottom) against pre-defined ratio approach (top).
segmentation: it robustly assigns static elements like walls,
stools, and cabinets to the static model, while accurately
capturing motion regions such as hands and moving dogs. This
optimized allocation directly contributes to superior overall
rendering quality. The necessity of this module is quantita-
tively confirmed in Tab. VI, whose removal leads to a -0.18
dB drop in PSNR and a slight model size increase of 1.7 MB,
underscoring its role in both compactness and fidelity.
Three-phase Training Strategy. We remove the first and
third training stages, respectively, to isolate their contributions.
As summarized in Tab. VII, the initialization stage provides
a reliable Gaussian translation prior, which is critical for an
accurate static-dynamic allocation. Ablating this stage leads
to a sub-optimal allocation that hinders effective training,
resulting in a performance drop of 1.24 dB. Conversely, the
fine-tuning stage further optimizes the Gaussian distributions
and attributes after the target count is reached, leading to a
notable refinement in reconstruction quality. Its removal causes
a drop of 0.28 dB, confirming its role in enhancing the final
rendering fidelity.
Compression Strategy. To minimize data redundancy, we
introduce a subsequent compression pipeline tailored for both
static and dynamic Gaussians. As summarized in Tab. VII, this
approach reduces the model size by 34.0 MB for static and
32.8 MB for dynamic Gaussians, with a marginal impact on
rendering quality. Fig. 10 provides a comparative visualization
of the storage distribution before and after compression. By
contrasting the two states, it is evident that our dual-mode
strategy not only significantly reduces the overall model size
but also effectively restructures the data. Specifically, the ’Af-
ter’ distribution highlights the successful isolation of ’Back-
ground’ static Gaussians and ’Outlier’ dynamic data, ensuring
that the majority of the storage budget is precisely allocated to
the perceptually critical foreground and motion details, thereby
validating our separation strategy. The regularization loss is so
critical for refining the data distribution that its removal leads
to a significant 0.75 dB drop. Furthermore, we identify that
separating outliers in both static and dynamic Gaussians is a
Fig. 10: Comparison of storage distribution before (left) and
after (right) applying our dual-mode hybrid compression on
the flame salmon sequence. The visualization highlights the
significant reduction in model size and the explicit isolation
of background and outlier components in the compressed
representation.
TABLE VII: Evaluation results of our three-phase training
strategy and dual-mode hybrid compression.
PSNR(dB)↑
Size(MB)↓
w/o Initialization
30.90
31.8
w/o Fine-tuning
31.86
31.5
w/o Compression
32.21
98.3
w/o Static compression
32.19
65.5
w/o Dynamic compression
32.16
64.3
w/o Regularization loss
31.39
31.2
w/o Outlier separation
29.74
26.2
Ours full
32.14
31.5
critical prerequisite. Neglecting this step severely hampers the
compression process, resulting in a substantial 2.4 dB quality
degradation.
V. CONCLUSION
In this paper, we presented Constrained Dynamic Gaussian
Splatting (CDGS), a framework that redefines dynamic scene
reconstruction from an unconstrained fitting task to a budget-
constrained optimization problem. Departing from heuristic
train-then-prune paradigms, CDGS demonstrates that integrat-
ing hardware constraints directly into the training loop is
essential for achieving optimal fidelity. Through our differen-
tiable budget controller and adaptive static-dynamic allocation,
underpinned by a progressive three-phase training strategy, we
successfully resolved the complex spatio-temporal resource
competition. Furthermore, coupled with a dual-mode hybrid
compression scheme, CDGS ensures that limited capacity is
intelligently invested in visually critical motion details while
minimizing storage footprint. Extensive experiments validate
that CDGS achieves superior rate-distortion performance, of-
fering a robust and practical solution for deploying high-
fidelity 4D immersive media on resource-constrained edge
devices.
REFERENCES
[1] Y. Wang, W. Wang, J. Ling, R. Xie, and L. Song, “Visibility-aware
human mesh recovery via balancing dense correspondence and proba-

<!-- page 12 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
12
bility model,” in 2024 IEEE International Conference on Multimedia
and Expo Workshops (ICMEW), 2024, pp. 1–6.
[2] Y. Xia, X. Zhou, E. Vouga, Q. Huang, and G. Pavlakos, “Reconstructing
humans with a biomechanically accurate skeleton,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2025.
[3] M. Yang, Z. Luo, M. Hu, M. Chen, and D. Wu, “A comparative
measurement study of point cloud-based volumetric video codecs,” IEEE
Transactions on Broadcasting, vol. 69, no. 3, pp. 715–726, 2023.
[4] A. Akhtar, Z. Li, G. Van der Auwera, and J. Chen, “Dynamic point cloud
interpolation,” in ICASSP 2022 - 2022 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 2574–
2578.
[5] D. Graziosi, O. Nakagami, S. Kuma, A. Zaghetto, T. Suzuki, and
A. Tabatabai, “An overview of ongoing point cloud compression stan-
dardization activities: Video-based (v-pcc) and geometry-based (g-pcc),”
APSIPA Transactions on Signal and Information Processing, vol. 9, p.
e13, 2020.
[6] S. Guo, J. Hu, K. Zhou, J. Wang, L. Song, R. Xie, and W. Zhang,
“Real-time free viewpoint video synthesis system based on dibr and a
depth estimation network,” IEEE Transactions on Multimedia, vol. 26,
pp. 6701–6716, 2024.
[7] J. M. Boyce, R. Dor´e, A. Dziembowski, J. Fleureau, J. Jung, B. Kroon,
B. Salahieh, V. K. M. Vadakital, and L. Yu, “Mpeg immersive video
coding standard,” Proceedings of the IEEE, vol. 109, no. 9, pp. 1521–
1536, 2021.
[8] Q. Hu, Q. He, H. Zhong, G. Lu, X. Zhang, G. Zhai, and Y. Wang, “Var-
fvv: View-adaptive real-time interactive free-view video streaming with
edge computing,” IEEE Journal on Selected Areas in Communications,
pp. 1–1, 2025.
[9] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[10] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in Proceedings of the IEEE/CVF inter-
national conference on computer vision, 2021, pp. 5855–5864.
[11] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[12] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in European conference on computer vision.
Springer, 2022,
pp. 333–350.
[13] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[14] D. Duckworth, P. Hedman, C. Reiser, P. Zhizhin, J.-F. Thibert, M. Luˇci´c,
R. Szeliski, and J. T. Barron, “Smerf: Streamable memory efficient
radiance fields for real-time large-scene exploration,” ACM Transactions
on Graphics (TOG), vol. 43, no. 4, pp. 1–13, 2024.
[15] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Zip-nerf: Anti-aliased grid-based neural radiance fields,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 19 697–19 705.
[16] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields
for 3d video synthesis,” Advances in Neural Information Processing
Systems, vol. 35, pp. 13 485–13 498, 2022.
[17] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and
M. Wu, “Neural residual radiance fields for streamably free-viewpoint
videos,” in CVPR, June 2023, pp. 76–87.
[18] L. Wang, K. Yao, C. Guo, Z. Zhang, Q. Hu, J. Yu, L. Xu, and
M. Wu, “Videorf: Rendering dynamic radiance fields as 2d feature video
streams,” 2023.
[19] Z. Zheng, H. Zhong, Q. Hu, X. Zhang, L. Song, Y. Zhang, and Y. Wang,
“Jointrf: End-to-end joint optimization for dynamic neural radiance field
representation and compression,” in 2024 IEEE International Conference
on Image Processing (ICIP), 2024, pp. 3292–3298.
[20] ——, “Hpc: Hierarchical progressive coding framework for volumetric
video,” in ACM MM, ser. MM ’24.
New York, NY, USA: Association
for Computing Machinery, 2024, p. 7937–7946.
[21] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic
dynamic scene representation and rendering with 4d gaussian splatting,”
in ICLR, 2024.
[22] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting
for real-time dynamic view synthesis,” in CVPR, June 2024, pp. 8508–
8520.
[23] J. Lee, C. Won, H. Jung, I. Bae, and H.-G. Jeon, “Fully explicit dynamic
guassian splatting,” in Proceedings of the Neural Information Processing
Systems, 2024.
[24] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[25] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in CVPR, June 2024, pp. 20 310–20 320.
[26] S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco, and
F. De La Torre, “Taming 3dgs: High-quality radiance fields with limited
resources,” in SIGGRAPH Asia 2024 Conference Papers, ser. SA ’24.
New York, NY, USA: Association for Computing Machinery, 2024.
[27] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing
and improving neural radiance fields,” arXiv preprint arXiv:2010.07492,
2020.
[28] C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Super-
fast convergence for radiance fields reconstruction,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5459–5469.
[29] A. Karnewar, T. Ritschel, O. Wang, and N. Mitra, “Relu fields: The
little non-linearity that could,” in ACM SIGGRAPH 2022 conference
proceedings, 2022, pp. 1–9.
[30] M. Niemeyer, F. Manhardt, M.-J. Rakotosaona, M. Oechsle, D. Duck-
worth, R. Gosula, K. Tateno, J. Bates, D. Kaeser, and F. Tombari,
“Radsplat: Radiance field-informed gaussian splatting for robust real-
time rendering with 900+ fps,” in 2025 International Conference on 3D
Vision (3DV).
IEEE, 2025, pp. 134–144.
[31] X. Zhang, S. Bi, K. Sunkavalli, H. Su, and Z. Xu, “Nerfusion: Fusing
radiance fields for large-scale scene reconstruction,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 5449–5458.
[32] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat:
3d gaussian splats from image pairs for scalable generalizable 3d re-
construction,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2024, pp. 19 457–19 467.
[33] G. Feng, S. Chen, R. Fu, Z. Liao, Y. Wang, T. Liu, B. Hu, L. Xu, Z. Pei,
H. Li et al., “Flashgs: Efficient 3d gaussian splatting for large-scale and
high-resolution rendering,” in Proceedings of the Computer Vision and
Pattern Recognition Conference, 2025, pp. 26 652–26 662.
[34] Z. Gao, B. Planche, M. Zheng, A. Choudhuri, T. Chen, and Z. Wu,
“6dgs: Enhanced direction-aware gaussian splatting for volumetric ren-
dering,” arXiv preprint arXiv:2410.04974, 2024.
[35] L. H¨ollein, A. Boˇziˇc, M. Zollh¨ofer, and M. Nießner, “3dgs-lm:
Faster gaussian-splatting optimization with levenberg-marquardt,” arXiv
preprint arXiv:2409.12892, 2024.
[36] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash,
“Compgs: Smaller and faster gaussian splatting with vector quantiza-
tion,” ECCV, 2024.
[37] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
2023.
[38] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-
gs: Structured 3d gaussians for view-adaptive rendering,” in CVPR,
2024, pp. 20 654–20 664.
[39] Y. Wang, Z. Li, L. Guo, W. Yang, A. C. Kot, and B. Wen, “Contextgs:
Compact 3d gaussian splatting with anchor level context model,” arXiv
preprint arXiv:2405.20721, 2024.
[40] H. Zhong, Z. Zheng, Q. Hu, Y. Tian, N. Cao, L. Xu, X. Zhang, Z. Cheng,
L. Song, and W. Zhang, “4d-mode: Towards editable and scalable
volumetric streaming via motion-decoupled 4d gaussian compression,”
2025.
[41] Y. Gao, H. Zhong, T. Zhu, Z. Cheng, Q. Hu, and L. Song, “Aligngs:
Aligning geometry and semantics for robust indoor reconstruction from
sparse views,” 2025.
[42] Y. Du, Y. Zhang, H.-X. Yu, J. B. Tenenbaum, and J. Wu, “Neural
radiance flow for 4d view synthesis and video processing,” in 2021
IEEE/CVF International Conference on Computer Vision (ICCV). IEEE
Computer Society, 2021, pp. 14 304–14 314.
[43] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz,
and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2021, pp. 5865–5874.

<!-- page 13 -->
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX
13
[44] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-
nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2021,
pp. 10 318–10 327.
[45] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and
A. Geiger, “Nerfplayer: A streamable dynamic scene representation with
decomposed neural radiance fields,” IEEE Transactions on Visualization
and Computer Graphics, vol. 29, no. 5, pp. 2732–2742, 2023.
[46] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic
scenes,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 130–141.
[47] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and
Q. Tian, “Fast dynamic radiance fields with time-aware neural voxels,”
in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.
[48] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa,
“K-planes: Explicit radiance fields in space, time, and appearance,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 12 479–12 488.
[49] M. Is¸ık, M. R¨unz, M. Georgopoulos, T. Khakhulin, J. Starck, L. Agapito,
and M. Nießner, “Humanrf: High-fidelity neural radiance fields for
humans in motion,” ACM Transactions on Graphics (TOG), vol. 42,
no. 4, pp. 1–12, 2023.
[50] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim,
T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al., “Neural 3d
video synthesis from multi-view video,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp. 5521–
5531.
[51] S. Park, M. Son, S. Jang, Y. C. Ahn, J.-Y. Kim, and N. Kang, “Temporal
interpolation is all you need for dynamic neural radiance fields,” in
Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2023, pp. 4212–4221.
[52] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, “Tensor4d:
Efficient neural 4d decomposition for high-fidelity dynamic reconstruc-
tion and rendering,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2023, pp. 16 632–16 642.
[53] F. Wang, S. Tan, X. Li, Z. Tian, Y. Song, and H. Liu, “Mixed neural
voxels for fast multi-view video synthesis,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2023, pp.
19 706–19 716.
[54] L. Wang, J. Zhang, X. Liu, F. Zhao, Y. Zhang, Y. Zhang, M. Wu, J. Yu,
and L. Xu, “Fourier plenoctrees for dynamic radiance field rendering in
real-time,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2022, pp. 13 524–13 534.
[55] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,”
arXiv preprint arXiv:2309.13101, 2023.
[56] J. Yan, R. Peng, L. Tang, and R. Wang, “4d gaussian splatting with scale-
aware residual field and adaptive optimization for real-time rendering of
temporally complex dynamic scenes,” in ACM MM, 2024, pp. 7871–
7880.
[57] Y. Wang, P. Yang, Z. Xu, J. Sun, Z. Zhang, Y. Chen, H. Bao, S. Peng,
and X. Zhou, “Freetimegs: Free gaussian primitives at anytime anywhere
for dynamic scene reconstruction,” in CVPR, 2025.
[58] X. Zhang, Z. Liu, Y. Zhang, X. Ge, D. He, T. Xu, Y. Wang, Z. Lin,
S. Yan, and J. Zhang, “Mega: Memory-efficient 4d gaussian splatting
for dynamic scenes,” arXiv preprint arXiv:2410.13613, 2024.
[59] W. O. Cho, I. Cho, S. Kim, J. Bae, Y. Uh, and S. J. Kim, “4d scaffold
gaussian splatting with dynamic-aware anchor growing for efficient and
high-fidelity dynamic scene reconstruction,” 2025.
[60] H. Lee and K. Baek, “Temporal smoothness-aware rate-distortion op-
timized 4d gaussian splatting,” in Advances in Neural Information
Processing Systems, 2025.
[61] H. Li, S. Li, X. Gao, A. Batuer, L. Yu, and Y. Liao, “Gifstream: 4d
gaussian-based immersive video with feature stream,” 2025.
[62] J. Wu, R. Peng, Z. Wang, L. Xiao, L. Tang, J. Yan, K. Xiong, and
R. Wang, “Swift4d: Adaptive divide-and-conquer gaussian splatting for
compact and efficient reconstruction of dynamic scene,” arXiv preprint
arXiv:2503.12307, 2025.
[63] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d
gaussians: Tracking by persistent dynamic view synthesis,” in 3DV,
2024.
[64] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in CVPR, June 2024, pp. 20 675–20 685.
[65] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai,
and Y. Wang, “4dgc: Rate-aware 4d gaussian compression for efficient
streamable free-viewpoint video,” in Proceedings of the Computer Vision
and Pattern Recognition Conference, 2025, pp. 875–885.
[66] Z. Zheng, Z. Wu, H. Zhong, Y. Tian, N. Cao, L. Xu, J. Yao, X. Zhang,
Q. Hu, and W. Zhang, “4dgcpro: Efficient hierarchical 4d gaussian
compression for progressive volumetric video streaming,” arXiv preprint
arXiv:2509.17513, 2025.
[67] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, “Hicom: Hierarchical
coherent motion for dynamic streamable scenes with 3d gaussian splat-
ting,” in Advances in Neural Information Processing Systems (NeurIPS),
2024.
[68] H. Zhong, Z. Wu, S. Fu, Z. Zheng, X. Jin, X. Zhang, L. Song, and Q. Hu,
“Prismgs: Physically-grounded anti-aliasing for high-fidelity large-scale
3d gaussian splatting,” 2025.
[69] R. Wang, Q. Lohmeyer, M. Meboldt, and S. Tang, “Degauss: Dynamic-
static decomposition with gaussian splatting for distractor-free 3d recon-
struction,” 2025.
[70] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated
3d gaussians with lightweight encodings,” 2024.
[71] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2024,
pp. 21 719–21 728.
[72] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed
3d gaussian splatting for accelerated novel view synthesis,” in 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024, pp. 10 349–10 358.
[73] M. Wu, Z. Wang, G. Kouros, and T. Tuytelaars, “Tetrirf: Temporal tri-
plane radiance fields for efficient free-viewpoint video,” in CVPR, June
2024, pp. 6487–6496.
[74] B. Attal, J.-B. Huang, C. Richardt, M. Zollhoefer, J. Kopf, M. O’Toole,
and C. Kim, “HyperReel: High-fidelity 6-DoF video with ray-
conditioned sampling,” in CVPR, 2023.
[75] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields
for 3d video synthesis,” Advances in Neural Information Processing
Systems, vol. 35, pp. 13 485–13 498, 2022.
[76] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assess-
ment: from error visibility to structural similarity,” IEEE Transactions
on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[77] B. S. Pateux and J. Jung, “An excel add-in for computing bjontegaard
metric and its evolution,” in vceg meeting,” 2007.
