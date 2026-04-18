<!-- page 1 -->
DIAL-GS: Dynamic Instance Aware Reconstruction for Label-free Street
Scenes with 4D Gaussian Splatting
Chenpeng Su∗, Wenhua Wu∗, Chensheng Peng, Tianchen Deng, Zhe Liu, Hesheng Wang†
∗Equal contribution.
†Corresponding author
Abstract— Urban scene reconstruction is critical for au-
tonomous driving, enabling structured 3D representations for
data synthesis and closed-loop testing. Supervised approaches
rely on costly human annotations and lack scalability, while
current self-supervised methods often confuse static and dynamic
elements and fail to distinguish individual dynamic objects, limit-
ing fine-grained editing. We propose DIAL-GS, a novel dynamic
instance-aware reconstruction method for label-free street scenes
with 4D Gaussian Splatting. We first accurately identify dy-
namic instances by exploiting appearance–position inconsistency
between warped rendering and actual observation. Guided
by instance-level dynamic perception, we employ instance-
aware 4D Gaussians as the unified volumetric representation,
realizing dynamic-adaptive and instance-aware reconstruction.
Furthermore, we introduce a reciprocal mechanism through
which identity and dynamics reinforce each other, enhancing
both integrity and consistency. Experiments on urban driving
scenarios show that DIAL-GS surpasses existing self-supervised
baselines in reconstruction quality and instance-level editing,
offering a concise yet powerful solution for urban scene modeling.
I. INTRODUCTION
Urban scene reconstruction has become a cornerstone tech-
nology in autonomous driving. By generating structured 3D
representations of complex urban environments, it provides
foundations for large-scale data synthesis, supporting both
algorithm development and closed-loop testing in safety-
critical scenarios [1], [2].
To address the challenges of road scenes, many existing
methods adopt supervised learning, which relies on labor-
intensive manual annotations to accurately capture the spatial
and semantic information of dynamic objects [3]–[11]. How-
ever, manual labeling is costly, and supervised models are
inherently limited to the scope of annotated datasets, hindering
its scalability. These drawbacks have motivated increasing
interest in self-supervised reconstruction [12]–[15].
Without explicit supervisory signals, self-supervised meth-
ods are prone to dynamic-static confusion: static objects
may be incorrectly modeled as dynamic due to data noise,
while slowly moving objects may be mistakenly treated
as static. To address this issue, DIAL-GS introduces an
inconsistency-driven approach for precise instance-level dy-
namic perception. Specifically, when dynamic objects are
forced to be represented by static Gaussians, the resulting
static field merely records their instantaneous states in the past,
which inevitably lag behind the current observations. This
discrepancy manifests as the inconsistency between rendering
and ground truth in both appearance and position, which
DIAL-GS leverages as a reliable cue to distinguish dynamic
instances from the static background.
Fig. 1.
Motivation. DIAL-GS overcomes the limits of supervised and self-
supervised methods with label-free, dynamic-adaptive and instance-aware
reconstruction.
Another challenge in self-supervised reconstruction lies
in constructing a unified representation for the entire scene.
Supervised paradigms often sidestep this difficulty by apply-
ing static Gaussians [16] to static background and employing
time-varying Gaussians for dynamic objects [1], [3], [5].
In contrast, self-supervised frameworks cannot pre-identify
elements as static or dynamic. This necessitates a represen-
tation that is simultaneously capable of preserving invariant
spatial attributes of static background while modeling the
spatiotemporal variations of dynamic objects.
Beyond unified representation, a further difficulty of self-
supervised reconstruction lies in enabling scene editing, which
is fundamental for data generation. Unfortunately, current
self-supervised approaches lack instance awareness, reducing
the edition to coarse static–dynamic decomposition. Without
the ability to distinguish between individual dynamic objects,
these methods cannot support per-instance modeling or fine-
grained editing, which severely limits their applicability.
DIAL-GS adopts instance-aware 4DGS to handle these lim-
itations. By jointly encoding identity and dynamic attributes,
DIAL-GS provides a unified framework where both static and
dynamic scene components are consistently modeled and each
Gaussian primitive is enriched with ID features. Furthermore,
we propose a reciprocal ID–dynamics training strategy. We
identify Gaussians belonging to the same instance via ID
embeddings and enforce their dynamics consistency, while
dynamic attributes are leveraged to select existing Gaussians
and cluster their ID embeddings. In this way, the integrity of
instance awareness and the consistency of dynamic modeling
are jointly enhanced.
arXiv:2511.06632v1  [cs.CV]  10 Nov 2025

<!-- page 2 -->
With these mechanisms, DIAL-GS realizes a dynamic-
adaptive and instance-aware 4D reconstruction within the self-
supervised regime. Our main contributions are summarized
as follows:
1) We introduce an intuitive and accurate instance-level
dynamic perception algorithm by exploiting the appear-
ance and position inconsistency caused by motion.
2) We empower self-supervised reconstruction with in-
stance awareness by proposing instance-aware 4DGS,
and introduce a reciprocal mechanism in which instance
awareness and dynamics mutually benefit.
3) Extensive experiments demonstrate that DIAL-GS
surpasses prior methods in image reconstruction and
novel view synthesis, while uniquely enabling instance-
level editing – a capability absent from existing self-
supervised approaches.
II. RELATED WORK
A. NeRF-based Driving Reconstruction
Numerous methods have investigated NeRF-based ap-
proaches for road-scene reconstruction. NSG [11] models
dynamic multi-object scenes with a scene graph, enabling
instance-level view synthesis and 3D detection. Block-
NeRF [17] adopts block-wise representations with semantic
masking and appearance codes to reconstruct large-scale
scenes, while READ [18] introduces a real-time rendering en-
gine with ω-net for photorealistic scene synthesis and editing.
SUDS [19] factorizes scenes into static, dynamic, and far-field
radiance fields with hash-grid acceleration, supporting scene
flow estimation and semantic manipulation. EmerNeRF [20]
advances self-supervised modeling by estimating flow-based
correspondences and leveraging 2D foundation features for
geometry and semantics.
Despite these advances, NeRF-based methods remain con-
strained by heavy computation, slow training and rendering,
and limited scalability to large dynamic environments. Their
reliance on dense sampling and implicit volumetric MLPs
further hinders real-time applications and fine-grained edit-
ing [21]. By contrast, Gaussian Splatting achieves comparable
visual quality with significantly faster performance and
naturally accommodates instance-aware extensions.
B. GS-based Driving Reconstruction
3D Gaussian Splatting (3DGS) [16] has recently emerged
as an efficient alternative to NeRFs, replacing implicit MLP-
based volumetric fields with explicit Gaussian primitives.
By rasterizing Gaussians, 3DGS enables fast rendering and,
thanks to its explicit structure, naturally extends beyond view
synthesis to tasks such as dynamic scene reconstruction, ge-
ometry editing, and physical simulation [22]. These properties
make 3DGS particularly well-suited for large-scale driving-
scene reconstruction, where efficiency is essential.
Supervised methods rely on labor-annotated datasets to
guide geometry, semantics, and dynamics, achieving highly
accurate and structured reconstructions. DrivingGaussian [4]
combines incremental static reconstruction with a dynamic
Gaussian graph for large-scale driving scenes. Street Gaus-
sians [3] leverages a 4D spherical harmonics appearance
model, tracked pose optimization, and point cloud initializa-
tion to improve rendering quality. AutoSplat [6] introduces
geometry-constrained background modeling, template-based
foreground initialization, and temporally adaptive appear-
ance modeling. OmniRe [5] builds a holistic scene graph
that unifies static backgrounds, vehicles, SMPL-modeled
humans [23], and other non-rigid actors. While effective,
supervised approaches are labor-intensive, expensive to scale,
and inherently constrained by the distribution of annotated
datasets, limiting their generalization to unseen scenarios.
Self-supervised approaches remove the need for annotations
by exploiting temporal consistency, geometric cues, and multi-
view signals in driving data. S3 Gaussians [24] decomposes
scenes with a spatio-temporal network that models Gaussian
deformation through feature planes. PVG [13] introduces 4D
Gaussians with periodic vibration and learnable lifespans,
avoiding Hexplane-based deformation [25]. DeSiRe-GS [12]
enhances separation with a motion-mask extraction mecha-
nism, 3D regularization, and temporal cross-view consistency.
Despite progress, these methods face two major limitations: (i)
dynamic–static confusion, where noise, pose perturbations, or
slow motion lead to misclassification of static versus dynamic
objects, and (ii) lack of instance awareness, as they only
coarsely separate static and dynamic components without
distinguishing or editing individual objects.
Our work directly addresses both issues by enabling
accurate instance-level dynamic perception and introducing
instance-aware 4D Gaussians.
C. Semantic Scene Modeling with Gaussians
Previous semantic scene modeling approaches are usually
NeRF-based [26]–[30]. However, with the rise of 3D Gaussian
Splatting (GS), increasing efforts focus on injecting 2D
semantic knowledge into 3D-GS. Gaussian Grouping [31]
transfers SAM’s segmentation capability [32] [33] into 3D,
enabling zero-shot segmentation without 3D mask annota-
tions. To resolve multi-granularity ambiguity, SAGA [34]
introduces a promptable 3D segmentation framework with a
scale-gated mechanism and contrastive distillation. Semantic
Gaussians [35] further projects diverse pre-trained 2D features
into 3D Gaussians and enriches them with semantic attributes.
Building on these approaches, a natural direction for self-
supervised road-scene reconstruction is to achieve instance
awareness by embedding trajectory-tracked instance IDs into
dynamic Gaussians. However, existing works are mostly
tailored for static or small-scale scenes, limiting their ap-
plicability to highly dynamic driving environments. To bridge
this gap, DIAL-GS integrates ID-embedding with dynamic
attributes and introduces a reciprocal training strategy that
allow both to enhance each other.
III. METHOD
We address the problem of road-scene reconstruction under
a self-supervised setting. Given a sequence of temporally
aligned multi-view observations—including RGB images,

<!-- page 3 -->
Fig. 2.
Method overview. (i) Stage 1 conducts instance-level dynamic perception with static GS by exploiting inconsistency between warped renderings and
ground-truth frames. Accumulated dynamic scores quantify inconsistency are used to obtain a dynamic ID list, according to which ID labels and dynamic
masks are derived. (ii) Stage 2 reconstructs the scene with instance-aware 4DGS as the unified representation. Guided by the ID labels and dynamic masks,
it achieves instance awareness and refines dynamic attributes. Then it performs reciprocal training to enhance both instance awareness integrity and dynamics
consistency. (iii) With instance awareness, DIAL-GS further enables instance-level editing, a capability not supported by previous self-supervised approaches.
LiDAR point clouds, camera intrinsics, and ego-poses—our
objective is to recover a scene representation that captures
both static structures and dynamic objects without any 3D
annotation.
As shown in Fig. 2, we design a two-stage pipeline. In
Stage 1, we first reconstruct an over-filtered static scene with
all frames and build dynamic candidate Gaussians for each
frame. By combining them, we establish an instantaneous
Gaussian field, where dynamic objects lag behind ground-truth
observations and exhibit inconsistency in both appearance
and position when rendered from new views. We quantify
the inconsistency as a dynamic score for each instance and
select dynamic IDs based on the aggregated scores. With the
dynamic ID list, we then derive ID labels and dynamic masks
from tracking sequence. In Stage 2, the scene is subsequently
reconstructed with instance-aware 4D Gaussians as a unified
representation. Guided by the labels and masks from Stage 1,
we realize instance awareness and clean dynamic attributes,
and further propose the reciprocal identity-dynamics training
to reinforce integrity of ID-embedding and consistency of
dynamic attributes. With these designs, DIAL-GS enables self-
supervised reconstruction to support instance-level editing.
A. Instance-level Dynamic Perception
When represented with static GS, only instantaneous states
of dynamic objects can be captured. As time progresses,
the previously recorded state of a dynamic object inevitably
diverges from the actual observation, much like how a frozen
image of a moving object always differs from its true current
state. Building on this intuition, we propose Instance-level
Dynamic Perception driven by inconsistency.
Instantaneous Gaussian Field Establishment. We start by
extracting the tracking sequence and semantic masks using
BoT-SORT [36]. The semantic masks divide each frame into
a filtered region (mainly static content, though some static
objects may be excluded) and a candidate region (where
potential dynamic objects remain). Using all filtered RGB
frames together with all filtered LiDAR point clouds, we
reconstruct an over-filtered static scene Sover with static GS,
which provides a temporally consistent background.
At time t, the point cloud is back-projected to recover
candidates’ spatial structure and, with RGB supervision,
dynamic candidate Gaussians Ct is obtained. Combined with
the static field Sover, they form the instantaneous Gaussian
field, which is then warped to frames t + 1, . . . , t + k to
simulate new-view observations:
ˆIt,t+k = Ft→t+k(Ct, Sover),
(1)
where F stands for the wrap process.
Dynamic Score and Instance Selection. To measure the
inconsistency of each candidate, we perform instance seg-
mentation on ˆIt,t+k and establish ID correspondences with
the ground-truth frame It+k. The appearance inconsistency

<!-- page 4 -->
of instance i is defined as:
Iapp
i,t
=
It+k −ˆIt,t+k
 ⊙
(Mi,t | ˆ
Mi,t+k)
(Mi,t | ˆ
Mi,t+k)

,
(2)
where Mi,t denotes the mask of instance i at frame t, and
(· | ·) represents the union of two masks.
Meanwhile, the position inconsistency is defined as
Ipos
i,t = |ci,t −ˆci,t+k|
p
Ai,t
+ |Ai,t −ˆAi,t+k|
Ai,t
+ σ(Ei,t,t+k)
p
Ai,t
,
(3)
where c denotes the bounding box center, A its area, Ei,t,t+k
the edge difference, and σ the standard deviation.
By combining these two inconsistency measures, we define
the dynamic score of instance i at frame t as Si,t =
Iapp
i,t
+ Ipos
i,t . We compute Si,t from the first frame and
accumulate them to obtain the final score: Si = 1
n
P
t Si,t,
where n denotes the total number of frames in which instance
i appears. To enhance the separation between dynamic and
static instances, we apply a cubic amplification and classify
an instance as dynamic if S3
i > δ. Finally, a dynamic ID list
D = {i|S3
i > δ} is obtained in stage one.
B. Self-supervised Reconstruction with instance-aware 4DGS
While existing self-supervised reconstruction methods have
shown promise in capturing geometry and motion, they
largely overlook instance awareness. Without distinguishing
which Gaussian belongs to which object, these approaches
often produce entangled representations that hinder reliable
decomposition and fine-grained editing. This limitation mo-
tivates us to introduce instance-aware 4DGS as the unified
representation.
Instance-aware 4D Gaussian. To consistently model both
static and dynamic scene components and realize instance
awareness, we embed ID vectors to PVG [13]. Thereby, the
Gaussian of stage two is formulated with position µ, color
c, opacity o, rotation q, scale s, ID-embedding e as static
attributes and velocity v, life-peak τ, life-span β as dynamic
attributes. The Gaussian vibrates around µ and fades away
according to τ and β:
˜µ(t) = µ + l
2π · sin
2π(t −τ)
l

· v,
(4)
˜o(t) = o · exp(−1
2(t −τ)2β−2).
(5)
Identity Loss. We formulate the ID-embedding e as a static
8-bit vector inspired by [31] and render it with the original
splatting pipeline.
E =
X
i∈N
eiαi
i−1
Y
j=1
(1 −αj),
(6)
where N is the total number of depth-sorted Gaussians, and
ei and αi denote the ID embedding and density of the i-th
Gaussian, respectively.
Fig. 3.
Dynamic Mask Comparison. DeSiRe-GS misclassifies static region
and incompletely capture dynamic parts. DIAL-GS obtains accurate and
sharp dynamic masks instead.
We then take a simple MLP l followed by a softmax as
the classifier: ˆIid = argmax(softmax(l(E))), where ˆIid
stands for ID rendering. By ignoring instances not in D, ID
label Iid can be derived from tracking sequence. The identity
loss is then defined as the pixel-wise cross-entropy between
the predicted ID renderings and the ID labels:
Lid = −1
P
P
X
p=1
C
X
c=1
Iid
p,c log

ˆIid
p,c

,
(7)
where P = h · w is pixel number of the rendering, C is the
max number of dynamic IDs, and ˆIid
p,c ∈[0, 1] is the predicted
probability that pixel p belongs to class c and Iid
p,c ∈{0, 1}
is the one-hot ground-truth indicator.
Dynamic Attributes Regularization. PVG [13] tends to
assign dynamics to static region inconsistent across time due
to data noise, and fails to capture the dynamics of slowly
moving objects. DeSiRe-GS [12] attempts to alleviate this
issue by introducing a motion mask, yet it still inherits similar
confusion since the mask itself is also learned in a fully
self-supervised manner. In contrast, DIAL-GS leverages 2D
trackers to directly generate instance mask and derive dynamic
mask M by selecting IDs in D. We then regulate dynamics
with precise and instance-level mask:
L¯v =
1
| ¯
M|I ¯v ⊙¯
M,
(8)
Lβ = −1
| ¯
M|Iβ ⊙¯
M,
(9)
where ¯v = v · exp(−β
2l) represents the instant velocity, and
I ¯v, Iβ represent the rendering of ¯v and β respectively.
C. Reciprocal Identity-Dynamics Training
Another limitation of existing self-supervised methods is
the inconsistency of dynamic attributes among Gaussians be-
longing to the same object. For example, different Gaussians
of a single car may exhibit significantly different instant
velocities or life-spans. Moreover, relying solely on Lid
often results in incomplete instance awareness, as it only
provides 2D supervision. To address this, DIAL-GS introduces
a reciprocal training scheme, enabling both more complete
ID-embedding and more coherent dynamic attributes.
3D Identity Loss. Due to the mechanism of PVG [13],
Gaussians fade out by decreasing opacity while retaining
their static attributes such as position. Therefore, clustering
all ID embeddings would incur prohibitive computation, so
we leverage dynamic attributes to filter Gaussians that actually

<!-- page 5 -->
TABLE I
COMPARISON ON WAYMO OPEN DATASET. FPS DENOTES RENDERING SPEED. ↑MEANS HIGHER IS BETTER, ↓MEANS LOWER IS BETTER.
Method
FPS
Image reconstruction
Novel view synthesis
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
S-NeRF [37]
0.0014
19.67
0.528
0.387
19.22
0.515
0.400
StreetSurf [38]
0.097
26.70
0.846
0.372
23.78
0.822
0.401
3DGS [16]
63
27.99
0.866
0.293
25.08
0.822
0.319
NSG [11]
0.032
24.08
0.656
0.441
21.01
0.571
0.487
Mars [39]
0.030
21.81
0.681
0.430
20.69
0.636
0.453
SUDS [19]
0.008
28.83
0.805
0.369
22.63
0.593
0.402
EmerNeRF [20]
0.053
28.11
0.786
0.373
25.92
0.763
0.384
PVG [13]
50
32.46
0.910
0.229
28.11
0.849
0.279
DeSiRe-GS [12]
36
33.61
0.919
0.204
29.75
0.878
0.213
Ours
34
36.88
0.948
0.113
30.14
0.880
0.183
Fig. 4.
Qualitative results. Decomposition with DeSiRe-GS [12] suffers from severe misclassification, whereas DIAL-GS achieves accurate decomposition
and clear instance separation.
exist in the current frame and perform ID clustering only on
them.
We select existing Gaussians of frame t as Gexist
t
=
{j|˜o(t) > ϵ}. For each Gaussian j ∈Gexist
t
, we extract its
predicted distribution as Pj = softmax(l(ej)) and predicted
ID as IDj = argmax(Pj). We collect dynamic Gaussians
as Gdyn
t
= {j|IDj ∈D, j ∈Gexist
t
}, and static Gaussians
as: Gstatic
t
= Gexist
t
−Gdyn
t
. For each Gaussian in Gdyn
t
,
we then search for its K nearest neighbors in Gstatic
t
and
obtain their predicted distributions Qk. Finally, we adopt
the KL divergence as the loss function to encourage nearby
static Gaussians to align with the ID-embedding of dynamic
Gaussians:
L3d =
1
K · |Gdyn
t
|
|Gdyn
t
|
X
j=1
K
X
k=1
DKL(Pj||Qk).
(10)
Note that L3d is activated in the later stage of training,
when both the dynamic attributes and the ID-embedding have
nearly converged. Such a scheduling strategy maximizes the
reliability of Gexist
t
while exploiting the stable ID-embedding
optimized by Lid. By enforcing 3D clustering, instance
awareness is no longer restricted to 2D alignment but instead
achieves a holistic embedding in the 3D space.
Dynamic Consistency Loss. After the ID-embedding has
stabilized through Lid and L3d, we leverage reliable instance
awareness to enforce consistency of dynamic attributes.
Similar to the computation of L3d, we first extract Gexist
t
and
their predicted IDs. For each instance in D, we gather its
Gaussians by Gi
t = {j | IDj = i, i ∈D}. For every Gaussian
in Gi
t, we then search K nearest neighbors within Gi
t and
obtain their instant velocities and life-spans. The consistency
losses are defined as follows:
Lmag =
1
K · |Gi
t|
|Gi
t|
X
j=1
K
X
k=1
∥¯vj −¯vk∥2
∥¯vmean∥2
,
(11)
Ldir =
1
2K · |Gi
t|
|Gi
t|
X
j=1
K
X
k=1

1 −
¯vj · ¯vk
∥¯vj∥2 · ∥¯vk∥2

,
(12)
Lbeta =
1
K · |Gi
t|
|Gi
t|
X
j=1
K
X
k=1
|βj −βk|
¯β
,
(13)
Lconsist = λmag·Lmag+λdir·Ldir+(1−λmag−λdir)·Lbeta. (14)
The reinforcement of consistency encourages the 4DGS of
the same dynamic object to form a coherent representation,
thereby reducing artifacts in novel view synthesis (NVS) and
allowing dynamic attributes to serve as reliable auxiliary cues
besides ID-embedding for decomposition.

<!-- page 6 -->
Fig. 5. Instance Edition. By realizing instance awareness, DIAL-GS supports
instance edition within the self-supervised regime.
D. Optimization
In the first stage, all Gaussians are treated as static. We
adopt the same densification and pruning strategy as 3DGS
[16], and the training loss consists:
LI = (1 −λssim)∥I −˜I∥+ λssimSSIM(I, ˜I),
(15)
LD = ∥ID −Dgt∥,
(16)
Lo = −1
hw
X
O·log O−1
hw
X
Msky ·log(1−O). (17)
The whole loss function of stage one is:
Lstage1 = λILI + λDLD + λoLO.
(18)
In the second stage, we gradually introduce the proposed
losses and the overall objective of stage 2 is formulated as:
Lstage2 = λILI + λDLD + λoLO + λ¯vL¯v + λβLβ+
λidLid + λ3dL3d + λconsistLconsist. (19)
E. Instance-level Scene Edition
Without instance awareness, former self-supervised works
can only perform coarse decomposition. On the contrary,
DIAL-GS empowers self-supervised reconstruction with the
ability to edit specific instances.
We jointly consider the e, β, ¯v and ˜o to select the Gaussians
belonging to instance i at frame t. By modifying ˜µ(t) and
color c, we change its position and appearance. We also use
ˆµ = ˜µ(t)+∆t· ¯v to change instance’s velocity while keeping
its original trajectory.
IV. EXPERIMENT
A. Experimental Setting
Dataset. For evaluation, we follow the experimental setup of
PVG [13] and DeSiRe-GS [12], focusing on highly dynamic
scenarios to enable comprehensive baseline comparisons.
Evaluation Metrics. We adopt PSNR, SSIM and LPIPS as
metrics for the evaluation of image rendering quality.
Implementation Details. We trained our model on one
NVIDIA A100 Tensor Core GPU. In the first stage, we use
BoT-SORT [36], [40], [41] as the 2D tracker and we directly
derive semantic mask from the instance mask in tracking
sequence. We train Sover for 30,000 iterations and each Ct
for 400 iterations. For efficiency and stability, we wrap 2
frames to check inconsistency. We take δ as 1e −3 to select
dynamic IDs and ϵ as 5e −4 to select existing Gaussians.
During the 55,000 iterations of stage 2, we gradually introduce
L2d, L¯v, Lβ, L3d and Lconsist in sequence. We set K to 5
for KNN search. Weights mentioned are λssim = 1, λI = 1,
λD = 1, λo = 0.05, λ¯v = 0.01, λβ = 0.001, λid = 0.1,
λ3d = 1.5, λconsist = 0.01, λmag = 0.4 and λdir = 0.2.
B. Experimental Results
We report quantitative results against other baselines in
Tab. I in both image reconstruction and novel view synthesis.
As shown in Table I, DIAL-GS achieves the best performance
with more than 3 PSNR improvement in image reconstruction
and competitive improvement in novel view synthesis. As for
the rendering speed, we achieve FPS close to DeSiRe-GS [12]
with extra ID-embedding.
In Fig. 4, we provide qualitative comparison against
DeSiRe-GS [12]. Following PVG [13], DeSiRe-GS relies
on staticness coefficient ρ = β
l for decomposition. It can be
observed that DeSiRe-GS [12] suffers severe misclassification
between dynamic and static regions: artifacts of dynamic
objects remain in the static part, small or slow-moving
dynamic objects are mistakenly treated as static, and the
dynamic part includes clearly static elements such as road
surfaces and parked cars. By contrast, DIAL-GS employs
ID embeddings as the primary criterion, with the staticness
coefficient as auxiliary support, thereby achieving precise
decomposition and enabling instance-level separation — a
capability that prior self-supervised methods could not realize.
C. Ablation Study
To verify the effectiveness of the instance-level dynamic
perception and loss functions, we perform ablation studies in
NVS task with the same sequences of Tab. I.
Dynamic score. In stage one, we test different methods to
select dynamic IDs within the score ranking list. As shown in
Fig. 6, subfigures (a)–(d) present the dynamic score ranking
Fig. 6.
Ablation studies on dynamic score. The horizontal axis denotes
instance IDs, and the vertical axis shows their dynamic scores. Stars represent
dynamic instances, while circles represent static ones. Red line represents
the threshold.

<!-- page 7 -->
Fig. 7.
Dynamic ID selection in different scenes. Axes and icons follow
the same convention as in Fig. 6.
TABLE II
ABLATION STUDY WITH LOSS CONFIGURATIONS.
Exp.
L¯v
Lβ
Lid
L3d
Lconsist
PSNR↑
SSIM↑
LPIPS↓
(a)
–
✓
✓
✓
✓
29.9262
0.8792
0.1817
(b)
✓
–
✓
✓
✓
29.8243
0.8674
0.1989
(c)
✓
✓
–
–
–
29.8331
0.8788
0.1833
(d)
✓
✓
✓
–
✓
29.8405
0.8788
0.1823
(e)
✓
✓
✓
✓
–
29.9032
0.8792
0.1813
(f)
✓
✓
✓
✓
✓
30.1368
0.8803
0.1826
using S3
i , S2
i , Si, and exp(Si), respectively. Subfigures (e)
and (f) further analyze S3
i when computed from only one type
of inconsistency: S(e)
i,t = Ipos
i,t
and S(f)
i,t = Iapp
i,t . The results
demonstrate that stable separation between dynamic and static
IDs is realized by jointly considering appearance and position
inconsistency with the cubic transform. We evaluate the
selection procedure across different scenes. As shown in
Fig. 7, the cubic transformation suppresses static-instance
scores toward zero while preserving relatively high scores for
dynamic instances, enabling stable thresholding for dynamic
ID separation. The few remaining misclassifications mainly
arise from distant or small objects with limited observations.
Dynamic Attribute Regularization. We ablate L¯v and Lβ to
examine the role of velocity and life-span regularization. As
shown in Tab. II(a)(b), both losses contribute to improving
novel-view rendering quality. Moreover, Fig. 8 illustrates
that removing the regularization leads to inaccurate dynamic
attributes in distant static regions and velocity grows exces-
sively and life-span shrinks, ultimately causing overfitting to
training views.
ID-embedding. The result in Tab. II(c) shows that ID-
embedding not only enables instance awareness but also
improves the reconstruction quality.
Reciprocal Identity–Dynamics Training. From Tab. II(d)(e),
we observe that the NVS quality drops when either L3d or
Lconsist is removed. Fig. 9 shows that removing L3d leads
Fig. 8.
Effect of Dynamic Attribute Regularization. The regularization
ensures clean static regions while constraining velocity and life-span to avoid
excessive growth or shrinkage.
Fig. 9.
Effect of L3d. With L3d, the ID-embedding is more complete and
the decomposition is more clean.
Fig. 10.
Effect of Lconsist. The dynamic attributes are more consistent and
reasonable with the guide of Lconsist.
to incomplete ID embeddings, resulting in residuals along
vehicle boundaries (first row) and artifacts within dynamic
objects (second row). Fig. 10 shows that Lconsist enhances the
consistency of dynamic attributes within the same instance.
As a result, the motion modeling adheres more closely to
realistic physical behavior, which in turn improves both novel
view synthesis and decomposition.
D. Discussion
While DIAL-GS achieves strong instance-aware reconstruc-
tion, it relies on an external 2D tracker, which may introduce
occasional errors such as broken IDs or false detections. We
nonetheless adopt this design because fully self-supervised
dynamic perception remains unreliable for complex road
scenes as shown by DeSiRe-GS [12] and discussed in Sec. III-
B. Furthermore, generic self-supervised features [42] adapted
by DeSiRe-GS [12] are not tailored to driving environments
and lack awareness of object-motion patterns. In contrast,
modern 2D trackers have already encoded rich semantic and
physical priors specific to traffic scenes, producing more
precise instance masks than those obtained from purely self-
supervised approaches, making them a pragmatic and effective
choice for our framework.
V. CONCLUSION
In this paper, we present DIAL-GS, a novel self-supervised
framework with dynamic instance awareness. DIAL-GS
achieves instance-level dynamic perception by leveraging
inconsistency caused by motion. By proposing instance-
aware 4DGS, DIAL-GS jointly encodes identity and dynamic
attributes and further enables them to benefit each other
in reciprocal identity–dynamics training strategy. Extensive
experiments validate its effectiveness, demonstrating that
DIAL-GS advances self-supervised reconstruction for real-
world autonomous driving scenarios.

<!-- page 8 -->
REFERENCES
[1] H. Zhu, Z. Zhang, J. Zhao, H. Duan, Y. Ding, X. Xiao, and J. Yuan,
“Scene reconstruction techniques for autonomous driving: a review of
3d gaussian splatting,” Artificial Intelligence Review, vol. 58, no. 1,
p. 30, 2024.
[2] L. Liao, W. Yan, M. Yang, and S. Zhang, “Learning-based 3d
reconstruction in autonomous driving: A comprehensive survey,” arXiv
preprint arXiv:2503.14537, 2025.
[3] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang,
X. Zhou, and S. Peng, “Street gaussians: Modeling dynamic urban
scenes with gaussian splatting,” in European Conference on Computer
Vision.
Springer, 2024, pp. 156–173.
[4] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang,
“Drivinggaussian: Composite gaussian splatting for surrounding dy-
namic autonomous driving scenes,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2024, pp.
21 634–21 643.
[5] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic,
O. Litany, Z. Gojcic, S. Fidler, M. Pavone, et al., “Omnire: Omni
urban scene reconstruction,” arXiv preprint arXiv:2408.16760, 2024.
[6] M. Khan, H. Fazlali, D. Sharma, T. Cao, D. Bai, Y. Ren, and B. Liu,
“Autosplat: Constrained gaussian splatting for autonomous driving scene
reconstruction,” arXiv preprint arXiv:2407.02598, 2024.
[7] T. Deng, S. Liu, X. Wang, Y. Liu, D. Wang, and W. Chen, “Prosgnerf:
Progressive dynamic neural scene graph with frequency modulated
auto-encoder in urban scenes,” arXiv preprint arXiv:2312.09076, 2023.
[8] H. Zhou, J. Shao, L. Xu, D. Bai, W. Qiu, B. Liu, Y. Wang, A. Geiger,
and Y. Liao, “Hugs: Holistic urban 3d scene understanding via gaussian
splatting,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 21 336–21 345.
[9] S. Hwang, M.-J. Kim, T. Kang, J. Kang, and J. Choo, “Vegs: View
extrapolation of urban scenes in 3d gaussian splatting using learned
priors,” in European Conference on Computer Vision.
Springer, 2024,
pp. 1–18.
[10] G. Hess, C. Lindstr¨om, M. Fatemi, C. Petersson, and L. Svensson,
“Splatad: Real-time lidar and camera rendering with 3d gaussian
splatting for autonomous driving,” in Proceedings of the Computer
Vision and Pattern Recognition Conference, 2025, pp. 11 982–11 992.
[11] J. Ost, F. Mannan, N. Thuerey, J. Knodt, and F. Heide, “Neural
scene graphs for dynamic scenes,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp.
2856–2865.
[12] C. Peng, C. Zhang, Y. Wang, C. Xu, Y. Xie, W. Zheng, K. Keutzer,
M. Tomizuka, and W. Zhan, “Desire-gs: 4d street gaussians for static-
dynamic decomposition and surface reconstruction for urban driving
scenes,” in Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025, pp. 6782–6791.
[13] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibration
gaussian: Dynamic urban scene reconstruction and real-time rendering,”
arXiv:2311.18561, 2023.
[14] S. Sun, C. Zhao, Z. Sun, Y. V. Chen, and M. Chen, “Splatflow: Self-
supervised dynamic gaussian splatting in neural motion flow field
for autonomous driving,” in Proceedings of the Computer Vision and
Pattern Recognition Conference, 2025, pp. 27 487–27 496.
[15] Y. Mao, R. Xiong, Y. Wang, and Y. Liao, “Unire: Unsupervised instance
decomposition for dynamic urban scene reconstruction,” arXiv preprint
arXiv:2504.00763, 2025.
[16] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[17] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan,
J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large scene
neural view synthesis,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2022, pp. 8248–8258.
[18] Z. Li, L. Li, and J. Zhu, “Read: Large-scale neural scene rendering
for autonomous driving,” in Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 37, no. 2, 2023, pp. 1522–1529.
[19] H. Turki, J. Y. Zhang, F. Ferroni, and D. Ramanan, “Suds: Scalable
urban dynamic scenes,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2023, pp. 12 375–12 385.
[20] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che,
D. Xu, S. Fidler, M. Pavone, et al., “Emernerf: Emergent spatial-
temporal scene decomposition via self-supervision,” arXiv preprint
arXiv:2311.02077, 2023.
[21] A. Rabby and C. Zhang, “Beyondpixels: A comprehensive review of the
evolution of neural radiance fields,” arXiv preprint arXiv:2306.03000,
2023.
[22] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and
L. Gao, “Recent advances in 3d gaussian splatting,” Computational
Visual Media, vol. 10, no. 4, pp. 613–642, 2024.
[23] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black,
“Smpl: A skinned multi-person linear model,” in Seminal Graphics
Papers: Pushing the Boundaries, Volume 2, 2023, pp. 851–866.
[24] N. Huang, X. Wei, W. Zheng, P. An, M. Lu, W. Zhan, M. Tomizuka,
K. Keutzer, and S. Zhang, “S3 gaussian: Self-supervised street gaussians
for autonomous driving,” arXiv preprint arXiv:2405.20323, 2024.
[25] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic
scenes,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 130–141.
[26] S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene
labelling and understanding with implicit scene representation,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 15 838–15 847.
[27] X. Fu, S. Zhang, T. Chen, Y. Lu, L. Zhu, X. Zhou, A. Geiger, and
Y. Liao, “Panoptic nerf: 3d-to-2d label transfer for panoptic urban
scene segmentation,” in 2022 International Conference on 3D Vision
(3DV).
IEEE, 2022, pp. 1–11.
[28] K. Mazur, E. Sucar, and A. J. Davison, “Feature-realistic neural
fusion for real-time, open set scene understanding,” arXiv preprint
arXiv:2210.03043, 2022.
[29] C. Peng, C. Xu, Y. Wang, M. Ding, H. Yang, M. Tomizuka, K. Keutzer,
M. Pavone, and W. Zhan, “Q-slam: Quadric representations for
monocular slam,” in Conference on Robot Learning.
PMLR, 2025,
pp. 1763–1781.
[30] Y. Siddiqui, L. Porzi, S. R. Bul´o, N. M¨uller, M. Nießner, A. Dai,
and P. Kontschieder, “Panoptic lifting for 3d scene understanding
with neural fields,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2023, pp. 9043–9052.
[31] M. Ye, M. Danelljan, F. Yu, and L. Ke, “Gaussian grouping: Segment
and edit anything in 3d scenes,” in European conference on computer
vision.
Springer, 2024, pp. 162–179.
[32] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., “Segment anything,”
in Proceedings of the IEEE/CVF international conference on computer
vision, 2023, pp. 4015–4026.
[33] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr,
R. R¨adle, C. Rolland, L. Gustafson, et al., “Sam 2: Segment anything
in images and videos,” arXiv preprint arXiv:2408.00714, 2024.
[34] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen, and Q. Tian,
“Segment any 3d gaussians,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 39, no. 2, 2025, pp. 1971–1979.
[35] J. Guo, X. Ma, Y. Fan, H. Liu, and Q. Li, “Semantic gaussians: Open-
vocabulary scene understanding with 3d gaussian splatting,” arXiv
preprint arXiv:2403.15624, 2024.
[36] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, “Bot-sort: Robust asso-
ciations multi-pedestrian tracking,” arXiv preprint arXiv:2206.14651,
2022.
[37] Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, “S-nerf: Neural
radiance fields for street views,” arXiv preprint arXiv:2303.00749,
2023.
[38] J. Guo, N. Deng, X. Li, Y. Bai, B. Shi, C. Wang, C. Ding, D. Wang, and
Y. Li, “Streetsurf: Extending multi-view implicit surface reconstruction
to street views,” arXiv preprint arXiv:2306.04988, 2023.
[39] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou,
Y. Chen, R. Yang, et al., “Mars: An instance-aware, modular and
realistic simulator for autonomous driving,” in CAAI International
Conference on Artificial Intelligence.
Springer, 2023, pp. 3–15.
[40] M. Brostr¨om, “BoxMOT: pluggable SOTA tracking modules for
object detection, segmentation and pose estimation models.” [Online].
Available: https://github.com/mikel-brostrom/boxmot
[41] W. Abdulla, “Mask r-cnn for object detection and instance segmentation
on keras and tensorflow,” https://github.com/matterport/Mask RCNN,
2017.
[42] Y. Yue, A. Das, F. Engelmann, S. Tang, and J. E. Lenssen, “Improving
2d feature representations by 3d-aware fine-tuning,” in European
Conference on Computer Vision.
Springer, 2024, pp. 57–74.
