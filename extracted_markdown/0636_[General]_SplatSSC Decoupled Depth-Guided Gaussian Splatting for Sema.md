<!-- page 1 -->
SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene
Completion
Rui Qian*1, Haozhi Cao*1, Tianchen Deng2, Shenghai Yuan1, Lihua Xie1
1Nanyang Technological University
2Shanghai Jiao Tong University
{rqian003, haozhi002, shyuan, elhxie}@ntu.edu.sg, dengtiancheng@sjtu.edu.cn
Abstract
Monocular 3D Semantic Scene Completion (SSC) is a chal-
lenging yet promising task that aims to infer dense geomet-
ric and semantic descriptions of a scene from a single im-
age. While recent object-centric paradigms significantly im-
prove efficiency by leveraging flexible 3D Gaussian primi-
tives, they still rely heavily on a large number of randomly
initialized primitives, which inevitably leads to 1) inefficient
primitive initialization and 2) outlier primitives that introduce
erroneous artifacts. In this paper, we propose SplatSSC, a
novel framework that resolves these limitations with a depth-
guided initialization strategy and a principled Gaussian ag-
gregator. Instead of random initialization, SplatSSC utilizes
a dedicated depth branch composed of a Group-wise Multi-
scale Fusion (GMF) module, which integrates multi-scale im-
age and depth features to generate a sparse yet representative
set of initial Gaussian primitives. To mitigate noise from out-
lier primitives, we develop the Decoupled Gaussian Aggrega-
tor (DGA), which enhances robustness by decomposing geo-
metric and semantic predictions during the Gaussian-to-voxel
splatting process. Complemented with a specialized Proba-
bility Scale Loss, our method achieves state-of-the-art per-
formance on the Occ-ScanNet dataset, outperforming prior
approaches by over 6.3% in IoU and 4.1% in mIoU, while
reducing both latency and memory cost by more than 9.3%.
Code — https://github.com/Made-Gpt/SplatSSC
Introduction
3D scene understanding has garnered significant atten-
tion with the rapid evolution of embodied agents and au-
tonomous driving (Li et al. 2024; Yi et al. 2025; Fusic and
Sitharthan 2024; Zheng et al. 2025; Zammit and van Kam-
pen 2023). As a key technology in this domain, 3D occu-
pancy prediction (Tong et al. 2023; Huang et al. 2023; Wei
et al. 2023; Tian et al. 2023; Wang et al. 2024c) and 3D Se-
mantic Scene Completion (SSC) (Cao and de Charette 2022;
Miao et al. 2023; Zhang, Zhu, and Du 2023; Li et al. 2023;
Mei et al. 2024) have made remarkable progress. Early and
conventional approaches for these tasks predominantly rely
on grid-based representations. However, processing dense
3D volumes incurs prohibitive computational and memory
costs. To mitigate this limitation, various efficiency-driven
*These authors contributed equally.
 16200 anchors
Random Init Gaussians
 1200 anchors
Depth Init Gaussians
redundant !
efficient !
 3.9% valid
(b)
(a)
Figure 1: Comparison with prior framework. (a) Recent
transformer-based SSC frameworks start with a large set of
randomly initialized Gaussian primitives, introducing redun-
dancy. (b) Our framework starts with a compact yet targeted
set of Gaussian primitives, guided by geometric priors.
strategies have been explored, such as accelerating process-
ing with Bird’s-Eye-View (BEV) projections (Yu et al. 2023;
Hou et al. 2024), or leveraging the natural sparsity of scenes
with sparse voxels (Tang et al. 2024; Li et al. 2023) and
points (Shi et al. 2024; Wang et al. 2024a).
Despite the devoted efforts, such methods remain inher-
ently constrained by their discrete and grid-aligned nature,
which struggle to model the sparse geometry efficiently.
A recent paradigm shift towards object-centric representa-
tions, pioneered by GaussianFormer (Huang et al. 2024),
has achieved a breakthrough. By utilizing flexible 3D Gaus-
sian primitives (Kerbl et al. 2023) to represent the scene,
this approach strikes a new balance between performance
and efficiency. Building upon this foundation, subsequent
works (Huang et al. 2025) have advanced this field by devel-
oping more principled aggregation methods based on Gaus-
sian Mixture Models (GMMs) and adapting the paradigm to
indoor scenes for incremental perception (Wu et al. 2025;
Zhang et al. 2025; Wang et al. 2025a).
While the object-centric paradigm offers a promising di-
rection, its application in vision-only settings faces a foun-
dational challenge: how to efficiently initialize and reliably
arXiv:2508.02261v3  [cs.CV]  31 Dec 2025

<!-- page 2 -->
supervise 3D primitives using only monocular cues. To en-
sure complete coverage of the target 3D space without ge-
ometric cues, the predominant strategy is to randomly dis-
tribute numerous primitives throughout the 3D volume, as
shown in Figure 1(a). This leads to two critical, coupled
limitations: 1) Inefficient Primitive Initialization. A signif-
icant portion of the model’s capacity is inevitably wasted
on representing empty or unknown space, making the ran-
dom distribution strategy inherently redundant. 2) Fragile
Aggregation of Outliers. Existing Gaussian-to-voxel splat-
ting strategies (Huang et al. 2024, 2025) lack an effective
rejection mechanism to mitigate the impact of outlier prim-
itives. This allows outliers to spurious semantics on distant
voxels, creating “floaters” in otherwise empty space.
To this end, we introduce SplatSSC, a novel framework
designed to tackle inefficient initialization and fragile ag-
gregation in object-centric SSC. Rather than starting with
a large set of random primitives, SplatSSC leverages geo-
metric priors to guide the primitive initialization, as shown
in Figure 1(b), reducing redundancy while maintaining rep-
resentational capacity. We generate this prior using a tai-
lored depth branch, equipped with our proposed Group-wise
Multi-scale Fusion (GMF) module. GMF integrates multi-
scale image features and depth features from a pretrained
depth estimator via Group Cross-Attention (GCA) for ef-
ficient multi-modal fusion. The resulting geometric priors
subsequently guide a lifter to initialize a sparse yet targeted
set of Gaussian primitives, which are then refined through
a standard multi-stage encoder. To address the “floaters”
that plague existing aggregators when dealing with sparse
outliers, we propose the Decoupled Gaussian Aggregator
(DGA), which renders the final semantic grid by completely
decomposing semantic and geometry prediction robustly.
Furthermore, to ensure stable geometric learning, we design
the specialized Probability Scale Loss to apply soft and pro-
gressive supervision to the intermediate encoder layers.
In summary, our contributions are as follows:
• We propose an efficient object-centric paradigm for
monocular SSC, namely SplatSSC, which features a
depth-guided strategy for initializing a sparse and tar-
geted set of Gaussian primitives.
• We introduce the Group-wise Multi-scale Fusion (GMF)
module with a Group Cross-Attention (GCA) core to ef-
ficiently generate a high-quality geometric prior.
• We design the Decoupled Gaussian Aggregator (DGA)
that decouples geometry and semantics to eliminate ag-
gregation artifacts from sparse primitives robustly.
• We propose a Probability Scale Loss to provide auxiliary
geometric supervision for robust end-to-end training.
Related Work
3D Semantic Scene Completion.
3D Semantic Scene
Completion (SSC) infers dense geometry and semantics
from partial observations. Early approaches (Song et al.
2017; Zhang et al. 2019; Wang et al. 2019) primarily focused
on indoor scenes using depth-only input, where deep convo-
lutional networks (CNNs) and Truncated Signed Distance
Function (TSDF) representations were widely employed. To
improve semantic understanding, subsequent methods (Li
et al. 2019, 2020; Wang et al. 2023) fuse features from both
RGB and depth inputs. In parallel, LiDAR-based SSC ap-
proaches (Rold˜ao, de Charette, and Verroust-Blondet 2020;
Yan et al. 2021; Yang et al. 2021) have been developed for
autonomous driving and also rely on CNN architectures.
A recent trend has shifted towards vision-only methods.
MonoScene (Cao and de Charette 2022) pioneered this di-
rection using a dense 2D-to-3D lifting with UNet architec-
ture (Ronneberger, Fischer, and Brox 2015), but this ap-
proach suffered from inherent depth ambiguity. To address
this, OccDepth (Miao et al. 2023) and ISO (Yu et al. 2024a)
introduced depth-aware strategies by leveraging stereo depth
and pretrained depth networks, respectively. Concurrently,
to tackle the inefficiency of dense voxel processing, Vox-
Former (Li et al. 2023), a two-stage model, proposed a
sparse-to-dense Transformer method based on generating
proposals from a geometry prior. Subsequent works con-
tinue to advance this paradigm (Mei et al. 2024; Yu et al.
2024b; Jiang et al. 2024), focusing on unified pipelines,
context-aware modeling, and instance-level reasoning.
While these Transformers improve accuracy, they remain
bound to grid-aligned voxel queries. Our work takes a dif-
ferent route through a flexible object-centric formulation in-
spired by (Wu et al. 2025; Huang et al. 2025).
Object-centric
3D
Scene
Representation.
A
recent
paradigm shift in occupancy prediction, pioneered by Gaus-
sianFormer (Huang et al. 2024), moves beyond grid-aligned
queries to object-centric representation using 3D Gaussian
primitives. This approach leverages the inherent sparsity of
3D scenes by representing them as a collection of continu-
ous ellipsoids, which are then rendered into a dense seman-
tic grid via an efficient Gaussian-to-voxel splatting mech-
anism. This marked a significant departure from discrete,
voxel-based frameworks (Li et al. 2023; Tang et al. 2024).
Sequential works (Huang et al. 2025; Zhao et al. 2025)
further advanced this paradigm by introducing a princi-
pled probabilistic framework via GMMs and incorporating
LiDAR-guided initialization to replace random placement.
In parallel, EmbodiedOcc (Wu et al. 2025) first adapted this
object-centric paradigm to the unique challenges of indoor
perception. It focuses on online incremental scene under-
standing, where confidence refinement is applied to contin-
uously update the Gaussian representation as an agent ex-
plores the environment. Following this, RoboOcc and Em-
bodiedOcc++ (Zhang et al. 2025; Wang et al. 2025a) ex-
tended this paradigm through geometry-aware refinement,
leveraging opacity cues and planar constraints to enhance
stability and structural fidelity.
However, object-centric approaches widely employ ran-
dom Gaussian primitive initialization, which introduces sig-
nificant redundancy, as most primitives are used to represent
empty space. In contrast, our method directly tackles this
problem by leveraging a depth prior to generate a compact
but more targeted set of primitives.

<!-- page 3 -->
add & norm
Gaussian
embeded
Gaussians
Occupancy
Occ
features
primitives
D.G.A
Encoder
Lifter
Image
Encoder
Depth
Anything
Initialize
G.C.A
FFN
G.M.F
Depth Branch
DPT Head
embed feature
sampling
Decoupled Gaussian Aggregator
sem prob
occ prob
Splatting
Mono Image
low-res depth
primitives
Initial primitives
Figure 2: An overview of our proposed SplatSSC architecture. Given a single input image, our model employs two parallel
branches: a trainable image encoder to extract multi-scale image features, and a frozen, pretrained Depth-Anything model to
extract depth features. After a sampling step, both features are fed into the proposed Group-wise Multi-scale Fusion (GMF)
block and a two-convolution layer depth head, yielding a refined feature map and a low-resolution depth map. These outputs
are then lifted to initialize a set of 3D Gaussian primitives. Subsequently, the primitives are processed by a multi-stage encoder
and finally passed to our Decoupled Gaussian Aggregator (DGA) to render the final semantic voxels.
Methodology
Problem Setup
Formally, given a single input RGB image Irgb, the local
prediction task is to infer the dense semantic voxel grid Vloc
and the underlying set of sparse Gaussian primitives Gloc
that represent the scene within the current camera frustum.
This process is defined as:
(Vloc, Gloc) = Mloc(Irgb),
(1)
where Mloc is our prediction model. The output grid Vloc ∈
{0, 1, ..., C −1}Xloc×Yloc×Zloc assigns each voxel a label
from C semantic classes, with class 0 denoting empty space.
The scene itself is represented by the set of N refined Gaus-
sian primitives Gloc = {Gi}N
i=1. Each primitive Gi is param-
eterized by its geometric and semantic properties: a mean
µi ∈R3, a scale vector si ∈R3, a rotation quaternion
qi ∈R4, an opacity ai ∈[0, 1], and a semantic logit vector
ci ∈RC−1. The scale and rotation are used to construct the
full anisotropic covariance matrix Σi:
Σi = RiSiST
i RT
i , Si = diag(si), Ri = q2r(qi).
(2)
where q2r(·) converts a quaternion into a rotation matrix and
diag(·) forms a diagonal scaling matrix.
Overview
The architecture of our approach is illustrated in Figure 2.
We first process the input image Irgb with an image en-
coder, composed of a lightweight image backbone Efficient-
Net (Tan and Le 2019) and FPN (Lin et al. 2017), to extract
multi-scale image features Frgb = {f l
rgb}L
l=1, where L is
the scale number. Simultaneously, a pretrained depth estima-
tion model Depth-Anything (Yang et al. 2024) is employed
to produce powerful depth features Fd. These two feature
streams are then fed into our specialized depth branch,
which employs the proposed GMF module to produce the
fused depth features F′
d and the refined depth map Id. The
resulting F′
d and Id are then fed to a lifting module to obtain
the initial Gaussian primitives Go with good geometry prior.
Subsequently, Go is refined by a series of encoder blocks
cyclically, following EmbodiedOcc. Given the refined primi-
tives, the 3D semantic voxels are obtained by our DGA ˆVagg.
By first leveraging the depth branch to generate a highly
compact set of primitives with geometrically grounded ini-
tial locations, we tackle the inefficiency inherent in random
initialization strategies. Subsequently, our DGA transforms
primitives into semantic voxels, overcoming the fragility of
prior aggregation methods. This enables our framework to
achieve state-of-the-art (SOTA) performance while main-
taining high efficiency with significantly fewer primitives.
Depth Branch
While recent monocular 3D completion methods (Wu et al.
2025; Yu et al. 2024a) leverage pretrained depth estima-
tors, they tend to utilize depth information as a secondary
guiding signal: either refining geometric distributions or in-
forming feature learning. However, this approach neglects
the rich latent features generated by depth networks. In con-
trast, our framework proposes a dual-pronged strategy: we
use the depth map as a direct geometric prior, while simul-
taneously employing the latent depth features as the initial
embeddings for 3D primitives. This not only ensures primi-
tives are grounded in both geometry (where) and semantics
(what), but necessitates a more advanced fusion mechanism.
To fulfill this demand, we design a dedicated depth branch.
Inspired by prior works (Ma et al. 2020; Jia et al. 2025), this
branch fuses multi-scale image features and depth cues via
our GMF mechanism. Specifically, GMF is a Transformer-
like block comprising the proposed GCA layer followed by
a point-wise FFN (Vaswani et al. 2017). The resulting fused
features F′
d are then processed by two convolutional layers
to produce the refined depth map Id.

<!-- page 4 -->
Figure 3: Illustration of the proposed GCA layer. The weight
matrix Wa is shared across different groups and scales, thus
reducing memory consumption and computational cost.
Group Cross-Attention.
The architecture of our GCA
module is illustrated in Figure 3. The process begins by sam-
pling features from the input depth features Fd and multi-
scale image features Frgb, using a set of predefined refer-
ence points normalized to the [0, 1] range. This step yields
the sampled features, denoted as Fs
d and F s
rgb = {f s,l
rgb}L
l=1
respectively. To balance performance and efficiency, we split
these features into G groups along the channel dimension,
where each group has a reduced feature dimensionality of
Dg = D/G. The Query Qg is projected from sampled depth
features, while the Key Kl
g and Value V l
g are projected from
sampled image features at each scale l:
Qg = (Fs
dWq)g, Kl
g = (f s,l
rgbWk)g, V l
g = (f s,l
rgbWv)g, (3)
where Wq, Wk, and Wv are linear projection matrices for
Query, Key, and Value, respectively. l ∈{1, ..., L} de-
notes the scale index. Inspired by the efficient design of De-
formable Attention (Zhu et al. 2021), we adopt a lightweight
linear projection mechanism in place of the standard dot-
product attention. To elaborate, the attention scores are com-
puted by feeding the element-wise sum of queries and keys
into a shared projection Wa ∈RDg×1:
Al
g = Sl
 Wa(Qg + Kl
g)

,
(4)
where Sl(·) denotes the Softmax operation across the scale
dimension, and g indexes feature groups. With the group-
wise formulation, both scale-wise attention and projection
are computed within each group, allowing Wa to be shared
across different groups and scales. This design significantly
reduces parameter overhead and computation.
The final fused representation is obtained by aggregating
value features V l
g using Hadamard product ◦with the at-
tention scores, followed by group concatenation Cg(·) and a
linear projection Wo:
F′
d = Cg
 L
X
l=1
Al
g ◦V l
g
!
Wo.
(5)
Efficiency Analysis.
The design of GCA is computa-
tionally lean. Standard cross-attention has a complexity of
O(LN 2D), where N is the sequence length. In contrast,
by employing a group-wise mechanism and replacing the
quadratic-cost dot-product with a linear-cost MLP, GCA sig-
nificantly reduces the complexity. The dominant cost of our
module becomes O(ND2(L + 2)/G), which is substantially
more efficient, especially for long sequences.
Decoupled Gaussian Aggregator
Gaussian-to-voxel splatting is a critical step for object-
centric approaches, which dictates the final quality of the oc-
cupancy output. While GaussianFormer first enabled object-
centric aggregation, its additive nature leads to redun-
dancy. The subsequent Probabilistic Gaussian Superposi-
tion (PGS) model proposed in GaussianFormer-2, though
theoretically elegant, introduces a flawed decoupling of ge-
ometry and semantics and therefore falls short when tackling
outlier primitives. To address these limitations, we propose
the DGA, a novel strategy that reformulates the task into two
distinct prediction pathways: Geometry Occupancy Predic-
tion and Conditional Semantic Distribution.
Analysis of Probabilistic Gaussian Superposition.
The
PGS models the semantic occupancy prediction at a point
x as a two-part process: a geometric occupancy probability
α(x) and a conditional semantic expectation e(x; G):
α(x) = 1 −
Y
i∈N (x)
(1 −α(x; Gi)) ,
(6)
e(x; G) =
N
X
i=1
p(Gi|x)˜ci =
PN
i=1 p(x|Gi)ai˜ci
PN
j=1 p(x|Gj)aj
,
(7)
p(x|Gi) =
1
(2π)3/2|Σi|1/2 α(x; Gi),
(8)
where p(x | Gi) denotes the conditional Gaussian probabil-
ity of point x under the ith primitive, and p(Gi | x) repre-
sents the normalized posterior of selecting Gi for x under a
uniform prior. α(x; Gi) = exp(−1
2(x−µi)T Σ−1
i (x−µi))
is the un-normalized Gaussian kernel. The key flaw in this
formulation lies in how the learned opacity ai is used. While
intended to represent a primitive’s existence confidence, it
is instead employed as the prior probability in the GMM.
The negative consequence of this choice becomes evident
when considering an isolated outlier primitive Gn. For any
point xf in its immediate vicinity, the likelihood p(xf|Gm)
for all other distant primitives Gm̸=n approaches zero. This
causes the normalization term in the posterior calculation to
be dominated by the outlier itself. Hence, the posterior prob-
ability p(Gn|xf) collapses to unity, regardless of the effect
of the low-confidence prior an:
p(Gn|xf) =
p(xf|Gn)an
PN
j=1 p(xf|Gj)aj
(9)
≈
p(xf|Gn)an
p(xf|Gn)an + 0 = 1.

<!-- page 5 -->
floaters
sem.
prob
Ours
D.G.A
floaters
occ.
prob
outliers
GF.agg
GF2.agg
Figure 4: Illustration of the proposed DGA. While
GF.agg (Huang et al. 2024) and GF2.agg (Huang et al. 2025)
wrongly produces the “floaters” from outliers, our DGA re-
mains robust, as the low occupancy probability directly sup-
presses its erroneous semantic contribution.
Accordingly, the semantic expectation at this point reduces
to e(xf; G) ≈˜cn, with the learned opacity an nullified by
the posterior normalization.
This issue is further exacerbated when considering the
geometry prediction, where the opacity ai is decomposed
and depends solely on the Gaussian kernel. As such, even a
low-confidence outlier can yield a high occupancy value for
nearby points. Consequently, the voxel xf is likely to be in-
correctly activated as occupied by the semantic label of the
outlier Gn, producing the characteristic “floaters”.
Geometric Occupancy Prediction.
Due to the exponen-
tial decay of the Gaussian kernel, only primitives in the lo-
cal vicinity of x have a meaningful influence. Therefore, we
only consider contributions from a neighborhood of relevant
Gaussian primitives for efficiency, denoted as N(x). The
occupancy is then modeled as a probabilistic OR operation
over this local set. Crucially, each primitive’s influence is
modulated by its learned opacity ai, which we interpret as
its existence confidence. This explicit use of opacity is a key
difference from PGS:
α′(x) = 1 −
Y
i∈N (x)
(1 −α(x; Gi) · ai) .
(10)
This natural gating mechanism suppresses the influence of
low-confidence outliers on the final occupancy probability.
Conditional Semantic Distribution.
Concurrently, we
predict the conditional semantic distribution e(x) under the
assumption that the position x is occupied. This is achieved
by using GMM, where we leverage the normalized semantic
weights of each Gaussian component. This design decou-
ples the semantic prediction from the opacity parameter ai,
forcing the model to rely solely on the geometric proximity
and the learned softmax-normalized semantic properties ˜ci
of each primitive. The posterior probability for each seman-
tic class k is then computed as:
ek(x) =
X
i∈N (x)
p(Gi|x) =
P
i∈N (x) p(x|Gi) · ˜ck
i
P
j∈N (x) p(x|Gj)
.
(11)
Probabilistic Fusion.
Finally, the two decoupled path-
ways are fused to compute the final probability distribution
ˆyx for each 3D position x. The probabilities for each valid
semantic class k and the empty class are defined as:

ˆyk
x = α′(x) · ek(x)
ˆyempty
x
= 1 −α′(x) .
(12)
This formulation serves as a principled and fully differ-
entiable gating mechanism. A low occupancy probability
α′(x), often resulting from an outlier primitive, directly sup-
presses any erroneous semantic prediction ek(x), thus ele-
gantly eliminating “floaters” without complex heuristics. We
demonstrate this effect in Figure 4.
Training Objective
Our model is trained via a two-stage strategy, where the first
stage establishes a robust geometric prior before training the
full network end-to-end. Throughout both stages, the pre-
trained model Depth-Anything-V2 is kept frozen.
Stage 1: Depth Branch Pre-training.
In this stage, we
exclusively train our depth branch to produce a high-quality
geometry prior. This module is supervised by a composite
depth loss Ld, similar with prior works (Laina et al. 2016;
Wang et al. 2025b):
Ld = λ1Ldepth
huber + λ2Lpts
huber + λ3Lgrad,
(13)
where the terms are the depth Huber loss, point cloud Huber
loss, and gradient matching Huber losses.
Stage 2: End-to-End SplatSSC Training.
In this stage,
we train the entire SplatSSC network. To prevent the model
from being overly constrained by the initial depth predic-
tions, while maintaining a robust geometric prior, we re-
move Ld and introduce our proposed Probability Scale Loss
Lprob
scal as a soft geometric supervision. The training objective
is therefore optimized with a final composite loss Lssc:
Lssc = Lsem + λ4Lprob
scal ,
(14)
where Lsem = λ5Lfocal + λ6Llovasz is the primary se-
mantic segmentation loss adopted by EmbodiedOcc. Our
loss Lprob
scal extends the geometry-aware scale loss Lgeo
scal from
MonoScene (Cao and de Charette 2022), adapting it to su-
pervise the predicted occupancy probability across all n
encoder layers. To account for the progressive refinement
across stages, we introduce a linear weighting schedule,
which imposes weaker constraints on early-stage predictions
and gradually enforces stronger consistency at deeper layers:
Lprob
scal = 1
2
n−1
X
i=1
i
n · Lgeo,i
scal + Lgeo,n
scal ,
(15)
where i is the layer index. The loss weights are set as λ1 =
10, λ2 = 20, λ3 = λ4 = 0.5, λ5 = 100, and λ6 = 2.
Experiments
To evaluate the effectiveness of our SplatSSC, we conduct
extensive experiments on the high-quality indoor datasets
Occ-ScanNet and Occ-ScanNet-mini (Yu et al. 2024a). De-
tails about datasets, implementation, and evaluation metrics
are included in our supplementary material from our code.

<!-- page 6 -->
Dataset
Method
Input
IoU
ceiling
floor
wall
window
chair
bed
sofa
table
tvs
furniture
objects
mIoU
Occ-ScanNet
TPVFormer
Irgb
33.39
6.96 32.97 14.41 9.10 24.01 41.49 45.44 28.61 10.66 35.37 25.31 24.94
GaussianFormer
Irgb
40.91 20.70 42.00 23.40 17.40 27.00 44.30 44.80 32.70 15.30 36.70 25.00 29.93
MonoScene
Irgb
41.60 15.17 44.71 22.41 12.55 26.11 27.03 35.91 28.32 6.57 32.16 19.84 24.62
ISO
Irgb
42.16 19.88 41.88 22.37 16.98 29.09 42.43 42.00 29.60 10.62 36.36 24.61 28.71
SurroundOcc
Irgb
42.52 18.90 49.30 24.80 18.00 26.80 42.00 44.10 32.90 18.60 36.80 26.90 30.83
EmbodiedOcc
Irgb
53.95 40.90 50.80 41.90 33.00 41.20 55.20 61.90 43.80 35.40 53.50 42.90 45.48
EmbodiedOcc++
Irgb
54.90 36.40 53.10 41.80 34.40 42.90 57.30 64.10 45.20 34.80 54.20 44.10 46.20
RoboOcc
Irgb
56.48 45.36 53.49 44.35 34.81 43.38 56.93 63.35 46.35 36.12 55.48 44.78 47.67
SplatSSC (Ours)
Irgb
62.83 49.10 59.00 48.30 38.80 47.40 62.40 67.00 49.50 42.60 60.70 45.40 51.83
Occ-ScanNet-mini
MonoScene
Irgb
41.90 17.00 46.20 23.90 12.70 27.00 29.10 34.80 29.10 9.70 34.50 20.40 25.90
ISO
Irgb
42.90 21.10 42.70 24.60 15.10 30.80 41.00 43.30 32.20 12.10 35.90 25.10 29.40
EmbodiedOcc
Irgb
55.13 29.50 49.40 41.70 36.30 41.90 60.40 59.60 46.30 34.50 58.00 43.50 45.57
EmbodiedOcc++
Irgb
55.70 23.30 51.00 42.80 39.30 43.50 65.60 64.00 50.70 40.70 60.30 48.90 48.20
SplatSSC (Ours)
Irgb
61.47 36.60 55.70 46.50 40.10 45.60 64.50 62.40 48.60 30.60 61.20 45.39 48.87
Table 1: Local Prediction Performance on the Occ-ScanNet dataset. The best results are highlighted in bold, while the second-
best are underlined.
Number Scale Range Mem.↓
(MiB)
Time↓
(ms)
Train
IoU
mIoU
19200
[0.01, 0.08]
3.122
135.18
✓
62.77 47.69
19200
[0.01, 0.16]
4.978
134.25
✓
60.64 43.31
19200
[0.01, 0.32]
14.380 134.51 OOM
/
/
4800
[0.01, 0.08]
3.158
123.27
✓
62.23 47.20
4800
[0.01, 0.16]
3.108
122.63
✓
61.53 46.74
4800
[0.01, 0.32]
5.854
122.70
✓
60.78 46.96
1200
[0.01, 0.08]
3.104
116.20
✓
60.18 48.32
1200
[0.01, 0.16]
3.112
115.56
✓
61.47 48.87
1200
[0.01, 0.32]
3.126
114.75
✓
57.09 42.38
Table 2: Ablation on Gaussian Parameters. Memory (Mem.)
usage and time are measured on one 3090 GPU.
GMF GF.agg GF2.agg DGA
IoU
mIoU
-
✓
-
-
11.64 12.62
-
-
✓
-
27.54 17.27
-
-
-
✓
48.85 36.91
✓
✓
-
-
16.63 10.45
✓
-
✓
-
57.70 45.13
✓
-
-
✓
60.61 48.01
Table 3: Ablation on the Components of SplatSSC.
Main Result
The main results on the Occ-ScanNet and Occ-ScanNet-
mini benchmarks are summarized in Table 1. Our SplatSSC
achieves SOTA performance, demonstrating strong robust-
ness and fine-grained scene understanding on both bench-
marks. For Occ-ScanNet, SplatSSC achieves 62.83% IoU
and 51.83% mIoU, surpassing the previous SOTA Ro-
boOcc (Zhang et al. 2025) by a substantial margin of 6.35%
and 4.16%, respectively. The per-class analysis further high-
lights the consistent improvements brought by SplatSSC
Lfocal Llovasz Lprob
scal
Lgeo
scal Lsem
scal Ld
IoU
mIoU
✓
✓
-
✓
✓
-
57.55 46.13
✓
✓
✓
-
-
✓
60.34 46.67
✓
✓
✓
-
✓
-
59.19 48.28
✓
✓
✓
-
-
-
60.61 48.01
Table 4: Ablation on Training Objective.
GMF DAv2 FT-DAv2
δ1 ↑
RMSE ↓C-l1 ↓
-
✓
-
0.075
50.314
1.996
✓
✓
-
0.981
4.944
0.182
-
-
✓
0.984
3.891
0.164
✓
-
✓
0.993
2.977
0.112
Table 5: Ablation on Depth Branch.
across diverse categories, from large structural elements
(e.g., walls and floor) to fine-grained objects (e.g., sofas
and chairs). These results underscore the strength of our
synergistic design. The depth-guided initialization facilitates
accurate geometric reconstruction, while our DGA ensures
sharp semantic boundaries. As illustrated in the qualitative
examples in Figure 5, SplatSSC yields superior 3D scene
perception capabilities that surpass the previous paradigm.
Ablation Studies
Ablation studies are conducted on the Occ-ScanNet-mini
dataset to assess the impact of design choice in our model.
Ablation on Gaussian Parameters.
We analyze the im-
pact of primitive count and scale range in Table 2, revealing
a clear trade-off between performance and efficiency. Our
setting achieves the highest semantic accuracy of 48.87%
mIoU with a remarkably compact configuration of just 1200
primitives. Increasing the count to 4800 and 19200 yields
marginal gains in geometric completeness but incurs higher

<!-- page 7 -->
ceiling
floor
wall
window
chair
bed
sofa
table
tvs
furniture
object
scene0272_01/0063
scene0000_00/0012
scene0107_00/0033
Input
Ground Truth
EmbodiedOcc
SplatSSC (Ours)
scene0623_00/0059
scene0468_00/0020
Figure 5: Qualitative results on the Occ-ScanNet-mini
dataset. Our method achieves superior performance in scene
completion and target object recall compared to others.
EmbodiedOcc
SplatSSC (Ours)
Parameters (M) ↓
127.51
115.63
Inference Time (ms) ↓
3464
3130
Memory Usage (MiB) ↓
134.0
133.8
133.6
133.4
133.2
130.0
126.0
122.0
118.0
114.0
3500
3400
3300
3200
3100
9.32%
9.64%
Increase
Decrease
133.604
133.857
0.19%
Figure 6: Efficiency Analysis.
latency and lower mIoU. The choice of scale range is equally
critical. Excessively large ranges degrade accuracy and trig-
ger Out-of-Memory (OOM) failures under dense configu-
rations, likely due to overlaps among oversized primitives.
In contrast, a moderate range [0.01, 0.16] offers the best
trade-off, effectively capturing both global layouts and fine-
grained details with minimal redundancy.
Ablation on Network Components.
We evaluate the im-
pact of our key components, GMF and DGA, in Table 3. The
analysis first highlights the necessity of a tailored aggrega-
tion mechanism. The standard GF.agg (Huang et al. 2024)
nearly fails in our sparse setting, yielding a prohibitively low
10.45% mIoU. While the more advanced GF2.agg (Huang
et al. 2025) performs significantly better, our DGA still sur-
passes it by over 2.8% in both IoU and mIoU. This con-
firms that “floaters” are the key bottleneck in sparse splat-
ting, and DGA is crucial for efficient and robust aggrega-
tion. The proposed GMF is equally important, as replacing
it with a naive depth-aware baseline (Wu et al. 2025) built on
Depth-Anything-V2 causes a substantial drop by more than
11% in both geometries and semantics, even when paired
with our DGA. The degradation becomes more severe with
other aggregators, leading to a near-collapse in performance.
This demonstrates the necessity of structured geometric pri-
ors for generating informative primitives.
Ablation on Training Objective.
Our validation on the
training objective design is shown in Table 4. The results
first confirm that the popular combination of geometry and
semantic scale losses (Lgeo
scal, Lsem
scal) is suboptimal for our
framework, yielding the lowest 46.13% mIoU. The explicit
depth loss Ld is also detrimental, as its inclusion consis-
tently degrades both geometric and semantic scores. Further-
more, while adding semantic scale loss provides a marginal
mIoU boost to a peak of 48.28%, it incurs over 1.42% IoU
drop. These findings lead to our final design: a simple yet ef-
fective objective incorporating our proposed Lprob
scal alongside
standard Focal and Lov´asz losses (Lfocal, Llovasz), which
achieves the best geometric performance of 60.61% IoU
while maintaining competitive semantic accuracy.
Ablation on Depth Branch.
The contribution of our GMF
module is validated in Table 5. The results highlight a dra-
matic impact of GMF on refining the geometric prior. When
applied to a frozen Depth-Anything-V2 (DAv2) backbone,
GMF boosts the δ1 score by a remarkable 0.906. Further-
more, it demonstrates its capability to enhance a fine-tuned
Depth-Anything-V2 (FT-DAv2) (Wu et al. 2025), pushing
the δ1 score to a new best of 0.993. This confirms that our
GMF is a powerful and versatile feature refiner, essential for
generating high-quality geometric representations.
Efficiency Analysis.
Beyond accuracy, we evaluate the
computational efficiency of SplatSSC against Embod-
iedOcc, with results detailed in Figure 6. Our method
demonstrates superior efficiency despite a negligible 0.19%
increase in parameter count. Specifically, SplatSSC achieves
a 9.32% reduction in inference latency and a 9.64% decrease
in memory usage. This advantage is primarily attributed to
our sparse design, which operates on significantly fewer
primitives than prior works.
Conclusion
In this paper, we introduced SplatSSC, a novel framework
for monocular 3D semantic scene completion. Our method
addresses the critical limitations of prior object-centric ap-
proaches through two core technical contributions: 1) a
depth-guided initialization strategy, powered by our group-
wise multi-scale fusion module, which generates a compact
and high-quality set of initial Gaussian primitives; and 2)
a decoupled Gaussian aggregator that robustly resolves ag-
gregation artifacts such as “floaters” from outlier primitives.
Extensive experiments demonstrate that SplatSSC estab-
lishes a new SOTA on the Occ-ScanNet benchmark, achiev-
ing superior accuracy while simultaneously reducing latency
and memory consumption.
Despite its outstanding performance, we acknowledge
several limitations that offer avenues for future work as dis-
cussed in the supplementary material from our code.

<!-- page 8 -->
Acknowledgments
This work was supported by National Research Foundation
of Singapore Medium-sized Centre for Advanced Robotics
Technology Innovation and Ministry of Education, Singa-
pore, under AcRF TIER 1 Grant RG64/23.
Appendix Overview
This technical appendix consists of the following sections.
• We detail the experimental setup for SplatSSC.
• We provide a detailed derivation of the semantic proba-
bility formulation for our proposed Decoupled Gaussian
Aggregator (DGA).
• We provide further visualization of qualitative results on
the Occ-ScanNet-mini and Occ-ScanNet validation sets.
• We provide further proof and analysis to support the ne-
cessity of our two-stage training.
• We conclude with a discussion of the current limitations
and potential applications of SplatSSC.
• We include a statement regarding our code availability
and its license.
Experimental Setup
Dataset
Occ-ScanNet (Yu et al. 2024a) comprises 45,755 training
frames and 19,764 validation frames, annotated with 12 se-
mantic classes, with one representing free space and eleven
corresponding to specific categories, including ceiling, floor,
wall, window, chair, bed, sofa, table, television, furniture,
and generic objects. The ground truth is provided as a voxel
grid covering a 4.8m×4.8m×2.88m region in front of the
camera, discretized into a resolution of 60×60×36. This
dataset serves as the benchmark for training and evalu-
ating local occupancy prediction. A smaller variant, Occ-
ScanNet-mini, is also available, containing 5,504 training
and 2,376 validation frames.
Evaluation metrics
Following common practice (Cao and de Charette 2022; Hu
et al. 2024), we evaluate the final semantic scene completion
performance using Intersection-over-Union (IoU) and mean
IoU (mIoU). These metrics are computed exclusively within
the current camera’s view frustum. To assess the quality of
the geometric prediction in our depth branch, we employ
three additional metrics: Chamfer l1 distance (C-l1), Root
Mean Squared Error (RMSE), and accuracy under thresh-
old (δ1) (Hu et al. 2024). For this geometric evaluation, the
ground truth point cloud is generated by down-sampling the
ground truth depth map using the indices from our GMF
module, then projecting the valid depth points into the cam-
era’s coordinate space.
Implementation Details
In our framework, the image encoder employs a pretrained
EfficientNet-B7 (Tan and Le 2019) as backbone, while the
depth branch utilizes a frozen fine-tuned Depth-Anything-
V2 (Wu et al. 2025) model. For both training stages, we use
the AdamW optimizer (Loshchilov and Hutter 2019) with
a weight decay of 0.01. We apply a learning rate multiplier
of 0.1 to the backbone. All input images are processed at a
resolution of 480 × 640.
Stage 1: Depth Branch Pretraining.
In the first stage, we
exclusively pretrain our depth branch to establish a robust
geometric prior. The down-sampled grid for our GMF has a
shape of 30×40. We employ a cosine learning rate schedule
with a 1000-iteration warmup, setting the peak learning rate
to 6 × 10−4. The model is trained for 10 epochs on the Occ-
ScanNet dataset using 2 NVIDIA 3090 GPUs with a per-
GPU batch size of 2 (total batch size of 4).
Stage 2: End-to-End SplatSSC Training.
In the second
stage, we train the full SplatSSC model, initializing the
depth branch with weights from stage one. The 30 × 40
down-sampled grid generates an initial set of 1200 Gaussian
primitives, with their scales initialized in the range [0.01m,
0.16m]. We train the model on 4 NVIDIA 4090 GPUs with a
per-GPU batch size of 2, resulting in a total batch size (bs) of
8. The learning rate follows a cosine schedule with a 1000-
iteration warmup, and the peak learning rate is determined
by a linear scaling rule: 2 × 10−4 · (bs/2). The model is
trained for 10 epochs on the full Occ-ScanNet dataset and
for 20 epochs on the Occ-ScanNet-mini subset.
Further experiments settings.
The experimental settings
for the ablation studies and efficiency analysis are summa-
rized in Table 6.
Derivation of Decoupled Gaussian Aggregator
This section presents a complete derivation of the pro-
posed Decoupled Gaussian Aggregator (DGA), clarifying
the probabilistic reasoning behind the semantic term that
models the probability of a point x belonging to class k,
given that it is occupied.
For clarity, we restate the definition of Gaussian primi-
tives, G = {Gi}N
i=1, with each Gaussian parameterized by
a mean µi ∈R3, a scale vector si ∈R3, a rotation quater-
nion qi ∈R4, a learned opacity ai ∈[0, 1], and a softmax-
normalized semantic vector ˜ci ∈RC.
Our DGA is designed to explicitly separate the prediction
of geometry and semantics. While define the final predic-
tion as ˆyk(x) = α′(x) · ek(x) for valid classes, we incorpo-
rate opacities ai into the occupancy probability α′(x) and
formulate a conditional semantic distribution ek(x).
We model the semantic distribution as a Gaussian mix-
ture model, where each primitive Gi in a local neighborhood
N(x) is a component. The likelihood of Gi contributing to
class k is determined by its semantic affinity ˜ck
i . Following
this, we can formulate the semantic probability for class k at
point x using Bayes’ theorem:
ek(x) =
P
i∈N (x) p(x|Gi)˜ck
i
P
j∈N (x)
PC
l=1 p(x|Gj)˜cl
j
.
(16)
This initial expression can be further simplified. By factor-

<!-- page 9 -->
Config
Ablation Studies
Efficiency Analysis
Gaussian Parameters Components of SplatSSC Training Objective
Depth Branch
Training Dataset
Occ-ScanNet-mini
Occ-ScanNet-mini
Occ-ScanNet-mini
Occ-ScanNet
Occ-ScanNet
Inference Dataset
Occ-ScanNet-mini
Occ-ScanNet-mini
Occ-ScanNet-mini Occ-ScanNet-mini
Occ-ScanNet
Training Device
4 RTX 3090
2 RTX 3090
2 RTX 3090
2 RTX 3090
4 RTX 4090
Inference Device
1 RTX 3090
1 RTX 3090
1 RTX 3090
1 RTX 3090
1 RTX 3090
Maximum Learning Rate
8 × 10−4
6 × 10−4
6 × 10−4
6 × 10−4
8 × 10−4
Weight Decay
0.01
0.01
0.01
0.01
0.01
Total Batch Size
8
6
6
6
8
Table 6: Experiment settings for different ablation studies and efficient analysis.
ing out the likelihood term in the denominator part, we have:
X
j∈N (x)
C
X
l=1
p(x|Gj)˜cl
j =
X
j∈N (x)
p(x|Gj)
 C
X
l=1
˜cl
j
!
. (17)
As the semantic vector ˜cj is softmax-normalized, the sum
of its components over all classes is unity, i.e., PC
l=1 ˜cl
j = 1.
This crucial property simplifies the normalization term to the
sum of only the geometric likelihoods.
Substituting this back, we arrive at the final expression for
our conditional semantic distribution:
ek(x) =
P
i∈N (x) p(x|Gi)˜ck
i
P
j∈N (x) p(x|Gj) .
(18)
Additional Visualization Results
Further Visualization on SSC
In Figure 10, we present additional qualitative results on
the Occ-ScanNet validation set. For each scene, we visual-
ize both the intermediate 3D semantic Gaussians and the fi-
nal dense occupancy prediction. The Gaussian views reveal
that SplatSSC forms a sparse, object-centric representation
that already captures well-aligned shapes and semantics for
major structures (e.g., furniture and walls), while the ren-
dered voxel grids show that this compact representation can
be faithfully translated into detailed 3D completions.
Furthermore, Figure 7 provides more visualizations on
the Occ-ScanNet-mini validation set using the same layout.
Again, we show the per-frame 3D semantic Gaussians to-
gether with the final occupancy predictions. These examples
highlight that our Gaussian scene representation remains sta-
ble and expressive in smaller and more cluttered environ-
ments, and that it consistently supports accurate reconstruc-
tion of fine-scale object geometry.
Further Visualization on Depth Branch
Figure 8 provides additional qualitative comparisons of our
depth branch. We visualize 30 × 40 depth maps from our
method and FT-DAv2 against downsampled ground truth,
which serve as spatial priors for Gaussian primitive ini-
tialization. Compared to FT-DAv2, our branch produces
depth maps with higher fidelity and sharper foreground-
background separation, aligning more closely with the
ground truth.
scene0168_01/0070
RGB Input
Ground Truth
SplatSSC (Ours)
scene0006_02/0033
3D Gaussians (Ours)
scene0626_01/0050
scene0706_00/0012
scene0673_02/0035
scene0362_01/0070
ceiling
floor
wall
window
chair
bed
sofa
table
tvs
furniture
object
Figure 7: Further visualization on Occ-ScanNet-mini.
These results indicate that our proposed depth branch can
effectively refine the depth prior into a more structured and
compact geometric representation, providing a stronger ini-
tialization for the subsequent Gaussian lifting stage.
Discussion
Analysis of Two-Stage Training Strategy
Figure 9 compares our two-stage schedule with an end-to-
end training variant. In both cases, we use the SSC loss:
Lssc = Lsem + λ1Lprob
scal ,
(19)
Lsem = λ2Lfocal + λ3Llovasz.
(20)

<!-- page 10 -->
RGB Input
Ground Truth
Depth Branch
FT-DAv2
(ours)
Figure 8: Qualitative visualization on the depth prediction.
Ours two-stage training
End-to-end training
0.48
0.40
0.32
0.24
2
4
6
8
10
10.0
7.5
5.0
2.5
Training Loss
mIoU
0
25000
50000
Iteration number
Epoch number
Figure 9: Comparison of different training strategies.
Our two-stage model in the second stage is optimized only
with Lssc, while the end-to-end baseline additionally uses
the depth loss:
Ld = λ4Ldepth
huber + λ5Lpts
huber + λ6Lgrad.
(21)
For a fair comparison, Figure 9 reports only Lssc together
with the validation mIoU. In our experiments, we set λ1 =
λ6 = 0.5, λ2 = 100, λ3 = 2, λ4 = 20, λ5 = 10.
We observe that the end-to-end variant converges to a
larger Lssc and a lower mIoU, whereas our two-stage sched-
ule achieves a smaller loss and a higher mIoU. We attribute
this gap to the absence of a precise depth prior in the end-
to-end setting: jointly training the depth branch from scratch
couples noisy depth supervision with SSC optimization and
makes the Gaussian lifting less stable. In contrast, the first
stage of our two-stage training learns a reliable depth prior
under metric supervision, and the second stage then refines
the Gaussian representation using Lssc only, which better
aligns the pretrained depth cues with the SSC objective.
Limitations
Despite the strong performance, our SplatSSC framework
has certain design constraints that highlight key areas for
further improvement.
Method
bs
lr
IoU
mIoU
EmbodiedOcc 2 2×10−4
52.59
42.61
4 2×10−4 55.13 (+2.54%) 45.57 (+2.96%)
SplatSSC
2 2×10−4
54.68
36.09
4 4×10−4 59.53 (+4.85%) 45.32 (+9.23%)
6 6×10−4 61.47 (+6.79%) 48.87 (+12.78%)
8 8×10−4 62.83 (+8.15%) 51.83 (+15.74%)
Table 7: Hyperparameter Sensitivity Analysis. We evaluate
the performance on different total batch sizes (bs) and max-
imum learning rates (lr).
Hyperparameter Sensitivity Analysis.
This experiment
validates our finding that SplatSSC’s performance is sub-
ject to a distinct threshold regarding its training hyperpa-
rameters, with results shown in Table 7. This effect is visi-
ble when comparing performance at different batch sizes. At
a total batch size of 2, our model’s performance is substan-
tially limited to 36.09% mIoU. However, upon increasing
the batch size to 4, the mIoU jumps dramatically to 45.32%,
reaching a competitive level. This demonstrates that a batch
size of at least 4 is necessary for effective optimization. Be-
yond this threshold, performance continues to scale robustly,
with the best results achieved at a batch size of 8. In contrast,
the baseline EmbodiedOcc (Wu et al. 2025) exhibits only
modest and linear gains. It is also worth noting that Embod-
iedOcc was designed for a per-GPU batch size of one, mak-
ing extensive scaling less applicable. This highlights that the
observed threshold effect is a unique characteristic of our
model’s interactive primitive optimization.
Local-View
Architectural
Constraint.
The
current
SplatSSC framework is designed to operate on a per-frame
basis, excelling at generating a high-quality scene repre-
sentation from a single view. However, this design presents
a scalability challenge when extending to global scene
perception. A naive extension of simply accumulating
primitives from consecutive frames would lead to an
unbounded growth in their total number, causing a rapid
escalation in both memory and computational costs. This
limitation reveals a critical need for a scalable online
primitive management strategy that leverages both pruning
and fusion techniques to prevent unbounded growth in
memory and computation. We leave this as a promising
direction for future work and will validate it on global scene
benchmarks (Wu et al. 2025; Wang et al. 2024b).
Future Outlook and Broader Applications
While SplatSSC establishes a new state-of-the-art, its under-
lying principles open up several exciting avenues for future
research. We discuss two key directions below.
Scaling to Unbounded and Large-Scale Environments.
A primary direction is adapting SplatSSC for large-scale
outdoor environments, particularly for applications like
autonomous driving. Unlike methods that rely on dense
grids (Wei et al. 2023; Zhang, Zhu, and Du 2023) or ran-
dom initialization across a predefined volume (Huang et al.

<!-- page 11 -->
Figure 10: Further visualization of predicted 3D Gaussians and voxels on the Occ-ScanNet dataset.
2025), our depth-guided approach naturally focuses compu-
tation on observed surfaces. This inherent efficiency makes
it exceptionally well-suited for sparse and large-scale set-
tings. To fully realize this potential, the fixed volumetric grid
could be replaced with more flexible spatial data structures,
such as hash-encoded grids (Deng et al. 2025b), to support
unbounded scenes. This extension would also need to ad-
dress the challenges unique to this domain, such as manag-
ing a dynamically growing set of primitives and handling the
presence of dynamic objects.
Application in Embodied AI and Robotics.
Moving be-
yond passive perception, a critical frontier in 3D vision is to
build representations that support active interaction, a cen-
tral theme in embodied and spatial intelligence (Wang et al.
2024b; Halacheva et al. 2025). Applying SplatSSC in em-
bodied AI requires moving from single-frame perception to
building a persistent and interactive world model. This de-
mands a higher level of detail than is currently captured;
for instance, an agent needs not just a semantic label for
a “door”, but also precise geometric information about its
handle for manipulation. This may necessitate using a larger
number of Gaussians or a finer-grained semantic taxonomy.
Furthermore, it requires a robust online framework where
the agent can continuously fuse new observations (Deng
et al. 2025a), prune outdated information, and refine its
Gaussian-based world map in real-time.
Code Availability and Licensing
The source code and trained models associated with this pa-
per will be released on GitHub upon acceptance, and the
specific URL will be provided.
All our source code is licensed under the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 In-
ternational (CC BY-NC-SA 4.0) license. This permits any
non-commercial use, distribution, and reproduction in any
medium, provided the original work is properly cited and
any derivative works are shared under the same license.
References
Cao, A.; and de Charette, R. 2022. MonoScene: Monoc-
ular 3D Semantic Scene Completion.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 3981–3991. CVPR.
Deng, T.; Shen, G.; Xun, C.; Yuan, S.; Jin, T.; Shen, H.;
Wang, Y.; Wang, J.; Wang, H.; Wang, D.; and Chen, W.
2025a. MNE-SLAM: Multi-Agent Neural SLAM for Mo-
bile Robots. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 1485–1494.
CVPR.
Deng, T.; Wu, W.; He, J.; Pan, Y.; Jiang, X.; Yuan, S.;
Wang, D.; Wang, H.; and Chen, W. 2025b. VPGS-SLAM:
Voxel-based Progressive 3D Gaussian SLAM in Large-Scale
Scenes. arxiv:2505.18992.
Fusic, S. J.; and Sitharthan, R. 2024.
Improved RRT*
Algorithm-Based Path Planning for Unmanned Aerial Vehi-

<!-- page 12 -->
cle in a 3D Metropolitan Environment. Unmanned Systems,
12(05): 859–875.
Halacheva, A.-M.; Zaech, J.-N.; Wang, X.; Paudel, D. P.;
and Van Gool, L. 2025.
GaussianVLM: Scene-centric
3D
Vision-Language
Models
using
Language-aligned
Gaussian Splats for Embodied Reasoning and Beyond.
arxiv:2507.00886.
Hou, J.; Li, X.; Guan, W.; Zhang, G.; Feng, D.; Du, Y.; Xue,
X.; and Pu, J. 2024. FastOcc: Accelerating 3D Occupancy
Prediction by Fusing the 2D Bird’s-Eye View and Perspec-
tive View. In IEEE International Conference on Robotics
and Automation, 16425–16431. ICRA.
Hu, M.; Yin, W.; Zhang, C.; Cai, Z.; Long, X.; Chen, H.;
Wang, K.; Yu, G.; Shen, C.; and Shen, S. 2024. Metric3D
v2: A Versatile Monocular Geometric Foundation Model for
Zero-Shot Metric Depth and Surface Normal Estimation.
IEEE Trans. Pattern Anal. Mach. Intell., 46(12): 10579–
10596.
Huang, Y.; Thammatadatrakoon, A.; Zheng, W.; Zhang, Y.;
Du, D.; and Lu, J. 2025. GaussianFormer-2: Probabilistic
Gaussian Superposition for Efficient 3D Occupancy Predic-
tion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 27477–27486. CVPR.
Huang, Y.; Zheng, W.; Zhang, Y.; Zhou, J.; and Lu, J. 2023.
Tri-Perspective View for Vision-Based 3D Semantic Occu-
pancy Prediction. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 9223–
9232. CVPR.
Huang, Y.; Zheng, W.; Zhang, Y.; Zhou, J.; and Lu, J. 2024.
GaussianFormer: Scene as Gaussians for Vision-Based 3D
Semantic Occupancy Prediction.
In Proceedings of the
European Conference on Computer Vision, volume 15085,
376–393. ECCV.
Jia, X.; Jian, S.; Tan, Y.; Che, Y.; Chen, W.; and Liang, Z.
2025. Gated Cross-Attention Network for Depth Comple-
tion. In Proceedings of the IEEE International Conference
on Acoustics, Speech and Signal Processing, 1–5. ICASSP.
Jiang, H.; Cheng, T.; Gao, N.; Zhang, H.; Lin, T.; Liu, W.;
and Wang, X. 2024. Symphonize 3D Semantic Scene Com-
pletion with Contextual Instance Queries. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 20258–20267. CVPR.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian Splatting for Real-Time Radiance Field
Rendering. ACM Trans. Graph., 42(4): 139:1–139:14.
Laina, I.; Rupprecht, C.; Belagiannis, V.; Tombari, F.; and
Navab, N. 2016. Deeper Depth Prediction with Fully Con-
volutional Residual Networks. In 14th International Con-
ference on 3D Vision, 239–248. 3DV.
Li, J.; Han, K.; Wang, P.; Liu, Y.; and Yuan, X. 2020.
Anisotropic Convolutional Networks for 3D Semantic Scene
Completion. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 3348–3356.
ICCV.
Li, J.; Liu, Y.; Gong, D.; Shi, Q.; Yuan, X.; Zhao, C.; and
Reid, I. D. 2019. RGBD Based Dimensional Decomposition
Residual Network for 3D Semantic Scene Completion. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 7693–7702. CVPR.
Li, R.; Lyu, J.; Wang, A.; Yu, R.; Wu, D.; and Xin, B. 2024.
FLAGDroneRacing: An Autonomous Drone Racing Sys-
tem. Unmanned Systems, 12(06): 985–1000.
Li, Y.; Yu, Z.; Choy, C. B.; Xiao, C.; ´Alvarez, J. M.; Fidler,
S.; Feng, C.; and Anandkumar, A. 2023. VoxFormer: Sparse
Voxel Transformer for Camera-Based 3D Semantic Scene
Completion. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 9087–9098.
CVPR.
Lin, T.; Doll´ar, P.; Girshick, R. B.; He, K.; Hariharan, B.;
and Belongie, S. J. 2017. Feature Pyramid Networks for
Object Detection. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, 936–944. CVPR.
Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight De-
cay Regularization. In seventh International Conference on
Learning Representations. ICLR.
Ma, B.; Zhang, J.; Xia, Y.; and Tao, D. 2020. Auto learn-
ing attention. In Advances in neural information processing
systems, volume 33, 1488–1500. NeurIPS.
Mei, J.; Yang, Y.; Wang, M.; Zhu, J.; Ra, J.; Ma, Y.; Li, L.;
and Liu, Y. 2024. Camera-Based 3D Semantic Scene Com-
pletion With Sparse Guidance Network. IEEE Transactions
on Image Processing, 33: 5468–5481.
Miao, R.; Liu, W.; Chen, M.; Gong, Z.; Xu, W.; Hu, C.; and
Zhou, S. 2023.
Occdepth: A depth-aware method for 3d
semantic scene completion. arxiv:2302.13540.
Rold˜ao, L.; de Charette, R.; and Verroust-Blondet, A. 2020.
LMSCNet: Lightweight Multiscale 3D Semantic Comple-
tion. In 8th International Conference on 3D Vision, 111–
119. 3DV.
Ronneberger, O.; Fischer, P.; and Brox, T. 2015.
U-Net:
Convolutional Networks for Biomedical Image Segmenta-
tion. In Medical Image Computing and Computer-Assisted
Intervention, volume 9351, 234–241. MICCAI.
Shi, Y.; Cheng, T.; Zhang, Q.; Liu, W.; and Wang, X. 2024.
Occupancy as set of points.
In Proceedings of the Euro-
pean Conference on Computer Vision, volume 15119, 72–
87. ECCV.
Song, S.; Yu, F.; Zeng, A.; Chang, A. X.; Savva, M.; and
Funkhouser, T. A. 2017. Semantic Scene Completion from
a Single Depth Image. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition, 190–198.
CVPR.
Tan, M.; and Le, Q. V. 2019. EfficientNet: Rethinking Model
Scaling for Convolutional Neural Networks. In Proceedings
of the 36th International Conference on Machine Learning,
volume 97, 6105–6114. PMLR.
Tang, P.; Wang, Z.; Wang, G.; Zheng, J.; Ren, X.; Feng, B.;
and Ma, C. 2024. SparseOcc: Rethinking Sparse Latent Rep-
resentation for Vision-Based Semantic Occupancy Predic-
tion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 15035–15044. CVPR.

<!-- page 13 -->
Tian, X.; Jiang, T.; Yun, L.; Mao, Y.; Yang, H.; Wang, Y.;
Wang, Y.; and Zhao, H. 2023. Occ3D: A Large-Scale 3D
Occupancy Prediction Benchmark for Autonomous Driving.
In Advances in Neural Information Processing Systems, vol-
ume 36, 64318–64330. NeurIPS.
Tong, W.; Sima, C.; Wang, T.; Chen, L.; Wu, S.; Deng, H.;
Gu, Y.; Lu, L.; Luo, P.; Lin, D.; and Li, H. 2023. Scene as
Occupancy. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, 8372–8381. ICCV.
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. At-
tention is All you Need. In Advances in Neural Information
Processing Systems, volume 30, 5998–6008. NeurIPS.
Wang, F.; Zhang, D.; Zhang, H.; Tang, J.; and Sun, Q. 2023.
Semantic Scene Completion with Cleaner Self. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 867–877. CVPR.
Wang, H.; Wei, X.; Zhang, X.; Li, J.; Bai, C.; Li, Y.; Lu, M.;
Zheng, W.; and Zhang, S. 2025a. EmbodiedOcc++: Boost-
ing Embodied 3D Occupancy Prediction with Plane Regu-
larization and Uncertainty Sampler. In Proceedings of the
33rd ACM International Conference on Multimedia. MM.
Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.;
and Novotn´y, D. 2025b. VGGT: Visual Geometry Grounded
Transformer. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 5294–5306.
CVPR.
Wang, J.; Liu, Z.; Meng, Q.; Yan, L.; Wang, K.; Yang, J.;
Liu, W.; Hou, Q.; and Cheng, M.-M. 2024a. Opus: occu-
pancy prediction using a sparse set. In Advances in Neu-
ral Information Processing Systems, volume 37, 119861–
119885. NeurIPS.
Wang, T.; Mao, X.; Zhu, C.; Xu, R.; Lyu, R.; Li, P.; Chen, X.;
Zhang, W.; Chen, K.; Xue, T.; Liu, X.; Lu, C.; Lin, D.; and
Pang, J. 2024b. EmbodiedScan: A Holistic Multi-Modal 3D
Perception Suite Towards Embodied AI. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 19757–19767. CVPR.
Wang, Y.; Chen, Y.; Liao, X.; Fan, L.; and Zhang, Z. 2024c.
PanoOcc: Unified Occupancy Representation for Camera-
based 3D Panoptic Segmentation.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 17158–17168. CVPR.
Wang, Y.; Tan, D. J.; Navab, N.; and Tombari, F. 2019.
ForkNet: Multi-Branch Volumetric Semantic Completion
From a Single Depth Image.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
8607–8616. ICCV.
Wei, Y.; Zhao, L.; Zheng, W.; Zhu, Z.; Zhou, J.; and Lu,
J. 2023. SurroundOcc: Multi-Camera 3D Occupancy Pre-
diction for Autonomous Driving.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
21672–21683. ICCV.
Wu, Y.; Zheng, W.; Zuo, S.; Huang, Y.; Zhou, J.; and Lu, J.
2025. EmbodiedOcc: Embodied 3D Occupancy Prediction
for Vision-based Online Scene Understanding. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision. ICCV.
Yan, X.; Gao, J.; Li, J.; Zhang, R.; Li, Z.; Huang, R.; and
Cui, S. 2021. Sparse Single Sweep LiDAR Point Cloud Seg-
mentation via Learning Contextual Shape Priors from Scene
Completion. In Proceedings of the AAAI Conference on Ar-
tificial Intelligence, 3101–3109. AAAI Press.
Yang, L.; Kang, B.; Huang, Z.; Zhao, Z.; Xu, X.; Feng, J.;
and Zhao, H. 2024.
Depth anything v2.
In Advances in
Neural Information Processing Systems, volume 37, 21875–
21911. NeurIPS.
Yang, X.; Zou, H.; Kong, X.; Huang, T.; Liu, Y.; Li, W.;
Wen, F.; and Zhang, H. 2021.
Semantic Segmentation-
assisted Scene Completion for LiDAR Point Clouds.
In
IEEE/RSJ International Conference on Intelligent Robots
and Systems, 3555–3562. IROS.
Yi, P.; Lei, J.; Hong, Y.; and Chen, J. 2025. Embodied Intel-
ligent Game: Models and Algorithms for Autonomous Inter-
actions Among Heterogeneous Agents. Unmanned Systems,
13(05): 1365–1394.
Yu, H.; Wang, Y.; Chen, Y.; and Zhang, Z. 2024a. Monocu-
lar occupancy prediction for scalable indoor scenes. In Pro-
ceedings of the European Conference on Computer Vision,
volume 15088, 38–54. ECCV.
Yu, Z.; Shu, C.; Deng, J.; Lu, K.; Liu, Z.; Yu, J.; Yang, D.;
Li, H.; and Chen, Y. 2023.
Flashocc: Fast and memory-
efficient occupancy prediction via channel-to-height plugin.
arxiv:2311.12058.
Yu, Z.; Zhang, R.; Ying, J.; Yu, J.; Hu, X.; Luo, L.; Cao, S.-
Y.; and Shen, H.-l. 2024b. Context and Geometry Aware
Voxel Transformer for Semantic Scene Completion.
In
Advances in Neural Information Processing Systems, vol-
ume 37, 1531–1555. NeurIPS.
Zammit, C.; and van Kampen, E.-J. 2023.
Real-time 3D
UAV Path Planning in Dynamic Environments with Uncer-
tainty. Unmanned Systems, 11(03): 203–219.
Zhang, P.; Liu, W.; Lei, Y.; Lu, H.; and Yang, X. 2019.
Cascaded Context Pyramid for Full-Resolution 3D Seman-
tic Scene Completion.
In Proceedings of the IEEE/CVF
International Conference on Computer Vision, 7800–7809.
ICCV.
Zhang, Y.; Zhu, Z.; and Du, D. 2023. Occformer: Dual-path
transformer for vision-based 3d semantic occupancy predic-
tion. In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision, 9433–9443. ICCV.
Zhang, Z.; Zhang, Q.; Cui, W.; Shi, S.; Guo, Y.; Han, G.;
Zhao, W.; Ren, H.; Xu, R.; and Tang, J. 2025. Roboocc:
Enhancing the geometric and semantic scene understanding
for robots. arxiv:2504.14604.
Zhao,
L.;
Wei,
S.;
Hays,
J.;
and
Gan,
L.
2025.
GaussianFormer3D: Multi-Modal Gaussian-based Seman-
tic Occupancy Prediction with 3D Deformable Attention.
arxiv:2505.10685.
Zheng, Z.; Cao, W.; Kubota, Y.; Nakano, Y.; Gao, S.; and
Suzuki, T. 2025. Robust and Energy-Efficient Torque Vec-
toring for a Four in-Wheel Motor Electric Vehicle Based

<!-- page 14 -->
on Sliding Mode and Model Predictive Control. Unmanned
Systems, 13(06): 1699–1712.
Zhu, X.; Su, W.; Lu, L.; Li, B.; Wang, X.; and Dai, J. 2021.
Deformable DETR: Deformable Transformers for End-to-
End Object Detection. In Proceedings of the nineth Interna-
tional Conference on Learning Representations. ICLR.
