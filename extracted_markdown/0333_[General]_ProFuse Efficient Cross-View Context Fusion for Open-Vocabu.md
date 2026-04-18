<!-- page 1 -->
PROFUSE: Efficient Cross-View Context Fusion for Open-Vocabulary 3D
Gaussian Splatting
Yen-Jen Chiou
Wei-Tse Cheng
Yuan-Fu Yang
National Yang Ming Chiao Tung University
remi.ii13@nycu.edu.tw, andy5552555.ii13@nycu.edu.tw, yfyangd@nycu.edu.tw
Figure 1. Overview of ProFuse. Left: A dense matcher supplies cross-view geometric and semantic correspondences. Top: Warped
masks are grouped into 3D Context Proposals with a shared global feature. Bottom: Triangulated matches initialize a compact Gaussian
scene, and proposal features are fused without render supervision for coherent open-vocabulary 3D semantics.
Abstract
We present ProFuse, an efficient context-aware frame-
work for open-vocabulary 3D scene understanding with 3D
Gaussian Splatting (3DGS). The pipeline enhances cross-
view consistency and intra-mask cohesion within a direct
registration setup, adding minimal overhead and requiring
no render-supervised fine-tuning. Instead of relying on a
pretrained 3DGS scene, we introduce a dense correspon-
dence–guided pre-registration phase that initializes Gaus-
sians with accurate geometry while jointly constructing 3D
Context Proposals via cross-view clustering. Each proposal
carries a global feature obtained through weighted aggre-
gation of member embeddings, and this feature is fused
onto Gaussians during direct registration to maintain per-
primitive language coherence across views.
With asso-
ciations established in advance, semantic fusion requires
no additional optimization beyond standard reconstruc-
tion, and the model retains geometric refinement without
densification.
ProFuse achieves strong open-vocabulary
3DGS understanding while completing semantic attach-
ment in about five minutes per scene, which is 2× faster
than SOTA. Additional details are available at our project
page https://chiou1203.github.io/ProFuse/.
1
arXiv:2601.04754v2  [cs.CV]  18 Jan 2026

<!-- page 2 -->
1. Introduction
Open-vocabulary 3D scene understanding aims to under-
stand a physical scene using free-form natural language
queries, with applications ranging from robotics and au-
tonomous navigation to augmented reality [5, 11, 30, 40,
41, 43]. The task remains challenging, as the system must
recover accurate geometry while also assigning meaning-
ful semantic concepts without being restricted to fixed la-
bels. Earlier efforts explored a range of 3D representations
[10, 12, 15, 23, 25, 33, 40]. Recent work has focused on 3D
Gaussian Splatting [14], which represents a scene as a set of
anisotropic Gaussians and enables photo-realistic, real-time
rendering.
Early work adopts 2D vision–language distillation in
which images are rendered during training and Gaussian
features are optimized to match 2D predictions [9, 27,
34, 42, 44]. This pipeline can propagate open-vocabulary
knowledge into 3D, but it also introduces two structural is-
sues. The supervision signal is delivered only after render-
ing and compositing, leading to mismatches with the origi-
nal language embedding that described the region. In addi-
tion, semantics are acquired and queried through individual
views, making reasoning less direct and less stable. These
limitations have motivated methods that operate directly in
3D Gaussian space [13, 20, 28, 39]. These approaches as-
sign language features to each Gaussian and answer a text
query by comparing the query embedding with those per-
Gaussian features in 3D.
More recent work has moved toward a registration-
based formulation [13].
This approach bypasses render-
supervised semantic training.
Language-aligned features
are directly registered in Gaussians using their visibility
along each viewing ray. The result is a compact, queryable
3D semantic field with high efficiency.
Despite such
progress, the direct registration paradigm is still in its early
stages. Our aim is to strengthen the registration framework
by injecting semantic consistency into the 3DGS represen-
tation without any additional render-supervised training.
We propose a registration-based framework ProFuse that
strengthens semantic coherence in 3D Gaussian Splatting.
Our key insight is to enforce two key factors highlighted by
previous work [32, 35, 39, 42], namely cross-view consis-
tency and intra-mask cohesion. Prior approaches typically
encourage these properties through render-supervised train-
ing on 2D feature maps or through explicit feature-learning
objectives. The registration pipeline does not impose these
constraints. Our approach injects these forms of semantic
consistency directly into the registration framework.
An overview of the proposed pipeline is shown in Fig-
ure 1. We introduce a pre-registration stage guided by dense
multi-view correspondence [8]. The correspondence signal
initializes the 3D Gaussian scene with accurate geometry
[17], which allows the representation to cover the scene
without relying on iterative densification. The same sig-
nal is also used to connect observations of the same object
across different viewpoints, consolidating them into con-
sistent, object-level groups that we refer to as 3D Context
Proposals. Each 3D Context Proposal encodes an object
as it appears across views, rather than as an isolated per-
frame mask, and provides a stable source of semantics that
is aligned across viewpoints.
During feature registration, each proposal carries a
global language feature computed from its mask members.
We then assign each Gaussian to its corresponding context
proposals and associate the global semantics to the Gaus-
sian. Notably, our method does not involve gradient-based
fine-tuning or backpropagation of language loss. Through
experiments across open-vocabulary 3D perception tasks,
we demonstrate effectiveness in 3D object selection, open-
vocabulary point cloud understanding, and optimizing effi-
ciency. Our contributions are summarized as follows:
• A registration-based semantic augmentation of 3D Gaus-
sian Splatting that introduces cross-view semantic con-
sistency and intra-mask coherence without any render-
supervised training for semantics.
• A pre-registration stage driven by dense multi-view cor-
respondence. The same correspondence signal initializes
a well-covered 3D Gaussian scene and assembles consis-
tent mask evidence across views into 3D Context Propos-
als.
• A unified open-vocabulary 3D scene representation that
improves object selection, point cloud understanding, and
training efficiency on existing benchmarks while main-
taining render-free semantic association efficiently.
Overall, ProFuse offers a compact and training-free route
to consistent open-vocabulary 3D scene understanding built
directly on correspondence-driven registration.
2. Related Work
Neural rendering has progressed from NeRFs to explicit
point-based primitives [1, 21, 22].
3DGS provides fast,
spatially local rendering and is now a common back-
bone for open-vocabulary understanding [14, 36]. Render-
supervised distillation methods transfer 2D vision-language
signals into 3D by supervising rendered feature maps [9,
12, 15, 25, 27, 29, 34, 35, 37, 44]. Direct 3D retrieval at-
taches language-aligned descriptors to Gaussians or points
for volumetric querying [13, 20, 28, 39]. To stabilize se-
mantics across views, recent works encourage cross-view
consistency and semantic cohesion [3, 4, 12, 18, 20, 25, 26,
35, 37, 39, 42].
Finally, dense correspondence provides
wide-baseline matches and confidences useful for multi-
view grouping and correspondence-driven 3DGS initializa-
tion [2, 7, 8, 17, 19, 31, 38]. We build on this direction
to couple correspondence-guided context association with
registration-based semantic field.
2

<!-- page 3 -->
Figure 2. Pre-registration. For each reference view we select K neighbors via view clustering, then apply a pre-trained dense matcher
to obtain per-pixel warps Wj→i and confidences αj→i. Bottom right: Given the warps of a pixel pair, we triangulate a 3D seed point for
Gaussian initialization. Top right: Warped IoU comparison on every reference–neighbor mask pair; masks that pass the selection form
edges of a bipartite graph.
3. Method
We construct a semantic 3D Gaussian scene that can
be queried with natural language without any render-
supervised semantic training. The pipeline begins with a
pre-registration stage via dense correspondence. This stage
initializes a dense Gaussian scene and links segmentation
masks across views to form 3D Context Proposals. Each
proposal records which masks across views are inferred to
refer to the same scene content, giving us cross-view group-
ings before any semantic fusion. A context-guided regis-
tration stage then uses these proposals to compute a global
language feature for each proposal. The features are then
assigned to the corresponding Gaussians using visibility-
based weights derived from transmittance and opacity along
camera rays. The final output is a 3D representation with
cross-view consistency and intra-mask cohesion that can be
searched directly in 3D by a text query.
3.1. Dense Correspondence Pre-registration
The pre-registration process begins from a set of posed
RGB images of a scene. Let {Ii}N
i=1 denote input views,
and let each image Ii have known camera intrinsics and ex-
trinsics. The goal of this stage is to initialize a dense set
of 3D Gaussians with accurate geometry and initial appear-
ance attributes, and to record cross-view evidence for se-
mantic grouping. As an overview, the full pre-registration
workflow is visualized in Figure 2.
For each image Ii, we obtain a set of non-overlapping
region masks {M k
i } using SAM [16], where M k
i
∈
{0, 1}H×W is a binary mask for the region k in view i. For
every mask M k
i , we extract a language-aligned feature vec-
tor f k
i ∈RD by cropping the corresponding region in Ii
and encoding it with CLIP [29]. The result is a per-view
dictionary Si = {(M k
i , f k
i ) | k = 1, . . . , Ki}, where Ki is
the number of predicted regions in view i. The sets Si will
later serve as semantic evidence.
Dense Feature Matching.
To relate content across views,
we compute dense correspondences between pairs of im-
ages using a pretrained dense matching network (see Fig-
ure 2) . The network was trained on a coarse layer using
DINOv2 [24] and a fine layer with pyramid convolution.
The result is a robust dense feature matching.
Given two images Ii and Ij, the dense matcher returns
C(Ii, Ij) →Wj→i, αj→i, where Wj→i ∈R2×H×W is a
dense warp field that maps each pixel coordinate (u, v) in
Ij to a subpixel coordinate in Ii, and αj→i ∈RH×W is
a confidence map. Intuitively, Wj→i(u, v) predicts where
the content seen at (u, v) in view j should appear in view
i. The value αj→i(u, v) measures how reliable that match
is. We discard correspondences whose confidence falls be-
low a threshold. The result is a dense set of pixel-to-pixel
matches across views that remains stable under wide view-
point change.
Gaussian Initialization.
We use the high-confidence cor-
respondences to seed 3D Gaussian primitives directly in
space. For a confident match between the pixel (uj, vj) in
view j and its mapped location (ui, vi) in view i, we back-
project both pixels into 3D using known camera poses and
triangulate their intersection. The resulting 3D point be-
comes the initial center of a Gaussian (see Figure 2, bottom
right). Its initial appearance attributes are taken from the
supporting image evidence, and its initial scale and orien-
tation are set to cover a small spatial neighborhood around
that 3D point. Repeating this over correspondences yields
the initial Gaussian set G0 = {gn}, where each gn is a Gaus-
sian primitive with position, scale, orientation, opacity, and
color. Because these Gaussians are instantiated from dense
correspondences rather than grown through iterative densi-
fication, G0 already provides broad and near-uniform spatial
coverage of the scene. Subsequent geometric refinement
adjusts these primitives but does not need to create a large
number of new Gaussians.
3

<!-- page 4 -->
Algorithm 1 Cross-view mask clustering
1: Inputs: per-view sets {Si} with Si = {(M k
i , f k
i )};
dense warp field Wj→i and certainties αj→i; visibility
mask; thresholds τα, τiou, τbox; size gates smin, vmin.
2: Initialize graph G = (V, E) with V ←{(i, k) ∀M k
i },
E ←∅
3: for all ordered view pairs (i, j) do
4:
Γj→i ←[αj→i ≥τα] ∧vis mask
5:
for all mask pairs (M a
i , M b
j ) do
6:
f
M b
j→i ←W(M b
j ; Wj→i)
7:
Oi,a; j,b ←IoU(M a
i ⊙Γj→i, f
M b
j→i ⊙Γj→i)
8:
f
M a
i→j ←W(M a
i ; Wi→j)
9:
Oj,b; i,a ←IoU(M b
j ⊙Γi→j, f
M a
i→j ⊙Γi→j)
10:
Bi,a; j,b ←BBoxIoU(M a
i , f
M b
j→i)
11:
Bj,b; i,a ←BBoxIoU(M b
j , f
M a
i→j)
12:
if Oi,a; j,b ≥τiou and Oj,b; i,a ≥τiou and
Bi,a; j,b ≥τbox and Bj,b; i,a ≥τbox then
13:
Add undirected edge between (i, a) and
(j, b) to E
14:
end if
15:
end for
16: end for
17: Extract connected components {Cm} of G
18: Filter Cm by |Cm| ≥smin and |views(Cm)| ≥vmin
19: P ←{Pm ≡Cm}
20: return P
Cross-view Context Association.
The same correspon-
dence field lets us record which masks from different views
refer to the same scene content. Consider two masks M a
i
from view i and M b
j from view j. We project M b
j into view
i using the warp field Wj→i, producing a warped support
mask in the coordinates of Ii. We then measure how well
this warped support overlaps M a
i , restricted to pixels with
high correspondence confidence αj→i. If the overlap ex-
ceeds a threshold, we register a link that these two masks
are consistent observations of the same underlying scene
content. Repeating this procedure over view pairs accumu-
lates the link set L = {(M a
i , ˜
M b
j→i)}, where each pair in L
indicates strong cross-view agreement between two masks
(see Figure 2, top right).
The pre-registration stage produces two artifacts. The
first is an initialized Gaussian scene G0 created by triangu-
lating dense correspondences. The second is a pool of mask
links across views L that captures which regions per-view
act as the same scene content between viewpoints. Sec-
tion 3.2 addresses how we cluster masks in L into 3D Con-
text Proposals.
3.2. 3D Context Proposals
3D Context Proposals are formed through grouping per-
view masks that mutually support one another under dense
correspondence into stable multi-view units. We realize this
by testing pairwise agreements under correspondence warps
and linking masks that pass mutual gates; connected com-
ponents in the resulting graph define the proposals.
Cross-view Mask Clustering.
Algorithm 1 demonstrates
the clustering procedure. Let a mask node be m = (i, k)
with M k
i ∈{0, 1}H×W . Given a candidate pair (i, a) and
(j, b) with a dense warp Wj→i from view j to i and a cer-
tainty map αj→i, we gate matches using a fixed certainty
threshold τα ∈[0, 1] together with a renderer-derived visi-
bility mask vis mask. The binary gate is defined as
Γj→i = [ αj→i ≥τα ] ∧vis mask.
(1)
The warped support in view i is obtained as
f
M b
j→i = W
 M b
j ; Wj→i

,
(2)
where W denotes bilinear sampling at sub-pixel accuracy.
The confidence-gated overlap in view i is
Oi,a; j,b = IoU

M a
i ⊙Γj→i, f
M b
j→i ⊙Γj→i

.
(3)
We compute a coarse bounding-box agreement Bi,a; j,b =
IoU
 box(M a
i ), box(f
M b
j→i)

and gate links with two
thresholds, τiou for mask overlap and τbox for box overlap.
Agreement is required in both directions, and an undirected
link is accepted only if
Oi,a; j,b ≥τiou
and
Oj,b; i,a ≥τiou,
Bi,a; j,b ≥τbox
and
Bj,b; i,a ≥τbox.
(4)
A graph G = (V, E) is then constructed with vertices
V = {(i, k)}. For every cross-view pair that passes the
mutual gates above, we add an undirected edge to E. The
connected components of G define the raw proposals. Very
small components are removed using two criteria: mini-
mal member count smin and minimal distinct-view support
vmin. Each proposal Pm is represented only by its member-
ship list (i, k), contributing view set, and compact per-view
label maps for efficient lookup.
3.3. Feature Registration
The goal of the registration stage is to assign a unit-
normalized language descriptor to every Gaussian, enabling
text queries to be evaluated directly in 3D. This stage oper-
ates on the initialized Gaussian set G0, calibrated cameras,
the per-view mask dictionary Si = {(M k
i , f k
i )}, and the
proposal set P = {Pm} constructed in §3.2.
4

<!-- page 5 -->
Figure 3. From context proposal to global feature. Left: masks of the same entity are grouped into a 3D Context Proposal. Center: for a
pixel p, the renderer returns the top-K Gaussians with contributions {ωi,p,t}K
t=1, from which the mask mass µ
 M k
i

is computed. Right:
a mass-weighted pool of member mask embeddings forms the proposal feature, which is registered to Gaussians via Eq. (8).
For a view i and a pixel p, the renderer returns the indices
and weights of the top-K Gaussians along the ray, denoted
{(gi,p,t, ωi,p,t)}K
t=1. Their blending contributions are
ωi,p,t = Ti,p,t αi,p,t,
Ti,p,t =
Y
s<t
 1 −αi,p,s

,
(5)
where αi,p,t is the effective opacity and Ti,p,t is the trans-
mittance of the preceding Gaussians on the ray.
Each proposal Pm contains member masks drawn from
multiple views. We compute a scalar mass for every mask
by integrating renderer contributions over the mask pixels
µ(M k
i ) =
X
p∈Ω(M k
i )
K
X
t=1
ωi,p,t.
(6)
The proposal descriptor is a mass-weighted pool of mask
embeddings followed by ℓ2 normalization,
¯fm =
P
(i,k)∈Pm µ(M k
i ) f k
i
P
(i,k)∈Pm µ(M k
i ) f k
i

2
.
(7)
An illustration of this aggregation is provided in Figure 3.
A pixel-wise proposal map Li(p) is constructed for every
training view, assigning each pixel inside a mask to the ID
of its corresponding proposal in P. Pixels outside all masks
receive a null label and are ignored. For each Gaussian g ∈
G0, a feature accumulator A[g] ∈RD and a scalar weight
sum S[g] ∈R≥0 are initialized to zero. For every pixel p
with valid proposal m = Li(p) and each of its top-K hits,
the accumulation step is
A[gi,p,t] ←A[gi,p,t] + ωi,p,t ¯fm,
S[gi,p,t] ←S[gi,p,t] + ωi,p,t.
(8)
This registration step consumes the proposal feature from
Figure 3 and weights it by contributions ωi,p,t.
After processing all views, the descriptor for Gaussian g
is computed as
fg =
A[g]
max(S[g], ε),
ˆfg =
fg
∥fg∥2
,
(9)
with a small ε for numerical stability. The implementation
uses batched gather–scatter operations and relies only on
renderer outputs.
3.4. Inference Procedure
A text query is encoded to fq ∈RD and normalized as ˆfq =
fq/∥fq∥2. Each Gaussian g stores a registered descriptor
from §3.3. Following Dr. Splat [13], Product Quantization
(PQ) is used for memory-efficient retrieval. Descriptors are
stored as FAISS product-quantized codes and decoded to
unit-normalized vectors at query time.
Cosine similarity is used to score Gaussians,
sg =
ˆf ⊤
q ˆfg. A FAISS PQ index over { ˆfg} produces a shortlist
that is re-scored using decoded (full-precision) descriptors.
Selection is performed directly in 3D without any render-
based fine-tuning: a Gaussian is considered active if sg ≥
τact. For visualization in view i, let {(gi,p,t, ωi,p,t)}K
t=1 de-
note the Top-K contributors to pixel p. The activation mask
is defined as
Mi(p) = 1[Ai(p) ≥γ],
(10)
where Ai(p) is the sum of contributions over Top-K hits.
4. Experiments
4.1. Implementation
Experiments are conducted on the LERF-OVS [15] and
ScanNet [6] datasets. All four LERF scenes are used, and
10 scenes are sampled from the ScanNet dataset. SAM-
based segmentation and mask embedding are preprocessed
on 8 NVIDIA H100 GPUs, while all remaining experiments
run on a single A100 GPU.
5

<!-- page 6 -->
Table 1. Evaluation of 3D object selection on LERF-OVS [15] dataset. Scores are averaged per scene and then across scenes. Bold
indicates the best performance.
mIoU ↑
mAcc@0.25 ↑
Method
waldo kitchen
figurines
ramen
teatime
mean
waldo kitchen
figurines
ramen
teatime
mean
LangSplat
9.18
10.16
7.92
11.38
9.66
9.09
11.27
8.93
20.34
12.41
LEGaussians
11.78
17.99
15.79
19.27
16.21
18.18
23.21
26.76
27.12
23.82
OpenGaussian
24.57
53.01
24.44
55.40
39.36
36.36
83.93
39.44
76.27
59.00
Dr. Splat
29.37
51.73
26.32
55.53
40.74
50.00
82.14
40.85
79.66
63.16
ProFuse (Ours)
36.91
56.13
28.16
62.78
46.00
68.18
85.71
39.44
79.66
68.25
Figure 4. Qualitative comparison of object-level semantic queries on the LERF-OVS [15] dataset. Our method produces more accurate
and cleaner object retrieval, showing sharper correspondence between the text query and the selected 3D content.
6

<!-- page 7 -->
Figure 5. Feature visualizations on the ScanNet [6] dataset using registration-based methods. Colors represent normalized language
features transferred to mesh vertices and rendered via a fixed RGB projection. ProFuse produces cleaner regions with sharper boundaries
and fewer speckles.
4.2. Open-Vocabulary 3D Object Selection
We evaluate open-vocabulary 3D object selection on the
four LERF scenes using the official text queries and splits.
Each method outputs a binary activation per frame, while
our pipeline performs selection directly in 3D. Let q ∈RD
be the CLIP text embedding, normalized as ˆq = q/∥q∥2.
Each Gaussian g stores a normalized language feature ˆfg
from registration. Active Gaussians are defined as Gτ =
{ g | ⟨ˆfg, ˆq⟩≥τ }, with a method-specific global threshold
τ. For view i and pixel p, the renderer provides the top-K
Gaussians and weights ωi,p,t . The activation is
Ai(p) =
K
X
t=1
ωi,p,t 1[gi,p,t ∈Gτ] ,
and the mask is c
Mi = 1[ Ai ≥γ ] using a fixed silhouette
threshold γ. A small grid search is used to determine the
global threshold τ for each method. mean IoU is computed
by evaluating intersection-over-union for each query–frame
pair and averaging across all queries and frames in a scene.
The final score is obtained by averaging across the four
scenes. Table 1 reports these quantitative results. The met-
ric mAcc@0.25 is also provided, defined as the fraction of
query–frame pairs with IoU at least 0.25, using the same τ.
Table 2. Open-vocabulary point cloud understanding on ScanNet.
Results use mIoU and mAcc for 19/15/10-class settings.
Method
19 classes
15 classes
10 classes
mIoU↑
mAcc↑
mIoU↑
mAcc↑
mIoU↑
mAcc↑
LangSplat
3.78
9.11
5.35
13.20
8.40
22.06
LEGaussians
3.84
10.87
9.01
22.22
12.82
28.62
OpenGaussian
24.73
41.54
30.13
48.25
38.29
55.19
Dr. Splat
28.40
52.77
32.67
58.53
36.81
66.41
ProFuse (Ours)
30.52
55.32
34.76
60.90
39.74
69.38
Qualitative results are presented in Figure 4. Our method
isolates the queried object with far fewer background activa-
tions, yielding cleaner and more semantically precise selec-
tions. In contrast, Dr. Splat often exhibit ray-like spillovers
into nearby clutter or textured areas.
For instance, the
“Toaster” query incorrectly highlights the entire kettle on
the left, while the “Glass of Water” query becomes dis-
tracted by specular reflections.
4.3. Open-Vocabulary Point Cloud Understanding
The evaluation is conducted on the ScanNet dataset using
the label spaces defined in OpenGaussian [39], considering
class sets of 19, 15, and 10 categories. Each mesh vertex in
the aligned reconstruction is assigned a semantic label, and
7

<!-- page 8 -->
Table 3. Comparison of training requirements and retrieval speed
across 3D scene understanding methods.
Method
Scene
Render supervision
Feature distill.
Query
LERF
NeRF
required
∼24 h
slow
LangSplat
SfM–3DGS
required
∼4 h
slow
LEGaussians
SfM–3DGS
required
∼4 h
slow
OpenGaussian
SfM–3DGS
required
∼1 h
fast
GOI
SfM–3DGS
required
∼12 min
fast
Dr. Splat
SfM–3DGS
none
∼10 min
fast
ProFuse (Ours)
Corr-init 3DGS
none
∼5 min
fast
class names are encoded once into language embeddings
and reused across all methods.
Per-Gaussian language codes are first decoded using
FAISS PQ to obtain cosine logits against class embeddings.
These logits are transferred to mesh vertices through a spa-
tially aware kernel that respects each Gaussian’s full el-
lipsoid. Candidate Gaussians are shortlisted by Euclidean
proximity (K=64), filtered by an elliptical Mahalanobis
gate (σ=3), and weighted by both exp(−1
2d2) and Gaussian
opacity. A softmax over class logits yields per-candidate
class probabilities, and vertex scores are computed as the
weighted sum of all candidates. Because predictions occur
directly in 3D, no rendering is involved during evaluation.
The same kernel and shortlist configuration is applied to ev-
ery method so that performance differences reflect the qual-
ity of the learned Gaussian features rather than variations
in the transfer rule. Ten scenes from ScanNet are sampled
for evaluation, and scores are computed with fixed hyper-
parameters to report average mIoU and mAcc for each class
set. Quantitative results for the 19-, 15-, and 10-class set-
tings are provided in Table 2.
To contextualize point-level scores, we visualize feature
colorings of ScanNet reconstructions and compare them
to the pioneer registration-based baseline Dr. Splat [13] in
Figure 5.
For each scene, we show the reference mesh
view and two pseudo-colored point clouds. Colors are ob-
tained by projecting normalized per-Gaussian features to
three channels and painting the transferred per-vertex fea-
tures; views are matched to the reference for consistent
framing. Dr.Splat tends to produce darker, patchy fragments
and color bleeding near corners, whereas our results exhibit
higher region consistency with large surfaces rendered in
coherent color swaths. We achieve cleaner boundaries at
furniture edges and fixtures with fewer mixed colors at ob-
ject–wall contacts.
4.4. Training Efficiency
The cost of attaching open-vocabulary semantics to a recon-
structed scene is measured in wall-clock time. As shown
in Table 3, render-supervised distillation methods require
hours of processing, and existing registration-based ap-
proaches [13] still take several minutes. ProFuse achieves
the fastest runtime through correspondence-guided initial-
Table 4. Wall-clock comparison of geometry, semantic processing,
and indexing time on the LERF dataset.
Method
Geometry
Semantics
Total
Indexing
OpenGaussian
∼20 m
∼40 m
∼1 h
Codebook
Dr. Splat
∼20 m
∼0 + 10 m
∼30 m
PQ
ProFuse (Ours)
∼2 + 15 m
∼2m + 20 s
∼19 m
PQ
Table 5. Top-K analysis on ScanNet showing mIoU and feature
registration time for registration-based methods.
Method
Top K=10
Top K=20
Top K=40
mIoU↑
time↓
mIoU↑
time↓
mIoU↑
time↓
Dr. Splat
33.82
∼45 s
35.57
∼85 s
36.81
∼165 s
ProFuse (Ours)
39.74
∼25 s
39.74
∼25 s
39.74
∼25 s
ization, which produces a compact Gaussian set without
densification, and through lightweight proposal-level fea-
ture fusion. These components reduce semantic attachment
to about five minutes per scene, making ProFuse 2× faster
than the prior SOTA. Table 4 provides a runtime breakdown
of direct 3D methods. ProFuse reduces scene-specific se-
mantic association to only a few minutes because proposal
construction is lightweight and registration uses simple con-
tribution accumulation without gradient updates. The com-
pact geometry from correspondence-guided initialization
removes densification and further shortens processing time.
4.5. Ablation Study
To isolate the effect of correspondence-guided geometry
and context proposals, we study the impact of the Top-K
Gaussian candidates used during feature registration. Ta-
ble 5 reports mIoU and registration time on ScanNet un-
der three settings K=10, 20, 40. Without context propos-
als, registration-based baselines typically require K=40 to
achieve saturation, indicating weak concentration of seman-
tic mass along the viewing ray. In contrast, ProFuse reaches
its maximum accuracy with K=10. The global proposal
features place most of the mass on the leading few Gaus-
sians, while our correspondence-initialized geometry fur-
ther reduces long-tail ambiguity. As a consequence, larger
K offers no additional benefit, and a compact K=10 is suf-
ficient for both accuracy and speed.
5. Conclusion
ProFuse enforces cross-view semantic consistency in 3DGS
without requiring any render-supervised learning for se-
mantics. Dense correspondences generate 3D Context Pro-
posals, and visibility-weighted fusion yields a coherent se-
mantic field.
Experiments on LERF and ScanNet con-
firm accurate open-vocabulary selection and point-level un-
derstanding, showing that correspondence-guided geometry
provides an efficient path to semantic association in 3DGS.
8

<!-- page 9 -->
References
[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In ICCV, 2021. 2
[2] Naijian Cao, Renjie He, Yuchao Dai, and Mingyi He. Loflat:
Local feature matching using focused linear attention trans-
former. arXiv preprint arXiv:2410.22710, 2024. 2
[3] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xi-
aopeng Zhang, Wei Shen, and Qi Tian.
Segment any 3d
gaussians. In AAAI, 2025. 2
[4] Rohan Chacko, Nicolai Haeni, Eldar Khaliullin, Lin Sun,
and Douglas Lee. Lifting by gaussians: A simple, fast and
flexible method for 3d instance segmentation. arXiv preprint
arXiv:2502.00173, 2025. 2
[5] Jianchuan Chen, Jingchuan Hu, Gaige Wang, Zhonghua
Jiang, Tiansong Zhou, Zhiwen Chen, and Chengfei Lv. Taoa-
vatar: Real-time lifelike full-body talking avatars for aug-
mented reality via 3d gaussian splatting. In CVPR, 2025. 2
[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
CVPR, 2017. 5, 7
[7] Johan Edstedt, Ioannis Athanasiadis, M˚arten Wadenb¨ack,
and Michael Felsberg.
Dkm:
Dense kernelized fea-
ture matching for geometry estimation.
arXiv preprint
arXiv:2202.00667, 2022. 2
[8] Johan
Edstedt,
Qiyu
Sun,
Georg
B¨okman,
M˚arten
Wadenb¨ack, and Michael Felsberg.
Roma: Robust dense
feature matching. arXiv preprint arXiv:2305.15404, 2023. 2
[9] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li.
Semantic gaussians: Open-vocabulary scene understanding
with 3d gaussian splatting. arXiv preprint arXiv:2403.15624,
2024. 2
[10] Qingdong He, Jinlong Peng, Zhengkai Jiang, Kai Wu, Xi-
aozhong Ji, Jiangning Zhang, Yabiao Wang, Chengjie Wang,
Mingang Chen, and Yunsheng Wu.
Unim-ov3d:
Uni-
modality open-vocabulary 3d scene understanding with fine-
grained feature representation. In IJCAI, 2024. 2
[11] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram
Burgard.
Visual language maps for robot navigation.
In
ICRA, London, UK, 2023. 2
[12] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala,
Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer,
Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B.
Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam
Paull, Florian Shkurti, and Antonio Torralba. Conceptfusion:
Open-set multimodal 3d mapping. Robotics: Science and
Systems (RSS), 2023. 2
[13] Kim Jun-Seong, Kim GeonU, Kim Yu-Ji, Yu-Chiang Frank
Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly
referring 3d gaussian splatting via direct language embed-
ding registration. In CVPR, 2025. 2, 5, 8
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2
[15] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In ICCV, 2023. 2, 5, 6
[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi
Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer
Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar,
and Ross Girshick.
Segment anything.
arXiv preprint
arXiv:2304.02643, 2023. 3
[17] Dmytro Kotovenko, Olga Grebenkova, and Bj¨orn Ommer.
Edgs: Eliminating densification for efficient convergence of
3dgs. arXiv preprint arXiv:2504.13204, 2025. 2
[18] Abhijit Kundu, Kyle Genova, Xiaoqi Yin, Alireza Fathi,
Caroline Pantofaru, Leonidas Guibas, Andrea Tagliasacchi,
Frank Dellaert, and Thomas Funkhouser. Panoptic neural
fields: A semantic object-aware neural scene representation.
arXiv preprint arXiv:2205.04334, 2022. 2
[19] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Ground-
ing image matching in 3d with mast3r. In ECCV, 2024. 2
[20] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao
Zhang, Ronggang Wang, and Jian Zhang.
Instancegaus-
sian: Appearance-semantic joint gaussian representation for
3d instance-level perception. In CVPR, 2025. 2
[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2
[22] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. In ACM Trans. Graph., 2022. 2
[23] Phuc D. A. Nguyen, Tuan Duc Ngo, Evangelos Kalogerakis,
Chuang Gan, Anh Tran, Cuong Pham, and Khoi Nguyen.
Open3dis: Open-vocabulary 3d instance segmentation with
2d mask guidance. In CVPR, 2024. 2
[24] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mah-
moud Assran, Nicolas Ballas, Wojciech Galuba, Russell
Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael
Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Je-
gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr
Bojanowski. Dinov2: Learning robust visual features with-
out supervision. arXiv preprint arXiv:2304.07193, 2024. 3
[25] Songyou Peng, Kyle Genova, Chiyu ”Max” Jiang, An-
drea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser.
Openscene: 3d scene understanding with open vocabularies.
In CVPR, 2023. 2
[26] Jens Piekenbrinck, Christian Schmidt, Alexander Hermans,
Narunas Vaskevicius, Timm Linder, and Bastian Leibe.
Opensplat3d: Open-vocabulary 3d instance segmentation us-
ing gaussian splatting.
arXiv preprint arXiv:2506.07697,
2025. 2
[27] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In CVPR, 2024. 2
[28] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Li-
ujuan Cao, Shengchuan Zhang, and Rongrong Ji.
Goi:
Find 3d gaussians of interest with an optimizable open-
9

<!-- page 10 -->
vocabulary semantic-space hyperplane.
arXiv preprint
arXiv:2405.17596, 2024. 2
[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision. arXiv preprint
arXiv:2103.00020, 2021. 2, 3
[30] Adam Rashid, Satvik Sharma, Chung Min Kim, Justin Kerr,
Lawrence Yunliang Chen, Angjoo Kanazawa, and Ken Gold-
berg. Language embedded radiance fields for zero-shot task-
oriented grasping. In CoRL, 2023. 2
[31] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich.
Superglue:
Learning feature
matching with graph neural networks. In CVPR, 2020. 2
[32] Hongyu Shen, Junfeng Ni, Yixin Chen, Weishuo Li, Mingtao
Pei, and Siyuan Huang. Trace3d: Consistent segmentation
lifting via gaussian instance tracing. In ICCV, 2025. 2
[33] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack
Kaelbling, and Phillip Isola.
Distilled feature fields en-
able few-shot language-guided manipulation. arXiv preprint
arXiv:2308.07931, 2023. 2
[34] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan.
Language embedded 3d gaussians for
open-vocabulary scene understanding.
arXiv preprint
arXiv:2311.18482, 2023. 2
[35] Wei Sun, Yanzhao Zhou, Jianbin Jiao, and Yuan Li. Cags:
Open-vocabulary 3d scene understanding with context-
aware gaussian splatting. arXiv preprint arXiv:2504.11893,
2025. 2
[36] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea
Vedaldi.
Splatter image: Ultra-fast single-view 3d recon-
struction. In CVPR, 2024. 2
[37] Ayc¸a Takmaz, Elisabetta Fedele, Robert W. Sumner, Marc
Pollefeys, Federico Tombari, and Francis Engelmann. Open-
mask3d: Open-vocabulary 3d instance segmentation.
In
NeurIPS, 2023. 2
[38] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In CVPR, 2024. 2
[39] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao
Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding,
Jingdong Wang, and Jian Zhang. Opengaussian: Towards
point-level 3d gaussian-based open vocabulary understand-
ing. In NeurIPS, 2024. 2, 7
[40] Kashu Yamazaki, Taisei Hanyu, Khoa Vo, Thang Pham,
Minh Tran, Gianfranco Doretto, Anh Nguyen, and Ngan
Le.
Open-fusion:
Real-time open-vocabulary 3d map-
ping and queryable scene representation.
arXiv preprint
arXiv:2310.03923, 2023. 2
[41] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In CVPR, 2024. 2
[42] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In ECCV, 2024. 2
[43] Hongjia Zhai, Xiyu Zhang, Boming Zhao, Hai Li, Yijia He,
Zhaopeng Cui, Hujun Bao, and Guofeng Zhang. Splatloc: 3d
gaussian splatting-based visual localization for augmented
reality. arXiv preprint arXiv:2409.14067, 2024. 2
[44] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields.
In
CVPR, 2024. 2
10

<!-- page 11 -->
1 
A. Reproducibility and Code Release 
The implementation of ProFuse will be released at 
https://github.com/chiou1203/ProFuse. This repository 
will include the training code for the dense correspondence 
guided Gaussian initialization, cross-view mask clustering, 
and feature registration stages.  
B. Discussion 
B.1. Limitation 
    Although cross-view mask clustering helps associate 
masks that likely refer to the same context, our method still 
remains bounded by SAM and CLIP. Mask embeddings 
still reflect the underlying segmentation quality. Moreover, 
even with accurately segmented masks, CLIP embedding 
errors directly affect 3D scene understanding, particularly 
for similar objects and uncommon text prompts. 
    Though pre-registration is correspondence driven, even 
with warps, mismatches can persist when masks are 
imperfect. For example, under an IoU threshold of 0.5, a 
mask that is 80% a large object and 20% a small object may 
be grouped with other masks of the large object; the pooled 
global feature then inherits contamination from the small 
object. This relates to over-coarse grouping in which masks 
group that are not clean and highly accurate washed out 
fine details and an inaccurate global feature can pollute all 
its mask members.  
B.2. Societal Impact 
The method lowers the barrier to open vocabulary 
understanding in 3D scenes. It attaches language 
descriptors to Gaussians through correspondence driven 
pre-registration and feature registration without a render 
supervised loop. The result is shorter iteration time and a 
lighter compute footprint. These gains translate into 
practical uses. Education and cultural heritage benefit from 
interactive exploration of reconstructed spaces where a 
user can ask for an object and see it in context. AR and VR 
authors gain a searchable index over large captures that 
enables precise selection and editing without project 
specific training. Robotics and digital twins obtain faster 
scene lookup for inventory, maintenance, and task setup in 
indoor environments. Assistive scenarios become more 
responsive since a user can request a target item and 
receive immediate guidance in a captured room. 
Responsible 
deployment 
remains 
straightforward. 
Capture and indexing should follow clear consent. Storage 
and sharing should use established governance in each 
setting. With these norms in place, ProFuse helps 
democratize semantic interaction with 3D content and 
broadens access to practical tools for learning, creation, 
and operation. 
C. Preliminaries 
C.1. 3D Gaussian Splatting 
3D Gaussian Splatting represents a scene with a set of 
anisotropic Gaussians. Each primitive has a mean in world 
space and a covariance that is factorized into a rotation and 
a diagonal scale. This factorization guarantees a valid 
positive semi-definite matrix and is convenient for 
optimization. 
Rendering proceeds by projecting each 3D covariance to 
image space through a first-order camera Jacobian, which 
yields a 2×2 covariance for splatting on the raster plane. 
The pixel color is then obtained by front-to-back alpha 
compositing. 
The 
formulation 
matches 
volumetric 
rendering and can be written as a sum of per-splat 
contributions, where the contribution of the i-th splat 
equals its transmittance times its effective opacity times its 
color. Transmittance accumulates along the ray as the 
product of one minus the previous opacities. 
The original system uses a differentiable tile-based 
rasterizer. Gaussians are culled against the frustum and 
tiles, sorted by depth, and blended per tile to maximize 
parallelism while maintaining the same alpha-compositing 
model. 
C.2. Product Quantization (PQ) 
    ProFuse 
follow 
Dr.Splat 
who 
utilize 
Product 
Quantization to store and search language features 
efficiently without per-scene codebook training. PQ 
partitions a D-dimensional vector into L sub-vectors, 
learns a codebook per subspace, and represents each sub-
vector by the index of its nearest centroid. This reduces 
memory and turns distance or similarity computation into 
table lookups across subspaces. 
    After training centroids, a lookup table stores all 
pairwise distances among centroids in each subspace. The 
distance between two PQ-encoded vectors becomes a sum 
ProFuse: Efficient Cross-View Context Fusion  
for Open-Vocabulary 3D Gaussian Splatting 
Supplementary Material

<!-- page 12 -->
2 
of L table entries, one per subspace. Cosine similarity can 
be computed in the same way using inner-product tables 
after normalizing subvectors. This design shifts the cost 
from high-dimensional arithmetic to indexed retrieval 
while preserving correlation with true distances within 
known quantization bounds. 
    Significant search-time gains over direct cosine 
similarity on CLIP features can be observed when varying 
the sub-vector size. These measurements demonstrate that 
LUT-based PQ search scales well for large 3D Gaussian 
sets and supports interactive text-to-3D queries. 
D. Implementation Details 
D.1. Correspondence Driven Gaussians 
    Unlike standard 3DGS, we initialize the scene from 
dense cross-view correspondences and then perform a 
pruning-only optimization without densification. A trainer 
calls a correspondence-based initializer and proceeds with 
a photometric objective under a fixed training schedule. 
We first sample a compact set of reference views by K-
means clustering in pose space to cover the trajectory with 
minimal redundancy. For each reference we attach a small 
pose-nearest neighbor set, selected by distance in the same 
pose space. We sample a set of reference views and attach 
a small pose-nearest set of neighbors to each reference. In 
our runs, we use 180 reference views and 3 neighbors per 
reference for maximum efficiency. The initializer 
computes a dense warp field and certainties of each warp, 
aggregates the most confident warp per pixel, and 
triangulates 15,000 correspondences per reference to seed 
Gaussians. The seed carries position, color, and scale from 
the paired views, followed by standard splat optimization. 
We enable pruning while disabling densification during 
training, removing poor splats without ever growing new 
ones. Opacity resets are effectively off through a very large 
reset interval. We use a batch size of 64 and run 30,000 
iterations per scene. This procedure typically yields around 
2×10⁶ initialized Gaussians and roughly 5×10⁵ to 10⁶ active 
Gaussians after pruning 
D.2. The ProFuse Framework 
ProFuse attaches language descriptors to a Gaussian 
scene that is initialized from dense correspondences and 
refined with pruning only. The representation follows 
standard 3D splatting for geometry and visibility. The 
semantic path operates on masks and text features and 
produces a per-Gaussian descriptor that supports open-
vocabulary queries without a render-supervised loop. 
Masks come from the object level of SAM. Each mask 
is encoded by CLIP ViT-H/14 with a 512-dimensional 
embedding. Per-view features are fused to Gaussians using 
the Top-K ray contributions from the renderer so the same 
weights that produce color also produce language features. 
The fusion creates a single descriptor per Gaussian that is 
shared across views and does not depend on any prompt at 
training time. 
Descriptors are stored with Product Quantization. We 
use a global codebook that is shared across all scenes and 
keep the PQ codes as the only per-Gaussian semantic 
payload during training and inference. The system 
reconstructs codes to unit-norm vectors for cosine scoring 
when answering a text query. This design reduces memory 
and enables fast similarity evaluation while keeping the 
scoring rule identical to the one used for visualization and 
selection. 
 
D.3.  Compare Model Settings 
LangSplat. LangSplat learns a 3D language field on 3D 
Gaussians and replaces NeRF rendering with tile-based 
splatting for language features. It builds a scene-wise 
language autoencoder and trains language features in a 
scene-specific latent space rather than directly on CLIP 
space, which reduces memory. Supervision comes from 
SAM to form hierarchical semantics so that subpart, part, 
and whole concepts are separable. The paper reports large 
speed gains over LERF at high resolution, which is 
consistent with the splatting design and the latent-space 
training. 
LEGaussians. 
LEGaussians 
discretizes 
language 
features with a learnable codebook and stores indices rather 
than full float descriptors. Quantization selects the nearest 
basis in a discrete feature space using a CLIP term and a 
controllable DINO term; optimization aligns dense image 
features to their quantized counterparts. During training the 
method renders compact semantic vectors from Gaussians 
and decodes them with a small MLP under a cross-entropy 
objective. It further adds adaptive spatial smoothing driven 
by a learned per-Gaussian uncertainty so that semantics 
vary smoothly where features are unstable. These design 
choices reduce storage and regularize multi-view 
inconsistency. 
OpenGaussian. 
OpenGaussian 
augments 
each 
Gaussian with a low-dimensional instance feature and 
learns it by rendering feature maps with alpha blending. 
Supervision uses SAM boolean masks without cross-view 
correlation. The loss encourages intra-mask smoothness 
and inter-mask separation so that features within an object 
cluster together while different objects separate in feature 
space. To discretize for efficient retrieval, a two-level 
codebook is constructed in a coarse-to-fine manner. The 
coarse stage clusters by concatenating position with 
features; the fine stage clusters by features only, which 
preserves geometry and improves scalability in larger 
scenes. The paper also proposes an instance-level 
association that links 2D CLIP to 3D points without 
additional training.

<!-- page 13 -->
3 
Dr. Splat. Dr. Splat performs direct feature registration 
on pre-trained 3DGS scenes. Per-pixel CLIP embeddings 
are aggregated onto the dominant top-k Gaussians along 
each camera ray with weights equal to transmittance times 
effective opacity from the volume rendering equation. The 
aggregated embeddings are product-quantized and stored 
as PQ indices, enabling compact storage and fast 3D search 
without per-scene feature distillation. The paper contrasts 
this registration-based pipeline with rendering-supervised 
methods and reports substantially shorter end-to-end 
preparation and query times. 
D.4. Training Details 
    Dense Correspondence. The settings for dense 
correspondence are shown in Table 1. We select 180 
reference views by K-means in pose space and attach 3 
pose-nearest neighbors to each reference by k-NN. We use 
RoMa as the pretrained network for dense matches.    We 
cap the sampling at 15,000 matches per reference. 
Certainty is aggregated by a per-pixel maximum across 
neighbors.  
Gaussian Initialization. The Gaussian initialization and 
optimization settings are summarized in Table 2. Each 
correspondence track is triangulated with calibrated 
cameras, and we keep only tracks whose mean reprojection 
error is below 0.01 in normalized image coordinates. For 
every surviving 3D point we create a Gaussian with 
spherical covariance, where the initial scale parameters are 
set to 0.001 in scene units.  
Gaussian Optimization. The initialized scene is then 
optimized with the 3DGS training loop for 30,000 
iterations with batch size 64. The position learning rate 
starts at 1.6×10⁻⁴ and decays to 1.6×10⁻⁶ over 30,000 steps 
with a delay multiplier of 0.01. The feature learning rate is 
0.0025, the opacity learning rate is 0.025, the scaling 
learning rate is 0.005, and the rotation learning rate is 0.001. 
We keep the dense ray sampling ratio at 1 % of pixels per 
iteration (percent_dense = 0.01) and weight the DSSIM 
term by 0.2. Densification itself is disabled by setting 
no_densify to True. The opacity reset interval is extended 
to 1,000,000 iterations so that no opacity reset occurs 
during optimization. 
Cross-View Mask Clustering. After dense feature 
matching, we cluster masks that likely depict the same 
scene region across views. For each reference image, we 
use SAM mask level 1 as the object-level partition. The 
reference segmentation is resized to the RoMa canvas 
resolution, and each neighbor segmentation is projected 
into this canvas using the warp. We prune projected labels 
with the Gaussian visibility mask, using a transmittance 
threshold of 0.05. The warped IoU threshold for cluster 
edges are set to 0.2 and bounding box IoU 0.08.  Very small 
masks that cover less than 0.5% of the canvas use a stricter 
IoU requirement of 0.30 in order to avoid spurious links 
caused by noise. Connected components in this graph 
define the cross-view clusters, and for each reference 
image we store the cluster assignments and corresponding 
SAM label ids in a NPZ file that is reused by the 
registration stage. 
Table 1: Dense correspondence setting. 
Config 
Value 
Dense Matching Network 
RoMa 
Total References 
180 
Neighbors per Reference 
3 
Max Matches per Reference 
15,000 
Table 2: Gaussian pre-training setting. 
Config 
Value 
Triangulation Reprojection 
Tolerance 
0.01 
Initial Scale 
0.001 
Training Iterations 
30,000 
Batch Size 
64 
Optimizer 
Adam 
Base Position Learning Rate 
1.6×10⁻⁴ 
Delay Multiplier 
0.01 
Feature Learning Rate 
0.0025 
Opacity Learning Rate 
0.0025 
Scaling Learning Rate 
0.005 
Rotation Learning Rate 
0.001 
Percent Dense Pixels 
0.01 
DSSIM weight 
0.2 
Densification 
Disabled 
Table 3: Cross-view mask clustering setting. 
Config 
Value 
SAM Mask Level 
1 (Object) 
Visibility Threshold 
0.05 
Warped IoU 
0.2 
Bounding Box IoU 
0.08 
Small Mask Fraction 
0.005 
Small Mask IoU 
0.3 
Edge Requirement 
Mutual Best Neighbor 
Table 4: Feature registration setting. 
Config 
Value 
Feature Level 
1  
Top-K 
10 
PQ Index 
128-D 
Pixel Stride 
1 
SpMM Cluster Block 
0 
Eps Contribution 
0

<!-- page 14 -->
4 
Feature Registration. Settings for registration are 
summarized in Table 4. During registration, we freeze the 
Gaussian scene and run a single registration pass that 
attaches a 512-dimensional language descriptor with 
feature level 1 to every Gaussian. Proposal-level feature 
registration is used whenever a valid NPZ file is available 
and the code falls back to per-mask accumulation only 
when a view has no mapped cluster metadata. 
 
For each camera, we render the top-10 Gaussian ids and 
their contributions per pixel from the pretrained scene.  To 
control sampling density on clustered pixels, we introduce 
pixel stride. Valid pixels inside clustered masks are sub-
sampled with a uniform stride in image space, keeping the 
option for heavier scenes if needed. After collecting top-10 
contributions for these pixels, we build a sparse weight 
matrix between Gaussians and cluster ids. Before forming 
this matrix, we apply a small threshold on contributions 
through eps_contrib. Entries with contribution below this 
threshold are discarded, which removes numerically tiny 
pairs that only add memory cost but almost no semantic 
signal. We set default stride to 1 and eps_contrib to 0 in 
reported experiments.  
The accumulation is implemented as a sparse matrix–
dense matrix multiplication over Gaussians and global 
features. An optional parameter spmm_cluster_block 
allows the cluster axis to be processed in blocks when GPU 
memory is tight. Each block builds a smaller sparse matrix 
for a subset of clusters and accumulates the result into the 
global Gaussian buffers. We set spmm_cluster_block = 0 
in all our runs since our scenes fit comfortably within 
memory at this resolution. 
Once the per-Gaussian float features are obtained, we 
normalize them and encode them with product quantization. 
We always enable PQ and load a pretrained FAISS index 
with code size 128. The final stored language descriptor of 
each Gaussian is a 128-byte PQ code. 
 
D.5. Evaluation Details 
Open-Vocabulary 3D Object Selection. We adopt the 
LERF object selection benchmark and use the same four 
scenes, prompts, and binary ground-truth masks as in 
Section 4.2. For each method we compute cosine similarity 
between every Gaussian language descriptor and the CLIP 
text embedding of the query. Gaussians are activated when 
the similarity exceeds a threshold 𝜏. In practice, 𝜏 is chosen 
for each method by grid search, sweeping values in steps 
of 0.01 and fixing the best value across all scenes. The re-
ranking stage follows the LERF relative-relevance 
formulation with the canonical word list {object, things, 
stuff, texture} and temperature 𝜏rerank = 3.0 . After 
selecting Gaussians, we aggregate their per-pixel 
contribution weights thresholding at a contribution level 
𝛾= 0.025. We then compute mIoU and mAcc@0.25 on 
the same set as LERF and average the scores over all 
scenes.  
Open-Vocabulary 3D Point Cloud Understanding. 
For ScanNet dataset, we evaluate all methods on the 19-
class, 15-class, and 10-class label sets with a strict point-
level protocol. For each checkpoint we decode the 512-
dimensional PQ language features back to float vectors 
using the original FAISS index. Codes that are all-255 or 
mapped to an invalid IVF list are treated as invalid, 
decoded features are L2-normalized, and invalid rows are 
kept as zeros. Class text features are loaded from an JSON, 
aligned to the class name list for each label set, and L2-
normalized. We form cosine logits between all Gaussians 
and all class text features and transfer these logits to mesh 
vertices using the rotation- and scale-aware Mahalanobis 
kernel with opacity weighting. The kernel uses a shortlist 
of 𝑘shortlist = 64 Gaussian candidates per point, a gating 
radius 𝜎gate = 3.0 in Mahalanobis distance, and a SoftMax 
temperature logit_temp = 1.0; these values are shared by 
the 19-class, 15-class, and 10-class evaluators. Points that 
fall outside the 𝜎gate fall back to the nearest valid Gaussian. 
We report point-mIoU and point-mAcc averaged over 
classes that are present in the ground truth and over the 
evaluation scenes listed in Section 4.3. 
 
D.6. Computing Resource Configuration 
  The experiments of ProFuse were conducted on a 
single NVIDIA A100 80 GB GPU. All methods were 
evaluated with their best model on their best threshold to 
maintain consistency. ScanNet scenes are down sample to 
170 ~ 210 images per scene, and each Gaussian scene was 
trained for 30k iterations with the same hyper-parameter 
setting during scene optimization. The experiments on 
registration-based methods were compared using the same 
PQ codebook with sub-vector size 128, and best threshold 
were picked method-wise to respect different model nature, 
ensuring a comprehensive and uniform assessment of 
performance across different architectures. 
 
Table 5: Evaluation setting. 
Config 
Value 
Contribution Threshold 
0.025 
Reranking Temperature 
3 
Gaussian Candidates 
64 
Gating Radius 
3 
SoftMax Temperature 
1

<!-- page 15 -->
5 
E. Additional Experiments 
E.1. Gaussian Scene Experiments 
We investigate how the choice of pretrained Gaussian 
scene affects both reconstruction quality and semantic 
association. Three variants are evaluated under the same 
ProFuse registration pipeline and object–selection protocol, 
as summarized in Tables 6–9. The first variant applies 
ProFuse on top of a standard SfM-based 3D Gaussian 
Splatting scene. Dense correspondence and cross-view 
mask clustering are still computed, but these tracks only 
influence the semantic side; the underlying geometry 
follows the original 3DGS training procedure. The second 
variant is the default ProFuse configuration used in the 
main paper. In this case the Gaussian scene is initialized 
from dense correspondence tracks without densification, 
followed by our 30k-step pruning-only optimization. The 
third variant enables densification on top of the 
correspondence-guided seeds. 
Scene-level mIoU and mAcc@0.25 for the LERF 
object-selection task are reported in Tables 6 and 7. The 
correspondence-guided 
scene 
without 
densification 
consistently improves over the pure 3DGS scene in both 
metrics, and achieves the best mean semantic performance 
across the four scenes. Enabling densification yields mixed 
behavior. Some scenes remain competitive, while others 
suffer from a noticeable drop in mIoU and mAcc. We 
attribute this to the strong variation of the similarity–
threshold curve across scenes. A single global activation 
threshold is applied within each variant, which is a 
reasonable 
choice 
for 
comparison 
but 
cannot 
simultaneously track the per-scene optimum once the 
density pattern of Gaussians changes significantly. 
Table 8 reports PSNR of the rendered views from the 
three pretrained scenes. Here the correspondence-guided 
initialization with densification achieves the highest mean 
PSNR, while the non-densified ProFuse scene is slightly 
below. This indicates that densification still brings benefits 
for pure reconstruction, even when the initial seeds already 
introduce highly accurate geometry. 
Table 9 further compares the optimization time across 
scenes. 
The 
correspondence-guided 
scene 
without 
densification shortens training to roughly 14 minutes on 
average. Enabling densification increases the mean time to 
about 22 minutes, almost doubling the cost on some scenes. 
Combining these trends, the default ProFuse configuration, 
which 
uses 
correspondence-guided 
seeds 
without 
densification, forms a practical compromise. It delivers the 
strongest semantic performance in the object-selection 
benchmark, preserves competitive reconstruction quality, 
and keeps the pretraining time significantly lower than the 
densified alternative. 
Qualitative example of the scene reconstruction 
progress for correspondence-guided 3DGS is illustrated 
in Figure 1. The left shows the reference image. The 
second column renders the raw seeds produced directly 
from dense correspondence before any optimization, 
already capturing the layout of major objects. The third 
column shows the scene after 7k optimization steps of 
pruning, where geometry and appearance become 
noticeably sharper. The right column viualize the result 
after 30k iterations, which mainly refines shading and 
small details. This qualitative behavior is consistent with 
the 
quantitative 
results 
and 
illustrates 
that 
correspondence-guided initialization provides a strong 
geometric prior even without densification 
 
Table 6: Ablation study on scene mIoU for different pretrain 
Gaussian scenes. 
Gaussian 
Scene 
Scene mIoU 
Waldo 
kitchen Figurines 
Ramen 
Teatime 
Mean 
3DGS 
24.45  
55.27  
24.70  
62.57 
41.75 
Corr-init 
36.91  
56.13  
28.16  
62.78 
46.00 
Corr-init 
+ 
Densify 
14.29 
41.31 
28.16 
60.62 
36.10 
Table 7: Ablation study on scene mAcc for different pretrain 
Gaussian scenes. 
Gaussian 
Scene 
Scene mAcc@0.25 
Waldo 
kitchen Figurines 
Ramen 
Teatime 
Mean 
3DGS 
36.36 
83.93 
40.85 
79.66 
59.85 
Corr-init 
68.18 
85.71 
39.44 
79.66 
68.25 
Corr-init 
+ 
Densify 
22.72 
67.86 
39.44 
79.66 
52.42 
 
Table 8: Ablation study on scene PSNR for different pretrain 
Gaussian scenes. 
 
Gaussian 
Scene 
Scene PSNR 
Waldo 
kitchen Figurines 
Ramen 
Teatime 
Mean 
3DGS 
32.89 
24.95 
28.73 
31.39 
29.49 
Corr-init 
32.18 
24.61 
28.84 
31.52 
29.29 
Corr-init 
+ 
Densify 
34.65 
26.34 
30.02 
32.73 
30.94 
 
Table 9: Optimization time analysis on enabling densification for 
Gaussian optimization of ProFuse pretrained scene. 
 
Densification 
Optimization Time 
Waldo 
kitchen Figurines Ramen Teatime 
Mean 
W/O Densify ~13m 
~12m 
~14m 
~15m 
~14m 
Densify 
~27m 
~23m 
~16m 
~24m 
~22m

<!-- page 16 -->
6

<!-- page 17 -->
7 
E.2. Neighbor per Reference  
We study how the number of neighbors per reference 
view influences both semantic understanding and the cost 
of dense correspondence. The experiment varies the 
neighbor count in pre-registration stage, using values 0, 3, 
5, 7, and 9 while keeping all other components fixed. 
Setting the neighbor count to 0 disables cross-view mask 
clustering and therefore removes 3D Context Proposals, 
leaving 
solely 
correspondence-guided 
Gaussian 
initialization. The remaining settings preserve both the 
correspondence-based initialization and the context 
proposal pipeline, with increasingly large neighborhood 
graphs. 
Tables 11 and 12 report scene-level mIoU and 
mAcc@0.25 on the LERF object selection task. 
Comparing 0 and 3 neighbors shows that introducing even 
a small neighborhood already improves mIoU noticeably, 
which confirms that the gain of ProFuse over the 
registration baseline is not explained only by a different 
pretrained Gaussian scene. The additional global feature 
injected by 3D context proposals genuinely strengthens 
semantic association. As the neighbor count grows beyond 
3, several scenes exhibit further improvements in mIoU, 
while the Figurines scene becomes slightly less stable, 
suggesting that very large neighborhoods may introduce 
noisy cross-view links for cluttered layouts. Overall 
increase in mIoU indicates that richer cross-view evidence 
can still benefit the registration stage. 
The mAcc@0.25 curves show a more nuanced 
behavior. Waldo kitchen is a representative example. Its 
mIoU increases significantly when the neighbor count 
exceeds 3, yet its mAcc decreases slightly. This apparent 
contradiction is explained by the threshold-selection 
procedure. For each variant we fix a single activation 
threshold shared across scenes. The mIoU of Waldo 
kitchen as a function of the threshold forms a curve whose 
peak 
shifts 
toward 
lower 
thresholds 
when 
the 
neighborhood grows. The global threshold chosen for the 
ablation lies farther from this new peak, so the reported 
mAcc does not fully reflect the best possible accuracy of 
the scene. In practice Waldo kitchen can reach mIoU above 
46 and mAcc above 68 under a threshold tuned specifically 
for that configuration, which is consistent with the 
improved curve. 
Table 13 summarizes the initialization time of the 
dense correspondence stage. The cost grows steadily with 
the neighbor count, since every additional neighbor 
requires extra RoMa evaluation and mask projection. 
Different scenes exhibit slightly different sensitivity, but 
the trend is consistent. When we compare the average 
semantic gains against the additional time, three neighbors 
per reference offers a favorable trade-off, keeping the pre-
registration stage within a practical budget. 
 
 
 
E.3. Additional Ablations 
More Details on Top-K Choice. Top-K ablations for 
ScanNet point-cloud understanding with 10 labeled classes 
were discussed in Section 4.5. The same behavior also 
applied to 19 and 15 classes label sets. Under the same 
evaluation protocol, ProFuse reaches its best or near-best 
mIoU and mAcc within top-10 Gaussians, while the Dr. 
Splat baseline continues to improve when K is increased 
and usually needs K=40 to approach its own peak. This 
contrast indicates that our proposal-based registration 
concentrates the useful semantic mass on a much smaller 
subset of Gaussians along each ray and is therefore far 
more efficient.  
Table 10: Ablation study on scene mIoU for different neighbor 
per reference. 
 
 
Neighbors  
Scene mIoU 
Waldo 
kitchen Figurines Ramen 
Teatime 
Mean 
0 
33.80 
48.28 
25.09 
57.57 
41.19 
3 
36.91 
56.13 
28.16 
62.78 
46.00 
5 
40.89 
50.06 
29.29 
64.99 
46.31 
7 
43.08 
49.31 
28.99 
65.10 
46.62 
9 
43.51 
49.15 
29.10 
65.04 
46.7 
 
Table 11: Ablation study on scene mAcc for different neighbor 
per reference. 
 
 
Neighbors  
Scene mAcc@0.25 
Waldo 
kitchen Figurines Ramen 
Teatime 
Mean 
0 
63.64 
82.14 
38.03 
77.97 
64.45 
3 
68.18 
85.71 
39.44 
79.66 
68.25 
5 
50.00 
83.93 
40.85 
79.66 
63.61 
7 
54.54 
82.14 
40.85 
79.66 
64.30 
9 
54.54 
82.14 
40.85 
79.66 
64.30 
 
Table 12: Ablation Study on Dense correspondence initialization 
time with different neighbor per reference 
 
 
Neighbors  
Scene Init Time 
Waldo 
kitchen Figurines Ramen 
Teatime 
Mean 
3 
2m 29s 
2m 37s 
1m 40s 
1m 47s 
2m 08s 
5 
4m 1s 
3m 43s 
3m 18s 
3m 51s 
3m 44s 
7 
3m 58s 
6m 21s 
3m 47s 
5m 35s 
4m 25s 
9 
7m 11s 
8m 4s 
4m 52s 
6m 53s 
6m 23s

<!-- page 18 -->
8 
We seek to find direct evidence for this concentration 
and whether the outcome is healthy by analyzing the 
contribution mass of each scene. For each ScanNet scene, 
we record the per-view “top-10 share” during registration. 
For each view, a vector is built where each entry is the total 
contribution weight assigned to one context proposal in 
that view. We then derive top 10 share as  
𝑚𝑎𝑠𝑠 𝑜𝑓 10 𝑙𝑎𝑟𝑔𝑒𝑠𝑡 𝑝𝑟𝑜𝑝𝑜𝑠𝑎𝑙𝑠
𝑡𝑜𝑡𝑎𝑙 𝑚𝑎𝑠𝑠 𝑜𝑓 𝑎𝑙𝑙 𝑝𝑟𝑜𝑝𝑜𝑠𝑎𝑙𝑠. 
The resulting mass for each view ranges between 0.87 and 
0.99, which means that the 10 most active proposals 
already account for the majority of the total proposal mass 
in a typical view. At the scene level, we also compare the 
fraction of total Gaussian mass that lies in the most heavily 
used 0.1% and 1% of Gaussians. The top 1% of Gaussians 
carries between 41 ~ 68% of the accumulated contribution 
mass, and the top 0.1% still carry 14 ~ 42%. These statistics 
show that both clusters and Gaussians exhibit a highly 
skewed distribution under our registration scheme, which 
explains why ProFuse saturates at K=10 on all three 
ScanNet label sets, whereas Dr. Splat requires much larger 
K to reach comparable point-cloud performance. 
 
Warped IoU. We ablate the warped intersection-over-
union threshold 𝜏iouthat decides whether a dense-warped 
mask pair contributes an edge to the cluster graph. The 
threshold is varied in {0.1,0.2,0.3,0.4,0.5} and for each 
value we recompute 3D context proposals and repeat the 
object-selection and point cloud evaluation. The resulting 
scene-level mIoU curves are not monotonic, and the 
behavior differs across scenes, which suggests that the 
effective operating point is shaped by a combination of 
warped IoU and the mutual best-neighbor rule. Once 
mutual best neighbors are enforced, many noisy 
correspondences are removed before the IoU gate is 
applied, resulting in 𝜏iou  controls mainly a secondary 
pruning stage. In this regime 𝜏iou = 0.2 provides a stable 
choice across scenes. It retains enough cross-view links to 
form reliable context proposals while still discarding 
clearly inconsistent warps, and we adopt this threshold as 
default setting for all main experiments. 
 
Inferencing Efficiency. Removing densification and 
relying on correspondence-guided initialization leads to a 
much more compact Gaussian scene. Table 13 summarized 
the comparison of average inferencing time and total 
Gaussians per scene. On ScanNet, Dr. Splat keeps 2.7 
times more Gaussians than our method while operating on 
the same data. Despite this reduction, the semantic and 
geometric metrics reported in the main paper remain 
competitive or improved, which indicates that the 
correspondence seeds and pruning schedule preserve the 
informative splats. 
We also measure the cost of point-cloud inference on 
ScanNet. Under the same evaluation pipeline, Dr. Splat 
requires on average about 99 seconds per scene, whereas 
ProFuse completes the same retrieval in about 59 seconds. 
The method therefore achieves faster inference together 
with a substantially smaller Gaussian set, which matches 
the reduction in Gaussian count and confirms that the 
correspondence-initialized, 
non-densified 
scenes 
are 
advantageous both for efficiency and for downstream 
open-vocabulary understanding. 
 
Table 13: Ablation study on inference efficiency. 
Method 
Inference Time/  
Total Gaussians 
Dr. Splat  
99s / 1.27M 
ProFuse 
59s / 470K 
 
E.4. More Qualitative Results 
We provide additional qualitative results of ProFuse in this 
section.  Figures 2 and 3 visualize cosine similarity activations for 
several text queries on LERF scenes. The results illustrate that 
global features sharpen the response on the queried object, 
suppress background clutter, and maintain consistent activation 
within the object extent. Figure 4 presents PCA projections of the 
per-Gaussian language features on ScanNet and LERF scenes. 
For each view we display the input image together with the 
projected features of Dr. Splat and ProFuse. The ProFuse features 
form more coherent regions that align much better with objects 
and surfaces and they remain stable across different viewpoints, 
which supports the improvements observed in the quantitative 
evaluations. Figure 5 illustrate feature visualization on more 
ScanNet scenes, demonstrating the strength of ProFuse in context 
understanding.

<!-- page 19 -->
9 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Figure 3. Qualitative results of cosine similarity activation on given queries “nori” and “spoon”. With global features, 
ProFuse carries context-level interpretation and injects consistency in 3D scene understanding. 
Figure 2. Qualitative results of cosine similarity activation on given queries “coffee mug” and “old camera”.

<!-- page 20 -->
10 
 
 
 
 
 
 
 
 
 
Figure 4. Qualitative results of PCA visualization on ScanNet and LERF scenes. For each view, we provide the reference 
image (left), and render PCA of Dr. Splat (middle), and ProFuse (right) for comparison. 
Figure 5. Additional rendering of feature visualization on ScanNet scenes.
