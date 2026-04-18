<!-- page 1 -->
ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary
Understanding in 3D Gaussian Splatting
Jiayu Ding1, Xinpeng Liu1, Zhiyi Pan2, Shiqiang Long3, Ge Li1*
1Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University
2School of Computer Science and Technology, Tianjin University 3Guangdong Bohua UHD Innovation Center Co., Ltd.
{jyding25@stu, lxpeng@stu, geli@ece}.pku.edu.cn, zhypan@tju.edu.cn, longsq@bohuauhd.com
Abstract
Lifting 2D open-vocabulary understanding into 3D Gaus-
sian Splatting (3DGS) scenes is a critical challenge. Main-
stream methods, built on an embedding paradigm, suffer
from three key flaws: (i) geometry-semantic inconsistency,
where points, rather than objects, serve as the semantic ba-
sis, limiting semantic fidelity; (ii) semantic bloat from in-
jecting gigabytes of feature data into the geometry; and
(iii) semantic rigidity, as one feature per Gaussian struggles
to capture rich polysemy. To overcome these limitations,
we introduce ExtrinSplat, a framework built on the extrin-
sic paradigm that decouples geometry from semantics. In-
stead of embedding features, ExtrinSplat clusters Gaussians
into multi-granularity, overlapping 3D object groups.
A
Vision-Language Model (VLM) then interprets these groups
to generate lightweight textual hypotheses, creating an ex-
trinsic index layer that natively supports complex polysemy.
By replacing costly feature embedding with lightweight in-
dices, ExtrinSplat reduces scene adaptation time from hours
to minutes and lowers storage overhead by several orders
of magnitude.
On benchmark tasks for open-vocabulary
3D object selection and semantic segmentation, ExtrinSplat
outperforms established embedding-based frameworks, val-
idating the efficacy and efficiency of the proposed extrinsic
paradigm.
1. Introduction
Open-vocabulary 3D scene understanding enables the pars-
ing of 3D scenes with arbitrary natural language queries,
moving beyond the limitations of predefined categories to
offer enhanced generalization and richer semantics for ap-
plications like autonomous driving [13, 36] and robotics [2,
27]. The primary challenge in this domain lies in finding an
efficient and effective 3D scene representation. Traditional
methods such as voxels, point clouds, and meshes, while
*Corresponding author
useful for structure modeling, struggle with the trade-off
between detail and computational expense. Recently, 3D
Gaussian Splatting (3DGS) [10] has been proposed, which
achieves high-fidelity modeling and rendering while main-
taining high rendering speeds, making it an ideal foundation
for next-generation 3D scene understanding.
Recently,
some methods have leveraged 3DGS to
achieve point-level open-vocabulary 3D scene understand-
ing. A majority of these methods [14, 16, 28, 30, 32] at-
tempt to embed high-dimensional semantic features into
each 3D Gaussian point, optimizing these features via con-
trastive learning. This requires tens of thousands of mask-
guided contrastive learning iterations, incurring substantial
optimization costs. More recently, a few approaches [8, 9,
19] have explored embedding semantic features into each
Gaussian via matching, which optimizes costs and achieves
strong results. However, all these methods are built upon
a common embedding paradigm that intrinsically embeds
semantic features into the 3D Gaussian points. While this
paradigm has shown promising results, it suffers from three
limitations: 1) Geometry-Semantic Inconsistency: Ob-
jects, not Gaussian points, should be the basic unit of se-
mantic understanding. Gaussians are designed to express
scene geometry, not specifically for semantics. Moreover,
this inherent disparity between geometric and semantic rep-
resentations manifests at object boundaries as what we term
“neutral points”: points driven solely by geometric opti-
mization for high-fidelity boundary rendering, which intrin-
sically lack a semantic assignment. Embedding paradigm,
by forcibly assigning semantic features to every Gaussian,
inevitably lead to semantic ambiguity and confusion at ob-
ject boundaries. 2) Semantic Bloat: The core of embed-
ding paradigm is to lift and store 2D visual features. This
results in 3DGS scenes being injected with Gigabytes (GB)
of feature data, dramatically increasing storage and down-
stream processing burdens. 3) Semantic Rigidity: A single
Gaussian can only store one visual feature vector. In real-
ity, a single Gaussian point may be part of multiple objects,
thereby possessing distinct semantic meanings that differ
arXiv:2509.22225v2  [cs.CV]  27 Mar 2026

<!-- page 2 -->
significantly in the feature space. For example, a Gaussian
point on a car’s window surface can be identified as “win-
dow”, but it is also correctly described as part of the “car”
itself. The existing embedding paradigm attempts to force-
fully fuse a point’s multiple, and often conflicting, seman-
tic identities into one feature vector via contrastive learning
or feature projection. This compromised representation not
only causes inherent semantic inaccuracy but also funda-
mentally limits the model’s ability to express the rich poly-
semy of complex scenes.
To overcome these limitations, we propose the extrin-
sic paradigm, a distinct, decoupled and layered architec-
ture. This paradigm avoids injecting semantic features into
the Gaussian points. Instead, it models semantics as an in-
dependent, abstract index layer. This semantic layer refer-
ences the underlying geometry, leaving its original struc-
ture intact. The core advantage of this decoupled design is
that it operates on the natural atomic units of each domain:
Gaussian points for geometry and objects for semantics.
Based on this extrinsic paradigm, we propose ExtrinSplat, a
point-level open-vocabulary 3D understanding framework.
ExtrinSplat addresses the three inherent drawbacks of the
embedding paradigm: 1) Addressing Geometry-Semantic
Inconsistency: ExtrinSplat shifts the semantic unit from
points to objects by clustering Gaussian points into distinct,
per-entity 3D groups via multi-view mask back-projection.
A dedicated mechanism further identifies and excludes neu-
tral points from semantic assignment. This enhances se-
mantic clarity at object boundaries. 2) Addressing Seman-
tic Bloat: We avoid storing high-dimensional feature vec-
tors. Instead, we use Vision-Language Models (VLMs) to
directly interpret these 3D object groups and generate can-
didate “textual hypotheses” for each group. Semantics are
then stored as lightweight extrinsic indices pointing to these
hypotheses, reducing storage overhead by several orders of
magnitude compared to feature embedding. 3) Addressing
Semantic Rigidity: We introduce a multi-granularity, over-
lapping object grouping mechanism. The same Gaussian
point can simultaneously belong to multiple groups with
different semantic identities. Each group links to multiple
textual hypotheses, natively supporting the rich polysemy
unattainable by the embedding paradigm.
Our contributions are summarized as follows:
• We propose ExtrinSplat, a new framework realizing
the extrinsic paradigm, which efficiently decouples 3D
geometry and semantics through object grouping and
lightweight textual indices.
• We introduce a multi-granularity, overlapping object
grouping strategy, enabling the framework to natively
support rich semantic polysemy.
• We define the concept of neutral points and propose a
dedicated handling mechanism to address this issue.
2. Related Works
2.1. Preliminary: 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) [10] models 3D scenes with
explicit 3D Gaussians, enabling high-quality, real-time ren-
dering. It represents a scene as a collection of 3D Gaussians
G = {gi}N
i=1, each defined by its position, covariance (gov-
erning scale and orientation), color, and opacity. To gener-
ate a 2D image, these 3D Gaussians are projected onto an
image plane and then blended in a depth-sorted order via
“splatting”. The final color C(p) for any pixel p is deter-
mined through alpha compositing [20]:
  \l a
bel 
{
eqw
} C
( p) 
=
 \s
u
m _
{i = 1}^
{ |\mathcal {G}_p|} c_{g_i^p} \alpha _{g_i^p} \prod _{j=1}^{i-1} (1 - \alpha _{g_j^p}), 
(1)
where cgp
i and αgp
i are the color and opacity of the i-th
Gaussian in the sorted set for pixel p. The product term,
Qi−1
j=1(1 −αgp
j ), calculates the accumulated transmittance,
which represents the light that reaches the i-th Gaussian af-
ter passing through all prior ones.
2.2. Open-Vocabulary Understanding in 3DGS
Prevailing methods for semanticizing 3DGS for open-
vocabulary understanding follow two primary paradigms:
pixel-based and point-based. Pixel-based methods employ
a “render-then-match” paradigm: they first render the entire
scene into dense 2D feature maps and subsequently perform
semantic matching in the image space. In contrast, point-
based methods adopt a “match-then-render” strategy, first
identifying a sparse set of semantically relevant 3D points
and then rendering only this pre-filtered subset.
In pixel-based methods, Feature-3DGS [35] distills se-
mantic features from 2D foundation models into 3DGS,
enabling fast semantic rendering. LEGaussians [26] adds
uncertainty and semantic to each Gaussian and compares
rendered semantic maps with quantized CLIP [24] and
DINO [33] features. LangSplat [22] learns language fea-
tures in a scene-specific latent space and renders them as
semantic maps. GS-Grouping [31] assigns a compact iden-
tity encoding to each Gaussian and leverages masks from
the Segment Anything Model (SAM) [12] for supervision.
GOI [23] introduces an optimizable semantic hyperplane
to separate pixels relevant to language queries, improving
open-vocabulary accuracy.
However, these methods rely on rendered 2D semantic
maps, so reasoning remains in 2D. They lack awareness of
3D structure and are thus less suited for tasks requiring di-
rect 3D interaction, such as embodied intelligence. To over-
come these limitations, point-based methods have been pro-
posed. These approaches operate on a foundational “select-
then-render” pipeline. OpenGaussian [30] uses SAM masks
to learn instance features with 3D consistency, introduces

<!-- page 3 -->
Figure 1. Overview of our method. (a) Multi-view 2D segmentation masks are first extracted from the input scene. (b) Based on these
masks, our method lifts the objects into 3D point groups via back-projection, refining their boundaries by filtering ambiguous neutral
points. (c) Each refined group is then grounded by using a VLM to generate textual hypotheses from its key views, which are encoded
into semantic features via a CLIP text encoder. (d) Finally, these geometric groups and semantic features are assembled into an extrinsic
semantic index layer, enabling open-vocabulary querying by matching a user’s text query against the pre-computed features.
a two-stage codebook for feature discretization, and links
3D points with 2D masks and CLIP features for open-
vocabulary selection.
InstanceGaussian [14], based on
Scaffold-GS [17], jointly learns appearance and seman-
tics and adaptively aggregates instances, reducing seman-
tic–appearance misalignment.
Dr.Splat [9] registers 2D
CLIP features to 3D Gaussian points via aggregation and
compresses them using a pre-trained Product Quantization
codebook. LUDVIG [19] uplifts 2D CLIP features to 3D
Gaussian points via weighted aggregation and refines them
using a DINOv2-based [21] graph diffusion mechanism.
All these approaches adhere to an embedding paradigm, in-
trinsically binding high-dimensional features to the Gaus-
sian geometry.
3. Method
3.1. Overall Architecture
We present ExtrinSplat, a training-free framework that re-
alizes the extrinsic paradigm by decoupling 3D geometry
from semantics, as shown in Fig. 1. The core of the Ex-
trinSplat framework is to treat each object in a 3D scene as
an independent entity. For every object, the model performs
separate 3D grouping and semantic feature assignment. The
assignment process for these objects is mutually indepen-
dent and can match the same Gaussian points simultane-
ously. User queries in natural language are matched directly
against the features of these independent object groups.
This approach eliminates the need to store a unique seman-
tic feature vector for every Gaussian point, enabling more
efficient and coherent object-level semantic understanding.
Our method takes an optimized 3DGS scene repre-
sentation and its corresponding image sequence as input.
First, the data preparation stage (§3.2) generates multi-
view, multi-granularity object masks. Next, the object-level
grouping stage (§3.3) links 2D masks to 3D Gaussian points
and purifies the resulting object boundaries by identifying
and excluding ambiguous neutral points. Then, the instance
feature extraction stage (§3.4) uses a VLM to generate tex-
tual hypotheses for each object group. Finally, these geo-
metric groups and semantic components are assembled into
the extrinsic semantic index layer (§3.5) to formalize the
decoupling and enable efficient open-vocabulary querying.
3.2. Data Preparation
Our goal is to extract a comprehensive set of multi-view
segmentation masks for all objects in a scene. To this end,
we first employ SAM on the initial frame, I0, leveraging
its ability to produce object masks at three distinct gran-
ularity levels (e.g., part, object, scene).
To ensure sta-
ble tracking throughout the sequence, especially in com-
plex scenes with visually similar distractors, we employ the
DAM2SAM [29] model. Its specialized distractor-aware
memory is crucial for maintaining accurate object identi-
ties where other trackers might fail. To capture new objects
that appear later, we introduce a periodic detection mecha-
nism that re-segments the scene at fixed intervals and identi-
fies new instances based on a minimal IoU overlap criterion
with existing tracks. The entire pipeline, from tracking to
new object detection, is executed independently for each of
the three granularity levels to yield a complete and hierar-
chical set of masks.
Our pipeline is designed for robustness, as potential
data preparation artifacts, such as tracking failures or re-

<!-- page 4 -->
identification errors, are gracefully handled by our down-
stream object grouping and query matching modules. This
design minimizes the requirements for perfect input data
(see Appendix for details).
This independent, multi-
granularity processing ensures the intrinsic overlap (e.g.,
fine-grained part within coarse-grained object) of the mask
set, providing a foundation for the model’s subsequent pol-
ysemous understanding.
3.3. Object-level Grouping
To realize multi-granularity grouping, we apply our group-
ing strategy independently to each object mask set gener-
ated in Sec. 3.2. This parallel and independent execution en-
ables a single 3D Gaussian to be assigned to multiple object
groups (e.g., “window” and “car”), thereby natively accom-
modating semantic polysemy. Specifically, for each group,
we first identify the object’s high-confidence core via mask
back-projection, then refine its boundaries by identifying
and excluding ambiguous points with our neutral point pro-
cessing module, ensuring a clean result.
Initial 3D Grouping via Mask Back-projection. We link
2D masks to 3D Gaussian points by back-projecting them
to estimate a per-point foreground probability. For each ob-
ject, we process its multi-view masks, first discarding any
null (entirely black) masks from viewpoints where the ob-
ject is unseen. For each valid mask, we then cast a ray
through each pixel r and sum the contributions of all inter-
sected Gaussians. The contribution of the j-th Gaussian Gj
along ray r is determined by its accumulated transmittance
and opacity, defined as:
  w( r, G _j) = T ( r, G _j) \cdot \alpha (r, G_j), 
(2)
where T(r, Gj) denotes the accumulated transmittance up
to Gj, and α(r, Gj) is its effective opacity. To ensure design
consistency, we define w(r, Gj), representing the contribu-
tion of Gaussian Gj to pixel r, to be identical to the forward
color rendering weight of 3DGS given in Eq. 1.
For each 3D Gaussian point Gj, we compute its to-
tal foreground (W1) and background (W0) weights, corre-
sponding to k = 1 and k = 0 respectively, by aggregating
contributions from multi-view 2D masks:
  W_k( G
_
j) 
=
 \su
m _{v \ i n \ mathc al {V}} \sum _{r \in \mathcal {P}_v} \delta (m_v(r) - k) \cdot w_v(r, G_j), 
(3)
where V is the set of visible views, Pv the pixels in a
view, mv(r) the mask value, δ(·) the indicator function, and
wv(r, Gj) the contribution weight. Based on these weights,
we form an initial foreground set, F, using a simple hard as-
signment: F = {Gj | W1(Gj) > W0(Gj)}. All remaining
points are consequently assigned to the background.
Neutral Point Processing.
During rendering, it is in-
evitable that some points lie at the boundaries between ob-
jects but do not semantically belong to any specific cate-
gory. We refer to these points as neutral points. Their se-
mantic assignment directly affects the accuracy of rendered
object boundaries. Existing methods typically assume that
each 3D Gaussian point belongs either to the foreground
or to the background, i.e., every point has a clear seman-
tic label. In practice, however, many points at boundaries
are transitional and may not carry a well-defined semantic
meaning. Such points should be considered neither fore-
ground nor background. Our goal is to identify and exclude
these neutral points from semantic supervision, thereby mit-
igating potential artifacts and improving the accuracy of the
final segmentation.
To identify neutral points, we leverage multi-view se-
mantic consistency. While points deep within an object are
consistently labeled across views, those near boundaries of-
ten exhibit conflicting semantics. To quantify this ambi-
guity, we treat each viewpoint as providing a discrete se-
mantic label for a given Gaussian point. Specifically, for
each point p, we project its center into every visible view
and record whether it lands inside (foreground) or outside
(background) the corresponding 2D mask.
This process
yields a set of binary labels {lv}v∈V for each 3D point. The
semantic entropy H(p), which quantifies the disagreement
among these discrete labels, is calculated as:
  H( p )
 = 
-  \le
ft
 (  \
f rac 
{V
_
f
}{V} \log _2 \frac {V_f}{V} + \frac {V_b}{V} \log _2 \frac {V_b}{V} \right ), 
(4)
where Vf and Vb are the respective counts of foreground and
background labels within the set {lv}v∈V, and V = |V| =
Vf + Vb. Points with entropy H(p) exceeding a threshold
τh form an initial candidate set C of ambiguous points.
This set C is impure, containing both true neutral points
used for smooth blending and mislabeled solid points that
belong to an object’s surface. To distinguish them, we use
a geometric property: opacity (α). Points on solid surfaces
typically have high opacity, while transitional points used
for anti-aliasing have low opacity. We filter C based on this
idea: if a point p ∈C has an opacity α(p) > τα, we classify
it as a mislabeled solid point. These points, identified as part
of a solid surface, are removed from the neutral candidate
set C, thereby retaining their initial classification as either
foreground or background.
The remaining points in C are confirmed as the final neu-
tral point set, which is excluded from all semantic super-
vision. The final set of foreground points is thus defined
by the expression F \ C. Likewise, the background set is
refined by removing these same points. We use fixed val-
ues for the thresholds τh and τα across all experiments for
simplicity and robustness. A detailed sensitivity analysis on
their selection is provided in Appendix.

<!-- page 5 -->
Figure 2. Comparison of 2D-3D feature association pipelines. (a)
Mainstream method (via direct extraction): All object masks, typ-
ically generated by SAM, are used to directly extract CLIP image
features. (b) Our method (via semantic distillation): We leverage
DAM2SAM to track a single instance. The top-N most visible
masks are then interpreted by a VLM, distilling volatile visual ap-
pearances into a stable CLIP text representation derived from the
generated object identity.
3.4. Instance Feature Extraction
The dominant embedding paradigm, which intrinsically ag-
gregates multi-view visual features into the 3D geometric
structure, as shown in Fig. 2(a), faces two key limitations:
1) Semantic instability. Aggregating visual features from
multi-view 2D masks often produces biased and inconsis-
tent 3D semantics. Due to viewpoint variance in 2D en-
coders (e.g., CLIP), the same 3D object can yield markedly
different embeddings across views (see Appendix for visu-
alization), reducing overall 3D accuracy. 2) Storage ineffi-
ciency. Extracting and storing high-dimensional visual fea-
tures for every object mask in all views introduces heavy
computational and storage costs, making the 3D scene rep-
resentation inefficient and inflexible. To avoid these draw-
backs, we propose semantic distillation: we use a VLM
to interpret key views and generate textual hypotheses of
the object’s identity. As shown in Fig. 2(b), this converts
volatile visual appearances into a stable, canonical text rep-
resentation.
Specifically, our semantic distillation process operates
for each object group i identified in Sec. 3.3. First, we se-
lect the top-N masked views with the largest visible areas
for this group. These views are fed into a Vision-Language
Model (VLM) along with a predefined text prompt, which
instructs the model to generate a set of candidate textual
hypotheses (i.e., object names). Our framework is VLM-
agnostic; for this study, we employ Gemini 2.5 Pro [5] , but
other models can be readily substituted. Next, these VLM-
generated candidate names are encoded using a pre-trained
CLIP text encoder. This process yields the group’s final se-
mantic component Qi, a set of feature vectors representing
its identity, which is then passed to the extrinsic index layer
described in the following section.
3.5. Extrinsic Semantic Index Layer
We construct the extrinsic semantic index layer to formalize
the decoupling of geometry and semantics proposed by the
extrinsic paradigm. This structure is a set of object maps,
L = {(Gi, Qi)}N
i=1, where N is the total number of object
groups. Each map i consists of a geometric component,
Gi = Fi \ Ci, which is the set of indices for all 3D Gaussian
points in the group, and a semantic component, Qi, which is
the set of pre-computed CLIP text features from the VLM’s
textual hypotheses.
Open-vocabulary querying becomes a fast lookup
against this index layer. A text query is encoded into a fea-
ture vector s using the CLIP text encoder. This query s is
then compared against the set of semantic features Qi for
each object group i. We use cosine similarity to quantify
semantic relevance:
  \tex t {
s i m
}(\math
b f  {s}, \mathbf {q}) = \frac {\mathbf {s} \cdot \mathbf {q}}{\|\mathbf {s}\|\|\mathbf {q}\|}, \quad \mathbf {q} \in \mathbf {Q}_i. 
(5)
An object group i matches if any of its semantic feature
vectors q ∈Qi exceeds a similarity threshold η. The set of
matching group indices Im is defined as:
  \
m
a t hca
l {I }_{\te xt  {
m
}} = \left \{ i \mid \max _{\mathbf {q} \in \mathbf {Q}_i} \text {sim}(\mathbf {s}, \mathbf {q}) > \eta \right \}. 
(6)
The final segmentation for the query is the union of the 3D
Gaussian points from the matched geometric components:
  \mat h
c
al {
G}_{\text {final}} = \bigcup _{i \in \mathcal {I}_{\text {m}}} \mathcal {G}_i 
(7)
4. Experiments
4.1. Open-Vocabulary Object Selection in 3D Space
Settings 1) Task. Given a text query as input, the task is
to produce multi-view renderings of the semantically cor-
responding 3D instance(s).
First, the textual feature of
the input query is extracted using the CLIP model. Then,
cosine similarity is computed between the query feature
and the textual features of each instance, and the most
similar instance(s) are selected. Finally, all 3D Gaussian
points belonging to the selected instances are rendered into
multi-view images through the 3DGS rasterization pipeline.
2) Baselines. We compare our method with several recent
representative approaches [1, 4, 9, 14, 19, 22, 23, 26, 30,
31, 35]. These approaches fall into the two primary cate-
gories of point-based and pixel-based methods. To provide
a clear comparison, we detail the comparative aspects such
as training time and search thresholds for these methods in
Tab. 1. 3) Dataset. We adopt the LERF [11] dataset, an-
notated by LangSplat. This dataset consists of multi-view
images capturing 3D scenes and provides ground-truth 2D

<!-- page 6 -->
Figure 3. Qualitative results on object selection from the LERF dataset. OpenGaussian fails to separate nearby objects or maintain sharp
boundaries, while Dr.Splat struggles to capture fine-grained details. In contrast, our method correctly interprets fine-grained instructions to
generate precise selections with well-defined boundaries.
Table 1. This caption compares computational resources for the LERF figurines scene, including per-scene optimization time, peak
VRAM use, and storage for CLIP features. By decoupling semantics from geometry as the only extrinsic method, our approach achieves
high efficiency, cutting CLIP feature storage from gigabytes to megabytes and using the least amount of VRAM. Note that “–” marks
methods that do not use CLIP features.
Method
Venue
Domain
Paradigm
Scene Opt.
Train Time
CLIP F.S.
Peak VRAM
LEGaussians
CVPR’24
2D
Embedding
Required
∼2h
∼3GB
∼20 GB
LangSplat
CVPR’24
2D
Embedding
Required
∼2h
∼3GB
∼20 GB
Feature-3DGS
CVPR’24
2D
Embedding
Required
∼1h
∼3GB
∼26 GB
GS-Grouping
ECCV’24
2D
Embedding
Required
∼1h
–
∼28 GB
GOI
MM’24
2D
Embedding
Required
∼1h
–
∼24 GB
3DVLGS
ICML’25
2D
Embedding
Required
∼1h
∼3GB
∼24 GB
Occam’s LGS
BMVC’25
2D
Embedding
None
None
∼3GB
∼12 GB
OpenGaussian
NIPS’25
3D
Embedding
Required
∼1h
∼3GB
∼22 GB
InstanceGaussian
CVPR’25
3D
Embedding
Required
∼2h
∼3GB
∼24 GB
Segment-then-Splat
NIPS’25
3D
Embedding
Required
∼1h
∼3GB
∼24 GB
LaGa
ICML’25
3D
Embedding
Required
∼2h
∼3GB
∼24 GB
Dr.Splat
CVPR’25
3D
Embedding
None
∼1h
∼3GB
∼24 GB
LUDVIG
ICCV’25
3D
Embedding
None
None
∼3GB
∼22 GB
Ours
–
3D
Extrinsic
None
None
∼3MB
∼8 GB
annotations for texture-level queries. For a fair comparison,
we use the same predefined query texts as in OpenGaussian.
Results 1) Quantitative Evaluation. As shown in Tab. 1,
our extrinsic, decoupled architecture eliminates per-scene
optimization, leading to a ∼1000x reduction in feature stor-
age and a significantly lower VRAM footprint. As shown in
Tab. 2, our method achieves a new state-of-the-art (SOTA)
result, outperforming the previous best-performing method
by 3.9 mIoU. This highlights the efficacy of our decoupled
architecture in maximizing both segmentation accuracy and
resource efficiency. 2) Qualitative Evaluation. As illus-
trated in Fig. 3, competing methods expose the inherent
flaws of current feature-embedding paradigms. OpenGaus-
sian aggregates semantically relevant Gaussians into singu-
lar representations, conflating part-level semantics and ne-
glecting neutral points, which yields chaotic boundaries and
irrelevant selections (e.g., spatula, apple). While Dr.Splat
compresses CLIP features to reduce bloat, this degrades lin-

<!-- page 7 -->
Figure 4. Qualitative results of our 3D object segmentation on the ScanNet dataset. OpenGaussian and InstanceGaussian rely on matching
CLIP features extracted from 2D images. This approach is susceptible to feature inconsistencies arising from different mask viewpoints,
often leading to incorrect matches (e.g., for the bed and chair). In contrast, our method achieves accurate 3D segmentation with sharp and
well-defined boundaries.
Table 2.
mIoU results for open-vocabulary object selection in
3D space on the LERF dataset.
Bold/Underline indicates the
best/second-best performance per category.
Method
Ramen
Teatime
Figurines
Waldo
Mean
2D Methods
LEGaussians
46.0
60.3
40.8
39.4
46.6
LangSplat
51.2
65.1
44.7
44.5
51.4
Feature-3DGS
43.7
58.8
40.5
39.6
45.7
GS-Grouping
45.5
60.9
40.0
38.7
46.3
GOI
52.6
63.7
44.5
41.4
50.6
Occam’s LGS
51.0
70.2
58.6
65.3
61.3
3DVLGS
61.4
73.5
58.1
54.8
62.0
3D Training-based
LangSplat-m
6.1
16.6
8.3
8.3
9.8
LEGaussians-m
15.8
19.3
18.0
11.8
16.2
OpenGaussian
31.0
60.4
39.3
22.7
38.4
InstanceGaussian
24.6
63.4
45.5
29.2
40.7
Dr.Splat(Top-40)
24.7
57.2
53.4
39.1
43.6
Segment-then-Splat
54.8
63.5
49.8
40.7
52.0
LAGA
55.6
70.9
64.1
65.6
64.0
3D Training-free
LUDVIG
42.3
58.6
58.0
42.8
50.4
Ours
45.6
64.4
66.4
40.9
54.3
guistic precision and retains the flaw of fusing multiple se-
mantic identities, failing on fine-grained queries (e.g., “bear
nose”, “noodles”). In contrast, our extrinsic architecture
inherently avoids feature entanglement. By operating on
semantic objects, utilizing uncompressed CLIP representa-
Table 3. Quantitative results for open-vocabulary 3D semantic
segmentation on the ScanNet dataset. Bold/Underline indicates
the best/second-best performance per category.
Method
19 classes
15 classes
10 classes
mIoU↑mAcc↑mIoU↑mAcc↑mIoU↑mAcc↑
LangSplat-m
3.8
9.1
5.4
13.2
8.4
22.1
LEGaussians-m
1.6
7.9
4.6
16.1
7.7
24.9
OpenGaussian
24.7
41.5
30.1
48.3
38.3
55.2
InstanceGaussian
40.7
54.0
42.5
59.1
47.9
64.0
Dr.Splat(Top-40)
29.6
47.7
38.2
60.4
50.8
73.5
LAGA
32.5
49.1
35.5
53.5
42.6
63.2
LUDVIG
33.9
51.4
37.4
57.2
46.4
66.2
Ours
45.5
58.4
47.2
61.7
53.7
74.9
tions for maximal linguistic fidelity, and explicitly handling
neutral points, our approach guarantees precise semantics
and sharp geometric boundaries.
4.2. Open-Vocabulary 3D Semantic Segmentation
Settings 1) Task.
The objective is to automatically ex-
tract 3D Gaussian points corresponding to input class names
(e.g., wall, chair, table). The segmented Gaussian points
are then converted into a point cloud to be evaluated against
the ground-truth annotated point cloud. To ensure a precise
correspondence between the converted point cloud and the
ground truth, we disable the 3D Gaussian densification pro-
cess during training. 2) Baselines. Consistent with the ob-
ject selection task, we compare our method against several

<!-- page 8 -->
Table 4. Ablation on neutral point processing. We evaluate the
impact of our two-stage filtering on the LERF dataset.
Case
Method
mIoU↑
#1
Initial Grouping
53.0
#2
+ Opacity Filter
52.6
#3
+ Entropy Filter
53.2
#4
+ Entropy & Opacity Filters (Ours)
54.3
recently proposed approaches [1, 9, 14, 19, 30]. LangSplat-
m and LEGaussians-m are adaptations of existing pixel-
based methods [22, 26], specifically modified to perform
direct 3D referring operations. As this task requires a direct
understanding of 3D points, pixel-based methods are not
applicable and are therefore excluded from our comparison.
3) Dataset. We employ the ScanNet [6] dataset, a bench-
mark comprising indoor scene data with calibrated RGB-
D images and 3D point clouds annotated with ground-truth
semantic labels. For a fair comparison, we adopt the same
scenes and evaluation categories used in OpenGaussian.
Results 1) Quantitative Analysis. Tab. 3 shows the perfor-
mance on the ScanNet dataset using text queries for 19, 15,
and 10 of its classes. The results show that our method con-
sistently achieves SOTA segmentation performance across
all scenes relative to the baselines. This consistent leader-
ship strongly suggests a more robust semantic understand-
ing, validating our hypothesis that our extrinsic, VLM-
driven approach provides a more stable representation than
competing embedding methods struggling with 3D seman-
tic inconsistencies. 2) Qualitative Analysis. Qualitative re-
sults are presented in Fig. 4. In complex scenes from Scan-
Net, both OpenGaussian and InstanceGaussian frequently
exhibit incorrect matches, which limits their accuracy. This
limitation arises from their reliance on matching masked
CLIP image features, as semantic inconsistencies across
different viewing angles make it difficult for such methods
to achieve high-precision results. In contrast, our method
instead leverages a VLM to distill an object’s varied and
often-corrupted visual appearances into a set of canonical
textual hypotheses, achieving a robust semantic understand-
ing that is invariant to occlusion.
4.3. Ablation Study
Neutral Point Processing. We ablate our neutral point pro-
cessing on the LERF dataset with results in Tab. 4. Case #1
is the baseline without any filtering. Case #2 applies only
the opacity filter, and Case #3 applies only the entropy filter.
Case #4 introduces our full model, which incorporates both
filters. As shown, entropy filtering alone provides a minor
gain by suppressing noise, but the opacity filter alone can
be aggressive and inadvertently remove valid foreground
points. Combining both filters resolves this issue, achiev-
ing the highest mIoU. This demonstrates that both stages
Table 5. Ablation on feature extraction. We compare VLM-based
text distillation against CLIP image baselines.
Case
Feature Source
View Aggregation
mIoU↑
#1
Image
Single (Max Area)
36.9
#2
Image
Average (All)
39.2
#3
Image
Average (Filtered)
50.1
#4
Text
Holistic
54.3
are essential for the final performance.
Instance Feature Extraction. To demonstrate the advan-
tages of using a VLM for language feature extraction, we
compare our approach with three baselines derived from
the CLIP image encoder. Case #1 uses the feature from
the single view with the largest mask area. Case #2 aver-
ages features from all valid views. Case #3 first renders the
class foreground points onto each view and computes the
IoU between the rendered foreground masks and candidate
masks. We then discard the low-IoU masks and average the
features of the remaining ones. The results are presented
in Tab. 5. Single-view methods struggle to capture com-
prehensive semantics, while multi-view averaging methods
often yield ambiguous features due to occlusions. Although
filtering-based methods significantly improve matching ac-
curacy, they require a rendering pass for each view, which
incurs high runtime costs. Moreover, these methods can
obscure discriminative details due to feature discrepancies
across different views (see Appendix for details). In con-
trast, our VLM-based method distills these multi-view cues
into a consistent textual representation, effectively captur-
ing the nuanced attributes required for abstract queries.
5. Conclusion and Limitation
In this work, we introduced ExtrinSplat, a training-free
framework that realizes the extrinsic paradigm for open-
vocabulary 3D Gaussian understanding. Our architecture
realizes this paradigm by constructing an extrinsic seman-
tic index layer that completely separates geometry from se-
mantics, associating purified 3D geometric groups with sta-
ble textual hypotheses to enable efficient open-vocabulary
matching. Evaluations on multiple benchmarks confirm that
ExtrinSplat delivers SOTA-level performance at a fraction
of the computational cost.
This work demonstrates that
our extrinsic paradigm is not just a viable alternative, but a
more efficient and semantically robust foundation for open-
vocabulary 3D Gaussian understanding.
Despite its strong performance, our method has certain
limitations: 1) The accuracy of our object-level grouping
can be compromised by substantially inaccurate initial seg-
mentation masks from SAM. 2) Rarely, the VLM may as-
sign incorrect semantic labels to objects. Addressing these
issues remains a promising direction for future work.

<!-- page 9 -->
Acknowledgements
This work was supported by the Natural Science Foun-
dation of China (Grant No.
62531022), the Guangdong
Provincial Key Laboratory of Ultra High Definition Immer-
sive Media Technology (Grant No.
2024B1212010006),
and the Outstanding Talents Training Fund in Shen-
zhen.
References
[1] Jiazhong Cen, Xudong Zhou, Jiemin Fang, Changsong Wen,
Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Tack-
ling view-dependent semantics in 3D language gaussian
splatting. arXiv preprint arXiv:2505.24746, 2025. 5, 8
[2] Shizhe Chen, Ricardo Garcia, Ivan Laptev, and Cordelia
Schmid. Sugar: Pre-training 3d visual representations for
robotics. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 18049–
18060, 2024. 1
[3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen,
Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu,
Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng
Dai.
Internvl: Scaling up vision foundation models and
aligning for generic visual-linguistic tasks. arXiv preprint
arXiv:2312.14238, 2024. 3
[4] Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, and
Danda Pani Paudel. Occam’s lgs: An efficient approach for
language gaussian splatting. In 36th British Machine Vision
Conference 2025, BMVC 2025, Sheffield, UK, November 24-
27, 2025. BMVA, 2025. 5
[5] Gheorghe Comanici et al. Gemini 2.5: Pushing the Frontier
with Advanced Reasoning, Multimodality, Long Context,
and Next Generation Agentic Capabilities. arXiv preprint
arXiv:2507.06261, 2025. 5, 3
[6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5828–5839, 2017. 8
[7] Hao-Shu Fang, Minghao Gou, Chenxi Wang, and Cewu
Lu.
Robust grasping across diverse sensor qualities: The
graspnet-1billion dataset.
The International Journal of
Robotics Research, 2023. 4
[8] Minchao Jiang, Shunyu Jia, Jiaming Gu, Xiaoyuan Lu,
Guangming Zhu, Anqi Dong, and Liang Zhang. Votesplat:
Hough voting gaussian splatting for 3d scene understanding.
arXiv preprint arXiv:2506.22799, 2025. 1
[9] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank
Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly
referring 3D gaussian splatting via direct language embed-
ding registration. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 14137–14146, 2025.
1, 3, 5, 8
[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and
George
Drettakis.
3D
Gaussian
Splatting
for
Real-Time Radiance Field Rendering.
arXiv preprint
arXiv:2308.04079, 2023. arXiv:2308.04079. 1, 2
[11] Justin* Kerr, Chung Min* Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In International Conference on Computer
Vision (ICCV), 2023. 5
[12] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C. Berg, and Wan-Yen Lo. Segment any-
thing. In Proceedings of the IEEE/CVF international con-
ference on computer vision, pages 4015–4026, 2023. 2
[13] Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang,
Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi-
modal data-efficient 3d scene understanding for autonomous
driving. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 2025. 1
[14] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao
Zhang, Ronggang Wang, and Jian Zhang. Instancegaussian:
Appearance-semantic joint gaussian representation for 3D
instance-level perception. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pages 14078–
14088, 2025. 1, 3, 5, 8
[15] Zehao Li, Wenwei Han, Yujun Cai, Hao Jiang, Baolong Bi,
Shuqin Gao, Honglong Zhao, and Zhaoqi Wang. GradiSeg:
Gradient-guided gaussian segmentation with enhanced 3D
boundary precision. arXiv preprint arXiv:2412.00392, 2024.
7
[16] Siyun Liang, Sen Wang, Kunyi Li, Michael Niemeyer, Ste-
fano Gasperini, Nassir Navab, and Federico Tombari. Su-
pergseg: Open-vocabulary 3d segmentation with structured
super-gaussians. arXiv preprint arXiv:2412.10231, 2024. 1
[17] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3D
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3
[18] Yiren Lu, Yunlai Zhou, Yiran Qiao, Chaoda Song, Tuo
Liang, Jing Ma, and Yu Yin. Segment then splat: A unified
approach for 3D open-vocabulary segmentation based on
gaussian splatting. arXiv preprint arXiv:2503.22204, 2025.
1
[19] Juliette Marrie, Romain Menegaux, Michael Arbel, Diane
Larlus, and Julien Mairal.
Ludvig: Learning-free uplift-
ing of 2d visual features to gaussian splatting scenes.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2025. 1, 3, 5, 8
[20] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao,
Wenzheng Chen, Alex Evans, Thomas M¨uller, and Sanja Fi-
dler. Extracting triangular 3d models, materials, and lighting
from images. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8280–
8290, 2022. 2
[21] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193, 2023. 3
[22] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3D language gaussian splat-

<!-- page 10 -->
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20051–20060,
2024. 2, 5, 8, 7
[23] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Li-
ujuan Cao, Shengchuan Zhang, and Rongrong Ji.
GOI:
Find 3D gaussians of interest with an optimizable open-
vocabulary semantic-space hyperplane.
In Proceedings of
the 32nd ACM International Conference on Multimedia,
Melbourne VIC Australia, 2024. ACM. 2, 5
[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, and Jack Clark. Learn-
ing transferable visual models from natural language super-
vision.
In International conference on machine learning,
pages 8748–8763. PmLR, 2021. 2
[25] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat:
2d to 3d gaussian splatting segmentation solved optimally. In
European Conference on Computer Vision, pages 456–472.
Springer, 2024. 7
[26] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan.
Language embedded 3D gaussians for open-
vocabulary scene understanding.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5333–5343, 2024. 2, 5, 8, 7
[27] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen
Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching
spatial understanding to 2d and 3d vision-language models
for robotics. In Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pages 15768–15780, 2025. 1
[28] Wei Sun, Yanzhao Zhou, Jianbin Jiao, and Yuan Li. Cags:
Open-vocabulary 3d scene understanding with context-
aware gaussian splatting. arXiv preprint arXiv:2504.11893,
2025. 1
[29] Jovana Videnovic, Alan Lukezic, and Matej Kristan.
A
distractor-aware memory for visual object tracking with
sam2. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 24255–24264, 2025. 3, 1
[30] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao
Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding,
Jingdong Wang, et al. Opengaussian: Towards point-level 3d
gaussian-based open vocabulary understanding.
Advances
in Neural Information Processing Systems, 37:19114–19138,
2024. 1, 2, 5, 8
[31] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3D scenes.
In European conference on computer vision, pages 162–179.
Springer, 2024. 2, 5
[32] Hairong Yin, Huangying Zhan, Yi Xu, and Raymond A
Yeh.
Semantic consistent language gaussian splatting
for point-level open-vocabulary querying.
arXiv preprint
arXiv:2503.21767, 2025. 1
[33] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun
Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR
with improved DeNoising anchor boxes for end-to-end ob-
ject detection. arXiv preprint arXiv:2203.03605, 2022. 2
[34] Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, and Xi-
anming Liu. Cob-gs: Clear object boundaries in 3dgs seg-
mentation based on boundary-adaptive gaussian splitting. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025. 7
[35] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3Dgs: Supercharging
3d gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21676–21685, 2024. 2, 5
[36] Zijian Zhu, Yichi Zhang, Hai Chen, Yinpeng Dong, Shu
Zhao, Wenbo Ding, Jiachen Zhong, and Shibao Zheng. Un-
derstanding the robustness of 3d object detection with bird’s-
eye-view representations in autonomous driving. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 21600–21610, 2023. 1

<!-- page 11 -->
ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary
Understanding in 3D Gaussian Splatting
Supplementary Material
6. Implementation Details
6.1. Model Implementation Details
Data Preparation. Initially, we employ SAM with grid-
based point prompting to acquire initial static object masks
at varying granularities from the first input frame, I0. Sub-
sequently, these masks extracted from I0 are utilized by the
DAM2SAM [29] model to track the corresponding objects
throughout the entire image sequence.
To ensure all objects appearing throughout the sequence
are captured, we introduce a periodic new-object detection
mechanism. This check is performed at a fixed interval of
∆t = 10 frames. At each check, we first compute the total
area of all tracked masks in the current frame, At. We then
trigger a full re-segmentation on this frame using SAM to
get a candidate mask area, Acand. A potential new object
event is flagged if the ratio At/Acand falls below a thresh-
old τarea = 0.9. When triggered, we identify a mask from
the candidate set as a “new” object if its maximum Inter-
section over Union (IoU) with any existing tracked mask is
below a threshold of τiou = 0.6. Once identified, these new
objects are added to the tracking pool and propagated by
DAM2SAM henceforth.
Existing research [18] suggests that such a detection
mechanism can introduce two potential drawbacks:
(1)
tracking failures for some objects, resulting in incomplete
object tracks, and (2) re-appearing objects being misiden-
tified as new after their tracking has been lost, leading to
a single object being assigned multiple instance IDs. Our
model, however, does not need to overcome these issues
during the data preparation stage.
Regarding the first issue, we simply discard views with
empty masks (i.e., where object tracking has failed) during
our object-level grouping stage. As demonstrated in Ap-
pendix 7.2, our model achieves robust performance even
with a reduced number of views per object. Consequently,
this issue has a negligible impact on the overall model ac-
curacy.
Regarding the second issue, the emergence of multiple
instances for a single object is handled by our matching pro-
cess. The matching between open-vocabulary queries and
instance point clusters is a one-to-many operation based on
similarity. In the event of multiple matches, we take the
union of their results as the final output. Therefore, the
presence of multiple instances for the same object does not
degrade the final matching accuracy.
In summary, our model imposes minimal requirements
on the data preparation stage and functions effectively even
with partial mask information for each object. This demon-
strates the robustness of our approach to imperfections in
the input data.
Object-Level Grouping. The object-level grouping pro-
cess is accomplished within a single forward rendering pass.
In our implementation, we simply accumulate the contri-
bution weights of all participating 3D Gaussians during the
forward pass of the 3D Gaussian Splatting render. Through-
out this process, the contribution weight of each Gaus-
sian is naturally aggregated, obviating the need for auxil-
iary data structures or redundant computations. By lever-
aging the highly optimized volumetric projection inherent
to 3D Gaussian Splatting, our method achieves exceptional
computational efficiency while maintaining semantic coher-
ence. For the subsequent neutral point processing, we use
fixed thresholds across all experiments to ensure robustness
and consistency. The semantic entropy threshold is set to
τh = 0.9, and the opacity threshold for filtering is set to
τα = 0.1. A detailed sensitivity analysis for these hyperpa-
rameters is provided in Appendix 7.2.
Instance Feature Extraction. We acquire features for each
object instance as follows.
First, we identify the three
largest masks for the instance based on pixel area. For each
selected mask, we highlight the corresponding object on the
original image with a green bounding box, creating three
distinct input images. These images are then processed by a
VLM, which generates a set of five nouns that describe the
instance.
To match an instance against a user’s text query, we com-
pute the cosine similarity between the CLIP feature em-
bedding of the query and the CLIP embeddings of the five
nouns associated with that instance.
This design allows
a single query to potentially match multiple instances. A
match is deemed successful if the similarity score for any
of an instance’s five candidate nouns surpasses a predefined
threshold of η = 0.9.
The specific prompt template used to elicit these nouns
from the VLM is defined as follows:
“In the images, identify the object that is enclosed
by a bright green outline.
Provide five distinct and ap-
propriate nouns to describe ONLY that specific object.
Return ONLY the five nouns separated by slashes (e.g.,
car/automobile/vehicle/motorcar/transport).
Do not add
any other explanatory text, titles, or formatting.”

<!-- page 12 -->
Figure 5. Additional qualitative results for open-vocabulary object selection on the LERF dataset.
Figure 6. Additional qualitative results for open-vocabulary 3D semantic segmentation on the ScanNet dataset.
6.2. Evaluation Details
LERF Dataset Evaluation We evaluate our model on the
LERF dataset, using annotations from LangSplat. Due to
the absence of 3D ground truth, we follow the 2D-based
evaluation protocol from OpenGaussian.
This protocol
measures 3D understanding by computing the multi-view
IoU accuracy between rendered occupancy masks from our
selected 3D Gaussians and the ground-truth masks, which
were manually annotated and provided by OpenGaussian
for a set of text queries.
ScanNet Dataset Evaluation For evaluation on the Scan-
Net dataset, we select the same 10 scenes as used in Open-
Gaussian:
scene0000 00, scene0062 00, scene0070 00,
scene0097 00,
scene0140 00,
scene0200 00,
scene0347 00,
scene0400 00,
scene0590 00,
and
scene0645 00.
The 19 categories defined by ScanNet
used for text queries are: wall, floor, cabinet, bed, chair,
sofa, table, door, window, bookshelf, picture, counter,
desk, curtain, refrigerator, shower curtain, toilet, sink, and
bathtub.
15 categories are without picture, refrigerator,
shower curtain, bathtub; 10 categories are further without
cabinet, counter, desk, curtain, sink.

<!-- page 13 -->
7. Additional Experimental Results
7.1. Additional Qualitative Results
Fig. 5 presents additional qualitative results for the task
of object selection in 3D space on the LERF dataset.
Fig. 6 showcases more results of our model on the open-
vocabulary 3D semantic segmentation task on the Scan-
Net dataset. These results were not included in the main
manuscript due to space limitations. Consistent with our
previous observations, both OpenGaussian and Instance-
Gaussian exhibit limitations in handling object boundaries
and in fine-grained semantic understanding. In contrast, our
model yields results with significantly sharper and more ac-
curate semantic interpretations.
7.2. Additional Ablation Studies
Scene Understanding with Limited Mask Supervision.
Our method leverages a mask-matching mechanism for se-
mantic understanding, a characteristic that enables it to per-
form 3D segmentation from only a sparse set of 2D masks.
To validate this capability, we conduct experiments using
progressively sparser subsets of 2D masks (corresponding
to 1/2, 1/4, 1/8, 1/16, and 1/32 of the total available
views), while all other model settings are held constant. Fi-
nally, we perform an open-vocabulary 3D object extraction
task and qualitatively evaluate the results. As illustrated in
Fig. 7, our method exhibits high robustness to the num-
ber of provided masks. Even with masks from only 1/8
of the views, it maintains high-quality segmentation. This
demonstrates our model’s high data efficiency and its abil-
ity to generalize from sparse supervision. However, when
the number of masks becomes excessively sparse, such as
at 1/16 or 1/32, a portion of the 3D Gaussians may not
be observed by any masked camera view. This lack of su-
pervision results in noticeable artifacts. Notably, the 1/32
subset often corresponds to merely 5–10 foreground masks.
While these extreme cases produce artifacts, the ability to
generate a coherent result from such minimal data under-
scores our method’s low reliance on dense supervision and
corroborates its strong generalization capabilities.
Ablation Study on Neutral Point Thresholds.
On the
LERF dataset, we investigate the influence of the entropy
threshold τh and the opacity threshold τα in our two-stage
neutral point processing module. The results of this sensi-
tivity analysis are presented in Tab. 8. The baseline config-
uration, which bypasses entropy-based filtering by setting
τh = 1.0, achieves an mIoU of 53.0. A notable improve-
ment is observed when τh is lowered to 0.9, underscoring
the efficacy of pruning points with high semantic ambigu-
ity. The necessity of the subsequent opacity-based filtering
is also validated. With τh = 0.9, setting τα = 0 removes all
high-entropy points indiscriminately and degrades perfor-
mance to 53.2 mIoU. This suggests that high-entropy points
with high opacity are geometrically significant and should
be retained. Peak performance is achieved at (τh, τα) =
(0.9, 0.1). This configuration strikes a favorable trade-off
between removing ambiguous transitional points and pre-
serving geometrically salient structures. While the model
demonstrates reasonable robustness to other settings, fur-
ther reductions in τh to 0.8 or 0.5 yield diminished returns,
likely due to the erroneous exclusion of valid surface points.
Based on these findings, we adopt τh = 0.9 and τα = 0.1
for all main experiments.
Instance Feature Extraction.
The core of our instance
feature extraction module is a VLM that grounds textual
queries to 3D visual features. The representational capacity
of the VLM is therefore a critical determinant of perfor-
mance. To investigate this dependency, we ablate the VLM
component with three different pre-trained models on the
LERF dataset: SenseNova 6.5 Pro, InternVL3-78B [3], and
Gemini 2.5 Pro [5] . The results, presented in Tab. 6, re-
veal a strong positive correlation between the representa-
tional power of the VLM and final segmentation accuracy.
More specifically, employing VLMs known for more robust
vision-language grounding consistently yields substantial
gains in mIoU. This indicates that the quality of the seman-
tic features provided by the VLM is a critical determinant of
performance in this task. Therefore, the performance ceil-
ing of our model is not static; it is set to rise in tandem with
the ongoing evolution of Vision-Language Models.
We further analyze the method’s sensitivity to the num-
ber of descriptive text prompts used for instance matching
on the LERF dataset. As shown in Tab. 7, the relation-
ship between prompt quantity and segmentation accuracy
is non-monotonic. Starting from a single prompt, perfor-
mance improves as the number of descriptors increases to
five. This suggests that a richer set of semantic cues helps
the VLM disambiguate instances, particularly for concepts
too nuanced to be captured by a single term. However, in-
creasing to 10 prompts leads to performance degradation.
We hypothesize that an excessive number of prompts may
introduce semantic noise or redundant information, thereby
interfering with the VLM’s feature matching process. Con-
sequently, we use five descriptive prompts, as this configu-
ration strikes a favorable balance between semantic richness
and feature ambiguity.
7.3. Open-Vocabulary 3D Object Editing
Our method enables open-vocabulary editing of objects in
3DGS scenes by mapping a language query to correspond-
ing instance IDs and then applying targeted manipulations.
Fig. 8 demonstrates the scene editing capabilities of our
method. Starting from an original scene reconstructed via
3DGS, we can select an object to perform operations such
as removal (Fig 8(a)), translation (Fig. 8(b)), or styliza-
tion (Fig. 8(c)).

<!-- page 14 -->
Figure 7. Open-Vocabulary 3D Object Extraction from Sparse Masks. We perform an open-vocabulary 3D object extraction task on the
figurines scene from the LERF dataset, providing a progressively smaller subset of 2D masks as supervision. The results demonstrate
that our model’s accuracy experiences negligible degradation when using ≥1/4 of the total masks. With only 1/8 of the masks, it still
exhibits a strong capability to capture the object’s geometry. Even in the extreme case with as few as 1/32 of the masks, our model can
still recover the object’s coarse shape.
Table 6. Ablation on the choice of VLM.
Model
mIoU↑
SenseNova 6.5 Pro
47.0
InternVL3-78B
50.2
Gemini 2.5 Pro
54.3
Table 7. Ablation on number of prompts.
Number of Prompts
mIoU↑
1
44.0
3
50.9
5
54.3
10
53.6
Table 8. Ablation on neutral point processing thresholds τh and
τα.
τh
τα
mIoU↑
1.00
/
53.0
0.99
0.1
53.8
0.90
0.5
53.8
0.90
0.1
54.3
0.90
0.01
54.2
0.90
0.0
53.2
0.80
0.1
53.8
0.50
0.1
53.1
7.4. Open-Vocabulary Object Extraction in Com-
plex and Real-World Scenes
To assess its practical applicability, we validate our method
on a real-world scene. We captured an office environment
using a standard mobile phone and tasked our model with
open-vocabulary object extraction. The qualitative results,
presented in Fig. 9, demonstrate that our model performs ro-
bustly on this in-the-wild data. This highlights the method’s
strong generalization capabilities and its potential for real-
world applications.
To evaluate our model’s comprehension capabilities in
complex scenes, we conduct experiments on the Grasp-Net
dataset [7].
This dataset is characterized by challenging
object arrangements, including overlapping, adjacent, and
contained instances. Despite the close proximity between
instances, our model successfully distinguishes and seg-
ments them. As shown in Fig. 10, our method produces
sharp, well-defined rendering boundaries, demonstrating its
effectiveness in such challenging scenarios.

<!-- page 15 -->
Figure 8. Demonstration of our scene editing capabilities. (a) Object Removal. (b) Object Translation. (c) Object Stylization. All
manipulations are applied directly to the 3D scene rather than on the 2D rendered images.
8. Efficiency Analysis
To dissect our method’s efficiency, we provide a detailed
component-wise runtime breakdown in Tab. 9, based on the
teatime scene in the LERF dataset, which contains 131
distinct instance categories. The total end-to-end process-
ing time for this complex scene is approximately 9.25 min-
utes (555.14s), including all computational and I/O stages.
The results clearly identify the primary computational bot-
tlenecks, with three stages accounting for over 99% of the
total computational workload: VLM Text Feature Acqui-
sition (37.7%), Backward Matching (32.0%), and the ini-
tial Mask Acquisition (29.7%). The analysis also highlights
the efficiency of the neutral point processing module, which
constitutes only 0.1% of the total computational cost. This
low figure indicates that the boundary refinement step is
achieved with minimal performance overhead.
Notably, despite the aforementioned bottlenecks, our
method’s runtime holds a significant advantage over main-
stream methods, which require hours of processing. For in-
stance, in our evaluation on the LERF dataset, we found that
InstanceGaussian [14] requires approximately 140 minutes
for the 3D Gaussian training phase alone. Furthermore, our
model offers potential for even greater speed. In principle,
it processes each category independently, allowing for sig-
nificant acceleration through parallelization. However, as
a key design goal is to ensure deployability on consumer-
grade hardware, this imposes a constraint on the model’s
total memory footprint. Consequently, we did not pursue
further parallelization in the current implementation.

<!-- page 16 -->
Figure 9. Qualitative results for the open-vocabulary object extraction task on a real-world scene captured with a mobile phone.
Figure 10. Qualitative results for the open-vocabulary object extraction task on the Grasp-Net dataset.
Figure 11. Qualitative results for rendering foreground, neutral, and background points on the figurines scene from the LERF dataset.

<!-- page 17 -->
Table 9. Component-wise runtime breakdown for our method on
the teatime scene in the LERF dataset. The analysis highlights
that VLM inference and backward matching are the primary com-
putational bottlenecks. All timings are in seconds, measured on a
single NVIDIA Tesla V100 (32GB) GPU.
Component
Time (s)
Time / Cat. (s)
Compute %
Computational Stages
Mask Acquisition
156.99
1.1984
29.7%
Backward Matching
169.23
1.2919
32.0%
Neutral Point Processing
0.54
0.0041
0.1%
Text Feature Acquisition
199.67
1.5242
37.7%
CLIP Feature Extraction
2.63
0.0201
0.5%
Total Computation
529.06
4.0386
100.0%
I/O Stages
Data Loading
16.12
–
–
Saving Output
9.96
–
–
Grand Total (incl. I/O)
555.14
–
–
9. Analysis of Failure Cases
Impact of Mask Inaccuracy. Our method demonstrates
considerable robustness to sporadic segmentation errors,
provided that the initial masks generated by DAM2SAM
are generally accurate. However, when these masks suf-
fer from large-scale or frequent inaccuracies, our model can
produce erroneous foreground-background distinctions dur-
ing the backward weight accumulation process. This, in
turn, adversely affects the final segmentation accuracy, as
illustrated in a failure case in Fig. 12(a).
Mismatches from the VLM. Incorrect matching can also
arise from the VLM itself, attributable to two primary
sources, as shown in Fig. 12(b).
First, ambiguous seg-
mentation masks or challenging viewing angles in the in-
put images can provide misleading guidance to the VLM.
Second, inherent limitations in the VLM’s comprehension
capabilities can lead to incorrect judgments even with clear
inputs. Either type of error can result in incorrect category
assignments, ultimately causing the point clusters to be mis-
matched with the intended text query.
10. Discussion
10.1. Neutral Points
Prior work on so-called “boundary points”
[15, 34] has
primarily focused on refining their positions through ded-
icated training strategies to enhance semantic understand-
ing. However, while repositioning these boundary points
can improve semantic segmentation accuracy, it often com-
promises the realism and fidelity of the final rendering. This
trade-off arises because boundary points include a special
Figure 12. Examples of Failure Cases. (a) Inaccurate Masks: The
segmentation model outputs incorrect 2D object masks. (b) VLM
Misunderstanding: The VLM provides an incorrect object name
for the given input images.
subset of points that belong neither to the foreground nor
the background. These points serve as transitional elements
that are crucial for ensuring rendering realism but lack spe-
cific semantic meaning. We term these as neutral points.
Neutral points are abundant in 3DGS scenes, making
them non-negligible for semantic understanding.
Never-
theless, accurately identifying and removing these neutral
points in an unsupervised manner remains a significant
challenge. In our implementation, precisely filtering out
these points during the matching stage is difficult due to
computational efficiency constraints. In Fig. 11, we present
the visualization of neutral points from our model on the
LERF dataset. Developing more effective methods to model
and eliminate neutral points is a key direction for future im-
provement of our method.
10.2. Diversity of Semantic Categories
Prior work has noted that a single Gaussian point can be-
long to multiple semantic categories [22, 25, 26]. To ver-
ify this phenomenon, we conduct a statistical analysis of
the semantic categories for all 3D Gaussian points within
the teatime scene of the LERF dataset, as illustrated in
Fig. 13. Our analysis reveals that approximately 25% of
all visible 3D points exhibit multi-dimensional semantic at-
tributes. In the context of our model, this means a sub-
stantial portion of 3D Gaussian points inherently possess
multiple semantic labels simultaneously.
For instance, a
single point on a tree branch may belong to the categories
of “branch”, “tree”, and “vegetation” all at once.This phe-
nomenon is consistent with how humans perceive 3D envi-
ronments.
This semantic diversity suggests that relying on a sin-
gle semantic label is often insufficient to comprehensively
describe the properties of a point. Therefore, this inherent
polysemy must be fully considered when performing 3D se-
mantic understanding.

<!-- page 18 -->
Figure 13. Category distribution of visible Gaussian points in the
teatime scene from the LERF dataset.
Figure 14. Illustration of the inconsistency of semantic features
across different viewpoints. (a) The same object can present dif-
ferent semantic characteristics from different viewpoints. (b) Vi-
sualization of feature similarity for the “Jake the Dog” object in
the figurines scene.
The plot shows the cosine similarity
scores between feature vectors from different views; a higher value
(closer to 1) indicates that the features are more similar.
10.3. Inconsistency of Semantic Features
Our work diverges from the common practice in related lit-
erature of feeding masked object regions into a CLIP image
encoder to obtain semantic features. This decision is based
on the observation that for the same object, its semantic fea-
tures can exhibit significant variations across different view-
points [1]. As shown in Fig. 14(a), acquiring accurate CLIP
image features becomes more challenging from certain an-
gles. Due to the existence of such views, strategies like se-
lecting the features from the view with the largest mask area
or averaging the features across all views inevitably intro-
duce errors.
To validate this phenomenon, we selected the “Jake the
Dog” object from the figurines scene in the LERF
dataset and extracted its CLIP image features from mul-
tiple viewpoints. A visualization of these features is pre-
sented in Fig. 14(b). The figure clearly shows that even
for the same object, the semantic features vary noticeably
with the observation angle. This feature inconsistency sug-
gests that conventional strategies based on single-mask or
averaged-mask feature extraction can lead to information
loss, thereby degrading matching performance. In contrast,
our VLM-based feature extraction approach alleviates this
issue to a certain extent, enhancing the stability and robust-
ness of the semantic representation.
