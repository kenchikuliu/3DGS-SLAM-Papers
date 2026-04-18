<!-- page 1 -->
Gaussian Semantic Field for One-shot LiDAR Global Localization
Pengyu Yin, Shenghai Yuan, Haozhi Cao, Xingyu Ji, Ruofei Bai, Siyu Chen, and Lihua Xie, Fellow, IEEE
Abstract— We present a one-shot LiDAR global localization
algorithm featuring semantic disambiguation ability based on a
lightweight tri-layered scene graph. While landmark semantic
registration-based methods have shown promising performance
improvements in global localization compared with geometric-
only methods, landmarks can be repetitive and misleading
for correspondence establishment. We propose to mitigate this
problem by modeling semantic distributions with continuous
functions learned from a population of Gaussian processes.
Compared with discrete semantic labels, the continuous func-
tions capture finer-grained geo-semantic information and also
provide more detailed metric information for correspondence
establishment. We insert this continuous function as the middle
layer between the object layer and the metric-semantic layer,
forming a tri-layered 3D scene graph, serving as a light-weight
yet performant backend for one-shot localization. We term our
global localization pipeline Outram-GSF (Gaussian semantic
field) and conduct a wide range of experiments on publicly
available data sets, validating the superior performance against
the current state-of-the-art.
I. INTRODUCTION
First proposed in [1], the 3D scene graph has gained
increasing attention over the years. A 3D scene graph is a
multilayered, compact representation of an environment that
organizes spatial-semantic information into interconnected
layers, typically comprising metric-semantic elements at the
lowest level, objects at intermediate levels, and places or
rooms at higher levels. This hierarchical structure captures
not only the geometric and semantic properties of individual
entities but also their topological and spatial relationships,
enabling efficient reasoning about scene structure. Given its
rich semantic information and compactness, several recent
works [2], [3], [4], [5] leverage it as a tool for large-scale
outdoor global localization.
In the literature, Outram [3] proposed to solve the LiDAR
global localization problem in a registration manner, where
a double-layered 3D scene graph is used for a coarse-to-fine
hierarchical search. The core idea behind it is that the query
point cloud can be parametrized as lightweight instance-
level triangle descriptors. And the global localization can
be achieved in a one-shot manner by a coarse-to-fine group-
to-individual search. Subsequent variants [4], [2], [6], [5]
focus on designing different descriptor generation manners
for better discrimination or efficiency. Nevertheless, these
methods rely solely on the centroids of semantic clusters and
their topological connections, overlooking the rich spatial-
semantic distributions within and between clusters. While
centroids provide coarse localization anchors, they fail to
capture the detailed geometric variations and semantic gra-
dients that characterize real-world environments. The contin-
uous spatial distribution of semantic information of how se-
Fig. 1.
A semantic annotated point cloud of a partially semantically
ambiguous scene and two Gaussian semantic fields of the corresponding
rectangle areas. The semantic field encodes the metric-semantic distribution
of a small area (purple for road, light blue for pole, green for vegetation,
and black for traffic sign). Note that the two areas are highly symmetric in
geometry, even in instance-level semantics, but naturally differentiable in
surrounding vegetation distributions. We leverage this continuous semantic
distribution for robust global localization.
mantics evolve across space has the potential to provide fine-
grained cues for robust correspondence establishment. To this
end, we propose a novel global localization approach that
integrates Gaussian semantic fields into 3D scene graphs. Un-
like discrete semantic labels, Gaussian semantic fields (GSF)
provide continuous semantic-geometric representations that
capture fine-grained environmental semantic information.
By incorporating GSF as the middle layer of a 3D scene
graph, we develop a one-shot global localization system with
superior robustness against semantic repetitiveness.
To summarize, we present Outram-GSF, a one-shot global
LiDAR localization framework that addresses semantic am-
biguity through continuous spatial-semantic modeling. Our
main contributions are:
• We introduce Gaussian Semantic Fields (GSF) as a
plug-in, intermediate layer in 3D scene graphs, enabling
continuous modeling of spatial-semantic distributions.
• We develop a probabilistic framework based on Gaus-
sian processes for learning semantic distributions from
local regions, along with a principled similarity met-
ric that leverages these continuous representations for
robust correspondence establishment.
arXiv:2510.12101v1  [cs.RO]  14 Oct 2025

<!-- page 2 -->
• We demonstrate through extensive experiments on pub-
lic datasets that our approach achieves state-of-the-art
performance in one-shot global localization, particularly
excelling in scenarios with repetitive semantic struc-
tures.
II. LITERATURE REVIEW
One-shot Semantic LiDAR Localization. Contrary to many
submap-based LiDAR global localization methods [7], [8],
[9] that require the mobile robot to move for point cloud
accumulation, one-shot methods aim at performing localiza-
tion with only one current LiDAR scan. These methods are
achieved by either bag-of-words (BoW) query-based meth-
ods [10], [11] or registration-based methods [12]. BoW-based
methods are usually extensions of loop closure detection
pipelines, as the same query-based nature can be shared.
However, these per-scan-based methods all share discretiza-
tion issues where the query scan can never exist in the map
database. Recent advances in point cloud semantic scene un-
derstanding have motivated the integration of semantic infor-
mation into one-shot semantic LiDAR localization methods.
By leveraging semantic information, these approaches can
construct 3D scene graphs [1] that provide a hierarchical
and abstract representation of the environment, resulting in
significantly more lightweight maps compared to dense point
clouds. Moreover, scene graphs enable registration-based
methods at the object level, where robust correspondences
can be established between semantic entities rather than raw
geometric features. The encoded instance-wise topological
relationships provide rich contextual constraints that signifi-
cantly reduce the search space for correspondence inference,
enabling more efficient and robust localization. Outram [3]
proposed to parametrize the semantic annotated query point
cloud can be parametrized as lightweight instance-level trian-
gle descriptors. And the global localization can be achieved
in a one-shot manner by a coarse-to-fine group-to-individual
search against the entire map. Subsequent variants [4], [2],
[6], [5] focus on designing different descriptor generation
manners exploiting instance-wise topological information for
better discrimination or efficiency. Nevertheless, they all
depend exclusively on the centroids of semantic instances
and the instance-wise topology, facing a severe performance
drop in semantic ambiguity environments (as shown in Fig.
(I)). As such, the instance-only semantic information can
never establish the correct correspondences.
Spatial Semantic Modeling Early spatial semantic modeling
approaches discretize 3D space into voxels or segments
with categorical labels [13], treating semantic properties as
spatially independent discrete variables. Recent advances
have shifted toward continuous spatial-semantic represen-
tations that model the spatial distribution and correlation
of semantic information, jointly encoding geometric and
semantic properties [14], [15]. Gaussian processes have been
successfully applied to model spatial semantic distributions
[16], [15], providing probabilistic frameworks that capture
both local semantic variations and global spatial correla-
tions. These methods leverage spatial kernels to encode the
assumption that semantically similar regions exhibit spatial
continuity [17]. Hilbert maps [18] and Bayesian generalized
kernel inference [14] further extend this concept by learning
discriminative models for semantic occupancy in continuous
space. More recently, implicit neural representations such
as semantic neural fields [19], [20] have demonstrated the
ability to encode fine-grained spatial-semantic information in
a continuous, differentiable manner. However, most existing
approaches focus on dense scene reconstruction rather than
efficient localization. The challenge remains in extracting
compact yet informative spatial-semantic representations that
preserve both local geometric details and global semantic
context for robust correspondence establishment [21].
The limitations of existing semantic localization ap-
proaches, coupled with recent advances in spatial semantic
modeling, motivate the integration of fine-grained semantic
information into the localization pipeline. Our work ad-
dresses this gap by introducing Gaussian semantic fields
(GSF) as an intermediate layer in 3D scene graphs, enabling
continuous spatial-semantic modeling while maintaining the
computational efficiency required for global localization. By
capturing detailed geo-semantic distributions within local
regions, GSF significantly enhances correspondence estab-
lishment in semantically ambiguous environments where
traditional discrete labels fail to provide sufficient discrimi-
native power.
III. METHODOLOGY
We formulate global localization as a global point cloud
registration problem between the query LiDAR scan P =

pi ∈R3	n
i=1 and the reference point cloud map M =

mj ∈R3	m
j=1. The optimal rigid transformation T
≜
[R, t] ∈SO(3)×R3, including the rotation R and translation
t, can be estimated in close-form [22] given the ground
truth correspondence set I⋆= {(i, j)} ∈[n] × [m] :=
{1, . . . , n}×{1, . . . , m}. Directly applying point cloud regis-
tration techniques to global localization faces two challenges:
1) A proper representation of both query scan and reference
map for scalability; 2) An extremely robust correspondence
establishment strategy for inlier selection. To address the
former challenge, we adhere to the pipeline presented in the
previous work [3] to use a compact 3D scene graph as the
scene representation. For the second challenge, a semantic
field layer is further injected for better correspondence setup
in semantic ambiguous environments.
A. Overview
The proposed global localization pipeline includes 4
stages: 1) A tri-layer 3D scene graph generation module to
produce the Gaussian semantic field layer between the upper
object layer and lower semantic point cloud layer from a
sparse Gaussian process; 2) A set of metric semantic features
is generated by grid probing approach; 3) A graph-based
substructure matching back-end that generates instance-level
correspondences; 4) A consistency unification process to
generate inlier correspondences and produce the final pose
estimation.

<!-- page 3 -->
Fig. 2.
We present Outram-GSF, a one-shot global LiDAR localization pipeline. In stage 1, we generate tri-layer scene graphs for both the current query
scan and the target reference map. Around each instance in the object layer C1, we train sparse Gaussian processes (GPs) using the semantic annotated
point clouds in the metric semantic layer. Populations of GPs for metric similarity comparison are generated by linear probing in stage 2. Combined with
a semantic stability mask, the populations of GP can be used as a similarity measurement in the Wasserstein sense in stage 3. Given the correspondences
established by the above process, inlier correspondences are selected by maximum clique (stage 4).
B. 3D Scene Graph with Local Gaussian Semantic Field
The 3D scene graph is a multi-layered, hierarchical rep-
resentation for joint modeling of low-level geometries and
high-level, human-perceivable semantics. We inject another
layer, the Gaussian semantic field layer, between the object
layer and metric semantic point layer in the commonly
used 3D scene graph structure [23]. The Gaussian semantic
field layer is created by training a sparse Gaussian process
GP. This layer is designed to fully capture the continuous
semantic-geometric distribution and provide more holistic
semantic information for global localization.
We first build a typical double-layered 3D scene graph
for both the current query scan and the reference map by
semantic segmentation [24] and instance clustering. Given a
set of points P = {xi}n
i=1, we define a mapping function λ
from an off-the-shelf segmentation network [25]: {li, yi} =
λ(xi), where li ∈L ⊂N is a finite set of semantic labels and
yi ∈RD is the network logits before the final classification
layer and encodes the uncertainty of one point belonging to
each specific semantic class. We define D = ∥L∥the dimen-
sion of the semantic logits and also the number of semantic
classes. The semantic logits serve as probabilistic semantic
distribution to construct the subsequent Gaussian semantic
field. Instances in the upper object layer are further extracted
by leveraging clustering algorithms within the instantiable
semantic classes, e.g., cars, trunks, and poles. Afterward, the
upper semantic object layer of the 3D scene graph is created:
O = {oj, lj}, where oj is the centroid of the j-th instance
and lj is the corresponding semantic label. The instance set
is then used for subsequent pose estimation. One may have
reservations about selecting moveable objects, e.g., cars, for
pose estimation in the global localization scenario. Rather
than directly filtering out all such objects, we propose to
rerank them by a semantic stability mask as static moveable
objects can be good features for data association.
Although the instance layer provides a compact repre-
sentation for data association and subsequent pose estima-
tion, it suffers from repetitiveness or ambiguity [3]. Such
a case is illustrated in Fig. 2 where a car park lobby is
full of repetitive columns. In this semi-symmetric scenario,
instance-level semantics can be insufficient to produce reli-
able data associations for a global optimal. Such scenarios
invoke us to use more detailed semantic information for data
association. As shown in Fig. 2, the mutual modeling of
geometry and background semantics (e.g., vegetation) can be
efficient in distinguishing the two repetitive columns apart.
As such, we aim to model all semantic classes and their
geometric distributions in a unified manner. We introduce
an intermediate Gaussian semantic field layer to model this
unified geo-semantic distribution of local areas by training a
sparse Gaussian process.
We start from the instance layer by querying the surround-
ing point cloud around each object centroid oi within a
fixed length r. After the radius search, a group of semantic
points is created: {xi, yi, li}M
i=1 with xi ∈R3 the 3D
location of the point and yi, li the semantic logits and label
respectively. Within the selected point cloud, a multi-layered
Gaussian semantic field is constructed to efficiently represent
the metric-semantic distribution.
A Gaussian process is a set of random variables such
that any finite combination of them is a joint multivariate
Gaussian distribution [26]. We design the Gaussian process
to be the mapping between the 3D location in the local
coordinate xi and the semantic logit yi parametrized as
fθ : R3 →R∥L∥where θ is the hyperparameter set. Assume
the latent generative function can be modeled as yi =
fθ(xi)+ϵi, with ϵi ∼N(0, σ2
y) the observation noise. Given
the observation set D = {(xi, yi)}N
i=1, the GP prior can be
written as f ∼GP (µ(x), k(x, x′)), with µ(x) the mean
function, which is usually set to zero in the absence of prior
semantic distribution information, and k(x, x′) the kernel
function, which defines the semantic correlation between
different positions x and x′. The mean and kernel functions
completely describe the Gaussian process [26]. As is com-

<!-- page 4 -->
mon, a Mat´ern 3/2 kernel is chosen as the kernel function:
k(x, x′) = 21−ν
Γ(ν)
 √
2ν ∥x−x′∥
κ
ν
Kν
 √
2ν ∥x−x′∥
κ

,
with Kν the modified Bessel function, ν the smoothness, κ
the length scale of its variation. The predictive mean and
covariance of the GP can be derived in closed form by using
the Bayes’ Rule:
µ(x∗) = k(x∗, X) K−1y,
σ2(x∗) = k(x∗, x∗) −k(x∗, X) K−1k(X, x∗),
(1)
where x∗is the 3-D location and y is the corresponding
semantic logits. And X is the known data point matrix, K =
k(X, X) + σ2I is the regularized kernel matrix.
Sparse Gaussian Process. Training a Gaussian process can
be computationally intensive when dealing with a large
amount of training data with high dimensions. This phe-
nomenon has become increasingly severe considering the
density of LiDAR data and the dimension of semantic logits.
Works in the literature try to reduce the size of the training
set by selecting representative pseudo-supporting points [27].
In our case, we empirically find the semantic labels to be
a natural supporting point selection criterion. More specif-
ically, we construct a semantic class-wise pseudo-training
data generation process as illustrated in Algorithm 1. We find
that such a semantic-based sparsification is computationally
more efficient and outperforms naive random sparsification.
We report the comparison in Fig. 3. We find that with the
proposed semantic-based sparsification, the trained Gaussian
process can better predict the semantic label of the points in
the local coverage with higher mIoU.
Fig. 3. Comparison of the semantic reconstruction quality of two downsam-
pling strategies to train the Sparse Gaussian Process in (1). We find that our
proposed Semantic-based sparsification consistently outperforms random
downsampling in terms of the reconstruction mIoU (mean Intersection over
Union) under the same sample number condition.
C. Triangulated GSF for Substructure Matching
Given the query scene graph and the reference map, the
global localization is achieved in a coarse-to-fine manner.
Following [3], [10], we initiate the localization process by
generating sets of triangle descriptors for coarse-level search.
We triangulate each scene graph (both the query scan and the
map) to form a series of triangles. Given a semantic cluster
Ai in the instance layer of the scene graph, we compute
two features from the lower layers: a Gaussian semantic
Algorithm 1: Semantic-based Sparcification
Input: Point cloud X = {xi}, Labels L = {ℓi},
Target downsampled point number N
Output: Downsampled point cloud X′, Labels L′
1 for each semantic class c do
2
nc ←count(ℓi = c);
3 for each semantic class c do
4
pc ←nc
|X| ; // Proportion of class c
5 for each semantic class c do
6
Nc ←round(pc × N) ; // Number of
points to keep for class c
7 for each semantic class c do
8
Randomly select Nc points from X where ℓi = c;
9 X′, L′ ←combine all selected points;
10 return X′, L′;
field from the intermediate layer GSF(Ai) and a centroid
ai computed by averaging all instance points in the lower
layer. To create the triangle descriptor, each vertex in the
object layer Ai = {ai, GSF(Ai)} is associated with K
nearest clusters {Aj}K
j=1. Afterward, we exhaustively select
two of the neighbors, together with the anchor cluster, i.e.,
A1, A2, and A3, to form one triangle representation of the
current scene graph. By an abuse of notation, we denote it
as ∆(A1,2,3) which comprises of the following attributes:
• a1, a2, a3: centroids of the semantic clusters;
• GSF(A1), GSF(A2), GSF(A3): corresponding Gaus-
sian semantic fields;
• d12, d23, d31: three side lengths, d12 ≤d23 ≤d31;
• l1, l2, l3: three semantic labels associated with each
vertex of the triangle.
Similar to STD [10] and Outram [3], a hash table is built
using the sorted side length d12, d23, and d31 as the key value
due to its permutation invariance. Other attributes are left for
verification purposes. In the searching process, we have the
triangulated scene graph in the query scan and reference map:
∆Query =

∆
 An
1,2,3
	N
n=1 ,
∆Map =

∆
 Bm
1,2,3
	M
m=1 ,
(2)
where n and m are indices for triangle descriptors in the
query and map scene graph, respectively. We drop the sub-
script and denote ∆
 An
1,2,3

as ∆An for clarity. Querying
each of the triangles (e.g., ∆A1) against the hash table con-
structed by the reference semantic scene graph will produce
multiple responses {∆Bq}Q
q=1 as similar substructures could
exist throughout the whole mapping region.
After the first round of coarse matching, we leverage the
GSF for fine-level filtering. While the Gaussian process itself
perfectly models the joint metric semantic distribution of
areas of interest, it is typically hard to use it directly as a
metric-level similarity measurement [28]. We lend the trained

<!-- page 5 -->
TABLE I
SEMANTIC STABILITY TABLE
Sem. Class
Volatile
S-Term
L-Term
Sta. Value wi
Truck, Bike, Person
✓
0.1
Car, Nature
✓
0.5
Infra.
✓
1.0
Gaussian semantic field the ability for similarity comparison
by a sampling process called grid probing. Given a trained
Gaussian process GP, a set of responses can be produced by
querying the process with inputs, where in our case is a set
of 3D locations.
Similarity Measurement for Gaussian processes. While
Gaussian processes are non-parametric representations, there
exists a 2-Wasserstein metric for them [28] achieved by sam-
pling the Gaussian measure and computing the 2-Wasserstein
between the multivariate Gaussian distributions accordingly.
Define the sampling supports S as a uniform 2D grid:
S =

(xi, yj)
 xi = x0 + i · ∆x, 0 ≤i < Nx,
yj = y0 + j · ∆y, 0 ≤j < Ny
	
,
where (x0, y0) denotes the centroid of one semantic cluster,
∆x and ∆y are the sampling intervals, and Nx, Ny are the
number of grid points along each axis. The sampling result is
a group of multivariate Gaussians defined as the populations
of the GP: N
 µ, K + σ2I

. The distance between two Se-
mantic distance fields can be defined using the 2-Wasserstein
distance between the populations of GPs:
W 2
2 (GSF(A1), GSF(B1)) := ∥µ1 −µ2∥2
2
+ Tr(Σ1 + Σ2)
−2 Tr

Σ1/2
1
Σ2Σ1/2
1
1/2
,
(3)
with Σ1 = K1 + σ2
1I and Σ2 = K2 + σ2
2I.
Semantic Stability Mask. Inspired by [29], [30], a stability
mask is proposed to tackle the possible scene changes.
Explicitly, instead of directly filtering out movable semantic
classes from segmentation results, we empirically assign
each semantic class a stability value wi according to its
possibility to move and divide the semantic classes into
types of volatile, short-term, and long-term. Computationally,
this is achieved by alternating the covariance matrix Σ with
a semantic-stability-weighted version W1/2ΣW1/2, with
W = diag(w1, w2, ..., wl). The weighted version can be
viewed as the semantic-based Mahalanobis distance.
With the help of the proposed Gaussian semantic field,
a more detailed semantic distribution is considered in a
continuous manner. Following the exhaustive enumeration of
triangular structures within the query scene graph, we con-
struct an initial collection of instance-level correspondences
Iraw by exploiting the inherent ordering of edge lengths,
which establishes a natural association mechanism between
semantic cluster pairs.
D. Global Localization
With the raw GSF-wise correspondence Iraw built in the
above section, we employ the maximum clique process to
find out the inlier correspondences. Intuitively, we seek to
find an area that maximizes the number of mutually con-
sistent correspondences as well as maintains the consistency
between these local structures:
max
I⊂Iraw
|I|
s.t. D (Ii, Ij) ≤ϵ, ∀Ii, Ij ∈I,
(4)
with D the metric consistency check which indicates whether
two correspondences are mutually consistent with each other
and ϵ the threshold. Namely, for two correspondences Ii and
Ij, with their corresponding semantic clusters Ai, Bi and
Aj, Bj, a consistency check is defined as
D (Ii, Ij) ≜dist (Aij, Bij) .
(5)
We define Aij := Ai −Aj and Bij := Bi −Bj as the
Euclidean distance between the corresponding two vertices.
The consistency check defines how two correspondences
D (Ii, Ij) agree with each other. We solve the objective
function 4 by parallel maximum clique [31].
Having established continuous semantic field correspon-
dences through our GSF framework, we proceed to estimate
the global pose by leveraging the weighted semantic field
similarities. Unlike traditional discrete matching approaches,
our continuous field representation provides probabilistic
correspondence confidences that naturally integrate into the
optimization process.
Given the set of semantic field correspondences with their
associated confidence scores from the GSF similarity com-
putation, we formulate the global localization as a weighted
least squares optimization problem:
ˆR,ˆt =
arg min
R∈SO(3),t∈R3
X
ij∈C
ωij ∥pi −Rqj −t∥2
2 ,
(6)
where C represents the set o f continuous semantic field
correspondences, and ωij denotes the confidence weight
derived from the GSF similarity measurement between se-
mantic field patches i and j from (3). The confidence
weights ωij are computed from the continuous semantic field
similarities, automatically emphasizing correspondences with
high semantic consistency while down-weighting ambiguous
matches.
To handle potential outliers that may arise from semantic
ambiguities in highly repetitive scenes, we incorporate a
robust truncated formulation:
ˆR,ˆt =
arg min
R∈SO(3),t∈R3
X
ij∈C
ωij min

∥pi −Rqj −t∥2
2 , τij

,
(7)
where τij represents an adaptive truncation threshold based
on the semantic field confidence. This formulation naturally
combines the benefits of continuous semantic field disam-
biguation with robust geometric optimization, enabling accu-
rate localization even in challenging repetitive environments.

<!-- page 6 -->
TABLE II
GLOBAL LOCALIZATION PERFORMANCE COMPARISON
Successful Global Localization Rate [%] ↑
Dataset
MulRan DCC
MulRan KAIST
MCD NTU
Localization Seq.
01
02
01
03
01
02
04
10
13
repet.
LCD
GOSMatch [32]
48.61
50.17
35.93
51.98
50.01
52.72
46.83
53.31
51.32
30.06
STD [10]
17.57
18.06
49.61
38.96
42.54
53.78
67.52
52.65
61.89
32.16
BTC [11]
20.08
32.25
52.97
65.17
46.72
55.85
67.23
52.47
68.12
34.08
BEVplaces++ [33]
46.67
51.47
86.85
90.15
71.12
80.31
69.48
73.53
82.72
52.15
Ring++ [34]
89.54
91.67
85.42
88.97
93.78
96.45
95.08
97.12
96.36
60.05
Regis.
Ankenbauer et al. (Original) [12]
-
-
-
-
75.32
78.01
80.42
72.84
83.35
45.89
Ankenbauer et al. (Cons.) [12]
0.072
0.032
0.025
0.012
-
-
-
-
-
-
SGTD [6]
81.15
92.37
86.87
92.06
93.65
92.31
95.26
96.51
96.27
55.26
Outram [3]
82.53
90.48
84.41
85.64
92.11
93.79
95.51
96.01
96.73
54.72
Outram-GSF (Ours)
90.83
94.27
91.09
93.26
98.51
98.43
96.76
99.32
98.53
87.36
We solve Eq. (7) using an iterative reweighted least
squares approach that alternates between updating the pose
estimate and refining the semantic field correspondences
based on geometric consistency.
IV. EXPERIMENTAL RESULTS
In this section, we compare our proposed method with
state-of-the-art LiDAR-based one-shot global localization
methods. For LCD-based methods, we involve handcrafted
sota descriptors BTC [11] and learning-based sota BEV-
places++ [33] and Ring++ [34]. For registration-based base-
lines, apart from previous baselines in Outram [12], [3], a
newly proposed SOTA SGDT [2] is included. All mentioned
algorithms are tested on a PC with Intel i9-13900 and 32Gb
RAM with Nvidia RTX4080.
Experiment Setup. We evaluate our proposed method,
Outram-GSF, against several state-of-the-art on two publicly
available datasets: MulRan [35] and MCD [36]. To further
investigate the performance of different algorithms in highly
repetitive environments, we derived a sequence out of MCD
NTU seq01 featuring highly semantic symmetry. Further-
more, to mimic a real global localization or relocalization
scenario, different from a loop closure detection setting, we
intentionally involve temporal diversity between the mapping
or descriptor generation session and the localization session
from days to months. For each mapping sequence, we
concatenate semantically annotated scans [25] by the ground
truth pose to generate the semantic segmented reference
map for registration-based methods. All semantic classes
with instance definitions are incorporated for localization.
For LCD-based global localization methods, frames in the
mapping sequences are encoded into a database for retrieval
using scans in the localization sequence. Statistics of the
benchmark datasets are presented in Table III. The criteria
for choosing the mapping sequence are the sequence that has
the most coverage of the target area.
A. Localization Success Rate
We adopt the same evaluation metrics as in prior works
[3], [6]. The principal evaluation metric in global localization
is the success rate of localization, where every frame in the
TABLE III
DETAILS OF EVALUATION DATASETS
Mapping/Loc. Sequence
Length
Scan Number
Time Diff.
Mapping:
MulRan DCC 03
5.7 km
7479
-
MulRan KAIST 02
6.3 km
8941
-
MCD NTU 08
3.8 km
6023
-
Localization:
MulRan DCC 01
4.9 km
5542
20 days
MulRan DCC 02
5.2 km
7561
1 month
MulRan KAIST 01
6.3 km
8226
2 months
MulRan KAIST 03
6.4 km
8629
10 days
MCD NTU 01
3.8 km
6023
2 hours
MCD NTU 02
0.64 km
2288
2 hours
MCD NTU 04
0.64 km
2288
2 hours
MCD NTU 10
0.64 km
2288
2 hours
MCD NTU 13
1.23 km
2337
2 days
MCD NTU 01 repet.
0.26 km
572
2 hours
localization sequence will be queried against a map built
by a mapping sequence. As demonstrated in Table II, our
proposed method, Outram-GSF, establishes consistent per-
formance advantages over all other counterparts. On MulRan
DCC, our method achieves 90.83% and 94.27% on sequences
01 and 02, respectively, outperforming previous best methods
including SGTD [6] and Outram [3]. On MulRan KAIST,
Outram-GSF again leads with up to 93.26% localization
success, demonstrating strong generalization across different
urban environments. Notably, on MCD NTU, our method
reaches 99.32% on sequence 10 and maintains robust per-
formance across all sequences, with an overall repetitive-
scene success rate of 87.36%, surpassing SGTD (55.26%)
and Outram (54.72%) by a significant margin.
Compared to LCD-based approaches, which tend to
struggle under appearance changes and repetitive struc-
tures, Outram-GSF shows substantial improvements. While
Ring++ [34] performs best among LCD methods, its
repetitive-scene performance (60.05%) still lags behind our
method by over 27%. Other LCD baselines such as GOS-
Match [32], STD [10], and BTC [11] perform noticeably
worse, particularly in challenging scenarios.

<!-- page 7 -->
TABLE IV
AVERAGE TRANSLATION/ROTATION ERROR AND RUNTIME
ATE/ARE [meter/◦]
t [ms]
MulRan DCC
MulRan Kaist
MCD NTU
STD
2.09/1.52
0.68/0.55
0.56/1.67
9.34
BTC
1.41/1.56
0.53/0.76
0.62/1.42
13.2
BEVplaces++
2.76/1.95
1.04/1.68
1.27/2.05
227.6
Ring++
1.62/0.75
0.63/0.50
0.48/1.15
382.6
SGDT
1.51/0.59
0.54/0.52
0.36/1.28
123.3
Outram
1.55/1.82
0.92/0.83
0.50/1.86
305.7
Outram-GSF
1.54/1.88
0.87/0.72
0.60/2.01
532.7
Ours w/ rf
1.38/0.52
0.55/0.48
0.41/1.28
563.6
These results highlight the superiority of Outram-GSF
in achieving robust and accurate global localization across
diverse and challenging environments. The gains over prior
registration-based methods confirm the effectiveness of our
framework in resolving long-range data associations and
handling semantic aliasing, making it well-suited for real-
world deployments. More importantly, our method performs
noticeably better on the sequence featuring repetitive scenes,
compared to previous semantic registration-based sota meth-
ods [3], [6]. This observation validates the effectiveness of
the Gaussian semantic field in modeling fine-grained geo-
semantic information for semantic disambiguation. It is also
worth noting that the proposed Outram-GSF works with-
out any post-geometric verification, whereas it outperforms
verification-based counterparts [6], [11].
B. Pose Estimation Accuracy and Runtime Results
We evaluate the localization accuracy and runtime effi-
ciency of our proposed method on all sequences of the
three datasets: MulRan DCC, MulRan KAIST, and MCD
NTU. A variant of our proposed method with subsequent
GICP [37] refinement (Ours w/ rf) is also presented. As
reported in Table IV, our method with refinement (Ours
w/ rf) achieves the lowest average translation and rotation
errors across all benchmarks, demonstrating superior pose
estimation accuracy under diverse urban conditions.
Specifically, our method achieves an average translation
error (ATE) of 1.38m, 0.55m, and 0.41m, and a rotation
error (ARE) of 0.52◦, 0.48◦, and 1.28◦on MulRan DCC,
KAIST, and MCD NTU, respectively. These results repre-
sent state-of-the-art performance, significantly outperforming
both retrieval-based methods (e.g., STD, BEVplaces++) and
other registration-based baselines (e.g., SGTD, Outram).
While our refined pipeline incurs higher computational
cost (563.6 ms/frame), the substantial gain in accuracy
justifies the trade-off for applications that prioritize robust
and precise global localization. Compared to fast but less
accurate methods such as STD [10] (9.34 ms/frame) and
BTC [11], our approach is better suited for long-range, large-
scale deployment where localization reliability is essential.
These results confirm the effectiveness of our global regis-
tration framework employing the Gaussian semantic field in
delivering high-fidelity 6-DoF pose estimates across complex
real-world environments.
Fig. 4.
A visual comparison of the correspondence establishment and
global localization performance in a symmetric environment of the proposed
method with (right) and without (left) the Gaussian semantic field. Pure
instance-level methods result in twisted correspondences (see the reversal
correspondences in red ).
V. LIMITATIONS
Although our proposed Outram-GSF demonstrates robust
performance across multiple datasets encompassing diverse
environmental conditions, it may encounter limitations in
certain scenarios, particularly in purely symmetric or highly
repetitive environments. This observation motivates the de-
velopment of symmetry-aware algorithms that can assess
and provide localizability certificates for given environments,
thereby predicting the feasibility of reliable localization prior
to deployment.
VI. CONCLUSION
This work introduces Outram-GSF, a one-shot LiDAR-
based global localization framework. Unlike previous ap-
proaches in the field, our method employs a Gaussian se-
mantic field representation to capture continuous semantic
distributions, moving beyond the limitations of conventional
cluster centroid-based techniques. The resulting localization
system demonstrates exceptional robustness and achieves
superior performance compared to state-of-the-art methods
across diverse benchmark datasets, particularly excelling in
challenging scenarios characterized by geometric repetition
and structural symmetry.
ACKNOWLEDGMENTS
REFERENCES
[1] I. Armeni, Z.-Y. He, J. Gwak, A. R. Zamir, M. Fischer, J. Malik, and
S. Savarese, “3d scene graph: A structure for unified semantics, 3d
space, and camera,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2019, pp. 5664–5673.
[2] S. Wang, F. Cao, T. Wang, X. Chen, and S. Shao, “Sgt-llc: Lidar
loop closing based on semantic graph with triangular spatial topology,”
IEEE Robotics and Automation Letters, 2025.
[3] P. Yin, H. Cao, T.-M. Nguyen, S. Yuan, S. Zhang, K. Liu, and L. Xie,
“Outram: One-shot global localization via triangulated scene graph
and global outlier pruning,” in 2024 IEEE International Conference
on Robotics and Automation (ICRA). IEEE, 2024, pp. 13 717–13 723.
[4] N. Wang, X. Chen, C. Shi, Z. Zheng, H. Yu, and H. Lu, “Sglc:
Semantic graph-guided coarse-fine-refine full loop closing for lidar
slam,” IEEE Robotics and Automation Letters, 2024.
[5] W. Ma, H. Yin, P. J. Wong, D. Wang, Y. Sun, and Z. Su, “Tripletloc:
One-shot global localization using semantic triplet in urban environ-
ments,” IEEE Robotics and Automation Letters, 2024.

<!-- page 8 -->
[6] F. Huang, W. Gao, S. Pan, H. Liu, and H. Zhao, “Sgtd: A semantic-
guided triangle descriptor for one-shot lidar-based global localization,”
IEEE Robotics and Automation Letters, 2025.
[7] R. Dub´e, D. Dugas, E. Stumm, J. Nieto, R. Siegwart, and C. Cadena,
“Segmatch: Segment based place recognition in 3d point clouds,”
in 2017 IEEE international conference on robotics and automation
(ICRA).
IEEE, 2017, pp. 5266–5272.
[8] R. Dube, A. Cramariuc, D. Dugas, H. Sommer, M. Dymczyk, J. Nieto,
R. Siegwart, and C. Cadena, “Segmap: Segment-based mapping and
localization using data-driven descriptors,” The International Journal
of Robotics Research, vol. 39, no. 2-3, pp. 339–355, 2020.
[9] D. N. Oliveira, J. Knights, S. B. Laina, S. Boche, W. Burgard, and
S. Leutenegger, “Regrace: A robust and efficient graph-based re-
localization algorithm using consistency evaluation,” arXiv preprint
arXiv:2503.03599, 2025.
[10] C. Yuan, J. Lin, Z. Zou, X. Hong, and F. Zhang, “Std: Stable triangle
descriptor for 3d place recognition,” in 2023 IEEE international
conference on robotics and automation (ICRA).
IEEE, 2023, pp.
1897–1903.
[11] C. Yuan, J. Lin, Z. Liu, H. Wei, X. Hong, and F. Zhang, “Btc: A
binary and triangle combined descriptor for 3-d place recognition,”
IEEE Transactions on Robotics, vol. 40, pp. 1580–1599, 2024.
[12] J. Ankenbauer, P. C. Lusk, A. Thomas, and J. P. How, “Global local-
ization in unstructured environments using semantic object maps built
from various viewpoints,” in 2023 IEEE/RSJ international conference
on intelligent robots and systems (IROS). IEEE, 2023, pp. 1358–1365.
[13] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stach-
niss, and J. Gall, “Semantickitti: A dataset for semantic scene un-
derstanding of lidar sequences,” in Proceedings of the IEEE/CVF
international conference on computer vision, 2019, pp. 9297–9307.
[14] L. Gan, R. Zhang, J. W. Grizzle, R. M. Eustice, and M. Ghaffari,
“Bayesian spatial kernel smoothing for scalable dense semantic map-
ping,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 790–
797, 2020.
[15] M. Ghaffari Jadidi, J. Valls Miro, and G. Dissanayake, “Gaussian
processes autonomous mapping and exploration for range-sensing
mobile robots,” Autonomous Robots, vol. 42, no. 2, pp. 273–290, 2018.
[16] S. T. O’Callaghan and F. T. Ramos, “Gaussian process occupancy
maps,” The International Journal of Robotics Research, vol. 31, no. 1,
pp. 42–62, 2012.
[17] S. Kim and J. Kim, “Continuous occupancy maps using overlapping
local gaussian processes,” in 2013 IEEE/RSJ international conference
on intelligent robots and systems.
IEEE, 2013, pp. 4709–4714.
[18] F. Ramos and L. Ott, “Hilbert maps: Scalable continuous occupancy
mapping with stochastic gradient descent,” The International Journal
of Robotics Research, vol. 35, no. 14, pp. 1717–1730, 2016.
[19] S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene
labelling and understanding with implicit scene representation,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 15 838–15 847.
[20] S. Vora, N. Radwan, K. Greff, H. Meyer, K. Genova, M. S. Sajjadi,
E. Pot, A. Tagliasacchi, and D. Duckworth, “Nesf: Neural semantic
fields for generalizable semantic segmentation of 3d scenes,” arXiv
preprint arXiv:2111.13260, 2021.
[21] Y. Chang, N. Hughes, A. Ray, and L. Carlone, “Hydra-multi: Collabo-
rative online construction of 3d scene graphs with multi-robot teams,”
in 2023 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS).
IEEE, 2023, pp. 10 995–11 002.
[22] F. Pomerleau, F. Colas, R. Siegwart, et al., “A review of point cloud
registration algorithms for mobile robotics,” Foundations and Trends®
in Robotics, vol. 4, no. 1, pp. 1–104, 2015.
[23] J. Strader, N. Hughes, W. Chen, A. Speranzon, and L. Carlone, “Indoor
and outdoor 3d scene graph generation via language-enabled spatial
ontologies,” IEEE Robotics and Automation Letters, 2024.
[24] H. Cao, Y. Xu, J. Yang, P. Yin, S. Yuan, and L. Xie, “Mopa: Multi-
modal prior aided domain adaptation for 3d semantic segmentation,”
in 2024 IEEE International Conference on Robotics and Automation
(ICRA).
IEEE, 2024, pp. 9463–9470.
[25] ——, “Multi-modal continual test-time adaptation for 3d semantic seg-
mentation,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 18 809–18 819.
[26] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine
learning.
MIT press Cambridge, MA, 2006, vol. 2, no. 3.
[27] E. Snelson and Z. Ghahramani, “Sparse gaussian processes using
pseudo-inputs,” Advances in neural information processing systems,
vol. 18, 2005.
[28] A. Mallasto and A. Feragen, “Learning from uncertain curves: The
2-wasserstein metric for gaussian processes,” Advances in Neural
Information Processing Systems, vol. 30, 2017.
[29] X. Chen, A. Milioto, E. Palazzolo, P. Giguere, J. Behley, and
C. Stachniss, “Suma++: Efficient lidar-based semantic slam,” in 2019
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS).
IEEE, 2019, pp. 4530–4537.
[30] F. Xue, I. Budvytis, and R. Cipolla, “Sfd2: Semantic-guided feature de-
tection and description,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2023, pp. 5206–5216.
[31] R. A. Rossi, D. F. Gleich, and A. H. Gebremedhin, “Parallel maxi-
mum clique algorithms with applications to network analysis,” SIAM
Journal on Scientific Computing, vol. 37, no. 5, pp. C589–C616, 2015.
[32] Y. Zhu, Y. Ma, L. Chen, C. Liu, M. Ye, and L. Li, “Gosmatch: Graph-
of-semantics matching for detecting loop closures in 3d lidar data,”
in 2020 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS).
IEEE, 2020, pp. 5151–5157.
[33] L. Luo, S.-Y. Cao, X. Li, J. Xu, R. Ai, Z. Yu, and X. Chen,
“Bevplace++: Fast, robust, and lightweight lidar global localization for
unmanned ground vehicles,” arXiv preprint arXiv:2408.01841, 2024.
[34] X. Xu, S. Lu, J. Wu, H. Lu, Q. Zhu, Y. Liao, R. Xiong, and Y. Wang,
“Ring++: Roto-translation invariant gram for global localization on a
sparse scan map,” IEEE Transactions on Robotics, vol. 39, no. 6, pp.
4616–4635, 2023.
[35] G. Kim, Y. S. Park, Y. Cho, J. Jeong, and A. Kim, “Mulran:
Multimodal range dataset for urban place recognition,” in 2020 IEEE
international conference on robotics and automation (ICRA).
IEEE,
2020, pp. 6246–6253.
[36] T.-M. Nguyen, S. Yuan, T. H. Nguyen, P. Yin, H. Cao, L. Xie,
M. Wozniak, P. Jensfelt, M. Thiel, J. Ziegenbein, et al., “Mcd:
Diverse large-scale multi-campus dataset for robot perception,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 22 304–22 313.
[37] K. Koide, “small gicp: Efficient and parallel algorithms for point cloud
registration,” Journal of Open Source Software, vol. 9, no. 100, p.
6948, 2024.
