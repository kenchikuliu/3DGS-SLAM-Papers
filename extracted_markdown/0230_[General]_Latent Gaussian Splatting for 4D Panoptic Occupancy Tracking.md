<!-- page 1 -->
Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking
Maximilian Luz1, Rohit Mohan1, Thomas N¨urnberg2, Yakov Miron2,3, Daniele Cattaneo1, Abhinav Valada1
Abstract— Capturing 4D spatiotemporal surroundings is
crucial for the safe and reliable operation of robots in dynamic
environments. However, most existing methods address only
one side of the problem: they either provide coarse geometric
tracking via bounding boxes, or detailed 3D structures like voxel-
based occupancy that lack explicit temporal association. In this
work, we present Latent Gaussian Splatting for 4D Panoptic
Occupancy Tracking (LaGS) that advances spatiotemporal scene
understanding in a holistic direction. Our approach incorporates
camera-based end-to-end tracking with mask-based multi-view
panoptic occupancy prediction, and addresses the key challenge
of efficiently aggregating multi-view information into 3D voxel
grids via a novel latent Gaussian splatting approach. Specifically,
we first fuse observations into 3D Gaussians that serve as a sparse
point-centric latent representation of the 3D scene, and then
splat the aggregated features onto a 3D voxel grid that is decoded
by a mask-based segmentation head. We evaluate LaGS on the
Occ3D nuScenes and Waymo datasets, achieving state-of-the-art
performance for 4D panoptic occupancy tracking. We make our
code available at https://lags.cs.uni-freiburg.de/.
I. INTRODUCTION
Camera-based 4D panoptic occupancy tracking (4D-
POT) [1] combines dense geometric reconstruction, semantic
understanding, and temporal consistency into a unified
framework. Through this holistic formulation, it promises
a comprehensive representation of dynamic environments,
addressing key shortcomings of prior paradigms [2]. Specif-
ically, classical box-based tracking methods model scenes
only with coarse cuboids [3]–[5], lacking fine-grained ge-
ometry and volumetric semantics. Conversely, standard 3D
occupancy prediction [6]–[8] typically operates per frame,
producing dense voxel grids but without instance identities or
temporal association. In contrast, 4D-POT assigns a semantic
class to every voxel in the 3D scene while simultaneously
distinguishing and tracking individual object instances across
time. By bridging the gap between geometric fidelity and
instance-level awareness, 4D-POT has the potential to serve
as the central perception backbone for autonomous agents.
Despite the relative novelty of 4D-POT, existing approaches
largely follow a direct composition of mask-based 3D
occupancy prediction with query-based end-to-end 3D multi-
object tracking (3D MOT) [1], primarily by adapting state-
of-the-art occupancy prediction approaches. While this is a
viable starting point, we argue that the additional temporal and
instance-level requirements motivate a reconsideration of the
underlying representation and encoder design. In particular,
we explore 3D Gaussians as an intermediate sparse latent
1Department of Computer Science, University of Freiburg, Germany.
2Bosch Research, Robert Bosch GmbH, Germany.
3University of Haifa, Israel
Fig. 1.
Illustration of our latent Gaussian representation. Bottom left:
panoptic voxel predictions. Center: latent Gaussian features colorized via
principal component analysis, splatted to 2D, and overlaid over voxel
predictions. Top right: latent Gaussians.
representation that replaces classical dense 3D voxel feature
volumes. Prior work has leveraged Gaussians in 3D occupancy
prediction either at the output level [9], [10] or for 2D
self-supervision [11]. Here, we instead use Gaussians as a
feature-bearing representation that shifts dense voxel-centric
architectures towards sparse point-centric ones.
Our key idea is to treat 3D Gaussians as dynamic volume-
oriented keypoints in space. We first efficiently aggregate
features for each Gaussian before splatting them back to
a 3D voxel grid for final refinement and decoding. This
design yields a sparse latent representation that preserves
dense scene context while improving encoder efficiency
and effectiveness. We term this concept as latent Gaussian
splatting (LaGS), illustrated in Figure 1. This point-centric
reformulation enables larger and more flexible data-dependent
attention neighborhoods, improving information exchange
and scaling behavior compared to more classical dense voxel
encoders such as COTR [8], which tend to have smaller
and more limited neighborhood relations. By exploiting
hierarchical representations, we further improve our encoder
and aggregate features into compressed Gaussian superpoints
that aid in both 2D-to-3D lifting and mask decoding, making
full quadratic-complexity attention operations feasible.
Beyond representation, we identify two major challenges in
mask-based 4D-POT. First, moving from semantic occupancy
to the panoptic setting requires predicting masks not only for
global semantic classes but also for individual local instances.
This leads to a considerable imbalance between the broad
arXiv:2602.23172v1  [cs.CV]  26 Feb 2026

<!-- page 2 -->
and global (non-instance stuff) masks and the locally con-
strained instance (thing) masks. While it may be conceptually
appealing to treat both semantic and instance queries in a
unified manner [1], we find that this imbalance must be
handled explicitly. Notably, we observe that instead of jointly
aggregating instance and semantic masks, aggregating them
separately before merging both aggregated volumes enhances
instance segmentation quality considerably. Second, this shift
also drastically increases memory requirements, as the number
of instance queries typically needs to be many times higher
than the maximum number of expected instances [12]–[15].
This is further aggravated when integrating classical query-
based end-to-end tracking approaches (such as MUTR3D [14])
directly, which generally backpropagate gradients across mul-
tiple frames, scaling resource requirements linearly with the
number of processed frames. However, our approach indicates
that this is not required. Instead, we find that propagating
detached queries forward and optimizing each frame in the
multi-frame training sequence independently frees significant
resources that are better spent in the decoder transformer.
In summary, our primary contributions are six-fold: (1) We
introduce Gaussians as a sparse intermediate feature repre-
sentation for dense 3D/4D prediction, extending Gaussian-to-
voxel splatting [10] from semantics to features, yielding a
more effective and scalable 3D voxel feature encoder. (2) We
streamline the integration of query-based end-to-end tracking
and mask-based panoptic occupancy prediction, resulting a
new state-of-the-art 4D-POT approach. (3) We re-evaluate
previously proposed metrics for 4D-POT [1] and address
inaccuracies in current implementations. (4) We extend 4D-
POT to the nuScenes dataset [16] more commonly used for 3D
occupancy prediction, reproduce baselines previously reported
only on Waymo [17], and provide groundtruth 4D panoptic
occupancy annotations. (5) Through extensive evaluations,
we demonstrate state-of-the-art results on both datasets by
up to +18.9 p.p. in occupancy segmentation and tracking
quality (STQ) [1]. (6) We make the code and models publicly
available upon acceptance.
II. RELATED WORK
In this section, we first briefly review methods for 3D
occupancy prediction, followed by a discussion of sparse
representations addressing the dense 3D nature of the task,
and finally cover 4D panoptic occupancy tracking.
3D Occupancy Prediction: While the majority of 3D
perception in autonomous driving still relies on bounding-
box representations [18], the promise of finer geometric
details has led to significant advances in 3D semantic
occupancy prediction in recent years [8], [19]. Notably,
Occ3D [20] extends the widely adopted nuScenes [16]
and Waymo [17] datasets to semantic occupancy prediction,
providing more challenging dynamic scenes. Current state-
of-the-art thereon [6]–[8] largely follows MaskFormer [21],
employing a mask-based transformer decoder and posing the
task as 3D segmentation. SparseOcc [22] and PaSCo [23]
show that, akin to 2D panoptic image segmentation, such
a decoder can be adapted straightforwardly to predict 3D
panoptic occupancy. However, while this can be conceptually
easy, we observe that treating (local) instance and (global)
semantic masks as equals (as done in these prior approaches)
can lead to underconfident instance segmentation due to the
larger semantic masks generally dominating, requiring careful
handling.
Sparse 3D Occupancy Prediction: A major practical
challenge in 3D semantic occupancy prediction is the dense
3D nature of the task. Several works explore more compressed
representations [8], [24] or the intrinsic sparsity of the
task by focusing on occupied regions only [22], [25], yet
retain largely voxel-aligned representations. Notably deviating
from this, GaussianFormer [9] proposes 3D Gaussians as a
representation for both occupied and free space, followed
by GaussianFormer-2 [10] modeling only occupied space
for a truly sparse object-centric approach. Both methods
splat the predicted 3D Gaussians to the final 3D voxel
grid for the task output. We believe that this represents an
opportunity to integrate concepts from recent point-based
3D perception methods [26], [27]. Specifically, it allows
treating Gaussians as superpoints in space, yielding a sparse
latent representation, alleviating costly dense 3D processing,
that can be converted into a dense representation at will via
splatting. Our novelty, therefore, lies in treating splatting
as an intermediate step in the encoder, splatting features
instead of appearance or semantics. Moreover, the sparse
representation allows our approach to reason more effectively,
across larger and more flexible neighborhoods, improving
exchange and aggregation of information, 2D-to-3D lifting,
and scalability. While sparsity has also been explored in
classical object detection [28], the prevalent architectures
remain dense networks.
4D Panoptic Occupancy Tracking: By extending 3D panop-
tic occupancy prediction temporally, TrackOcc [1] introduces
the task of 4D panoptic occupancy tracking (4D-POT). While
it can be conceptually understood as a 3D extension to the
task of video panoptic segmentation (VPS) [29], its dense
3D nature requires specifically tailored approaches due to its
inherent computational complexity. To this end, TrackOcc
incorporates principles previously explored in end-to-end
3D multi-object tracking [12]–[15], ensuring temporally
consistent instance assignments by propagating instance-
mask-queries forward across frames. While TrackOcc relies
on MUTR3D [14] for query-based tracking, we base our
approach on PF-Track [15] to incorporate gains from their
comparatively lightweight temporal query refinement and
further streamline our approach to allow for more than one
decoder layer, significantly improving tracking performance.
III. METHOD
4D panoptic occupancy tracking is a highly challenging
task, involving close interaction of several components
(Section III-A). At its core, our approach (illustrated in Fig. 2)
combines concepts from mask-based segmentation [21], [30],
[31] with transformer-based end-to-end tracking [12], [14],
[15], treating each object instance as a separate tracked query.
It can be divided into two major parts: a 3D occupancy volume

<!-- page 3 -->
Multi-View Images
Depth
Features
D
F
Image Encoder
lift
N
pool
D
F
V0
V2
Voxel Backbone
SA
CA
FFN
WSA
SCA
FFN
SMSA
FFN
FFN
N×
F
sample
sample
V2
V0
G2
G0
splat features to voxels
merge
V0
V2
V
Latent Gaussian Encoder
SA
CA
VCA
CA
FFN
F
V
G2
M×
Sem.
Detect
Track
N
V
Occupancy & IDs
Panoptic Mask Decoder
Memory
refine
to t + 1
Query Propagation
Fig. 2.
Architecture overview (left to right). An image encoder ( ) produces image features F and depth D. Features are lifted via depth to a 3D volume
and further encoded into a 3D voxel feature pyramid (V0, V2,
). Our latent Gaussian encoder ( ) samples points from the pyramid volumes and processes
them in a coarse (left) and fine (right) stream, using self-attention (SA), windowed self-attention (WSA), cross-attention (CA), spatial cross-attention (SCA),
and feed-forward networks (FFN). Our novel Serialized Multi-Stream Attention (SMSA) facilitates information exchange between streams. Refined points
are decoded as Gaussians (G0, G2) and splatted back to a 3D feature volume, which is then refined to the final voxel volume V . Our transformer decoder
( ) then decodes this volume into semantic and instance masks using volume cross-attention (VCA) for efficient query-to-3D-volume attention. Tracking is
facilitated by the tracking-by-attention paradigm. We refine track queries by spatio-temporal reasoning before passing them onto the next frame ( ).
encoder (Sections III-B and III-C) and a 4D panoptic mask
decoder (Sections III-D and III-E). Notably, our occupancy
encoder is founded on the same principles as COTR [8],
combining explicit (Section III-B) and implicit (Section III-
C) view transforms, adhering to the same general structure.
Our overall approach begins by lifting and aggregating
image features into 3D volume features (Section III-B), which
are further refined by our novel latent Gaussian encoder
(Section III-C). Given the refined 3D feature volume, a
panoptic mask decoder (Section III-D) then uses a query-
centric approach to decode semantic and instance masks.
Tracking (Section III-E) is facilitated via the same queries
following the tracking-by-attention paradigm [12], [13] and
enhanced by an additional spatio-temporal query refinement
module based on PF-Track [15]. Sections III-F and III-G
provide further details for training and inference, respectively.
A. Task Definition
Given a consecutive sequence of multi-view images I =
{Ij
t , Ij
t−1, . . . Ij
t−T }N
j=1, where t is the current timestep, T the
sequence length, and N the number of camera views, camera-
based 4D panoptic occupancy tracking requires joint predic-
tion of occupancy, semantics, and instance associations for
each 3D voxel surrounding the ego-vehicle [1]. Specifically,
the task requires predicting (cp,t, ip,t) for each voxel position
p, where cp,t represents the semantic class and occupancy
state (i.e., category including unknown/other and free) and ip,t
the associated instance ID at timestep t. Notably, semantic
predictions incorporate both tracked thing and non-tracked
stuff classes, while instance IDs are only assigned for tracked
thing classes. Intrinsic and extrinsic parameters of all cameras
as well as ego-motion of the vehicle are assumed to be known.
B. Image Encoder, Explicit Lifting, and Voxel Backbone
We begin by extracting multi-view image features F with
an off-the-shelf image encoder (Fig. 2,
). A small head is
trained to predict a binned depth distribution D independently
for each image, which is then used to lift the image features
(explicitly) to 3D via the outer product P = F ⊗D, creating a
pseudo point cloud. This is subsequently pooled to a 3D voxel
grid representation and further refined into a voxel feature
pyramid (V0, V2) using standard 3D convolutions (Fig. 2,
).
C. Latent Gaussian Occupancy Feature Encoder
Instead of further refining the dense voxel representation,
as done in COTR [8], we explore a novel way of representing
3D features as latent Gaussians, treating them as volumetric
keypoints. This point-centric representation allows us to
leverage insights from recent point transformer networks [27].
By serializing points via space-filling curves, we can achieve
larger and more flexible receptive fields than dedicated
dense 3D operations (such as COTR’s voxel self-attention
[8]), improving scalability, and replacing fixed neighborhood
relations with learned ones through positional encodings. Our
final encoder (Fig. 2,
) further extends this hierarchically and
employs two parallel streams: a higher-resolution (fine) stream
(G0) with more points, capturing details, and a coarse stream
(G2), allowing us to aggregate a smaller set of super-points
that can subsequently aid query refinement and decoding.
Point-Based Refinement (Fine Stream): Given a voxel
feature volume V0, we use multinomial sampling with
probability scores based on the feature magnitude to obtain a
set of k voxels. These voxels act as seed points for our latent
Gaussian representation (G0) and allow us to shift from the
dense domain to sparse point-based transformer architectures.
Specifically, we follow Point Transformer V3 [27] and first
serialize our seed points via a space-filling curve before

<!-- page 4 -->
enriching them with a multi-layer transformer architecture. We
employ sliding-window self-attention (WSA) for efficiency,
ensuring required memory scales linearly with the number of
points k. To efficiently integrate image features for a large
number of points, we rely on spatial cross-attention (SCA)
[32], projecting the points into the images and thereon using
2D deformable attention.
Hierarchical Extension (Coarse Stream): We further en-
hance our encoder by introducing the concept of hierarchical
streams: instead of sampling k points from a single voxel
feature volume, we rely on the feature pyramid (V0, V2),
sampling from multiple scales, serializing and processing
streams (G0, G2) independently. This allows us to create a
coarser stream (G2) with fewer points, derived from a coarser
scale (V2), where full point-wise self-attention and full image-
point cross-attention are feasible, while, at the same time,
retaining details via a higher-resolution finer stream (G0).
To facilitate efficient stream-to-stream communication, we
devise a novel cross-stream attention operation: Serialized
Multi-Stream Attention (SMSA). SMSA first merges all
streams before re-serializing all points with a single space-
filling curve to obtain a single linearized stream. Through this,
it can then jointly refine features and exchange information via
windowed self-attention. Finally, it splits the unified stream
back up into the original hierarchical streams. Notably, SMSA
handles varying densities of points and streams intrinsically,
without special consideration.
Gaussian Feature Aggregation: Our downstream archi-
tecture depends on a 3D feature volume. To this end, we
adapt 3D Gaussian occupancy splatting [10] for voxel feature
aggregation. Specifically, for each point j, we predict centers
µj, covariances Σj (composed from scale and rotation),
opacities αj, and feature embeddings ej ∈RC. From this,
occupancy o and voxel features f are computed as
o(x) = 1 −
Y
j

1 −exp
 −1
2∥x −µj∥2
Σ−1
j

,
(1)
f(x) = o(x) ·
P
j αjGj(x)ej
P
j αjGj(x) ,
(2)
where Gj(x) = N(x | µj, Σj) is the 3D Gaussian PDF,
and ∥v∥2
M = v⊺Mv denotes Mahalanobis distance. Note
that the decoded Gaussians only represent occupied space.
Hierarchical streams are concatenated before aggregation.
Subsequently, we merge the aggregated feature volume with
the initial feature pyramid volumes, essentially introducing
skip connections and creating a U-Net-like structure.
D. Panoptic Mask Decoder
We use a mask transformer decoder (Fig. 2,
) with detec-
tion queries for instance (thing) segmentation and semantic
queries for global instance-less (stuff) semantic masks. As this
aligns conceptually well with detection transformers, we adapt
PETR [33]. PETR uses cross-attention between multi-view
images and queries, which we further extend by letting queries
attend to the voxel features via 3D deformable cross-attention
and, in the case of hierarchical Gaussian encoding, also to the
refined coarse encoder point features. After refinement by the
transformer, queries are decoded into a mask embedding and
semantic class scores. Similar to MaskFormer [21], binary
occupancy masks are computed via a dot product between
the mask embedding and the voxel features.
E. Tracking, Query Propagation, and Refinement
To facilitate tracking, we propagate successfully decoded
detection queries from the decoder forward to the next frame,
following the established tracking-by-attention paradigm [12],
[13]. These forwarded track queries are then used together
with newly initialized detection as well as (non-temporal) se-
mantic/stuff queries as input to the decoder in this subsequent
frame. Newly instantiated and successfully decoded detection
queries introduce new tracks, while successfully decoded
track queries continue old ones. To further improve tracking
performance, we integrate the spatio-temporal refinement
module of PF-Track [15] (Fig. 2,
), refining queries based
on memory (past) and trajectory prediction (future). Predicted
trajectories are also used to fill gaps for intermittently missed
or low-confidence detections.
F. Training and Supervision
We supervise our approach on multiple levels. Following
COTR [8], we supervise the depth prediction for explicit
feature lifting in the encoder via sparse depth from LiDAR.
Further, we add a small head to decode semantic scores
for each Gaussian, splatting them to a semantic voxel grid
for direct supervision via a cross-entropy loss, akin to
GaussianFormer-2 [10]. The decoder is supervised via both
semantic and instance masks as well as box predictions
(akin to center supervision in TrackOcc [1]). For detection
queries, we use bipartite matching, considering both predicted
boxes and masks to assign ground truth instances. Once a
ground-truth instance has been assigned, it is kept across
all subsequent training frames. For semantic queries, we
use bipartite matching with masks only. Supervision of the
spatio-temporal refinement module follows PF-Track [15].
To enable temporal supervision, we train on short multi-
frame sequences. We prevent linear scaling of gradient buffers
by detaching track and detection queries after decoding and
before refinement, meaning gradients from subsequent frames
still flow back to the refinement stage, but not the decoder
itself, decoupling individual frames.
G. Inference
The raw predictions of our approach are class scores cq
and occupancy mask scores mq,x for both instance and stuff
queries. We score queries via the maximum class score, i.e.,
sq = ∥cq∥∞, filter out inactive ones via a threshold, and
compute the dominant query ˆqx = arg maxq{sq · mq,x} for
each voxel x. In contrast to previous methods [1], [21], [30],
we compute dominant queries independently for instance and
stuff classes, merging both afterward by overriding the stuff
predictions with instance ones.

<!-- page 5 -->
IV. EXPERIMENTS
The effectiveness of our proposed approach is demonstrated
by extensive evaluation. To this end, we first describe
the datasets used for benchmarking in Section IV-A and
the evaluation metric in Section IV-B, including necessary
revisions to correct existing inaccuracies. Brief descriptions
of the baselines follow in Section IV-C, along with implemen-
tation and training details in Section IV-D. Benchmarking
results are presented in Section IV-E, followed by a detailed
ablation study of the architectural components (Section IV-F),
concluding with qualitative comparisons (Section IV-G). All
evaluations are performed on the respective validation splits.
A. Datasets
We evaluate our approach on both nuScenes [16] and
Waymo [17], with 3D occupancy ground-truth provided by
Occ3D [20]. For both datasets, the spatial range is bounded
from −40 m to 40 m for x and y and from −1 m to 5.4 m
for z, with a voxel size of 0.4 m in each axis, resulting in
a voxel grid resolution of 200×200×16. For Waymo, we
follow TrackOcc [1] and subsample the dataset at every 5th
frame, yielding 789 training scenes and 202 validation scenes
with 40 samples each. For nuScenes, we train and evaluate on
the full dataset with 700 training and 150 validation scenes,
with each scene containing around 40 samples.
To obtain 4D panoptic occupancy labels for nuScenes, we
assign instance IDs to all thing-class voxels of the Occ3D
semantic occupancy data by using the ground-truth box
labels. Specifically, instance IDs are assigned based on the
intersecting box of the same class. Ambiguities, such as voxels
being intersected by none or multiple boxes, are resolved by
choosing the closest instance. Notably, the assigned instance
IDs are consistent with the nuScenes box instance IDs,
facilitating direct box-to-voxel correspondence. We make
this data preprocessing available alongside our code. For
Waymo, we rely on the data provided by TrackOcc [1].
B. Evaluation Metrics
Following TrackOcc [1], we adapt the Segmentation and
Tracking Quality (STQ), originally introduced for video
panoptic segmentation [34], to 4D occupancy prediction
and tracking. STQ is defined as the geometric mean of
Segmentation Quality (SQ) and Association Quality (AQ),
STQ =
p
SQ · AQ,
(3)
where SQ is the classical mean intersection over union (mIoU)
of semantic occupancy prediction [20], [35].
AQ represents the mean IoU between each ground-truth
and predicted 4D tube, where each individual IoU is weighted
by the respective intersected fraction to facilitate a soft
assignment and bound the AQ score by one. Mathematically,
we define the 4D panoptic occupancy predictions P =
{(p, t, i, c)} as the set over tuples of 3D voxel position p,
time step t, instance ID i, and class c. Equivalently, we define
G as the set of ground-truth tuples. From this, we derive
Pi = {(p, t, i) | (p, t, i, c) ∈P} and equivalently Gi as the
instance prediction and ground-truth for instance i. Finally,
we define
AQ =
1
|IG|
X
i∈IG
1
|Gi|
X
j∈IP
|Gi ∩Pj| · |Gi ∩Pj|
|Gi ∪Pj|,
(4)
where IG = {i | (p, t, i, c) ∈G} is the set of ground-truth
instance IDs and IP is the set of predicted instance IDs.
Extending over TrackOcc, we further propose the AQ1
metric for single-frame panoptic assessment. AQ1 is con-
structed analogously to Eq. (4), with the exception that we do
not consider tracking tubes but only single-frame instances
for matching, i.e., enforce ∀(p1, t1, i1, c1), (p2, t2, i2, c2) ∈
G : t1 ̸= t2 ⇒i1 ̸= i2. STQ1 follows analogously again
as the geometric mean over the already non-temporal mIoU
and AQ1. This avoids the well-documented shortcomings of
the panoptic quality (PQ) metric [34]. In addition to STQ,
AQ, and mIoU/SQ, we also use the binary IoU to assess the
quality of the binary free/non-free occupancy prediction.
Following Occ3D [20], we evaluate only on visible regions
(using the “camera” mask). Notably, AQ does not depend
on any class assignments, and SQ does not depend on any
instance information, strictly separating semantic and instance
segmentation between SQ and AQ. While mathematically
sound, however, we find that the metric implementations of
TrackOcc [1] are flawed: they solely consider areas occupied
in the ground-truth data and ignore regions marked as free
space. This skews the metric significantly, as any false
positives in known free space are disregarded, essentially only
counting true positives and false negatives. Mathematically,
this is equivalent to applying Eq. (4) to P ′
i = Pi ∩M and
G′
i = Gi ∩M, where M is a mask indicating occupied
space in G. We therefore reconstruct and re-evaluate the
baselines presented by TrackOcc, as well as TrackOcc itself,
and provide corrected implementations alongside our code.
C. Baselines
To fairly evaluate our approach under the revised metrics,
we reconstruct the baselines proposed by TrackOcc [1]:
MinVIS-inspired bipartite matching of queries across subse-
quent frames by cosine similarity [36], additionally extended
to a CTVIS-based approach where embeddings used for
matching are trained contrastively end-to-end [37], heuristic
box extraction from instance occupancy data and tracking
via AB3DMOT [38], and 4D-LiDAR-panoptic-segmentation-
inspired IoU-matching of subsequent predictions [39]. To
evaluate the actual effectiveness of the tracking approaches,
we propose an additional baseline: assigning new and inde-
pendent instance IDs each frame (Per-Frame). Through this,
we create a minimal metric target for the tracking approaches,
below which tracking is measurably ineffective and instead
hinders per-frame instance segmentation.
D. Implementation and Training Details
Following PF-Track [15], we choose VoVNetV2 [40] as our
image backbone with input resolution 800×320 for nuScenes
and 704×256 for Waymo. Image-to-3D lifting follows the
BEVDet4D variant of COTR [8] and TrackOcc [1], with

<!-- page 6 -->
TABLE I
4D-POT PERFORMANCE ON OCC3D-NUSCENES.
mIoU
Approach
STQ
AQ
STQ1
AQ1
all
things stuff
IoU
Per-Frame
9.0
2.5
21.8
14.7
32.5
26.4
41.2 59.2
MinVIS† [36]
11.8
4.3
21.8
14.7
32.5
26.4
41.2 59.2
CTVIS† [37]
11.4
3.9
22.5
15.4
33.0
27.0
41.5 59.9
4D-LCA† [39]
12.5
4.8
21.8
14.7
32.5
26.4
41.2 59.2
AB3DMOT† [38]
13.1
5.3
21.8
14.7
32.5
26.4
41.2 59.2
TrackOcc‡ [1]
12.2
4.7
19.7
12.1
32.1
25.3
41.8 59.8
LaGS-2s (Ours)
31.2
24.6
35.0
31.0
39.5
35.1
45.6 64.1
†: Baselines reproduced by us. ‡: Official code adapted for nuScenes.
TABLE II
4D-POT PERFORMANCE ON OCC3D-WAYMO.
mIoU
Approach
STQ
AQ
STQ1
AQ1
all
things stuff
IoU
Per-Frame
11.9
4.7
—
—
30.0
32.7
29.3
—
MinVIS† [36]
15.0
7.5
—
—
30.0
32.7
29.3
—
CTVIS† [37]
16.4
9.3
—
—
28.9
31.9
28.1
—
4D-LCA† [39]
16.2
8.7
—
—
30.0
32.7
29.3
—
AB3DMOT† [38]
18.0
10.8
—
—
30.0
32.7
29.3
—
TrackOcc‡ [1]
20.2
13.8
—
—
29.4
29.7
29.4
—
Per-Frame
9.1
4.0
18.1
15.6
20.9
21.9
20.6 55.5
MinVIS† [36]
11.0
5.8
18.1
15.6
20.9
21.9
20.6 55.5
CTVIS† [37]
12.5
7.3
18.7
16.5
21.2
22.3
20.9 56.9
4D-LCA† [39]
12.1
7.0
18.1
15.6
20.9
21.9
20.6 55.5
AB3DMOT† [38]
13.3
8.5
18.1
15.6
20.9
21.9
20.6 55.5
TrackOcc‡ [1]
15.2
10.7
18.1
15.2
21.6
21.9
21.5 57.5
LaGS-2s (Ours)
20.3
18.6
22.2
22.2
22.2
26.1
21.3 60.2
†: Baselines reproduced by us. ‡: Results reproduced with official code and weights.
Gray text: metrics as implemented by TrackOcc [1], ignoring false positives.
memory-intensive stereo and temporal multi-frame aggrega-
tion of input frames disabled. All approaches and baselines
are trained on 8 NVIDIA L40s for 24 epochs with a batch
size of 1. All baselines are based on a single-frame-adapted
version of TrackOcc and follow its training procedure [1].
Training of our method is split into 12 epochs of single-frame
pre-training and 12 epochs of tracking training, akin to PF-
Track [15]. We use a two-stream latent Gaussian encoder with
512 (coarse) and 8192 (fine) points and a window size of
1024 for all window-based attention operations. The encoder
and decoder both employ 4 transformer layers.
E. Comparison with State-of-the-Art
4D panoptic occupancy tracking results for nuScenes and
Waymo are presented in Tables I and II, respectively. LaGS
achieves significant improvements in both overall (STQ) and
pure tracking performance (AQ), with up to +18.9 p.p. STQ
and +19.8 p.p. AQ for nuScenes, and +5.1 p.p. STQ and
+7.9 p.p. AQ for Waymo. The larger increases on nuScenes
can be attributed to a more diverse set of tracked (thing)
classes, on which our method excels specifically (cf. mIoU-
things in Tables I and II). Qualitative evaluation (Fig. 3,
Fig. 4) shows a similar picture: The instance segmentation
of LaGS is significantly clearer, whereas instance masks in
TABLE III
3D OCCUPANCY PREDICTION PERFORMANCE ON OCC3D-NUSCENES.
Approach
mIoU
IoU
TPVFormer [24]
34.2
66.8
SurroundOcc [7]
34.6
65.5
OccFormer [6]
37.4
70.1
BEVDet4D [41]
39.3
73.8
BEVDet4D + COTR [8]
44.5
75.0
BEVDet4D + COTR (w/o longterm, stereo)∗[8]
34.6
66.3
TrackOcc‡ [1]
32.1
59.8
LaGS-2s (Ours)
39.5
64.1
‡: Official code adapted for nuScenes. ∗: Methodologically closest baseline.
Front Cam
Ground Truth
TrackOcc [1]
LaGS-2s (Ours)
time
2
1
3
3
3
4
Fig. 3.
Qualitative results on the Occ3D-nuScenes validation split. Our
approach shows clear improvements in (1) instance separation, (2) instance
association, (3) missing detections, and (4) underconfident detections.
Front Cam
Ground Truth
TrackOcc [1]
LaGS-2s (Ours)
time
1
1
2
1
3
Fig. 4.
Qualitative results on the Occ3D-Waymo validation split. Our
approach shows clear improvements in (1) instance association, (2) instance
separation, and (3) ID switches.

<!-- page 7 -->
TABLE IV
ABLATION STUDY ON THE LATENT GAUSSIAN ENCODER.
pre-training
tracking
Type
Gaussians
Layers AQ1
mIoU
IoU
AQ
mIoU
IoU
COTR
N/A
1
29.4
36.2
61.4 23.6
38.6
61.8
LaGS
8192
1
29.7
35.9
61.5 23.4
38.2
61.5
COTR
N/A
2
28.5
36.0
61.1 23.4
38.4
61.6
LaGS
8192
2
30.1
36.2
61.7 23.9
38.5
61.9
COTR
N/A
4
28.7
36.2
61.2 23.2
38.3
61.7
LaGS
2048
4
30.2
36.6
62.3 24.2
39.2
63.1
LaGS
4096
4
30.5
36.7
62.0 24.1
39.0
62.4
LaGS
8192
4
30.4
36.4
62.0 24.1
38.9
62.0
LaGS-2s {512, 2048}
4
30.9
37.0
62.7 24.6
39.4
63.8
LaGS-2s {512, 4096}
4
31.0
37.4
63.2 24.4
39.1
63.9
LaGS-2s {512, 8192}
4
31.1
37.2
63.0 24.6
39.5
64.1
Performance reported after both single-frame pre-training (12 epochs) and full
tracking training (12 epochs pre-training + 12 epochs tracking training). Shaded gray
background indicates the original COTR encoder. Shaded blue background indicates
our chosen configuration.
TABLE V
ABLATION STUDY ON THE DECODER TRANSFORMER.
mIoU
Layers
Refinement
STQ
AQ
all
things
stuff
IoU
1
✓
26.1
18.5
36.8
31.1
44.9
61.5
4
✗
29.1
22.7
37.2
32.2
44.3
61.7
4
✓
30.2
23.6
38.6
34.6
44.3
61.8
Performance reported using the COTR encoder with 1 encoder layer. Shaded
background indicates our chosen configuration.
TrackOcc [1] often seem comparatively underconfident, or
instance assignments are mixed. An adverse effect of this can
be seen on Waymo, where stuff classes are more diverse than
on nuScenes and gains in thing-class segmentation come at
a minor expense of stuff-class performance.
Notably, LaGS also makes a significant step towards closing
the gap in semantic occupancy performance (mIoU) between
single-frame and 4D-POT methods (see Table III). We
outperform the non-temporal non-stereo BEVDet4D+COTR
baseline (Table III, gray shade) used as a starting point for
TrackOcc [1] and our 4D-POT implementation by +4.9 p.p.
mIoU, achieving scores close to and higher than prior state-
of-the-art methods like SurroundOcc [7], OccFormer [6],
and BEVDet4D [41]. Bringing back temporal aggregation
and stereo-based lifting, dropped by TrackOcc [1] and our
method due to resource constraints, would likely bridge this
gap entirely.
F. Ablation Studies
Latent Gaussian Encoder: Effective aggregation of infor-
mation from images and spatial neighborhoods is crucial
for both 3D panoptic occupancy prediction and 4D panoptic
occupancy tracking. Hence, we validate the efficacy of our
proposed latent Gaussian encoder in both single-frame 3D
panoptic occupancy (pre-training) and 4D tracking scenarios
(Table IV). While our proposed encoder performs largely
similarly to the COTR [8] encoder with a single transformer
layer, its benefits lie in scaling, outperforming the COTR
encoder with both 2 and 4 layers. We believe that this is due
TABLE VI
ABLATION STUDY ON MASK AGGREGATION.
mIoU
Approach
STQ
AQ
AQ1
all
things
stuff
IoU
Unified
29.5
22.3
28.8
39.2
34.6
45.7
64.1
Split
31.1
24.5
31.0
39.5
35.1
45.6
64.1
+/−
+1.6
+2.2
+2.2
+0.3
+0.5
-0.1
0.0
Shaded background indicates our chosen configuration.
to the more dynamic neighborhoods of our approach. Both
our encoder and COTR rely on spatial cross-attention [32]
for efficiency and feasibility, incurring the drawback that all
points on a given view ray will be projected onto the same
image coordinate. This leads to image information being
erroneously attributed not just to the front surface but to
the full ray, especially when working with dense grids like
COTR. To reason about this, finding the correct attribution
and deduplicating the information, networks likely benefit
from exchanging information across larger data-dependent
regions. Therein, COTR is limited again: its use of deformable
attention [42] for voxel self-attention restricts queries to
attend to only a small (kp = 8) set of query-dependent
points around the query location. Through its point-based
design, our approach, however, allows for neighborhoods of
kw = 1024, in which queries can freely interact. This is
further improved upon in our dual-stream encoder, where
the additional coarse stream a) extends the neighborhood
hierarchically and b) facilitates the use of full cross-attention
between coarse points and voxels, leading to gains in binary
occupancy of up to +2.3 p.p. IoU and tracking quality of up
to +1.0 p.p. AQ.
Decoder Transformer: The decoder transformer (Sec-
tions III-D and III-E) plays a major role in both instance
segmentation and tracking. Results of our ablation studies
are shown in Table V. While previous works [1], [8]
employ only a single decoder layer, we find that using
multiple layers yields a significant increase in performance
for semantic segmentation of tracked classes (mIoU-things)
as well as instance-track associations (AQ). Further, spatio-
temporal refinement of tracked queries enables considerable
improvements in thing-class semantic segmentation with up to
+2.4 p.p. mIoU-things, showing that temporal refinement can
effectively aggregate useful semantic occupancy and instance
information across time steps.
Mask Aggregation: Moving from pure semantic to panoptic
masks introduces a discrepancy: stuff-class semantic masks
remain global, whereas thing-class masks are now locally
constrained, comparatively small, and act independently for
each instance. Visual inspection indicates that instance masks
are likely less confident at boundary regions; hence, when
treating stuff- and thing-class masks jointly, stuff-class masks
tend to dominate. We can counteract this by computing
dominant masks separately for both types (cf. Section III-
G), leading to improved instance segmentation (AQ1) and
tracking (AQ) metrics, as shown in Table VI.

<!-- page 8 -->
G. Qualitative Evaluation
Qualitative results are presented in Fig. 3 (nuScenes) and
Fig. 4 (Waymo). We demonstrate improvements across five
categories: (1) instance separation, correctly separating nearby
instances, (2) instance association, concisely assigning an
object to a single instance, (3) missing detections, leading
to incorrect free-space predictions, (4) underconfident mask
predictions, leading to incomplete instance segmentation, and
(5) ID switches, where the same instance ID is wrongly
assigned to different objects in subsequent frames.
V. CONCLUSION
We proposed a novel Gaussian-driven architecture, out-
performing the state-of-the-art in 4D panoptic occupancy
tracking. Inspired by recent advancements in Gaussian
splatting and point-transformer methods, we designed a novel
3D occupancy feature encoder. Using Gaussians as a 3D
representation allows us to convert the classically dense voxel-
grid-based encoders of occupancy prediction tasks into a
sparse, point-wise format, suitable for standard transformer-
based architectures. This allows for more flexible, data-
driven information aggregation, effective 2D-to-3D lifting,
and improved scalability. By utilizing splatting to convert
the sparse point-wise representation back into a voxel grid,
our encoder can replace any classical voxel feature encoder.
Extensive evaluations on both Occ3D nuScenes and Waymo
datasets demonstrate the effectiveness of our approach. We
hope that this paves the way for further exploration into more
dynamic, effective, and saliency-driven 3D representations.
ACKNOWLEDGMENTS
This work was funded by the Bosch Research collaboration
on AI-driven automated driving. A.V. was funded by the
Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) under grant number 539134284, through EFRE
(FEIH 2698644), and the state of Baden-W¨urttemberg.
REFERENCES
[1] Z. Chen, K. Li, X. Yang, T. Jiang, Y. Li, and H. Zhao, “TrackOcc:
Camera-based 4d panoptic occupancy tracking,” in ICRA, 2025.
[2] M. Luz, R. Mohan, A. R. Sekkat, O. Sawade, E. Matthes, T. Brox,
and A. Valada, “Amodal optical flow,” in ICRA, 2024.
[3] M. B¨uchner and A. Valada, “3d multi-object tracking using graph
neural networks with cross-edge modality attention,” IEEE Robotics
and Automation Letters, vol. 7, no. 4, pp. 9707–9714, 2022.
[4] C. Lang, A. Braun, L. Schillingmann, and A. Valada, “Self-supervised
multi-object tracking for autonomous driving from consistency across
timescales,” IEEE Robotics and Automation Letters, vol. 8, no. 11, pp.
7711–7718, 2023.
[5] M. K¨appeler, ¨O. C¸ ic¸ek, D. Cattaneo, C. Gl¨aser, Y. Miron, and A. Valada,
“Bridging perspectives: Foundation model guided bev maps for 3d object
detection and tracking,” arXiv preprint arXiv:2510.10287, 2025.
[6] Y. Zhang, Z. Zhu, and D. Du, “OccFormer: Dual-path transformer for
vision-based 3D semantic occupancy prediction,” in ICCV, 2023.
[7] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “SurroundOcc:
Multi-camera 3D occupancy prediction for autonomous driving,” in
ICCV, 2023, pp. 21 729–21 740.
[8] Q. Ma, X. Tan, Y. Qu, L. Ma, Z. Zhang, and Y. Xie, “COTR: Compact
occupancy transformer for vision-based 3D occupancy prediction,” in
CVPR, 2024, pp. 19 936–19 945.
[9] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “GaussianFormer:
Scene as gaussians for vision-based 3D semantic occupancy prediction,”
in ECCV, 2025, pp. 376–393.
[10] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and
J. Lu, “GaussianFormer-2: Probabilistic gaussian superposition for
efficient 3D occupancy prediction,” in CVPR, 2025, pp. 27 477–27 486.
[11] W. Gan, F. Liu, H. Xu, N. Mo, and N. Yokoya, “GaussianOcc: Fully
self-supervised and efficient 3D occupancy estimation with gaussian
splatting,” in ICCV, 2025, pp. 28 980–28 990.
[12] T. Meinhardt, A. Kirillov, L. Leal-Taix´e, and C. Feichtenhofer,
“TrackFormer: Multi-object tracking with transformers,” in CVPR, 2022,
pp. 8844–8854.
[13] F. Zeng, B. Dong, Y. Zhang, T. Wang, X. Zhang, and Y. Wei, “MOTR:
End-to-end multiple-object tracking with transformer,” in ECCV, 2022.
[14] T. Zhang, X. Chen, Y. Wang, Y. Wang, and H. Zhao, “MUTR3D: A
multi-camera tracking framework via 3D-to-2D queries,” in CVPRW,
2022.
[15] Z. Pang, J. Li, P. Tokmakov, D. Chen, S. Zagoruyko, and Y.-X. Wang,
“Standing between past and future: Spatio-temporal modeling for multi-
camera 3D multi-object tracking,” in CVPR, 2023, pp. 17 928–17 938.
[16] W. K. Fong, R. Mohan, J. V. Hurtado, L. Zhou, H. Caesar, O. Beijbom,
and A. Valada, “Panoptic nuscenes: A large-scale benchmark for lidar
panoptic segmentation and tracking,” IEEE Robotics and Automation
Letters, vol. 7, no. 2, pp. 3795–3802, 2022.
[17] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui,
J. Guo, Y. Zhou, Y. Chai, B. Caine, et al., “Scalability in perception
for autonomous driving: Waymo open dataset,” in CVPR, 2020, pp.
2446–2454.
[18] R. Mohan, D. Cattaneo, F. Drews, and A. Valada, “Progressive multi-
modal fusion for robust 3d object detection,” in CoRL, 2024.
[19] R. Mohan, J. V. Hurtado, R. Mohan, and A. Valada, “Forecas-
tOcc: Vision-based semantic occupancy forecasting,” arXiv preprint
arXiv:2602.08006, 2026.
[20] X. Tian, T. Jiang, L. Yun, Y. Mao, H. Yang, Y. Wang, Y. Wang, and
H. Zhao, “Occ3d: A large-scale 3d occupancy prediction benchmark
for autonomous driving,” in NeurIPS, 2023.
[21] B. Cheng, A. Schwing, and A. Kirillov, “Per-pixel classification is not
all you need for semantic segmentation,” in NeurIPS, vol. 34, 2021,
pp. 17 864–17 875.
[22] P. Tang, Z. Wang, G. Wang, J. Zheng, X. Ren, B. Feng, and C. Ma,
“SparseOcc: Rethinking sparse latent representation for vision-based
semantic occupancy prediction,” in CVPR, 2024, pp. 15 035–15 044.
[23] A.-Q. Cao, A. Dai, and R. de Charette, “PaSCo: Urban 3D panoptic
scene completion with uncertainty awareness,” in CVPR, 2024, pp.
14 554–14 564.
[24] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “Tri-perspective
view for vision-based 3D semantic occupancy prediction,” in CVPR,
2023, pp. 9223–9232.
[25] J. Wang, Z. Liu, Q. Meng, L. Yan, K. Wang, J. Yang, W. Liu, Q. Hou,
and M.-M. Cheng, “OPUS: Occupancy prediction using a sparse set,”
in NeurIPS, 2024, pp. 119 861–119 885.
[26] Z. Liu, J. Hou, X. Wang, X. Ye, J. Wang, H. Zhao, and X. Bai, “LION:
Linear group RNN for 3D object detection in point clouds,” in NeurIPS,
vol. 37, 2024, pp. 13 601–13 626.
[27] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang,
T. He, and H. Zhao, “Point transformer v3: Simpler faster stronger,”
in CVPR, 2024, pp. 4840–4851.
[28] Y. Chen, J. Liu, X. Zhang, X. Qi, and J. Jia, “VoxelNeXt: Fully sparse
voxelnet for 3D object detection and tracking,” in CVPR, 2023, pp.
21 674–21 683.
[29] D. Kim, S. Woo, J.-Y. Lee, and I. S. Kweon, “Video Panoptic
Segmentation,” in CVPR, 2020, pp. 9859–9868.
[30] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar,
“Masked-attention mask transformer for universal image segmentation,”
in CVPR, 2022, pp. 1290–1299.
[31] N. V¨odisch, K. Petek, W. Burgard, and A. Valada, “CoDEPS: Online
continual learning for depth estimation and panoptic segmentation,”
RSS, 2023.
[32] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai,
“BEVFormer: Learning bird’s-eye-view representation from LiDAR-
camera via spatiotemporal transformers,” IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 47, no. 3, pp. 2020–2036, 2025.
[33] Y. Liu, T. Wang, X. Zhang, and J. Sun, “PETR: Position embedding
transformation for multi-view 3D object detection,” in ECCV, 2022,
pp. 531–548.

<!-- page 9 -->
[34] M. Weber, J. Xie, M. D. Collins, Y. Zhu, P. Voigtlaender, H. Adam,
B. Green, A. Geiger, B. Leibe, D. Cremers, et al., “STEP: Segmenting
and tracking every pixel,” in NeurIPS, 2021.
[35] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss,
and J. Gall, “SemanticKITTI: A dataset for semantic scene understand-
ing of lidar sequences,” in ICCV, 2019, pp. 9297–9307.
[36] D.-A. Huang, Z. Yu, and A. Anandkumar, “MinVIS: A minimal video
instance segmentation framework without video-based training,” in
NeurIPS, vol. 35, 2022, pp. 31 265–31 277.
[37] K. Ying, Q. Zhong, W. Mao, Z. Wang, H. Chen, L. Y. Wu, Y. Liu,
C. Fan, Y. Zhuge, and C. Shen, “CTVIS: Consistent training for online
video instance segmentation,” in CVPR, 2023, pp. 899–908.
[38] X. Weng, J. Wang, D. Held, and K. Kitani, “3D multi-object tracking:
A baseline and new evaluation metrics,” in IROS, 2020.
[39] M. Ayg¨un, A. Osep, M. Weber, M. Maximov, C. Stachniss, J. Behley,
and L. Leal-Taix´e, “4d panoptic lidar segmentation,” in CVPR, 2021,
pp. 5527–5537.
[40] Y. Lee, J.-w. Hwang, S. Lee, Y. Bae, and J. Park, “An energy and GPU-
computation efficient backbone network for real-time object detection,”
in CVPRW, 2019, pp. 752–760.
[41] J. Huang and G. Huang, “BEVDet4D: Exploit temporal cues in multi-
camera 3D object detection,” arXiv preprint, arXiv:2203.17054, 2022.
[42] Z. Xia, X. Pan, S. Song, L. E. Li, and G. Huang, “Vision transformer
with deformable attention,” in CVPR, 2022, pp. 4794–4803.
