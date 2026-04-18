<!-- page 1 -->
1
CAD-SLAM: Consistency-Aware Dynamic SLAM
with Dynamic-Static Decoupled Mapping
Wenhua Wu, Chenpeng Su, Siting Zhu, Tianchen Deng, Jianhao Jiao, Guangming Wang,
Dimitrios Kanoulas, Zhe Liu, Hesheng Wang* Senior Member, IEEE,
Abstract—Recent advances in neural radiation fields (NeRF)
and 3D Gaussian-based SLAM have achieved impressive lo-
calization accuracy and high-quality dense mapping in static
scenes. However, these methods remain challenged in dynamic
environments, where moving objects violate the static-world
assumption and introduce inconsistent observations that degrade
both camera tracking and map reconstruction. This motivates
two fundamental problems: robustly identifying dynamic objects
and modeling them online. To address these limitations, we
propose CAD-SLAM, a Consistency-Aware Dynamic SLAM
framework with dynamic-static decoupled mapping. Our key
insight is that dynamic objects inherently violate cross-view
and cross-time scene consistency. We detect object motion by
analyzing geometric and texture discrepancies between historical
map renderings and real-world observations. Once a moving
object is identified, we perform bidirectional dynamic object
tracking (both backward and forward in time) to achieve
complete sequence-wise dynamic recognition. Our consistency-
aware dynamic detection model achieves category-agnostic, in-
stantaneous dynamic identification, which effectively mitigates
motion-induced interference during localization and mapping.
In addition, we introduce a dynamic–static decoupled mapping
strategy that employs a temporal Gaussian model for online in-
cremental dynamic modeling. Experiments conducted on multiple
dynamic datasets demonstrate the flexible and accurate dynamic
segmentation capabilities of our method, along with the state-of-
the-art performance in both localization and mapping.
Index Terms—Dense visual SLAM, 3D Gaussian Splatting,
Consistency-Aware, Adaptive dynamic perception and modeling.
I. INTRODUCTION
A. Motivations
D
ENSE visual Simultaneous Localization and Mapping
(SLAM) is the foundation of perception, navigation,
and planning that finds wide applications in areas such as
autonomous driving, mobile robotics, and virtual reality [1],
[2]. SLAM consists of two primary components: estimating
the position of the sensing system within an unknown envi-
ronment, and constructing a map of that environment.
Although existing dense visual SLAM methods based on
Neural Radiance Field (NeRF) and 3D Gaussians have shown
Wenhua Wu, Chenpeng Su, Siting Zhu, Tianchen Deng, and Zhe Liu are
with the Department of Automation, Shanghai Jiao Tong University, China.
Jianhao Jiao and Dimitrios Kanoulas are with the Department of Computer
Science, University College London, UK. Guangming Wang is with the
Department of Engineering, University of Cambridge, UK. Hesheng Wang
is with the Department of Automation, Key Laboratory of System Control
and Information Processing of Ministry of Education, State Key Laboratory
of Avionics Integration and Aviation System-of-Systems Synthesis, Shanghai
Key Laboratory of Navigation and Location Based Services, Shanghai Jiao
Tong University, Shanghai, China.
* Corresponding Author
promising results in static scenes, their performance often
deteriorates in complex dynamic environments, especially
when there are highly dynamic objects in the observations [3].
This degradation arises primarily because such highly-dynamic
objects severely violate the static-world assumption underlying
both geometric and photometric optimization. Their rapid
and irregular motion produces prominent inconsistent inter-
frame correspondences, which directly corrupts camera pose
estimation and distorts the recovered 3D geometry [4].
B. Challenges
To support robots in maintaining stability within dynamic
environments, the core objective of dynamic SLAM is not only
to estimate camera poses and reconstruct static backgrounds
but also to support the reliable perception and modeling
of dynamic elements, allowing downstream tasks such as
safe navigation, obstacle avoidance, and human-robot inter-
action [4]. This objective presents two inherent challenges.
The first challenge is correctly and completely identifying
general dynamic objects. Real-world environments exhibit
open-world semantic characteristics. dynamic objects are not
limited to predefined categories (e.g. pedestrians, vehicles)
but also include arbitrary moving entities (e.g., manipulated
boxes, floating balloons). This renders semantic-prior-based
methods [5]–[8] ineffective for unseen categories and inca-
pable of accurately distinguishing motion states, for instance,
differentiating between a parked car and a moving car. Fur-
thermore, observation changes induced by camera movement
are easily confounded with the actual motion of objects. As a
result, methods utilizing optical flow [9] or multi-view depth
warping masks [10] fail to accurately segment truly dynamic
objects. Third, ambiguity of the motion boundary. fast-moving
objects produce blurred boundaries in consecutive frames, and
low-texture regions further exacerbate the difficulty of precise
delineation [11]. These limitations highlight the need for a
dynamic object identification mechanism that is independent
of brittle semantic priors while capable of capturing fine-
grained motion boundaries.
The second challenge is effectively tracking and modeling
dynamic objects. Dynamic objects are essential for understand-
ing the environment in downstream interactions. However,
dynamic modeling faces several obstacles. First, the coupling
between the motion of the camera and the object. The SLAM
system must simultaneously estimate the camera pose and
model object motion, where errors in either task propagate to
the other. Second, real-time performance constraints. online
arXiv:2505.19420v2  [cs.CV]  3 Feb 2026

<!-- page 2 -->
2
RGB-D Stream
Dynamic Tracking and Mapping
Ours Static
Ours Dynamic
SplaTAM
MonoGS
Fig. 1. CAD-SLAM. Given RGB-D stream, our method achieves precise camera pose tracking while constructing dynamic-static decoupled maps. Our method
can adaptively segment dynamic objects of any category without any semantic priors. The illustration presents effective tracking and mapping performance
under dynamic scenarios.
SLAM requires that dynamic modeling be completed within
the frame interval, which precludes heavy offline optimiza-
tion methods [12]–[14] that rely on pre-computed poses or
full-sequence data. Finally, moving objects may undergo de-
formation, which requires the modeling framework to adapt to
temporal changes. These fundamental difficulties highlight the
urgent need for an online, SLAM-native dynamic modeling
solution—one that can decouple camera and object motion
while efficiently capturing temporal dynamics.
C. Contributions
The aforementioned challenges still remain unsolved, di-
rectly motivating our CAD-SLAM, an adaptive dynamic dense
visual SLAM with Gaussian splatting. Our key insight is that
dynamic objects inherently introduce inconsistency between
historical mapping and actual observation. To be specific,
observations from different viewpoints and timestamps exhibit
consistency under static-world assumption. The movement of
objects breaks this consistency, leading to discrepancies in
both geometry and texture between the rendered historical map
and the real-time observation. We then leverage this clue to
prompt fine-grained and class-agnostic segmentation. Unlike
semantic-based strategies or iterative dynamic identification
pipelines, our dynamic identification mechanism enables both
high adaptiveness and instant responsiveness.
Upon detecting a dynamic object, we initiate a bidirectional
tracklet for concise and effective tracking, endowing the
SLAM system with precise and timely dynamic cues. In cam-
era tracking, dynamic object pixels are excluded to mitigate
motion-induced corruption, ensuring more accurate and robust
pose estimation. Beyond mere suppression of dynamic regions,
we propose a dynamic–static composite mapping strategy. For
dynamic objects, we construct a temporal Gaussian model that
supports online incremental modeling, capturing their motion
and appearance evolution over time. For the static background,
dynamic regions are removed, and previously occluded areas
are progressively completed as the object moves, enabling the
static map to grow cleanly and consistently over time.
Overall, the main contributions of this paper are as follows:
• We propose CAD-SLAM, a novel dynamic dense visual
SLAM that accurately performs tracking and mapping
in complex dynamic environments while simultaneously
tracking and modeling dynamic objects.
• We introduce a semantic-free, consistency-aware dynamic
identification method that enables category-agnostic, fine-
grained, and instantaneous detection of dynamic objects
without any priors or iterative learning.
• We develop a hybrid static–dynamic mapping frame-
work that incorporates a temporal Gaussian model for
online incremental modeling of dynamic objects while
maintaining a clean and progressively completed static
background.
• Extensive experiments conducted on multiple real-world
dynamic datasets demonstrate that our method achieves
state-of-the-art performance in both camera tracking
and dense mapping, while exhibiting strong adaptability
across diverse scenarios.
II. RELATED WORK
A. Traditional Visual SLAM
The foundation of dense visual SLAM lies in DTAM [15],
which pioneers real-time tracking and mapping via dense
scene representation. In the same period, the KinectFusion
method [16] makes notable strides by using ICP algo-
rithms [17] and volumetric TSDF to achieve accurate and
real-time reconstruction of dense surfaces for indoor scenes.
Several data structures, including Surfels [18], [19] and Oc-
trees [20], [21], have been proposed to improve scalability and
reduce memory. In contrast to these methods, which rely on
per-frame pose optimization, BAD-SLAM [22] is the first to
propose a full Bundle Adjustment (BA) to jointly optimize
the keyframes. Traditional visual SLAM has matured over
decades into a well-established framework, with representa-
tive systems such as ORB-SLAM [23]–[25], DSO [26], and
VINS-Mono [27]. These systems typically follow a pipeline
of feature extraction and matching, pose estimation, local

<!-- page 3 -->
3
mapping, and loop closure detection, relying on sparse feature
points (e.g. ORB [28], SIFT [29]) or direct pixel intensities
to optimize camera poses and sparse 3D structure. In recent
years, numerous deep learning-based SLAM methods [30]–
[34] have been introduced to improve the precision and
robustness of traditional SLAM methods. However, the point
or mesh maps generated by traditional SLAM do not meet
the demands of high-density reconstruction and photorealistic
rendering.
B. Neural Implicit SLAM
The advent of Neural Radiance Fields (NeRF) [35] has
revolutionized scene representation by modeling scenes as
implicit neural networks, enabling high-fidelity dense recon-
struction. This paradigm naturally overcomes the problem of
the sparse structure of traditional SLAM, spurring the devel-
opment of NeRF-based visual SLAM. iMAP [36] pioneered
the integration of NeRF into the SLAM framework, achieving
joint optimization of camera poses and an implicit map.
However, its single Multi-Layer Perceptron (MLP) design
often restricts reconstruction detail and is prone to catas-
trophic forgetting. This challenge inspired NICE-SLAM [37]
to introduce a hierarchical feature-grid scene representation,
improving scalability and efficiency. Co-SLAM [38] combines
the smoothness and fast convergence of coordinate encoding
with the local-detail representation advantages of sparse para-
metric encoding. ESLAM [39] improves SLAM performance
by implementing multi-scale axis-aligned feature planes and
using TSDF for faster and more refined mapping. Subsequent
studies [40]–[47] have further advanced scene representation
and camera tracking through a series of refinements. How-
ever, NeRF-based SLAM still faces a critical bottleneck: the
computational overhead of ray sampling and MLP inference
is substantial, making it difficult to meet the demands of real-
time applications.
C. 3D Gaussian Splatting SLAM
3D Gaussian Splatting (3DGS) [48] is a breakthrough scene
representation that has emerged in recent years, combining
the explicit nature of traditional point clouds with the high-
quality rendering capability of NeRF [49]. It represents a
scene as a set of 3D Gaussians, each parameterized by its
position, covariance, and radiance attributes, and achieves real-
time photorealistic rendering through a splatting operation.
This effectively addresses the efficiency bottleneck of NeRF.
Due to these advantages, 3DGS has been rapidly integrated
into the field of visual SLAM, leading to a series of 3DGS-
based visual SLAM methods [50]. GS-SLAM [51] was the
first to employ fast splatting techniques and introduced a
dynamic, adaptive 3D Gaussian expansion strategy. Photo-
SLAM [52] integrates both explicit geometric features and
implicit texture representations, utilizing geometry-based den-
sification and Gaussian-pyramid-based learning within a multi-
threaded framework. Gaussian-SLAM [53] organizes scenes
into independently optimized sub-maps, enhancing efficiency
and scalability. SplaTAM [54] and GSSLAM [55] repre-
sent scenes using 3DGS and fundamentally redefine dense
SLAM processes, further enhancing robustness by incorpo-
rating geometric verification, regularization techniques, and
covisibility-based keyframe selection. Although these methods
have achieved impressive performance in static scenes, they
struggle in dynamic environments primarily because dynamic
objects violate scene consistency.
D. SLAM in Dynamic Environments.
The interference of dynamic objects has long been one
of the core challenges in the development of SLAM tech-
nology. For traditional SLAM, on the one hand, object mo-
tion leads to erroneous feature matches, causing pose drift;
on the other hand, dynamic regions introduce outliers into
bundle adjustment, degrading optimization accuracy. Existing
improvements mostly rely on semantic segmentation or optical
flow to identify dynamic areas, which are then removed during
feature matching and bundle adjustment [56]–[62]. Similarly,
for neural implicit SLAM and 3DGS SLAM, object motion
induces photometric and geometric inconsistencies between
frames, resulting in localization drift and rendering artifacts.
To handle dynamics, several methods [6], [7], [63] employ
semantic segmentation or object detection. However, they rely
on predefined dynamic category priors, leading to two inherent
limitations: (1) inability to process unknown object categories,
and (2) misclassification of stationary objects from predefined
dynamic categories as moving objects. DynaMoN [5] and
RoDyn-SLAM [9] incorporate optical flow estimation, but the
inherent ambiguity between object motion and camera mo-
tion measurements remains challenging to disambiguate. DG-
SLAM [10] designs a multi-view depth warp mask to compen-
sate for missing objects. However, the occlusion caused by the
view change is contained in the mask. Gassidy [64] performs
instance segmentation of the scene and relies on object-by-
object iterative analysis to distinguish dynamics. The latest
WildGS-SLAM [11] introduces an uncertainty-aware approach
that eliminates dependency on prior. However, uncertainties
in dynamic object boundaries remain prone to ambiguity,
and the incrementally trained MLP performs poorly at the
beginning stage and short sequence case. In contrast, our
method adaptively segments arbitrary dynamic objects through
scene consistency analysis, requiring no prior knowledge
while achieving precise boundary delineation. Furthermore,
while existing methods simply filter out dynamic elements to
construct static maps only, ours simultaneously builds both
dynamic and static maps.
III. METHOD
Our proposed adaptive dynamic dense SLAM framework
is illustrated in Fig. 2. The input consists of an RGB-
D image sequence {(It, Dt)|, t = 0, 1, . . . , n} and camera
intrinsics K, while the output includes estimated camera poses
and a dynamic–static decoupled map. The system comprises
three tightly coupled modules: a tracking module for pose
estimation, a CAD module for dynamic object identifica-
tion, and a mapping module for dynamic-static decoupled
dense reconstruction. The pipeline begins with static map
initialization using the first frame (Sect. III-A). As SLAM

<!-- page 4 -->
4
RGB
D epth
H istorical M ap 
Render
Forward
Com parison
D iscrepancy
Prom pt
...
...
CAD  Model
Tracking Model
Segm entation
Object Tracking
D ata
M apping Model
Separation
D ynam ic-Static D ecoupled M apping
Update
Bidirectional
...
...
Pipeline
Prelim inary Pose
M asked
Com posite
Fram e t
T
Inconsistency D etected
 ! 
C 
! 
M
Fram e 0
Initialization
Fram e 2
T
C
M
...
Fram e t+1
T
C
M
Fram e t+2
T
C
M
Fram e 1
T
C
M
...
INCONSISTENCY ! 
Fram e-to-Model
Refined Pose
DBA
ConvGRU
T
C
M
 Growing +
Purning -
...
...
Fig. 2. Overview of our complete SLAM system. CAD-SLAM consists of three tightly coupled models and takes RGB-D as input to produce camera poses
and a dense Gaussian map. For each incoming frame, the tracking model first predicts an initial pose using frame-to-model tracking, then refines it with a
pretrained ConvGRU after DBA. Using this refined pose, the Consistency-Aware Dynamic (CAD) model renders the historical map forward and compares
it with the current observation to compute an inconsistency map that reveals dynamic regions. Guided by this prompt, CAD model adaptively segments
dynamic masks and performs bidirectional temporal tracking of dynamic objects. The dynamic masks allow the mapping model to separate static and dynamic
components and maintain dynamic–static decoupled mapping. Throughout SLAM, whenever CAD model detects strong inconsistency and produces or updates
the mask, both tracking and mapping are subsequently guided by it.
progresses, dynamic objects manifest motion and are adap-
tively segmented via scene-consistency analysis, triggering
bidirectional dynamic object tracking (Sect. III-B). Based on
accurate dynamic identification, dynamic–static separation is
applied to the initial static map. Camera poses are subse-
quently optimized using the masked tracking loss (Sect. III-C),
followed by dynamic–static decoupled mapping to update both
the static background and dynamic object models (Sect. III-D).
A. 3D Gaussian Splatting
We adopt 3D Gaussian Splatting [48] as the fundamental
scene representation for our dynamic SLAM system. The
scene is represented as a set of 3D Gaussians G = {Gi}N
i=1,
where each Gaussian Gi is parameterized by its mean position
µi ∈R3, covariance matrix Σi ∈R3×3, opacity oi ∈[0, 1],
and spherical harmonic (SH) coefficients hi ∈Rk for view-
dependent color modeling. The spatial influence of the i-th
Gaussian is defined by the probability density function:
gi(x) = exp

−1
2(x −µi)⊤Σ−1
i (x −µi)

.
(1)
Following standard practice, the covariance matrix Σi is
decomposed into a rotation matrix Ri and a scaling matrix
Si to ensure semi-positive definiteness during optimization,
such that Σi = RiSiS⊤
i R⊤
i . Following prior Gaussian-based
SLAM approaches, we initialize the static Gaussian map from
the first RGB-D frame. Given the depth image and camera
intrinsics, the initial point cloud is reconstructed and each
point is converted into a 3D Gaussian.
Differentiable Rendering. To render the scene from a camera
pose T, 3D Gaussians are projected onto the 2D image
plane. The 2D covariance matrix Σ′
i in the screen space
is approximated using the affine transformation of the view
transformation:
Σ′
i = JWΣiW⊤J⊤,
(2)
where J is the Jacobian of the affine approximation of the pro-
jective transformation, and W is the viewing transformation
matrix calculated from T.
The rendered pixel values are computed using front-to-back
alpha blending. Specifically, we sort the Gaussians based on
their depth in the camera frame. The final color ˆC, depth ˆD,
and opacity mask ˆO for a pixel are computed by accumulating
the contributions of N ordered Gaussians overlapping the

<!-- page 5 -->
5
Before Motion
After Motion
Motion in 
Physical 
World
GT
Rendering
D iscrepancy
Motionless 
H istorical 
M ap
D iscrepancy
Em erges
Occlude
Mobile 
SAM
Trace Back
Recursively
Stream ingly
Tracking
New Input
T hresholding
Select
Foreground
Shallower 
D epth
...
Prom pt 
Point
t-1
t-2
t-3
Separation
Consistent
Inconsist ent  !
Fig. 3. CAD Model. Motion disrupts scene consistency, causing photometric and geometric discrepancies between the historical map and current observations.
These inconsistencies are used to prompt MobileSAM for fine-grained mask extraction, facilitating bidirectional object tracking and dynamic–static separation.
pixel:
ˆI =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(3)
ˆD =
X
i∈N
diαi
i−1
Y
j=1
(1 −αj),
(4)
ˆO =
X
i∈N
αi
i−1
Y
j=1
(1 −αj),
(5)
where ci is the view-dependent color computed from SH
coefficients, di is the projected depth of the Gaussian center,
and αi is the effective opacity derived from oi and the 2D
Gaussian probability. This differentiable formulation allows
for the simultaneous optimization of geometry and appearance
via gradient descent.
B. Consistency-Aware Dynamic Detection Model
Most existing dynamic object recognition methods [6]–
[8], [10] rely heavily on pre-trained detection and semantic
segmentation models. However, these approaches are inher-
ently limited to a closed set of predefined dynamic categories,
making SLAM systems weak in open-world scenarios. While
recent frameworks attempt to mitigate this dependency by
leveraging loss analysis or uncertainty learning, they often
incur non-negligible computational overhead or require extra
online training. To address these limitations, we propose a
Consistency-Aware Dynamic (CAD) model. Crucially, our
approach operates without semantic priors and maintains high
computational efficiency, ensuring the low latency required for
real-time SLAM.
Scene Consistency Analysis. Assuming that the motion oc-
curs at t, the historical Gaussian map Ht−1 therefore rep-
resents a static reconstruction of the scene. Under the static
assumption, the historical map accurately models the physical
scene, which means that the renderings (ˆI, ˆD) of Ht−1 should
exhibit consistency with current observations (I,D). However,
the presence of dynamic objects violates this alignment: the
static map built from former t −1 frames fails to reflect
the motion taking place at t. When rendered from pose
estimated at t, significant discrepancies are produced in the
areas undergoing motion. This inconsistency serves as a robust
and generalizable cue for identifying dynamic regions within
the scene. To capture the misalignment comprehensively, we
simultaneously consider both photometric and geometric in-
consistencies which are inherently complementary:
Ierr = ∥I −ˆI∥2,
Derr = ∥D −ˆD∥1,
(6)
Mic = (Ierr > τI) ∪(Derr > τD),
(7)
where τI and τD represent the thresholds for color and geo-
metric inconsistencies, respectively. Mic represents the mask
of the inconsistent region.
Dynamic Regions Detection. Dynamic objects in the field of
view can be classified into two types: one is a dynamic object
moving from outside the field of view into it, and the other is
an object moving within the field of view. For the first type,
the inconsistent region in the observed image corresponds
to the dynamic object that has entered the field of view.
For the second type, the inconsistent region in the observed
image manifests as the area where the dynamic object has
newly moved or the previously occluded background has been
exposed. The new occlusion caused by the dynamic object in
the background will make the observed depth smaller than the
rendered depth. In contrast, the newly exposed background
region will have an observed depth larger than the rendered
depth. This allows for the distinction between dynamic objects
and the background.
M d
ic = Mic ∩((D −ˆD) < 0),
(8)
where M d
ic denotes the inconsistent region corresponding to
dynamic objects.
Dynamic Segmentation. With the dynamic regions detected,
we then utilize an off-the-shelf model to perform detailed
segmentation. Considering efficiency and latency, we choose
MobileSAM [65], a lightweight and fast segment model
guided by prompts. Due to the minimal object movement
between consecutive frames (with a time interval of 1/30s),
the resulting dynamic inconsistency region represents only a

<!-- page 6 -->
6
small portion of the object. Therefore, we use the center of
the inconsistent region as a prompt input for MobileSAM fθ
to obtain the complete dynamic object segmentation.
Md = fθ(I, o(M d
ic)),
(9)
where o(·) represents the center of the inscribed circle.
Dynamic Object Tracking. For all detected dynamic objects
D, a unique ID is assigned to each instance, and we maintain
an object state directory recording their historical states as
the tuple (Ci, Ai, Mi). Specifically, Mi = {M k
i }t
k=0 de-
notes the set of dynamic segmentation masks across time;
Ci = {ck
i }t
k=0 stores the sequence of mask centers; and
Ai = {ak
i }t
k=0 is the temporal visibility vector indicating the
presence of object i at each timestamp k ∈[0, t]. For clarity,
we denote the newly detected object at time t by dropping the
object index i in the following analysis.
Since dynamic objects are detected via inconsistency analy-
sis, the detection typically occurs after the object has initiated
motion. To achieve a complete capture of the object’s motion
history, we employ a bidirectional tracking mechanism. For
past frames(k < t), we recursively trace back: M k−1 =
fθ(Ik−1, ck) until the object exits the view frustum or the
initial frame(k = 0). For future frames (k > t), the tracking
process is integrated streamingly into the SLAM pipeline:
M t+1
d
= fθ(It+1, ct).
The bidirectional tracking mechanism not only reconstructs
a complete temporal sequence of dynamic masks, but also
provides reliable cross-frame associations for each object.
Such temporally coherent trajectories are crucial for masked
pose estimation and decoupled mapping.
In contrast to frame-wise semantic detection, which over-
looks the temporal coherence of dynamic objects and remains
constrained by closed-set category assumptions, our CAD
module provides a more principled and efficient alternative. By
leveraging violations of spatiotemporal scene consistency as a
unified motion cue, the proposed approach achieves category-
agnostic and temporally consistent tracking without resorting
to heavyweight multi-stage pipelines or additional semantic
priors. This lightweight yet effective formulation satisfies the
latency and efficiency demands of real-time SLAM while
preserving the continuity and completeness of dynamic object
behavior across time.
C. Camera Tracking
For each incoming frame, we first perform frame-to-model
camera tracking. The pose Tt is initialized using a constant-
velocity motion model and then refined by aligning the ren-
dered observations from the static Gaussian map with the
incoming RGB-D frame.
To ensure that pose estimation is not contaminated by
moving objects, we restrict optimization to reliable static
regions. Specifically, we define the tracking mask
Mtrack = (¬Md) ∩( ˆO > τtrack),
(10)
where ¬ denotes mask negation. Md = S
i∈D{Mi | ai = 1}
denotes the union of active dynamic masks identified by our
CAD module, and ˆO is the accumulated opacity of the ren-
dered static map. The threshold τtrack filters out low-opacity
areas that correspond to poorly reconstructed or sparsely
observed regions.
Within this stable region, camera tracking is supervised by
both photometric and geometric loss.
LI =
X
Mtrack · ∥ˆI −I∥1,
(11)
LD =
X
(Mtrack ∩Mv) · ∥ˆD −D∥1,
(12)
where Mv indicates the valid pixels of the input depth map,
accounting for missing values or sensor-induced holes.
The final camera pose is obtained by minimizing a weighted
combination of the two terms:
Tt = arg min
T
λtrackLI + (1 −λtrack)LD,
(13)
where λtrack balances the contribution of photometric and
geometric cues. This formulation ensures that pose refinement
is guided exclusively by reliable static observations, yielding
robust tracking even in highly dynamic environments.
To mitigate pose drift in extended sequences, we incorporate
loop detection and bundle adjustment (BA). Following Droid-
SLAM [34], we integrate a pre-trained optical flow model
with Dense Bundle Adjustment Layer (DBA) to optimize
keyframe camera poses and depth. In contrast to WildGS-
SLAM [11] which introduces uncertainty maps during BA
optimization, we take advantage of acquired dynamic masks to
eliminate interference from moving entities while preserving
the integrity of the static scene. Following DG-SLAM [10],
the cost function over the keyframe graph is defined as:
E(T, d) =
X
(i,j)∈E
∥p∗
ij −Πc(Tij ◦Π−1
c (pi, di))∥2
Σij·¬Md,
Σij = diag ωij,
(14)
where p∗
ij where Πc denotes the projection transformation
from 3D coordinates to the image plane. pi represents pixel
coordinates, di indicates inverse depth values. Tij corresponds
to the relative camera pose between frames i and j. p∗
ij
represents the propagated coordinates of pixel pi in frame
j through optical flow estimation. ∥· ∥2
Σij·¬Md denotes the
Mahalanobis distance weighted by confidence metric Σij,
while filtering the dynamic.
D. Dynamic-static Decoupled Mapping
Unlike existing dynamic SLAM [10], [11], [64] that perform
elimination of dynamic objects, we propose a dynamic-static
decoupled mapping strategy that leverages temporal Gaussian
models to achieve online incremental dynamic modeling.
Dynamic-Static Separation. At the beginning of SLAM, the
scene is assumed to be entirely static, and the initial map
is constructed accordingly. Once the CAD module detects a
dynamic object, we immediately decouple it from the static
map. Following Eq. 9, we obtain a complete dynamic-object
mask from the image using the inconsistency-guided prompt.
With its 2D support region identified, the corresponding 3D

<!-- page 7 -->
7
points are recovered via back-projection using the depth map
and camera intrinsics, yielding a point cloud of the dynamic
object. We then extract all Gaussian primitives whose centers
fall within this dynamic region from the static map, thereby
achieving adaptive and precise dynamic–static separation.
Static Mapping. For each newly added keyframe, we insert
Gaussian ellipsoids into the static Gaussian map to fill regions
with holes or poor quality in the rendered output.
Minstert = (¬Md) ∩Mv ∩( ˆO < τmap),
(15)
where τmap represents the opacity threshold for the static
mapping region. The RGB-D pixels in Minsert will be ini-
tialized as Gaussian ellipsoids and incorporated into the static
Gaussian map. Subsequently, keyframes are selected from the
keyframe set to optimize the static map. We compute the
rendering loss for the valid static regions:
LI = (1 −λssim)
1
Mmap
X
Mmap · ∥ˆI −I∥1
+λssimSSIM(ˆI, I, Mmap),
(16)
LD =
1
Mmap
X
Mmap · ∥ˆD −D∥1,
(17)
Mmap = (¬Md) ∩Mv,
(18)
where λssim is the weight of ssim loss. The final static
mapping loss is:
Lstatic
map
= λcolorLI + λdepthLD + λregLreg,
(19)
where Lreg is the Gaussian ellipsoid scale regularization loss
from [66].
Dynamic Mapping. For each tracked dynamic object i ∈D,
we construct a temporal Gaussian model:
Gi(t) = {(µt
j, Σt
j, ot
j, ht
j)}.
(20)
Gi(t) is initialized with point cloud back-projected from
It, Dt using the mask M t
i . Then the dynamic object i at t is
rendered to yield the predicted color ˆIt
i and depth ˆDt
i:
ˆIt
i =
X
j∈Gi(t)
ct
jαt
j
j−1
Y
l=1
(1 −αt
l),
(21)
ˆDt
i =
X
j∈Gi(t)
dt
jαt
j
j−1
Y
l=1
(1 −αt
l).
(22)
The parameters of the dynamic Gaussians Gi(t) are op-
timized using loss defined over the valid mapping region
M i
map = M i ∩Mv. Here we omit t for clarity.
Li
I = (1 −λssim)
1
M imap
X
M i
map · ∥ˆId −I∥1
+λssimSSIM(ˆIi, I, M i
map),
(23)
Li
D =
1
M imap
X
M i
map · ∥ˆDi −D∥1.
(24)
The final dynamic mapping loss of object i is:
Li
map = λcolorLi
I + λdepthLi
D + λregLreg.
(25)
IV. EXPERIMENT
A. Experimental Setting
1) Dataset: We evaluate CAD-SLAM on four real-world
dynamic datasets: the TUM RGB-D [74] dataset, the Bonn
dataset [67], the Wild-SLAM Dataset [11], and the DAVIS
dataset [75].
• TUM RGB-D dataset [74] offers a comprehensive col-
lection of indoor RGB-D sequences recorded using a
Microsoft Kinect sensor at 30 Hz with a resolution of
640×480. Ground-truth trajectories were captured using
a high-precision motion capture system operating at 100
Hz. We select 4 dynamic sequences and 2 static sequences
for evaluation, covering diverse human motion patterns.
• Bonn dataset [67] is a dataset for RGB-D SLAM,
containing highly dynamic sequences capturing human
activities such as object manipulation and interaction with
balloons. Each sequence includes ground-truth camera
poses obtained via an OptiTrack Prime 13 motion capture
system. We select 6 representative dynamic sequences
for evaluation, including interactions between human and
various objects.
• Wild-SLAM Dataset [11] is primarily designed for
benchmarking dynamic SLAM performance in uncon-
strained real-world environments. It incorporates multiple
moving objects as dynamic interference sources to sim-
ulate common challenges such as motion-induced distur-
bances and occlusions encountered in practical scenarios.
The data were captured using an Intel RealSense D455
camera. We evaluate on 10 core sequences.
• DAVIS Dataset [75] comprises high-resolution video
sequences capturing a wide array of dynamic scenes and
moving objects. It includes scenarios ranging from hu-
man activities and animal movements to complex object
interactions and natural phenomena, providing a wide
variety of motion patterns and object appearances. DAVIS
doesn’t contain depth information. We leverage Depth
Anything V2 [76] to estimate depth, ensuring compati-
bility with our pipeline, and select 6 dynamic sequences
for evaluation.
2) Evaluation Metrics: For camera tracking performance,
we use the Root Mean Square Error (RMSE) and Standard
Deviation (S.D.) of the Absolute Trajectory Error (ATE).
For the quantitative evaluation of 3D reconstruction, we use
accuracy (cm), completeness (cm), and completion rate (%)
thresholds set to 5 cm. In addition, we employ image ren-
dering quality metrics, including Peak Signal-to-Noise Ratio
(PSNR), Structure Similarity Index Measure (SSIM), and
Learned Perceptual Image Patch Similarity (LPIPS). Since the
original images contain both static backgrounds and dynamic
foregrounds, the rendering metrics can effectively reflect the
performance of dynamic-static composition mapping.
3) Baselines: We conduct extensive and comprehensive
comparisons between CAD-SLAM and traditional SLAM
methods, NeRF-based SLAM methods, and 3D Gaussian-
based SLAM methods to highlight the superiority of our
method.

<!-- page 8 -->
8
Fig. 4.
Rendering visualization. SLAM methods assuming a static world fail to reconstruct dynamic scenes, highlighting the necessity of dynamic-object
identification. Although uncertainty-based method can detect motion to some extent, they still leave residual artifacts of dynamic objects (red circles in the left
column) and introduce blurring in static regions (right column). In contrast, CAD-SLAM realizes dynamic-static decoupled mapping, demonstrating superior
mapping performance in highly dynamic environments.
Fig. 5. Comparison between the uncertainty maps of WildGS-SLAM [11] and the dynamic masks produced by CAD-SLAM (Ours). Yellow regions denote
true dynamic objects, while red circles highlight uncertainty failure cases. As shown, uncertainty-based detection exhibits notable false negatives—either
missing entire dynamic objects or only partially segmenting them. It also suffers from false positives, where high uncertainty incorrectly appears in static
regions with sparse observations or temporary occlusions. In contrast, our dynamic masks produced by CAD-model are relatively robust and accurate.
4) Implementation Details: All experiments are conducted
on a server equipped with an Intel Platinum 8362 CPU and
an NVIDIA A100 GPU.
Tracking settings. We employ a weighted combination of
color and depth losses for tracking, with weights set to
λtrack = 0.6 and λssim = 0.2. Each frame is optimized for

<!-- page 9 -->
9
TABLE I
CAMERA TRACKING RESULTS ON BONN DATASET. ”*” DENOTES THE VERSION REPRODUCED BY NICE-SLAM. ”-” DENOTES THE ABSENCE OF
MENTION. ”X” DENOTES A FAILURE IN EXECUTION, WITH NO VALID RESULT. ”†” INDICATES THE REPLACEMENT OF DEPTH ESTIMATION WITH
GROUND-TRUTH DEPTH FOR FAIR COMPARISON. THE METRIC UNIT IS [CM]. BEST RESULTS ARE HIGHLIGHTED AS
FIRST ,
SECOND , AND
THIRD .
Methods
balloon
balloon2
ps_track
ps_track2
ball_track
mv_box2
Avg.
Traditional
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
ORB-SLAM3 [25]
5.8
2.8
17.7
8.6
70.7
32.6
77.9
43.8
3.1
1.6
3.5
1.5
29.8
15.2
Droid-VO [34]
5.4
-
4.6
-
21.34
-
46.0
-
8.9
-
5.9
-
15.4
-
DynaSLAM [56]
3.0
-
2.9
-
6.1
-
7.8
-
4.9
-
3.9
-
4.77
-
ReFusion [67]
17.5
-
25.4
-
28.9
-
46.3
-
30.2
-
17.9
-
27.7
-
NeRF based
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
iMAP* [36]
14.9
5.4
67.0
19.2
28.3
12.9
52.8
20.9
24.8
11.2
28.3
35.3
36.1
17.5
NICE-SLAM [37]
X
X
66.8
20.0
54.9
27.5
45.3
17.5
21.2
13.1
31.9
13.6
-
-
Vox-Fusion [40]
65.7
30.9
82.1
52.0
128.6
52.5
162.2
46.2
43.9
16.5
47.5
19.5
88.4
36.3
Co-SLAM [38]
28.8
9.6
20.6
8.1
61.0
22.2
59.1
24.0
38.3
17.4
70.0
25.5
46.3
17.8
ESLAM [39]
22.6
12.2
36.2
19.9
48.0
18.7
51.4
23.2
12.4
6.6
17.7
7.5
31.4
14.7
RoDyn-SLAM [9]
7.9
2.7
11.5
6.1
14.5
4.6
13.8
3.5
13.3
4.7
12.6
4.7
12.3
4.38
DynaMoN(MS&SS) [5]
2.8
-
2.7
-
14.8
-
2.2
-
3.4
-
2.7
-
4.77
-
3DGS based
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
SplaTAM [54]
40.0
14.6
39.5
15.8
217.9
81.2
131.0
33.1
20.2
16.3
17.1
9.3
77.6
28.4
MonoGS [55]
31.2
15.3
26.7
13.5
43.8
16.8
48.4
16.6
4.7
2.4
7.1
3.5
27.0
11.4
GS-ICP SLAM [68]
42.2
14.4
57.5
22.4
87.8
40.6
49.8
21.2
32.1
11.7
26.0
12.4
49.2
20.5
PG-SLAM [8]
6.4
2.2
7.3
3.4
5.0
1.9
8.5
2.8
-
-
7.0
2.0
-
-
DG-SLAM [10]
3.7
-
4.1
-
4.5
-
6.9
-
10.0
-
3.5
-
5.45
-
WildGS-SLAM [11]
2.9
1.2
2.5
1.2
3.6
1.9
3.1
1.4
3.1
1.6
2.4
1.3
2.93
1.43
WildGS-SLAM† [11]
2.4
1.0
2.4
1.2
3.4
1.9
3.1
1.4
3.4
1.9
2.6
1.3
2.88
1.45
CAD-SLAM(Ours)
2.7
1.1
2.3
0.8
2.4
1.1
3.7
1.4
3.4
1.2
2.1
0.8
2.77
1.05
Fig. 6. Comparison of the masks produced by our method and DG-SLAM.
Red circles highlight the noise artifacts in DG-SLAM’s mask, which caused
by inaccurate depth warping.
100 iterations. The learning rates for camera pose optimization
are set to 0.002 for rotation and 0.01 for translation. To
mitigate the influence of outliers, alpha and depth filtering
are applied to generate masks during the computation of the
tracking loss.
Mapping settings. For mapping, the order of spherical har-
monics L is 0. The position learning rate decays from an
initial value of 0.001 to a final value of 1.6 × 10−6. The
learning rates for color, opacity, scaling, and rotation are set
to 0.0025, 0.05, 0.005, and 0.001, respectively. Similar to
tracking, each frame undergoes 100 iterations. The opacity
threshold is set to 0.8 during densification and 0.3 during
pruning. The weights for the composite losses are configured
as λssim = 0.2, λcolor = 1.0, λdepth = 1.0, and λreg = 1.0.
Adaptive dynamic detection and tracking settings. The
thresholds for color and geometric inconsistencies τI = 20 ·
median(Ierr) and τD = 20 · median(Derr). The opacity
thresholds for tracking τtrack and mapping τmap are set to
0.7 and 0.8, respectively. Adaptive dynamic object detection
is performed every 5 frames for the Bonn, Wild-SLAM, and
DAVIS datasets, and every 10 frames for the TUM RGB-D
dataset. For 2D dynamic tracking, tracking is terminated if the
center of a dynamic object approaches within 4% of the image
boundary. Additionally, tracking is considered erroneous and
is terminated if the dynamic object’s mask area increases by
more than 1.5× or its center moves over 20% of the field of
view within a single frame.
B. Experimental Results
1) Camera Tracking Evaluation: Tab. I, Tab. II, and Tab. III
present the camera tracking results of different methods on the
Bonn, TUM RGB-D, and Wild-SLAM datasets, respectively.
Results for other methods are taken directly from their original
publications or obtained by running their official open-source
code. Our method achieves the best tracking accuracy in most
scenarios, with overall performance surpassing the current
state-of-the-art methods. This is attributed to the precise mask-
ing of dynamic regions, which effectively avoids pose drift
caused by motion interference. Compared to the semantic-
prior-based methods DG-SLAM and DynaMoN, our method
reduces tracking errors by 49.2% and 41.9%, respectively,
demonstrating its advantage in adaptively identifying dynamic
objects without relying on pre-defined dynamic categories.
2) 3D Reconstruction Evaluation: Most existing methods
focus primarily on static scene reconstruction and neglect
dynamic objects, whereas our method achieves composite
reconstruction of both dynamic and static components. For a
fair comparison, we evaluate only the static map. We employ
Accuracy, Completeness, and Completion Ratio to assess
geometric reconstruction quality, as shown in Tab. IV. In the
dynamic sequences of the Bonn dataset, CAD-SLAM achieves
an average reconstruction accuracy of 6.30 cm, which is 21.8%
higher than that of DG-SLAM (8.06 cm), and a completeness

<!-- page 10 -->
10
TABLE II
CAMERA TRACKING RESULTS ON TUM RGB-D DATASET.”*” DENOTES THE VERSION REPRODUCED BY NICE-SLAM. ”-” DENOTES THE ABSENCE OF
MENTION. ”X” DENOTES A FAILURE IN EXECUTION, WITH NO VALID RESULT. ”†” INDICATES THE REPLACEMENT OF DEPTH ESTIMATION WITH
GROUND-TRUTH DEPTH FOR FAIR COMPARISON. THE METRIC UNIT IS [CM]. BEST RESULTS ARE HIGHLIGHTED AS
FIRST ,
SECOND , AND
THIRD .
Methods
Dynamic
Static
Avg.
fr3/wk_xyz
fr3/wk_hf
fr3/wk_st
fr3/st_hf
fr1/xyz
fr1/rpy
Traditional
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
ORB-SLAM3 [25]
28.1
12.2
30.5
9.0
2.0
1.1
2.6
1.6
1.1
0.6
2.2
1.3
11.1
4.3
DVO-SLAM [69]
59.7
-
52.9
-
21.2
-
6.2
-
1.1
-
2.0
-
22.9
-
DynaSLAM [56]
1.7
-
2.6
-
0.7
-
2.8
-
-
-
-
-
-
-
ReFusion [67]
9.9
-
10.4
-
1.7
-
11.0
-
-
-
-
-
-
-
NeRF based
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
iMAP* [36]
111.5
43.9
X
X
137.3
21.7
93.0
35.3
7.9
7.3
16.0
13.8
-
-
NICE-SLAM [37]
113.8
42.9
X
X
88.2
27.8
45.0
14.4
4.6
3.8
3.4
2.5
-
-
Vox-Fusion [40]
146.6
32.1
X
X
109.9
25.5
89.1
28.5
1.8
0.9
4.3
3.0
-
-
Co-SLAM [38]
51.8
25.3
105.1
42.0
49.5
10.8
4.7
2.2
2.3
1.2
3.9
2.8
36.3
14.1
ESLAM [39]
45.7
28.5
60.8
27.9
93.6
20.7
3.6
1.6
1.1
0.6
2.2
1.2
34.5
13.5
RoDyn-SLAM [9]
8.3
5.5
5.6
2.8
1.7
0.9
4.4
2.2
1.5
0.8
2.8
1.5
4.05
2.28
DynaMoN(MS&SS) [5]
1.4
-
1.9
-
0.7
-
2.3
-
-
-
-
-
-
-
-
3DGS based
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
RMSE
S.D.
SplaTAM [54]
160.5
42.4
X
X
42.6
13.0
14.5
6.2
1.1
0.6
3.2
1.5
44.4
12.8
MonoGS [55]
28.4
12.3
47.8
16.5
15.3
8.4
13.9
3.2
1.0
0.4
2.6
1.3
18.2
7.0
GS-ICP [68]
68.9
50.8
84.6
34.3
87.5
16.9
11.2
2.7
1.4
0.7
4.2
3.9
43.0
18.2
PG-SLAM [8]
6.8
2.9
11.7
4.4
1.4
0.6
4.0
1.5
-
-
-
-
-
DG-SLAM [10]
1.6
-
0.6
-
-
-
-
-
-
-
-
-
-
-
WildGS-SLAM [11]
1.2
0.6
1.5
0.8
0.5
0.2
1.8
1.1
0.9
0.5
2.3
1.3
1.37
0.75
WildGS-SLAM† [11]
1.2
0.6
1.4
0.7
0.5
0.2
1.7
1.0
0.9
0.5
2.2
1.0
1.32
0.67
CAD-SLAM(Ours)
1.4
0.9
1.6
0.8
0.5
0.2
1.3
0.6
1.0
0.5
1.7
0.9
1.25
0.65
TABLE III
CAMERA TRACKING RESULTS ON WILD-SLAM MOCAP DATASET (ATE RMSE ↓[CM]). ”X” DENOTES A FAILURE IN EXECUTION, WITH NO VALID
RESULT. BEST RESULTS ARE HIGHLIGHTED AS
FIRST ,
SECOND , AND
THIRD .
Method
ANymal1
ANymal2
Ball
Crowd
Person
Racket
Stones
Table1
Table2
Umbrella
Avg
ReFusion [67]
4.2
5.6
5.0
91.9
5.0
10.4
39.4
99.1
101.0
10.7
37.23
DynaSLAM (N+G) [56]
1.6
0.5
0.5
1.7
0.5
0.8
2.1
1.2
34.8
34.7
7.84
NICE-SLAM [37]
X
123.6
21.1
X
150.2
X
134.4
138.4
X
23.8
-
DSO [70]
12.0
2.5
1.0
88.6
9.3
3.1
41.5
50.6
85.3
26.0
32.99
DROID-SLAM [34]
0.6
4.7
1.2
2.3
0.6
1.5
3.4
48.0
95.6
3.8
16.17
DynaSLAM (RGB) [56]
0.6
0.5
0.5
0.5
0.4
0.6
1.7
1.8
42.1
1.2
5.19
MonoGS [55]
8.8
51.6
7.4
70.3
55.6
67.6
39.9
24.9
118.4
35.3
47.99
SplatGS [71]
0.4
0.4
0.3
0.7
0.8
0.6
1.9
2.5
73.6
5.9
8.71
MonST3R-SW [72]
3.5
21.6
6.1
14.4
7.2
13.2
11.2
4.8
33.7
5.5
12.12
MegaSAM [73]
0.6
2.7
0.6
1.0
3.2
1.6
3.2
1.0
9.4
0.6
2.40
WildGS-SLAM [11]
0.2
0.3
0.2
0.3
0.8
0.4
0.3
0.6
1.3
0.2
0.46
CAD-SLAM(Ours)
0.4
0.3
0.2
1.5
0.8
0.5
0.7
0.6
1.5
0.4
0.68
of 7.64 cm, representing a 50.6% improvement over DG-
SLAM (15.46 cm). These gains stem from our method’s more
accurate adaptive dynamic identification and the dynamic-
static separation mapping strategy, which prevents interference
from missed dynamic objects and avoids missing static map
areas due to false detections.
3) Dynamic Separation Evaluation: Fig. 8 visualizes the
adaptive dynamic segmentation and dynamic-static separation
mapping process during the execution of CAD-SLAM. The
first frame is initialized as a static Gaussian map. It should
be noted that our method does not enforce the first frame
scene to be completely static. Subsequently, adaptive dynamic
segmentation is performed based on inconsistency detection.
When a dynamic object is detected, it is separated from
the static map, and a sequential dynamic Gaussian model is
constructed for dynamic mapping. As the object moves, holes
in the static map are progressively filled in.
Fig. 6 compares CAD-SLAM’s masks with those of DG-
SLAM [10]. DG-SLAM [10] employs semantic segmentation
priors combined with a multi-view depth-warping mask to
compensate for missing objects. Even objects that are static
according to the prior are incorrectly segmented as dynamic
regions. Although the multi-view depth-warping mask can
compensate for missing object information, occlusions caused
by viewpoint changes are also included in this mask and
cannot be distinguished from truly dynamic areas, as indicated
by the red circles in Fig. 6.
Fig. 5 compares our dynamic masks with the uncertainty
of WildGS-SLAM [11] on the Bonn dataset. As highlighted
by the red circles, WildGS-SLAM exhibits significant under-
detection and false detection. Blurred and incomplete uncer-
tainty boundaries can lead to residual artifacts from dynamic

<!-- page 11 -->
11
Time
Time
Ground 
Truth
WildGS-SLAM
Uncertainty
WildGS-SLAM
Rendered RGB
Ours Mask
Ours
Rendered RGB
Ground 
Truth
WildGS-SLAM
Uncertainty
WildGS-SLAM
Rendered RGB
Ours Mask
Ours
Rendered RGB
Ground 
Truth
WildGS-SLAM
Uncertainty
WildGS-SLAM
Rendered RGB
Ours Mask
Ours
Rendered RGB
a. Camel
b. Parkour
c. Tennis
d. Soccer ball
e. Car-turn
f. Crossing
Fig. 7. Visualization of dynamic mask and rendering results on the DAVIS Dataset. Our dynamic masks are more complete and precise than the uncertainty
masks of WildGS-SLAM [11] and remain unaffected by background interference. Due to the failure of uncertainty, WildGS-SLAM results in blurriness in
the dynamic area. In contrast, our method can accurately identify and model various types of dynamic objects, shown ours stronger adaptability.

<!-- page 12 -->
12
TABLE IV
RECONSTRUCTION RESULTS ON DYNAMIC SCENE SEQUENCES IN THE BONN DATASET. INSTANCES OF TRACKING FAILURES ARE DENOTED BY ”X”.
BEST RESULTS ARE HIGHLIGHTED AS
FIRST
AND
SECOND
Method
Metric
ball
ball2
ps trk
ps trk2
mv box2
Avg.
NICE-SLAM [37]
Acc. [cm] ↓
X
24.30
43.11
74.92
17.56
39.97
Comp. [cm] ↓
X
16.65
117.95
172.20
18.19
81.25
Comp. Ratio [≤5cm%] ↑
X
29.68
15.89
13.96
32.18
22.93
Co-SLAM [38]
Acc. [cm] ↓
10.61
14.49
26.46
26.00
12.73
18.06
Comp. [cm] ↓
10.65
40.23
124.86
118.35
10.22
60.86
Comp. Ratio [≤5cm%] ↑
34.10
3.21
2.05
2.90
39.10
16.27
ESLAM [39]
Acc. [cm] ↓
17.17
26.82
59.18
89.22
12.32
40.94
Comp. [cm] ↓
9.11
13.58
145.78
186.65
10.03
73.03
Comp. Ratio [≤5cm%] ↑
47.44
47.94
20.53
17.33
41.41
34.93
DG-SLAM [10]
Acc. [cm] ↓
7.00
5.80
9.14
11.78
6.56
8.06
Comp. [cm] ↓
9.80
8.05
17.99
20.10
7.61
15.46
Comp. Ratio [≤5cm%] ↑
49.46
52.41
34.62
32.81
49.02
43.67
CAD-SLAM(Ours)
Acc. [cm] ↓
6.01
7.21
6.12
6.28
5.86
6.30
Comp. [cm] ↓
7.84
6.40
8.42
8.78
6.77
7.64
Comp. Ratio [≤5cm%] ↑
45.82
57.34
40.72
42.58
47.75
46.84
Ground Truth
RGB & Depth
Rendered Static
RGB & Depth
&
&
Time 
Frame 0
Rendered Dynamic 
RGB & Depth
Fig. 8. Visualizations of the adaptive dynamic segmentation and dynamic-static separation mapping process during CAD-SLAM execution on Scene person
tracking of the Bonn dataset. The first frame is initialized as a static Gaussian map. Subsequently, adaptive dynamic segmentation is performed based on
inconsistency detection. When a dynamic object is detected, it is separated from the static map, and a sequential dynamic Gaussian model is constructed for
dynamic mapping. As the object moves, holes in the static map are progressively filled in.
objects persisting in the static map, while incorrectly over-
estimated uncertainty results in insufficient optimization of
the corresponding static map regions. In contrast, our method
accurately segments dynamic objects. Fig. 7 compares our
dynamic masks with the uncertainty of WildGS-SLAM [11]
across richer scenes on the DAVIS dataset. The results show
that our dynamic masks are more complete and precise than
the uncertainty masks of WildGS-SLAM [11] and remain
unaffected by background interference. The successful seg-
mentation of various objects across multiple environments
robustly demonstrates the adaptability of our method.
4) Rendering Quality Evaluation:
We employ PSNR,
SSIM, and LPIPS to quantitatively evaluate rendering quality,
with results presented in Tab. V. Our method achieves state-
of-the-art rendering performance, benefiting from the precise
modeling of foreground details through dynamic-static decou-
pled mapping and the artifact-free reconstruction of the static
background. Fig. 4 provides a visual comparison of renderings
from different methods. SplaTAM [54] and MonoGS [55]
exhibit severe blurring and geometric distortion. WildGS-
SLAM [11] can only render the static scene, and residual arti-
facts remain due to ambiguity at dynamic boundaries. In con-
trast, our CAD-SLAM accurately reconstructs texture details
of the static background while preserving the complete form
of dynamic objects (e.g., balloon contours, human posture),
achieving high-quality rendering that seamlessly integrates
static and dynamic elements. Fig. 7 compares our renderings
with those of the current state-of-the-art method, WildGS-

<!-- page 13 -->
13
TABLE V
QUANTITATIVE COMPARISON OF RENDERING PERFORMANCE ON BONN AND TUM DATASET. THE BEST RESULTS ARE DENOTED IN BOLD FONT.
Methods
Metrics
Bonn
TUM RGBD
balloon
person track
person track2
fr3/walk xyz
fr3/walk static
fr3/sit hf
Avg.
SplaTAM [54]
PSNR[dB] ↑
16.92
17.11
15.54
17.15
18.54
18.44
17.28
SSIM ↑
0.78
0.62
0.58
0.66
0.75
0.75
0.69
LPIPS ↓
0.22
0.33
0.37
0.35
0.26
0.25
0.30
MonoGS [55]
PSNR[dB] ↑
20.25
19.64
18.47
12.82
16.35
19.61
17.86
SSIM ↑
0.77
0.76
0.71
0.38
0.66
0.72
0.67
LPIPS ↓
0.34
0.37
0.40
0.55
0.27
0.29
0.37
CAD-SLAM(Ours)
PSNR[dB] ↑
22.91
23.53
22.95
23.03
27.51
25.58
24.25
SSIM ↑
0.91
0.91
0.91
0.89
0.97
0.93
0.92
LPIPS ↓
0.22
0.25
0.24
0.24
0.09
0.19
0.21
Fig. 9. Inconsistent regions Md
ic and dynamic masks Md under different multiples of the error median. From left to right: 10×, 15×, 20×, 25×, and 30× the
error median. The dynamic mask segmentation results remain unaffected, with true motion regions consistently identifiable. This demonstrates the effectiveness
and robustness of our method.
TABLE VI
COMPARISON OF DIFFERENT METHODS IN TERMS OF RUNNING TIME (MS).
Methods
Dynamic Seg.
Tracking
Mapping
Rodyn-SLAM [9]
278.66
875.70
1083.60
SplaTAM [54]
-
2630.36
548.06
WildGS-SLAM [11]
25.77
2467.28
2948.27
CAD-SLAM(Ours)
68.79
1025.39
1108.78
TABLE VII
COMPARISON OF TRACKING METRICS OF DIFFERENT WAYS TO MANAGE
DYNAMIC SEGMENTATION.
Methods
balloon
ball_track
mv_box2
RMSE
SD
RMSE
SD
RMSE
SD
a. w/o dynamic seg.
52.5
21.6
15.2
5.2
18.3
5.4
b. w/ MaskDINO [77]
5.5
2.1
11.4
4.6
12.5
3.9
c. w/o keyframe DBA
3.3
1.0
6.9
3.9
7.4
3.0
d. CAD-SLAM (Ours)
2.7
1.0
3.4
1.2
2.1
0.8
SLAM [11], across diverse scenarios. As shown, due to the
failure of its dynamic uncertainty prediction, WildGS-SLAM
cannot effectively distinguish dynamic objects, resulting in
blurry dynamic regions, such as the camel area in the ”camel”
sequence and the car area in the ”car-turn” sequence. In
comparison, our method can accurately identify and model
various categories of dynamic objects, demonstrating superior
adaptability.
5) Runtime Analysis.: Tab. VI presents the runtime anal-
ysis, including the time for dynamic segmentation, camera
ball_track
balloon
mv_box2
RGB Image
Ours
MaskDINO
Fig. 10. Comparison of the dynamic segmentation results of our method with
those obtained using a semantic segmentation network. Unlike prior-based
semantic segmentation, our method can adaptively detect atypical moving
objects, such as boxes and balloons, enabling more precise and flexible
dynamic segmentation.
tracking, and mapping. Our tracking and mapping times are
comparable to other methods, while additionally performing
dynamic mapping. For dynamic segmentation, our method is
more efficient than Rodyn-SLAM [9], which relies on pre-
trained semantic segmentation and optical flow networks.

<!-- page 14 -->
14
Fig. 11. Rendered error distribution analysis. The red dashed lines indicate
error medians, while the yellow shaded areas represent the range from 10× to
30× the error median. Selecting 20× the median as the operational threshold
effectively distinguishes genuine motion areas. Notably, the error threshold
maintains a wide acceptable range (10×-30×, even wider), ensuring robust
applicability in practice.
C. Ablation Study
We conduct an ablation study on our adaptive dynamic
segmentation, to demonstrate whether an explicit segmentation
is necessary. Fig. 10 presents a comparison between our
method’s dynamic segmentation results and those obtained by
a semantic segmentation network. We utilize MaskDINO [77],
with the predefined dynamic category set to “human”. Unlike
prior-based semantic segmentation, our method can adaptively
detect atypical moving objects, such as boxes and balloons,
enabling more precise and flexible dynamic segmentation.
Tab. VII shows the camera tracking results under different
settings: a. without dynamic segmentation, b. using the prior-
based semantic segmentation, and d. employing our adaptive
dynamic segmentation. With more accurate and flexible dy-
namic segmentation, camera tracking accuracy is significantly
improved. Additionally, we validated the effect of keyframe
DBA. As shown in row c. of Tab. VII, keyframe DBA can
significantly improve tracking accuracy.
Additionally, we conducted a parameter sensitivity analysis
on the adaptive dynamic segmentation module. For geometric
and color inconsistency thresholds, we statistically analyze the
rendered geometric and color error distributions, as shown in
Fig. 11. The red dashed lines indicate error medians, while
the yellow shaded areas represent the range from 10× to 30×
the error median. The observed long-tailed error distribution
reveals two distinct characteristics: minimal errors in static
regions (distribution head) and significantly larger errors in
dynamic regions (distribution tail). Selecting 20× the median
as the operational threshold effectively distinguishes genuine
motion areas. Notably, the error threshold maintains a wide
acceptable range (10×-30×, even wider), ensuring robust ap-
plicability in practice. Fig. 9 presents a sensitivity analysis of
the threshold selection, demonstrating that while the inconsis-
tent mask area slightly decreases from 10× to 30× median
thresholds. The dynamic mask segmentation results remain
unaffected, with true motion regions consistently identifiable.
This analysis demonstrates the effectiveness and robustness of
our threshold Settings.
V. CONCLUSION
We propose CAD-SLAM, a novel dynamic dense visual
SLAM system that addresses the core challenges of dy-
namic environment perception and modeling. By introducing
a consistency-aware dynamic detection mechanism, we elimi-
nate reliance on brittle semantic priors and achieve category-
agnostic, real-time dynamic object identification. Comple-
mented by a bidirectional tracking strategy and dynamic-static
decoupled mapping framework, our method not only maintains
state-of-the-art performance in camera localization and static
scene reconstruction but also enables accurate modeling of
dynamic objects. Extensive experiments across four real-world
datasets validate the robustness, adaptability, and efficiency
of CAD-SLAM, providing a reliable technical foundation
for robotic perception and interaction in complex dynamic
scenarios. Despite these advances, several promising directions
remain for further refinement. First, develop a unified end-to-
end network that integrates scene consistency analysis and dy-
namic segmentation, eliminating reliance on external models
like MobileSAM. Second, integrate motion prediction models
into the tracking pipeline. By predicting the future motion of
dynamic objects, the system can handle temporary occlusion
and provide more support for downstream tasks. Additionally,
the real-time performance could be further improved.
REFERENCES
[1] Y. Wang, Y. Tian, J. Chen, K. Xu, and X. Ding, “A survey of visual
slam in dynamic environment: the evolution from geometric to semantic
approaches,” IEEE Transactions on Instrumentation and Measurement,
2024.
[2] B. Al-Tawil, T. Hempel, A. Abdelrahman, and A. Al-Hamadi, “A
review of visual slam for robotics: evolution, properties, and future
applications,” Frontiers in Robotics and AI, vol. 11, p. 1347985, 2024.
[3] F. Tosi, Y. Zhang, Z. Gong, E. Sandstr¨om, S. Mattoccia, M. R. Oswald,
and M. Poggi, “How nerfs and 3d gaussian splatting are reshaping slam:
a survey,” arXiv preprint arXiv:2402.13255, vol. 4, p. 1, 2024.
[4] L. Carlone, A. Kim, T. Barfoot, D. Cremers, and F. Dellaert, “Slam
handbook: From localization and mapping to spatial intelligence,” 2025.
[5] N. Schischka, H. Schieber, M. A. Karaoglu, M. Gorgulu, F. Gr¨otzner,
A. Ladikos, N. Navab, D. Roth, and B. Busam, “Dynamon: Motion-
aware fast and robust camera localization for dynamic neural radiance
fields,” IEEE Robotics and Automation Letters, 2024.
[6] Z. Xu, J. Niu, Q. Li, T. Ren, and C. Chen, “Nid-slam: Neural implicit
representation-based rgb-d slam in dynamic environments,” in 2024
IEEE International Conference on Multimedia and Expo (ICME). IEEE,
2024, pp. 1–6.
[7] C. Ruan, Q. Zang, K. Zhang, and K. Huang, “Dn-slam: A visual slam
with orb features and nerf mapping in dynamic environments,” IEEE
Sensors Journal, 2023.
[8] H. Li, X. Meng, X. Zuo, Z. Liu, H. Wang, and D. Cremers, “Pg-
slam: Photo-realistic and geometry-aware rgb-d slam in dynamic en-
vironments,” IEEE Transactions on Robotics, 2025.
[9] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, “Rodyn-slam: Robust
dynamic dense rgb-d slam with neural radiance fields,” IEEE Robotics
and Automation Letters, 2024.
[10] Y. Xu, H. Jiang, Z. Xiao, J. Feng, and L. Zhang, “Dg-slam: Robust dy-
namic gaussian splatting slam with hybrid pose optimization,” Advances
in Neural Information Processing Systems, vol. 37, pp. 51 577–51 596,
2024.

<!-- page 15 -->
15
[11] J. Zheng, Z. Zhu, V. Bieri, M. Pollefeys, S. Peng, and I. Armeni,
“Wildgs-slam: Monocular gaussian splatting slam in dynamic environ-
ments,” in Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025, pp. 11 461–11 471.
[12] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d
gaussians: Tracking by persistent dynamic view synthesis,” in 2024
International Conference on 3D Vision (3DV).
IEEE, 2024, pp. 800–
809.
[13] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 310–20 320.
[14] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic
dynamic scene representation and rendering with 4d gaussian splatting,”
in The Twelfth International Conference on Learning Representations,
2024.
[15] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “Dtam: Dense
tracking and mapping in real-time,” in 2011 international conference on
computer vision.
IEEE, 2011, pp. 2320–2327.
[16] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim,
A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon,
“Kinectfusion: Real-time dense surface mapping and tracking,” in 2011
10th IEEE international symposium on mixed and augmented reality.
Ieee, 2011, pp. 127–136.
[17] P. J. Besl and N. D. McKay, “Method for registration of 3-d shapes,”
in Sensor fusion IV: control paradigms and data structures, vol. 1611.
Spie, 1992, pp. 586–606.
[18] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J.
Davison, “Elasticfusion: Dense slam without a pose graph.” in Robotics:
science and systems, vol. 11, no. 3.
Rome, 2015.
[19] T. Sch¨ops, T. Sattler, and M. Pollefeys, “Surfelmeshing: Online surfel-
based mesh reconstruction,” IEEE transactions on pattern analysis and
machine intelligence, vol. 42, no. 10, pp. 2494–2507, 2019.
[20] E. Vespa, N. Nikolov, M. Grimm, L. Nardi, P. H. Kelly, and S. Leuteneg-
ger, “Efficient octree-based volumetric slam supporting signed-distance
and occupancy mapping,” IEEE Robotics and Automation Letters, vol. 3,
no. 2, pp. 1144–1151, 2018.
[21] B. Xu, W. Li, D. Tzoumanikas, M. Bloesch, A. Davison, and
S. Leutenegger, “Mid-fusion: Octree-based object-level multi-instance
dynamic slam,” in 2019 International Conference on Robotics and
Automation (ICRA).
IEEE, 2019, pp. 5231–5237.
[22] T. Schops, T. Sattler, and M. Pollefeys, “Bad slam: Bundle adjusted
direct rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2019, pp. 134–144.
[23] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: A versatile
and accurate monocular slam system,” IEEE transactions on robotics,
vol. 31, no. 5, pp. 1147–1163, 2015.
[24] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras,” IEEE transactions
on robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
[25] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D.
Tard´os, “Orb-slam3: An accurate open-source library for visual, visual–
inertial, and multimap slam,” IEEE Transactions on Robotics, vol. 37,
no. 6, pp. 1874–1890, 2021.
[26] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
transactions on pattern analysis and machine intelligence, vol. 40, no. 3,
pp. 611–625, 2017.
[27] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monocular
visual-inertial state estimator,” IEEE transactions on robotics, vol. 34,
no. 4, pp. 1004–1020, 2018.
[28] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “Orb: An efficient
alternative to sift or surf,” in 2011 International conference on computer
vision.
Ieee, 2011, pp. 2564–2571.
[29] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,”
International journal of computer vision, vol. 60, no. 2, pp. 91–110,
2004.
[30] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison,
“Codeslam—learning a compact, optimisable representation for dense
visual slam,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2018, pp. 2560–2568.
[31] R. Li, S. Wang, and D. Gu, “Deepslam: A robust monocular slam system
with unsupervised deep learning,” IEEE Transactions on Industrial
Electronics, vol. 68, no. 4, pp. 3577–3587, 2020.
[32] L. Koestler, N. Yang, N. Zeller, and D. Cremers, “Tandem: Tracking and
dense mapping in real-time using deep multi-view stereo,” in Conference
on Robot Learning.
PMLR, 2022, pp. 34–45.
[33] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger, “Con-
volutional occupancy networks,” in Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020, Proceedings,
Part III 16.
Springer, 2020, pp. 523–540.
[34] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular,
stereo, and rgb-d cameras,” Advances in neural information processing
systems, vol. 34, pp. 16 558–16 569, 2021.
[35] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[36] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit mapping and
positioning in real-time,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 6229–6238.
[37] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 786–12 796.
[38] H. Wang, J. Wang, and L. Agapito, “Co-slam: Joint coordinate and
sparse parametric encodings for neural real-time slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 293–13 302.
[39] M. M. Johari, C. Carta, and F. Fleuret, “Eslam: Efficient dense slam
system based on hybrid representation of signed distance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 17 408–17 419.
[40] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, “Vox-
fusion: Dense tracking and mapping with voxel-based neural implicit
representation,” in 2022 IEEE International Symposium on Mixed and
Augmented Reality (ISMAR).
IEEE, 2022, pp. 499–507.
[41] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global
optimization for consistent 3d instant reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV),
October 2023, pp. 3727–3737.
[42] X. Liu, Y. Li, Y. Teng, H. Bao, G. Zhang, Y. Zhang, and Z. Cui, “Multi-
modal neural radiance field for monocular dense slam with a light-weight
tof sensor,” in Proceedings of the ieee/cvf international conference on
computer vision, 2023, pp. 1–11.
[43] M. Li, J. He, Y. Wang, and H. Wang, “End-to-end rgb-d slam with
multi-mlps dense neural implicit representations,” IEEE Robotics and
Automation Letters, 2023.
[44] Y. Ming, W. Ye, and A. Calway, “idf-slam: End-to-end rgb-d slam
with neural implicit mapping and deep feature tracking,” arXiv preprint
arXiv:2209.07919, 2022.
[45] H. Wang, Y. Cao, X. Wei, Y. Shou, L. Shen, Z. Xu, and K. Ren,
“Structerf-slam: Neural implicit representation slam for structural en-
vironments,” Computers & Graphics, vol. 119, p. 103893, 2024.
[46] W. Wu, G. Wang, T. Deng, S. Ægidiu, S. Shanks, V. Modugno,
D. Kanoulas, and H. Wang, “Dvn-slam: Dynamic visual neural slam
based on local-global encoding,” in 2025 IEEE International Conference
on Robotics and Automation (ICRA).
IEEE, 2025, pp. 14 564–14 571.
[47] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and
W. Chen, “Plgslam: Progressive neural scene represenation with local to
global bundle adjustment,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 19 657–19 666.
[48] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[49] T. Deng, Y. Pan, S. Yuan, D. Li, C. Wang, M. Li, L. Chen, L. Xie,
D. Wang, J. Wang, J. Civera, H. Wang, and W. Chen, “What is the
best 3d scene representation for robotics? from geometric to foundation
models,” arXiv preprint arXiv:2512.03422, 2025.
[50] T. Deng, W. Wu, J. He, Y. Pan, X. Jiang, S. Yuan, D. Wang, H. Wang,
and W. Chen, “Vpgs-slam: Voxel-based progressive 3d gaussian slam in
large-scale scenes,” arXiv preprint arXiv:2505.18992, 2025.
[51] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “Gs-
slam: Dense visual slam with 3d gaussian splatting,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 19 595–19 604.
[52] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-
time simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 21 584–21 593.
[53] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam:
Photo-realistic dense slam with gaussian splatting,” arXiv preprint
arXiv:2312.10070, 2023.

<!-- page 16 -->
16
[54] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 21 357–21 366.
[55] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian
Splatting SLAM,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024.
[56] B. Bescos, J. M. F´acil, J. Civera, and J. Neira, “Dynaslam: Tracking,
mapping, and inpainting in dynamic scenes,” IEEE Robotics and Au-
tomation Letters, vol. 3, no. 4, pp. 4076–4083, 2018.
[57] C. Yu, Z. Liu, X.-J. Liu, F. Xie, Y. Yang, Q. Wei, and Q. Fei, “Ds-
slam: A semantic visual slam towards dynamic environments,” in 2018
IEEE/RSJ international conference on intelligent robots and systems
(IROS).
IEEE, 2018, pp. 1168–1174.
[58] B. Bescos, C. Campos, J. D. Tard´os, and J. Neira, “Dynaslam ii: Tightly-
coupled multi-object tracking and slam,” IEEE robotics and automation
letters, vol. 6, no. 3, pp. 5191–5198, 2021.
[59] J. Zhang, M. Henein, R. Mahony, and V. Ila, “Vdo-slam: A visual
dynamic object-aware slam system,” arXiv preprint arXiv:2005.11052,
2020.
[60] L. Xiao, J. Wang, X. Qiu, Z. Rong, and X. Zou, “Dynamic-slam:
Semantic monocular visual localization and mapping based on deep
learning in dynamic environment,” Robotics and Autonomous Systems,
vol. 117, pp. 1–16, 2019.
[61] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, “Flowfusion:
Dynamic dense rgb-d slam based on optical flow,” in 2020 IEEE
international conference on robotics and automation (ICRA).
IEEE,
2020, pp. 7322–7328.
[62] T. Deng, G. Shen, C. Xun, S. Yuan, T. Jin, H. Shen, Y. Wang, J. Wang,
H. Wang, D. Wang et al., “Mne-slam: Multi-agent neural slam for mobile
robots,” in Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025, pp. 1485–1494.
[63] M. Li, Z. Guo, T. Deng, Y. Zhou, Y. Ren, and H. Wang, “Ddn-slam:
Real time dense dynamic neural implicit slam,” IEEE Robotics and
Automation Letters, 2025.
[64] L. Wen, S. Li, Y. Zhang, Y. Huang, J. Lin, F. Pan, Z. Bing, and A. Knoll,
“Gassidy: Gaussian splatting slam in dynamic environments,” in 2025
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2025, pp. 8471–8477.
[65] C. Zhang, D. Han, Y. Qiao, J. U. Kim, S.-H. Bae, S. Lee, and C. S.
Hong, “Faster segment anything: Towards lightweight sam for mobile
applications,” arXiv preprint arXiv:2306.14289, 2023.
[66] L. Zhu, Y. Li, E. Sandstr¨om, S. Huang, K. Schindler, and I. Armeni,
“Loopsplat: Loop closure by registering 3d gaussian splats,” in 2025
International Conference on 3D Vision (3DV).
IEEE, 2025, pp. 156–
167.
[67] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss, “Re-
fusion: 3d reconstruction in dynamic environments for rgb-d cameras
exploiting residuals,” in 2019 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS).
IEEE, 2019, pp. 7855–7862.
[68] S. Ha, J. Yeon, and H. Yu, “Rgbd gs-icp slam,” in European Conference
on Computer Vision.
Springer, 2025, pp. 180–197.
[69] C. Kerl, J. Sturm, and D. Cremers, “Dense visual slam for rgb-d
cameras,” in 2013 IEEE/RSJ International Conference on Intelligent
Robots and Systems.
IEEE, 2013, pp. 2100–2106.
[70] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
transactions on pattern analysis and machine intelligence, vol. 40, no. 3,
pp. 611–625, 2017.
[71] E. Sandstr¨om, G. Zhang, K. Tateno, M. Oechsle, M. Niemeyer, Y. Zhang,
M. Patel, L. Van Gool, M. Oswald, and F. Tombari, “Splat-slam:
Globally optimized rgb-only slam with 3d gaussians,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
1680–1691.
[72] J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole, D. Sun,
and M.-H. Yang, “Monst3r: A simple approach for estimating geometry
in the presence of motion,” in The Thirteenth International Conference
on Learning Representations, 2025.
[73] Z. Li, R. Tucker, F. Cole, Q. Wang, L. Jin, V. Ye, A. Kanazawa,
A. Holynski, and N. Snavely, “Megasam: Accurate, fast and robust
structure and motion from casual dynamic videos,” in Proceedings of
the Computer Vision and Pattern Recognition Conference, 2025, pp.
10 486–10 496.
[74] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A
benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ
international conference on intelligent robots and systems. IEEE, 2012,
pp. 573–580.
[75] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and
A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology
for video object segmentation,” in Proceedings of the IEEE conference
on computer vision and pattern recognition, 2016, pp. 724–732.
[76] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao,
“Depth anything v2,” Advances in Neural Information Processing Sys-
tems, vol. 37, pp. 21 875–21 911, 2024.
[77] F. Li, H. Zhang, H. Xu, S. Liu, L. Zhang, L. M. Ni, and H.-Y.
Shum, “Mask dino: Towards a unified transformer-based framework for
object detection and segmentation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2023, pp. 3041–
3050.
