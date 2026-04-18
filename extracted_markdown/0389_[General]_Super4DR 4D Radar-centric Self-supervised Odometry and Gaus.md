<!-- page 1 -->
UNDER REVIEW
1
Super4DR: 4D Radar-centric Self-supervised Odometry and
Gaussian-based Map Optimization
Zhiheng Li, Weihua Wang, Qiang Shen, Yichen Zhao, and Zheng Fang*
Abstract—Conventional SLAM systems using visual or LiDAR
data often struggle in poor lighting and severe weather. Although
4D radar is suited for such environments, its sparse and noisy
point clouds hinder accurate odometry estimation, while the
radar maps suffer from obscure and incomplete structures.
Thus, we propose Super4DR, a 4D radar-centric framework
for learning-based odometry estimation and gaussian-based map
optimization. First, we design a cluster-aware odometry network
that incorporates object-level cues from the clustered radar
points for inter-frame matching, alongside a hierarchical self-
supervision mechanism to overcome outliers through spatio-
temporal consistency, knowledge transfer, and feature contrast.
Second, we propose using 3D gaussians as an intermediate
representation, coupled with a radar-specific growth strategy,
selective separation, and multi-view regularization, to recover
blurry map areas and those undetected based on image texture.
Experiments show that Super4DR achieves a 67% performance
gain over prior self-supervised methods, nearly matches super-
vised odometry, and narrows the map quality disparity with
LiDAR while enabling multi-modal image rendering.
Index Terms—4D radar, Self-supervised odometry estimation,
Gaussian splatting, Simultaneous localization and mapping.
I. INTRODUCTION
S
IMULTANEOUS localization and mapping is essential for
autonomous robots. Over the past decade, LiDAR and vi-
sual SLAM have achieved significant progress, enabling real-
time operation with high precision in favorable environments.
However, LiDAR is susceptible to atmospheric scattering and
particulate matter in inclement weather, causing degraded data
integrity or a complete loss of functionality. Similarly, cameras
become ineffective in poor illumination or dense smoke. These
limitations pose fundamental challenges to reliable localization
and scene reconstruction in such adverse situations.
Fortunately, 4D radar has emerged as a promising solution
to these issues. Its longer wavelengths provide superior signal
penetration and all-weather robustness compared with LiDAR.
Additionally, unlike high-cost scanning radar limited to coarse
planar imaging and 3D single-chip automotive radar with low
resolution, 4D radar provides point cloud data that can describe
the approximate contour of objects. Consequently, recent stud-
ies have developed geometric and end-to-end frameworks for
4D radar odometry, aiming for reliable pose estimation in chal-
lenging weather. However, geometry-based algorithms [1]–[4]
that rely on dense and precise structural association encounter
limitations due to the characteristics of radar points (Fig. 1(a)).
1) Limited geometric primitives: The sparser radar data lacks
distinct geometric elements (e.g., lines and planes), which act
as essential constraints for traditional registration approaches.
2) Measurement uncertainty: More severe positional noise in
range and azimuth violates the rigid geometric assumptions
4D radar points
(Sparse / Noisy / Uneven)
16-beam LiDAR points 
(Dense / Clean / Uniform)
(a) Comparison of LiDAR and 4D radar frames
LiDAR map
4D radar map
(b) Comparison of LiDAR and 4D radar maps
Fig. 1. Comparison of LiDAR and 4D radar data. Unlike LiDAR, 4D radar
points usually suffer from greater sparsity, noise, and uneven density, thereby
posing challenges for robust odometry estimation and high-integrity mapping.
Here, the maps are constructed based on ground-truth poses provided by the
NTU4DRadLM [9] dataset.
of conventional methods. In comparison, learning-based meth-
ods [5]–[8] are capable of extracting information-rich features
from low-quality radar points. They further model inter-frame
relationships using adaptive matching within high-dimensional
feature space to alleviate the impact of noise, thereby pushing
the boundaries of 4D radar odometry.
However, supervised learning depends on expensive ground
truths that are difficult to obtain in harsh conditions. Although
self-supervised algorithms [7], [8] are label-free, their accuracy
lags notably behind supervised approaches due to over-reliance
on rigid geometric constraints that are easily disrupted by radar
noise and measurement errors. Meanwhile, they inherit point-
level matching originally designed for uniform LiDAR points,
which may be suboptimal for the blocky distribution of radar
points and fail to exploit latent object cues. Beyond odometry,
the maps built from 4D radar points are too loose and indistinct
(Fig. 1(b)). While utilizing diffusion networks [10], [11] is a
way for point cloud completion, they struggle to reconstruct
objects outside radar’s field of view (e.g., inferring a complete
tree from a single trunk without rich geometric priors) and face
a critical bottleneck in scene generalization. Furthermore, most
existing works focus on enhancing radar frames independently,
leaving the optimization of the entire map as a challenging and
open topic. Thus, in this article, we propose a 4D radar-centric
framework called Super4DR, as illustrated in Fig. 2, to address
the radar limitations in accurate pose estimation and complete
map reconstruction.
For radar odometry, building upon radar properties, we pro-
pose a cluster-aware network with multi-level self-supervision,
arXiv:2512.09608v1  [cs.RO]  10 Dec 2025

<!-- page 2 -->
UNDER REVIEW
2
Geometric 
Constraint
Poses
RGB Images
Inital Radar Map
Growth Strategy
(Sec. IV-B)
Selective Separation
(Sec. IV-C)
Regularization
(Sec. IV-D)
Gaussian-based
Map Optimization
Radar Points
Radar Cluster
Clear Depth 
Refined Radar Map
Vague Depth 
Sparse Point Clouds
Dense Point Clouds
Complete Rendered Image
Odometry
Network
RaFlow
CMFlow
Ground Truth Ours
Painting
Self-supervised 
Radar Odometry
Multi-view
Constraint
Incomplete Rendered Image
...
...
...
Teacher Guidance
(Sec. III-C.3)
Feature Space and
Temporal Constraints
(Sec. III-C.4 & III-C.5)
Geometric Consistency
(Sec. III-C.1 & III-C.2)
Odometry stream 
Map stream 
Trajectory
Vision Model
Fragmented Ground
Continuous Ground
Exceeding radar FOV
Fig. 2. Overview of Super4DR framework with end-to-end radar odometry and gaussian-based map optimization. First, a cluster-aware network, trained with
multi-level self-supervised losses, processes radar points along with cluster grouping to estimate poses. Afterwards, an initial map with incomplete structures is
constructed based on the trajectory and painted through image pixels. We further propose a map optimizer adopting gaussians as an intermediate representation,
with a radar-specific growth strategy, selective separation and multi-view regularization for complete map reconstruction and detailed image rendering.
moving beyond sole reliance on geometric constraints. Specif-
ically, we use clustering information derived from point distri-
bution to enable the network to learn inter-frame relationships
at the object level while suppressing the point-level noise. We
further introduce a cluster-weighted distance loss to supervise
geometric alignment, with particular emphasis on large, stable
point clusters. Next, we present a column occupancy loss that
represents points as grid occupation, allowing for tolerance of
noisy positions and ensuring large-scale consistency through
column-wise similarity. Beyond rigid spatial consistency rules,
a geometry-based approach serves as a teacher to produce
soft pose labels based on a secondary refinement and selection
mechanism, distilling meaningful knowledge into the network.
Additionally, we adopt a feature contrast loss to maximize the
feature difference of non-corresponding point pairs for reliable
matching and design a constant-acceleration motion model for
trajectory smoothing.
For map optimization, encouraged by 3D gaussian splatting
(3DGS [12]), we believe that converting the 4D radar map into
gaussian representation and optimizing it could enhance map
quality, grounded in several capabilities of 3DGS framework.
First, benefiting from splitting and cloning mechanisms, radar
gaussians can multiply adaptively based on gradients to occupy
structural voids, while their positions are adjusted to generate
dense geometry under image supervision. Second, because of
the image’s larger vertical FOV, gaussians are forced to expand
and recover undetected regions of radar to guarantee that their
rendered output aligns with image pixels correctly, thereby
improving map integrity. Third, with the help of cross-modal
and full-map optimization without neural network, it can avoid
generalization problems and ensure greater global consistency
than frame-wise point refinement. Once optimization finishes,
refined gaussians can be reverted into points according to their
centroids, producing a complete and structurally coherent map.
Nevertheless, directly using 3DGS [12] is “insufficient” as
it prioritizes rendering quality over geometric accuracy, while
gaussians initialized from radar maps with uneven point distri-
bution and noise cause challenges for subsequent optimization.
Thus, we propose a growth strategy that first creates synthetic
gaussians using depth priors derived from a visual fundamental
model to solve the ground fragmentation issue unique to radar
maps. Unlike the densification manner that only meets image
rendering requirements in most GS-based methods, we adopt a
geometric-aware strategy that splits and interpolates gaussians
based on local geometry to generate denser structures. Besides,
selective separation decouples sky gaussians by maintaining a
mask to decrease floaters and saves valuable large gaussians
from periodic pruning for subsequent completion. Rather than
optimizing gaussians based on a single view with ambiguous
spatial information one by one, we introduce a multi-view reg-
ularization that leverages overlapping regions to jointly refine
noisy radar gaussians into deterministic states. Furthermore,
geometric priors from visual foundation models are utilized as
extra supervision to improve the geometric accuracy of maps.
In summary, the contributions of this paper are as follows:
• To the best of our knowledge, Super4DR is the first inte-
grated framework that considers both radar odometry and
map optimization, empowering pose prediction from low-
quality radar points and reconstruction of clear structures
from initially indistinct maps.
• We propose a cluster-aware odometry network trained by
a multi-level self-supervision strategy, including cluster-
weighted distance, column occupancy, teacher guidance,
and the constraints on feature-space and temporal aspects,
to mitigate the impact of outliers on the learning process.
• We innovatively employ 3D gaussian representation as an
intermediary to enhance map quality, developing gaussian
growth strategy with depth-guided ground completion and
geometry-aware densification, complemented by selective
separation and multi-view overlap regularization, thereby
recovering blurry structures and radar-undetected areas.
• Experiments on public and self-collected datasets indicate

<!-- page 3 -->
UNDER REVIEW
3
that our radar odometry achieves state-of-the-art results
among self-supervised algorithms while remarkably clos-
ing the gap to recent supervised methods. Super4DR also
exceeds previous GS-based works in radar map and ren-
dered image quality, validating that gaussian splatting can
be a viable new approach for 4D radar map optimization.
• We additionally explore the potential of 4D radar-thermal
gaussian splatting in low-light conditions and demonstrate
the applicability of our algorithm across different visual
sensors, including both RGB and thermal cameras. Our
code and multi-sensor dataset based on a handheld plat-
form will be released in https://github.com/NEU-REAL/
Super4DR to promote the development of related fields.
II. RELATED WORK
A. LiDAR-based Odometry
The purpose of odometry is to calculate pose transformation
between adjacent frames. As a pioneering method, the iterative
closest point (ICP) [13] is widely adopted for point cloud reg-
istration by minimizing point-to-point distance, but it exhibits
sensitivity to noise. Its variants, like point-to-line and point-to-
plane ICP, exploit local geometry to enhance robustness, while
GICP [14] merges their advantages. KISS-ICP [15] unleashes
the potential of point-to-point matching by employing adaptive
thresholds for data association and outlier rejection. NDT [16]
forms a voxel-based probability distribution from point clouds,
bypassing point correspondences. For efficiency, LOAM [17]
extracts edge and plane features to perform matching. Building
on it, Lego-LOAM [18] incorporates ground segmentation for
noise filtering, then F-LOAM [19] reduces computation burden
through non-iterative two-step compensation.
With the rise of deep learning, many studies are dedicated to
end-to-end odometry methods, achieving competitive results.
PWCLO-Net [20] and EfficientLO [21] introduce a coarse-to-
fine framework to iteratively match point clouds and leverage
masks to suppress the effect of dynamic objects. TransLO [22]
employs window-based attention to extract global features for
reliable registration, and DSLO [23] utilizes historical features
as hidden states to raise prediction continuity. While effective,
these supervised learning methods require costly labels. Thus,
DeLORA [24] uses a KD-Tree to search for target pairs based
on predicted pose and adopts point-to-plane and plane-to-plane
error metrics to optimize model. HPPLO-Net [25] then boosts
network robustness using hierarchical point-to-plane distances
as supervision. RSLO [26] partitions point clouds into multiple
sub-regions, estimates ego-motion by regional voting, and uses
uncertainty-aware geometric loss for self-supervised training.
Although these LiDAR-based algorithms work well in normal
conditions, their performance markedly drops with point cloud
degradation from smoke and adverse weather.
B. Radar-based Odometry
To achieve odometry estimation in challenging scenes, early
works utilize scanning radar as the primary sensor, employing
point-to-line metric minimization [27], feature point extraction
and matching [28] or improved NDT with the outlier filter [29]
to estimate ego-motion. However, the blurred 2D imaging of
the sensor limits odometry precision and map quality. Its large
volume and high cost also hinder the practical application. By
contrast, compact, low-cost and higher-precision 4D radar has
attracted research attention. As an initial study, APDGICP [1]
extends GICP with probabilistic distributions to improve scan
matching. EFEAR-4D [2] applies a scan-to-map mechanism to
form denser spatial correspondences for pose solving. Besides,
a doppler-based ICP [3] is presented to harness radar velocity
information, and RCS-weighted registration [4] is proposed to
handle inherent sparsity and measurement noise in radar data.
Recently, end-to-end methods have demonstrated strong per-
formance. For example, 4DRONet [5] separately encodes radar
information with different properties and constructs two-stage
pose refinement. CMFlow [30] leverages cross-modal supervi-
sion from multi-sensor data to jointly train a network for scene
flow and ego-motion estimation. Then, CAO-RONet [31] uses
local completion to supply denser constraints for matching and
adopts multi-level resilient registration with feature similarity
to suppress noise influence. Besides, 4DRVO-Net [6] presents
a multi-modal method that integrates rich image information
to assist sparse point matching. To remove reliance on labels,
RaFlow [8] computes relative poses from predicted scene flow,
subsequently optimizing the model based on chamfer distance.
SelfRO [7] further applies spherical reprojection and velocity-
aware loss to achieve notable improvement, but there is still a
significant gap against supervised methods. More importantly,
most methods focus on the pose accuracy but neglect the map
quality. Thus, the constructed radar map remains noisy, vague
and incomplete as a result of the low-quality 4D radar points.
C. 3D Gaussian Splatting
Unlike Neural Radiance Fields (NeRFs) [32] that use MLP
to learn implicit neural scenes, 3DGS [12] represents scenarios
as explicit gaussian ellipsoids whose parameters are optimized
by differentiable rasterization and image reconstruction losses,
enabling faster convergence and rendering speed. On this basis,
GaussianPro [33] proposes a propagation operation to generate
reliable rendered depths and normals, which are further applied
to yield new gaussians to populate under-reconstructed regions
and enhance image details. To promote geometric optimization
in sparse views, DNGaussian [34] presents hard and soft depth
regularization to drive the movement of gaussians and reshape
space structure. As 3DGS struggles with precise surface fitting,
2DGS [35] proposes planar 2D gaussians with view-consistent
geometry to exactly model thin surfaces. To reconstruct scenes
at real scale, some works [36]–[38] integrate LiDAR points to
initialize gaussians. LiV-GS [36] uses a conditional constraint
to guide gaussian optimization in areas lacking LiDAR depth.
HGS-Mapping [37] optimizes hybrid gaussians with different
classes, while DrivingGaussian [38] represents dynamic scenes
by decoupling gaussians for static and moving objects.
Although advances have been made in LiDAR-based scene
reconstruction [36]–[38] and map post-processing [39], [40],
little attention has been paid to enhancing 4D radar maps, over-
looking their potential for mapping under adverse conditions.
Moreover, no existing studies have explored gaussian splitting
for optimizing maps built by 4D radar. This is a distinct topic

<!-- page 4 -->
UNDER REVIEW
4
Cluster Weighting
(a) Cluster-weighted Distance
Selection Mechanism
Cluster-aware
Matching
Uncertainty
Head
Point-Cluster Feature Encoder
Ego-motion Decoder
C
C
Cluster Feature 
Generator
Temporal 
Fusion Module 
Point-to-Point 
Distance
(b) Column Occupancy
Column-wise 
Similarity
(d) Feature Contrast
Nearest pair
Farthest pair
Contrastive Learning
Diminish
Magnify
(c) Teacher Guidence
Distance-
weighted Kabsch
s
F
t
F
s
H
t
H
s
C
tC
ˆ
M
Predicted Pose T
ˆ s
P
t
P
t
S
s
S
ˆ s
ip
fc
L
cm
L
(e)
Continuous Motion
2
kT 
1
kT 
kT
tg
T
Gaussian 
Mixture Models
t
P
ˆ s
P
T
ˆ s
tg
P
·
s
P
Source Points
Target Points
tg
L
Column
Sector
Ring
cd
L
co
L
ga
L
t
jp 
t
jp 
t
jf 
t
jf 
s
if
s
if
Cauchy-Schwarz 
 Divergence
Cluster Feature 
Generator
M
t
P
Cluster Labels
Weight Sharing
+
Loss Constraint C Concatenation + Addition T Transformation
Fig. 3. Framework of end-to-end radar odometry with multi-level constraints. The odometry network consists of a point-cluster feature encoder and an ego-
motion decoder. For self-supervised training, the loss functions first include cluster-weighted distance and column-wise occupancy comparison, which account
for the distribution and noise properties of radar points. Network learning is further selectively guided via soft labels generated by a geometry-based algorithm.
We also facilitate discriminative feature extraction using feature contrast, while enforcing motion smoothness through a constant-acceleration assumption.
because the sparsity and non-uniformity of radar points, unlike
LiDAR data, leave numerous spatial positions uninitialized and
hinder later optimization due to missing geometric priors.
III. SELF-SUPERVISED RADAR ODOMETRY WITH
MULTI-LEVEL CONSTRAINTS
A. Problem Definition
Odometry estimation aims to calculate pose transformation
T = {R, t} between consecutive timesteps using sensor data,
where R and t mean rotation and translation components. The
source and target points P ∈{Ps, Pt} are expressed as P =
{pi = {xi, fi}}N
i=1, where N is the number of points, x ∈R3
means 3D coordinates, and f ∈R2 denotes point feature with
relative radial velocity (RRV) and radar cross section (RCS).
For end-to-end odometry, a neural network Γ is used to encode
inter-frame relationship and decode pose T. Unlike supervised
learning using ground-truth pose (Rgt, tgt) as a constraint to
train model parameters θ, i.e., minθ L(Γθ(Ps, Pt), (Rgt, tgt)),
we propose a hierarchical self-supervised loss with geometric
(Lcd, Lco), temporal Lcm and feature-level Lfc terms, along
with teacher guidance Ltg and Lga, to generate comprehensive
supervision signals. Our optimization objective is as follows:
min
θ (Lcd( ˆPs, Pt) + Lco( ˆPs, Pt) + Ltg(T, Ttg) +
Lga( ˆPs, Pt) + Lfc(Fs, Ft) + Lcm(Tk−2:k)),
(1)
where ˆPs = T (Ps, T), and T is the transformation operation.
Ttg and Tk−2:k are soft labels and sequential predicted poses.
B. Cluster-aware Network Architecture
Even though radar data is much sparser than LiDAR points,
it possesses a block-like spatial distribution, making it easier
to segment into distinct objects or regions based on geometric
distances. These grouping labels then enable the integration of
Points with 
Cluster Labels
Point Features
Point Coordinates
Geometric 
Features
Cluster 
Coordinates
C
Convolution
Layer
Cluster Features H
c
F
c
X
Scatter
P
P
Concatenation
C
P Average Pooling
Group
Fig. 4. Details of cluster feature generator. It takes point features with cluster
labels as input to produce per-point cluster features.
instance features for matching, which reduces search space for
potential matches and enhances outlier rejection by enforcing
consistency within similar clusters. Motivated by this insight,
we design a cluster-aware network structured as follows:
1) Point-cluster Feature Encoder: In Fig. 3, we first employ
DBSCAN [41] to cluster raw points P ∈{Ps, Pt} and assign
cluster labels. Then, we utilize farthest point sampling (FPS) to
unify two frames into the same number N of points and obtain
the corresponding labels Cs, Ct ∈RN. The set convolution
layer [42] is further used to extract multi-scale features from
downsampled points and combine them into Fs, Ft ∈RN×D
along channel dimension.
To acquire cluster information, we propose a cluster feature
generator that exploits cluster labels to integrate point features
and coordinates belonging to the same group to produce local
geometric features Fc and centroids Xc of clusters. Both are
concatenated and processed through two convolutional layers
to obtain cluster features, which are then redistributed to their
corresponding points, generating per-point cluster features H.
This process is illustrated in Fig. 4 and can be formulated as:
Fc =
1
|Cc|
X
i∈Cc
Fi, Xc =
1
|Cc|
X
i∈Cc
Xi, ∀c ∈{1, . . . , L} (2)
Hi = Conv(Fc(i) ⊕Xc(i)), ∀i ∈{1, . . . , N}
(3)

<!-- page 5 -->
UNDER REVIEW
5
where Cc represents the set of points assigned to the cluster
c, and L signifies the total number of clusters, while Fi and
Xi denote the feature and coordinate of the i-th point from P.
Next, we leverage cost volume layers [43] to conduct feature
correlation, which compares point pairs and learns the nonlin-
ear relationship by MLP. However, rather than relying solely
on point features, we employ cluster features H to implement
cluster-aware matching as described in Eq. 4. Since cluster
features of the same object are similar in adjacent frames,
this cue strengthens recognition of specific-object relationships
and suppresses mismatches with irrelevant objects. We further
aggregate cost features in a patch-to-patch manner [43], getting
an inter-frame motion feature M.
Cost(ps
i, pt
j) = MLP(f s
i ⊕f t
j ⊕hs
i ⊕ht
j ⊕(xs
i −xt
j))
(4)
2) Ego-motion Decoder: To ensure the continuity of motion
prediction, we first employ max-pooling to motion feature M,
obtaining a global motion representation Mg ∈RD. It is then
merged with historical global feature using GRU [44], thereby
constraining the current prediction using temporal consistency
from adjacent states implicitly. The updated global feature is
further combined with M to form
ˆ
M ∈RN×2D.
Unlike previous methods [5], [7], [31] that directly average
motion features for pose prediction, we argue that focusing on
critical points rather than all is necessary, as noisy points and
dynamic objects cause errors in results. Thus, we calculate the
confidence score of each point based on its geometric feature
F and motion feature M, which implies density condition and
movement relative to neighbours. The score-weighted feature
is then used to estimate rotation R and translation t as follows:
W = SoftMax(Conv(F ⊕M))
(5)
R = Conv
N
X
i=1
(Wi · ˆ
Mi), t = Conv
N
X
i=1
(Wi · ˆ
Mi)
(6)
C. Multi-level Self-supervised Signals
1) Cluster-weighted Distance Loss: Since radar points lack
distinct structures (e.g., lines and planes), employing point re-
lationship constraints for network learning is a basic approach.
Specifically, the source points Ps are transformed by T to ˆPs,
where each ˆps
i is paired with its nearest neighbor pt
j ∈Pt, and
their distances serve as supervision. To mitigate the impact of
incorrect pairs caused by isolated and noisy points, we discard
outliers using local density and define inter-frame distance as:
ds→t
i
= I(ρ(ˆps
i) > δ)max( min
pt
j∈Pt ∥ˆps
i −pt
j∥2
2 −ϵ, 0)
(7)
where ρ(·) is a density function, while δ and ϵ mean the density
and distance thresholds. Although averaging all distances di is
convenient, this overlooks the contribution of different regions.
For example, larger point clusters typically exhibit more stable
and complete geometries across frames, enabling more precise
consistency assessment than the fragmented clusters. Thus, we
enhance point distances in large clusters and suppress those in
small clusters through cluster labels Cc as weighting factors:
Ls→t
cd
=
L−1
X
c=0
(nc
N · 1
nc
X
i∈Cc
ds→t
i
), Lcd = Ls→t
cd
+ Lt→s
cd
(8)
where nc indicates the number of points in the cluster labeled
c. The bidirectional matching strategy is also used to eliminate
ambiguity in unidirectional matching, as displayed in Fig. 3(a).
2) Column Occupancy Loss: While the point-to-point con-
straint facilitates network convergence, it is inherently difficult
for matched pairs to correspond to identical physical locations
in 3D space, resulting in geometric inconsistency. Meanwhile,
distance uncertainty of point pairs caused by inaccurate radar
measurement amplifies this issue. Thus, we propose a column
occupancy loss in Fig. 3(b), whose core idea is to transform
point clouds into voxel occupancy representation with a fixed
resolution that has tolerance for position noise and fuzzy space
correspondences. Subsequently, column-wise comparison en-
ables a more reliable consistency evaluation by operating over
a larger range, which effectively reduces the influence of local
noise that often plagues small-scale point-pair comparisons.
Specifically, given a point cloud P ∈{ ˆPs, Pt}, we first map
each point pi = (xi, yi, zi) into a 2D polar representation by
computing its azimuth angle θi and radial distance ri:
θi = 90 + atan2(yi, xi) · 180
π , ri =
q
x2
i + y2
i
(9)
The angle θi and distance ri are discretized into sectors and
rings with the resolutions controlled by ∆θ = 360◦
Ns and ∆r =
rmax
Nr , where Ns and Nr mean the number of sectors and rings.
We then generate binary occupancy matrices S ∈{0, 1}Nr×Ns
for ˆPs and Pt, respectively. The element S(k, l) indicates the
presence of at least one point in the k-th ring and l-th sector:
S(k, l) = I

∃pi s.t.
 ri
∆r

= k and
 θi
∆θ

= l

(10)
For the occupancy matrices from two frames, we calculate the
cosine similarity between corresponding columns Ss[:, l] and
St[:, l] to signify structural differences within a large receptive
field. Columns where all elements are equal to 0 due to point
sparsity and limited FOV are removed. The column occupancy
loss is expressed as the average dissimilarity across all valid
columns in Eq. 11, where L is the non-empty column indices.
Lco = 1 −1
|L|
X
l∈L
Ss[:, l] · St[:, l]
∥Ss[:, l]∥2 · ∥St[:, l]∥2
(11)
3) Teacher Guidance Loss: Beyond the above losses, which
propel model learning by rigid geometric alignment, we think
that creating pseudo labels as soft supervision for the network
can transfer some meaningful knowledge based on distillation
learning theory [45], [46]. Meanwhile, unlike common distil-
lation methods (e.g., [47]) that use a large network to generate
pseudo labels for training a lightweight model, we propose a
teacher guidance loss (Fig. 3(c)) that regards a geometry-based
method as the teacher. It uses the network’s predicted pose as
an initial guess for the pose refinement. A selective distillation
is then adopted to identify and discard unreliable poses, while
using trustworthy ones as soft labels to drive model learning
towards better geometric solutions.
Concretely, based on the transformed point cloud ˆPs by T,
we apply the distance-weighted Kabsch algorithm to solve the

<!-- page 6 -->
UNDER REVIEW
6
Gaussian Map
Sparse 4D Radar Map
Initialize
Restore
Multi-view Overlap Regularization
Depth & Normal Maps
Geometry-aware 
Densification
Depth-assisted 
Ground Completion
Gaussian Growth Strategy
Neighborhood-
based Pruning
Sky Floater 
Decoupling
Gaussian Selective Separation
Pre-trained Model
Resplitting
Interpolation
Save
Prune
d

mvs

Update
Multi-view Images
Optimize
Current View
Render
Surrounding View
(b) Our Densification 
(a) Traditional Densification 
Splitting
n

Dense 4D Radar Map
No Ground
Visual Scale
Spatial Scale
Fig. 5. Overview of gaussian-based map optimization. We first initialize a sparse 4D radar map as gaussians and perform attribute optimization by gradients
derived from multi-view image rendering with the depth and normal maps. During optimization, we adopt ground completion and geometry-aware densification
to grow the number of gaussians to reconstruct missing structure, while employing selective separation to decouple sky floaters and avoid excessive pruning.
Finally, the optimized gaussians are restored into a dense radar map.
pose adjustment ∆T = {∆R, ∆t} by minimizing the nearest-
neighbor distance between ˆPs and Pt, defined as:
dmax = max
i ( min
pt
j∈Pt ∥ˆps
i −pt
j∥2), wi = dmax −di
dmax
(12)
∆T = argmin
N
X
i=1
wi∥∆Rˆps
i + ∆t −NN(ˆps
i, Pt)∥2
2
(13)
where wi and NN(·) are the distance weight and nearest neigh-
bor search, and the updated pose is Ttg = ∆T ·T = {Rtg, ttg}.
Despite using T as an initial guess, the optimized pose Ttg may
fail to improve on T because of the challenge posed by radar
point sparsity to the Kabsch algorithm. Therefore, a selection
mechanism is adopted to map Ps to ˆP s
tg by Ttg, and to convert
{ ˆP s, ˆP s
tg, P t} into gaussian mixture models (GMMs) [48] and
evaluate the alignment of GMMs through Cauchy-Schwarz
(CS) divergence. When Ttg produces a lower CS divergence
between the projected and target points Pt compared to T, it
can be regarded as a valid refined pose and applied to constrain
the model training as follows:
M = I(g(G( ˆPs), G(Pt)) > g(G( ˆPs
tg), G(Pt)))
(14)
Ltg = M · (∥R −Rtg∥1 + ∥t −ttg∥1)
(15)
where g and G mean functions of CS divergence and gaussian
mixture. As a result, selective guidance avoids forcing model
to imitate erroneous poses. In addition, we use CS divergence
of G( ˆPs) and G(Pt) as a loss term Lga to drive model learning
when there are no reliable soft labels available (i.e. M = 0).
4) Feature Contrast Loss: Apart from the result-level con-
straints, the feature-level regularization is also critical, as pose
estimation relies on inter-frame feature correlation. Thus, we
introduce a feature contrast loss in Fig. 3(d) to enhance feature
discrimination among points and help perceive subtle motion
between frames.
We initially consider the backbone’s outputs Fs and Ft as
feature sources. For each point ˆps
i = (ˆxs
i, f s
i ) ∈ˆPs, we define
its nearest point pt
j+ = (xt
j+, f t
j+) ∈Pt as a positive sample,
whereas the farthest point pt
j−= (xt
j−, f t
j−) is selected as the
negative sample instead of neighboring points. This is because
the neighbors’ features are usually similar to ˆps
i, causing weak
contrastive signals, which also may generate a false perception
that neighbors exhibit larger discrepancies from ˆps
i than non-
neighboring points. Thus, using the farthest point from a global
perspective as a negative sample is more reasonable. Next, we
use information noise contrastive estimation as a loss function:
j+ = argmin
j∈[1,N]
∥ˆxs
i −xt
j∥2, j−= argmax
j∈[1,N]
∥ˆxs
i −xt
j∥2
(16)
Lfc = −1
N
N
X
i=1
log
es(fi,fj+)/τ
es(fi,fj+)/τ + es(fi,fj−)/τ
(17)
where j+ and j−stand for the positive and negative sample
indices. s represents the cosine similarity of features between
point ˆps
i and its positive and negative samples. τ denotes the
temperature coefficient.
5) Continuous Motion Loss: Supervised learning methods
such as [31] implicitly model temporal motion by a network
trained with continuous ground truth, which enforces predicted
poses to conform to short-term motion patterns. However, for
self-supervised learning, only relying on feature-level temporal
fusion is still insufficient owing to missing explicit consecutive
constraints. To address this issue, we apply the assumption of
constant acceleration to ensure that the pose predictions main-
tain physical rationality. To be specific, we divide the training
sequence into L-length frame segments and store the historical
results in a buffer. For each predicted pose Tk(2 ≤k < L)
in the segment, we retrieve poses {Tk−2, Tk−1} from the
buffer and optimize the acceleration consistency for prediction
stability utilizing Lcm:
αR
k = (Rk −Rk−1), αt
k = (tk −tk−1), Tk = [Rk|tk] (18)
Lcm = λcm(max(∥αR
k −αR
k−1∥1 −ϵR, 0)
+ max(∥αt
k −αt
k−1∥1 −ϵt, 0))
(19)
where the relative rotation Rk is parameterized in Euler angles.
To account for accelerated motion in real-world driving, we
apply thresholds ϵR and ϵt to avoid too tight constraints, while
enabling a gradual decay of the loss weight λcm for Lcm.

<!-- page 7 -->
UNDER REVIEW
7
IV. GAUSSIAN-BASED MAP OPTIMIZATION
A. Preliminary
The radar map is transformed into a representation signified
by anisotropic gaussian ellipsoids G. Each gaussian contains
its center µ ∈R3 in the world coordinate, opacity o ∈R, scale
S ∈R3, rotation R ∈R3, and spherical harmonics SH ∈R3.
Here, µ corresponds to radar point coordinates. SH is obtained
by mapping the RGB pixel values. S is initialized by distances
to neighbors. The covariance matrix Σ is calculated to define
the gaussian shape. Thus, the gaussian distribution is given by:
Σ = RSST RT , G(x) = e(−1
2 (x−µ)T Σ−1(x−µ))
(20)
Afterwards, given the transformation matrix T C
W = {RC
W , tC
W }
between the camera and world coordinate systems, 3D gaus-
sian ellipsoids (µ, Σ) in the world frame are projected onto the
image plane as 2D gaussian (µ′, Σ′) by the following equation:
µ′ = π(T C
W · µ), Σ′ = JRC
W Σ(RC
W )T (J)T
(21)
where RC
W and tC
W are the rotation and translation components.
J ∈R2×3 is the Jacobian of the affine approximation of T C
W .
π means the 3D-to-2D projection operation. The influence of
2D gaussian on pixel ρ = [u, v]T is determined by weight α:
α = o · e(−1
2 (µ′−ρ)T Σ′−1(µ′−ρ))
(22)
Gaussians are then sorted by depth and rendered by volumetric
α-blending to calculate the color of pixel ρ, as expressed by:
C(ρ) =
n
X
i=1
ciαi
i−1
Y
j=1
(1 −αj)
(23)
where c denotes the RGB color from spherical harmonics SH.
B. Gaussian Growth Strategy
1) Depth-assisted Ground Completion: Caused by specular
reflection, the original radar map typically has fewer points on
the ground compared to other areas (Fig. 5), posing a challenge
for ground reconstruction with poor gaussian initialization. At
the same time, standard gaussian densification in 3DGS [12]
only generates a few ground gaussians for low-texture ground
rendering in the image, leading to discontinuous ground in the
optimized map (Fig. 6(a)). Thus, we propose a depth-assisted
ground completion mechanism as described in Fig. 6(b), which
integrates synthetic ground gaussians into optimization process
through the depth estimation of the visual foundation model.
The positions of these gaussians are adjusted through gradients
from the differentiable rasterization to form a realistic ground
in the final radar map. For optimization within a specific view,
the ground completion consists of three steps:
a) Virtual Points Generation: DepthAnythingV2 [49] is first
utilized to estimate a metric depth map D of image I from the
current view, which is further mapped to the world coordinate
by a transformation matrix T C
W , generating virtual points Pa.
b) Ground Fitting: We transform gaussians Gb ⊂G within
the current view into points Pb according to their center µb.
Owing to the sparsity of ground points, we design an iterative
RANSAC with normal-based filtering to fit an initial ground
plane to Pb. If the plane’s normal departs from vertical, nearby
(a) The visual comparison of ground reconstruction
3DGS
Ours w/ Ground Completion
Ours w/o Ground Completion
Project
Iterative
RANSAC 
Iterative
RANSAC 
Ground 
Projection
Transform

Depth Image
a

b

b
G
a

b

g
a

g
b

(b) The process of depth-assisted ground completion 
Add
Fig. 6. The detail of depth-assisted ground completion and visual comparison.
outlier points are removed, and the plane is refitted until the
normal approaches vertical direction. Afterwards, the resulting
ground plane Qb = [nb, db] is determined, where nb and db are
the normal vector and distance from the plane to the origin. For
the virtual points Pa, we apply the same approach to calculate
Qa = [na, da] and nearby ground points Pg
a.
c) Ground Projection: We obtain the rotation matrix R and
translation t between planes via Rodrigues’ rotation formula:
v = ˆna × ˆnb, ˆna =
na
∥na∥2
, ˆnb =
nb
∥nb∥2
(24)
R = I + [v]× + [v]2
×
1 −ˆna · ˆnb
∥v∥2
2
, t = (da −db) · ˆnb
(25)
where [v]× means the skew-symmetric matrix of v. Then, we
incorporate new ground gaussians that are initialized from the
projected points Pg
b = R·Pg
a +t into the original gaussian set
G. Finally, by iteratively applying this process across multiple
views, we achieve a dense reconstruction of ground structure.
2) Geometry-aware Densification: Due to the limited FOV
and uneven distribution of radar points, initial gaussians only
cover partial image regions, which forces them to be enlarged
before being split into tile vacant regions. However, the com-
mon binary splitting (in Fig. 5(a)) struggles to efficiently deal
with large-scale gaussians for rapid densification. Meanwhile,
post-split positions are sampled from the rendering-optimized
visual scales rather than physical spatial scales. This means the
splitting results are tailored for image rendering but may fail
to accurately restore the fine 3D structure. Thus, we present a
geometry-aware splitting and interpolation scheme that yields
gaussians adhering to local distributions and improves struc-
tural integrity.
For efficiency and a trade-off between image rendering and
map quality, we extend initial splitting in [12] with a geometry-
aware resplitting stage as illustrated in Fig. 5(b). This process
operates on a subset Gs of G, which comprises I gaussians
that still retain large scales S after the first splitting. For each
Gi
s, we find its nearest gaussian Gj
s and use their distance as
a spatial scale ˆSi. Then, we sample the M offsets σm
i
from a
normal distribution with a mean of zero and variance of ˆSi:
ˆSi = ∥µi −µj∥2, {σm
i }M
m=1 ∼N(0, ˆSi)
(26)

<!-- page 8 -->
UNDER REVIEW
8
(a) Raw Points 
(only trunk)
(c) Ours w/o          and  
GA Densification 
mvs
L
(b) 3DGS
(e) Ours 
(d) Ours w/o GA 
Densification
(f) Corresponding Image
4453
4055
1871
2647
73
0
1000
2000
3000
4000
5000
e
d
c
b
a
Point Number
Point Number: 61.0 times
Sky Floaters
Fig. 7. Comparison of a tree reconstruction result by 3DGS and our method.
We also display the total number of points to reveal the impact of our modules
on structural integrity and density. GA is the abbreviation for geometry-aware.
Afterwards, the center µi of gaussian Gi
s is transformed using
offset σm
i , and its scale Si is decayed by ratio α. This parallel
process generates M × I new gaussians from Gs, collectively
denoted as ˆGs = { ˆGk
s}M×I
k=1 . Upon creation of ˆGs, the original
set Gs is removed from G to eliminate redundancy, getting an
updated gaussian set G′. This procedure is denoted as follows:
ˆµk = µi + Riσm
i , ˆSk = α · Si, G′ = (G \ Gs) ∪ˆGs
(27)
Next, we adopt an interpolation strategy to improve the local
density of gaussians. The gaussians Gv ⊂G with high opacity
are first selected. For each Gi
v, a k-nearest neighbor set Ni is
constructed, and distant gaussians are discarded to produce ˆ
Ni.
Within the neighborhood, new ellipsoid centers ˆµi and colors
ˆci are obtained through the following interpolation operation:
ˆ
Ni =

Gk
v ∈Ni | ∥µi −µk∥2 ≤dmax
	
(28)
ˆµi =
1
| ˆ
Ni|
X
Gj
v∈ˆ
Ni
µj, ˆci =
X
Gj
v∈ˆ
Ni
P ∥µi −µj∥2
∥µi −µj∥2
· cj
(29)
In the end, we incorporate new gaussians ˆGv into the original
gaussians G to fill the void of local structures.
C. Gaussian Selective Separation
1) Sky Floater Decoupling: Radar map inherently contains
virtual structures formed by noise, where the gaussians initial-
ized from these artifacts are usually projected into sky regions
and undergo splitting, resulting in numerous floaters near the
actual structures. In addition, as shown in Fig. 7(b, f), the area
that intersects with the sky in the image also tends to produce
sky floaters. To solve these problems, we maintain a sky mask
M3D
sky ∈{0, 1}N (in Fig. 8), which is updated gradually as new
sky gaussians appear by splitting or existing ones are pruned
during the optimization process. To determine which gaussian
ellipsoids belong to the sky, we feed the current image I into
MaskFormerv2 [50] to yield a 2D mask M2D
sky ∈{0, 1}H×W ,
where H and W are the image height and width. We then
Split
Prune
Sky Floater Identification
Sky Mask Maintenance
Transform
Project
Update
MaskFormerv2
2D
sky

3D
sky

i

Fig. 8. The procedures of sky floater identification and sky mask maintenance.
project the center µi of each gaussian to the image plane, and
update the mask M3D
sky by the corresponding values in M2D
sky:
(ui, vi) = Π(µi), M3D
sky(i) = M2D
sky(⌊ui⌋, ⌊vi⌋)
(30)
where Π(·) and (ui, vi) mean the projection operator and pixel
coordinates. As a result, unlike directly deleting sky gaussians,
our decoupling strategy using a mask not only reduces floaters
in the final radar map but also ensures that sky information is
not lost during image rendering.
2) Neighborhood-based Pruning: As noted in Sec. IV-B2,
uneven distribution of radar points causes oversized gaussians
during the initial phase, later shrunk through splitting. How-
ever, most approaches directly remove all large gaussians at a
fixed interval, which will mistakenly prune gaussians that are
waiting to be split to recover the missing structure beyond the
radar FOV. Fortunately, we notice that valuable large gaussians
usually extend from existing structures with denser neighbors.
Building upon this, we present a neighborhood-based pruning
criterion Mp that determines whether the gaussians should be
removed by incorporating spatial relationships. Specifically, a
gaussian is discarded when its average center distance relative
to neighbors exceeds a threshold τd, and either its scale Si or
projected 2D radius r2D
i
surpasses thresholds τs or τr:
Mp = I(d(µi) > τd) ∧(I(Si > τs) ∨I(r2D
i
> τr))
(31)
where d(·) denotes the KNN-distance function. In this way, the
large gaussians utilized for subsequent completion are avoided
from being deleted incorrectly.
D. Multi-view Overlap Regularization
Existing methods [12], [33], [35] usually optimize gaussians
by minimizing the photometric loss between the single-view
rendered image ˆIt and the ground truth It. However, geomet-
ric ambiguity in single-view observations makes it challenging
for both the initial gaussians (with radar noise) and the post-
split ones to shift quickly to accurate spatial locations, while
risking overfitting in their distribution. We thus present a multi-
view overlap regularization that renders images ˆIt+w (w ∈
{−2L, −L, L, 2L}) from nearby timesteps and compares them
with It+w as additional supervision signals for gaussians in
the current view. Since co-visible regions At+w typically exist
between ˆIt+w and ˆIt, and satisfy an overlap relationship
S At+w = ˆIt in most cases, this means that the corresponding
gaussians for ˆIt are constrained by at least two perspectives, as
shown in Fig. 5. Thus, gaussian positions and other attributes
can be optimized toward unambiguous states more easily. The

<!-- page 9 -->
UNDER REVIEW
9
Thermal 
Camera
RGB 
Camera
IMU
LiDAR
4D Radar
LiDAR-RGB
4D Radar-RGB
LiDAR-Thermal
4D Radar-Thermal
Fig. 9. The visualization of our handheld equipment and the projection results among different sensors. Multi-sensor data are collected across various campus
scenes under both daytime and nighttime conditions.
multi-view overlap regularization is defined by the following
functions:
Lc = 1 −λ
N
N
X
i=1
|ˆIi
t −Ii
t|γ + λLSSIM(ˆIt, It)
(32)
Lmvc = λmvc
4
4
X
i=1
Lc(ˆIt+wi, It+wi) + Lc(ˆIt, It)
(33)
where λ = 0.1, λmvc = 3.0 indicate the loss weights, and we
adopt L1 loss with exponential weight to strengthen the fitting
on difficult samples. We also intuitively exhibit the influence
of multi-view overlap regularization in Fig. 7(c, d).
E. Loss Functions
Except for multi-view regularization, we apply prior knowl-
edge derived from visual fundamental models [49], [51] to op-
timize the geometric distribution of gaussians. We first render
depths ˆD and normals ˆ
N through Eq. 34. We then supervise
depths ˆD with L1 loss against the predicted depths D obtained
from DepthAnythingv2 [49], while adopting Metric3D [51] to
get normals N for regularizing ˆ
N by normal similarity loss.
Both losses are defined in Eq. 35.
ˆD =
n
X
i=1
diαi
i−1
Y
j=1
(1 −αj),
ˆ
N =
n
X
i=1
niαi
i−1
Y
j=1
(1 −αj) (34)
Ld = ∥(D −ˆD)∥1, Ln = 1 −N · ˆ
N
(35)
To sum up, the overall loss function is formulated as follows:
L = Lmvc + Ln + Ld
(36)
where Lmvc means the multi-view overlap regularization loss
(Eq. 33), enforcing consistency of gaussian attributes across
multiple views to resolve spatial ambiguity. Then, Ln and Ld
incorporate monocular geometric priors to guide the gaussians
towards forming accurate local structures and smooth surfaces.
V. EXPERIMENT
A. Datasets
To thoroughly evaluate our method, we conduct comprehen-
sive experiments not only on public datasets, including View-
of-Delft [52] and NTU4DRadLM [9], but also on our self-
collected campus dataset to assess both odometry estimation
accuracy and map optimization performance in various scenes.
The details of the used datasets are shown in Tab. I.
TABLE I
THE DETAILS OF THE USED DATASETS. 3DL: 3D LIDAR, 4DR: 4D
RADAR, VC: VISUAL CAMERA, TC: THERMAL CAMERA. DENSITY
DENOTES THE DENSITY OF RADAR POINTS IN EACH FRAME.
Dataset
Sensors
Scenes
Density Radar Frame# Sequence#
3DL 4DR VC TC Day Night
VoD [52]
✓
✓
✓
✓
Low
8,693
24
NTU [9]
✓
✓
✓
✓
✓
High
61,233
6
Ours
✓
✓
✓
✓
✓
✓
Low
38,892
10
1) View-of-Delft: The VoD dataset [52] provides synchro-
nized multi-sensor data, consisting of 64-beam LiDAR, RGB
camera and 4D radar measurements captured in complex urban
traffic environments. With a total of 8,693 frames, the dataset
is structured into 24 continuous sequences. Consistent with the
data partitioning scheme of [6], we divide these sequences into
13 for training, 4 for validation, and 7 for testing. Since each
radar frame contains only 300-500 points, VoD presents severe
challenges for odometry estimation and map optimization.
2) NTU4DRadLM: The NTU4DRadLM dataset [9] is col-
lected using an Oculii Eagle 4D radar, Livox Horizon LiDAR,
an RGB camera, and a thermal camera, which are not time-
synchronized. It captures six trajectories across NTU campus,
including structured, semi-structured, and unstructured scenar-
ios. Data are collected from both a handcart platform moving
at around 1m/s and vehicle-mounted systems traveling at 25-
30km/h. For odometry estimation, we partition the sequences
into training (Garden, Nyl, Loop1), validation (Loop2), and
testing (Loop3, Cp) sets. In particular, the radar point clouds
in NTU4DRadLM exhibit a higher density relative to the VoD
dataset, typically comprising thousands of points per frame.
3) Self-Collected Campus Dataset: To conduct more exten-
sive experiments, we build a handheld data collection system
that is more compact and portable compared to the handcart
platform in NTU4DRadLM. Our system integrates a 16-beam
LiDAR, GPAL 4D radar, RGB camera, thermal camera, and
IMU. Here, the 4D radar operates at around 15 Hz and collects
around 500 points per frame. The setup of our multi-sensor
equipment and the projection of the point cloud onto the image
are shown in Fig. 9. Using this platform, we build a campus
dataset with 8 daytime sequences and 2 nighttime sequences
under various scenes. For daytime sequences (34,241 frames
total), we divide them into training, validation, and testing sets
based on a 6:1:3 ratio. The validation set (Gym) includes 3,124
frames, while the testing set (Building2, Parking2, Straight)

<!-- page 10 -->
UNDER REVIEW
10
TABLE II
4D RADAR ODOMETRY EXPERIMENT RESULTS ON VIEW-OF-DELFT (VOD) DATASET.
Method
03
04
09
17
19
22
24
Mean
trel
rrel
trel
rrel
trel
rrel
trel
rrel
trel
rrel
trel
rrel
trel
rrel
trel
rrel
Classical-based
ICP-po2po
0.39
1.00
0.21
1.14
0.15
0.72
0.16
0.53
1.40
4.70
0.44
0.76
0.24
0.77
0.43
1.37
ICP-po2pl
0.42
2.19
0.37
1.83
0.50
1.32
0.23
0.68
3.04
5.62
0.42
1.20
0.35
0.67
0.76
1.93
GICP
0.46
0.68
0.30
0.39
0.51
0.32
0.40
0.10
0.51
1.23
0.34
0.57
0.15
0.30
0.38
0.51
NDT
0.55
1.60
0.47
0.91
0.46
0.56
0.44
0.40
1.33
2.58
0.47
1.10
0.36
1.84
0.58
1.28
LiDAR-based
Full A-LOAM
NA
NA
0.03
0.09
0.04
0.19
0.02
0.04
0.38
1.35
0.06
0.18
0.06
0.20
0.10
0.34
A-LOAM w/o mapping
NA
NA
0.14
0.35
0.16
1.23
0.09
0.26
1.17
4.63
0.27
0.92
0.16
0.81
0.33
1.37
LO-Net
1.05
1.78
0.26
0.49
0.30
0.36
0.57
0.14
3.29
3.07
1.00
1.12
0.77
1.45
1.03
1.20
PWCLO-Net
0.26
0.37
0.31
0.40
0.38
0.55
0.27
0.39
1.23
0.91
0.23
0.35
0.46
0.82
0.45
0.54
4D Radar-based
Supervised
CMFlow
0.06
0.10
0.05
0.09
0.09
0.14
0.06
0.03
0.28
0.94
0.14
0.29
0.12
0.58
0.11
0.31
4DRO-Net
0.08
0.10
0.04
0.07
0.13
0.38
0.09
0.10
0.91
0.62
0.23
0.32
0.28
1.20
0.25
0.40
CAO-RONet
0.05
0.03
0.04
0.04
0.06
0.09
0.10
0.01
0.02
0.05
0.08
0.06
0.14
0.08
0.07
0.05
DNOI-4DRO
0.02
0.02
0.02
0.03
0.02
0.02
0.02
0.02
0.02
0.04
0.03
0.02
0.03
0.05
0.02
0.03
4D Radar-based
Self-supervised
RaFlow
0.87
2.09
0.07
0.44
0.11
0.09
0.13
0.03
1.22
4.09
0.72
1.34
0.25
1.14
0.48
1.32
Ours
0.04
0.07
0.04
0.03
0.04
0.05
0.03
0.02
0.04
0.04
0.06
0.10
0.12
0.05
0.05
0.05
TABLE III
4D RADAR ODOMETRY EXPERIMENT RESULTS ON VOD DATASET USING
THE SAME CONFIGURATION AS SELFRO. BLUE AND RED ARE USED TO
DISTINGUISH SUPERVISED AND SELF-SUPERVISED METHODS.
Method
00
03
04
07
23
Mean
trel rrel trel rrel trel rrel trel rrel trel rrel trel rrel
CMFlow
0.04 0.05 0.07 0.09 0.06 0.09 0.03 0.04 0.09 0.14 0.06 0.08
4DRO-Net
0.08 0.03 0.06 0.05 0.08 0.07 0.05 0.03 0.10 0.15 0.07 0.07
CAO-RONet 0.05 0.03 0.02 0.03 0.03 0.05 0.02 0.02 0.04 0.06 0.03 0.04
RaFlow
0.61 0.84 0.87 1.98 0.07 0.45 0.07 0.04 0.42 1.16 0.41 0.90
SelfRO
0.07 0.11 0.10 0.16 0.05 0.13 0.07 0.14 0.13 0.14 0.08 0.14
Ours
0.03 0.04 0.04 0.07 0.03 0.03 0.01 0.02 0.05 0.04 0.03 0.04
Impr. (%)
57.1 63.6 60.0 56.3 40.0 76.9 85.7 50.0 61.5 71.4 62.5 71.4
TABLE IV
4D RADAR ODOMETRY EXPERIMENT RESULTS ON NTU4DRADLM. “R”
AND “A” MEAN THE RELATIVE POSE ERROR (RPE) AND ABSOLUTE
TRAJECTORY ERROR (ATE), RESPECTIVELY.
Method
Loop2 (4.79km)
Loop3 (4.23km)
Cp (0.25km)
R (m) R (°) A (m) R (m) R (°) A (m) R (m) R (°) A (m)
ICP
12.357 8.003 613.073 13.772 3.818 304.793 0.543 1.274 3.639
NDT
3.406 5.775 652.073 2.798 3.579 169.034 0.223 1.368 3.294
GICP
2.448 2.517 58.155
3.161 2.052 29.224
0.192 1.184 2.256
APDGICP
2.459 2.616 145.674 3.072 2.209 27.871
0.216 1.241 2.408
CAO-RONet 1.504 4.215 265.864 4.228 2.604 481.916 0.342 2.471 15.557
RaFlow
NA
NA
NA
NA
NA
NA
1.310 9.367 44.228
Ours
1.521 2.448 38.876
2.104 1.769 22.077
0.248 1.121 1.749
contains 9,701 frames. Moreover, nighttime sequences are uti-
lized to evaluate the performance of thermal image rendering
and map optimization in low-light conditions.
B. Implementation Details
1) Data Processing: For odometry data preprocessing, we
first apply a height-based filtering to the point clouds, retaining
only those points within the vertical range of [-3m, 3m]. To
augment the training data diversity in the VoD dataset, we flip
training sequences to generate reversed-motion data. Random
translations within ±1m along each axis are also employed to
point clouds during training. For DBSCAN-based clustering,
we use a 3-meter search radius and a minimum cluster size of
3 points for VoD and self-collected datasets, while configuring
a 1.5-meter radius and a 5-point minimum for NTU4DRadLM.
In the map optimization pipeline, we downsample the points
of radar maps in the NTU4DRadLM to 25% of their original
10
15
20
25
30
35
40
45
Runtime (ms)
0.0
0.2
0.4
0.6
0.8
1.0
Odometry Estimation Error
LO-Net
4DRO-Net
Ours
CAO-RONet
CMFlow
DNOI-4DRO
RaFlow
Fig. 10. Comparison of runtime and odometry estimation error across several
methods on Delft dataset. The error is computed as the mean of rrel and trel.
Our method achieves an optimal balance between performance and efficiency.
Ground Truth
Ours
GICP
APDGICP
Loop2
Loop3
Fig. 11. The trajectory visualization on the long sequences of NTU4DRadLM.
density for gaussian initialization and select image keyframes
at a 5-frame interval to optimize maps. For the VoD dataset,
no sampling is applied to either images or point clouds.
2) Network Training & Gaussian Optimization: For odom-
etry, our model is trained for 60 epochs on a single NVIDIA
RTX 4090 GPU through the Adam optimizer with a batch size
of 24. Meanwhile, we assign an initial learning rate of 1×10−3
with a per-epoch decay rate of 0.9 and employ equal weights
for all self-supervised loss terms. For map optimization, the
gaussian parameters are updated utilizing Adam optimizer for
15,000 iterations, with gaussian densification every 100 steps
starting at iteration 500 with 5×10−4 gradient threshold.
3) Evaluation Metrics: For odometry evaluation on the
VoD, we adopt the relative pose error (RPE) metric, following
previous methods [5], [6], [31]. With this metric, we calculate
the root mean square error (RMSE) for rotation (◦/m) and
translation (m/m), with 20m intervals ranging from 20m to
160m. For the NTU4DRadLM and self-collected datasets, we
employ the same approach as 4DRadarSLAM [1] to evaluate
absolute trajectory error (ATE) and RPE. In our self-collected

<!-- page 11 -->
UNDER REVIEW
11
Ours
3DGS
GaussianPro
Ground Truth
2DGS
Seq. 03
Seq. 24
Garden
Cp
Fig. 12. Qualitative comparisons of rendering quality on the VoD and NTU4DRadLM datasets. We mark poor rendering regions of 3DGS with yellow circles.
TABLE V
THE PERFORMANCE OF MAP OPTIMIZATION AND IMAGE RENDERING ON VIEW-OF-DELFT (VOD) DATASET. “ODOM” IS DEFINED AS USING POSES
ESTIMATED BY OUR SELF-SUPERVISED ODOMETRY TO CONSTRUCT INITIAL RADAR MAPS.
Method
03
04
09
22
24
F-Score↑
CD↓
MHD↓F-Score↑
CD↓
MHD↓F-Score↑
CD↓
MHD↓F-Score↑
CD↓
MHD↓F-Score↑
CD↓
MHD↓
GaussianPro
0.378
1.482
1.148
0.297
1.881
1.486
0.253
2.514
1.801
0.357
1.481
0.975
0.241
3.085
6.817
2DGS
0.271
1.441
1.316
0.263
1.092
0.537
0.254
1.113
0.358
0.186
1.422
0.704
0.125
2.827
17.587
3DGS
0.339
1.171
1.642
0.351
0.951
0.721
0.364
0.987
0.589
0.349
1.171
0.831
0.318
1.728
1.485
Ours
0.509
0.955
0.271
0.501
0.733
0.164
0.556
0.624
0.105
0.571
0.664
0.099
0.428
1.128
0.304
3DGS (Odom)
0.208
1.507
3.983
0.285
1.229
0.977
0.311
1.049
0.918
0.241
1.548
1.923
0.254
1.809
3.017
Ours (Odom)
0.394
1.196
0.670
0.438
0.731
0.188
0.389
0.892
0.267
0.433
1.083
0.473
0.402
1.161
0.352
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
GaussianPro
15.783
0.494
0.488
18.721
0.647
0.391
16.805
0.664
0.434
17.171
0.615
0.466
20.551
0.733
0.339
2DGS
14.713
0.469
0.569
17.877
0.641
0.433
16.849
0.669
0.454
14.901
0.582
0.514
21.235
0.744
0.369
3DGS
15.965
0.505
0.511
19.184
0.666
0.389
18.271
0.696
0.408
16.727
0.609
0.476
21.858
0.762
0.321
Ours
19.473
0.571
0.453
21.366
0.683
0.371
20.862
0.725
0.387
20.938
0.659
0.432
24.208
0.787
0.316
3DGS (Odom)
15.702
0.504
0.516
19.002
0.661
0.397
17.956
0.693
0.413
16.631
0.605
0.488
21.845
0.760
0.323
Ours (Odom)
19.306
0.561
0.467
21.295
0.681
0.376
20.821
0.724
0.388
20.879
0.657
0.438
24.218
0.788
0.316
dataset, we use FAST-LIO [53] as the ground-truth trajectory
because of the multipath errors caused by tree and building ob-
structions, which will degrade the accuracy of RTK GPS. For
map quality assessment, we construct LiDAR maps as ground
truth and follow previous point cloud completion approaches,
such as [10], [11], to utilize the L1 Chamfer Distance (CD),
Modified Hausdorff Distance (MHD), and F-Score (with a
distance threshold of 0.3m) as metrics for comprehensive
evaluation. Furthermore, to evaluate image rendering quality,
we adopt three metrics, namely the peak signal-to-noise ratio
(PSNR), learned perceptual image patch similarity (LPIPS),
and structural similarity index measure (SSIM), as utilized in
3DGS [12].
C. Performance Comparisons on Public Datasets
1) 4D Radar Odometry: As displayed in Tab. II, since each
radar frame of the VoD dataset only includes 300-500 points,
classical approaches such as ICP-po2po, ICP-po2pl, GICP, and
NDT are difficult to find sufficient geometric correspondences
for pose estimation, resulting in unsatisfactory results. Besides,
the differences in sensor characteristics make it challenging for
the geometric-based method (A-LOAM [17]) and learning-
based algorithms (LO-Net [54] and PWCLO-Net [20]) de-
signed for LiDAR to be applicable to 4D radar. Among radar-
based learning works, our self-supervised framework surpasses
RaFlow [8] notably based on constraints from various aspects
and reduces the gap compared to supervised learning methods
obviously in a label-free manner. Besides, as shown in Fig. 10,
our self-supervised approach runs at 15.9ms, outperforming
supervised CAO-RONet [31] (20.2ms) and DNOI-4DRO [55]
(43.2ms). In Tab. III, since SelfRO [7] is not open-sourced, we
employ the same sequence division protocol as described in
its paper to train our model for a fair comparison. The results
validate the superiority of our method over SelfRO.
Due to lacking time synchronization in NTU4DRadLM, we
first perform an interpolation operation to compute the ground-
truth poses corresponding to the 4D radar timestamps and then
train CAO-RONet [31]. However, the results in Tab. IV show
that [31] suffers from poor performance, implying that super-
vised algorithms heavily rely on precise time synchronization

<!-- page 12 -->
UNDER REVIEW
12
TABLE VI
THE EVALUATION OF MAP OPTIMIZATION AND IMAGE RENDERING ON
NTU4DRADLM. GSPRO IS THE ABBREVIATION FOR GAUSSIANPRO.
Method
Cp (Structured)
Garden (Unstructured)
Nyl (Semi-structured)
F-Score↑CD↓MHD↓F-Score↑CD↓MHD↓F-Score↑CD↓MHD↓
GSPro
0.297
3.512
3.212
0.501
0.877
0.221
0.198
2.087
1.384
2DGS
0.222
1.109
1.219
0.287
1.037
0.268
0.061
2.073
1.723
3DGS
0.382
1.104 0.521
0.545
0.749
0.132
0.228
2.592
2.039
Ours
0.615
0.692
0.091
0.688
0.467
0.041
0.372
1.016
0.345
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
GSPro
17.411
0.546
0.506
15.908
0.473
0.547
14.793
0.466
0.604
2DGS
17.363
0.555
0.519
14.741
0.461
0.625
14.456
0.474
0.613
3DGS
18.385
0.578
0.466
17.635
0.521
0.544
15.319
0.488
0.591
Ours
19.821
0.601
0.445
19.608
0.549
0.502
18.788
0.536
0.529
Raw Radar Map
3DGS 
Ours
Seq. 04
Seq. 24
Cp
Fig. 13. Visualization of the optimized radar maps. Our algorithm generates
denser and more complete scenes compared to 3DGS.
and accurate poses as supervision signals. In contrast, our self-
supervised method breaks free from these limitations and still
works well. Especially for long sequences (Loop2 and Loop3)
with 4.79 km and 4.23km, we achieve ATE of 38.876m and
22.077m, representing a reduction of 73% and 21% compared
to APDGICP [1] (145.674m and 27.871m), respectively.
In addition, Fig. 11 further demonstrates that our trajectories
in kilometer-level sequences are closer to the ground truth
than traditional GICP and APDGICP designed for 4D radar,
especially in rotation. This improvement could be attributed to
our method’s ability to learn effective inter-frame correspon-
dence through multi-level constraints, including noise-tolerant
geometric consistency, teacher guidance, feature contrast, and
constant-acceleration motion modeling. And the effectiveness
of these constraints is validated in ablation studies (Sec. V-D1).
We also show more visualizations of trajectories based on our
self-collected dataset in Fig. 18.
2) Map Optimization: In this part, we select five sequences
from test sets of the VoD dataset, then use ground-truth poses
and predicted odometry to build initial radar maps respectively.
The former isolates map optimization from odometry errors,
enabling a more focused and pure evaluation of our proposed
method. The radar maps are further optimized by differentiable
rasterization. The results in Tab. V indicate that our algorithm
achieves better geometric accuracy in the reconstructed maps,
with mean CD and MHD of 0.821 and 0.189, considerably
GaussianPro
Ours
RGB Image
Cp
Seq. 03
Seq. 04
Fig. 14. The visualization of rendered depths. Our depth maps show smoother
surfaces, proving the superior geometric quality of reconstructed maps.
Ours w/o sky separation
Ours w/ sky separation
Fig. 15. The effect of sky separation. Floaters are marked using yellow circles.
lower than that of 3DGS [12] (1.202 CD and 1.054 MHD).
We attribute these advantages to geometry-aware densification
for dense structure generation, selective separation for precise
gaussian pruning, and the multi-view regularization for opti-
mization with reduced ambiguity. Then, these well-distributed
gaussians enable accurate scene representation, consequently
resulting in superior rendering quality compared with previous
approaches [12], [33], [35]. Moreover, we conduct experiments
on NTU4DRadLM across diverse scenes in Tab. VI, showing
that our method maintains obvious advantages in map quality
and rendering fidelity relative to other algorithms.
As shown in Fig. 12, our method achieves a higher-fidelity
rendering of environmental details. Especially, it better recon-
structs building extending beyond the radar’s FOV and ground
with weak initialization, because of the proposed densification
strategy. To intuitively demonstrate reconstruction results, we
present the original input and optimized radar maps in Fig. 13.
Although 3DGS recovers missing structures to some extent,
the resulting geometry remains a sparse distribution with fuzzy
artifacts. In comparison, our approach reconstructs maps with
clearer structures and higher point density using radar-specific
optimization, while ground surfaces are also restored by depth-
assisted completion. Furthermore, we show the rendered depth
maps in Fig. 14, which demonstrates that our method produces
smoother surfaces and higher local consistency.
D. Ablation Studies
In this section, we conduct ablation studies through progres-
sively adding modules and replacing core strategies to evaluate

<!-- page 13 -->
UNDER REVIEW
13
w/o Normal Constraint
w/ Normal Constraint
RGB Image
w/o Depth Constraint
w/ Depth Constraint
RGB Image
Fig. 16. Effects of normal and depth constraints. Results display that the former smooth surfaces, while the latter enhance the spatial accuracy of structures.
TABLE VII
THE EFFECTS OF DIFFERENT SELF-SUPERVISED LOSS TERMS.
Lcd
Lco
Lga
Ltg
Lfc
Lcm
Cluster
Mean trel
Mean rrel
✓
✓
0.219
0.332
✓
✓
✓
0.190
0.235
✓
✓
✓
✓
0.124
0.175
✓
✓
✓
✓
✓
0.078
0.091
✓
✓
✓
✓
✓
✓
0.056
0.068
✓
✓
✓
✓
✓
✓
0.068
0.108
✓
✓
✓
✓
✓
✓
✓
0.053
0.053
TABLE VIII
THE EFFECTS OF OUR SELF-SUPERVISED STRATEGY SELECTION.
Strategy
Mean trel
Mean rrel
Lcd: w/o Cluster Weighting
0.063
0.085
Lco: Polar Coordinate →Cartesian Coordinate
0.083
0.086
Ltg: w/o Selection Mechanism
0.088
0.106
Lfc: w/ Neighbor as Negative Sample
0.064
0.139
Lcm: Acc. Consistency →Vel. Consistency
0.066
0.068
Ours
0.053
0.053
TABLE IX
THE EFFECTS OF GAUSSIAN UPDATE AND REGULARIZATION STRATEGY.
Strategy
Geometric Quality
Rendering Quality
F-Score↑CD↓MHD↓PSNR↑SSIM↑LPIPS↓
Baseline
0.401
1.063 0.583
17.546 0.525
0.547
+ Ground Completion
0.422
0.937 0.506
17.885 0.531
0.540
+ NB Pruning
0.465
0.898 0.377
18.064 0.538
0.535
+ GA Densification
0.511
0.837 0.270
18.565 0.542
0.527
+ Sky Decoupling
0.528
0.787 0.227
18.582 0.542
0.526
+ Depth & Normal Loss
0.542
0.750 0.186
18.738 0.549
0.513
+ Multi-view Constraint
0.558
0.725 0.159
19.406 0.562
0.492
their contributions and the rationality of our framework design.
1) Self-supervised Signals: In Tab. VII, we perform a series
of experiments on the VoD with extremely sparse points. Lcd
and Lco are cluster-based distance and column occupancy loss.
Ltg and Lga stand for teacher guidance and GMM alignment
loss, while Lfc and Lcm represent feature-level and temporal
constraints. At first, the use of Lco leads to reduced odometry
errors, especially in rrel, as its larger-scale column comparison
with occupancy representation is more sensitive to rotational
discrepancies and can ease noise from point-to-point constraint
Lcd. Then, by incorporating soft labels derived from geometry-
based pose refinement, Ltg guides the network to predict more
reasonable results and avoid getting stuck in local optima due
to rigid geometric constraints. Using Lfc further reduces trel
and rrel, validating that the discriminative features derived by
contrastive learning of farthest and nearest points make model
easier to identify inter-frame relationships. Besides, the results
in the 5th, 6th and 7th rows demonstrate that the acceleration
constraint Lcm is crucial for suppressing abnormal poses, and
that the cluster cues are critical for reducing feature correlation
between unrelated instances.
2) Self-supervised Strategy Selection: We show the impact
of various self-supervised strategies in Tab. VIII. The 1st row
TABLE X
THE EFFECTS OF VIEW COUNT ON MULTI-VIEW REGULARIZATION.
Number of views F-Score↑
CD↓
MHD↓PSNR↑SSIM↑LPIPS↓
Two
0.549
0.736
0.171
19.121
0.554
0.509
Three
0.553
0.729
0.166
19.291
0.559
0.499
Ours (Five)
0.558
0.725
0.159
19.406
0.562
0.492
TABLE XI
GEOMETRIC AND RENDERING QUALITY EVALUATION ON THE GARDEN
SEQUENCE OF NTU4DRADLM BASED ON THERMAL IMAGES. *
INDICATES NOT USING NORMAL CONSTRAINTS.
Method
F-Score↑
CD↓
MHD↓
PSNR↑
SSIM↑
LPIPS↓
GaussianPro
0.633
0.590
0.080
22.355
0.680
0.478
2DGS
0.295
0.830
0.224
18.456
0.634
0.556
3DGS
0.572
0.561
0.102
21.984
0.676
0.491
Ours*
0.666
0.445
0.058
25.875
0.715
0.436
shows that assigning higher weights to larger clusters is effec-
tive in Lcd. Polar coordinates exhibit superiority over cartesian
coordinates when transforming points into BEV for column
comparison in Lco. Besides, without a selection mechanism in
Ltg, suboptimal soft labels would misguide the model, causing
performance degradation. Using the farthest point rather than
the neighbor with similar features as a negative sample in Lfc
is more reasonable, as demonstrated in the 4th row. Finally,
the constant velocity assumption adversely affects the network
learning, due to frequent speed variations in real-world driving.
3) Gaussian Update and Regularization: We conduct abla-
tion studies on the NTU4DRadLM dataset with various scenes
to evaluate map optimization. As illustrated in Tab. IX, ground
completion improves geometric quality because the radar map
lacks ground structure initially. The improvement in ground
rendering is also evident in 3rd and 4th rows of Fig. 12. Since
neighborhood-based (NB) pruning preserves useful gaussians
for splitting and geometry-aware (GA) densification produces
more gaussians in structurally appropriate positions to com-
plete voids, all metrics further achieve a consistent boost. The
results in the 5th row of Tab. IX exhibit that the sky decoupling
strategy improves map quality by using a sky mask to remove
floaters. A qualitative example of this enhancement is shown
in Fig. 15. The 6th row of Tab. IX and Fig. 16 demonstrate that
geometric knowledge of visual fundamental models serves as
effective constraints. Finally, we mitigate the spatial ambiguity
of a single view through multi-view collaborative constraints
in overlap regions, resulting in the best performance. In Tab. X,
we further explore the effect of view count. The results prove
that using constraints from more perspectives facilitates proper
convergence of gaussian attributes, leading to better results.
E. Application Potential with Thermal Image
While RGB images are available in most environments, they
degrade in low-light nights. In contrast, thermal images exhibit

<!-- page 14 -->
UNDER REVIEW
14
3DGS
GaussianPro
2DGS
Inital Rendered Image
Depth
Value
(m)
Rendering Depth
''Deficient Depth''
''Ambiguous''
''Clear''
''Complete Depth''
Near
Far
Ground-truth Thermal Image
Ours
Fig. 17. Comparisons of thermal images rendered in the garden sequence of NTU4DRadLM. We highlight the rendered depth of nearby and distant objects.
TABLE XII
4D RADAR ODOMETRY EXPERIMENT RESULTS ON SELF-COLLECTED CAMPUS DATASET.
Method
Gym (300m)
Building2 (370m)
Straight (205m)
Parking2 (234m)
Mean
RPE(m) RPE(°) ATE(m) RPE(m) RPE(°) ATE(m) RPE(m) RPE(°) ATE(m) RPE(m) RPE(°) ATE(m) RPE(m) RPE(°) ATE(m)
ICP
0.886
1.913
7.677
0.903
2.149
10.330
1.414
2.612
13.945
0.770
1.939
5.595
0.993
2.153
9.387
NDT
0.785
3.715
4.751
2.161
8.463
24.089
5.617
32.763
26.836
2.046
9.028
20.961
2.652
13.492
19.159
GICP
3.337
5.031
26.357
3.721
6.532
22.384
5.754
10.316
16.307
3.983
5.893
16.490
4.199
6.943
20.384
APDGICP
1.608
2.914
11.619
2.196
3.932
17.340
5.049
6.047
29.351
2.359
3.082
12.517
2.803
3.994
17.707
RaFlow
5.623
9.010
50.921
5.012
10.804
48.504
5.112
2.569
46.497
4.666
13.803
32.208
5.103
9.047
44.533
Ours
0.440
1.878
2.316
0.477
1.778
7.166
0.542
1.913
4.882
0.626
2.728
2.663
0.521
2.074
4.257
TABLE XIII
THE PERFORMANCE OF MAP OPTIMIZATION AND IMAGE RENDERING ON SELF-COLLECTED CAMPUS DATASET.
Method
Gym
Building2
Straight
Parking2
Mean
F-Score↑
CD↓
MHD↓
F-Score↑
CD↓
MHD↓
F-Score↑
CD↓
MHD↓
F-Score↑
CD↓
MHD↓
F-Score↑
CD↓
MHD↓
GaussianPro
0.276
5.973
0.673
0.230
5.733
1.164
0.297
2.290
0.366
0.349
4.902
0.678
0.288
4.724
0.720
2DGS
0.118
1.669
0.959
0.088
1.835
1.262
0.144
1.467
0.692
0.181
1.370
1.824
0.133
1.585
1.184
3DGS
0.267
1.210
0.422
0.198
3.531
1.012
0.298
0.898
0.319
0.389
0.938
0.274
0.288
1.644
0.507
Ours
0.341
0.974
0.377
0.283
1.577
0.586
0.362
0.818
0.328
0.484
0.854
0.173
0.367
1.056
0.366
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
PSNR↑
SSIM↑LPIPS↓
GaussianPro
20.081
0.584
0.459
20.970
0.671
0.437
19.461
0.498
0.512
21.122
0.683
0.346
20.408
0.609
0.439
2DGS
18.535
0.566
0.548
17.127
0.606
0.532
18.695
0.491
0.600
19.093
0.647
0.487
18.362
0.577
0.542
3DGS
18.467
0.557
0.516
18.911
0.640
0.474
19.131
0.509
0.547
21.141
0.705
0.383
19.412
0.602
0.480
Ours
21.273
0.602
0.468
23.177
0.699
0.408
20.756
0.524
0.519
22.853
0.724
0.366
22.015
0.637
0.440
greater stability across diverse scenarios. To reveal the infrared
potential and prove the adaptability of our method to different
visual sensors, we adopt thermal images for map optimization
in this part. As NTU4DRadLM does not provide thermal data
for Cp and Nyl sequences, we evaluate solely on the Garden in
Tab. XI, where our approach omits normal constraints owing
to Metric3D’s [51] inability to estimate normals from thermal
images. Nonetheless, the results indicate that our method still
outperforms GaussianPro by 3.520 PSNR and achieves advan-
tages across all metrics of geometric quality, demonstrating its
generalization and prospects for radar-thermal reconstruction.
In Fig. 17, we illustrate the infrared rendering results, and our
framework can better restore the details of ground and building
while getting reasonable depths for nearby and distant objects.
F. More Experiments on Self-collected Campus Dataset
To validate our method’s effectiveness in real-world scenar-
ios beyond public datasets and in nighttime scene reconstruc-
tion not supported by NTU4DRadLM [9], we conduct exten-
sive experiments based on our self-collected dataset acquired
from a handheld platform, thereby providing a more thorough
validation of our algorithm alongside the results presented in
Sec. V-C.
1) 4D Radar Odometry: We present the evaluation results
of our self-collected dataset in Tab. XII. It indicates that our
method achieves state-of-the-art performance with translation
and rotation RPEs of 0.521 and 2.074, respectively. In contrast
to the results in Tab. IV, both APDGICP and GICP exhibit sub-
stantially inferior performance to ICP. We attribute this to the
sparser points in our radar data compared to NTU4DRadLM,
which significantly influences the effectiveness of GICP-based
methods due to their dependence on distinct local structures.
Furthermore, we show the trajectories of odometry in Fig. 18.
As a result of unstable inter-frame registration caused by low-
quality radar points, the paths of GICP-based methods display
significant fluctuations. By comparison, benefiting from radar-
specific designs, namely the cluster-aware matching for robust
data association and multi-level constraints for reliable pose
prediction, our method obtains smoother trajectories in Fig. 18.
Meanwhile, by using a cluster-weighted point-to-point loss and
enforcing large-scale consistency through column comparison,
our proposed algorithm outperforms ICP in both quantitative
metrics (4.257 vs. 9.387 ATE) and visual comparisons.

<!-- page 15 -->
UNDER REVIEW
15
Ground Truth
Ours
GICP
APDGICP
ICP
Gym
Building2
Parking2
Straight
Canteen*
Fig. 18. Trajectory comparison of various methods using different colors on our self-collected campus dataset. * indicates the sequence from the training set.
Raw Radar Map
3DGS 
Ours
Straight
Parking2
GaussianPro
Ours
Ground Truth
Gym
Building2
Fig. 19. Comparison of optimized radar maps and rendered images on self-collected dataset. Yellow circles mark the defective rendering areas in GaussianPro.
Ours
3DGS
GaussianPro
2DGS
Ground-truth Thermal Image
RGB Image
Canteen-night
Library-night
Fig. 20. Comparison of thermal images rendered from the night sequences of our self-collected dataset. We also display the corresponding RGB images.
TABLE XIV
MEAN GEOMETRIC AND RENDERING QUALITY OVER NIGHT SEQUENCES
IN THE SELF-COLLECTED DATASET BASED ON THERMAL IMAGES. *
INDICATES NOT USING NORMAL CONSTRAINTS.
Method
F-Score↑
CD↓
MHD↓
PSNR↑
SSIM↑
LPIPS↓
GaussianPro
0.171
5.052
4.965
23.650
0.756
0.418
2DGS
0.113
1.764
1.849
22.936
0.762
0.463
3DGS
0.232
1.613
1.111
23.026
0.765
0.433
Ours*
0.264
1.322
0.559
25.993
0.792
0.414
2) Map Optimization: In Tab. XIII, we present a compre-
hensive comparison of both geometric reconstruction accuracy
and novel view rendering quality across different approaches.
Results show that GaussianPro achieves competitive rendering
quality but poorer geometric precision, with a chamfer distance
(CD) three times greater than 3DGS. This may be caused by
inaccurate propagation depth of the newly added gaussians. In
contrast, our algorithm achieves a better balance between the
geometry and rendering quality, with a CD merely 64% that of
3DGS. For a more intuitive comparison, we further visualize
the optimized local radar maps and rendered images in Fig. 19.
Benefiting from ground completion and geometry-aware den-
sification that generates new gaussians with reasonable spatial
positions, our algorithm not only reconstructs maps with more
complete and denser structures but also renders images with
higher clarity.
3) Radar-Thermal Reconstruction: As the NTU4DRadLM
contains only daytime sequences, we collect nighttime data for
our campus dataset and conduct more experiments to explore
the potential for radar-thermal reconstruction under low light.
As displayed in Tab. XIV, our algorithm achieves an MHD
of 0.559 for geometric quality, about half that of 3DGS, and
surpasses GaussianPro by 2.343 in PSNR for image rendering.

<!-- page 16 -->
UNDER REVIEW
16
We further visualize rendering results in Fig. 20, where our
method also generates novel view thermal images with clearer
details than other approaches, proving the effectiveness of the
proposed modules across different image modalities.
VI. CONCLUSIONS
In this article, we introduce the first 4D radar-based frame-
work, called Super4DR, which contains learning-based odom-
etry estimation and map optimization while explicitly account-
ing for radar-specific characteristics. It can predict poses based
on noisy and sparse radar points, reconstruct dense structures
from low-quality radar maps, and render multi-modal images
with depth and normal maps. For odometry estimation, we first
propose a cluster-aware network, which leverages object-level
cues to perform robust inter-frame matching and is trained by
multi-level self-supervised losses for label-free learning. Un-
der consideration of radar point distribution, cluster-weighted
distance and column occupancy losses are adopted to constrain
geometric consistency. A teacher guidance loss then uses soft
labels as additional supervision to perform knowledge transfer.
The feature contrast and temporal constraint are also applied to
enable effective matching and improve trajectory smoothness.
For map optimization, we propose a gaussian-based optimizer
that views 3D gaussians as an intermediate map representation.
It contains a depth-guided completion module for the ground
recovery, combined with a radar-specific densification module
aimed at improving map integrity. The selective separation and
multi-view regularization are further used to decrease floaters,
avoid mistaken deletion and accelerate optimization. Extensive
experiments on public and self-collected datasets show that our
method achieves superior results in various aspects, including
odometry accuracy, map geometric quality and rendering detail
fidelity. We further explore the potential of 4D radar-thermal
reconstruction in poor illumination and verify the applicability
of our method across different visual sensors.
REFERENCES
[1] J. Zhang, H. Zhuge, Z. Wu, G. Peng, M. Wen, Y. Liu, and D. Wang,
“4dradarslam: A 4d imaging radar slam system for large-scale environ-
ments based on pose graph optimization,” in 2023 IEEE International
Conference on Robotics and Automation (ICRA), pp. 8333–8340, IEEE,
2023.
[2] X. Wu, Y. Chen, Z. Li, Z. Hong, and L. Hu, “Efear-4d: Ego-velocity
filtering for efficient and accurate 4d radar odometry,” IEEE Robotics
and Automation Letters, vol. 9, no. 11, pp. 9828–9835, 2024.
[3] D. C. Herraez, M. Zeller, L. Chang, I. Vizzo, M. Heidingsfeld, and
C. Stachniss, “Radar-only odometry and mapping for autonomous
vehicles,” in 2024 IEEE International Conference on Robotics and
Automation (ICRA), pp. 10275–10282, 2024.
[4] S. Kim, J. Seok, J. Lee, and K. Jo, “Radar4motion: Imu-free 4d radar
odometry with robust dynamic filtering and rcs-weighted matching,”
IEEE Transactions on Intelligent Vehicles, pp. 1–11, 2024.
[5] S. Lu, G. Zhuo, L. Xiong, X. Zhu, L. Zheng, Z. He, M. Zhou, X. Lu, and
J. Bai, “Efficient deep-learning 4d automotive radar odometry method,”
IEEE Transactions on Intelligent Vehicles, 2023.
[6] G. Zhuoins, S. Lu, L. Xiong, H. Zhouins, L. Zheng, and M. Zhou,
“4drvo-net: Deep 4d radar–visual odometry using multi-modal and
multi-scale adaptive fusion,” IEEE Transactions on Intelligent Vehicles,
2023.
[7] H. Zhou, S. Lu, and G. Zhuo, “Self-supervised 4-d radar odometry for
autonomous vehicles,” in 2023 IEEE 26th International Conference on
Intelligent Transportation Systems (ITSC), pp. 764–769, 2023.
[8] F. Ding, Z. Pan, Y. Deng, J. Deng, and C. X. Lu, “Self-supervised
scene flow estimation with 4-d automotive radar,” IEEE Robotics and
Automation Letters, vol. 7, no. 3, pp. 8233–8240, 2022.
[9] J. Zhang, H. Zhuge, Y. Liu, G. Peng, Z. Wu, H. Zhang, Q. Lyu, H. Li,
C. Zhao, D. Kircali, et al., “Ntu4dradlm: 4d radar-centric multi-modal
dataset for localization and mapping,” in 2023 IEEE 26th International
Conference on Intelligent Transportation Systems (ITSC), pp. 4291–
4296, IEEE, 2023.
[10] R. Zhang, D. Xue, Y. Wang, R. Geng, and F. Gao, “Towards dense
and accurate radar perception via efficient cross-modal diffusion model,”
IEEE Robotics and Automation Letters, vol. 9, no. 9, pp. 7429–7436,
2024.
[11] K. Luan, C. Shi, N. Wang, Y. Cheng, H. Lu, and X. Chen, “Diffusion-
based point cloud super-resolution for mmwave radar data,” in 2024
IEEE International Conference on Robotics and Automation (ICRA),
pp. 11171–11177, 2024.
[12] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.,” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[13] P. J. Besl and N. D. McKay, “Method for registration of 3-d shapes,”
in Sensor fusion IV: control paradigms and data structures, vol. 1611,
pp. 586–606, Spie, 1992.
[14] A. Segal, D. Haehnel, and S. Thrun, “Generalized-icp.,” in Robotics:
science and systems, vol. 2, p. 435, Seattle, WA, 2009.
[15] I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, and
C. Stachniss, “Kiss-icp: In defense of point-to-point icp–simple, accu-
rate, and robust registration if done the right way,” IEEE Robotics and
Automation Letters, vol. 8, no. 2, pp. 1029–1036, 2023.
[16] P. Biber and W. Straßer, “The normal distributions transform: A new
approach to laser scan matching,” in Proceedings 2003 IEEE/RSJ Inter-
national Conference on Intelligent Robots and Systems (IROS 2003)(Cat.
No. 03CH37453), vol. 3, pp. 2743–2748, IEEE, 2003.
[17] J. Zhang, S. Singh, et al., “Loam: Lidar odometry and mapping in real-
time.,” in Robotics: Science and systems, vol. 2, pp. 1–9, Berkeley, CA,
2014.
[18] T. Shan and B. Englot, “Lego-loam: Lightweight and ground-optimized
lidar odometry and mapping on variable terrain,” in 2018 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS),
pp. 4758–4765, IEEE, 2018.
[19] H. Wang, C. Wang, C.-L. Chen, and L. Xie, “F-loam: Fast lidar odometry
and mapping,” in 2021 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pp. 4390–4396, IEEE, 2021.
[20] G. Wang, X. Wu, Z. Liu, and H. Wang, “Pwclo-net: Deep lidar odometry
in 3d point clouds using hierarchical embedding mask optimization,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 15910–15919, 2021.
[21] G. Wang, X. Wu, S. Jiang, Z. Liu, and H. Wang, “Efficient 3d deep
lidar odometry,” IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 45, no. 5, pp. 5749–5765, 2023.
[22] J. Liu, G. Wang, C. Jiang, Z. Liu, and H. Wang, “Translo: A window-
based masked point transformer framework for large-scale lidar odom-
etry,” in Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 37, pp. 1683–1691, 2023.
[23] H. Zhang, G. Wang, X. Wu, C. Xu, M. Ding, M. Tomizuka, W. Zhan, and
H. Wang, “Dslo: Deep sequence lidar odometry based on inconsistent
spatio-temporal propagation,” in 2024 IEEE/RSJ International Confer-
ence on Intelligent Robots and Systems (IROS), pp. 10672–10677, 2024.
[24] J. Nubert, S. Khattak, and M. Hutter, “Self-supervised learning of
lidar odometry for robotic applications,” in 2021 IEEE International
Conference on Robotics and Automation (ICRA), pp. 9601–9607, 2021.
[25] B. Zhou, Y. Tu, Z. Jin, C. Xu, and H. Kong, “Hpplo-net: Unsupervised
lidar odometry using a hierarchical point-to-plane solver,” IEEE Trans-
actions on Intelligent Vehicles, vol. 9, no. 1, pp. 2727–2739, 2024.
[26] Y. Xu, J. Lin, J. Shi, G. Zhang, X. Wang, and H. Li, “Robust self-
supervised lidar odometry via representative structure discovery and 3d
inherent error modeling,” IEEE Robotics and Automation Letters, vol. 7,
no. 2, pp. 1651–1658, 2022.
[27] D. Adolfsson, M. Magnusson, A. Alhashimi, A. J. Lilienthal, and H. An-
dreasson, “Cfear radarodometry - conservative filtering for efficient and
accurate radar odometry,” in 2021 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pp. 5462–5469, 2021.
[28] H. Lim, K. Han, G. Shin, G. Kim, S. Hong, and H. Myung, “Orora:
Outlier-robust radar odometry,” in 2023 IEEE International Conference
on Robotics and Automation (ICRA), pp. 2046–2053, 2023.
[29] R. Zhang, Y. Zhang, D. Fu, and K. Liu, “Scan denoising and normal
distribution transform for accurate radar odometry and positioning,”

<!-- page 17 -->
UNDER REVIEW
17
IEEE Robotics and Automation Letters, vol. 8, no. 3, pp. 1199–1206,
2023.
[30] F. Ding, A. Palffy, D. M. Gavrila, and C. X. Lu, “Hidden gems: 4d radar
scene flow learning using cross-modal supervision,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 9340–9349, 2023.
[31] Z. Li, Y. Cui, N. Huang, C. Pang, and Z. Fang, “Cao-ronet: A robust
4d radar odometry with exploring more information from low-quality
points,” arXiv preprint arXiv:2503.01438, 2025.
[32] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[33] K. Cheng, X. Long, K. Yang, Y. Yao, W. Yin, Y. Ma, W. Wang, and
X. Chen, “Gaussianpro: 3d gaussian splatting with progressive propa-
gation,” in Forty-first International Conference on Machine Learning,
2024.
[34] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
“Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with
global-local depth normalization,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pp. 20775–
20785, 2024.
[35] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
conference papers, pp. 1–11, 2024.
[36] R. Xiao, W. Liu, Y. Chen, and L. Hu, “Liv-gs: Lidar-vision integration
for 3d gaussian splatting slam in outdoor environments,” IEEE Robotics
and Automation Letters, vol. 10, no. 1, pp. 421–428, 2025.
[37] K. Wu, K. Zhang, Z. Zhang, M. Tie, S. Yuan, J. Zhao, Z. Gan, and
W. Ding, “Hgs-mapping: Online dense mapping using hybrid gaussian
representation in urban scenes,” IEEE Robotics and Automation Letters,
vol. 9, no. 11, pp. 9573–9580, 2024.
[38] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Driv-
inggaussian: Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 21634–21643, 2024.
[39] X. Liu, Z. Liu, F. Kong, and F. Zhang, “Large-scale lidar consistent
mapping using hierarchical lidar bundle adjustment,” IEEE Robotics and
Automation Letters, vol. 8, no. 3, pp. 1523–1530, 2023.
[40] C. Pang, Z. Shen, R. Yuan, C. Xu, and Z. Fang, “Lm-mapping: Large-
scale and multi-session point cloud consistent mapping,” IEEE Robotics
and Automation Letters, vol. 9, no. 12, pp. 10866–10873, 2024.
[41] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “Density-based spatial
clustering of applications with noise,” in Int. Conf. knowledge discovery
and data mining, vol. 240, 1996.
[42] X. Liu, C. R. Qi, and L. J. Guibas, “Flownet3d: Learning scene flow
in 3d point clouds,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 529–537, 2019.
[43] W. Wu, Z. Y. Wang, Z. Li, W. Liu, and L. Fuxin, “Pointpwc-net: Cost
volume on point clouds for (self-) supervised scene flow estimation,”
in Computer Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part V 16, pp. 88–107, Springer,
2020.
[44] K. Cho, B. Van Merri¨enboer, C. Gulcehre, D. Bahdanau, F. Bougares,
H. Schwenk, and Y. Bengio, “Learning phrase representations using
rnn encoder-decoder for statistical machine translation,” arXiv preprint
arXiv:1406.1078, 2014.
[45] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural
network,” arXiv preprint arXiv:1503.02531, 2015.
[46] A. Mishra and D. Marr, “Apprentice: Using knowledge distillation
techniques to improve low-precision network accuracy,” arXiv preprint
arXiv:1711.05852, 2017.
[47] Z. Fang, J. Wang, L. Wang, L. Zhang, Y. Yang, and Z. Liu, “Seed:
Self-supervised distillation for visual representation,” arXiv preprint
arXiv:2101.04731, 2021.
[48] P. He, P. Emami, S. Ranka, and A. Rangarajan, “Self-supervised robust
scene flow estimation via the alignment of probability density functions,”
in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36,
pp. 861–869, 2022.
[49] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao,
“Depth anything v2,” Advances in Neural Information Processing Sys-
tems, vol. 37, pp. 21875–21911, 2024.
[50] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar,
“Masked-attention mask transformer for universal image segmentation,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 1290–1299, 2022.
[51] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang,
G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular
geometric foundation model for zero-shot metric depth and surface
normal estimation,” IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 46, no. 12, pp. 10579–10596, 2024.
[52] A. Palffy, E. Pool, S. Baratam, J. F. Kooij, and D. M. Gavrila, “Multi-
class road user detection with 3+ 1d radar in the view-of-delft dataset,”
IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4961–4968,
2022.
[53] W. Xu and F. Zhang, “Fast-lio: A fast, robust lidar-inertial odometry
package by tightly-coupled iterated kalman filter,” IEEE Robotics and
Automation Letters, vol. 6, no. 2, pp. 3317–3324, 2021.
[54] Q. Li, S. Chen, C. Wang, X. Li, C. Wen, M. Cheng, and J. Li, “Lo-
net: Deep real-time lidar odometry,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 8473–
8482, 2019.
[55] S. Lu, H. Zhou, and G. Zhuo, “Dnoi-4dro: Deep 4d radar odome-
try with differentiable neural-optimization iterations,” arXiv preprint
arXiv:2505.12310, 2025.
