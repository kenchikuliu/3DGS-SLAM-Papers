<!-- page 1 -->
DriveSplat: Unified Neural Gaussian Reconstruction for
Dynamic Driving Scenes
Cong Wang1,2, Ruiqi Song1,3, Wei Tian4, Chenming Zhang5, Lingxi Li6,
Long Chen1,3*
1the State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of
Automation, Chinese Academy of Sciences, Beijing, China.
2Zhongguancun Academy, Beijing, China.
3Waytous, Beijing, China.
4School of Automotive Studies, Tongji University, Shanghai, China.
5Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an, China.
6School of Electrical and Computer Engineering, Purdue University, Indianapolis, USA.
*Corresponding author(s). E-mail(s): long.chen@ia.ac.cn;
Abstract
Reconstructing large-scale dynamic driving scenes remains challenging due to the coexistence of static
environments with extreme depth variation and diverse dynamic actors exhibiting complex motions.
Existing Gaussian Splatting based methods have primarily focused on limited-scale or object-centric
settings, and their applicability to large-scale dynamic driving scenes remains underexplored, par-
ticularly in the presence of extreme scale variation and non-rigid motions. In this work, we propose
DriveSplat, a unified neural Gaussian framework for reconstructing dynamic driving scenes within
a unified Gaussian-based representation. For static backgrounds, we introduce a scene-aware learn-
able level-of-detail (LOD) modeling strategy that explicitly accounts for near, intermediate, and
far depth ranges in driving environments, enabling adaptive multi-scale Gaussian allocation. For
dynamic actors, we use an object-centric formulation with neural Gaussian primitives, modeling
motion through a global rigid transformation and handling non-rigid dynamics via a two-stage defor-
mation that first adjusts anchors and subsequently updates the Gaussians. To further regularize
the optimization, we incorporate dense depth and surface normal priors from pre-trained models as
auxiliary supervision. Extensive experiments on the Waymo and KITTI benchmarks demonstrate
that DriveSplat achieves state-of-the-art performance in novel-view synthesis while producing tem-
porally stable and geometrically consistent reconstructions of dynamic driving scenes. Project page:
https://physwm.github.io/drivesplat.
Keywords: 3D Reconstruction, Novel-view Synthesis, Gaussian Splatting, Driving Scenario, Geometry Priors
1 Introduction
High-fidelity 3D simulation of autonomous driv-
ing scenarios plays a crucial role in closed-loop
testing and validation of autonomous driving sys-
tems. By enabling the flexible construction of
complex traffic environments, such simulations
also provide valuable training data for perception
1
arXiv:2508.15376v4  [cs.CV]  19 Mar 2026

<!-- page 2 -->
StreetGS
Ours
Reconstructed 3D Scene
RGB
Depth
Normal
PSNR: 30.57 dB
PSNR: 36.67 dB
Viewpoint
Viewpoint
Fig. 1 Comparison with StreetGS. StreetGS generates an excessive number of redundant Gaussians (yellow circles)
in the reconstructed 3D scene. The right panel presents the rendered image, depth and normal map from a novel view (red
dots).
and decision-making models. Compared to tradi-
tional solutions such as oblique photography or
manually designed simulators, 3D reconstruction
and novel view synthesis (NVS) methods offer a
more scalable and realistic alternative by recov-
ering 3D scenes directly from multi-view image
observations [1–3].
Recent advances in neural scene represen-
tations have significantly accelerated progress
in this direction [4]. Neural Radiance Fields
(NeRF) [5, 6] introduce an implicit volumet-
ric representation that achieves high-fidelity view
synthesis through differentiable ray sampling [7,
8]. Building upon this paradigm, 3D Gaussian
Splatting (3D-GS) [9] explicitly represents scenes
using anisotropic Gaussian primitives, substan-
tially improving rendering efficiency while main-
taining competitive visual quality [10]. While
these methods [5, 9] demonstrate impressive per-
formance in object-centric and small-scale indoor
environments, their scalability to large outdoor
scenes remains limited. Subsequent works [11–14]
improve robustness to viewpoint variations and
introduce hierarchical or neural Gaussian repre-
sentations to better balance reconstruction quality
and computational efficiency.
Despite this progress, 3D reconstruction in
autonomous driving scenarios remains particu-
larly challenging [15–20]. Driving scenes are char-
acterized by large spatial extents, strong scale
variation, and the presence of dynamic actors such
as vehicles, pedestrians, and cyclists. To address
these challenges, recent methods such as Street-
Gaussian [19] and DrivingGaussian [20] adopt
dynamic–static decoupling strategies, reconstruct-
ing static backgrounds and dynamic foreground
actors
separately.
Follow-up
studies
further
explore non-rigid actor modeling [21], trajec-
tory refinement [22], and motion-aware decoupling
using optical flow or semantic cues [23, 24]. How-
ever, existing representations are not explicitly
designed to handle the scale, viewpoint, and geo-
metric characteristics of driving scenes, which
manifests as limited robustness under novel view-
points and suboptimal geometric consistency.
A key limitation of current Gaussian-based
driving scene reconstruction methods lies in their
treatment of background geometry in large-scale,
multi-depth environments. Driving scenes exhibit
significant scale variation, where near-range struc-
tures demand fine-grained modeling while distant
regions primarily require coherent global geome-
try. However, existing approaches typically rely on
single-scale Gaussian densification strategies that
improve rendering quality at training viewpoints
but tend to introduce redundant primitives in
distant regions, degrading visual clarity and geo-
metric consistency under novel viewpoints. This
issue is further exacerbated in dynamic driving
scenarios, where background reconstruction must
remain stable while accommodating the pres-
ence of moving and non-rigid actors. Moreover,
depth supervision from LiDAR, commonly used
in driving datasets, provides sparse and uneven
constraints [25] that are insufficient for supervis-
ing distant structures such as tall buildings and
large-scale facades. Surface geometry, particularly
normal consistency across scales and viewpoints,
2

<!-- page 3 -->
is also largely underexplored in existing methods,
leading to reconstructions that appear visually
plausible yet lack geometric fidelity (see Fig. 1).
To
address
these
challenges,
we
propose
DriveSplat, a unified neural Gaussian repre-
sentation framework for robust reconstruction of
dynamic driving scenes. Our approach models
static backgrounds and dynamic scene elements
within a single optimization framework, avoid-
ing ad-hoc design choices tailored to individual
components. For background reconstruction, we
introduce a scene-aware learnable level-of-detail
(LOD) modeling strategy that explicitly accounts
for the characteristic near-, mid-, and far-range
structure of driving environments. By adaptively
allocating multi-scale Gaussian representations,
the proposed method enhances the reconstruction
of fine-grained geometry in close-range regions
while preserving global consistency. Within the
same framework, we further extend the repre-
sentation to dynamic actors. Instead of directly
entangling motion with per-Gaussian parameters,
we model non-rigid dynamics via an anchor-level
deformation mechanism that captures temporal
motion through a continuous deformation field
and propagates it coherently to neural Gaussian
attributes. This design enables stable reconstruc-
tion of non-rigid actors while remaining fully
compatible with the unified background represen-
tation. To further improve geometric accuracy, we
incorporate geometry-aware regularization using
dense depth and surface normal priors predicted
by pretrained monocular models, which effectively
enhances both rendering quality and surface con-
sistency. Extensive experiments on the Waymo
and KITTI benchmarks demonstrate that our
method achieves state-of-the-art performance in
novel view synthesis for large-scale driving scenes.
In summary, our main contributions are as
follows:
• We propose a unified representation for large-
scale driving scenes that models both static
backgrounds and dynamic actors using neu-
ral Gaussians, enabling consistent scene recon-
struction under a unified Gaussian-based repre-
sentation.
• For background reconstruction, we introduce
a scene-aware learnable level-of-detail (LOD)
modeling strategy tailored to driving environ-
ments, which explicitly accounts for near-, mid-,
and far-range scene structures and adaptively
allocates multi-scale Gaussian representations
to enhance close-range geometric fidelity.
• For dynamic actors, we design an anchor-level
non-rigid deformation mechanism that models
temporal motion through a continuous defor-
mation field and propagates it coherently to
neural Gaussian parameters, allowing stable
reconstruction of non-rigid dynamics within the
same representation framework.
• We further incorporate geometry-aware reg-
ularization
using
dense
depth
and
surface
normal priors to improve geometric consis-
tency. Extensive experiments on the Waymo
and KITTI benchmarks demonstrate that our
method achieves state-of-the-art performance in
novel view synthesis for driving scenes.
2 Related Works
2.1 Large-scale Scene
Reconstruction
The naive NeRF [5] struggles with large-scale
scenes due to blurry close-ups and jagged distant
edges. Improvements include Mip-NeRF [6] with
multi-scale IPE, NeRF-W [26] for lighting vari-
ations, and Block-NeRF [27], which trains local
blocks separately. Recently, 3D-GS-based meth-
ods [2, 9, 11, 28, 29] have achieved remarkable
breakthroughs in both reconstruction speed and
quality. The initial Gaussian Splatting [9] is specif-
ically designed for scenes with an object-centered
view, and subsequent efforts have extended it
to large-scale scenes. Neural Gaussian [12, 13]
incorporates the advantage of Gaussian Splatting
and neural fields, and achieves real-time rendering
with robust viewpoint invariance. Hierarchical-GS
[15] introduces a hierarchical structure for driving
scenes to optimize the effect of real-time recon-
struction and combines the blocking strategy to
select different levels. For the urban scene recon-
struction, some methods
[13, 30–32] propose to
divide point clouds into cells, and introduce the
Level-of-Details to optimize reconstruction effi-
ciency and detail performance. Above methods
neglect dynamic object optimization, while our
approach improves reconstruction by decoupling
dynamic and static components.
3

<!-- page 4 -->
Color Supervision
RGB Images
Initial Point Cloud
Geometry Supervision
Bounding Boxes
ℒ!
ℒ"#$%&
ℒ'()*+,
Semantic Supervision
ℒ*+-.
Splatting
Input 
Views
Given 
Views
Dynamic Non-rigid Actors
LOD 0 Voxel
LOD 1 Voxel
𝑅/, 𝑇/
Original Pose
Current Pose
Dynamic Rigid Vehicles
Gaussian
Update
Anchor
Flow
𝑮𝒂
𝑮𝒂(𝒕)
Near
Middle
Far
Principal Direction
Background Model
Fig. 2 Overall pipeline of DriveSplat. A dynamic-static decoupling paradigm is adopted, where neural Gaussian
representations with partitioned voxel structures are applied for background reconstruction, while a deformation field
network models the temporal dynamics of each non-rigid actor. Depth maps and normal priors are incorporated to enhance
geometric accuracy.
2.2 Dynamic Scene Reconstruction
Traditional reconstruction methods [5, 9, 11] pri-
marily focus on static scenes and are unable to
represent dynamic scenes or objects with tempo-
ral variations, leading to issues such as motion
blur. The NeRF [5] leverages MLPs for implicit
modeling of static environments. This concept has
been extended to animate scenarios through the
integration of deformation fields [33–35]. Alter-
natively, certain strategies [36] conceptualize ani-
mate scenes as 4D radiance landscapes, albeit
at the cost of significant computational resources
attributed to ray-point sampling and volumetric
rendering. To mitigate these issues, acceleration
techniques [37, 38] have been devised for the
depiction of dynamic environments. Some meth-
ods include the use of geometry priors [39], the
projection of MLP-derived mappings [40], or the
implementation of grid/plane-oriented architec-
tures [41] to elevate both the speed and efficacy.
Several works adapt 3D Gaussians for dynamic
scenes [22–24]. Luiten et al.[42] train frame-by-
frame for multi-view scenes, while Yang et al.[43]
use a deformation field to represent the tem-
poral changes of objects. 4D-GS [44] propose
to use multi-resolution hex-planes[41] to encode
deformed motion. We learn from the above 4D
reconstruction methods and adopt a deformation
field to model the temporal evolution of neural
Gaussians for non-rigid actors.
2.3 Geometry Optimization in 3D
Reconstruction
Depth and normal supervision enhance scene
reconstruction by improving geometric accuracy
and surface orientation, enabling high-fidelity cap-
ture of complex scenes [45, 46]. Several methods
[28, 47] propose to integrate depth priors to guide
the reconstruction process. The following works
[48, 49] propose embedding depth supervision
into the NeRF framework to boost training effi-
ciency and reduce multi-view input dependency.
MVSGaussian [28] combines MVS with Gaussian
Splatting to improve reconstruction in sparse-view
settings. DN-Splatter [50] presents an innova-
tive approach by utilizing depth-normal fusion to
enhance point cloud precision in complex envi-
ronments, while 2D-GS [51] leverages 2D depth
maps to refine the Gaussian Splatting technique
for more efficient reconstruction in real-time appli-
cations. In driving scenarios, GaussianPro [52]
introduces a progressive propagation strategy that
focuses on optimizing geometric properties. And
Desire-GS [53] proposes the combination of geo-
metric priors for enhanced supervision, but faces
problems of very slow training speeds. Drawing on
the above methods, we utilize depth and normal
priors to guide neural Gaussian reconstruction,
enhancing geometric quality while maintaining
reconstruction efficiency.
4

<!-- page 5 -->
3 Methology
As shown in Fig. 2, the inputs to DriveSplat
include RGB images, an initialized 3D point cloud,
and bounding boxes of dynamic actors provided
by the dataset [54, 55]. Depth and normal pri-
ors predicted by pre-trained models [56, 57] are
utilized during the supervised optimization stage.
3.1 Point Cloud Initialization
DriveSplat supports multiple types of point cloud
initialization, including SfM, LiDAR, and dense
DUSt3R [58] input. Our model separates dynamic
actors by leveraging tracked bounding boxes to
estimate their pose parameters (qobj, tobj), which
are used to compute a transformation matrix Tobj.
When LiDAR data is available, each frame’s point
cloud Pframe is then transformed into the actor’s
local coordinate system for consistent modeling.
Within the local coordinate system, an axis-
aligned bounding box B is constructed to identify
and filter the points Pobj enclosed within it. This
process is applied to all tracked actors, generating
dynamic actor point cloud masks Mobj.
Pobj =
[
i
n
P(i)
frameT−1
obj | M(i)
obj = 1
o
.
(1)
The remaining unmasked points are classified
as static points, defined as:
Ps = {p | p ∈Pframe, Mobj(p) = 0}.
(2)
In the absence of LiDAR data, our method
initializes the point cloud of dynamic actors by
randomly initializing points within the bound-
ing boxes in the relative coordinate system. For
static background points, when LiDAR is not
used, we directly use COLMAP’s point cloud for
initialization, as it contains only static points.
3.2 Large-scale Driving Background
Representation
We propose a scene-aware background reconstruc-
tion framework for large-scale driving environ-
ments based on neural Gaussian representations.
The framework explicitly accounts for the geo-
metric characteristics of driving scenes, including
large spatial extent, strong depth variation, and
continuous camera motion.
a) Multi-scale Partitioned Background
Point Cloud
LOD 
Initialize
Multi-level Grids
Hash Table
Neural Gaussians
Anchor
b) Learnable LOD Allocation 
Far
Mid
View-adaptive 
LOD Selection
Camera View
📷
Selected Levels
Near
Far
Mid
Near
Fig. 3 Overview of the proposed multi-scale back-
ground
representation
and
view-adaptive
LOD
allocation. (a) The static background is modeled using a
multi-scale Gaussian representation. (b) Geometry-guided
near/mid/far regions provide structural priors, while the
effective LOD of each anchor is dynamically selected based
on the current camera viewpoint.
Our approach integrates a unified multi-scale
Gaussian feature representation, geometry-guided
scene partitioning, and a view-adaptive LOD
allocation strategy. Together, these components
enable robust background reconstruction under
significant viewpoint changes. An overview of the
proposed framework is illustrated in Fig. 3.
Scene-aware multi-scale Gaussian
representation.
To effectively model the large-scale variation
inherent in driving scenes, we adopt a multi-
scale Gaussian representation in which each spa-
tial anchor is associated with level-specific latent
features. Rather than relying on explicit hierarchi-
cal spatial data structures [13], we parameterize
anchor features across multiple scales using a uni-
fied and learnable representation that remains
flexible to scene complexity.
Specifically, we maintain a set of L resolu-
tion levels, each corresponding to a distinct spatial
scale. At each level l ∈{0, . . . , L −1}, anchor
features are parameterized by a learnable table
T (l) ∈RH×F , where H denotes the table capac-
ity and F the feature dimensionality. The spatial
5

<!-- page 6 -->
resolution associated with level l is defined as
r(l) = ⌊rbase · bl/(L−1)⌋,
(3)
where rbase and rfinest denote the minimum and
maximum spatial resolutions considered in the
multi-scale representation, and b = rfinest/rbase
defines
the
geometric
scaling
factor
between
consecutive resolution levels. This formulation
enables the representation to accommodate spa-
tial structures of varying scale within a single
parameterization, which is essential for modeling
both fine-grained geometry and large-scale layout
in driving environments.
To associate spatial anchors with the cor-
responding level-specific feature parameters, we
define a mapping from continuous 3D positions
to discrete feature indices at each resolution level.
Given a 3D anchor position x ∈R3, we normalize
it within the scene bounding box and discretize it
into integer grid coordinates at level l as:
g(l) =

x −xmin
xmax −xmin
· r(l)

.
(4)
The resulting grid coordinates are then mapped
to feature indices using a spatial hashing function
h(g) = (gx · π1 ⊕gy · π2 ⊕gz · π3) mod H,
(5)
where π1, π2, π3 are fixed large prime numbers and
⊕denotes the bitwise XOR operator.
Each anchor is associated with a single level-of-
detail assignment l, which is optimized by a scene-
aware LOD allocation strategy discussed later.
Given an anchor position a and its assigned level
l, the corresponding feature vector is obtained as:
fhash(a, l) = T (l)
h(g(l)(a))

.
(6)
By enforcing level-specific feature access rather
than aggregating features across multiple resolu-
tions, each anchor is represented at a single, well-
defined spatial scale. This design reduces implicit
parameter coupling across scales and leads to a
more stable and interpretable multi-scale Gaus-
sian representation, which is particularly suitable
for large-scale driving scenes.
Near-region voxels
Mid-region voxels
Far-region voxels
Level 0
Level 1
Level 2
Fig. 4 Visualization of voxel representation at dif-
ferent levels. As the level increases, the voxel resolution
gradually improves. The corresponding partitioned neural
Gaussians are shown in the bottom row.
Geometry-guided scene partitioning for
driving environments.
Driving scenes span a wide range of depths, with
scene elements ranging from regions close to the
ego vehicle to distant urban structures. Such
characteristics are not well captured by uniform
spatial discretization strategies, which treat all
spatial regions equivalently and ignore the depth-
dependent structure inherent in driving environ-
ments. To better align the representation with
scene geometry, we introduce a geometry-guided
partitioning strategy that decomposes the back-
ground into three depth-ordered regions, referred
to as the near, mid, and far regions.
Given a set of 3D points P = {pi}N
i=1 sam-
pled from the background point cloud Ps, we first
estimate the dominant geometric direction of the
scene via principal component analysis (PCA):
dmain = arg max
∥v∥=1 Var
 v⊤(P −¯P)

,
(7)
where ¯P denotes the centroid of the point cloud.
In typical driving scenarios, the resulting prin-
cipal direction dmain aligns closely with the for-
ward viewing direction of the ego vehicle, provid-
ing a meaningful reference for depth-based scene
decomposition.
We then project each point pi onto the princi-
pal direction to obtain a scalar depth coordinate
zi, and partition the depth distribution into three
regions using quantile-based thresholds:
Rnear = {pi : zi ≤τ1},
Rmid = {pi : τ1 < zi ≤τ2},
Rfar = {pi : zi > τ2},
(8)
6

<!-- page 7 -->
where τ1 and τ2 correspond to the q1-th and q2-th
percentiles of the depth distribution, respectively.
Unless otherwise specified, we set q1 = 0.33 and
q2 = 0.67 in our experiments. An example of the
resulting partitioning is visualized in Fig. 4.
Based on this partitioning, we assign region-
specific spatial resolutions to better match the
geometric characteristics of different depth ranges.
Specifically, the effective voxel size for region r ∈
{near, mid, far} is defined as:
sr = sbase/αr,
(9)
where sbase denotes a reference spatial scale and
αnear > αmid > αfar are region-dependent scaling
factors. This design enables finer geometric rep-
resentation in regions closer to the camera while
maintaining coherent structure in more distant
areas, and serves as a geometry-aware foundation
for subsequent level-of-detail allocation.
Learnable Level-of-Detail allocation.
Assigning LOD based solely on static heuristics
or initialization-time region labels is insufficient
for large-scale driving scenes, as camera motion
continuously alters the visual relevance of dif-
ferent spatial regions. Although the background
scene is first decomposed into near-, mid-, and far-
range regions using geometry-guided partitioning,
this coarse structural prior alone does not fully
determine the appropriate representation granu-
larity. Anchors initially assigned to distant regions
may later become visually prominent as the cam-
era approaches them, motivating a view-adaptive
LOD allocation strategy that dynamically selects
the effective representation level of each anchor.
To account for this effect, we dynamically
select anchor visibility and LOD levels at each
rendering frame based on the current camera view-
point. For an anchor i, we compute a target LOD
level as
Ltarget
i
=

logs
dmax
di

,
(10)
where di denotes the Euclidean distance between
anchor i and the camera center of current view,
dmax is a scene-dependent normalization constant,
and s controls the LOD scaling factor. An anchor
is rendered if and only if its assigned resolution
level Li satisfies Li ≤Ltarget
i
. This formulation
ensures that anchors are selected according to
their instantaneous visual relevance rather than
fixed depth labels.
The geometry-guided scene partitioning intro-
duced in Sec. 3.2 provides a coarse structural
prior by decomposing the background into distinct
regions. While this partitioning offers a reasonable
initialization, fixed region boundaries are insuffi-
cient to capture the continuously changing visual
relevance induced by camera motion in driving
scenes. We therefore lift the partition boundaries
to learnable parameters and refine them jointly
with the LOD allocation during training.
Concretely, the region boundaries τ1 and τ2 are
treated as continuous learnable variables, initial-
ized using the quantile-based geometry partition-
ing. Given the projected depth coordinate zi of
anchor i, its soft region assignment is defined as
wi,r = σ
zi −τr−1
ℓr

−σ
zi −τr
ℓr

,
(11)
where r ∈{near, mid, far}, σ(·) denotes the sig-
moid function, ℓr controls the transition sharpness
between adjacent regions, and we define τ0 = −∞
and τ3 = +∞. This formulation yields a differen-
tiable, region-aware weighting that allows anchors
to smoothly adjust their effective LOD preferences
as the region boundaries evolve.
The
parameters
{τ1, τ2, ℓr}
are
optimized
jointly with the neural Gaussian representa-
tion via gradients propagated from the recon-
struction objective, which consists of photomet-
ric and geometry-aware losses. Through this
reconstruction-driven
supervision,
the
initially
geometry-defined partitions are gradually refined
to better align with view-dependent visibility and
scene content, enabling a data-driven and adap-
tive LOD allocation without relying on manually
tuned heuristics.
To ensure stable optimization, we enforce the
ordering constraint τ1 < τ2 by parameterizing
them as cumulative offsets. Unless otherwise spec-
ified, all learnable partition parameters are initial-
ized to match the geometry-guided partitioning
described in Sec. 3.2.
3.3 Dynamic Actor Modeling
Dynamic actors in driving scenes include both
rigid objects, such as vehicles, and non-rigid
actors, such as pedestrians, which exhibit complex
7

<!-- page 8 -->
Anchor 𝑃!(0)
Time  𝑡
GaussianUpdate
(a) Anchor-centric Non-rigid Motion Modeling
(b) Deformation Propagation to Neural Gaussians.
Anchor
𝑃!(𝑡)
Anchor 
𝑃!(𝑡)
Time  𝑡
Gaussians 
G"(0)
Gaussians G"(𝑡)
AnchorFlow
Fig. 5 Pipeline for dynamic non-rigid actor modeling. Our two-stage pipeline first estimates anchor-level motion
and then updates the corresponding neural Gaussian parameters.
combinations of global motion and local defor-
mation. To model these dynamics in a unified
manner, we adopt an object-centric formulation
that decomposes actor motion into a shared global
rigid transformation and an optional non-rigid
deformation component.
Global actor motion.
For each actor, we first model its global motion as
a time-varying rigid transformation that aligns the
actor from a canonical coordinate system to the
world frame. This global transformation is shared
by both rigid and non-rigid actors and captures
their overall translation and rotation over time.
The position µo and rotation Ro for an actor
are defined within the actor’s local coordinate
system. To align with the background, these vari-
ables must be transformed into the world coordi-
nate system, which requires applying the actor’s
tracked poses. Specifically, these tracked poses are
represented by a set of rotation matrices {Rt}Nt
t=1
and translation vectors {Tt}Nt
t=1, where Nt denotes
the total number of frames. This transformation
is expressed as
µ = Rtµo + Tt,
R = RoRT
t ,
(12)
where µ and R denote the position and rotation
of the dynamic actor’s Gaussians Ga within the
world coordinate system.
At each time step t, these canonical Gaussian
parameters are mapped to the world coordinate
system via the rigid transformation, ensuring con-
sistent pose alignment throughout the sequence.
Anchor-centric non-rigid motion modeling.
While rigid actors are fully described by the global
transformation alone, non-rigid actors require
additional modeling to capture local deformations.
Rather than directly modeling dense per-Gaussian
deformations in 4D space, we observe that, in
anchor-based neural Gaussian representations, the
spatiotemporal evolution of Gaussians is primar-
ily governed by the motion of their associated
anchors.
Motivated by this observation, we introduce
AnchorFlow, an N-layer MLP that learns a con-
tinuous flow field over a sparse set of anchors
in canonical space. Conditioned on anchor rep-
resentations and an encoded time variable t,
AnchorFlow predicts the updated anchor posi-
tions Pa(t), providing a compact and differentiable
parameterization of non-rigid motion over time, as
illustrated in Fig. 5(a).
This
anchor-centric
formulation
introduces
a strong inductive bias that enforces coherent
motion among neural Gaussians linked to the
same anchor, while substantially reducing the
number of learnable parameters compared to
dense 4D Gaussian deformation models.
Deformation propagation to neural
Gaussians.
Given the updated anchor states at time t, we
propagate anchor motion to the associated neural
Gaussians through a multi-head update scheme.
Specifically, a geometry update head synchro-
nizes the spatial attributes of each Gaussian
with its corresponding anchor, updating spatial
attributes based on the anchor displacement and
temporal encoding, as shown in Fig. 5(b). This
anchor-driven synchronization ensures coherent
geometric deformation among Gaussians linked to
the same anchor.
Following the geometric update, the appear-
ance heads independently update opacity α and
8

<!-- page 9 -->
color c for each neural Gaussian, producing time-
dependent appearance parameters α(t) and c(t).
By decoupling geometry propagation from appear-
ance evolution, our design enables stable opti-
mization while preserving flexible, per-Gaussian
appearance dynamics under non-rigid motion.
Compositional scene rendering.
After applying the rigid transformation and the
optional non-rigid deformation, we obtain the
updated dynamic actor representation Ga(t) at
timestamp t. We then combine the dynamic Gaus-
sians of all actors with the static background
Gaussians Gb to form the complete scene repre-
sentation:
G(t) = (∪K
i=1Gi
a(t)) ∪Gb,
(13)
where K denotes the number of dynamic actors
reconstructed in the scene. Finally, given the cam-
era view matrix Mt of the current frame, we render
the RGB image It, depth map Dt, and normal
map Nt using the splatting operator S:
(It, Dt, Nt) = S(Mt, G(t)).
(14)
3.4 Geometry Enhanced
Optimization
Geometry prior estimation.
We utilize depth and normal priors to guide the
geometry optimization process. To obtain per-
pixel referenceable depth values, monocular depth
estimation can be employed to predict either
absolute depth Dm or relative depth Dr. We
use the DepthAnything-V2 [56] model for rel-
ative depth estimation and the ZoeDepth [59]
model for absolute depth estimation. To obtain
reliable surface normal priors, we leverage a pre-
trained normal estimation model [57] to generate
normal maps Nm. These estimated normals pro-
vide crucial geometric constraints, contributing
to improved surface quality of the reconstructed
driving scenario.
Loss functions.
To achieve a high-quality rendering, we define a
comprehensive loss function as:
L = Lr + λdLdepth + λnLnormal + λmLmask, (15)
where Lr, Ldepth, Lnormal, Lmask represent the
rendered color loss, depth loss, normal loss, and
dynamic actor mask loss, respectively. Here, λd,
λn, and λm are hyperparameters that control the
weights of each loss term during optimization.
The rendering color loss Lr incorporates an L1
loss LL1(I, Igt) and a structural similarity index
loss LSSIM(I, Igt), defined as:
Lr = LL1(I, Igt) + λLSSIM(I, Igt),
(16)
where λ balances the contribution of the L1 loss
and SSIM loss to achieve high-quality rendering
results.
To further enhance geometric accuracy, we
include the depth loss Ldepth, to enforce con-
sistency between our estimated depth map D
and the predicted relative depth map Dr. This
loss is scaled by weight λd and computed using
correlation loss:
Ldepth = ∥
Cov(D, Dr)
p
Var(D)Var(Dr)
∥1,
(17)
where Cov(·) and Var(·) denoting the covariance
and variance, respectively. Our method also sup-
ports supervision using absolute depth Dm, where
the L1 loss is used rather than the correlation loss.
To ensure that the predicted surface normals
align with the real surface normal distribution,
we incorporate the normal loss Lnormal. The loss
includes L1 loss LnL1 for error between our esti-
mated normal N and predicted reference normal
Nm, and cosine similarity loss Lncos for normal
direction alignment. The total normal loss is:
Lnormal = LnL1(N, Nm) + Lncos(N, Nm).
(18)
In addition, the mask loss Lmask employs cross-
entropy to compare predicted and ground-truth
masks, helping to improve rendering quality of
dynamic actors.
4 Experiments
4.1 Datasets
Waymo [54] provides a diverse collection of
sensor data covering both urban and suburban
environments. We select 12 sequences captured
under a variety of conditions, including differ-
ent weather settings (e.g., foggy and sunny) and
9

<!-- page 10 -->
GaussianPro
OmniRe
StreetGS
Ours
Ground Truth
Fig. 6 Qualitative novel-view synthesis results on the Waymo dataset. Patches that highlight the visual differences
are emphasized with red boxes and enlarged for clearer visibility. The rendering image resolution is 1066 × 1600.
Table 1 Quantitative comparison results on Waymo dataset. S and D indicate methods that model static scenes
only and dynamic scenes, respectively. Dopt and Nopt denote depth and normal optimization, respectively. The image
resolution is 1066 × 1600. LPIPS uniformly adopts the AlexNet. Bold: Best. Underline: Second Best.
Model
Type
Dopt
Nopt
Reconstruction
NVS
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
3D-GS [9]
S
×
×
33.15
0.929
0.108
30.57
0.923
0.113
GaussianPro [52]
S
✓
✓
32.79
0.928
0.113
31.28
0.915
0.121
Octree-GS [13]
S
×
×
33.12
0.934
0.109
31.84
0.917
0.109
4D-GS [44]
D
×
×
31.15
0.897
0.197
28.35
0.873
0.208
Deform-GS [43]
D
×
×
32.58
0.909
0.162
30.04
0.892
0.174
S3Gaussian [18]
D
✓
×
33.64
0.931
0.117
31.32
0.912
0.127
PVG [17]
D
✓
×
34.37
0.934
0.102
31.89
0.912
0.118
StreetGS [19]
D
×
×
35.15
0.935
0.110
30.24
0.878
0.125
OmniRe [21]
D
✓
×
34.57
0.939
0.112
31.19
0.897
0.126
Desire-GS [53]
D
✓
✓
34.35
0.925
0.109
32.35
0.917
0.122
AD-GS [24]
D
✓
×
35.26
0.936
0.105
33.08
0.920
0.112
Ours
D
✓
✓
35.41
0.937
0.096
33.83
0.923
0.103
traffic scenarios (e.g., urban low-speed roads and
highways). Among them, the representative set
of 12 sequences consists of 8 sequences from
StreetGS [19] and 4 sequences from OmniRe [21].
This combination is intentionally chosen to lever-
age the complementary characteristics of the two
datasets. StreetGS [19] sequences feature chal-
lenging background variations but do not contain
non-rigid actors, whereas OmniRe [21] sequences
include non-rigid actors while presenting relatively
less challenging backgrounds due to the stable
ego-view.
KITTI [55] includes numerous scenes with sig-
nificant lighting variations, ranging from high-
exposure areas to shadowed regions, creating
substantial challenges for reconstruction. There-
fore, evaluating the reconstruction performance
on sequences from the KITTI dataset provides a
robustness test of the model’s resilience to varying
environmental conditions. We select 3 sequences
containing both dynamic non-rigid actors and
10

<!-- page 11 -->
Front Left
Front
Front Right
38.13 dB
37.33 dB
36.30 dB
Normal
Depth
Normal
Depth
Normal
Depth
Fig. 7 Front 3-views rendering results of our method on the Waymo dataset. The above shows the rendered
depth and normal maps, and below shows the rendered images. Our method can preserve the intricate details of the scene,
and generate accurate depth and normal maps.
Table 2 Quantative results on KITTI dataset. The rendering image resolution is 370 × 1226.
Model
Type
Scene Reconstruction
Novel View Synthesis
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
3D-GS [9]
S
24.43
0.817
0.162
20.10
0.678
0.224
Octree-GS [13]
S
23.51
0.738
0.288
21.90
0.679
0.300
GaussianPro [52]
S
23.51
0.788
0.181
18.47
0.607
0.277
4D-GS [44]
D
24.43
0.735
0.280
18.12
0.562
0.338
Deform-GS [43]
D
27.64
0.810
0.113
22.01
0.727
0.156
OmniRe [21]
D
28.68
0.874
0.115
20.86
0.592
0.187
Ours
D
28.59
0.895
0.100
24.53
0.767
0.142
challenging backgrounds to perform comparative
experiments.
4.2 Implementation Details
We evaluate the performance of our method
in both training-view reconstruction and novel-
view synthesis. The novel views are selected by
sampling every fourth frame from the original
sequence, ensuring that these views are excluded
from the model’s training process. Training a
scene of 30K iterations takes 68 minutes on a sin-
gle NVIDIA L20 GPU, far more efficient than
Desire-GS’s over 180 minutes (see Table 5). We
employ DepthAnything-V2 [56] to generate rela-
tive depth priors, ZoeDepth [59] to obtain metric
depth priors, and Omnidata [57] to provide surface
normal priors. During point cloud initialization,
we employ COLMAP [60] to obtain SfM points,
which are then fused with LiDAR points. During
training, the gradient information is incorporated
after 500 iterations to update the neural Gaus-
sians and adjust the visibility of anchor points.
After 1500 iterations, neural Gaussians are fur-
ther refined, including the removal of redundant
or invisible anchors, addition of new anchors, and
updates to properties such as opacity. We train our
model with 30,000 iterations on both the Waymo
and KITTI datasets. In our training process, we
set the hyperparameters as follows: ϕ1 = 10, ϕ2 =
4, λd = 0.01, λn = 0.01, λm = 0.01, λ = 0.2.
11

<!-- page 12 -->
GaussianPro
OmniRe
Ours
Ground Truth
Fig. 8 Qualitative NVS comparison results on the KITTI dataset. The rendering image resolution is 370 × 1226.
4.3 Benchmark Evaluation
Results on Waymo.
We evaluate our approach against established
methods, including both static methods and
dynamic methods, and use both reconstruction
and novel view synthesis metrics on the Waymo
dataset (Table 1). Our method surpasses all
baselines in both PSNR and LPIPS reconstruc-
tion metrics, showcasing high precision. While
StreetGS [19] and OmniRe [21] perform well
in reconstruction metrics, they struggle with
novel view synthesis, highlighting their limitations
in handling viewpoint transitions. In contrast,
DriveSplat excels in novel view synthesis tasks,
outperforming all baselines across three evaluation
metrics. Visual analysis (Fig. 6) highlights DriveS-
plat’s superior artifact-free rendering of vehicles,
enhanced clarity of challenging static background
details, and accurate depiction of dynamic vehi-
cles.
We compare our method with representa-
tive static and dynamic baselines on the Waymo
dataset using both reconstruction and novel view
synthesis metrics, as reported in Table 1. DriveS-
plat consistently achieves the best performance in
terms of SSIM and LPIPS for reconstruction, indi-
cating accurate fitting of observed views. While
recent dynamic methods such as StreetGS [19]
and OmniRe [21] attain competitive reconstruc-
tion scores, their performance degrades noticeably
under novel viewpoint evaluation, reflecting lim-
ited robustness to viewpoint changes. In contrast,
DriveSplat demonstrates strong generalization in
novel view synthesis, outperforming all baselines
across all evaluated metrics. Qualitative results
Table 3 Quantitative comparison results focusing
on sequences with predominantly non-rigid actors.
PSNR* denotes the rendering quality at dynamic regions.
Model
PSNR↑
SSIM↑
LPIPS↓
PSNR*↑
Deform-GS [43]
28.19
0.903
0.135
22.58
4D-GS [44]
23.77
0.860
0.121
17.90
OmniRe [21]
30.69
0.906
0.107
28.79
Ours
30.96
0.913
0.104
29.46
in Fig. 6 further illustrate that DriveSplat pro-
duces artifact-free renderings of dynamic vehicles,
preserves fine-grained static background struc-
tures, and maintains consistent geometry under
challenging viewpoint transitions.
To specifically evaluate the effectiveness of our
method in reconstructing non-rigid actors com-
pared to other dynamic reconstruction techniques,
we conducted assessments on two select sequences
from the Waymo dataset, which contain extensive
pedestrian activity. We utilized the PSNR* metric
to focus specifically on the performance pertain-
ing to dynamic components, and compare with
recent non-rigid optimization methods, includ-
ing Deform-GS [43], 4D-GS [44], and OmniRe
[21]. As shown in Table 3, our method surpasses
other reconstruction approaches that incorporate
deformation networks.
Results on KITTI.
The performance of DriveSplat is also evaluated
on the KITTI dataset. As detailed in Table 2,
DriveSplat outperforms baselines in both recon-
struction and novel view synthesis tasks. Fig.
8 highlights rendering results, where DriveSplat
demonstrates superior background clarity and
12

<!-- page 13 -->
Table 4 Quantitative evaluation on a challenging
novel-view synthesis task. Results are reported under
three camera viewpoint shift settings: 1 m, 2 m, and 5 m.
Model
1m
2m
5m
FID↓FVD↓FID↓FVD↓
FID↓
FVD↓
StreetGS [19]
38.3
16.8
61.9
36.7
149.6
64.9
OmniRe [21]
41.8
25.4
60.4
34.5
155.9
68.4
Desire-GS [53]
40.8
23.6
60.5
40.4
151.6
65.8
AD-GS [24]
37.4
15.9
57.2
31.9
152.6
67.9
Ours
35.8
12.9
50.0
27.2
149.3
64.0
accurate rendering of dynamic vehicles. Com-
pared to OmniRe [21], our method demonstrates
improved performance in novel view synthesis,
particularly in preserving background details, as
illustrated in the second row of Fig. 8.
4.4 Challenging NVS Comparison
To evaluate the rendering performance of our
method under challenging novel view conditions,
we conduct experiments on the Waymo dataset
by modifying the original training viewpoints and
visualizing the outcomes. Specifically, we apply
a 1.0-meter viewpoint shift and compare our
method with StreetGS [19], as illustrated in Fig.
9. Our method consistently maintains high-quality
rendering of static background elements, such as
parked vehicles and road surfaces. For instance,
under a 1.0-meter rightward viewpoint shift (Fig.
9), our approach preserves fine-grained details of
road markings, including white lane lines, whereas
StreetGS [19] exhibits noticeable rendering arti-
facts, particularly on road markings.
For quantitative evaluation, we conduct a com-
prehensive comparison under camera viewpoint
shifts of 1 m, 2 m, and 5 m, corresponding to
increasing degrees of viewpoint change in driving
scenarios. As shown in Table 4, our method consis-
tently outperforms competing approaches under
small viewpoint shifts (1 m and 2 m), demon-
strating robust novel-view synthesis for moderate
viewpoint changes commonly encountered in real-
world driving scenes. Under the largest shift of 5
m, all methods exhibit comparable performance,
reflecting the inherent difficulty of large view
extrapolation in driving scenarios when relying
solely on reconstruction-based approaches.
Forward 1.0m
Up 1.0m
Left 1.0m
Right 1.0m
StreetGS
Ours
Reference View
Fig. 9 Qualitative comparison of NVS rendering
results on Waymo. Four challenging novel views are
obtained by applying a 1.0-meter shift in viewpoint.
Table 5 Training and rendering speed comparison.
PSNR shows the rendering result from test views.
Model
Train (min)↓
FPS↑
PSNR (dB)↑
StreetGS [19]
84.6
44.1
30.24
OmniRe [21]
68.2
51.5
31.19
Desire-GS [53]
182.9
37.8
32.35
AD-GS [24]
88.6
54.2
33.08
Ours
69.6
73.5
33.83
4.5 Time Efficiency Analysis
To evaluate the time efficiency of our method,
we
conduct
a
comparative
study
with
sev-
eral state-of-the-art dynamic–static disentangled
driving-scene reconstruction approaches, includ-
ing StreetGS [19], OmniRe [21], Desire-GS [53],
and AD-GS [24]. The comparison covers both
training efficiency and rendering performance, as
well as the corresponding reconstruction qual-
ity. The quantitative results are summarized in
Table 5. Our model demonstrates a balanced per-
formance with a training time of 69.6 minutes
and a rendering speed of 73.5 FPS, outperform-
ing StreetGS [19] and AD-GS [24]. Desire-GS [53]
employs a two-stage training strategy and requires
over 50,000 iterations, resulting in a lengthy train-
ing process. Additionally, our model achieves a
superior rendering quality, indicating better novel-
view synthesis quality.
13

<!-- page 14 -->
4.6 Ablation Study
Ablation study on initialization module.
We evaluate the impact of different initialization
methods, as shown in Table 6. LiDAR delivers
the most precise point clouds but lacks coverage
for tall buildings and distant areas. Although SfM
provides sparser points, it offers broader scene
coverage and thus slightly outperforms LiDAR
alone. DUSt3R yields the densest point clouds,
but due to misalignment in scale and position with
real-world coordinates, even after transformation,
its performance is suboptimal. Consequently, we
selected the SfM+LiDAR combination for ini-
tialization, which produced the best rendering
results.
Ablation study on background
representation module.
Our background representation module incorpo-
rates three key components: scene-aware multi-
scale Gaussian representation (SMG), geometry-
guided scene partitioning (GSP), and learnable
level-of-detail allocation (LLOD). To evaluate the
contribution of each component, we perform a
comprehensive ablation study, with the results
summarized in Table 7. As a baseline, we replace
the proposed background representation module
with that adopted in StreetGS [19], while keeping
all other settings unchanged. As shown in Table 7,
we add these components cumulatively, and each
addition consistently improves rendering quality.
Moreover, SMG and LLOD notably improve ren-
dering efficiency, indicating their effectiveness in
reducing computational overhead while preserving
reconstruction fidelity.
Furthermore, we provide more visualization
results to show the improvement of our method.
As shown in Fig. 10, the use of our background
optimization method can better preserve the back-
ground details in close-range regions and reduce
the artifacts.
Ablation study on geometry optimization.
We conducted depth and normal rendering exper-
iments using GaussianPro [52] as a baseline due
to its high geometric accuracy in driving scenar-
ios. For depth evaluation, we compare our method
using depth priors (including metric depth Dm
and relative depth Dr) and without, as shown in
Table 8.
We use three standard metrics: Abso-
Baseline
Ours
Ground Truth
Fig. 10 Qualitative ablation study of background
representation module. Our method yields improved
rendering details compared to the baseline.
Table 6 Ablation study of the point cloud
initialization module. Fusion* indicates our final
choice of combining SfM and LiDAR data.
Initialize
PSNR↑
SSIM↑
LPIPS↓
Abs Rel↓
Random
29.64
0.894
0.127
0.269
SfM
33.16
0.912
0.112
0.239
LiDAR
32.98
0.913
0.114
0.174
DUSt3R
31.42
0.887
0.118
0.209
Fusion*
33.83
0.923
0.103
0.185
lute Relative Error (Abs Rel), Mean Absolute
Error (MAE), and the accuracy under threshold
(δ < 1.25), together with PSNR to assess the qual-
ity of rendered images. Our method surpasses the
baseline across all metrics. While metric depth
Dm yields the most accurate depth, it reduces
rendering quality. Conversely, relative depth Dr
enhances rendering but lessens depth accuracy.
This trade-off is due to the limited accuracy of
absolute depth data, leading us to prefer relative
depth for better overall quality. For normal evalua-
tion, we assess our method both with and without
normal priors, using the estimated normals Nm as
referenced GT. We select three normal evaluation
metrics: MAE, Root Mean Square Error (RMSE)
and cosine similarity (Simi.). The result shows
that our method outperforms the baseline in all
metrics and proves the effectiveness of the normal
14

<!-- page 15 -->
Table 7 Ablation study on background
representation module. SMG, GSP, and LLOD denote
the scene-aware multi-scale Gaussian representation,
geometry-guided partitioning, and learnable
level-of-detail allocation, respectively.
Model
PSNR↑
SSIM↑
LPIPS↓
FPS↑
Baseline
31.98
0.915
0.119
50.87
+ SMG
32.94
0.918
0.109
66.72
+ GSP
33.47
0.919
0.107
61.60
+ LLOD
33.83
0.923
0.103
73.52
Table 8 Quantitative ablation study results of
depth and normal optimization module.
Depth Model
AbsRel↓MAE↓δ1.25↑PSNR↑
GaussianPro [52]
0.527
9.53
56.27
31.28
Ours (w/o Ldepth)
0.288
5.26
69.23
33.20
Ours (Dm)
0.124
3.41
81.99 33.12
Ours (Dr) *
0.179
3.87
79.38 33.83
Normal Model
MAE↓RMSE↓Simi.↑PSNR↑
GaussianPro [52]
1.89
2.65
0.082
31.28
Ours (w/o Lnormal)
1.21
2.24
0.276
33.31
Ours (w/ Lnormal) *
0.84
1.78
0.498 33.83
w/o ℒ!"#$%, ℒ&'()*+
Ours
RGB
Depth
Normal
Fig. 11 Qualitative ablation study results on geom-
etry supervision.
priors. For qualitative comparison, we show the
rendered depth and normal results of our method
and those without priors in Fig. 11.
Ablation study on dynamic representation
module.
We utilize four Waymo sequences that contain
many pedestrians in front views to evaluate
our non-rigid actor reconstruction performance.
Decoupling dynamic and static elements in the
Table 9 Quantitative ablation study of the
dynamic representation module. Deform denotes the
deformable network proposed in Deform-GS [43], while
deform. denotes our proposed anchor-centric deformable
module.
Model
PSNR↑SSIM↑LPIPS↓FPS↑
OmniRe [21]
30.69
0.906
0.107
37.2
Ours (w/o deform.)
28.37
0.891
0.124
58.9
Ours (w/ Deform)
30.84
0.912
0.107
39.3
Ours (w/ deform.) * 30.96
0.913
0.104
48.7
w/o obj.
w/o deform.
Ours
Fig. 12 Qualitative ablation study results on the
dynamic non-rigid actor representation module.
scene effectively improves rendering quality for
moving actors. Adding individual dynamic object
representation can improve the rendering qual-
ity of moving vehicles, but it still suffers from
motion blur for non-rigid actors. After adding
the deformable module, the rendering quality
of non-rigid actors is significantly improved, as
shown in Fig. 12. Compared to the OmniRe [21]
that additionally incorporates SMPL [61], our
method achieves comparable reconstruction per-
formance (Table 9). To evaluate the effectiveness
of our proposed non-rigid deformation module,
we replace it with the deformable network used
in Deform-GS [43] while keeping all other com-
ponents unchanged. As shown by the results,
our anchor-centric deformation module achieves
higher rendering efficiency compared to existing
4D representation methods [43].
15

<!-- page 16 -->
Table 10 Ablation study of supervision modules.
Results are derived from reconstruction experiments
conducted on the Waymo dataset.
Model
PSNR↑
SSIM↑
LPIPS↓
Ours (w/o Lnormal)
30.51
0.911
0.105
Ours (w/o Ldepth)
30.39
0.910
0.106
Ours (w/o Lmask)
30.14
0.908
0.108
Ours
30.96
0.913
0.104
Reference View
Translation
Rotation
Fig. 13 Editing operations on the Waymo. Our
method supports dynamic actors editing, including trans-
lation and rotation.
Ablation Study of Loss Functions
We further verify the effectiveness of the loss func-
tions used in our method, as shown in Table 10.
The ablation study is conducted on four Waymo
sequences containing non-rigid actors. These four
sequences are also utilized in Table 9. All eval-
uation metrics are obtained from reconstruction
experiments using the test views. Among all the
loss functions, the Lmask performs the most signif-
icant improvement on the rendering quality since
it can ensure the accuracy of dynamic actors.
5 Application and Discussion
5.1 Application of Scene Editing
Based on the reconstructed scene, our method
enables the editing of foreground dynamic objects,
which includes operations such as translation and
rotation of specified targets, as illustrated in Fig.
13. By allowing precise manipulation of these ele-
ments, our approach offers enhanced flexibility
and control over scene composition and dynamic
interaction.
5.2 Limitation and Discussion
Our current approach to 3D reconstruction faces
limitations primarily in the domain of scene edit-
ing. Specifically, the challenge arises from the
insufficient perspective awareness of foreground
objects such as vehicles. Due to the lack of full-
view perception, these objects cannot be wholly
reconstructed, leading to potential deficiencies
wherein some aspects of the foreground objects
may be missing during editing processes.
To address this limitation, our future work
plans to integrate existing advancements in image-
to-3D technology. By generating comprehensive
3D representations from single-image foreground
inputs, we aim to provide complete 3D assets for
editing. This will enable seamless replacement and
manipulation using full 3D models, therefore over-
coming the present constraint of incomplete object
representation during editing.
However, it is important to note that our cur-
rent scene editing framework does not yet support
this enhancement. We are committed to contin-
uous improvement and anticipate implementing
this functionality in subsequent versions of the
publicly available code. This effort will ensure that
users can leverage fully reconstructed 3D assets
in their scene editing endeavors, resulting in more
robust and versatile applications.
6 Conclusion
We have introduced the DriveSplat, a novel
approach for 3D reconstruction in driving sce-
narios that enhances the accuracy of both static
and dynamic elements. By integrating scene-
aware multi-scale Gaussian representation with
depth and normal priors, our method captures
detailed scene geometry for a large-scale back-
ground. By tracking the poses of moving vehi-
cles and applying anchor-centric motion modeling
and deformation propagation to non-rigid actors,
dynamic elements achieve accurate and efficient
reconstruction. DriveSplat achieves state-of-the-
art performance in novel-view synthesis tasks on
two autonomous driving datasets, allowing high-
quality geometry representation and large-scale
scene reconstruction.
16

<!-- page 17 -->
Selected Frame
a) OmniRe
b) StreetGS
c) Ours
d) GT
28.07 dB
22.97 dB
15.91 dB
30.97 dB
33.36 dB
22.77 dB
24.12 dB
27.34 dB
19.26 dB
30.71 dB
34.01 dB
20.67 dB
Fig. 14 Qualitative comparison of detail rendering results on Waymo. Our method demonstrates superior detail
preservation and visual clarity compared to StreetGS [19] and OmniRe [21], particularly in challenging detailed regions.
LOD 0
LOD 1
LOD 2
LOD 3
LOD 4
LOD 5
LOD 6
LOD 7
Fig. 15 Rendering results under different LOD. Increasing the LOD reveals finer details in near-range regions.
7 Appendix
7.1 Analysis of LOD
We adopt a scene-aware multi-scale learnable
Gaussian representation for background recon-
struction. Our LOD allocation strategy assigns
higher levels of detail to near-range regions and
lower levels to far-range regions, thereby allo-
cating more representational capacity to visually
important areas that occupy a larger portion of
the rendered images. As illustrated in Fig. 15,
increasing the LOD gradually shifts the render-
ing focus from far-range to near-range regions.
This scene-aware multi-scale representation leads
to more detailed rendering results. As shown in
Fig. 14, we compare the rendering quality of fine
details and demonstrate that our method outper-
forms existing approaches both quantitatively and
qualitatively.
7.2 More Visualization Results
Our model generates high-quality RGB images,
depth maps, and normal maps, providing a com-
prehensive representation of the scene for applica-
tions like autonomous driving and digital twins.
The rendering results are shown in Fig. 16. The
RGB outputs showcase photorealistic rendering
with fine details, the depth maps capture accu-
rate geometric relationships, and the normal maps
highlight precise surface orientations, demonstrat-
ing the model’s versatility and 3D structural
accuracy.
17

<!-- page 18 -->
RGB
Depth
Normal
Fig. 16 Rendering results on the Waymo dataset. Our method produces high-quality RGB images, depth maps,
and normal maps across diverse environmental conditions.
Acknowledgements This work was supported
by the National Natural Science Foundation of
China under Grant 62373356 and the Joint Funds
of the National Natural Science Foundation of
China under U24B20162.
Data Availability The Waymo dataset [54] is
publicly available at https://waymo.com/open.
The KITTI dataset [55] is available at https://
www.cvlibs.net/datasets/kitti.
Conflict of interest. The authors have no rele-
vant financial or non-financial interests to disclose.
References
[1] Yang, Z., Chen, Y., Wang, J., Manivasagam,
S., Ma, W.-C., Yang, A.J., Urtasun, R.:
Unisim: A neural closed-loop sensor sim-
ulator. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern
Recognition, pp. 1389–1399 (2023)
[2] Zhang, Z., Song, T., Lee, Y., Yang, L., Peng,
C., Chellappa, R., Fan, D.: Lp-3dgs: Learning
to prune 3d gaussian splatting. Advances in
Neural Information Processing Systems 37,
122434–122457 (2024)
[3] Han, K., Wong, K.-Y.K., Liu, M.: Dense
reconstruction
of
transparent
objects
by
altering incident light paths through refrac-
tion.
International
Journal
of
Computer
Vision 126(5), 460–475 (2018)
18

<!-- page 19 -->
[4] Cho, G., Kang, C., Soon, D., Joo, K.: Dogre-
con: Canine prior-guided animatable 3d gaus-
sian dog reconstruction from a single image:
Dogrecon: Canine prior-guided animatable 3d
gaussian... International Journal of Computer
Vision 133(9), 6332–6346 (2025)
[5] Mildenhall, B., Srinivasan, P.P., Tancik, M.,
Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf:
Representing scenes as neural radiance fields
for view synthesis. Communications of the
ACM 65(1), 99–106 (2021)
[6] Barron, J.T., Mildenhall, B., Tancik, M.,
Hedman, P., Martin-Brualla, R., Srinivasan,
P.P.: Mip-nerf: A multiscale representation
for anti-aliasing neural radiance fields. In:
Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 5855–
5864 (2021)
[7] Li, J., Yu, J., Wang, R., Gao, S.: Pseudo-
plane regularized signed distance field for
neural indoor scene reconstruction. Interna-
tional Journal of Computer Vision 133(6),
3203–3221 (2025)
[8] Xu, R., Yao, M., Chen, C., Wang, L., Xiong,
Z.: Continuous spatial-spectral reconstruc-
tion via implicit neural representation. Inter-
national Journal of Computer Vision 133(1),
106–128 (2025)
[9] Kerbl, B., Kopanas, G., Leimk¨uhler, T., Dret-
takis, G.: 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions
on Graphics (TOG) 42(4), 1–14 (2023)
[10] Zhan, C., Zhang, Y., Lin, Y., Wang, G.,
Wang, H.: Rdg-gs: Relative depth guidance
with gaussian splatting for real-time sparse-
view 3d rendering. International Journal of
Computer Vision (2026)
[11] Yu,
Z.,
Chen,
A.,
Huang,
B.,
Sattler,
T., Geiger, A.: Mip-splatting: Alias-free 3d
gaussian splatting. In: Proceedings of the
IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 19447–19456
(2024)
[12] Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang,
L., Lin, D., Dai, B.: Scaffold-gs: Structured
3d gaussians for view-adaptive rendering. In:
Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition,
pp. 20654–20664 (2024)
[13] Ren, K., Jiang, L., Lu, T., Yu, M., Xu, L.,
Ni, Z., Dai, B.: Octree-gs: Towards consis-
tent real-time rendering with lod-structured
3d gaussians. IEEE Transactions on Pattern
Analysis and Machine Intelligence (2025)
[14] Zhang, D., Wang, C., Wang, W., Li, P., Qin,
M., Wang, H.: Gaussian in the wild: 3d gaus-
sian splatting for unconstrained image collec-
tions. In: European Conference on Computer
Vision, pp. 341–359 (2024). Springer
[15] Kerbl, B., Meuleman, A., Kopanas, G., Wim-
mer, M., Lanvin, A., Drettakis, G.: A hier-
archical 3d gaussian representation for real-
time rendering of very large datasets. ACM
Transactions on Graphics (TOG) 43(4), 1–15
(2024)
[16] Li, H., Li, J., Zhang, D., Wu, C., Shi,
J., Zhao, C., Feng, H., Ding, E., Wang,
J., Han, J.: Vdg: vision-only dynamic gaus-
sian for driving simulation. arXiv preprint
arXiv:2406.18198 (2024)
[17] Chen, Y., Gu, C., Jiang, J., Zhu, X., Zhang,
L.: Periodic vibration gaussian: Dynamic
urban scene reconstruction and real-time
rendering. arXiv preprint arXiv:2311.18561
(2023)
[18] Huang, N., Wei, X., Zheng, W., An, P., Lu,
M., Zhan, W., Tomizuka, M., Keutzer, K.,
Zhang, S.: S3gaussian: Self-supervised street
gaussians
for
autonomous
driving.
arXiv
preprint arXiv:2405.20323 (2024)
[19] Yan, Y., Lin, H., Zhou, C., Wang, W.,
Sun, H., Zhan, K., Lang, X., Zhou, X.,
Peng,
S.:
Street
gaussians
for
model-
ing dynamic urban scenes. arXiv preprint
arXiv:2401.01339 (2024)
[20] Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun,
D., Yang, M.-H.: Drivinggaussian: Composite
gaussian splatting for surrounding dynamic
19

<!-- page 20 -->
autonomous driving scenes. In: Proceedings
of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 21634–
21643 (2024)
[21] Chen, Z., Yang, J., Huang, J., Lutio, R.,
Esturo, J.M., Ivanovic, B., Litany, O., Goj-
cic,
Z.,
Fidler,
S.,
Pavone,
M.,
et
al.:
Omnire: Omni urban scene reconstruction.
In: The Thirteenth International Conference
on Learning Representations (2025)
[22] Ma, Z., Jiang, J., Chen, Y., Zhang, L.:
B\’eziergs: Dynamic urban scene reconstruc-
tion with b\’ezier curve gaussian splatting.
arXiv preprint arXiv:2506.22099 (2025)
[23] Sun, S., Zhao, C., Sun, Z., Chen, Y.V., Chen,
M.: Splatflow: Self-supervised dynamic gaus-
sian splatting in neural motion flow field for
autonomous driving. In: Proceedings of the
Computer Vision and Pattern Recognition
Conference, pp. 27487–27496 (2025)
[24] Xu, J., Deng, K., Fan, Z., Wang, S., Xie, J.,
Yang, J.: Ad-gs: Object-aware b-spline gaus-
sian splatting for self-supervised autonomous
driving.
arXiv
preprint
arXiv:2507.12137
(2025)
[25] Chakravarthy, A.S., Ganesina, M.R., Hu, P.,
Leal-Taix´e, L., Kong, S., Ramanan, D., Osep,
A.: Lidar panoptic segmentation in an open
world. International Journal of Computer
Vision 133(3), 1153–1174 (2025)
[26] Martin-Brualla, R., Radwan, N., Sajjadi,
M.S., Barron, J.T., Dosovitskiy, A., Duck-
worth, D.: Nerf in the wild: Neural radiance
fields for unconstrained photo collections. In:
Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition,
pp. 7210–7219 (2021)
[27] Tancik, M., Casser, V., Yan, X., Pradhan,
S., Mildenhall, B., Srinivasan, P.P., Barron,
J.T., Kretzschmar, H.: Block-nerf: Scalable
large scene neural view synthesis. In: Pro-
ceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition,
pp. 8248–8258 (2022)
[28] Liu, T., Wang, G., Hu, S., Shen, L., Ye, X.,
Zang, Y., Cao, Z., Li, W., Liu, Z.: Mvsgaus-
sian: Fast generalizable gaussian splatting
reconstruction from multi-view stereo. In:
European Conference on Computer Vision,
pp. 37–53 (2025). Springer
[29] Jiang, K., Sivaram, V., Peng, C., Ramamoor-
thi, R.: Geometry field splatting with gaus-
sian surfels. In: Proceedings of the Computer
Vision and Pattern Recognition Conference,
pp. 5752–5762 (2025)
[30] Lin, J., Li, Z., Tang, X., Liu, J., Liu, S.,
Liu, J., Lu, Y., Wu, X., Xu, S., Yan, Y.,
et al.: Vastgaussian: Vast 3d gaussians for
large scene reconstruction. In: Proceedings
of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 5166–
5175 (2024)
[31] Liu, Y., Luo, C., Fan, L., Wang, N., Peng,
J., Zhang, Z.: Citygaussian: Real-time high-
quality large-scale scene rendering with gaus-
sians. In: European Conference on Computer
Vision, pp. 265–282 (2025)
[32] Peng, C., Tang, Y., Zhou, Y., Wang, N., Liu,
X., Li, D., Chellappa, R.: Bags: Blur agnostic
gaussian splatting through multi-scale kernel
modeling. In: European Conference on Com-
puter Vision, pp. 293–310 (2024). Springer
[33] Guo, X., Sun, J., Dai, Y., Chen, G., Ye,
X., Tan, X., Ding, E., Zhang, Y., Wang,
J.: Forward flow for novel view synthe-
sis of dynamic scenes. In: Proceedings of
the IEEE/CVF International Conference on
Computer Vision, pp. 16022–16033 (2023)
[34] Park, K., Sinha, U., Hedman, P., Barron,
J.T., Bouaziz, S., Goldman, D.B., Martin-
Brualla, R., Seitz, S.M.: Hypernerf: A higher-
dimensional representation for topologically
varying neural radiance fields. arXiv preprint
arXiv:2106.13228 (2021)
[35] Pumarola, A., Corona, E., Pons-Moll, G.,
Moreno-Noguer, F.: D-nerf: Neural radiance
fields for dynamic scenes. In: Proceedings
of the IEEE/CVF Conference on Computer
20

<!-- page 21 -->
Vision and Pattern Recognition, pp. 10318–
10327 (2021)
[36] Park, S., Son, M., Jang, S., Ahn, Y.C., Kim,
J.-Y., Kang, N.: Temporal interpolation is all
you need for dynamic neural radiance fields.
In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recogni-
tion, pp. 4212–4221 (2023)
[37] Li, Z., Wang, Q., Cole, F., Tucker, R.,
Snavely, N.: Dynibar: Neural dynamic image-
based
rendering.
In:
Proceedings
of
the
IEEE/CVF Conference on Computer Vision
and
Pattern
Recognition,
pp.
4273–4284
(2023)
[38] Lin, H., Peng, S., Xu, Z., Yan, Y., Shuai, Q.,
Bao, H., Zhou, X.: Efficient neural radiance
fields for interactive free-viewpoint video. In:
SIGGRAPH Asia 2022 Conference Papers,
pp. 1–9 (2022)
[39] Lombardi, S., Simon, T., Schwartz, G., Zoll-
hoefer, M., Sheikh, Y., Saragih, J.: Mixture
of volumetric primitives for efficient neural
rendering. ACM Transactions on Graphics
(ToG) 40(4), 1–13 (2021)
[40] Peng, S., Yan, Y., Shuai, Q., Bao, H.,
Zhou, X.: Representing volumetric videos
as dynamic mlp maps. In: Proceedings of
the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 4252–
4262 (2023)
[41] Cao, A., Johnson, J.: Hexplane: A fast repre-
sentation for dynamic scenes. In: Proceedings
of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 130–141
(2023)
[42] Luiten, J., Kopanas, G., Leibe, B., Ramanan,
D.:
Dynamic
3d
gaussians:
Tracking
by
persistent dynamic view synthesis. arXiv
preprint arXiv:2308.09713 (2023)
[43] Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang,
Y., Jin, X.: Deformable 3d gaussians for
high-fidelity monocular dynamic scene recon-
struction. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern
Recognition, pp. 20331–20341 (2024)
[44] Wu, G., Yi, T., Fang, J., Xie, L., Zhang,
X., Wei, W., Liu, W., Tian, Q., Wang, X.:
4d gaussian splatting for real-time dynamic
scene
rendering.
In:
Proceedings
of
the
IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 20310–20320
(2024)
[45] Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.:
Fsgs:
Real-time
few-shot
view
synthesis
using
gaussian
splatting.
arXiv
preprint
arXiv:2312.00451 (2023)
[46] Jiang, Y., Tu, J., Liu, Y., Gao, X., Long,
X., Wang, W., Ma, Y.: Gaussianshader:
3d gaussian splatting with shading func-
tions for reflective surfaces. arXiv preprint
arXiv:2311.17977 (2023)
[47] Wei, Y., Liu, S., Rao, Y., Zhao, W., Lu,
J., Zhou, J.: Nerfingmvs: Guided optimiza-
tion of neural radiance fields for indoor
multi-view stereo. In: Proceedings of the
IEEE/CVF
International
Conference
on
Computer Vision, pp. 5610–5619 (2021)
[48] Roessle, B., Barron, J.T., Mildenhall, B.,
Srinivasan, P.P., Nießner, M.: Dense depth
priors
for
neural
radiance
fields
from
sparse input views. In: Proceedings of the
IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 12892–12901
(2022)
[49] Wang, G., Chen, Z., Loy, C.C., Liu, Z.:
Sparsenerf: Distilling depth ranking for few-
shot novel view synthesis. arXiv preprint
arXiv:2303.16196 (2023)
[50] Turkulainen, M., Ren, X., Melekhov, I.,
Seiskari, O., Rahtu, E., Kannala, J.: Dn-
splatter: Depth and normal priors for gaus-
sian splatting and meshing. arXiv preprint
arXiv:2403.17822 (2024)
[51] Huang, B., Yu, Z., Chen, A., Geiger, A.,
Gao, S.: 2d gaussian splatting for geometri-
cally accurate radiance fields. In: ACM SIG-
GRAPH 2024 Conference Papers, pp. 1–11
(2024)
21

<!-- page 22 -->
[52] Cheng, K., Long, X., Yang, K., Yao, Y., Yin,
W., Ma, Y., Wang, W., Chen, X.: Gaussian-
pro: 3d gaussian splatting with progressive
propagation. In: International Conference on
Machine Learning, pp. 8123–8140 (2024)
[53] Peng, C., Zhang, C., Wang, Y., Xu, C.,
Xie, Y., Zheng, W., Keutzer, K., Tomizuka,
M., Zhan, W.: Desire-gs: 4d street gaussians
for static-dynamic decomposition and surface
reconstruction for urban driving scenes. In:
Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pp. 6782–6791
(2025)
[54] Sun, P., Kretzschmar, H., Dotiwalla, X.,
Chouard, A., Patnaik, V., Tsui, P., Guo, J.,
Zhou, Y., Chai, Y., Caine, B., et al.: Scal-
ability in perception for autonomous driv-
ing: Waymo open dataset. In: Proceedings
of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 2446–
2454 (2020)
[55] Geiger, A., Lenz, P., Urtasun, R.: Are we
ready for autonomous driving? the kitti
vision benchmark suite. In: 2012 IEEE Con-
ference on Computer Vision and Pattern
Recognition, pp. 3354–3361 (2012)
[56] Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu,
X., Feng, J., Zhao, H.: Depth anything v2.
Advances in Neural Information Processing
Systems 37, 21875–21911 (2024)
[57] Eftekhar, A., Sax, A., Malik, J., Zamir,
A.: Omnidata: A scalable pipeline for mak-
ing
multi-task
mid-level
vision
datasets
from
3d
scans.
In:
Proceedings
of
the
IEEE/CVF
International
Conference
on
Computer Vision, pp. 10786–10796 (2021)
[58] Wang, S., Leroy, V., Cabon, Y., Chidlovskii,
B., Revaud, J.: Dust3r: Geometric 3d vision
made easy. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern
Recognition, pp. 20697–20709 (2024)
[59] Bhat, S.F., Birkl, R., Wofk, D., Wonka, P.,
M¨uller, M.: Zoedepth: Zero-shot transfer by
combining relative and metric depth. arXiv
preprint arXiv:2302.12288 (2023)
[60] Schonberger, J.L., Frahm, J.-M.: Structure-
from-motion revisited. In: Proceedings of the
IEEE Conference on Computer Vision and
Pattern Recognition, pp. 4104–4113 (2016)
[61] Loper, M., Mahmood, N., Romero, J., Pons-
Moll, G., Black, M.J.: Smpl: a skinned multi-
person linear model. ACM Transactions on
Graphics (TOG) 34(6), 1–16 (2015)
22
