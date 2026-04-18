<!-- page 1 -->
Ditto: Building Digital Twins of Articulated Objects from Interaction
Zhenyu Jiang
Cheng-Chun Hsu
Yuke Zhu
Department of Computer Science, The University of Texas at Austin
{zhenyu,hsucc,yukez}@cs.utexas.edu
Abstract
Digitizing physical objects into the virtual world has the
potential to unlock new research and applications in em-
bodied AI and mixed reality. This work focuses on recre-
ating interactive digital twins of real-world articulated ob-
jects, which can be directly imported into virtual environ-
ments. We introduce Ditto to learn articulation model es-
timation and 3D geometry reconstruction of an articulated
object through interactive perception. Given a pair of vi-
sual observations of an articulated object before and af-
ter interaction, Ditto reconstructs part-level geometry and
estimates the articulation model of the object.
We em-
ploy implicit neural representations for joint geometry and
articulation modeling.
Our experiments show that Ditto
effectively builds digital twins of articulated objects in a
category-agnostic way. We also apply Ditto to real-world
objects and deploy the recreated digital twins in physical
simulation. Code and additional results are available at
https://ut-austin-rpl.github.io/Ditto/
1. Introduction
Synthetic data has played a steadily more vital role in
fueling emerging AI applications, from training and proto-
typing computer vision models [21, 45] to teaching robots
to perform physical tasks [2,30,61]. As modern AI models
become larger and increasingly data-hungry, virtual plat-
forms and synthetic datasets supply a massive amount of
cheap training data. For vision models to benefit from syn-
thetic data, realism is key — the distribution mismatch be-
tween the real and virtual worlds hinders the generalization
of models trained in simulation. A promising path towards
closing the reality gap is digitizing physical objects and
recreating them in virtual environments. Research on 3D
vision and SLAM [3,13,36,37,56] has made significant ad-
vances in capturing realistic objects and scenes with static
3D models. Nonetheless, the burgeoning body of embodied
AI and mixed reality research calls for interactive digital
twins of physical objects that can be spawned in simulated
Ditto
Interactive
Digital Twin
interaction
after
before
Robot Simulation
AR/VR Application
Figure 1. We build digital twins of articulated objects through in-
teractive perception. Given visual observations before and after
interaction, our method jointly reconstructs the part-level geom-
etry and articulation model of the object. Our recreated digital
twins can be spawned in physics engines and are fully interactive
in robot simulation and AR/VR applications.
environments and interact with virtual agents. Building dig-
ital twins of articulated objects is particularly challenging
as it requires not only a good understanding of its overall
geometry but the part compositions as well as the kinematic
relations between the parts.
Recent efforts in embodied AI platforms [24,25,53] have
incorporated interactive articulated objects, such as cabinets
and drawers, in simulated household environments and em-
ployed them for training virtual agents. Even so, they heav-
ily rely on graphics designers and engineers to author and
curate the object models, limiting the scalability of the as-
set acquisition process. Developing vision-based methods
to automate the estimation [1, 18] and reconstruction [35]
of articulated objects has been an active line of research, ac-
celerated by new tools developed from the 3D vision com-
munity, including geometric deep learning [20, 39, 42] and
implicit neural representations [10, 38].
The majority of
prior work focuses on solving individual components of the
problem rather than constructing a full-fledged model. Sev-
eral recent works [26,55] have studied the joint learning of

<!-- page 2 -->
part segmentation and joint estimation. However, they in-
fer part-level geometry on the point cloud which cannot be
used for physical simulation, because physical simulation
requires compact geometry of the object such as the mesh
for collision computation.
Departing from prior work, we seek the full-fledged vir-
tual recreation of articulated physical objects from unknown
categories. These digital twins of articulated objects rep-
resent the geometry and physics of individual object parts
and their articulation relations (e.g., prismatic or revolute
joints).
Category-agnostic articulation estimation from a
single image is inherently ambiguous. The parts may move
along a prismatic axis or rotate around a revolute axis de-
pending on the underlying kinematic joints. Following pi-
oneer work on the interactive perception of articulated ob-
jects [15,31], we propose to infer the digital twins from vi-
sual observations collected before and after articulated mo-
tions (see Fig. 1). This task comprises three intimately con-
nected challenges: object part segmentation based on mo-
tion cues, part reconstruction from a partial point cloud, and
articulation estimation of unknown joint types.
We introduce Ditto (Digital twin of articulated objects),
an implicit neural representation-based model that jointly
predicts part-level geometry and kinematic articulation be-
tween the parts.
We employ implicit neural representa-
tions [6,33,40,48] to encode continuous and high-resolution
3D information. Ditto is built on top of ConvONets [43],
which learns local implicit fields based on convolutional
feature grids. The input to Ditto is partial point cloud ob-
servations of an articulated object before and after interac-
tion with one of its parts. The key technical challenge is to
establish correspondences between these two partial obser-
vations. To achieve this, we encode the point clouds with
PointNet++ [44] into two sets of subsampled point features.
Then we fuse these two sets of point features with a self-
attention layer [54] and decode the fused subsampled fea-
tures into dense point features. We construct structured fea-
ture grids from the decoded point features. A local feature
can be computed from the feature grids at a query 3D co-
ordinate. We learn an implicit occupancy decoder and an
implicit segmentation decoder that maps from a 3D coordi-
nate and its local feature to the occupancy/part segmentation
label at that coordinate to reconstruct part-level geometry.
We use another set of implicit decoders that densely predict
the relative joint parameters at each query point to estimate
articulated joints. Such dense articulation prediction brings
forth more robust articulation estimation than predicting the
joint parameters globally. All the implicit decoders can be
trained end-to-end.
We evaluate our approach on two datasets of articulated
objects [1,55]. Our results demonstrate that Ditto accurately
reconstructs part-level geometry and articulation model in a
category-agnostic fashion. Ditto achieves superior results
across all datasets and metrics compared with the baselines.
Furthermore, we apply our method to real-world articulated
objects for recreating digital twins. We provide examples of
instantiating digital twins in simulation for a virtual robot to
interact and transfer the interaction back to the real world.
2. Related Work
Articulation Model Estimation. Probabilistic methods [8,
49–52] are first used to estimate the articulation relation-
ships between different parts of an articulated object. As ar-
ticulation can be ambiguous to infer from a single observa-
tion, interactive perception methods [4,12,15,22,31,32,60]
have been employed to estimate articulation from action-
generated visual stimuli. Conventional methods take a se-
ries of sensory observations as input and rely on markers
or handcrafted features to track the mobile parts. Recently,
deep learning methods have been developed for articulation
estimation from raw sensory data [1, 17, 18, 28]. Most of
these works primarily focus on predicting the articulation
parameters. In contrast, our method jointly reconstructs the
full 3D geometry and estimates the articulation model.
3D Reconstruction of Articulated Objects.
3D mod-
els of articulated objects encode articulation and geome-
try properties of the objects. Pioneer work from Huang et
al. [16] uses structure from motion to reconstruct the full
point cloud of the object and segment the point cloud us-
ing feature-based correspondence. More recently, learning-
based methods [26, 55] are developed to predict part seg-
mentation together with joint parameters. These works rea-
son about the part-level geometry on the point cloud, which
lacks the mesh information required for physical simula-
tion. A series of methods on reconstructing deformable ob-
ject [5, 58, 59] use articulated bones to represent articula-
tion. These representations loosely constrain the motions of
object parts. In contrast, the digital twins require accurate
part-level geometry and precise articulation modeling to be
simulated in a physics engine.
Closest to our work is A-SDF [35], which learns a deep
signed function for articulated objects from which a 3D
mesh can be extracted. It uses a separate latent code to
model the articulation state implicitly. Instead, our method
builds full 3D meshes for each part and models their artic-
ulations explicitly. The resultant digital twin can spawn in
virtual environments for physical interaction.
Implicit Neural Representations. Our method builds on
top of recent work on implicit neural representations [6,
33, 40].
These works encode a 3D shape with the iso-
surface of an implicit function. These implicit models are
parametrized with deep networks so that they are capable
of representing complex shapes smoothly and continuously
in high resolution. Aiming at better scalability and finer
details, several approaches [14, 27, 43] learn local implicit

<!-- page 3 -->
Pooling
Projection
Pooling
PointNet++
Encoder
Attention
Layer
Occupancy
Decoder
Articulation
Decoder
Segmentation
Decoder
PointNet++
Encoder
query
local 
feature
0
1
local 
feature
0
1
before
interaction
after
interaction
fused
point features
subsampled
point features
dense
point features
3D feature grid
query
2D feature planes
occupancy
probability
mobile
probability
dense joint
prediction
Marching 
Cube
Joint
Voting
(𝑥, 𝑦, 𝑧)
(𝑥, 𝑦, 𝑧)
Geometry
Feature
Decoder
Motion
Feature
Decoder
3D-UNet
2D-UNets
Part-Level
Reconstruction
Joint
Estimation
3D point cloud
Two-Stream Encoder
Implicit Decoders
Explicit Articulated 
Object Extraction 
Figure 2. Model architecture of Ditto. The input consists of point cloud observations before and after interaction. After a PointNet++ [44]
encoder, we fuse the subsampled point features with a simple attention layer. Then we use two independent decoders to propagate the fused
point features into two sets of dense point features for geometry reconstruction and articulation estimation separately. We construct feature
grid/planes by projecting and pooling the point features and query local features from the constructed feature grid/planes. Conditioning on
local features, we use different decoders to predict occupancy, segmentation, and joint parameters with respect to the query points.
decoders and condition the implicit representations on local
features instead of a global shape feature. Specifically, our
model extends ConvONets [43] with a stronger encoder and
a fusing module for processing two input data streams.
Physical Simulation with Articulated Objects. Physical
simulators have become a vital tool for embodied AI re-
search. A growing trend is shifting from static 3D scenes
for visual navigation [24,47,61] to interactive environments
that support physical interaction between the robot and the
objects [9,25,53]. Interactive 3D assets are the key elements
to construct these simulators. Existing interactive 3D assets
are mostly authored and refined by 3D artists [34,53,55,57]
or procedurally generated [1]. Our method builds interac-
tive digital twins of daily articulated objects directly from
visual observations. It has the potential to accelerate the
acquisition of realistic interactive 3D assets.
3. Problem Formulation
We study the problem of recreating interactive digital
twins of articulated objects from a pair of sensory observa-
tions before and after an interaction. Digital twins are com-
monly represented in standard 3D formats, such as URDF1,
such that they can be imported into physics engines. To
enable physical interaction in the virtual world, a digital
twin of an articulated object constitutes a kinematic tree,
where the nodes define the geometry and physical proper-
ties (e.g. mass and friction) of individual parts and the edges
define the kinematic joints between the parts. This work fo-
cuses on estimating part geometry and kinematic articula-
tion while setting the physical properties to default values
based on real-world statistics.
Given an articulated object from an unknown category,
1http://wiki.ros.org/urdf
we interact with the object to change the articulation state.
Without the loss of generality, we assume only one part is
moved after the interaction, which we call the mobile part.
The input to our method is a pair of point cloud observations
P1, P2 ∈RN×3 of the articulated object before and after an
interaction. N is the number of input points. The objective
is to segment and reconstruct the 3D geometry for static
and mobile parts, estimate the joint parameters that connect
these two parts, and relative change of the joint states.
For articulation estimation, we consider the 1D revolute
joints and 1D prismatic joints. We follow Li et al. [26] and
parameterize the two types of joints as follows. The param-
eters of a prismatic joint consist of the direction of the trans-
lation axis up ∈R3 and the joint state cp. The joint state
cp is defined as the relative translation distance between the
two observations. The parameters of a revolute joint consist
of the direction of the revolute axis ur ∈R3, a pivot point
q ∈R3 on the revolute axis and the joint state cr. The joint
state cr is defined as the relative rotation angle between the
two observations.
4. Method
We now present Ditto, a learning framework that builds
digital twins of articulated objects through interactive per-
ception. Ditto jointly learns part-level geometry reconstruc-
tion and articulation model estimation with structured fea-
ture grids and unified implicit neural representations. Fig. 2
illustrates the overall model architecture. Ditto consists of a
two-stream encoder that fuses two input point clouds and
multiple implicit decoders for geometry and articulation.
The model is jointly optimized with a combination of loss
functions on geometry reconstruction and articulation esti-
mation. Upon inference, we extract explicit models of ar-
ticulated objects from the implicit decoders.

<!-- page 4 -->
4.1. Two-Stream Encoder
To jointly learn the 3D reconstruction and articulation
model estimation, we need to extract features that fuse the
information from the input pair of point clouds. We build
our encoder based on ConvONets [43], the state-of-the-art
implicit representation-based 3D reconstruction method.
We use an attention layer [54] to fuse the two sets of
point features of two input point clouds. The complexity of
attention operations exhibits quadratic growth with respect
to the number of points. To process more dense point clouds
which capture finer details of the object, we use a Point-
Net++ [44] encoder µenc to obtain two sets of subsampled
point features f1 = µenc(P1) and f2 = µenc(P2), where
f1, f2 ∈RN ′×dsub, N ′ < N is the number of the subsam-
pled points, and dsub is the dimension of the subsampled
point features. A scaled dot-product attention operation is
applied to these subsampled feature points
  Attn _ {12} = \ tex t
 
{soft
max}
(\f r ac { f_1 f_2^T}{\sqrt {d_\text {sub}}})f_2, \quad f_{12} = [f_1, Attn_{12}]. (1)
The fused subsampled point features f12 ∈RN ′×2dsub is the
concatenation of the f1 and the output of attention. Then we
use two PointNet++ decoder νgeo and νart to propagate the
fused subsampled point features into dense features aligned
with the original points
  f_ \ text {geo
} =
 \nu  _\text {geo}(f_{12})\quad \text {and} \quad f_\text {art} = \nu _\text {art}(f_{12}), 
(2)
where fgeo, fart ∈RN×ddense are ddense-dim point features
aligned with P1. We use two separate sets of dense point
features because geometry reconstruction mainly exploits
the static observation, while the articulation estimation re-
lies more on the correspondence between two observations.
Also, these features are processed separately. fart is pro-
jected into 2D feature planes and fgeo is projected into voxel
grids as in the ConvONets [43]. The points that fall into
the same pixel cell or voxel cell are aggregated together via
max pooling. This projection operation greatly reduces the
computation cost while keeping the spatial distribution of
feature points. We apply the projection to three canonical
planes c and a coarse voxel grid v. The resulting feature
planes and grid are processed with independent 2D and 3D
UNets [46]. The output voxel feature grid is used for the ge-
ometry implicit decoder, and the feature planes are used for
the articulation implicit decoders. Geometry reconstruction
requires dense feature grids for fine-grained and local rea-
soning, while sparse feature planes are sufficient for articu-
lation estimation. Therefore we choose this separate feature
representation.
4.2. Implicit Decoders
Motivated by recent works that demonstrate the conti-
nuity and versatility of implicit neural representations [11,
19, 33, 40], we design implicit decoders for both geometry
and articulation reasoning. As both tasks require reason-
ing about fine-grained geometry details, we condition the
implicit decoders on local features. These local features
can be computed from the feature grid/planes using trilin-
ear/bilinear sampling given a query 3D coordinate p ∈R3.
4.2.1
Geometry Implicit Decoder
Our geometry implicit decoder is a mapping from a coor-
dinate p ∈R3 to the occupancy probability o(p) at the
coordinate. The occupancy o(p) should be 1 if the point
p is occupied by the object and 0 otherwise. We query the
local feature ψv
p from the feature grid v using trilinear sam-
pling. Conditioned on the query point coordinate p and lo-
cal feature ψv
p, our geometry implicit decoder predicts the
occupancy probability:
  \beg in
 { a ligned} f_{\theta _{o}}(\mathbf {p}, \psi _{\mathbf {p}}^\mathbf {v}) \rightarrow o(\mathbf {p}). \end {aligned} 
(3)
4.2.2
Articulation Implicit Decoders
Our articulation implicit decoders map from an arbitrary
point pin inside the object to the segmentation label and
joint parameters with respect to this point. We only con-
sider the space inside the object because articulation is only
meaningful for points in this space. We query the local fea-
ture ψc
pin from the feature planes c using bilinear sampling.
Segmentation. Since we assume that only one joint’s state
is changed due to the interaction, we can segment the ob-
ject into the static and mobile parts during each interaction.
Therefore we predict a binary segmentation label s(pin)
where 0 stands for the static part and 1 stands for the mo-
bile part. Our segmentation implicit decoder predicts the
segmentation probability conditioning on the local feature:
  \begin { al
igne d } f_{\theta _\text {seg}}(\mathbf {p}_\text {in}, \psi _{\mathbf {p}_\text {in}}^c) \rightarrow s(\mathbf {p}_\text {in}). \end {aligned} 
(4)
Joint Parameters. Even though the joint is a global prop-
erty of an articulated object, we use a per-point represen-
tation to better utilize our structured feature representation
and get a more robust estimation through voting. We share
the feature planes for articulation and segmentation predic-
tion since the articulation can be inferred from motion cues
like segmentation. First, we use an implicit decoder to pre-
dict joint type pjtype:
  \begin {a li
gned }  f_{\theta _\text {type}}(\mathbf {p}_\text {in}, \psi _{\mathbf {p}_\text {in}}^c) \rightarrow p_{j_\text {type}}(\mathbf {p}_\text {in}). \end {aligned} 
(5)
Then we use two implicit decoders to predict parameters
and states of prismatic joints and revolute joints. The pris-
matic joint is defined by its translation axis direction, a 3D
unit vector up. The joint state is the translation distance
cp resulting from the interaction. Revolute joint parameters
include the rotation axis direction ur. Different from the

<!-- page 5 -->
prismatic joint, the position of the revolute joint axis also
matters. We follow Li et al. [26] and define the joint po-
sition with respect to point pin as the projection of pin to
the axis, represented by a 3D unit vector for the projection
direction dr
pin and a scalar hr
pin for the projection distance.
The joint state of the revolute joint is the rotation angle cr
resulting from the interaction. We directly predict these pa-
rameters with the implicit decoders:
  \begin  {ali gn
ed} f _{\t heta
 _{\text  {par am
}_p} } (\ma th
bf { p}
_\te xt {in}, \psi _{\mathbf {p}_\text {in}}^c) &\rightarrow [\mathbf {u}^p, c^p], \\ f_{\theta _{\text {param}_r}}(\mathbf {p}_\text {in}, \psi _{\mathbf {p}_\text {in}}^c) &\rightarrow [\mathbf {u}^r, \mathbf {d}_{\mathbf {p}_\text {in}} ^r, h_{\mathbf {p}_\text {in}}^r, c^r] . \end {aligned} 
(6)
4.3. Training
Our method does not assume known joint types during
inference. Therefore, the model can be trained with data
from different categories. The loss for training consists of
two parts: the geometry loss and the joint loss. The geom-
etry loss optimizes the part-level geometry reconstruction,
and the joint estimation loss optimizes joint estimation.
Geometry Loss. We apply standard binary cross-entropy
losses on occupancy and segmentation predictions,
  \b e
g
i
n {aligne d} \mat
hcal  
{
L}_
\text {occ}  &= \sum _{\mathbf {p}} BCE(o(\mathbf {p}), \hat {o}(\mathbf {p})), \\ \mathcal {L}_\text {seg} &= \sum _{\mathbf {p}_\text {in}} BCE(s(\mathbf {p}_\text {in}), \hat {s}(\mathbf {p}_\text {in})), \end {aligned} 
(7)
where ˆo(p) is the ground truth occupancy at p and ˆs(pin) is
the ground truth segmentation label at pin.
Joint Loss. We have three implicit decoders that predict
joint type, prismatic joint parameters, and revolute joint pa-
rameters respectively. For joint type prediction, we also ap-
ply the standard binary cross entropy loss. The joint type
loss is denoted as Ltype = P
pin BCE(pjtype(pin), ˆt), where
ˆt is the ground truth joint type.
Prismatic Joint. We penalize the orientation difference be-
tween the estimated joint axis and the ground truth one with
loss Lorip = arccos(up · ˆup). The state prediction is opti-
mized with simple ℓ1 loss Lstatep = |cp−ˆcp|, where ˆcp is the
ground truth joint state. Besides, we also minimize the dif-
ferent between the predicted displacement and ground truth
one. The state prediction and parameter prediction can be
jointly optimized with this loss Ldispp = ||cpup −ˆcpˆup||.
All together we have joint loss of the prismatic joint
  \begi n
 
{al
igned}  \mathca l  {L}_{\text {param}_p} = \sum _{\mathbf {p}_\text {in}} (\mathcal {L}_{\text {ori}_p} + \mathcal {L}_{\text {state}_p} + \mathcal {L}_{\text {disp}_p}). \end {aligned} 
(8)
Revolute Joint. The loss for axis orientation and joint state
of the revolute joint is the same as the prismatic joint, de-
noted as Lorir and Lstater. We apply the same loss for ori-
entation of the projection direction dr
pin and projection dis-
tance hr
pin, which are added together to form Lposr. Thanks
to our dense joint representation, displacement based loss
can be also applied to revolute joint parameters prediction.
For each point pin, we compute predicted rotation matrix
Rpin and ground truth one ˆRpin based on predicted and
ground truth axis orientation and rotation angle. We also
locate the estimated pivot point on the axis qpin = pin +
hr
pindr
pin. Then the displacement can be computed as lpin =
Rpin(pin −qpin) + qpin. The ground truth displacement ˆlpin
can be computed similarly with the ground truth parameters.
And the displacement loss Ldispr = ||lpin −ˆlpin||. More-
over, we apply an extra loss on rotation matrix following
ScrewNet [18] Lrotr = ||I3,3 −Rpin ˆRT
pin||. All together we
have joint loss of the revolute joint
  \begi n
 
{al
igned}  \mathca l  {L}_ { \text { param}_r} = \sum _{\mathbf {p}_\text {in}} (\mathcal {L}_{\text {ori}_r} + \mathcal {L}_{\text {state}_r} + \mathcal {L}_{\text {pos}_r} + \mathcal {L}_{\text {disp}_r} + \mathcal {L}_{\text {rot}_r}). \end {aligned} 
(9)
Since the joint type is unknown, we need to dynamically
apply the prismatic or revolute joint loss based on the joint
type. The full loss is
  \beg i n {a l igned }  \mathcal  {L} = \mathcal {L}_\text {occ} + \mathcal {L}_\text {seg} + \mathcal {L}_\text {type} + \mathbbm {1}_{p} \mathcal {L}_{\text {param}_p} + \mathbbm {1}_{r} \mathcal {L}_{\text {param}_r}, \end {aligned} 
(10)
where Ip and Ir are indicators of whether the ground truth
joint type is prismatic or revolute.
4.4. Explicit Articulated Object Extraction
We need to extract the explicit part-level meshes and ar-
ticulation model from the learned implicit representation to
build interactive digital twins that can be spawned in a vir-
tual environment.
Part-Level Mesh Extraction.
To reconstruct part-level
meshes, we mask the occupancy query results with segmen-
tation query results. The occupancy query results of mobile
part and static part are
  \be g in {al i gned} o_{m} ( \mathb
f {p} )  &= \m a thbbm {1}[o ( \mathbf {p}) > t_\text {occ}] \mathbbm {1}[s (\mathbf {p}) > t_\text {seg}], \\ o_{s}(\mathbf {p}) &= \mathbbm {1}[o(\mathbf {p}) > t_\text {occ}] \mathbbm {1}[s (\mathbf {p}) \leq t_\text {seg}], \end {aligned} 
(11)
where tocc and tseg are thresholds for the predicted oc-
cupancy and segmentation probability.
Then we apply
Multiresolution IsoSurface Extraction [33] and Marching
Cube [29] to extract per-part surface meshes.
Global Articulation Model Extraction. We use a simple
average voting strategy to aggregate the dense joint predic-
tion. During the mesh extraction, we can sample numerous
points inside the mesh with their predicted label. Because
the object’s motion determines the articulation model, we
only let points inside the mobile part vote for the global
joint. For joint axis direction and joint state of both types of
joint, we average all mobile points’ predictions. As for the
position of the revolute axis, we compute the pivot point co-
ordinate of each mobile point using the predicted projection
direction and distance. Then we average the results of all
mobile points and get the estimated pivot point on the axis.

<!-- page 6 -->
Dataset
Method
Geometry
Joint
Whole
Chamfer Distance ↓
Mobile
Chamfer Distance ↓
Prismatic
Revoulute
Angle Err ↓
Angle Err ↓
Pos Err ↓
Synthetic
Dataset [1]
A-SDF [35]
2.48
-
-
-
-
Correspondence [7]
2.13
93.2
10.3
46.5
0.46
Global Joint [18]
0.54
37.7
0.69
52.0
0.13
Ditto (Ours)
0.38
0.21
0.06
0.72
0.03
Shape2Motion
[55]
Correspondence [7]
2.22
35.7
15.2
45.5
0.28
Global Joint [18]
0.90
64.9
1.36
79.8
0.17
Share Feature
0.75
10.7
0.07
2.22
0.04
Concat Fusion
0.97
3.09
0.17
3.13
0.03
Share Decoder
0.68
3.30
0.19
1.93
0.02
Ditto (Ours)
0.72
0.42
0.08
1.36
0.02
Table 1. Quantitative results of geometry reconstruction and articulation estimation on Shape2Motion [55] and synthetic [1] datasets.
5. Experiments
We examine Ditto’s ability to recreate digital twins of
articulated objects. We first perform systematic quantitative
evaluations on two 3D asset datasets, showing that Ditto can
accurately reconstruct the geometry and estimate the artic-
ulation model. We then qualitatively show that our method
generalizes well to objects in the real world.
5.1. Datasets
We conduct experiments on two 3D articulated object
datasets, the synthetic objects dataset provided by Ab-
batematteo et al. [1] and the Shape2Motion dataset [55].
The synthetic dataset contains procedurally generated ar-
ticulated objects. Shape2Motion contains human-designed
objects. We select four categories from each dataset. For
Shape2Motion dataset, we choose four categories with
more than 30 instances. We choose 4 out of 6 categories
for the synthetic dataset because the other two are very sim-
ilar to the chosen ones. During data generation, we ran-
domly spawn an object in simulation and set the object
into random start and end states to mimic articulated mo-
tions. In each state, we fuse multi-view depth images into
point cloud observation. Even though we use multi-view
depth images, the point cloud may still be incomplete due
to the self-occlusion of the objects. We generate occupancy
data points for each part separately for ground truth geome-
try and aggregate the samples to get shape-level occupancy
and segmentation. Ground-truth articulation is directly ac-
quired from the simulator and the ground-truth occupancy
is queried from the ground truth mesh as in [43].
5.2. Baselines
A-SDF. A-SDF [35] is the closest work to ours, given that
no existing method is designed specifically for the full-
fledged virtual recreation of articulated objects. There are
two main differences between A-SDF and our work. First,
A-SDF is a category-level model that assumes the same
kinematic tree structures of objects in the same category.
Second, it estimates the articulation model implicitly rather
than explicitly. Accordingly, we train one A-SDF model for
each category and evaluate only the geometry reconstruc-
tion result on the synthetic dataset.
Correspondence. We first train an FCGF [7] feature ex-
tractor on the whole dataset. Then we use the extracted
features to find point correspondence across the observa-
tions before and after the interaction. An articulation model
is fitted based on correspondence and using the non-linear
least square algorithm. Besides, we use correspondence to
compute the moving distance of every point and segment
the mobile points with a threshold of 0.02 on this distance.
We reconstruct the mesh of segmented points using a Con-
vONet [43] trained for part reconstruction. This baseline
has the same output as Ditto.
Global Joint. To validate our choice of dense joint repre-
sentation, we modify our model and use decoders that pre-
dict joint parameters from a global feature. Since the pivot
point of a revolute joint axis is ambiguous, i.e., it can move
along the axis, we adopt the screw-based joint parametriza-
tion in ScrewNet [18]. We also apply the loss function of
ScrewNet to train the model.
In addition to the external baselines above, we also use
the following ablated versions of our model to validate our
design choices:
Concat Fusion. Instead of the attention-based fusion, it
directly concatenates the structured features of the pair of
point clouds and conditions the local implicit decoders on
the concatenated features.
Share Feature. We use 3D feature grids for occupancy pre-
diction and 2D feature planes for segmentation and joint
prediction in our current model. This ablated version shares

<!-- page 7 -->
Correspondence
Global joint
Ditto
Ground 
Truth
After
interaction
Before
interaction
Correspondence
Global joint
Ditto
Ground 
Truth
After
interaction
Before
interaction
A-SDF
Figure 3. Reconstructed unseen articulated objects in Shape2Motion [55] (top) and synthetic [1] (bottom) dataset. Static parts are colored
grey while mobile parts are colored green. We also visualize the estimated joint with the red arrow.
the 3D and 2D features for both geometry and articulation
prediction.
Share Decoder. In our current model, we use two separate
decoders in PointNet++ for geometry and articulation. This
ablated version uses a shared decoder instead.
5.3. Evaluation Metrics
Part-level Geometry. To evaluate the quality of the recon-
structed part-level mesh, we use Chamfer-ℓ1 distance (CD)
as the evaluation metric. Apart from the CD between the
whole reconstructed mesh and the ground truth, we also
evaluate the CD of the segmented mobile part because it
is the only interactable region of the object. CD shown are
multiplied by 1000 as in A-SDF [35].
Articulation Model. For both types of joints, we measure
the axis orientation error (Angle Err). For the revolute joint,
we also measure the axis position error (Pos Err) using the
minimum distance between the predicted and ground truth
rotation axis.
5.4. Articulated Object Reconstruction
The quantitative results are shown in Tab. 1. On both
datasets, Ditto gets significantly better results on all metrics
compared with the baselines. Both the Correspondence [7]
and Global Joint [18] baselines perform poorly on articu-
lation estimation. As shown in Fig. 3, while the baseline
methods produce overall well-reconstructed shapes, the pre-
dicted mobile parts have many artifacts. In contrast, Ditto
achieves precise part-level geometry reconstruction as well
as accurate joint estimation.
Due to the two-stage design, the Correspondence base-
line highly relies on a learned disentangled feature repre-
sentation at the beginning. Bad initial feature representation
is prone to inaccurate correspondence and articulation esti-
mation. In comparison, Ditto does not suffer from such a
bottleneck as an end-to-end method. The Global Joint base-
line performs poorly mainly due to the high variance of di-
rect global joint regression. Failure of joint estimation also
harms segmentation prediction because the joint parameter
decoders and the segmentation decoder share the same fea-
ture planes. Differently, Ditto predicts joint parameters with
respect to each point. The dense predictions are aggregated

<!-- page 8 -->
Real object
Before
interaction
After
interaction
Digital
twin
Import
Simulated
interaction
Real-robot
interaction
Transfer
Figure 4. Real-world results. We use Ditto trained in simulated
datasets to build the digital twin of these physical objects. The
recreated faucet model is imported into a physical simulator. The
robot interacts with the virtual faucet and transfers its actions back
into the real world to manipulate the physical faucet.
into the final joint estimation and thus lead to a more robust
and accurate result.
To compare with A-SDF [35], we provide the shape re-
construction results on the synthetic dataset. When it comes
to the whole Chamfer distance, Ditto surpasses A-SDF by a
notably large margin. As visualized in Fig. 3, A-SDF fails
to reconstruct the shape details of unseen objects, especially
the objects with prismatic joints. In contrast, Ditto accu-
rately reconstructs the whole object and the fine-grained
geometry detail like the handles of the cabinet door and
drawer. A key difference between A-SDF and ours is that
we use a feedforward model while A-SDF uses test-time
optimization to find the articulation and shape codes. The
inferior performance of A-SDF is due to the interference
of articulation code and shape code in test-time optimiza-
tion. Note that A-SDF requires separate training for each
category while Ditto is a category-agnostic method. Thus,
the task should be more challenging for Ditto. Furthermore,
Ditto can extract explicit articulation and part-level geome-
try while A-SDF encodes the articulation model implicitly.
More comparison with A-SDF is presented in the appendix.
5.5. Ablation Studies
As shown in Tab. 1, Ditto achieves superior or at least
on-par performance on all metrics. The mobile Chamfer
distance (CD) of Ditto is substantially lower (better) than
the ablated versions. Mobile CD measures the quality of
the reconstructed mobile part, which is vital for simulating
interaction. Share Feature baseline has the worst perfor-
mance in Mobile CD. We observe that using the same 3D
and 2D features for geometry and articulation makes train-
ing unstable, and 2D features would harm the reconstruc-
tion due to the loss of spatial information after projection.
Concat Fusion does not reason about correspondence ex-
plicitly and thus shows inferior performance on all metrics
compared with Ditto. Finally, the Share Decoder baseline
applies one decoder for both geometry and motion features.
This decoder needs to reason about geometry and articu-
lation simultaneously. Suffering from a limited capacity,
this baseline obtains sub-optimal performance on Mobile-
CD and joint angle errors. Qualitative results and analysis
of ablation study are in the appendix.
5.6. Real-World Experiments
Finally, we use Ditto to recreate digital twins of real-
world objects. We choose three daily objects, a toy cabinet,
a laptop, and a faucet. The results are shown in Fig. 4. The
results have some artifacts due to the noisy and incomplete
input point clouds from depth cameras. Despite these arti-
facts, Ditto can generally reconstruct the geometry and ar-
ticulation of these physical objects. Moreover, we import
the digital twin of the faucet into Robosuite [62], a robot
learning simulation framework. We use a simulated robot
arm to interact with the digital twin and transfer the actions
back to the real world after calibrating the simulated and
real robot frames. The video of this experiment is provided
on the project website. With Ditto we can recreate a real-
world articulated object to the digital twin in a virtual envi-
ronment and map the interactions with the digital twin back
to actions in the real world.
5.7. Limitations
Kinematic tree. Currently Ditto only segments the object
into two parts, the mobile and static ones. We hope to ex-
tend our method to reconstruct the full kinematic tree of a
composite object with multiple joints and parts via consec-
utive interactions and aggregation of model inference after
each interaction.
Active perception. We use interactions to create novel sen-
sory data for inferring articulation. These interactions are
either by setting the joint states (in simulation) or by human

<!-- page 9 -->
(real world). We hope to develop algorithms that enable an
agent to autonomously interact with objects to actively col-
lect data.
6. Conclusion
We introduce Ditto, an implicit neural representation-
based model for recreating digital twins of articulated ob-
jects through interactive perception. Ditto is an end-to-end
model that jointly learns full-fledged geometry reconstruc-
tion and articulation estimation from two visual inputs be-
fore and after articulated motions. Results show that Ditto
achieves significantly more accurate results on geometry
and articulation reasoning over baselines. Furthermore, we
demonstrate Ditto generalizes to real-world objects, and we
can directly spawn the recreated digital twins in interac-
tive simulation. These results manifest the potential of au-
tonomous digital twin building in empowering embodied AI
research and AR/VR applications.
Acknowledgments
We would like to thank Yifeng Zhu for his help with real
robot experiments. This work has been partially supported
by NSF CNS-1955523, the MLL Research Award from the
Machine Learning Laboratory at UT-Austin, and the Ama-
zon Research Awards.
References
[1] Ben Abbatematteo, Stefanie Tellex, and George Konidaris.
Learning to generalize kinematic models to novel objects. In
Proceedings of the 3rd Conference on Robot Learning, 2019.
1, 2, 3, 6, 7, 12, 13
[2] OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek
Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki,
Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray,
et al. Learning dexterous in-hand manipulation. The Inter-
national Journal of Robotics Research, 39(1):3–20, 2020. 1
[3] Armen Avetisyan, Manuel Dahnert, Angela Dai, Manolis
Savva, Angel X Chang, and Matthias Nießner. Scan2cad:
Learning cad model alignment in rgb-d scans. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 2614–2623, 2019. 1
[4] Jeannette Bohg, Karol Hausman, Bharath Sankaran, Oliver
Brock,
Danica Kragic,
Stefan Schaal,
and Gaurav S
Sukhatme. Interactive perception: Leveraging action in per-
ception and perception in action.
IEEE Transactions on
Robotics, 33(6):1273–1291, 2017. 2
[5] Aljaz Bozic, Pablo Palafox, Michael Zollhofer, Justus Thies,
Angela Dai, and Matthias Nießner.
Neural deformation
graphs for globally-consistent non-rigid reconstruction. In
CVPR, pages 1450–1459, 2021. 2
[6] Zhiqin Chen and Hao Zhang. Learning implicit fields for
generative shape modeling. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 5939–5948, 2019. 2
[7] Christopher Choy, Jaesik Park, and Vladlen Koltun. Fully
convolutional geometric features.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 8958–8966, 2019. 6, 7, 12
[8] Anthony Dearden and Yiannis Demiris. Learning forward
models for robots. In IJCAI, volume 5, page 1440, 2005. 2
[9] Matt Deitke, Winson Han, Alvaro Herrasti, Aniruddha
Kembhavi, Eric Kolve, Roozbeh Mottaghi, Jordi Salvador,
Dustin Schwenk, Eli VanderBilt, Matthew Wallingford, et al.
Robothor: An open simulation-to-real embodied ai platform.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 3164–3174, 2020. 3
[10] Boyang Deng, John P Lewis, Timothy Jeruzalski, Gerard
Pons-Moll, Geoffrey Hinton, Mohammad Norouzi, and An-
drea Tagliasacchi. Nasa: Neural articulated shape approxi-
mation. In European Conference on Computer Vision, pages
612–628, 2020. 1
[11] Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez,
Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee,
Igor Mordatch, and Jonathan Tompson. Implicit behavioral
cloning. arXiv preprint arXiv:2109.00137, 2021. 4
[12] Samir Yitzhak Gadre, Kiana Ehsani, and Shuran Song. Act
the part: Learning interaction strategies for articulated object
part discovery.
In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 15752–15761,
2021. 2
[13] Andreas Geiger, Julius Ziegler, and Christoph Stiller. Stere-
oscan: Dense 3d reconstruction in real-time. In 2011 IEEE
intelligent vehicles symposium (IV), pages 963–968. Ieee,
2011. 1
[14] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna,
and Thomas Funkhouser. Local deep implicit functions for
3d shape.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4857–
4866, 2020. 2
[15] Karol Hausman, Scott Niekum, Sarah Osentoski, and Gau-
rav S Sukhatme.
Active articulation model estimation
through interactive perception. In 2015 IEEE International
Conference on Robotics and Automation (ICRA), pages
3305–3312. IEEE, 2015. 2
[16] Xiaoxia Huang, Ian Walker, and Stan Birchfield. Occlusion-
aware reconstruction and manipulation of 3d articulated ob-
jects. In 2012 IEEE International Conference on Robotics
and Automation, pages 1365–1371. IEEE, 2012. 2
[17] Ajinkya Jain, Stephen Giguere, Rudolf Lioutikov, and Scott
Niekum. Distributional depth-based estimation of object ar-
ticulation models. arXiv preprint arXiv:2108.05875, 2021.
2
[18] Ajinkya
Jain,
Rudolf
Lioutikov,
and
Scott
Niekum.
Screwnet: Category-independent articulation model estima-
tion from depth images using screw theory. arXiv preprint
arXiv:2008.10518, 2020. 1, 2, 5, 6, 7
[19] Zhenyu Jiang, Yifeng Zhu, Maxwell Svetlik, Kuan Fang,
and Yuke Zhu. Synergies between affordance and geome-
try: 6-dof grasp detection via implicit representations. arXiv
preprint arXiv:2104.01542, 2021. 4
[20] Angjoo Kanazawa, Michael J Black, David W Jacobs, and
Jitendra Malik. End-to-end recovery of human shape and

<!-- page 10 -->
pose. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 7122–7131, 2018. 1
[21] Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci,
Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba,
and Sanja Fidler. Meta-sim: Learning to generate synthetic
datasets.
In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 4551–4560, 2019. 1
[22] Dov Katz and Oliver Brock. Manipulating articulated objects
with interactive perception. In 2008 IEEE International Con-
ference on Robotics and Automation, pages 272–277. IEEE,
2008. 2
[23] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization.
arXiv preprint arXiv:1412.6980,
2014. 12
[24] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt,
Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Ab-
hinav Gupta, and Ali Farhadi. Ai2-thor: An interactive 3d
environment for visual ai. arXiv preprint arXiv:1712.05474,
2017. 1, 3
[25] Chengshu Li, Fei Xia, Roberto Mart´ın-Mart´ın, Michael Lin-
gelbach, Sanjana Srivastava, Bokui Shen, Kent Vainio, Cem
Gokmen, Gokul Dharan, Tanish Jain, et al.
igibson 2.0:
Object-centric simulation for robot learning of everyday
household tasks. arXiv preprint arXiv:2108.03272, 2021. 1,
3
[26] Xiaolong Li, He Wang, Li Yi, Leonidas J Guibas, A Lynn
Abbott, and Shuran Song. Category-level articulated object
pose estimation. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
3706–3715, 2020. 1, 2, 3, 5
[27] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua,
and Christian Theobalt. Neural sparse voxel fields. arXiv
preprint arXiv:2007.11571, 2020. 2
[28] Qihao Liu, Weichao Qiu, Weiyao Wang, Gregory D Hager,
and Alan L Yuille. Nothing but geometric constraints: A
model-free method for articulated object pose estimation.
arXiv preprint arXiv:2012.00088, 2020. 2
[29] William E Lorensen and Harvey E Cline. Marching cubes:
A high resolution 3d surface construction algorithm. ACM
siggraph computer graphics, 21(4):163–169, 1987. 5
[30] Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael Laskey,
Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and Ken
Goldberg. Dex-net 2.0: Deep learning to plan robust grasps
with synthetic point clouds and analytic grasp metrics. arXiv
preprint arXiv:1703.09312, 2017. 1
[31] Roberto Mart´ın-Mart´ın and Oliver Brock. Online interactive
perception of articulated objects with multi-level recursive
estimation based on task-specific priors. In 2014 IEEE/RSJ
International Conference on Intelligent Robots and Systems,
pages 2494–2501. IEEE, 2014. 2
[32] Roberto Mart´ın-Mart´ın, Sebastian H¨ofer, and Oliver Brock.
An integrated approach to visual perception of articulated
objects. In 2016 IEEE International Conference on Robotics
and Automation (ICRA), pages 5091–5097. IEEE, 2016. 2
[33] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Se-
bastian Nowozin, and Andreas Geiger. Occupancy networks:
Learning 3d reconstruction in function space. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 4460–4470, 2019. 2, 4, 5
[34] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna
Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A large-
scale benchmark for fine-grained and hierarchical part-level
3d object understanding. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 909–918, 2019. 3
[35] Jiteng Mu, Weichao Qiu, Adam Kortylewski, Alan Yuille,
Nuno Vasconcelos, and Xiaolong Wang. A-sdf: Learning
disentangled signed distance functions for articulated shape
representation. arXiv preprint arXiv:2104.07645, 2021. 1,
2, 6, 7, 8, 12
[36] Richard A Newcombe, Dieter Fox, and Steven M Seitz.
Dynamicfusion: Reconstruction and tracking of non-rigid
scenes in real-time. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 343–352,
2015. 1
[37] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Pushmeet
Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon.
Kinectfusion: Real-time dense surface mapping and track-
ing. In 2011 10th IEEE international symposium on mixed
and augmented reality, pages 127–136. IEEE, 2011. 1
[38] Atsuhiro Noguchi, Xiao Sun, Stephen Lin, and Tatsuya
Harada.
Neural articulated radiance field.
arXiv preprint
arXiv:2104.03110, 2021. 1
[39] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Pe-
ter Gehler, and Bernt Schiele. Neural body fitting: Unify-
ing deep learning and model based human pose and shape
estimation. In 2018 international conference on 3D vision
(3DV), pages 484–494. IEEE, 2018. 1
[40] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove. Deepsdf: Learning con-
tinuous signed distance functions for shape representation.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 165–174, 2019. 2, 4
[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,
Andreas Kopf, Edward Yang, Zachary DeVito, Martin Rai-
son, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An im-
perative style, high-performance deep learning library. In H.
Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc, E.
Fox, and R. Garnett, editors, Advances in Neural Informa-
tion Processing Systems 32, pages 8024–8035. Curran Asso-
ciates, Inc., 2019. 12
[42] Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, and Kostas
Daniilidis. Learning to estimate 3d human pose and shape
from a single color image. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
459–468, 2018. 1
[43] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc
Pollefeys, and Andreas Geiger.
Convolutional occupancy
networks. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceed-

<!-- page 11 -->
ings, Part III 16, pages 523–540. Springer, 2020. 2, 3, 4,
6
[44] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Point-
net++: Deep hierarchical feature learning on point sets in a
metric space. arXiv preprint arXiv:1706.02413, 2017. 2, 3,
4
[45] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit
Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb,
and Joshua M. Susskind. Hypersim: A photorealistic syn-
thetic dataset for holistic indoor scene understanding. In In-
ternational Conference on Computer Vision (ICCV), 2021.
1
[46] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
Net: Convolutional networks for biomedical image segmen-
tation. In International Conference on Medical image com-
puting and Computer-Assisted Intervention, pages 234–241,
2015. 4
[47] Manolis Savva, Angel X Chang, Alexey Dosovitskiy,
Thomas Funkhouser, and Vladlen Koltun.
Minos: Multi-
modal indoor simulator for navigation in complex environ-
ments. arXiv preprint arXiv:1712.03931, 2017. 3
[48] Vincent Sitzmann, Julien Martel, Alexander Bergman, David
Lindell, and Gordon Wetzstein. Implicit neural representa-
tions with periodic activation functions. Advances in Neural
Information Processing Systems, 33, 2020. 2
[49] J¨urgen Sturm, Christian Plagemann, and Wolfram Burgard.
Adaptive body scheme models for robust robotic manipula-
tion. In Robotics: Science and systems. Zurich, 2008. 2
[50] Jurgen Sturm, Christian Plagemann, and Wolfram Burgard.
Unsupervised body scheme learning through self-perception.
In 2008 IEEE International Conference on Robotics and Au-
tomation, pages 3328–3333. IEEE, 2008. 2
[51] J¨urgen Sturm, Vijay Pradeep, Cyrill Stachniss, Christian
Plagemann, Kurt Konolige, and Wolfram Burgard. Learning
kinematic models for articulated objects. In Twenty-First In-
ternational Joint Conference on Artificial Intelligence, 2009.
2
[52] J¨urgen Sturm, Cyrill Stachniss, and Wolfram Burgard.
A
probabilistic framework for learning kinematic models of ar-
ticulated objects. Journal of Artificial Intelligence Research,
41:477–526, 2011. 2
[53] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans,
Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam,
Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan,
Vladimir Vondrus, Sameer Dharur, Franziska Meier, Woj-
ciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Ji-
tendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0:
Training home assistants to rearrange their habitat.
arXiv
preprint arXiv:2106.14405, 2021. 1, 3
[54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In Advances in neural
information processing systems, pages 5998–6008, 2017. 2,
4
[55] Xiaogang Wang, Bin Zhou, Yahao Shi, Xiaowu Chen, Qin-
ping Zhao, and Kai Xu. Shape2motion: Joint analysis of
motion parts and attributes from 3d shapes. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 8876–8884, 2019. 1, 2, 3, 6, 7, 12,
13
[56] Jiajun Wu, Yifan Wang, Tianfan Xue, Xingyuan Sun,
William T Freeman, and Joshua B Tenenbaum.
Marrnet:
3d shape reconstruction via 2.5 d sketches. arXiv preprint
arXiv:1711.03129, 2017. 1
[57] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao
Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan,
He Wang, et al. Sapien: A simulated part-based interactive
environment. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 11097–
11107, 2020. 3
[58] Gengshan Yang, Deqing Sun, Varun Jampani, Daniel Vlasic,
Forrester Cole, Huiwen Chang, Deva Ramanan, William T
Freeman, and Ce Liu.
Lasr: Learning articulated shape
reconstruction from a monocular video.
In CVPR, pages
15980–15989, 2021. 2
[59] Gengshan Yang, Minh Vo, Neverova Natalia, Deva Ra-
manan, Vedaldi Andrea, and Joo Hanbyul. Banmo: Building
animatable 3d neural models from many casual videos. arXiv
preprint arXiv:2112.12761, 2021. 2
[60] Li Yi, Haibin Huang, Difan Liu, Evangelos Kalogerakis, Hao
Su, and Leonidas Guibas. Deep part induction from articu-
lated object pairs. arXiv preprint arXiv:1809.07417, 2018.
2
[61] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Ab-
hinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven vi-
sual navigation in indoor scenes using deep reinforcement
learning. In IEEE International Conference on Robotics and
Automation (ICRA), pages 3357–3364. IEEE, 2017. 1, 3
[62] Yuke Zhu, Josiah Wong, Ajay Mandlekar, and Roberto
Mart´ın-Mart´ın.
robosuite: A modular simulation frame-
work and benchmark for robot learning. In arXiv preprint
arXiv:2009.12293, 2020. 8

<!-- page 12 -->
Ditto
Ground Truth
Ditto
Ground Truth
Drawer
Fridge
Microwave
Stapler
Figure 5. Qualitative results of generalizing to unseen categories.
A. Implementation Details
We use the Shape2Motion dataset [55] and the Synthetic
dataset [1]. The Shape2Motion dataset is licensed under
the GNU General Public License v3.0. We sample 8,192
points for each input point cloud. In each iteration, we sam-
ple 2,048 pairs of p and corresponding occupancy. We also
512 pairs of pin inside the object and corresponding seg-
mentation and joint parameters as query points and ground
truths. p and pin are input query points for geometry de-
coder and articulation decoders separately, as described in
Sec. 4.2.
We implement the models with Pytorch [41] and train the
models with the Adam [23] optimizer and a learning rate of
10−4 and batch sizes of 8.
B. Ablation Study
Ditto uses 3D feature grid for geometry reconstruction
and 2D feature plane for articulation estimation. To vali-
date the advantage of this design, we evaluate another ab-
lated version where two 3D feature grid are used for geom-
etry and articulation respectively. As in Tab. 2, this ablated
version has similar performance to Ditto. But it requires
around 20% more memory usage and training time com-
pared with Ditto.
We show some qualitative results in Fig. 6. Our full Ditto
model can recreate the articulated objects more accurately,
especially the mobile part, benefiting from the attention-
based fusion, separate decoders and features. In compari-
son, Concat Fusion and Share Feature are not able to recon-
struct the smooth and complete surface. The ablated ver-
sion with Share Feature uses 2D feature planes along with
3D feature grids for geometry reconstruction. This projec-
tion operation results in artifacts as in the faucet result in
Fig. 6 (red circle in the second row, first column). The ab-
lated version with Share Decoder has a problem segmenting
the mobile and the static parts correctly. Overall, Ditto can
achieve the best performance on the reconstruction of the
Method
Whole CD ↓
Mobile CD
A-SDF (oracle code) [35]
0.66
-
Ditto (3D+3D feature)
0.27
0.12
Ditto
0.25
0.16
Table 2. Quantitative results of reconstruction on the Synthetic [1]
dataset.
Method
Chamfer Distance ↓
A-SDF [35]
3.57
Ditto
0.37
Table 3. Quantitative results of articulated motion synthesis on the
Synthetic [1] dataset.
Method
Joint type accuracy (%)
Global Joint [7]
88
Share Feature
100
Concat Fusion
96
Share Decoder
100
Ditto (Ours)
100
Table 4. Quantitative results of joint type prediction accuracy on
the Shape2Motion [55] dataset.
articulated objects.
C. Comparison with A-SDF
To explore the possible reason behind the inferior per-
formance in Tab. 1, we try fixing the A-SDF’s articulation
code to the ground-truth one. The reconstruction result is
improved and close to Ditto as in Tab. 2. It indicates that the
inferior performance of A-SDF is caused by the in- terfer-
ence between articulation and shape codes in test-time op-
timization. For example, the shape code degrades when the
articulation code is in a local minimum far from the ground
truth. In contrast, the articulation and geometry predictions
do not interfere with each other in our model.
A-SDF [35] can control the joint state by changing the
articulation code. On the other hand, Ditto explicitly re-
constructs the explicit part-level meshes and the articula-
tion model, where the joint state can also be easily con-
trolled. It is thus possible to compare the performance of
articulated motion synthesis of A-SDF and Ditto. We first
reconstruct the articulated object and then manipulate the
articulated object to a new joint state. And we measure the
whole Chamfer distance between manipulated results and
the ground truth objects after such an articulated motion.
The quantitative results are in Tab. 3.
Ditto achieves
significantly better results compared with A-SDF. We also

<!-- page 13 -->
Share Feature
Concat Fusion
Share Decoder
Ditto
Ground Truth
Figure 6. Reconstructed unseen articulated objects in the Shape2Motion [55] dataset of ablated versions. Static parts are colored grey
while mobile parts are colored green. We also visualize the estimated joint with the red arrow.
A-SDF
Reconstruction
Ditto
Reconstruction
GT
A-SDF
After Motion
Ditto
After Motion
GT
After Motion
Figure 7. Objects after articulated motion on the Synthetic [1] dataset. Static parts are colored grey while mobile parts are colored green.
We also visualize the estimated joint with the red arrow.
show some qualitative results in Fig. 7. Even though A-
SDF can generally reconstruct the articulated object from
the observation, the results after the articulated motion are
not consistent with the initial state due to its latent repre-
sentation of articulation. For example, the whole drawer
body is widened after the motion. In contrast, Ditto explic-
itly extracts mobile part mesh and the corresponding joint
parameters. Apart from the rigid transformation induced by
articulated motion, there is no unexpected distortion after
the motion.

<!-- page 14 -->
D. Joint Type Prediction
Our model is also predicting the joint type. All meth-
ods give 100% joint type accuracy on the Synthetic dataset.
We provide the results on the accuracy of joint type predic-
tion on the Shape2Motion dataset in Tab. 4. Most methods
also acquire 100% accuracy on this dataset except the global
joint baseline and the ablated version with concat fusion.
E. Generalization to Unseen Categories
In our experiments, we trained our model for four cat-
egories altogether and evaluated it on the same four cate-
gories. To test generalization to novel categories, we run our
model, trained on Shape2Motion, on four unseen categories
(drawer, microwave, fridge, and stapler). Fig. 5 shows that
our model generalizes robustly to geometrically similar cat-
egories (drawer, microwave, and fridge) but slightly worse
on the new categories of more significant differences (sta-
pler) as it learns shape priors from the training data.
