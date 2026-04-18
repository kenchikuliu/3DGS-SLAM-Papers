<!-- page 1 -->
Deep Visual Constraints: Neural Implicit Models
for Manipulation Planning from Visual Input
Jung-Su Ha
Danny Driess
Marc Toussaint
Learning & Intelligent Systems Lab, TU Berlin, Germany
 
 
(a) No object model
(b) See
(c) Plan
(d) Act
Camera
Fig. 1: Unlike static objects and a robot’s own body, 3D models of objects to manipulate are often unavailable. The proposed Deep Visual
Constraints represent an object, directly from images, as a continuous function over the 3D space and predict the task constraint values based
on such representation. This so-called implicit representation can naturally describe object’s rigid transformations in SE(3), enabling efﬁcient
optimization-based manipulation planning. The overall pipeline as well as demonstrations are well-visualized in https://youtu.be/r
mIGTu6Jg
Abstract—Manipulation planning is the problem of ﬁnding a
sequence of robot conﬁgurations that involves interactions with
objects in the scene, e.g., grasping and placing an object, or
more general tool-use. To achieve such interactions, traditional
approaches require hand-engineering of object representations
and interaction constraints, which easily becomes tedious when
complex objects/interactions are considered. Inspired by recent
advances in 3D modeling, e.g. NeRF, we propose a method to
represent objects as continuous functions upon which constraint
features are deﬁned and jointly trained. In particular, the
proposed pixel-aligned representation is directly inferred from
images with known camera geometry and naturally acts as a per-
ception component in the whole manipulation pipeline, thereby
enabling long-horizon planning only from visual input. Project
page: https://sites.google.com/view/deep-visual-constraints
I. INTRODUCTION
Dexterous robots should be able to ﬂexibly interact with
objects in the environment, such as grasping and placing an
object, or more general tool-use, to achieve a certain goal.
Such instances are formalized as manipulation planning, a type
of motion planning problem that solves not only for the robot’s
own movement but also for the objects’ motions subject to
their interaction constraints. Therefore, designing interaction
constraint functions, which we also call interaction features,
is at the core of achieving the robot dexterity. Traditional
approaches rely on hand-crafted constraint functions based on
geometric object representations such as meshes or combina-
tions of shape primitives. However, when considering large
varieties of objects and interaction modes, such traditional
approaches have long-standing limitations in two aspects: (i)
The representations have to be inferred from raw sensory
inputs like images or point clouds – raising the fundamental
problem of perception and shape estimation. (ii) With increas-
ing generality of object shapes and interaction, representation’s
complexity grows, thereby making hand-engineering of the
interaction features inefﬁcient. However, if the aim is manipu-
lation skills, the hard problem of precise shape estimation and
the feature engineering might be unnecessary.
What is a good object representation? Considering the
representation will be used to predict interaction features, we
expect it to encode primarily task-speciﬁc information rather
than only geometric. We also expect some of the information
to be shared across different interaction modes. In other words,
good representations should be task-speciﬁc so that the feature
prediction can be simpliﬁed and, at the same time, be task-
agnostic to enable synergies between the tasks. E.g., mug
handles are called handles because we can handle the mug
through them and also, once we learn the notion of a handle,
we can play around with the mug through the handle in many
different ways. Also, from the perception standpoint, good
representations should be easy to infer from raw sensory inputs
and should be able to trade their accuracy (if bounded) in favor
of the feature prediction.
To this end, we propose a data-driven approach to learning
interaction features that are conditioned on object images. The
whole pipeline is trained end-to-end directly with the task
supervisions so as to make the representation and perception
task-speciﬁc and thus to simplify the interaction prediction.
The object representation acts as a bottleneck and is shared
arXiv:2112.04812v3  [cs.RO]  28 Jul 2022

<!-- page 2 -->
across multiple features so that the task-agnostic aspects can
emerge. We propose the representation to be a d-dimensional
continuous function over the 3D space [30, 25]. In particular,
the proposed implicit neural representation is pixel-aligned,
meaning that the function takes as input images from multiple
cameras (e.g. stereo) and, assuming known camera poses
and intrinsics, computes a representation at a certain 3D
location using image features at the corresponding 2D pixel
coordinates. Once learned, the interaction features can be
used by a typical constrained optimal control framework to
plan dexterous object-robot interaction. We show that making
use of the learned constraint models within Logic-Geometric
Programming (LGP) [43] enables planning various types of
interactions with complex-shaped objects only from images.
Since the representations generalize well, the learned con-
straint models are directly applicable to manipulation tasks
involving unseen objects. To summarize, our main contribu-
tions are
• To represent objects as neural implicit functions upon
which interaction constraint functions are trained,
• An image-based manipulation planning framework with
the learned features as constraints,
• Comparison to non pixel-aligned, non implicit function,
and geometric representations,
• Demonstration in various manipulation scenarios rang-
ing from basic pick-and-hang [video1] to long-horizon
manipulation [video2], zero-shot imitation [video3], and
sim-to-real transfer [video4].
II. RELATED WORK
A. Neural Implicit Representations in 3D Modeling
Implicit neural representations have recently gained increas-
ing attention in 3D modeling. The core idea is to encode an
object or a scene in the weights of a neural network, where
the network acts as a direct mapping from 3D spatial location
to an implicit representation of the model, such as occupancy
measures [24], signed distance ﬁelds (SDF) [30, 1], or ra-
diance ﬁelds [25]. In contrast to explicit representations like
voxels, meshes or point clouds, the implicit representations
don’t require discretization of the 3D space nor ﬁxed shape
topology but rather continuously represent the 3D geometry,
thereby allowing for capturing complex shape geometry at
high resolutions in a memory efﬁcient way.
There have been attempts to associate these 3D representa-
tions with 2D images using the principle of camera geometry.
Exploiting the camera geometry in a forward direction, i.e.,
2D projection of 3D representations, yields a differentiable
image rendering procedure and this idea can be used to
get rid of 3D supervisions. For example, Sitzmann et al.
[38], Niemeyer et al. [29], Yariv et al. [50], Mildenhall et al.
[25], Henzler et al. [18], Reizenstein et al. [34] showed
that the representation networks can be trained without the
3D supervision by deﬁning a loss function to be difference
between the rendered images and the ground-truth. Another
notable application of this idea is view synthesis. Based on the
differentiable rendering, Park et al. [31], Chen et al. [4], Yen-
Chen et al. [51] addressed unseen object pose estimation
problems, where the goal is to ﬁnd object’s pose relative
to the camera that produces a rendered image closest to the
ground truth. By conditioning 3D representations on 2D input
images, one can expect the amortized encoder network to
directly generalize to novel 3D geometries without requiring
any test-time optimization. This can be done by introducing a
bottleneck of a ﬁnite-dimensional global latent vector between
the images and representations, but these global features often
fail to capture ﬁne-grained details of the 3D models [39]. To
address this, the camera geometry can be exploited in the
inverse direction to obtain pixel-aligned local representations,
i.e., 3D reprojection of 2D image features. Saito et al. [36]
and Xu et al. [49] showed that the pixel-aligned methods can
establish rich latent features because they can easily preserve
high-frequency components in the input images. Also, Yu
et al. [53] and Trevithick and Yang [45] incorporated this idea
within the view-synthesis framework and showed that their
convolutional encoders have strong generalizations.
While the above work investigates implicit neural represen-
tation to model shapes or appearances, our work makes use of
it to model physical interaction feasibility and thereby to pro-
vide a differentiable constraint model for robot manipulation
planning.
B. Object/Scene Representations for Robotic Manipulations
Several works have proposed data-driven approaches to
learning object representations and/or interaction features
which are conditioned on raw sensory inputs, especially for
grasping of diverse objects. One popular approach is to train
discriminative models for grasp assessments. For example, ten
Pas et al. [41], Mahler et al. [20], Van der Merwe et al. [47]
trained a neural network that, for given candidate grasp poses,
predicts their grasp qualities from point clouds. In addition,
Breyer et al. [2], Jiang et al. [19] proposed 3D convolutional
networks that take as inputs a truncated SDF and candidate
grasp poses and return the grasp affordances. Similarly, Zeng
et al. [56, 55] addressed more general manipulation scenarios
such as throwing or pick-and-place, where a convolutional
network outputs a task score image. On the other hand,
neural networks also have been used as generative models.
For example, Mousavian et al. [26] and Murali et al. [27]
adopted the approach of conditional variational autoencoders
to model the feasible grasp pose distribution conditioned on
the point cloud. Sundermeyer et al. [40] proposed a somewhat
hybrid method, where the network densely generates grasp
candidates by assigning grasp scores and orientations to the
point cloud. You et al. [52] addressed the object hanging tasks
from point clouds where the framework ﬁrst makes dense
predictions of the candidate poses among which one is picked
and reﬁned. Compared to these works, our framework takes
advantage of a trajectory optimization to jointly optimize an
interaction pose sequence instead of relying on exhaustive
search or heuristic sampling schemes, thus not suffering from
the high dimensionality nor the combinatorial complexity of

<!-- page 3 -->
long-horizon planning problems.
Another important line of research is learning and utilizing
keypoint object representations. Manuelli et al. [22], Gao and
Tedrake [14], Qin et al. [33], Turpin et al. [46] represented
objects using a set of 3D semantic keypoints and formulated
manipulation problems in terms of such the keypoints. Sim-
ilarly, Manuelli et al. [23] learned the object dynamics as a
function of keypoints upon which a model predictive controller
is implemented. Despite their strong generalizations to unseen
objects, the keypoint representations require semantics of the
keypoints to be predeﬁned.
The representation part of our framework is closely re-
lated to dense object descriptions proposed by
Florence
et al. [12, 13]. The idea is to train fully-convolutional neural
networks that maps a raw input image to pixelwise object
representations which directly generalize to unseen objects.
As a recent concurrent work from Simeonov et al. [37] also
proposed, our implicit representation extends this pixel-wise
representation to the 3D space. Compared with those existing
work, ours is learned via task supervisions in conjunction
with the task feature heads and thus can be seamlessly inte-
grated into general sequential manipulation planning problems.
Another recent related work from Yuan et al. [54] proposed
learning object-centric representations that are used to predict
the symbolic predicates of the scene which in turn enables
symbolic-level task planning. Driess et al. [7] formulated
manipulation planning problems solely in terms of SDFs as
representations and proposed to learn manipulation constraints
as functionals of SDFs. More recently, Driess et al. [9] trained
implicit object encoders together with differentiable image
rendering decoders and used a graph neural network to model
dynamics, based on which an RRT-based method can plan
sequential manipulations in the latent space. In contrast to the
above, our proposed representations are trained in conjunction
with multiple task prediction heads and can be seamlessly
integrated into sequential manipulation planning schemes that
generate motions ﬂexibly blending diverse interactions to-
gether.
III. DEEP VISUAL CONSTRAINTS (DVC) VIA IMPLICIT
OBJECT REPRESENTATION
robot or static
frame’s pose
q ∈SE(3)
obj images,
cam poses,
intrinsics: V
interaction points
{p1, ..., pK}
pk
ψ(p; V)
backbone
(PIFO)
yk
L
Concat.
task
head
h
shared
(Sec.III-A)
task-speciﬁc
(Sec.III-B)
Fig. 2: Feature prediction of DVC. The backbone, conditioned on a
set of posed images, computes representation vectors at queried 3D
spatial locations pk ∈R3, and the task head predicts the interaction
feature based on the obtained representation vectors yk ∈Rd.
Given Nview images with their camera poses/intrinsics,
V = {(I1, T 1, K1), ..., (INview, T Nview, KNview)} with I ∈
R3×H×W (we considered H = W = 128) and T , K ∈R4×4,
we build an interaction feature as a neural network:
h = φtask(q; V),
(1)
where q ∈SE(3) is the pose of the robot/static frame
interacting with the object; the interaction feature h ∈R,
analogous to energy potentials, is zero when feasible and
non-zero otherwise, which will act as an equality constraint
in manipulation planning. As shown in Fig. 2, the feature
prediction framework consists of two parts: the representation
backbone which serves as an implicit representation of an
object, and the task heads that make feature predictions.
Notably, while the multiple task heads individually model
different interaction constraints, the backbone is shared across
them, allowing for learning more general object representation.
A. Pixel-Aligned Implicit Functional Object (PIFO)
The proposed implicit object representation is a mapping:
y = ψ(p; V),
(2)
where p ∈R3 and y ∈Rd are a queried 3D position and a
representation vector at that point, respectively. This function,
implemented as a neural network as depicted in Fig. 3, consists
of three parts: image encoder, 3D reprojector, and feature
aggregator. The ﬁrst two compute a representation vector from
each image and the last one combines them.
Image Encoder: This module takes as input an image and
computes a feature map (the pathway from In to Fn in Fig. 3).
We adopted the hourglass network architecture, especially with
ResNet-34 as its downward path and two residual layers with
3 × 3 convolutions followed by up-convolution as the upward
path:
Fn = UNet(In), ∀n ∈{1, ..., Nview},
(3)
which results in a feature map Fn ∈R64×64×64 that captures
both local and global information in the input image.
3D Reprojector: To endow the network with the multi-view
consistency, all the 3D operations are performed in the view
space. The 3D reprojector, the pathway from (T n, Kn), Fn
and p to yn in Fig. 3, transforms a queried point, p, into the
image coordinate including depth, π(p; T , K) = z ∈R3 and
extracts the local image feature at the projected point from the
feature map, F, via bilinear interpolation. Finally, the extracted
feature and the coordinate feature, which is computed through
a couple of fully connected layers (FCLs), are passed to a
couple of FCLs to get a representation vector at p for a single
image, i.e., ∀n ∈{1, ..., Nview},
yn = MLP(Fn(zn), zn), zn = π(p; T n, Kn).
(4)
Feature Aggregator: This module is the pathway from
yn to y in Fig. 3, which aggregates the representation
vectors from multiple views into one vector. Among many
permutation-invariant options, like summation or more sophis-
ticated attention mechanisms, we simply take the averaging
operation for it, i.e.,
y =
1
Nview
Nview
X
n=1
yn.
(5)

<!-- page 4 -->
In
(T n, Kn)
obj images,
cam poses,
intrinsics: V
64
64
64
64
32
128
128
16
256
256
8
512
512
4
Bottleneck
Conv
256
256
256
8
128
128
128
16
64
64
64
32
64 64 64
64
Fn
1
64
local
image
fea-
ture
p ∈R3
camera
projection
π(p; T n, Kn)
1
3
zn
1
32
coord
fea-
ture
L
Concat.
1
256
local2
1
128
local3
yn
¯Σ
Average
y ∈Rd
Fig. 3: PIFO (i) encodes the images I as pixel-wise feature maps F via U-net, (ii) projects the query point p into the pixel coordinate z
using camera geometry (T , K), and (iii) computes the representation vector yd by extracting the image features at the projected points.
B. Interaction Task Feature Prediction
A task head evaluates the interaction constraint violation,
h, for a given robot/static frame’s pose, q, using the object
representation function over 3D, ψ(·). To this end, we rigidly
attach a set of keypoints to the robot frame at which the
backbone is queried, i.e., ∀k ∈{1, ..., K},
yk = ψ(pk; V), pk = R(q)ˆpk + t(q),
(6)
where ˆpk is kth keypoint’s local coordinate, and R(q) and
t(q) denote the rotation matrix and the translation vector of
q, respectively. Finally, the task head, based on the resulting
representation vectors, predicts a constraint value through a
couple of FCLs:
h = MLP(y1, ..., yK).
(7)
IV. TRAINING
In this paper, we consider manipulation scenarios where
a robot arm, Franka Emika Panda, or two manipulate mugs.
The shapes of mugs are diverse and the scene contains
multiple hooks on which a mug can be hung. Formulating such
problems requires three types of learned interaction features:
an SDF feature for collision avoidance and grasping/hanging
features, so we prepared the dataset for each.
A. Data Generation
We took 131 mesh models of mugs from ShapeNet [3] and
convex-decomposed those meshes. The meshes are translated
and randomly scaled so that they can ﬁt in a bounding sphere
with a radius of 10 ∼15 cm at the origin. For each mug, we
created the following dataset.
Posed Images: The posed image data consists of 100
images (128 × 128) with the corresponding camera poses
and intrinsic matrices generated by the OpenGL rendering.
Azimuths and elevations of the cameras are sampled such that
they are uniformly projected onto the unit sphere, while their
distances from the object center are random. The azimuth,
elevation and distance fully determine the camera’s positions,
and the camera’s orientations are set such that the cameras
are upright and face the object center. For the intrinsics, we
used the ﬁeld of view fov = 2 arcsin(d/r), where d is the
camera distance from the object center and r is the radius
of the object’s bounding sphere, so that the object spans the
entire image. Lighting is also randomized.
SDF: We sampled 12,500 3D points and precomputed their
signed distance values, i.e., the distance of a point from the
object surface with the sign indicating whether or not the point
is inside the surface. Following the approach of DeepSDF [30],
we sampled more aggressively near the object surface to foster
the learning of the object geometry.
Grasping & Hanging: The grasping and hanging data
are 1,000 feasible grasping and hanging poses of the gripper
and the hook, respectively. For grasping, we used an antipo-
dal sampling scheme, similarly to [10], to create candidate
gripper poses and checked their feasibility using Bullet [5].
For hanging, we randomly sampled collision-free hook poses
and checked if it’s kinematically trapped by the mug in the
directions perpendicular to the hook’s main axis.
Fig. 16 shows some rough looks of the generated data. In
the end, we have a dataset of:

I1:100, T 1:100, K1:100, p1:12500, SDF 1:12500, q1:1000
grasp
, q1:1000
hang
(i)131
i=1
,
which we divided into 78 training, 25 validation, 28 test sets.
B. Data Augmentation
While randomizing the azimuth, elevation and distance of
the camera provides all possible appearances of the object, it
still cannot account for varying roll angles of the camera (i.e.
image rotations) and off-centered images. To show the network
all possible images that it can encounter when deployed later

<!-- page 5 -->
(a) Before augmentation
(b) After augmentation
Fig. 4: Image Data Augmentation
and to mitigate the size-ambiguity issue, we propose to use a
data augmentation technique based on Homography warping:
In each iteration, for a randomly sampled set of images,
we artiﬁcially perturb the roll angle of each camera and the
estimated object center position (at which the cameras are
looking). Also, fov is modiﬁed as if the radius of the bounding
sphere is 15 cm so that smaller objects can appear smaller in
the transformed images. This results in new rotation matrices,
ˆR, and intrinsic matrices,
ˆ
K, of the cameras. Because the
original and new cameras are at the same position, images
taken from them can be transformed one another through the
Homography warping, as also illustrated in Fig. 7. Therefore,
we compute the corresponding Homography transformation
matrix and warp the images accordingly:
W( ˆR, ˆ
K) :


u
v
1

7→w ˆ
K ˆRT RK−1


u
v
1

.
(8)
Random cutouts are also applied to address occlusion. Fig. 4
depicts how this image augmentation works.
For grasping and hanging, i.e., task ∈{grasp, hang}, we
generate random poses ˆqtask ∈SE(3) in each iteration as
a weighted sum of a (randomly picked) feasible pose and a
random pose ˆqtask = tqfeasible +(1−t)qrand, t ∼U(0, 1) where
the position of qrand is from the normal distribution and its
quaternion is sampled uniformly, to encourage more precise
prediction around the constraint manifolds. The training target
is then, similarly to [1], the unsigned distances (in SE(3)) of
ˆqtask from the set of the feasible poses:
dtask =
min
j∈{1,...,1000} ||q −qj
task||2.
(9)
C. Loss Function
The whole architecture, backbone and three task heads,
is
trained
end-to-end.
In
each
iteration,
we
choose
a
minibatch
of
mugs
for
which
a
subset
of
aug-
mented
images
with
their
camera
parameters,
ˆV
=
{(ˆI1, ˆT 1, ˆ
K1), ..., (ˆINview, ˆT Nview, ˆ
KNview)}, a subset of SDF
data,
 p1:NSDF, SDF 1:NSDF
, and the grasping/hanging data,

ˆq1:Ntask
task
, d1:Ntask
task

, are sampled. The images are encoded only
Fig. 5: The grasp and hang interaction points are deﬁned as (3×3×3)
grid points around the gripper center and 5 points along the hook’s
main axis, respectively.
once per iteration and then the SDF, grasping, hanging features
are queried at the sampled points and poses. The overall loss
is given as
Ltotal = Lsdf + Lgrasp + Lhang,
(10)
where we used a typical L1 loss for SDFs, i.e.
Lsdf =
1
NSDF
NSDF
X
i=1
|φsdf(pi) −SDF i|,
(11)
and the sign-agnostic L1 loss in [1] for grasping and hanging,
i.e., ∀task ∈{grasp, hang}
Ltask =
1
Ntask
Ntask
X
i=1

φtask(ˆqi
task; ˆV)
 −di
task
 .
(12)
We used Nviews = 4, NSDF = 300, Ngrasp = 100, Nhang = 100
and the considered interaction points are shown in Fig. 5.
V. SEQUENTIAL MANIPULATION PLANNING WITH DVC
robot
conﬁguration
x ∈Rnx
forward
kinematics
robot or static
frame’s pose
q ∈SE(3)
raw images&masks,
camera poses,
intrinsics: Vraw
multi-view
processing
(Sec.V-A)
obj images,
cam poses,
intrinsics: Vobj
Deep
Visual
Constraint
(Sec.V-B)
h
object’s rigid
transformation
δq ∈SE(3)
Fig. 6: DVCs in manipulation planning. The multi-view preprocess-
ing converts the scene images into the object-centric images. The
robot or static frames’ poses are computed via forward kinematics.
In order to compute a full trajectory of the robot and objects
that it interacts with, the learned features can be integrated as
differentiable constraints into any constraint-based trajectory
optimization framework, for which we adopt Logic-Geometric
Programming (LGP) [43]. In typical manipulation scenes,
however, cameras are equipped such that their views cover
a wide range of the environment, so we need to transform
the raw images of the entire scene into object-centric ones
to pass them to the network. To that end, as depicted in
Fig. 6, we propose the multi-view preprocessing to compute

<!-- page 6 -->
the object-centric images and corresponding camera extrin-
sics/intrinsics. In addition to the multi-view preprocessing, we
wrap the learned DVCs with forward kinematics to evaluate
the learned constraints from the scene images and the opti-
mization variables, i.e., robot’s joint conﬁguration and object’s
transformation. Section V-A discusses the proposed multi-
view warping procedure and Section V-B presents how the
learned features serve as constraints to sequential manipulation
planning problems.
A. Multi-View Preprocessing
Fig. 7: Illustration of Multi-View Preprocessing: Two images taken
at the same location but different orientations & fov are related by a
homography
As illustrated in Fig. 7, multi-view processing ﬁnds a
bounding ball and warps the raw images via the Homography
warping. Let Mn ∈{0, 1}W ×H be the object masks available
along with the raw images In
raw, ∀n = 1, ..., Ncam. We ﬁrst ﬁnd
a position and radius of the minimal bounding sphere such that
the warped images contain all the object pixels in the original
images by solving the following optimization problem:
min
p∈R3,r∈R+ r,
(13)
s.t. ∀(un,vn,n)∈{(u′,v′,n′);Mn′(u′,v′)=1,∀n′∈{1,...,Ncam}} :
||W( ˆRn, ˆ
Kn)(un, vn)||2 < 1,
where ˆR can be obtained from the sphere center p and the
camera position t, and ˆ
K is computed as fov = 2 arcsin(||t−
p||2/r); the warping W can then be deﬁned as in (8). After
solving the above optimization, we ﬁx the camera orientations
ˆR, change the intrinsics as if the bounding sphere has a radius
of 15 cm and ﬁnally warp the raw images accordingly. Fig. 8
shows the raw images from an example environment and the
results of the multi-view processing.
B. Logic-Geometric Programming for Manipulation Planning
The core concept of manipulation is the rigid transforma-
tions of objects [43, 7]; e.g., while grasped, the object moves
with the gripper. For an object transformed by δq ∈SE(3),
we deﬁne a rigid transformation of the interaction feature as:
T(δq)[φtask](·) := φtask
 δq−1·

,
(14)
which is equivalent to rigidly transforming the representation
function as T(δq)[ψ](·) = ψ
 R(δq)T (· −t(δq))
1. Through
the function composition of the forward kinematics, FK, and
the (transformed) feature, φtask, i.e.,
Htask(x, δq) := (T(δq)[φtask] ◦FK) (x),
(15)
1We dropped V from φtask(q; V) for the simplicity of notation.
(a) Raw images and masks
(b) Object-centric images warped via the multi-view processing
Fig. 8: Multi-view processing
we obtain an interaction feature as a function of a robot joint
conﬁguration x and object’s rigid transformation δq.
Now we are ready to formalize manipulation planning
problems. For an nx-joint robot and no rigid objects, LGP
is a hybrid optimization problem over the number of phases
K ∈N, a sequence of discrete actions a1:K and sequences of
the robot joint conﬁgurations x1:KT , x ∈Rnx and the object’s
rigid transformations δq1:KT , δq ∈SE(3)no. The trajectory
is discretized into T steps per phase. A discrete action ak
describes which interaction should be fulﬁlled at the end of
the phase k, i.e., which mug to pick or on which hook to hang
the grasped mug, and uniquely determines a symbolic state
sk = succ(sk−1, ak), i.e., whether each mug is grasped or hung
on a particular hook. Suppose that a discrete action sequence
a1:K and the corresponding modes s1:K with sK ∈Sgoal are
proposed by a logic tree search. We then deﬁne the geometric
path problem as a 2nd order Markov optimization [42]:
min
x1:KT
δq1:KT
KT
X
t=1
f (xt−2:t) ,
(16)
s.t.∀H∈H(sk,ak),
k∈{1,...,K}
: H
 (xt−2:t, δqi
t−2:t)(t,i)∈IH(sk,ak)

= 0,
where the initial joint states x−1:0 and objects’ transforma-
tions δq−1:0 = 0 are given. Note that δq denotes rigid
transformations applied to objects’ implicit representations,
not their absolute poses. f is a path cost that penalizes
squared accelerations of the robot joints, but it can be more
general if necessary. H(sk, ak) is a set of constraints the

<!-- page 7 -->
symbolic state and action impose on the geometric path at
each phase k(t) = ⌊t/T⌋; these constraints include physical
consistency, collision avoidance, and the learned interaction
constraints that ensure the success of the discrete action ak.
Lastly, IH(sk, ak) decides the time slice and object index that
are subject to the constraint H. Appendix VII-B introduces
the set of imposed constraints in detail. As all the cost and
constraint terms are differentiable and their Jacobians/Hessians
are sparse, we can solve this optimization problem efﬁciently
using the augmented Lagrangian method with the Gauss-
Newton approximation [42].
VI. EXPERIMENTS
A. Performance of Learned Features
Baselines: The key techniques of the proposed framework
are threefold: the pixel-aligned local feature extraction, the
implicit object representation over 3D and the task-guided
learning scheme. To examine the beneﬁts from each compo-
nent, three baselines are considered. (i) Global image features:
The ﬁrst baseline still represents an object as a function but
the image encoder outputs a global image feature (as shown in
Fig. 19(b)) rather than having the pixel-aligned feature locally
extracted; we used the ResNet-34 architecture as the image
encoder and ﬁxed the other model speciﬁcations. (ii) Vector
object representations: The second baseline represents an
object as a ﬁnite-dimensional vector instead of a function; as
shown in Fig. 19(c), the representation network ﬁrst computes
the image features from the images using ResNet-34 and the
camera features from the camera parameters using a couple of
FCLs. Two features are then passed to another couple of FCLs
to produce the object representation vector. The task heads take
as input the frame’s pose as well as the object representation
vector. (iii) SDF representations: The last baseline uses SDFs
as object representations; the network architecture for the SDF
feature remains the same, but the grasping and hanging heads
take as input a set of the keypoints’ SDF values instead of
the d-dimensional representation vectors. The SDF values are
detached when passed to the grasping/hanging heads so the
backbone is trained by the geometry (SDF) data only.
Evaluation Metric:
Regarding the shape reconstruction,
we report the Volumetric IoU and the Chamfer distance. To
measure these metrics, we randomly sampled 4 images from
the dataset and reconstructed the meshes from the learned
SDF feature using the marching cube algorithm (See Fig. 17).
The volumetric IoU is the ratio between the intersection
and the union of the reconstructed and ground-truth meshes
which is (approximately) computed on the 1003 grid points
around the objects. To compute the Chamfer distance, we
sampled 10,000 surface points from each mesh and averaged
the forward and backward closest pair distances. To evaluate
the learned task features, we solved the unconstrained opti-
mization ˆq∗= arg minq ||φtask(q)||2, task ∈{grasp, hang}
using the Gauss-Newton method. Starting from this solu-
tion, we then solved the second optimization problem by
including the collision feature (details in Appendix VII-C),
q∗= arg minq ||φtask(q)||2 + wcoll||φcoll(q)||2. Because the
(a)
(b)
(c)
Fig. 9: SDFs predicted by (b) PIFO and (c) the global image feature
model.
(a)
(b)
(c)
Fig. 10: Some failure cases of hand-engineered features. (a) The
hand-engineered feature lead the optimizer to hang the mug through
the wrongly generated hole. (Green transparent meshes represent the
ground truth.) (b) The handle disappeared in reconstruction, so this
part would never be grasped and the mug never be hung. (c) The
hand-engineered feature generated a wrong grasping pose on the
ground truth mesh.
local optimization method can be stuck at local optima, we
ran the algorithm from 10 random initial guesses in parallel
and picked the best one. The optimized pose is ﬁnally tested
in simulation and the success rates (feasibility) are reported in
Table I.
Result:
Table I shows that the SDF representation has
the best shape reconstruction performance; PIFO is slightly
worse, followed by the other two baselines. On the other
hand, the task performances of PIFO are signiﬁcantly better
than the others. The SDF representation is especially worse
in the hanging task, which implies that SDFs along the line
are not sufﬁcient for its feature prediction and our task-guided
representation simpliﬁes the feature prediction. In addition, it
can be observed from Fig. 9 that the pixel-aligned method was
better able to capture ﬁne-grained details than the global image
feature which reconstructed the handle shape as being more
“typical”. PIFO was also trained with the different numbers of
input images and it can be seen that the more images we put
in, the better performance the network shows. Tables II and
III report all combinations of the metrics and the number of
views.
Hand-Engineered Constraint Models: We also compared
our model to hand-engineered constraint models, (iv) GT Mesh
+ HE and (v) Recon. + HE, each of which computes constraint
values based on the ground-truth meshes and the meshes
reconstructed by the above SDF representations. Notably,
Figs. 10(a)-(b) show how vulnerable the hand-engineered
constraints can be to the reconstruction error; i.e., the error

<!-- page 8 -->
IoU
Chamfer-L1 (×10−3)
Grasp+c (%)
Hang+c (%)
PIFO
0.816 / 0.656
5.26 / 6.90
88.1 / 82.5
94.0 / 78.9
Global Image Feature
0.697 / 0.581
7.42 / 9.49
82.7 / 75.7
91.2 / 78.2
Vector Object Representation
0.036 / 0.014
38.6 / 39.7
0.5 / 0.4
0.0 / 0.0
SDF Object Representation
0.845 / 0.667
4.90 / 6.83
67.9 / 64.3
3.7 / 4.3
PIFO (2 views)
0.760 / 0.577
6.14 / 8.84
82.9 / 77.1
88.2 / 72.1
PIFO (8 views)
0.851 / 0.683
4.78 / 6.34
88.7 / 85.0
96.5 / 82.5
GT Mesh + HE
-
-
62.8 / 75.0
94.9 / 92.9
Recon. + HE
-
-
66.7 / 42.9
78.2 / 60.7
TABLE I: Individual Feature Evaluation with 4 views (Training / Test)
(a) Single mug hanging
(b) Three-mug hanging
(c) Handover
Fig. 11: Sequential manipulation scenarios
is directly associated with the planning result. While the
perception pipeline for this geometric representation is never
encouraged to reconstruct the “graspable/hangable parts” more
accurately, we can view our end-to-end representation learning
via task supervision as a way to do so. Moreover, the hand-
engineered feature sometimes produces a wrong grasping pose
even for the ground truth mesh (e.g., Fig. 10(c)). One can argue
that a better interaction feature could be hand-designed by
investigating the physics and kinematic structures more deeply,
but that would require a huge amount of human insights/efforts
and thus is inevitably less scalable. In contrast, our data-
driven approach eliminates this procedure and directly learns
the interaction constraint models from empirical success data
of physical interactions.
B. Sequential Manipulation Planning via LGP
We ﬁrst considered a basic pick & hang task as shown in
Fig. 11(a). The environment contains one robot arm, one hook,
one mug and 4 cameras (as in Fig. 18(a)), and the interaction
modes are constrained by the discrete action sequence of
[(GRASP, gripper, mug), (HANG, hook, mug)]. 10 mugs were
picked from each of the training and test data sets and their
initial poses are randomized.2 When the optimized trajectories
are executed in the Bullet simulation, the success rates on the
train and test mugs were 50 % and 40 %, respectively. If we
allow the method to re-plan and execute when it failed, the
success rates increased to 90% and 70%, respectively [video1].
2Before solving the full trajectory optimization, we ﬁrst optimized each
feature as in Sec. VI-A and added small regularization terms using the
optimized poses to guide the optimizer away from local optima.
(a) Sampled grasp
(b) Sampled hang
(c) IK (out-of-reach)
(d) Sampled grasp1
(e) Sampled grasp2
(f) IK (collision)
Fig. 12: Inverse kinematics with generative models
To showcase the long-horizon planning capability of LGP,
we considered the following two scenarios: (i) The three-mug
scenario consists of 6 discrete phases with [(GRASP, gripper,
mug1), (HANG, M hook, mug1), (GRASP, gripper, mug2),
(HANG, U hook, mug2), (GRASP, gripper, mug3), (HANG,
L hook, mug3)]. (ii) The handover scenario has two arms at
different heights and the target hook is placed high, requiring
two arms to coordinate a handover motion with the discrete
actions [(GRASP, R gripper, mug), (GRASP, L gripper, mug),
(HANG, U hook, mug)]. Fig. 11 shows the last conﬁgurations
of the optimized plans; we refer readers to Figs. 20–21 and
videos [video2] for clearer views.
Inverse Kinematics with Generative Models: One im-
portant attribute of our framework is that, while most ex-
isting works train generative models that directly produce
the interaction poses, ours models interactions as equality
constraints where multiple constraints can be jointly optimized
with other planning features. To see the beneﬁts of such joint
optimization, we considered the following inverse kinematics
problems with a generative model: For the basic pick & hang
and handover scenarios, we optimized each interaction pose
separately as in Sec. VI-A and checked if these individually
optimized poses are kinematically feasible when combined

<!-- page 9 -->
Fig. 13: First 3 principal components from PCA on image features.
Each component distinguishes the overall object areas, the handle or
rim parts, etc. More images can be found in Figs. 24–25.
together, i.e., whether or not the inverse kinematics problems
have a solution. Even though the mug’s initial pose was given
such that the ﬁrst gasping is ensured feasible, 53 out of 100
pairs of grasp and hang poses were infeasible for the pick
& hang scenario and 86 out of 100 sets were infeasible
for the handover scenario, i.e., many of the sampled poses
led to a collision or an infeasible robot conﬁguration for
hanging/handover. Some failure cases are depicted in Fig. 12
(more in Figs. 22–23). As the sequence length gets longer,
not only should an exponentially larger number of planning
problems be solved to ﬁnd a set of feasible poses, but also
the found poses are not guaranteed to be optimal. The joint
optimization with our constraint models doesn’t raise such
issues.
C. Exploiting Learned Representations: 6D Pose Estimation
and Zero-shot Imitation
Fig. 13 visualizes three components of the image feature
vectors (the outputs of the U-net encoder) from the principal
component analysis (PCA). It can be observed that each
component represents a certain property of the objects, such as
inside vs. outside, handle vs. other parts, or above vs. below.
This enables the image-based pose estimation which we call
feature-based closest point (FCP) matching, i.e., the problem
of ﬁnding the relative pose of a target mesh w.r.t. a model
mesh, without deﬁning any canonical coordinate of the objects.
Speciﬁcally, the FCP matching works as follows:
1) It ﬁrst queries the backbone at 103 and 53 grid points
around the target and the model, respectively, (as shown
in Fig. 14(c)) with their own images.
2) For each model grid point, the target point is obtained
such that their representations are closest.
3) Finally, it computes a SE(3) pose that minimizes the
sum of the model-target pairwise Euclidean distances.
We compared this to the conventional iterative closest point
(ICP) algorithm on point clouds, i.e., the problem of ﬁnding
the relative pose minimizing the Euclidean distance of two sets
of point clouds. The point clouds can be obtained from depth
cameras (ICP) or on the surface of the meshes reconstructed
via the learned SDF features (ICP2). The point clouds’ size
was 1000. Fig. 14 (h, i) shows the position and orientation
errors when 131 mugs with random poses were tested. No-
tably, as visualized in Fig. 14 (d-g), FCP performs much better
(a) Model (right) and
target (left) mugs
(b) Point clouds for ICP (c) Grid points for FCP
(d) ICP
(e) ICP2
(f) FCP
(g) F+ICP2
ICP
ICP2
FCP
F+ICP2
0.000
0.025
0.050
0.075
0.100
0.125
(h) Position error
ICP
ICP2
FCP
F+ICP2
0
1
2
3
(i) Orientation errors
Fig. 14: 6D Pose Estimation Results - the estimated poses are applied
to the green meshes. ICP easily gets stuck at local optima while FCP
produces fairly accurate poses which help F+ICP2 escape the local
optima; note that FCP does not iterate to get the results. More images
are shown in Fig. 27.
(a) Reference
(b) Imitation 1
(c) Imitation 2
Fig. 15: Zero shot imitation. Detailed views are in Figs. 28 – 29.
especially in orientation because, notoriously, ICP easily gets
stuck in local optima. A signiﬁcant improvement was observed
in F+ICP2 where we used the FCP results as starting points
of ICP2 (which is performed without depth images).
Another important observation from Fig. 13 is that the
semantics of representations are consistent across different
objects as well, e.g. the handle parts of different mugs have
similar representations. It implies that a pose of one object
can be transferred into another through the representation.
We therefore considered an image-based zero-shot imitation
scenario, where the environment contains one robot arm,
one target mug (ﬁlled with small balls) and 4 cameras as
shown in Fig. 15. We manually designed a pouring motion

<!-- page 10 -->
for one mug and stored the images of pre- and post-pouring
postures of the mug, Vpre = (Ipre, Tpre, Kpre) and Vpost =
(Ipost, Tpost, Kpost), respectively. For a new mug, we solved
LGP with [(GRASP, gripper, mug), (POSEFCP, Vpre, mug),
(POSEFCP, Vpost, mug)], where (POSEFCP, ·, ·) imposes the
aforementioned FCP constraint at the end of each phase. That
is, the trajectory optimizer tries to match each part of the
new object to the corresponding part of the target mug while
coordinating the global consistency of the full trajectory (e.g.,
determining a proper grasp pose for pouring). Fig. 15 shows
the optimized post-pouring posture, which implies that the
learned representation allows for imitation of the reference
motions only from the posed images. The videos can be found
at [video3].
D. Real Robot Demonstration
Fig. 1 shows our complete framework in the real robot
system. To successfuly apply the learned DVCs to the real
robot by closing the sim-to-real gap, we had to extend training
to a larger dataset; speciﬁcally, we randomized the material
of mugs by adjusting metalness and roughness to get more
diverse appearances and also applied more extensive data
augmentations, e.g. ColorJitter or GaussianBlur. At test time,
we attached RealSense D435 on the gripper and took 8 color
images from predeﬁned shooting poses. We used the pre-
trained Mask R-CNN [17] to get object masks, which we
found provides clear enough segmentation for the network
to come up with sensible manipulation plans. We refer the
readers to the supplementary video (in the caption of Fig. 1)
for visualization of the whole manipulation pipeline.
VII. DISCUSSION
The main idea of the proposed Deep Visual Constrains is
twofold:
1) Implicit object representations to which manipulation
planning algorithms can apply rigid transformations in
SE(3) and
2) the implicit representations trained as a shared backbone
of multiple task features, directly via task-supervisions.
Throughout the experiments, we demonstrated the proposed
visual manipulation framework both in simulation and with the
real robot. The ablation studies examined each of the proposed
techniques, compared to the non-pixel aligned, explicit, and
geometric representations as well as the traditional hand-
engineered features. The IK experiments demonstrated the
advantage of DVC’s joint optimization capability. We ana-
lyzed the learned representations via PCA and found that the
generalizable sementics emerged in the representation during
training, which enables 6D pose estimation and zero-shot
imitation.
Notably, the last ﬁnding implies that considering more
diverse tasks and objects in our multi-task learning would
lead to more generalized representations as well as stronger
synergies between individual feature learning, which we leave
for future work. All those task features don’t necessarily model
physical interaction feasibility for planning; e.g., they can
also serve as a value or energy function of a direct control
policy and be trained via imitation or reinforcement learning
as well [11, 6].
While we have only demonstrated one-robot/static-frame vs.
one-object interactions, addressing interactions between mul-
tiple frames vs. object, e.g. grasping an object with two hands
or elbows, is straightforward by attaching further key inter-
action points on those frames. However, interactions between
two or more functional objects would require to extend our
framework, e.g. by concatenating the objects’ representation
vectors obtained in some pre-deﬁned interaction region, and
predicting features based on this concatenation, the details of
which we need to leave for future work. Alternatively, the
notion of Point-of-Attack can be introduced for some primitive
object-object interactions, like touching, inserting, placing and
pushing, solely from the geometric feature, SDFs [48, 44, 7].
Lastly, we would like to emphasize that the idea of DVCs
is not limited to RGB input. Point clouds can be considered
by replacing the U-net encoder with PointNet [32], which
could be a better choice depending on the setting, e.g., whether
reliable depth perception is available [37]. Incorporating non-
visual, like tactile, input would be another exciting direction
to explore.
ACKNOWLEDGMENTS
This research has been supported by the German Research
Foundation (DFG) under Germany’s Excellence Strategy –
EXC 2002/1–390523135 “Science of Intelligence”.
REFERENCES
[1] Matan Atzmon and Yaron Lipman. SAL: Sign agnostic
learning of shapes from raw data. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2565–2574, 2020.
[2] Michel Breyer, Jen Jen Chung, Lionel Ott, Siegwart
Roland, and Nieto Juan. Volumetric grasping network:
Real-time 6 dof grasp detection in clutter. In Conference
on Robot Learning, 2020.
[3] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas,
Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese,
Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao,
Li Yi, and Fisher Yu. ShapeNet: An Information-Rich 3D
Model Repository. Technical Report arXiv:1512.03012
[cs.GR], Stanford University — Princeton University —
Toyota Technological Institute at Chicago, 2015.
[4] Xu Chen, Zijian Dong, Jie Song, Andreas Geiger, and
Otmar Hilliges. Category level object pose estimation via
neural analysis-by-synthesis. In European Conference on
Computer Vision, pages 139–156. Springer, 2020.
[5] Erwin Coumans and Yunfei Bai.
Pybullet, a python
module for physics simulation for games, robotics and
machine learning, 2016–2021.
[6] Danny Driess*, Jung-Su Ha*, Russ Tedrake, and Marc
Toussaint. Learning geometric reasoning and control for
long-horizon tasks from visual input.
In Proc. of the
IEEE Int. Conf. on Robotics and Automation (ICRA),
2021.

<!-- page 11 -->
[7] Danny Driess, Jung-Su Ha, Marc Toussaint, and Russ
Tedrake.
Learning models as functionals of signed-
distance ﬁelds for manipulation planning.
In Proc. of
the Annual Conf. on Robot Learning (CORL), 2021.
[8] Danny Driess, Jung-Su Ha, Marc Toussaint, and Russ
Tedrake.
Learning models as functionals of signed-
distance ﬁelds for manipulation planning. In 5th Annual
Conference on Robot Learning, 2021.
URL https://
openreview.net/forum?id=FS30JeiGG3h.
[9] Danny Driess, Zhiao Huang, Yunzhu Li, Russ Tedrake,
and Marc Toussaint.
Learning multi-object dynamics
with compositional neural radiance ﬁelds. arXiv preprint
arXiv:2202.11855, 2022.
[10] Clemens Eppner, Arsalan Mousavian, and Dieter Fox.
ACRONYM: A large-scale grasp dataset based on sim-
ulation.
In 2021 IEEE Int. Conf. on Robotics and
Automation, ICRA, 2021.
[11] Pete Florence, Corey Lynch, Andy Zeng, Oscar A
Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong,
Johnny Lee, Igor Mordatch, and Jonathan Tompson.
Implicit behavioral cloning.
In Conference on Robot
Learning. PMLR, 2021.
[12] Peter Florence, Lucas Manuelli, and Russ Tedrake.
Dense object nets: Learning dense visual object descrip-
tors by and for robotic manipulation.
Conference on
Robot Learning, 2018.
[13] Peter Florence, Lucas Manuelli, and Russ Tedrake. Self-
supervised correspondence in visuomotor policy learn-
ing. IEEE Robotics and Automation Letters, 5(2):492–
499, 2019.
[14] Wei Gao and Russ Tedrake. kPAM 2.0: Feedback control
for category-level robotic manipulation. IEEE Robotics
and Automation Letters, 6(2):2962–2969, 2021.
[15] Jung-Su Ha, Danny Driess, and Marc Toussaint.
A
probabilistic framework for constrained manipulations
and task and motion planning under uncertainty.
In
Proc. of the IEEE Int. Conf. on Robotics and Automation
(ICRA), 2020. doi: 10.1109/LRA.2020.3010462.
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Deep residual learning for image recognition, 2016.
[17] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross
Girshick. Mask R-CNN. In IEEE international confer-
ence on computer vision, 2017.
[18] Philipp Henzler, Jeremy Reizenstein, Patrick Labatut,
Roman Shapovalov, Tobias Ritschel, Andrea Vedaldi, and
David Novotny.
Unsupervised learning of 3d object
categories from videos in the wild. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 4700–4709, 2021.
[19] Zhenyu Jiang, Yifeng Zhu, Maxwell Svetlik, Kuan Fang,
and Yuke Zhu. Synergies between affordance and geom-
etry: 6-dof grasp detection via implicit representations.
In Robotics: Science and Systems (RSS), 2021.
[20] Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael
Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea,
and Ken Goldberg. Dex-net 2.0: Deep learning to plan
robust grasps with synthetic point clouds and analytic
grasp metrics. In Robotics: Science and Systems (RSS),
2017.
[21] Khaled Mamou, E Lengyel, and AK Peters. Volumet-
ric hierarchical approximate convex decomposition. In
Game Engine Gems 3, pages 141–158. AK Peters, 2016.
[22] Lucas Manuelli, Wei Gao, Peter Florence, and Russ
Tedrake. kPAM: Keypoint affordances for category-level
robotic manipulation.
In International Symposium on
Robotics Research (ISRR), 2019.
[23] Lucas Manuelli, Yunzhu Li, Pete Florence, and Russ
Tedrake.
Keypoints into the future: Self-supervised
correspondence in model-based reinforcement learning.
Conference on Robot Learning, 2020.
[24] Lars Mescheder, Michael Oechsle, Michael Niemeyer,
Sebastian Nowozin, and Andreas Geiger. Occupancy net-
works: Learning 3d reconstruction in function space. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 4460–4470, 2019.
[25] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.
NeRF: Representing scenes as neural radiance ﬁelds for
view synthesis.
In European conference on computer
vision, pages 405–421. Springer, 2020.
[26] Arsalan Mousavian, Clemens Eppner, and Dieter Fox.
6-dof GraspNet: Variational grasp generation for object
manipulation.
In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 2901–
2910, 2019.
[27] Adithyavairavan Murali, Arsalan Mousavian, Clemens
Eppner, Chris Paxton, and Dieter Fox. 6-dof grasping for
target-driven object manipulation in clutter. In 2020 IEEE
International Conference on Robotics and Automation
(ICRA), pages 6232–6238. IEEE, 2020.
[28] Richard M Murray, Zexiang Li, and S Shankar Sastry. A
mathematical introduction to robotic manipulation. CRC
press, 2017.
[29] Michael Niemeyer, Lars Mescheder, Michael Oechsle,
and Andreas Geiger. Differentiable volumetric rendering:
Learning implicit 3d representations without 3d supervi-
sion. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 3504–
3515, 2020.
[30] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove. DeepSDF: Learning
continuous signed distance functions for shape represen-
tation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 165–
174, 2019.
[31] Keunhong Park, Arsalan Mousavian, Yu Xiang, and
Dieter Fox.
LatentFusion: End-to-end differentiable
reconstruction and rendering for unseen object pose esti-
mation. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 10710–
10719, 2020.
[32] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J

<!-- page 12 -->
Guibas.
Pointnet: Deep learning on point sets for 3d
classiﬁcation and segmentation. In Proceedings of the
IEEE conference on computer vision and pattern recog-
nition, pages 652–660, 2017.
[33] Zengyi Qin, Kuan Fang, Yuke Zhu, Li Fei-Fei, and Silvio
Savarese. KETO: Learning keypoint representations for
tool manipulation. In 2020 IEEE International Confer-
ence on Robotics and Automation (ICRA), pages 7278–
7285. IEEE, 2020.
[34] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,
Luca Sbordone, Patrick Labatut, and David Novotny.
Common objects in 3d: Large-scale learning and eval-
uation of real-life 3d category reconstruction. In Pro-
ceedings of the IEEE/CVF International Conference on
Computer Vision, pages 10901–10911, 2021.
[35] Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
U-net: Convolutional networks for biomedical image
segmentation. In International Conference on Medical
image computing and computer-assisted intervention,
pages 234–241. Springer, 2015.
[36] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo
Morishima, Angjoo Kanazawa, and Hao Li. PIFu: Pixel-
aligned implicit function for high-resolution clothed hu-
man digitization. In International Conference on Com-
puter Vision (ICCV), pages 2304–2314, 2019.
[37] Anthony
Simeonov,
Yilun
Du,
Andrea
Tagliasac-
chi, Joshua B. Tenenbaum, Alberto Rodriguez, Pulkit
Agrawal, and Vincent Sitzmann. Neural descriptor ﬁelds:
Se(3)-equivariant object representations for manipula-
tion. arXiv preprint arXiv:2112.05124, 2021.
[38] Vincent Sitzmann, Michael Zollhoefer, and Gordon Wet-
zstein. Scene representation networks: Continuous 3d-
structure-aware neural scene representations. Advances in
Neural Information Processing Systems, 32:1121–1132,
2019.
[39] Lars
Mescheder
Marc
Pollefeys
Andreas
Geiger
Songyou Peng, Michael Niemeyer. Convolutional occu-
pancy networks. In European Conference on Computer
Vision (ECCV), 2020.
[40] Martin
Sundermeyer,
Arsalan
Mousavian,
Rudolph
Triebel, and Dieter Fox.
Contact-GraspNet: Efﬁcient
6-dof grasp generation in cluttered scenes.
In IEEE
International Conference on Robotics and Automation
(ICRA), 2021.
[41] Andreas ten Pas, Marcus Gualtieri, Kate Saenko, and
Robert Platt. Grasp pose detection in point clouds. The
International Journal of Robotics Research, 36(13-14):
1455–1473, 2017.
[42] Marc Toussaint. A tutorial on Newton methods for con-
strained trajectory optimization and relations to SLAM,
Gaussian Process smoothing, optimal control, and prob-
abilistic inference. In Geometric and Numerical Foun-
dations of Movements. Springer, 2017.
[43] Marc Toussaint, Kelsey Allen, Kevin A Smith, and
Joshua B Tenenbaum.
Differentiable physics and sta-
ble modes for tool-use and manipulation planning. In
Robotics: Science and Systems, 2018.
[44] Marc Toussaint, Jung-Su Ha, and Danny Driess. Describ-
ing physics for physical reasoning: Force-based sequen-
tial manipulation planning. IEEE Robotics and Automa-
tion Letters, 2020. doi: 10.1109/LRA.2020.3010462.
[45] Alex Trevithick and Bo Yang. GRF: Learning a general
radiance ﬁeld for 3d scene representation and rendering.
In International Conference on Computer Vision (ICCV),
2021.
[46] Dylan Turpin, Liquan Wang, Stavros Tsogkas, Sven
Dickinson, and Animesh Garg.
GIFT: Generalizable
interaction-aware functional tool affordances without la-
bels. In Robotics: Science and Systems, 2021.
[47] Mark Van der Merwe, Qingkai Lu, Balakumar Sundar-
alingam, Martin Matak, and Tucker Hermans. Learning
continuous 3d reconstructions for geometrically aware
grasping.
In 2020 IEEE International Conference on
Robotics and Automation (ICRA), pages 11516–11522.
IEEE, 2020.
[48] Jiayin Xie and Nilanjan Chakraborty.
Rigid body dy-
namic simulation with line and surface contact. In 2016
IEEE International Conference on Simulation, Modeling,
and Programming for Autonomous Robots (SIMPAR),
pages 9–15. IEEE, 2016.
[49] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir
Mech, and Ulrich Neumann. DISN: Deep implicit surface
network for high-quality single-view 3d reconstruction.
Advances in Neural Information Processing Systems, 32:
492–502, 2019.
[50] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun,
Matan Atzmon, Basri Ronen, and Yaron Lipman. Multi-
view neural surface reconstruction by disentangling ge-
ometry and appearance. Advances in Neural Information
Processing Systems, 33, 2020.
[51] Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Al-
berto Rodriguez, Phillip Isola, and Tsung-Yi Lin. iNeRF:
Inverting neural radiance ﬁelds for pose estimation. In
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), 2021.
[52] Yifan You, Lin Shao, Toki Migimatsu, and Jeannette
Bohg.
OmniHang: Learning to hang arbitrary objects
using contact point correspondences and neural colli-
sion estimation.
In IEEE International Conference on
Robotics and Automation (ICRA). IEEE, 2021.
[53] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo
Kanazawa.
pixelNeRF: Neural radiance ﬁelds from
one or few images.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2021.
[54] Wentao Yuan, Chris Paxton, Karthik Desingh, and Dieter
Fox.
SORNet: Spatial object-centric representations
for sequential manipulation. In 5th Annual Conference
on Robot Learning, 2021. URL https://openreview.net/
forum?id=mOLu2rODIJF.
[55] Andy Zeng, Pete Florence, Jonathan Tompson, Stefan
Welker, Jonathan Chien, Maria Attarian, Travis Arm-

<!-- page 13 -->
strong, Ivan Krasin, Dan Duong, Vikas Sindhwani, and
Johnny Lee.
Transporter networks: Rearranging the
visual world for robotic manipulation.
Conference on
Robot Learning (CoRL), 2020.
[56] Andy Zeng, Shuran Song, Johnny Lee, Alberto Ro-
driguez, and Thomas Funkhouser. Tossingbot: Learning
to throw arbitrary objects with residual physics. IEEE
Transactions on Robotics, 36(4):1307–1319, 2020.

<!-- page 14 -->
APPENDIX
A. Homography Transformation
The idea of the Homography warping is that two images
taken by cameras at the same position but with different
orientations and intrinsics can be transformed into each other.
Suppose that we have a source image I with the camera
position t, rotation matrix R and projection matrix K and
that an object is inside a bounding sphere at p ∈R3 with a
radius r ∈R+. An image focusing on the bounding sphere
can be taken from a (synthetic) camera at the same position t
with the view direction as t−p and the ﬁeld of view angle as
2 arcsin(||t −p||2/r), from which we can compute the new
camera rotation matrix ˆR and the intrinsic ˆ
K.
Given ˆR and ˆ
K, the new ﬁeld warped by the corresponding
Homography can be obtained as follows: First, a pixel in the
source image, p1 = (u, v, 1), is reprojected into a ray in the
3D space: P1 = K−1p1. Next, the ray is viewed in the new
camera coordinate: P2 = ˆRT RP1. Lastly, this ray is projected
back into a pixel in the new camera: p2 = ˆKP2. Putting all
together, the Homography warping is given as:
W( ˆR, ˆ
K) :


u
v
1

7→w ˆ
K ˆRT RK−1


u
v
1

,
(17)
where w is the parameter that makes the last element of the
output homogeneous coordinate 1, which results in the warped
image ˆI with its camera pose ˆT =
 ˆR
t
0
1

and intrinsic
matrix ˆ
K.
B. Manipulation Constraints
In this work, we consider two discrete actions, (GRASP,
gripper, mug) and (HANG, hook, mug), for grasping and
hanging, respectively. Each action imposes three constraints
on the path as follows.
• The action ak = (GRASP, gripper, mug) ﬁrst imposes
the learned grasping constraint at the end of its phase,
Hi
grasp(xt, δqi
t) = 0, t = kT, i.e.,
 T(δqi
t)[φi
grasp] ◦FKj

(xt) = 0,
(18)
where i and j are indices of the mug and the gripper,
respectively. It also imposes the zero-impact switching
constraint at t = kT for the smooth transition, i.e.,
ˆvt = 0,
(19)
where ˆvt is a joint velocity computed from xt−1 and
xt via ﬁnite difference. Lastly, it introduces an equality
constraint on the gripper’s approaching direction for
collision-safe grasping; more precisely, the constraint is
imposed at t ∈{kT −2, kT −1, kT} as:
j ˆai
t = aapproach


0
0
−1

,
(20)
where j ˆai
t is the mug’s acceleration in the gripper’s coor-
dinate computed from jt(δqi
t−2), jt(δqi
t−1) and jt(δqi
t)
via ﬁnite difference, and aapproach ∈R+ is the predeﬁned
approaching acceleration magnitude. The gripper’s z axis
is depicted in Fig. 5(a) as a blue arrow. Combined with
the above zero-impact constraint, this constraint enforces
the gripper to approach the mug in the gripper’s -z axis
direction and to stop moving at the end of the phase.
• Similarly, the action ak = (HANG, hook, mug) consists
of the learned hanging constraint, the zero-impact and
hanging approaching constraints as
 T(δqi
kT )[φhang] ◦FKj

(xkT ) = 0,
(21)
ˆvkT = 0,
(22)
j ˆai
t = aapproach


0
0
1

, ∀t ∈{kT −2, kT −1, kT}, (23)
where i and j are indices of the mug and the hook,
respectively, and the hook’s z axis is the blue arrow in
Fig. 5(b) (or outer product of the red and green arrow).
The discrete actions above affect the consecutive symbolic
states. While sk indicates a mug is grasped by a gripper or
hung on a hook at the phase k, we impose the following path
constraint: ∀t ∈{(k −1)T + 1, · · · , kT}
δqi
t −δqi
t−1 = FKj(xt) −FKj(xt−1),
(24)
where i and j are indices of the mug and the gripper/hook,
respectively. Effectively this introduces a static joint between
the two frames [43] so the mug moves along with its parent
frame (the gripper or hook). The collision constraints are also
imposed along the trajectory, where the pair collisions with the
mug are computed by the learned SDF feature. We introduce
the collision feature in the following section.
We would like to emphasize that our manipulation planning
framework is not limited by the constraints we introduced
above, but it can incorporate any existing constraint models
and methods for general dexterous manipulation, e.g., [43, 15,
44, 8].
C. Deﬁning Pair-Collision Constraints with SDFs
For manipulation planning problems written only by convex
meshes, the distance or penetration of two objects, which
we call pair-collision features, are computed with either
Gilbert-Johnson-Keerthi (GJK) for non-penetrating objects or
Minkowski Portal Reﬁnement (MPR) for penetrating objects.
In this section, we introduce how to deﬁne pair-collision
features when one or both objects are given as SDFs.
SDF vs. Sphere:
Let δqi, qj and rj be the rigid trans-
formation of PIFO, the sphere’s pose and radius, respectively.
Then the pair-collision feature is simply given by:
dij = T(δqi)[φSDF](t(qj)) −rj.
(25)
SDF vs. Capsule:
Let δqi, qj, hj and rj be the rigid
transformation of PIFO, the capsule’s pose, height and radius,

<!-- page 15 -->
respectively. The pair-collision feature is given by the solution
of the following optimization:
dij =
min
−hj/2≤z≤hj/2 T(δqi)[φSDF]

R(qj)


0
0
z

+ t(qj)

−rj.
(26)
SDF vs. Mesh: Let δqi and qj be the rigid transformation
of PIFO, the mesh’s pose, respectively.
dij =
min
p1∈R3,p2∈R3
T (δqi)[φSDF](p1)=0
dj(p2)=0
nT
1 (p2 −p1),
(27)
where n1 is the normal vector of φSDF at p1 and dj(p2) is the
signed distance of p2 to the mesh computed by GJK/MPR.
SDF vs. SDF: Let δqi and δqj be the rigid transformations
of two PIFOs.
dij =
min
p1∈R3,p2∈R3
T (δqi)[φi
SDF](p1)=0
T (δqj)[φj
SDF](p2)=0
nT
1 (p2 −p1).
(28)
The optimizations in (26)–(28) should be run multiple
times from different initial guesses because the object shape
represented as SDF can be non-convex. In practice, we found
approximating the meshes by a number of spheres and comput-
ing the collision feature much more efﬁcient because querying
the network φSDF at multiple points can be done in parallel on
GPUs.
D. Network Parameters
Image encoder has the U-net architecture [35], especially
with the headless ResNet-34 [16] as its downward path and
two residual 3 × 3 convolutions followed by up-convolution
as the upward path. The number of output channels is 64.
3D reprojector computes the coordinate feature as 32-
dimensional vector using one linear+ReLU layer and concate-
nate it with the local image feature. They are passed to two
hidden layers with the width of (256, 128) followed by ReLUs.
Therefore, the dimension of the representation vector is 128.
SDF head takes as input one representation vector and
computes the output through one hidden layer with the width
of 128 followed by ReLU.
Grasp and hang heads take as input 27 and 5 representa-
tion vectors at their interaction points (depicted in Fig. 5) and
predict the feature through two hidden layers with the widths
of (256, 128) followed by ReLUs.
As shown in Figure 19, the network structures for com-
parison in Section VI-A was kept similar to the above as
possible. Image encoders of the global image feature and
vector representation networks are the ResNet-34 returning
64-dimensional vector. The feature head structures remain the
same, but, because the vector representation scheme doesn’t
represent objects as implicit functions, the input of their
feature head is the frame’s pose as 7-dimensional vector
(3D translation+4D quaternion). The grasp and hang heads
of the SDF representation scheme take as input 27- and 5-
dimensional vectors of their interaction points SDF values.
E. Hand-Designed Constraint Models
Throughout the experiments, objects are represented by
meshes, especially with convex-decomposition using the V-
HACD library [21] for non-convex shapes, and thus pair-
distance and collision between meshes can be computed via
the GJK/MPR algorithm. On top of this mesh representation,
the grasping and hanging constraints are deﬁned and optimized
as follows.
The grasping constraint consists of the aforementioned
collision constraints and the so-called oppose constraint. The
oppose feature takes as input three meshes, FINGER1, FIN-
GER2, and (a set of decomposed) OBJECT meshes to grasp.
It computes the minimum pair-distances from FINGER1 and
FINGER2 to OBJECT and returns summation of those two
vectors, i.e., vFINGER1→OBJECT + vFINGER2→OBJECT. Making the
oppose feature 0 places the object in the middle of two ﬁngers
with proper orientation. This geometric heuristic is inspired by
the notion of force-closure for two-point grasping [28, 41] and
works very well for simple shapes, such as spheres, capsules,
etc. Because the mug shapes are highly non-convex we ran
the optimization from 100 initial seeds and took the best one
with the minimum constraint violation.
The hanging feature, given the object mesh, iteratively
generates a collision-free pose (up to 10,000 iterations) and
checks if the hook is kinematically trapped by the mug
(as done in data generation). If trapped, it returns the pose
difference so that optimizer can output the found pose.
F. Additional Tables and Figures
The content is in the next pages.

<!-- page 16 -->
# of views
Method
SDF error (×10−3)
Volumetric IoU
Chamfer-L1 (×10−3)
2
PIFO
2.91 / 4.63
0.760 / 0.577
6.14 / 8.84
Global Image Feature
3.58 / 4.96
0.642 / 0.515
8.50 / 10.8
Vector Representation
15.6 / 15.8
0.045 / 0.046
39.1 / 40.4
SDF Representation
2.11 / 3.48
0.786 / 0.622
5.78 / 8.13
4
PIFO
2.20 / 3.38
0.816 / 0.656
5.26 / 6.90
Global Image Feature
2.82 / 3.93
0.697 / 0.581
7.42 / 9.49
Vector Representation
15.0 / 15.2
0.036 / 0.014
38.6 / 39.7
SDF Representation
1.43 / 2.73
0.845 / 0.667
4.90 / 6.83
8
PIFO
1.68 / 2.72
0.851 / 0.683
4.78 / 6.34
Global Image Feature
2.31 / 3.51
0.728 / 0.607
6.75 / 8.80
Vector Representation
14.6 / 15.3
0.033 / 0.006
38.7 / 40.6
SDF Representation
1.07 / 2.07
0.878 / 0.703
4.51 / 6.06
TABLE II: SDF Feature Evaluation (Training / Test). The SDF errors were also measured at the same grid points as IoU.
# of views
Method
Grasp (%)
Grasp+c (%)
Hang (%)
Hang+c (%)
2
PIFO
65.8 / 55.4
82.9 / 77.1
87.2 / 71.4
88.2 / 72.1
Global Image Feature
67.6 / 63.9
80.9 / 70.4
88.3 / 70.4
86.3 / 71.8
Vector Representation
13.2 / 12.9
0.8 / 0.4
25.6 / 21.8
0.0 / 0.0
SDF Representation
41.2 / 55.3
49.6/ 45.7
2.6 / 1.1
3.3 / 2.1
4
PIFO
69.0 / 63.9
88.1 / 82.5
88.7 / 75.4
94.0 / 78.9
Global Image Feature
62.3 / 61.8
82.7 / 75.7
90.3 / 75.7
91.2 / 78.2
Vector Representation
21.2 / 22.5
0.5 / 0.4
55.1 / 46.4
0.0 / 0.0
SDF Representation
49.1 / 46.1
67.9 / 64.3
3.3 / 2.9
3.7 / 4.3
8
PIFO
71.9 / 69.3
88.7 / 85.0
91.7 / 80.4
96.5 / 82.5
Global Image Feature
71.3 / 67.1
84.0 / 79.3
91.3 / 77.5
92.9 / 80.4
Vector Representation
29.0 / 23.9
0.5 / 0.7
65.9 / 49.6
0.0 / 0.0
SDF Representation
51.4 / 52.1
75.5 / 70.4
4.6 / 6.1
6.3/ 5.7
TABLE III: Task Feature Evaluation (Training / Test).

<!-- page 17 -->
(a) SDF
(b) Grasp
(c) Hang
(d) Camera
(e) Image
Fig. 16: Data Generation

<!-- page 18 -->
(a) Train Mugs
(b) Test Mugs
Fig. 17: Reconstruction via marching cube. Red: ground truth, Blue: reconstructed

<!-- page 19 -->
(a) Scene: Four cameras’ poses are depicted as coordinate axes where the origin is the camera location, −z axis
(blue) is pointing the view direction, and x and −y axes (red and blue) are the directions of (u, v) coordinate
of images.
(b) Raw images and masks
(c) Warped images (via the multi-view processing)
Fig. 18: Multi-view processing

<!-- page 20 -->
64
64
64
64
32
128
128
16
256
256
8
512
512
4
Bottleneck Conv
256
256
256
8
128
128
128
16
64
64
64
32
64 64 64
64
1
64
local
img
feat
1
3
p
1
3
z
1
32
coord
feat
×
1
256
local2
1
128
local3
projection
(a) PIFO
64
64
64
64
32
128
128
16
256
256
8
512
512
4
1
64
global
img
feat
1
3
p
1
3
z
1
32
coord
feat
×
1
256
local2
1
128
local3
projection
(b) Global Image Feature
64
64
64
64
32
128
128
16
256
256
8
512
512
4
1
64
img
feat
1
6
cam
params
1
32
cam
feat
×
1
256
1
128
(c) Vector Object Represnetation
Fig. 19: Baseline Networks used for comparison.

<!-- page 21 -->
(a) t=5
(b) t=8.5
(c) t=10
(d) t=15
(e) t=18.5
(f) t=20
(g) t=25
(h) t=28.5
(i) t=30
Fig. 20: The three-mug scenario. 60 steps of robot conﬁgurations and rigid transformations of three mugs are jointly optimized via the
proposed manipulation framework. This optimization is a 1071-dimensional decision problem (one 7DOF arm for 60 steps and one 7DOF
mug for 51, 31, 11 steps = 1071, the mug’s rigid transformations before grasped are not included in optimization) and is solved within 1
minute on a standard laptop.

<!-- page 22 -->
(a) t=0
(b) t=5
(c) t=10
(d) t=13.5
(e) t=15
Fig. 21: The handover scenario. 30 steps of the two arms’ conﬁgurations and rigid transformations of the mug are jointly optimize dvia the
proposed manipulation framework. This optimization is a 567-dimensional decision problem (two 7DOF arms for 30 steps and one 7DOF
mug for 21 steps = 567, the mug’s rigid transformations at the ﬁrst phase are not included in optimization) and is solved within 1 minute
on a standard laptop.

<!-- page 23 -->
(a) Sampled grasp pose
(b) Sampled hang pose
(c) IK result (inﬁseable)
(d) Sampled grasp pose
(e) Sampled hang pose
(f) IK result (inﬁseable)
(g) Sampled grasp pose
(h) Sampled hang pose
(i) IK result (inﬁseable)
Fig. 22: IK with generative models - Pick & Hang. Separately generated poses often can not be coordinated due to the kinematic infeasibility,
i.e., the robot joint angle limits, or the collision constraints.

<!-- page 24 -->
(a) Sampled grasp1 pose
(b) Sampled grasp2 pose
(c) IK result (inﬁseable)
(d) Sampled grasp1 pose
(e) Sampled grasp2 pose
(f) IK result (inﬁseable)
(g) Sampled grasp1 pose
(h) Sampled grasp2 pose
(i) IK result (inﬁseable)
Fig. 23: IK with generative models - Handover. Separately generated poses often can not be coordinated due to the kinematic infeasibility,
i.e., the robot joint angle limits, or the collision constraints.

<!-- page 25 -->
(a)
(b)
(c)
Fig. 24: First 5 principal components from PCA on image features. The ﬁrst component indicates the object vs. non-object areas, the second
component distinguishes the handle parts, and the third one spots the above vs. below of the mugs, etc. Note that the network is trained
only via the task feature supervisions.

<!-- page 26 -->
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(a) Train mugs (1st)
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(b) Test mugs (1st)
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(c) Train mugs (2nd)
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(d) Test mugs (2nd)
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(e) Train mugs (3rd)
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
0.10.0 0.1
0.1
0.0
0.1
0.1
0.0
0.1
(f) Test mugs (3rd)
Fig. 25: First 3 principal components from PCA on representation vectors of the 3D surface points. It distinguishes the handles of the mugs
from the other parts and is consistent across different mugs.

<!-- page 27 -->
(a) Model (right) and target (left) mugs
(b) Point clouds for ICP
(c) Point clouds for ICP2 obtained from meshes reconstructed via φSDF
(d) Grid points for FCP
Fig. 26: 6D Pose Estimation. (b) Point clouds for ICP are obtained from depth cameras at the same locations/orientations as the RGB
cameras. The size of the point clouds is 1000. (c) Point clouds for ICP are sampled from the surfaces of the meshes reconstructed via the
learned φSDF. The size of the point clouds is 1000. (d) FCP uses 103 grid points for the target and 53 grid points (in smaller area) for the
model, respectively.

<!-- page 28 -->
(a) ICP
(b) ICP2
(c) FCP
(d) F+ICP2
(e) ICP
(f) ICP2
(g) FCP
(h) F+ICP2
(i) ICP
(j) ICP2
(k) FCP
(l) F+ICP2
(m) ICP
(n) ICP2
(o) FCP
(p) F+ICP2
Fig. 27: 6D Pose Estimation Results - the estimated poses are applied to the green meshes. ICP easily gets stuck at local optima while FCP
produces fairly accurate poses which help F+ICP2 escape the local optima; note that FCP does not iterate to get the results.

<!-- page 29 -->
(a) t=5
(b) t=10
(c) t=15
Fig. 28: Zero-shot Imitation - reference motion. Two sets of posed images are obtained at t = 10, 15.

<!-- page 30 -->
(a) t=5
(b) t=10
(c) t=15
(d) t=5
(e) t=10
(f) t=15
(g) t=5
(h) t=10
(i) t=15
Fig. 29: Zero-shot imitation - optimized motions. The FCP constraints are imposed at t = 10, 15. The imitations are achieved only from
images, without deﬁning the canonical coordinate/pose of the objects.

<!-- page 31 -->
Fig. 30: Real robot transfer: Mug materials used by domain randomization.
Fig. 31: Real robot transfer: Multi-view processing with real images. The object masks are obtained from Mask R-CNN. 8 images were
taken and the mask detection failed in one image.
