<!-- page 1 -->
Synergies Between Affordance and Geometry:
6-DoF Grasp Detection via Implicit Representations
Zhenyu Jiang†, Yifeng Zhu†, Maxwell Svetlik†, Kuan Fang‡, Yuke Zhu†
†The University of Texas at Austin, ‡Stanford University
Abstract—Grasp detection in clutter requires the robot to
reason about the 3D scene from incomplete and noisy perception.
In this work, we draw insight that 3D reconstruction and grasp
learning are two intimately connected tasks, both of which
require a ﬁne-grained understanding of local geometry details.
We thus propose to utilize the synergies between grasp affordance
and 3D reconstruction through multi-task learning of a shared
representation. Our model takes advantage of deep implicit
functions, a continuous and memory-efﬁcient representation, to
enable differentiable training of both tasks. We train the model
on self-supervised grasp trials data in simulation. Evaluation is
conducted on a clutter removal task, where the robot clears clut-
tered objects by grasping them one at a time. The experimental
results in simulation and on the real robot have demonstrated
that the use of implicit neural representations and joint learning
of grasp affordance and 3D reconstruction have led to state-of-
the-art grasping results. Our method outperforms baselines by
over 10% in terms of grasp success rate. Additional results and
videos can be found at https://sites.google.com/view/rpl-giga2021
I. INTRODUCTION
Generating robust grasps from raw perception is an essential
task for robots to physically interact with objects in unstruc-
tured environments. This task demands the robots to reason
about the geometry and physical properties of objects from
partially observed visual data, infer a proper grasp pose (3D
position and 3D orientation), and move the gripper to the
desired grasp conﬁguration for execution. Here we consider
the problem of 6-DoF grasp detection in clutter from 3D point
cloud of the robot’s on-board depth camera. Our goal is to
predict a set of candidate grasps on a clutter of objects from
partial point cloud for grasping and decluttering.
Robot grasping is a long-standing challenge with decades
of research. Pioneer work [12, 47] has cast it as a geometry-
centric task, typically assuming access to the full 3D model of
the objects. Grasps are thereby generated through optimization
on analytical models of constraints derived from geometry and
physics. In practice, the requirement of ground-truth models
has impeded their applicability in unstructured scenes. One
remedy is to integrate these model-driven methods with a
3D reconstruction pipeline that builds the object models from
perception as the precursor step [2, 54, 31]. However, it
demands solving a full-ﬂedged 3D reconstruction problem,
an open challenge in computer vision. Motivated by the new
development of machine learning, in particular deep learning,
recent work on grasping has shifted focus towards a data-
driven paradigm [3, 20, 32], where deep networks are trained
Grasp Detection in Clutter
3D Reconstruction
Partial Observation
Grasp detection 
for occluded regions
3D reconstruction
of graspable parts
Fig. 1: We harness the synergies between affordance and geometry for 6-DoF
grasp detection in clutter. Our model jointly learns grasp affordance prediction
and 3D reconstruction. Supervision from reconstruction facilitates our model
to learn geometrically-aware features for accurate grasps in occluded regions
from partial observation. Supervision from grasp, in turn, produce better 3D
reconstruction in graspable regions.
end-to-end on large-scale grasping datasets, either through
manual labeling [19] or self-exploration [20, 43]. Data-driven
methods have enabled direct grasp prediction from noisy
perception. However, end-to-end deep learning for grasping
often suffers from limited generalization within the training
domains.
Inspired by the two threads of research on geometry-centric
and data-driven approaches to grasping, we investigate the
synergistic relations between geometry reasoning and grasp
learning. Our key intuition is that a learned representation
capable of reconstructing the 3D scene encodes relevant ge-
ometry information for predicting grasp points and vice versa.
In this work, we develop a uniﬁed learning framework that
enables a shared scene representation for both tasks of grasp
prediction and 3D reconstruction, where grasps are represented
by a landscape of grasp affordance over the scene. By grasp
affordance, we refer to the likelihood of grasp success and the
corresponding grasp parameters at each location.
The primary challenge here is to develop a shared rep-
resentation that effectively encodes 3D geometry and grasp
affordance information. Recent work from the 3D vision and
graphics communities has shed light on the merits of implicit
representations for geometry reasoning tasks [34, 40, 6, 42].
Deep implicit functions deﬁne a scene through continuous
and differentiable representations parameterized by a deep
network. The network maps each spatial location to a cor-
arXiv:2104.01542v2  [cs.RO]  21 Jul 2021

<!-- page 2 -->
responding local feature, which can further be decoded to
geometry quantities, such as occupancy [34], signed distance
functions [40], probability density, and emitted color [28, 35].
Implicit neural representations have demonstrated state-of-the-
art results in a variety of 3D reconstruction tasks [34, 55, 7],
due to their ability to represent smooth surfaces in high resolu-
tion. On top of the strengths of encoding 3D geometry, the im-
plicit representations are desirable for our problem for two ad-
ditional reasons: 1) They are differentiable and thus amenable
to gradient-based multi-task training of 3D reconstruction and
grasp prediction, enabling to learn a shared representation for
both tasks; and 2) The representations parameterized by deep
networks can adaptively allocate computational budgets to
regions of importance. Hence, implicit neural representations
can ﬂexibly encode action-related information for the parts of
the scene where grasps are likely to succeed.
To this end, we introduce our model: Grasp detection
via Implicit Geometry and Affordance (GIGA). We develop
a structured implicit neural representation for 6-DoF grasp
detection. Our method extracts structured feature grids from
the Truncated Signed Distance Function (TSDF) voxel grid
fused from the input depth image. A local feature can be
computed from the feature grids given a query 3D coordi-
nate. This local feature is used by the implicit functions for
estimating the grasp affordance (in the form of grasp quality,
grasp orientation, and gripper width of a parallel jaw) and
the 3D geometry (in the form of binary occupancy) at the
query location. The model is jointly trained in simulation with
known 3D geometry and self-supervised grasp trials. With
multi-task supervision of affordance and geometry, our model
takes advantage of the synergies between them for more robust
grasp detection.
We conduct experiments on a clutter removal task [4] in
simulation and on physical hardware. In the experiments,
multiple objects are piled or packed in clutter, and the robot is
tasked to remove the objects by grasping them one at a time.
In each round, the robot receives a single-view depth image
from the on-board depth camera and predicts a 6-DoF grasp
conﬁguration. The ability to reconstruct the 3D scene from
a single view enables GIGA to achieve grasp performance
on par with multi-view input as employed in prior work [4].
Empirical results have conﬁrmed the beneﬁts of implicit neural
representations and the exploitation of the synergies between
affordance and geometry. Our model achieved 87.9% and
69.2% grasp success rates on the packed and pile scenes,
outperforming 74.5% and 60.7% reported by the state-of-
the-art VGN model. We provide qualitative visualizations
of the learned grasp affordance landscape over the entire
scene, showing that our implicit representations have encoded
scene-level context information for collision and occlusion
reasoning. Meanwhile, grasp prediction guides the learned
implicit representations to produce better reconstruction to the
graspable regions of the scene.
We summarize the main contributions of our work below:
• We exploit the synergistic relationships between grasp
affordance prediction and 3D reconstruction for 6-DoF
grasp detection in clutter.
• We introduce structured implicit neural representations
that effectively encode 3D scenes and jointly train the
representations with simulated self-supervision.
• We demonstrate signiﬁcant improvements of our model
over the state-of-the-art in the clutter removal task in
simulation and on the real robot.
II. RELATED WORK
A. Learning Grasp Detection
Grasping has been studied for decades. Pioneer work has
developed analytical methods based on the object models [10,
29, 36, 45, 49, 59]. However, complete models of object
geometry and physics are usually unavailable in real-world
applications. To bridge these analytical methods with raw
perception, prior works have resorted to various 3D recon-
struction methods, such as by ﬁtting known CAD models
to partial observations [24, 26] or completing full 3D model
from partial observations based on symmetry analysis [2, 46].
In recent years, learning methods, especially deep learning
models, have gained increased attention for the grasping prob-
lem [2, 18, 20, 38, 39, 11]. Dex-Net [32, 33] introduced a two-
stage pipeline for top-down antipodal grasping. It ﬁrst samples
candidate 4-DoF grasps. The grasp quality of each grasp
candidate is then assessed by a convolutional neural network.
6-DoF GraspNet [39] extends grasp generation to the SE(3)
space with a variational autoencoder for grasp proposals on a
singulated object. GPD [16, 52] and PointGPD [27] tackled 6-
DoF grasp detection in clutter with a two-stage grasp pipeline.
VGN [4] predicts 6-DoF grasps in clutter with a one-stage
pipeline from input depth images. There is also a line of works
that estimate affordance of an object or a scene ﬁrst and then
detect grasps based on estimated affordance [44, 25, 57]. In
most of the prior works, deep networks are trained end-to-end
with only grasp supervision. In contrast, our method learns
a structured neural representation jointly with self-supervised
geometry and grasp supervisions.
B. Geometry-Aware Grasping
The intimate connection between grasp detection and geom-
etry reasoning has inspired a line of work on geometry-aware
grasping. Bohg and Kragic [1] use a descriptor based on shape
context with a non-linear classiﬁcation algorithm to predict
grasps. DGGN [56] regularized grasps through 3D geometry
prediction. It learns to predict a voxel occupancy grid from
partial observations and evaluates grasp quality from feature
of the reconstructed grid. Most relevant to us is PointSDF
[53], which learns 3D reconstruction via implicit functions
and shares the learned geometry features with the grasp
evaluation network. It used a global feature for the shared
shape representation, and the quantitative results did not show
that geometry-aware features improve grasp performances. On
the contrary, our work demonstrates that through the use of
structured feature grids and implicit functions, GIGA improves
grasp detection and focuses more on the 3D reconstruction of
graspable regions via joint training of both tasks.

<!-- page 3 -->
C. Implicit Neural Representations
Our method is inspired by recent work from the vision
and graphics communities on representing 3D objects and
scenes with deep implicit functions [6, 34, 40]. Rather than
explicit 3D representations such as voxels, point clouds, or
meshes, these works have used the isosurface of an implicit
function to represent the surface of a shape. By parametrizing
these implicit functions with deep networks, they are capable
of representing complex shapes smoothly and continuously
in high resolution. The most common architecture for deep
implicit functions is multi-layer perceptions (MLP), which
encode the geometry information of the entire scene into the
model parameters of the MLP. However, they have difﬁculty in
preserving the ﬁne-grained geometric details of local regions.
To mitigate this problem, hybrid representations have been
introduced to combine feature grid structures and neural
representations [28, 42]. These structured representations are
appealing for the grasping problem, which requires geometry
reasoning in local object parts. Our model extends the archi-
tecture of convolutional occupancy network [42], the state-of-
the-art scene reconstruction model, as the backbone for joint
learning of affordance and geometry.
III. PRELIMINARIES
A. Implicit Neural Representations
Implicit neural representations [51] are continuous functions
Φ that are parameterized by neural networks and satisfy
equations of the form:
F(p, Φθ) = 0,
Φθ : p →Φθ(p)
(1)
where p ∈Rm is a spatial or spatio-temporal coordinate, Φθ
is implicitly deﬁned by relations deﬁned by F, and θ refers
to the parameters of the neural networks. As Φθ is deﬁned
on continuous domains of p (e.g., Cartesian coordinates), it
serves as a memory-efﬁcient and differentiable representation
of high-resolution 3D data compared to explicit and discrete
representations.
B. Occupancy Networks
Here we introduce Occupancy Network (ONet) [34], one of
the pioneer works that learn implicit neural representations for
3D reconstruction. Given an observation of a 3D object, ONet
ﬁts a continuous occupancy ﬁeld deﬁned over the bounding
3D volume of the object. The occupancy ﬁeld is deﬁned as:
o : R3 →{0, 1}.
(2)
Following Equation (1), the occupancy function in ONet is
deﬁned as:
Φθ(p) −o(p) = 0.
(3)
The resulting occupancy function Φθ(p) maps a 3D location
p ∈R3 to the occupancy probability between 0 and 1.
To generalize over shapes, we need to estimate occupancy
functions conditioning on a context vector x ∈X computed
from visual observations of the shape. A function that con-
ditions on the context x ∈X and instantiates an implicit
function Φ : R3 →R is equivalent to a function that takes
a pair (p, x) ∈R3 × X as input and has a real number as
output. The latter function parameterized by a neural network
fθ is the Occupancy Network:
fθ : R3 × X →[0, 1].
(4)
Our model builds on top of the architecture of Convolutional
Occupancy Networks (ConvONets) [42], which is designed
for scene-level 3D reconstruction. ConvONets extend ONet
with a convolutional encoder and a structured representation.
ConvONets encode input observation (point cloud or voxel
grid) into structured feature grids, and condition implicit
functions on local features retrieved from these feature grids.
They demonstrate more accurate 3D reconstructions of local
details in large-scale environments.
IV. PROBLEM FORMULATION
We consider the problem of 6-DoF grasp detection for
unknown rigid objects in clutter from a single-view depth
image.
A. Assumptions
We assume a robot arm equipped with a parallel-jaw gripper
in a cubic workspace with a planar tabletop. We initialize the
workspace by placing multiple rigid objects on the tabletop.
A single-view depth image taken with a ﬁxed side view depth
camera is fused into a Truncated Signed Distance Function
(TSDF) and then passed into the model. The model outputs
6-DoF grasp pose predictions and associated grasp quality. We
train our model with grasp trials generated in a self-supervised
fashion from a physics engine. In simulated training, we
assume the pose and shape of the objects are known.
B. Notations
Observations Given the depth image captured by the depth
camera with known intrinsic and extrinsic, we fuse it into
a TSDF, which is an N 3 voxel grid V where each cell Vi
contains the truncated signed distance to the nearest surface.
We believe this additional distance-to-surface information pro-
vided by TSDF can improve grasp detection performance.
Therefore we convert the raw depth image into TSDF volume
V and pass it into the model.
Grasps
We deﬁne a 6-DoF grasp g as the grasp center
position t ∈R3, the orientation r ∈SO(3) of the gripper,
and the opening width w ∈R between the ﬁngers.
Grasp Quality A scalar grasp quality q ∈[0, 1] estimates
the probability of grasp success. We learn to predict the grasp
quality of a grasp with binary success labels of executing the
grasp trial in simulation.
Occupancy For an arbitrary point p ∈R3, the occupancy
b ∈{0, 1} is a binary value indicating whether this point is
occupied by any of the objects in the scene.

<!-- page 4 -->
Grasp Selection
3D Reconstruction
: grasp quality
q
<latexit sha1_base64="tr0+Q1qhYt0WsSNRDyNJBVRr/k=">AB6HicbVDLSgNBEOyNrxhfUY9eBoPgKexKQL0FvXhM
wDwgWcLspDcZMzu7zswKIeQLvHhQxKuf5M2/cZLsQRMLGoqbrq7gkRwbVz328mtrW9sbuW3Czu7e/sHxcOjpo5TxbDBYhGrdkA1Ci6xYbgR2E4U0igQ2ApGtzO/9YRK81jem3GCfkQHkoecUWOl+mOvWHL7hxklXgZKUGWq/41e3HLI1QGiao1h3PTYw
/ocpwJnBa6KYaE8pGdIAdSyWNUPuT+aFTcmaVPgljZUsaMld/T0xopPU4CmxnRM1QL3sz8T+vk5rwyp9wmaQGJVsClNBTExmX5M+V8iMGFtCmeL2VsKGVFmbDYFG4K3/PIqaV6UvUr5ul4pVW+yOPJwAqdwDh5cQhXuoAYNYIDwDK/w5jw4L86787Foz
TnZzDH8gfP5A9jQE=</latexit>
: gripper width
w
<latexit sha1_base64="kb+UKGUJj5Ow3856JUZe/QgwfHs=">AB6HicbVDLSgNBEOyNrxhfUY9eBoPgKexKQL0FvX
hMwDwgWcLspDcZMzu7zMwqIeQLvHhQxKuf5M2/cZLsQRMLGoqbrq7gkRwbVz328mtrW9sbuW3Czu7e/sHxcOjpo5TxbDBYhGrdkA1Ci6xYbgR2E4U0igQ2ApGtzO/9YhK81jem3GCfkQHkoecUWOl+lOvWHL7hxklXgZKUGWq/41e3HLI1QGiao1
h3PTYw/ocpwJnBa6KYaE8pGdIAdSyWNUPuT+aFTcmaVPgljZUsaMld/T0xopPU4CmxnRM1QL3sz8T+vk5rwyp9wmaQGJVsClNBTExmX5M+V8iMGFtCmeL2VsKGVFmbDYFG4K3/PIqaV6UvUr5ul4pVW+yOPJwAqdwDh5cQhXuoAYNYIDwDK/w5jw4L
86787FozTnZzDH8gfP5A+iVjQc=</latexit>
Affordance
Implicit 
Functions
Input TSDF
𝑥
𝑦
𝑧
Structured feature grids
Geometry
Implicit 
Function
Occupancy probability
Grasp center 
Query point
3D Conv
𝑥
𝑦
𝑧
Projection
Aggregation
𝑥
𝑦
𝑧
2D U-Nets
(𝑥′, 𝑦′, 𝑧′)
(𝑥, 𝑦, 𝑧)
3D feature grid
Projected 2D feature grids
0
1
Local feature
Structured feature grids
q
<latexit sha1_base64="OnFnKB83dxh9t32j2hSsVIBInWk=">AB6Hi
cbVDLTgJBEOzF+IL9ehlIjHxRHYNRo4kXjxCIo8ENmR26IWR2dl1ZtaEL7AiweN8eonefNvHGAPClbSaWqO91dQSK4Nq7eQ2Nre2d/K7hb39g8Oj4
vFJS8epYthksYhVJ6AaBZfYNwI7CQKaRQIbAfj27nfkKleSzvzSRBP6JDyUPOqLFS47FfLldwGyTryMlCBDvV/86g1ilkYoDRNU67nJsafUmU4Ezg
r9FKNCWVjOsSupZJGqP3p4tAZubDKgISxsiUNWai/J6Y0noSBbYzomakV725+J/XTU1Y9adcJqlByZaLwlQE5P512TAFTIjJpZQpri9lbARVZQZm03Bh
uCtvrxOWldlr1K+blRKtWoWRx7O4BwuwYMbqMEd1KEJDBCe4RXenAfnxXl3PpatOSebOYU/cD5/ANszjPM=</latexit>
w
<latexit sha1_base64="vZCHE/t7iJd03U7aLqTPFbIMR4=">AB6Hi
cbVDLTgJBEOzF+IL9ehlIjHxRHYNRo4kXjxCIo8ENmR26IWR2dnNzKyGEL7AiweN8eonefNvHGAPClbSaWqO91dQSK4Nq7eQ2Nre2d/K7hb39g8Oj4
vFJS8epYthksYhVJ6AaBZfYNwI7CQKaRQIbAfj27nfkSleSzvzSRBP6JDyUPOqLFS46lfLldwGyTryMlCBDvV/86g1ilkYoDRNU67nJsafUmU4Ezg
r9FKNCWVjOsSupZJGqP3p4tAZubDKgISxsiUNWai/J6Y0noSBbYzomakV725+J/XTU1Y9adcJqlByZaLwlQE5P512TAFTIjJpZQpri9lbARVZQZm03Bh
uCtvrxOWldlr1K+blRKtWoWRx7O4BwuwYMbqMEd1KEJDBCe4RXenAfnxXl3PpatOSebOYU/cD5/AORLjPk=</latexit>
: orientation
 t
<latexit sha1_base64="q3BiAnGj
A7GuXOvNo69/ct4M2w=">AB+HicbVBNS8NAFHypX7V+NOrRy2IRPJWkV
PRY8OKxgm2FJoTNdtMu3WzC7kaob/EiwdFvPpTvPlv3LQ5aOvAwjDzHm92
wpQzpR3n26psbG5t71R3a3v7B4d1+i4r5JMEtojCU/kQ4gV5UzQnma04d
UhyHnA7C6U3hDx6pVCwR93qWUj/GY8EiRrA2UmDXvVSxwIuxnoRrueB3X
CazgJonbglaUCJbmB/eaOEZDEVmnCs1NB1Uu3nWGpGOJ3XvEzRFJMpHtOh
oQLHVPn5IvgcnRtlhKJEmic0Wqi/N3IcKzWLQzNZJFSrXiH+5w0zHV37ORN
pqkgy0NRxpFOUNECGjFJieYzQzCRzGRFZIlJtp0VTMluKtfXif9VtNtNy
/v2o1Oq6yjCqdwBhfgwhV04Ba60AMCGTzDK7xZT9aL9W59LEcrVrlzAn9gf
f4AUMCTeg=</latexit>
 p
<latexit sha1_b
ase64="Ar/ga5XuN6wrSUfCJF/H+Eg
yRBU=">AB+HicbVBNS8NAFHypX7
V+NOrRy2IRPJWkVPRY8OKxgm2FJoTN
dtMu3WzC7kaob/EiwdFvPpTvPlv3
LQ5aOvAwjDzHm92wpQzpR3n26psbG
5t71R3a3v7B4d1+i4r5JMEtojCU/k
Q4gV5UzQnma04dUhyHnA7C6U3hD
x6pVCwR93qWUj/GY8EiRrA2UmDXvVS
xwIuxnoRns4Du+E0nQXQOnFL0oAS
3cD+8kYJyWIqNOFYqaHrpNrPsdSMcD
qveZmiKSZTPKZDQwWOqfLzRfA5Ojf
KCEWJNE9otFB/b+Q4VmoWh2aySKhWv
UL8zxtmOr2cybSTFNBloeijCOdoK
IFNGKSEs1nhmAimcmKyARLTLTpqmZK
cFe/vE76rabl7etRudVlHFU7hD
C7AhSvowC10oQcEMniGV3iznqwX69
36WI5WrHLnBP7A+vwBSqyTdg=</la
texit>
 t,  p
<latexit sha1_base64="YtmGJc7v
NsFXuheu+Kdh7a7+8vc=">ACXicbVBNS8NAEN3Ur1q/oh69LBbBg5REW
vRY8OKxgm2FJoTNdtMu3WyW3Y1Qq5e/CtePCji1X/gzX/jps3Btj4YeLw3
w8y8UDCqtOP8WJW19Y3Nrep2bWd3b/APjzqSVmHRxwhL5ECJFGOWkq6l
m5EFIguKQkX4uSn8/iORib8Xk8F8WM04jSiGkjBTb0hKBFyM9DqNM5x
eLgsgDu+40nBngKnFLUgclOoH97Q0TnMaEa8yQUgPXEdrPkNQUM5LXvFQR
gfAEjcjAUI5iovxs9kOz4wyhFEiTXENZ+rfiQzFSk3j0HQWF6plrxD/8wa
pjq79jHKRasLxfFGUMqgTWMQCh1QSrNnUEIQlNbdCPEYSYW3Cq5kQ3OWXV0
nvsuE2G627Zr3dKuOoghNwCs6BC65AG9yCDugCDJ7AC3gD79az9Wp9WJ/z1
opVzhyDBVhfv0uemrA=</latexit>
t
<latexit sha1_base64="p4ZaKqML
tTsxts2DozHbhNDBkW8=">AB8XicbVDLSgMxFL1TX7W+qi7dBIvgqsxIf
SwLblxWsA9sh5JM21oJjMkd4Qy9C/cuFDErX/jzr8xbWehrQcCh3PuJe
IJHCoOt+O4W19Y3NreJ2aWd3b/+gfHjUMnGqGW+yWMa6E1DpVC8iQIl7yS
a0yiQvB2Mb2d+4lrI2L1gJOE+xEdKhEKRtFKj72I4igIM5z2yxW36s5BVo
mXkwrkaPTLX71BzNKIK2SGtP13AT9jGoUTPJpqZcanlA2pkPetVTRiBs/
myekjOrDEgYa/sUkrn6eyOjkTGTKLCTs4Rm2ZuJ/3ndFMbPxMqSZErtvg
oTCXBmMzOJwOhOUM5sYQyLWxWwkZU4a2pJItwVs+eZW0LqperXp5X6vUr/
I6inACp3AOHlxDHe6gAU1goOAZXuHNMc6L8+58LEYLTr5zDH/gfP4A9iGRF
A=</latexit>
p
<latexit sha1_base64="tBkbnR2a
+sPJ/zS5hxu5YF8cbgc=">AB8XicbVDLSgMxFL1TX7W+qi7dBIvgqsxIf
SwLblxWsA9sh5J7ShmcyQZIQy9C/cuFDErX/jzr8xbWehrQcCh3PuJe
IBFcG9f9dgpr6xubW8Xt0s7u3v5B+fCopeNUMWyWMSqE1CNgktsGm4EdhK
FNAoEtoPx7cxvP6HSPJYPZpKgH9Gh5CFn1FjpsRdRMwrCLJn2yxW36s5BVo
mXkwrkaPTLX71BzNIpWGCat313MT4GVWGM4HTUi/VmFA2pkPsWiphNrP
5omn5MwqAxLGyj5pyFz9vZHRSOtJFNjJWUK97M3E/7xuasIbP+MySQ1Ktvg
oTAUxMZmdTwZcITNiYglitushI2oszYkq2BG/5FXSuqh6terlfa1Sv8
rKMIJnMI5eHANdbiDBjSBgYRneIU3RzsvzrvzsRgtOPnOMfyB8/kD8A2RE
A=</latexit>
Fig. 2: Model architecture of GIGA. The input is a TSDF fused from the depth image. After a 3D convolution layer, the output 3D voxel features are projected
to canonical planes and aggregated into 2D feature grids. After passing each of the three feature planes through three independent U-Nets, we query the local
feature at grasp center/occupancy query point with bilinear interpolation. The affordance implicit functions predict grasp parameters from the local feature at
the grasp center. The geometry implicit function predicts occupancy probability from the local feature at the query point.
C. Objectives
The primary goal is to detect 6-DoF grasp conﬁgurations
that allow the robot arm to successfully grasp and remove
the objects from the workspace. To foster multi-task learning
of geometry reasoning and grasp affordance, we perform
simultaneous 3D reconstruction. Given the input observation
V, our goal is to learn two functions:
fa : t →q, r, w,
fg : p →b.
(5)
The ﬁrst function fa maps from a grasp center to the rotation,
gripper width, and grasp quality of the best grasp at that
location. Once fa is trained, we can select which grasp to
execute based on the grasp quality at a set of grasp centers.
The second function fg maps any point in the workspace to the
estimated occupancy value at that point. We can extract a 3D
mesh from the learned occupancy function with the Marching
Cube algorithm [30].
V. METHOD
We now present GIGA, a learning framework that exploits
synergies between affordance and geometry for 6-DoF grasp
detection from partial observation. We learn grasp affordance
prediction and 3D occupancy prediction jointly with shared
feature grids and a uniﬁed implicit neural representation.
Figure 2 illustrates the overall model architecture.
A. Structured Feature Grids
To jointly learn the grasp affordance prediction and 3D re-
construction, we need to extract a shared feature from the input
TSDF. In previous implicit-function-based 3D reconstruction
works [34, 40, 53], a ﬂat global feature is extracted from the
input observation and it is used to infer the implicit function.
However, this simple representation falls short of encoding
local spatial information, nor for incorporating inductive biases
such as translational equivariance into the model [42]. For this
reason, they lead to overly smooth surface reconstructions.
For grasp detection, local geometry reasoning is even more
important. The spatial distribution of grasp affordance is very
sparse and highly localized: most viable grasps cluster around
a few graspable regions. Therefore, we adopt the encoder ar-
chitecture from ConvONets [42] and learn to extract structured
feature grids from partial observation.
Our encoder takes as input a TSDF voxel ﬁeld and processes
it with a 3D CNN layer to obtain a feature embedding for
every voxel. Given these features, we construct planar feature
representations by performing an orthographic projection onto
a canonical plane for each input voxel. The canonical plane is
discretized into pixel cells. Then we aggregate the features of
voxels projected onto the same pixel cell using average pool-
ing, which gives us a feature plane. The projection operation
greatly reduces the computation cost while keeping the spatial
distribution of feature points. We apply this feature projection
and aggregation process to all three canonical frames. The
feature plane might have empty features due to the incomplete
observation. We therefore process each of these feature planes
with a 2D U-Net [48] which is composed of a series of down-
sampling and up-sampling convolutions with skip connections.
The U-Net integrates both local and global information and
acts as a feature inpainting network. The output feature grids
denoted as c, are shared for affordance and geometry learning.

<!-- page 5 -->
B. Implicit Neural Representations
A uniﬁed representation of affordance and geometry facili-
tates the joint learning of grasp detection and 3D occupancy
prediction. Recent works in the 3D vision community have
shown that deep implicit functions are capable of representing
shapes in a continuous and memory-efﬁcient way and they
can adaptively allocate computational and memory resources
to regions of importance [14, 28, 42]. Therefore, we are
motivated to use implicit neural representations to encode
both affordance and geometry. As both require reasoning
about local geometry details, we condition the deep implicit
functions on local features. We query the local feature from the
shared feature planes c. Given a query position p, we project
it to each feature plane and query the features at the projected
locations using bilinear interpolation. Unlike ConvONets [42]
where features from different planes are summed together,
we concatenate them instead in order to preserve the spatial
information. The local feature ψp can be formulated as:
ψp = [φ(cxy, pxy), φ(cyz, pyz), φ(cxz, pxz)],
(6)
where cij, pij (i, j ∈{x, y, z}) are the plane feature and point
projected onto the corresponding plane, and φ means bilinear
interpolation of feature plane at the projected point.
a) Affordance Implicit Functions: The affordance im-
plicit functions represent the grasp affordance ﬁeld of grasp
parameters and grasp quality. They map the grasp center t to
grasp parameters (r and w) and the grasp quality metric q.
These implicit neural representations enable learning directly
from data with continuous grasp centers. In contrast, VGN
has to snap grasp centers to the nearest voxel as it uses an
explicit voxel-based grasp ﬁeld. The snapping operation leads
to information loss while our model does not.
We implement implicit neural representations as functions
that map a pair of point and corresponding local feature (t, ψt)
to target values. We parametrize the functions with small
fully-connected occupancy networks with multiple ResNet
blocks [17]. Grasp center, grasp orientation, and gripper width
are output from three separate implicit functions:
fθq(t, ψt) →[0, 1],
fθr(t, ψt) →SO(3),
fθw(t, ψt) →[0, wmax].
(7)
Here wmax is the maximum gripper width. θq, θr, θw are
neural networks parameters of each deep implicit function.
We represent grasp orientation with quaternions.
b) Geometry Implicit Function: Our geometry implicit
function maps from an arbitrary query point p inside the
bounded volume to the occupancy probability o(p) at the
point. Similar to affordance implicit functions, we learn a func-
tion that predicts occupancy based on input point coordinate
and the corresponding local feature:
fθo(p, ψp) →[0, 1].
(8)
Notice that the query points of occupancy p can be different
from the grasp center points t.
C. Grasp Detection
GIGA takes as input a TSDF voxel grid, a grasp center, and
multiple occupancy query points and predicts grasp parameters
corresponding to the grasp center and occupancy probabilities
at the query points.
Given the trained GIGA model, we use a sampling pro-
cedure to select the ﬁnal grasp pose. Grasp affordance is
implicitly deﬁned by the learned neural networks, so we need
to query it from the learned implicit functions. To cover all
possible graspable regions, we discretize the volume of the
workspace into voxel grids and use the position of all the
voxel cells as grasp centers. Then we query the grasp quality
and grasp parameters corresponding to these grasp centers in
parallel. Next, we mask out impractical grasps and apply non-
maxima suppression as done in VGN [4]. Finally, we select
a grasp with the highest quality if the quality is beyond a
threshold. If no grasp has the quality above the threshold, we
don’t make grasp predictions and give up the current scene.
D. Training
The loss for training consists of two parts: the affordance
loss and the geometry loss. For the affordance loss, we adopt
the same training objective as VGN [4]:
LA(ˆg, g) = Lq(ˆq, q) + q(Lr(ˆr, r) + Lw( ˆw, w)).
(9)
Here ˆg denotes predicted grasp parameters and g denotes
ground-truth parameters. q ∈{0, 1} is the ground-truth grasp
quality (0 for failure, 1 for success) and ˆq ∈[0, 1] is the pre-
dicted grasp quality. Lq is a binary cross-entropy loss between
the predicted and ground-truth grasp quality. Lw is the ℓ2-
distance between predicted gripper width ˆw and ground-truth
one w. For orientation, Lquat between predicted quaternion ˆr
and target quaternion r is given by Lquat(ˆr, r) = 1 −|ˆr · r|.
However, the parallel-jaw gripper is symmetric, which means
a grasp conﬁguration corresponds to itself after rotated by
180◦about the gripper’s wrist axis. To handle this symmetry
during training, both mirrored rotations r and rπ are deemed
as ground-truth. Thus the orientation loss is deﬁned as:
Lr(ˆr, r) = min(Lquat(ˆr, r), Lquat(ˆr, rπ)).
(10)
We only supervise the grasp orientation and gripper width
when a grasp is successful (q = 1).
For the geometry loss, we apply the standard binary cross-
entropy loss between the predicted occupancy ˆo ∈[0, 1] and
the ground-truth occupancy label o ∈{0, 1}. The loss is
denoted as LG. The ﬁnal loss is simply the direct sum of
the affordance loss and geometry loss:
L = LA + LG.
(11)
We implement the models with Pytorch [41] and train the
models with the Adam optimizer [23] and a learning rate of
2 × 10−4 and batch sizes of 32.

<!-- page 6 -->
Fig. 3: Visualization of packed (left) and pile (right) scenarios. In the packed
scenario, objects are placed on the table at their canonical poses. In the pile
scenario, objects are dropped on the workspace with random poses. These
objects are from Google Scanned Objects [15] and the scenes are rendered
with NVISII [37].
TABLE I: Quantitative results of clutter removal. We report mean and standard
deviation of grasp success rates (GSR) and declutter rates (DR). HR denotes
high resolution.
Method
Packed
Pile
GSR (%)
DR (%)
GSR (%)
DR (%)
SHAF [13]
56.6 ± 2.0
58.0 ± 3.0
50.7 ± 1.7
42.6 ± 2.8
GPD [16]
35.4 ± 1.9
30.7 ± 2.0
17.7 ± 2.3
9.2 ± 1.3
VGN [4]
74.5 ± 1.3
79.2 ± 2.3
60.7 ± 4.2
44.0 ± 4.9
GIGA-Aff
77.2 ± 2.3
78.9 ± 1.7
67.8 ± 3.0
49.7 ± 1.9
GIGA
83.5 ± 2.4
84.3 ± 2.2
69.3 ± 3.3
49.8 ± 3.9
GIGA (HR)
87.9 ± 3.0
86.0 ± 3.2
69.8 ± 3.2
51.1 ± 2.8
VI. EXPERIMENTS
We study the efﬁcacy of synergies between affordance
geometry on grasp detection in clutter. Speciﬁcally, we would
like to investigate three complementary questions: 1) Can
the structured implicit neural representations encode action-
related information for grasping? 2) Does joint learning of
geometry and affordance improve grasp detection? 3) How
does grasp affordance learning impact the performance of 3D
reconstruction?
A. Experimental Setup
Our model is trained in a self-supervised manner with
ground-truth grasp labels collected from physical trials in sim-
ulation and occupancy data obtained from the object meshes.
The use of TSDF enables zero-shot transfer of our model from
simulation to a real Panda arm from Franka Emika.
a) Simulation Environment: Our simulated environment
is built on PyBullet [8]. We use a free gripper to sample grasps
in a 30×30×30cm3 tabletop workspace. For a fair comparison,
we use the same object assets as VGN, including 303 training
and 40 test objects from different sources [5, 21, 22, 50]. The
simulation grasp evaluations are all done with the test objects,
which are excluded from training.
We collect grasp data in a self-supervised fashion in two
type of simulated scenes, pile and packed as in VGN [4].
In the pile scenario, objects are randomly dropped to a box
of the same size as the workspace. Removing the box leaves
a cluttered pile of objects. In the packed scenario, a subset
of taller objects is placed at random locations on the table
at their canonical pose. Examples of these two scenarios are
shown in Figure 3. Once the scene is created, we randomly
sample grasp centers and grasp orientations near the surface of
the objects and execute these grasp samples in simulation. We
store grasp parameters and the corresponding outcomes of the
grasp trials and balance the dataset by discarding redundant
negative samples.
We collect the occupancy training data in the same scenes
where grasp trials are performed. Upon the creation of a
simulation scene, we query the binary occupancy of a large
number of points uniformly distributed in the cubic workspace
as the training data.
b) Camera Observations: To evaluate the model’s ro-
bustness against noise and occlusion in real-world clutter, we
assume that the robot perceives the workspace by a single
depth image from a ﬁxed side view. If we denote the length
of the workspace as l and use the spherical coordinates with
the workspace center as the origin, the viewpoint of the virtual
camera points towards the origin at r = 2l, θ = π
3 , φ = 0. To
expedite sim-to-real transfer, we add noise to the rendered
images in simulation using the additive noise model [32]:
y = αˆy + ϵ,
(12)
where ˆy is a rendered depth image, α is a Gamma random
variable with k = 1000 and θ = 0.001 and ϵ is a Gaussian
Process noise drawn with measurement noise σ = 0.005 and
kernel bandwidth l =
√
2px. The input to the our algorithm
is a 40 × 40 × 40 TSDF [9] fused from this noisy single-view
depth image using the Open3D library [58].
c) Grasp Execution: We select top grasps to execute by
querying grasp parameters from the learned implicit functions
with a set of grasp centers. For a fair comparison with
VGN [4], our GIGA model samples 40 × 40 × 40 uniformly
distributed grasp centers in the workspace and query the grasp
parameters. However, our implicit representations are continu-
ous, so we can query grasp samples in arbitrary resolutions. In
GIGA (HR), we query at a higher resolution of 60×60×60.
We use a set of clutter removal scenarios to evaluate GIGA
and other baselines. Each round, a pile or packed scene with
5 objects is generated. We take a depth image from the same
viewpoint as training. The grasp detection algorithm generates
a grasp proposal given the input TSDF. We execute the grasp
and remove the grasped object from the workspace. If all
objects are cleared, two consecutive failures happen, or no
grasp is detected, we terminate the current scene. Otherwise,
we collect the new observation and predict the next grasp.
In our experiments, grasp proposals with a predicted grasp
quality below 0.5 are discarded.
B. Baselines
We compare the performance of our method and the fol-
lowing baselines:
• SHAF: We use the highest point heuristic [13] by classic
work of grasping in clutter, rather than the learned grasp
quality, for grasp selection. Among all grasps of quality
over 0.5 predicted by our model, we select the highest
one to grasp from the clutter.

<!-- page 7 -->
Input view
VGN
GIGA-Aff
GIGA
Fig. 4: Visualization of the grasp affordance landscape and predicted grasps.
Red indicates that the method predicts high grasp affordance near the
corresponding area. Green indicates successful grasps and Blue failures.
The circles highlight interesting examples, such as asymmetric affordance
heatmaps and highly occluded objects.
• GPD [16]: Grasp Pose Detection, a two-stage 6-DoF
grasp detection algorithm that generates a large set of
grasp candidates and classiﬁes each of them.
• VGN [4]: Volumetric Grasping Network, a single-stage
6-DoF grasp detection algorithm that generates a large
number of grasp parameters in parallel given input TSDF
volume.
• GIGA-Aff: An ablated version of our method with
only affordance implicit function branch. The network is
trained with only grasp supervision but no reconstruction.
Performance is measured using the following metrics av-
eraged over 100 simulation rounds: 1) Grasp success rate
(GSR), the ratio of success grasp executions; and 2) Declutter
rate (DR), the average ratio of objects removed. The original
VGN uses multi-view inputs, we re-train the VGN model on
the same single-view data we used for training GIGA for fair
comparisons.
C. Grasp Detection Results
We report grasp success rate and declutter rate for different
scenarios in Table I. We can see that GIGA and GIGA-
Aff outperform other baselines in almost all scenarios and
metrics. Even though GIGA-Aff does not utilize the synergies
between affordance and geometry and is trained without geom-
etry supervision, it still outperforms the state-of-the-art VGN
baseline. We attribute this to the high expressiveness of our
implicit neural representations. VGN snaps the grasp center
to voxel grid cells during training. In contrast, our implicit
GIGA
GIGA-Geo
Ground truth
Fig. 5: Qualitative 3D reconstruction results of a scene rendered from the top
view. The circles highlight the contrast.
neural representations learn to ﬁt grasp affordance ﬁeld with
continuous functions. It allows us to query grasp parameters at
a higher resolution as done in GIGA (HR). GIGA (HR) gives
the highest performance in all cases.
Next, we compare the results of GIGA-Aff with GIGA.
In the pile scenario, the gain from geometry supervision is
relatively small (around 2% grasp success rate). However, in
the packed scenario, GIGA outperforms GIGA-Aff by a large
margin of around 5%. We believe this is due to the different
characteristics of these two scenarios. From Figure 3, we can
see that in the packed scene, some tall objects standing in the
workspace would occlude the objects behind them and the oc-
cluded objects are partially visible. We hypothesize that in this
case, the geometrically-aware feature representation learned
via geometry supervision facilitates the model to predict grasps
on partially visible objects. Such occlusion is, however, less
frequent in pile scenarios. These results demonstrate that
synergies between affordance and geometry improve grasp
detection, especially in the presence of occlusion.
We show examples of the learned grasp affordance land-
scape (grasp quality at different locations of the scene) and
predicted top grasps in Figure 4. Our ﬁrst observation is that
our method is able to take into account the context of the scene
and predict collision-free grasps. It often predicts asymmetric
affordance heatmap over symmetric objects (e.g., boxes and
cylinders), where one part of the object is graspable (red)
while the symmetric part is not (grey), and grasping these grey
regions is likely to lead to a collision with the neighboring
objects. This indicates that our model encodes scene-level
information from training on self-supervised grasp trials and
takes into account practical constraints when making grasp
predictions.
Our second observation is that GIGA produces more diverse
and accurate grasp detections compared with the baselines.
In the ﬁrst two rows, we visualize the affordance and grasp
of two packed scenes. In regions marked by the black circle,
GIGA predicts high affordance and more accurate grasps. The
objects in these regions are largely occluded from the view but

<!-- page 8 -->
TABLE II: Quantitative results of 3D reconstruction. We see that models
trained with grasp supervision tend to have better reconstruction performance
near graspable regions than average by a larger margin.
Method
IoU (%)
IoU-Grasp (%)
∆% (IoU-Grasp−IoU)
GIGA-Detach
53.2
68.8
+15.6
GIGA
70.0
78.1
+8.1
GIGA-Geo
80.0
84.0
+4.0
are easy to grasp given full 3D information. For example, the
thin box in the circle of the ﬁrst row affords a wide range
of grasp poses. These easy grasps are not detected in GIGA-
Aff or VGN due to occlusion but are successfully predicted
by GIGA. The last two rows show the affordance landscape
and top grasps for two pile scenes. We see that baselines
without the multi-task training of 3D reconstruction tend to
generate failed or no grasp, whereas GIGA produces more
diverse and accurate grasps due to the learned geometrically-
aware representations.
Another advantage of GIGA is the improvement of the
grasping efﬁciency. The planning time(time between receiving
depth image(s) and returning a list of grasps) of GIGA and
original multi-view VGN is similar, which are 46ms(GIGA)
and 46ms(VGN) on an NVIDIA Titan RTX 2080Ti. However,
GIGA uses single-view depth input, which can be collected
from a ﬁxed camera instantaneously. Meanwhile, the original
VGN model has to collect multi-view inputs by moving
the wrist-mounted camera along a scanning trajectory. The
estimated time cost of the scanning process before each grasp
is about 16-20s. Therefore, GIGA grasps at least one order of
magnitude faster than the VGN.
To further verify GIGA’s ability to predict grasps from
partial observation, we retrain GIGA on a fused TSDF of
multi-view depth images taken from six randomly distributed
viewpoints along a circle in the workspace, the same setup as
VGN [4]. Compared to the single-view setup, acquiring multi-
view observations is more time-consuming and subject to
environmental constraints. Yet with multi-view inputs, GIGA
achieves 88.8% and 69.6% grasp success rates in packed and
pile scenarios, which is on par with the performances of GIGA
on single-view input reported in Table I. We hypothesize that
supervision from 3D reconstruction plays a key role for GIGA
to reason about the occluded parts of the scene.
D. 3D Reconstruction
In the previous section, we have demonstrated that geometry
supervision beneﬁts grasp affordance learning. We now exam-
ine how grasp affordance learning affects 3D reconstruction.
To this end, we evaluate the performance of 3D reconstruction
on pile scenes. We further compare two ablated versions of
our method: GIGA-Geo, which is trained solely for 3D re-
construction without the affordance implicit functions branch,
and GIGA-Detach, which is trained to reconstruct the scene
on the ﬁxed feature grids from GIGA-Aff.
We extract mesh from the learned implicit function through
marching cube algorithm [30]. We use the Volumetric IoU
metric for evaluation, which is the intersection over the union
TABLE III: Quantitative results of clutter removal in real world. We report
GSR, DR, the number of successful grasps, and the number of total grasp
trials (in bracket).
Method
Packed
Pile
GSR (%)
DR (%)
GSR (%)
DR (%)
VGN [4]
77.2 (61 / 79)
81.3
79.0 (64 / 81)
85.3
GIGA
83.3 (65 / 78)
86.6
86.9 (73 / 84)
97.3
(a)
(b)
(c)
(d)
Fig. 6: Examples of real-world grasps by GIGA. (a) and (b) show two
examples of grasps of partially occluded objects. The tilted camera looks
at the workspace from the left. (c) shows an example where GIGA picks up
the bear doll from a localized graspable part. (d) illustrates a typical failure
where the gripper slips off the object due to the small contact surface.
between predicted mesh and ground-truth one. For compar-
ison, we also report the IoU near graspable parts given by
grasp trials, denoted as IoU-Grasp. Speciﬁcally, we sample the
successful grasps and evaluate the IoU in the regions between
the two ﬁngers of these grasps.
Table II shows the results of 3D reconstruction. GIGA-
Geo gives the highest overall IoU and the highest IoU in
graspable parts. This result is unsurprising since GIGA-Geo
is trained to specialize in this task. In contrast, both GIGA
and GIGA-Detach are trained to predict grasps, of which the
spatial distribution is highly localized. Thus, we expect them to
biased their representational resources towards these graspable
regions.
Interesting observations arise when we look at the per-
formance gaps between IoU-Grasp and IoU for the same
method. GIGA-Detach gives the largest delta, then comes the
GIGA. It indicates that scene features learned from grasp
supervision dynamically allocate the representational budget
to actionable regions. In Figure 5, GIGA-Geo shows more
accurate overall reconstruction results than GIGA, consistent
with the quantitative results. In comparison, some parts are
missing in the GIGA results, such as the bottom of the bowl.
However, the missing parts are mostly non-graspable parts,
while graspable parts such as the edges of the bowls are clearly
reconstructed. In addition, we see that GIGA successfully
reconstructs the handle of the cup, while GIGA-Geo either

<!-- page 9 -->
ignores the handle (in the ﬁrst row) or reconstructs a small
proportion of the handle. These results imply that GIGA
attends to the action-related regions of high grasp affordance,
driven by the grasp supervision.
E. Real Robot Experiments
Finally, we test our method in the clutter removal exper-
iments on the real hardware. 15 rounds of experiments are
performed with GIGA and VGN for both the Packed and Pile
scenarios respectively. In each round, 5 objects are randomly
selected and placed on the table. In each grasp trial, we pass
the TSDF from a side view depth image to the model and
execute the predicted top grasp. A grasp trial is marked as a
success if the robot grasps the object and places it into a bin
next to the workspace. We repeatedly plan and execute grasps
till either all objects are cleared or two consecutive failures
occur.
Table III reports the real-world evaluations. GIGA achieves
higher success rates and clears more objects. We show some
grasp examples in Figure 6. Compared with VGN, GIGA is
better at detecting grasps for partially occluded objects and
ﬁnding graspable object parts such as edges and handles. We
also show a failure case. In Figure 6(d), the grasp fails due to
the insufﬁcient contact surface. We believe this is attributed
to the unrealistic contact and friction models in the simulated
environments that we used for generating training data. Videos
of clutter removal can be found in the supplementary video
and on our website.
VII. CONCLUSION
We introduced GIGA, a method for 6-DoF grasp detection
in a clutter removal task. Our model learns grasp detection
and 3D reconstruction simultaneously using implicit neural
representations, exploiting the synergies between affordance
and geometry. Concretely, we represent the grasp affordance
and occupancy ﬁeld of an input scene with continuous implicit
functions, parametrized with neural networks. We learn shared
structured feature grids on multi-task grasp and 3D super-
vision. In experiments, we study the inﬂuence of geometry
supervision on grasp affordance learning and vice versa.
The results demonstrate that utilizing the synergies between
affordance and geometry can improve 6-DoF grasp detection,
especially in the case of large occlusion, and grasp supervision
produces better reconstruction in graspable regions.
Our method can be extended and improved in several fronts.
First, in the training process, we assume only a single ground-
truth grasp pose is provided for each grasp center. We hope
to extend the current work and learn to predict the full distri-
bution of viable grasp parameters with generative modeling.
Second, though our model implicitly learns to avoid collisions
from the training data where collided grasps are marked as
negative, we do not explicitly reason about collision-free paths
to grasps. We plan to utilize the reconstructed 3D scene to
constrain the grasp prediction to be collision-free. We also plan
to adapt this learning paradigm to close-loop grasp planning
by integrating real-time feedback.
ACKNOWLEDGMENTS
We would like to thank Zhiyao Bao for efforts on affordance
visualization. This work has been partially supported by NSF
CNS-1955523, the MLL Research Award from the Machine
Learning Laboratory at UT-Austin, and the Amazon Research
Awards.
REFERENCES
[1] Jeannette Bohg and Danica Kragic. Learning grasping
points with shape context.
Robotics and Autonomous
Systems, 58(4):362–377, 2010.
[2] Jeannette Bohg, Matthew Johnson-Roberson, Beatriz
Le´on, Javier Felip, Xavi Gratal, Niklas Bergstr¨om, Dan-
ica Kragic, and Antonio Morales. Mind the gap-robotic
grasping under incomplete observation. In IEEE Inter-
national Conference on Robotics and Automation, pages
686–693, 2011.
[3] Jeannette Bohg, Antonio Morales, Tamim Asfour, and
Danica Kragic. Data-driven grasp synthesis—a survey.
IEEE Transactions on Robotics, 30(2):289–309, 2013.
[4] Michel Breyer, Jen Jen Chung, Lionel Ott, Siegwart
Roland, and Nieto Juan. Volumetric grasping network:
Real-time 6 dof grasp detection in clutter. In Conference
on Robot Learning, 2020.
[5] Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha
Srinivasa, Pieter Abbeel, and Aaron M Dollar. The ycb
object and model set: Towards common benchmarks for
manipulation research. In International Conference on
Advanced Robotics, pages 510–517. IEEE, 2015.
[6] Zhiqin Chen and Hao Zhang. Learning implicit ﬁelds for
generative shape modeling. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
pages 5939–5948, 2019.
[7] Julian Chibane, Thiemo Alldieck, and Gerard Pons-Moll.
Implicit functions in feature space for 3d shape recon-
struction and completion. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 6970–
6981, 2020.
[8] Erwin Coumans and Yunfei Bai.
Pybullet, a python
module for physics simulation for games, robotics and
machine learning. http://pybullet.org, 2016–2019.
[9] Brian Curless and Marc Levoy.
A volumetric method
for building complex models from range images. In An-
nual Conference on Computer Graphics and Interactive
Techniques, pages 303–312, 1996.
[10] Dan Ding, Yun-Hui Liu, and Shuguo Wang. Computing
3-d optimal form-closure grasps. In IEEE International
Conference on Robotics and Automation, volume 4,
pages 3573–3578. IEEE, 2000.
[11] Kuan Fang, Yuke Zhu, Animesh Garg, Andrey Kurenkov,
Viraj Mehta, Li Fei-Fei, and Silvio Savarese.
Learn-
ing task-oriented grasping for tool manipulation from
simulated self-supervision. The International Journal of
Robotics Research, 39(2-3):202–216, 2020.
[12] Carlo Ferrari and John F Canny. Planning optimal grasps.
In ICRA, volume 3, pages 2290–2295, 1992.

<!-- page 10 -->
[13] David Fischinger, Markus Vincze, and Yun Jiang. Learn-
ing grasps for unknown objects in cluttered scenes. In
2013 IEEE international conference on robotics and
automation, pages 609–616. IEEE, 2013.
[14] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna,
and Thomas Funkhouser. Local deep implicit functions
for 3d shape. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
4857–4866, 2020.
[15] GoogleResearch.
Google
scanned
objects,
September .
URL https://fuel.ignitionrobotics.org/1.0/
GoogleResearch/fuel/collections/Google%20Scanned%
20Objects.
[16] Marcus Gualtieri, Andreas Ten Pas, Kate Saenko, and
Robert Platt.
High precision grasp pose detection in
dense clutter. In International Conference on Intelligent
Robots and Systems, pages 598–605, 2016.
[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Deep residual learning for image recognition. In
Proceedings of the IEEE conference on computer vision
and pattern recognition, pages 770–778, 2016.
[18] Stephen James, Paul Wohlhart, Mrinal Kalakrishnan,
Dmitry Kalashnikov, Alex Irpan, Julian Ibarz, Sergey
Levine, Raia Hadsell, and Konstantinos Bousmalis. Sim-
to-real via sim-to-sim: Data-efﬁcient robotic grasping via
randomized-to-canonical adaptation networks. In IEEE
Conference on Computer Vision and Pattern Recognition,
pages 12627–12637, 2019.
[19] Yun Jiang, Stephen Moseson, and Ashutosh Saxena.
Efﬁcient grasping from rgbd images: Learning using a
new rectangle representation.
In IEEE International
Conference on Robotics and Automation, pages 3304–
3311, 2011.
[20] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian
Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen,
Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke,
et al.
Qt-opt: Scalable deep reinforcement learning
for vision-based robotic manipulation.
arXiv preprint
arXiv:1806.10293, 2018.
[21] Daniel Kappler, Jeannette Bohg, and Stefan Schaal.
Leveraging big data for grasp planning. In IEEE Inter-
national Conference on Robotics and Automation, pages
4304–4311, 2015.
[22] Alexander Kasper, Zhixing Xue, and R¨udiger Dill-
mann. The kit object models database: An object model
database for object recognition, localization and manip-
ulation in service robotics. The International Journal of
Robotics Research, 31(8):927–934, 2012.
[23] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. arXiv preprint arXiv:1412.6980,
2014.
[24] Ulrich Klank, Dejan Pangercic, Radu Bogdan Rusu, and
Michael Beetz. Real-time cad model matching for mobile
manipulation and grasping.
In 2009 9th IEEE-RAS
International Conference on Humanoid Robots, pages
290–296. IEEE, 2009.
[25] Mia Kokic, Johannes A Stork, Joshua A Haustein, and
Danica Kragic.
Affordance detection for task-speciﬁc
grasping using deep learning. In International Confer-
ence on Humanoid Robotics, pages 91–98. IEEE, 2017.
[26] Danica Kragic, Andrew T Miller, and Peter K Allen.
Real-time tracking meets online grasp planning.
In
International Conference on Robotics and Automation,
volume 3, pages 2460–2465, 2001.
[27] Hongzhuo Liang, Xiaojian Ma, Shuang Li, Michael
G¨orner, Song Tang, Bin Fang, Fuchun Sun, and Jianwei
Zhang. Pointnetgpd: Detecting grasp conﬁgurations from
point sets. In International Conference on Robotics and
Automation, pages 3629–3635, 2019.
[28] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua,
and Christian Theobalt. Neural sparse voxel ﬁelds. arXiv
preprint arXiv:2007.11571, 2020.
[29] Yun-Hui Liu. Qualitative test and force optimization of
3-d frictional form-closure grasps using linear program-
ming. IEEE Transactions on Robotics and Automation,
15(1):163–173, 1999.
[30] William E Lorensen and Harvey E Cline.
Marching
cubes: A high resolution 3d surface construction algo-
rithm. ACM Siggraph Computer Graphics, 21(4):163–
169, 1987.
[31] Jens Lundell, Francesco Verdoja, and Ville Kyrki. Robust
grasp planning over uncertain shape completions. arXiv
preprint arXiv:1903.00645, 2019.
[32] Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael
Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea,
and Ken Goldberg. Dex-net 2.0: Deep learning to plan
robust grasps with synthetic point clouds and analytic
grasp metrics. arXiv preprint arXiv:1703.09312, 2017.
[33] Jeffrey Mahler, Matthew Matl, Vishal Satish, Michael
Danielczuk, Bill DeRose, Stephen McKinley, and Ken
Goldberg. Learning ambidextrous robot grasping poli-
cies. Science Robotics, 4(26):eaau4984, 2019.
[34] Lars Mescheder, Michael Oechsle, Michael Niemeyer,
Sebastian Nowozin, and Andreas Geiger.
Occupancy
networks: Learning 3d reconstruction in function space.
In IEEE Conference on Computer Vision and Pattern
Recognition, pages 4460–4470, 2019.
[35] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.
Nerf: Representing scenes as neural radiance ﬁelds for
view synthesis. In European Conference on Computer
Vision, pages 405–421. Springer, 2020.
[36] Brian Mirtich and John Canny.
Easily computable
optimum grasps in 2-d and 3-d. In Proceedings of the
1994 IEEE International Conference on Robotics and
Automation, pages 739–747. IEEE, 1994.
[37] Nathan Morrical, Jonathan Tremblay, Stan Birchﬁeld,
and Ingo Wald. NVISII: Nvidia scene imaging interface,
2020. https://github.com/owl-project/NVISII/.
[38] Douglas Morrison, Peter Corke, and J¨urgen Leitner.
Closing the loop for robotic grasping: A real-time,
generative grasp synthesis approach.
arXiv preprint

<!-- page 11 -->
arXiv:1804.05172, 2018.
[39] Arsalan Mousavian, Clemens Eppner, and Dieter Fox.
6-dof graspnet: Variational grasp generation for object
manipulation.
In IEEE International Conference on
Computer Vision, pages 2901–2910, 2019.
[40] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove. Deepsdf: Learning
continuous signed distance functions for shape repre-
sentation.
In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 165–
174, 2019.
[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zem-
ing Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch:
An imperative style, high-performance deep learning
library.
Advances in Neural Information Processing
Systems, 32:8026–8037, 2019.
[42] Songyou Peng, Michael Niemeyer, Lars Mescheder,
Marc Pollefeys, and Andreas Geiger.
Convolutional
occupancy networks. arXiv preprint arXiv:2003.04618,
2020.
[43] Lerrel Pinto and Abhinav Gupta.
Supersizing self-
supervision: Learning to grasp from 50k tries and 700
robot hours.
In 2016 IEEE international conference
on robotics and automation (ICRA), pages 3406–3413.
IEEE, 2016.
[44] Christoph Pohl, Kevin Hitzler, Raphael Grimm, Antonio
Zea, Uwe D Hanebeck, and Tamim Asfour. Affordance-
based grasping and manipulation in real world applica-
tions. In International Conference on Intelligent Robots
and Systems, pages 9569–9576, 2020.
[45] Jean Ponce, Steve Sullivan, J-D Boissonnat, and J-P
Merlet. On characterizing and computing three-and four-
ﬁnger force-closure grasps of polyhedral objects.
In
[1993] Proceedings IEEE International Conference on
Robotics and Automation, pages 821–827. IEEE, 1993.
[46] Ana
Huam´an
Quispe,
Benoˆıt
Milville,
Marco
A
Guti´errez, Can Erdogan, Mike Stilman, Henrik Chris-
tensen, and Heni Ben Amor. Exploiting symmetries and
extrusions for grasping household objects. In Interna-
tional Conference on Robotics and Automation, pages
3702–3708, 2015.
[47] Alberto Rodriguez, Matthew T Mason, and Steve Ferry.
From caging to grasping. The International Journal of
Robotics Research, 31(7), 2012.
[48] Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
U-Net: Convolutional networks for biomedical image
segmentation. In International Conference on Medical
image computing and Computer-Assisted Intervention,
pages 234–241, 2015.
[49] Anis Sahbani, Sahar El-Khoury, and Philippe Bidaud.
An overview of 3d object grasp synthesis algorithms.
Robotics and Autonomous Systems, 60(3):326–336, 2012.
[50] Arjun Singh, James Sha, Karthik S Narayan, Tudor
Achim, and Pieter Abbeel.
Bigbird: A large-scale 3d
database of object instances.
In IEEE International
Conference on Robotics and Automation, pages 509–516,
2014.
[51] Vincent Sitzmann, Julien NP Martel, Alexander W
Bergman, David B Lindell, and Gordon Wetzstein.
Implicit neural representations with periodic activation
functions. arXiv preprint arXiv:2006.09661, 2020.
[52] Andreas ten Pas, Marcus Gualtieri, Kate Saenko, and
Robert Platt. Grasp pose detection in point clouds. The
International Journal of Robotics Research, 36(13-14):
1455–1473, 2017.
[53] Mark Van der Merwe, Qingkai Lu, Balakumar Sundar-
alingam, Martin Matak, and Tucker Hermans. Learning
continuous 3d reconstructions for geometrically aware
grasping. In International Conference on Robotics and
Automation, 2020.
[54] Jacob Varley, Chad DeChant, Adam Richardson, Joaqu´ın
Ruales, and Peter Allen.
Shape completion enabled
robotic grasping. In 2017 IEEE/RSJ international con-
ference on intelligent robots and systems (IROS), pages
2442–2447. IEEE, 2017.
[55] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir
Mech, and Ulrich Neumann. Disn: Deep implicit surface
network for high-quality single-view 3d reconstruction.
arXiv preprint arXiv:1905.10711, 2019.
[56] Xinchen Yan, Jasmined Hsu, Mohammad Khansari, Yun-
fei Bai, Arkanath Pathak, Abhinav Gupta, James David-
son, and Honglak Lee. Learning 6-dof grasping inter-
action via deep geometry-aware 3d representations. In
IEEE International Conference on Robotics and Automa-
tion, pages 3766–3773, 2018.
[57] Lin Yen-Chen, Andy Zeng, Shuran Song, Phillip Isola,
and Tsung-Yi Lin.
Learning to see before learning
to act: Visual pre-training for manipulation.
In IEEE
International Conference on Robotics and Automation,
pages 7286–7293. IEEE, 2020.
[58] Qian-Yi
Zhou,
Jaesik
Park,
and
Vladlen
Koltun.
Open3D: A modern library for 3D data processing.
arXiv:1801.09847, 2018.
[59] Xiangyang Zhu and Han Ding. Planning force-closure
grasps on 3-d objects. In IEEE International Conference
on Robotics and Automation, volume 2, pages 1258–
1263, 2004.
