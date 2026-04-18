<!-- page 1 -->
ATLAS Navigator: Active Task-driven
LAnguage-embedded Gaussian Splatting
Dexter Ong, Yuezhan Tao, Varun Murali, Igor Spasojevic, Vijay Kumar, Pratik Chaudhari
GRASP Laboratory, University of Pennsylvania
Email:{odexter, yztao, mvarun, igorspas, kumar, pratikac}@seas.upenn.edu
[B] Hierarchical Mapping
Sensors
& Visual 
Odometry
[C] Planning
Trajectory
Dense
Sparse
Trajectory
Optimization
Discrete
Planning
[A] Front-end
Processing
Gaussian
Splatting
Submaps
Regions
Gaussians
G1 G2
G3 G4
G5
Clusters
Is task complete?
Agglomerative
Clustering
Path
Vision Language
Model
Yes. In this image, there is a 
building with a brick wall and 
glass windows. There are 
two doors ...
Find the 
entrance to 
the building
Compressed Language
Features
Fig. 1: Our framework consists of three components. The front-end processing [A] extracts and compresses dense pixel-level language
features from the image. The module also clusters features based on geometry and semantics in the map. The hierarchical mapper [B]
runs bottom-up, ingesting the RGB and depth images and the odometric path from the robot to build a map. The top level of the map
contains the submaps, the middle level the regions, and the bottom level the objects. The local map compsises the loaded submaps.
The other submaps are unloaded to save memory (shown here in gray). The planning module [C] consists of a discrete planner that
operates on the sparse map and generates a reference path, while the dense Gaussians in the local map are used to find the trajectory
to be executed on the robot.
Abstract—We address the challenge of task-oriented navigation
in unstructured and unknown environments, where robots must
incrementally build and reason on rich, metric-semantic maps in
real time. Since tasks may require clarification or re-specification,
it is necessary for the information in the map to be rich enough to
enable generalization across a wide range of tasks. To effectively
execute tasks specified in natural language, we propose a hi-
erarchical representation built on language-embedded Gaussian
splatting that enables both sparse semantic planning that lends
itself to online operation and dense geometric representation
for collision-free navigation. We validate the effectiveness of our
method through real-world robot experiments conducted in both
cluttered indoor and kilometer-scale outdoor environments, with
a competitive ratio of about 60% against privileged baselines.
Experiment videos and more details can be found on our project
page: https://atlasnav.github.io
I. INTRODUCTION
The increasing deployment of robots in real-world environ-
ments for infrastructure inspection [1], search-and-rescue [2,
3] and agriculture [4] necessitates the development of systems
capable of natural language interaction with humans, providing
both textual and visual feedback. This, in turn, requires robots
to autonomously perceive their surroundings, gather relevant
information, and make safe and efficient decisions – capabili-
ties crucial for a variety of open-world tasking approaches
over kilometer-scale environments with sparse semantics.
To enable these capabilities on-board robots with privacy &
compute constraints, we develop a framework to efficiently
store and plan on hierarchical metric-semantic maps with
visual and inertial sensors only. An overview of our method
is shown in Fig. 1.
A cornerstone of autonomous navigation is the creation
of actionable maps that effectively represent the environment
and support diverse navigation and task-specific operations.
Such maps must possess several key desiderata: (1) a consis-
tent association between semantic and geometric information
derived from observations, enabling a holistic understanding
of the environment; (2) an efficient storage mechanism for
navigation, such as a hierarchical organization of semantic
information into submaps coupled with the representation of
geometric information using Gaussian distributions, providing
scalability and precise spatial modeling; and (3) task-relevant
arXiv:2502.20386v1  [cs.RO]  27 Feb 2025

<!-- page 2 -->
identification and adaptability, achieved through scoring Gaus-
sian components to facilitate generalization across various
tasks and adaptation to new objectives.
These properties collectively ensure that the proposed map
is not only manageable but also capable of supporting large-
scale autonomous navigation to complete tasks provided in
natural language. To achieve these goals, we propose an
agglomerative data structure that is consistent across both
geometric and semantic scales built upon 3D Gaussian Splat-
ting [5] (3DGS). We extract semantics using a backbone
vision-language model [6] and store a compressed feature
vector alongside the gaussian points (see Sec. IV-A). This
data structure integrates semantic and geometric information
seamlessly, allowing for efficient navigation, and task-driven
retrieval. The design addresses the key challenges of map
storage, scalability, and task adaptability, making it suitable for
real-world applications. A critical consideration in autonomous
navigation is the ability to efficiently access and process
relevant map information for motion planning. Therefore,
the proposed map structure is designed to enable rapid, in-
silico retrieval of the map region immediately surrounding the
robot (required for efficient planning, see Sec. IV-B), allowing
for timely and efficient local path planning. Simultaneously,
the map must retain and manage long-range environmental
information (see Sec. IV-A), providing a comprehensive un-
derstanding of the overall environment for tasks such as global
path planning and loop closure.
Beyond these map storage requirements, we recognize the
need for this actionable map to complement typical hierarchi-
cal motion-planning strategies, which often employ a discrete
planner for high-level guidance and a low-level planner for
dynamically-feasible and safe navigation. This allows for a
unified framework for mapping and planning from the same
set of sensors which allows our method to be deployed on low
SWaP (size, weight & power) robots. In addition to the reduced
computational and power burden of operation, this formulation
allows us to mitigate any discrepancies in the maps used
between mapping and planning. We formulate the planning
problem within this two-stage framework (see Sec. IV-B) and
leverage task-relevant scoring for utility calculation in the
discrete planning problem, coupled with a sampling-based
motion planner for dynamically-feasible trajectory generation.
The key contributions of this work are summarized as
follows:
• Memory-efficient online language-embedded Gaussian
splatting: We introduce a method for Gaussian splat-
ting that incorporates language embeddings efficiently
to optimize memory usage. This approach combines
dimensionality reduction techniques, such as principal
component analysis (PCA), with efficient online updates,
ensuring that the map maintains a compact and scalable
representation while preserving the necessary geometric
and semantic fidelity.
• Submapping with sparse semantic hierarchical clus-
ters and dense geometry: The proposed map is struc-
tured into sparse semantic clusters that represent regions
or objects and dense geometric representations for navi-
gation. This design leverages a tectonic structure, which
updates submap anchor poses after loop closure. As a
result, the map is compatible with external odometry
sources and pre-built maps, enabling consistent updates
and large-scale navigation.
• Large-scale (>km) autonomous task-driven naviga-
tion with reusable maps: Our approach supports task-
driven navigation by dynamically identifying and re-
trieving map regions relevant to specific objectives. The
system is also compatible with real-time, interactive task
re-specification, allowing for adaptive exploration and
navigation in large-scale environments.
We demonstrate these contributions through experiments
that show the efficiency and semantic retrieval capability of
our mapping framework on datasets (Sec. V-A), and real-
world demonstrations of the full framework on a mobile robot
(Sec. V-B).
II. RELATED WORK
Our proposed work lies at the intersection of active percep-
tion, efficient language-embedded Gaussian splatting, hierar-
chical semantic-metric mapping and planning, and task-driven
navigation. We propose a compact hierarchical language-
embedded map representation built upon Gaussian splatting
that can enable task-driven autonomous navigation while in-
crementally exploring and building a map of the environment.
A. Active Perception
Active perception methods allow the robot to actively select
actions and viewpoints that maximize the information gain
relevant to a given task. This problem has been widely studied
in volumetric representations such as voxel maps or Signed
Distance Field (SDF) maps. In [7, 8, 9, 10], the information
gain is evaluated as the mutual information between the current
map and expected map given future observations. To accelerate
the computation, information gain can be approximated with
cell counting-based methods [11, 12, 13, 14, 15, 16, 17]. The
information-driven approach has also been widely adopted for
autonomous exploration with learned representations such as
Neural Radiance Fields (NeRFs) or neural SDFs. Several ap-
proaches [18] estimate information gain in NeRFs by selecting
future observations from a training dataset or by sampling
viewpoints in the radiance fields [19, 20, 21, 22, 23]. In neural
SDFs, [24] evaluates information gain as the variance of the
model induced by parameter perturbations. The approach in
[25] quantifies information gain by learning the reconstruction
uncertainty with an MLP. Fisher Information has been used as
the proxy for evaluating mutual information for 3D Gaussian
splatting [26]. Combining a voxel map with 3D Gaussian
Splatting, in [27], the information gain is evaluated as the
weighted sum of unobserved volume between rays in the
occupancy map, where the weights are determined by the
transmittance of the Gaussians. The method has been further
extended in [28], by incorporating Fisher Information as part
of the information gain of a candidate viewpoint. Motivated by

<!-- page 3 -->
the theory of Kalman filtering, [29] approximates uncertainty
with the magnitude of the parameter updates of the map, which
is then used for estimating the information gain of candidate
viewpoints.
B. 3D Gaussian Splatting
3D Gaussian splatting proposed in [5] provides a unique
representation that captures geometry with Gaussians while
encoding color and opacity information for high-quality ren-
dering of scenes. Building upon this representation, several
approaches focus on real-time construction of such maps with
color and depth measurements in simultaneous localization
and mapping (SLAM) frameworks. The method in [30] em-
ploys a silhouette mask on rendering for efficient optimization
of the scene. The work in [31] addresses the monocular
SLAM problem with 3DGS along with an analytical Jacobians
for pose optimization. The authors of [32] propose handling
Gaussian parameters differently for color and depth rendering
by more explicitly representing surfaces for depth rendering,
resulting in a more memory-efficient representation of the
scene. Similarly, [33] estimates stability and uncertainty of
the Gaussians for efficient representation of the scene. An
advantage of the explicit representation of the Gaussian points
is the ease of collision checking. In comparison with other
works, most notably [34], we handle collision avoidance as
chance constraints - in our case the size of the confidence
ellipsoids depends on the number of Gaussians in the environ-
ment. There are also numerous methods that address the prob-
lem of scene understanding with language-embedded 3DGS.
These approaches typically obtain 2D language features from
Contrastive Language–Image Pre-training [35] (CLIP) and
other foundation models, and distill these image features into
3D Gaussians. Methods in [36, 37, 38] use scene-specific
autoencoders or quantization to obtain compact representations
of the language features. Other approaches [39, 40] leverage
feature fields similar to the method proposed in [41] for Neural
Radiance Fields (NeRFs). In this work, we focus on an explicit
representation that is amenable to planning. We associate every
Gaussian point in the map with a language embedding that
is subsequently compressed using its principal components,
yielding a scene-agnostic language-embedded map.
A sub-class of approaches [42, 43] processes the scene as
submaps by only optimizing the Gaussians on the submap
level. The approach in [42] performs submap alignment using
multi-view pose refinement on keyframes images. Similar to
these methods, we use a submapping approach for efficient
storage of the map, focusing instead on the problem of
optimizing Gaussians across multiple submaps to facilitate
large-scale navigation.
C. Semantic Mapping and Task-based Navigation
Metric-semantic maps, which integrate both geometric
and semantic information, provide actionable and informa-
tive representations of the environment. Common forms of
metric-semantic maps include semantics-augmented occu-
pancy maps [9, 44], object-based semantic maps [45] and
3D scene graphs [46, 47, 48, 49]. Among these, 3D scene
graphs have emerged as a powerful representation, capable
of capturing broader semantic concepts and the underlying
contextual relationships within environments [46]. These mod-
els provide a compact, symbolic representation of semantic
entities in the environments and their relationships [47, 48].
The concept of hierarchical representations of semantics and
geometry was established by [46], which integrated multiple
layers of information at increasing levels of abstraction. Oth-
ers [49] incorporated additional information such as free space,
object detections, and room categories on top of geometric
representations. This was extended by [50] to incorporate
structural elements (e.g. walls) and clustering rooms. Large
Language Models (LLMs) have also been utilized to infer
semantic relationships in the hierarchy [51] where structure
is not readily available. To further generalize the use of
scene graphs and capture a broader range of concepts for
more complex robotic tasks, language features have been
integrated to enable open-set scene understanding. Clio [52]
constructs a hierarchical scene graph, where the set of ob-
jects and regions are inferred from the list of given tasks.
Similarly, OrionNav [53] leverages open-set segmentation and
LLM-based room clustering to build open-vocabulary scene
graphs. ConceptGraphs [54] constructs open-vocabulary 3D
scene graphs from RGB-D image sequences by leveraging 2D
foundation models for instance segmentation and projecting
semantic features into 3D point clouds. It fuses multi-view
information to create 3D objects annotated with vision and
language descriptors, and uses large vision-language models
to generate object captions and infer inter-object relations,
resulting in comprehensive 3D scene graphs. HOV-SG (Hi-
erarchical Open-Vocabulary 3D Scene Graph) [55] extends
these capabilities to large-scale, multi-story environments. It
introduces a hierarchical structure encompassing floors, rooms,
and objects, each enriched with open-vocabulary features
derived from pre-trained vision-language models. While these
methods that obtain an explicit semantic label at each level
of a sparse, hierarchical representation capture the structure
of known indoor environments, such approaches cannot be
directly applied to unknown and unstructured environments.
We instead try to find a balance between the sparsity provided
by the hierarchical representation in scene graphs and the
flexibility provided by dense, continuous feature embeddings.
D. Task and motion planning
A fundamental challenge in task-driven planning (beyond
mapping and exploration) is the identification of objects and
following paths likely to result in localizing the object in
the minimum possible time.
[56] proposed a planner that
prioritizes the discovery of objects of interest and ensures
their complete and high-resolution reconstruction. Learning
based approaches that estimate semantic maps beyond line
of sight have also been proposed [57] utilizing confidence in
semantic labels from unobserved regions to guide the search.
Recent work has also focused on leveraging hierarchical scene
graphs for semantic search tasks [58] using LLMs to guide

<!-- page 4 -->
an optimal hierarchical planner for semantic search, although
these approaches assume that the map is available a priori.
Partially Observable Markov Decision Processes (POMDPs)
have also been considered for planning on partially unknown
scene graphs [59]. Reinforcement Learning based methods
[60] have also been used to leverage scene graphs and graph
neural networks to compute navigation policies, biasing the
search toward regions of interest. Recently, Vision Language
Frontier Maps (VLFM) [61] utilized a Vision-Language Model
(VLM) for estimating the similarity between the task and
images projected onto a bird’s eye view to score frontiers
by their relevancy. Large Language Models have also been
directly used to select robot behaviors from a fixed library
to select actions that minimize the time required to complete
a task [62]. While these methods either focus entirely on
completing the task with minimal maps along the way, we
instead focus on generating a rich map that is reusable across
a variety of tasks.
III. PROBLEM FORMULATION
We consider the problem of optimizing the policy to mini-
mize the time T it takes a robot to complete a given task. The
robot dynamics are given by
xt+1 = f(xt, ut) + wt,
(1)
0 where xt ∈X and ut ∈U are the state and control input of
the robot at time t, wt is dynamics noise distributed according
to N(0, Σdyn), and the function f : X × U →X represents
the disturbance-free equation of motion. The state space X is
a subset of Rdx and the feasible control set U is a bounded
region in Rdu.
The true, initially unknown, map of the environment m∗
belongs to the space of maps M. In this work, M is a vector
of 3D Gaussian point parameters. The measurement received
from the camera on board the robot at time t is given by
yt = h(xt, m∗) + vt,
(2)
where h : X × M →Y is the observation function, Y is
the vector space of images captured by the camera, and vt is
sensing noise distributed according to N(0, Σobs). We assume
that (vt)t≥1 and (wt)t≥1 are independent random variables.
A policy is a sequence of functions π = (π1, π2, · · · ) such
that for each t ≥1, πt is a function mapping the history of
observations received up to time t to a control input u:
πt : Y × · · · × Y
|
{z
}
=:Yt
→U
with
ut := πt(y1:t).
(3)
We assume that we are provided with an oracle function Ψ that
determines whether the task has been completed or not. Each
task is encoded by a feature vector z that belongs to the space
of all tasks Z, which lies in the same space as the natural
language embedding vector. Since a task can be completed
after a different number of observations, we define the set
of all finite sequences of observations as Y∗= ∪n∈NYn. In
this way, the oracle takes as input the task embedding and the
sequence of observations, and outputs the confidence regarding
whether the task has been completed or not:
Ψ : Z × Y∗→[0, 1].
(4)
Ultimately, we are interested in solving the following problem:
min
π,T
E[T]
subject to
Ψ(z, y0:T ) = 1,
and ∀1 ≤t ≤T :
xt+1 = f(xt, ut) + wt
yt = h(xt, m∗) + vt
P(xt ∈Xfree) ≥1 −η.
(5)
The last constraint requires the robot to remain within the set
of collision-free states Xfree ⊆X above a certain probability.
In this way η ∈(0, 1) represents the tolerance on the collision
probability at each point in time. The expectation is taken over
the randomness generated by the dynamics and sensing noise.
IV. METHOD
We address the problem of task-based navigation and map-
ping while incrementally building a map of the environment
that (1) provides a sparse, hierarchical and semantic represen-
tation for navigation that can be adapted for different tasks in
real-time and (2) contains a dense geometric representation
for collision-free planning. We first present the bottom-up
hierarchical mapping approach and then the top-down planning
approach. The methods interact with each other through new
images acquired by taking an action.
We seek a solution to the problem within a subclass of
policies. These policies first map the history of observations to
a representation of the map as a vector of Gaussian parameters.
They subsequently solve a trajectory planning problem that
maximizes the progress towards achieving the specified task
while retaining safety. Such a policy might be summarized in
the following steps:
1)
ˆ
Mt, ˆxt ←MAPPER(y1:t) (described in section IV-A)
2) Pinfo ←DISC( ˆ
Mt, ˆxt) (described in section IV-B)
3) Pdyn
←CTS( ˆ
Mt, ˆxt, Pinfo) (described in section
IV-B),
where DISC refers to the discrete planner and CTS refers
to the continuous planner.
A. Hierarchical Semantic Perception
Gaussian splatting. A map of the scene is built with a set
of Gaussians, each containing parameters comprising the mean
µ, covariance Σ, color c and opacity o. To render an image
at a given camera pose, the Gaussians are sorted and splatted
on to the image space. Each splatted 3D Gaussian will have a
corresponding 2D mean µ2D and covariance Σ2D in the image
space. The color of a pixel p is obtained according to:
Cp =
n
X
i
ciα(p)i
i−1
Y
j
(1 −αj)

<!-- page 5 -->
where
αi(p) = oi · exp(−1
2(p −µ2D,i)T Σ−1
2D,i(p −µ2D,i).
Depth can be similarly obtained by splatting the means of the
Gaussians. In an incremental mapping approach, Gaussians are
initialized at each mapping step and subsequently optimized.
From a single color and depth observation, each pixel is back-
projected to form a 3D pointcloud. A Gaussian is initialized at
each 3D point with its corresponding color. The parameters of
the Gaussians are optimized by rendering color and depth im-
age through the differentiable rasterization process and a loss
is computed against the original color and depth observations.
Dense language features. For each observation yt, we
extract dense pixel-level language features Ft ∈RNf via the
feature map Φ : Y →Z. In contrast to other approaches
that compute and fuse multi-scale embeddings from CLIP and
other foundation models, we use a lightweight approach from
CLIP-DINOiser [6], which refines MaskCLIP [63] features by
incorporating localization priors extracted with self-supervised
features from DINOv2 [64].
Language-embedded 3D Gaussian splatting. We build
our semantic mapping approach on language-embedded 3D
Gaussian splatting. We optimize isotropic Gaussian parameters
to enable rendering of scenes. In addition to color and opacity,
we embed a compressed language feature vector ˜Ft ∈RNc
in each Gaussian’s parameters. For a given camera pose, a
reconstructed feature image ˆFt ∈RNc is rendered from the
scene and the loss is computed against the original compressed
feature image ˜Ft to optimize the Gaussian parameters.
Feature compression. To capture rich semantic informa-
tion, language features are typically high-dimensional vectors.
For efficient computation and storage of language features in
the Gaussians’ parameters, we require a compact representa-
tion of the language features in the form of ˜F.
Most language-embedded Gaussian splatting approaches
leverage alternate representations like features fields [41, 39]
or scene-specific autoencoders [37] to obtain ˜F. However,
these approaches do not generalize well to new and unknown
environments. We perform dimensionality reduction on the
feature space via Principal Component Analysis (PCA). To
obtain a representative distribution of CLIP embeddings, we
use Incremental Principal Component Analyis [65] (IPCA) to
fit the COCO [66] 2017 dataset. We note that different and
larger datasets can be used as well. The computation of the
basis vectors is done offline. At runtime, we simply need to
project the features on to these vectors to obtain the principal
components.
For each image, we extract the full Nf-dimensional lan-
guage features, obtain the first Nc principal components and
embed them in the map through the 3D Gaussian splatting
process. We can then recover the original dimensions of the
features with the inverse of the PCA for computing similarity
with the task embeddings.
Submapping. In this work, we augment the 3D Gaussian
splatting framework with submaps. As the robot navigates
Xgoal
Xs
Xl
Rloc
H
Loaded Submaps
Unloaded Submaps
Fig. 2: An illustration of the different parameters that are
relevant to the submapping and collision checking process. Xs
denotes the current position of the robot, Xl is the local goal
along the path to the final goal Xgoal. Submaps are loaded
within the region bounded by Rloc.
the environment, it receives color and depth measurements
together with its estimated odometry, and incrementally builds
a map. Since Gaussians are added at every mapping iteration,
the number of Gaussians in the map can quickly grow to the
order of millions of Gaussians and this can be challenging to
manage on a SWaP-constrained robot. To enable large-scale
operations, our proposed mapping system efficiently creates
submaps as the robot explores the environment.
By design, each submap contains a 6 degree-of-freedom
(DoF) anchor pose which serves as the reference frame of its
local coordinate system. The submap stores all parameters of
the 3D Gaussians relative to the anchor pose. In every mapping
iteration, the robot loads all submaps within the range Rloc
into the GPU memory and offloads any other submaps that
are currently on the GPU. If there are no submaps within
a specified range rsubmap, the robot creates a new submap
with a unique ID and the last odometry measurement as the
anchor pose. Every Gaussian instantiated from a measurement
yt is associated with the submap corresponding to xt. These
Gaussians are stored in the local reference frame of the
submap.
The submap design ensures safe and efficient navigation
while significantly reducing the number of Gaussians that need
to be considered by the continuous planning module IV-B. As
an example, in Fig. 2, the continuous planning module only
plans on a map with 369,513 Gaussians instead of the global
map which contains 2.47 million Gaussians. This drastically
reduces the number of operations for collision checking and
hence improves the planning efficiency. In contrast to tradi-

<!-- page 6 -->
tional voxel-based mapping that has a fixed discretization, our
map design does not preallocate memory for the Gaussians,
allowing for a flexible storage structure. This flexibility allows
for denser maps in cluttered environments and sparser maps
in large open spaces.
Our submap design can also accomodate pose corrections
from any underlying Simultaneous Localization and Mapping
(SLAM) algorithm. This ensures map consistency across large-
scale operations with multiple loop closures. Our method
incorporates external pose graph corrections by updating the
anchor poses of the submaps. For efficiency, we assume
that the keyframes associated with each submap are locally
consistent and are rigidly attached to the submap reference
frame.
Metric-semantic clustering In unstructured environments,
approaches such as those in [54, 52, 55] that use image
segmentation priors may fail to give meaningful object clus-
ters. We instead perform metric-semantic clustering on the
language-embedded Gaussians directly. For the set of Gaus-
sians in each submap, we compute the pairwise distances
q between all points using a sum of Euclidean distance qe
and cosine-similarity qs of the compressed language feature
vectors of each Gaussian, weighted by a parameter λ.
q = qe + λ · (1 −qs)
With the computed distance matrix, we perform agglomerative
clustering to create object-level clusters in a hierarchy. For this
work, we partition these clusters into three levels – the submap,
region and object levels. We note that unlike other feature
compression methods, we can perform this clustering with our
embedded features directly since the components from IPCA
still retain important semantic features. This negates the need
to recover the full feature dimensions, allowing for efficient
dense clustering online.
For a given task, we compute the utility of each submap by
querying the object-level clusters to obtain a task-relevancy
score. These clusters would correspond to the leaf nodes of
the tree. From these clusters, we propagate the object-level
relevancy up the tree to obtain utility scores for region- and
submap-level nodes. Following Eq. 8, the utility of each node
in the tree is the sum of the utilities of its children. The utility
at the root of the tree serves as the corresponding utility of
that submap. This submap-level cluster information is stored
in each submap along with the Gaussian splatting parameters.
B. Hierarchical Planning
Since we care about potentially large-scale environments,
we turn our attention to a hierarchical planning strategy
with a discrete planner responsible for long-term reasoning
and a continuous trajectory planner responsible for finding
dynamically-feasible trajectories.
Problem 1: Discrete planning. Given a set of vantage points
and their associated utilities (X, Γ) and task embedding z, we
find the optimal sequence of vantage points P = x1x2...xk
that maximizes the utility along the path while staying within
a given travel budget B.
Letting w be a function on pairs of points w : (xi, xj) →
[0, ∞), we solve
max
P=(x1:k) ψ (z, y0:k)
subject to
k−1
X
i=1
w(xi, xi+1) ≤B
yk = h(xk, m) ∀xk ∈P
(6)
Problem 1 requires us to find a sequence of vantage points
(through the sparse components of the map in Fig. 1) from
which the collected images y0:k maximize the chance of
completing the task. We make a few simplifying assumptions
at this stage: (i) that the locations are not correlated i.e. that
taking a measurement at xi does not affect the utility of
xj ∀j ̸= i; (ii) the path between a pair of points with weight
w ≤dwire lies in free-space and (iii) visiting a vertex is
enough to collect the information regardless of the orientation
required for acquiring the image from the vantage point.
With these assumptions, we can construct this problem as
a graph search problem. We construct a graph G = (V, E),
whose vertices are the set of vantage points X. The utility
function ⊓: V →[0, ∞) specifies the utility of each vertex.
The weight function on the edges is then w : E →[0, ∞)
which is the Euclidean distance between the vertices. To find
the optimal path we seek to solve the following problem:
max
P=(x1:k)
k
X
i=1
⊓(xi)
subject to
k−1
X
i=1
w(xi, xi+1) ≤B
(7)
Since the number of possible vertices is large & we desire
an efficient solution on-board low SWaP robots, we solve the
latter problem in a hierarchical fashion. We assume the set
of vertices is partitioned according to V = ∪p
i=1Vi where
Vi ∩Vj = ∅for all i ̸= j. We form a high level graph ˜G =
( ˜V , ˜E) whose vertex set ˜V = {˜v1, ˜v2, . . . , ˜vp} corresponds to
centroids of vertices in each partition. For each vertex ˜vi in ˜V ,
we add an edge between ˜vi and ˜vj if ||˜vi −˜vj||2 ≤dwire for a
user-defined parameter dwire. Our path planning problem then
involves finding the optimal path through this weighted graph
whose cost does not exceed a given budget. The set of edges
˜E ⊆˜V × ˜V . The weight function ˜w : ˜E →[0, ∞) specifies the
weight of every edge in ˜G as the distance between centroids
corresponding to the endpoint vertices. The utility of every
vertex in ˜G is given by the function ˜⊓: ˜V →[0, ∞), with
˜⊓(˜vi) :=
X
v∈Vi
⊓(v).
(8)

<!-- page 7 -->
We first solve the high level problem:
max
˜
P=(˜x1:k)
k
X
i=1
˜⊓(˜xi)
subject to
k−1
X
i=1
˜w(˜xi, ˜xi+1) ≤B.
(9)
We then plan the path through G that visits the partitions Vi
in the order specified by the sequence of edges in P.
Continuous trajectory planning. For the purpose of syn-
thesizing feasible trajectories, we model the robot as an agent
with unicycle dynamics. Its state x = [p, θ] ∈R2×S1 consists
of its position (on the ground plane z = 0), and heading angle
θ. The control input u = [v, ω] ∈R2 consists of the transla-
tional and angular speed of the robot. Its equations of motion
are given by ˙x = f(x, u), where f : R × S1 × R2 →R2 × R
is defined as
f(x, u) =


cos θ
0
sin θ
0
0
1


v
ω

.
(10)
The trajectory generation problem involves finding a dy-
namically feasible trajectory (per eq 11) that interpolates a
given pair of boundary conditions. We solve the problem as
part of a receding horizon control scheme. It is resolved at
a set frequency, with the boundary conditions and free space
updated using information based on accrued measurements.
We choose the furthest point along the path provided by the
Discrete Planner that is within our planning horizon H and
treat this as the center of our goal region Xgoal for each
iteration.
Problem 2: Continuous Trajectory Planning. Given an
initial robot state x0 ∈Xfree, and goal region Xgoal, find the
control inputs u(·) defined on [0, τ] that solve:
min
u(·),τ J(u(·), τ)
s.t. for all t ∈[0, τ]
˙x(t) = f(x(t), u(t)), x(t) ∈Xfree,
||v(t)||2 ≤vmax, |ω(t)| ≤ωmax,
x(0), θ(0) = x0, θ0, x(τ) ∈Xgoal.
(11)
Problem 2 is a high-dimensional non-convex optimization
problem. Two key hurdles from a computational standpoint are
the nonlinear dynamics of the robot as well as the presence
of obstacles in the environment. We therefore adopt a search-
based procedure, motivated by the idea presented in [67]. In
particular, we seek the minimum weight path in a directed
graph formed by discretizing the continuous state space of the
robot. The vertices of the graph correspond to select states,
whereas the edges correspond to motion primitives emanating
from a particular state that remain within Xfree. Each motion
primitive is obtained by integrating a feasible fixed control
input for a time interval δt. This ensures that any path through
the graph represents a dynamically viable trajectory without
taking collisions into account. The set of fixed controls is
represented as the Cartesian product of uniform discretizations
of spaces of feasible translational and angular speeds. The
set of feasible translational (angular) speeds, [−vmax, vmax]
([−ωmax, ωmax]), is discretized by a grid of size Nv (Nω).
In order to determine whether an edge of the graph lies in
free space, we first discuss how we check whether the robot
collides with an object in the environment. To this end, we treat
the robot as a normal random variable R ∼N(µrob, Σrob)
with mean µrob ∈R3 and covariance Σrob ∈S3
++. We
consider the robot to be in collision with a Gaussian point
represented by the normal random variable Gi ∼N(µi, Σi) if
their distance is below a specified collision radius rcoll > 0.
Defining the predicate collide(R, G) that evaluates to true if
the realizations of R and G represent a configuration in which
a collision occurs, we have
collide(R, Gi) ⇔||R −Gi||2 ≤rcoll.
(12)
Since we have an uncertain map, the chance constraint P(x ∈
Xfree) ≥1 −η amounts to
P(∪Gi∈ˆ
M collide(R, Gi)) ≤η,
(13)
where R is a random normal variable whose mean consists of
the translational components of x, and whose variance is equal
to Σrob. We consider an edge of the graph corresponding to
a motion primitive to lie within Xfree just when each of the
discretization points along the edge are collision-free per the
equation above.
Since the number of Gaussians representing the environ-
ment is often very large, we compute approximate collision
probabilities in a way that does not require iterating through
all the Guassians for every potential collision point. To prune
away a large number of collision checks at the expense of a
specified (small) tolerance in the error of estimation of the
collision probability, we seek the smallest “local radius” Rloc
such that the probability of colliding with obstacles whose
means lie within the radius is to within ptol of colliding with
any of the obstacles in the whole environment. The main point
is that a smaller value of Rloc allows us to focus on collision
checking within a localized region of space, thus significantly
decreasing both the running time and memory complexity of
the collision-checking procedure without sacrificing safety of
the robot. To determine a meaningful value of Rloc, we seek
an upper bound on the probability of collision with a Gaussian
point whose mean lies at distance at least Rloc from the robot.
From the union bound, we have
P(∃Gi ∈ˆ
M : ||µi −µrob||2 ≥Rloc, collide(R, Gi)) ≤
X
i : ||µi−µrob||2≥Rloc
P(collide(R, Gi)).
(14)
Assuming that R and Gi are independent normal variables,
we have that
R −Gi ∼N(µrob −µi, Σrob + Σi),
(15)

<!-- page 8 -->
resulting in
P(collide(R, Gi)) =
P(||(µrob −µi) + (Σrob + Σi)
1
2 Zstd||2 ≤rcoll),
(16)
where Zstd ∼N(0, I3) is a standard normal random variable
used throughout the subsection. To make the calculation above
more tractable, we assume that Σrob = diag(σ2
rob, σ2
rob, σ2
rob)
and Σi = diag(σ2
i , σ2
i , σ2
i ). Hence we get
P(collide(R, Gi)) =
= P
 

µrob −µi
p
σ2
rob + σ2
i
+ Zstd


2
≤
rcoll
p
σ2
rob + σ2
i
!
= P
 
Zstd ∈B
 
e1
||µrob −µi||2
p
σ2
rob + σ2
i
;
rcoll
p
σ2
rob + σ2
i
!!
,
(17)
where the last inequality follows from the isotropic nature of
the standard normal random variable. Plugging in the latter
equality into the union bound, and denoting
Sfar(Rloc) = {Gi ∈ˆ
M : ||µi −µrob||2 ≥Rloc},
(18)
we get
P(∃Gi ∈ˆ
M : ||µi −µrob||2 ≥Rloc, collide(R, Gi)) ≤
X
Sfar
P
 
Zstd ∈B
 
e1
||µrob −µi||2
p
σ2
rob + σ2
i
;
rcoll
p
σ2
rob + σ2
i
!!
≤
X
Sfar
P
 
Zstd ∈B
 
e1
Rloc
p
σ2
rob + σ2
i
;
rcoll
p
σ2
rob + σ2
i
!!
≈
|{Sfar(Rloc)}| P

Zstd ∈B(e1Rloc; rcoll)
q
σ2
rob + σ2avg

,
(19)
where in the last equality we made the approximation that
the characteristic size of Gaussian points is σavg. Finally, we
assume that the density of Gaussian points is approximately
uniform, and equal to ρ. Therefore, denoting the number of
all Gaussians in the global map by N, we make the last
approximation
|{Gi ∈ˆ
M : ||µi −µrob||2 ≥Rloc}| ≈

N −4
3R3
locπρ

,
(20)
resulting in the following estimate
P(∃Gi ∈ˆ
M : ||µi −µrob||2 ≥Rloc, collide(R, Gi)) ≲

N −4
3R3
locπρ

P

N(0, I3) ∈B(e1Rloc; rcoll)
q
σ2
rob + σ2avg

.
(21)
We are interested in finding the smallest Rloc so that the very
last expression, a monotonically decreasing function of Rloc,
is below a specified tolerance ptol. Algorithmically we find
this value via binary search (on the value of Rloc).
The expression for the probability that a standard normal
random variable in 3D lies inside the ball B(e1a; b) may be
obtained as
prob(a, b) =
=
Z a+b
a−b
dx
√
2π e−x2
2

1 −exp

−1
2(b2 −(x −a)2)

=
Z a+b
a−b
dx
√
2π e−x2
2
−exp
 1
2(a2 −b2)
	
√
2π
Z a+b
a−b
e−axdx
= Fnormal(c)

c=a+b
c=a−b −exp

−1
2((a −b)2 −(a + b)2)
	
√
2π a
,
(22)
where Fnormal(c) :=
R c
−∞
dx
√
2π exp{−x2
2 } is the cumulative
distribution funtion of the 1D standard normal variable. In
the previous equation, we used the elementary fact that the
probability that the standard normal variable in 2D lies outside
a sphere (in this case - disk) of radius r centered on the
origin is exp{−r2
2 }. Furthermore, for a fixed value of b > 0,
prob(a, b) is a monotonically decreasing function of a on
[0, ∞). The latter observation may easily be seen by noting
that the reflection about the plane that passes through the
midpoint of two spheres (Snear and Sfar) of the same radius
whose centers lie on a ray emanating from the origin, with the
center of Snear being closer, maps Snear\Sfar to Sfar\Snear
in a way that leaves the Lebesgue measure intact while
decreasing the density of the standard normal variable.
The cost of each valid motion primitive is defined as
J(u(·), τ) := λtτ + (v2 + w2)τ,
where λt weights the time cost with the control efforts. Since
the maximum velocity of the robot is bounded by vmax, we
consider the minimum time heuristic as
h(p) := ||pgoal −p||2/vmax.
Finally, we use A* to search through the motion primitives
tree.
C. Implementation details
1) Mapping: We build our mapping module on the Gaus-
sian splatting method presented by [30]. The most signifi-
cant modifications are the submapping framework with pose
estimates from visual-inertial odometry and the addition of
Gaussian parameters for the language features. We note that
any other Gaussian splatting method can be used as well if it
can meet the compute and latency requirements. We compress
the Nf
= 512 dimensional CLIP features to Nc = 24
dimensions for embedding in the Gaussian parameters. We set
the submap size rsubmap to 2 meters for indoor experiments
and 5 meters for outdoor experiments. We resize the image
stream to 320 × 240 for mapping and set the maximum depth
range to 5 meters. To balance compute resources between the
various modules, we limit the map update rate to 1 Hz.

<!-- page 9 -->
2) Planning: We implemented both the discrete and con-
tinuous planning module in Python 3.8 with Pytorch 2.4.1 and
implement the motion planning library from [67]. We set ptol
to 0.001 and σrob to 0.7 meters, and compute Rloc to be
10 meters. The planning horizon H is set to 5 meters. We
adopted the idea of parallelized collision checks from [29]
to enable efficient continuous planning online. The discrete
planner runs onboard the robot every 5 mapping iterations
while the continuous planner runs at 1 Hz.
3) Task specification:
The task is specified in natural
language by the user. We use a Vit-B-16 encoder for the
CLIP backbone. Text embeddings are computed from the
task prompt using the CLIP text encoder [38]. These text
embeddings are used for computing the relevancy of the fea-
ture embeddings in the map to the task. Task re-specification
is done in the same way. In addition, the relevancy of all
submap object clusters are recomputed and their relevancy
scores are propagated up the hierarchical tree of each submap.
We note that we do not need to re-cluster the submaps for
each new task and only need to recompute the relevancy of
the features embeddings of the clusters. Hence we have a
consistent hierarchical submap structure that can be efficiently
re-queried for various tasks.
4) Task termination: We use LLaVA OneVison [68] to
determine if the task has been completed. Given the language
features of the current observation, we mask out pixels that are
beyond the maximum depth range and compute the relevancy
of the remaining pixels against the task. If the relevancy is
beyond a threshold, we query the VLM module with the image
and the task. We ask it to reply with a yes or no and a
description of the scene. One such example is visualized in
Fig. 7.
5) Robot Platform: In all of our robot experiments, we
used a Clearpath Jackal robot platform. It is modified to
carry an onboard computer with AMD Ryzen 5 3600 CPU,
NVIDIA RTX 4000 Ada SFF GPU and 32 GB of RAM.
Additionally, the robot is equipped with a ZED 2i stereo
camera for obtaining RGB, depth and pose measurements
using the first generation position tracking module. It also
carries an Ouster OS1-64 LiDAR to generate ground truth
odometry for evaluation.
V. EXPERIMENTAL EVALUATION
We design our experimental protocol to first evaluate each
component of our method with prior work for fair comparison
and then demonstrate the full framework on a real robot. The
key attributes of our mapping method are: (i) 3D Semantic
Segmentation; (ii) Submapping; (iii) Memory Efficiency and
(iv) Retrieval of objects in the pre-built maps. Further, we
demonstrate closed-loop autonomy through real robot experi-
ments, with and without a prior map.
A. Dataset Experiments
1) Open-vocabulary 3D semantic segmentation: We evalu-
ate the 3D semantic segmentation performance of our map-
ping approach on the ScanNet [69] dataset, using scenes:
scene0011 00, scene0050 00, scene0231 00, scene0378 00,
scene0518 00.
Metrics. We compute the mean Intersection Over Union
(mIOU), mean accuracy (mAcc) and frequency-weighted
mIOU (F-mIOU) following the same procedure in [55].
Results. The results are presented in Table I. Concept-
Graphs [54] and HOV-SG [55] leverage image segmentation
priors for creating object clusters and maintain sparse language
feature vectors for each object. In contrast, ConceptFusion [70]
is more similar to our work and stores dense language features
in the map. We note that ConceptFusion also uses image
segmentation priors for extracting local features on an object
level. With our compressed language feature representation
and without using image segmentation priors, we are able to
achieve comparable performance with the baselines. We note
that we also use a smaller CLIP backbone compared to the
baseline methods.
Method
Feature
embeddings
mIOU
mAcc
F-mIOU
ConceptGraphs [54]
Sparse
0.16
0.20
0.28
HOV-SG [55]
0.22
0.30
0.43
ConceptFusion [70]
Dense
0.11
0.12
0.21
ATLAS (ours)
0.15
0.27
0.16
TABLE I: Open-vocabulary 3D semantic segmentation on
ScanNet. Baseline results from [55]. Baselines use Vit-H-14
while our method uses Vit-B-16 for the CLIP backbone.
2) Loop closure: To evaluate the effectiveness of pose
correction with submaps, we generate maps from the same
set of data. We generate three maps, one with just VIO pose
estimates, one with the addition of PGO, and finally one with
submap anchor pose updates on top of the PGO. We use
LiDAR odometry [71] to obtain groundtruth poses for the
test set. We compare the rendered images from each map and
evaluate the rendering quality. Finally, to present the upper
bound of the Gaussian splatting approach on this dataset, we
also generate a map using the test set poses.
Metrics. We evaluate the Peak Signal-to-Noise Ratio
(PSNR), Structural Similarity Index [72] (SSIM), Learned
Perceptual Image Patch Similarity [73] (LPIPS) on the color
images and Root Mean-Square-Error (RMSE) on the depth
images.
Results. The results are presented in Table II. We show that
we are able to leverage PGO with our submap-based mapping
framework to achieve better reconstruction quality.
3) Memory: To highlight the value of the dynamic loading
of submaps, we evaluate several 3DGS approaches on a large
indoor scene. We use scene 00824 from the Habitat Matterport
3D Semantics Dataset [74] (HM3D), following the method
in [55] to generate groundtruth observations and poses. If
they support submapping, methods are evaluated with different
submap sizes, at 2m and 5m distances between each submap.
We note that Gaussian-SLAM and LoopSplat both employ
submapping but only unload submaps and do not handle
reloading and updating of submaps.

<!-- page 10 -->
Method
PSNR ↑
SSIM ↑
LPIPS ↓
RMSE ↓
(dB)
(m)
VIO
12.62
0.53
0.48
0.89
VIO + PGO
14.09
0.54
0.47
0.85
VIO + PGO + SU
14.31
0.59
0.44
0.76
GT
16.13
0.65
0.37
0.39
TABLE II: Evaluation of loop closure-aware submap pose
graph update. VIO refers to visual-inertia odometry, PGO
refers to pose graph optimization, SU refers to submap anchor
pose updates and GT refers to the ground truth poses. This
experiment shows that our method is able to correct past poses
by shifting the submap anchors.
Results. The results are presented in Table. III. Gaussian-
SLAM [43] and LoopSplat [42] require large amounts of mem-
ory even at relatively small submap sizes. The results highlight
that many Gaussian splatting SLAM methods cannot support
memory-efficient mapping of large-scale environments, even
on the scale of indoor environments. Ensuring efficient scaling
of memory is crucial for operation on robots with limited
compute.
Method
Memory Allocated / Reserved (GB)
2m submap
5m submap
No submap
Gaussian-SLAM [43]
5.71 / 36.21
×
×
LoopSplat [42]
9.03 / 21.33
×
×
SplaTAM [30]
–
–
13.36 / 16.32
ATLAS (ours)
8.28 / 9.68
9.77 / 12.51
13.36 / 16.32
TABLE III: Comparison of memory performance on scene
0011 of the HM3D dataset. – indicates that the method does
not support submapping and is not evaluated with submaps.
× indicates that the method failed due to excessive memory
requirements or otherwise. Gaussian-SLAM and LoopSplat are
evaluated with submaps.
4) Image rendering from built map: Finally, a feature of
our method is the ability to query images of relevance and
provide textual feedback. We compare this functionality with
HOV-SG [55] since they generate a dense colored point cloud.
The resolution of the point cloud generated by HOV-SG is set
to the default of 0.05m. Since HOV-SG stores dense features
in each point, increasing the resolution of the point cloud is too
expensive in terms of both compute and memory. We generate
images from the point cloud by projecting the points using
the corresponding camera intrinsics of the scene. We render
images with both methods for all poses in the dataset. For
this comparison, we use scene 0011 from the ScanNet [69]
dataset. In addition to this, we show a qualitative result of
the retrieval capability of our method from a pre-built map,
constructed from data collected by tele-operating our robot.
Metrics. We evaluate the image reconstruction quality using
the following metrics – Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index [72] (SSIM), Learned Perceptual
Image Patch Similarity [73] (LPIPS) on the color images and
Root Mean-Square-Error (RMSE) on the depth images.
Results. We evaluate the rendered images from each
method using the original images of the ScanNet dataset as
groundtruth. The results are presented in Table IV. While we
Method
PSNR ↑
SSIM ↑
LPIPS ↓
RMSE ↓
(dB)
(m)
HOV-SG [55]
6.86
0.22
0.90
1.95
ATLAS (ours)
20.44
0.80
0.26
0.05
TABLE IV: Image rendering quality on ScanNet.
acknowledge that the map representation used in HOV-SG
does not prioritize rendering of the scene, we present these
results to highlight the value of storing the map as Gaussian
parameters. The qualitative evaluation of the difference be-
tween a rendered image and the ground truth image for the
task is shown in Fig. 4.
B. Robot experiments
To demonstrate the flexibility and efficiency of our method,
we conduct several real-world experiments across both indoor
and outdoor environments. We measure the ability of our
framework to load, localize and then navigate to the target.
Then, we demonstrate that our method can be applied to
completely unknown maps and follow the utility signal to
complete the task. The task completion oracle ψ uses a VLM
with a prompt requesting yes or no to terminate the task.
Example outputs from our experiments are shown in Fig. 3.
Metrics. We compute the path length PL of the trajectory
taken by the robot in each experiment. We compare this against
two other path lengths, SP and GT. After the experiment is
complete and the map is available, we compute the shortest
path SP on the full planning graph. This ablates away any
odometry error since our traveled path and the planning graph
are built on the same set of odometry measurements. To obtain
the length of the optimal path GT from the starting position
to the inferred goal in the outdoor experiments, we measure
the path length from GPS coordinates annotated by a human
on Google Earth. For indoor experiments, we measure the
shortest path from the start to goal with a rangefinder. We
measure the competitive ratio (listed ratio in Tab. V) of the
distance measured by the visual odometry on our robot and
these privileged distance measurements, given by SP
P L and GT
P L
respectively. To show the scale of our experiments, we measure
and report the approximate area of the operational area of the
environment for each experiment from Google Earth, and the
number of Gaussians in the map.
Experiment Areas. We conduct our experiments in an
urban office complex (shown in Fig. 5) with office space,
parking lots, and an outdoor park.
1) Navigation in pre-built map: To show how our sparse
hierarchical map can be used for navigation, we conduct an
experiment Outdoor1 where the robot uses a pre-built map to
identify and plan a path to a region of interest. An example
is shown in Fig. 8. In this experiment, we first tele-operate
the robot to a dock several hundred meters from the starting
position and save the map. On a separate run, the map is
loaded on the robot. The robot is given the task ‘navigate to
entrance to pier’. The task relevancy is computed across the
submaps and their regions, and the submap with the highest

<!-- page 11 -->
Is there a boardwalk near the road? Please answer with yes 
or no. Provide a description of the scene
Yes. In this image we can see a road, a boardwalk, a tree, 
a trash bin, a railing, a river, and the sky.
(a) The output of Vision Language Model for
the “find boardwalk near road” task
Is there a road blockage in the image? Please answer with 
yes or no. Provide a description of the scene
Yes. In the image, there are two orange trafc cones lying on 
the ground on a paved road. The cones are positioned in the
 middle of the road, indicating a possible road closure or a 
temporary obstruction...
(b) The output of Vision Language Model for
the “inspect road blockage” task.
 Is there a parking lot in the image? Please answer with yes 
or no. Provide a description of the scene
Yes. The image shows a parking lot at night. The parking lot 
is empty and there are no cars parked in it. The ground 
is covered with snow. There are streetlights and buildings 
in the background.
(c) The output of Vision Language Model for
the “find parking lot” task.
Fig. 3: Qualitative results showing the output of the VLM when the task terminates.
Experiment
Task
Prior
Distance (m)
Area
Num. of
Map
PL (m)
SP (m / ratio)
GT (m / ratio)
(m2)
Gaussians
Outdoor1
Navigate to entrance to pier
Yes
185.78
–
–
184.86
0.99
3346.5
2661562
Indoor1
Find cushions
No
72.53
34.87
0.48
29.72
0.41
973.28
633089
Indoor2
Find plants, Exit building
No
44.55
31.60
0.71
30.16
0.68
973.28
446361
Outdoor2
Inspect road blockage
No
94.25
45.16
0.48
43.52
0.46
1063.57
1087489
Outdoor3
Find parking lot
No
69.19
46.439
0.67
43.22
0.62
1280.10
1473420
Outdoor4
Find river near road
No
742.42
473.64
0.64
472.55
0.64
17791.14
7769214
Outdoor5
Find boardwalk near road
No
1257.53
688.90
0.55
671.76
0.53
21870.34
11279776
TABLE V: Overview of robot experiments. Prior indicates that a prior PL refers to path length, SP refers to the shortest path computed
on the full planning graph after the experiment is complete, and GT refers to the ground truth.
“Find white minivan”
[A] Task
[B] Selected submap 
[C] Rendered image 
[D] Ground Truth Image
Fig. 4: [A] shows the task provided to our method. [B] shows
the selected submap and region in the bottom-left with highest
relevance. [C] shows the rendered image from the vantage point
with the highest relevance to the task. [D] shows the ground
truth image.
utility is identified. The robot plans a path to the submap of
interest and navigates to the goal.
Fig. 5: The outdoor experiment areas for our experiments.
Our park experiments are in the highlighted yellow areas. The
parking lots are in red and blue.
Results. Our results for this experiment are reported in
Table V. As expected, our method performs close to the
ground truth since it also has access to the true map of the
environment. The qualitative results of the VLM query for the
termination of the task is shown in Fig. 6.

<!-- page 12 -->
Is there an entrance to pier? Please answer with yes or no. 
Provide a description of the scene
Yes. The image shows a pathway leading to a gate. The 
pathway is made of asphalt and is surrounded by trees and 
bushes. There are piles of leaves on the sides of the pathway... 
Relevancy
Fig. 6: The user-specified task and the description received on
termination of the task. The VLM is queried when the relevancy
of the language features obtained from the image exceeds a
threshold.
2) Navigation with no prior map: We conduct six real-
world experiments – two indoor and four outdoor. In these
experiments, the robot starts off with no prior map or infor-
mation of the environment. Given a user-specified task, the
robot proceeds to incrementally build a metric-semantic map
of the environment and uses relevant information in the map
to complete the task.
Indoor. For Indoor1, we perform a simple object search
of ‘find plants’ and then demonstrate re-tasking the robot to
‘exit the building’. In the second indoor experiment Indoor2,
we show that our approach is able to leverage the semantic
relationships between objects in the environment to complete
tasks. With the task ‘find cushions’, the robot is able to identify
chairs and couches as areas of high relevance to the task
and proceeds to inspect them, eventually successfully locating
the cushions. Visualizations of the experiment are provided in
Fig. 7.
Outdoor. For our outdoor experiments, we consider the
following:
1) In Outdoor2, we use an under-specified task ‘inspect
road blockage’ and the robot leverages semantics of the
pavement to search for and identify obstructions.
2) In Outdoor3, the robot is tasked with ‘find parking lot’
while starting close to the entrance of a building.
3) In Outdoor4 and Outdoor5, we conduct large-scale ex-
periments in the park. A visualization of task relevancy
and the ground truth trajectory is shown in Fig. 8.
Results. We present an overview of the robot experiments
in Table V. The competitive ratio of our method is ∼0.59 on
Are there cushions in the image? 
Please answer with yes or no. 
Provide a description of the scene
Yes. The image shows a modern ofce or commercial hallway 
with a seating area that features an L-shaped sectional sofa in 
orange and gray with decorative cushions
[D] Cushions found!
[A] Start
[B] Couch found but no cushion
[C] Couch found but no cushion
Fig. 7: Robot executing the task ‘find cushions’ starting at
[A]. The robot incrementally constructs a map with language-
embedded Gaussian splatting and identifies and navigates to
regions in the map with high relevance. The VLM is queried at
vantage points [B], [C], [D]. The images acquired at [B] and [C]
have the second-order association of couches but no cushions.
The task only terminates at [D] when the cushion is found.
SP and ∼0.56 on GT. This shows that our tasks require some
exploration but in general, our method performs at least half as
well as a privileged baseline. In our experiments, our method
is also able to store over an order of a million Gaussians on-
board the robot.
VI. LIMITATIONS
One of the primary limitations of this method is its heavy
dependence on the quality of the external odometry solution.
While it is possible to incorporate loop closures into the
process, the system is fundamentally reliant on the exter-
nal module’s ability to accurately detect and handle these
loop closures. Furthermore, the success of the method is
dependent on the quality of the visual data—both the images
and depth information—as well as the viewpoint from which
they are captured. Poor-quality images, inaccurate depth mea-

<!-- page 13 -->
Start
Goal
[C] Ground truth trajectory
[B] Relevancy
[A] Submap graph
Fig. 8: [A] The map built from the task ”Find boardwalk near
road”. The colored Gaussian points and the submap nodes (red
and green circles) are visualized. The prior hierarchical graph
can then be used to retrieve and navigate in the map. The
shortest path (SP) to the boardwalk is shown with the green
nodes. [B] The task relevancy of the Gaussians is colored blue
(low) to red (high). [C] The ground truth trajectory (GT) is
overlaid in yellow on an image from Google Earth to highlight
the scale of the experiments.
surements, or suboptimal viewpoints can severely impact the
system’s ability to generate accurate task-driven semantics,
as these factors form the core of the data used to interpret
and solve tasks. Empirically, the compressed CLIP features
retain sufficient information for the tasks we consider but may
not work for more abstract tasks. We also do not consider
the orientation of the robot at the discrete planning stage,
which might result in the robot failing to complete the task by
mapping only a portion of the object of interest. Lastly, the
method faces challenges when dealing with very abstract tasks.
While it performs well for tasks with concrete and well-defined
semantics, it may struggle with tasks that require a deeper
understanding or decomposition of more abstract concepts. In
such cases, additional tools, such as large language models
(LLMs), may be necessary to decompose the tasks into more
manageable components, further complicating the overall pro-
cess and potentially adding another layer of complexity.
VII. CONCLUSION
In this paper, we develop a framework and methodology that
allows robots to autonomously explore and navigate unstruc-
tured environments to accomplish tasks that are specified using
natural language. Central to our approach is a hierarchical
representation built on language-embedded Gaussian splatting
that can be run on-board a robot in real time. We use indoor
and outdoor experiments traversing hundreds of meters to
show the robustness of our approach and our ability to build
large-scale maps and identify semantically relevant objects
in diverse environments. Our method does not rely on the
presence of structure or rich semantics in the scene and can
be applied in general settings. In future work, we plan to
deploy a distilled LLM on the robot for task decomposition
to enable more abstract reasoning of tasks. While this paper
empirically evaluates this method on largely 2D environments,
in future work we will consider multi-floor buildings and large
environments with elevation changes.
REFERENCES
[1] David Lattanzi and Gregory Miller. Review of robotic in-
frastructure inspection systems. Journal of Infrastructure
Systems, 23(3):04017004, 2017.
[2] Timothy H Chung, Viktor Orekhov, and Angela Maio.
Into the robotic depths: analysis and insights from the
darpa subterranean challenge. Annual Review of Control,
Robotics, and Autonomous Systems, 6(1):477–502, 2023.
[3] Geert-Jan M Kruijff, Fiora Pirri, Mario Gianni, Panagi-
otis Papadakis, Matia Pizzoli, Arnab Sinha, Viatcheslav
Tretyakov, Thorsten Linder, Emanuele Pianese, Salvatore
Corrao, et al. Rescue robots at earthquake-hit mirandola,
Italy: A field report. In 2012 IEEE international sym-
posium on safety, security, and rescue robotics (SSRR),
pages 1–8. IEEE, 2012.
[4] Spyros Fountas, Nikos Mylonas, Ioannis Malounas,
Efthymios Rodias, Christoph Hellmann Santos, and Erik
Pekkeriet.
Agricultural robotics for field operations.
Sensors, 20(9):2672, 2020.
[5] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
42(4):1–14, 2023.
[6] Monika Wysocza´nska, Oriane Sim´eoni, Micha¨el Rama-
monjisoa, Andrei Bursuc, Tomasz Trzci´nski, and Patrick
P´erez.
Clip-dinoiser: Teaching clip a few dino tricks
for open-vocabulary semantic segmentation.
In Euro-
pean Conference on Computer Vision, pages 320–337.
Springer, 2025.
[7] Benjamin Charrow, Sikang Liu, Vijay Kumar, and
Nathan Michael.
Information-theoretic mapping using
cauchy-schwarz quadratic mutual information. In 2015
IEEE International Conference on Robotics and Automa-
tion (ICRA), pages 4791–4798. IEEE, 2015.
[8] Kelsey Saulnier, Nikolay Atanasov, George J Pappas, and
Vijay Kumar.
Information theoretic active exploration
in signed distance fields.
In 2020 IEEE International
Conference on Robotics and Automation (ICRA), pages
4080–4085. IEEE, 2020.
[9] Arash Asgharivaskasi and Nikolay Atanasov. Semantic
octree mapping and shannon mutual information com-
putation for robot exploration.
IEEE Transactions on
Robotics, 39(3):1910–1928, 2023.
[10] Shi Bai, Jinkun Wang, Fanfei Chen, and Brendan En-
glot. Information-theoretic exploration with bayesian op-
timization. In 2016 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pages 1816–
1822, 2016.

<!-- page 14 -->
[11] Lukas Schmid, Michael Pantic, Raghav Khanna, Lionel
Ott, Roland Siegwart, and Juan Nieto.
An efficient
sampling-based method for online informative path plan-
ning in unknown environments.
IEEE Robotics and
Automation Letters, 5(2):1500–1507, 2020.
[12] Andreas Bircher, Mina Kamel, Kostas Alexis, Helen
Oleynikova, and Roland Siegwart.
Receding horizon”
next-best-view” planner for 3d exploration. In 2016 IEEE
international conference on robotics and automation
(ICRA), pages 1462–1468. IEEE, 2016.
[13] Christos Papachristos, Shehryar Khattak, and Kostas
Alexis. Uncertainty-aware receding horizon exploration
and mapping using aerial robots. In 2017 IEEE Interna-
tional Conference on Robotics and Automation (ICRA),
pages 4568–4575, 2017.
[14] Mihir Dharmadhikari, Tung Dang, Lukas Solanka, Jo-
hannes Loje, Huan Nguyen, Nikhil Khedekar, and Kostas
Alexis. Motion primitives-based path planning for fast
and agile exploration using aerial robots. In 2020 IEEE
International Conference on Robotics and Automation
(ICRA), pages 179–185, 2020.
[15] Lukas Schmid, Victor Reijgwart, Lionel Ott, Juan Nieto,
Roland Siegwart, and Cesar Cadena. A unified approach
for autonomous volumetric exploration of large scale en-
vironments under severe odometry drift. IEEE Robotics
and Automation Letters, 6(3):4504–4511, 2021.
[16] Yuezhan Tao, Yuwei Wu, Beiming Li, Fernando Cladera,
Alex Zhou, Dinesh Thakur, and Vijay Kumar. SEER:
Safe efficient exploration for aerial robots using learning
to predict information gain. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages
1235–1241. IEEE, 2023.
[17] Yuezhan Tao, Xu Liu, Igor Spasojevic, Saurav Agarwal,
and Vijay Kumar. 3d active metric-semantic slam. IEEE
Robotics and Automation Letters, pages 1–8, 2024.
[18] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang.
Activenerf: Learning where to see with uncertainty es-
timation. In European Conference on Computer Vision,
pages 230–246. Springer, 2022.
[19] Soomin Lee, Le Chen, Jiahao Wang, Alexander Liniger,
Suryansh Kumar, and Fisher Yu.
Uncertainty guided
policy for active robotic 3d reconstruction using neural
radiance fields. IEEE Robotics and Automation Letters,
7(4):12070–12077, 2022.
[20] Huangying Zhan, Jiyang Zheng, Yi Xu, Ian Reid, and
Hamid Rezatofighi. Activermap: Radiance field for active
mapping and planning. arXiv preprint arXiv:2211.12656,
2022.
[21] Yunlong Ran, Jing Zeng, Shibo He, Jiming Chen,
Lincheng Li, Yingfeng Chen, Gimhee Lee, and Qi Ye.
Neurar: Neural uncertainty for autonomous 3d recon-
struction with implicit neural representations.
IEEE
Robotics and Automation Letters, 8(2):1125–1132, 2023.
[22] Siming He, Christopher D Hsu, Dexter Ong, Yifei Simon
Shao, and Pratik Chaudhari.
Active perception using
neural radiance fields. arXiv preprint arXiv:2310.09892,
2023.
[23] Siming He, Yuezhan Tao, Igor Spasojevic, Vijay Kumar,
and Pratik Chaudhari.
An active perception game for
robust autonomous exploration, 2024.
[24] Zike Yan, Haoxiang Yang, and Hongbin Zha.
Active
neural mapping. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 10981–
10992, 2023.
[25] Ziyue Feng, Huangying Zhan, Zheng Chen, Qingan Yan,
Xiangyu Xu, Changjiang Cai, Bing Li, Qilun Zhu, and
Yi Xu. Naruto: Neural active reconstruction from uncer-
tain target observations. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21572–21583, 2024.
[26] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fisherrf:
Active view selection and uncertainty quantification for
radiance fields using fisher information, 2023.
[27] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei
Gao.
Gs-planner: A gaussian-splatting-based planning
framework for active high-fidelity reconstruction, 2024.
[28] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang,
Jieru Zhao, Fei Gao, Zhongxue Gan, and Wenchao Ding.
Hgs-planner: Hierarchical planning framework for active
scene reconstruction using 3d gaussian splatting. arXiv
preprint arXiv:2409.17624, 2024.
[29] Yuezhan Tao, Dexter Ong, Varun Murali, Igor Spaso-
jevic, Pratik Chaudhari, and Vijay Kumar.
Rt-guide:
Real-time gaussian splatting for information-driven ex-
ploration. arXiv preprint arXiv:2409.18122, 2024.
[30] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallab-
hula, Gengshan Yang, Sebastian Scherer, Deva Ramanan,
and Jonathon Luiten. Splatam: Splat, track & map 3d
gaussians for dense rgb-d slam. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024.
[31] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 18039–18048, 2024.
[32] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin
Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-
time 3d reconstruction at scale using gaussian splatting.
In ACM SIGGRAPH 2024 Conference Papers, pages 1–
11, 2024.
[33] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li,
Liangjing Yang, Hujun Bao, Guofeng Zhang, and
Zhaopeng Cui.
Cg-slam: Efficient dense rgb-d slam
in a consistent uncertainty-aware 3d gaussian field. In
European Conference on Computer Vision, pages 93–
112. Springer, 2025.
[34] Timothy Chen, Ola Shorinwa, Weijia Zeng, Joseph
Bruno, Philip Dames, and Mac Schwager. Splat-nav: Safe
real-time robot navigation in gaussian splatting maps.
arXiv preprint arXiv:2403.02751, 2024.
[35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

<!-- page 15 -->
Amanda Askell, Pamela Mishkin, Jack Clark, et al.
Learning transferable visual models from natural lan-
guage supervision. In International conference on ma-
chine learning, pages 8748–8763. PMLR, 2021.
[36] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan. Language embedded 3d gaussians for open-
vocabulary scene understanding. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5333–5343, 2024.
[37] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang,
and Hanspeter Pfister. Langsplat: 3d language gaussian
splatting. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
20051–20060, 2024.
[38] Guibiao Liao, Jiankun Li, Zhenyu Bao, Xiaoqing Ye,
Jingdong Wang, Qing Li, and Kanglin Liu.
Clip-gs:
Clip-informed gaussian splatting for real-time and view-
consistent 3d semantic understanding.
arXiv preprint
arXiv:2404.14249, 2024.
[39] Justin Yu, Kush Hari, Kishore Srinivas, Karim El-Refai,
Adam Rashid, Chung Min Kim, Justin Kerr, Richard
Cheng, Muhammad Zubair Irshad, Ashwin Balakrishna,
et al. Language-embedded gaussian splats (legs): Incre-
mentally building room-scale representations with a mo-
bile robot. In 2024 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pages 13326–
13332. IEEE, 2024.
[40] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan
Di, and Mingyang Li.
Fmgs: Foundation model em-
bedded 3d gaussian splatting for holistic 3d scene un-
derstanding. International Journal of Computer Vision,
pages 1–17, 2024.
[41] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik.
Lerf: Language em-
bedded radiance fields. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages
19729–19739, 2023.
[42] Liyuan Zhu, Yue Li, Erik Sandstr¨om, Shengyu Huang,
Konrad Schindler, and Iro Armeni.
Loopsplat: Loop
closure by registering 3d gaussian splats. arXiv preprint
arXiv:2408.10154, 2024.
[43] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R
Oswald. Gaussian-slam: Photo-realistic dense slam with
gaussian splatting.
arXiv preprint arXiv:2312.10070,
2023.
[44] Tung Dang, Christos Papachristos, and Kostas Alexis.
Autonomous exploration and simultaneous object search
using aerial robots. In 2018 IEEE Aerospace Conference,
pages 1–7. IEEE, 2018.
[45] Xu Liu, Jiuzhou Lei, Ankit Prabhu, Yuezhan Tao, Igor
Spasojevic, Pratik Chaudhari, Nikolay Atanasov, and Vi-
jay Kumar. Slideslam: Sparse, lightweight, decentralized
metric-semantic slam for multi-robot navigation. arXiv
preprint arXiv:2406.17249, 2024.
[46] Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Za-
mir, Martin Fischer, Jitendra Malik, and Silvio Savarese.
3d scene graph: A structure for unified semantics, 3d
space, and camera. In IEEE/CVF Int. Conf. on Computer
Vision, pages 5664–5673, 2019.
[47] Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir
Navab, and Federico Tombari. Scenegraphfusion: Incre-
mental 3d scene graph prediction from rgb-d sequences.
In Proc. of IEEE Conf. on Computer Vision and Pattern
Recognition, pages 7515–7525, 2021.
[48] Samuel Looper, Javier Rodriguez-Puigvert, Roland Sieg-
wart, Cesar Cadena, and Lukas Schmid. 3d vsg: Long-
term semantic scene change prediction through 3d vari-
able scene graphs. In IEEE Int. Conf. on Robotics &
Automation, pages 8179–8186. IEEE, 2023.
[49] Nathan Hughes, Yun Chang, Siyi Hu, Rajat Talak, Ru-
maia Abdulhai, Jared Strader, and Luca Carlone. Foun-
dations of spatial perception for robotics: Hierarchical
representations and real-time systems. The International
Journal of Robotics Research, 2024.
[50] Hriday Bavle, Jose Luis Sanchez-Lopez, Muhammad
Shaheer, Javier Civera, and Holger Voos.
S-graphs+:
Real-time localization and mapping leveraging hierar-
chical representations. IEEE Robotics and Automation
Letters, 8(8):4927–4934, 2023.
[51] Jared Strader, Nathan Hughes, William Chen, Alberto
Speranzon, and Luca Carlone.
Indoor and outdoor
3d scene graph generation via language-enabled spatial
ontologies.
IEEE Robotics and Automation Letters,
9(6):4886–4893, 2024.
[52] Dominic Maggio, Yun Chang, Nathan Hughes, Matthew
Trang, Dan Griffith, Carlyn Dougherty, Eric Cristofalo,
Lukas Schmid, and Luca Carlone.
Clio: Real-time
task-driven open-set 3d scene graphs.
arXiv preprint
arXiv:2404.13696, 2024.
[53] Venkata Naren Devarakonda, Raktim Gautam Goswami,
Ali Umut Kaypak, Naman Patel, Rooholla Khorram-
bakht, Prashanth Krishnamurthy, and Farshad Khorrami.
Orionnav: Online planning for robot autonomy with
context-aware llm and open-vocabulary semantic scene
graphs. arXiv preprint arXiv:2410.06239, 2024.
[54] Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna
Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal,
Corban Rivera, William Paul, Kirsty Ellis, Rama Chel-
lappa, Chuang Gan, Celso Miguel de Melo, Joshua B.
Tenenbaum, Antonio Torralba, Florian Shkurti, and Liam
Paull. Conceptgraphs: Open-vocabulary 3d scene graphs
for perception and planning. International Conference
on Robotics and Automation, 2024.
[55] Abdelrhman Werby, Chenguang Huang, Martin B¨uchner,
Abhinav Valada, and Wolfram Burgard.
Hierarchical
open-vocabulary 3d scene graphs for language-grounded
robot navigation. In First Workshop on Vision-Language
Models for Navigation and Manipulation at ICRA 2024,
2024.
[56] Sotiris Papatheodorou, Nils Funk, Dimos Tzoumanikas,
Christopher Choi, Binbin Xu, and Stefan Leutenegger.
Finding things in the unknown: Semantic object-centric

<!-- page 16 -->
exploration with an mav.
In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages
3339–3345. IEEE, 2023.
[57] Georgios Georgakis, Bernadette Bucher, Karl Schmeck-
peper, Siddharth Singh, and Kostas Daniilidis. Learning
to map for active semantic goal navigation.
arXiv
preprint arXiv:2106.15648, 2021.
[58] Zhirui Dai, Arash Asgharivaskasi, Thai Duong, Shusen
Lin, Maria-Elizabeth Tzes, George Pappas, and Nikolay
Atanasov.
Optimal scene graph planning with large
language model guidance, 2024.
[59] Saeid Amiri, Kishan Chandan, and Shiqi Zhang. Reason-
ing with scene graphs for robot planning under partial
observability.
IEEE Robotics and Automation Letters,
7(2):5560–5567, 2022.
[60] Zachary Ravichandran, Lisa Peng, Nathan Hughes,
J Daniel Griffith, and Luca Carlone. Hierarchical rep-
resentations and explicit memory: Learning effective
navigation policies on 3d scene graphs using graph neural
networks. In 2022 International Conference on Robotics
and Automation (ICRA), pages 9272–9279. IEEE, 2022.
[61] Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang
Wang, and Bernadette Bucher.
Vlfm: Vision-language
frontier maps for zero-shot semantic navigation.
In
2024 IEEE International Conference on Robotics and
Automation (ICRA), pages 42–48. IEEE, 2024.
[62] Zachary Ravichandran, Varun Murali, Mariliza Tzes,
George J Pappas, and Vijay Kumar.
Spine: Online
semantic planning for missions with incomplete natural
language specifications in unstructured environments.
arXiv preprint arXiv:2410.03035, 2024.
[63] Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang,
Dongdong Chen, Hao Yang, Ming Zeng, Weiming
Zhang, Lu Yuan, Dong Chen, et al. Maskclip: Masked
self-distillation advances contrastive language-image pre-
training. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 10995–
11005, 2023.
[64] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni,
Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fer-
nandez, Daniel Haziza, Francisco Massa, Alaaeldin El-
Nouby, et al. Dinov2: Learning robust visual features
without supervision. arXiv preprint arXiv:2304.07193,
2023.
[65] David A Ross, Jongwoo Lim, Ruei-Sung Lin, and Ming-
Hsuan Yang. Incremental learning for robust visual track-
ing. International journal of computer vision, 77:125–
141, 2008.
[66] Tsung-Yi Lin, Michael Maire, Serge Belongie, James
Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and
C Lawrence Zitnick. Microsoft coco: Common objects in
context. In Computer Vision–ECCV 2014: 13th European
Conference, Zurich, Switzerland, September 6-12, 2014,
Proceedings, Part V 13, pages 740–755. Springer, 2014.
[67] Sikang Liu, Nikolay Atanasov, Kartik Mohta, and Vijay
Kumar.
Search-based motion planning for quadrotors
using linear quadratic minimum time control. In 2017
IEEE/RSJ international conference on intelligent robots
and systems (IROS), pages 2872–2879. IEEE, 2017.
[68] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng
Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei
Li, Ziwei Liu, et al. Llava-onevision: Easy visual task
transfer. arXiv preprint arXiv:2408.03326, 2024.
[69] Angela Dai, Angel X. Chang, Manolis Savva, Ma-
ciej Halber, Thomas Funkhouser, and Matthias Nießner.
Scannet: Richly-annotated 3d reconstructions of indoor
scenes. In Proc. Computer Vision and Pattern Recogni-
tion (CVPR), IEEE, 2017.
[70] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala,
Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf,
Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil
Keetha, et al. Conceptfusion: Open-set multimodal 3d
mapping. arXiv preprint arXiv:2302.07241, 2023.
[71] Chunge Bai, Tao Xiao, Yajie Chen, Haoqian Wang, Fang
Zhang, and Xiang Gao. Faster-lio: Lightweight tightly
coupled lidar-inertial odometry using parallel sparse in-
cremental voxels. IEEE Robotics and Automation Letters,
7(2):4861–4868, 2022.
[72] Z. Wang, E.P. Simoncelli, and A.C. Bovik. Multiscale
structural similarity for image quality assessment. In The
Thrity-Seventh Asilomar Conference on Signals, Systems
& Computers, 2003, volume 2, pages 1398–1402 Vol.2,
2003.
[73] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness
of deep features as a perceptual metric. In CVPR, 2018.
[74] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ra-
makrishnan, Theo Gervet, John Turner, Aaron Gokaslan,
Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis
Savva, et al. Habitat-matterport 3d semantics dataset. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 4927–4936, 2023.
