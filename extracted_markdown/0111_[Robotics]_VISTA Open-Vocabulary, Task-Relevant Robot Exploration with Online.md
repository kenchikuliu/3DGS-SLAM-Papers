<!-- page 1 -->
arXiv:2507.01125v1  [cs.RO]  1 Jul 2025
VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with
Online Semantic Gaussian Splatting
Keiko Nagami1, Timothy Chen1, Javier Yu1, Ola Shorinwa1, Maximilian Adang1,
Carlyn Dougherty2, Eric Cristofalo2, and Mac Schwager1
Abstract— We present VISTA (Viewpoint-based Image se-
lection with Semantic Task Awareness), an active exploration
method for robots to plan informative trajectories that improve
3D map quality in areas most relevant for task completion.
Given an open-vocabulary search instruction (e.g., “find a
person”), VISTA enables a robot to explore its environment to
search for the object of interest, while simultaneously building
a real-time semantic 3D Gaussian Splatting reconstruction of
the scene. The robot navigates its environment by planning
receding-horizon trajectories that prioritize semantic similarity
to the query and exploration of unseen regions of the envi-
ronment. To evaluate trajectories, VISTA introduces a novel,
efficient viewpoint-semantic coverage metric that quantifies
both the geometric view diversity and task relevance in the
3D scene. On static datasets, our coverage metric outper-
forms state-of-the-art baselines, FisherRF and Bayes’ Rays, in
computation speed and reconstruction quality. In quadrotor
hardware experiments, VISTA achieves 6x higher success rates
in challenging maps, compared to baseline methods, while
matching baseline performance in less challenging maps. Lastly,
we show that VISTA is platform-agnostic by deploying it on
a quadrotor drone and a Spot quadruped robot. Open-source
code will be released upon acceptance of the paper.
I. INTRODUCTION
Research advances in vision and language foundation
models, e.g., [1], [2], have enabled language-guided object
localization in pre-mapped real-world environments [3]. For
example, a user can task a robot with the word apple and
it will find an apple in a pre-mapped concept graph [4].
However, to find query objects efficiently in unstructured,
unmapped environments, robots must be capable of exploring
their environments intelligently, with a bias toward finding
the object of interest. Prior work in robot exploration broadly
uses traditional 3D scene representations, such as occupancy
grids and voxel grids. We build upon these traditional rep-
resentations by introducing a Gaussian Splat embedded with
1Department of Aeronautics and Astronautics, Stanford University,
Stanford, CA 94305, USA knagami, chengine, javieryu,
shorinwa, madang, schwager@stanford.edu.
2MIT
Lincoln
Laboratory,
Lexington,
MA
02421,
USA
carlyn.dougherty, eric.cristofalo@ll.mit.edu.
This
work
was
supported
in
part
by
MIT
Lincoln
Laboratory
ACC grant 7000603941, NSF project FRR 2342246, and ONR project
N00014-23-1-2354. The NASA University Leadership initiative (grant
#80NSSC20M0163) provided funds to assist the authors with their research,
but this article solely reflects the opinions and conclusions of its authors and
not any NASA entity.
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited. This material is based
upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-
25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Department of the Air Force. ©2025 Massachusetts Institute of Technology.
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014).
Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or
DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
Fig. 1.
System overview of VISTA. Real-time sensor data is gathered from
a robot hardware platform to train a semantic 3DGS map. The semantic
and RGB information from the 3DGS map are transferred to a 3D Voxel
Grid, and training poses are used to store geometric information about
directions from which each voxel has already been viewed. In the planner,
the 3D voxel grid is flattened into a top-down 2D voxel grid, where frontier
cells and semantic information are used to fit a Gaussian Mixture Model
that is sampled to generate candidate trajectories. The trajectory with the
highest semantic + geometric information gain is then executed in a receding
horizon loop.
semantic information that can advance downstream tasks to
be performed by the robot.
We present VISTA, an algorithm for Viewpoint-based
Image Selection with Semantic Task Awareness. To en-
able task-relevant exploration, VISTA introduces two key
innovations: (i) a semantics-aware mapping and information-
gain pipeline that leverages open-vocabulary semantics for
task-relevant exploration, and (ii) a scalable information-gain
metric based on view angle diversity that can be computed
efficiently in real-time. Our key insight is that, for vision-
based mapping, the variety and multiplicity of viewing
directions from which an environment point has appeared in
the image history is a strong proxy for the geometric quality
of the reconstruction at that point.
First, VISTA builds a high-fidelity photorealistic map of

<!-- page 2 -->
the robot’s environment online using a Gaussian Splatting
(3DGS) representation [5].1 To enable open-vocabulary, task-
relevant robot exploration, VISTA distills semantic features
from vision-language models, e.g., CLIP [1], into the 3DGS
map incrementally as new observations are obtained by the
robot. The semantic features encode the relevancy between
each point in the environment and the specified exploration
task, giving a task relevancy 3D heatmap.
Using the 3DGS map, VISTA simultaneously constructs
a voxel grid, capturing the viewed regions of the scene and
the semantic relevance of these regions. With this grid, we
measure view diversity through a conceptually simple, com-
putationally efficient coverage metric with advantages like
recursive updating and informative trajectory optimization.
In each voxel, the geometric uncertainty is the minimum
angular separation between the test viewpoint and all view
angles from which that voxel has appeared in the image
history accounting for occlusions.
Finally, VISTA samples trajectories, and selects those
with viewpoints that maximize a weighted combination of
geometric uncertainty and semantic relevance, ultimately
guiding the robot toward a specified query object. We illus-
trate these components in Figure 1. Through an experimental
campaign with a total of 36 hardware executions, we show
that VISTA outperforms state-of-the-art baselines, achieving
6x better success rates in environments where the object
is not visible from the initial robot pose, and matching
performance when it is visible.
Our contributions are as follows. We introduce:
1) an efficient information metric that combines view
angle diversity and semantic task relevance stored on
a voxel grid that can be recursively updated,
2) a real-time informative trajectory planning algorithm
to drive robot exploration using this metric,
3) and a full stack ROS implementation of VISTA and
online 3DGS training demonstrated on robot hardware.
II. RELATED WORK
Robot Exploration. The objective in robot exploration is
to traverse through an environment efficiently to build a
map, without colliding into obstacles. The maps used in
these problems have traditionally used occupancy grids, point
clouds, signed distance fields (SDFs), and voxel maps [7]–
[9]. Within these map environments, exploration is typically
performed through frontier-based, and viewpoint sampling
methods. Frontiers are defined within the map as boundaries
between free and unknown space. The robot then plans to
the frontiers to collect observations of the unknown regions
[10]–[12]. In viewpoint sampling methods, each view is
scored by an information metric where the highest scoring
are prioritized destinations [13], [14]. These two methods are
often used together, in order to improve sample efficiency of
the candidate viewpoints [15]–[17].
1VISTA can equivalently use a neural radiance field (NeRF) [6] or any
other high fidelity scene representation that can embed semantic codes.
Building upon these methods, VISTA incorporates both
frontier-based and viewpoint sampling methods for its ex-
ploration pipeline, using a high-fidelity 3DGS map repre-
sentation, while simultaneously incorporating semantic in-
formation in both its map and trajectory scoring to search
for objects in the scene.
Active Planning and View Selection in Radiance Fields.
Recently, radiance field methods, such as NeRFs [6], [18],
[19] and 3DGS [5], [20] have been introduced to robot
mapping [21]–[23], to generate high fidelity representations
of environments. Of these methods, 3DGS provides much
faster training and rendering rates, enabled by an explicit,
interpretable representation of the scene, and a tile-based
rasterization procedure, which is more efficient than the
volumetric ray-marching procedure used by NeRFs.
Active planning algorithms use view selection methods
in order to gather the most information. Several methods
for radiance fields [24]–[31] perform active view planning
to improve localization accuracy while reducing the risk of
collision in navigation. A notable body of work [29], [30],
[32] leverage the Fisher information of radiance fields to
estimate the uncertainty of 3DGS models, optimizing over
a trajectory of viewpoints to maximize the information gain
while minimizing localization errors [30], [32]. In [33], the
authors use a similar mechanism to quantify epistemic uncer-
tainty in NeRFs, however these methods are not used directly
for active mapping. In RT-GuIDE [34], the authors compute
the magnitude of the change in the parameters of a 3DGS
map over consecutive updates to estimate the uncertainty of
the map in active exploration, and showcase demonstrations
of their approach on a robot hardware platform.
Many of these methods rely on the radiance field map
to obtain information metrics, and as such are not easily
transferrable to other map representations. Additionally, none
of these methods account for semantic information in the
map, nor do they reason about task-relevance in planning.
Semantics-Guided Exploration. The methods used for ac-
tive planning and view selection focus largely on improving
the quality of the map, and typically do not reason about
any specified task. In this work, we not only hope to obtain
a well constructed map, but additionally require our robot to
find specific objects queried by a user. 2D vision language
foundation models enable distillation of information into
3D scene geometry. Specifically, semantic radiance fields
[35]–[40], capture 2D semantic information from vision
foundation models, e.g., CLIP [1], DINO [2], SAM [41], and
LLaVA [42] into maps expressing semantics in 3D space.
Several works address the use of semantic maps for object
search, [43]–[47]. In [43], the authors compare a number of
different exploration methods, combined with object detec-
tion modules that are used to define a switch from explo-
ration to exploitation. However, none of these methods are
performed on robot hardware platforms. In [44], the authors
also use a switching mode from exploration to exploitation,
and present a novel geometry-based metric to quantify the
confidence in the map to represent their information gain to
select frontiers for exploration. In [45], the authors present

<!-- page 3 -->
a method for active search with task specification on large-
scale maps by using smaller local 3DGS submaps as the
robot moves through a global scene. The paper demonstrates
the method used on real robot platforms in multiple different
maps in both indoor and outdoor environments. While the
results show great versatility in multiple environments, the
method does not account for geometric uncertainty in the
map, so it is unclear if this method would be able to perform
well if the query object were not in the initial map. In
contrast to these methods, VISTA integrates both semantics
and geometric guidance into the exploration task robustly
without relying on switching behavior modes, addressing
these limitations.
III. PROBLEM FORMULATION
We consider a robotic exploration problem in which a
robot has an onboard, forward-facing RGB-D camera with
reliable state estimation. The robot is placed into a previously
unseen environment and is given an open-vocabulary query
to locate and retrieve a certain object in the scene by a
user. Once the robot receives the input query, it must then
construct a map of its environment as it moves, while simul-
taneously searching for the query object. In this informative
planning task, the robot must balance the requirement of
finding the object while generating a map during exploration
to confidently determine whether or not the identified object
satisfies the user’s input query.
To train the 3DGS and render images in the voxel grid, the
camera pose of the robot’s onboard camera is represented as:
x =
x
y
z
ϕ
θ
ψT , representing the position and
Euler angles of the camera in the global frame. As the robot
moves, it collects full pose odometry information along with
RGB and depth images in order to train a 3DGS map of the
environment. 3DGS [5] represents non-empty space using
Gaussian primitives, each parametrized by a mean (center)
µ ∈R3, a covariance Σ = RSSTRT ∈Sn
++ (parameterized
by a rotation matrix R and a diagonal scaling matrix S), an
opacity α ∈R+, and spherical harmonics (SH) parameters
to capture view-dependent visual effects like reflections. This
explicit representation not only enables Gaussian Splatting to
avoid unnecessary computation involving empty space, but
it also enables the utilization of fast tile-based rasterization.
The rasterization procedure uses α-blending, computing the
color of each pixel.
As the map updates, we assume that the motion of the
robot is restricted in the z, ϕ, and θ axes. The robot’s motion
is then modeled as a planar single integrator with a heading
angle in the yaw direction. The state and control vectors for
planning s ∈R3 and u ∈R3 are as follows:
s =
x
y
ψT ,
u =

˙x
˙y
˙ψ
T ,
(1)
where the velocity control vector is subject to control limits.
IV. VISTA
We propose a method for efficient exploration of an
environment by a robot, guided by a natural language se-
mantic query provided by a user. Our robot constructs a
3DGS map in real-time as it moves, collecting odometry
and RGB-D image information. We extract a voxel grid
from the underlying radiance field for computational effi-
ciency when computing information gain metrics. Within this
voxelized representation, we distinguish between observed
and unobserved space, where the observed space is further
segmented into occupied and free space. To enable task-
aligned scene coverage, we fuse geometric and semantic
information-gain metrics, guiding the robot toward regions
with high geometric uncertainty and semantic relevancy to
the input query. Our method is visualized in Fig. 1.
A. Real-Time Semantic Radiance Field Training
To construct semantic 3DGS maps in real-time, VISTA
builds upon NerfBridge [48], [49] and its 3DGS extension
Splatbridge [50], both real-time methods for online training
of radiance fields. In both, images from the robot’s onboard
cameras and poses are aggregated into a streaming dataset
that is used to continuously optimize the radiance field.
To add semantic information to these platforms, we then
distill semantic embeddings from the 2D vision-language
model CLIP [1] into the online radiance field using the
distillation procedure used in semantic NeRF literature [51],
computing the CLIP image embeddings for each incoming
image. Specifically, the trained semantic field f : R3 7→Rl,
parametrized by a multi-resolution hashgrid followed by
a multilayer perceptron, maps a 3D point to a semantic
embedding.
To optimize the semantic field, 3D points back-projected
from the predicted depth image are used to generate in-
puts for training f. The parameters of f are optimized
using gradient-based optimization of the mean-squared error
(MSE) between the predicted and ground-truth CLIP seman-
tic embeddings. We also include the cosine similarity as a
component in the loss function in 3DGS. The parameters of
f and those of the base 3DGS are trained simultaneously.
While the 3DGS map is training, a point cloud repre-
sentation of the field containing RGB colors and semantic
embeddings is published in real-time. Simultaneously, a
subset of the training poses is also returned. If the dataset
size is below some maximum number of poses N, the full
dataset is returned, otherwise N poses are sampled from the
training dataset.
B. VISTA-Map: 3D Voxel Grid Representation
The published point cloud from the 3DGS training pro-
cedure is voxelized into a grid of fixed size and resolution
centered on the robot’s current position. In this way, the map
representation is restricted from growing with the number of
Gaussians in the 3DGS map.
The voxels in the grid are characterized into one of three
categories: occupied, free space, and unobserved. All point
cloud information (e.g. color and semantic embeddings) is
registered into occupied voxels. In order to separate the
remaining voxel grid cells into either unobserved or free
cells, VISTA employs voxel traversal [52] from the training
cameras of the 3DGS pipeline. All rays corresponding to

<!-- page 4 -->
Voxel Traversal
Past Trajectory
Sub-optimal Gain
Optimal Gain
Observed 
and Free
Observed 
and Occupied
Unobserved
Termination
Dataset
Fig. 2. Geometric information gain based on view diversity coverage. Given
a point cloud, voxels are characterized as free, occupied, and unobserved.
These categories are determined through voxel traversal over the camera
rays. All occupied voxels contain a list of view directions derived from the
dataset (blue). Pixel-level geometric gain can be rendered from arbitrary
viewpoints through voxel traversal, terminating at either occupied or unob-
served voxels. A simple coverage metric is computed between the candidate
view direction and the per-voxel list of dataset-derived directions. Rays that
hit unobserved voxels always render the highest gain. The gain metric first
prioritizes viewing unobserved regions and then viewing occupied regions
from different directions. As a result, viewpoints that are similar to the
dataset render to a sub-optimal gain (red), while those that are different
return a high gain (green). The mechanism is visualized in the inset, with
low gain directions being red and pointing to denser areas of the unit sphere,
while high gain directions are green and point towards sparser regions.
pixels of the training camera originate at the camera origin
and either terminate at an occupied voxel or exit the voxel
grid. All voxels that intersect with rays up until termination
are deemed free space. The remaining voxels are assigned
as unobserved voxels.
Voxel traversal of the voxel map can similarly be used to
render RGB, depth, semantic, and geometric gain images.
From camera pose x, rays parametrized as r(t) = o + t ·
d corresponding to each pixel are cast into the voxel map,
terminating when either an occupied or unobserved voxel is
traversed or some maximum draw distance is reached. Rays
that terminate at an occupied or unobserved voxel render
that voxel’s attributes into the pixel. The geometric gain and
semantic images are used in active view planning, detailed
in Section IV-C.
C. VISTA-Score: Information Gain Quantification
Once the voxel grid representation of the environment
is obtained, we compute information gain metrics from
the voxel grid. As described in Section IV-B, the training
poses and the voxel traversal mechanism are used to store
information about the directions from which camera rays of
the training views intersect with the respective voxel.
With this stored information, we evaluate the proposed
geometric information gain of a candidate camera pose for
a particular voxel by comparing the new proposed view
direction with the existing view directions for that voxel.
Specifically, we compute the dot product of each of the
existing voxel’s direction vectors against the new camera
pose rays that intersect with that voxel. For new rays that
contrast greatly from the closest training ray, the dot product
will approach −1. For new rays that are very similar to the
closest training ray, the dot product will approach 1. We
then normalize this information between 0 and 1 for each
ray from the proposed camera for the following pixel-wise
gain metric,
gI(dn
x ) = min(−dT
v dn
x ) + 1
2
,
(2)
where dx is an array of all direction vectors of rays generated
from camera pose x, and the superscript n denotes a single
ray index in this array. Similarly, dv is an array of all
direction vectors v stored in that voxel from previous camera
views. The min is taken over indices of the resulting vector,
to give the cosine similarity between the proposed view and
closest existing view. Subsequently, we compute an image-
wise geometric information gain for the candidate pose by
taking the mean of the pixel values,
GI(x) = 1
Nr
Nr
X
n=1
gI(dn
x ),
(3)
where Nr is the number of rays from the image taken at
pose x. For the semantic information gain, we also use the
voxel traversal mechanism to produce a per-pixel semantic
value in images rendered in the voxelized map. The semantic
information gain is then computed as the mean value of all
pixels in the image, denoted by GS(x), and the two metrics
can be used together to compute a VISTA-Score along a
trajectory of viewing directions as follows,
G(¯x) =
X
x∈¯x
γK−k(cGI(x) + GS(x)),
(4)
where the path ¯x is described by a sequence of waypoints
x, K is the number of waypoints in the path, k is the index
of the waypoint in the path, γ is a discount factor, and c
is a weighting factor of the geometric information gain. In
Section IV-D, we detail how paths are sampled for scoring.
D. VISTA-Plan: Informative Planning
To generate candidate paths for exploration (4), we sample
trajectories that are biased toward regions of high information
gain. The algorithm used for planning is detailed in Algo.
1. In this planning pipeline, we first create a 2D voxel grid,
V′ from the 3D voxel grid, V, by slicing a band of V in
the z-direction in which the robot will operate, omitting

<!-- page 5 -->
the floor and ceiling. The voxels are then assigned in V′
by priority in order of observed-occupied, unobserved, then
observed-free, along the z-dimension. Semantic information
is similarly encoded in V′, by slicing a band of the 3D voxel
grid, and summing the semantic values of each voxel grid
cell in the height dimension. In Algo. 1, this procedure is
captured in the function FlattenVoxelGrid.
Using this 2D representation of the environment, we
encode global geometric and semantic information from the
scene to bias trajectories toward regions of the environment
with highest information gain. For geometric information
gain, we search for frontier cells, Df from V′, where
observed-free cells are bordered by unobserved cells. For
semantic information gain, the top-m 2D grid cells with
the highest semantic similarity values are used to create
a Categorical distribution that is then sampled to generate
data, Ds. The frontier cells along with the samples from
the Categorical distribution are then used to fit a Gaussian
Mixture Model (GMM) probability distribution across the
environment.
To generate candidate plans that are biased toward regions
of high geometric uncertainty and semantic information gain,
we use Dijkstra’s algorithm on 2D position coordinates to
compute the shortest path between each observed-free cell
in the occupancy grid and the robot’s current state si, where
i is the MPC replanning index. From the set of all candidate
plans P, we randomly sample from the GMM for target
positions. This allows the updated set of randomly sampled
trajectories, ˆP, to be biased toward the frontiers and the
highest scoring semantic regions. This trajectory sampling
procedure is shown in lines 2 through 5 of Algo.1. While
this procedure accounts for the 2D path of the robot in the
environment, the viewing angles along the trajectory, ¯ψ, are
determined by pointing the robot toward the closest frontier
cell or GMM mean. These viewing angles are computed to
be dynamically feasible from the robot’s current orientation
with control limits.
After generating the candidate trajectories, we score each
trajectory using (4) and select the path with the highest
score for the robot to track. This planning procedure is
repeated in a Model-Predictive Control (MPC) loop. As the
replanning loop progresses, we additionally decay c in (4)
with parameter β and replanning index i to gradually weigh
the semantic score higher than the geometric score.
V. RESULTS
To demonstrate the contributions of our method, we first
compare our geometric information gain method to two
baseline methods: FisherRF [17] and Bayes’ Rays [33], on
static image-pose datasets that are collected from the real
world. This evaluation setup allows us to directly compare
our proposed information gain metric with prior work. We
then incorporate semantic information and our proposed
planning approach to implement the full pipeline in hard-
ware on a quadrotor platform. On this hardware platform,
we compare our method to two baselines; one using only
semantic relevance and the other only geometric information
Algorithm 1 VISTA-Plan
Input: si, V, c, β, z
1: V′ = FlattenVoxelGrid(V)
2: Df = GetFrontiers(V′)
3: Ds = GetSemanticSamples(V′)
4: P = Dijkstra(si, V′)
5: ˆP ∼SampleTrajectories(P, GMM(Df, Ds))
6: G = ∅
7: for ¯p ∈ˆP do
8:
¯ψ = FeasibleHeadings(Df, Ds, ¯p, si)
9:
¯s ←¯p ∪¯ψ
10:
¯x ←ConstructFullPose(¯s, z)
11:
G = VISTAScore(¯x, V, c)
12:
G ←G ∪G
13: end for
14: c = βic
15: ¯s∗= GetBestTrajectory(G, ˆP)
gain. The semantic information-only baseline plans greedily
toward the highest semantic information point in the 3DGS
map, while the geometric information gain baseline is our
reimplementation of RT-Guide [34]. This experiment eval-
uates the advantages of fusing semantic information and
geometric uncertainty in robot exploration problems. Lastly,
we demonstrate our full pipeline in hardware on a Boston
Dynamics Spot quadruped robot to show the versatility of our
method to different types of hardware platforms. We show
third-person views of the Quadrotor and Spot robot in their
respective testing environments in Fig. 4.
A. Next Best View Selection Baseline Comparisons
To evaluate our geometric information gain metric, we
compare against baseline approaches FisherRF [17] and
Bayes’ Rays [33]. In our baseline comparisons, we train a
radiance field using a predetermined set of training views
for a fixed number of iterations (1000). We then apply each
geometric information gain metric to select a fixed number of
additional views. After including these additional views, we
train the models for 1000 iterations and render images from
fixed test viewpoints. We evaluate each method using the
standard metrics: Peak-Signal-Noise-Ratio (PSNR), Learned
Perceptuation Image Patch Similarity (LPIPS), and Structural
Similarity Index Measure (SSIM).
We evaluate each method across six scenes: three bench-
mark scenes in Nerfstudio (Plane, Kitchen, and Poster) and
three additional datasets (Flight, Clutter, and Adirondacks),
shown in Fig. 3. We provide the performance results of each
method in Fig. 3. We find that VISTA achieves the highest
PSNR and SSIM scores and the lowest LPIPS score across
all scenes. Moreover, VISTA requires the fewest number
of training iterations to reach the best PSNR, SSIM, and
LPIPS scores in many scenes, demonstrating the VISTA’s
superiority in selecting informative views compared to the
other methods. For example, the best-competing method,
FisherRF, requires almost twice as many training iterations to
achieve the same photometric scores as VISTA, in the Poster

<!-- page 6 -->
PSNR
SSIM
LPIPS
Time (s)
Training Iteration
VISTA
FisherRF
Bayes Rays
Ground-truth
Fig. 3.
Our geometric information gain metric significantly outperforms baselines FisherRF and Bayes Rays in the next best view selection task for about
50K iterations in three visual reconstruction metrics and computation time. Six real scenes were used in this comparison, three from the Nerfstudio dataset
{Plane, Kitchen, Poster} and three additional datasets {Flight, Clutter, Adirondacks}.
and Adirondacks scenes. Meanwhile, VISTA requires about
the same or lower computation time as the best-competing
methods. These results indicate that VISTA’s information-
gain metric accurately captures the information content of
candidate views, compared to prior work.
B. Quadrotor Hardware Experiments
We first demonstrate our full method in hardware on a
custom-built quadrotor that uses a ZED-Mini stereo camera
for images, and a Jetson Orin Nano onboard computer. For
pose feedback, we use an OptiTrack external motion capture
system, and all 3DGS training and planning is done on a
desktop computer that has an Intel(R) Core(TM) i9-13900K
CPU, and an NVIDIA GeForce RTX 4090 GPU. The quadro-
tor receives waypoints from the offboard computer and uses
an onboard state machine and PX4 flight controller for lower
level control.
We test our full system on a quadrotor platform and
compare against two different baseline approaches. The
first baseline solely uses the geometric information gain to
quantify trajectories that are sampled uniformly throughout
the environment, and is based off the work in RT-Guide [34].
Specifically, we compute the change in the means of the
Gaussians in the scene, and use this signal to find the regions
of the environment where the Gaussians are changing the
most. Instead of using the planning method that is used in
the original paper, we adapt our method to seed the GMM
with these high changing Gaussians. When trajectories are
sampled from the GMM and scored, they are scored by the
counts of the high uncertainty and low uncertainty Gaussians
visible from each view point along the sampled trajectory.
In the second baseline, the method greedily plans toward the
point from the 3DGS point cloud with the highest semantic
similarity and points its heading along the velocity vector of
the path.
We compare our full pipeline against these two baselines
using three different query objects, with two different map
configurations, illustrated in Fig. 4. In the first map configu-
ration, the query objects are not occluded in the environment.
In the second map configuration, we intentionally occlude
the objects from the drone. We expect most methods to be
able to succeed in the first map, as the query object should
be relatively easy to find and have the stopping condition
trigger. In the second map, we expect methods that do not
account for geometric information gain to struggle to find
the query object.
We evaluate all methods on success rate (SR), time to
reach (TTR), and success weighted by inverse path length
(SPL), as done in [43] and [44]. We enforce a maximum
amount of time for the quadrotor to find the query object
based on battery life. Each method is tested on a query and
map for two trials, totaling 12 trials for each method, six
on each map. All methods are given an initialization phase,
where the robot turns in a small circle about its starting pose.
The results are shown in Table I.
Through these experiments, we find that all methods have
some successes on the easy low-occlusion map domain. Our
method has the highest success rate on this map with an
83.33% success rate over the RT-Guide baseline success
rate of 66.67%, and semantic baseline success rate of 50%.
On the more challenging map domain, we find that our
method has a significant improvement over the baseline
methods, where our method has a 100% success rate while
both baselines each have a 16.67% success rate. The results
suggest that our method is able to outperform both baselines
on both maps because we reason about both semantic and
geometric information gain.
C. Spot Quadruped Hardware Experiments
For our second hardware platform, we use a Boston
Dynamics Spot quadruped robot fitted with RGB-D cameras
and onboard odometry. In these experiments, only the front
two cameras on the Spot robot body are used to train the
3DGS map. The offboard computer is equipped with a
4.2 GHz AMD Ryzen 7 7800X3D CPU and an NVIDIA
GeForce RTX 4090 (24GB memory). We communicate with

<!-- page 7 -->
Fig. 4.
The top row shows our three environments and two robots, with the search object in a green circle. The second row shows an example trajectory
of the robot as it executes VISTA. Trajectories are color coded from blue (beginning) to red (end). In the first two maps (columns 1 and 2), we prompt
the quadrotor with the search term “wagon.” In the last map (column 3), we prompt the spot quadruped with the term “cone.”
TABLE I
COMPARISON METRICS OF TIME TO TASK COMPLETION BETWEEN OUR METHOD AND BASELINE METHODS IN THE LOW-OCCLUSION MAP (EASY
MAP DOMAIN) AND HIGH-OCCLUSION MAP (HARD MAP DOMAIN)
Ladder (close)
Sofa (medium)
Wagon (far)
Map
Methods
SR % ↑
TTR ↓
SPL % ↑
SR % ↑
TTR ↓
SPL % ↑
SR % ↑
TTR ↓
SPL % ↑
RT-GuIDE [34]
50
154.32
7.19
100
85.22
46.31
50
57.20
33.52
Easy
Semantic
100
74.23
31.19
50
123.18
12.81
0
N/A
0
VISTA [ours]
100
83.72
29.04
100
72.61
38.42
50
56.21
31.27
RT-GuIDE [34]
50
145.25
10.59
0
N/A
0
0
N/A
0
Hard
Semantic
50
159.21
8.97
0
N/A
0
0
N/A
0
VISTA [ours]
100
141.69
16.26
100
123.24
38.92
100
109.89
33.82
the Boston Dynamics Spot via the SDK and execute the
task-aware plans using desired waypoint control. Qualitative
results of our method on the Spot robot are shown in Fig.
4.
VI. CONCLUSION
In this work, we present an information gain metric
combining both geometric information as well as semantic
gain, and demonstrate how these metrics can be used on
a real-time hardware platform to simultaneously map an
environment and find an object in the map specified through
natural language. We find that our method, VISTA, is fast
enough to run real-time on robot hardware supported by
an offboard GPU workstation, and show that by using a
combined metric for semantic and geometric information
gain, we can more quickly focus on areas of the map that
have higher relevance to the search task. We compared our
geometric information gain metric to previously published
baseline methods using pre-collected datasets of images and
poses, and demonstrated VISTA on two different hardware
platforms in exploration tasks with varying map difficulty.
Limitations: VISTA uses CLIP features, which are
known to mostly encode object-centric semantics. This limits
VISTA to performing object search tasks. VISTA cannot
disambiguate between multiple instances of the same object.
More grammatically sophisticated VLM embeddings could
enable more nuanced search tasks (avoiding danger, using
landmark hints, more dynamic tasks like target following).
VISTA currently requires offboard GPU compute, limiting its
range and potential for field operations. VISTA also requires
that the robot has its own low-level control and localization
stack. In our experiments, this is accomplished with a motion
capture system for the quadrotor, and onboard SLAM system
for the spot quadruped.
REFERENCES
[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable
visual models from natural language supervision,” in International
Conference on Machine Learning (ICML).
PMLR, 2021, pp. 8748–
8763.
[2] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski,
and A. Joulin, “Emerging properties in self-supervised vision trans-
formers,” in Proceedings of the IEEE/CVF international conference
on computer vision, 2021, pp. 9650–9660.
[3] R. Firoozi, J. Tucker, S. Tian, A. Majumdar, J. Sun, W. Liu, Y. Zhu,
S. Song, A. Kapoor, K. Hausman et al., “Foundation models in
robotics: Applications, challenges, and the future,” The International
Journal of Robotics Research, p. 02783649241281508, 2023.
[4] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen,
A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa et al.,
“Conceptgraphs: Open-vocabulary 3d scene graphs for perception and
planning,” in 2024 IEEE International Conference on Robotics and
Automation (ICRA).
IEEE, 2024, pp. 5021–5028.
[5] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.

<!-- page 8 -->
[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp.
99–106, 2021.
[7] A. Elfes, “Using occupancy grids for mobile robot perception and
navigation,” Computer, vol. 22, no. 6, pp. 46–57, Jun. 1989.
[8] P. Kim, J. Chen, and Y. K. Cho, “Slam-driven robotic mapping and
registration of 3d point clouds,” Automation in Construction, vol. 89,
pp. 38–48, 2018.
[9] J. Ryde and H. Hu, “3d mapping with multi-resolution occupied voxel
lists,” Autonomous Robots, vol. 28, pp. 169–185, 2010.
[10] B. Yamauchi, “A frontier-based approach for autonomous exploration,”
in Proceedings 1997 IEEE International Symposium on Computational
Intelligence in Robotics and Automation CIRA’97.’Towards New Com-
putational Principles for Robotics and Automation’.
IEEE, 1997, pp.
146–151.
[11] J. Yu, H. Shen, J. Xu, and T. Zhang, “Echo: An efficient heuristic
viewpoint determination method on frontier-based autonomous explo-
ration for quadrotors,” IEEE Robotics and Automation Letters, vol. 8,
no. 8, pp. 5047–5054, 2023.
[12] T. Cieslewski, E. Kaufmann, and D. Scaramuzza, “Rapid exploration
with multi-rotors: A frontier selection method for high speed flight,”
in 2017 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS).
IEEE, 2017, pp. 2135–2142.
[13] A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and R. Siegwart,
“Receding horizon path planning for 3d exploration and surface
inspection,” Autonomous Robots, vol. 42, pp. 291–306, 2018.
[14] K. Saulnier, N. Atanasov, G. J. Pappas, and V. Kumar, “Information
theoretic active exploration in signed distance fields,” in 2020 IEEE
International Conference on Robotics and Automation (ICRA). IEEE,
2020, pp. 4080–4085.
[15] Y. Tao, Y. Wu, B. Li, F. Cladera, A. Zhou, D. Thakur, and V. Kumar,
“Seer: Safe efficient exploration for aerial robots using learning to
predict information gain,” in 2023 IEEE International Conference on
Robotics and Automation (ICRA).
IEEE, 2023, pp. 1235–1241.
[16] Y. Kompis, L. Bartolomei, R. Mascaro, L. Teixeira, and M. Chli,
“Informed sampling exploration path planner for 3d reconstruction
of large scenes,” IEEE Robotics and Automation Letters, vol. 6, no. 4,
pp. 7893–7900, 2021.
[17] W. Jiang, B. Lei, and K. Daniilidis, “Fisherrf: Active view selection
and uncertainty quantification for radiance fields using fisher informa-
tion,” arXiv preprint arXiv:2311.17874, 2023.
[18] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hed-
man, “Mip-NeRF360: Unbounded anti-aliased neural radiance fields,”
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2022.
[19] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph.,
vol. 41, no. 4, pp. 102:1–102:15, Jul. 2022.
[20] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-splatting:
Alias-free 3d gaussian splatting,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
19 447–19 456.
[21] A. Rosinol, J. J. Leonard, and L. Carlone, “NeRF-SLAM: Real-Time
Dense Monocular SLAM with Neural Radiance Fields,” arXiv preprint
arXiv:2210.13641, 2022.
[22] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “Gs-
slam: Dense visual slam with 3d gaussian splatting,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, 2024, pp. 19 595–19 604.
[23] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian
splatting slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 18 039–18 048.
[24] H. Zhan, J. Zheng, Y. Xu, I. Reid, and H. Rezatofighi, “Activermap:
Radiance field for active mapping and planning,” arXiv preprint
arXiv:2211.12656, 2022.
[25] S. Lee, L. Chen, J. Wang, A. Liniger, S. Kumar, and F. Yu, “Uncer-
tainty guided policy for active robotic 3d reconstruction using neural
radiance fields,” IEEE Robotics and Automation Letters, vol. 7, no. 4,
pp. 12 070–12 077, 2022.
[26] Z. Yan, H. Yang, and H. Zha, “Active neural mapping,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 10 981–10 992.
[27] D. Yan, J. Liu, F. Quan, H. Chen, and M. Fu, “Active implicit object
reconstruction using uncertainty-guided next-best-view optimization,”
IEEE Robotics and Automation Letters, 2023.
[28] R. Jin, Y. Gao, H. Lu, and F. Gao, “Gs-planner: A gaussian-splatting-
based planning framework for active high-fidelity reconstruction,”
arXiv preprint arXiv:2405.10142, 2024.
[29] Z. Xu, R. Jin, K. Wu, Y. Zhao, Z. Zhang, J. Zhao, Z. Gan, and
W. Ding, “Hgs-planner: Hierarchical planning framework for active
scene reconstruction using 3d gaussian splatting,” arXiv preprint
arXiv:2409.17624, 2024.
[30] W. Jiang, B. Lei, K. Ashton, and K. Daniilidis, “Ag-slam: Active
gaussian splatting slam,” arXiv preprint arXiv:2410.17422, 2024.
[31] L. Jin, X. Zhong, Y. Pan, J. Behley, C. Stachniss, and M. Popovi´c,
“Activegs: Active scene reconstruction using gaussian splatting,” IEEE
Robotics and Automation Letters, 2025.
[32] G. Liu, W. Jiang, B. Lei, V. Pandey, K. Daniilidis, and N. Motee,
“Beyond uncertainty: Risk-aware active view acquisition for safe robot
navigation and 3d scene understanding with fisherrf,” arXiv preprint
arXiv:2403.11396, 2024.
[33] L. Goli, C. Reading, S. Sell´an, A. Jacobson, and A. Tagliasacchi,
“Bayes’ rays: Uncertainty quantification for neural radiance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 061–20 070.
[34] Y. Tao, D. Ong, V. Murali, I. Spasojevic, P. Chaudhari, and V. Kumar,
“Rt-guide: Real-time gaussian splatting for information-driven explo-
ration,” arXiv preprint arXiv:2409.18122, 2024.
[35] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik,
“LERF: Language embedded radiance fields,” in IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), 2023, pp. 19 729–
19 739.
[36] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You,
Z. Wang, and A. Kadambi, “Feature 3dgs: Supercharging 3d gaussian
splatting to enable distilled feature fields,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 21 676–21 685.
[37] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, “Langsplat:
3d language gaussian splatting,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
20 051–20 060.
[38] O. Shorinwa, J. Sun, and M. Schwager, “Fast-splat: Fast, ambiguity-
free
semantics
transfer
in
gaussian
splatting,”
arXiv
preprint
arXiv:2411.13753, 2024.
[39] J. Yu, T. Chen, and M. Schwager, “Hammer: Heterogeneous, multi-
robot semantic gaussian splatting,” arXiv preprint arXiv:2501.14147,
2025.
[40] O. Shorinwa, J. Sun, M. Schwager, and A. Majumdar, “Siren: Se-
mantic, initialization-free registration of multi-robot gaussian splatting
maps,” arXiv preprint arXiv:2502.06519, 2025.
[41] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment
anything,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 4015–4026.
[42] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,”
Advances in neural information processing systems, vol. 36, 2024.
[43] S. Y. Gadre, M. Wortsman, G. Ilharco, L. Schmidt, and S. Song, “Cows
on pasture: Baselines and benchmarks for language-driven zero-shot
object navigation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2023, pp. 23 171–23 181.
[44] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “Vlfm: Vision-
language frontier maps for zero-shot semantic navigation,” in 2024
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2024, pp. 42–48.
[45] D. Ong, Y. Tao, V. Murali, I. Spasojevic, V. Kumar, and P. Chaudhari,
“Atlas navigator: Active task-driven language-embedded gaussian
splatting,” 2025. [Online]. Available: https://arxiv.org/abs/2502.20386
[46] S. Papatheodorou, N. Funk, D. Tzoumanikas, C. Choi, B. Xu, and
S. Leutenegger, “Finding things in the unknown: Semantic object-
centric exploration with an mav,” in 2023 IEEE International Confer-
ence on Robotics and Automation (ICRA).
IEEE, 2023, pp. 3339–
3345.
[47] S. Barbas Laina, S. Boche, S. Papatheodorou, S. Schaefer, J. Jung, and
S. Leutenegger, “Findanything: Open-vocabulary and object-centric
mapping for robot exploration in any environment,” arXiv e-prints,
pp. arXiv–2504, 2025.
[48] J. Yu, J. E. Low, K. Nagami, and M. Schwager, “Nerfbridge: Bringing

<!-- page 9 -->
real-time, online neural radiance field training to robotics,” arXiv
preprint arXiv:2305.09761, 2023.
[49] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, J. Kerr, T. Wang,
A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja, D. McAllister,
and A. Kanazawa, “Nerfstudio: A modular framework for neural
radiance field development,” in ACM SIGGRAPH 2023 Conference
Proceedings, ser. SIGGRAPH ’23, 2023.
[50] T. Chen, A. Swann, J. Yu, O. Shorinwa, R. Murai, M. Kennedy III,
and M. Schwager, “Safer-splat: A control barrier function for safe
navigation with online gaussian splatting maps,” arXiv preprint
arXiv:2409.09868, 2024.
[51] W. Shen, G. Yang, A. Yu, J. Wong, L. P. Kaelbling, and P. Isola, “Dis-
tilled feature fields enable few-shot language-guided manipulation,” in
7th Annual Conference on Robot Learning, 2023.
[52] J. Amanatides and A. Woo, “A fast voxel traversal algorithm for ray
tracing,” Proceedings of EuroGraphics, vol. 87, 08 1987.
