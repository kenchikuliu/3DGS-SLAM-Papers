<!-- page 1 -->
YOPO-Nav: Visual Navigation using 3DGS Graphs from One-Pass Videos
Ryan Meegan
Rutgers University
ryan.meegan@rutgers.edu
Adam D’Souza
Rutgers University
adam.dsouza@rutgers.edu
Bryan Bo Cao
Stony Brook University
boccao@cs.stonybrook.edu
Shubham Jain
Stony Brook University
jain@cs.stonybrook.edu
Kristin Dana
Rutgers University
kristin.dana@rutgers.edu
Abstract
Visual navigation has emerged as a practical alternative
to traditional robotic navigation pipelines that rely on de-
tailed mapping and path planning. However, constructing
and maintaining 3D maps is often computationally expen-
sive and memory-intensive. We address the problem of vi-
sual navigation when exploration videos of a large environ-
ment are available. The videos serve as a visual reference,
allowing a robot to retrace the explored trajectories with-
out relying on metric maps. Our proposed method, YOPO-
Nav (You Only Pass Once), encodes an environment into a
compact spatial representation composed of interconnected
local 3D Gaussian Splatting (3DGS) models. During navi-
gation, the framework aligns the robot’s current visual ob-
servation with this representation and predicts actions that
guide it back toward the demonstrated trajectory. YOPO-
Nav employs a hierarchical design: a visual place recogni-
tion (VPR) module provides coarse localization, while the
local 3DGS models refine the goal and intermediate poses
to generate control actions. To evaluate our approach, we
introduce the YOPO-Campus dataset, comprising ∼4 hours
of egocentric video and robot controller inputs from over
6 km of human-teleoperated robot trajectories. We bench-
mark recent visual navigation methods on trajectories from
YOPO-Campus using a Clearpath Jackal robot. Experi-
mental results show YOPO-Nav provides excellent perfor-
mance in image-goal navigation for real-world scenes on a
physical robot. The dataset and code will be made publicly
available for visual navigation and scene representation re-
search.
1. Introduction
In unfamiliar environments, humans naturally explore to
learn the scene, forming a mental map organized around
key landmarks, salient objects, topology, and spatial rela-
tionships [16, 58, 80].
The concept of cognitive mental
Figure 1. YOPO-Campus Dataset Birds-eye-view (BEV) of the
paths traversed by the Jackal robot under human teleoperation
across Rutgers University, Busch Campus. Egocentric video from
the robot and action control sequences are captured over 4 hours
across 6 km.
maps has driven new computational scene representations
and methods for visual navigation, enabling navigation on
low-cost mobile robots equipped with only an onboard cam-
era and no odometry, GPS, or multi-modal sensing.
Traditionally, robotic navigation is treated as a geometric
problem: exhaustively traversing the environment to build a
detailed 3D reconstruction, then planning using the gener-
ated representation. This large-scale mapping approach is
computationally intensive and often impractical for time-
sensitive applications, where prolonged exploration is not
feasible. Moreover, human activity in crowded settings fur-
ther complicates robot exploration and mapping. Unlike hu-
mans, who enter new environments with prior knowledge of
scene semantics and appropriate movement patterns, robots
using traditional SLAM lack this higher-level contextual
awareness.
Early efforts to leverage visual information for scene rep-
resentations primarily employed topological graphs [11, 25,
69–71, 74]. Recent approaches leverage generative mod-
els to anticipate future trajectories, enabling more accurate
1
arXiv:2512.09903v1  [cs.RO]  10 Dec 2025

<!-- page 2 -->
Figure 2. YOPO-Nav: Videos of human-teleoperated robot trajectories are used to construct a scene representation as a graph of local
3DGS nodes. When the embodied agent revisits the scene, coarse localization is performed using Visual Place Recognition (VPR), linking
frames in the the recorded trajectory to a corresponding node in the 3DGS graph. The robot’s real-world pose, p′, is localized in the 3DGS
node via PnP RANSAC, and the difference from the desired pose, p, yields a transformation matrix, directing actions to align with p (see
Section 3.2.)
navigation decisions [6, 38, 85]. Currently, 3D Gaussian
Splatting (3DGS) models [36] are changing visual naviga-
tion by supporting real-time, photo-realistic rendering, and
novel view synthesis for algorithmic navigation decisions.
Scene representations with large global 3DGS models are
a shift from both classic SLAM and topological graphs
[54, 76, 84].
Building 3D scene representations directly from one-
pass videos is highly compelling, as videos are often easy
to capture [6, 11, 50, 66, 83]. However, a key limitation of
videos is that they lack true 3D structure: the scene is en-
tirely constrained to the trajectory originally taken. Since
3D Gaussian Splatting can transform a sequence of video
frames into Gaussian primitives that encode depth and in-
terpolate radiance across viewpoints, it allows free move-
ment along the recorded trajectory and exploration beyond
the original perspectives. In other words, a robot or agent
that strays from the recorded trajectory can leverage the
constructed 3D representation to compute geometry-based
corrective actions and realign with the trajectory. This in-
sight motivates our approach: an exploration video defines
a traversable path through the scene, a “video breadcrumb
trail”, that the agent can later follow to navigate the envi-
ronment.
In this work, we merge the concepts of topological
graphs and 3DGS methods to construct a graph of lo-
cal 3DGS models from one-pass videos.
We propose a
framework called YOPO-Nav (You Only Pass Once), en-
coding an environment into a compact spatial representa-
tion composed of interconnected local 3D Gaussian Splat-
ting (3DGS) models (see Figs. 2 and 3).
Our approach
is lightweight, interpretable, and scalable, relying only on
single-pass videos of the scene; no GPS or odometry is as-
sumed. The YOPO-Nav scene representation draws inspira-
tion from human mental maps: identifying points of inter-
est, partitioning the environment into manageable chunks,
and organizing these chunks into an interconnected struc-
ture (see Fig. 3). Navigation proceeds hierarchically: Vi-
sual Place Recognition (VPR) localizes the start and goal re-
gions, recalls the surroundings, and links connecting chunks
into a traversable, continuous route.
Key to our approach is the integration of frameworks that
can create 3DGS models from un-calibrated multiview im-
ages [27, 29, 32, 47, 86]. By jointly estimating 3D Gaus-
sians and poses, we obtain a stable representation that en-
ables scene learning from one-pass videos and supports
navigation on a physical robot (Clearpath Jackal).
As a contribution of this paper, and to test YOPO-
2

<!-- page 3 -->
Figure 3.
YOPO-Nav Scene Representation YOPO-Nav rep-
resents a scene as a graph of 3DGS models,
built from
∼50–55 frames at 448×336 resolution, using videos of human-
teleoperated robot trajectories. Edges connect nodes by frame con-
tinuity (within each video) or by visual similarity (across different
videos). Navigation proceeds by localizing new camera observa-
tions in the 3DGS and aligning them to the estimated poses from
the videos.
Nav, we introduce YOPO-Campus (see Fig. 1), a dataset
collected on a college campus using a Clearpath Jackal
robot equipped with an Intel RealSense D435 camera and
teleoperated by a human via a wireless DualShock con-
troller.
The robot traversed 35 unique trajectories along
sidewalks connecting campus buildings, covering 6 km (4
hours of footage).
We evaluate YOPO-Nav in the real
world on YOPO-Campus trajectories, demonstrating excel-
lent performance in visual navigation on the common task
of image-goal navigation, without reliance on odometry and
GPS. We further demonstrate strong transfer of pre-trained
VPR networks, where a geographically distinct and unaffili-
ated dataset (GND: Global Navigation Dataset [46]) is used
for pre-training but tested on the YOPO-Campus trajecto-
ries.
Our contributions are: (i) YOPO-Campus, a 6 km (4
hours) dataset collected by a human-teleoperated robot in an
outdoor college campus environment (ii) Scene representa-
tion comprised of a graph of 3D Gaussian Splats, where
small spatial regions are represented by their own local
3DGS models (iii) YOPO-Nav, an algorithm utilizing this
graph of 3DGS models that enables visual navigation across
scenes with minimal human intervention.
2. Related Work
Our work relates to four main lines of research: (1) Scene
Representations, (2) 3D Gaussian Splatting, (3) Visual
Place Recognition, and (4) Embodied Visual Navigation.
Scene Representations for Navigation
Beyond full met-
ric maps, alternative scene representations have been intro-
duced to learn navigation policies. Scene graphs represent
visual environments as structured graphs, with nodes de-
noting objects and edges capturing semantic or spatial rela-
tionships. Scene graphs, like topological graphs, are typi-
cally non-metric; but while topological graphs map spatial
regions and connectivity, scene graphs span broader entities
such as objects, people, and regions. Recent work [73, 81]
extends scene graphs for navigation by encoding both ob-
ject relations and spatial connectivity.
Bird’s-eye-view (BEV) representations provide an alter-
native using occupancy grids or similar structures to capture
metric spatial relations. BEV scene graphs [49] unify these
ideas by constructing a scene graph on top of a BEV repre-
sentation, combining BEV’s geometric grounding with the
relational reasoning of scene graphs. Topological graphs
have evolved from early cognitive-map-inspired models,
which were symbolic abstractions of connected places, to
recent approaches [31, 37, 44, 52, 73] that integrate learned
spatial embeddings, semantic grounding, and semantic aug-
mentation. Our method models the scene as a graph that
captures spatial relationships without encoding global met-
rics, assigning each node to a local 3D Gaussian Splat.
3D Gaussian Splatting
3D Gaussian splatting [36] has
revolutionized scene modeling. Unlike traditional SLAM,
which outputs point clouds that are later converted to sur-
faces, 3DGS captures the true visual appearance of a scene
using efficient geometric models composed of Gaussian
primitives.
3DGS methods rasterize Gaussian primitives
iteratively, producing high-quality 3D models with novel
view synthesis.
By leveraging established rasterization
techniques, 3DGS methods achieve substantially faster ren-
dering than NeRF [56]. With recent advancements in 3D
Gaussian Splatting, the technology has been applied to mul-
tiple downstream tasks, including navigation. 3DGS has
been integrated into SLAM algorithms to construct detailed
volumetric maps of the environment for localization and
planning [45, 54, 76, 84]. Recent work applies Gaussian
Splatting to embodied visual navigation for fast novel view
synthesis [43] and pose estimation [14, 24]. Building on
recent advances in 3DGS, we avoid global modeling and
instead propose a graph framework of local 3DGS nodes,
combining the strengths of topological graphs and 3DGS
models.
Visual Place Recognition
The goal of visual place recog-
nition (VPR) [4] is to determine whether an image has been
seen before, and this framework is useful in robot localiza-
tion and navigation. Modern VPR models typically com-
bine a deep learning feature extractor with an aggregator.
Common optimizations include enhancing the feature ex-
tractor [2, 34], reformulating aggregator clustering [28], and
improving viewpoint robustness [8]. These advances have
enabled the use of VPR in various embodied robotic navi-
gation tasks [18, 23]. VPR not only provides an effective
method for localization, but can also serve as a foundation
3

<!-- page 4 -->
for representing the environment in navigation tasks. We
build on recent VPR advancements to identify the node in
the 3DGS graph where the agent is currently located. Our
VPR model achieves strong performance when trained on a
domain-similar (but geographically distinct) dataset, GND:
Global Navigation Dataset [46].
Embodied Visual Navigation
Several tasks have been
proposed in embodied visual navigation, including object-
goal navigation [75], multi-object-goal navigation [13, 79],
image-goal navigation [12, 40, 41, 87], and instance-image-
goal navigation [9, 39], with most evaluated in simulation
environments such as Habitat [65] and Gibson [82]. De-
ploying visual navigation algorithms on physical robots re-
mains a core challenge, and key objectives are high success
rates for tasks in real environments.
Various strategies have been applied to embodied visual
navigation, including deep reinforcement learning [19, 51,
87], spatial attention [53, 55, 67], and novel view synthe-
sis [38]. Recent approaches also employ diffusion models
to sample from the action space [63] or to predict future
observations along a trajectory [6, 85] to determine opti-
mal navigation actions. ViNT [71] introduced a foundation
transformer model for visual navigation that uses diffusion
to predict intermediate goal images and a topological graph
structure for mapping and planning.
Several subsequent
works [70, 74] proposed similar generalizable navigation
policies, combining transformer architectures with diffu-
sion models. Unlike generalist navigation foundation mod-
els, our method focuses on learning a specific environment
from a single exploration video. Our results indicate that
YOPO-Nav has better performance than pre-trained foun-
dation models, suggesting that when one or more explo-
ration videos are available, a strong representation can be
produced.
3. Methods
3.1. YOPO-Campus
YOPO-Campus, our dataset of robot trajectories, was
collected at Rutgers University, Busch Campus, during
the Summer and Fall of 2024 and 2025, primarily in
low-activity periods (summer break, weekends, or early
mornings) to ensure that no people appear in the videos.
The dataset consists of timestamps, synchronized RGB
(8-bit) and depth (16-bit) imagery at VGA resolution
(640 × 480), controller inputs, compass directions, and
ground-truth GPS and Wi-Fi Fine Timing Measurements
(FTM). Data was gathered using a Clearpath Jackal robot,
equipped with an Intel RealSense D435 camera and a
Google Pixel 3a smartphone, that was manually teleoper-
ated by a human using a wireless DualShock controller. The
Pixel 3a, mounted on the robot and connected via Blue-
tooth to the Jackal robot, logged GPS, compass direction,
Figure 4. YOPO-Campus Dataset Viewer GUI for efficient visu-
alization of YOPO-Campus: left: bird’s-eye view annotated with
routers, planned path, and robot position based on the current
frame; right: RGB/depth images; center: frame data (timestamp,
action, FTM/RSSI, compass, GPS) with a player to view the data
associated with each frame in the video.
and Wi-Fi measurements (FTM and RSSI) through a cus-
tom Android app we developed.
To support the collec-
tion of FTM and RSSI data, 3 Google Nest Pro routers
were placed along each trajectory, and their GPS coordi-
nates were recorded. Wi-Fi FTM and RSSI measurements
are intended for future research and are currently not used
in YOPO-Nav. GPS is used only for evaluation of our VPR
model.
Controller inputs were limited to discrete actions:
0.25 m forward and backward using the up and down arrows
on the D-pad, and ±15◦rotations using the square and cir-
cle buttons, respectively. After every action, a timestamp is
logged, RGB and depth images are captured, and the Pixel
3a is queried for GPS, compass direction, and Wi-Fi FTM
and RSSI data, which is sent back to the Jackal robot over a
Bluetooth connection. Data was collected along 35 side-
walk trajectories (100 m–250 m, avg. 170 m), each with
unique start and end points.
Trajectories were traversed
bidirectionally to capture complementary perspectives, last-
ing 3–16 minutes (avg. 6.5 minutes) per path. The dataset
covers most of the campus sidewalk network and includes
academic buildings, statues, and open fields. At sidewalk
junctions, the Jackal robot was rotated in place by the hu-
man teleoperator to more easily align overlapping path seg-
ments. In total, the dataset contains 4 hours of data (26,500
images, actions, etc.) that covers 6 km of the campus. The
dataset has a file size of 26.5 GB (13.1 GB with compres-
sion). A Qt-based GUI was created to view the dataset in an
efficient manner (see Fig. 4).
3.2. YOPO-Nav
YOPO-Nav integrates two complementary components. A
visual place recognition model offers a coarse global esti-
mate of location, while a graph of compressed 3DGS mod-
els (see Fig. 3) captures fine local geometry and appear-
ance. Together they establish a global-to-local hierarchy for
navigation (see Fig. 2): global image retrieval selects the
4

<!-- page 5 -->
Dataset
R@1 ↑
R@5 ↑
R@10 ↑
R@15 ↑
∆R@1 ↑
GND Val Set [46]
92.27
97.16
98.37
98.64
+7%
YOPO-Campus
68.54
93.68
96.55
97.37
+3%
Table 1.
Recall results of YOPO-Loc on YOPO-Campus and
the GND [46] validation set.
R@1 improved by 3% on
YOPO-Campus and 7% on the GND validation set. The perfor-
mance gain on YOPO-Campus provides evidence of transfer learn-
ing from GND. The R@1–R@5 gain on YOPO-Campus motivates
a top-5 KNN search in the FAISS [20] index built from YOPO-Loc
features.
Model
Params (M) ↓FLOPs (G) ↓R@1 ↑R@5 ↑R@10 ↑R@15 ↑
SALAD [28]
87.99
22.22
67.89
92.95
96.26
97.41
MegaLOC [7]
228.64
22.37
68.54
93.51
96.53
97.31
FoL [77]
308.83
164.75
66.54
92.52
95.94
97.17
YOPO-Loc
99.85
15.74
68.62
93.95
96.55
97.39
Table 2.
Comparison of visual place recognition models on
YOPO-Campus. Best values in each column are in bold. YOPO-
Loc achieves competitive recall compared to general-purpose
models while requiring fewer parameters (with the exception of
SALAD [28]) and FLOPs.
relevant 3DGS, and local pose estimation determines the
robot’s position within that 3DGS and computes the action
needed to align with the estimated poses from the recorded
trajectories. This representation is lightweight, avoiding the
cost of full 3D reconstruction in SLAM approaches and loss
of detail in topological graph approaches, while remaining
interpretable for human guidance and correction.
VPR
The visual place recognition model, hereafter re-
ferred to as YOPO-Loc, was trained following the open-
source framework created by OpenVPRLab [1].
This
framework consists of a datamodule, feature backbone, ag-
gregator, and loss module. DINOv3 [72], the current state-
of-the-art feature extractor, was chosen as the backbone
while Bag of Queries (BoQ) [3] was chosen as the aggre-
gator due to its strong performance and deployment sim-
plicity [3, 77]. Multi-similarity loss was selected as the loss
module because of its stable training behavior. The Global
Navigation Dataset (GND) [46] was selected as the data-
module. GND is a large-scale dataset collected across ten
university campuses, covering approximately 2.7 km2 and
containing RGB images and associated GPS. We selected
GND for training because it resembles our dataset (a Jackal
robot navigating a campus environment) and we expected
successful transfer learning.
Only 6 of the 10 campuses in GND [46] were used, since
the others lacked GPS. After preprocessing, we split at the
campus level, assigning all images from a given campus
exclusively to either training (80%) or validation (20%) to
prevent cross-contamination between sets. In total, there
were 55,317 training and 14,584 validation images. Within
each campus, images were defined as belonging in the same
place if their associated GPS coordinates were within a
1.0 m radius of each other. For validation, database images
were defined as images within a 0.5 m radius of each other,
while query images were sampled from a 0.5 m band out-
side each database region. Queries were matched against
the database and performance was evaluated using recall@k
(e.g., R@1, R@5), the percentage of queries that success-
fully retrieved the matching database within the top-k re-
sults. See Section 4.1 for training parameters.
YOPO-Campus served as an additional validation set,
following the same protocol as GND [46]. Validation re-
sults for both sets are shown in Table 1.
Evidence of
transfer learning from GND to YOPO-Campus was demon-
strated in YOPO-Loc. YOPO-Loc was compared to sev-
eral notable VPR models on the YOPO-Campus dataset in
Table 2.
YOPO-Loc achieves similar results to general-
purpose models while being considerably more efficient.
3DGS
We
employ
AnySplat
[29],
an
efficient
feed-forward network that generates 3D Gaussian prim-
itives along with camera intrinsics and extrinsics in a
single pass, to construct the graph of 3DGS models. We
selected AnySplat because the lack of camera poses in
our dataset requires a 3DGS method capable of estimating
robust camera poses. We found that AnySplat produced the
least sparse reconstruction with the most accurate camera
pose estimations compared to other pose-free alternatives,
such as VGGT [78], MapAnything [35], and LongSplat
[47].
Frames from each trajectory in YOPO-Campus
were processed in AnySplat [29] and exported as .ply
files along with JSON files containing camera extrinsics,
intrinsics, and the filenames of the frames used in the
3D reconstruction. See Section 4.1 for more details. We
converted all .ply files into the Spatially Ordered Gaussians
(SOG) format using the PlayCanvas splat-transform library
[59], an enhanced implementation of Self-Organizing
Gaussians [57], to reduce the storage footprint of all the
model reconstructions.
Across all 35 trajectories, 547
3DGS models were produced, with a total file size of 17.8
GB after SOG compression (JSONs total 30 MB). During
runtime, SOG files are reconverted to .ply files, which takes
1-3 seconds, so they can be properly decoded and rendered
by AnySplat.
All images in YOPO-Campus, contained within the 547
3DGS models, are used to built a graph that involves two
stages: within each trajectory video, frames are connected
sequentially based on the known order, and across different
trajectory videos, visually similar frames are connected. For
cross-trajectory linking, YOPO-Loc features for all frames
in YOPO-Campus were stored in a GpuIndexFlat FAISS
[20] database, allowing each frame from one trajectory to
retrieve its top-5 candidates from other trajectories via a
simple KNN search.
The resulting top-5 candidates are
5

<!-- page 6 -->
Figure 5. YOPO-Nav GUI The YOPO-Nav GUI is comprised of five widgets: 1) the top-left displays the Jackal robot’s live camera feed;
(2) the bottom-left shows the camera feed’s closest matched frame in the FAISS [20] index; (3) the center presents a BEV of the campus
with the robot’s position, desired goal, and planned path; (4) the top-right displays the start and goal images, next frame in the planned
path, and performance metrics (actions, time, interventions); and (5) the bottom-right renders the 3DGS model and simulated actions in
real-time.
re-ranked using point correspondences from XFeat [60] and
LighterGlue, an improved version of LightGlue [48] by the
authors of XFeat. An edge is retained if the image with
the highest number of correspondences exceeds a prede-
fined threshold (i.e. 500-900, 800 was determined to be
optimal). The resulting graph supports path planning, e.g.
with Dijkstra’s algorithm.
Pose Estimation and Action Generation
Navigation be-
gins by loading the 3DGS model and pose for each frame
in the planned path. Using the current camera observation,
the Jackal robot is then localized within the 3DGS model.
Translation and rotation errors between the estimated pose
and the target pose (next frame in the planned path) are
computed and corrective actions are executed to align with
the target pose in 3DGS environment. These actions are
transferred from the 3DGS environment into the real-world
and are repeated for each frame in the planned path until
the goal is reached.
Two strategies enable this transfer:
(1) deriving a scale factor from the camera height in the
3DGS node relative to its real-world height to map transla-
tions in the 3DGS model to metric units, and (2) applying
Perspective-n-Point (PnP) [26] RANSAC [22] to localize
new camera observations within the 3DGS model.
Upon reaching a frame linked to a new 3DGS model,
an image is rendered at each estimated camera pose (im-
age inputs used in reconstruction) in the model.
Seg-
ment Anything 2 [62] (specifically multi-mask version of
SAM-2.1-base-plus) isolates the ground plane in each ren-
dered image using a single point click positioned at 90%
of the image height and centered along the width.
It is
assumed that, across each image in YOPO-Campus, the
ground planes will exist in this region—and it should exist
based on our data collection procedure. The largest mask
is always used to ensure the as much of the ground plane is
segmented as possible. We define the ground plane as side-
walk, road, or brick pavement that forms the trajectory the
Jackal robot follows. For each rendered image, the depth
values within the ground-plane mask are back-projected
into 3D points using the camera intrinsics, and a plane is
fit to these points via RANSAC. The offset of each fitted
plane is then compared to the Jackal robot’s fixed camera
height of 250 mm to compute a scale factor that converts
3DGS translations into real-world meters (rotations require
no scaling). The final scale factor is obtained by taking the
median across the plane offsets for each estimated camera
pose in the 3DGS model.
To localize within the 3DGS model, feature correspon-
dences are established between the current camera obser-
vation and the next frame in the planned path (target) us-
ing XFeat [60] with LighterGlue. The matched 2D key-
points from the current observation are paired with 3D
points reconstructed from the depth values at the target
pose in the 3DGS. These depth values are back-projected
into 3D camera coordinates using the target pose intrin-
sics and then transformed into the global coordinate frame
with the target pose extrinsics, yielding the 2D–3D cor-
respondences required for PnP-RANSAC pose estimation.
These correspondences are then used by PnP [26] with
RANSAC [22] to estimate the camera pose of the current
observation within the 3DGS. The solution is then refined
6

<!-- page 7 -->
via Gauss-Newton non-linear minimization, yielding a final
4 × 4 transformation matrix that aligns the current observa-
tion to the next frame in the planned path. This matrix is
used in a simple kinematic model for the Jackal robot that
cannot move sideways: rotate to face the goal, translate for-
ward, and rotate in place to match the target orientation.
GUI
A GUI was developed to make YOPO-Nav both in-
tuitive and interpretable (see Fig. 5). The interface provides
transparency into the YOPO-Nav’s decisions and progress,
while also supporting human intervention when needed. At
any point, a human operator (using a controller connected
locally or remotely) can issue discrete corrections of 0.25 m
forward/backward or 15◦left/right rotations, where the GUI
helps to determine when such corrections are appropriate.
Following interventions, YOPO-Nav resumes autonomous
traversal by re-localizing the Jackal robot using YOPO-Loc
and updating its position within the planned path.
4. Experiments and Results
4.1. Comparisons to SOTA
Baselines
As a preliminary experiment, we tested replay-
ing the controller actions from a YOPO-Campus trajectory
on a physical robot in the same environment. As expected,
this naive approach failed within the first few actions due to
drift and error accumulation. To establish a benchmark, we
compare YOPO-Nav to other image-goal navigation meth-
ods, namely ViNT [71] and NoMad [74], using real-world
robot deployment. Both ViNT and NoMad boast generaliz-
able navigation policies that are intended to work as zero-
shot foundation models. While ViNT supports fine-tuning,
we do not fine-tune YOPO-Loc on our own data, so we elect
to compare ViNT and NoMad zero-shot. We run both algo-
rithms using the author’s implementation and their default
navigation parameters.
Experimental Setup
We evaluate YOPO-Nav on image-
goal navigation across several different distance thresholds,
with distances ranging from 1.5 m to 12 m. We evaluate
three sample exploration videos in our dataset, and follow-
ing CityWalker [50], we evaluate the performance of the
two algorithms on each trajectory at each distance threshold
for 8-10 trials. These three exploration videos were chosen
based on the trajectory shape and visual differences; they
vary in the relative level of foliage, the number of build-
ings in the scene, and the number of turns. We deploy the
robot in the same environment as the exploration videos, in
which the scene appearance has significantly (i.e. tested in
winter 2025, versus summer and fall 2024 when the dataset
was collected). Minimal pedestrian traffic was also present
during evaluation. Each algorithm was initialized with a
modified version of the evaluation trajectory, in which the
rotations at intersections (see Section 3.1) were removed,
and an image in the trajectory was specified as the goal.
Metrics
To measure performance we use success rate,
where success is defined as the navigation algorithm reach-
ing the location where the goal image was taken and rec-
ognizing it as the goal. Failure cases end the trial; failure
cases include the navigation algorithm taking incorrect ac-
tions (i.e. turning right instead of left at an intersection),
colliding with an object, or not recognizing the target loca-
tion as the goal.
Real-World Deployment
Evaluation was conducted us-
ing a Clearpath Jackal UGV robot, whose onboard com-
puter (i3-9100TE CPU, no GPU) required a low-latency
link to a remote GPU workstation; we implemented this us-
ing s2n-quic [5], AWS’s Rust implementation of the QUIC
protocol, selected for its broad compatibility, modern fea-
tures (TLS 1.3, post-quantum cryptography), and mutual
TLS authentication over the public campus network. To
optimize throughput, we built minimal client-server scripts
for sending and receiving strings and images. Interoperabil-
ity between Rust and Python was achieved through PyO3
[61].
On the Jackal, the Python code queried ROS top-
ics (RealSense D435 and velocity controller) while Rust
code managed communication; on the remote workstation,
Python code ran YOPO-Nav while Rust code again handled
communication. Persistent connectivity across campus was
maintained using a MiFi X Pro 5G hotspot on Calyx Insti-
tute’s unlimited mobile data plan.
Implementation Details
The visual place recognition
model used a DINOv3 (ViT-L) [72] backbone with the fi-
nal two layers unfrozen for fine-tuning.
A BoQ [3] ag-
gregator consisting of two layers (512 features, 128 learn-
able queries) produced 4096 descriptors from images that
were resized to 224×224 during training and validation. The
model was trained on GND [46] for 30 epochs with a batch
size of 64, learning rate of 1 × 10−4 (with a warmup fac-
tor of 0.1), and weight decay of 1 × 10−4. Frames from
each exploration video were processed in AnySplat [29] se-
quentially in batches of 55, to balance accuracy and qual-
ity within VRAM limits of the NVIDIA RTX 4090, at
448 × 336 resolution and saved as .ply files without spher-
ical harmonics; SOG compression reduced file sizes from
Method
SR@1.5 m ↑SR@3 m ↑SR@7 m ↑SR@12 m ↑
ViNT [71]
0.94
0.67
0.63
0.42
NoMaD [74]
0.83
0.53
0.53
0.20
YOPO-Nav
1.0
0.94
0.81
0.75
Table 3. Comparison of image-goal success rate at different trajec-
tory lengths across state-of-the-art navigation methods, with best
performing results in bold. YOPO-Nav significantly outperforms
zero-shot transformer-based models at both short and long trajec-
tories in real-world campus environments.
7

<!-- page 8 -->
Path Type
Length (m) Human Interventions Actions per Int. Robot Actions Time (min) GT Actions GT Time (min)
Straight Trajectory
222.6
5
17.6
2280
25.8
908
15.8
Varying Ground Type
113.6
7
18.7
1281
10.4
535
3.3
Construction Area
167.0
15
16.5
1968
21.1
787
4.5
Seasonal Changes
92.4
8
18
1245
13.4
446
4.9
Combined Trajectory/Downward Slope
87.2
3
14
1302
14.2
572
4.6
Table 4. Summary of collected metrics testing image-goal navigation using the YOPO-Nav algorithm with human interventions and
accumulated human actions at each intervention. Metrics include length of the exploration path, the number of human interventions for a
successful navigation, the mean number of controller micro-actions per intervention (Actions per Int.), the number of total robot actions
to reach the goal image, time to completion, the ground truth number of actions (during the teleoperation stage), and the ground truth
time. Notice the simple trajectories (e.g. Straight Trajectory) require fewer human interventions, while exploration paths with large scene
changes (e.g. Construction Area) require more human interventions.
150–250 MB to 25–40 MB in 15–30 seconds. Reconver-
sion to .ply was performed during inference. Pose estima-
tion was performed using OpenCV’s [10] PnP [26] with
RANSAC [22], configured to use SQPNP with 500 iter-
ations, a 5-pixel reprojection threshold, and a 0.99 confi-
dence threshold. Ground plane estimation was restricted
to the lower 30% of the renders at each known pose in the
3DGS, subsampled to 20,000 points, and filtered to the clos-
est fiftieth percentile of depths, with RANSAC using 500
iterations and an inlier threshold of 0.01 to fit the plane.
Results
Table 3 shows the relative success rates of
YOPO-Nav compared to ViNT [71] and NoMad [74] at
each distance threshold.
Notice that YOPO-Nav outper-
forms its competitors at short trajectories, and significantly
outperforms its competitors at long trajectories. It is un-
likely that a lack of appropriate training data is the cause
of this performance difference, since ViNT and NoMad in-
clude similar campus environments in their training data
such as SCAND [33] and Berkeley [68].
Differences in scene dynamics (e.g., seasonal variation,
lighting conditions, and activity such as pedestrian traffic)
can affect the performance of end-to-end solutions such as
ViNT and NoMad. Our method, however, is robust to these
variations and other dynamics that differ from the explo-
ration video, so long as they do not dominate the scene, and
it produces stronger success rate metrics under similar nav-
igation conditions.
4.2. Human Interventions
Motivation
Operating robots in real-world environments
is challenging. Intuitively, human-in-the-loop intervention
can help robots achieve their navigation goals. When such
intervention can occur without retraining the robot, interac-
tion becomes more seamless, enabling navigation to func-
tion as a true human–robot collaboration.
Human assis-
tance has generally proven effective in many recent works
[15, 17, 42, 64]. Building on this idea, we enable a com-
panion human to provide simple interventions via a con-
troller when the robot deviates from its intended path. Us-
ing this setup, we evaluated YOPO-Nav on five full-length
paths in YOPO-Campus to examine how well the algorithm
can scale while keeping human interventions to a minimum.
Experimental Setup
We used the same real-world de-
ployment and implementation details from earlier. For these
trials, we initialize YOPO-Nav with an unmodified explo-
ration video from YOPO-Campus, and place the Clearpath
Jackal robot in a similar starting point as in the exploration
video. Each trajectory was tested for one trial. A human-
in-the-loop monitored each trial and teleoperated the robot
with small discrete actions, effectively a nudge, if it went
off the sidewalk, was about to collide, or could not deter-
mine the next action. Interventions to prevent potential col-
lisions were overly cautious. We choose to test YOPO-Nav
using five trajectories that contain varying scenes, trajec-
tories, and challenges: (1) Straight Trajectory: a trajectory
between two buildings where the primary action to reach the
image goal is to move forward in a straight line. (2) Vary-
ing Ground Type: a trajectory with varying ground types
including brick and cement, as well as multiple turns. (3)
Construction Area: a trajectory whose scene changed sig-
nificantly due to construction in the area.
(4) Seasonal
Changes: a trajectory with seasonal foliage in the scene
which changed between the exploration video and the trial.
(5) Combined Trajectory/Downward Slope: two trajectories
stitched together into a longer video, with a sloping ground
plane present in a section.
Metrics
We collect four distinct metrics to evaluate the
effect of human interventions on the performance of YOPO-
Nav: (1) Human Interventions: the number of manual ac-
tions by a human necessary to successfully navigate to the
goal image. A human intervention is defined as any instance
in which the human-in-the-loop collaborator took control of
the mobile robot during one of the three failure cases pre-
viously discussed. (2) Actions per Intervention: the mean
number of controller action inputs by the human collabo-
rator per each intervention. Following the same discrete
actions as the YOPO-Campus dataset, each controller ac-
tion is limited to 0.25 m forward or backward, or 15◦left or
8

<!-- page 9 -->
right. (3) Robot Actions: the number of robot actions taken
to successfully navigate to the goal image. Human interven-
tion actions were not included in the robot actions metric.
(4) Time: the amount of time the robot took to successfully
navigate to the goal image in minutes.
These metrics are listed along with ground truth data of
the exploration path, including: (1) Length: the length of the
exploration path in meters. (2) Ground Truth (GT) Actions:
the number of controller input actions taken by the human
teleoperator in the exploration path. (3) GT Time (min): the
amount of time it took for the human teleoperator to com-
plete the exploration path.
Results
Table 4 shows the human intervention evaluation
metrics for each path type trial. The length of the path gen-
erally does not imply a higher number of human interven-
tions. This implies that YOPO-Nav is resilient to poten-
tial noise present in a trial that would be compounded over
long trajectories. The number of human interventions corre-
lates with the difficulty of the task, and is higher in cases of
scene changes between the exploration video and the trial
environment. YOPO-Nav is limited in its ability to adapt
to changes that overly dominate the scene, as shown in the
Construction Area case, since its scene representation is a
snapshot at a fixed point in time; as shown in the other cases,
however, YOPO-Nav remains largely effective when evalu-
ated on paths whose challenges do not involve relatively ex-
treme scene change. Despite this shortcoming, YOPO-Nav
can successfully complete image-goal navigation tasks over
long trajectories with a human-in-the loop collaborator.
5. Conclusion
We present a novel scene representation for visual naviga-
tion, composed of a graph of 3DGS that partition an en-
vironment into small, manageable chunks. We introduce
the YOPO-Nav framework using a 3DGS graph and a VPR
model to perform image-goal navigation. Our results show
that YOPO-Nav outperforms zero-shot foundation models
and performs successful visual navigation on long trajecto-
ries with limited human-in-the-loop intervention.
The YOPO-Campus dataset has a companion dataset
Ego-Campus [30], with egocentric video and eye-gaze from
pedestrians walking identical campus paths (shown in Fig. 1
wearing Project Aria glasses [21]. While YOPO-campus
shows the robot view, Ego-campus shows a pedestrian view
and captures eye-gaze. Together, these datasets provide an
important data resource for future studies of navigation in
human-robot systems.
Acknowledgments
This
work
was
supported
by
the
NSF-NRT
grant:
Socially
Cognizant
Robotics
for
a
Technology
En-
hanced Society
(SOCRATES), No.
2021628,
and
by
the
National
Science
Foundation
(NSF)
un-
der
Grant
Nos.
CNS-1901355,
CNS-1901133.
References
[1] Amar Ali-bey. Openvprlab. https://github.com/
amaralibey/OpenVPRLab, 2025. Accessed: 2025-11-
14. 5
[2] Amar Ali-Bey, Brahim Chaib-Draa, and Philippe Giguere.
Mixvpr: Feature mixing for visual place recognition. In Pro-
ceedings of the IEEE/CVF winter conference on applications
of computer vision, pages 2998–3007, 2023. 3
[3] Amar Ali-Bey, Brahim Chaib-draa, and Philippe Giguere.
Boq: A place is worth a bag of learnable queries. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 17794–17803, 2024. 5, 7
[4] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pa-
jdla, and Josef Sivic. Netvlad: Cnn architecture for weakly
supervised place recognition. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pages 5297–5307, 2016. 3
[5] AWS. s2n-quic. https://github.com/aws/s2n-
quic, 2025. Accessed: 2025-11-14. 7
[6] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and
Yann LeCun. Navigation world models. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 15791–15801, 2025. 2, 4
[7] Gabriele Berton and Carlo Masone. Megaloc: One retrieval
to place them all. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 2861–2867, 2025. 5
[8] Gabriele Berton, Gabriele Trivigno, Barbara Caputo, and
Carlo Masone.
Eigenplaces:
Training viewpoint robust
models for visual place recognition. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 11080–11090, 2023. 3
[9] Guillaume Bono,
Leonid Antsfeld,
Boris Chidlovskii,
Philippe Weinzaepfel, and Christian Wolf.
End-to-end
(instance)-image goal navigation through correspondence as
an emergent phenomenon. arXiv preprint arXiv:2309.16634,
2023. 4
[10] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of
Software Tools, 2000. 8
[11] Matthew Chang, Arjun Gupta, and Saurabh Gupta. Seman-
tic visual navigation by watching youtube videos. Advances
in Neural Information Processing Systems, 33:4283–4294,
2020. 1, 2
[12] Devendra Singh Chaplot, Ruslan Salakhutdinov, Abhinav
Gupta, and Saurabh Gupta. Neural topological slam for vi-
sual navigation. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 12875–
12884, 2020. 4
[13] Peihao Chen, Dongyu Ji, Kunyang Lin, Weiwen Hu, Wen-
bing Huang, Thomas Li, Mingkui Tan, and Chuang Gan.
Learning active camera for multi-object navigation.
Ad-
vances in Neural Information Processing Systems, 35:
28670–28682, 2022. 4
9

<!-- page 10 -->
[14] Timothy Chen, Ola Shorinwa, Joseph Bruno, Aiden Swann,
Javier Yu, Weijia Zeng, Keiko Nagami, Philip Dames, and
Mac Schwager. Splat-nav: Safe real-time robot navigation
in gaussian splatting maps. IEEE Transactions on Robotics,
2025. 3
[15] Valerie Chen, Abhinav Gupta, and Kenneth Marino.
Ask
your humans: Using human instructions to improve gen-
eralization in reinforcement learning.
arXiv preprint
arXiv:2011.00517, 2020. 8
[16] Elizabeth R Chrastil and William H Warren. From cognitive
maps to cognitive graphs. PloS one, 9(11):e112544, 2014. 1
[17] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic,
Shane Legg, and Dario Amodei. Deep reinforcement learn-
ing from human preferences. Advances in neural information
processing systems, 30, 2017. 8
[18] Owen Claxton, Connor Malone, Helen Carson, Jason J Ford,
Gabe Bolton, Iman Shames, and Michael Milford. Improv-
ing visual place recognition based robot navigation by veri-
fying localization estimates. IEEE Robotics and Automation
Letters, 2024. 3
[19] Alessandro Devo, Giacomo Mezzetti, Gabriele Costante,
Mario L Fravolini, and Paolo Valigi. Towards generaliza-
tion in target-driven visual navigation by using deep rein-
forcement learning. IEEE Transactions on Robotics, 36(5):
1546–1561, 2020. 4
[20] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria
Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library.
IEEE Transactions on Big Data, 2025. 5, 6
[21] Jakob Engel, Kiran Somasundaram, Michael Goesele, Albert
Sun, Alexander Gamino, Andrew Turner, Arjang Talattof,
Arnie Yuan, Bilal Souti, Brighid Meredith, et al. Project aria:
A new tool for egocentric multi-modal ai research. arXiv
preprint arXiv:2308.13561, 2023. 9
[22] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 6, 8
[23] Sourav Garg, Tobias Fischer, and Michael Milford. Where
is your place, visual place recognition?
arXiv preprint
arXiv:2103.06443, 2021. 3
[24] Wenxuan Guo, Xiuwei Xu, Hang Yin, Ziwei Wang, Jianjiang
Feng, Jie Zhou, and Jiwen Lu. Igl-nav: Incremental 3d gaus-
sian localization for image-goal navigation. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 6808–6817, 2025. 3
[25] Saurabh Gupta, James Davidson, Sergey Levine, Rahul Suk-
thankar, and Jitendra Malik. Cognitive mapping and plan-
ning for visual navigation. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
2616–2625, 2017. 1
[26] Richard Hartley and Andrew Zisserman. Multiple view ge-
ometry in computer vision.
Cambridge university press,
2003. 6, 8
[27] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han,
Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat:
Pose-free feed-forward 3d gaussian splatting for novel view
synthesis. In Proceedings of the International Conference on
Machine Learning (ICML), 2025. 2
[28] Sergio Izquierdo and Javier Civera. Optimal transport ag-
gregation for visual place recognition. In Proceedings of the
ieee/cvf conference on computer vision and pattern recogni-
tion, pages 17658–17668, 2024. 3, 5
[29] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 2, 5, 7
[30] Ronan John, Aditya Kesari, Vincenzo DiMatteo, and Kristin
Dana. Egocampus: Egocentric pedestrian eye gaze model
and dataset, 2025. 9
[31] Faith Johnson, Bryan Bo Cao, Ashwin Ashok, Shubham
Jain, and Kristin Dana. Feudal networks for visual naviga-
tion. arXiv preprint arXiv:2402.12498, 2024. 3
[32] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam,
Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung
Park. Selfsplat: Pose-free and 3d prior-free generalizable 3d
gaussian splatting. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 22012–22022,
2025. 2
[33] Haresh Karnan, Anirudh Nair, Xuesu Xiao, Garrett War-
nell, S¨oren Pirk, Alexander Toshev, Justin Hart, Joydeep
Biswas, and Peter Stone.
Socially compliant navigation
dataset (scand): A large-scale dataset of demonstrations for
social navigation. IEEE Robotics and Automation Letters, 7
(4):11807–11814, 2022. 8
[34] Nikhil
Keetha,
Avneesh
Mishra,
Jay
Karhade,
Kr-
ishna Murthy Jatavallabhula, Sebastian Scherer, Madhava
Krishna, and Sourav Garg. Anyloc: Towards universal visual
place recognition. IEEE Robotics and Automation Letters, 9
(2):1286–1293, 2023. 3
[35] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi,
Yuchen Zhang,
Tobias Fischer,
Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
et al. Mapanything: Universal feed-forward metric 3d re-
construction. arXiv preprint arXiv:2509.13414, 2025. 5
[36] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[37] Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho
Park, and Songhwai Oh. Topological semantic graph mem-
ory for image-goal navigation.
In Conference on Robot
Learning, pages 393–402. PMLR, 2023. 3
[38] Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge,
and Peter Anderson. Pathdreamer: A world model for indoor
navigation. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 14738–14748, 2021.
2, 4
[39] Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, and
Devendra Singh Chaplot. Instance-specific image goal nav-
igation: Training embodied agents to find object instances.
arXiv preprint arXiv:2211.15876, 2022. 4
10

<!-- page 11 -->
[40] Obin Kwon, Nuri Kim, Yunho Choi, Hwiyeon Yoo, Jeongho
Park, and Songhwai Oh. Visual graph memory with unsuper-
vised representation for visual navigation. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 15890–15899, 2021. 4
[41] Obin Kwon, Jeongho Park, and Songhwai Oh. Renderable
neural radiance map for visual navigation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9099–9108, 2023. 4
[42] Kimin Lee, Laura Smith, and Pieter Abbeel.
Pebble:
Feedback-efficient interactive reinforcement learning via re-
labeling experience and unsupervised pre-training.
arXiv
preprint arXiv:2106.05091, 2021. 8
[43] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li.
Gaussnav: Gaussian splatting for visual navigation. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
2025. 3
[44] Hongxin Li, Zeyu Wang, Xu Yang, Yuran Yang, Shuqi Mei,
and Zhaoxiang Zhang. Memonav: Working memory model
for visual navigation. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
17913–17922, 2024. 3
[45] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Se-
mantic gaussian splatting for neural dense slam. In European
Conference on Computer Vision, pages 163–179. Springer,
2024. 3
[46] Jing Liang, Dibyendu Das, Daeun Song, Md Nahid Hasan
Shuvo, Mohammad Durrani, Karthik Taranath, Ivan Penskiy,
Dinesh Manocha, and Xuesu Xiao.
Gnd: Global naviga-
tion dataset with multi-modal perception and multi-category
traversability in outdoor campus environments.
In 2025
IEEE International Conference on Robotics and Automation
(ICRA), pages 2383–2390. IEEE, 2025. 3, 4, 5, 7
[47] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen,
Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed
3d gaussian splatting for casual long videos. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 27412–27422, 2025. 2, 5
[48] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. Lightglue: Local feature matching at light speed. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 17627–17638, 2023. 6
[49] Rui Liu, Xiaohan Wang, Wenguan Wang, and Yi Yang.
Bird’s-eye-view scene graph for vision-language navigation.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 10968–10980, 2023. 3
[50] Xinhao Liu, Jintong Li, Yicheng Jiang, Niranjan Sujay,
Zhicheng Yang, Juexiao Zhang, John Abanes, Jing Zhang,
and Chen Feng.
Citywalker:
Learning embodied ur-
ban navigation from web-scale videos.
arXiv preprint
arXiv:2411.17820, 2024. 2, 7
[51] Kenzo Lobos-Tsunekawa, Francisco Leiva, and Javier Ruiz-
del Solar. Visual navigation for biped humanoid robots using
deep reinforcement learning. IEEE Robotics and Automation
Letters, 3(4):3247–3254, 2018. 4
[52] Joel Loo, Zhanxin Wu, and David Hsu. Open scene graphs
for open-world object-goal navigation.
The International
Journal of Robotics Research, page 02783649251369549,
2025. 3
[53] Yunlian Lyu, Yimin Shi, and Xianggang Zhang. Improving
target-driven visual navigation with attention on 3d spatial
relationships. Neural Processing Letters, 54(5):3979–3998,
2022. 4
[54] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 18039–18048, 2024. 2, 3
[55] Bar Mayo, Tamir Hazan, and Ayellet Tal. Visual navigation
with spatial attention. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
16898–16907, 2021. 4
[56] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
3
[57] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and
Peter Eisert.
Compact 3d scene representation via self-
organizing gaussian grids. In ECCV, 2024. 5
[58] Michael Peer, Iva K Brunec, Nora S Newcombe, and Rus-
sell A Epstein. Structuring knowledge with cognitive maps
and cognitive graphs. Trends in cognitive sciences, 25(1):
37–54, 2021. 1
[59] PlayCanvas. Splattransform. https://github.com/
playcanvas/splat-transform, 2025.
Accessed:
2025-11-14. 5
[60] Guilherme Potje, Felipe Cadar, Andr´e Araujo, Renato Mar-
tins, and Erickson R Nascimento. Xfeat: Accelerated fea-
tures for lightweight image matching.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2682–2691, 2024. 6
[61] PyO3.
Pyo3.
https://github.com/PyO3/pyo3,
2025. Accessed: 2025-11-14. 7
[62] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos,
2024. 6
[63] Hao Ren, Yiming Zeng, Zetong Bi, Zhaoliang Wan, Junlong
Huang, and Hui Cheng. Prior does matter: Visual navigation
via denoising diffusion bridge models. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
12100–12110, 2025. 4
[64] S Saunders, G Sastry, A Stuhlm¨uller, and O Evans. Trial
without error: Towards safe reinforcement learning via hu-
man intervention. In 17th International Conference on Au-
tonomous Agents and MultiAgent Systems. ACM Digital Li-
brary, 2018. 8
[65] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets,
Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia
Liu, Vladlen Koltun, Jitendra Malik, et al.
Habitat: A
platform for embodied ai research.
In Proceedings of
11

<!-- page 12 -->
the IEEE/CVF international conference on computer vision,
pages 9339–9347, 2019. 4
[66] Karl Schmeckpeper, Oleh Rybkin, Kostas Daniilidis, Sergey
Levine, and Chelsea Finn.
Reinforcement learning with
videos:
Combining offline observations with interaction.
arXiv preprint arXiv:2011.06507, 2020. 2
[67] Zachary Seymour, Kowshik Thopalli, Niluthpol Mithun,
Han-Pang Chiu, Supun Samarasekera, and Rakesh Kumar.
Maast: Map attention with semantic transformers for effi-
cient visual navigation.
In 2021 IEEE international con-
ference on robotics and automation (ICRA), pages 13223–
13230. IEEE, 2021. 4
[68] Dhruv Shah and Sergey Levine.
Viking:
Vision-based
kilometer-scale navigation with geographic hints.
arXiv
preprint arXiv:2202.11271, 2022. 8
[69] Dhruv Shah, Benjamin Eysenbach, Gregory Kahn, Nicholas
Rhinehart, and Sergey Levine.
Ving:
Learning open-
world navigation with visual goals.
In 2021 IEEE Inter-
national Conference on Robotics and Automation (ICRA),
pages 13215–13222. IEEE, 2021. 1
[70] Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose,
and Sergey Levine. Gnm: A general navigation model to
drive any robot. arXiv preprint arXiv:2210.03370, 2022. 4
[71] Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachow-
icz, Kevin Black, Noriaki Hirose, and Sergey Levine. Vint:
A foundation model for visual navigation. arXiv preprint
arXiv:2306.14846, 2023. 1, 4, 7, 8
[72] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa,
et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 5,
7
[73] Kunal Pratap Singh, Jordi Salvador, Luca Weihs, and
Aniruddha Kembhavi. Scene graph contrastive learning for
embodied navigation. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages 10884–
10894, 2023. 3
[74] Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey
Levine. Nomad: Goal masked diffusion policies for nav-
igation and exploration. In 2024 IEEE International Con-
ference on Robotics and Automation (ICRA), pages 63–70.
IEEE, 2024. 1, 4, 7, 8
[75] Jingwen Sun, Jing Wu, Ze Ji, and Yu-Kun Lai. A survey of
object goal navigation. IEEE Transactions on Automation
Science and Engineering, 22:2292–2308, 2024. 4
[76] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstr¨om,
Stefano Mattoccia, Martin R Oswald, and Matteo Poggi.
How nerfs and 3d gaussian splatting are reshaping slam: a
survey. arXiv preprint arXiv:2402.13255, 4:1, 2024. 2, 3
[77] Changwei Wang, Shunpeng Chen, Yukun Song, Rongtao
Xu, Zherui Zhang, Jiguang Zhang, Haoran Yang, Yu Zhang,
Kexue Fu, Shide Du, et al. Focus on local: Finding reliable
discriminative regions for visual place recognition. In Pro-
ceedings of the AAAI Conference on Artificial Intelligence,
pages 7536–7544, 2025. 5
[78] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 5
[79] Saim Wani, Shivansh Patel, Unnat Jain, Angel Chang, and
Manolis Savva.
Multion:
Benchmarking semantic map
memory using multi-object navigation. Advances in Neural
Information Processing Systems, 33:9700–9712, 2020. 4
[80] Steven M Weisberg and Nora S Newcombe. How do (some)
people make a cognitive map? routes, places, and working
memory. Journal of Experimental Psychology: Learning,
Memory, and Cognition, 42(5):768, 2016. 1
[81] Abdelrhman Werby, Chenguang Huang, Martin B¨uchner,
Abhinav Valada, and Wolfram Burgard. Hierarchical open-
vocabulary 3d scene graphs for language-grounded robot
navigation. In First Workshop on Vision-Language Models
for Navigation and Manipulation at ICRA 2024, 2024. 3
[82] Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra
Malik, and Silvio Savarese. Gibson env: Real-world percep-
tion for embodied agents. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
9068–9079, 2018. 4
[83] Ziyang Xie, Zhizheng Liu, Zhenghao Peng, Wayne Wu, and
Bolei Zhou. Vid2sim: Realistic and interactive simulation
from video for urban navigation.
In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
1581–1591, 2025. 2
[84] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 2, 3
[85] Sixian Zhang, Xinyao Yu, Xinhang Song, Xiaohan Wang,
and Shuqiang Jiang.
Imagine before go: Self-supervised
generative map for object goal navigation. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 16414–16425, 2024. 2, 4
[86] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
don Wetzstein. Flare: Feed-forward geometry, appearance
and camera estimation from uncalibrated sparse views. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 21936–21947, 2025. 2
[87] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Ab-
hinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven vi-
sual navigation in indoor scenes using deep reinforcement
learning. In 2017 IEEE international conference on robotics
and automation (ICRA), pages 3357–3364. IEEE, 2017. 4
12
