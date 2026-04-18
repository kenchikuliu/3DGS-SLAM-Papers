<!-- page 1 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
1
VPGS-SLAM: Voxel-based Progressive 3D Gaussian
SLAM in Large-Scale Scenes
Tianchen Deng, Wenhua Wu, Junjie He, Yue Pan, Shenghai Yuan,
Danwei Wang, Life Fellow, IEEE, Hesheng Wang Senior Member, IEEE
Abstract—3D Gaussian Splatting has recently shown promising
results in dense visual SLAM. However, existing 3DGS-based
SLAM methods are all constrained to small-room scenarios and
struggle with memory explosion in large-scale urban scenes and
long sequences. To this end, we propose VPGS-SLAM, a novel
3DGS-based large-scale RGBD SLAM framework for both indoor
and outdoor scenarios. We design a novel voxel-based progressive
3D Gaussian mapping method with multiple submaps for compact
and accurate scene representation in large-scale and long-sequence
scenes. This allows us to scale up to arbitrary scenes and improves
robustness (even under pose drifts). In addition, we propose a 2D-
3D fusion camera tracking method to achieve robust and accurate
camera tracking in both indoor and outdoor large-scale scenes.
Furthermore, we design a 2D-3D Gaussian loop closure method to
eliminate pose drift. We further propose a submap fusion method
with online distillation to achieve global consistency in large-scale
scenes when detecting a loop. Experiments on various indoor and
outdoor datasets demonstrate the superiority and generalizability
of the proposed framework. The code will be open-sourced on
https://github.com/dtc111111/vpgs-slam.
Index Terms—Dense SLAM, Urban Scene Reconstruction, 3D
Gaussian Splatting.
I. INTRODUCTION
Visual Simultaneous localization and mapping (SLAM) [1],
[2] has been a fundamental problem with wide applications such
as autonomous driving [3], robotics [4], and remote sensing.
In intelligent transportation systems, accurate pose estimation
constitutes a fundamental prerequisite for safe and reliable
autonomous driving. Errors in vehicle pose estimation can
directly compromise motion planning and control, leading
to hazardous behaviors such as premature or unnecessary
braking, unsafe following distances, and incorrect obstacle
avoidance decisions. More critically, the consequences of pose
misestimation extend beyond the affected vehicle itself. In real-
world traffic environments, abnormal driving actions triggered
by inaccurate pose estimates can influence surrounding vehicles
through interaction and response, potentially propagating as
Tianchen Deng, Wenhua Wu, Hesheng Wang are with the School of
Automation and Intelligent Sensing, Shanghai Jiao Tong University, and
Key Laboratory of System Control and Information Processing, Ministry of
Education, Shanghai 200240, China. Junjie He is at the Thrust of Robotics and
Autonomous Systems, The Hong Kong University of Science and Technology
(Guangzhou). Yue Pan is with the University of Bonn. Danwei Wang and
Shenghai Yuan are with the School of Electrical and Electronic Engineering,
Nanyang Technological University, Singapore. This research is supported by
the National Research Foundation, Singapore, under the NRF Medium Sized
Centre scheme (CARTIN), Maritime and Port Authority of Singapore under
its Maritime Transformation Programme (Project No. SMI-2022-MTP-04),
ASTAR under National Robotics Programme with Grant No.M22NBK0109.
The first two authors contibute equal to this paper. (*corresponding author:
wanghesheng@sjtu.edu.cn)
cascading unsafe maneuvers, large-scale braking waves, and
traffic flow instability. Such error amplification mechanisms
may significantly elevate the risk of traffic accidents and
degrade overall transportation safety, underscoring the essential
role of robust and accurate pose estimation in autonomous
driving systems.
Several traditional methods [5]–[10] have been introduced
over the years. They use handcraft descriptors for image match-
ing and represent scenes using sparse feature point maps. Due to
the sparse nature of such point cloud, it is difficult for humans
to understand how machines interact with the scene, and these
methods cannot meet the demands of collision avoidance
and motion planning. Attention then turns to dense scene
reconstruction, exemplified by DTAM [11], Kintinuous [12],
and ElasticFusion [13]. However, their performance is limited
by high memory consumption, slow processing speeds.
Following the introduction of Neural Radiance Fields
(NeRF), numerous research efforts have focused on com-
bining implicit scene representation with SLAM systems
and autonomous driving [14]. iMAP [15] pioneered the use
of a single MLP to represent the scene, while [16]–[19]
have further enhanced scene representation through hybrid
feature grids, axis-aligned feature planes, joint coordinate-
parametric encoding, and multiple implicit submaps. NeS-
LAM [20] use a depth completion and denoising method to
improve scene representation. To further improve rendering
speed, recent methods have started to explore 3D Gaussian
Splatting (3DGS) [21] in SLAM systems, as demonstrated by
[22], [23]. GS-based SLAM methods leverage a point-based
representation associated with 3D Gaussian attributes and adopt
the rasterization pipeline to render the images, achieving fast
rendering speed and promising image quality.
However, existing SLAM systems primarily focus on small-
scale indoor environments; they face significant challenges in
representing large-scale scenes (e.g., multi-room apartments
and urban scenes). Some We outline the key challenges for
indoor and outdoor large-scale 3DGS-based SLAM systems: a)
Redundant 3D Gaussian ellipsoids: existing methods employ
a substantial number of 3D Gaussian ellipsoids to represent
the scene. These methods typically require more than 500MB
to represent a small room-scale scene, which severely limits
their applicability and results in memory overload in large-
scale environments. b) Accumulation of errors and pose drift:
Existing works rely on rendered loss for camera tracking,
which proves inaccurate and unreliable in large-scale real-
world environments with dramatic movement, motion blur and
exposure change. c) Global inconsistency: In long sequences
arXiv:2505.18992v2  [cs.CV]  10 Jan 2026

<!-- page 2 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
2
Rendered RGB
Rendered Depth
Rendered RGB
Rendered Depth
Fig. 1: We present VPGS-SLAM, a novel large-scale SLAM framework with voxel-based progressive 3D Gaussian representation,
2D-3D assisted camera tracking and 3D Gaussian loop closure. Depicted in the middle, we demonstrate the large-scale globally
consistent 3D Gaussian map built with our approach. At the top and bottom of the figure, we include zoomed-in views of the
map with RGB and depth images rendered by our method, indicated by dashed blue and yellow boxes.
and large-scale scenes, when the robot revisits the same location,
it is essential to ensure spatial correlations, and long-term
memory for global consistency.
To this end, we propose VPGS-SLAM, a novel 3DGS-based
large-scale SLAM framework with voxel-based progressive
3D Gaussian representation, 2D-3D fusion camera tracking,
and 2D-3D Gaussian based loop detection and correction, and
submap fusion. We use a collection of submaps to represent
the entire scene, which dynamically initializes local scene
representation when the camera moves to the bounds of the
local submaps. The entire scene is divided into multiple
local submaps, which can significantly improve the scene
representation capacity of large-scale scenes. The submap
parameters do not need to be retained in memory after
optimization, which can significantly reduce online memory
requirements and enhances the scalability of the framework.
In local scene representation, a sequentialized voxel-based
3D Gaussian representation is proposed tailored for online
SLAM framework. We design an efficient hybrid data structure
that combines the multi-resolution voxel representation with 3D
Gaussian ellipsoids. The local scene is sequentially initialized
with sparse voxels. When a keyframe arrives, we initialize
voxels in the regions observed by this frame. Each voxel is
assigned a corresponding anchor point. Each anchor spawns a
set of neural Gaussians with learnable offsets, where attributes
such as opacity, color, rotation, and scale are predicted based
on the feature of the anchor point and the viewing position.
We leverage scene geometry structure to guide and constrain
the distribution of 3D Gaussian ellipsoids. We also design a
novel sequentially growing and pruning strategy for voxels and
anchor points.
For camera tracking, we propose a 2D-3D fusion tracking
method to combine the 3D Gaussian ellipsoids geometric
information with 2D photometric information. We optimize the
pose with 2D photometric loss using RGB and depth loss in the
coarse-level optimization stage. We then incorporate a 3D voxel
based ICP method to perform frame-to-map pose estimation in
the fine-level stage. We provide a good initial estimate for 3D
point matching through the 2D rendering loss, enabling further
accurate pose estimation. To handle both indoor and outdoor
environments, we propose an adaptive 2D information assessing
and parameter selection strategy. By assessing the richness of
2D information in the scene, our method will choose different
key parameters in the tracking and loop closure modules,
enabling robust performance across diverse environments.
Moreover, to alleviate the cumulative pose drift in large-scale
scenes, we propose a 2D-3D Gaussian based loop closure
method with loop detection, pose graph optimization with voxel

<!-- page 3 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
3
Fig. 2: System Overview. Our system is a large-scale SLAM framework, with voxel-based progressive 3D Gaussian representation,
2D-3D fusion camera tracking, and 3D Gaussian loop closure. Our framework takes color images and 3D point clouds as input.
Our method can achieve accurate and efficient scene reconstruction, camera tracking, and global map generation.
ICP and rendering loss. We also design an online distillation
method for submap fusion that is triggered upon loop closure
detection. This method merges the multiple submaps involved
in the loop, thereby enhancing global consistency of the global
map. Overall, our contributions are shown as follows:
• We propose VPGS-SLAM, a novel 3DGS-based large-
scale RGBD SLAM framework that enables efficient and
accurate scene reconstruction and pose estimation in both
indoor and city-scale environments.
• A novel sequentialized voxel-based progressive 3D Gaus-
sian representation is proposed for compact and efficient
scene representation in large-scale scenes. We design
multiple local submaps with multi-resolution voxel repre-
sentation to achieve efficient and accurate reconstruction
in large-scale scenes.
• A novel 2D-3D fusion tracking method is proposed to
combine the 3D Gaussian geometric information with 2D
photometric information for accurate pose estimation. We
also propose a 2D-3D Gaussian loop closure method with
loop detection, and pose graph optimization. A submap
fusion method with online distillation is proposed to
achieve global consistency. Experiments on various indoor
and outdoor datasets demonstrate the superiority of the
proposed method in both mapping and tracking.
II. RELATED WORK
Traditional SLAM. SLAM has been an active research
field for the past two decades. Traditional visual SLAM
algorithms [24] estimate accurate camera poses and represent
the scene using sparse point clouds. [25] utilizes tightly-coupled
LiDAR-Visual-Inertial odometry with multi-modal semantic
information to enhance the robustness and accuracy of SLAM.
DTAM [11] was the first RGB-D approach to achieve dense
scene reconstruction. Some learning-based methods [26], [27],
integrate traditional geometric frameworks with deep learning
networks for improved camera tracking and mapping.For 3D
LiDAR odometry and mapping, similar to feature-matching
methods widely used in visual SLAM, the seminal work
LOAM [28] proposes extracting sparse planar or edge feature
points from the scan point cloud and registering them to the
previous frame or the feature point map using ICP. Recently,
CT-ICP [29] and KISS-ICP [30] have achieved robust LiDAR
odometry performance without the need for feature point
extraction.
NeRF-based SLAM. With the introduction of NeRF [31],
iMAP [15] pioneered the use of a single multi-layer percep-
tron (MLP) to represent the scene, while NICE-SLAM [16]
introduced learnable hierarchical feature grids. ESLAM [17]
and Co-SLAM [18] further enhanced scene representation
using tri-planes and joint coordinate-parametric encoding.
Some methods [19], [32], [33] proposed a novel progressive
scene representation that dynamically allocates new local
representations. Go-SLAM [34], Loopy-SLAM [35] use loop
closure to enhance the camera tracking performance. SNI-
SLAM [36] leverages semantic information. [37]–[40] use
neural point-based neural radiance fields for large-scale scenes
and high-accuracy reconstruction. DDN-SLAM [41] focus
on dynamic scene representation with masks. Unlike these
methods, which use neural implicit features, our approach
adopts an explicit 3D Gaussian representation, significantly
improving the scalability of our method.
GS-based SLAM. Recently, 3D Gaussian Splatting [21] has
emerged using 3D Gaussians as primitives for real-time neural
rendering. SplaTAM [22], MonoGS [23], Gaussian-SLAM [42],
and other works [43]–[51] are the pioneer works that success-
fully combine the advantages of 3D Gaussian Splatting with
SLAM. These methods achieve fast rendering speed and high-

<!-- page 4 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
4
fidelity reconstruction performance. GigaSLAM [52] is the
concurrent work of our method which use depth estimation
method for dense RGB SLAM in large scenes. However, the
memory and storage usage are intensive in these GS-based
SLAM systems, which makes them difficult to use in large-
scale and long-sequence scenarios.
III. METHOD
The pipeline of our system is shown in Fig. 2. The inputs
of this framework are RGB frames and 3D points {Ii, Pi}M
i=1
with known camera intrinsics K ∈R3×3. Our model predicts
camera poses {Ri|ti}M
i=1, color c, and a structured 3D Gaussian
scene representation. The system consists of three main
modules: (i) Voxel-based progressive scene representation
(Sec. III-A), (ii) 2D-3D fusion camera tracking (Sec. III-B),
(iii) 3D Gaussian-based loop closure (Sec. III-C). The network
is incrementally updated with the system operation.
A. Voxel-based Progressive Scene Representation
Recently, existing methods only focus on small room
scenarios and have difficulties in large-scale scenes due to
their redundant representation and the cumulative growth in
the number of ellipsoids. To address this issue, we design a
novel sequentialized voxel-based progressive 3D Gaussian with
multiple submaps for compact and efficient scene representation.
We utilize the scene structure prior to guide the distribution
of Gaussians to remove unnecessary 3D Gaussian ellipsoids,
maintaining a low compute cost while avoiding unrestricted
growth as the scene expands.
Sequentialized Multi-Resolution Voxel-based Scene Rep-
resentation Although there are some voxel-based mapping
methods, such as [53], the existing method is not well-suited to
the incremental nature of SLAM systems. To better accommo-
date this characteristic, we further reformulated our approach
into a sequentialized multi-resolution voxel-based mapping
framework. When a keyframe arrives, the regions observed by
this frame is voxelized with the point cloud P i ∈RN×3, i
denotes the ID of the current frame. We use Vi ∈RN ′×3 to
denote voxel centers. The center of each voxel is initialized
as an anchor point xa
i ∈R3. Each anchor is characterized by
its attributes Ai =

f a
i ∈R32, li ∈R3, Oi ∈Rk×3	
, where
each component represents the anchor feature, scaling, and
offsets, respectively. Then, we derive 3D Gaussians attributes
from anchor points. The attributes of a neural Gaussian are
defined as: position µi ∈R3, opacity αi ∈R, quaternion
qi ∈R4, scaling si ∈R3, and color ci ∈R3. The positions
of the corresponding k 3D Gaussians of current frame Ii are
calculated as:
{µm
i }k−1
m=0 = xa
i + {Om
i }k−1
m=0 · li
(1)
where {Om
i }k−1
m=0 ∈Rk×3 are the learnable offsets and li is
the scaling factor associated with anchor. Then, the attributes
of the Gaussians are decoded from the anchor feature f a, the
viewing distance δi, directiondi through individual MLPs:
{f a
i , δi, di} 7→{{αm
i }k−1
m=0, {qm
i }k−1
m=0, {sm
i }k−1
m=0, {cm
i }k−1
m=0}
(2)
δi = ∥xa
i −xc
i∥2 ,
di =
xa
i −xc
i
∥xa
i −xc
i∥2
(3)
where xc
i denotes camera position of current frame Ii. The
core MLPs include the opacity MLP , the color MLP and the
covariance MLP. All of these F* are implemented in a LINEAR
7→RELU 7→LINEAR style with the hidden dimension of 32.
Each branch’s output is activated with a head layer. For opacity,
the output is activated by Tanh, where value 0 serves as a natural
threshold for selecting valid samples and the final valid values
can cover the full range of [0,1). For color, we activate the
output with Sigmoid function:
{c0, . . . , ck−1} = Sigmoid (Fc)
(4)
which canstrains the color into a range of (0,1). For rotation,
we follow 3D-GS and activate it with a normalization to obtain
a valid quaternion. For scaling, we adjust the base scaling of
each anchor with the MLP output as follows:
{s0, . . . , sk−1} = Sigmoid (Fs) · sv
(5)
In order to improve the efficiency, we use multiple levels
voxel size based on camera distance, from fine to coarse.
All attributes are decoded in a single pass. We achieve
online reconstruction through the sequentialized voxel map-
ping method, where new voxels are continuously initialized,
registered, aligned, and iteratively updated, ultimately enabling
the reconstruction of the entire sequence.
Progressive Voxel-based Scene Representation In order to
improve the scalability of scene representation in large-scale
scenes, we propose a progressive mapping method that uses
multiple submaps to represent the scene to avoid the cumulative
growth of 3D Gaussian ellipsoids, similar with. Each submap
covers several keyframes that observe it and is represented by
a voxelized scene representation.
{Ii, Pi}M
i=1 7→{M1
θ1, M2
θ2, . . . , Mn
θn} 7→{c, α}
(6)
where Mn
θn denotes the n-th local submap. Starting with
the first keyframe, each submap models a specific region.
Whenever the estimated camera pose trajectory leaves the space
of the submap, we dynamically allocate a new local scene
representation trained with a small set of frames. Subsequently,
we progressively introduce additional local frames to the
optimization. We use the distance threshold d and the rotation
threshold ω to trigger the initialization of a new submap.
Voxel-Based Submap Expansion and Activation Keyframes
are selected at fixed intervals for the submap, and we define
the first keyframe of the submap as the anchor frame. Every
new keyframe adds new anchor points and 3D Gaussians to
the active submap for the newly observed parts of the scene.
At the beginning of each submap, we first compute a posed
point cloud from the input, and then sample Mk points from
the regions where the accumulated α is below a threshold or
where significant depth discrepancies occur. These points are
voxelized and initialized as anchor points. New 3D Gaussian
anchors are added to the current submap only if there is no
existing 3D Gaussian mean within a radius ρ. Then, for each
voxel, we compute the averaged gradients of the included neural
Gaussians when we move to the bound of the submap, denoted

<!-- page 5 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
5
Fig. 3: Rendering Performance on Replica dataset [54] compared with SplaTAM [22] and Loop-splat [45].
∇g. If the ∇g > τg, we grow new anchor points in these
voxels. To remove trivial anchors, we accumulate the opacity
values of their associated neural Gaussians in the submap. To
avoid redundant submap creation, we assign the current frame
to the most relevant submap by measuring its distance to the
anchor frames of all existing submaps.
Online Submap Optimization In the mapping thread, we
optimize the scene representation with the rendering loss. The
rendering loss consists of four components:
Lm = Lc + LSSIM + λdLd + λvolLvol
(7)
where Lc and LSSIM are the color losses, and Ld is the
depth loss computed using L1 distance. The SSIM color loss
and regularization loss are defined as:
LSSIM = (1 −λSSIM)·|ˆI−I|1+λSSIM(1−SSIM(ˆI, I)) (8)
Lvol =
Nng
X
i=1
Prod (si)
(9)
where Nng denotes the number of neural Gaussians in the
submap. Prod(·) denotes the product of the values of a
vector. The volume regularization term promotes small neural
Gaussians with minimal overlap. We optimize the parameters
of the two relevant submaps only in the overlapping regions. In
other areas, only a single submap’s parameters are maintained,
while non-essential submaps can be deactivated. Notably, the
number of submap parameters and Gaussian ellipsoids is
O(N). Thus, our method significantly reduces online memory
consumption, which is crucial for SLAM systems, as existing
approaches often encounter GPU out-of-memory issues in
large-scale environments.
B. 2D-3D Fusion Camera Tracking
Most existing 3DGS methods rely on single-modality super-
vision, typically using rendering loss. However, our findings
indicate that 2D photometric loss performs better in indoor and
simulated environments, whereas 3D geometric information
are more beneficial for outdoor scene reconstruction with
motion blur and exposure change. To this end, we design
a coarse-to-fine pose estimation, utilizing both 3D Gaussian
information and 2D rendering information for both indoor
and large-scale outdoor scenes. In coarse-level optimization,
we use 2D photometric information Lc, Ld to optimize the
camera pose, which provides a good initial estimate. In the
fine-level stage, we leverage 3D geometric information to
further refine the pose. However, directly applying 2D-3D
optimization is not always suitable for diverse indoor and
outdoor environments. To address this, we design a dynamic
adaptive method to dynamically adjust the reliance on 2D/3D
information during tracking. We assess the quality of rendering
results with rendering loss Lc, Ld and input frames Ii. If the
2D information quality lower than the threshold ζ, we will
discard the coarse-level optimization and directly employ 3D
Gaussian-based voxel ICP for pose refinement. This strategy
significantly improves robustness in large-scale outdoor scenes,
where visual inputs may suffer from illumination changes,
exposure, and motion blur, enabling the system to rely more
heavily on 3D geometric information when necessary.
In the fine-level optimization, we incorporate a voxel-based
3D Gaussian ICP method, inspired by [30]. We perform frame-
to-map registration with the voxel-based 3D Gaussian map
and adopt a double downsampling strategy, which retains
only a single original point per voxel. Compared to most
voxelization strategies that select the center of each voxel
for downsampling, we find it more advantageous to retain
the original point coordinates during the mapping process,
which can avoid discretization errors. For the input points S,
we transform it into the global coordinate frame using the
previous pose estimate {Ri−1, ti−1} and predicted relative
pose{Rpred, tpred}, resulting in the source points:
S = {si = {Ri−1, ti−1}{Rpred, tpred}p | p ∈P } .
(10)
where P denotes the scan in the local frame. We obtain a set
of correspondences between the point cloud S and the local
map M =

mi | mi ∈R3	
through nearest neighbor(NN)

<!-- page 6 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
6
Methods
LC Rm0 Rm1 Rm2 Off0 Off1 Off2 Off3 Off4 Avg.
NeRF-based Methods
NICE-SLAM [16]
✗
0.93 1.28 1.07 0.93 1.04 1.08 1.12 1.13 1.07
ESLAM [17]
✗
0.72 0.68 0.54 0.55 0.55 0.60 0.70 0.65 0.65
Co-SLAM [18]
✗
0.96 1.02 0.95 0.83 0.87 0.94 1.01 0.93 0.95
Point-SLAM [37]
✓
0.52 0.38 0.30 0.50 0.44 1.26 0.77 0.57 0.59
PLGSLAM [19]
✗
0.64 0.65 0.49 0.51 0.52 0.54 0.65 0.57 0.57
Loopy-SLAM [35]
✓
0.27 0.25 0.29 0.29 0.41 0.31 0.24 0.34 0.31
3DGS-based Methods
SplaTAM [22]
✗
0.33 0.44 0.33 0.51 0.29 0.29 0.35 0.75 0.41
MonoGS [23]
✗
0.35 0.24 0.32 0.38 0.22 0.26 0.16 0.82 0.34
Gaussian-SLAM [42]
✗
0.29 0.29 0.23 0.38 0.24 0.42 0.31 0.36 0.32
Photo-SLAM [55]
✗
0.65 0.65 0.50 0.53 0.52 0.55 0.66 0.58 0.58
Loop-Splat [45]
✓
0.28 0.22 0.17 0.22 0.15 0.49 0.20 0.30 0.26
Ours
✓
0.26 0.21 0.16 0.21 0.15 0.25 0.18 0.28 0.21
TABLE I: Camera Tracking Performance on Replica
dataset [54]. We use ATE RMSE (cm) as the metric. Best
results are highlighted as first , second , and third .
Methods
0000 0059 0106 0169 0181 0207 0054 0233 Avg.
NeRF-based Methods
NICE-SLAM [16]
12.3
14.2
7.9
10.9
13.6
6.8
20.9
9.4
13.2
ESLAM [17]
7.7
8.6
7.8
6.5
9.4
6.2
36.5
4.6
10.7
Co-SLAM [18]
7.2
11.4
9.5
5.9
11.9
7.3
-
-
-
Point-SLAM [37]
10.4
7.9
8.8
22.2
14.9
9.7
28.2
6.3
14.4
GO-SLAM [34]
5.7
7.5
7.0
7.8
6.8
6.7
8.8
4.9
6.8
PLGSLAM [19]
7.3
8.2
7.4
6.3
9.2
5.8
30.8
4.2
9.9
Loopy-SLAM [35]
4.9
7.7
8.5
7.7
10.6
7.9
14.5
5.2
7.7
3DGS-based Methods
SplaTAM [22]
12.8
10.1
17.7
12.1
11.1
7.5
56.8
4.8
16.6
MonoGS [23]
9.8
32.1
8.9
10.7
21.8
7.9
17.5
12.4 15.2
Gaussian-SLAM [42] 21.2
12.8
13.5
16.3
21.0
14.3
37.1
11.1 18.4
Loop-Splat [45]
6.2
7.2
7.4
10.8
8.5
6.7
16.3
4.8
8.4
Ours
5.7
7.2
7.1
9.1
7.5
6.2
13.9
4.6
7.6
TABLE II: Camera Tracking Performance on ScanNet
dataset [56]. We use ATE RMSE (cm) as the metric.
We evaluate on eight sequences following the experimental
settings of previous methods.
Ours
Ground Truth
SplaTAM
Loop-Splat
Fig. 4: Qualitative comparison between our proposed method and existing SOTA methods: SplaTAM [22] and Loop-Splat [45].
We demonstrate RGB image rendering results on the KITTI odometry dataset [57]. Our method shows improved rendering
quality compared to these existing methods.
search over the 3D Gaussian voxel map considering only
correspondences with a point-to-point distance below threshold.
To compute the current pose registration, we perform a robust
optimization minimizing the residuals:
∆{R, t}est = argmin
{R,t}
X
(s,q)∈C(τt)
ρ (∥{R, t}s −m∥2)
(11)
ρ(e) =
e2/2
σt/3 + e2
(12)
where C (τt) is the set of nearest neighbor correspondences
with a distance smaller than τt and ρ is the Geman-McClure
robust kernel. The scale factor σt of the kernel is adapted
online. Furthermore, we use a hash table to represent the voxel
structure, which offers memory-efficient storage and enables
fast nearest neighbor search.
C. 2D-3D Gaussian Loop Closure
In large-scale environments, pose drift accumulates sig-
nificantly, posing a major challenge for existing methods.
Furthermore, the fusion of multiple submaps also requires
alignment to achieve global consistency. We incorporate a
novel loop closure method into our framework based on 3D
Gaussians to identify pose corrections for past submaps and
keyframes. Although some existing GS-SLAM methods [45]
have incorporated loop closure, we improve the loop closure
optimization by combining 2D photometric loss with voxel-
based ICP, which allows better adaptation to domain differences

<!-- page 7 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
7
Ours
Ground Truth
SplaTAM
Loop-splat
Fig. 5: Visualization of RGB image rendering results on the KITTI odometry dataset sequence 02. Qualitative comparison
between our proposed method and existing SOTA methods: SplaTAM [22] and Loop-Splat [45]. We demonstrate RGB image
rendering results on the KITTI odometry dataset. Our method shows improved rendering quality compared to these existing
methods.
between indoor and outdoor environments. Furthermore, we
design an online distillation methods for submap fusion upon
loop closure detection, enabling the integration of information
from different submaps to achieve global consistency.
Loop Detection and Correction In order to detect when the
robot revisits the same place, we extract the visual descriptor
with a lightweight method Netvlad [59]. We then compute the
cosine similarities of the descriptor across different submap
α, β. Upon detecting a loop, we construct pose optimization
graph with the loop closure constraints. We utilize both 2D
rendering loss and voxel based ICP as loop closure constraints
to achieve better adaptation to domain differences. We obtain a
set of pose corrections {Ri|ti}M from PGO, where M denotes
the correction for submap M. Then we update both the camera
pose and attributes of 3D Gaussians:
{Ri|ti}M·{Ri|ti} 7→{Ri|ti},
{Ri|ti}M{qi}k−1
i=0 7→{qi}k−1
i=0
(13)
where {qi}k−1
i=0 denotes the quaternion of 3D Gaussians anchor
in submap M.
Submap Fusion After aligning the relative poses between
submaps, we further propose a submap fusion method to
integrate information across different submaps and ensure
global consistency. Specifically, for two submaps where a loop
closure is detected, we propose an online distillation approach.
we identify highly similar image pairs between the two submaps
as overlapping regions, and then construct a distillation loss to
merge the corresponding 3D Gaussians from the two submaps
effectively and improve spatial correlation. We use the poses
of the overlapping region keyframe {KF}m
i=1 to render the
RGB and depth images.
Lα−β = 1
m
m
X
α,β∈{KF }

(ˆcα −ˆcβ)2 +

ˆ
dα −ˆdβ
2
(14)
where ˆcα, ˆcβ, and ˆdα and ˆdβ denotes the rendered color and
depth images from different submap α and β from different
networks.
Methods
Accuracy
Completion
Comp. Ratio
SplaTAM [22]
2.74
4.02
84.61
MonoGS [23]
3.16
4.45
81.52
Gaussian-SLAM [42]
2.53
3.77
84.65
Splat-SLAM [47]
2.49
3.68
84.79
Ours
2.47
3.64
85.75
TABLE III: Scene Reconstruction Performance on Replica
dataset [54]. We use accuracy [cm], completion [cm] and
completion ratio (%) as the metrics for mesh evaluation.
Methods
VKITTI2 [60]
KITTI [57]
PSNR ↑
SSIM ↑
PSNR ↑
SSIM ↑
GO-SLAM [34]
19.29
0.73
15.71
0.51
SplaTAM [22]
18.29
0.69
14.68
0.48
MonoGS [23]
18.45
0.70
14.73
0.49
Gaussian-SLAM [42]
18.78
0.69
14.78
0.50
Loop-Splat [45]
19.47
0.76
16.77
0.71
Ours
25.45
0.84
21.37
0.81
TABLE IV: Scene Reconstruction Performance comparison
on KITTI [57] and VKITTI 2 [60] datasets. We use PSNR
and SSIM as the metrics.

<!-- page 8 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
8
Ours
Ground Truth
SplaTAM
Loop-splat
Fig. 6: Visualization of RGB image rendering results on the KITTI odometry dataset sequence 03. Qualitative comparison
between our proposed method and existing SOTA methods: SplaTAM [22] and Loop-Splat [45]. We demonstrate RGB image
rendering results on the KITTI odometry dataset. Our method shows improved rendering quality compared to these existing
methods.
Fig. 7: Mesh reconstruction results on Newer College dataset [58].
IV. EXPERIMENTS
We validate that our method outperforms existing implicit
representation-based methods in scene reconstruction, pose
estimation, and real-time performance.
Datasets. We evaluate VPGS-SLAM on a variety of scenes
from different datasets:
• Replica Dataset [54]. 8 small room scenes (approximately
6.5m × 4.2m × 2.7m ). We use this dataset to evaluate
the reconstruction and localization accuracy in small-scale
environments.
• ScanNet dataset [56]. Real-world scenes with long se-
quences (more than 5000 images) and large-scale indoor
scenarios (approximately 7.5m × 6.6m × 3.5m). We use
this dataset for large-scale real-world indoor environments.
• KITTI dataset [57] and VKITTI 2 dataset [60]. Ur-
ban scenes dataset with long sequences (approximately
500m × 400m). We use these datasets to validate the
effectiveness of our method in virtual and real-world city-
scale scenes.
• Newer College datasets [58]. This dataset provides diverse
trajectories and complex scenes, making it well-suited

<!-- page 9 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
9
Rendered RGB
Rendered Depth
Rendered RGB
Rendered Depth
Rendered RGB
Rendered Depth
Rendered RGB
Rendered Depth
Fig. 8: Extensive Experiments on KITTI datasets [57]. Our system is a large-scale SLAM framework, with voxel-based
progressive 3D Gaussian representation, 2D-3D fusion camera tracking, and 3D Gaussian loop closure. Our framework takes
color images and 3D point clouds as input. Our method can achieve accurate and efficient scene reconstruction, camera tracking,
and global map generation.
for assessing large-scale and high-fidelity reconstruction
performance.
We use these datasets to validate the effectiveness of our method
in indoor and real-world outdoor city-scale scenes.
Implementation Details We run our system on a desktop
PC with NVIDIA RTX 3090 GPU. We set the voxelized
corresponding parameter k = 10. All the MLPs employed
in our approach are 2-layer MLPs with ReLU activation; the
dimensions of the hidden units are all 32. We set the submap
initialization parameters with a distance threshold d = 0.5m
in indoor scenes and d = 10m for outdoor scenes, and a
rotation threshold ω = 50 degree. The two loss weight λSSIM
and λvol are set to 0.2 and 0.001 in our experiments. We use
DepthLab [68] for dense depth image in outdoor scene.
Scene Reconstruction To evaluate the performance of scene re-
construction, we utilized three datasets: Replica for small room
scenes, KITTI and vKITTI for large-scale urban scenes, and
Newer College for campus environments. We report the scene
reconstruction results on Replica [54] in Tab. V. The best results
are highlighted as first , second , and third . Our method
outperforms all 3DGS-based methods with superior rendering
performance, demonstrating its effectiveness in reconstructing
small room scenes. We also visualize the rendering performance
in the Replica scene in Fig. 3. To evaluate the effectiveness
of our method in large-scale urban scene reconstruction, we
test it on the KITTI [57] and VKITTI2 [60] dataset, as shown
in Tab. IV. In Fig. 4, Fig. 5, and Fig. 6, we visualize the
rendered results in different KITTI sequence and compare
them with SplaTAM [22] and MonoGS [23], demonstrating
our clearly superior performance on scene reconstruction and
rendering. We also visualize the mesh reconstruction results
on the Newer College dataset in Fig. 1. The left image shows
the reconstruction of the entire sequence, and the right image
shows a zoomed-in view of a local area. As can be seen, in
addition to the rendering performance, our mesh reconstruction
also achieves excellent results.
Camera Tracking We present our camera tracking performance
on Replica [54], ScanNet [56], and KITTI [57] datasets in
Tab. I, II, VI and Fig. 8. In the synthetic dataset Replica,
our method surpasses existing state-of-the-art approaches,
attributable to our 2D-3D fusion tracking method, which
effectively integrates 3D Gaussian and 2D information. Our
method achieves a 30% accuracy improvement on Replica
dataset over current NeRF-based and 3DGS-based approaches.

<!-- page 10 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
10
Method
Metrics
Room0
Room1
Room2
Office0
Office1
Office2
Office3
Office4
Avgerage
NeRF-based Methods
ESLAM [17]
Depth
0.97
1.07
1.28
0.86
1.26
1.71
1.43
1.06
1.18
PSNR
25.25
27.39
28.09
30.33
27.04
27.99
29.27
29.15
28.06
SSIM
0.874
0.891
0.935
0.934
0.910
0.942
0.953
0.948
0.923
Co-SLAM [18]
Depth
1.05
0.85
2.37
1.24
1.48
1.86
1.66
1.54
1.51
PSNR
27.27
27.45
28.12
30.31
28.47
28.91
29.75
29.91
28.77
SSIM
0.898
0.905
0.937
0.946
0.923
0.952
0.961
0.957
0.934
GO-SLAM [34]
Depth
2.56
1.57
2.43
1.47
1.63
2.31
1.71
1.47
4.68
PSNR
24.32
25.37
26.52
30.17
30.34
24.16
28.23
27.64
24.42
SSIM
0.854
0.871
0.905
0.904
0.893
0.912
0.927
0.918
0.898
Point-
SLAM [37]
Depth
0.53
0.22
0.46
0.30
0.57
0.49
0.51
0.46
0.44
PSNR
32.40
34.08
35.50
38.26
39.16
33.99
33.48
33.49
35.17
SSIM
0.974
0.977
0.982
0.983
0.986
0.960
0.960
0.979
0.975
Loopy-
SLAM [35]
Depth
0.33
0.21
0.43
0.24
0.47
0.62
0.37
0.26
0.37
PSNR
32.71
34.28
35.70
38.39
38.91
34.09
33.48
33.79
35.19
SSIM
0.984
0.980
0.989
0.985
0.991
0.974
0.967
0.985
0.980
3DGS-based Methods
SplaTAM [22]
Depth
0.43
0.38
0.54
0.44
0.66
1.05
1.60
0.68
0.72
PSNR
32.86
33.89
35.25
38.26
39.17
31.97
29.70
31.81
34.11
SSIM
0.982
0.974
0.983
0.981
0.981
0.971
0.951
0.953
0.972
Loop-
Splat [45]
Depth
0.39
0.23
0.52
0.32
0.51
0.63
1.09
0.40
0.51
PSNR
33.07
35.32
36.16
39.12
39.81
34.67
33.93
33.98
35.75
SSIM
0.971
0.978
0.981
0.989
0.988
0.981
0.987
0.984
0.982
Ours
Depth
0.34
0.22
0.50
0.30
0.49
0.59
0.98
0.38
0.51
PSNR
33.10
35.09
36.21
39.12
39.79
35.07
34.09
34.12
35.82
SSIM
0.973
0.981
0.985
0.991
0.991
0.981
0.988
0.985
0.984
TABLE V: Scene Reconstruction Performance on Replica dataset [54]. We use depth L1, PSNR, SSIM as our metics.
Methods
Map Type
00
01
02
03
04
05
06
07
08
09
10
Average
Suma [61]
Surfel
2.94
13.85
8.43
0.94
0.43
1.26
0.47
0.54
2.87
2.95
1.37
3.61
Litamin2 [62]
NDT
5.84
15.93
10.74
0.85
0.77
2.46
0.91
0.65
2.57
2.13
1.05
4.39
FLOAM [63]
Point Cloud
5.03
3.27
8.64
0.74
0.35
3.43
0.53
0.63
2.57
2.14
1.08
2.84
KISS-ICP [30]
3.72
10.35
7.86
2.13
0.51
1.37
0.65
0.63
3.64
2.35
1.49
3.47
Mesh-Loam [64]
(point-to-plane)
5.54
4.09
7.05
0.57
1.53
1.74
8.94
0.56
3.07
1.84
0.95
3.59
Puma [65]
Mesh
6.64
32.61
18.55
2.25
0.94
3.34
2.41
0.93
6.34
3.94
4.43
8.23
SLAMesh [66]
5.51
10.93
13.29
0.83
0.33
3.74
0.71
0.83
5.13
1.15
1.17
4.36
Mesh-Loam [64]
5.37
3.25
7.45
0.54
0.37
1.74
0.38
0.45
3.35
1.74
0.95
2.77
NeRF-Loam [39]
NeRF
8.64
20.58
7.45
1.79
0.80
5.01
2.47
0.79
4.76
3.47
3.02
5.88
PIN-LO [40]
5.84
4.37
9.35
0.85
0.19
1.87
0.53
0.55
3.27
2.35
0.97
3.01
PIN-SLAM [40]
1.18
3.46
2.69
0.80
0.17
0.37
0.48
0.30
2.58
1.34
0.96
1.43
Ours
3DGS
1.09
3.26
2.65
0.95
0.15
0.35
0.44
0.39
2.47
1.19
0.95
1.40
TABLE VI: Tracking performance comparison on KITTI dataset [57]. The evaluation is conducted on the LiDAR dataset
with motion compensated point cloud. We use ATE RMSE (m) as our metric. “Avg." denotes the average results of all sequences.
Note that ours is the only 3DGS-based method able to run successfully on all sequences, so we present only our method as a
representative of GS-based approaches.
On the real-world indoor dataset ScanNet, our method also
performs better than current NeRF-based and 3DGS-based
approaches. In challenging sequences such as ScanNet00 and
ScanNet69, where pose error accumulates continuously and
loop closures occur, our method accurately identifies loop
closure points and effectively mitigates accumulated pose errors.
In large-scale outdoor datasets KITTI [57], where most GS-
based and NeRF-based methods fail, we compare our approach
with pose estimation methods for large scenes, demonstrating
a 10% accuracy improvement over NeRF-LOAM.
Memory and Time analysis In Tab. VII, we present the
runtime, memory usage, and Peak GPU memory of our
method and other SOTA methods. In contrast to GS-based
methods, our method employs a progressive map representation
with multiple submaps, reducing excessive online GPU usage.
This approach reduces GPU requirements by a factor of 2-
3 compared to SplaTAM [22], enabling our algorithm to be
effectively utilized on edge computing platforms. Our voxelized
scene representation method also significantly improves map
representation efficiency, reducing the required memory six
times compared to SplaTAM [22].
A. Ablation Study
In this section, we validate the effectiveness of each module
in our algorithm in both indoor and outdoor datasets, shown
in Tab. VIII. First, we explore the impact of voxelized scene
representation and find that this module significantly enhances
mapping efficiency in dense scenes. Next, we compare the effec-
tiveness of progressive mapping with multiple submaps, which
considerably reduces GPU computation requirements while
improving tracking accuracy. This multi-map representation
confines pose errors within each submap, effectively mitigating
cumulative drift. We then assess the effectiveness of 2D-3D
fusion tracking. Our findings indicate that 2D photometric

<!-- page 11 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
11
Methods
Track/Iter.
Map/Iter.
Track/Frame(s) ↓
Map/Frame(s) ↓
Render FPS
Decoder Param.
Memory
Peak GPU
NeRF-based Methods
NICE-SLAM [16]
6.98ms
28.88ms
68.54ms
1.23s
0.30
0.06M
48.48MB
12.0GB
Vox-Fusion [67]
7.02ms
23.34ms
350.9ms
0.47s
1.31
0.06M
43.48MB
17.6GB
Point-SLAM [37]
6.85ms
19.98ms
25.72ms
10.47s
1.33
0.127M
55.42MB
8.5GB
ESLAM [17]
6.85ms
19.98ms
54.80ms
0.29s
2.82
0.003M
27.12MB
17.3GB
Co-SLAM [18]
6.38ms
14.25ms
63.93ms
0.15s
3.68
0.013M
24.85MB
16.7GB
Loopy-SLAM [35]
55.53ms
10.24ms
1.11s
4.08s
2.12
0.127M
60.98MB
13.3GB
3DGS-based Methods
SplaTAM [22]
24.23ms
22.83ms
2.18s
1.37s
175.64
0M
273.09MB
18.5GB
Loop-Splat [45]
10.28ms
17.53ms
1.03s
1.05s
315.48
0M
93.87MB
7.4GB
Ours
-
26.85ms
10.75ms
0.93s
322.45
0.003M
70.81MB
7.8GB
TABLE VII: Runtime and Memory Usage on Replica Room 0. Per-frame runtime is calculated as the total optimization
time divided by the sequence length. “-" is because our method does not use iterative tracking, making it infeasible to calculate
per-iteration time. The memory usage represents the total memory of the map representation. “Decoder Param." denotes decoder
parameters. Note that implicit field-based methods require additional space for their decoders.
Method
Replica [54]
KITTI [57]
Accuracy
Real-time performance and Memory
Accuracy
Real-time performance and Memory
PSNR ↑RMSE ↓Time ↓Render FPS↑Memory↓
GPU
PSNR ↑RMSE ↓Time ↓Render FPS↑
Memory↓
GPU
(a)w/o Vox.
33.08
0.24
0.53H
317.43
94.45MB 10.9GB
21.01
1.15
3.71H
261.39
371.49MB 30.9GB
(b)w/o Prog.
31.49
0.27
0.51H
321.45
73.98MB 18.8GB
-
-
-
-
-
-
(c)only 2Dtrack
30.45
0.28
0.50H
322.45
70.81MB
7.6GB
17.92
5.96
3.02H
301.14
264.59MB 19.9GB
(c)only 3Dtrack
30.21
0.41
0.41H
322.45
70.81MB
7.3GB
20.39
1.15
2.61H
301.14
264.59MB 18.5GB
(d)w/o LC
32.87
0.23
0.49H
322.45
70.81MB
7.7GB
18.13
3.15
3.09H
301.14
264.59MB 20.2GB
Full model
33.10
0.21
0.51H
322.45
70.81MB
7.8GB
21.37
1.09
3.17H
301.14
264.59MB 20.4GB
TABLE VIII: Ablation study on the Replica [54] and KITTI [57] dataset. “H" denotes hours. “-" denotes fail. The full
model demonstrates superior pose estimation accuracy while maintaining faster training/rendering speed and lower memory
consumption.
loss performs better in indoor and simulated environments,
whereas 3D geometric cues are more beneficial for outdoor
scene tracking. Compared with 3D gaussian-based tracking,
our tracking method can achieve more accurate matching with
2D-3D information fusion. Finally, we validate the loop closure
module’s effectiveness, showing that it significantly reduces
cumulative pose errors.
V. CONCLUSION
In this paper, we propose a novel large-scale 3DGS SLAM
framework, VPGS-SLAM, which achieves accurate scene
reconstruction and pose estimation in both small and large-scale
scenes. Our voxelized progressive mapping method achieves
compact and accurate scene representation. The novel 2D-
3D fusion camera tracking fully leverages the 3D Gaussian
attributes with photometric information to efficiently match
the 3D points and achieve accurate camera tracking. The 3D
Gaussian loop closure enables global map consistency across
multiple submaps and eliminates the accumulative pose error.
The extensive experiments demonstrate the effectiveness and
accuracy of our system in scene reconstruction, view synthesis,
and pose estimation in various scenes.
REFERENCES
[1] Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide
Scaramuzza, José Neira, Ian Reid, and John J Leonard. Past, present,
and future of simultaneous localization and mapping: Toward the robust-
perception age. IEEE Transactions on robotics, 32(6):1309–1332, 2016.
[2] Weinan Chen, Shilang Chen, Jiewu Leng, Jiankun Wang, Yisheng Guan,
Max Q.-H. Meng, and Hong Zhang. A review of cloud-edge slam: Toward
asynchronous collaboration and implicit representation transmission.
IEEE Transactions on Intelligent Transportation Systems, 25(11):15437–
15453, 2024.
[3] Chih-Chung Chou and Cheng-Fu Chou. Efficient and accurate tightly-
coupled visual-lidar slam. IEEE Transactions on Intelligent Transporta-
tion Systems, 23(9):14509–14523, 2022.
[4] Tianchen Deng, Yue Pan, Shenghai Yuan, Dong Li, Chen Wang, Mingrui
Li, Long Chen, Lihua Xie, Danwei Wang, Jingchuan Wang, Javier
Civera, Hesheng Wang, and Weidong Chen. What is the best 3d scene
representation for robotics? from geometric to foundation models. arXiv
preprint arXiv:2512.03422, 2025.
[5] Raúl Mur-Artal and Juan D. Tardós. Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras. IEEE Transactions on
Robotics, 33(5):1255–1262, 2017.
[6] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and
versatile monocular visual-inertial state estimator. IEEE Transactions on
Robotics, 34(4):1004–1020, 2018.
[7] Hongle Xie, Tianchen Deng, Jingchuan Wang, and Weidong Chen.
Robust incremental long-term visual topological localization in changing
environments. IEEE Transactions on Instrumentation and Measurement,
72:1–14, 2022.
[8] Hongming Shen, Zhenyu Wu, Yulin Hui, Wei Wang, Qiyang Lyu,
Tianchen Deng, Yeqing Zhu, Bailing Tian, and Danwei Wang. Cte-mlo:
Continuous-time and efficient multi-lidar odometry with localizability-
aware point cloud sampling.
IEEE Transactions on Field Robotics,
2025.
[9] Qi Chen, Yu Cao, Jiawei Hou, Guanghao Li, Shoumeng Qiu, Bo Chen,
Xiangyang Xue, Hong Lu, and Jian Pu.
Vpl-slam: A vertical line
supported point line monocular slam system. IEEE Transactions on
Intelligent Transportation Systems, 25(8):9749–9761, 2024.
[10] Zhiqi Zhao, Chang Wu, Xiaotong Kong, Qiyan Li, Zifan Guo, Zejie
Lv, and Xiaoqi Du. Light-slam: A robust deep-learning visual slam
system based on lightglue under challenging lighting conditions. IEEE
Transactions on Intelligent Transportation Systems, 26(7):9918–9931,
2025.
[11] Richard A Newcombe, Steven J Lovegrove, and Andrew J Davison.
Dtam: Dense tracking and mapping in real-time. In 2011 international
conference on computer vision, pages 2320–2327. IEEE, 2011.
[12] Thomas Whelan, Michael Kaess, Hordur Johannsson, Maurice Fallon,
John J Leonard, and John McDonald. Real-time large-scale dense rgb-d
slam with volumetric fusion. The International Journal of Robotics
Research, 34(4-5):598–626, 2015.
[13] Thomas Whelan, Renato F Salas-Moreno, Ben Glocker, Andrew J
Davison, and Stefan Leutenegger. Elasticfusion: Real-time dense slam

<!-- page 12 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
12
and light source estimation.
The International Journal of Robotics
Research, 35(14):1697–1716, 2016.
[14] Tong Qin, Changze Li, Haoyang Ye, Shaowei Wan, Minzhen Li, Hongwei
Liu, and Ming Yang.
Crowd-sourced nerf: Collecting data from
production vehicles for 3d street view reconstruction. IEEE Transactions
on Intelligent Transportation Systems, 25(11):16145–16156, 2024.
[15] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. imap:
Implicit mapping and positioning in real-time. In ICCV, pages 6229–6238,
October 2021.
[16] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao,
Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural
implicit scalable encoding for slam. In CVPR, pages 12786–12796, June
2022.
[17] Mohammad Mahdi Johari, Camilla Carta, and François Fleuret. Eslam:
Efficient dense slam system based on hybrid representation of signed
distance fields. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 17408–17419, 2023.
[18] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-slam: Joint
coordinate and sparse parametric encodings for neural real-time slam.
In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 13293–13302, 2023.
[19] Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wentao Zhao,
Jingchuan Wang, Danwei Wang, and Weidong Chen. Plgslam: Progressive
neural scene represenation with local to global bundle adjustment. In
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 19657–19666, June 2024.
[20] Tianchen Deng, Yanbo Wang, Hongle Xie, Hesheng Wang, Rui Guo,
Jingchuan Wang, Danwei Wang, and Weidong Chen. Neslam: Neural
implicit mapping and self-supervised feature tracking with depth com-
pletion and denoising. IEEE Transactions on Automation Science and
Engineering, 2025.
[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George
Drettakis. 3d gaussian splatting for real-time radiance field rendering.
ACM Transactions on Graphics, 42(4), 2023.
[22] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan
Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam:
Splat, track & map 3d gaussians for dense rgb-d slam. arXiv preprint
arXiv:2312.02126, 2023.
[23] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison.
Gaussian splatting slam. arXiv preprint arXiv:2312.06741, 2023.
[24] Raúl Mur-Artal, J. M. M. Montiel, and Juan D. Tardós. Orb-slam: A
versatile and accurate monocular slam system. IEEE Transactions on
Robotics, 31(5):1147–1163, 2015.
[25] Hanbiao Xiao, Zhaozheng Hu, Chen Lv, Jie Meng, Jianan Zhang, and
Ji’an You. Progressive multi-modal semantic segmentation guided slam
using tightly-coupled lidar-visual-inertial odometry. IEEE Transactions
on Intelligent Transportation Systems, 26(2):1645–1656, 2025.
[26] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular,
stereo, and rgb-d cameras. Advances in neural information processing
systems, 34:16558–16569, 2021.
[27] Jiuming Liu, Dong Zhuo, Zhiheng Feng, Siting Zhu, Chensheng Peng,
Zhe Liu, and Hesheng Wang. Dvlo: Deep visual-lidar odometry with
local-to-global feature fusion and bi-directional structure alignment. In
European Conference on Computer Vision, pages 475–493. Springer,
2024.
[28] Ji Zhang, Sanjiv Singh, et al. Loam: Lidar odometry and mapping
in real-time. In Robotics: Science and systems, volume 2, pages 1–9.
Berkeley, CA, 2014.
[29] Pierre Dellenbach, Jean-Emmanuel Deschaud, Bastien Jacquet, and
François Goulette. Ct-icp: Real-time elastic lidar odometry with loop
closure. In 2022 International Conference on Robotics and Automation
(ICRA), pages 5580–5586. IEEE, 2022.
[30] Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Louis Wiesmann,
Jens Behley, and Cyrill Stachniss. Kiss-icp: In defense of point-to-point
icp–simple, accurate, and robust registration if done the right way. IEEE
Robotics and Automation Letters, 8(2):1029–1036, 2023.
[31] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron,
Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural
radiance fields for view synthesis. In ECCV, 2020.
[32] Tianchen Deng, Nailin Wang, Chongdi Wang, Shenghai Yuan, Jingchuan
Wang, Danwei Wang, and Weidong Chen. Incremental joint learning of
depth, pose and implicit scene representation on monocular camera in
large-scale scenes. arXiv preprint arXiv:2404.06050, 2024.
[33] Zhong Wang, Lin Zhang, and Hesheng Wang. S²kan-slam: Elastic neural
lidar slam with sdf submaps and kolmogorov-arnold networks. IEEE
Transactions on Circuits and Systems for Video Technology, 35(8):7618–
7630, 2025.
[34] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. Go-
slam: Global optimization for consistent 3d instant reconstruction. In
Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 3727–3737, 2023.
[35] Lorenzo Liso, Erik Sandström, Vladimir Yugay, Luc Van Gool, and
Martin R Oswald. Loopy-slam: Dense neural slam with loop closures.
In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 20363–20373, 2024.
[36] Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu, Liang Song,
Marc Pollefeys, and Hesheng Wang. Sni-slam: Semantic neural implicit
slam. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21167–21177, 2024.
[37] Erik Sandström, Yue Li, Luc Van Gool, and Martin R Oswald. Point-slam:
Dense neural point cloud-based slam. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 18433–18444, 2023.
[38] Ganlin Zhang, Erik Sandström, Youmin Zhang, Manthan Patel, Luc
Van Gool, and Martin R Oswald. Glorie-slam: Globally optimized rgb-
only implicit encoding point cloud slam. arXiv preprint arXiv:2403.19549,
2024.
[39] Junyuan Deng, Qi Wu, Xieyuanli Chen, Songpengcheng Xia, Zhen Sun,
Guoqing Liu, Wenxian Yu, and Ling Pei. Nerf-loam: Neural implicit
representation for large-scale incremental lidar odometry and mapping.
In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 8218–8227, 2023.
[40] Yue Pan, Xingguang Zhong, Louis Wiesmann, Thorbjörn Posewsky, Jens
Behley, and Cyrill Stachniss. Pin-slam: Lidar slam using a point-based
implicit neural representation for achieving global map consistency. arXiv
preprint arXiv:2401.09101, 2024.
[41] Mingrui Li, Zhetao Guo, Tianchen Deng, Yiming Zhou, Yuxiang Ren,
and Hongyu Wang. Ddn-slam: Real time dense dynamic neural implicit
slam. IEEE Robotics and Automation Letters, 2025.
[42] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-
slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint
arXiv:2312.10070, 2023.
[43] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao,
and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting.
arXiv preprint arXiv:2311.11700, 2023.
[44] Tianchen Deng, Yaohui Chen, Leyan Zhang, Jianfei Yang, Shenghai
Yuan, Danwei Wang, and Weidong Chen. Compact 3d gaussian splatting
for dense visual slam. arXiv preprint arXiv:2403.11247, 2024.
[45] Liyuan Zhu, Yue Li, Erik Sandström, Konrad Schindler, and Iro Armeni.
Loopsplat: Loop closure by registering 3d gaussian splats. arXiv preprint
arXiv:2408.10154, 2024.
[46] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp slam. arXiv
preprint arXiv:2403.12550, 2024.
[47] Erik Sandström, Keisuke Tateno, Michael Oechsle, Michael Niemeyer,
Luc Van Gool, Martin R Oswald, and Federico Tombari. Splat-slam:
Globally optimized rgb-only slam with 3d gaussians. arXiv preprint
arXiv:2405.16544, 2024.
[48] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong
Wang, and Kun Zhou. Rtg-slam: Real-time 3d reconstruction at scale
using gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers,
pages 1–11, 2024.
[49] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na Cheng, Tianchen
Deng, and Hongyu Wang. Sgs-slam: Semantic gaussian splatting for
neural dense slam. In European Conference on Computer Vision, pages
163–179. Springer, 2024.
[50] Shuhong Liu, Tianchen Deng, Heng Zhou, Liuzhuozheng Li, Hongyu
Wang, Danwei Wang, and Mingrui Li. Mg-slam: Structure gaussian
splatting slam with manhattan world hypothesis. IEEE Transactions on
Automation Science and Engineering, 2025.
[51] Mingrui Li, Shuhong Liu, Tianchen Deng, and Hongyu Wang. Densesplat:
Densifying gaussian splatting slam with neural radiance prior. arXiv
preprint arXiv:2502.09111, 2025.
[52] Kai Deng, Jian Yang, Shenlong Wang, and Jin Xie. Gigaslam: Large-
scale monocular slam with hierachical gaussian splats. arXiv preprint
arXiv:2503.08071, 2025.
[53] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua
Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive
rendering. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20654–20664, 2024.
[54] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans,
Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma,
et al. The replica dataset: A digital replica of indoor spaces. arXiv
preprint arXiv:1906.05797, 2019.
[55] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-
slam: Real-time simultaneous localization and photorealistic mapping for

<!-- page 13 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
13
monocular stereo and rgb-d cameras. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pages
21584–21593, June 2024.
[56] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas
Funkhouser, and Matthias Niessner.
Scannet: Richly-annotated 3d
reconstructions of indoor scenes. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), July 2017.
[57] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun.
Vision meets robotics: The kitti dataset. International Journal of Robotics
Research (IJRR), 2013.
[58] Milad Ramezani, Yiduo Wang, Marco Camurri, David Wisth, Matias
Mattamala, and Maurice Fallon. The newer college dataset: Handheld
lidar, inertial and vision with ground truth.
In 2020 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS),
pages 4353–4360. IEEE, 2020.
[59] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef
Sivic. Netvlad: Cnn architecture for weakly supervised place recognition.
In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 5297–5307, 2016.
[60] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2.
arXiv preprint arXiv:2001.10773, 2020.
[61] Jens Behley and Cyrill Stachniss. Efficient surfel-based slam using 3d
laser range data in urban environments. In Robotics: science and systems,
volume 2018, page 59, 2018.
[62] Masashi Yokozuka, Kenji Koide, Shuji Oishi, and Atsuhiko Banno.
Litamin2: Ultra light lidar-based slam using geometric approximation
applied with kl-divergence. In 2021 IEEE international conference on
robotics and automation (ICRA), pages 11619–11625. IEEE, 2021.
[63] Han Wang, Chen Wang, Chun-Lin Chen, and Lihua Xie. F-loam: Fast
lidar odometry and mapping. In 2021 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pages 4390–4396. IEEE, 2021.
[64] Yanjin Zhu, Xin Zheng, and Jianke Zhu. Mesh-loam: Real-time mesh-
based lidar odometry and mapping. IEEE Transactions on Intelligent
Vehicles, 2024.
[65] Ignacio Vizzo, Xieyuanli Chen, Nived Chebrolu, Jens Behley, and Cyrill
Stachniss. Poisson surface reconstruction for lidar odometry and mapping.
In 2021 IEEE international conference on robotics and automation
(ICRA), pages 5624–5630. IEEE, 2021.
[66] Jianyuan Ruan, Bo Li, Yibo Wang, and Yuxiang Sun. Slamesh: Real-time
lidar simultaneous localization and meshing. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages 3546–3552. IEEE,
2023.
[67] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and
Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-
based neural implicit representation.
In 2022 IEEE International
Symposium on Mixed and Augmented Reality (ISMAR), pages 499–507,
2022.
[68] Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang, Hao Ouyang,
Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen, and Ping Luo. Depthlab:
From partial to complete. arXiv preprint arXiv:2412.18153, 2024.
Tianchen Deng received the B.Eng. degree in control
science and engineering from the Harbin Institute of
Technology, Harbin, China, in 2021. He is currently
pursuing the Joint Ph.D. degree in control science and
engineering with Shanghai Jiao Tong University and
Nanyang Technological University. His main research
interests include visual SLAM, 3D Reconstruction,
world model, and Embodied AI.
Wenhua Wu received the B.S. degree in Depart-
ment of Automation, Shanghai Jiao Tong University,
Shanghai, China, in 2023. He is currently pursuing
the Ph.D. degree in Computer Science and Technol-
ogy with Shanghai Jiao Tong University. His current
research interests include robot learning and computer
vision.
Junjie He received the B.E. degree in Automation
from Guangdong University of Technology, in 2022,
and the M.E. degree in artificial intelligence from
Xi’an Jiaotong University, in 2025. From 2025, He
works as a Research Assistant at the Thrust of
Robotics and Autonomous Systems, The Hong Kong
University of Science and Technology (Guangzhou).
His research interests include 3D computer vision.
Yue Pan is a Ph.D. student at the Photogrammetry &
Robotics Lab at the University of Bonn, Germany. He
obtained his B.Sc. degree in Geomatics Engineering
from Wuhan University, China in 2019 and received
his MSc degree in Geomatics Engineering from ETH
Zurich, Switzerland in 2022. His research focuses
on SLAM, 3D reconstruction, and navigation.
Shenghai Yuan is a senior research fellow at the
Centre for Advanced Robotics Technology Innova-
tion (CARTIN), Nanyang Technological University,
Singapore. He received his B.S. and Ph.D. degrees
in Electrical and Electronic Engineering in 2013 and
2019, respectively. His research focuses on robotics
perception and navigation. Currently, he serves as an
associate editor for the Unmanned Systems Journal
and as a guest editor of the Electronics Special
Issue on Advanced Technologies of Navigation for
Intelligent Vehicles. He received the Outstanding
Reviewer Award at ICRA 2024.
Danwei Wang (Life Fellow, IEEE) received the
B.E. degree from the South China University of
Technology, China, in 1982, and the M.S.E. and
Ph.D. degrees from the University of Michigan, Ann
Arbor, MI, USA, in 1984 and 1989, respectively.
He is a fellow of the Academy of Engineering
Singapore. He was a recipient of the Alexander von
Humboldt Fellowship, Germany. He served as the
general chairperson, the technical chairperson, and
various positions for several international conferences.
He was an invited guest editor of various international
journals. He is a Distinguished Lecturer of the IEEE Robotics and Automation
Society.

<!-- page 14 -->
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS
14
Hesheng Wang (Senior Member, IEEE) received
the B.Eng. degree in electrical engineering from
the Harbin Institute of Technology in 2002 and
the M.Phil. and Ph.D. degrees in automation and
computer-aided engineering from The Chinese Uni-
versity of Hong Kong, China, in 2004 and 2007,
respectively. He is currently a Professor with the
Department of Automation, Shanghai Jiao Tong
University. He is an Associate Editor of Assembly Au-
tomation and the International Journal of Humanoid
Robotics and an Senior Editor of IEEE/ASME
TRANSACTIONS ON MECHATRONICS. He served as an Associate Editor
for IEEE TRANSACTIONS ON ROBOTICS from 2015 to 2019. He was the
General Chair of IEEE ROBIO 2022 and IEEE RCAR 2016, and the Program
Chair of the IEEE ROBIO 2014 and IEEE/ASME AIM 2019. He will be the
General Chair of IEEE/RSJ IROS 2025.
