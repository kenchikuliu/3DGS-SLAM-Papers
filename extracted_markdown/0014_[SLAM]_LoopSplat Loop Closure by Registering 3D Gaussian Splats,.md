<!-- page 1 -->
LoopSplat: Loop Closure by Registering 3D Gaussian Splats
Liyuan Zhu1
Yue Li2
Erik Sandstr¨om3
Shengyu Huang3
Konrad Schindler3
Iro Armeni1
1Stanford University
2University of Amsterdam
3ETH Zurich
PSNR: 22.33 dB
PSNR:  25.59 dB
PSNR: 16.21 dB
PSNR: 28.52 dB
GO-SLAM [101]
Gaussian-SLAM [95]
Loopy-SLAM [40]
LoopSplat (Ours)
Ground Truth
Figure 1. Dense Reconstruction on ScanNet [17] scene0054. LoopSplat demonstrates superior performance in geometric accuracy,
robust tracking, and high-quality re-rendering. This is enabled by our globally consistent reconstruction approach utilizing 3DGS [38].
Abstract
Simultaneous Localization and Mapping (SLAM) based
on 3D Gaussian Splats (3DGS) has recently shown promise
towards more accurate, dense 3D scene maps. However, ex-
isting 3DGS-based methods fail to address the global con-
sistency of the scene via loop closure and/or global bun-
dle adjustment. To this end, we propose LoopSplat, which
takes RGB-D images as input and performs dense mapping
with 3DGS submaps and frame-to-model tracking. Loop-
Splat triggers loop closure online and computes relative
loop edge constraints between submaps directly via 3DGS
registration, leading to improvements in efficiency and ac-
curacy over traditional global-to-local point cloud registra-
tion. It uses a robust pose graph optimization formulation
and rigidly aligns the submaps to achieve global consis-
tency. Evaluation on the synthetic Replica and real-world
TUM-RGBD, ScanNet, and ScanNet++ datasets demon-
strates competitive or superior tracking, mapping, and ren-
dering compared to existing methods for dense RGB-D
SLAM. Code is available at loopsplat.github.io.
1. Introduction
Dense Simultaneous Localization and Mapping (SLAM)
with RGB-D cameras has seen steady progress throughout
the years from traditional approaches [8, 18, 53, 54, 68, 87]
to neural implicit methods [33, 40, 46, 63, 64, 75, 78, 83,
90, 101, 103] and recent methods that employ 3D Gaus-
sians [38] as the scene representation [30, 36, 48, 88, 95].
Existing methods can be split into two categories, decou-
pled and coupled, where decoupled methods [15, 30, 49,
61, 101] do not leverage the dense map for the tracking
task, while the coupled methods [36, 40, 43, 48, 63, 64,
75, 78, 83, 88, 90, 95, 103] employ frame-to-model track-
ing using the dense map. Decoupling mapping and tracking
generally creates undesirable redundancies in the system,
such as inefficient information sharing and increased com-
putational overhead. On the other hand, all coupled 3DGS
SLAM methods lack strategies for achieving global consis-
tency on the map and the poses, which leads to an accumu-
lation of pose errors and distorted maps. Among the recent
methods that enforce global consistency via loop closure
and/or global bundle adjustment (BA), GO-SLAM [101]
requires costly retraining of the hash grid features to de-
1
arXiv:2408.10154v2  [cs.CV]  20 Aug 2024

<!-- page 2 -->
form the map and Photo-SLAM [30] similarly requires
additional optimization of the 3D Gaussian parameters to
resolve pose updates from the ORB-SLAM [52] tracker.
These re-integration techniques need to save all mapped
frames in memory, which limits their scalability. To avoid
saving all mapped frames, Loopy-SLAM [40] uses submaps
of neural point clouds and rigidly updates them after loop
closure. However, to compute the loop edge constraints,
Loopy-SLAM uses traditional global-to-local point cloud
registration. This is not only slow, but also fails to lever-
age the property of the scene representation itself.
To address limitations of current systems, we seek a cou-
pled SLAM system that avoids saving all mapped input
frames and is able to extract loop constraints directly from
the dense map, without redundant compute. Framed as a re-
search question, we ask: Can we use the map representation
(i.e., 3DGS) itself for loop closure in a SLAM system? To
this end, we propose a dense RGB-D SLAM system that
uses submaps of 3D Gaussians for local frame-to-model
tracking and dense mapping and is based on existing sys-
tems [48, 95]. Different to the latter, we achieve global con-
sistency via online loop closure detection and pose graph
optimization. Importantly, we show that traditional point
cloud registration techniques are not suitable to derive the
loop edge constraints from 3D Gaussians and propose a new
registration method that directly operates on the 3DGS rep-
resentation, hence using 3DGS as a unified scene represen-
tation for tracking, mapping, and maintaining global con-
sistency. Our key contributions are:
1. We introduce LoopSplat, a coupled RGB-D SLAM sys-
tem based on Gaussian Splatting, featuring a novel loop
closure module. This module operates directly on Gaus-
sian splats, integrating both 3D geometry and visual
scene content for robust loop detection and closure.
2. We develop an effective way to register two 3DGS rep-
resentations, so as to efficiently extract edge constraints
for pose graph optimization. Leveraging the fast ras-
terization of 3DGS, it is seamlessly integrated into the
system, outperforming traditional techniques in terms of
both speed and accuracy.
3. We enhance the tracking and reconstruction performance
of 3DGS-based RGB-D SLAM system, demonstrating
marked improvements and increased robustness across
diverse real-world datasets.
2. Related Work
Dense Visual SLAM. The seminal work of Curless and
Levoy [16] paved the way for dense 3D mapping with
truncated signed distance functions. Using frame-to-model
tracking, KinectFusion [53] showed that real-time SLAM
is possible from a commodity depth sensor.
To address
the cubic memory scaling to the scene size, numerous
works utilized voxel hashing [18, 34, 49, 54, 55] and oc-
trees [10, 41, 47, 69, 90] for map compression. Point-based
representations have also been popular [8, 12, 34, 37, 40, 63,
68, 87, 98, 99], with surfels and lately using neural points
or 3D Gaussians [36, 48, 65, 88, 95, 98]. To tackle the is-
sue of accumulating pose errors, globally consistent dense
SLAM methods have been developed, where a subdivision
of the global map into submaps is common [5, 8, 13, 18, 22,
27, 34, 35, 40, 44–46, 49, 60, 72, 78, 78], followed by pose
graph optimization [8, 13, 19, 20, 26–28, 35, 39, 40, 45, 46,
49, 49, 60, 68, 72, 78, 82, 89] to deform the submaps be-
tween them. Additionally, some works employ global BA
for refinement [8, 15, 18, 28, 49, 68, 78, 79, 89, 91, 101].
3D Gaussian SLAM with RGB-D input has also been
shown, however, methods fail to consider global consis-
tency [36, 48, 88, 95], leading to error accumulation in
the map and pose estimates. Most similar to our work is
Loopy-SLAM [40], which uses the explicit neural point
cloud representation of Point-SLAM [63] and equips it with
global consistency via loop closure on submaps.
Loop-
Splat differentiates itself from Loopy-SLAM and demon-
strates improvements in three key areas: (i) We improve
the accuracy and efficiency of the relative pose constraints
by directly registering 3DGS, instead of resorting to clas-
sical techniques like FPFH [62] with RANSAC, followed
by ICP [4]. (ii) We avoid having to mesh the submaps in a
separate process for registration and use the 3D Gaussians
directly. (iii) For loop detection, we rely on a combination
of image matching and overlap between submaps, leading
to better detections than using only image content as in [40].
Geometric Registration. Geometric registration is an im-
portant component of building edge constraints for pose
graphs. Specifically, point cloud registration aims to find
a rigid transformation that aligns two point cloud fragments
into the same coordinate framework. Traditional methods
leverage hand-crafted local descriptors [62, 80] for feature
matching, followed by RANSAC for pose estimation. Re-
cent learning-based methods either use patch-based local
descriptors [23, 96] or efficient fully-convolutional ones [3,
14]. BUFFER [1] balances the efficiency and generaliza-
tion of local descriptors by combining fully-convolutional
backbones for key-point detection with a patch-based net-
work for feature description. To address fragment registra-
tion with low overlap, Predator [32] uses attention mech-
anisms [81] to guide key-point sampling, significantly im-
proving the robustness of algorithms. This has been fur-
ther enhanced through coarse-to-fine matching [59]. Point
clouds lack the continuous, view-dependent, and multi-
scale representation capabilities of NeRFs, limiting their
ability to fully capture complex 3D scene in SLAM.
Neural Radiance Fields (NeRF) [51] have been widely
adopted for various applications beyond scene reconstruc-
tion, including scene understanding [21], autonomous driv-
ing [85], and SLAM [56, 76]. When modeling large-scale
2

<!-- page 3 -->
Input: RGB-D Video
Submap Tracking & Mapping
w/ 3D Gaussian Splats
Loop Detection
Loop Edge Constraint
Pose Graph Optimization
Pose & Map Update
Loop Closure
3DGS Registration
Viewpoint Selection
Diff. Rasterization
Multi-view Pose Opt.
Rotation Averaging
R, t
Aligned GS Point Cloud
Figure 2. LoopSplat Overview. LoopSplat is a coupled RGB-D SLAM system that uses Gaussian splats as a unified scene representation
for tracking, mapping, and maintaining global consistency. In the front-end, it continuously estimates the camera position while construct-
ing the scene using Gaussian splats. When the camera traverses beyond a predefined threshold, the current submap is finalized, and a new
one is initiated. Concurrently, the back-end loop closure module monitors for location revisits. Upon detecting a loop, the system generates
a pose graph, incorporating loop edge constraints derived from our proposed 3DGS registration. Subsequently, pose graph optimization
(PGO) is executed to refine both camera poses and submaps, ensuring overall spatial coherence.
scenes with NeRF, it is necessary to partition the scene
into blocks to manage memory constraints and to ensure
sufficient representation power. Consequently, registering
NeRFs to merge different partitions emerged as a research
problem. iNeRF [92] aligns a query image to the NeRF
map through analysis-by-synthesis: it optimizes the camera
pose so that the rendered image matches the query. How-
ever, this method is only suitable for local refinement due
to its non-convex nature, which can cause the model to get
stuck in local minima. NeRF2NeRF [24] aims to align two
NeRFs by extracting surface points from the density field
and aligning manually selected keypoints to estimate the
pose. DReg-NeRF [11] addresses NeRF registration sim-
ilarly to point cloud registration, by first extracting surface
points and then applying a fully convolutional feature ex-
traction backbone. Recently, Gaussian Splatting [38] has
started to replace NeRFs due to its efficient rasterization
and flexible editing capabilities, afforded by the explicit
representation. GaussReg [9] pioneered learning-based 3D
Gaussian Splatting (3DGS) registration, drawing on the fast
rendering of 3DGS. However, all previous NeRF and 3DGS
registration methods [9, 11, 24, 92] assume ground truth
camera poses for training views, which is not suitable for
real-world SLAM scenarios. Moreover, these methods have
only explored pairwise registration in small-scale scenes.
Our method, without any training or preprocessing, directly
operates on estimated camera poses from the SLAM front-
end and can be integrated into loop closure on the fly.
3. LoopSplat
LoopSplat is an RGB-D SLAM system that simultaneously
estimates the camera poses and builds a 3D Gaussian map
from input frames in a globally consistent manner. This
section begins with a recap of the Gaussian-SLAM system
described in [95] (Sec. 3.1) – which is the base of Loop-
Splat, followed by the introduction of the proposed 3DGS
registration module (Sec. 3.2). Finally, the integration of
loop closure into the Gaussian-SLAM system, enabled by
the registration module, is presented in Sec. 3.3. Please
see Fig. 2 for an overview of the proposed system.
3.1. Gaussian Splatting SLAM
We follow [40, 95] and represent the scene using a collec-
tion of submaps, each modeling several keyframes with a
3D Gaussian point cloud Ps, where
Ps = {Gi(µ, Σ, o, C)|, i = 1, . . . , N},
(1)
with individual Gaussian mean µ ∈R3, covariance matrix
Σ ∈R3×3, opacity value o ∈R, and RGB color C ∈R3.
Submap Initialization.
Starting from the first keyframe
Is
f, each submap models a sequence of keyframes observ-
ing a specific region. As the explored scene space expands,
a new submap is initialized to avoid processing the entire
global map simultaneously.
Unlike previous approaches
that use a fixed number of keyframes [13, 18, 44], we dy-
namically trigger new submap initialization when the cur-
rent frame’s relative displacement or rotation to the first
keyframe Is
f exceeds the predefined thresholds, dthre or θthre.
Frame-to-model Tracking.
To localize a incoming frame
Is
j within the current submap Ps, we first initialize the cam-
era pose Tj based on the constant motion assumption as:
Tj = Tj−1 · T−1
j−2 · Tj−1. Next, we optimize Tj by mini-
mizing the tracking loss Ltracking(ˆIs
j, ˆDs
j, Is
j, Ds
j, Tj), which
measures the discrepancy between the rendered color ˆIj and
depth ˆDs
j images at viewpoint Tj, and the input color Is
j
3

<!-- page 4 -->
and depth Ds
j. To stabilize tracking, we use an alpha mask
Ma and an inlier mask Min to address gross errors caused by
poorly reconstructed or previously unobserved areas. The
final tracking loss is a sum over the valid pixels as
Ltracking =
X
Min·Ma·(λc|ˆIs
j−Is
j|1+(1−λc)| ˆDs
j−Ds
j|1),
(2)
where λc is a weight that balances the color and depth
losses, and ∥· ∥denotes the L1 loss between two images.
Please refer to the supplementary material for more details.
Submap Expansion.
Keyframes are selected by fixed in-
terval for the submap.
Once the current keyframe Is
j is
localized, we expand the 3D Gaussian map primarily in
sparsely covered regions for efficient mapping.
We first
compute a posed dense point cloud from the RGB-D in-
put and then uniformly sample Mk points from areas where
the accumulated alpha values are below a threshold αthre or
where significant depth discrepancies occur. These points
are initialized as anisotropic 3D Gaussians, with scales de-
fined based on the nearest neighbor distance within the cur-
rent submap. New 3D Gaussian splats are added to the cur-
rent submap only if there is no existing 3D Gaussian mean
within a radius ρ.
Submap Update.
After new Gaussians are added, all
Gaussians in the current submap are optimized for a fixed
number of iterations by minimizing the rendering loss
Lrender, computed over all keyframes of the submap, with
at least 40% of the compute allocated to the most recent
keyframe. The rendering loss is of three components: color
loss Lcolor, depth loss Ldepth, and a regularization term Lreg:
Lrender = λcolor · Lcolor + λdepth · Ldepth + λreg · Lreg,
(3)
where λ∗are hyperparamters. Similar to the tracking loss,
the depth loss is the L1 loss between rendered and ground
truth depth maps. For color supervision, we use a weighted
combination of the L1 and SSIM [84] loss:
Lcol = (1−λSSIM)·|ˆI−I|1+λSSIM
 1−SSIM(ˆI, I)

, (4)
where λSSIM ∈[0, 1]. To regularize overly elongated 3D
Gaussians in sparsely covered or barely observed regions,
we add an isotropic regularization term [48]
Lreg = 1
K
X
k∈K
|sk −sk|1,
(5)
where sk ∈R3 is the scale of a 3D Gaussian, sk is its mean,
and K is the number of Gaussians in the submap. During
optimization, to preserve geometry directly measured from
the depth sensor and reduce computation time, we do not
clone or prune the Gaussians [38].
3.2. Registration of Gaussian Splats
LoopSplat’s first contribution relates to the registration of
Gaussian splats which is formulated as following. Consider
two overlapping 3D Gaussian submaps P and Q, each re-
constructed using different keyframes and not aligned. The
goal is to estimate a rigid transformation TP→Q ∈SE(3)
that aligns P with Q. Each submap is also associated with
a set of viewpoints VP as:
VP = {vp
i = (I, D, T)i|i = 0, . . . , N},
(6)
where I and D are the individual RGB and depth measure-
ments, respectively, and T is the estimated camera pose
in Sec. 3.1.
Overlap Estimation.
Knowing the approximate overlap
between the source and target submaps P and Q is crucial
for robust and accurate registration, and this co-contextual
information can be extracted by comparing feature similari-
ties [32]. While the means of the Gaussian splats do form a
point cloud, we found that estimating the overlap region di-
rectly from them by matching local features does not work
well (cf. Sec. 4.5). Instead, we identify viewpoints from
each submap that share similar visual content. Specifically,
we first pass all keyframes through NetVLAD [2] to extract
their global descriptors. We then compute the cosine simi-
larity between the two sets of keyframes and retain the top-k
pairs for registration.
Registration as Keyframe Localization.
Given that the
3DGS submap and its viewpoints can be treated as one
rigid body, we propose to approach 3DGS registration as
a keyframe localization problem. For a selected viewpoint
vp
i , determining its camera pose Tq
i within Q allows one to
render the same RGB-D image from Q as vp
i . Hence, the
rigid transformation TP→Q can be computed as Tq
i · T−1
i .
During keyframe localization, we keep the parameters
of Q fixed and optimize the rigid transformation TP→Q
by minimizing the rendering loss L = Lcol + Ldepth [50],
where both Lcol and Ldepth are L1 losses.
We estimate the rigid transformations for the selected
viewpoints, from P to Q for viewpoints in VP and vice
versa for VQ, in parallel. The rendering residuals ϵ are also
saved upon completion of the optimization. By using the
sampled top-k viewpoints from the estimated overlap re-
gion as the selected viewpoints, the registration efficiency
is greatly improved without redundancy in non-overlapping
viewpoints. Viewpoint transformations are estimated first,
then used to compute the submap’s global transformation.
Multi-view Pose Refinement.
Given a set of transforma-
tions {(TP⇌Q, ε)i}2k
i=1, where the first k estimates are from
P →Q and the last k estimates from Q →P, one must
4

<!-- page 5 -->
find a global consensus for the transformation ¯TP→Q. As
the rendering residual indicates how well the transformed
viewpoint fits the original observation, we take the recipro-
cal of the residuals as a weight for each estimate and apply
weighted rotation averaging [6, 58] to compute the global
rotation:
¯R = arg min
R∈SO3
k
X
i=1
1
εi
∥R−Ri∥2
F +
2k
X
i=k+1
1
εi
∥R−R−1
i ∥2
F ,
(7)
where ∥· ∥2
F denotes the Frobenius norm. The global trans-
lation is found as the weighted mean over individual esti-
mates.
3.3. Loop Closure with 3DGS
Loop closure aims to identify pose corrections (i.e. rela-
tive transformations w.r.t. the current estimates) for past
submaps and keyframes to ensure global consistency. This
process is initiated when a new submap is created, and upon
detecting a new loop, the pose graph, which includes all his-
torical submaps, is constructed. The loop edge constraints
for the pose graph are then computed using 3DGS reg-
istration (Sec. 3.2). Subsequently, Pose Graph Optimiza-
tion (PGO) [13] is performed to achieve globally consistent
multi-way registration of 3DGS.
Loop Closure Detection.
To effectively detect system re-
visits to the same place, we first extract a global descrip-
tor d ∈R1024 using a pretrained NetVLAD [2]. We com-
pute the cosine similarities of all keyframes within the i-th
submap and determine the self-similarity score si
self corre-
sponding to their p-th percentile. We then apply the same
method to compute the cross-similarity si,j
cross between the
i-th and j-th submaps. A new loop is added if si,j
cross >
min(si
self, sj
self). However, relying solely on visual similar-
ity for loop closure [40] can generate false loop edges, po-
tentially degrading PGO performance. To mitigate that risk,
we additionally evaluate the initial geometric overlap ratio
r [32] between the Gaussians of two submaps, and retain
only loops with r > 0.2. See Supp. for more details.
Pose Graph Optimization.
We create a new pose graph
every time a new loop is detected and ensure that its con-
nections match the previous one, besides the new edges in-
troduced by the new submap. The relative pose corrections
{Tci ∈SE(3)} to each submap are defined as nodes in
the pose graph, which are connected with odometry edges
and loop edges. Here Tci denotes the correction applied
to i-th submap. The nodes and edges connecting adjacent
nodes (i.e., odometry edges) are initialized with identity ma-
trices. Loop edge constraints are added at detected loops
and initialized according to the Gaussian splatting regis-
tration (Sec. 3.2). The information matrices for edges are
Method
LC Rm 0 Rm 1 Rm 2 Off 0 Off 1 Off 2 Off 3 Off 4 Avg.
Neural Implicit Fields
NICE-SLAM [103]
✗
0.97
1.31
1.07
0.88
1.00
1.06
1.10
1.13
1.06
Vox-Fusion [90]
✗
1.37
4.70
1.47
8.48
2.04
2.58
1.11
2.94
3.09
ESLAM [33]
✗
0.71
0.70
0.52
0.57
0.55
0.58
0.72
0.63
0.63
Point-SLAM [63]
✗
0.61
0.41
0.37
0.38
0.48
0.54
0.69
0.72
0.52
MIPS-Fusion [77]
✓
1.10
1.20
1.10
0.70
0.80
1.30
2.20
1.10
1.19
GO-SLAM [101]
✓
0.34
0.29
0.29
0.32
0.30
0.39
0.39
0.46
0.35
Loopy-SLAM [40]
✓
0.24
0.24
0.28
0.26
0.40
0.29
0.22
0.35
0.29
3D Gaussian Splatting
SplaTAM [36]
✗
0.31
0.40
0.29
0.47
0.27
0.29
0.32
0.72
0.38
MonoGS [48]
✗
0.33
0.22
0.29
0.36
0.19
0.25
0.12
0.81
0.32
Gaussian-SLAM [95]
✗
0.29
0.29
0.22
0.37
0.23
0.41
0.30
0.35
0.31
∗Photo-SLAM [30]
✓
0.54
0.39
0.31
0.52
0.44
1.28
0.78
0.58
0.60
LoopSplat (Ours)
✓
0.28
0.22
0.17
0.22
0.16
0.49
0.20
0.30
0.26
Table 1. Tracking Performance on Replica [70] (ATE RMSE
↓[cm]). LC indicates loop closure. The best results are high-
lighted as first , second , and third . LoopSplat performs the
best.
∗Photo-SLAM [30] is a decoupled method using ORB-
SLAM3 [7] for tracking and loop closure.
Method
a
b
c
d
e
Avg.
Neural Implicit Fields
Point-SLAM [63]
246.16
632.99
830.79
271.42
574.86
511.24
ESLAM [33]
25.15
2.15
27.02
20.89
35.47
22.14
GO-SLAM [101]
176.28
145.45
38.74
85.48
106.47
110.49
Loopy-SLAM [33]
N/A
N/A
25.16
234.25
81.48
113.63
3D Gaussian Splatting
SplaTAM [36]
1.50
0.57
0.31
443.10
1.58
89.41
MonoGS [95]
7.00
3.66
6.37
3.28
44.09
12.88
Gaussian SLAM [95]
1.37
2.82
6.80
3.51
0.88
3.08
LoopSplat (Ours)
1.14
3.16
3.16
1.68
0.91
2.05
Table 2. Tracking Performance on ScanNet++ [93] (ATE RMSE
↓[cm]). LoopSplat achieves the highest accuracy and can robustly
deal with the large camera motions in the sequence.
computed directly from the Gaussian centers and incorpo-
rated into the pose graph. PGO is triggered after loop de-
tection and we use a robust formulation based on line pro-
cesses [13].
Globally Consistent Map Adjustment.
From the PGO
output, we obtain a set of pose corrections {Tci
=
[Rci|tci]}Ns
i=1 for Ns submaps, with ci denoting correction
for submap i. For each submap, we update camera poses,
the Gaussian means and covariances
Tj ←TciTj ,
(8)
µi ←RciµSi + tci, Σi ←RciΣSiRT
ci .
(9)
Here, µi and Σi represent the sets of centers and covariance
matrices, respectively, of the Gaussians in the i-th submap
Si, index j is iterated over the keyframe span of the submap.
We omit spherical harmonics (SH) to reduce the Gaussian
map size and improve pose estimation accuracy [48].
4. Experiments
Here we describe our experimental setup and compare our
method to state-of-the-art baselines. We evaluate tracking,
reconstruction, and rendering performance on synthetic and
5

<!-- page 6 -->
Method
00
59
106
169
181
207
54
233
Avg.
Neural Implicit Fields
Vox-Fusion [90]
16.6
24.2
8.4
27.3
23.3
9.4
-
-
-
Co-SLAM [83]
7.1
11.1
9.4
5.9
11.8
7.1
-
-
-
MIPS-Fusion [77]
7.9
10.7
9.7
9.7
14.2
7.8
-
-
-
NICE-SLAM [103]
12.0
14.0
7.9
10.9
13.4
6.2
20.9
9.0
13.0
ESLAM [33]
7.3
8.5
7.5
6.5
9.0
5.7
36.3
4.3
10.6
Point-SLAM [63]
10.2
7.8
8.7
22.2
14.8
9.5
28.0
6.1
14.3
GO-SLAM [101]
5.4
7.5
7.0
7.7
6.8
6.9
8.8
4.8
6.9
Loopy-SLAM [40]
4.2
7.5
8.3
7.5
10.6
7.9
7.5
5.2
7.7
3D Gaussian Splatting
MonoGS [48]
9.8
32.1
8.9
10.7
21.8
7.9
17.5
12.4
15.2
SplaTAM [36]
12.8
10.1
17.7
12.1
11.1
7.5
56.8
4.8
16.6
Gaussian-SLAM [95]
21.2
12.8
13.5
16.3
21.0
14.3
37.1
11.1
18.4
LoopSplat (Ours)
6.2
7.1
7.4
10.6
8.5
6.6
16.0
4.7
8.4
Table 3. Tracking Performance on ScanNet [17]. LoopSplat
outperforms 3DGS-based systems by a large margin and is on par
with the state-of-the-art baselines.
real-world datasets, with a dedicated ablation study for loop
closure. For implementation details, please refer to Supp.
Datasets.
We evaluate on four datasets: Replica [71] is
a synthetic dataset with high-quality 3D indoor reconstruc-
tions. We use the same RGB-D sequences as [75]. Scan-
Net [17] is a real-world dataset with its poses estimated by
BundleFusion [18]. We evaluate on eight scenes with loops
following [40, 101]. ScanNet++ [93] is a real, high-quality
dataset. We use five DSLR-captured sequences where poses
are estimated with COLMAP [67] and refined with the help
of laser scans. TUM-RGBD [73] is a real-world dataset with
accurate poses obtained from a motion capture system.
Baselines.
We compare LoopSplat with state-of-the-
art coupled RGB-D SLAM methods, categorized into
two groups based on the underlying scene represen-
tation:
(i) Neural implicit fields:
MIPS-Fusion [77],
GO-SLAM [101], and Loopy-SLAM [40], all of which in-
corporate loop closure; and (ii) 3DGS: MonoGS [48],
SplaTAM
[36],
Gaussian-SLAM
[95],
and
Photo-
SLAM [30].
For completeness,
we include Photo-
SLAM [30] in our evaluation, noting that it utilizes
ORB-SLAM3 [7] for tracking and loop closure, setting it
apart from all other tested methods.
Evaluation Metrics.
Tracking accuracy is measured by
the root mean square absolute trajectory error (ATE
RMSE) [73]. For reconstruction, we follow [63] and evalu-
ate via meshes extracted with marching cubes [42], using a
voxel size of 1 cm. We measure rendered mesh depth error
at sampled novel views as in [103] and the F1-score, i.e., the
harmonic mean of precision and recall w.r.t. ground truth
mesh vertices. Rendering quality is evaluated by compar-
ing full-resolution rendered images to input training views
in terms of PSNR, SSIM [84], and LPIPS [100]. We note
that comparing to training views may yield too optimistic
Method
LC
fr1/
fr1/
fr1/
fr2/
fr3/
Avg.
desk
desk2
room
xyz
off.
Neural Implicit Fields
DI-Fusion [31]
✗
4.4
N/A
N/A
2.0
5.8
N/A
NICE-SLAM [103]
✗
4.26
4.99
34.49
6.19
3.87
10.76
Vox-Fusion [90]
✗
3.52
6.00
19.53
1.49
26.01
11.31
MIPS-Fusion [77]
✓
3.0
N/A
N/A
1.4
4.6
N/A
Point-SLAM [63]
✗
4.34
4.54
30.92
1.31
3.48
8.92
ESLAM [33]
✗
2.47
3.69
29.73
1.11
2.42
7.89
Co-SLAM [83]
✗
2.40
N/A
N/A
1.70
2.40
N/A
GO-SLAM [101]
✓
1.50
N/A
4.64
0.60
1.30
N/A
Loopy-SLAM [40]
✓
3.79
3.38
7.03
1.62
3.41
3.85
3D Gaussian Splatting
SplaTAM [36]
✗
3.35
6.54
11.13
1.24
5.16
5.48
MonoGS [48]
✗
1.59
7.03
8.55
1.44
1.49
4.02
Gaussian-SLAM [95]
✗
2.73
6.03
14.92
1.39
5.31
6.08
∗Photo-SLAM [30]
✓
2.60
N/A
N/A
0.35
1.00
N/A
LoopSplat (Ours)
✓
2.08
3.54
6.24
1.58
3.22
3.33
Classical
BAD-SLAM [68]
✓
1.7
N/A
N/A
1.1
1.7
N/A
Kintinuous [86]
✓
3.7
7.1
7.5
2.9
3.0
4.84
ORB-SLAM2 [52]
✓
1.6
2.2
4.7
0.4
1.0
1.98
ElasticFusion [87]
✓
2.53
6.83
21.49
1.17
2.52
6.91
BundleFusion [18]
✓
1.6
N/A
N/A
1.1
2.2
N/A
Cao et al. [8]
✓
1.5
N/A
N/A
0.6
0.9
N/A
Yan et al. [89]
✓
1.6
N/A
5.1
N/A
3.1
N/A
Table 4.
Tracking Performance on TUM-RGBD [74] (ATE
RMSE ↓[cm]).
∗indicates using ORB-SLAM3 [7] for tracking
and loop closure. LoopSplat performs the best among coupled
SLAM, further closing the gap to sparse solver-based SLAM.
results, but it enables a consistent comparison with existing
methods. To assess map size, we measure the total mem-
ory needed for the map and the peak GPU memory usage.
Runtime is reported as average per-frame tracking and map
optimization time, as well as loop edge registration runtime.
4.1. Tracking
We report the camera tracking performance in Tabs. 1 to 4.
On Replica, we outperform all the baselines, achieving a
10% higher accuracy compared to the second best one. On
real-world datasets, we achieve the highest pose accuracy
on TUM-RGBD and ScanNet++ among all neural implicit
field-based and 3DGS-based baselines, improving tracking
accuracy by 14% and 33%, respectively. It is worth not-
ing that, for all 3DGS-based baselines [36, 48, 95], tra-
jectory errors accumulate as trajectories grow longer in
larger scenes with loops and motion blur, e.g., ScanNet 00,
59, 181 and TUM-RGBD fr1/desk2 and fr1/room.
We attribute our superior tracking performance to the ro-
bust 3DGS registration that underpins our loop closure.
On ScanNet, we obtain the third-best performance.
We
note that the ground truth poses in ScanNet, derived from
BundleFusion [18], appear to have limited accuracy: visual
inspection suggests that our method achieves better align-
ment and reconstruction than the ground truth; see Fig. 1,
Fig. 3, and scene 233 in Fig. G.3. Additional qualitative
examples are in Supp. Besides superior tracking accuracy,
our coupled method avoids redundant computations for sep-
arate tracking and map reconstruction, in contrast to decou-
6

<!-- page 7 -->
Scene 54
Scene 233
Gaussian-SLAM [95]
Ours
Ground Truth
Figure 3. Comparison of Submap Alignment on ScanNet [17].
We visualize the centers of 3D Gaussians as point clouds. Two
submaps are colorized differently. LoopSplat consistently aligns
the submaps better than Gaussian-SLAM [95].
pled ones like GO-SLAM [101] and Photo-SLAM [30].
4.2. Reconstruction
We evaluate the mesh reconstruction quality on Replica,
the only dataset with high accuracy ground truth mesh, in
Tab. 51. LoopSplat outperforms all 3DGS-based baselines
attributed to more accurate pose estimates. LoopSplat falls
behind Loopy-SLAM [40] and Point-SLAM [63], but note
that the latter two require ground truth depth to determine
where to sample points during ray-marching, thus assuming
perfect input depth. Fig. 4 compares ScanNet meshes re-
constructed with LoopSplat to those of the best-performing
baselines, GO-SLAM and Loopy-SLAM (both also includ-
ing loop closure), as well as to a 3DGS baseline, Gaussian-
SLAM (which does not perform loop closure). Our method
recovers more geometric details (e.g., on the chairs). On
ScanNet 233, the visual quality and completeness of our
reconstruction appears even better than the ground truth, es-
pecially on the floor, desk and bed.
4.3. Rendering
Tab. 6 reports our rendering performance on training views.
To conduct a fair comparison, we merge all the submaps
into a global one and optimize the global map with esti-
mated cameras pose, to avoid local overfitting on submaps2.
LoopSplat surpasses all competing methods in terms of
PSNR and LPIPS on Replica and ScanNet, and is compet-
itive with SplaTAM on TUM-RGBD. Note the significant
1∗Depth L1 for GO-SLAM is based on results reproduced by [40] using
random poses, as GO-SLAM originally evaluates on ground truth poses.
2Gaussian-SLAM evaluates rendering on local submaps.
Method
Metric
Rm 0 Rm 1 Rm 2 Off 0 Off 1 Off 2 Off 3 Off 4 Avg.
Neural Implicit Fields
NICE-
SLAM [103]
Depth L1 [cm] ↓
1.81 1.44 2.04
1.39
1.76
8.33
4.99
2.01
2.97
F1 [%] ↑
45.0 44.8 43.6
50.0
51.9
39.2
39.9
36.5
43.9
Vox-
Fusion [90]
Depth L1 [cm] ↓
1.09 1.90 2.21
2.32
3.40
4.19
2.96
1.61
2.46
F1 [%] ↑
69.9 34.4 59.7
46.5
40.8
51.0
64.6
50.7
52.2
ESLAM [33]
Depth L1 [cm] ↓
0.97 1.07 1.28
0.86
1.26
1.71
1.43
1.06
1.18
F1 [%] ↑
81.0 82.2 83.9
78.4
75.5
77.1
75.5
79.1
79.1
Co-SLAM [83]
Depth L1 [cm] ↓
1.05 0.85 2.37
1.24
1.48
1.86
1.66
1.54
1.51
GO-
SLAM [101]
Depth L1 [cm] ↓
-
-
-
-
-
-
-
-
3.38
∗Depth L1 [cm] ↓4.56 1.97 3.43
2.47
3.03
10.3
7.31
4.34
4.68
F1 [%] ↑
17.3 33.4 24.0
43.0
31.8
21.8
17.3
22.0
26.3
Point-
SLAM [63]
Depth L1 [cm] ↓
0.53 0.22 0.46
0.30
0.57
0.49
0.51
0.46
0.44
F1 [%] ↑
86.9 92.3 90.8
93.8
91.6
89.0
88.2
85.6
89.8
Loopy-SLAM [40]
Depth L1 [cm] ↓
0.30 0.20 0.42
0.23
0.46
0.60
0.37
0.24
0.35
F1 [%] ↑
91.6 92.4 90.6
93.9
91.6
88.5
89.0
88.7
90.8
3D Gaussian Splatting
SplaTAM [36]
Depth L1 [cm]↓
0.43 0.38 0.54
0.44
0.66
1.05
1.60
0.68
0.72
F1 [%]↑
89.3 88.2 88.0
91.7
90.0
85.1
77.1
80.1
86.1
Gaussian SLAM [95] Depth L1 [cm]↓
0.61 0.25 0.54
0.50
0.52
0.98
1.63
0.42
0.68
F1 [%]↑
88.8 91.4 90.5
91.7
90.1
87.3
84.2
87.4
88.9
LoopSplat (Ours)
Depth L1 [cm] ↓
0.39 0.23 0.52
0.32
0.51
0.63
1.09
0.40
0.51
F1 [%] ↑
90.6 91.9 91.1
93.3
90.4
88.9
88.7
88.3
90.4
Table 5. Reconstruction Performance on Replica [70]. Loop-
Splat obtains the second-best F1-score, falling behind only to
Loopy-SLAM. It is noteworthy that both the NeRF-based Loopy-
SLAM and Point-SLAM methods require ground truth depth in-
put to guide the depth rendering, whereas our method, leveraging
3DGS, only requires estimated camera poses at rendering time.
Dataset
Replica [70]
TUM [74]
ScanNet [17]
Method
PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓PSNR ↑SSIM ↑LPIPS ↓
NICE-SLAM [103]
24.42
0.892
0.233
14.86
0.614
0.441
17.54
0.621
0.548
Vox-Fusion [90]
24.41
0.801
0.236
16.46
0.677
0.471
18.17
0.673
0.504
ESLAM [33]
28.06
0.923
0.245
15.26
0.478
0.569
15.29
0.658
0.488
Point-SLAM [63]
35.17
0.975
0.124
16.62
0.696
0.526
19.82
0.751
0.514
Loopy-SLAM [40]
35.47
0.981
0.109
12.94
0.489
0.645
15.23
0.629
0.671
SplaTAM [36]
34.11
0.970
0.100
22.80
0.893
0.178
19.14
0.716
0.358
Gaussian-SLAM [95]
42.08
0.996
0.018
25.05
0.929
0.168
27.67
0.923
0.248
LoopSplat (Ours)
36.63
0.985
0.112
22.72
0.873
0.259
24.92
0.845
0.425
Table 6.
Rendering Performance on 3 Datasets.
LoopSplat
achieves competitive results on synthetic and real-world datasets.
Gray indicates evaluation on submaps instead of a global map.
margin over baselines that employ implicit neural represen-
tations. We report the per-scene rendering results in Supp.
4.4. Memory and Runtime Analysis
Tab. 7 profiles the runtime and memory usage of LoopSplat.
While our per-frame tracking and map optimization time
falls behind the fastest baselines, our Gaussian Splatting-
based registration significantly shortens the loop edge reg-
istration time compared to Loopy-SLAM. Through careful
control of submap growth, our Gaussian splats embedding
size is 8× smaller than that of the 3DGS baseline SplaTAM.
Additionally, we require the least GPU memory to process a
room-sized scene. In contrast, baselines like ESLAM, GO-
SLAM or SplaTAM require >15 GB of GPU memory.
4.5. Ablations
We first demonstrate that straightforward point cloud regis-
tration is not suitable to derive loop edge constraints from
7

<!-- page 8 -->
Scene 181
Scene 233
GO-SLAM [101]
Gaussian-SLAM [95]
Loopy-SLAM [40]
LoopSplat (Ours)
Ground Truth
Figure 4. Comparison of Mesh Reconstruction on two ScanNet [17] scenes. For the first scene, we highlight shape details with normal
shading, showing that LoopSplat yields the best geometry (e.g. the chairs). For the second one, we display renderings of the colored mesh.
Note the distortions at the desk in ground truth that are not present in ours, indicating accuracy limitations of ScanNet ground truth poses.
Method
Tracking
Mapping
Registration
Embedding
Peak GPU
/Frame(s) ↓
/Frame(s) ↓
/Edge(s) ↓
Size(MiB)↓
Use(GiB)↓
NICE-SLAM [103]
1.06
1.15
-
95.9
12.0
Vox-Fusion [90]
1.92
1.47
-
0.15
17.6
Point-SLAM [63]
1.11
3.52
-
27.2
7.7
ESLAM [43]
0.15
0.62
-
45.5
17.3
GO-SLAM [101]
0.125
-
48.1
18.4
SplaTAM [36]
2.70
4.89
-
404.5
18.5
Loopy-SLAM[40]
1.11
3.52
12.0
60.9
9.3
LoopSplat (Ours)
0.83
0.93
1.36
49.7
7.0
Table 7. Runtime and Memory Usage on Replica office 0.
Per-frame runtime is calculated as the total optimization time di-
vided by the sequence length, profiled on a RTX A6000 GPU.
The embedding size is the total memory of the map representation.
Note that implicit field-based methods require additional space for
their decoders. We take runtime values from [95] and embedding
values from [40] for the baselines.
3DGS. To illustrate this, we replace the proposed 3DGS
registration in our SLAM system with FPFH+ICP [102] and
evaluate the trajectory error (ATE) on Replica. As shown in
the last row of Tab. 8, FPFH+ICP applied directly to the
center points of 3D Gaussians leads to less accurate loop
edges compared to our method and deteriorates loop clo-
sure. We hypothesize that this is because the center points
do not accurately represent the scene surfaces, as previously
discussed in [29, 94, 97]. Furthermore, the pre-processing
of [102] involves re-rendering and back-projecting 3DGS to
obtain 3D points, downsampling the point clouds and vox-
elizing them. This heavy pre-processing makes [40] more
than 8× slower than our method. In contrast, LoopSplat ef-
Mul. Opt.
Ove. Est.
Rot. Ave.
ATE (cm)
Runtime (s)
✗
✗
✗
0.31
-
✗
✓
✗
0.31
1.25
✓
✓
✗
0.27
1.36
✓
✗
✓
0.37
11.02
✓
✓
✓
0.26
1.36
FPFH+ICP [102]
0.40
12.0
Table 8.
Ablation Study on 3DGS Registration.
The num-
bers are computed based on average performance of 8 scenes on
Replica [71]. Mul. Opt. denotes multi-view optimization, Ove.
Est. and Rot. Ave. denote view selection and rotation averaging.
ficiently reuses the native map representation without any
pre-processing, answering the research question we asked
in Sec. 1. We also explore the impact of different mod-
ules in our registration method. The ablation study confirms
that every component contributes to the final performance:
Multi-view optimization and rotation averaging greatly im-
prove registration accuracy by fusing information from dif-
ferent viewpoints. View selection via overlap estimation
(Sec. 3.2) is crucial to identify informative viewpoints and
ensure the efficiency of the SLAM system.
5. Conclusion
We presented LoopSplat, a novel dense RGB-D SLAM sys-
tem that exclusively uses 3D Gaussian Splats for scene
representation, achieving global consistency through loop
closure. Built around 3DGS submaps, LoopSplat enables
dense mapping, frame-to-model tracking, and online loop
closure via direct 3DGS submap registration. Comprehen-
8

<!-- page 9 -->
sive evaluation on four datasets shows competitive or supe-
rior performance in tracking, mapping, and rendering. We
discuss limitations and future work in Supp.
References
[1] Sheng Ao, Qingyong Hu, Hanyun Wang, Kai Xu, and Yulan
Guo. Buffer: Balancing accuracy, efficiency, and general-
izability in point cloud registration. In CVPR, 2023. 2
[2] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pa-
jdla, and Josef Sivic. Netvlad: Cnn architecture for weakly
supervised place recognition. In CVPR, 2016. 4, 5, 2
[3] Xuyang Bai, Zixin Luo, Lei Zhou, Hongbo Fu, Long Quan,
and Chiew-Lan Tai. D3feat: Joint learning of dense detec-
tion and description of 3d local features. In CVPR, 2020.
2
[4] Paul J Besl and Neil D McKay. Method for registration
of 3-d shapes. In Sensor fusion IV: control paradigms and
data structures, 1992. 2
[5] Michael Bosse, Paul Newman, John Leonard, Martin Soika,
Wendelin Feiten, and Seth Teller. An atlas framework for
scalable mapping. In ICRA, 2003. 2
[6] Romain Br´egier. Deep regression on manifolds: a 3D rota-
tion case study. 2021. 5
[7] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez,
Jos´e MM Montiel, and Juan D Tard´os. Orb-slam3: An ac-
curate open-source library for visual, visual–inertial, and
multimap slam. IEEE Transactions on Robotics, 2021. 5, 6
[8] Yan-Pei Cao, Leif Kobbelt, and Shi-Min Hu.
Real-time
high-accuracy three-dimensional reconstruction with con-
sumer rgb-d cameras. ACM TOG, 2018. 1, 2, 6
[9] Jiahao Chang, Yinglin Xu, Yihao Li, Yuantao Chen, and Xi-
aoguang Han. Gaussreg: Fast 3d registration with gaussian
splatting. In ECCV, 2024. 3
[10] Jiawen Chen, Dennis Bautembach, and Shahram Izadi.
Scalable real-time volumetric surface reconstruction. ACM
TOG, 2013. 2
[11] Yu Chen and Gim Hee Lee. Dreg-nerf: Deep registration
for neural radiance fields. In ICCV, 2023. 3
[12] Hae Min Cho, HyungGi Jo, and Euntai Kim.
Sp-
slam: Surfel-point simultaneous localization and mapping.
IEEE/ASME Transactions on Mechatronics, 2021. 2
[13] Sungjoon Choi, Qian-Yi Zhou, and Vladlen Koltun. Robust
reconstruction of indoor scenes. In CVPR, 2015. 2, 3, 5
[14] Christopher Choy, Jaesik Park, and Vladlen Koltun. Fully
convolutional geometric features. In ICCV, 2019. 2
[15] Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-
Qian Shi, Yun-Hung Hua, Jia-Fong Yeh, Wen-Chin Chen,
Yi-Ting Chen, and Winston H Hsu. Orbeez-slam: A real-
time monocular visual slam with orb features and nerf-
realized mapping. arXiv preprint arXiv:2209.13274, 2022.
1, 2
[16] Brian Curless and Marc Levoy.
Volumetric method for
building complex models from range images.
In SIG-
GRAPH Conference on Computer Graphics, 1996. 2
[17] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes. In
CVPR, 2017. 1, 6, 7, 8, 2, 4
[18] Angela Dai, Matthias Nießner, Michael Zollh¨ofer, Shahram
Izadi, and Christian Theobalt.
Bundlefusion: Real-time
globally consistent 3d reconstruction using on-the-fly sur-
face reintegration. ACM TOG, 2017. 1, 2, 3, 6
[19] Felix Endres, J¨urgen Hess, Nikolas Engelhard, J¨urgen
Sturm, Daniel Cremers, and Wolfram Burgard. An eval-
uation of the rgb-d slam system. In ICRA, 2012. 2
[20] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers. Lsd-
slam: Large-scale direct monocular slam. In ECCV, 2014.
2
[21] Francis Engelmann, Fabian Manhardt, Michael Niemeyer,
Keisuke Tateno, Marc Pollefeys, and Federico Tombari.
Opennerf: Open set 3d neural scene segmentation with
pixel-wise features and rendered novel views.
arXiv
preprint arXiv:2404.03650, 2024. 2
[22] Nicola Fioraio, Jonathan Taylor, Andrew Fitzgibbon, Luigi
Di Stefano, and Shahram Izadi. Large-scale and drift-free
surface reconstruction using online subvolume registration.
In CVPR, 2015. 2
[23] Zan Gojcic, Caifa Zhou, Jan D Wegner, and Andreas
Wieser. The perfect match: 3d point cloud matching with
smoothed densities. In CVPR, 2019. 2
[24] Lily Goli, Daniel Rebain, Sara Sabour, Animesh Garg, and
Andrea Tagliasacchi.
nerf2nerf: Pairwise registration of
neural radiance fields. In ICRA, 2023. 3
[25] Antoine Gu´edon and Vincent Lepetit.
Sugar: Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering. CVPR, 2024. 4
[26] Peter Henry, Michael Krainin, Evan Herbst, Xiaofeng Ren,
and Dieter Fox. Rgb-d mapping: Using kinect-style depth
cameras for dense 3d modeling of indoor environments.
The International Journal of Robotics Research, 2012. 2
[27] Peter Henry, Dieter Fox, Achintya Bhowmik, and Rajiv
Mongia.
Patch volumes: Segmentation-based consistent
mapping with rgb-d cameras. In 3DV, 2013. 2
[28] Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, and
Zhaopeng Cui.
CP-SLAM: Collaborative neural point-
based SLAM system. In NeurIPS, 2023. 2
[29] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In SIGGRAPH, 2024. 8
[30] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Ye-
ung.
Photo-slam:
Real-time simultaneous localization
and photorealistic mapping for monocular stereo and rgb-
d cameras. In CVPR, 2024. 1, 2, 5, 6, 7
[31] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and Shi-
Min Hu. Di-fusion: Online implicit 3d reconstruction with
deep priors. In CVPR, 2021. 6
[32] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas
Wieser, and Konrad Schindler. Predator: Registration of 3d
point clouds with low overlap. In CVPR, 2021. 2, 4, 5
[33] M. M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient
dense slam system based on hybrid representation of signed
distance fields. In CVPR, 2023. 1, 5, 6, 7, 3, 4
9

<!-- page 10 -->
[34] Olaf K¨ahler, Victor Adrian Prisacariu, Carl Yuheng Ren,
Xin Sun, Philip H. S. Torr, and David William Murray. Very
high frame rate volumetric integration of depth images on
mobile devices. IEEE Trans. Vis. Comput. Graph., 2015. 2
[35] Olaf K¨ahler, Victor A Prisacariu, and David W Murray.
Real-time large-scale dense 3d reconstruction with loop
closure. In ECCV, 2016. 2
[36] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallab-
hula, Gengshan Yang, Sebastian Scherer, Deva Ramanan,
and Jonathon Luiten. Splatam: Splat, track & map 3d gaus-
sians for dense rgb-d slam. CVPR, 2024. 1, 2, 5, 6, 7, 8, 3,
4
[37] Maik Keller, Damien Lefloch, Martin Lambers, Shahram
Izadi, Tim Weyrich, and Andreas Kolb. Real-time 3d re-
construction in dynamic scenes using point-based fusion.
In International Conference on 3D Vision (3DV), 2013. 2
[38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM TOG, 2023. 1, 3, 4, 2
[39] Christian Kerl, J¨urgen Sturm, and Daniel Cremers. Dense
visual slam for rgb-d cameras. In IROS, 2013. 2
[40] Lorenzo Liso, Erik Sandstr¨om, Vladimir Yugay, Luc Van
Gool, and Martin R. Oswald. Loopy-slam: Dense neural
slam with loop closures. In CVPR, 2024. 1, 2, 3, 5, 6, 7, 8
[41] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and
Christian Theobalt. Neural sparse voxel fields. NeurIPS,
2020. 2
[42] William E Lorensen and Harvey E Cline. Marching cubes:
A high resolution 3d surface construction algorithm. ACM
siggraph computer graphics, 1987. 6
[43] Mohammad Mahdi Johari, Camilla Carta, and Franc¸ois
Fleuret. Eslam: Efficient dense slam system based on hy-
brid representation of signed distance fields. pages arXiv–
2211, 2022. 1, 8, 3
[44] Robert Maier, J¨urgen Sturm, and Daniel Cremers. Submap-
based bundle adjustment for 3d reconstruction from rgb-d
data. In Pattern Recognition: 36th German Conference,
GCPR 2014, M¨unster, Germany, September 2-5, 2014, Pro-
ceedings 36, 2014. 2, 3
[45] R Maier, R Schaller, and D Cremers. Efficient online sur-
face correction for real-time large-scale 3d reconstruction.
arxiv 2017. arXiv preprint arXiv:1709.03763, 2017. 2
[46] Yunxuan Mao, Xuan Yu, Kai Wang, Yue Wang, Rong
Xiong, and Yiyi Liao.
Ngel-slam:
Neural implicit
representation-based global consistent low-latency slam
system. arXiv preprint arXiv:2311.09525, 2023. 1, 2
[47] Nico Marniok, Ole Johannsen, and Bastian Goldluecke.
An efficient octree design for local variational range im-
age fusion. In German Conference on Pattern Recognition
(GCPR), 2017. 2
[48] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. arXiv preprint
arXiv:2312.06741, 2023. 1, 2, 4, 5, 6
[49] Hidenobu Matsuki, Keisuke Tateno, Michael Niemeyer,
and Federic Tombari.
Newton:
Neural view-centric
mapping for on-the-fly large-scale slam.
arXiv preprint
arXiv:2303.13654, 2023. 1, 2
[50] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. CVPR, 2024. 4
[51] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2021. 2
[52] Raul Mur-Artal and Juan D. Tardos. ORB-SLAM2: An
Open-Source SLAM System for Monocular, Stereo, and
RGB-D Cameras. IEEE Transactions on Robotics, 2017.
2, 6
[53] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Push-
meet Kohli, Jamie Shotton, Steve Hodges, and Andrew W
Fitzgibbon. Kinectfusion: Real-time dense surface map-
ping and tracking. In ISMAR, 2011. 1, 2
[54] Matthias Nießner, Michael Zollh¨ofer, Shahram Izadi, and
Marc Stamminger. Real-time 3d reconstruction at scale us-
ing voxel hashing. ACM TOG, 2013. 1, 2
[55] Helen Oleynikova, Zachary Taylor, Marius Fehr, Roland
Siegwart, and Juan I. Nieto. Voxblox: Incremental 3d eu-
clidean signed distance fields for on-board MAV planning.
In IROS, 2017. 2
[56] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley,
and C. Stachniss.
PIN-SLAM: LiDAR SLAM Using a
Point-Based Implicit Neural Representation for Achieving
Global Map Consistency. IEEE Transactions on Robotics
(TRO), 2024. 2
[57] Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Colored
point cloud registration revisited. In ICCV, pages 143–152,
2017. 1
[58] Valentin Peretroukhin, Matthew Giamou, David M Rosen,
W Nicholas Greene, Nicholas Roy, and Jonathan Kelly.
A smooth representation of belief over so (3) for
deep rotation learning with uncertainty.
arXiv preprint
arXiv:2006.01031, 2020. 5
[59] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing
Peng, Slobodan Ilic, Dewen Hu, and Kai Xu. Geotrans-
former: Fast and robust point cloud registration with geo-
metric transformer. IEEE TPAMI, 2023. 2
[60] Victor Reijgwart, Alexander Millane, Helen Oleynikova,
Roland Siegwart, Cesar Cadena, and Juan Nieto. Voxgraph:
Globally consistent, volumetric mapping using signed dis-
tance function submaps. IEEE Robotics and Automation
Letters, 2019. 2
[61] Antoni Rosinol, John J. Leonard, and Luca Carlone. NeRF-
SLAM: Real-Time Dense Monocular SLAM with Neural
Radiance Fields. arXiv, 2022. 1
[62] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast
point feature histograms (FPFH) for 3d registration.
In
ICRA, 2009. 2
[63] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R Os-
wald. Point-slam: Dense neural point cloud-based slam. In
ICCV, 2023. 1, 2, 5, 6, 7, 8, 3, 4
[64] Erik Sandstr¨om, Kevin Ta, Luc Van Gool, and Martin R
Oswald. Uncle-slam: Uncertainty learning for dense neural
slam. In Int. Conf. Comput. Vis. Worksh., 2023. 1
10

<!-- page 11 -->
[65] Erik
Sandstr¨om,
Keisuke
Tateno,
Michael
Oechsle,
Michael Niemeyer, Luc Van Gool, Martin R Oswald,
and Federico Tombari.
Splat-slam:
Globally opti-
mized rgb-only slam with 3d gaussians.
arXiv preprint
arXiv:2405.16544, 2024. 2
[66] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and
Marcin Dymczyk. From coarse to fine: Robust hierarchical
localization at large scale. In CVPR, 2019. 2
[67] Johannes Lutz Sch¨onberger and Jan-Michael Frahm.
Structure-from-motion revisited. In CVPR, 2016. 6
[68] Thomas Schops, Torsten Sattler, and Marc Pollefeys. BAD
SLAM: Bundle adjusted direct RGB-D SLAM. In CVPR,
2019. 1, 2, 6
[69] Frank Steinbrucker, Christian Kerl, and Daniel Cremers.
Large-scale multi-resolution surface reconstruction from
rgb-d sequences. In ICCV, 2013. 2
[70] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen,
Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-
Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei
Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon,
Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales,
Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis
Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi,
Michael Goesele, Steven Lovegrove, and Richard New-
combe. The Replica dataset: A digital replica of indoor
spaces. arXiv preprint arXiv:1906.05797, 2019. 5, 7, 2, 3
[71] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen,
Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-
Artal, Carl Ren, Shobhit Verma, et al.
The replica
dataset: A digital replica of indoor spaces. arXiv preprint
arXiv:1906.05797, 2019. 6, 8, 1
[72] J¨org St¨uckler and Sven Behnke.
Multi-resolution surfel
maps for efficient dense 3d modeling and tracking. Journal
of Visual Communication and Image Representation, 2014.
2
[73] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalu-
ation of RGB-D SLAM systems. In International Confer-
ence on Intelligent Robots and Systems (IROS), 2012. 6,
1
[74] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalu-
ation of rgb-d slam systems. In IROS, 2012. 6, 7, 2, 3
[75] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J.
Davison. iMAP: Implicit Mapping and Positioning in Real-
Time. In ICCV, 2021. 1, 6
[76] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davi-
son. imap: Implicit mapping and positioning in real-time.
In ICCV, 2021. 2
[77] Yijie Tang, Jiazhao Zhang, Zhinan Yu, He Wang, and Kai
Xu. Mips-fusion: Multi-implicit-submaps for scalable and
robust online neural rgb-d reconstruction. ACM TOG, 2023.
5, 6
[78] Yijie Tang, Jiazhao Zhang, Zhinan Yu, He Wang, and Kai
Xu. Mips-fusion: Multi-implicit-submaps for scalable and
robust online neural rgb-d reconstruction. arXiv preprint
arXiv:2308.08741, 2023. 1, 2
[79] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras. NeurIPS, 2021.
2
[80] Federico Tombari, Samuele Salti, and Luigi Di Stefano.
Unique signatures of histograms for local surface descrip-
tion. In ECCV, 2010. 2
[81] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,
and Illia Polosukhin. Attention is all you need. In NeurIPS,
2017. 2
[82] Hao Wang, Jun Wang, and Wang Liang. Online reconstruc-
tion of indoor scenes from rgb-d streams. In CVPR, 2016.
2
[83] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-
slam: Joint coordinate and sparse parametric encodings for
neural real-time slam. In CVPR, 2023. 1, 6, 7
[84] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. IEEE transactions on image pro-
cessing, 2004. 4, 6
[85] Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang,
Jacob Munkberg, Jon Hasselgren, Zan Gojcic, Wenzheng
Chen, and Sanja Fidler. Neural fields meet explicit geomet-
ric representations for inverse rendering of urban scenes. In
CVPR, 2023. 2
[86] Thomas Whelan, John McDonald, Michael Kaess, Maurice
Fallon, Hordur Johannsson, and John J. Leonard. Kintin-
uous: Spatially extended kinectfusion. In Proceedings of
RSS ’12 Workshop on RGB-D: Advanced Reasoning with
Depth Cameras, 2012. 6
[87] Thomas Whelan,
Stefan Leutenegger,
Renato Salas-
Moreno, Ben Glocker, and Andrew Davison. Elasticfusion:
Dense slam without a pose graph. In Robotics: Science and
Systems (RSS), 2015. 1, 2, 6
[88] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang,
Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam
with 3d gaussian splatting. In CVPR, 2024. 1, 2
[89] Zhixin Yan, Mao Ye, and Liu Ren. Dense visual slam with
probabilistic surfel map. IEEE TVCG, 2017. 2, 6
[90] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian
Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and
mapping with voxel-based neural implicit representation. In
IEEE International Symposium on Mixed and Augmented
Reality (ISMAR), 2022. 1, 2, 5, 6, 7, 8, 3, 4
[91] Xingrui Yang, Yuhang Ming, Zhaopeng Cui, and Andrew
Calway.
Fd-slam: 3-d reconstruction using features and
dense matching.
In 2022 International Conference on
Robotics and Automation (ICRA), 2022. 2
[92] Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Alberto
Rodriguez, Phillip Isola, and Tsung-Yi Lin. iNeRF: Invert-
ing neural radiance fields for pose estimation. In (IROS),
2021. 3
[93] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d
indoor scenes. In ICCV, 2023. 5, 6, 1, 3
[94] Zehao Yu, Torsten Sattler, and Andreas Geiger.
Gaus-
sian opacity fields: Efficient high-quality compact surface
11

<!-- page 12 -->
reconstruction in unbounded scenes.
arXiv:2404.10772,
2024. 8, 4
[95] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting. arXiv preprint arXiv:2312.10070, 2023. 1,
2, 3, 5, 6, 7, 8, 4
[96] Andy Zeng, Shuran Song, Matthias Nießner, Matthew
Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3dmatch:
Learning local geometric descriptors from rgb-d recon-
structions. In CVPR, 2017. 2
[97] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun
Liang, Xiaoxiao Long, and Ping Tan.
Rade-gs:
Ras-
terizing depth in gaussian splatting.
arXiv preprint
arXiv:2406.01467, 2024. 8
[98] Ganlin Zhang, Erik Sandstr¨om, Youmin Zhang, Manthan
Patel, Luc Van Gool, and Martin R Oswald. Glorie-slam:
Globally optimized rgb-only implicit encoding point cloud
slam. arXiv preprint arXiv:2403.19549, 2024. 2
[99] Heng Zhang, Guodong Chen, Zheng Wang, Zhenhua Wang,
and Lining Sun. Dense 3d mapping for indoor environment
based on feature-point slam method. In 2020 the 4th In-
ternational Conference on Innovation in Artificial Intelli-
gence, 2020. 2
[100] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In CVPR, 2018. 6
[101] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d in-
stant reconstruction. In ICCV, 2023. 1, 2, 5, 6, 7, 8
[102] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Fast global
registration. In ECCV, 2016. 8
[103] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu,
Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc
Pollefeys.
Nice-slam: Neural implicit scalable encoding
for slam. In CVPR, 2022. 1, 5, 6, 7, 8, 3, 4
12

<!-- page 13 -->
LoopSplat: Loop Closure by Registering 3D Gaussian Splats
Supplementary Material
Abstract
This supplementary material includes a video of Loop-
Splat running on a multi-room scene, showcasing the ef-
fectiveness of the online loop closure module of Loop-
Splat. Furthermore, we provide the implementation details
and statistics on loop closure and pose graph optimization
(PGO). We also present more qualitative results and abla-
tion studies. Lastly, we discuss the limitations and future
work.
A. Video
We submit a video loopsplat 0054.mp4, demonstrat-
ing LoopSplat’s online tracking and reconstruction capabil-
ities on ScanNet [17] scene0054. This video showcases
the effectiveness of our globally consistent reconstruction
process. The visualization includes the reconstructed mesh,
the colorized camera trajectory that denotes average trans-
lation error from the ground truth trajectory – see heatmap
legend on the right, and the point cloud observed from the
current frame colored in blue. As the camera completes its
scan of the first room, one can clearly observe the signif-
icant improvements achieved through loop closure. While
substantial drift occurs in the bathroom and storage room
(the leftmost room), the online loop closure (LC) mod-
ule in LoopSplat successfully corrects the accumulated er-
ror when the camera revisits the first room at the end of
the video. This correction highlights the robustness of our
method in maintaining global consistency throughout the
reconstruction process.
B. Implementation Details
Hyperparameters. Tab. B.1 lists the hyperparameters used
in our system, including λc in the tracking loss, learning
rates lr for rotation and lt for translation, and the number of
optimization iterations itert for tracking and iterm for map-
ping on the reported Replica [71], TUM-RGBD [73], Scan-
Net [17], and ScanNet++ [93] datasets. Additionally we set
λcolor, λdepth, and λreg to 1 in the mapping loss Lrender for all
datasets.
Params
Replica
TUM-RGBD
ScanNet
ScanNet++
λc
0.95
0.6
0.6
0.5
lr
0.0002
0.002
0.002
0.002
lt
0.002
0.01
0.01
0.01
itert
60
200
200
300
iterm
100
100
100
500
Table B.1. Per-dataset Hyperparameters.
Submap Initialization. A new submap is triggered based
on motion heuristics with the displacement threshold dthre =
0.5 [m] and rotation threshold θthre = 50◦. For the ScanNet
and ScanNet++ datasets, we adopted a different approach
to submap initialization. Motion heuristics were not em-
ployed, primarily due to two factors: significant motion blur
in ScanNet and substantial per-frame motion in ScanNet++
(cf. Tab. C.1). Instead, we implemented a fixed interval
system for triggering new submaps. Specifically, we set in-
tervals of 50 frames for ScanNet and 100 for ScanNet++.
Frame-to-model Tracking. Instead of estimating the cur-
rent camera pose Tj directly, we optimize the relative cam-
era pose Tj−1,j of frame j with respect to j −1. To achieve
the equivalent of rendering at the current pose Tj, we trans-
form the submap with the relative transformation T−1
j−1,j
and render from the last camera pose Tj−1 to get the ren-
dered color ˆIj and depth ˆDj.
Tracking Loss.
The inlier mask Minlier in the tracking
loss filters out pixels with depth errors 50 times larger than
the median depth error of the current re-rendered depth
map. Pixels without valid depth input are also excluded
as the inconsistent re-rendering in those areas can hinder
the pose optimization. For the soft alpha mask, we adopt
Malpha = α3 for per-pixel loss weighting. On the Scan-
Net++ dataset, if at the initialized pose the re-rendering loss
is 50 times larger than the running average during tracking
optimization, we use ICP odometry [57] to re-initialize the
pose for the current frame.
Submap Expansion. When selecting candidates to add to
the submap at a new keyframe, we uniformly sample Mk
points from pixels that meet either the alpha value condi-
tion or the depth discrepancy condition. Mk is set to 30K
for TUM-RGBD and ScanNet datasets, 100K for Scan-
Net++, and all available points that meet either condition
for Replica. The alpha threshold αthre is set to 0.98 across
all datasets. The depth discrepancy condition masks pixels
where the depth error exceeds 40 times the median depth
error of the current frame.
Submap Update. The radius ρ for the neighborhood check
when adding new Gaussians to the submap is set to 1cm.
Newly added Gaussians are initialized with opacity val-
ues 0.5 and their initial scales are set to the nearest neigh-
bor distances within the submap.
As mentioned in the
main paper, the Gaussians are not pruned until optimiza-
tion finishes. After the mapping optimization for the new
keyframe, we prune Gaussians that have opacity values
lower than a threshold othre. We set othre = 0.1 for Replica
and 0.5 for all other datasets.
1

<!-- page 14 -->
Submap Merging.
Upon completing the mapping and
tracking of all frames for the input sequence, we merge the
saved submaps into a global map. The mesh is extracted
by TSDF fusion [16] using the rendered depth maps and
estimated poses from the submaps. Then we use the recon-
structed mesh vertices to initialize the Gaussian centers of
the global map, providing a good starting point as they rep-
resent the scene geometry. We perform color refinement on
the global map for 30K iterations using the same hyperpa-
rameters as in [38]. The Gaussian parameters of the global
map are optimized from scratch using the RGB-D input and
our estimated camera poses.
Params
Replica
TUM RGB-D
ScanNet
ScanNet++
lrrotation
0.003
0.015
0.015
0.015
lrtranslation
0.001
0.005
0.005
0.005
lrexposure
0.1
0.1
0.1
0.1
overlapmin
0.1
0.2
0.2
0.2
intervalmin
2
4
3
1
Table B.2. Per-dataset Hyperparameters on Loop Closure.
Loop Detection. For NetVLAD [2], we use the pretrained
weights VGG16-NetVLAD-Pitts30K from HLoc [66].
We compute the cosine similarities of all keyframes within
the i-th submap and determine the self-similarity score si
self
corresponding to their p-th percentile. We set p = 50 on
Replica, TUM RGB-D, and ScanNet and p = 33 on Scan-
Net++. After getting the initial loops from the visual simi-
larity between submaps, we further filter detected loops by
computing their overlap ratio (OR) using the pose estimated
from the front-end. The overlapping region between P and
Q is defined as in [32]:
OR =
1
|Kij|
X
(p,q)∈Kij

∥TP→Q(p) −q∥2 < τ1

,
(10)
with [·] the Iverson bracket and (p ∈P, q ∈Q) ∈Kij the
set of putative correspondences found by reciprocal match-
ing the closest point between P and Q. We set τ1 = 0.1m
on Replica and τ1 = 0.2m on TUM RGB-D, ScanNet, and
ScanNet++. The selected thresholds are quite loose com-
pared to standard point cloud registration, as we only need
to ensure that two submaps have a spatial overlap for the
next step. We also remove the loops where two submaps
are too temporally close to each other to avoid redundant
computations.
We set the minimum submap id interval
(intervalmin) (cf. Tab. B.2) and remove the loop edges
whose submap id distances are smaller than intervalmin.
3DGS Registration. We first find the overlapping view-
points between two submaps using NetVLAD, as discussed
in the main paper. For all datasets, we select the top-k pairs
as the overlapping viewpoints, k = 2. In multi-view pose
estimation, we optimize the camera pose parameters (i.e.
translation and rotation) and the exposure coefficients for
selected viewpoints because the exposure of renders in dif-
ferent submaps can differ. We set different learning rates
of parameters in Tab. B.2. The learning rates of camera
pose parameters are significantly smaller because Replica
is a synthetic dataset with high-quality RGB-D measure-
ments from rendering; thus, the step size for optimization
should be smaller. The learning rates on the three real-world
datasets are consistent with each other.
Number of LCs.
We report the number of frames,
submaps, and loop closures (LCs) for each scene in our
LoopSplat system. On Replica scenes, LCs occur on aver-
age every 500 frames, about 4 times per scene (Tab. B.3a).
The relatively low frequency of LCs in Replica is due to its
single-room layouts and shorter sequences (approximately
2000 frames).
In contrast, ScanNet [17] scenes feature
longer sequences, averaging 4000 frames per scene (cf.
Tab. B.3b).
More challenging scenes like Scene 00,
54, and 233 require LoopSplat to create over 100 submaps
and perform more than 30 pose graph optimizations (PGOs)
per scene, which is attributed to their high sequence lengths.
The TUM RGB-D dataset presents a mix of long and short
sequences (cf. Tab. B.3c), resulting in varied numbers of
submaps and PGOs across its scenes. This diversity in scene
complexity and sequence length across datasets showcases
the adaptability of LoopSplat to different scene capturing
scenarios.
Method
r0
r1
r2
o0
o1
o2
o3
o4
Avg.
# Frames
2000
2000
2000 2000 2000
2000
2000
2000
2000
# Submaps
38
25
33
27
11
39
45
39
32
# LCs
2
8
4
3
4
1
2
6
4
(a) Replica [70]
Method
00
54
59
106
169
181
207
233
Avg.
# Frames
5578
6629
1807 2324 2034
2349
1988
7643
4073
# Submaps
112
132
36
47
41
47
39
153
76
# LCs
48
36
17
4
11
15
19
55
26
(b) ScanNet [17]
Method
fr1/desk1 fr1/desk2 fr1/room fr2/xyz fr3/office
Avg.
# Frames
595
640
1362
3669
2585
1770
# Submaps
14
15
24
6
39
20
# LCs
7
7
6
2
5
5
(c) TUM-RGBD [74]
Table B.3. Number of Submaps and PGOs Across Different
Datasets.
C. Datasets
We first specify the ScanNet++ sequences used through-
out our evaluation: (a) b20a261fdf, (b) 8b5caf3398,
(c) fb05e13ad1, (d) 2e74812d00, (e) 281bc17764.
Some sudden large motions occur in the DSLR-captured se-
quences. To avoid this, we only use the first 250 frames of
2

<!-- page 15 -->
each sequence. Tab. C.1 shows the average ground truth
frame translation distance and rotation degree per dataset
on the scenes (and frame length) we evaluated. The aver-
age motion on ScanNet++ is about 10× larger than in other
datasets, making it a challenging dataset for accurate pose
estimation and, hence, highlighting the robustness of Loop-
Splat given its superior performance on it.
Dataset
Replica
TUM-RGBD
ScanNet
ScanNet++
Translation (cm)
1.07
1.39
1.34
14.77
Rotation (◦)
0.50
1.37
0.69
13.43
Table C.1. Average Frame Motion Across Datasets.
D. Novel View Synthesis
We evaluate the novel view synthesis (NVS) performance
using the test set of the ScanNet++ sequences, where the
test views are held-out and distant from training views.
PSNR is evaluated on all test views after 10K iterations
of global map refinement using the image resolution of
876×584. Tab. D.1 shows that ours yields the best NVS re-
sults. For the baselines, we implement the evaluation using
their open-sourced code.
Method
a
b
c
d
e
Avg.
ESLAM [43]
13.63
11.86
11.83
10.59
10.64
11.71
SplaTAM [36]
23.95
22.66
13.95
8.47
20.06
17.82
Gaussian-SLAM [95]
26.66
24.42
15.01
18.35
21.91
21.27
LoopSplat (Ours)
25.60
23.65
15.87
18.86
22.51
21.30
Table D.1. Novel View Synthesis on ScanNet++ [93] (PSNR ↑
[dB]). For the baselines, results were obtained using the open-
sourced code with our implementation for the NVS evaluation.
PSNR calculations include all pixels, regardless of whether they
have valid depth input. LoopSplat yields the best results.
E. Additional Analysis
Rendering Performance at Scene Level. In the main pa-
per, we only report the average rendering performance on
each dataset. Tab. E.1, Tab. E.2, and Tab. E.3 report the per-
scene rendering performance on Replica, TUM RGB-D,
and ScanNet, respectively. On Replica and ScanNet, Loop-
Splat has the best performance on most of the scenes and on
TUM RGB-D, LoopSplat is only second to SplaTAM [36].
Online LC.
We investigate the significance of applying
LC and PGO online in LoopSplat, as opposed to applying
them only after the entire run concludes. The online mode,
as presented in our main paper, continuously performs LC
and PGO during the SLAM process. In contrast, the offline
mode delays these operations until the input stream ends,
applying them only once. Results in Tab. E.4 reveal that for
smaller scenes, such as those in Replica, online LC does
not significantly improve performance due to the limited
number of loops. However, in more complex environments
Method
Metric
Rm0
Rm1
Rm2 Off0 Off1 Off2 Off3 Off4
Avg.
NICE-SLAM [103]
PSNR↑
22.12 22.47 24.52 29.07 30.34 19.66 22.23 24.94 24.42
SSIM ↑
0.689 0.757 0.814 0.874 0.886 0.797 0.801 0.856 0.809
LPIPS ↓0.330 0.271 0.208 0.229 0.181 0.235 0.209 0.198 0.233
Vox-Fusion [90]
PSNR↑
22.39 22.36 23.92 27.79 29.83 20.33 23.47 25.21 24.41
SSIM↑
0.683 0.751 0.798 0.857 0.876 0.794 0.803 0.847 0.801
LPIPS↓
0.303 0.269 0.234 0.241 0.184 0.243 0.213 0.199 0.236
ESLAM [33]
PSNR↑
25.25 27.39 28.09 30.33 27.04 27.99 29.27 29.15 28.06
SSIM↑
0.874
0.89
0.935 0.934 0.910 0.942 0.953 0.948 0.923
LPIPS↓
0.315 0.296 0.245 0.213 0.254 0.238 0.186 0.210 0.245
Point-SLAM [63]
PSNR↑
32.40 34.08 35.50 38.26 39.16 33.99 33.48 33.49 35.17
SSIM↑
0.974 0.977 0.982 0.983 0.986 0.960 0.960 0.979 0.975
LPIPS↓
0.113 0.116 0.111 0.100 0.118 0.156 0.132 0.142 0.124
SplaTAM [36]
PSNR↑
32.86 33.89 35.25 38.26 39.17 31.97 29.70 31.81 34.11
SSIM↑
0.98
0.97
0.98
0.98
0.98
0.97
0.95
0.95
0.97
LPIPS↓
0.07
0.10
0.08
0.09
0.09
0.10
0.12
0.15
0.10
∗Gaussian-SLAM [95]
PSNR↑
38.88 41.80 42.44 46.40 45.29 40.10 39.06 42.65 42.08
SSIM↑
0.993 0.996 0.996 0.998 0.997 0.997 0.997 0.997 0.996
LPIPS↓
0.017 0.018 0.019 0.015 0.016 0.020 0.020 0.020 0.018
LoopSplat
PSNR↑
33.07 35.32 36.16 40.82 40.21 34.67 35.67 37.10 36.63
SSIM↑
0.973 0.978 0.985 0.992 0.990 0.985 0.990 0.989 0.985
LPIPS↓
0.116 0.122 0.111 0.085 0.123 0.140 0.096 0.106 0.112
Table E.1. Rendering Performance on Replica [70]. ∗denotes
evaluating on submaps instead of a global one.
Method
Metric
fr1/desk
fr2/xyz
fr3/office
Avg.
NICE-SLAM [103]
PSNR↑
13.83
17.87
12.890
14.86
SSIM↑
0.569
0.718
0.554
0.614
LPIPS↓
0.482
0.344
0.498
0.441
Vox-Fusion [90]
PSNR↑
15.79
16.32
17.27
16.46
SSIM↑
0.647
0.706
0.677
0.677
LPIPS↓
0.523
0.433
0.456
0.471
ESLAM [33]
PSNR↑
11.29
17.46
17.02
15.26
SSIM↑
0.666
0.310
0.457
0.478
LPIPS↓
0.358
0.698
0.652
0.569
Point-SLAM [63]
PSNR↑
13.87
17.56
18.43
16.62
SSIM↑
0.627
0.708
0.754
0.696
LPIPS↓
0.544
0.585
0.448
0.526
SplaTAM [36]
PSNR↑
22.00
24.50
21.90
22.80
SSIM↑
0.857
0.947
0.876
0.893
LPIPS↓
0.232
0.100
0.202
0.178
∗Gaussian-SLAM [95]
PSNR↑
24.01
25.02
26.13
25.05
SSIM↑
0.924
0.924
0.939
0.929
LPIPS↓
0.178
0.186
0.141
0.168
LoopSplat
PSNR↑
22.03
22.68
23.47
22.72
SSIM↑
0.849
0.892
0.879
0.873
LPIPS↓
0.307
0.217
0.253
0.259
Table E.2. Rendering Performance on TUM RGB-D [74].
∗
denotes evaluating on submaps instead of a global one.
like ScanNet and TUM RGB-D, online LC proves crucial to
LoopSplat’s superior performance. This is because it con-
stantly corrects map drift, preventing cumulative errors that
would otherwise degrade accuracy over time.
Average Number of Gaussians Per Scene. Tab. E.5 re-
ports the average number of Gaussians after global map re-
finement for each dataset. For a room-sized scene, we ob-
tain on average around 300K Gaussian splats, which is a
reasonable number. The number of Gaussians is dependent
on the scale of the scenes, the number of vertices used to
initialize the Gaussians, and the number of densification it-
erations during the optimization of 3DGS.
3

<!-- page 16 -->
Method
Metric
0000
0059
0106
0169
0181
0207
Avg.
NICE-SLAM [103]
PSNR↑
18.71
16.55
17.29
18.75
15.56
18.38
17.54
SSIM↑
0.641
0.605
0.646
0.629
0.562
0.646
0.621
LPIPS↓
0.561
0.534
0.510
0.534
0.602
0.552
0.548
Vox-Fusion [90]
PSNR↑
19.06
16.38
18.46
18.69
16.75
19.66
18.17
SSIM↑
0.662
0.615
0.753
0.650
0.666
0.696
0.673
LPIPS↓
0.515
0.528
0.439
0.513
0.532
0.500
0.504
ESLAM [33]
PSNR↑
15.70
14.48
15.44
14.56
14.22
17.32
15.29
SSIM↑
0.687
0.632
0.628
0.656
0.696
0.653
0.658
LPIPS↓
0.449
0.450
0.529
0.486
0.482
0.534
0.488
Point-SLAM [63]
PSNR↑
21.30
19.48
16.80
18.53
22.27
20.56
19.82
SSIM↑
0.806
0.765
0.676
0.686
0.823
0.750
0.751
LPIPS↓
0.485
0.499
0.544
0.542
0.471
0.544
0.514
SplaTAM [36]
PSNR↑
19.33
19.27
17.73
21.97
16.76
19.8
19.14
SSIM↑
0.660
0.792
0.690
0.776
0.683
0.696
0.716
LPIPS↓
0.438
0.289
0.376
0.281
0.420
0.341
0.358
∗Gaussian-SLAM [95]
PSNR↑28.539 26.208 26.258 28.604 27.789 28.627 27.67
SSIM↑
0.926
0.9336 0.9259
0.917
0.9223 0.9135 0.923
LPIPS↓
0.271
0.211
0.217
0.226
0.277
0.288
0.248
LoopSplat (Ours)
PSNR↑
24.99
23.23
23.35
26.80
24.82
26.33
24.92
SSIM↑
0.840
0.831
0.846
0.877
0.824
0.854
0.845
LPIPS↓
0.450
0.400
0.409
0.346
0.514
0.430
0.425
Table E.3. Rendering Performance on ScanNet [17]. ∗denotes
evaluating on submaps instead of a global one. We exclude these
results from the comparison for not being fair and for evaluating
an easier setting.
LC Mode
Replica
ScanNet
TUM RGB-D
Offline
0.26
15.27
12.54
Online
0.26
8.39
3.33
Table E.4. Ablation Study on Offline LC. (ATE [cm]↓)
Dataset
Replica
TUM-RGBD
ScanNet
ScanNet++
# Gaussians
295K
219K
331K
330K
Table E.5. Average Number of Gaussians Per-scene.
F. Additional Qualitative Results
In this section, we present additional qualitative results.
Overlap Ratio.
We first illustrate the overlap ratio we
adopt to determine if a detected loop is added to the pose
graph. In Fig. G.1, we showcase three representative Scan-
Net submap pairs with descending overlap ratios.
3DGS Registration.
Fig. G.2 presents more registration
results on the submaps. The red arrows highlight the dif-
ferences between odometry, ours, and ground truth. The
odometry results have the most misalignment, whereas esti-
mates from LoopSplat are closer to, or even better than, the
ground truth through visual inspection.
Mesh Reconstruction.
We present additional qualitative
results for mesh reconstruction on ScanNet scenes 0059
and 0207 in Fig. G.3. Our analysis concentrates on re-
gions with high geometric complexity. As evident from the
results, LoopSplat consistently produces higher-quality and
more consistent reconstructions compared to baseline meth-
ods, particularly in these challenging areas.
G. Limitations and Future Work
Limitations.
LoopSplat still faces certain limitations. As
the number of submaps exceeds 100, the computational de-
mands for pairwise registrations during pose graph opti-
mization increase significantly, reducing the efficiency of
the loop closure module. While LoopSplat demonstrates
competitive performance and achieves the lowest peak GPU
usage among all compared methods, there remains signifi-
cant room to improve the system’s overall efficiency. The
iterative nature of optimizing 3D Gaussians and camera
poses limits the speed of the system.
The pose initial-
ization is based on the constant speed assumption, which
can be improved with Kalman Filters. In terms of submap
construction, we use different hyperparameters for different
datasets, which is a standard practice in the SLAM commu-
nity, but we believe it hinders the generalization ability of
the system to in-the-wild data.
Future Work.
Several promising avenues for future re-
search emerge from this work. First, employing advanced
mesh extraction methods that directly operate on 3DGS,
such as SuGAR [25] or GOF [94], can improve the recon-
struction performance. Second, integrating uncertainty es-
timates for each viewpoint could improve both overlap es-
timation and multi-view optimization in 3DGS registration.
Additionally, exploring techniques to refine 3DGS recon-
struction in overlapping regions between submaps presents
another intriguing direction.
4

<!-- page 17 -->
Figure G.1. Qualitative Results of Overlap Ratio between Submaps. We visualize the centers of 3D Gaussians as point clouds, with
two submaps only colorized in the overlapping region. The top row demonstrates a large overlap between submaps with OR = 0.9. The
middle row showcases a medium overlap of OR = 0.6, while the bottom row exhibits an extremely low overlap of OR = 0.1. This last
case was rejected as a loop due to its insufficient overlap, which typically leads to low-accuracy registration or even complete failure.
5

<!-- page 18 -->
Odometry
LoopSplat (Ours)
Ground Truth
Figure G.2. Qualitative Results on Submap Registration. We visualize the centers of 3D Gaussians as point clouds, with two submaps
colorized differently. LoopSplat consistently improves upon the initial odometry-based alignment and outperforms the pseudo ground
truth. In the first row, LoopSplat (middle) achieves better alignment of the chair’s back compared to both odometry and ground truth.
Similar improvements are observed in the second row. The last row demonstrates LoopSplat’s superior alignment of walls and trash cans.
These results, representative of ScanNet and not cherry-picked, consistently showcase the method’s effectiveness across various scenes.
6

<!-- page 19 -->
GO-SLAM [101]
Gaussian-SLAM [95]
Loopy-SLAM [40]
LoopSplat (Ours)
Ground Truth
Figure G.3. Mesh Reconstruction on ScanNet [17] scenes 0059 and 0207. Per example, the first row displays the colored mesh, while
the second row shows the corresponding normals. LoopSplat demonstrates superior performance compared to baseline methods, excelling
in both texture fidelity and geometric detail. Notably, our approach yields smoother and more complete mesh reconstructions than the
strongest baseline, Loopy-SLAM.
7
