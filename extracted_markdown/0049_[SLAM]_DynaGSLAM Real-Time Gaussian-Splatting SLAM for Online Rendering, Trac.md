<!-- page 1 -->
DynaGSLAM: Real-Time Gaussian-Splatting SLAM for Online Rendering,
Tracking, Motion Predictions of Moving Objects in Dynamic Scenes
Runfa Blark Li*†
Mahdi Shaghaghi†
Keito Suzuki*
Xinshuang Liu*
Varun Moparthi*
Bang Du*
Walker Curtis†
Martin Renschler†
Ki Myung Brian Lee*
Nikolay Atanasov*
Truong Nguyen*
*UC San Diego
†Qualcomm XR Advanced Technology
{runfa, k3suzuki, xil235, vmoparthi, b7du, kmblee, natanaso, tqn001}@ucsd.edu
{shaghagh, wcurtis}@qti.qualcomm.com
https://blarklee.github.io/dynagslam/
Figure 1. DynaGSLAM is the first real-time Gaussian-Splatting (GS) based SLAM for online high-quality rendering of dynamic objects
(moving people in this scene) in dynamic scenes, while jointly estimating ego motion. With the online RGBD frames, DynaGSLAM
enables us to track(interpolate)/predict(extrapolate) the continuous object motions in the past/future. This figure shows the rendering of
GS mapping on TUM dataset [42] 1. First row: RGB rendering. Second row: Absolute error between the rendering and the ground truth.
Abstract
Simultaneous Localization and Mapping (SLAM) is one
of the most important environment-perception and navi-
gation algorithms for computer vision, robotics, and au-
tonomous cars/drones. Hence, high quality and fast map-
ping becomes a fundamental problem. With the advent of
3D Gaussian Splatting (3DGS) as an explicit representa-
tion with excellent rendering quality and speed, state-of-
the-art (SOTA) works introduce GS to SLAM. Compared to
classical pointcloud-SLAM, GS-SLAM generates photomet-
ric information by learning from input camera views and
synthesize unseen views with high-quality textures. How-
ever, these GS-SLAM fail when moving objects occupy the
1TUM RGB-D SLAM dataset and benchmark is licensed under Cre-
ative Commons 4.0 Attribution License (CC BY 4.0).
scene that violate the static assumption of bundle adjust-
ment. The failed updates of moving GS affects the static GS
and contaminates the full map over long frames. Although
some efforts have been made by concurrent works to con-
sider moving objects for GS-SLAM, they simply detect and
remove the moving regions from GS rendering (“anti” dy-
namic GS-SLAM), where only the static background could
benefit from GS. To this end, we propose the first real-time
GS-SLAM, “DynaGSLAM”, that achieves high-quality on-
line GS rendering, tracking, motion predictions of moving
objects in dynamic scenes while jointly estimating accurate
ego motion. Our DynaGSLAM outperforms SOTA static &
“Anti” dynamic GS-SLAM on three dynamic real datasets,
while keeping speed and memory efficiency in practice.
1
arXiv:2503.11979v1  [cs.CV]  15 Mar 2025

<!-- page 2 -->
Figure 2. Failures of static GS-SLAM over long frames. “Avs.
Error” is the absolute error of the rendered RGB to the ground
truth RGB. Typical static GS-SLAM works (like RTGSLAM [36])
do not consider moving objects, not only the rendering quality
of the moving objects becomes worse over long frames, but the
static regions get contaminated by the failed dynamic GS. Our Dy-
naGSLAM consistently renders both dynamic and static regions in
high quality over long frames.
1. Introduction
Simultaneous localization and mapping (SLAM) is a funda-
mental problem with wide areas of application in robotics,
virtual reality, and autonomous vehicles. Given an input
RGBD video sequence, a SLAM algorithm must jointly
estimate the ego motion (camera pose) and build the map
of the environment, conventionally represented as a point
cloud. Although existing SLAM algorithms are robust and
real-time in simple cases [33], there remain two major lim-
itations that cannot be neglected: 1. Standard SLAM as-
sumes a static scene, and cannot gracefully handle dynamic
objects; 2. The pointcloud representation conveys basic 3D
structure of the environment, but offers no photometric in-
formation on graphics or texture other than the input RGB,
limiting downstream tasks.
Many efforts [5, 13, 15, 32, 40, 53, 55] have been made
to improve SLAM with moving objects. While implemen-
tations vary, the main idea is to detect and remove the
moving objects that violate static assumptions for bundle
adjustment. Some SOTA works [32, 53] take a step fur-
ther to track object motions. This may be at the expense
of computational speed; however, explicit consideration of
moving objects improve robustness in many challenging in-
door/outdoor scenes. Nonetheless, the pointcloud represen-
tation still offers limited photometric information.
With the rise of Gaussian splats (GS) [18] as an explicit
3D representation with effective rendering speed and qual-
ity, recent work adopted GS to capture photometric infor-
mation in SLAM [2, 12, 17, 31, 36, 44, 48, 51]. This body
of work replaces the conventional pointclouds with GS, so
that RGB images from novel views can be synthesized be-
yond the input camera views using GS. Such a representa-
tion retains the 3D structure with explicit GS, while adding
significant photometric information. However, these SOTA
methods only consider with static scenes.
Therefore, as
shown in Fig. 2, even static regions are corrupted by the
mishandling of dynamic regions. A few concurrent meth-
ods [20, 25, 26, 46, 47] realize ”anti” dynamic SLAM with
GS representation, where dynamic objects are detected and
removed. Although this benefits localization, removing dy-
namic objects means only the static background can be ren-
dered from the GS map.
To represent dynamic objects with GS, SOTA methods
[1, 8, 16, 19, 21, 23, 24, 27, 28, 30, 35, 38, 39, 43, 50, 52]
explore directly adding a time dimension to GS. However,
these methods train GS in an offline manner, for hours per
video sequence, and are hence unsuitable for online SLAM.
In contrast to all concurrent SOTA GS-SLAM or dynamic
GS, our contributions can be summarized as:
• We propose DynaGSLAM, the first real-time Gaussian-
Splatting based SLAM that achieves high-quality online
GS rendering, tracking, motion predictions of moving ob-
jects with dynamic ego motions in the scene.
• We propose a novel dynamic GS management algorithm
for adding, deleting, tracking, updating and predicting dy-
namic GS, specifically our novel online GS tracker and
the motion interpolation/extrapolation algorithms, which
enable online real-time accurate dynamic GS mapping
with humble memory requirements.
2. Related Works
Dynamic SLAM. To handle dynamic objects, the usual ap-
proach is to detect and subtract the dynamic regions of the
image. Earlier work [5, 40] use RANSAC [9] or point cor-
relations [7] for motion detection, and the recent learning-
based methods [3, 54] learn to semantically segment mov-
ing objects. These methods improve the quality of camera
pose estimation by removing dynamic objects, but lose the
object motion information. Hence, we refer to these meth-
ods as “anti” dynamic SLAM. To extract the object motion,
some dynamic SLAM incorporate and track the dynamic
objects. These methods assume the objects are rigid, and
assign a tracklet to every object. DynaSLAM2 [4] tracks
rigid objects by estimating the motion of centroids, which
also improves camera localization. SOTA work DynoSAM
[32] proposes a world-centric factor-graph optimization for
accurate but suffers from time-consuming object motion es-
timation. However, all these attempts are based on classical
point cloud mapping, it is non-trivial to directly extend the
ideas to GS-SLAM since GS have more complex attributes,
2

<!-- page 3 -->
to be optimized other than the point position, such as spher-
ical harmonic and the shape (covariance).
GS-based SLAM. GS [18] has become a promising alter-
native to point cloud for SLAM [17, 31, 36, 48, 51]. Initial-
ized from an RGBD point cloud, new GS are incrementally
added with ego motion to complete the scene. Compared to
point cloud, 3DGS contains high-quality photometric infor-
mation, at the expense of additional storage and computa-
tion. Thus, SOTA GS-SLAM focus on GS management al-
gorithms, where RTGSLAM [36] designed a representative
real-time algorithm for the tasks of adding, deleting, and
reusing of static GS by converting GS in stable and unsta-
ble status. However, these SOTA GS-SLAM methods only
work for static scenes.
There are concurrent efforts on extending GS-SLAM to
dynamic scenes [20, 25, 26, 46, 47]. However, these meth-
ods fall under the “anti” dynamic SLAM category because
they segment and discard dynamic objects. Our method,
DynaGSLAM, is the first to construct a dynamic GS that
models dynamic objects in the online SLAM setting.
Offline Dynamic GS. Dynamic GS [1, 8, 16, 19, 21, 23, 24,
27, 28, 30, 35, 38, 39, 43, 50, 52] has been also attracting
lots of attention. With video and camera poses over frames,
dynamic GS aims to train GS in “4D” such that the well-
trained GS can be rendered from unseen views at any given
timestamps in the video. [8, 50] explored an explicit addi-
tional time dimension for GS position and shape, and design
4D rotation matrix with 4D-rotor and extend Spherical Har-
monics to 4D, but the extra time dimension on all GS intro-
duce speed and memory burden. The motion-function based
dynamic GS [16, 21, 29] leveraged Fourier series& cubic
polynomials for translation and SLERP (Spherical Linear
Interpolation) for rotation, and embed the motion function
parameters as GS attributes. However, these strategies in-
troduce additional channels to all Gaussians, and require
accurate supervision over time to learn the motion. Some
methods can also segment moving GS [10, 23, 28, 52], but
the motion-awareness is only achieved when the dynamic
GS is fully trained over all frames. Worse still, these meth-
ods all require long, offline training, and are not suitable for
online SLAM. Compared to the offline dynamic GS meth-
ods, our dynamic GS-SLAM method performs equally well
with three challenging constraints: 1) the target images are
presented online, so future frames are inaccessible; 2) GS
is optimized in real-time, while capturing dynamic objects;
and 3) the camera trajectory is unknown or inaccurate.
3. Problem Formulation
We formulate the dynamic GS SLAM problem as follows.
We are given a streaming sequence of RGB and depth im-
ages Ct ∈RW ×H×3 and Dt ∈RW ×H of a scene, taken
from unknown camera poses Tt ∈SE(3). The scene con-
tains moving objects.
The objective is to recover the unknown camera poses
Tt (i.e., localization) and to find a time-varying scene repre-
sentation Gt that models the moving objects (i.e., mapping).
We use the GS to represent Gt, so that RGB and depth im-
ages ˆC(Gt, Tt) and ˆD(Gt, Tt) can be synthesized to match
the scene at time t seen from camera pose Tt. To this end, at
each time t, we aim to find the camera trajectory {Tτ} for
τ ∈[0, t], and a time-varying scene representation {Gτ},
such that:
min
Gτ ,Tτ
t
X
τ=0
ℓc( ˆC(Gτ, Tτ), Cτ) + ℓd( ˆD(Gτ, Tτ), Dτ)
(1)
where ℓc and ℓd are color and depth image losses measur-
ing the similarity between images ˆCt, ˆDt reconstructed by
GS and the images Ct, Dt provided by the camera. Fur-
thermore, we focus on tracking a time-varying GS Gt over
a time horizon, rather than only creating new GS for each
timestep. This allows photorealistic synthesis of images not
only at novel viewpoints (as in static GS), but also at con-
tinuous novel times. For simplicity, our notations reflect
the case of regular, unit time intervals; however, we aim to
predict and track motion over continuous time, given data
arriving at irregular time intervals.
To ensure online, real-time performance, we treat the lo-
calization and mapping problems separately. For localiza-
tion, we rely on DynoSAM [32], which is a SOTA graph-
based visual localization method in dynamic scenes based
on point cloud representation. Using the camera trajectory
Tt estimated by DynoSAM, we focus on the mapping prob-
lem of finding the dynamic GS Gt, which allows photoreal-
istic image rendering that is not possible with point clouds.
For example, this is analogous to the use of ORB-SLAM2
[33] in RTGSLAM [36]. We defer the correction of camera
trajectory using the dynamic GS Gt to future work, with the
expectation that the improvements will be incremental.
4. Dynamic GS Architecture
To solve the problem of photorealistic synthesis of dynamic
objects, we introduce a new variant of GS with a dynamic
mean that moves over time. We define a dynamic GS as a set
of Gaussian blobs defined as Gt =

(mi
t(τ), Σi
t, αi
t, shi
t)
	
, where Σi
t ∈R3×3, αi
t ∈R and shi
t ∈R16 are the covari-
ance matrix, opacity and spherical harmonics. Importantly,
mi
t(τ) is a time-varying mean allowing novel-view synthe-
sis at an unobserved time τ, modeled as a cubic Hermite
spline:
mi
t(τ) = (2τ ′3 −3τ ′2 + 1)mi
t−+ (τ ′3 −2τ ′2 + τ ′)vi
t−
+ (−2τ ′3 + 3τ ′2)mi
t+ + (τ ′3 −τ ′2)vi
t+,
(2)
where τ ′ = τ −t −1, mi
t−,t+ and vi
t−,t+ are interpolation
parameters. Notably, these parameters can be updated an-
3

<!-- page 4 -->
Figure 3. Overview of DynaGSLAM Mapping. We focus on three modules - Dynamic GS (a) Segmentation & Flow, (b) Management
and (c) Tracking & Prediction. DynaGSLAM takes RGBD sequence as input to construct map with GS, (a) segment dynamic GS from
static GS in 3D, and estimate dynamic GS 3D motion flow between frames. (b) Dynamic GS are managed separately from static GS with
GS flow, but combined to jointly optimize. Case 1 &2 are the rules for dynamic GS adding; “Cond. 1&2” denotes the conditions for
dynamic GS deletion. (c) The optimized dynamic GS at current and past frames are used to interpolate/extrapolate dynamic GS in the
continuous timeline from past to future. “CHS” refers to “cubic Hermite spline” and “LF” refers to “linear function”. Localization details
are not included in the figure since our main contribution is on mapping.
alytically without iterative optimization, as we detail later.
Extrapolation into the future is achieved by querying τ > t.
RGB
images
are
rendered
similarly
to
original
3DGS [18] through alpha-blending of projected Gaus-
sians at each time t:
ˆCt =
X
i
ci
tf(gi
t)
i−1
Y
j=1
(1 −f(gj
t )),
(3)
where f(gi
t) = αi
tN2D(Pmi
t(t), PΣi
tP T ) is the weight of
the i-th Gaussian at time t, after affine projection P.
For rendering accurate depth images ˆDt, we adopt the
ideas from 2DGS [11], and discard the shortest principal
axis of 3D Gaussians for higher efficiency and better sur-
face representation. We also adopt the “surface rendering”
technique for depth from [11] as it is faster than alpha blend-
ing. This takes the depth d(gi
t) of the closest GS that is over
an opacity threshold λα:
ˆD = min
gi
t∈Gt
d(gi
t), s.t. f(gi
t)
i−1
Y
j=1
(1 −f(gj
t )) > λα.
(4)
5. Online Training of Dynamic GS
When training a static GS, the conventional approach is to
first render RGB & depth images at target views as usual,
and then minimize the loss between the rendered and ob-
served ground truth, as is done in [17, 31, 36, 51]. How-
ever, this approach is insufficient for real-time training of
dynamic GS because of the objects’ motion. Here, we intro-
duce an improved training method that first explicitly mod-
ifies the GS to match the current observations to accurately
account for object motion.
Prerequisites We assume that there are suitable submod-
ules for 1) 2D optical flow calculation; 2) motion segmen-
tation; and 3) localization. Although challenging, suitable
prior methods exist for these tasks. For 2D optical flow,
we adopted the real-time optical flow (RAFT) [45]. For
motion segmentation, we combined a coarse motion blobs
computed from the RAFT optical flow image with real-time
online SAM2 [37] (Sec.
A). For localization, we adopt
DynoSAM [32], as aforementioned in Sec. 3, and use its
estimated camera poses without modification.
4

<!-- page 5 -->
5.1. Dynamic GS Flow
To make the most of the current observations, we utilize the
optical flow at the current frame t to associate and propa-
gate the existing GS from time t −1. To do so, we lift the
2D optical flow to 3D (Fig. 3(a)), and call it dynamic GS
flow. To obtain the dynamic GS flow in 3D from the current
to last frame t →t −1, we mask out static optical flow
with 2D motion mask Mt, project the moving optical flow
ft−1←t(u, v) to 3D dynamic GS flow FDyna
t−1←t using depth
Dt, and compensate the ego motion:
FDyna
t−1←t = Mt ·
 DtK−1ft−1←t

−(Tt−1←tPt −Pt) ,
(5)
where K is the camera intrinsic, P is the camera pose, and
Tt−1←t is the ego motion transformation
5.2. Dynamic GS Management
Maintaining an appropriate number of GS is important for
GS-SLAM, and even more so in the presence of dynamic
objects. Adding new GS allows capturing the latest infor-
mation, whereas adding too many will lead to prohibitively
high memory usage (n.b. results of SplaTAM [17] on “rpy”
in Table 2). Similarly, deleting old GS is essential for cap-
ping the memory usage, and more importantly to avoid in-
troducing outliers in the future when it is outdated (n.b. re-
sult of RTGSLAM [36] in Fig. 2).
In this work, we present a management strategy for ad-
dition and deletion of the dynamic GS. We store dynamic
and static GS separately, with different addition and dele-
tion strategies, although they are rendered jointly.
For static GS, we follow the strategy of SOTA static GS-
SLAM [17, 36]. This strategy adds and optimizes new GS
when new areas are seen, while keeping the old GS un-
changed. If a GS is not seen for a certain number of frames,
it is deleted.
For dynamic GS, this strategy is insufficient because of
the objects’ motion. We thus introduce a novel dynamic
GS management algorithm shown in Fig. 3(b) that over-
comes the limitations above. Let Gt−1 = {gi
t−1} be the
GS up to time t −1, and let Pt = {pj
t} be the current
motion-segmented RGBD pointcloud. We first transform
the current pointcloud Pt (red) to the previous timestep t−1
(blue), using the GS flow FDyna
t−1←t (green) (5). Transforming
the current pointcloud back in time using the latest optical
flow observation FDyna
t−1←t is better than using the cubic Her-
mite spline (2) for extrapolation, because the cubic Hermite
spline only contains information up to time t −1. More-
over, by filtering the GS at time t−1, we avoid unnecessary
foward-propagation of unnecessary GS to time t.
With the transformed pointcloud FDyna
t−1←tPt (green), we
search for the nearest neighbor in the existing dynamic GS
(blue), and compare the nearest neighbor distance dmin(pi
t)
the average nearest neighbor distance ¯d, computed as:
¯d =
1
NW
NW
X
i=1
dmin(pi
t) =
1
NW
NW
X
i=1
min
g∈Gt∥FDyna
t−1←tpi
t −g∥.
(6)
We check if the nearest neighbor distance exceeds some ra-
tio threshold λd of ¯d, resulting in two cases:
Case 1 (prev. observed points): dmin(pi
t) ≤λd ¯d. The
nearest past GS (blue) is within the distance threshold of
the transformed point (green). In this case, we simply reuse
the past GS (blue), by replacing their mean with the current
point matched (red), and optimize using the current RGBD.
This explicit modification is the key to higher performance,
as it moves the past GS to the right location in one step,
whereas the usual gradient updates as in the static GS case
can only provide minor, insufficient displacements.
Case 2 (new points): dmin(pi
t) > λd ¯d. There is no past
GS (blue) near the transformed point (green). In this case,
a new GS (red) is initialized for the point pi
t. This allows
complete coverage of the whole scene, as some objects are
unseen before the current frame (e.g. occluded sides of the
moving box in Fig. 2).
We also check the validity of the existing GS against two
conditions: observability and longevity.
Cond. 1 (observability): ∃pi
t ∈Pt, ∥FDyna
t−1←tpi
t −gj∥≤
λd ¯d.
We only keep GS (blue) that are within distance
threshold of the currently observed points (green). This step
is essential for dynamic GS because old, unobserved GS be-
come outlier noise in the future if they are not displaced or
deleted, unlike in static GS where the scene is unchanged.
Cond. 2 (longevity): We delete any GS that persisted for a
longer period of time than a set longevity threshold.
As observed, the distance threshold ratio λd plays a cru-
cial role in the dynamic GS management. We thoroughly
study its impact in Sec. 6.
Point trackers may seem to serve the same purpose, but
they do not, let alone being too slow for online SLAM (n.b,
[22, Table 1]). We are searching for correspondences from
the current pointcloud to the past GS, whereas point track-
ers predict the location of past points in the current frame,
necessitating the management logic we presented.
5.3. Rendering and Optimization
Although being managed separately, we jointly render dy-
namic & static GS because it improves their interactions to
better handle occlusions, lighting consistency, and spatial
coherence. We follow the SOTA GS-SLAM [17, 31, 36, 48,
51] with similar supervision between the rendered ((3), (4))
and input RGBD over a small time window of past views.
Although jointly rendered, the optimization is decoupled
in that different learning rates are used between dynamic
and static GS, as well as longevity windows. Static GS at-
tributes remain more stable across frames, while dynamic
5

<!-- page 6 -->
GS undergo abrupt changes, decoupling ensures that their
updates do not interfere with the optimization of static struc-
tures. Without decoupling, motion inconsistencies from dy-
namic objects introduce ghosting effects, blending, or in-
correct updates in the static region (as shown in Fig. 2),
where remnants of dynamic objects appear in static regions
due to incorrect optimization of photometric and geometric
consistency.
Optimization-free update of motion spline.
After as-
sociating and training the dynamic GS with respect to the
current frame, the cubic Hermite spline (2) can be updated
analytically without optimization. This is because the pa-
rameters mi
t−,t+ and vi
t−,t+ in (2) correspond exactly to the
3D position and the velocity of the center of GS gi
t at the last
(t−) and current (t+) frame. Thus, we directly set mi
t−,t+
from the optimized GS center at the last (t−) and current
(t+) frames. The velocity term at the last frame vi
t−is set as
the negative of the GS flow (5), so that vi
t−= −FDyna
(t−1)←t.
The velocity term at the current frame vi
t+ is estimated us-
ing the constant acceleration assumption, by extrapolating
between vi
(t−1)+ and vi
t−. If vi
(t−1)+ is unavailable (e.g.
when the GS gi
t was just initialized), we fall back to the
constant velocity assumption.
6. Experiments
Baselines:
We compare the performance of our Dy-
naGSLAM algorithm against four other SOTA GS-SLAM
methods, which are: RTGSLAM [36], SplatTAM [17],
GSSLAM [31] and GSLAM [51]. Although there are other
concurrent ”anti” dynamic GS-SLAM methods [20, 25, 26,
46, 47] that remove dynamic objects, we could not repro-
duce these methods because their code is unavailable. We
compare against the reported results where possible.
Datasets: Prior works on static GS-SLAM are evaluated on
synthetic datasets with static scenes [6, 41]. In contrast, we
evaluate our method on real datasets with dynamic scenes.
We use OMD [14] and the dynamic scenes from the TUM
[42] and the BONN [34] datasets.
Experimental Setup: For TUM and OMD datasets, we use
DepthAnythingV2 [49] to get smooth depth and recover the
real scale with the original depth map because the raw depth
sensor measurements come with large portion of invalid re-
gions in these datasets. For the Bonn dataset, we use the raw
depth sensors measurements. In addition to the common
metrics (PSNR, SSIM, LPIPS) used for mapping evalua-
tion, we evaluate “DynaPSNR” as PSNR only for dynamic
objects within 2D motion masks. We evaluate the Absolute
Trajectory Error (ATE) of camera localization. The exper-
iments are conducted on a desktop with a single NVIDIA
3090Ti (24GB).
Dynamic Mapping Results. Tables 1 and 2 show the quan-
titative comparison of the GS mapping, our DynaGSLAM
Metrics
Scene
[36]
[31]
[17]
[51]
[46]∗
Ours
balloon
13.7
19.3
18.8
24.5
24.0
28.4
balloon2
12.9
17.8
16.4
23.1
22.9
28.3
PSNR↑
ps track
13.2
14.9
15.6
24.7
24.6
28.0
ps track2
13.5
15.9
13.7
24.6
24.2
27.4
balloon
38.4
73.0
73.5
85.8
77.5
93.1
ballon2
32.0
67.5
60.6
83.2
71.5
93.3
SSIM↑
ps track
36.1
59.2
54.9
86.4
78.7
93.0
ps track2
38.1
69.9
46.9
86.2
77.3
91.6
balloon
67.9
43.9
38.8
26.6
32.5
29.3
balloon2
71.2
46.2
51.2
27.2
39.4
26.6
LPIPS↓
ps track
69.2
54.8
53.4
24.3
32.8
28.9
ps track2
66.5
45.4
56.2
24.3
32.0
31.6
balloon
18.8
14.6
15.2
19.8
-
32.5
balloon2
16.6
14.1
14.6
20.0
-
32.8
DynaPSNR↑
ps track
18.1
8.8
10.3
21.6
-
32.1
ps track2
17.4
7.6
8.7
22.2
-
32.6
Table 1. Comparison on Bonn Dataset. Our method outperforms
all other baselines. ∗: Reported results from [46] listed without re-
production due to unavailability of code. DynaPSNR unavailable
for [46], because dynamic objects are removed.
Metrics
Scene
[36]
[17]
[31]
[51]
Ours
fr3 wk xyz
14.3
13.8
13.5
22.0
27.5
PSNR↑
fr3 wk static
13.9
15.5
15.9
20.2
26.9
fr3 wk rpy
15.2
OOM
13.7
25.0
27.4
fr3 wk hs
13.1
11.9
12.3
24.7
27.2
fr3 wk xyz
45.4
40.8
38.2
80.7
95.7
SSIM↑
fr3 wk static
52.9
60.5
54.7
73.5
96.1
fr3 wk rpy
51.7
OOM
40.3
88.3
94.7
fr3 wk hs
34.3
33.4
37.0
87.9
94.5
fr3 wk xyz
59.7
64.1
58.0
23.7
16.0
LPIPS↓
fr3 wk static
53.9
41.7
42.4
29.4
14.0
fr3 wk rpy
56.7
OOM
53.9
16.8
21.1
fr3 wk hs
69.9
67.3
66.8
16.3
20.0
fr3 wk xyz
17.0
12.3
12.9
23.5
31.5
DynaPSNR↑
fr3 wk static
16.6
12.1
12.7
22.5
30.3
fr3 wk rpy
16.5
OOM
12.8
26.1
30.1
fr3 wk hs
14.9
12.1
12.7
26.2
30.7
Table 2. Comparison on TUM Dataset. Our method outperforms
others on all metrics. Best results boldfaced. OOM indicates out
of memory.
achieves superior results that outperforms other SOTA GS-
SLAM on all dynamic sequences. The superior results in
DynaPSNR illustrates the efficacy of our dynamic GS man-
agement algorithm. Figures 1 and 4 show some qualita-
tive comparisons, where our rendering quality is better than
other works, especially around dynamic objects such as the
two people (Fig. 1) and the balloon (Fig. 4). Please confer
the videos in Supplementary Material for full compar-
ison. Our rendering results exhibit some minor “floater”
artifacts, because we use very few numbers of GS for effi-
ciency compared to others (Table 3). In contrast, SplaTAM
[17] runs out-of-memory(OOM) after 500 frames because
it fails to delete outlier dynamic GS and release memory.
Dynamic Motion Tracking & Prediction. Figure 5 shows
a qualitative comparison of tracking and prediction with
DynaGSLAM. Tracking is evaluated by interpolating and
6

<!-- page 7 -->
Figure 4. Qualitative Results on Bonn Dataset. First row: RGB rendering. Second row: Difference between the rendering and the ground
truth. Our DynaGSLAM outperforms all baseline static GS-SLAM on the mapping quality, especially at the moving balloon and person.
Figure 5. Tracking & Prediction results on OMD and TUM dataset. “Start/End Frame” denotes two consecutive frames (t0 & t5).
“Target Frame” is the ground truth of t3 (for tracking) and t10 (for prediction). Only “Start/End Frame” can be seen by SLAM, “Target
Frame” cannot be seen by SLAM. When DynaGSLAM online proceeds to the ”End Frame” (t5) as the current frame, it interpolates the
missing past frame t3 and extrapolates the unseen future frame t10. To better visualize motion quality, we overlap the ground-truth motion
mask (white transparent) of the target frames to all frames of ground truths and estimations; A better overlapping between the moving
objects and the masks indicates better motion estimation and rendering. Please zoom in to check details.
rendering an intermediate target timestamp (t3) given two
start and end frames (t0 and t5). For prediction, we extrap-
olate and render a future (t10) timestamp given the same in-
put. This is an extremely difficult task, as we are only given
one out of every five frames, which is temporally sparse, to
reconstruct the 3rd and 10th frames that are unseen. Since
previous methods do not model dynamic objects’ motion
in GS, we take RTGSLAM [36] as baseline, assuming no
moving entities at t5 and render at the target viewpoint.
The results in Fig. 5 shows that our method accurately
predicts the moving objects (moving boxes and people),
which overlap with the motion mask (transparent white).
A quantitative comparison of tracking & prediction is pro-
vided in the supplementary.
7

<!-- page 8 -->
Figure 6. Camera Tracking Results on OMD dataset (S4U). Top
Left: RTGSLAM [36]’s ICP, 90 frames. Top Right: RTGSLAM’s
ICP refined ORBSLAM2 [33], 90 frames. Bottom Left: Ours, 90
frames. Bottom Right: Ours, 500 frames.
Methods
SplaTAM [17]
RTGSLAM [36]
Ours
Mapping
(ms/frame)
1027
555
347
Localization
(ms/frame)
5179
59
95
GS number
310K
520K
22K
Memory
(GB)
0.82
1.7
2.6
Table 3. Comparison of Inference Speed & Memory on TUM
fr3 walking xyz with a single 3090Ti GPU.
Localization Results. Figure 6 visualizes the camera tra-
jectory and ATE on OMD dataset (S4U). We directly adopt
the world-centric graph optimization strategy from [32],
which considers the moving objects. In contrast, ICP and
ORBSLAM2 used in RTGSLAM do not distinguish be-
tween static and dynamic objects. The result validates the
importance of static/dynamic separation.
Online Speed & Memory. Computational speed and mem-
ory usage are important for online SLAM. We compare on-
line speed and memory usage on TUM “fr3 wk xyz” se-
quence. The results are shown in Table 3. For SplaTAM
[17], we follow their official configurations for TUM
with 200 iters/frame for mapping optimization. For both
RTGSLAM [36] and our DynaGSLAM, we follow RT-
GSLAM’s official configuration with 50 iters/frame for
mapping optimization. SplatTAM and RTGSLAM only up-
date active GS to reduce the memory usage, but their static
GS management fails on dynamic scenes yielding much
Dist Threshold λd
0
0.01
0.1
0.5
PSNR↑
30.63
29.15
26.15
23.95
DynaPSNR↑
34.58
28.20
20.21
17.04
SSIM↑
95.13
93.88
90.89
87.49
LPIPS↓
15.13
17.51
22.88
27.50
Reuse Rate (%)↑
0
22.75
56.36
83.53
Dyna GS Num ↓
48.0k
42.8k
38.3k
25.4k
Table 4. Ablation Study on the distance threshold of the nearest
neighbor in Dynamic GS management. Reuse Rate is the ratio
of dynamic GS that are used for at least two consecutive frames.
With higher distance threshold, the GS reuse rate is higher, and
less new GS are initialized, yielding lower mapping quality but
higher efficiency with lower number of dynamic GS.
larger number of GS than ours. Thanks to our dynamic
GS management strategy, we achieve better mapping with
much fewer GS. Our mapping runs at ∼347 ms/frame, in-
cluding the online SAM2 segmentation (∼36ms) and RAFT
[45] optical flow (∼55ms). Our memory usage is higher
than baselines mainly due to online SAM2 [37] (∼1100mb)
and RAFT (∼980mb), but it is acceptable given the benefits
of dynamic rendering over baselines. The DynoSAM lo-
calzation [32] costs ∼450 ms/frame. Overall, the fast com-
putation guarantees real-time operation with efficient mem-
ory usage, while ensuring accurate mapping and localiza-
tion.
Ablation Study. The distance threshold λd is the most im-
portant hyper-parameter for our dynamic GS management
algorithm (Fig. 3(b) & Section 5.2). Table 4 shows the
effect of distance threshold on mapping quality and compu-
tational efficiency. In our experiments, we chose a low dis-
tance threshold of λd = 0.05 with a limit of 50k dynamic
GS for the best mapping quality. In other applications, it
may be beneficial to trade off the mapping quality for com-
putational efficiency. Ablation studies of other factors are
presented in the Supplementary Material due to space.
7. Conclusion and Future Works
We build the first online GS-based SLAM system - Dy-
naGSLAM that render, track, and predict the motions of dy-
namic objects with ego motion estimation in real time. Our
experiments on three real datasets validate the high qual-
ity, efficiency and robustness of our dynamic Gaussian map-
ping, along with accurate real-time ego motion estimation.
To enable the online real-time usage, we pursue efficiency
and sacrifice some complexity of the motion model. Future
works should explore to various motion models/functions
while keeping the system’s efficiency.
References
[1] 4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024. 2, 3
8

<!-- page 9 -->
[2] Lizhi Bai, Chunqi Tian, Jun Yang, Siyu Zhang, Masanori
Suganuma, and Takayuki Okatani. Rp-slam: Real-time pho-
torealistic slam with efficient 3d gaussian splatting, 2024. 2
[3] Berta Bescos, JM. F´acil, Javier Civera, and Jos´e Neira. Dy-
naSLAM: Tracking, mapping and inpainting in dynamic en-
vironments. IEEE RA-L, 2018. 2
[4] Berta Bescos, Carlos Campos, Juan D. Tard´os, and Jos´e
Neira. Dynaslam ii: Tightly-coupled multi-object tracking
and slam.
IEEE Robotics and Automation Letters, 6(3):
5191–5198, 2021. 2
[5] Carlos Campos, Richard Elvira, Juan J. G´omez Rodr´ıguez,
Jos´e M. M. Montiel, and Juan D. Tard´os. Orb-slam3: An
accurate open-source library for visual, visual–inertial, and
multimap slam. IEEE Transactions on Robotics, 2021. 2
[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proc. Computer Vision and Pattern Recognition (CVPR),
IEEE, 2017. 6
[7] Weichen Dai, Yu Zhang, Ping Li, Zheng Fang, and Sebas-
tian Scherer.
Rgb-d slam in dynamic environments using
point correlations. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 2022. 2
[8] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
Towards efficient novel view synthesis for dynamic scenes.
In Proc. SIGGRAPH, 2024. 2, 3
[9] Martin A. Fischler and Robert C. Bolles. Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Commun. ACM,
1981. 2
[10] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for effi-
cient dynamic scene reconstruction. IEEE Transactions on
Circuits and Systems for Video Technology, 2024. 3
[11] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 4
[12] Huajian Huang, Longwei Li, Cheng Hui, and Sai-Kit Yeung.
Photo-slam: Real-time simultaneous localization and photo-
realistic mapping for monocular, stereo, and rgb-d cameras.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024. 2
[13] Jiahui Huang, Sheng Yang, Zishuo Zhao, Yu-Kun Lai, and
Shimin Hu.
Clusterslam: A slam backend for simultane-
ous rigid body clustering and motion estimation. In 2019
IEEE/CVF International Conference on Computer Vision
(ICCV), 2019. 2
[14] Kevin Michael Judd and Jonathan D. Gammell. The oxford
multimotion dataset: Multiple se(3) motions with ground
truth. IEEE Robotics and Automation Letters, 2019. 6
[15] Kevin M. Judd, Jonathan D. Gammell, and Paul Newman.
Multimotion visual odometry (mvo): Simultaneous estima-
tion of camera and third-party motions. In 2018 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS), 2018. 2
[16] Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. A
compact dynamic 3d gaussian representation for real-time
dynamic view synthesis. In ECCV 2024, 2024. 2, 3
[17] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. SplaTAM: Splat, track & map 3d gaussians
for dense rgb-d slam. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024. 2, 3, 4, 5, 6, 8, 1
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3, 4
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
2023. 2, 3
[20] Mangyu Kong, Jaewon Lee, Seongwon Lee, and Euntai Kim.
Dgs-slam: Gaussian splatting slam in dynamic environment,
2024. 2, 3, 6
[21] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis.
Dynmf: Neural motion factorization for real-time dynamic
view synthesis with 3d gaussian splatting. ECCV, 2024. 2, 3
[22] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid.
Dense optical tracking: Connecting the dots. In CVPR, 2024.
5
[23] Junoh Lee, Chang-Yeon Won, Hyunjun Jung, Inhwan Bae,
and Hae-Gon Jeon. Fully explicit dynamic gaussian splat-
ting, 2024. 2, 3
[24] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas,
and Kostas Daniilidis.
MoSca: Dynamic gaussian fusion
from casual videos via 4D motion scaffolds. arXiv preprint
arXiv:2405.17421, 2024. 2, 3
[25] Haoang Li, Xiangqi Meng, Xingxing Zuo, Zhe Liu, Hesheng
Wang, and Daniel Cremers.
Pg-slam: Photo-realistic and
geometry-aware rgb-d slam in dynamic environments, 2024.
2, 3, 6
[26] Mingrui Li, Weijian Chen, Na Cheng, Jingyuan Xu, Dong
Li, and Hongyu Wang. Garad-slam: 3d gaussian splatting
for real-time anti dynamic slam, 2025. 2, 3, 6
[27] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024. 2, 3
[28] Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-
Phuoc, Douglas Lanman, James Tompkin, and Lei Xiao.
Gaufre: Gaussian deformation fields for real-time dynamic
novel view synthesis. In Proc. IEEE/CVF Winter Confer-
ence on Applications of Computer Vision (WACV), 2025. 2,
3
[29] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao.
Gaussian-flow: 4d reconstruction with dynamic 3d gaussian
particle. Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 21136–21145,
2024. 3
[30] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 3DV, 2024. 2, 3
9

<!-- page 10 -->
[31] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison. Gaussian Splatting SLAM. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, 2024. 2, 3, 4, 5, 6
[32] Jesse Morris, Yiduo Wang, Mikolaj Kliniewski, and Viorela
Ila. Dynosam: Open-source smoothing and mapping frame-
work for dynamic slam, 2025. 2, 3, 4, 8
[33] Ra´ul Mur-Artal and Juan D. Tard´os.
ORB-SLAM2: an
open-source SLAM system for monocular, stereo and RGB-
D cameras.
IEEE Transactions on Robotics, 33(5):1255–
1262, 2017. 2, 3, 8
[34] E. Palazzolo, J. Behley, P. Lottes, P. Gigu`ere, and C. Stach-
niss.
ReFusion: 3D Reconstruction in Dynamic Environ-
ments for RGB-D Cameras Exploiting Residuals. 2019. 6
[35] Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonza-
lez Bello, Jaeho Moon, Jihyong Oh, and Munchurl Kim.
Splinegs: Robust motion-adaptive spline for real-time dy-
namic 3d gaussians from monocular video, 2024. 2, 3
[36] Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin Yang,
Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d re-
construction at scale using gaussian splatting. 2024. 2, 3, 4,
5, 6, 7, 8
[37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos.
arXiv preprint arXiv:2408.00714, 2024. 4, 8, 1
[38] Richard Shaw,
Michal Nazarczuk,
Jifei Song,
Arthur
Moreau, Sibi Catley-Chandar, Helisa Dhamo, and Eduardo
P´erez-Pellitero. Swings: Sliding windows for dynamic 3d
gaussian splatting. In ECCV 2024, 2024. 2, 3
[39] Nagabhushan Somraj, Kapil Choudhary, Sai Harsha Mup-
paraju, and Rajiv Soundararajan. Factorized motion fields
for fast sparse input dynamic view synthesis. In ACM SIG-
GRAPH 2024 Conference Papers, 2024. 2, 3
[40] Seungwon Song, Hyungtae Lim, Alex Junho Lee, and Hyun
Myung. Dynavins: A visual-inertial slam for dynamic en-
vironments. IEEE Robotics and Automation Letters, 2022.
2
[41] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal,
Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan,
Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang
Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler
Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva,
Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael
Goesele, Steven Lovegrove, and Richard Newcombe. The
Replica dataset: A digital replica of indoor spaces. arXiv
preprint arXiv:1906.05797, 2019. 6
[42] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cre-
mers. A benchmark for the evaluation of rgb-d slam systems.
In Proc. of the International Conference on Intelligent Robot
Systems (IROS), 2012. 1, 6
[43] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei
Zhao, and Wei Xing.
3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-
viewpoint videos. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
2024. 2, 3
[44] Lisong C. Sun, Neel P. Bhatt, Jonathan C. Liu, Zhiwen Fan,
Zhangyang Wang, Todd E. Humphreys, and Ufuk Topcu.
Mm3dgs slam: Multi-modal 3d gaussian splatting for slam
using vision, depth, and inertial measurements, 2024. 2
[45] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow. In ECCV 2020, 2020. 4, 8, 1
[46] Long Wen, Shixin Li, Yu Zhang, Yuhong Huang, Jianjie Lin,
Fengjunjie Pan, Zhenshan Bing, and Alois Knoll. Gassidy:
Gaussian splatting slam in dynamic environments, 2024. 2,
3, 6, 1
[47] Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng
Feng, and Li Zhang.
DG-SLAM: Robust dynamic gaus-
sian splatting SLAM with hybrid pose optimization.
In
The Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. 2, 3, 6
[48] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In CVPR, 2024. 2, 3, 5
[49] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. arXiv:2406.09414, 2024. 6
[50] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting. 2024. 2, 3
[51] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting, 2023. 2, 3, 4, 5, 6
[52] Daiwei Zhang, Gengyan Li, Jiajie Li, Micka¨el Bressieux,
Otmar Hilliges, Marc Pollefeys, Luc Van Gool, and Xi
Wang.
Egogaussian: Dynamic scene understanding from
egocentric video with 3d gaussian splatting. arXiv preprint
arXiv:2406.19811, 2024. 2, 3
[53] Jun Zhang, Mina Henein, Robert Mahony, and Viorela Ila.
Vdo-slam:
A visual dynamic object-aware slam system,
2021. 2
[54] Tianwei Zhang, Huayan Zhang, Yang Li, Yoshihiko Naka-
mura, and Lei Zhang. Flowfusion: Dynamic dense rgb-d
slam based on optical flow. In 2020 IEEE International Con-
ference on Robotics and Automation (ICRA), pages 7322–
7328, 2020. 2
[55] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022. 2
10

<!-- page 11 -->
DynaGSLAM: Real-Time Gaussian-Splatting SLAM for Online Rendering,
Tracking, Motion Predictions of Moving Objects in Dynamic Scenes
Supplementary Material
A. Dynamic GS Segmentation & Flow
While online 3D motion segmentation is challenging and
slow today, we propose a novel Dynamic GS segmenta-
tion strategy (Fig.
3).
Our GS is initialized from the
point cloud of RGBD, , so we estimate 2D pixel motion
and align them to the GS motion. When our SLAM pro-
ceeds to current frame t, given two consecutive images
Ct −1, Ct ∈RW ×H×3, we first use a real-time optical
flow model (RAFT [45]) to estimate a flow image ft−1←t ∈
RW ×H×2, where each pixel stores its own 2D velocity, and
the velocities of moving pixels are distinct from the static
pixels. We then, compute the gradient of ft−1←t, which
detect edges and close the shapes to get a coarse motion
mask. By setting a click prompt at the object centroid, we
use prompt-based model SAM2 [37] to segment 2D mo-
tion. Our strategy enables an automatic pipeline to segment
moving pixels and filter out the static objects Whereas the
strategy handles well in general cases, it relies on robust
optical flow.
Incorrect segmentation of static objects as dynamic does
not deprecate GS quality since our dynamic GS manage-
ment also handles static GS, it only introduces minor ex-
tra computation that could be ignored in practice. On the
contrary, treating moving objects as static causes problems
as classical static GS management cannot manage dynamic
GS. However, our experiments show that our DynaGSLAM
has tolerance to the low quality of 2D motion segmenta-
tions, which is further discussed in Sec C and Fig. 7.
B. Trick of Dynamic Mapping Results
A popular un-written trick when evaluating mapping met-
rics (PSNR, SSIM, LPIPS) in previous works (like [17]) is
to set all invalid pixels to be 0, where the invalid mask is
defined by the invalid regions of the original depth maps.
However, the trick is unfair since setting estimation and
ground truth pixel values both to 0 significantly benefits all
metrics: 0 and 0 is infinitely similar. For a fair evaluation,
we cancel the cheating benefits by evaluating without mask,
which is why our implementation results of SplaTAM [17]
in table 1 is worse than the results proposed in table 3 of
[46]. However, even without the boost of the trick, our map-
ping accuracy is significantly better than SOTA baselines.
C. Ablation Studies
Robustness to Motion Segmentation Noise. As discussed
in Sec. 5 and Sec. A, 2D motion segmentation is an impor-
Figure 7. Failure of 2D Motion Segmentation further validates
the robustness of our dynamic GS management algorithm under
inaccurate motion priors. With some imperfect motion mask, our
DynaGSLAM still enables to reasonably manage dynamic GS, and
obtain outstanding mapping quality.
PSNR↑
SSIM↑
LPIPS↓
DynaPSNR↑
OMD (S4U)
30.6/31.0
95.1/95.7
15.1/15.1
34.6/31.1
TUM (xyz)
27.5/16.3
95.7/79.3
16.0/37.1
31.5/28.7
TUM (static)
26.9/12.8
96.1/77.7
14.0/37.8
30.3/27.0
TUM (rpy)
27.4/20.3
94.7/84.6
21.1/34.6
30.1/29.6
TUM (halfsphere)
27.2/19.4
94.5/83.1
20.0/35.4
30.7/31.6
Table 5. Ablation Study on the Impact of the Depth Quality. In
each cell, the metric is “refined (DepthAnythingV2) depth/original
sensor-depth”.
tant prior for our architecture, however, so far there is no
perfect solution for online real-time motion segmentation.
Although we made improvements on “automatic” segmen-
tation, the poor masks can be generated due to: 1. SAM2
[37] loses the moving objects in the tracking process, as
shown in Fig. 7(abc). 2. The optical flow gradients are
not strong enough to initialize any click prompts when the
speed of moving object is low, as shown in Fig. 7(d) 3.
The SAM2 tracker fails to perfectly segment the fast mov-
ing objects due to highly blurred shapes (as shown in 7(ef)).
However, with noisy motion masks, our DynaGSLAM still
achieves outstanding GS mapping quality, which further
validates the robustness of our dynamic GS management
algorithm under inaccurate motion priors.
Impact of depth quality. While our model is not very sen-
sitive to noisy 2D motion priors, it relies on good depth.
While the three datasets used in our work are all real
datasets with the depth from sensor, TUM’s depth is unre-
liable. We achieved outstanding results on Bonn and OMD
with their original depth (table 1 and 5). However, the depth
1

<!-- page 12 -->
maps from TUM include too large invalid regions, resulting
point cloud in poor quality. Nonetheless, we use the orig-
inal sensor-depth from OMD (table 5) and Bonn (table 1)
to get outstanding mapping quality, which still proves the
robustness of our DynaGSLAM with practical depth sensor
resource. Moreover, although the PSNR with noisy depth is
not ideal, its counterpart “DynaPSNR” is still competitive.
The good mapping quality of the dynamic region regard-
less of the static scene further validates our proposed novel
dynamic GS management.
D. Additional Results
Dynamic Mapping results.
We show additional qual-
itative comparisons of the GS mapping quality be-
tween our DynaGSLAM with SOTA baseline GS-SLAM
works (RTGSLAM[36], SplaTAM[17], GSSLAM [31], and
GSLAM [51]). Fig. 8 is an extension of Fig. 4 on the
Bonn Dataset. Fig. 9 is an extension of Fig. 1 on the TUM
Dataset. Our DynaGSLAM significantly outperforms these
baselines, especially around the moving object such as the
balloon and moving people. The failures of the baseline
GS-SLAM works can be attributed to two aspects: 1. The
past dynamic GS cannot be effectively deleted with static
GS management, which become outlier GS in the back-
ground and contaminate static GS, such as the remnant red
GS noises of RTGSLAM in Fig. 8, which belong to the
red hoodie of the person in the past frames. 2. The new
GS cannot be effectively added with static GS management,
such as the missing left leg of GSLAM in 9 (row 1). Our
novel proposed dynamic GS management algorithm over-
comes all these limitations proves to be robust and accurate
in the real dataset.
Dynamic Motion Tracking & Prediction Results.
We
show additional qualitative results of tracking & prediction
in Fig. 10. In all of the three datasets, our DynaGSLAM
shows the ability to synthesize unseen views by traversing
the time dimension. As an extension of Fig. 5, we show the
tracking (interpolation) and extremely hard prediction (ex-
trapolation) over long missing frames. With the transparent
white mask as the ground truth motion, we show that our
motion model successfully brings GS to the desirable posi-
tion, and the overlap of the dynamic entities (balloons and
people) with the ground truth motion mask shows the qual-
ity of our proposed novel motion function. By contrast, the
SOTA static GS-SLAM “RTGSLAM” [36] fails to correct
the motion. Due to the lack of dynamic GS management,
their background static GS are also contaminated by mov-
ing GS. Our DynaGSLAM generates some minor artifacts
under the extrapolation of long “Motion Horizon”, which
is mainly because we use an extremely low number of GS
for real-time efficiency, so that individual GS can adjust the
position and shape to cover more space whereas diminish
their photometric textures, this issue can be moderated by
trading-off the number and efficiency of GS.
We also conduct quantitative ablation study on the “Mo-
tion Horizon” for GS tracking and prediction on OMD and
TUM datasets, as shown in table 6. As the input frame in-
terval or the motion horizon grows, the difficulty for motion
estimation is increasing, and tracking(interpolation) always
performs better than prediction(extrapolation). While our
DynaGSLAM’s performance edge is huge under small mo-
tion horizon, the advantage gets unclear while the motion
horizon is growing. This is because the PSNR metric is very
sensitive to even a small displacement - The exact same two
patterns get low PSNR if overlapping with a minor displace-
ment. With the motion horizon grows, displacement errors
are zoomed out. However, we argue that for a long “Mo-
tion Horizon”, visual results give fairer comparisons, such
as the accurate fitting of the moving object’s contour with
the ground truth motion mask in Fig. 10.
2

<!-- page 13 -->
Figure 8. GS Mapping Rendering Comparisons on the Bonn Dataset. From top to bottom, the four scenes are: balloon, balloon2,
ps track, ps track2. For each scene, the first row shows the RGB rendering results, the second row shows the absolute error between the
rendered RGB to the ground truth. Our DynaGSLAM is obviously better than other SOTA GS-SLAM, especially at the moving entities
such as the yellow balloon and the person.
3

<!-- page 14 -->
Figure 9. GS Mapping Rendering Comparisons on the TUM Dataset. From top to bottom, the four scenes are: fr3 walking xyz,
fr3 walking static, fr3 walking static, fr3 walking halfsphere. For each scene, the first row shows the RGB rendering results,
the second row shows the absolute error between the rendered RGB to the ground truth. Our DynaGSLAM is obviously better than other
SOTA GS-SLAM, especially at the moving people.
4

<!-- page 15 -->
OMD (S4U)
TUM (fr3 walking static)
Method
DynaGSLAM
RTGSLAM
DynaGSLAM
RTGSLAM
Mapping
30.63/34.58
17.12/15.66
26.88/30.30
13.92/16.59
Interval = 2 frames, Interpolate middle frame (1st)
20.69/16.69
16.90/15.19
20.83/19.92
14.25/17.07
Interval = 5 frames, Interpolate middle frame (3rd)
18.80/14.54
16.70/14.46
19.98/17.82
14.54/16.54
Interval = 2 frames, Extrapolate next 1st frame
20.45/16.10
16.51/14.55
18.64/15.51
14.09/16.58
Interval = 5 frames, Extrapolate next 1st frame
20.57/16.30
17.05/15.32
20.01/17.86
14.50/16.65
Interval = 5 frames, Extrapolate next 2nd frame
18.73/14.33
16.44/14.27
17.76/14.85
14.24/15.82
Interval = 5 frames, Extrapolate next 5th frame
17.24/13.30
15.65/13.14
15.82/14.84
13.99/12.27
Table 6.
Ablation Study on Motion Horizon for Tracking and Prediction.
In each cell the metric is represented as
“PSNR↑/DynaPSNR↑”. The conditions on “Interval = 5 frames, Interpolate middle frame (3rd)” and “Interval = 5 frames, Extrapolate
next 5th frame” are the tracking and prediction corresponding to Fig. 5 and Fig. 10. Please check the annotation explaination in Fig. 5 to
understand the condition in the table.
Figure 10. Tracking & Prediction results on OMD, TUM and Bonn datasets. This figure is an extension of Fig. 5 showing the tracking
and prediction quality of our DynaGSLAM with our proposed novel GS motion function. Please check the annotation explanation in Fig.
5.
5
