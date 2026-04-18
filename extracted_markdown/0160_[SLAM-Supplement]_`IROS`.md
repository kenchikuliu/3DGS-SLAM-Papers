<!-- page 1 -->
MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using
Vision, Depth, and Inertial Measurements
Lisong C. Sun, Neel P. Bhatt, Member, IEEE, Jonathan C. Liu, Zhiwen Fan,
Zhangyang Wang, Senior Member, IEEE, Todd E. Humphreys, Senior Member, IEEE, and Ufuk Topcu, Fellow, IEEE
MM3DGS Framework
IMU@100Hz
Camera@30Hz
Pre-integration
Depth
Pose Optimization
ℒ𝑡𝑟𝑎𝑐𝑘𝑖𝑛𝑔= 𝟙𝑂𝐺,𝑇𝐶
(ℒ𝐶+ 𝜆𝐷ℒ𝑑𝑒𝑝𝑡ℎ)
Keyframe Selection
Classify as keyframe if:
1.
Covisibility < 95%
2.
Min NIQE in sliding window
Mapping
Gaussian Initialization
Add new Gaussian per pixel of 
keyframe if:
1.
Opacity < 0.5
2.
Depth error > 50x median 
depth error
ℒ𝑚𝑎𝑝𝑝𝑖𝑛𝑔= 𝜆𝐶ℒ𝑝ℎ𝑜𝑡𝑜+ 𝜆𝑆ℒ𝑆𝑆𝐼𝑀+ 𝜆𝐷ℒ𝑑𝑒𝑝𝑡ℎ
Mobile Robot
Fig. 1: Overview of the MM3DGS framework. We receive camera images and inertial measurements from a mobile robot. We utilize depth
measurements and IMU pre-integration for pose optimization using a combined tracking loss. We apply a keyframe selection approach
based on image covisibility and the NIQE metric across a sliding window and initialize new 3D Gaussians for keyframes with low opacity
and high depth error [1]. Finally, we optimize parameters of the 3D Gaussians according the mapping loss for the selected keyframes.
Abstract— Simultaneous localization and mapping is essential
for position tracking and scene understanding. 3D Gaussian-
based map representations enable photorealistic reconstruction
and real-time rendering of scenes using multiple posed cameras.
We show for the first time that using 3D Gaussians for
map representation with unposed camera images and iner-
tial measurements can enable accurate SLAM. Our method,
MM3DGS, addresses the limitations of prior neural radiance
field-based representations by enabling faster rendering, scale
awareness, and improved trajectory tracking. Our framework
enables keyframe-based mapping and tracking utilizing loss
functions that incorporate relative pose transformations from
pre-integrated inertial measurements, depth estimates, and
measures of photometric rendering quality. We also release a
multi-modal dataset, UT-MM, collected from a mobile robot
equipped with a camera and an inertial measurement unit.
Experimental evaluation on several scenes from the dataset
shows that MM3DGS achieves 3× improvement in tracking and
5% improvement in photometric rendering quality compared to
the current 3DGS SLAM state-of-the-art, while allowing real-
time rendering of a high-resolution dense 3D map.
Lisong C. Sun, Neel P. Bhatt, Jonathan C. Liu, Zhiwen Fan, Zhangyang
Wang, Todd E. Humphreys, and Ufuk Topcu are with the University
of
Texas
at
Austin,
Austin,
TX,
USA.
Email: {codey.sun,
npbhatt, jonathanliu88, zhiwenfan, atlaswang,
todd.humphreys, utopcu}@utexas.edu
I. INTRODUCTION
Simultaneous localization and mapping (SLAM), the task
of generating a map of the environment along with estimating
the pose of a sensor, is an essential enabler in applications
such as aerial mapping, augmented reality, and autonomous
mobile robotics [2], [3]. 3D scene reconstruction and sensor
localization are essential capabilities for an autonomous sys-
tem to perform downstream tasks such as decision-making
and navigation [4]. As a result, SLAM plays a pivotal role
in advancing the capabilities of autonomous systems.
The representation used for mapping and the type of sensor
input utilized is a fundamental choice that has a direct impact
on SLAM performance. Although SLAM approaches using
sparse point clouds to represent the operating environment
yield state-of-the-art tracking accuracy [5], [6], the generated
maps are disjoint due to sparsity and visually inferior to
newer 3D reconstruction methods. While visual quality is
irrelevant for the sole purpose of navigation, the creation
of photorealistic maps is valuable for human consumption,
semantic segmentation, and post-processing. Neural radiance
Webpage: https://vita-group.github.io/MM3DGS-SLAM
arXiv:2404.00923v1  [cs.CV]  1 Apr 2024

<!-- page 2 -->
field (NeRF)-based SLAM approaches yield output maps that
are spatially and photorealistically dense [7], [8], but are
computationally expensive at inference time and hence are
not capable of accurately tracking in real-time. To address
this shortcoming, recent works utilize the real-time rendering
capability of a 3D Gaussian-based map representation and
perform 3D Gaussian splatting (3DGS) to generate 2D
renderings [9], [10].
A precursor to 3DGS is depth initialization of the Gaus-
sians. Most approaches use depth inputs from relatively
expensive sensors such as LiDARs, which may not be readily
available on a device such as a consumer phone [11]. Other
approaches use an RGB-D camera providing depth informa-
tion through stereoscopic depth estimation [10]. However,
relying solely on RGB-D cameras may lead to erroneous
depth, especially as the distance from the camera increases,
leading to degraded tracking accuracy.
To address these shortcomings, we present the first real-
time visual-inertial SLAM framework using 3D Gaussians
for efficient and explicit map representation with inputs from
a single monocular camera or RGB-D camera along with
inertial measurements that may be readily obtained from
most modern smartphones. Our approach is capable of real-
time photorealistic 3D reconstructions of the environment,
yielding accurate camera pose tracking for SLAM. We
release a multi-modal dataset consisting of several scenes
and the required sensory inputs which we use to evaluate our
approach. Our approach outperforms SplaTAM, the state-of-
the-art RGB-D-based 3DGS SLAM approach, by 3x in tra-
jectory tracking and achieves a 5% increase in photorealistic
rendering quality.
In summary, our key contributions are as follows:
• We integrate inertial measurements and depth estimates
from an unposed monocular RGB or RGB-D camera
into our real-time MM3DGS SLAM framework using
3D Gaussians for scene representation. Our framework
enables scale awareness as well as lateral, longitudinal,
and vertical trajectory alignment. Our framework can
utilize inputs from inexpensive sensors available on
most consumer smartphones.
• We release a multi-modal dataset collected using a
mobile robot consisting of several indoor scenarios
with RGB and RGB-D images, LiDAR depth, 6-DOF
inertial measurement unit (IMU) measurements, as well
as ground truth trajectories for error analysis.
• We achieve superior quantitative and qualitative trajec-
tory tracking (3x improvement) and photometric render-
ing (5% improvement) results compared to the current
state-of-the-art 3DGS SLAM baseline.
II. RELATED WORKS
A. SLAM Map Representations
Sparse visual SLAM algorithms, such as ORB-SLAM [5]
and OKVIS [6], are designed to estimate precise camera
poses while producing only sparse point clouds for map
representation. Conversely, dense visual SLAM methodolo-
gies [12], [13] aim to construct a dense representation of
the scene. Map representations in dense SLAM are broadly
classified into two categories: view-centric and world-centric.
View-centric representations encode 3D information using
keyframes accompanied by depth maps [12]. On the other
hand, world-centric approaches anchor the 3D geometry of
the entire scene within a consistent global coordinate system,
typically represented through surfels [13] or by utilizing
occupancy values within voxel grids [14].
B. Efficient 3D Representation
Recent advancements in neural rendering, exemplified by
NeRFs [15] and subsequent developments [16], [17], have
demonstrated significant progress in novel view synthesis
using foundational 3D representations such as MLPs, voxel
grids, or hash tables. Although these NeRF-inspired models
exhibit impressive results, they often necessitate extensive
training periods for individual scenes. The introduction of
3DGS, which this paper employs, offers a solution to the
issue of training efficiency. This volumetric rendering ap-
proach depicts a 3D scene as a collection of explicit Gaus-
sian distributions [18]. Initially, reconstructing a scene with
3DGS required applying structure-from-motion techniques
like COLMAP [19], [20] to determine camera poses before
optimization, but recent studies have aimed to bypass this
prerequisite by leveraging depth measurements or monocular
depth estimators [10], [21]. Concurrently, several research
initiatives have explored the utilization of 3DGS within
SLAM frameworks, with works in [22], [10], [9], [23] em-
ploying RGB-D sequences. Specifically, Gaussian Splatting
SLAM [9] considers both RGB and RGB-D settings, but
none of these frameworks fuse inertial measurements. At the
time of writing, SplaTAM, an RGB-D-based 3DGS method,
is the only method with publicly available source code and
is thus considered as the baseline [10].
C. Multi-modal SLAM Frameworks
Visual sensing, while effective, has its limitations, such
as susceptibility to motion blur and exposure changes. To
mitigate the vulnerabilities of individual sensors, a standard
approach is to employ multi-modal sensing through sensor
fusion. Prior research has delved into enhancing system
robustness by integrating semantic mapping and developing
LiDAR-camera SLAM systems augmented with laser range
finders [11], [24], [25]. In contrast, our focus is on the
fusion of inertial measurements, which are favored for their
high data acquisition rates and proficiency in tracking rapid
movements within brief timeframes.
III. METHOD
The MM3DGS SLAM framework consists of four main
stages: pose optimization (tracking), keyframe selection,
Gaussian initialization, and mapping. An overview of the
framework is illustrated in Fig. 1.
A. 3D Gaussian Splatting
The underlying scene map is represented using a set of 3D
Gaussians G, where the ith Gaussian is defined by position

<!-- page 3 -->
µi, shape Σi, opacity oi, and color ci. Recall that given a
mean, µ, and covariance matrix, Σ, a Gaussian distribution
is defined as
G(x) = exp(−1
2(x −µ)Σ−1(x −µ)T)
(1)
By applying eigenvalue decomposition to Σ, the covari-
ance can be decomposed into the form, Σ = RSSTRT,
where R represents an orthonormal rotation matrix and S
represents a diagonal scaling matrix. In this way, the shape of
3D Gaussians can be optimized while keeping Σ symmetrical
and positive-semidefinite.
A set of 3D Gaussians can be rasterized into an image via
“splatting,” i.e., projecting the Gaussians onto a 2D image
plane. The 2D view-space covariance matrix, Σ′, can be
computed as Σ′ = JWΣ(JW)T, where J is the Jacobian of
the affine approximation of the projection matrix, W is the
world to view frame transformation matrix, and Σ is the 3D
covariance matrix of the respective Gaussian.
Specifically, given a set of Gaussian features, G, and a
camera pose, Tc, the color, C, of a pixel is calculated by
blending N Gaussians that overlap the pixel, ordered by non-
increasing depth given by
C(G, Tc) =
N
X
i=1
ciαi
i−1
Y
j=1
(1 −αj)
(2)
where ci is the color of the ith Gaussian and αi is sampled
from the ith splatted 2D Gaussian distribution at the pixel
location. Similarly, the opacity of a pixel can be calculated
as
O(G, Tc) =
N
X
i=1
αi
i−1
Y
j=1
(1 −αj)
(3)
Since this rendering process is differentiable, G can be
optimized using the L1 loss between the rendered image and
the ground truth undistorted image, I:
Lphoto = L1(I, C(G, Tc))
(4)
B. Tracking
The tracking process consists of camera pose optimiza-
tion given a fixed 3D Gaussian map. To enable gradient
backpropagation to the camera pose, the inverse camera
transformation is applied to the Gaussian map while keeping
the camera fixed. This achieves identical rendering without
implementing camera pose gradients in the 3DGS rasterizer.
Subsequently, the 3D Gaussian map is frozen, and the camera
pose is optimized according to the following loss function:
Ltracking = (1O(G,Tc))(Lphoto + λDLdepth)
(5)
where 1O(G,Tc) is an indicator function defined as
1O(G,Tc) =
(
1
if O(G, Tc) > 0.99
0
otherwise
and Ldepth is the depth loss described in detail in Sec. III-C.
Since the map is not guaranteed to cover the entire extent of
the current frame, pixels with an opacity < 0.99 are masked.
To aid in convergence, a dynamics model can be applied
prior to optimization to provide an initial guess for the
camera pose. In most cases, a constant velocity model is
used. However, in the presence of an inertial sensor, tracking
accuracy can be improved by utilizing inertial measurements
as is later described in Sec. III-D. Note that tracking is
skipped for the first frame as there is no existing map yet
and an identify transformation matrix is assumed as an initial
guess.
C. Depth Supervision
When tracking camera poses using a 3D Gaussian map,
it is essential for the map to encode accurate geometric
information absent which the tracking may diverge. This
is particularly true for an underoptimized map, in which
Gaussians are not trained long enough to converge to the
correct position. The use of depth priors solves this problem
by providing reasonable initial estimates for Gaussian po-
sitions, minimizing both inconsistent geometry and training
time. Further, depth priors can supervise the map training
loss to prevent geometric artifacts from overfitting on limited
views.
To render depth in a differentiable manner, a second ras-
terization pass is performed with color ci of each Gaussian
replaced with its projected depth on the image plane.
In the RGB-D case, these depth priors can be gathered
directly through a depth sensor or using stereoscopic depth.
However, in the absence of a such sensors, monocular
dense depth estimation networks, such as DPT [26], can be
used. Since dense depth estimators output a relative inverse
depth, the estimated and rendered depths cannot be directly
compared. Following [27], the depth loss is instead computed
using the linear correlation (Pearson correlation coefficient)
between the estimated and rendered depth maps De and Dr:
Ldepth =
Cov(De, Dr)
p
Var(De)Var(Dr)
(6)
This depth correlation term is appended to the loss functions
used in Eqs. (5) and (12).
For initializing Gaussian positions, one must first resolve
the scale ambiguity of the depth estimate. This can be done
by solving for a scaling σ and shift θ that fits the depth
estimate to the current map. This can be modeled as a linear
least squares problem of the form
de
1 σ
θ

= dr
(7)
where de and dr are the flattened vectors of De and Dr.
Once the estimated depths are properly fitted to the existing
map, new Gaussians can be initialized for unseen areas, ini-
tializing underoptimized maps with geometric information.

<!-- page 4 -->
Fig. 2: Our dataset provides RGB images (top left), depth
images (top right), IMU measurements (bottom left), and
LIDAR point clouds (bottom right). The above examples
were taken from the Ego-drive scene.
D. Inertial Fusion
Prior to optimizing the camera pose at the current frame,
an initial pose estimate is required to guide optimization. A
good initial estimate can lead to faster optimization times.
In addition, in underoptimized areas where the convergence
basins may be small, good initial estimates are essential to
prevent tracking divergence.
In a monocular setting, pose estimates can be extrapolated
by assuming constant velocity between consecutive frames.
However, this model breaks down during presence of vigor-
ous camera motion and low image frame rate.
Inertial measurements obtained via an IMU can be in-
tegrated to accurately propagate the camera pose between
frames and yield meaningful initial estimates. Most 6-degree-
of-freedom (6-DOF) IMUs provide linear acceleration mea-
surements through accelerometers, a = [¨x, ¨y, ¨z], as well as
angular velocities, ˙Θ = [ ˙α, ˙β, ˙γ] , in 3D space. Given IMU
measurements are readily available in almost all consumer
phones and are relatively inexpensive compared to cameras
and LiDARs, this valuable information can be practically
availed in any setting.
The change in position at time, t, expressed in the previous
coordinate frame, t−1∆pt, can be computed using Eq. (8),
t−1∆pt = vt−1 × t + 1
2at2
(8)
where velocity is computed as vt−1 = vt−2 + at−1 × t.
Similarly, the change in angular position at time, t, ex-
pressed in the previous coordinate frame, t−1∆pt, can be
computed using Eq. (9),
t−1∆Θt = Θt−1 × t
(9)
Using, Eq. (8) and Eq. (9) the relative transformation
between two consecutive coordinate frames, t−1
t
TI, can be
constructed as in Eq. (10),
t−1
t
TI = [t−1
t
R | t−1∆pt]
(10)
where the relative rotation matrix, t−1
t
R, is constructed using
t−1∆Θt from Eq. (9). Using the static transform between
Fig. 3: A depiction of the mobile robot platform (left)
equipped with a RGB-D camera, IMU, and a LiDAR and
the test environment (right) featuring a 16 camera Vicon-
based ground truth system.
the IMU and camera frame, the relative transform between
consecutive camera frames can be computed as per Eq. (11),
t−1
t
Tc =C
I T t−1
t
TI
(11)
To obtain the transform between the two arbitrary frames
within a sliding window, the relative transform in Eq. (11),
can be chained from the current frame to the destination
frame of interest.
Note that this open-loop method does not estimate internal
IMU biases. This is because errors in t−1
t
Tc are small within
short time deltas, and these small errors are optimized away
by the visual camera pose optimization in Section III-B.
However, this method becomes less robust as both video
frame rate and IMU quality decreases. We leave closed-loop
inertial fusion with bias estimation as future work.
E. Gaussian Initialization
To cover unseen areas, new Gaussians are added each
keyframe at pixels where the opacity is below 0.5 and the
depth error exceeds 50 times the median depth error. These
new Gaussians are added per-pixel, with RGB initialized at
the pixel color, position initialized at the depth measure-
ment/estimate (elaborated in Sec. III-C), opacity at 0.5, and
scaling initialized isotropically to cover the extent of a single
pixel at the initialized depth.
F. Keyframe Selection
In a real-time setting, it is impractical to optimize the 3D
Gaussian map over the entire set of video frames. However,
one can exploit the temporal locality of video frames to prune
away most frames from the optimization pool. This process
of selectively choosing a subset of frames to optimize is
known as keyframing.
In general, keyframes should be chosen to minimize the
number of redundant frames processed and maximize the
information gain. MM3DGS achieves this by selecting an
input frame as a keyframe when the map does not contain
enough information to track the current frame. This is
calculated using a covisibility metric, which defines which
Gaussians are visible in multiple keyframes.
To do so, first, a depth rendering is created using the
current frame’s estimated pose. This depth rendering can be
backprojected into a point cloud. This point cloud can be
projected onto the image plane of any keyframe’s estimated

<!-- page 5 -->
GT
SplaTAM
Ours
Square-1
Ego-centric-1
Ego-drive
Fast-straight
Fig. 4: Qualitative results on UT-MM dataset:
RGB and depth renderings of UT-MM scenes. Note that the ground truth
(GT) depths are captured with depth cameras, and thus are imperfect. Our method exhibits geometric details not present in
the GT depth, as well as fewer RGB artifacts compared to SplaTAM.
TABLE I: Multi-modal SLAM results on the UT-MM dataset: ATE RMSE ↓is in cm and PSNR ↑is in dB, with SplaTAM
is used as a baseline. Best results are in bold. Both depth and inertial measurements benefit tracking and image quality.
Method
Avg
Square-1
Ego-centric-1
Ego-drive
Fast-straight
ATE
PSNR
ATE
PSNR
ATE
PSNR
ATE
PSNR
ATE
PSNR
SplaTAM (RGB-D)
12.06
22.03
32.86
18.67
4.40
22.78
4.20
20.61
6.78
26.07
Ours (RGB)
39.14
19.73
59.48
16.54
4.09
23.151
67.20
17.51
25.78
21.71
Ours (RGB+IMU)
33.23
19.58
44.26
17.01
3.41
22.96
68.50
17.12
16.78
21.24
Ours (RGB-D)
8.75
22.20
20.38
16.55
6.86
22.24
4.25
23.58
3.52
26.42
Ours (RGB-D+IMU)
3.98
23.30
7.11
18.59
1.15
24.95
4.54
23.61
3.13
26.05
pose. The covisibility can then be defined as the percentage
of points visible in the keyframe. If this covisibility drops
below 95%, the current frame is added as a keyframe.
In the presence of visual noise, such as when an input
frame depicts motion blur, keyframes with low image quality
may persist and degrade the reconstruction and tracking
quality. To prevent outlier noisy images from being selected,
the Naturalness Image Quality Evaluator (NIQE) metric [1]
is used to select the highest quality frame across a sliding
window.
During the mapping phase, Gaussians are optimized over
the set of keyframes covisible with the current frame. In this
way, the optimization affects all of the relevant measurements
available while minimizing processing of redundant frames
and reducing computational load.
G. Mapping
The mapping process optimizes the Gaussian features
visible within the current covisible set of keyframes. For the
current frame and each selected keyframe, the 3D Gaussians
are optimized according to
Lmapping = λCLphoto + λSLSSIM + λDLdepth
(12)
where LSSIM is the D-SSIM loss [28]. The other terms
are identical to those in Eq. (5). To prevent unconstrained
growth of Gaussians into unobserved areas, Gaussians are
constrained to be isotropic.
IV. EXPERIMENTAL SETUP
A. Datasets
Several visual-intertial SLAM datasets exist to test visual-
inertial fusion, however they provide grayscale images rather
than RGB images, which fails to test reconstruction capa-
bilities [29], [30]. An exception is the AR Table dataset,
which contains RGB images along with inertial measure-
ments. However, it is limited to egocentric scenes focused
at only tables [31]. To bridge this gap due to the absence
of RGB visual-inertial datasets, we release such a dataset,
dubbed UT Multi-modal (UT-MM), captured in the Anna
Hiss Gymnasium at the University of Texas at Austin. A
Clearpath Jackal unmanned ground vehicle (UGV), shown
in Fig. 3, is equipped with a Lord Microstrain 3DM-GX5-
25 IMU, an Intel Realsense D455 camera, and an Oster 64
line LiDAR. The UGV captures RGB and depth images at
30 frames per second, inertial measurements at 100 Hz and
point clouds at 10 Hz. In addition, the ground truth trajectory
of the UGV is captured using a 16 camera Vicon motion
capture system with sub-mm precision. A view from one of
the 16 cameras is shown in Fig. 3.
The UT-MM dataset contains eight different scenes la-
beled: Ego-centric-1, Ego-centric-2, Ego-drive, Square-1,
Square-2, Fast-straight, Slow-straight-1, and Slow-straight-
2. The three straight scenes are short scenes with the Jackal
driving along a straight path with different initial positions
and speeds. Square-1 and Square-2 consist of straight driving
with three turns to create a loop depicting a square-shaped
trajectory. Ego-drive consists of the robot being driven
around several obstacles, capturing the obstacles from 360◦.
The Ego-centric-1 and Ego-centric-2 datasets were captured
with the Jackal mounted on a omnidirectional trolley revolv-
ing around an object of interest while keeping it focused
nearly at the center of the image. In some scenes, such as
in the beginning of Ego-centric-1, dynamic objects are also
present. These may either be cropped out or be included to

<!-- page 6 -->
Fig. 5: Tracking results for the UT-MM Square-1 scene.
The blue solid line denotes the tracked trajectory, while the
red dotted line denotes the ground truth. Top: monocular
RGB case exhibits substantial drift. Middle: RGB-D case
fixes Z drift, but XY drift persists. Bottom: Adding IMU
measurements to RGB-D fixes XY drift.
test the robustness of a SLAM method to dynamic objects.
As shown in Fig. 2, there are several sensing modalities
available in the UT-MM dataset. For each frame in the
dataset, a corresponding RGB image and depth map is
available. At the same time, IMU measurements are provided
at a rate of 100 Hz along with accurate point clouds cap-
tured. RGB-D images and IMU measurements are software-
synchronized; an additional hardware synchronization board
takes Pulse Per Second (PPS) input to synchronize LiDAR
and IMU measurements. Our framework does not make use
of highly accurate LiDAR measurements given the high cost
of the sensor and lack of easy access by a common user.
Nonetheless, the LiDAR can be used as ground truth during
evaluation of a depth estimator.
In addition to UT-MM, we test our framework on the
TUM RGB-D dataset to evaluate the performance of the
monocular SLAM model in a different setting [32]. Note that
TUM RGB-D provides accelerometer data but no gyroscope
measurements and thus is not suitable for 6-DOF inertial
fusion.
B. Metrics
To evaluate our model, we use two main metrics: 1)
tracking accuracy, measured via absolute tracking error root
Fig. 6: RGB-D+IMU tracking trajectory results for selected
scenes in the UT-MM dataset. Top: Fast-straight. Middle:
Ego-drive. Bottom: Ego-centric-1.
mean square error (ATE RMSE), and 2) scene reconstruction
quality, measured via peak signal-to-noise ratio (PSNR). An
Umeyama point alignment algorithm aligns the trajectory
generated using the model of interest with the ground truth
trajectory [33]. SplaTAM, an RGB-D 3DGS SLAM model
that this work extends, is used as a baseline [10].
C. Implementation
We run our framework on an RTX A5000 GPU with 24GB
VRAM. We do not multi-thread our framework and hence the
tracking and mapping threads run sequentially. For all scenes,
we run pose optimization for 100 iterations followed by
150 iterations of map optimization. We set the loss function
hyperparameters to λC = 0.8, λS = 0.2, and λD = 0.05.
V. RESULTS AND DISCUSSION
We conduct a comprehensive qualitative and quantitative
evaluation of our framework on the UT-MM dataset and
perform an ablation study on the TUM RGB-D dataset to
justify the use of NIQE keyframing and Pearson correlation
loss.
A. Effect of Incorporating Multi-modal Sensor Information
Table I shows the tracking and rendering results for
SplaTAM, a state-of-the-art RGB-D 3DGS baseline, and var-
ious configurations of MM3DGS on scenes from the UT-MM
dataset. Considering the average ATE and PSNR metrics, it

<!-- page 7 -->
Fig. 7: Top: depth initialized by DPT. Bottom: depth initial-
ized by depth camera. The DPT estimate exhibits warping
along the Z axis compared to the depth camera.
is evident that addition of inertial measurements improves
tracking and image quality for both the RGB and RGB-D
configurations. The RGB-D+IMU configuration consistently
outperforms others, demonstrating a 3x improvement in ATE
RMSE and 5% improvement in PSNR compared to the base-
line on average. This demonstrates the value of integrating
inertial measurements in our framework enhancing trajectory
tracking and rendering quality as previously claimed.
Furthermore, we showcase qualitative comparisons of the
rendered RGB and depth images against SplaTAM and
ground truth in Fig. 4. The superior image rendering quality
of our framework on all of the scenes compared to SplaTAM
signifies the value of the RGB-D+IMU configuration of
MM3DGS. Due to the use of lightly weighted depth cor-
relation loss rather than a heavily weighted direct depth
loss implemented in SplaTAM, MM3DGS is able to capture
geometric details that are not present in the noisy depth input
through the photometric loss in Eq. (4). For instance, in the
Ego-centric-1 scene, our depth rendering includes the many
small holes on the back of the chair while SplaTAM instead
overfits to the depth input and is not able to capture the holes.
MM3DGS also exhibits significantly fewer visual artifacts
in its RGB renderings compared to SplaTAM which depicts
considerably more missing colors and floaters. Both methods
render at 90 fps on an RTX A5000 GPU.
To visualize the effect of incorporating IMU and depth
measurements on tracking, we plot the trajectory of the RGB,
RGB-D, and RGB-D+IMU configurations on the UT-MM
Square-1 scene as shown in Fig. 5. The integration of depth
measurements helps with trajectory alignment by eliminating
drift in the Z-axis. The dense depth estimates provided by
DPT are indeed warped in the Z direction as shown in
Fig. 7. However, drift along the XY plane persists and is
addressed only with the addition of inertial measurements in
our framework. This highlights the limitations of monocular
depth estimators and the importance of integrating both
inertial and depth sensing modalities.
w/o Pearson corr.
w/ Pearson corr.
Fig. 8: RGB and depth renderings of TUM RGB-D with and
without Pearson correlation loss. The addition of Pearson
correlation results in similar RGB quality while increasing
geometric consistency, particularly along edges. Note the
shift in brightness on the right side of the RGB renders;
this is due to a sudden exposure change in the input camera
frames.
TABLE II: Monocular RGB configuration results on the
TUM RGB-D dataset. ATE RMSE ↓is in cm. Monocular
RGB SLAM provides comparable performance as RGB-D
baselines.
Method
fr1/desk
fr1/desk2
fr2/xyz
SplaTAM (RGB-D)
3.35
6.54
1.24
Ours (RGB)
3.51
5.78
2.04
TABLE III: Monocular RGB configuration ablation results
on the TUM RGB-D freiburg1/desk2 scene. ATE
RMSE ↓is in cm and PSNR ↑is in dB. Performance is
best with both Pearson Corr. loss and NIQE keyframing.
NIQE Keyframing
Pearson Corr. Loss
ATE RMSE
PSNR
✓
✗
Fails
Fails
✗
✓
7.8
18.05
✓
✓
5.78
18.33
B. Performance of the Monocular RGB Configuration
To further gauge the performance of the proposed monoc-
ular RGB SLAM configuration, we evaluate its performance
on the TUM RGB-D dataset. The ATE RMSE results on
select scenes from the dataset are shown in Table II. In
ego-centric scenes, the RGB model exhibits comparable per-
formance to SplaTAM’s RGB-D model, demonstrating that
depth measurements do not provide much added information
given a wide range of multi-view constraints. However, the
RGB model fails on scenes that involve longer trajectories
which may be alleviated by the addition of loop closure
methods.

<!-- page 8 -->
C. Ablation Study
We
also
perform
an
ablation
study
on
the
freiburg1/desk2
scene
due
to
presence
of
high
motion blur and exposure changes. Table III showcases
RGB tracking results with Pearson correlation depth loss and
NIQE keyframing ablated. Without the Pearson correlation
depth loss, tracking fails. The effect of the depth correlation
loss in improving geometric consistency is illustrated in
Fig. 8. Further, the addition of NIQE keyframing increases
both tracking accuracy and image quality.
VI. CONCLUSIONS
We presented MM3DGS, a multi-modal SLAM framework
built on a 3D Gaussian map representation that utilizes
visual, inertial, and depth measurements to enable real-time
photorealistic rendering and improved trajectory tracking.
We evaluate our framework on a new multi-modal dataset,
UT-MM, that includes RGB-D images, IMU measurements,
LiDAR depth, and ground truth trajectories. MM3DGS
achieves superior tracking accuracy and rendering qual-
ity compared to state-of-the-art baselines. In addition, we
present an ablation study to highlight the importance of
our framework. MM3DGS can be implemented in a wide
range of applications in robotics, augmented reality, and
mobile computing due to its use of commonly available
and inexpensive sensors. As future work, MM3DGS can be
extended to include tightly-coupled IMU fusion and loop
closure to further enhance tracking performance.
REFERENCES
[1] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “com-
pletely blind” image quality analyzer,” IEEE Signal Processing Letters,
vol. 20, no. 3, pp. 209–212, 2013.
[2] M. Contreras, N. P. Bhatt, and E. Hashemi, “A stereo visual odometry
framework with augmented perception for dynamic urban environ-
ments,” in 2023 IEEE 26th International Conference on Intelligent
Transportation Systems (ITSC).
IEEE, 2023, pp. 4094–4099.
[3] J. Polvi, T. Taketomi, G. Yamamoto, A. Dey, C. Sandor, and H. Kato,
“SlidAR: A 3d positioning method for SLAM-based handheld aug-
mented reality,” Computers & Graphics, vol. 55, pp. 33–43, 2016.
[4] H. Bavle, P. De La Puente, J. P. How, and P. Campoy, “VPS-
SLAM: Visual planar semantic SLAM for aerial robotic systems,”
IEEE Access, vol. 8, pp. 60 704–60 718, 2020.
[5] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM: a
versatile and accurate monocular SLAM system,” IEEE transactions
on robotics, vol. 31, no. 5, pp. 1147–1163, 2015.
[6] S. Leutenegger, S. Lynen, M. Bosse, R. Siegwart, and P. Furgale,
“Keyframe-based visual–inertial odometry using nonlinear optimiza-
tion,” The International Journal of Robotics Research, vol. 34, no. 3,
pp. 314–334, 2015.
[7] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “iMAP: Implicit map-
ping and positioning in real-time,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 6229–6238.
[8] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald,
and M. Pollefeys, “NICE-SLAM: Neural implicit scalable encoding
for SLAM,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2022, pp. 12 786–12 796.
[9] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian
Splatting SLAM,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024.
[10] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “SplaTAM: Splat, track & map 3d
gaussians for dense RGB-D SLAM,” arXiv, 2023.
[11] S. Hong, J. He, X. Zheng, H. Wang, H. Fang, K. Liu, C. Zheng, and
S. Shen, “LIV-GaussMap: LiDAR-inertial-visual fusion for real-time
3d radiance field map rendering,” arXiv preprint arXiv:2401.14857,
2024.
[12] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense
tracking and mapping in real-time,” in 2011 international conference
on computer vision.
IEEE, 2011, pp. 2320–2327.
[13] T. Schops, T. Sattler, and M. Pollefeys, “BAD SLAM: Bundle adjusted
direct RGB-D SLAM,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2019, pp. 134–144.
[14] A. Dai, M. Nießner, M. Zollh¨ofer, S. Izadi, and C. Theobalt, “Bundle-
fusion: Real-time globally consistent 3d reconstruction using on-
the-fly surface reintegration,” ACM Transactions on Graphics (ToG),
vol. 36, no. 4, p. 1, 2017.
[15] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp.
99–106, 2021.
[16] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 5501–5510.
[17] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions
on Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022.
[18] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics (ToG), vol. 42, no. 4, pp. 1–14, 2023.
[19] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Conference on Computer Vision and Pattern Recognition (CVPR),
2016.
[20] J. L. Sch¨onberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixel-
wise view selection for unstructured multi-view stereo,” in European
Conference on Computer Vision (ECCV), 2016.
[21] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang,
“Colmap-free 3d gaussian splatting,” arXiv preprint arXiv:2312.07504,
2023.
[22] C. Yan, D. Qu, D. Wang, D. Xu, Z. Wang, B. Zhao, and X. Li, “GS-
SLAM: Dense visual SLAM with 3d gaussian splatting,” 2024.
[23] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-SLAM:
Photo-realistic dense SLAM with gaussian splatting,” 2023.
[24] J. Jeong, T. S. Yoon, and J. B. Park, “Towards a meaningful 3d map
using a 3d lidar and a camera,” Sensors, vol. 18, no. 8, p. 2571, 2018.
[25] C. Jiang, D. P. Paudel, Y. Fougerolle, D. Fofi, and C. Demonceaux,
“Static-map and dynamic object reconstruction in outdoor scenes using
3-d motion segmentation,” IEEE Robotics and Automation Letters,
vol. 1, no. 1, pp. 324–331, 2016.
[26] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for
dense prediction,” ArXiv preprint, 2021.
[27] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, “FSGS: Real-time few-shot
view synthesis using gaussian splatting,” 2023.
[28] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality
assessment: from error visibility to structural similarity,” IEEE Trans-
actions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[29] M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari,
M. W. Achtelik, and R. Siegwart, “The EuRoC micro aerial vehicle
datasets,” The International Journal of Robotics Research, 2016.
[Online]. Available: http://ijr.sagepub.com/content/early/2016/01/21/
0278364915620033.abstract
[30] D. Schubert, T. Goll, N. Demmel, V. Usenko, J. Stuckler, and
D. Cremers, “The TUM VI benchmark for evaluating visual-inertial
odometry,” in 2018 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS).
IEEE, Oct. 2018. [Online]. Available:
http://dx.doi.org/10.1109/IROS.2018.8593419
[31] C. Chen, P. Geneva, Y. Peng, W. Lee, and G. Huang, “Monocular
visual-inertial odometry with planar regularities,” in Proc. of the
IEEE International Conference on Robotics and Automation, London,
UK, 2023. [Online]. Available: https://github.com/rpng/ov plane
[32] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A
benchmark for the evaluation of RGB-D SLAM systems,” in Proc.
of the International Conference on Intelligent Robot Systems (IROS),
Oct. 2012.
[33] S. Umeyama, “Least-squares estimation of transformation parameters
between two point patterns,” IEEE Transactions on Pattern Analysis
& Machine Intelligence, vol. 13, no. 04, pp. 376–380, 1991.
