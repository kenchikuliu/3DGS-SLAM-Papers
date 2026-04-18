<!-- page 1 -->
ToF-Splatting: Dense SLAM using Sparse Time-of-Flight Depth
and Multi-Frame Integration
Andrea Conti 1,2* Matteo Poggi 1 Valerio Cambareri 2
Martin R. Oswald 3 Stefano Mattoccia 1
1University of Bologna, Italy 2Sony DepthSensing Solutions, Belgium 3University of Amsterdam, Netherlands
Figure 1. Overview of our ToF-Splatting method. Our method combines sparse ToF depth, multi-view geometry from a buffer of
keyframes, and monocular cues (left) to perform into a unique end-to-end dense SLAM framework enabled by a Gaussian Splatting.
Abstract
Time-of-Flight (ToF) sensors provide efficient active
depth sensing at relatively low power budgets; among
such designs, only very sparse measurements from low-
resolution sensors are considered to meet the increasingly
limited power constraints of mobile and AR/VR devices.
However, such extreme sparsity levels limit the seamless
usage of ToF depth in SLAM. In this work, we propose
ToF-Splatting, the first 3D Gaussian Splatting-based SLAM
pipeline tailored for using effectively very sparse ToF input
data. Our approach improves upon the state of the art by
introducing a multi-frame integration module, which pro-
duces dense depth maps by merging cues from extremely
sparse ToF depth, monocular color, and multi-view geome-
try. Extensive experiments on both real and synthetic sparse
ToF datasets demonstrate the advantages of our approach,
as it achieves state-of-the-art tracking and mapping perfor-
mances on reference datasets.
1. Introduction
Simultaneous Localization and Mapping (SLAM) [6, 8, 36,
43] entails the joint estimation of the current camera pose
∗work started while visiting University of Amsterdam.
against a 3D scene representation (i.e., the map) and the up-
date of this representation with data from the current camera
view. It is a fundamental task with applications in AR/VR,
robotics, and autonomous navigation [14, 18]. In recent
years, SLAM witnessed a revolution [33] following the suc-
cess of Neural Radiance Fields (NeRF) [23] and 3D Gaus-
sian Splatting (3DGS) [13] as scene representations from
the adjacent field of novel view synthesis.
In this paper we focus on SLAM pipelines that leverage
RGB-D video inputs, i.e., where color images and dense
depth maps are collected by synchronized, accurate color
and depth sensors. Beyond global scale, depth sensors pro-
vide accurate local geometry that grants better tracking and
mapping quality. A common choice for active depth sensing
are Time-of-Flight (ToF) cameras. Despite the increased ac-
curacy provided by reliable depth measurements, the power
consumption and cost of such sensors constrain their adop-
tion to high-end industrial or automotive applications. A re-
cent trend to favor ToF sensor integration in handheld con-
sumer devices is to reduce the specifications of ToF sensors
to provide low-resolution but more reliable depth measure-
ments. Among other examples, LiDAR (an instance of ToF)
appears in consumer mobile phones [20] and is known to
feature up to at most 576 dots (interpolated to 256 × 192
resolution via depth completion). Other recent ToF sen-
sors provide only up to 64 depth points but consume a re-
1
arXiv:2504.16545v1  [cs.CV]  23 Apr 2025

<!-- page 2 -->
ported 200 mW [19]. In such settings a SLAM pipeline
can expect to receive, e.g., 8 × 8 very noisy, very sparse
depth measurements per frame. These specifications pre-
vent their straightforward integration in SLAM systems, as
low-resolution noisy depth is known to harm the accuracy
of dense SLAM systems built with NeRF and not originally
designed to deal with this scenario [30, 36, 46]. This is even
more evident with the more recent 3DGS-based SLAM sys-
tems [22], where accurate and dense depth is paramount to
achieve satisfactory results.
To overcome these limitations, a straightforward strat-
egy [38] would consist in recovering dense and accurate
depth maps using a single-image depth completion frame-
work [25]. However, models trained for this task are known
to generalize poorly across domains and across input depth
noise levels, potentially affecting the downstream SLAM
pipeline accuracy and yielding both drifted camera tra-
jectory and inaccurate mapping.
Therefore, we explore
3DGS as an explicit representation that allows for inject-
ing domain-specific knowledge and that is easily extended
to, e.g., dynamic scenes [21] and accurate large mesh recon-
struction [2, 9]. On the one hand, we argue that depth com-
pletion alone is insufficient to allow sparse ToF depth usage
in SLAM systems. On the other, we acknowledge that even
sparse and noisy depth measurements can bootstrap SLAM
if properly assisted with geometry. Accordingly, inspired
by frame-to-frame SLAM systems relying on pre-trained
trackers for localization, we believe that multi-frame inte-
gration fusing the noisy depth measurements with multi-
view geometry across a small set of local frames could sup-
ply the dense SLAM system with the supervision necessary
to fill this gap.
In this paper, we propose ToF-Splatting, a novel 3DGS-
based SLAM system enabling dense reconstruction from
sparse ToF sensors. ToF-Splatting harmoniously alternates
tracking and mapping, with each one benefiting from the
other. The former is carried out through backpropagation
during the optimization of the 3DGS model on color im-
ages; the latter exploits a pre-trained network [3] as a multi-
frame integration module, combining sparse depth mea-
surements with multi-view geometry according to a buffer
of keyframes. For each of the keyframes, we use the camera
pose estimated through tracking and providing dense depth
maps to improve the supervision given to the 3DGS model
itself. Figure 1 shows an overview of the setting in which
ToF-Splatting operates, as well as the quality of the 3D re-
constructions it yields.
ToF-Splatting is tested on the real-world ZJUL5
dataset [19], where we measure tracking and mapping qual-
ity and find the superiority of our framework with respect to
existing solutions built over sparse ToF sensors [38]. More-
over, standard benchmarks such as Replica [28] are also
provided and confirm the effectiveness of our mapping strat-
egy. Our main contributions are as follows:
• We propose ToF-Splatting, the first 3DGS-based SLAM
system suited for sparse ToF sensors.
• We introduce a novel multi-frame integration method
combining sparse depth,
multi-view geometry,
and
monocular cues for robust noise and outlier handling.
• We establish a new state-of-the-art for dense SLAM sys-
tems built on top of sparse ToF depth sensors.
2. Related Work
We briefly review the literature relevant to our work, refer-
ring to [33] for a detailed overview of the latest advances.
Traditional SLAM. Traditional visual SLAM pipelines, or
the more recent ones introducing deep modules [15, 31, 32],
usually perform tracking in frame-to-frame fashion, based
on the visual features extracted from past frames either us-
ing handcrafted feature extractors [24] or deep networks
[31, 32]. These systems are usually defined by four main
steps: tracking, mapping, global bundle adjustment, and
loop closure. Among the different representations used by
traditional SLAM frameworks, depth points [1, 24], surfels
[27], and volumetric representations [6] allows for achiev-
ing globally consistent 3D reconstruction.
NeRF-based SLAM. The advent of NeRF in the field of
novel view synthesis also reached adjacent research areas,
as well as SLAM a few years later. NeRF-based SLAM sys-
tems estimate camera poses and reconstruct dense meshes
by modeling scenes via MLPs.
iMAP [30] and NICE-
SLAM [45] are the first SLAM frameworks following this
approach, with the latter introducing dense feature grids
supporting the MLPs. Point-SLAM [26] and Loopy-SLAM
[17] switched feature grids with neural point embeddings,
allowing for higher flexibility and correcting or adjusting
local maps after global optimization, yet with slow pro-
cessing prohibiting deployment in robotics. ESLAM [11]
and Co-SLAM [34] deploy tri-planes and hash grid em-
beddings, improving both processing speed and reconstruc-
tion accuracy, with PLGSLAM [7] further improving the
representation capacity in large indoor scenes. GO-SLAM
[44] and KN-SLAM [37] implement frame-to-frame sys-
tems thanks to external trackers, yielding a good trade-off
between tracking/mapping accuracy and frame rate.
3DGS-based SLAM. More recent advances in novel view
synthesis brought to the development of new SLAM sys-
tems using 3D Gaussian Splatting [13] to represent the
mapped scene.
Several concurrent frameworks emerged
built over 3DGS, such as MonoGS [22], GaussianSLAM
[42], SplaTAM [12], and GS-SLAM [39]. Different from
NeRF-based SLAM systems, these frameworks allow for
higher interpretability and higher rendering speed.
Sparse ToF-based SLAM. Only lately, the use of more
compact and energy-efficient depth sensors has gained at-
tention in the SLAM research community. ToF-SLAM [38]
2

<!-- page 3 -->
Figure 2. ToF-Splatting Pipeline. Our method involves three main modules: a Tracking frontend estimating camera poses, a Multi-Frame
Integration module that predicts dense depth maps from sparse ToF measurements and multi-view geometry, and a Mapping backend
modeling the 3D scene representation via 3D Gaussian Splatting.
represents the only attempt to exploit “lightweight”, sparse
depth measurements within a dense SLAM system, aided
by a depth completion model [16]. However, depth comple-
tion alone fails to deal with noise in the sparse input depth,
a shortcoming we aim to specifically overcome within our
ToF-Splatting framework, the first one based on 3DGS and
suited for sparse ToF sensors.
3. Pipeline Architecture
In this section, we present the core components of our
framework. ToF-Splatting is a frame-to-model SLAM sys-
tem conceptually divided into three modules: (i) a mapping
backend, (ii) a tracking frontend, and (iii) a peculiar multi-
frame integration module yielding dense depth maps based
on a subset of the keyframe graph, the current RGB frame,
and the current sparse ToF depth map.
Figure 2 provides an overview of our pipeline.
Our
framework is characterized by three heterogeneous mod-
ules that are tightly related and influence each other. The
tracking module stands as the most influencing element
in the pipeline, as it selects the keyframes used by both
the mapping and multi-frame integration parts, moreover,
it computes the poses used for multi-frame depth estima-
tion. Nonetheless, it relies on the quality of the mapping
step to perform correct ego-motion estimation and exploits
the depth cues to properly behave when photometric infor-
mation is lacking or ambiguous.
The depth perception part is mainly sustained by the
tracking step and highly influences the mapping part since
it is used to seed new Gaussians and as supervision. Fi-
nally, the mapping phase not only creates a unique smooth
representation but also defines the rendered opacity that is
used for keyframe selection.
These connections lead to
a smoothly integrated framework that not only exploits a
multi-frame backbone but more importantly, defines an ef-
fective way to use such information in the SLAM scenario.
3.1. Background: 3D Gaussian Splatting
3D Gaussian Splatting [13] (3DGS) represents one of the
latest advances in novel view synthesis, fitting a dynami-
cally defined set of multivariate Gaussian distributions over
the scene to enable image rendering from arbitrary view-
points. Each Gaussian G(x) is parametrized by its mean
µ ∈R3 and covariance matrix Σ ∈R3×3 , reading
G(x) = exp

−1
2(x −µ)⊤Σ−1(x −µ)

(1)
Moreover, opacity o ∈R and RGB color c ∈R3 are as-
signed to each Gaussian.
Given a set of images from different viewpoints all
around the scene, the Gaussian parameters are optimized
using the splatting technique [35]. With such a method, 3D
Gaussians are projected through rasterization over the 2D
image plane, containing a 2D Gaussian distribution
µ2D = π(Twc ¯µ)
Σ2D = JRwcΣR⊤
wcJ⊤
(2)
where ¯µ is µ in homogeneous coordinates, Twc
∈
R4×4, Rwc ∈R3×3 are respectively the world-to-camera
transformation and its rotational component, π is the projec-
tion matrix and J ∈R2×3 is the Jacobian of the projective
transformation [49]. The color C of a pixel is then given by
the combination of the M weighted Gaussians, i.e.,
C =
X
i∈M
ciαi
i−1
Y
j=1
(1 −αj)
(3)
with αi = oiG(xi). Since Σ is positive semi-definite, it
is parametrized with a diagonal scaling matrix S and a ro-
tation matrix R, so that Σ = RSS⊤R⊤. R is internally
represented with quaternions.
3.2. Multi-Frame Integration
Dense depth cues are pivotal for proper initialization of the
Gaussians position µ ∈R3, as well as to provide super-
vision to both tracking and mapping threads – as already
3

<!-- page 4 -->
known in the literature [22, 47].
However, sparse ToF
sensors alone are insufficient for this purpose, because of
the very sparse and noisy depth measurements they pro-
vide (e.g., only 64 points for a sensor such as that used
in [19]). To overcome this limitation, ToF-SLAM [19] ex-
ploits an auxiliary completion model, DELTAR [16], to re-
cover dense depth maps out of ToF measurements. How-
ever, depth completion is highly dependent on the quality
of the sparse depth in input, rapidly degrading the accuracy
of the densified depth maps in the presence of noise.
In this paper, we follow a different path:
inspired
by frame-to-frame SLAM systems deploying an external
tracker, we believe that multi-frame integration can com-
pensate for the noise in the ToF measurements by exploit-
ing both single-view and multi-view geometry cues on a set
of local frames, given the camera poses estimated by the
tracking frontend. We choose to extend the Depth on De-
mand (DoD) framework [3] for this purpose.
Multi-Frame Integration.
DoD [3] integrates sparse
depth and two-view stereo cues for dense depth prediction.
Specifically – given a frame Fk – it processes the associated
RGB image Ik (the target view), another RGB view Ij from
a different frame Fj (a source view), the relative pose P jk
between the two, and the sparse depth points Hk to predict
a dense metric depth map Dk. Differently from monocu-
lar depth prediction, DoD is aware of the scene scale ei-
ther from Hk or P jk, simplifying the problem of integrat-
ing scale-inconsistent geometries in the SLAM frontend.
However, DoD is limited to two-view processing: this may
lead to suboptimal results in the SLAM scenario, where an
abundant amount of keyframes largely overlap and cover
the same portion of the scene. Thus, we extend [3] to multi-
view processing. This is achieved by observing that the pre-
dicted depth map Dk depends on the source view Ij through
its relative pose P jk only. DoD iteratively updates Dk for
multiple iterations, exploiting this intrinsic property we can
integrate multiple views (Ij, P jk) using a different one at
each iteration [4]. This approach allows smooth integration
of multi-view cues avoiding order dependency. To select the
source views Ij, at each prediction we order the keyframes
selected by the SLAM pipeline according to their relative
pose similarity and select the first N frames with baseline
distance close to b = 15cm to ensure enough parallax. The
model is trained from scratch on ScanNetv2 [5] and follow-
ing the protocol of [3].
Monocular Cues Integration. As shown in [3], DoD heav-
ily relies on both geometry and depth measurements, thus
struggling at generalizing in challenging scenarios where
the two lack. This may happen in cases where the tight
field of view, noisy sparse points, and textureless surfaces
hamper multi-view matching. To mitigate such limitations,
we integrate explicit monocular cues into DoD by feeding
its monocular encoder with Ik and the normalized depth
map ˜Dk ∈[0, 1]H×W obtained through the single-image
depth estimator Depth Anything v2 [40]. This approach al-
lows the injection of a robust bias, avoiding issues related to
monocular estimation, e.g., slanted surfaces or noisy scale
estimates, and delegating to DoD their handling. These cues
are included while retraining on ScanNetv2 [5].
Outlier Handling. Furthermore, the noisy depth measure-
ments in input to DoD can severely affect its accuracy.
Therefore, we explicitly focus on dealing with outliers in
the input sparse depth, that may be perceived by ToF de-
vices over specific surfaces. Such errors are particularly ev-
ident in cheap sensors and may largely prejudice the infor-
mation supplied by the few points they measure. Purposely,
we deploy a simple yet effective method to identify such
outliers by exploiting multi-view geometry again. When-
ever depth needs to be predicted for a new frame, DoD pre-
dicts a depth map ˆDk without processing the ToF measure-
ments. Then, the ℓ1 error between the latter and the depth
measured by the ToF sensor Hk is computed and finally
sparse measures with error higher than a given quantile q
are discarded, and depth is predicted again by DoD by also
processing the filtered sparse depth ˜Hk this time. This way,
we exploit multi-view cues to extract a clean depth prior to
filtering out inconsistent sparse depth measures, and then
we integrate the remaining points to improve the original
prediction. Even without processing the sparse depth, DoD
still predicts metric depth, as the metric scale is enforced by
the relative camera poses.
3.3. Tracking Frontend
The tracking part of ToF-Splatting estimates the ego-motion
of the camera for each new frame Fk and builds a keyframe
graph G = {KFk} ⊂{Fk} containing a set of meaningful
frames for mapping and multi-frame depth estimation.
Pose Estimation. We initialize each new frame Fk of our
frame-to-model tracker with the pose of the previous frame
Fk−1 and then minimizing the tracking loss with respect
to the relative pose between Fk and Fk−1, following [22].
The tracking loss consists of two terms, Lrgb and Ldepth, re-
spectively accounting for photometric and geometric errors
Ltrack = λtrackLrgb + (1 −λtrack)Ldepth
(4)
Lrgb = ∥Ik −ˆI(Fk)∥1
Ldepth = ∥Dk −ˆD(Fk)∥1 (5)
where Ik is the RGB image from Fk, ˆI(·) and ˆD(·) denote
the rendered RGB and Depth and, Dk is the depth predic-
tion produced by the multi-frame integration module.
Unlike other methods, the tracking process happens in
two stages. Initially, for a fixed number of steps ηrgb only
the photometric loss LT
rgb is optimized. Then, such a pose
is used for multi-frame integration, as described in Sec. 3.2,
to generate Dk. Finally, both losses in Eq. (5) are used for
ηT
rgbd further steps.
4

<!-- page 5 -->
Keyframe Selection Policy. We deploy a simple yet effec-
tive keyframe selection policy that considers i) novel con-
tent to be mapped and ii) instabilities in the already mapped
areas. Unlike other methods using complex approaches –
e.g., frustum intersection, point clouds and pose analysis
[22, 41] – we observe that the rendered opacity for a novel
frame contains low opacity values where either Gaussians
are not present or unstable. The first case highlights the
presence of novel areas of the environment to be mapped,
the second happens due to the pruning procedure involved
in the mapping step (see Sec. 3.4). We set a frame Fk as
a keyframe KFk if its novelty factor νk is higher than a
threshold νth, where we define νk as
νk =
P
i∈KFk M(KFk
i )
P
i∈KFk
 1 −M(KFk
i )

(6)
M(KFk
i ) =
(
1
if
ˆO(KFk
i ) < σ
0
otherwise
(7)
with ˆO(KFk
i ) being the rendered opacity at pixel i, and
M(KFk
i ) a binary uncertainty map.
This approach allows for skipping several frames when
the camera moves slowly or focusing deeply on areas where
the ToF-Splatting struggles the most, by mapping it consec-
utively. Nonetheless, we enforce mapping after skipping a
certain number of frames to avoid mapping too few frames.
3.4. Mapping Backend
Whenever a new keyframe KFk is identified, mapping is
triggered. This phase seeks to embed the new frame in the
global 3DGS model, involving the following two steps.
Initialization. In this step, new Gaussians are seeded in the
3DGS model. To limit the size of the model and reduce out-
liers, only areas where the rendered uncertainty M(KFk)
is high are seeded with new points, with the new Gaus-
sians also being randomly downsampled by a constant fac-
tor. Each Gaussian is initialized using depth Dk predicted in
the multi-frame integration step, with color from the RGB
frame Ik, and scale initialized with a constant value. Unlike
other 3DGS pipelines [22, 41], we prove this approach ef-
fective and particularly efficient, replacing any complex 3D
heuristics to a simple use of rendered opacity O(·), already
a side-product of the rasterization process.
Optimization. Firstly, ToF-Splatting smoothly integrates
the new information in the global model and secondly tunes
the existing representation. This is achieved by optimizing
the new keyframe KFk and a random subset of 5 keyframes
among the last N keyframes {KFj | j ≥max(k −N, 0)}.
These are optimized for a fixed number of steps ηM
rgbd. More-
over, we enforce a sampling rate of 60% for KFk, with the
remaining 40% for the remaining KFj to avoid forgetting.
During this step, gradients are also back-propagated to the
camera poses of the sampled keyframes, thus acting as a
global bundle adjustment. Finally, we prune Gaussians with
opacities lower than 0.5 every ηM
rgbd/2 steps. In addition to
the terms in Eq. (5), we optimize structural similarity as in
Eq. (8) and, following [2], the normals as in Eq. (9), i.e.,
Ldssim = 1 −SSIM
 Ik, ˆI(KFk)

2
(8)
Lnormals = ∥N k −ˆN∥1 + 1 −⟨N k, ˆN⟩
(9)
with ˆN := ¯∇ˆD(KFk), N k := ¯∇Dk denoting the normals
estimated via 3D gradients ¯∇for the rendered and multi-
frame predicted depth respectively, ⟨·, ·⟩their inner product,
and · the average. Lnormals is masked on edges to preserve
sharpness, as identified by running a Sobel filter.
Finally, we introduce a regularization loss Liso to pe-
nalize elongated Gaussians (i.e., promote their isotropy)
that usually leads to rendering artifacts as already observed
in [22]. Eq. (10) defines Liso, where diag(·) extracts the
diagonal values from its argument, and diag(Sj) is the av-
erage of the resulting vector, i.e., of the elements of the j-th
diagonal scale matrix Sj as defined in Sec. 3.1. Thus,
Liso = 1
|G|
|G|
X
j
∥diag(Sj) −diag(Sj) · 13×1∥1
(10)
The final mapping loss aggregates the aforementioned loss
terms with corresponding weights as follows:
Lmap =λmap
 λvisualLrgb + (1 −λvisual)Ldssim

(11)
+ (1 −λmap)Ldepth + λnormalsLnormals + λisoLiso
4. Experiments
This section assesses both the tracking and mapping perfor-
mance of ToF-Splatting on the following datasets.
ZJUL5. ZJUL5 [19] is the only existing public-domain real
dataset providing sparse depth data from a low-resolution,
low-cost ToF sensor. A VL53L5CX ToF sensor is assem-
bled on a calibrated rig with an Intel RealSense 435i to pro-
vide dense depth ground truth. This benchmark comprises
7 diverse indoor scene recordings. Ground truth 3D meshes
and camera poses are obtained following the ScanNet pro-
tocol [5]. This sensor provides 8 × 8 depth points by count-
ing the number of photons returned in each discretized time
range. Of the at most 64 depth points provided, many are
extremely noisy and introduce large outliers. This challeng-
ing real-world dataset provides real measurement noise that
is not accurately represented in mainstream datasets.
TUM RGB-D. This is a real dataset [29] providing indoor
sequence captures and widely used as benchmark for RGB-
D SLAM pipelines. The depth stream provides dense VGA
depth maps from a high-resolution Kinect v1 depth sensor.
5

<!-- page 6 -->
Method
Kitchen
Sofa
Office
Reception
Living Room
Office2
Sofa2
Avg.
KinectFusion [10]
✗
0.146
0.209
0.157
✗
0.321
0.125
0.192
iMAP [30]
✗
1.658
0.338
0.648
0.679
0.344
0.214
0.647
NICE-SLAM [46]
0.745
0.144
0.155
0.251
0.289
0.228
0.421
0.319
BundleFusion [6]
0.176
0.102
0.135
0.101
✗
0.163
0.120
0.132
ElasticFusion [36]
0.253
0.110
0.070
0.193
0.530
0.121
0.146
0.203
MonoGS [22]
0.231
0.032
0.041
0.044
0.153
0.035
0.073
0.087
ToF-SLAM [19]
0.113
0.081
0.056
0.114
0.200
0.101
0.085
0.107
ToF-Splatting (ours)
0.088
0.024
0.022
0.041
0.122
0.029
0.030
0.051
Table 1. ZJUL5 Tracking Results. We show here the tracking performance on the ZJUL5 dataset [19] dataset with the ATE RMSE [m]
(↓) metric on the 8 sequences available where ✗indicates lost tracking. ToF-Splatting provides the better performance by a good margin
on each sequence, demonstrating the optimal capability of our proposal to track the camera ego-motion accurately.
Replica. Replica [28] is a synthetic dataset providing ex-
tremely realistic indoor sequences. It provides dense depth
maps, which we use at different densities to perform exper-
iments that prove the robustness and generalization capabil-
ities of ToF-Splatting in challenging scenarios.
Baselines. We compare our method with both learning-
based [19, 30, 46] and traditional pipelines [6, 10, 36].
Following ToF-SLAM [19], baselines requiring RGB-D
frames use DELTAR [16] to densify the sparse ToF mea-
surements. Moreover, we adapt MonoGS [22] densifying
depth with the state-of-the-art depth completion framework
OGNI-DC [48] for a Gaussian Splatting SLAM baseline.
Implementation Details. In all our experiments, we use
N = 4 source views Ij (i.e., keyframe buffer elements) for
the multi-view integration module and a baseline b = 15cm
for selecting keyframes. We use a quantile q = 0.75 to filter
outliers that occur frequently in the sparse ToF depth of the
ZJUL5 dataset [19]. Concerning tracking, we use a novelty
factor threshold of νth = 0.1, σ = 0.98. The total track-
ing loss uses λtrack = 0.9, a total number of ηT
rgb
=
30
and ηT
rgbd = 70 steps are performed at each iteration. For
mapping, we set the loss weights λmap = 60, λvisual =
0.20, λnormals = 0.01, λiso = 1.0 and the number of steps
per iteration ηM
rgbd = 60. We test on a single RTX3090.
4.1. Comparison with the State of the Art
In this section, we compare ToF-Splatting with existing
methods in terms of tracking and mapping performances.
Tracking. In Table 1, we report the tracking performance
by ToF-Splatting and various baselines on each sequence
of the ZJUL5 dataset [19].
Following common practice
we use the Absolute Trajectory Error metric [29], which
directly compares the global consistency of a trajectory.
While various frameworks [6, 10, 30] fail on the most chal-
lenging scenes, ToF-Splatting delivers accurate and consis-
tent tracking. This is evident from the good margin it gains
against the baselines. In Figure 3, we show the 3D trajecto-
ries on a subset of the scenes, with error below 3 cm.
Mapping. In Table 2, we report the mapping performance
by ToF-Splatting compared with the other baselines. For
each scene, we collect the predicted pose of each frame
Office
Reception
Sofa
Figure 3. Qualitative results on the ZJUL5 dataset [19]. We
show meshes obtained by fusing rendered depth maps with TSDF
and marching cubes (left), and 3D trajectories (right) on 3 scenes
selected from the ZJUL5 dataset [19].
and render depth and color from 3DGS. Then, we perform
TSDF integration and extract the final mesh through march-
ing cubes. Following [19], the meshes are then evaluated
by computing Accuracy, Completion, and F-score versus
the ground truth meshes. Accuracy and Completion respec-
tively evaluate the mean distance between each predicted
vertex and the nearest ground truth one and vice versa. The
F-score takes into account both Accuracy and Completion
with an aggregate metric. ToF-Splatting delivers consis-
tent results for mapping, always achieving the highest ac-
curacy and significantly exceeding the baselines on average
and specifically on the Sofa, Office, and Reception scenes
which account for five over seven sequences. [19] is the
second-best method and the main competitor in mapping.
6

<!-- page 7 -->
Method
Metric
Kitchen
Sofa
Office
Reception
Living Room
Office2
Sofa2
Avg.
KinectFusion [10]
Accuracy ↓
✗
0.190
0.211
0.261
✗
0.267
0.135
-
Completion ↓
✗
0.048
0.046
0.064
✗
0.078
0.064
-
F-score ↑
✗
0.278
0.288
0.285
✗
0.274
0.381
-
ElasticFusion [36]
Accuracy ↓
0.092
0.135
0.084
0.297
0.151
0.096
0.122
0.140
Completion ↓
0.065
0.048
0.082
0.305
0.216
0.147
0.047
0.130
F-score ↑
0.553
0.420
0.529
0.274
0.382
0.416
0.481
0.436
BundleFusion [6]
Accuracy ↓
0.170
0.100
0.103
0.122
✗
0.121
0.123
-
Completion ↓
0.088
0.030
0.038
0.057
✗
0.214
0.034
-
F-score ↑
0.373
0.571
0.474
0.470
✗
0.442
0.527
-
iMAP [30]
Accuracy ↓
✗
0.135
0.229
0.365
0.225
0.233
0.139
-
Completion ↓
✗
0.054
0.103
0.245
0.291
0.139
0.069
-
F-score ↑
✗
0.445
0.315
0.238
0.170
0.255
0.416
-
NICE-SLAM [46]
Accuracy ↓
0.303
0.119
0.116
0.216
0.103
0.156
0.464
0.211
Completion ↓
0.456
0.042
0.070
0.199
0.089
0.163
0.045
0.152
F-score ↑
0.221
0.554
0.411
0.402
0.400
0.273
0.401
0.380
MonoGS [22]
Accuracy ↓
0.090
0.086
0.083
0.071
0.069
0.081
0.089
0.081
Completion ↓
0.246
0.122
0.104
0.238
0.193
0.157
0.170
0.175
F-score ↑
0.290
0.299
0.346
0.367
0.259
0.374
0.289
0.318
ToF-SLAM [19]
Accuracy ↓
0.081
0.068
0.067
0.079
0.078
0.113
0.121
0.087
Completion ↓
0.071
0.041
0.045
0.062
0.122
0.085
0.033
0.066
F-score ↑
0.559
0.661
0.646
0.643
0.496
0.557
0.656
0.604
ToF-Splatting (ours)
Accuracy ↓
0.064
0.031
0.032
0.041
0.059
0.034
0.043
0.043
Completion ↓
0.095
0.038
0.029
0.056
0.131
0.054
0.046
0.064
F-score ↑
0.527
0.791
0.840
0.710
0.359
0.779
0.642
0.664
Table 2. ZJUL5 Mapping Results. We perform the mapping evaluation on the ZJUL5 dataset (7 indoor sequences) and report results of
three metrics including accuracy (Acc.), completion (Comp.), and F-score. The failure cases are marked as ✗.
Monocular Cues
Multi-View Cues
Metric
Kitchen
Sofa
Office
Reception
Living Room
Office2
Sofa2
Avg.
✓
✗
ATE ↓
0.103
0.054
0.046
0.050
0.110
0.041
0.055
0.066
F-score ↑
0.216
0.675
0.564
0.645
0.241
0.611
0.557
0.501
✗
✓
ATE ↓
0.234
0.024
0.025
0.037
0.109
0.034
0.028
0.072
F-score ↑
0.284
0.781
0.826
0.703
0.253
0.766
0.683
0.614
✓
✓
ATE ↓
0.088
0.024
0.022
0.041
0.122
0.029
0.030
0.051
F-score ↑
0.527
0.791
0.840
0.710
0.359
0.779
0.642
0.664
Table 3. Multi-Frame Integration Ablation Study. Analysis of the impact of monocular and multi-view cues in the multi-frame integra-
tion step. Monocular cues are less impactful than multi-view ones, boosting performance in the challenging scenes where multi-view cues
struggle most. Nonetheless, their combination yields the best results.
Method
LC Input
Density fr1/desk fr2/xyz fr3/office
Avg.
DROID-VO
✗
RGB
0.00%
0.052
0.107
0.073
0.077
MonoGS
✗
RGB
0.00%
0.038
0.046
0.035
0.040
ToF-Splatting ✗
RGB-D 0.02%
0.030
0.022
0.030
0.027
0.04%
0.027
0.019
0.021
0.022
MonoGS
✗
RGB-D 100%
0.015
0.014
0.015
0.015
ORB-SLAM2 ✓
RGB
0.00%
0.019
0.006
0.024
0.016
ORB-SLAM2 ✓
RGB-D 100%
0.016
0.040
0.010
0.010
Table 4. Results on TUM RGB-D dataset. Top: competitors w/o
loop-closure (LC); middle: most representative competitor with
100% density; bottom: competitors w/ loop-closure.
Notably, the other methods – like [6, 36] – may provide
good performance on specific scenes but demonstrate unre-
liability on average usually failing in the particularly chal-
lenging Kitchen and Living Room scenes. In Figure 3, we
show Office, Reception, and Sofa meshes on the left.
4.2. Ablation Studies
In this section, we perform additional experiments aimed at
studying the impact of different factors on ToF-Splatting.
Multi-Frame Integration. In Table 3, we ablate the main
components of the multi-frame integration and assess their
contribution to the whole SLAM framework. By retaining
monocular cues alone at the expense of multi-view geome-
try, we observe severe drops in both tracking and mapping
accuracy, confirming the paramount importance of the lat-
ter; monocular cues alone are effective only where multi-
view cues are ineffective – e.g., in the absence of texture, as
occurs mostly in the Kitchen scene.
TUM RGB-D. In Table 4, we simulate sparse ToF data by
sampling 0.02% and 0.04% depth points, and compare ToF-
Splatting with existing RGB and RGB-D methods. With
as few as 0.02% of points, ToF-Splatting largely improves
over RGB methods [22, 32], approaching the 100% density
RGB-D approach of the most representative competitor [22]
at only 0.04% density.
Replica: Input Depth Sparsity. We conduct additional
studies on the Replica dataset, for which Figure 4 provides
a qualitative overview of our 3D reconstructions.
ToF-
7

<!-- page 8 -->
Figure 4. Replica Qualitatives. We provide qualitative results on Replica [28] to demonstrate the generalization capabilities of our method.
On the left from top to bottom, meshes obtained respectively on scenes Office2 and Room2. On the right, details from the scene Room0.
ToF-Splatting delivers accurate details and allows for nice photometric and depth rendering.
0.03
0.04
0.05
0.02%
0.04%
0.08%
0.16%
(a) MAE [m]
27.0
27.5
28.0
0.02%
0.04%
0.08%
0.16%
(b) PSNR [dB]
0.02
0.03
0.04
0.02%
0.04%
0.08%
0.16%
(c) ATE [m]
Figure 5. Impact of depth sparsity. We test on Replica [28] with
different simulated depth sparsity levels to assess the capability to
exploit higher input densities. MAE and ATE smoothly decrease,
whereas rendering metrics appear to be less affected.
0.06
0.08
0.10
0.12
0.14
0%
0.01%
0.05%
0.10%
(a) MAE [m]
24.0
26.0
0%
0.01%
0.05%
0.10%
(b) PSNR [dB]
0.05
0.10
0.15
0%
0.01%
0.05%
0.10%
(c) ATE [m]
Figure 6. Impact of depth noise. We test on Replica [28] injecting
different amounts of noise ξ
∈
[0.00, 0.01, 0.05, 0.10]. ToF-
Splatting demonstrates to be effective at dealing with noise, with
error increasing almost linearly with the injected amount of noise.
Splatting enables performing SLAM with very few sparse
points – the 64 sparse points provided by a 8 × 8 sparse
account only for 0.02% of a 640 × 480 image. We here
explore the impact of higher sparse points density on its
overall performance.
We simulate on Replica densities
of {0.02%, 0.04%, 0.08%, 0.16%} corresponding to about
{64, 128, 256, 512} points. Figure 5 shows the performance
variation on both tracking and mapping metrics using the
aforementioned sparsities.
Replica: Input Depth Noise. We study various noise lev-
els impact in the input depth to assess our pipeline robust-
ness. We model noise as additive but dependent on depth,
injecting heteroscedastic Gaussian noise N(d, ϵd) where ϵ
modulates the variance of the noise source on Replica. Fig-
ure 6 shows that MAE, PSNR, and ATE all expose linear
degradation trends as ϵ increases.
Runtime.
In Table 5 we measure the runtime of ToF-
Method Name
Tracking Time
Mapping Time
iMAP [30]
101 ms
448 ms
NICE-SLAM [46]
470 ms
1300 ms
MonoGS [22]
1169 ms
650 ms
ToF-SLAM [19]
116 ms
380 ms
ToF-Splatting (ours)
614 ms
865 ms
Table 5. Runtime Comparison. We compare the runtime of ToF-
Splatting and other learned baselines measuring their runtime for
tracking and mapping. MonoGS [22] tracking time contains also
the depth completion inference time [48].
Splatting in comparison with the main baselines. The track-
ing time of ToF-Splatting already includes depth estimates
with DoD, with average runtime ¯tDoD ≈90 ms (compris-
ing DepthAnything v2 for monocular cues inference [40]).
To date, 3DGS-based mapping methods are not capable
of real-time performances.
Nevertheless, exploiting the
fact that the mapping step accounts only for just a few
frames (typically 4% of the total in sequences of the ZJUL5
dataset [19]) and can be parallelized, the pipeline achieves
1.5 fps without further optimization.
Limitations.
Even with state-of-the-art quality, ToF-
Splatting manifests some issues that will be addressed in
future developments. The most concerning is the runtime,
which is too slow for real-time applications and requires
memory and compute-intensive backpropagation at deploy-
ment.
Even though fast 3DGS frameworks exist, such
methodologies have not been transferred yet into the SLAM
community, that being outside the scope of this work.
5. Conclusion
We presented the first 3DGS-based SLAM pipeline relying
on sparse ToF depth sensing, as it provides accurate tracking
and mapping from cheap and low-power cameras. More-
over, we showed how this result can be achieved through
the integration of multi-view geometry, sparse depth data,
and monocular cues, yielding an end-to-end 3DGS-based
SLAM system. Finally, we assessed the effectiveness of
our approach on the ZJUL5 and Replica datasets.
8

<!-- page 9 -->
References
[1] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez,
Jos´e MM Montiel, and Juan D Tard´os. Orb-slam3: An accu-
rate open-source library for visual, visual–inertial, and mul-
timap slam. IEEE Transactions on Robotics, 37(6):1874–
1890, 2021. 2
[2] Hanlin Chen, Fangyin Wei, Chen Li, Tianxin Huang, Yun-
song Wang, and Gim Hee Lee. Vcr-gaus: View consistent
depth-normal regularizer for gaussian surface reconstruction.
arXiv preprint arXiv:2406.05774, 2024. 2, 5
[3] Andrea Conti, Matteo Poggi, Valerio Cambareri, and Stefano
Mattoccia. Depth on demand: Streaming dense depth from
a low frame-rate active sensor. In European Conference on
Computer Vision (ECCV), 2024. 2, 4, 1
[4] Andrea Conti, Matteo Poggi, Valerio Cambareri, and S. Mat-
toccia.
Range-agnostic multi-view depth estimation with
keyframe selection. 2024 International Conference on 3D
Vision (3DV), pages 1350–1359, 2024. 4
[5] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. ScanNet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 5828–5839, 2017. 4, 5,
2
[6] Angela Dai, Matthias Nießner, Michael Zollh¨ofer, Shahram
Izadi, and Christian Theobalt.
BundleFusion: Real-time
globally consistent 3d reconstruction using on-the-fly surface
reintegration. ACM Transactions on Graphics (ToG), 36(4):
1, 2017. 1, 2, 6, 7
[7] Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wen-
tao Zhao, Jingchuan Wang, Danwei Wang, and Weidong
Chen.
Plgslam:
Progressive neural scene represenation
with local to global bundle adjustment. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19657–19666, 2024. 2
[8] Zilong Dong, Guofeng Zhang, Jiaya Jia, and Hujun Bao.
Keyframe-based real-time camera tracking. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 1538–1545. IEEE, 2009. 1
[9] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering. CVPR, 2024. 2
[10] Shahram
Izadi,
David
Kim,
Otmar
Hilliges,
David
Molyneaux, Richard Newcombe, Pushmeet Kohli, Jamie
Shotton, Steve Hodges, Dustin Freeman, Andrew Davison,
et al. KinectFusion: real-time 3d reconstruction and interac-
tion using a moving depth camera. In Proceedings of the
24th annual ACM symposium on User Interface Software
and Technology, pages 559–568, 2011. 6, 7
[11] Mohammad Mahdi Johari, Camilla Carta, and Franc¸ois
Fleuret. Eslam: Efficient dense slam system based on hybrid
representation of signed distance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17408–17419, 2023. 2
[12] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 2
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(TOG), 2023. 1, 2, 3
[14] Georg Klein and David Murray. Parallel tracking and map-
ping for small AR workspaces.
In Proceedings of the
IEEE and ACM International Symposium on Mixed and Aug-
mented Reality, pages 225–234. IEEE, 2007. 1
[15] Yanyan Li, Nikolas Brasch, Yida Wang, Nassir Navab, and
Federico Tombari.
Structure-slam: Low-drift monocular
slam in indoor environments. IEEE Robotics and Automa-
tion Letters, 5(4):6583–6590, 2020. 2
[16] Yijin Li, Xinyang Liu, Wenqian Dong, Han Zhou, Hujun
Bao, Guofeng Zhang, Yinda Zhang, and Zhaopeng Cui.
Deltar: Depth estimation from a light-weight tof sensor and
rgb image. In European Conference on Computer Vision,
2022. 3, 4, 6
[17] Lorenzo Liso,
Erik Sandstr¨om,
Vladimir Yugay,
Luc
Van Gool, and Martin R Oswald. Loopy-slam: Dense neural
slam with loop closures. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 20363–20373, 2024. 2
[18] Haomin Liu, Guofeng Zhang, and Hujun Bao.
Robust
keyframe-based monocular SLAM for augmented reality. In
Proceedings of the IEEE International Symposium on Mixed
and Augmented Reality, pages 1–10. IEEE, 2016. 1
[19] Xinyang Liu, Yijin Li, Yanbin Teng, Hujun Bao, Guofeng
Zhang, Yinda Zhang, and Zhaopeng Cui. Multi-modal neural
radiance field for monocular dense slam with a light-weight
tof sensor.
2023 IEEE/CVF International Conference on
Computer Vision (ICCV), pages 1–11, 2023. 2, 4, 5, 6, 7,
8, 1
[20] Gregor Luetzenburg, Aart Kroon, and Anders A. Bjørk.
Evaluation of the Apple iPhone 12 Pro LiDAR for an Ap-
plication in Geosciences. Scientific Reports, 11(1), 2021. 1
[21] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by persis-
tent dynamic view synthesis. 2024 International Conference
on 3D Vision (3DV), pages 800–809, 2023. 2
[22] Hidenobu Matsuki, Riku Murai, Paul H.J. Kelly, and An-
drew J. Davison. Gaussian splatting slam. 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 18039–18048, 2023. 2, 4, 5, 6, 7, 8
[23] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing scenes as neural radiance fields for view syn-
thesis. In Proceedings of the European Conference on Com-
puter Vision, pages 405–421. Springer, 2020. 1
[24] Ra´ul Mur-Artal and Juan D. Tard´os.
ORB-SLAM2: An
Open-Source SLAM System for Monocular, Stereo, and
RGB-D Cameras. IEEE Transactions on Robotics, 33(5):
1255–1262, 2017. 2
[25] Xin Qiao,
Matteo Poggi,
Pengchao Deng,
Hao Wei,
Chenyang Ge, and Stefano Mattoccia. Rgb guided tof imag-
9

<!-- page 10 -->
ing system: A survey of deep learning-based methods. In-
ternational Journal of Computer Vision, pages 1–38, 2024.
2
[26] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 18433–18444, 2023. 2
[27] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Bad
slam: Bundle adjusted direct rgb-d slam. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 134–144, 2019. 2
[28] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal,
Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan,
Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang
Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler
Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva,
Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael
Goesele, Steven Lovegrove, and Richard Newcombe. The
Replica dataset: A digital replica of indoor spaces. arXiv
preprint arXiv:1906.05797, 2019. 2, 6, 8, 1
[29] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the eval-
uation of RGB-D SLAM systems.
In Proceedings of the
IEEE/RSJ International Conference on Intelligent Robots
and Systems, pages 573–580. IEEE, 2012. 5, 6
[30] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davi-
son. iMAP: Implicit mapping and positioning in real-time.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 6229–6238, 2021. 2, 6, 7, 8
[31] Keisuke Tateno, Federico Tombari, Iro Laina, and Nassir
Navab.
Cnn-slam: Real-time dense monocular slam with
learned depth prediction. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
6243–6252, 2017. 2
[32] Zachary Teed and Jia Deng.
Droid-SLAM: Deep vi-
sual SLAM for monocular, stereo, and RGB-D cameras.
Advances in Neural Information Processing Systems, 34:
16558–16569, 2021. 2, 7
[33] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstr¨om,
Stefano Mattoccia, Martin R. Oswald, and Matteo Poggi.
How nerfs and 3d gaussian splatting are reshaping slam: a
survey, 2024. 1, 2
[34] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-
slam: Joint coordinate and sparse parametric encodings for
neural real-time slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
13293–13302, 2023. 2
[35] Yifan Wang, Felice Serena, Shihao Wu, Cengiz ¨Oztireli,
and Olga Sorkine-Hornung. Differentiable surface splatting
for point-based geometry processing. ACM Transactions on
Graphics (TOG), 38:1 – 14, 2019. 3
[36] Thomas Whelan, Renato F Salas-Moreno, Ben Glocker, An-
drew J Davison, and Stefan Leutenegger.
ElasticFusion:
Real-time dense SLAM and light source estimation.
The
International Journal of Robotics Research, 35(14):1697–
1716, 2016. 1, 2, 6, 7
[37] Xingming Wu, Zimeng Liu, Yuxin Tian, Zhong Liu, and
Weihai Chen. Kn-slam: Keypoints and neural implicit en-
coding slam.
IEEE Transactions on Instrumentation and
Measurement, 73:1–12, 2024. 2
[38] Liu Xinyang, Li Yijin, Teng Yanbin, Bao Hujun, Zhang
Guofeng, Zhang Yinda, and Cui Zhaopeng.
Multi-modal
neural radiance field for monocular dense slam with a light-
weight tof sensor. In International Conference on Computer
Vision (ICCV), 2023. 2
[39] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 2
[40] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. ArXiv, abs/2406.09414, 2024. 4, 8
[41] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting. ArXiv, abs/2312.10070, 2023. 5
[42] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting, 2023. 2
[43] Guofeng Zhang, Jiaya Jia, Tien-Tsin Wong, and Hujun Bao.
Recovering consistent video depth maps via bundle opti-
mization. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 1–8. IEEE,
2008. 1
[44] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d in-
stant reconstruction. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 3727–3737,
2023. 2
[45] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 12786–12796, 2022. 2
[46] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. NICE-SLAM: Neural implicit scalable encoding for
SLAM.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 12786–
12796, 2022. 2, 6, 7, 8
[47] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R. Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural implicit scene encoding for rgb slam.
2024 International Conference on 3D Vision (3DV), pages
42–52, 2023. 4
[48] Yiming Zuo and Jia Deng.
Ogni-dc: Robust depth com-
pletion with optimization-guided neural iterations.
arXiv
preprint arXiv:2406.11711, 2024. 6, 8
[49] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and
Markus H. Gross.
Surface splatting.
Proceedings of the
28th annual conference on Computer graphics and interac-
tive techniques, 2001. 3
10

<!-- page 11 -->
ToF-Splatting: Dense SLAM using Sparse Time-of-Flight Depth
and Multi-Frame Integration
Supplementary Material
We provide this manuscript as a supplementary resource
to the CVPR submission #6364 titled “ToF-Splatting:
Dense SLAM using Sparse Time-of-Flight Depth and
Multi-Frame Integration” to provide a deeper understand-
ing of the proposed framework through extended insights,
detailed explanations, and additional qualitative results that
complement the findings presented in the main paper. By
including these extended materials, we hope to facilitate a
more comprehensive appreciation of the contributions and
practical relevance of the proposed approach.
6. TUM RGB-D DoD Qualitative Results
In Figure 7 we present three pairs of color views and re-
constructed depth maps to assess that our method [3], as
integrated in the ToF-splatting pipeline, generalizes well to
unseen datasets such as TUM RGB-D at test time. Indeed,
we obtain high-quality depth maps from sparse inputs.
7. Replica Qualitative Results
In Figure 8, we present the reconstructed mesh and the pre-
dicted trajectory for each scene in the Replica [28] dataset.
To achieve this, we first fit the entire scene and render depth
and color images for each pose estimated by ToF-Splatting.
These rendered outputs are then fused using Truncated
Signed Distance Function (TSDF) integration.
Once the
integration is complete, we extract the 3D mesh using the
marching cubes algorithm as implemented in Open3D. The
reconstruction process employs a voxel size of 0.02 meters,
with depth values truncated at a maximum distance of 4 me-
ters to ensure robustness. On the right side of the figure,
we visualize the xy plane projections of the predicted and
ground truth trajectories, where the ground truth is repre-
sented by a dashed gray line. Additionally, a color bar indi-
cates the positional error between the corresponding frame
poses along the trajectories. The results demonstrate that
ToF-Splatting effectively achieves high-quality reconstruc-
tions and accurate tracking, highlighting its robustness and
precision in this context.
8. ZJUL5 Qualitative Results
Figure 9 illustrates the reconstructed meshes and predicted
trajectories for the scenes included in the ZJUL5 [19]
dataset.
Unlike the Replica dataset [28], which primar-
ily focuses on synthetic environments, the ZJUL5 dataset
presents real-world scenarios that introduce a wide range of
practical challenges. These include substantial noise and
Figure 7. Qualitative results on TUM RGB-D dataset. Depth
maps for fr1/office (left), fr2/xyz (center), fr3/desk (right) as esti-
mated by DoD [3].
a high density of outliers resulting from sparse Time-of-
Flight (ToF) data, as well as environmental difficulties such
as poor texture, suboptimal lighting conditions, and lim-
ited fields of view. Despite these hurdles, ToF-Splatting
demonstrates remarkable robustness and adaptability, con-
sistently outperforming competing approaches. It is capable
of producing high-quality mesh reconstructions and accu-
rately predicting trajectories even under such adverse con-
ditions. These results highlight the strength and versatility
of ToF-Splatting in real-world applications, where it effec-
tively addresses the challenges posed by noisy and sparse
data while still delivering meaningful and reliable outputs.
9. Temporal Sparsity
Finally, we study ToF-Splatting performance under tempo-
ral sparsity, which refers to scenarios where the ToF sensor
frame rate is lower than that of the RGB camera. Thus, only
a subset of the RGB frames is coupled with sparse depth in-
formation. Such a situation is particularly relevant in real
use case scenarios where the ToF sensor may operate at a
reduced frame rate either due to hardware or power con-
straints. To investigate this, we simulate this scenario on the
ZJUL5 dataset [19] by subsampling the ToF frames at ratios
of [2, 3, 4, 6, 8], as shown in Figure 10. ToF-Splatting main-
tains consistent performance despite the increasing tempo-
ral sparsification. The availability of multi-view informa-
tion enables robustness in the framework, compensating ef-
fectively for the lack of sparse depth for a subset of frames.
To ensure that the system has access to the scene scale, we
provide ToF depth for every frame in the first 50 frames to
provide a reliable starting point. The results highlight the
adaptability of ToF-Splatting, demonstrating its ability to
1

<!-- page 12 -->
Office 0
Office 1
Office 2
Office 3
Office 4
Room 0
Room 1
Room 2
Figure 8. Replica Qualitatives. We provide the trajectory and mesh reconstruction qualitative results on each scene provided by Replica.
ToF-Splatting enables effective mesh reconstructions and accurate tracking.
operate effectively even when the ToF sensor’s frame rate is
significantly lower than that of the RGB camera.
10. Depth on Demand Training Details
In ToF-Splatting,
we perform multi-frame integration
adapting the Depth on Demand framework [3] to our spe-
cific use case. Specifically, we significantly modify its in-
nermost logic to integrate monocular cues and handle a
larger number of frames to overcome the original two-frame
configuration. Moreover, we retrain the framework with
adjustments designed to optimize its performance in sce-
narios characterized by extreme input depth sparsity. The
architectural improvements made to the framework are de-
tailed in the main paper. To train the model, we utilized
the ScanNetV2 dataset [5], which provides a robust and di-
verse set of scenes. However, our training procedure di-
verges from the one described in the original paper [3]. In-
deed, at each iteration, we sample a set of source views be-
tween 0 and 5 and directly extract sparse depth measure-
ments from the target view instead of performing a repro-
2

<!-- page 13 -->
Reception
Kitchen
Office
Living Room
Sofa
Sofa 2
Figure 9. ZJUL5 Qualitatives. We provide the trajectory and mesh reconstruction qualitative results on the scenes provided by ZJUL5.
ToF-Splatting enables effective mesh reconstructions and accurate tracking.
jection from one of the previous source views. This adjust-
ment aligns better with our use case and eliminates reliance
on source-to-target projections for depth data. Addition-
ally, we introduced variability in the density of sparse depth
samples, randomly selecting a density within the range
[0%, . . . , 0.03%]. Such sparsification is important since it
effectively mimics the extremely sparse depth scenarios en-
countered in our application, ensuring that the model is
robust to real-world conditions of depth sparsity.
These
enhancements collectively enable ToF-Splatting to achieve
high-quality performance, even in environments where data
sparsity and noise are significant challenges.
3

<!-- page 14 -->
0.08
0.10
0.12
0.14
2
3
4
6
8
(a) MAE [m]
0.5
0.6
0.6
2
3
4
6
8
(b) F-score [%]
0.06
0.08
0.10
0.12
2
3
4
6
8
(c) ATE [m]
Figure 10. Impact of Temporal Sparsity. The three line plots represent respectively mean absolute error, F-score, and absolute trajectory
error as the subsampling ratio of the ToF frames increases. As temporal sparsity grows, a gradual decline in overall performance is ob-
served, reflecting the reduced availability of depth information. Despite this, ToF-Splatting demonstrates resilience, maintaining reasonable
performance by effectively leveraging multi-view cues to mitigate the challenges of sparse depth sampling.
4
