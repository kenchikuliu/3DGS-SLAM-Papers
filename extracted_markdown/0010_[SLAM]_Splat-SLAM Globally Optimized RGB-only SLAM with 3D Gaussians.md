<!-- page 1 -->
Splat-SLAM: Globally Optimized RGB-only SLAM
with 3D Gaussians
Erik Sandström∗
Google
ETH Zürich
Keisuke Tateno
Google
Michael Oechsle
Google
Michael Niemeyer
Google
Luc Van Gool
ETH Zürich
INSAIT
Martin R. Oswald
ETH Zürich
University of Amsterdam
Federico Tombari
Google
TU München
Abstract
3D Gaussian Splatting has emerged as a powerful representation of geometry and
appearance for RGB-only dense Simultaneous Localization and Mapping (SLAM),
as it provides a compact dense map representation while enabling efficient and
high-quality map rendering. However, existing methods show significantly worse
reconstruction quality than competing methods using other 3D representations, e.g.
neural points clouds, since they either do not employ global map and pose optimiza-
tion or make use of monocular depth. In response, we propose the first RGB-only
SLAM system with a dense 3D Gaussian map representation that utilizes all bene-
fits of globally optimized tracking by adapting dynamically to keyframe pose and
depth updates by actively deforming the 3D Gaussian map. Moreover, we find that
refining the depth updates in inaccurate areas with a monocular depth estimator
further improves the accuracy of the 3D reconstruction. Our experiments on the
Replica, TUM-RGBD, and ScanNet datasets indicate the effectiveness of globally
optimized 3D Gaussians, as the approach achieves superior or on par performance
with existing RGB-only SLAM methods methods in tracking, mapping and render-
ing accuracy while yielding small map sizes and fast runtimes. The source code is
available at https://github.com/eriksandstroem/Splat-SLAM.
1
Introduction
A common factor within the recent trend of dense SLAM is that the majority of works reconstruct a
dense map by optimizing a neural implicit encoding of the scene, either as weights of an MLP [1,
57, 39, 45], as features anchored in dense grids [82, 42, 66, 67, 58, 3, 29, 83, 51], using hierarchical
octrees [72], via voxel hashing [79, 78, 8, 49, 40], point clouds [18, 50, 30, 75] or axis-aligned feature
planes [33, 47]. We have also seen the introduction of 3D Gaussian Splatting (3DGS) to the dense
SLAM field [74, 24, 69, 38, 21].
Out of this 3D representation race there is, however, not yet a clear winner. In the context of dense
SLAM, a careful modeling choice needs to be made to achieve accurate surface reconstruction as
well as low tracking errors. Some takeaways can be deduced from the literature: neural implicit
point cloud representations achieve state-of-the-art reconstruction accuracy [30, 75, 50], especially
with RGBD input. At the same time, 3D Gaussian splatting methods yield the highest fidelity
renderings [38, 74, 24, 21, 69] and show promise in the RGB-only setting due to their flexibility in
optimizing the surface location [21, 38]. However, they are not leveraging any multi-view depth or
geometric prior leading to poor geometry in the RGB-only setting. The majority of the aforementioned
∗This work was conducted during an internship at Google.
arXiv:2405.16544v1  [cs.CV]  26 May 2024

<!-- page 2 -->
Ground Truth
GlORIE-SLAM [75]
MonoGS [38]
Splat-SLAM (Ours)
Depth L1 [cm]↓PSNR↑
22.19
18.78
116.71
18.41
15.05
24.06
ATE RMSE [cm]↓
4.2
76.56
4.2
Map Size [MB]↓
382.4
5.2
10.8
Figure 1: Splat-SLAM. Our system yields accurate scene reconstruction (rendering depth L1) and
rendering (PSNR) and on par tracking accuracy (ATE RMSE) to GlORIE-SLAM and map size to
MonoGS. The results averaged over all keyframes. The scene is from TUM-RGBD [56] fr1 room.
works only deploy so called frame-to-model tracking, and do not implement global trajectory and
map optimization, leading to excessive drift, especially in real world conditions. Instead, to this
date, frame-to-frame tracking methods, coupled with loop closure and global bundle adjustment
(BA) achieve state-of-the-art tracking accuracy [79, 78, 75]. However, they either use hierarchical
feature grids [79, 78], not suitable for map deformations at e.g. loop closure as they require expensive
reintegration strategies, or neural point clouds as in GlORIE-SLAM [75]. While the neural point
cloud is straightforward to deform, the depth guided rendering leads to artifacts when the depth
is noisy and the surface estimation can only be adjusted locally since the point locations are not
optimized directly.
In this work we propose an RGB-only SLAM system that combines the strengths of frame-to-
frame tracking using recurrent dense optical flow [61] with the fidelity of 3D Gaussians as the map
representation [38] (see fig. 1). The point-based 3D Gaussian map enables online map deformations
at loop closure and global BA. To enable accurate surface reconstruction, we leverage consistent so
called proxy depth that combines multi-view depth estimation with learned monocular depth.
Our contribution comprises, for the first time, a SLAM pipeline encompassing all the following parts:
• A frame-to-frame RGB-only tracker with global consistency.
• A dense deformable 3D Gaussian map that adapts online to loop closure and global BA.
• A proxy depth map consisting of on-the-fly optimized multi-view depth and a monocular depth
estimator leading to improved rendering and reconstruction quality.
• Improved map sizes and runtimes compared to other dense SLAM approaches.
2
Related Work
Dense Visual SLAM. Curless and Levoy [9] pioneered dense online 3D mapping with truncated
signed distance functions, with KinectFusion [42] demonstrating real-time SLAM via depth maps.
Enhancements like voxel hashing [43, 23, 44, 11, 40] and octrees [53, 72, 37, 5, 31] improved
scalability, while point-based SLAM [68, 52, 4, 23, 25, 6, 76, 50, 30, 75] has also been effective.
To address pose drift, globally consistent pose estimation and dense mapping techniques have been
developed, often dividing the global map into submaps [4, 11, 15, 59, 40, 34, 22, 55, 7, 23, 48, 16,
2, 35, 59, 36, 30]. Loop detection triggers submap deformation via pose graph optimization [4, 34,
59, 40, 22, 13, 14, 27, 7, 17, 70, 52, 48, 16, 55, 63, 40, 18, 36, 30]. Sometimes global BA is used for
refinement [11, 52, 4, 61, 70, 73, 40, 8, 59, 18]. 3D Gaussian SLAM with RGBD input has also been
shown, but these methods do not consider global consistency via e.g. loop closure [74, 24, 69]. Other
approaches to global consistency minimize reprojection errors directly, with DROID-SLAM [61]
refining dense optical flow and camera poses iteratively, and recent enhancements like GO-SLAM [79],
HI-SLAM [78], and GlORIE-SLAM [75] optimizing factor graphs for accurate tracking. For a recent
survey on NeRF-inspired dense SLAM, see [62].
RGB-only Dense Visual SLAM. The majority of NeRF inspired dense SLAM works using only RGB
cameras do not address the problem of global map consistency or requires expensive reintegration
strategies via backpropagation [49, 8, 28, 81, 46, 79, 78, 20, 41, 19]. Instead, the concurrent GlORIE-
2

<!-- page 3 -->
Rendering
Mapping
Keyframe Pose
RGB
Depth
Proxy Depth 𝐷
Estimated Keyframe
Depth 𝐷𝑚𝑜𝑛𝑜
Monocular Depth 
Estimator
Compute Proxy Depth
Tracking
Multi-view
Filter
Keyframe
Buffer
Loop Closure
Global BA
Local BA
Keyframe
Selection
DSPO Layer
Conv GRU
Flow Revision 
∆scale, ∆pose
∆depth
Optimize Depth, Scale, Pose
Minimize
RGB Error
Depth Error
3D Gaussian Map
RGB Video
Depth ෡𝐷
Fixed Neural Network
Graph Structure
Optimization Layer
Memory Buffer
Figure 2: Splat-SLAM Architecture. Given an RGB input stream, we track and map each keyframe,
initially estimating poses through local bundle adjustment (BA) using a DSPO (Disparity, Scale and
Pose Optimization) layer. This layer integrates pose and depth estimation, enhancing depth with
monocular depth. It further refines poses globally via online loop closure and global BA. The proxy
depth map merges keyframe depths ˜D from the tracking with monocular depth Dmono to fill gaps.
Mapping employs a deformable 3D Gaussian map, optimizing its parameters through a re-rendering
loss. Notably, the 3D map adjusts for global pose and depth updates before each mapping phase.
SLAM [75] uses a feature based point cloud which can adapt to global map changes in a straight
forward way. However, redundant points are not pruned, leading to large map sizes. Furthermore,
the depth guided sampling during rendering leads to rendering artifacts when noise is present in
the estimated depth. MonoGS [38] and Photo-SLAM [21] pioneered RGB-only SLAM with 3D
Gaussians. However, they lack proxy depth which prevents them from achieving high accuracy
mapping. MonoGS [38] also lacks global consistency. Concurrent to our work, MoD-SLAM [80]
uses an MLP to parameterize the map via a unique reparameterization.
3
Method
Splat-SLAM is a monocular SLAM system which tracks the camera pose while reconstructing the
dense geometry of the scene in an online manner. This is achieved through the following steps: We
first track the camera by performing local BA on selected keyframes by fitting them to dense optical
flow estimates. The local BA optimizes the camera pose as well as the dense depth of the keyframe.
For global consistency, when loop closure is detected, loop BA is performed on an extended graph
including the loop nodes and edges (section 3.1). Interleaved with tracking, mapping is done on a
progressively growing 3D Gaussian map which deforms online to the keyframe poses and so called
proxy depth maps (section 3.2). For an overview of our method, see fig. 2.
3.1
Tracking
To predict the motion of the camera during scene exploration, we use a pretrained recurrent optical
flow model [60] coupled with a Disparity, Scale and Pose Optimization (DSPO) layer [75] to jointly
optimize camera poses and per pixel disparities. In the following, we describe this process in detail.
Optimization is done with the Gauss-Newton algorithm over a factor graph G(V, E), where the nodes
V store the keyframe pose and disparity, and edges E store the optical flow between keyframes.
Odometry keyframe edges are added to G by computing the optical flow to the last added keyframe.
If the mean flow is larger than a threshold τ ∈R, the new keyframe is added to G. Edges for loop
closure and global BA are discussed later. Importantly, the same objective is optimized for local BA,
loop closure and global BA, but over factor graphs with different structures.
The DSPO layer consists of two optimization objectives that are optimized alternatively. The first
objective, typically termed Dense Bundle Adjustment (DBA) [61] optimizes the pose and disparity
of the keyframes jointly, eq. (1). Specifically, the objective is optimized over a local graph defined
within a sliding window over the current frame.
arg min
ω,d
X
(i,j)∈E
˜pij −Kω−1
j (ωi(1/di)K−1[pi, 1]T )
2
Σij ,
(1)
with ˜pij ∈R(W ×H×2)×1 being the flattened predicted pixel coordinates when the pixels pi ∈
R(W ×H×2)×1 from keyframe i are projected into keyframe j using optical flow. Further, K is the
3

<!-- page 4 -->
camera intrinsics, ωj and ωi the camera-to-world extrinsics for keyframes j and i, di the disparity
of pixel pi and ∥· ∥Σij is the Mahalanobis distance with diagonal weighting matrix Σij. Each
weight denotes the confidence of the optical flow prediction for each pixel in ˜pij. For clarity of the
presentation, we omit homogeneous coordinates.
The second objective introduces monocular depth Dmono as two additional data terms. The monocular
depth Dmono is predicted at runtime by a pretrained relative depth DPT model [12].
arg min
dh,θ,γ
X
(i,j)∈E
˜pij −Kω−1
j (ωi(1/dh
i )K−1[pi, 1]T )
2
Σij
(2)
+α1
X
i∈V
dh
i −(θi(1/Dmono
i
) + γi)
2 + α2
X
i∈V
dl
i −(θi(1/Dmono
i
) + γi)
2 .
Here, the optimizable parameters are the scales θ ∈R, shifts γ ∈R and a subset of the disparities
dh classified as being high error (explained later). This is done since the monocular depth is only
deemed useful where the multi-view disparity di optimization is inaccurate. Furthermore, α1 <α2,
which is done to ensure that the scales θ and shifts γ are optimized with the preserved low error
disparities dl. The scale θi and shift γi are initialized using least squares fitting
{θi, γi} = arg min
θ,γ
X
(u,v)
 θ(1/Dmono
i
) + γ

−dl
i
2
.
(3)
Equation (1) and eq. (2) are optimized alternatingly to avoid the scale ambiguity encountered if d, θ,
γ and ω are optimized jointly.
Next, we describe how high and low error disparities are classified. For a given disparity map di
(separated into low and high error parts {dl
i, dh
i }) for frame i, we denote the corresponding depth
˜Di = 1/di. Pixel correspondences (u, v) and (ˆu, ˆv) between keyframes i and j respectively are
established by warping (u, v) into frame j with depth ˜Di as
pi = ωi ˜Di(u, v)K−1[u, v, 1]T ,
[ˆu, ˆv, 1]T ∝Kω−1
j [pi, 1]T .
(4)
The corresponding 3D point to (ˆu, ˆv) is computed from the depth at (ˆu, ˆv) as
pj = ωj ˜Dj(ˆu, ˆv)K−1[ˆu, ˆv, 1]T .
(5)
If the L2 distance between pi and pj is smaller than a threshold, the depth ˜Di(u, v) is consistent
between i and j. By looping over all keyframes except i, the global two-view consistency ni can be
computed for frame i as
ni(u, v) =
X
k∈KFs,
k̸=i
1

∥pi −pk∥2 < η · average( ˜Di)

.
(6)
Here, 1(·) is the indicator function and η ∈R≥0 is a hyperparameter and ni is the total two-view
consistency for pixel (u, v) in keyframe i. ˜Di(u, v) is valid if ni is larger than a threshold.
Loop Closure. To mitigate scale and pose drift, we incorporate loop closure along with online global
bundle adjustment (BA) in addition to local window frame tracking. Loop detection is achieved by
calculating the mean optical flow magnitude between the current active keyframes (within the local
window) and all previous keyframes. Two criteria are evaluated for each keyframe pair: First, the
optical flow must be below a specified threshold τloop, ensuring sufficient co-visibility between the
views. Second, the time interval between the frames must exceed a predefined threshold τt to prevent
the introduction of redundant edges into the graph. When both criteria are met, a unidirectional edge
is added to the graph. During the loop closure optimization process, only the active keyframes and
their connected loop nodes are optimized to keep the computational load manageable.
Global BA. For the online global BA, a separate graph that includes all keyframes up to the present
is constructed. Edges are introduced based on the temporal and spatial relationships between the
keyframes, as outlined in [79]. Following the approach detailed in [75], we execute online global
BA every 20 keyframes. To maintain numerical stability, the scales of the disparities and poses
are normalized prior to each global BA optimization. This normalization involves calculating the
average disparity ¯d across all keyframes and then adjusting the disparity to dnorm = d/ ¯d and the
pose translation to tnorm = ¯dt.
4

<!-- page 5 -->
3.2
Deformable 3D Gaussian Scene Representation
We adopt a 3D Gaussian Splatting representation [26] which deforms under DSPO or loop closure
optimizations to achieve global consistency. Thus, the scene is represented by a set G = {gi}N
i=1 of
3D Gaussians. Each Gaussian primitive gi, is parameterized by a covariance matrix Σi ∈R3×3, a
mean µi ∈R3, opacity oi ∈[0, 1], and color ci ∈R3. All attributes of each Gaussian are optimized
through back-propagation. The density function of a single Gaussian is described as
gi(x) = exp

−1
2(x −µi)⊤Σ−1
i (x −µi)

.
(7)
Here, the spatial covariance Σi defines an ellipsoid and is decomposed as Σi = RiSiST
i RT
i , where
Si = diag(si) ∈R3×3 is the spatial scale and Ri ∈R3×3 represents the rotation.
Rendering. Rendering color and depth from G, given a camera pose, involves first projecting (known
as “splatting”) 3D Gaussians onto the 2D image plane. This is done by projecting the covariance
matrix Σ and mean µ as Σ′ = JRΣRT JT and µ′ = Kω−1µ, where R is the rotation component of
world-to-camera extrinsics ω−1 and J is the Jacobian of the affine approximation of the projective
transformation [84]. The final pixel color C and depth Dr at pixel x′ is computed by blending 3D
Gaussian splats that overlap at a given pixel, sorted by their depth as
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj)
Dr =
X
i∈N
ˆdiαi
i−1
Y
j=1
(1 −αj) ,
(8)
where ˆdi is the z-axis depth of the center of the i-th 3D Gaussian and the final opacity αi is the
product of the opacity oi and the 2D Gaussian density as
αi = oi exp

−1
2(x′ −µ′
i)⊤Σ′−1
i
(x′ −µ′
i)

.
(9)
Map Initialization. For every new keyframe, we adopt the RGBD strategy of MonoGS [38] for
adding new Gaussians to the unexplored scene space. As we do not have access to a depth sensor, we
construct a proxy depth map D by combining the inlier multi-view depth ˜D and the monocular depth
Dmono as
D(u, v) =
 ˜D(u, v)
if ˜D(u, v) is valid
θDmono(u, v) + γ
otherwise
(10)
Here, θ and γ are computed as in eq. (3) but using depth instead of disparity.
Keyframe Selection and Optimization. Apart from the keyframe selection based on a mean optical
flow threshold τ, we additionally adopt the keyframe selection strategy from [38] to avoid mapping
redundant frames.
To optimize the 3D Gaussian parameters, we batch the parameter updates to a local window similar
to [38] and apply a photometric and geometric loss to the proxy depth as well as a scale regularizer to
avoid artifacts from elongated Gaussians. Inspired by [38], we further use exposure compensation by
optimizing an affine transformation for each keyframe. The final loss is
min
G,a,b
X
k∈KFs
λ
Nk
|(akCk + bk) −Cgt
k |1 + 1 −λ
Nk
|Dr
k −Dk|1 + λreg
|G|
|G|
X
i
|si −˜si|1 ,
(11)
where KFs contains the set of keyframes in the local window, Nk is the number of pixels per keyframe,
λ and λreg are hyperparameters, a = {a1, . . . , ak, . . . } and b = {b1, . . . , bk, . . . } are the parameters
for the exposure compensation and ˜s is the mean scaling, repeated over the three dimensions.
Map Deformation. Since our tracking framework is globally consistent, changes in the keyframe
poses and proxy depth maps need to be accounted for in the 3D Gaussian map by a non-rigid
deformation. Though the Gaussian means are directly optimized, one could in theory let the optimizer
deform the map as refined poses and proxy depth maps are provided. We find, however, that in
particular rendering is aided by actively deforming the 3D Gaussian map. We apply the deformation
to all Gaussians which receive updated poses and depths before mapping.
Each Gaussian gi is associated with a keyframe that anchored it to the map G. Assume that a keyframe
with camera-to-world pose ω and proxy depth D is updated such that ω →ω′ and D →D′. We
5

<!-- page 6 -->
Metric
GO-
SLAM [79]
NICER-
SLAM [81]
MoD-
SLAM∗[28]
Photo-
SLAM [21]
Mono-
GS [38]
GlORIE-
SLAM∗[75]
Q-SLAM
∗[46]
Ours
PSNR↑
22.13
25.41
27.31
33.30
31.22
31.04
32.49
36.45
SSIM ↑
0.73
0.83
0.85
0.93
0.91
0.91
0.89
0.95
LPIPS↓
-
0.19
-
-
0.21
0.12
0.17
0.06
ATE RMSE↓
0.39
1.88
0.35
1.09
14.54
0.35
-
0.35
Table 1:
Rendering and Tracking Results on Replica [54] for RGB-Methods. Our method
outperforms all methods on rendering and performs on par for tracking accuracy. Results are
from [62] except ours (average over 8 scenes). Best results are highlighted as first , second , third .
update the mean, scale and rotation of all Gaussians gi associated with the keyframe. Association is
determined by what keyframe added the Gaussian to the scene. The mean µi is projected into ω to
find the pixel correspondence (u, v). Since the Gaussians are not necessarily anchored on the surface,
instead of re-anchoring the mean at D′, we opt to shift the mean by D′(u, v) −D(u, v) along the
optical axis. We update Ri and si accordingly as
µ′
i =

1 + D′(u, v) −D(u, v)
(ω−1µi)z

ω′ω−1µi , R′
i = R′R−1Ri , s′
i =

1 + D′(u, v) −D(u, v)
(ω−1µi)z

si .
(12)
Here, (·)z denotes the z-axis depth. For Gaussians which project into pixels with missing depth or
outside the viewing frustum, we only rigidly deform them. After the final global BA optimization, we
additionally deform the Gaussian map and perform a set of final refinements (see suppl. material).
4
Experiments
We first describe our experimental setup and then evaluate our method against state-of-the-art dense
RGB and RGBD SLAM methods on Replica [54] as well as the real world TUM-RGBD [56] and the
ScanNet [10] datasets. For more experiments and details, we refer to the supplementary material.
Implementation Details. For the proxy depth, we use η = 0.01 to filter points and use the condition
nc ≥2 to ensure multi-view consistency. For the mapping loss function, we use λ = 0.8, λreg = 10.0.
We use 60 iterations during mapping. For tracking, we use α1 = 0.01 and α2 = 0.1 as weights
for the DSPO layer. We use the flow threshold τ = 4.0 on ScanNet, τ = 3.0 on TUM-RGBD and
τ = 2.25 on Replica. The threshold for loop detection is τloop = 25.0. The time interval threshold is
τt = 20. We conducted the experiments on a cluster with an NVIDIA A100 GPU.
Evaluation Metrics. For rendering we report PSNR, SSIM [65] and LPIPS [77] on the rendered
keyframe images against the sensor images. For reconstruction, we first extract the meshes with
marching cubes [32] as in [50] and evaluate the meshes using accuracy [cm], completion [cm] and
completion ratio [%] (threshold 5 cm) against the ground truth meshes. We also report the re-rendering
depth L1 [cm] metric to the ground truth sensor depth as in [49]. We use ATE RMSE [cm] [56] to
evaluate the estimated trajectory.
Datasets. We use the RGBD trajectories from [57] captured from the synthetic Replica dataset [54].
We also test on real-world data using the TUM-RGBD [56] and the ScanNet [10] datasets.
Baseline Methods. We compare our method to numerous published and concurrent works on dense
RGB and RGBD SLAM. Concurrent works are denoted with an asterix∗. The main baselines are
GlORIE-SLAM [75] and MonoGS [38].
Rendering. In tab. 1, we evaluate the rendering performance on Replica [54] and find that our
method performs superior among all baseline RGB-methods. Tab. 2 and tab. 3 show the rendering
accuracy on the ScanNet [10] and TUM-RGBD [56] datasets. In particular, we outperform existing
RGB-only works with a clear margin, while even beating the currently best RGBD method, Gaussian-
SLAM [74] on most metrics, despite the fact that we do not implement view-dependent rendering in
the form of spherical harmonics. We attribute this to our deformable 3D Gaussian map, optimized
with strong proxy depth along a globally consistent tracking backend. In fig. 3 and fig. 1 we show
renderings on the real-world ScanNet [10] and TUM-RGBD [56] datasets. Due to high tracking errors,
MonoGS [38] performs poorly on some scenes, yet fails to achieve the same fidelity as our method
when the tracking error is low, as a result of the weak geometric constraints during optimization. Our
method avoids the artifacts produced by GlORIE-SLAM [75] and yields high quality renderings.
6

<!-- page 7 -->
Method
Metric
0000
0059
0106
0169
0181
0207
Avg.
RGB-D Input
SplaTaM [24]
PSNR↑
19.33
19.27
17.73
21.97
16.76
19.80
19.14
SSIM ↑
0.66
0.79
0.69
0.78
0.68
0.70
0.72
LPIPS↓
0.44
0.29
0.38
0.28
0.42
0.34
0.36
MonoGS [38]
PSNR↑
18.70
20.91
19.84
22.16
22.01
18.90
20.42
SSIM ↑
0.71
0.79
0.81
0.78
0.82
0.75
0.78
LPIPS↓
0.48
0.32
0.32
0.34
0.42
0.41
0.38
Gaussian-
SLAM [74]
PSNR↑
28.54
26.21
26.26
28.60
27.79
28.63
27.67
SSIM ↑
0.93
0.93
0.93
0.92
0.92
0.91
0.92
LPIPS↓
0.27
0.21
0.22
0.23
0.28
0.29
0.25
RGB Input
GO-
SLAM [79]
PSNR↑
15.74
13.15
14.58
14.49
15.72
15.37
14.84
SSIM ↑
0.42
0.32
0.46
0.42
0.53
0.39
0.42
LPIPS↓
0.61
0.60
0.59
0.57
0.62
0.60
0.60
MonoGS [38]
PSNR↑
16.91
19.15
18.57
20.21
19.51
18.37
18.79
SSIM ↑
0.62
0.69
0.74
0.74
0.75
0.70
0.71
LPIPS↓
0.70
0.51
0.55
0.54
0.63
0.58
0.59
GlORIE-
SLAM∗[75]
PSNR↑
23.42
20.66
20.41
25.23
21.28
23.68
22.45
SSIM ↑
0.87
0.87
0.83
0.84
0.91
0.76
0.85
LPIPS↓
0.26
0.31
0.31
0.21
0.44
0.29
0.30
Splat-SLAM
(Ours)
PSNR↑
28.68
27.69
27.70
31.14
31.15
30.49
29.48
SSIM ↑
0.83
0.87
0.86
0.87
0.84
0.84
0.85
LPIPS ↓
0.19
0.15
0.18
0.15
0.23
0.19
0.18
Table 2: Rendering Performance on ScanNet [10]. Our method performs even better or on par with
all RGB-D methods. We take the numbers for SplaTaM and Gaussian-SLAM from [74].
Method
Method
f1/desk
f2/xyz
f3/off
f1/desk2
f1/room
Avg.
RGB-D Input
SplaTaM [24]
PSNR↑
22.00
24.50
21.90
-
-
-
SSIM ↑
0.86
0.95
0.88
-
-
-
LPIPS ↓
0.23
0.10
0.20
-
-
-
Gaussian-
SLAM [74]
PSNR↑
24.01
25.02
26.13
23.15
22.98
24.26
SSIM ↑
0.92
0.92
0.94
0.91
0.89
0.92
LPIPS ↓
0.18
0.19
0.14
0.20
0.24
0.19
RGB Input
Photo-SLAM [21]
PSNR↑
20.97
21.07
19.59
-
-
-
SSIM ↑
0.74
0.73
0.69
-
-
-
LPIPS ↓
0.23
0.17
0.24
-
-
-
MonoGS [38]
PSNR↑
19.67
16.17
20.63
19.16
18.41
18.81
SSIM ↑
0.73
0.72
0.77
0.66
0.64
0.70
LPIPS ↓
0.33
0.31
0.34
0.48
0.51
0.39
GlORIE-
SLAM∗[75]
PSNR↑
20.26
25.62
21.21
19.09
18.78
20.99
SSIM ↑
0.79
0.72
0.72
0.92
0.73
0.77
LPIPS ↓
0.31
0.09
0.32
0.38
0.38
0.30
Splat-SLAM
(Ours)
PSNR↑
25.61
29.53
26.05
23.98
24.06
25.85
SSIM ↑
0.84
0.90
0.84
0.81
0.80
0.84
LPIPS ↓
0.18
0.08
0.20
0.23
0.24
0.19
Table 3: Rendering Performance on TUM-RGBD [56]. Our method performs competitively or
better than RGB-D methods. For all RGB-D methods, we take the numbers from [74].
Reconstruction. We show quantitative and qualitative results on the Replica [54] dataset in tab. 4
and fig. 4 respectively. Our method achieves the best performance on all metrics. Qualitatively, we
show normal shaded meshes from different viewpoints. Our method can reconstruct finer details
than existing works, especially around thin structures (e.g. second row), where our strong proxy
depth coupled with the 3D Gaussian map representation yields superior depth rendering, which
directly influences the mesh quality. In contrast, e.g. GlORIE-SLAM [75] uses depth guided volume
rendering, which is sensitive to input depth noise, resulting in inconistent depth rendering with
floating artifacts. MonoGS [38] suffers significantly from the lack of proxy depth, visible in all
scenes. Fig. 1 shows depth rendering on the real-world TUM-RGBD [56] room scene. We compute
the average depth L1 error over all keyframes, achieving 15.05 cm, beating existing works.
Ablation Study. In tab. 5, we conduct a set of ablation studies related to our method, by enabling
and disabling certain parts. We find that the combination of filtered multiview depth completed with
monocular depth yields the best performance in terms of rendering and reconstruction metrics.
7

<!-- page 8 -->
Scene
0000
Scene
0054
fr3
office
GlORIE-SLAM∗[75]
MonoGS [38]
Splat-SLAM (Ours)
Ground Truth
Figure 3: Rendering Results on ScanNet [10] and TUM-RGBD [56]. Our method yields better
rendering quality than GlORIE-SLAM and MonoGS. Top row: the orange box shows artifacts from
GlORIE-SLAM, partly due to the depth guided volume rendering. The yellow box shows an area with
redundant floating points. The red box shows a rendering distortion, likely from the large trajectory
error. The green boxes show that our method fuses information from multiple views to avoid motion
blur, present in the input. Fourth row: The rendering is from the pose of the red box in the third row.
Metrics
NeRF-
SLAM [62]
DIM-
SLAM [28]
GO-
SLAM [79]
NICER-
SLAM [81]
HI-
SLAM [78]
MoD-
SLAM∗[80]
GlORIE-
SLAM∗[75]
Mono-
GS[38]
Q-SLAM
∗[46]
Ours
Render Depth L1↓
4.49
-
-
-
-
-
-
27.24
2.76
2.41
Accuracy ↓
-
4.03
3.81
3.65
3.62
2.48
2.96
30.61
-
2.43
Completion ↓
-
4.20
4.79
4.16
4.59
-
3.95
12.19
-
3.64
Comp. Rat. ↑
-
79.60
78.00
79.37
80.60
-
83.72
40.53
-
84.69
Table 4: Reconstruction Results on Replica [54] for RGB-Methods. Our method outperforms
existing works on all metrics. Results are averaged over 8 scenes.
Memory and Runtime. In tab. 6, we evaluate the peak GPU memory usage, map size and runtime of
our method. We achieve a comparable GPU memory usage with GO-SLAM [79] and SplaTaM [24].
Our map size is similar to MonoGS [38] and much smaller than GlORIE-SLAM, which does not
prune redundant neural points. In fig. 1 we also show similar map size to MonoGS on the real-world
TUM-RGBD [56] room scene. Regarding runtime, we are faster than SplaTaM and GlORIE-SLAM
and comparable to MonoGS. GO-SLAM has the fastest runtime, but as shown in tab. 1 and tab. 4, it
sacrifices rendering and reconstruction quality for speed.
Limitations. We currently do not model the appearance with spherical harmonics, since it only
yields a marginal gains in rendering accuracy, while requiring more memory. It is is straightforward
to add. We only make use of globally optimized frame-to-frame tracking, which fails to leverage
frame-to-model queues from the 3D Gaussian map. Another limitation is that our construction of
the final proxy depth D is quite simple and does not fuse the monocular and keyframe depths in an
informed manner, e.g. using normal consistency. Finally, as future work, it is interesting to study how
surface regularization can be enforced via e.g. quadric surface elements as in [46].
8

<!-- page 9 -->
Office 0
Office 4
Room 0
GlORIE-SLAM∗[75]
MonoGS [38]
Splat-SLAM (Ours)
Ground Truth
Figure 4: Reconstruction Results on Replica [54] on Normal Shaded Meshes. Our method
achieves higher geometric accuracy compared to existing works. In particular, GlORIE-SLAM
suffers from floating point artifacts (e.g. second row) where our method reconstructs even the
individual legs of the table. MonoGS suffers significantly from a lack of proxy depth, despite
multiview optimization.
Mono
Depth
Multiview
Depth
Multiview
Filtering
PSNR
[dB] ↑
Acc.
[cm] ↓
Comp.
[cm] ↓
Comp. Ratio
[cm] ↑
✓
✗
✗
36.02
3.62
4.08
81.16
✗
✓
✓
36.17
2.64
4.73
80.12
✗
✓
✗
36.21
18.71
4.06
80.29
✓
✓
✓
36.45
2.43
3.64
84.69
Table 5: Ablation Study on Replica [54]. We show that the combination of filtered multiview depth
completed with monocular depth yields the best performance on all metrics. Mono Depth refers to
Dmono, Multiview Depth refers to ˜D and Multiview Filtering means enabling eq. (6). All results are
averaged over 8 scenes.
GO-SLAM [79]
SplaTAM [24]
GlORIE-SLAM∗[75]
MonoGS [38]
Ours
GPU Usage [GiB]
18.50
18.54
15.22
14.62
17.57
Map Size [MB]
-
-
114.0
6.8
6.5
Avg. FPS
8.36
0.14
0.23
0.32
1.24
Table 6: Memory and Running Time Evaluation on Replica [54] room0. Our peak memory usage
and runtime are comparable to existing works. We take the numbers from [62] except for ours and
MonoGS and we add the Map Size, which denotes the size of the final 3D representation. GPU Usage
denotes the peak usage during runtime. All methods are evaluated on an NVIDIA RTX 3090 GPU
using single threading for fairness.
5
Conclusion
We proposed Splat-SLAM, a dense RGB-only SLAM system which uses a deformable 3D Gaussian
map for mapping and globally optimized frame-to-frame tracking via optical flow. Importantly, the
inclusion of monocular depth into the tracking loop, to refine the scale and to correct the erroneous
keyframe depth predictions, leads to better rendering and mapping. By using the monocular depth
for completion, mapping is further improved. Our experiments demonstrate that Splat-SLAM
outperforms existing solutions regarding reconstruction and rendering accuracy while being on par or
better with respect to tracking as well as runtime and memory usage.
9

<!-- page 10 -->
Supplementary Material
Splat-SLAM: Globally Optimized RGB-only SLAM
with 3D Gaussians
Erik Sandström*
Keisuke Tateno
Michael Oechsle
Michael Niemeyer
Google
Google
Google
Google
ETH Zürich
Luc Van Gool
Martin R. Oswald
Federico Tombari
ETH Zürich
ETH Zürich
Google
INSAIT
University of Amsterdam
TU München
This supplementary material accompanies the main paper and provides more details on the methodol-
ogy and additional experimental results.
A
Method
We describe further details about our method that were left out from the main paper.
Comparison to Existing Works. To further clarify the differences between our method and existing
3DGS SLAM works, we classify each method in tab. S8 based on important characteristics. It shows
that our work is the first to include loop closure, proxy depth, RGB-only and online 3D Gaussian
deformations.
RGB-only
Loop
Closure
Proxy
Depth
Online 3DGS
Deformations
GS-SLAM [69]
✗
✗
✓
✗
Gaussian-SLAM [74]
✗
✗
✓
✗
SplaTaM [24]
✗
✗
✓
✗
MonoGS [38]
✓
✗
✗
✗
Photo-SLAM [21]
✓
✓
✗
✗
Splat-SLAM (ours)
✓
✓
✓
✓
Table S8: Method Classification. We show that our method is the first to combine 3D Gaussian
SLAM with loop closure, proxy depth and online 3D Gaussian map deformations in an RGB-only
SLAM system.
Map Initialization. With map initialization, we refer to the process of anchoring new Gaussians dur-
ing scene exploration. For every new keyframe to be mapped, we adopt the strategy that MonoGS [38]
uses in pure RGBD mode. It works by unprojecting the depth reading per pixel to 3D and then
downsampling this point cloud by a factor θ. New Gaussians are then assigned their means as the
point cloud. The rotations are initialized to identity, the opacity to 0.5 and the scales are initialized
related to their distance to the nearest neighbor point in the point cloud.
Keyframe Selection and Local Windowing. As mentioned in the main paper, we adopt the keyframe
selection strategy from MonoGS [38]. We describe this strategy in the following.
Keyframes are selected based on the covisibility of the Gaussians. Between two keyframes i and j,
the covisibility is defined using the Intersection over Union (IOU) and Overlap Coefficient (OC):
IOUcov(i, j) = |Gi
v ∩Gj
v|
|Giv ∪Gj
v|
,
(16)
*This work was conducted during an internship at Google.
10

<!-- page 11 -->
OCcov(i, j) =
|Gi
v ∩Gj
v|
min(|Giv|, |Gj
v|)
,
(17)
where Gi
v are the Gaussians visible in keyframe i, based on the following definition of visibility.
A Gaussian is seen as visible from a camera pose if it is used in the rasterization pipeline when
rendering and if the accumulated transmittance Qi−1
j=1(1 −αj) has not yet reached 0.5.
A keyframe i is added to the keyframe window KFs if, given the last keyframe j, IOUcov(i, j) < kfcov
or if the relative translation tij > kfm ˆDi, where ˆDi is the median depth of frame i. For Replica,
kfcov = 0.95, kfm = 0.04 and for TUM and ScanNet, kfcov = 0.90, kfm = 0.08. The registered
keyframe j in KFs is removed if OCcov(i, j) < kfc, where keyframe i is the latest added keyframe.
For all datasets, the cutoff is set to kfc = 0.3. The size of the keyframe window is set to |KFs| = 10
for Replica and |KFs| = 8 for TUM and ScanNet.
Pruning and Densification We also follow [38] when it comes to Gaussian pruning and densification.
Pruning is done based on the visibility: if new Gaussians inserted within the last 3 keyframes are not
visible by at least 3 other frames in the keyframe window KFs, they are removed. Visibility-based
pruning is only done when the keyframe window KFs is full. Additionally, every 150 mapping
iterations, Gaussians with opacity lower than 0.7 are removed globally. Also Gaussians which project
in 2D with a too large scale are removed. Densification is done as in [26], also at an interval of every
150 mapping iterations.
Final Refinement. Similar to GlORIE-SLAM [75], which performs a final refinement after the last
final global BA at the end of the trajectory, we also perform a few refinement iterations after the
last final global BA. Also MonoGS [38] performs a set of final iterations at the end of the SLAM
trajectory to refine the colors.
Our refinement strategy is straight forward. We disable pruning and densification of the Gaussians
and perform a set of optimization iterations β using the same loss function as in the main paper, but
only sampling random single frames per iteration.
Differences to GlORIE-SLAM. We briefly discuss some differences to GlORIE-SLAM [75] not
covered in the main paper. GlORIE-SLAM uses an additional point cloud called Pd consisting of
all inlier mullti-view depth maps unprojected into a point cloud. We found that this is not needed
and it saves memory and compute to not use it. GlORIE-SLAM also re-anchors the neural points at
the depth reading. We do not do this as the Gaussians do not necessarily lie on the surface exactly.
Finally, GlORIE-SLAM requires input depth to guide the sampling of points to render color and
depth. If the depth is noisy or if the map is used for tracking (i.e. frame-to-model tracking), the
depth guiding strategy is not favorable as it leads to artifacts when sampling the wrong points (when
noisy depth is encountered) and to a much smaller basin of convergence when tracking (because the
rendering is conditioned on the current view point). With 3D Gaussians, we can avoid depth guidance
during rendering.
B
More Experiments
To accompany the evaluations provided in the main paper, we provide further experiments in this
section.
Implementation Details. As the point cloud downsampling factor, we use θ = 32 for all frames
but the first frame where θ = 16 is used. We use β = 2000, the number of iterations for the
final refinement optimization, on the Replica dataset and β = 26000 on the TUM-RGBD [56] and
ScanNet [10] datasets (same as MonoGS [38]). We benchmark the runtime on an AMD Ryzen
Threadripper Pro 3945WX 12-Cores with an NVIDIA GeForce RTX 3090 Ti with 24 GB of memory.
For the remaining hyperparameters, we refer to MonoGS [38] for the Gaussian mapping and GlORIE-
SLAM for tracking [75].
A Note on Rendering and Runtime with MonoGS. By default, MonoGS [38] does not evaluate
the rendering error on the mapped keyframes nor implement the exposure compensation during
rendering evaluation. To compare our results fairly to MonoGS, we implement these details and run
the experiments with these settings enabled. Further, we report the runtime for MonoGS using a
single process (same as us) compared to the reported number in the paper, which was using multiple
processes at once.
11

<!-- page 12 -->
A Note on Gaussian Deformation with Photo-SLAM. Though not fully clear from reading the
paper, after discussing with the authors of Photo-SLAM [21], we find that they do, in fact, not
deform the Gaussians as a result of global BA or loop closure. They found this to be unstable in their
experiments. This suggests that our deformation strategy is non-trivial.
Justification of Monocular Depth Estimator. There are already numerous monocular depth
estimators, but most of them are limited by speed, memory or quality. We use Omnidata [12] since
empirically we found it still provides the best trade-off between output performance and runtime. We
also tested our system with Depth Anything [71], but found that it was marginally worse in terms of
the final reconstructed mesh accuracy.
B.1
Tracking on ScanNet and TUM-RGBD
We do not put the results on tracking for ScanNet and TUM-RGBD since we use the tracking
framework from GlORIE-SLAM [75], but we provide the numbers here. Tab. S9 and tab. S10 show
the tracking accuracy of the estimated trajectory on ScanNet [10] and TUM-RGBD [56] respectively.
Our method shows competitive results in every single scene and gives the best average value among
the RGB and RGB-D methods.
Method
0000
0059
0106
0169
0181
0207
Avg.-6
0054
0233
Avg.-8
RGB-D Input
NICE-SLAM [82]
12.0
14.0
7.9
10.9
13.4
6.2
10.7
20.9
9.0
11.8
Co-SLAM [64]
7.1
11.1
9.4
5.9
11.8
7.1
8.7
-
-
-
ESLAM [33]
7.3
8.5
7.5
6.5
9.0
5.7
7.4
36.3
4.3
10.6
MonoGS[38]
16.1
6.4
8.1
8.7
26.4
9.2
12.5
20.6
13.1
13.6
RGB Input
MonoGS[38]
149.2
96.8
155.5
140.3
92.6
101.9
122.7
206.4
89.1
129.0
GO-SLAM [79]
5.9
8.3
8.1
8.4
8.3
6.9
7.7
13.3
5.3
8.1
HI-SLAM[78]
6.4
7.2
6.5
8.5
7.6
8.4
7.4
-
-
-
Q-SLAM∗[46]
5.8
8.5
8.4
8.7
8.8
-
-
12.6
5.3
-
GlORIE-SLAM∗[75]
5.5
9.1
7.0
8.2
8.3
7.5
7.6
9.4
5.1
7.5
Ours
5.5
9.1
7.0
8.2
8.3
7.5
7.6
9.4
5.1
7.5
Table S9:
Tracking Accuracy ATE RMSE [cm] ↓on ScanNet [10]. Our method equals to
GlORIE-SLAM [75], giving the average lowest trajectory error. Results for the RGB-D methods are
from [30]. Note that all methods with a ∗are concurrent works.
Method
f1/desk
f2/xyz
f3/off
Avg.-3
f1/desk2
f1/room
Avg.-5
RGB-D Input
SplaTAM [24]
3.4
1.2
5.2
3.3
6.5
11.1
5.5
GS-SLAM∗[69]
1.5
1.6
1.7
1.6
-
-
-
GO-SLAM [79]
1.5
0.6
1.3
1.1
-
4.7
-
MonoGS [38]
1.4
1.4
1.5
1.5
5.1
6.3
3.1
RGB Input
MonoGS [38]
3.8
5.2
2.9
4.0
75.7
76.6
32.8
Photo-SLAM [21]
1.5
1.0
1.3
1.3
-
-
-
DIM-SLAM [28]
2.0
0.6
2.3
1.6
-
-
-
GO-SLAM [79]
1.6
0.6
1.5
1.2
2.8
5.2
2.3
MoD-SLAM∗[80]
1.5
0.7
1.1
1.1
-
-
-
Q-SLAM∗[46]
1.3
0.9
-
-
2.3
4.9
-
GlORIE-SLAM∗[75]
1.6
0.2
1.4
1.1
2.8
4.2
2.1
Ours
1.6
0.2
1.4
1.1
2.8
4.2
2.1
Table S10: Tracking Accuracy ATE RMSE [cm] ↓on TUM-RGBD [56]. Our method equals to
GlORIE-SLAM [75], giving the average lowest trajectory error. Note that all methods with a ∗are
concurrent works.
B.2
Full Evaluations Data
In tab. S11, tab. S12 and tab. S13, we provide the full per scene results on all commonly reported
metrics on Replica [54], TUM-RGBD [56] and ScanNet [10].
The reconstruction results are only measured on Replica since the other two datasets are real world
datasets which lack quality ground truth meshes.
12

<!-- page 13 -->
Metric
R-0
R-1
R-2
O-0
O-1
O-2
O-3
O-4
Avg.
Reconstruction
Render Depth L1 ↓
2.90
2.16
2.18
2.44
1.97
2.46
2.62
2.53
2.41
Accuracy ↓
1.99
1.91
2.06
3.96
2.03
3.45
2.15
1.89
2.43
Completion ↓
3.78
3.38
3.34
2.75
3.33
4.36
3.96
4.25
3.64
Comp. Rat. ↑
85.47
86.88
86.12
87.32
85.17
81.37
82.25
82.95
84.69
Rendering
Keyframes
PSNR ↑
32.25
34.31
35.95
40.81
40.64
35.19
35.03
37.40
36.45
SSIM ↑
0.91
0.93
0.95
0.98
0.97
0.96
0.95
0.98
0.95
LPIPS ↓
0.10
0.09
0.06
0.05
0.05
0.07
0.06
0.04
0.06
Tracking
Keyframes
Trajectory
ATE
RMSE ↓
0.29
0.38
0.24
0.27
0.35
0.34
0.42
0.43
0.34
Full
Trajectory
ATE
RMSE ↓
0.29
0.33
0.25
0.29
0.35
0.34
0.42
0.43
0.34
Number of
Gaussians
1000x
116
116
91
76
66
134
114
106
102
Table S11: Full Evaluation on Replica [54]. We show the ATE RMSE [cm] evaluation on the
keyframes as well as on the full trajectory.
Metric
f1/desk
f1/desk2
f1/room
f2/xyz
f3/office
Avg.
Rendering
Keyframes
PSNR ↑
25.61
23.98
24.06
29.53
26.05
25.85
SSIM ↑
0.84
0.81
0.80
0.90
0.84
0.84
LPIPS ↓
0.18
0.23
0.24
0.08
0.20
0.19
Depth
Rendering
Keyframes
Depth
L1↓[cm]
8.05
15.70
15.05
14.53
25.59
15.78
Tracking
Key Frames
Trajectory
ATE
RMSE ↓
1.92
3.05
4.43
0.23
1.41
2.21
Full
Trajectory
ATE
RMSE ↓
1.65
2.79
4.16
0.22
1.44
2.05
Number of
Gaussians
1000x
88
78
211
173
114
133
Table S12: Full Evaluation on TUM-RGBD [56].
Metric
0000
0054
0059
0106
0169
0181
0207
0233
Avg.
Rendering
Keyframes
PSNR↑
28.68
30.21
27.69
27.70
31.14
31.15
30.49
27.48
29.32
SSIM ↑
0.83
0.85
0.87
0.86
0.87
0.84
0.84
0.78
0.84
LPIPS ↓
0.19
0.22
0.15
0.18
0.15
0.23
0.19
0.22
0.19
Depth
Rendering
Keyframes
Depth
L1↓[cm]
8.24
18.24
13.39
23.5
11.49
18.35
13.78
10.19
11.37
Tracking
Key Frames
Trajectory
ATE
RMSE ↓
5.66
9.17
9.48
7.03
8.72
8.42
7.47
4.97
7.61
Full
Trajectory
ATE
RMSE ↓
5.57
9.50
9.11
7.09
8.26
8.39
7.53
5.17
7.58
Number of
Gaussians
1000x
144
157
84
108
52
127
121
191
123
Table S13: Full Evaluation on ScanNet [10].
We show the trajectory accuracy measurement of both keyframes and the full trajectory, which is
obtained by first linear interpolation between keyframes and using optical flow to refine. The accuracy
of these two trajectories are similar. In the main paper, the data we report is always measured on the
full trajectory.
B.3
Influence of Monocular Depth
While we show that the monocular depth improves the geometric estimation capability of our
framework, it may still be erroneous. To better understand the accuracy of the monocular depth, we
replace it with the ground truth sensor depth instead. This experiment acts as the upper bound of
our method if the monocular depth is perfect. The experiments are done on Replica [54] and are
shown in tab. S14. Compared with the standard setting with the monocular depth, the ground truth
depth setting gives improvements on both reconstruction and rendering quality, which reveals that our
method still has potential to achieve better mapping results once better monocular depth is available.
13

<!-- page 14 -->
Metric
R-0
R-1
R-2
O-0
O-1
O-2
O-3
O-4
Avg.
Recon-
struction
Render Depth L1 ↓
2.38
1.31
1.73
1.15
1.60
1.29
5.71
1.93
2.14
Accuracy ↓
1.29
0.91
1.05
1.22
0.83
0.96
1.24
1.07
1.07
Completion ↓
3.43
2.83
2.66
1.50
2.46
3.57
3.46
3.61
2.94
Comp. Rat. ↑
86.61
88.69
88.70
93.44
89.09
85.20
84.60
85.32
87.71
Rendering
PSNR ↑
35.66
37.65
38.87
43.95
43.28
37.93
37.41
39.88
39.33
SSIM ↑
0.96
0.96
0.97
0.99
0.98
0.96
0.96
0.98
0.97
LPIPS ↓
0.04
0.05
0.03
0.02
0.02
0.06
0.04
0.03
0.04
Tracking
ATE
RMSE ↓
0.29
0.38
0.24
0.28
0.39
0.35
0.45
0.40
0.35
Table S14: Full Evaluations on Replica [54] with ground truth depth. Both reconstruction and
rendering results improve significantly with the ground truth depth, suggesting that our method is
bounded by the quality of current day monocular depth estimation. Since we do not require any extra
training or fine-tuning of the monocular depth estimator, it is easy to plug in a better estimator once
available. Tracking performance does not change much.
Since our method does not require further training or fine-tuning for the monocular depth, it is quite
easy to just replace the current off-the-shelf monocular depth estimator with a better one.
B.4
Impact of Deformation
During runtime, we deform the 3D Gaussian map to account for adjustments to poses and depth that
have already been integrated into the existing map. An alternative to performing the deformation
is to solely rely on optimization to resolve the new map. We conduct two experiments to show the
benefit of performing the deformation, especially when it comes to rendering accuracy. In tab. S15,
we vary the number of final refinement iterations and evaluate the rendering depth L1 and PSNR on
the Replica office 0 scene. We find that utilizing online 3D Gaussian deformations yields better
rendering and depth L1 accuracy regardless of the number of iterations. In tab. S16 we conduct the
same experiment, but over a set of scenes on ScanNet. We find that on average, by enabling the
deformation, we achieve higher rendering accuracy and lower depth L1 error. The improvement is,
however, more significant when it comes to the rendering accuracy.
Nbr of Final Iterations β
Metric
0K
0.5K
1K
2K
Reconstruction
W/O Deform
W Deform
Render Depth L1 ↓
8.84
3.49
2.64
2.6
Render Depth L1 ↓
6.55
2.37
2.34
2.40
Rendering
W/O Deform
W Deform
PSNR ↑
22.86
34.30
37.66
37.86
PSNR ↑
30.50
39.87
40.59
41.20
Table S15: Gaussian Deformation Ablation on Replica [54] office 0.
Metric
0000
0054
0059
0106
0169
0181
0207
Avg.
Rendering
W/O Deform
W Deform
PSNR↑
25.15
28.39
27.77
25.25
29.41
30.38
29.30
27.95
PSNR↑
28.68
30.21
27.69
27.70
31.14
31.15
30.49
29.58
Depth
Rendering
W/O Deform
W Deform
L1↓[cm]
7.86
22.81
10.51
24.19
11.54
18.48
13.66
15.58
8.24
18.24
13.39
23.5
11.49
18.35
13.78
15.28
Table S16: Gaussian Deformation Ablation on ScanNet [10].
B.5
Final Refinement Iterations
After the final global BA step, we perform a final refinement, similar to MonoGS[38], but include
the geometric depth loss as well and do not only refine with a color loss. We ablate the influence
on the results by varying the number of iterations of the final refinement in tab. S17. We find that
the rendering accuracy increases monotonically with the number of iterations while the geometric
accuracy decreases with more than 2K iterations. We believe this to be a result of fitting to the noisy
monocular depth. We choose to use 2K iterations since this provides the best trade-off between
rendering and geometric accuracy. 2K iterations takes around 15 seconds on our benchmark hardware
which consists of an AMD Ryzen Threadripper Pro 3945WX 12-Cores with an NVIDIA GeForce
RTX 3090 Ti with 24 GB of memory.
14

<!-- page 15 -->
Nbr of Final Iterations β
Metric
2K
5K
10K
26K
Reconstruction
Render Depth L1 ↓
2.36
2.45
2.51
2.59
Accuracy ↓
2.46
2.66
2.84
3.02
Completion ↓
3.60
3.61
3.59
3.60
Comp. Rat. ↑
84.87
84.71
84.80
84.77
Rendering
Keyframes
PSNR ↑
36.77
37.80
38.41
38.95
Table S17: Final Refinement Iterations Ablation on Replica [54]. The results are averaged over
the 8 scenes.
B.6
Impact of Downsampling Factor
During mapping, the point cloud formed from unprojecting the depth input is downsampled to avoid
adding redundant Gaussians to the scene representation. We investigate the impact of using stronger
versus weaker downsampling in tab. S18 where we also compare to the sensitivity of MonoGS[38]
with respect to the same parameter. Tab. S18 shows that both systems are not very sensitive to the
model compression as a result of a larger downsampling factor θ. When both systems use the same
number of Gaussians on average (θ = 32 for MonoGS and θ = 64 for our method), we find that our
method performs significantly better in terms of depth rerendering and photometric accuracy. For all
results in the main paper, we use θ = 32.
Downsampling Factor θ
Metric
16
32
64
Reconstruction
Ours
MonoGS [38]
Render Depth L1 ↓
2.38
2.40
2.46
33.43
28.47
28.09
Rendering
Ours
MonoGS [38]
PSNR ↑
36.63
36.45
36.31
31.17
30.87
29.64
Number of
Gaussians
Ours
MonoGS [38]
1000x↓
141
102
83
97
83
73
Table S18: Downsampling Factor θ Ablation on Replica [54]. The results are averaged over the 8
scenes.
B.7
Runtime Evaluation
To be consistent with the keyframe selection hyperparameters of MonoGS [38], we report on the
same parameters as MonoGS uses by default. In practice, this means that few keyframes from the
tracking system (determined via mean optical flow thresholding) are actually filtered out and not
mapped. In tab. S19, we show that by altering the hyperparamters, we can speed up the system during
runtime, while still rendering and reconstructing the scene well. Note that we evaluate the rendering
performance on the same set of views for all runs. We benchmark the runtime on an AMD Ryzen
Threadripper Pro 3945WX 12-Cores with an NVIDIA GeForce RTX 3090 Ti with 24 GB of memory.
We note that we currently do not leverage multiprocessing to the amount possible in practice i.e.
currently we first do tracking and then mapping i.e. there is no simultaneous tracking and mapping.
This is, however, straightforward to include, which should further speed up the runtime.
kfcov, kfm
0.95, 0.04
0.90, 0.08
0.85, 0.08
0.80, 0.12
0.70, 0.16
0.60, 0.20
0.50, 0.30
Reconstruction
Render Depth L1 ↓
2.90
2.94
2.97
3.08
3.37
3.53
4.78
Accuracy ↓
1.99
1.94
2.06
2.04
2.54
3.20
6.20
Completion ↓
3.78
3.76
3.79
3.77
3.86
3.93
5.23
Comp. Rat. ↑
85.47
85.58
85.39
85.53
85.03
84.33
80.38
Rendering
PSNR ↑
32.25
31.65
31.31
30.59
30.12
29.25
27.59
Runtime
FPS ↑
1.24
1.45
1.62
2.02
2.50
3.03
3.67
Table S19: Keyframe Hyperparameter Search on Replica [54] room 0. By changing the keyframe
selection hyperparameters, we can speed up our runtime without impacting reconstruction and
rendering too much. We evaluate the rendering performance on the same set of frames for all runs. In
comparison, with the default kfcov = 0.95, kfm = 0.04, MonoGS [38] yields PSNR: 26.12 and render
depth L1: 17.38 cm.
15

<!-- page 16 -->
B.8
Additional Qualitative Reconstructions
In fig. S6 we show additional qualitative results from the Replica dataset on normal shaded meshes.
Office 4
GlORIE-SLAM∗[75]
MonoGS [38]
Splat-SLAM (Ours)
Ground Truth
Figure S6: Reconstruction Results on Replica [54]. Our method improves upon the geometric
accuracy compared to existing works, when observing the normal shaded meshes. In particular,
GlORIE-SLAM suffers from floating point artifacts. MonoGS suffers badly from a lack of proxy
depth, despite multiview optimization.
16

<!-- page 17 -->
References
[1] Azinovi´c, D., Martin-Brualla, R., Goldman, D.B., Nießner, M., Thies, J.: Neural rgb-d surface reconstruc-
tion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6290–6301 (2022)
[2] Bosse, M., Newman, P., Leonard, J., Soika, M., Feiten, W., Teller, S.: An atlas framework for scalable
mapping. In: 2003 IEEE International Conference on Robotics and Automation (Cat. No. 03CH37422).
vol. 2, pp. 1899–1906. IEEE (2003)
[3] Božiˇc, A., Palafox, P., Thies, J., Dai, A., Nießner, M.: Transformerfusion: Monocular rgb scene reconstruc-
tion using transformers. arXiv preprint arXiv:2107.02191 (2021)
[4] Cao, Y.P., Kobbelt, L., Hu, S.M.: Real-time high-accuracy three-dimensional reconstruction with consumer
rgb-d cameras. ACM Transactions on Graphics (TOG) 37(5), 1–16 (2018)
[5] Chen, J., Bautembach, D., Izadi, S.: Scalable real-time volumetric surface reconstruction. ACM Transac-
tions on Graphics (ToG) 32(4), 1–16 (2013)
[6] Cho, H.M., Jo, H., Kim, E.: Sp-slam: Surfel-point simultaneous localization and mapping. IEEE/ASME
Transactions on Mechatronics 27(5), 2568–2579 (2021)
[7] Choi, S., Zhou, Q.Y., Koltun, V.: Robust reconstruction of indoor scenes. In: IEEE Conference on Computer
Vision and Pattern Recognition. pp. 5556–5565 (2015)
[8] Chung, C.M., Tseng, Y.C., Hsu, Y.C., Shi, X.Q., Hua, Y.H., Yeh, J.F., Chen, W.C., Chen, Y.T., Hsu,
W.H.: Orbeez-slam: A real-time monocular visual slam with orb features and nerf-realized mapping. arXiv
preprint arXiv:2209.13274 (2022)
[9] Curless, B., Levoy, M.: Volumetric method for building complex models from range images. In: SIG-
GRAPH Conference on Computer Graphics. ACM (1996)
[10] Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: ScanNet: Richly-annotated 3D
reconstructions of indoor scenes. In: Conference on Computer Vision and Pattern Recognition (CVPR).
IEEE/CVF (2017). https://doi.org/10.1109/CVPR.2017.261, http://arxiv.org/abs/1702.04405
[11] Dai, A., Nießner, M., Zollhöfer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-time globally consistent
3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG) 36(4), 1
(2017)
[12] Eftekhar, A., Sax, A., Malik, J., Zamir, A.: Omnidata: A scalable pipeline for making multi-task mid-level
vision datasets from 3d scans. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision. pp. 10786–10796 (2021)
[13] Endres, F., Hess, J., Engelhard, N., Sturm, J., Cremers, D., Burgard, W.: An evaluation of the rgb-d slam
system. In: 2012 IEEE international conference on robotics and automation. pp. 1691–1696. IEEE (2012)
[14] Engel, J., Schöps, T., Cremers, D.: Lsd-slam: Large-scale direct monocular slam. In: European conference
on computer vision. pp. 834–849. Springer (2014)
[15] Fioraio, N., Taylor, J., Fitzgibbon, A., Di Stefano, L., Izadi, S.: Large-scale and drift-free surface
reconstruction using online subvolume registration. In: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition. pp. 4475–4483 (2015)
[16] Henry, P., Fox, D., Bhowmik, A., Mongia, R.: Patch volumes: Segmentation-based consistent mapping
with rgb-d cameras. In: 2013 International Conference on 3D Vision-3DV 2013. pp. 398–405. IEEE (2013)
[17] Henry, P., Krainin, M., Herbst, E., Ren, X., Fox, D.: Rgb-d mapping: Using kinect-style depth cameras for
dense 3d modeling of indoor environments. The international journal of Robotics Research 31(5), 647–663
(2012)
[18] Hu, J., Mao, M., Bao, H., Zhang, G., Cui, Z.: CP-SLAM: Collaborative neural point-based SLAM system.
In: Thirty-seventh Conference on Neural Information Processing Systems (2023), https://openreview.
net/forum?id=dFSeZm6dTC
[19] Hua, T., Bai, H., Cao, Z., Liu, M., Tao, D., Wang, L.: Hi-map: Hierarchical factorized radiance field for
high-fidelity monocular dense mapping. arXiv preprint arXiv:2401.03203 (2024)
[20] Hua, T., Bai, H., Cao, Z., Wang, L.: Fmapping: Factorized efficient neural field mapping for real-time
dense rgb slam. arXiv preprint arXiv:2306.00579 (2023)
17

<!-- page 18 -->
[21] Huang, H., Li, L., Cheng, H., Yeung, S.K.: Photo-slam: Real-time simultaneous localization and photoreal-
istic mapping for monocular, stereo, and rgb-d cameras. arXiv preprint arXiv:2311.16728 (2023)
[22] Kähler, O., Prisacariu, V.A., Murray, D.W.: Real-time large-scale dense 3d reconstruction with loop closure.
In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October
11-14, 2016, Proceedings, Part VIII 14. pp. 500–516. Springer (2016)
[23] Kähler, O., Prisacariu, V.A., Ren, C.Y., Sun, X., Torr, P.H.S., Murray, D.W.: Very high frame rate volumetric
integration of depth images on mobile devices. IEEE Trans. Vis. Comput. Graph. 21(11), 1241–1250 (2015).
https://doi.org/10.1109/TVCG.2015.2459891, https://doi.org/10.1109/TVCG.2015.2459891
[24] Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan, D., Luiten, J.: Splatam:
Splat, track and map 3d gaussians for dense rgb-d slam. arXiv preprint (2023)
[25] Keller, M., Lefloch, D., Lambers, M., Izadi, S., Weyrich, T., Kolb, A.: Real-time 3d reconstruction in
dynamic scenes using point-based fusion. In: International Conference on 3D Vision (3DV). pp. 1–8. IEEE
(2013)
[26] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field
rendering. ACM Transactions on Graphics 42(4) (2023)
[27] Kerl, C., Sturm, J., Cremers, D.: Dense visual slam for rgb-d cameras. In: 2013 IEEE/RSJ International
Conference on Intelligent Robots and Systems. pp. 2100–2106. IEEE (2013)
[28] Li, H., Gu, X., Yuan, W., Yang, L., Dong, Z., Tan, P.: Dense rgb slam with neural implicit maps. In:
Proceedings of the International Conference on Learning Representations (2023), https://openreview.
net/forum?id=QUK1ExlbbA
[29] Li, K., Tang, Y., Prisacariu, V.A., Torr, P.H.: Bnv-fusion: Dense 3d reconstruction using bi-level neural
volume fusion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6166–6175
(2022)
[30] Liso, L., Sandström, E., Yugay, V., Van Gool, L., Oswald, M.R.: Loopy-slam: Dense neural slam with loop
closures. arXiv preprint arXiv:2402.09944 (2024)
[31] Liu, L., Gu, J., Zaw Lin, K., Chua, T.S., Theobalt, C.: Neural sparse voxel fields. Advances in Neural
Information Processing Systems 33, 15651–15663 (2020)
[32] Lorensen, W.E., Cline, H.E.: Marching cubes: A high resolution 3d surface construction algorithm. ACM
siggraph computer graphics 21(4), 163–169 (1987)
[33] Mahdi Johari, M., Carta, C., Fleuret, F.: Eslam: Efficient dense slam system based on hybrid representation
of signed distance fields. arXiv e-prints pp. arXiv–2211 (2022)
[34] Maier, R., Schaller, R., Cremers, D.: Efficient online surface correction for real-time large-scale 3d
reconstruction. arxiv 2017. arXiv preprint arXiv:1709.03763 (2017)
[35] Maier, R., Sturm, J., Cremers, D.: Submap-based bundle adjustment for 3d reconstruction from rgb-d data.
In: Pattern Recognition: 36th German Conference, GCPR 2014, Münster, Germany, September 2-5, 2014,
Proceedings 36. pp. 54–65. Springer (2014)
[36] Mao, Y., Yu, X., Wang, K., Wang, Y., Xiong, R., Liao, Y.: Ngel-slam: Neural implicit representation-based
global consistent low-latency slam system. arXiv preprint arXiv:2311.09525 (2023)
[37] Marniok, N., Johannsen, O., Goldluecke, B.: An efficient octree design for local variational range image
fusion. In: German Conference on Pattern Recognition (GCPR). pp. 401–412. Springer (2017)
[38] Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.:
Gaussian splatting slam. arXiv preprint
arXiv:2312.06741 (2023)
[39] Matsuki, H., Sucar, E., Laidow, T., Wada, K., Scona, R., Davison, A.J.: imode: Real-time incremental
monocular dense mapping using neural field. In: 2023 IEEE International Conference on Robotics and
Automation (ICRA). pp. 4171–4177. IEEE (2023)
[40] Matsuki, H., Tateno, K., Niemeyer, M., Tombari, F.: Newton: Neural view-centric mapping for on-the-fly
large-scale slam. arXiv preprint arXiv:2303.13654 (2023)
[41] Naumann, J., Xu, B., Leutenegger, S., Zuo, X.: Nerf-vo: Real-time sparse visual odometry with neural
radiance fields. arXiv preprint arXiv:2312.13471 (2023)
18

<!-- page 19 -->
[42] Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J., Kohli, P., Shotton, J.,
Hodges, S., Fitzgibbon, A.W.: Kinectfusion: Real-time dense surface mapping and tracking. In: ISMAR.
vol. 11, pp. 127–136 (2011)
[43] Nießner, M., Zollhöfer, M., Izadi, S., Stamminger, M.: Real-time 3d reconstruction at scale using voxel
hashing. ACM Transactions on Graphics (TOG) 32 (11 2013). https://doi.org/10.1145/2508363.2508374
[44] Oleynikova, H., Taylor, Z., Fehr, M., Siegwart, R., Nieto, J.I.: Voxblox: Incremental 3d euclidean signed
distance fields for on-board MAV planning. In: 2017 IEEE/RSJ International Conference on Intelligent
Robots and Systems, IROS 2017, Vancouver, BC, Canada, September 24-28, 2017. pp. 1366–1373.
IEEE (2017). https://doi.org/10.1109/IROS.2017.8202315, https://doi.org/10.1109/IROS.2017.
8202315
[45] Ortiz, J., Clegg, A., Dong, J., Sucar, E., Novotny, D., Zollhoefer, M., Mukadam, M.: isdf: Real-time neural
signed distance fields for robot perception. arXiv preprint arXiv:2204.02296 (2022)
[46] Peng, C., Xu, C., Wang, Y., Ding, M., Yang, H., Tomizuka, M., Keutzer, K., Pavone, M., Zhan, W.: Q-slam:
Quadric representations for monocular slam. arXiv preprint arXiv:2403.08125 (2024)
[47] Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., Geiger, A.: Convolutional Occupancy Networks. In:
European Conference Computer Vision (ECCV). CVF (2020), https://www.microsoft.com/en-us/
research/publication/convolutional-occupancy-networks/
[48] Reijgwart, V., Millane, A., Oleynikova, H., Siegwart, R., Cadena, C., Nieto, J.: Voxgraph: Globally
consistent, volumetric mapping using signed distance function submaps. IEEE Robotics and Automation
Letters 5(1), 227–234 (2019)
[49] Rosinol, A., Leonard, J.J., Carlone, L.: NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural
Radiance Fields. arXiv (2022), http://arxiv.org/abs/2210.13641
[50] Sandström, E., Li, Y., Van Gool, L., Oswald, M.R.: Point-slam: Dense neural point cloud-based slam. In:
International Conference on Computer Vision (ICCV). IEEE/CVF (2023)
[51] Sandström, E., Ta, K., Gool, L.V., Oswald, M.R.: Uncle-SLAM: Uncertainty learning for dense neural
SLAM. In: International Conference on Computer Vision Workshops (ICCVW) (2023)
[52] Schops, T., Sattler, T., Pollefeys, M.: BAD SLAM: Bundle adjusted direct RGB-D SLAM. In: CVF/IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) (2019)
[53] Steinbrucker, F., Kerl, C., Cremers, D.: Large-scale multi-resolution surface reconstruction from rgb-d
sequences. In: IEEE International Conference on Computer Vision. pp. 3264–3271 (2013)
[54] Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-Artal, R., Ren, C., Verma,
S., et al.: The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)
[55] Stückler, J., Behnke, S.: Multi-resolution surfel maps for efficient dense 3d modeling and tracking. Journal
of Visual Communication and Image Representation 25(1), 137–147 (2014)
[56] Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for the evaluation of RGB-D
SLAM systems. In: International Conference on Intelligent Robots and Systems (IROS). IEEE/RSJ (2012).
https://doi.org/10.1109/IROS.2012.6385773, http://ieeexplore.ieee.org/document/6385773/
[57] Sucar,
E.,
Liu,
S.,
Ortiz,
J.,
Davison,
A.J.:
iMAP:
Implicit
Mapping
and
Position-
ing in Real-Time. In:
International Conference on Computer Vision (ICCV). IEEE/CVF
(2021). https://doi.org/10.1109/ICCV48922.2021.00617, https://ieeexplore.ieee.org/document/
9710431/
[58] Sun, J., Xie, Y., Chen, L., Zhou, X., Bao, H.: Neuralrecon: Real-time coherent 3d reconstruction from
monocular video. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15598–
15607 (2021)
[59] Tang, Y., Zhang, J., Yu, Z., Wang, H., Xu, K.: Mips-fusion: Multi-implicit-submaps for scalable and robust
online neural rgb-d reconstruction. arXiv preprint arXiv:2308.08741 (2023)
[60] Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV
2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419.
Springer (2020)
19

<!-- page 20 -->
[61] Teed, Z., Deng, J.: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in
neural information processing systems 34, 16558–16569 (2021)
[62] Tosi, F., Zhang, Y., Gong, Z., Sandström, E., Mattoccia, S., Oswald, M.R., Poggi, M.: How nerfs and 3d
gaussian splatting are reshaping slam: a survey (2024)
[63] Wang, H., Wang, J., Liang, W.: Online reconstruction of indoor scenes from rgb-d streams. In: Proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 3271–3279 (2016)
[64] Wang, H., Wang, J., Agapito, L.: Co-slam: Joint coordinate and sparse parametric encodings for neural
real-time slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). pp. 13293–13302 (June 2023)
[65] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)
[66] Weder, S., Schonberger, J., Pollefeys, M., Oswald, M.R.: Routedfusion: Learning real-time depth map
fusion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4887–4897 (2020)
[67] Weder, S., Schonberger, J.L., Pollefeys, M., Oswald, M.R.: Neuralfusion: Online depth fusion in latent
space. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3162–3172 (2021)
[68] Whelan, T., Leutenegger, S., Salas-Moreno, R., Glocker, B., Davison, A.: Elasticfusion: Dense slam
without a pose graph. In: Robotics: Science and Systems (RSS) (2015)
[69] Yan, C., Qu, D., Wang, D., Xu, D., Wang, Z., Zhao, B., Li, X.: Gs-slam: Dense visual slam with 3d
gaussian splatting. arXiv preprint arXiv:2311.11700 (2023)
[70] Yan, Z., Ye, M., Ren, L.: Dense visual slam with probabilistic surfel map. IEEE transactions on visualization
and computer graphics 23(11), 2389–2398 (2017)
[71] Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth anything: Unleashing the power of
large-scale unlabeled data. arXiv preprint arXiv:2401.10891 (2024)
[72] Yang, X., Li, H., Zhai, H., Ming, Y., Liu, Y., Zhang, G.: Vox-fusion: Dense tracking and mapping with
voxel-based neural implicit representation. In: IEEE International Symposium on Mixed and Augmented
Reality (ISMAR). pp. 499–507. IEEE (2022)
[73] Yang, X., Ming, Y., Cui, Z., Calway, A.: Fd-slam: 3-d reconstruction using features and dense matching.
In: 2022 International Conference on Robotics and Automation (ICRA). pp. 8040–8046. IEEE (2022)
[74] Yugay, V., Li, Y., Gevers, T., Oswald, M.R.: Gaussian-slam: Photo-realistic dense slam with gaussian
splatting (2023)
[75] Zhang, G., Sandström, E., Zhang, Y., Patel, M., Van Gool, L., Oswald, M.R.: Glorie-slam: Globally
optimized rgb-only implicit encoding point cloud slam. arXiv preprint arXiv:2403.19549 (2024)
[76] Zhang, H., Chen, G., Wang, Z., Wang, Z., Sun, L.: Dense 3d mapping for indoor environment based
on feature-point slam method. In: 2020 the 4th International Conference on Innovation in Artificial
Intelligence. pp. 42–46 (2020)
[77] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features
as a perceptual metric. In: IEEE conference on computer vision and pattern recognition. pp. 586–595
(2018)
[78] Zhang, W., Sun, T., Wang, S., Cheng, Q., Haala, N.: Hi-slam: Monocular real-time dense mapping with
hybrid implicit fields. IEEE Robotics and Automation Letters (2023)
[79] Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: Go-slam: Global optimization for consistent 3d instant
reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp.
3727–3737 (2023)
[80] Zhou, H., Guo, Z., Liu, S., Zhang, L., Wang, Q., Ren, Y., Li, M.: Mod-slam: Monocular dense mapping
for unbounded 3d scene reconstruction (2024)
[81] Zhu, Z., Peng, S., Larsson, V., Cui, Z., Oswald, M.R., Geiger, A., Pollefeys, M.: Nicer-slam: Neural
implicit scene encoding for rgb slam. arXiv preprint arXiv:2302.03594 (2023)
20

<!-- page 21 -->
[82] Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys, M.: Nice-slam:
Neural implicit scalable encoding for slam. In: IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 12786–12796 (2022)
[83] Zou, Z.X., Huang, S.S., Cao, Y.P., Mu, T.J., Shan, Y., Fu, H.: Mononeuralfusion: Online monocular neural
3d reconstruction with geometric priors. arXiv preprint arXiv:2209.15153 (2022)
[84] Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Surface splatting. In: Proceedings of the 28th annual
conference on Computer graphics and interactive techniques. pp. 371–378 (2001)
21
