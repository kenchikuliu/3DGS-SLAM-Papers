<!-- page 1 -->
This paper has been accepted for publication at the IEEE International Conference on Robotics and Automation (ICRA), 2026 © IEEE
MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian
Splatting for High-Fidelity Mapping
Zhihao Cao1, Hanyu Wu2, Li Wa Tang2, Zizhou Luo3,
Wei Zhang4, Marc Pollefeys5,6, Zihan Zhu5,∗, and Martin R. Oswald7
Tracking and Mapping
Multi Camera Stream
Multi Camera Reconstruction
Cam0
Front
Cam1
Left
Cam2
Right
. . .
. . .
. . .
Rendering
1
2
3
Rendering Image
Rendering Depth
1
2
3
Multi Camera Rendering
Rendering Image
Rendering Depth
Rendering Depth
Rendering Image
Fig. 1: MCGS-SLAM synchronizes RGB inputs from the front, left, and right cameras of the multi-camera rig in the
Waymo dataset and fuses them into a unified 3D Gaussian Splatting map. The system performs real-time tracking and
mapping, enabling high-fidelity reconstruction of both color and depth views from each individual camera. Through joint
multi-camera optimization, MCGS-SLAM ensures accurate pose and geometry alignment, while supporting comprehensive
multi-view rendering for photorealistic visualization. https://zhihao-ethz.github.io/mcgs-slam/
Abstract— Recent progress in dense SLAM has primarily
targeted monocular setups, often at the expense of robustness
and geometric coverage. We present MCGS-SLAM, the first
purely RGB-based multi-camera SLAM system built on 3D
Gaussian Splatting (3DGS). Unlike prior methods relying on
sparse maps or inertial data, MCGS-SLAM fuses dense RGB
inputs from multiple viewpoints into a unified, continuously
optimized Gaussian map. A multi-camera bundle adjustment
(MCBA) jointly refines poses and depths via dense photometric
and geometric residuals, while a scale consistency module
enforces metric alignment across views using low-rank priors.
The system supports RGB input and maintains real-time
performance at large scale. Experiments on synthetic and real-
world datasets show that MCGS-SLAM consistently yields
accurate trajectories and photorealistic reconstructions, usually
outperforming monocular baselines. Notably, the wide field
of view from multi-camera input enables reconstruction of
side-view regions that monocular setups miss, critical for safe
autonomous operation. These results highlight the promise
of multi-camera Gaussian Splatting SLAM for high-fidelity
∗Zihan Zhu is the Project Lead of this work.
1Zhihao Cao is with the Department of Mathematics, ETH Zurich,
Switzerland. (e-mail: zhicao@student.ethz.ch)
2Hanyu Wu and Li Wa Tang are with the Department of Me-
chanical and Process Engineering, ETH Zurich, Switzerland. (e-mail:
hanywu@student.ethz.ch; litang1@student.ethz.ch)
3Zizhou Luo is with the Department of Informatics, University of Zurich,
Switzerland. (e-mail: zizhou.luo@uzh.ch)
4Wei Zhang is with the Institute for Photogrammetry, University of
Stuttgart, Germany (e-mail: wei.zhang@ifp.uni-stuttgart.de)
5Marc Pollefeys and Zihan Zhu are with Computer Vision and Ge-
ometry Group, ETH Zurich, 8092 Zurich, Switzerland. (e-mail: zi-
han.zhu@inf.ethz.ch; marc.pollefeys@inf.ethz.ch)
6Marc Pollefeys is also with Microsoft Spatial AI Lab, 8038 Zurich,
Switzerland (e-mail: mapoll@microsoft.com)
7Martin R. Oswald is with Computer Vision Research Group, University
of Amsterdam, Netherlands (e-mail: m.r.oswald@uva.nl)
mapping in robotics and autonomous driving.
I. INTRODUCTION
Simultaneous Localization and Mapping (SLAM) remains
a foundational component in robotic navigation and 3D scene
reconstruction. Early monocular SLAM systems, such as
ORB-SLAM [1], [2], LSD-SLAM [3], and DSO [4], achieve
real-time camera tracking by minimizing sparse geometric
or photometric residuals. However, their reliance on a single
narrow field-of-view (FoV) camera renders them suscepti-
ble to scale drift, motion blur, and occlusions. Learning-
augmented approaches, including DROID-SLAM [5] and
MAC-VO [6], alleviate some of these issues, yet the core
limitation of monocular viewpoint remains a fundamen-
tal bottleneck for scene completeness and depth accuracy.
A natural solution is to employ overlapping multi-camera
systems. Early visual-inertial odometry pipelines improved
robustness through fisheye clusters [7]. More recently, Kuo
et al. [8] proposed a generalization of visual-inertial bun-
dle adjustment (BA) to wide-baseline multi-camera sys-
tems through adaptive initialization and keyframe selection.
BAMF-SLAM [9] introduced a scalable BA formulation for
general camera networks, achieving state-of-the-art odometry
accuracy. Nevertheless, these systems typically yield only
sparse landmarks, and some systems heavily rely on inertial
sensors, relegating high-fidelity geometry and photorealistic
rendering to costly offline post-processing.
In parallel, dense scene representations have made re-
markable strides, though predominantly in monocular set-
tings, thus underutilizing the potential of multi-camera
platforms. Traditional map structures such as surfels and
arXiv:2509.14191v3  [cs.RO]  9 Mar 2026

<!-- page 2 -->
TSDF volumes [10], [11] have evolved towards neural
implicit fields. NeRF-based methods [12], [13] enable im-
pressive photorealism, while SLAM variants like NICER-
SLAM [14] and GLORIE-SLAM [15] integrate neural fields
into SLAM pipelines for high-quality novel view synthesis.
However, these methods remain computationally expensive
and lack explicit geometric control. In contrast, 3D Gaus-
sian Splatting (3DGS) [16] offers an efficient alternative
that combines explicit geometry, differentiable rasteriza-
tion, and fast optimization. Recent extensions, including
MonoGS [17] for dense tracking, Loop-Splat [18] for loop
closure, Splat-SLAM [19] for global joint optimization,
and HI-SLAM2 [20] for monocular refinement, demonstrate
strong results. Still, they inherit the limitations of monocular
input: limited FoV, scale ambiguity, and degraded perfor-
mance in low-texture or occluded regions. These drawbacks
highlight the unmet potential of fusing multi-view observa-
tions with the efficiency of Gaussian splatting. While multi-
agent extensions [21] also support multiple cameras, they
cannot benefit from calibrated rigs.
Leveraging a calibrated multi-camera rig with k spa-
tially overlapping views offers rich observational redundancy
but presents challenges in fusing dense RGB streams into
a unified Gaussian representation, specifically, maintaining
inter-camera scale consistency, achieving drift-free tracking,
and enabling efficient online mapping with large numbers
of Gaussians. We propose MCGS-SLAM, to the best of
our knowledge, the first fully vision-based multi-camera
SLAM system built upon 3D Gaussian Splatting with purely
RGB input. MCGS-SLAM jointly estimates accurate camera
trajectories and high-fidelity 3D reconstructions by fusing
raw RGB inputs into a globally consistent Gaussian map.
Our framework also supports RGB-D inputs, but this paper
focuses on the RGB-only setting. Central to our framework
is a Multi-Camera Bundle Adjustment (MCBA) module that
jointly optimizes pose and dense depth across views via
photometric and geometric consistency. To ensure metric-
scale alignment, we introduce a complementary module that
leverages low-rank geometric priors from a learned network.
These components enable scalable Gaussian optimization
and pruning across large anisotropic fields, yielding recon-
structions with sharp geometry and photorealistic textures
under wide baselines. Our contributions are as follows.
• An efficient multi-camera Gaussian SLAM system sup-
porting RGB inputs, with joint optimization over camera
poses and 3DGS maps.
• A
unified
multi-camera
framework
that
combines
Multi-Camera Bundle Adjustment (MCBA) and Joint
Depth–Scale
Alignment
(JDSA),
jointly
optimizing
photometric consistency, geometric priors, and global
scale alignment across views.
• A practical and scalable implementation that generalizes
across real-world and synthetic benchmarks, demonstrat-
ing strong performance in both geometry and appearance.
Through these innovations, MCGS-SLAM bridges the
gap between wide-baseline multi-camera tracking and dense
Fig. 2: The sensor suite integrates multiple wide-angle
RGB cameras centrally mounted on the vehicle’s roof in
Waymo Open Dataset [22], whose fan-shaped fields of view
collectively provide full 240◦coverage. This configuration
enables high-density observations for multi-camera SLAM
and autonomous driving algorithms.
3D Gaussian mapping, laying the groundwork for next-
generation robotic perception, digital twin construction, and
autonomous systems at scale.
II. PRELIMINARIES
This section introduces the core concepts underpinning
our multi-camera Gaussian Splatting SLAM framework. We
first review the dense SLAM formulation and the multi-
camera setting, followed by an overview of Recurrent Field
Transforms and learning-based SLAM front-ends such as
DROID-SLAM and BAMF-SLAM. Finally, we present the
3D Gaussian Splatting representation, which serves as the
foundational structure of our mapping system.
A. Problem Setting: Dense Multi-Camera SLAM
1) Dense SLAM: Given a temporally ordered stream of
color (or color–depth) images It captured at time t by a
calibrated rig with k camera views, dense SLAM jointly
estimates the metric camera trajectory T = {Tt}L
t=0 with
Tt ∈SE(3) and a continuous scene map M by minimizing
photometric and geometric residuals across all pixels. To
ensure both temporal and spatial consistency, we define a
set of frame pairs (t, t′) ∈E, where t′ denotes either
a temporally adjacent frame or one selected via keyframe
heuristics. The overall objective is formulated as
arg min
T,M
X
(t,t′)∈E
It −It′ ◦Π
 Ttt′ Π−1(pt, dt)

ρ,
(1)
where Π and Π−1 denote the projection and back-projection
functions, dt is the depth at pixel pt, and ρ(·) is the robust ℓ2
penalty function. The transformation Ttt′ denotes the relative
camera pose from frame t to t′. Unlike pipelines based on
sparse features, optimization in (1) is performed at full image
resolution, enabling recovery of dense scene geometry.
2) Multi-Camera Setting: The calibrated multi-camera
system is defined by fixed extrinsic transformations TB
C ∈
SE(3), which map points from each individual camera frame
C to a shared body frame B. As illustrated in Fig. 2, modern

<!-- page 3 -->
RGB
Normal
Multi-Camera
Keyframe Buffer
Init Gaussians
Densification
and Pruning
Differentiable
Surface
Rasterization
3DGS Refinement
Pose Refinement
Offine Global Refinement
Global BA
Multi-Camera's
Pose and Depth
Keyframe Estimation T, D
Global Update ΔT,ΔD
Map Update Δθ
Multi-Camera Refined
Keyframe Attributes I, D, T
Multi-Camera Gaussian Mapping
Joint Depth and Scale
Alignment
(JDSA)
Depth
Multi-Camera Online Tracking
...
...
Keyframe Selection
Depth & Normal
Estimation
Cam 0
Keyframe Selection
Cam k
Synchronous
Multi-Camera Bundle
Adjustment
(MCBA)
Depth & Normal
Estimation
...
...
Metric3Dv2
Metric3Dv2
Temporal Pairs
Cross-view Pairs
...
Fig. 3: Our method performs real-time SLAM by fusing synchronized inputs from a multi-camera rig into a unified 3D
Gaussian map. It first selects keyframes and estimates depth and normal maps for each camera, then jointly optimizes
poses and depths via multi-camera bundle adjustment and scale-consistent depth alignment. Refined keyframes are fused
into a dense Gaussian map using differentiable rasterization, interleaved with densification and pruning. An optional offline
stage further refines camera trajectories and map quality. The system supports RGB inputs, enabling accurate tracking and
photorealistic reconstruction.
automotive datasets such as the Waymo Open Dataset pro-
vide time-synchronized, wide-baseline camera clusters com-
posed of multiple global-shutter RGB sensors with accurate
intrinsic and extrinsic calibrations. These offer a compelling
testbed for SLAM systems, as they introduce strong parallax,
large fields of view, and a complex environment.
B. Recurrent Field Transforms and Learning-based SLAM
Recurrent Field Transforms (RFT) extend the RAFT fam-
ily of recurrent optical flow networks to iteratively refine
dense correspondences between two views. Given a current
reprojection ˆpij, RFT predicts a flow increment δij and
an associated per-pixel confidence weight wij. The refined
target location is defined as ˜pij = ˆpij + δij and is used
to minimize the reprojection error. During optimization,
the resulting weighted residual is inserted into the normal
equations of bundle adjustment (BA) [5] as
rij =
˜pij −Π
 TijΠ−1(pi, di)
2
wij
(2)
where Π and Π−1 denote projection and back-projection,
respectively. Equation (2) forms the foundation of the dense,
differentiable front-end in DROID-SLAM. To tightly cou-
ple correspondence estimation and geometric optimization,
DROID-SLAM augments classical photometric BA with
RFT, treating optical flow as a latent variable updated via
a gated recurrent unit (GRU). This formulation enables
joint, real-time optimization of camera poses, per-frame
depths, and inter-frame flow, achieving state-of-the-art ac-
curacy in monocular visual odometry. BAMF-SLAM builds
upon DROID-SLAM by generalizing it to wide-baseline,
multi-fisheye camera systems, with optional visual–inertial
integration. It fuses dense intra- and inter-view residuals with
inertial pre-integration factors within a unified optimization
graph. The system further leverages the large field of view
for memory-efficient loop closures via semi-pose-graph BA.
C. 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) represents the scene as a
set {(µi, Σi, αi, ci)}M
i=1 of M anisotropic Gaussians, where
each Gaussian is defined by its mean µi ∈R3, covariance
Σi ∈R3×3, opacity αi ∈R, and RGB color ci ∈R3. Under
a camera pose Ti ∈SE(3), a 3D Gaussian is projected into
an elliptical footprint on the image plane as
µ′
i = π(Tiµi),
Σ′
i = J R ΣiR⊤J⊤,
(3)
where R is the rotational component of T and J denotes the
Jacobian of the perspective projection [16]. The resulting
splats are composited in a front-to-back order using α-
blending to produce color and depth images as
ˆC(p) =
X
i∈Np
ciαi
Y
j<i
 1 −αj

,
ˆD(p) =
X
i∈Np
diαi
Y
j<i
 1 −αj

.
(4)
where Np denotes the set of Gaussians intersecting the
ray. Here, ci and di are the color and depth of the i-th
Gaussian, respectively, and αi represents its contribution to
pixel translucency, obtained from the Gaussian’s opacity at
the ray Gaussian intersection. The projection and blending
operations in Equations (3) and (4) are fully differentiable,
allowing gradients to be backpropagated with respect to both
the camera pose T and all Gaussian parameters.
III. METHOD
This section presents MCGS-SLAM, a dense multi-camera
SLAM pipeline that integrates learning-based tracking with a
differentiable 3D Gaussian map representation. An overview
of the system architecture is shown in Figure 3. The pipeline
operates in two stages. In the online tracking stage, the
system estimates the camera rig’s trajectory in real time,

<!-- page 4 -->
resolves the scale ambiguity associated with monocular pri-
ors, and performs MCBA to jointly optimize per-view depths
and poses. Refined keyframes are incrementally fused into a
global 3DGS map. In the optional offline refinement stage,
all rig poses and Gaussian parameters are jointly optimized to
enforce global consistency and further improve the geometric
and photometric fidelity of the reconstruction.
A. Online Multi-Camera Tracking
1) Key-Frame Selection, Depth and Normal Estimation:
For each synchronized RGB frame, we compute the aver-
age Recurrent Field Transform (RFT) flow relative to the
current reference keyframe. If the flow magnitude exceeds a
threshold, the frame is promoted to a multi-camera keyframe,
Kt := {It, d+
t, n+
t}, where d+
t and n+
t denote the per-pixel
depth and surface normal maps. These are obtained from
Metric3Dv2 [23], which also allows our system to support
RGB-D input. Keyframes are stored in a shared buffer ac-
cessible to both tracking and mapping threads. Although the
depths from Metric3Dv2 are metric, they often suffer from
noise, bias, and inconsistent scaling across viewpoints, lead-
ing to misaligned poses and depths. Our proposed MCBA
module corrects this by jointly refining poses and depths,
enforcing geometric consistency and scale alignment across
the rig.
2) Joint Depth and Scale Alignment (JDSA): Depth maps
predicted for each RGB camera are only defined up to an
unknown, spatially varying scale. To compensate for this
ambiguity, [24] introduce a learnable m × n scale grid st
for each key-frame. This grid is bilinearly interpolated to
yield a per-pixel scale factor Bt(p, st), which relates the
predicted and optimized depths as ˜dt(p) = d+
t(p)·Bt(p, st),
where d+
t denotes the monocular depth map and ˜dt the
rescaled depth used during optimization. However, directly
coupling the scale factors with bundle adjustment, by jointly
optimizing camera poses, depths, and scale coefficients, has
been shown to cause unstable convergence and scale drift
[20]. To mitigate this, we adopt the Joint Depth and Scale
Alignment (JDSA) formulation proposed in [20], which
introduces a dedicated loss function as
arg min
s,d
X
(i,j)∈E
˜pij −Π
 TijΠ−1(pi, di)
2
ωij +
X
i∈V
˜di · Bi(pi, si) −di

2
,
(5)
where the node set V consists of keyframes, each associated
with a pose T ∈SE(3) and an estimated depth map d. The
edge set E connects keyframes that exhibit sufficient overlap,
as determined by their optical flow correspondences. The
first term enforces multi-view photometric and geometric
consistency, and the second term aligns scaled depths to the
optimized depths. By interleaving JDSA with local multi-
camera bundle adjustment, our system achieves stable scale
calibration and improved depth initialization.
3) Multi-Camera Bundle Adjustment (MCBA): To jointly
optimize camera poses and dense depth maps, we minimize
a weighted photometric reprojection loss over both temporal
and cross-view image pairs. Specifically, for each valid
correspondence between a source view (i, Ci) and a target
view (j, Cj), we define the following objective:
arg min
T,d
X
(i,j)∈E
˜pij −ΠCj

ˆTij · Π−1
Ci (pi, di)

2
wij ,
(6)
where T ∈SE(3) denotes the body pose, and di is the
estimated inverse depth parametrization in view (i, Ci). The
function Π−1
Ci (·) back-projects the pixel using the intrinsics of
camera Ci, while ΠCj(·) reprojects it into the target view.
The norm ∥· ∥wij incorporates a confidence wij per pixel
predicted by the RFT module in multi-camera settings. The
transformation ˆTij maps 3D points from the source to the
target camera frame, and is defined differently based on the
type of correspondence as
• Temporal pairs (i.e., same camera across time):
ˆTij = TB
C T−1
j
Ti TB
C
−1,
(7)
where TB
C is the known extrinsic between the camera
frame C and the body frame B.
• Cross-view pairs (i.e., different cameras at the same
timestamp):
ˆTij = TCiCj,
(8)
which is the pre-calibrated extrinsic between camera Ci
and camera Cj.
This unified formulation allows for simultaneous optimiza-
tion over both time-varying motion and multi-camera geom-
etry in a single bundle adjustment framework. The resulting
non-linear least-squares problem is solved via a damped
Gauss–Newton method, yielding a block-structured linear
system of the form as
 B
E
E⊤
C
 ∆ξ
∆d

=
v
w

,
(9)
where ∆ξ ∈R6 represents the pose update in the Lie algebra
of SE(3), applied via ∆T = exp(∆ξ) as in [5]. Matrices B,
C, and E correspond to the Hessian blocks with respect to
pose, depth, and their coupling terms, while v and w are
the respective residual gradients. Since the pose block B
is typically much smaller than the depth block C, we solve
the system efficiently using the Schur complement. The pose
update is obtained via
∆ξ =

B −EC−1E⊤−1  v −EC−1w

,
∆d = C−1  w −E⊤∆ξ

.
(10)
In the implementation, the depth Hessian C is diagonal and
thus admits a cheap closed-form inverse C−1 = 1/C.
B. Multi-Camera Gaussian Mapping
1) Gaussian Initialization and Maintenance: After each
MCBA and JDSA update, we back-project the depth map
of the latest keyframe Kt into 3D space to initialize new
Gaussian primitives. For each valid pixel p, a Gaussian gi
is created with mean µi ∈R3 corresponding to the back-
projected 3D point, and covariance Σi ∈R3×3 estimated

<!-- page 5 -->
from the average distance to its three nearest neighbors. To
keep the map compact yet expressive, the system alternates
every few iterations between two complementary operations:
(1) densification, which adds Gaussians at previously un-
observed pixels to grow underrepresented regions; and (2)
pruning, which removes nearly transparent Gaussians to
reduce redundancy and computational overhead.
2) Differentiable Rasterization and Losses: We follow
[20] and avoid depth bias by analytically intersecting each
viewing ray with the ellipsoidal surface defined by the
anisotropic Gaussian, yielding a more accurate intersection
depth. Each Gaussian is jointly optimized through a multi-
term loss function per keyframe Kt as
L = λc
 ˆCt −It

2 + λd
 ˆDt −dt

2
+ λn
1 −
D
ˆnt, npri
t
E
2 + λs ∥st −¯s∥2 ,
(11)
where
ˆCt and
ˆDt denote the rendered color and depth
from the viewpoint of the MCBA-refined camera pose, dt
is the depth refined via MCBA and JDSA, ˆnt and n+
t
represent the rendered and estimated surface normals by
Metric3Dv2, and st and ¯s denote the current and average
scale of the corresponding Gaussian ellipsoids, respectively.
Optimization is performed using the optimizer for a fixed
number of iterations per keyframe.
3) Pose-Consistent Gaussian Updates: When the pose of
a keyframe Kt is updated via MCBA or loop closure by a
relative transform ∆Tt ∈SE(3), we propagate the update to
all Gaussians gi anchored in that frame as
µi ←∆Tt · µi,
Σi ←R(∆Tt) · Σi · R⊤(∆Tt),
(12)
where R(·) extracts the rotational component of ∆Tt. If
scale changes are introduced via scale updates, we addition-
ally rescale the ellipsoids as
si ←st · si.
(13)
This deformation ensures consistency of the 3D map without
requiring re-initialization or re-rendering, enabling efficient
and flexible map maintenance.
C. Offline Global Refinement
after the real-time pipeline finishes, we apply two global
refinement stages to enhance the consistency and overall
quality of the reconstruction.
1) Global Bundle Adjustment: All keyframes that in-
cludes synthetically inserted views, are jointly optimized via
global bundle adjustment. The optimization minimizes both
photometric and geometric residuals across all overlapping
image pairs, refining the camera poses and improving the
consistency of the reconstructed scene geometry.
2) Joint Pose and 3DGS Map Refinement: In the final
stage, we jointly optimize all 3D Gaussian parameters Θ :=
{µ, Σ, α, c}, along with per-frame exposure matrices At and
camera poses Tt. Gradients are backpropagated through the
differentiable rasterization pipeline to minimize a weighted
combination of photometric, depth, normal, and scale regu-
larization losses. This optimization stage effectively reduces
global drift and improves both the geometric accuracy and
photometric consistency of the final reconstruction.
IV. RESULTS
A. Datasets, Metrics, and Protocol
We evaluated MCGS-SLAM on both real-world and syn-
thetic datasets. For real-world experiments, we employ the
Waymo Open Dataset [22], which provides urban driving
sequences with five synchronized wide-angle roof cameras.
We select three of them, as this already ensures a sufficiently
wide front-facing field of view while keeping GPU mem-
ory usage manageable. We further use the Oxford Spires
Dataset[25], which contains large-scale Oxford landmarks
recorded by three fisheye cameras with LiDAR/IMU ground
truth. For synthetic evaluation, we adopt the AirSim [26]
simulator with three photorealistic UE5 environments, cap-
tured using a four-camera aircraft rig in the simulation
setting. Reconstruction quality is quantified using standard
image-based metrics: PSNR (↑), SSIM (↑) and LPIPS (↓) -
computed over all keyframes after mapping. The trajectory
accuracy is measured by the absolute trajectory error (ATE,
meters; ↓) after Sim(3)-alignment with the ground truth. For
better readability, the result tables highlight the top three
results with first , second , and third .
B. Rendering Results Study
Tables I and III present quantitative appearance metrics,
while Figures 4 and 5 show qualitative reconstruction results
in different Waymo and AirSim environments. On the four
held-out urban sequences from the Waymo dataset, MCGS-
SLAM consistently ranks among the top two performers,
demonstrating strong photometric fidelity and perceptual
quality. In contrast, competing methods report inferior LPIPS
values and fail to reconstruct critical side-view structures,
such as alley facades, that are clearly recovered by MCGS-
SLAM (see Fig.4). This advantage stems from the wide
field of view (FoV) provided by the multi-camera rig
(Fig. 2), which enables MCGS-SLAM to resolve occluded
elements such as building corners and overhead traffic lights.
Furthermore, the resulting 3D maps exhibit substantially
fewer floating artifacts, highlighting the effectiveness of
cross-view depth consistency enforced by our MCBA and
JDSA modules. Although GLORIE-SLAM and DROID-
Splat occasionally reconstruct sharper specular surfaces, their
limited spatial coverage leads to incomplete scene geometry.
Overall, MCGS-SLAM achieves a better balance between
reconstruction fidelity and spatial completeness, making it
particularly well suited for complex urban environments.
Similar trends are observed in the AirSim benchmark
(Table III and Fig. 5), where MCGS-SLAM consistently
ranks among the top two methods across all environments.
In the low-parallax Garden scene, it surpasses all single-
camera baselines by approximately 4 dB PSNR, highlighting
the benefit of leveraging complementary viewpoints from
a wide-baseline rig. In the Factory scene, although Photo-
SLAM achieves the highest PSNR and SSIM, MCGS-SLAM

<!-- page 6 -->
w/ L Camera
w/ R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/ L Camera
w/ R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/ L Camera
w/ R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
100613
158686
106762
MonoGS [17]
HI-SLAM2 [20]
MCGS-SLAM (Ours)
DROID-Splat [27]
220
134763
w/ L Camera
w/ R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
w/o L Camera
w/o R Camera
w/ F Camera
Fig. 4: Qualitative results on the Waymo dataset [22] (Real-World Dataset). MCGS-SLAM reconstructs urban scenes with
higher fidelity and completeness, preserving structural details and textures that are often missed by monocular methods.
Method
Metric
100613
132384
134763
152706
158686
153495
106762
163453
Avg.
NICER-SLAM [14]
PSNR ↑
12.91
15.48
8.79
11.32
11.68
13.25
13.09
16.41
12.87
SSIM ↑
0.498
0.775
0.330
0.611
0.438
0.541
0.587
0.712
0.562
LPIPS ↓
0.695
0.518
0.791
0.754
0.686
0.691
0.626
0.657
0.677
GLORIE-SLAM [15]
PSNR ↑
25.78
25.52
25.71
24.90
25.09
23.79
27.35
23.72
25.23
SSIM ↑
0.916
0.902
0.883
0.878
0.908
0.891
0.918
0.903
0.900
LPIPS ↓
0.282
0.287
0.365
0.338
0.291
0.309
0.272
0.279
0.303
MonoGS [17]
PSNR ↑
20.58
23.53
21.41
22.34
21.87
21.08
22.31
19.41
21.57
SSIM ↑
0.674
0.862
0.620
0.784
0.684
0.772
0.741
0.753
0.737
LPIPS ↓
0.607
0.421
0.625
0.641
0.514
0.646
0.503
0.657
0.577
DROID-Splat [27]
PSNR ↑
26.77
25.02
26.20
25.92
26.81
24.01
27.21
23.02
25.62
SSIM ↑
0.829
0.823
0.792
0.782
0.850
0.748
0.864
0.720
0.801
LPIPS ↓
0.273
0.384
0.376
0.512
0.297
0.482
0.281
0.451
0.382
Photo-SLAM [28]
PSNR ↑
19.03
20.49
21.28
20.84
21.44
18.14
20.26
19.12
20.08
SSIM ↑
0.640
0.824
0.624
0.759
0.674
0.726
0.712
0.758
0.715
LPIPS ↓
0.527
0.307
0.471
0.466
0.367
0.453
0.440
0.444
0.434
MCGS-SLAM (Ours)
PSNR ↑
27.09
26.26
27.20
28.45
21.91
26.48
27.70
26.92
26.50
SSIM ↑
0.830
0.826
0.813
0.797
0.682
0.813
0.819
0.829
0.801
LPIPS ↓
0.223
0.284
0.233
0.330
0.547
0.231
0.262
0.234
0.293
TABLE I: Appearance reconstruction comparison of different methods on 8 scenes of the Waymo dataset [22] (Real-World
Dataset). MCGS-SLAM achieves the best PSNR and LPIPS results, highlighted as first , second and third .
yields cleaner and more geometrically consistent reconstruc-
tions thanks to dense cross-view constraints and broader
visual coverage. The Village scene, characterized by abrupt
turns and large FoV discontinuities, remains challenging
for single-camera baselines (e.g., MonoGS, DROID-Splat),
which exhibit holes and blending artifacts. By exploiting
multi-view priors and robust depth-scale alignment, MCGS-
SLAM reconstructs sharper structures and more complete
geometry even under wide-baseline motion.
C. Tracking Results Study
Tables II and IV summarize the quantitative tracking accu-
racy on diverse real-world datasets. These Waymo sequences
use original images without distortion correction, providing
a more challenging and realistic setting to evaluate tracking
robustness in autonomous driving conditions. For the Oxford
Spires dataset [25], the original fisheye images were undis-
torted to fit the pinhole camera model, and sequences with
severe distortion were excluded. MCGS-SLAM achieves the
lowest average ATE and ranks first in five of eight Waymo
sequences, demonstrating strong robustness to wide baselines
and complex environments. It maintains low drift even in
difficult cases such as 100613 and 106762, where methods
like MonoGS show large trajectory errors. To account for
the scene-dependent behavior of the JDSA module, which

<!-- page 7 -->
w/o R Camera
w/ F Camera (Mono)
w/ R Camera
w/ F Camera (Stereo)
w/o L Cam
w/o R Camera
w/ F Camera (Mono)
w/o L Cam
w/o L Cam
w/o R Camera
w/ F Camera (Mono)
w/ L Cam
w/o R Camera
w/ F Camera (Mono)
w/o L Camera
w/o R Camera
w/ F Camera (Mono)
w/o L Camera
w/o R Camera
w/ F Camera (Mono)
w/o L Camera
w/ R Camera
w/ F Camera (Stereo)
w/ L Camera
MonoGS [17]
HI-SLAM2 [20]
MCGS-SLAM (Ours)
DROID-Splat [27]
Garden (Inside)
Factory
Village
Fig. 5: MCGS-SLAM produces faithful and complete reconstructions on AirSim [26] (Synthetic Dataset).
Method
Metric
100613
158686
132384
134763
152706
153495
106762
163453
Avg.
NICER-SLAM [14]
ATE [m] ↓
2.351
2.362
56.363
2.642
19.409
19.782
1.634
14.708
14.906
MonoGS [17]
ATE [m] ↓
10.727
10.101
12.033
3.394
9.073
1.628
19.532
9.189
9.459
Splat-SLAM [19]
ATE [m] ↓
0.802
2.575
1.133
1.625
1.092
2.572
1.973
3.115
1.861
HI-SLAM2 [20]
ATE [m] ↓
0.790
1.782
0.888
1.281
0.964
1.389
2.132
2.558
1.473
MCGS-SLAM (Ours)
ATE [m] ↓
0.398
0.612
1.242
1.107
2.554
1.180
2.366
0.927
1.298
TABLE II: Quantitative comparison of tracking accuracy (ATE RMSE) across different methods and scenes on the Waymo
dataset [22]. MCGS-SLAM yields the best average results. Best results are highlighted as first , second and third .
Method
Metric
Garden
Factory
Village
Avg.
NICER-
SLAM [14]
PSNR ↑
12.30
9.84
11.18
11.11
SSIM ↑
0.450
0.332
0.504
0.429
LPIPS ↓
0.801
0.690
0.653
0.715
GLORIE-
SLAM [15]
PSNR ↑
24.50
23.39
17.56
21.82
SSIM ↑
0.849
0.888
0.494
0.744
LPIPS ↓
0.351
0.346
0.712
0.470
MonoGS [17]
PSNR ↑
25.59
21.45
21.39
22.81
SSIM ↑
0.766
0.760
0.689
0.738
LPIPS ↓
0.258
0.175
0.444
0.292
DROID-
Splat [27]
PSNR ↑
24.12
26.50
17.25
22.62
SSIM ↑
0.822
0.898
0.669
0.796
LPIPS ↓
0.242
0.107
0.652
0.334
Photo-
SLAM [28]
PSNR ↑
25.47
28.38
26.77
26.87
SSIM ↑
0.775
0.923
0.805
0.834
LPIPS ↓
0.156
0.041
0.205
0.134
MCGS-
SLAM (Ours)
PSNR ↑
29.36
28.37
28.10
28.64
SSIM ↑
0.879
0.924
0.853
0.885
LPIPS ↓
0.126
0.083
0.219
0.143
TABLE III: Quantitative comparison of appearance recon-
structions of different methods on 3 scenes of the AirSim
dataset [26] (Synthetic Dataset). Best results are highlighted
as first , second and third .
improves metric-scale consistency, but can occasionally in-
crease drift, we evaluated both configurations and reported
the better result. The performance gains mainly stem from
the joint optimization of inter-camera depth and pose in
the MCBA module, supported by effective scale alignment
via JDSA. Although HI-SLAM2 and Splat-SLAM perform
competitively in terms of ATE, their monocular design leads
to greater drift in long or wide-baseline sequences.
Method
Library
Palace
College
Observatory
NICER-SLAM [14]
77.593
41.593
24.580
23.621
MonoGS [17]
FAILED
29.451
30.794
11.814
Splat-SLAM [19]
11.890
37.853
5.756
19.727
HI-SLAM2 [20]
9.001
31.601
1.694
0.262
MCGS-SLAM (Ours)
7.665
3.391
1.551
0.924
TABLE IV: Quantitative comparison of tracking accuracy
(ATE RMSE) across different methods and scenes on the
Oxford Spires Dataset (Bodleian Library, Blenheim Palace,
Christ Church College, and Observatory Quarter). Best re-
sults are highlighted as first , second and third .
Similar trends appear in the Oxford Spires dataset, which
features complex large-scale outdoor scenes with frequent
occlusions and strong parallax. MCGS-SLAM again delivers
superior performance, achieving the lowest average ATE and
outperforming all baselines by a significant margin. In con-
trast, MonoGS fails on several sequences, leading to heavily
degraded ATE values, while HI-SLAM2 and Splat-SLAM
suffer from scale ambiguity and tracking discontinuities. The
ability of MCGS-SLAM to leverage multi-view redundancy
and consistently recover occluded structures from multiple
viewpoints proves essential in these challenging, large-scale
environments. Overall, the results underscore the robustness
and accuracy of our multi-camera framework, which achieves
drift-resilient tracking and significantly outperforms prior
monocular and single-camera systems.
D. Ablation Study
Table
V
analyzes
the
contributions
of
the
Joint
Depth–Scale Alignment (JDSA) module and the monocular

<!-- page 8 -->
HI-SLAM2 [20]
MCGS-SLAM (Ours)
Splat-SLAM [19]
Christ Church College
Bodleian Library
Blenheim Palace
Observatory Quarter
Fig. 6: Tracking performance on the Oxford Spires Dataset [25], evaluated across 4 representative sequences. Ground
truth trajectories are compared against Splat-SLAM [19], HI-SLAM2 [20], and our MCGS-SLAM. MCGS-SLAM remains
closely aligned with ground truth across all sequences, usually achieving the lowest ATE RMSE values and demonstrating
the robustness and accuracy of our multi-camera framework in large-scale outdoor environments.
depth maps predicted by Metric3Dv2 [23]. Removing both
components leads to a notable degradation in performance,
with PSNR dropping, SSIM decreasing, and LPIPS increas-
ing substantially. Introducing only the previous depth im-
proves PSNR, but results in subtle double-edge artifacts due
to inconsistent estimates of the per-camera scale. In contrast,
the full configuration of MCGS-SLAM, with both JDSA and
estimated monocular depth, achieves the best scores across
all three metrics, justifying its superior photometric accuracy.
Although this combination improves reconstruction quality,
in a few challenging cases the ATE slightly worsens, likely
due to errors in the depth maps, suggesting that stronger
predictors could further enhance consistency. Qualitatively,
the depth estimates enhance depth initialization, providing
a better starting point for optimization in multi-camera
configurations. However, without JDSA, inter-camera scale
inconsistencies persist, leading to visible artifacts. The JDSA
module corrects these inconsistencies by performing per-
camera scale alignment and compensates for missing or
noisy depth estimates from MCBA. The combined effect
of depth initialization and scale-consistent optimization im-
proves depth reliability. Their synergy is key to exploiting
the wide field-of-view, enabling cross-view alignment that
densifies Gaussians in regions unseen by a single lens.
V. CONCLUSION AND FUTURE WORK
In this work, we introduced MCGS-SLAM, a fully vision-
based SLAM framework that constructs unified 3D Gaus-
sian maps from synchronized multi-camera RGB inputs. By
jointly optimizing camera poses and dense depths through
Multi-Camera Bundle Adjustment (MCBA) and enforc-
Method
PSNR↑
SSIM↑
LPIPS↓
w/o Depth∗+ w/o JDSA
25.01
0.751
0.404
w/ Depth∗+ w/o JDSA
27.02
0.809
0.271
MCGS-SLAM (full)
27.17
0.816
0.262
∗From Metric3Dv2 [23]
TABLE V: Ablation of Joint Depth–Scale Alignment (JDSA)
on the Waymo dataset [22], averaged over 6 sequences
(134763, 106762, 132384, 152706, 153495, and 163453).
Best results are highlighted as first , second and third .
ing inter-camera scale consistency via our proposed Joint
Depth–Scale Alignment (JDSA) module, the system achieves
real-time, photorealistic, and geometrically consistent recon-
structions. MCGS-SLAM performs well in both synthetic
and real-world scenarios. Our analysis highlights the critical
role of wide-baseline, overlapping views in enhancing scene
completeness and robustness, particularly under occlusion
and viewpoint discontinuities where monocular systems of-
ten fail. Looking ahead, promising directions include inte-
grating inertial or event-based sensing for improved perfor-
mance in dynamic or low-texture environments, extending
the system to support uncalibrated or asynchronous rigs, and
further incorporating semantic or instance-level understand-
ing for object-aware mapping.
REFERENCES
[1] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos.
Orb-slam: A versatile and accurate monocular slam system.
IEEE
transactions on robotics, 31(5), 2015.
[2] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez, Jos´e MM
Montiel, and Juan D Tard´os. Orb-slam3: An accurate open-source li-

<!-- page 9 -->
brary for visual, visual–inertial, and multimap slam. IEEE transactions
on robotics, 37(6), 2021.
[3] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers. Lsd-slam: Large-
scale direct monocular slam. In European conference on computer
vision. Springer, 2014.
[4] Rui Wang, Martin Schworer, and Daniel Cremers.
Stereo dso:
Large-scale direct sparse visual odometry with stereo cameras.
In
Proceedings of the IEEE international conference on computer vision,
2017.
[5] Zachary Teed and Jia Deng.
Droid-slam: Deep visual slam for
monocular, stereo, and rgb-d cameras. Advances in neural information
processing systems, 34, 2021.
[6] Yuheng Qiu, Yutian Chen, Zihao Zhang, Wenshan Wang, and Sebas-
tian Scherer.
Mac-vo: Metrics-aware covariance for learning-based
stereo visual odometry. arXiv preprint:2409.09479, 2024.
[7] Steffen Urban and Stefan Hinz. Multicol-slam-a modular real-time
multi-camera slam system. arXiv preprint:1610.07336, 2016.
[8] Juichung Kuo, Manasi Muglikar, Zichao Zhang, and Davide Scara-
muzza. Redesigning slam for arbitrary multi-camera systems. In IEEE
International Conference on Robotics and Automation (ICRA), 2020.
[9] Wei Zhang, Sen Wang, Xingliang Dong, Rongwei Guo, and Norbert
Haala. Bamf-slam: Bundle adjusted multi-fisheye visual-inertial slam
using recurrent field transforms. In IEEE international conference on
robotics and automation (ICRA), 2023.
[10] Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David
Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie
Shotton, Steve Hodges, and Andrew Fitzgibbon. Kinectfusion: Real-
time dense surface mapping and tracking.
In 2011 10th IEEE
international symposium on mixed and augmented reality. Ieee, 2011.
[11] Christian Kerl, J¨urgen Sturm, and Daniel Cremers. Dense visual slam
for rgb-d cameras.
In 2013 IEEE/RSJ international conference on
intelligent robots and systems. IEEE, 2013.
[12] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T
Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes
as neural radiance fields for view synthesis. Communications of the
ACM, 65(1), 2021.
[13] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller.
Instant neural graphics primitives with a multiresolution hash encod-
ing. ACM transactions on graphics (TOG), 41(4), 2022.
[14] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R
Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural
implicit scene encoding for rgb slam. In 2024 International Conference
on 3D Vision (3DV). IEEE, 2024.
[15] Ganlin Zhang, Erik Sandstr¨om, Youmin Zhang, Manthan Patel, Luc
Van Gool, and Martin R Oswald. Glorie-slam: Globally optimized rgb-
only implicit encoding point cloud slam. arXiv:2403.19549, 2024.
[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George
Drettakis. 3d gaussian splatting for real-time radiance field rendering.
ACM Trans. Graph., 42(4), 2023.
[17] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison.
Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024.
[18] Liyuan Zhu, Yue Li, Erik Sandstr¨om, Shengyu Huang, Konrad
Schindler, and Iro Armeni. Loopsplat: Loop closure by registering
3d gaussian splats. arXiv preprint:2408.10154, 2024.
[19] Erik Sandstr¨om, Ganlin Zhang, Keisuke Tateno, Michael Oechsle,
Michael Niemeyer, Youmin Zhang, Manthan Patel, Luc Van Gool,
Martin Oswald, and Federico Tombari. Splat-slam: Globally optimized
rgb-only slam with 3d gaussians.
In Proceedings of the Computer
Vision and Pattern Recognition Conference, 2025.
[20] Wei Zhang, Qing Cheng, David Skuddis, Niclas Zeller, Daniel Cre-
mers, and Norbert Haala. Hi-slam2: Geometry-aware gaussian slam
for fast monocular scene reconstruction. arXiv:2411.17982, 2024.
[21] Vladimir Yugay, Theo Gevers, and Martin R Oswald. Magic-slam:
Multi-agent gaussian globally consistent slam. In Proceedings of the
Computer Vision and Pattern Recognition Conference, 2025.
[22] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard,
Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai,
Benjamin Caine, et al.
Scalability in perception for autonomous
driving: Waymo open dataset.
In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020.
[23] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen,
Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d
v2: A versatile monocular geometric foundation model for zero-shot
metric depth and surface normal estimation. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2024.
[24] Wei Zhang, Tiecheng Sun, Sen Wang, Qing Cheng, and Norbert Haala.
Hi-slam: Monocular real-time dense mapping with hybrid implicit
fields. IEEE Robotics and Automation Letters, 9(2), 2023.
[25] Yifu Tao, Miguel ´Angel Mu˜noz-Ba˜n´on, Lintong Zhang, Jiahao Wang,
Lanke Frank Tarimo Fu, and Maurice Fallon.
The oxford spires
dataset: Benchmarking large-scale lidar-visual localisation, reconstruc-
tion and radiance field methods. International Journal of Robotics
Research, 2025.
[26] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor.
Airsim: High-fidelity visual and physical simulation for autonomous
vehicles. In Field and Service Robotics, 2017.
[27] Christian Homeyer, Leon Begiristain, and Christoph Schn¨orr. Droid-
splat: Combining end-to-end slam with 3d gaussian splatting. arXiv
preprint:2411.17660, 2024.
[28] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-
slam: Real-time simultaneous localization and photorealistic mapping
for monocular stereo and rgb-d cameras.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024.
