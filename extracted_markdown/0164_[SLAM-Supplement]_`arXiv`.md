<!-- page 1 -->
AgriGS-SLAM: Orchard Mapping Across Seasons
via Multi-View Gaussian Splatting SLAM
Mirko Usuelli1,†, David Rapado-Rincon2, Gert Kootstra2, Matteo Matteucci1
Pear
Orchard
Apple
Orchard
Agricultural
Robot
Dormancy
Flowering
Harvesting
Phenological
Phases
GAUSSIAN
SPLATTING
MULTI-MODAL
LOSS
sparser
denser
 Mapping
 Localization
Photometric Supervision
Geometric Supervision
Template Trajectory
DIRECT
ODOMETRY
KEYFRAME
SELECTION
FACTOR
GRAPH
LOOP
CLOSURE
GPU-CPU
VOXEL
SEARCH
RGB DEPTH
INIT. GUESS
MAPPER
Novel-View Trajectory
Training-View Trajectory
START
END
Experimental Setup
AgriGS-SLAM
1
2
3
4
5
6
Multiple RGB-D
Cameras
LiDAR
Point Cloud
RENDERED
IMAGE
RENDERED
DEPTH
CAMERA
POSES
KEYFRAME
GAUSSIAN
SUBMAP
3DGS thread
MAPPER thread
GRAPHER thread
TRACKER thread
: start|end
: input|output
: process
Background
Optimization
: flow
: gradient
7
Fig. 1.
The agricultural robot follows a standardized trajectory (Training- and Novel-View) in both apple and pear orchards, across dormancy, flowering,
and harvesting stages. LiDAR odometry keyframes update a loop-closing factor graph that triggers background 3DGS optimization, jointly refining Gaussian
submaps and camera poses. Rendered depths and images are corrected via a multimodal geometric and photometric loss, enabling simultaneous gradient-
based refinement for both localization and mapping.
Abstract— Autonomous robots in orchards require real-time
3D scene understanding despite repetitive row geometry, sea-
sonal appearance changes, and wind-driven foliage motion.
We present AgriGS-SLAM, a Visual–LiDAR SLAM frame-
work that couples direct LiDAR odometry and loop closures
with multi-camera 3D Gaussian Splatting (3DGS) rendering.
Batch rasterization across complementary viewpoints recovers
orchard structure under occlusions, while a unified gradient-
driven map lifecycle executed between keyframes preserves
fine details and bounds memory. Pose refinement is guided
by a probabilistic LiDAR-based depth consistency term, back-
propagated through the camera projection to tighten geometry-
appearance coupling. We deploy the system on a field plat-
form in apple and pear orchards across dormancy, flowering,
and harvesting, using a standardized trajectory protocol that
evaluates both training-view and novel-view synthesis to re-
duce 3DGS overfitting in evaluation. Across seasons and sites,
AgriGS-SLAM delivers sharper, more stable reconstructions
and steadier trajectories than recent state-of-the-art 3DGS-
SLAM baselines while maintaining real-time performance on-
tractor. While demonstrated in orchard monitoring, the ap-
proach can be applied to other outdoor domains requiring
robust multimodal perception.
† Corresponding author;
1Mirko Usuelli and Matteo Matteucci are
with the Dipartimento di Bioingegneria, Elettronica e Informazione,
Politecnico
di Milano,
20133 Milano,
Italy {mirko.usuelli,
matteo.matteucci}@polimi.it
2David
Rapado-Rincon
and
Gert
Kootstra
are
with
Agricultural
Biosystems
Engineering,
Wageningen
University
&
Research,
6708
PB
Wageningen,
The
Netherlands
{david.rapadorincon,
gert.kootstra}@wur.nl
I. INTRODUCTION
The rapid growth of the global population is driving an
urgent demand for increased food production, while labor
shortages and strict sustainability regulations increase the
need for autonomous farming technologies [1]. In response,
autonomous systems equipped with advanced sensing and
decision-making capabilities are being developed to reduce
dependence on seasonal labor and improve the efficient
use of resources. Robotic automation in orchards presents
particular challenges that necessitate accurate 3D reconstruc-
tion capabilities. Every tree develops differently and fruit
trees undergo continuous transformations across the growing
season: foliage density shifts dramatically, blossoms emerge
and fade, and fruit develops over time. Temporal changes
require methods that can capture and maintain accurate
spatial representations.
While offline Structure-from-Motion (SfM) approaches
can generate high-quality reconstructions in post-processing,
agricultural applications demand online and real-time capa-
bilities for several critical reasons. First, autonomous agri-
cultural robots require immediate spatial understanding for
navigation and manipulation tasks: they cannot wait for of-
fline processing to complete harvesting, spraying, or pruning
operations. Second, farmers need immediate feedback during
field operations to adjust interventions on the fly, such as
modifying spray patterns based on current canopy density
arXiv:2510.26358v1  [cs.RO]  30 Oct 2025

<!-- page 2 -->
or identifying areas requiring instant attention. Third, the
concept of digital twins in agriculture envisions continuous
synchronization between the physical world and its digital
representations, enabling interactive monitoring. To simulta-
neously localize the robot and build maps in real-time, these
requirements necessitate SLAM (Simultaneous Localization
and Mapping).
There is a significant research gap in the development
of SLAM systems suitable for large-scale agricultural en-
vironments, such as orchards. The repetitive geometry of
tree rows introduces systematic ambiguities in short- and
long-term associations, where the scarcity of distinctive
landmarks often leads to incorrect data associations. Existing
SLAM approaches remain limited in this context: vision-only
methods fail under repetitive crop patterns and vegetation
motion [2], while LiDAR-only systems are hampered by
geometric sparsity compared to the structured surfaces of
man-made environments [3].
Recent advances in neural rendering have introduced
promising paradigms for dense 3D reconstruction. Neural
Radiance Fields (NeRF) [4] encode scenes as neural func-
tions but suffer from expensive training and poor scalabil-
ity. In contrast, 3D Gaussian Splatting (3DGS) [5] offers
explicit point-based representations with discrete Gaussian
primitives, called splats, enabling localized updates and effi-
cient rasterization, properties well-suited for the incremental
nature of SLAM in large scenes. However, most NeRF and
3DGS-based SLAM methods remain restricted to controlled
room-scale indoor scenes [6].
Building on recent advances in SLAM and 3D reconstruc-
tion, we propose a Visual–LiDAR SLAM framework (Fig. 1)
tailored to orchards. A key contribution is the multi-camera
setup, which provides complementary perspectives around
the vehicle: lateral views improve canopy coverage under
occlusions, while the inline view supports traversability
perception along narrow rows (Fig. 2). On the geometric side,
direct LiDAR odometry delivers robust tracking, remaining
largely unaffected by wind-induced foliage motion or illumi-
nation changes that can disrupt vision methods, since LiDAR
operates on the point cloud structure as a whole. The two
modalities are fused to both improve 3DGS rendering and
refine camera poses, with loop closures detected and refined
through scan-matching to correct drift during extended row
traversals and tight inter-row maneuvers.
The main contributions of this work are:
• Real-Time Visual–LiDAR 3DGS-SLAM (III):
Our method combines LiDAR odometry, loop closure,
multimodal optimization, and memory-aware incremen-
tal 3D Gaussian Splatting, providing real-time percep-
tion, 3D reconstruction, and trajectory estimation for
agricultural robots in orchards.
• Cross-Seasonal Applicability Benchmark (IV, V):
We compare against diverse 3DGS-SLAM state-of-
the-art pipelines (Photo-SLAM [7], Splat-SLAM [8],
OpenGS-SLAM [9], PINGS [10]). Our pipeline evalu-
ates both training- and novel-view trajectories, overcom-
ing prior methods that only assess training-view render-
Fig. 2.
Agricultural platform equipped with three cameras (one horizontal
and two vertical), a 32-beam LiDAR, and a VIO–GNSS–RTK system for
ground truth acquisition.
ing. This enables a fair joint evaluation of reconstruction
and localization, highlighting the generalization and
performance gains of our approach across seasons.
• ∇Mapping & ∇Localization (V-C §1, §2):
A unified, gradient-driven 3DGS-SLAM that (i) up-
dates the map between keyframes via image-gradient
densification, opacity/scale-aware pruning, and sched-
uled opacity resets under per-batch budgets; and (ii)
refines poses using a probabilistic LiDAR-guided depth-
consistency loss, defined as a Kullback–Leibler (KL)
divergence between LiDAR and rendered depth, en-
abling sensor fusion with cameras and back-propagation
through the projection model.
• Multi-Camera 3DGS-SLAM (V-C §3):
To the best of the authors’ knowledge, the first real-
time 3DGS-SLAM framework supporting multi-camera
setups in outdoor environments, enabled by a batch
rasterization strategy. Multiple cameras are leveraged to
overcome occlusions and limited viewpoints in orchard.
The entire implementation of this work is open source
and available on GitHub at: https://github.com/
AIRLab-POLIMI/agri-gs-slam.
II. RELATED WORK
Recent advances in neural rendering have enriched the
traditional SLAM pipeline, coupling accurate localization
with photorealistic 3D reconstruction. Early implicit methods
based on neural radiance fields, such as NeRF-SLAM [11],
encoded geometry and appearance directly in network
weights, achieving dense reconstructions with strong photo-
metric consistency. However, these approaches proved com-
putationally prohibitive for real-time robotics applications,
scaling poorly to large environments [6].
To overcome these limitations, 3DGS [5] emerged as an
explicit alternative. Originally developed as an extension
of classical photogrammetry, 3DGS reformulates multi-view
reconstruction into a differentiable scene representation. By
modeling environments as collections of Gaussian primitives,
it preserves the flexibility of optimization while allowing
localized updates. This formulation has since provided the

<!-- page 3 -->
TABLE I
MULTI-VIEW IMAGE COMPARISON GROUPED BY ORCHARD TYPE, PHENOLOGICAL STAGE AND RENDERING MODALITY.
Apple Orchard
Pear Orchard
GT / Training-View Rendering
GT / Novel-View Rendering
GT / Training-View Rendering
GT / Novel-View Rendering
Dormancy
Flowering
Harvesting
foundation for adapting 3DGS to SLAM, enabling practical
real-time systems that couple reconstruction and localization.
Adapting 3DGS-SLAM to outdoor settings introduced
substantial challenges, including variable illumination, dy-
namic elements, and sparse or absent depth measurements.
Two primary strategies have emerged to address these con-
straints: sensor-fusion methods, such as LiV-GS [12], which
integrate point clouds into 3DGS with geometric and normal
consistency under inertial correction, and classical visual
pipelines, such as Splat-SLAM [8], which rely solely on
RGB inputs to infer geometry via multi-view optimization
or learned depth priors without dense depth supervision.
Recent systems exhibit considerable architectural diver-
sity, reflecting different integration strategies and algorith-
mic philosophies. Photo-SLAM [7] extends the classical
ORB-SLAM3 [13] framework by augmenting feature-based
tracking with Gaussian rendering for enhanced photometric
refinement. Splat-SLAM [8] integrates 3DGS directly into
the dense bundle adjustment pipeline of DROID-SLAM [14].
OpenGS-SLAM [9] combines transformer-based depth es-
timation with RANSAC-PnP for robust pose estimation in
challenging conditions, while PINGS [10] fuses LiDAR and
SDF implicit points into splats to achieve joint geometric
and photometric alignment. This heterogeneous landscape
illustrates the rapid maturation of 3DGS-SLAM from initial
proof-of-concept systems to a diverse family of pipelines
spanning classical and multimodal paradigms.
Within agricultural robotics, 3D reconstruction has pri-
marily served offline tasks such as phenotyping, yield es-
timation, and crop perception [15]–[18]. While NeRF-based
approaches have been explored for agricultural scenes, the
distinctive challenges of orchard environments remain largely
unaddressed in online 3DGS-SLAM research. These chal-
lenges include highly repetitive row structures, frequent
occlusions from dense foliage, and pronounced seasonal
appearance variations that violate typical photometric con-
sistency assumptions.
III. PROPOSED APPROACH
A. Simultaneous Localization and Mapping
1) SLAM Frontend: We use Direct LiDAR Odometry
(DLO) [19] as the scan-to-submap odometry module, since

<!-- page 4 -->
Dormancy
Flowering
Harvesting
Training-View
Novel-View
Training-View
Novel-View
Training-View
Novel-View
Method
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
ATE (m) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
ATE (m) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
ATE (m) ↓
Apple Orchard
Photo-SLAM [7]
8.1567
0.2273
0.7058
8.3705
0.2137
0.7466
18.9545
8.9779
0.2990
0.7077
9.4701
0.2604
0.7424
15.1569
9.2807
0.3442
0.6823
9.7843
0.3211
0.7278
19.0975
Splat-SLAM [8]
19.6937
0.1761
0.9562
11.0302
0.1575
0.9997
5.2648
19.7930
0.1893
0.9879
12.3704
0.1255
0.9952
28.1355
19.7298
0.2730
0.9561
14.5519
0.2567
0.9844
21.8882
PINGS [10]
10.7060
0.3030
0.6343
10.1925
0.2448
0.6908
171.4970
6.2423
0.1822
0.6839
6.8555
0.1528
0.7208
132.9355
5.3410
0.2325
0.6865
4.3199
0.2064
0.7112
89.4429
OpenGS-SLAM [9]
12.9748
0.3486
0.7627
14.5637
0.4231
0.7772
20.6953
14.3540
0.3732
0.7746
14.2038
0.3553
0.7995
20.9450
14.3598
0.4346
0.7033
14.0214
0.3984
0.7403
20.2172
DLO+3DGS [5], [19]
25.0465
0.7157
0.3854
15.8479
0.3781
0.8092
0.5756
23.2178
0.6680
0.4989
16.5182
0.3664
0.7929
0.4084
31.6495
0.8595
0.2141
18.0344
0.4541
0.7998
0.6578
AgriGS-SLAM
29.8954
0.8959
0.1316
15.7438
0.3890
0.5915
0.5192
25.5346
0.8239
0.2319
16.7627
0.3878
0.5410
0.3543
31.8057
0.8755
0.1599
17.8304
0.4605
0.6237
0.7370
AgriGS w/2 cameras
26.2831
0.9129
0.1341
14.4759
0.3739
0.6018
0.5192
26.8023
0.9006
0.1571
15.6297
0.3808
0.5642
0.3561
33.5170
0.9157
0.1127
15.7548
0.4077
0.6646
0.7348
AgriGS w/3 cameras
30.0287
0.9249
0.1253
14.6592
0.4153
0.5500
0.5104
28.6355
0.9137
0.1367
14.7663
0.3411
0.5667
0.3410
35.1030
0.9307
0.1064
15.1278
0.3727
0.6705
0.7281
Pear Orchard
Photo-SLAM [7]
7.9990
0.2739
0.6536
7.9589
0.2678
0.6382
16.2391
7.8072
0.2801
0.6565
8.2488
0.3417
0.6619
23.7226
8.3235
0.2839
0.6513
8.1888
0.2462
0.6713
18.9968
Splat-SLAM [8]
21.5711*
0.2170*
0.9078*
11.1528*
0.1626*
0.9448*
6.6386*
23.5328
0.2829
0.6885
13.9598
0.2822
0.8749
20.8523
18.4055
0.2114
0.9446
13.5946
0.1635
0.9468
18.2869
PINGS [10]
9.2289
0.3402
0.6027
9.7279
0.3449
0.5964
157.2260
13.0218
0.4558
0.5374
12.4485
0.4538
0.5615
87.3478
11.7895
0.3322
0.6089
10.8935
0.2549
0.6858
97.7556
OpenGS-SLAM [9]
12.8747
0.3920
0.6911
15.2439
0.5080
0.6410
21.0662
14.3015
0.4604
0.6722
16.1294
0.5783
0.6608
10.9321
15.3887
0.4354
0.6580
15.4138
0.4125
0.7008
20.8209
DLO+3DGS [5], [19]
18.5079
0.6724
0.5774
18.6592
0.5243
0.6613
0.4402
30.5728
0.8867
0.2095
19.0106
0.5111
0.6550
0.4478
33.1395
0.8498
0.2607
17.5047
0.3615
0.7678
0.7551
AgriGS-SLAM
18.3281
0.6728
0.5648
17.7175
0.5539
0.5244
0.4264
33.1657
0.9206
0.1339
18.8224
0.5635
0.5289
0.4398
35.0662
0.9038
0.1531
17.4914
0.4006
0.5920
0.7201
AgriGS w/2 cameras
19.3567
0.7223
0.4141
15.9485
0.5111
0.5023
0.4313
29.6681
0.9176
0.1252
17.1290
0.5393
0.5056
0.4321
34.9862
0.9210
0.1152
17.2648
0.4655
0.5937
0.7150
AgriGS w/3 cameras
16.8865
0.5625
0.4830
16.1695
0.4271
0.5631
0.4303
34.3933
0.9300
0.1019
16.7364
0.5559
0.4864
0.4305
34.3865
0.9146
0.1238
16.2442
0.4664
0.5566
0.7120
TABLE II
COMPARISON OF APPLE AND PEAR ORCHARDS ACROSS PHENOLOGICAL PHASES. TRAINING/VALIDATION METRICS (PSNR, SSIM, LPIPS) AND
ABSOLUTE TRAJECTORY ERROR (ATE, M) ARE SHOWN. ARROWS MARK WHETHER HIGHER (↑) OR LOWER (↓) IS BETTER. BEST, SECOND, AND THIRD
RESULTS ARE MARKED IN BOLD (RED), UNDERLINE (ORANGE), AND YELLOW. ASTERISK (∗) DENOTES PARTIAL TRAJECTORIES DUE TO DIVERGENCE.
conventional LiDAR odometry degrades in off-road and
orchard conditions [3], [20]. We re-implement DLO as a
standalone (non-ROS) library to (i) keep the pipeline modular
and (ii) inject a vehicle prior tailored to agricultural driving,
which is dominated by wheel slip, uneven ground, and sharp
local maneuvers.
Let Pk = {pk
i }Nk
i=1 be the k-th LiDAR sweep with pose
Xk ∈SE(3). From past keyframes we maintain an active local
submap Sk−1, selected using nearest-neighbor and convex-
hull criteria. The frontend estimates the relative motion
ˆ
∆Xk
between Pk and Sk−1 using Generalized ICP (GICP).
Standard DLO assumes constant body velocity during a
LiDAR sweep. We instead use a second-order motion prior
that includes both velocity and acceleration of the platform.
Let ξ(t) ∈se(3) be the instantaneous body twist (linear and
angular velocity) and ˙ξ(t) its time derivative (accelerations).
The pose at time t +∆t is predicted as
ˆX(t +∆t) ≈X(t) exp

∆t ξ(t)+ 1
2∆t2 ˙ξ(t)

,
(1)
where exp(·) is the exponential map from se(3) to SE(3).
This model is used both as the registration prior and for
LiDAR deskewing. In practice it stabilizes convergence when
the tractor undergoes sudden acceleration or pitch/roll excur-
sions on rough soil.
2) SLAM Backend: The backend maintains a factor graph
over keyframe poses and solves it incrementally with iSAM2.
The first keyframe X0 is anchored by a prior factor with
covariance Σprior. For each new keyframe k, we add an odom-
etry factor that enforces consistency between (Xk−1,Xk) and
the frontend estimate
ˆ
∆Xk. The residual is computed in the
Lie algebra of SE(3) via the log map, and is weighted by an
odometry covariance Σodo.
Loop closures are introduced as non-consecutive pose
constraints. Loop candidates are first proposed using (i) a
KD-tree search in pose space and (ii) Scan Context de-
scriptors [21]. Candidates are ranked by cosine similarity
between descriptors, then verified geometrically with GICP.
Only matches that pass GICP are inserted as loop-closure
factors. This reduces long-term drift and enforces global
consistency of the trajectory.
B. Multi-View 3D Gaussian Splatting
We extend 3DGS to a synchronized multi-view setting to
handle occlusions in orchards. The platform carries multi-
ple cameras with complementary views (Fig. 2). At each
keyframe, all cameras are optimized jointly: their gradients
update a single shared set of splats. Gaussians are not
duplicated per camera; each splat lives once in the global
map and may be seen by one or more views.
Each Gaussian i is defined by
µi ∈R3,
Σi ∈R3×3,
oi ∈R,
ci ∈RdSH,
(2)
where µi is its 3D center, Σi its 3D covariance, oi its opacity,
and ci spherical-harmonics (SH) color coefficients.
As in standard 3DGS, covariance comes from a learned
scale Si and rotation Ri:
Σi = Ri Si S⊤
i R⊤
i .
(3)
For each camera b ∈{1,...,B}, splats are projected into
that camera via approximate rasterization. With world-to-
camera transform Wb and projection Jacobian Jb, the image-
space covariance is
Σ′
i,b = Jb Wb Σi W⊤
b J⊤
b ,
(4)
and its 2×2 top-left block is used for rendering in view b.
Joint multi-view supervision lets all cameras refine the
same map and the camera poses, improving consistency
under dense foliage and repetitive row structure without per-
camera splat copies.
1) Incremental
Gaussian
Mapping:
The
incremental
Gaussian mapper maintains an explicit scene representation
as a set of splats, stored in CPU memory for long–term
retention and partially streamed to the GPU as an active set
for real–time optimization.
Whenever the SLAM system selects a new LiDAR
keyframe, the corresponding RGB–D images from onboard
cameras are refined to the LiDAR frame via GICP. This

<!-- page 5 -->
TABLE III
COMPARISON OF SLAM METHODS ACROSS THREE PHENOLOGICAL STAGES FOR APPLE ORCHARD AND PEAR ORCHARD.
Ground Truth
AgriGS-SLAM
DLO+3DGS
OpenGS-SLAM
PINGS
Splat-SLAM
Photo-SLAM
Apple Orchard
Dormancy
Flowering
Harvesting
Pear OrchardDormancy
Flowering
Harvesting
alignment yields a dense set of 3D points that (i) provide
an initialization for the 3DGS process and (ii) compensate
for small timestamp offsets across sensors, since camera and
LiDAR frames recorded simultaneously on the same unit
may exhibit slight delays.
The fused points {pi} are inserted into the submap by
hashing each point into a voxel index:
ki =

pi,x/v

P1 +

pi,y/v

P2 +

pi,z/v

P3,
(5)
where v is the current voxel size and {P1,P2,P3} are large
primes chosen to minimize collisions. This voxel hashing
strategy enables efficient coarse retrieval in large environ-
ments, where fast access to local neighborhoods is critical.
Points whose voxel keys match existing splats are marked as
active, while unmatched points instantiate new splats.
2) Memory Management: To bound GPU memory usage,
the mapper employs an adaptive active–set strategy. When
the number of active splats exceeds a threshold τ, the
least–recently–used elements are offloaded to CPU memory.
Conversely, splats stored on the CPU are reloaded when
their voxel keys collide with new incoming points. This
policy sustains real-time operation while retaining a complete
global map, ensuring scalability along long orchard rows and
supporting loop closures at revisited locations.
3) Gaussian Splats Lifecycle Operations: The Gaussian
map is updated online between keyframes. For each splat
j, three lifecycle operations maintain detail while keeping
memory bounded.
Pruning. Every ∆p iterations, the opacity is updated as the
sigmoid of its trainable logit ω j. A splat is removed if its
opacity oj is below τopa = 0.01, or if its size sj is either too
large (sj > 0.3) or too small (sj < 5 × 10−5). This discards
dim, unstable, or oversized splats while retaining potentially
useful weak ones.
Densification. Every ∆d iterations, the image–space gradient
magnitude at the projected splat center is computed. If the
gradient exceeds a threshold γg, refinement is triggered: when
the splat is relatively large (s j > κ) it is split into smaller
ones, whereas when the splat is still small (sj ≤2sinit) it is
duplicated. This increases local density around fine structures
such as leaves and thin branches, without unbounded growth.
Opacity Reset. Every ∆r iterations, splats that are weak but
not clear outliers are reset. Specifically, if the opacity lies
between τopa and a reset threshold τreset, and the size is within
valid bounds (smin < sj < smax), the opacity is refreshed.
This allows low-opacity splats to recover under changing
illumination and motion.
4) Optimization Loop:
At each iteration, the mapper
schedules: opacity reset every ∆r steps, pruning every
∆p steps, densification every ∆d steps, and GPU memory
cleanup every 10 steps. These routines are designed to run
in real time between keyframes, combining gradient-driven
densification, conservative pruning, and periodic resets to
preserve fine detail while keeping memory usage bounded.

<!-- page 6 -->
C. Multimodal Loss Function
We propose a composite objective that jointly aligns the
rendered Gaussian splats with ground-truth RGB imagery
and LiDAR observations. Let Ipred and Igt denote the rendered
and ground-truth RGB images, respectively;
1) Photometric Consistency: Following Kerbl et al. [5],
the parameters of the 3D Gaussians are optimized by
minimizing a weighted combination of an L1 loss and a
Structural Similarity Index (SSIM) loss between rendered
and ground-truth images. The balance between these two
terms is controlled by λ:
L3DGS = (1−λ)∥Ipred −Igt∥1 +λ
 1−SSIM(Ipred,Igt)

. (6)
2) Probabilistic Depth Consistency: To handle LiDAR
sparsity and noise, we introduce a probabilistic depth-
consistency loss. Each LiDAR return at pixel u is modeled
as a Gaussian with mean DLiDAR(u) and fixed variance σ2
L,
while the renderer predicts a Gaussian N (µu,σ2
u ). The loss
averages the KL divergence between these distributions over
all valid pixels Ω:
LLiDAR = λL
1
|Ω| ∑
u∈Ω
KL
 N (DLiDAR(u),σ2
L) ∥N (µu,σ2
u )

.
(7)
This enforces alignment between LiDAR geometry and
rendered depth, enabling sensor fusion at the loss level.
3) Total Multimodal Loss: The final objective LAgriGS
combines photometric and geometric supervision:
LAgriGS = L3DGS +LLiDAR.
(8)
During training, the multimodal loss guides the simul-
taneous optimization of the Gaussian scene representation
and the camera or robot poses, integrating complementary
photometric and geometric cues for consistent supervision.
IV. EXPERIMENTS
We conducted experiments across three phenological
phases, dormancy, flowering, and harvesting, capturing the
impact of seasonal changes in foliage, occlusions, and il-
lumination. Two contrasting orchards were considered: an
apple orchard with espalier canopies that simplify naviga-
tion, yet grow dense with leaves, and a pear orchard with
globular trees whose volumetric structure creates persistent
occlusions and spatial complexity. Data collection followed
a standardized protocol: one clockwise pass for training and
two counter-clockwise passes, the final reserved for novel-
view validation (Fig. 1). This design introduced viewpoint
variation while maintaining temporal consistency and maxi-
mizing overlap across the multi-camera platform (Fig. 2). To
focus on orchard reconstruction, we explicitly remove the sky
from observations, avoiding large uninformative regions that
would bias optimization and degrade map consistency.
The literature on 3DGS-SLAM has expanded rapidly in
the past year. To capture the breadth of current approaches,
we selected four highly recent methods, published within
the last 12 months at leading robotics and computer vision
Fig. 3.
Comparison of SLAM trajectories against ground truth.
venues, balancing indoor and outdoor settings as well as
diverse tracking strategies. For indoor environments, we
included Photo-SLAM [7], which extends the classical ORB-
SLAM3 [13] pipeline with handcrafted feature tracking,
and Splat-SLAM [8], which leverages the deep learning-
based tracking of DROID-SLAM [14] to represent the neural
counterpart of the classical paradigm. For outdoor settings,
we selected OpenGS-SLAM [9], which employs transformer-
based depth prediction followed by RANSAC and PnP
alignment for robust pose estimation, and PINGS [10], a
LiDAR-based system that integrates scan-matching with SDF
representations. We excluded categories beyond our scope,
such as inertial-based methods requiring ROS/ROS 2 inte-
gration, as well as earlier works superseded by more recent
contributions. Incremental variants (e.g., DROID-Splat [22])
that offer only minor deviations or are outperformed were
also omitted. Similarly, we excluded methods without public
implementations (e.g., LSG-SLAM [23]) and those reliant on
ground-truth supervision (e.g., Point-SLAM [24]), as they are
unsuitable for real-world deployment.
V. RESULTS AND DISCUSSION
Our primary baseline, DLO+3DGS, serves a twofold pur-
pose. First, by disabling pose correction from 3DGS back to
odometry, it isolates the effect of LiDAR-guided localization
via LLiDAR, allowing a direct comparison between gradient-
driven pose refinement and the DLO prior (see Subsection V-

<!-- page 7 -->
C §1). Second, it benchmarks the 3DGS training strategy
introduced in Section III (see Subsection V-C §2) against
the original schedule of Kerbl et al. [5].
A. Quantitative Results
Overall, AgriGS-SLAM outperforms the state-of-the-art
across both orchard scenarios (Table II). Performance is
weaker in the dormancy domain, where the sparsity of trunks
and thin branches makes optimization harder and produces
more splatting artifacts. The highest results are obtained in
the harvesting stage, with similarly strong results in flowering
due to blossom density.
Multi-camera setups generally enhance localization ac-
curacy (ATE), with two cameras outperforming one and
three outperforming two. The sole exception arises in ap-
ple harvesting, where the DLO+3DGS baseline marginally
surpasses AgriGS-SLAM despite the overall trend. Notably,
although the three-camera configuration often achieves the
best scores, the single-camera AgriGS-SLAM attains higher
average performance across all metrics, underscoring the
robustness of the baseline. This can be attributed to the fact
that multiple overlapping views increase the likelihood of
rendering artifacts, whereas the single-view setting avoids
such inconsistencies while still delivering strong localization.
On novel-view trajectories, OpenGS-SLAM achieves the
best results among state-of-the-art methods, with superior
SSIM in pear flowering and apple dormancy. Neverthe-
less, its performance is consistently below AgriGS-SLAM
in other novel-view and training-view metrics. Conversely,
on training-view trajectories, Splat-SLAM is the strongest
competitor, surpassing OpenGS-SLAM in both localization
and rendering, though still inferior to AgriGS-SLAM. Im-
portantly, Splat-SLAM diverged in one sequence (pear dor-
mancy) even after extensive tuning, whereas AgriGS-SLAM
completed all runs without failure.
Among the remaining baselines, DLO+3DGS obtains
higher PSNR in novel view but performs poorly in SSIM and
LPIPS due to noisier splats designed for long offline opti-
mizations rather than our real-time 3DGS scheduling. PINGS
achieves the worst results overall, despite being LiDAR-
based and designed for outdoor environments. Finally, Photo-
SLAM (ORB-SLAM3 based) proved unsuitable for orchards,
consistently crashing even after extensive parameter tuning.
B. Qualitative Results
Table III reports a visual comparison across different
SLAM baselines in orchard environments under varying sea-
sonal and structural conditions. Reconstructions are shown
for both apple and pear orchards during dormancy, flowering,
and harvesting.
Across all conditions, Photo-SLAM and Splat-SLAM
struggle to preserve structural consistency, producing overly
blurred or distorted representations that fail to capture the
tree rows and inter-row spacing. Similarly, PINGS frequently
collapses the orchard geometry, resulting in severe artifacts
and large black voids where structural details are lost.
OpenGS-SLAM yields more coherent layouts, yet recon-
structions remain unstable, with wavy ground surfaces and
over-smoothed vegetation.
In Fig. 3, we show the 2D projection of the trajectories
reported in Table II. Most methods fail in both orchard
scenarios, with only Splat-SLAM achieving comparatively
better results. Although proposed for outdoor tasks, PINGS
yields the poorest performance, even underperforming the
indoor-oriented methods Photo-SLAM and Splat-SLAM.
Conversely, OpenGS-SLAM achieves the best rendering
quality but fails to maintain accurate localization, despite
being designed for outdoor environments.
C. Ablation Studies
We perform ablation studies to assess the contribution
of each component of our pipeline (§1,
§2,
§3); as part
of this analysis, we also evaluated the effect of adding
supplementary loss functions to the core objective (§4),
which did not provide further improvements and confirms
the robustness of our design:
1) ∇Localization (Table II). A direct comparison to our
primary baseline, DLO+3DGS, highlights the impact
of the LiDAR supervision in AgriGS-SLAM. While
the former achieves strong reconstructions, the latter
improves both mapping and localization simultane-
ously. The effect is most pronounced in the dor-
mancy and flowering phase, where LLiDAR reduces
ATE despite sparsity. Similar gains are observed during
harvesting, where dense canopies often degrade odom-
etry but AgriGS-SLAM preserves scene coherence and
trajectory accuracy.
2) ∇Mapping (Table II). We compare our proposed
training schedule with the original formulation [5].
While keeping data, poses, and loss terms unchanged,
our strategy consistently delivers improved rendering
metrics across both seasons and orchard types, reflect-
ing more stable optimization and a stronger coupling
between geometry and appearance.
3) Multi-Camera Views (Table I, Table II) The multi-
camera ablation reveals a clear trend: while adding
views progressively sharpens reconstructed geometry
and reduces drift on training trajectories, it simulta-
neously degrades novel-view synthesis by increasing
overfitting.
4) Multimodal Loss (Table IV). A complementary study
on the multimodal loss evaluates the added contribu-
tions of Chamfer distance for point-cloud alignment,
LOS constraints for geometry, surface-normal regu-
larization for structure, and Raydrops for occlusion,
with Table IV reporting the average results of each
sequence. The complete formulation consistently im-
proves PSNR, SSIM, and LPIPS over reduced vari-
ants, while normal regularization proves ineffective in
orchards with irregular surfaces and Chamfer distance
adds overhead without clear benefits.

<!-- page 8 -->
TABLE IV
SUPPLEMENTARY ANALYSIS OF MULTIMODAL LOSS.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
AgriGS-SLAM
26.7204
0.8870
0.2177
AgriGS-SLAM w/ Chamfer
25.9230
0.8626
0.2638
AgriGS-SLAM w/ LOS
26.5652
0.8827
0.2290
AgriGS-SLAM w/ Normals
25.8807
0.8544
0.2800
AgriGS-SLAM w/ Raydrops
26.5247
0.8813
0.2292
VI. CONCLUSION
We presented AgriGS-SLAM, a Visual–LiDAR 3DGS-
SLAM framework for orchards that integrates direct Li-
DAR odometry, loop-closure refinement, and incremental
memory-aware mapping under a multimodal optimization
scheme. The multi-camera design mitigates occlusions, while
a gradient-driven map lifecycle and a LiDAR-guided depth
term couple geometry and appearance more effectively than
prior pipelines. An orchard-specific trajectory template that
evaluates both training- and novel-view synthesis reduces
reconstruction bias. Tested across apple and pear orchards
at multiple growth stages, AgriGS-SLAM achieves higher
reconstruction quality and trajectory accuracy than recent
3DGS-SLAM baselines, with ablations confirming the im-
pact of the LiDAR loss, scheduled training, and multi-camera
input, while working real-time under seasonal variation.
Limitations include sensitivity to LiDAR–camera calibra-
tion and reduced performance in sparser scenes, where mis-
alignments may occur from sensor delays and motion. The
KL-divergence depth-consistency term alleviates but cannot
fully remove these cross-modal temporal discrepancies.
As future work, we plan to extend recordings to additional
growth stages with semantic annotations (e.g., flowering and
fruit counts), enabling broader use in AgriGS-SLAM, and
to incorporate inertial cues for tighter motion priors and full
on-tractor deployment. While compatible with digital-twin
workflows, this study establishes improved SLAM and 3D
reconstruction in real orchards, providing a foundation for
precision-agriculture applications from on-tractor perception
to farm-management support.
ACKNOWLEDGMENT
The authors thank the Fruit Research Center (FRC) in
Randwijk for access to the orchards. Mirko Usuelli’s work
was carried out within the Agritech National Research Cen-
ter and funded by the EU Next-GenerationEU (PNRR –
M4C2, Inv. 1.4 – D.D. 1032 17/06/2022, CN00000022).
This manuscript reflects only the authors’ views; the EU
and Commission are not responsible. Contributions from
Matteo Matteucci, Gert Kootstra, and David Rapado-Rincon
were co-funded by the EU Digital Europe Programme (Agri-
foodTEF, GA Nº 101100622).
REFERENCES
[1] G. Kootstra, X. Wang, P. M. Blok, J. Hemming, and E. Van Henten,
“Selective harvesting robotics: current research, trends, and future
directions,” Current Robotics Reports, vol. 2, no. 1, pp. 95–104, 2021.
[2] F. Shu, P. Lesur, Y. Xie, A. Pagani, and D. Stricker, “SLAM in the
Field: An Evaluation of Monocular Mapping and Localization on
Challenging Dynamic Agricultural Environment,” in Proceedings of
the IEEE/CVF WACV, January 2021, pp. 1761–1771.
[3] S. W. Chen, G. V. Nardari, E. S. Lee, C. Qu, X. Liu, R. A. F. Romero,
and V. Kumar, “SLOAM: Semantic Lidar Odometry and Mapping for
Forest Inventory,” IEEE Robotics and Automation Letters, vol. 5, no. 2,
pp. 612–619, 2020.
[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” in ECCV, 2020.
[5] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[6] A. J. Davison, The Computational Structure of Spatial AI Systems.
Cambridge University Press, 2025.
[7] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-time
simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in IEEE/CVF CVPR, June 2024, pp.
21 584–21 593.
[8] E. Sandstr¨om, G. Zhang, K. Tateno, M. Oechsle, M. Niemeyer,
Y. Zhang, M. Patel, L. Van Gool, M. Oswald, and F. Tombari,
“Splat-slam: Globally optimized rgb-only slam with 3d gaussians,”
in IEEE/CVF CVPR, 2025, pp. 1680–1691.
[9] S. Yu, C. Cheng, Y. Zhou, X. Yang, and H. Wang, “Rgb-only
gaussian splatting slam for unbounded outdoor scenes,” arXiv preprint
arXiv:2502.15633, 2025.
[10] Y. Pan, X. Zhong, L. Jin, L. Wiesmann, M. Popovi´c, J. Behley, and
C. Stachniss, “Pings: Gaussian splatting meets distance fields within
a point-based implicit neural map,” arXiv preprint arXiv:2502.05752,
2025.
[11] A. Rosinol, J. J. Leonard, and L. Carlone, “Nerf-slam: Real-time dense
monocular slam with neural radiance fields,” in 2023 IEEE/RSJ IROS.
IEEE, 2023, pp. 3437–3444.
[12] R. Xiao, W. Liu, Y. Chen, and L. Hu, “Liv-gs: Lidar-vision integra-
tion for 3d gaussian splatting slam in outdoor environments,” IEEE
Robotics and Automation Letters, 2024.
[13] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and
J. D. Tard´os, “Orb-slam3: An accurate open-source library for visual,
visual–inertial, and multimap slam,” IEEE transactions on robotics,
vol. 37, no. 6, pp. 1874–1890, 2021.
[14] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular,
stereo, and rgb-d cameras,” Advances in neural information processing
systems, vol. 34, pp. 16 558–16 569, 2021.
[15] L. Meyer, A. Gilson, U. Schmid, and M. Stamminger, “Fruitnerf: A
unified neural radiance field based fruit counting framework,” in 2024
IEEE/RSJ IROS.
IEEE, 2024, pp. 1–8.
[16] J. Gr¨un, L. Meyer, M. Weiherer, B. Egger, M. Stamminger, and
L. Franke, “Towards integrating multi-spectral imaging with gaussian
splatting,” arXiv preprint arXiv:2509.00989, 2025.
[17] S. Adebola, S. Xie, C. M. Kim, J. Kerr, B. M. van Marrewijk,
M. van Vlaardingen, T. van Daalen, E. van Loo, J. L. Susa Rincon,
E. Solowjow, R. van Zedde, and K. Goldberg, “Growsplat: Construct-
ing temporal digital twins of plants with gaussian splats,” in IEEE
CASE, 2025, pp. 1766–1773.
[18] S. Chopra, F. Cladera, V. Murali, and V. Kumar, “Agrinerf: Neural
radiance fields for agriculture in challenging lighting conditions,”
arXiv preprint arXiv:2409.15487, 2024.
[19] K. Chen, B. T. Lopez, A.-a. Agha-mohammadi, and A. Mehta, “Direct
lidar odometry: Fast localization with dense point clouds,” IEEE
Robotics and Automation Letters, vol. 7, no. 2, pp. 2000–2007, 2022.
[20] R. Huaman, C. Gonzalez, and S. Prado, “Comparative analysis of
lidar inertial odometry algorithms in blueberry crops,” Engineering
Proceedings, vol. 83, no. 1, p. 9, 2025.
[21] G. Kim and A. Kim, “Scan context: Egocentric spatial descriptor for
place recognition within 3d point cloud map,” in 2018 IEEE/RSJ IROS.
IEEE, 2018, pp. 4802–4809.
[22] L. Homeyer, C. Begiristain and C. Schn¨orr, “Droid-splat: Com-
bining
end-to-end
slam
with
3d
gaussian
splatting,”
preprint
arXiv:2411.17660, 2024.
[23] Z. Xin, C. Wu, P. Huang, Y. Zhang, Y. Mao, and G. Huang, “Large-
scale gaussian splatting slam,” preprint arXiv:2505.09915, 2025.
[24] E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald, “Point-
slam: Dense neural point cloud-based slam,” in Proceedings of the
IEEE/CVF ICCV, 2023, pp. 18 433–18 444.
