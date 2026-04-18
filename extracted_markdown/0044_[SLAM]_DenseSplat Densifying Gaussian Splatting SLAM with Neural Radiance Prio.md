<!-- page 1 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
1
DenseSplat: Densifying Gaussian Splatting SLAM with Neural
Radiance Prior
Mingrui Li∗, Shuhong Liu∗, Tianchen Deng, Hongyu Wang†
Abstract—Gaussian SLAM systems excel in real-time render-
ing and fine-grained reconstruction compared to NeRF-based
systems. However, their reliance on extensive keyframes is im-
practical for deployment in real-world robotic systems, which
typically operate under sparse-view conditions that can result
in substantial holes in the map. To address these challenges,
we introduce DenseSplat, the first SLAM system that effectively
combines the advantages of NeRF and 3DGS. DenseSplat utilizes
sparse keyframes and NeRF priors for initializing primitives that
densely populate maps and seamlessly fill gaps. It also implements
geometry-aware primitive sampling and pruning strategies to
manage granularity and enhance rendering efficiency. Moreover,
DenseSplat integrates loop closure and bundle adjustment, sig-
nificantly enhancing frame-to-frame tracking accuracy. Exten-
sive experiments on multiple large-scale datasets demonstrate
that DenseSplat achieves superior performance in tracking and
mapping compared to current state-of-the-art methods.
Index Terms—Visual Dense SLAM, Neural Radiance Field, 3D
Gaussian Splatting
I. INTRODUCTION
Visual Dense Simultaneous Localization and Mapping
(SLAM) is a core area of study in 3D computer vision,
focusing on real-time localization of the camera and generating
a fine-grained map of the surrounding environment. It plays a
crucial role in robot localization and navigation, autonomous
vehicles, and Virtual/Augmented Reality (VR/AR).
Recent breakthroughs in differential rendering, specifically
with Neural Radiance Fields (NeRF) [1]–[3] and 3D Gaussian
Splatting (3DGS) [4]–[6], have significantly advanced visual
dense SLAM systems [7]–[12]. NeRF-based neural implicit
SLAM [13]–[17] integrates the NeRF model for simultaneous
tracking and mapping, which facilitates high-quality, dense
online map reconstruction and remarkably improves geometric
accuracy. Building on this, Gaussian-based SLAM systems
[18]–[24] have pushed the boundaries by delivering higher-
fidelity map reconstruction. By utilizing explicit Gaussian
primitives, 3DGS provides substantial benefits in terms of de-
tailed texture representation [19], explicit scene manipulation
[25], and remarkable real-time rendering capabilities [24].
However, 3DGS faces particular challenges within the con-
text of SLAM systems. Unlike offline reconstruction often
∗First two authors have contributed equally to this work.
† Corresponding author: whyu@dlut.edu.cn
Mingrui Li and Hongyu Wang are with the Department of Computer
Science, Dalian University of Technology, Dalian, China. Shuhong Liu
is with the Department of Mechano-informatics, Information Science and
Technology, The University of Tokyo, Tokyo, Japan. Tianchen Deng is with
Institute of Medical Robotics and Department of Automation, Shanghai Jiao
Tong University, and Key Laboratory of System Control and Information
Processing, Ministry of Education. This work was partially supported by JST
SPRING, Grant Number JPMJSP2108.
Fig. 1. DenseSplat leverages NeRF priors into the Gaussian SLAM system,
offering superior tracking, fine-grained mapping, and extraordinary real-time
performance using sparse keyframes.
applied to exhaustive image collections [14], [26], [27], online
SLAM systems are typically deployed in complex environ-
ments with insufficient observations, leading to partially ob-
served or obstructed views. This shortage significantly affects
the completeness of the Gaussian map, as it struggles to
effectively interpolate missing geometry of unobserved areas.
Moreover, current Gaussian SLAM systems rely on per-frame
pixel backprojection from the input stream, which fails to cap-
ture the 3D structure of the scene. Such 2D-based initialization
results in a map representation where foreground elements
are detailed with denser primitives, while the backgrounds,
encompassing broader scene geometries, are less defined.
Figure 2 illustrates these limitations, where imbalanced point
distribution, erroneous depth projection, and undersampling
due to occlusions lead to an uneven and challenging opti-
mization landscape for multiview geometry [28]. Given these
limitations, current Gaussian SLAM systems often maintain a
dense keyframe list, such as one in every four frames, during
the mapping process that attempts to engage more viewpoints
for reliable map optimization [29]. This reliance on a dense
keyframe list demands extensive memory and reduces real-
time processing capabilities, both of which are crucial for
online SLAM systems.
To address these challenges, we propose employing NeRF
priors to densify the Gaussian SLAM system. This densifi-
cation is achieved through two primary mechanisms using
the NeRF model: (1) its interpolation capabilities efficiently
close the gaps in the map with densely positioned Gaussian
primitives, and (2) it offers a robust initialization of Gaus-
sian primitives that can be densified using extremely sparse
keyframes. Moreover, NeRF-based sampling ensures an even
distribution of Gaussian primitives aligned with the scene
arXiv:2502.09111v2  [cs.CV]  6 Jan 2026

<!-- page 2 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
2
Fig. 2.
Gaussian primitives initialized by direct backprojection from RGB-D streams suffer from several drawbacks. The left pixel image illustrates that
closer objects occupy more pixels, whereas farther regions receive fewer rays. (a) Imbalanced point sampling occurs due to the uneven distribution of rays.
(b) Erroneous projections arise from inaccurate depth estimates in distant regions. (c) Undersampled areas result from occlusions or insufficient views.
geometry and allows for manageable granularity in the scene
representation through specific sampling ratios. Coupled with
loop closure and bundle adjustment (BA), our method delivers
state-of-the-art performance across large-scale synthetic and
real-world datasets as illustrated in Figure 1. Overall, our
contributions can be summarized as follows:
• We propose DenseSplat, the first SLAM system combin-
ing the advantages of NeRF and 3DGS, capable of real-
time tracking, mapping, and loop correction using only
sparsely sampled keyframes.
• By leveraging NeRF priors, DenseSplat effectively fills
the gaps from unobserved or obstructed viewpoints.
• We implement geometry-aware primitive sampling and
pruning strategies that control the granularity of the
Gaussian representation and prune inactive primitives,
ensuring a high-fidelity map and real-time processing.
• We compare our method with state-of-the-art (SOTA)
approaches on multiple datasets, including a challenging
large-scale apartment dataset, where DenseSplat achieves
superior performance.
II. RELATED WORK
A. Neural Dense SLAM
In contrast to traditional SLAM systems [30]–[36] that
utilize point-clouds or voxels for sparse map representation,
neural dense SLAM systems [37]–[46] offer substantial advan-
tages through their dense neural radiance maps, providing a
robust foundation for downstream tasks in robotics and AR/XR
[47]. iMAP [48] pioneered neural implicit SLAM but suffers
from large tracking and mapping error using a single MLP.
NICE-SLAM [49] employs multiple MLPs for coarser-to-finer
mapping, effectively filling the gaps in reconstruction. ESLAM
[50] leverages tri-plane features for efficient scene represen-
tation, while Co-SLAM [51] employs multi-resolution hash-
grids for real-time performance. Point-SLAM [52] relies on
neural point clouds for dense scene reconstruction, and Loopy-
SLAM [53] introduces map corrections to address scene drift
caused by accumulated tracking errors. However, maps recon-
structed using NeRF typically lack the quality seen in more
recent systems built upon 3D Gaussian Splatting. Furthermore,
high-resolution NeRF models often require extensive training
and exhibit slower real-time rendering performance [4], which
significantly reduces their practical efficiency.
B. Gaussian Splatting SLAM
Propelled by advancements in 3DGS [4], recent Gaussian
SLAM systems [18], [24], [25], [29], [54] have shown re-
markable capabilities in high-fidelity map reconstruction and
efficient real-time rendering. Notably, SplaTAM [19] utilizes
isotropic Gaussian representation coupled with dense point-
cloud sampling to ensure geometric precision. Conversely,
MonoGS [20] employs anisotropic Gaussians to accelerate
map reconstruction and enhance texture rendering. Neverthe-
less, the frame-to-frame tracking mechanisms of these systems
do not incorporate loop closure or bundle adjustment, which
leads to significant tracking discrepancies in real-world envi-
ronments. Concurrently, Photo-SLAM [21] and RTG-SLAM
[22], which integrate feature-based tracking [31], [33] with
dense Gaussian maps, achieve superior tracking accuracy and
real-time performance. However, they take the trade-off of
diminished rendering quality due to the sparse sampling of
scene representation. Moreover, systems such as Gaussian-
SLAM [55] and LoopSplat [56] propose to implement submap
division and fusion strategies to tackle the high memory
consumption of fine-grained Gaussian maps. While Gaussian
SLAM systems offer superior rendering quality compared to
NeRF models, the explicit and discrete nature of their scene
representation often results in substantial gaps and holes in
the reconstructed map due to unobserved or obstructed views
commonly seen in online systems. These deficiencies severely
affect their efficiency in real-world applications.
C. Gaussian Splatting with Neural Radiance Prior
Vanilla 3DGS [4] utilizes the sparse point cloud derived
from Structure-from-Motion [57], [58], whose reconstruction
quality heavily relies on the accuracy of the initial point
cloud [59]. As an alternative, RadSplat [28] introduces NeRF
model [1] into the Gaussian Splatting framework, aiming
to achieve robust real-time rendering of complex real-world
scenes. RadSplat [28] specifically employs NeRF as a prior
for initialization and supervision, enabling fast training con-
vergence and enhanced quality.
Compared to RadSplat [28], our method differs fundamen-
tally in task objectives and methods. RadSplat [28] focuses
on offline reconstruction scenarios where datasets typically
consist of 360-degree or dense viewpoint coverage [26], [60],
making completeness and rendering quality the primary goals.
In contrast, our method addresses the challenges of real-time

<!-- page 3 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
3
Fig. 3. DenseSplat comprises tracking and mapping modules. The tracking module computes camera poses by optimizing the NeRF model and streaming
sparse keyframes to the mapping module. Gaussian primitives are produced via geometry-aware sampling, effectively capturing the scene geometry and
seamlessly filling gaps. Enhanced by BA-induced map refinement and ray-guided Gaussian pruning strategies, DenseSplat delivers high-quality reconstructions
at remarkable real-time speeds.
SLAM systems deployed in robotic systems, where the limited
sensor field of view and navigation path lead to sparse views
and sequential observations. These factors frequently result in
unobserved or partially observed geometry due to obstacles,
posing significant challenges for explicit Gaussian represen-
tations that often leave critical gaps in the reconstruction. To
overcome these issues, we leverage the NeRF prior not only
for initialization but as a robust interpolation mechanism to
interpolate unobserved regions and enable real-time adapt-
ability in environments with sparse views. Moreover, unlike
RadSplat [28] that naively samples one million points from
all cast rays, we propose a geometry-aware sampling strategy
that operates directly in 3D space. This approach uniformly al-
locates primitives across object surfaces, effectively mitigating
the imbalanced sampling caused by foreground bias.
III. METHODS
Figure 3 illustrates the overall pipeline of DenseSplat.
Starting with an RGB-D stream {Ii, Di}N
i=1 of N frames,
tracking is initiated by simultaneously optimizing the camera
pose and the neural radiance fields fθ (Section III-A). We then
initialize Gaussian primitives guided by points sampled from
the implicit radiance fields for fine-grained map reconstruction
and scene interpolation (Section III-B). To mitigate drift errors,
we implement a local loop closure detection and bundle
optimization strategy on the Gaussian map (Section III-C).
Finally, Section III-D explains the overall mapping loss and
our submap division strategy that effectively reduces memory
consumption in the system. Further details on each component
are discussed in the subsequent sections.
A. Neural Radiance Prior
Preliminaries of Neural Radiance Rendering The NeRF
model fθ [1] is a continuous function that predicts colors
c ∈R3 and volume density σ ∈R+ along the sampled rays r.
Specifically, given a camera origin o ∈R3 and ray direction
v, by uniformly sampling points x = o + djv|j∈{1...}, the
pixel color CN can be rendered by NeRF using ray marching:
CN =
X
j=1
cjαjTNj,
and
TNj =
j−1
Y
t=1
(1 −αt) ,
(1)
where TNi denotes the transmittance and αi = 1 −e−σiδi is
the alpha values at point xi. δj represents the spacing between
successive sample points. The radiance filed fθ, parameterized
as an MLP with ReLU activation, is trained using gradient
descent to minimize the photometric loss:
LN =
X
r∈Rb
Cθ
N(r) −CGT(r)
2 .
(2)
In this context, r ∈Rb denotes a batch of rays drawn from
the entire set of rays that have valid depth measurements,
CGT is the ground-truth color.
NeRF-based Camera Tracking We track the camera-to-world
transformation matrix Twc ∈SE(3) by optimizing the pose
using the objective function of NeRF defined in Equation (2).
The pose is initialized based on the principle of constant
velocity:
Ti = Ti−1T−1
i−2Ti−1 .
(3)
To enable tracking, we first bootstrap the NeRF model using a
small number of initial frames, whose poses are estimated via
the constant velocity assumption. These early frames provide
rough supervision to begin learning coarse scene geometry
and appearance. Once initialized, the NeRF is incrementally
updated as more frames are integrated. For each new frame
i, a two-stage optimization strategy is adopted following the
conventional NeRF-based SLAM system [51]. We begin with
pose-only optimization while keeping the NeRF parameters
fixed. After the pose stabilizes, we perform joint optimization

<!-- page 4 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
4
of both the camera pose and NeRF weights. The NeRF model
is used exclusively for per-frame camera tracking. Once
tracking is complete, loop closure and bundle adjustment are
applied based on the fine-grained Gaussian map to correct
cumulative drift, as detailed in Section III-C.
Geometry-aware
Point
Sampling Compared to explicit
Gaussian representations that require dense viewpoints for
thorough scene optimization, NeRF-based models offer re-
markable advantages in terms of interpolation, as they can
deduce unseen geometry [49], [61], [62]. To harness this
capability for efficient real-time performance, we utilize a
multi-resolution hash radiance field as proposed by [63].
This grid system enables precise interrogation of volumetric
data across various resolutions, effectively capturing detailed
surface geometries even in sparsely sampled areas. We identify
critical surface transitions by setting a density threshold, τgrid,
which examines each grid edge to detect transitions where the
density at one corner surpasses τgrid, while at its adjacent
corner, it does not. At these transitions, we interpolate the
surface crossing points using:
x = x1 + τgrid −σ(x1)
σ(x2) −σ(x1) · (x2 −x1) .
(4)
Here, x1 and x2 represent sampled grid points, and σ de-
notes volumn density. We then compile these points into a
point cloud. This approach leverages the robust interpolation
capabilities of NeRF to provide dense and geometry-aware
initialization of Gaussian primitives, which are further refined
in subsequent mapping steps.
B. Fine-grained Gaussian Map
Multi-scale Gaussian Rendering By initializing through grid
sampling from th NeRF model, we represent the scene using
a set of anisotropic 3D Gaussian primitives {Gj} [4]. Each
primitive Gj is defined by a mean µj ∈R3, a covariance
matrix Σj, an opacity value αj ∈[0, 1], coefficients for third-
order spherical harmonics SHj ∈R16, and a scaling factor
sj ∈R3. During the rendering process, these primitives are
first projected onto a 2D plane, transforming them into 2D
Gaussians. The transformation utilizes a viewing matrix W,
and the resulting 2D covariance matrix Σ′
j in the image space
is computed as follows [64]:
Σ′
j = (JWΣjW T JT )1:2,1:2 ,
(5)
where J represents the Jacobian of an affine projection ap-
proximation. The mean µ′
j of the 2D Gaussian is derived by
projecting µj onto the image plane using W. Subsequently,
these projected Gaussians are sorted from the nearest to the
farthest and rendered using an alpha-blending process akin to
Equation (1). This results in rasterized pixel color CG and
depth value DG:
CG =
X
j=1
cjαjTGj
and
DG =
X
j=1
djαjTGj ,
(6)
where αj denotes the blending weight, dj represents the depth
of each Gaussian relative to the image plane, and TGj is the
transmittance, calculated similar to Equation (1), using the
opacity αj of each Gaussian Gj.
The radiance field enables interpolated point initialization
for Gaussian primitives, yet it can also lead to aliasing effects
that diminish the quality of the map during sampling [14],
[65]. This issue becomes particularly acute at the edges of
objects where small Gaussian floaters result in significant
artifacts. Drawing inspiration from [66], we adopt a multi-
scale Gaussian rendering strategy that consolidates smaller
Gaussians into larger ones to enhance scene consistency.
Specifically, we use Gaussian functions across four levels
of detail, corresponding to down-sampling resolutions of
1×, 4×, 16×, and 64×. Throughout the training phase, we
merge smaller, fine-level Gaussians into larger, coarse-level
Gaussians.
The
selection
of
Gaussians
for
merging
is
determined by pixel coverage, which picks Gaussians based
on the coverage range set by the inverse of the highest
frequency component in the region, denoted as fmax = 1/sj,
where sj is the scaling factor.
Ray-Guided Gaussian Pruning Sampling from the NeRF
model may also introduce erroneous Gaussian floaters and
outliers that affect the quality of the reconstruction. To reduce
redundant Gaussians produced by the densification process and
enhance rendering efficiency, we implement a pruning strategy
based on sampled rays from the NeRF model. Specifically, we
use an importance assessment to identify and remove inactive
Gaussians from the map during optimization. The importance
of each Gaussian is quantified based on its contribution
to sampled rays across all input images {Ii}N
i=1. Drawing
inspiration from [28], we implement the score function for
each primitive as:
E(Gi) = max
r∈Ii (αr
i T r
Gi) ,
(7)
where αr
i T r
Gi captures the Gaussian Gi’s contribution to the
final color prediction of a pixel along the ray r. We then
compute a pruning mask:
M(Gi) = 1(E(Gi) < τprune) ,
(8)
where primitives under a pruning threshold τprune ∈[0, 1] are
removed from the map. Note that the Gaussians initialized by
the NeRF model are exempt from this pruning process to avoid
removing those that bridge the gaps for obstructed viewpoints
and to ensure a manageable granularity.
C. Loop Closure and Bundle Adjustment
In the bundle adjustment (BA) process, we use the Bag of
Words (BoW) model [33] to determine the relevance between
keyframes. Upon detecting a loop, this triggers a BA procedure
for the involved keyframes, in a manner akin to [53], [56]. The
implementation is detailed in Algorithm 1.
While BA refines the camera trajectories, the dense map
cannot trivially incorporate these updates and remains syn-
chronized with the pre-optimized poses, which leads to spatial
drift. To maintain the geometric and visual consistency of the
map after the BA process, we adjust the color CG⟨Ti⟩and

<!-- page 5 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
5
Algorithm 1 Loop Closure and Bundle Adjustment
1: Input: Keyframe stream {Ik, Dk, Tk}, BoW database B
2: Output: Updated keyframe poses {Tk}
3: for each new keyframe (Ik, Dk, Tk) do
4:
// Add keyframe to BoW database
5:
B ←B ∪{Ik}
6:
// Retrieve loop closure candidates
7:
{j} ←QueryBoW(B, Ik)
8:
for each candidate j do
9:
if LoopGeometricCheck(Ik, Ij) then
10:
// Loop closure confirmed
11:
Tk→j ←EstimateRelativePose(Ik, Ij)
12:
// Collect covisible keyframes
13:
Klocal ←GetCovisibleKeyframes(k, j)
14:
// Pose-only Bundle Adjustment
15:
{Tp} ←RunPoseBA(Klocal)
16:
// Update keyframe poses
17:
for each p ∈Klocal do
18:
Tp ←OptimizedPose(p)
19:
end for
20:
end if
21:
end for
22: end for
depth DG⟨Ti⟩rendered at current pose Ti to warp the co-
visible keyframe pose Tk using the estimated relative pose
transformation ⟨Ti→k⟩. We then construct the BA-induced
mapping loss using the following equations:
LC
BA =
N−1
X
k=1
N
X
i=k+1
∥CG⟨Ti→k⟩−CG⟨Tk⟩∥
(9)
LD
BA =
N−1
X
k=1
N
X
i=k+1
∥DG⟨Ti→k⟩−DG⟨Tk⟩∥
(10)
Leveraging the rapid rendering capabilities of 3D Gaussian
Splatting, our method supports real-time re-rendering and the
correction of drift errors.
D. Gaussian Map Optimization
Mapping Objective Function In our experiments, we observe
that the aggregated Gaussians can experience scale explosion
during the BA process, potentially introducing artifacts into the
map. To mitigate this issue, we introduce an L2 regularization
loss Lreg for Gaussian primitives whose scales exceed a
threshold of τscale = 1. The overall mapping loss is thereby
defined as:
LG = λc∥CG −CGT∥+ λd∥DG −DGT∥
+ λssimSSIM(CG, CGT) + λregLreg ,
(11)
where CGT and DGT denote the ground-truth color and depth
from the input stream. The SSIM loss [67] calculates the
structural similarity between the rendered and ground-truth
images. The coefficients λc, λd, λssim, and λreg are weighting
hyperparameters.
Algorithm 2 Submap Division and Fusion Strategy with
Submap Bundle Adjustment
1: Input: RGB-D sequences {Ii, Di} and poses {Ti}
2: Output:
3:
Submaps {SF⟨fθ,G⟩}
4:
keyframe set {Ii, Di, Ti} ∈ΩSF
5:
anchor-frame set {Ii, Di, Ti} ∈Λ
6: Initiate frame index i ←0 and submap index j ←0
7: repeat
8:
// Insert Keyframes:
9:
if CheckKeyframe({Ii, Di, Ti}) then
10:
ΩSFj ←InsertKeyFrame(Ii, Di, Ti)
11:
end if
12:
// NeRF Submap Creation:
13:
if CheckSubmapCreation(i) then
14:
SFj
fθ ←CreateNeRFSubmap(Ii, Di, Ti)
15:
Λ ←InsertAnchorFrame(Ii, Di, Ti)
16:
end if
17:
// Gaussian Submap Creation:
18:
if Length(Λ) > j then
19:
SFj
G ←RenderGaussianSubmap(SFj
fθ, ΩSFj)
20:
// Gaussian Submaps BA:
21:
{SF⟨fθ,G⟩} ←AnchorFrameBA(Λ, {SFn
⟨fθ,G⟩|n≤j})
22:
j ←j + 1
23:
end if
24:
i ←i + 1
25: until All frames are processed
Submap Division and Fusion When deploying a SLAM
system in large-scale environments, managing the excessive
memory consumption of the dense mapping is critical for
practical applications. To address this issue, we employ an
effective submap division and fusion strategy presented in Al-
gorithm 2. Specifically, we partition input frames into submaps
at intervals of every 400 frames, structured as follows:
{Ii, Di}N
i=1 7→
n
SF1
⟨f 1
θ ,G1⟩, SF2
⟨f 2
θ ,G2⟩, . . . , SFn
⟨f n
θ ,Gn⟩
o
,
(12)
where SFn
⟨fθ,G⟩represents each submap used to develop NeRF
models and subsequent Gaussian maps. While the explicit
Gaussian representation enables the seamless combination of
submaps into a global map, directly fusing submaps remains
a challenging task. Drawing inspiration from Mipsfusion [69],
we utilize anchor-frame BA during our submap fusion process
to achieve precise alignment and seamless fusion at submap
boundaries. Each submap is anchored based on the estimated
pose of its first frame. Following BA, we precisely adjust the
central pose of each submap to ensure accurate re-anchoring.
Our submap strategy effectively reduces memory consumption
by enabling the parallel construction of each submap, thereby
mitigating the issues associated with the continuous expansion
of the global map.
IV. EXPERIMENTS
Implementation Details The experiments were conducted
on a single NVIDIA A100 GPU with 80 GB of VRAM.

<!-- page 6 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
6
TABLE I
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN TRAINING VIEW RENDERING ON THE REPLICA DATASET [68]. THE UNDERLINE
INDICATES THAT RELOCALIZATION WAS TRIGGERED DUE TO ACCUMULATED TRACKING ERRORS IN THE ORB TRACKING SYSTEM [33]. DASH PRESENTS
SYSTEM FAILURE.
Methods
Metrics room0
room1
room2
office0
office1
office2
office3
office4
apart0
apart1
apart2
frl0
frl4
Avg.
PSNR↑
28.88
28.51
29.37
35.44
34.63
26.56
28.79
32.16
30.10
22.86
23.29
23.52
25.33
28.42
NeRF-SLAM
Co-SLAM [51]
SSIM↑
0.892
0.843
0.851
0.854
0.826
0.814
0.866
0.856
0.905
0.766
0.771
0.822
0.814
0.837
LPIPS↓
0.213
0.205
0.215
0.177
0.181
0.172
0.163
0.176
0.321
0.440
0.462
0.367
0.461
0.273
PSNR↑
31.80
32.70
32.70
38.66
15.96
15.00
33.61
34.15
33.71
23.74
25.96
25.55
34.55
29.08
Loopy-SLAM [53]
SSIM↑
0.912
0.914
0.917
0.960
0.088
0.583
0.921
0.935
0.927
0.833
0.842
0.889
0.930
0.819
LPIPS↓
0.167
0.198
0.205
0.126
0.369
0.538
0.202
0.187
0.248
0.293
0.288
0.195
0.226
0.249
PSNR↑
32.40
34.08
35.50
38.26
39.16
33.99
33.48
33.49
34.95
32.27
33.31
36.01
34.87
34.75
Point-SLAM [52]
SSIM↑
0.974
0.977
0.982
0.983
0.986
0.960
0.960
0.979
0.970
0.929
0.944
0.960
0.970
0.965
LPIPS↓
0.113
0.116
0.111
0.100
0.118
0.156
0.132
0.142
0.153
0.205
0.211
0.156
0.176
0.147
PSNR↑
32.49
33.72
34.65
38.29
39.04
31.91
30.05
31.83
13.12
24.57
28.52
31.82
32.71
30.98
SplaTAM [19]
SSIM↑
0.975
0.970
0.980
0.982
0.982
0.965
0.952
0.949
0.415
0.821
0.883
0.930
0.945
0.904
LPIPS↓
0.072
0.096
0.078
0.086
0.093
0.100
0.110
0.150
0.656
0.302
0.241
0.164
0.184
0.179
PSNR↑
29.57
31.61
33.46
38.39
39.62
32.91
33.62
34.26
27.07
24.93
24.34
25.19
26.70
30.90
Gauss-SLAM [55]
SSIM↑
0.944
0.952
0.973
0.985
0.991
0.974
0.982
0.979
0.864
0.850
0.828
0.836
0.839
0.923
LPIPS↓
0.197
0.184
0.148
0.099
0.097
0.158
0.123
0.138
0.345
0.381
0.410
0.358
0.343
0.229
PSNR↑
30.71
33.51
35.02
38.47
39.08
33.03
33.78
36.02
29.07
22.73
24.59
34.16
33.36
32.58
Gaussian-SLAM
Photo-SLAM [21]
SSIM↑
0.899
0.934
0.951
0.964
0.961
0.938
0.938
0.952
0.922
0.796
0.848
0.940
0.932
0.921
LPIPS↓
0.075
0.057
0.043
0.050
0.047
0.077
0.066
0.054
0.227
0.293
0.354
0.115
0.129
0.122
PSNR↑
34.83
36.43
37.49
39.95
42.09
36.24
36.70
36.07
22.91
26.88
27.93
31.72
27.98
33.63
MonoGS [20]
SSIM↑
0.954
0.959
0.965
0.971
0.977
0.964
0.963
0.957
0.864
0.835
0.836
0.886
0.873
0.923
LPIPS↓
0.068
0.076
0.075
0.072
0.055
0.078
0.065
0.099
0.385
0.284
0.272
0.225
0.245
0.154
PSNR↑
31.56
34.21
35.57
39.11
40.27
33.54
32.76
36.48
-
29.08
29.14
33.88
-
34.14
RTG-SLAM [22]
SSIM↑
0.967
0.979
0.981
0.990
0.992
0.981
0.981
0.984
-
0.900
0.909
0.933
-
0.963
LPIPS↓
0.131
0.105
0.115
0.068
0.075
0.134
0.128
0.117
-
0.232
0.233
0.181
-
0.138
PSNR↑
37.11
36.64
36.32
39.97
41.69
37.84
37.52
40.76
38.44
37.31
37.37
40.66
41.89
38.73
Ours
SSIM↑
0.958
0.954
0.956
0.986
0.977
0.976
0.966
0.985
0.972
0.945
0.951
0.986
0.988
0.969
LPIPS↓
0.065
0.060
0.062
0.055
0.053
0.074
0.064
0.049
0.057
0.055
0.054
0.046
0.038
0.056
The keyframe interval was set to 20. The tracking and
mapping iterations for the NeRF model are both set to
10, and the mapping iteration for the Gaussian map is 30.
Parameters of the NeRF model and Gaussian primitives were
optimized using the Adam optimizer, with the learning rate
following the settings in [63] and [4]. The initial pose is
estimated purely using NeRF with a tracking iteration of
10. For mapping, NeRF undergoes 10 iterations, while the
Gaussian map is optimized over 30 iterations for fine-grained
reconstruction. The keyframe intervals for both NeRF and
Gaussian models are 20 frames for the Replica [68] and
TUM RGB-D [70] datasets, and 25 frames for the ScanNet
dataset [71]. For NeRF, we employ a sampling ratio of 64
in our geometry-warping initialization. The density threshold
τgrid for selecting transmission points is set at 0.001. In the
ray-guided pruning process, the pruning threshold τprune
is set at 0.001 to optimize memory and computational
resources. During map optimization, the regularization scale
threshold τscale is maintained at 1.0, where scales exceeding
this threshold are penalized. The weighting hyperparameters
λc, λd, λssim, and λreg are set at 0.8, 0.1, 0.2, and 0.001
respectively, promoting consistency across various metrics
including color accuracy, depth precision, structural similarity,
and regularization.
Datasets and Metrics We conduct comprehensive evaluations
using
both
synthesized
and
real-world
scenes
from
Replica [68], ScanNet [71], and TUM RGB-D [70] datasets
(supplementary material). In addition, we use four large-scale
apartment scenes from the Replica dataset [68], with the
largest scene containing up to two floors and eight rooms,
presenting challenging indoor layouts with complex corridors
and stairs. To evaluate tracking accuracy, we employ ATE
RMSE (cm). For reconstruction results on training views, we
adhere to PSNR, SSIM, and LPIPS to quantitatively evaluate
rendering quality. To qualitatively compare mapping quality,
we visualize the reconstruction outcomes from novel views.
The running speed and computation usage are assessed using
FPS and GPU consumption. Best results are shaded as first ,
second , and third .
Baseline Methods We compare DenseSplat with NeRF-based
RGB-D SLAM systems, including Co-SLAM [51], Point-
SLAM [52], Loopy-SLAM [53]; as well as recent 3DGS-based
systems such as SplaTAM [19], MonoGS [20], Gaussian-
SLAM [55], Photo-SLAM [21], and RTG-SLAM [22]. To
ensure a fair comparison, all results are assessed based on
the final global maps produced by the SLAM systems.
A. Evaluation of Tracking and Mapping
Evaluations on Replica Dataset We quantitatively assessed
the rendering quality of our method against NeRF-based and
Gaussian-based SLAM systems on the Replica dataset [68],
as presented in Table I. DenseSplat exhibits competitive ren-
dering quality in single-room scale scenes, achieving state-
of-the-art performance in some instances. Moreover, our
method substantially outperforms baseline approaches in large-
scale, multi-room environments, such as apartment scenes.
Specifically, Gaussian SLAM baselines struggle due to inef-
fective bundle adjustment and cumbersome map representa-
tions, which do not adequately address the extensive camera

<!-- page 7 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
7
Fig. 4. Novel-view synthesis (NVS) comparison of DenseSplat and Gaussian-based baseline methods on the selected scenes of Replica apartment dataset [68].
Our method demonstrates superior performance in geometric accuracy, hole filling, and fine-grained texture rendering. Crucially, DenseSplat utilizes a sparse
keyframe (kf) interval of 20, offering a more efficient and practical setup compared to the dense keyframe lists employed by the baseline methods.
TABLE II
EVALUATION OF AVERAGED TRACKING PERFORMANCE, SYSTEM FRAME
RATE, MEMORY CONSUMPTION, AND MAP SIZE ACROSS THE EIGHT
SCENES OF THE REPLICA DATASET [68]. WE SEPARATELY PRESENT THE
SYSTEMS THAT UTILIZE ORB-SLAM [31], [33] AND FRAME-TO-FRAME
TRACKING.
Methods
ATE
[cm]↓
Track.
FPS↑
Map.
FPS↑
System
FPS↑
Memory
[GB]↓
Size
[MB]↓
ORB
Photo-SLAM [21]
0.59
41.64
30.36
20.71
8.02
59
RTG-SLAM [22]
0.49
50.33
20.09
17.30
10.18
71
Co-SLAM [51]
1.12
10.2
10.0
9.26
7.90
7
Frame-to-frame
Point-SLAM [52]
0.54
0.95
0.87
0.44
18.86
154
Loopy-SLAM [53]
0.29
0.95
0.87
0.43
18.91
177
SplaTAM [19]
0.55
0.86
0.51
0.51
11.27
331
MonoGS [20]
0.58
3.41
3.16
3.09
13.88
42
Ours
0.33
7.96
23.35
7.67
6.67
117
movement and complex geometry characteristic of large-scale
scenes. NeRF baselines provide robust tracking and efficient
implicit scene representation but do not match our method in
detailed reconstruction and rendering quality. DenseSplat inte-
grates the strengths of both NeRF and Gaussian maps, utilizing
the robust NeRF prior to stabilize scene representation. In
addition, our submap strategy facilitates parallel computation
of maps, significantly reducing memory consumption that
often leads to system failures in Gaussian systems, such
as SplaTAM [19] in apartment 0. The novel-view render-
ing results, shown in Figure 4, qualitatively compare our
reconstructed map with Gaussian-based systems. DenseSplat
demonstrates superior mapping quality compared to baselines
that suffer from significant scene drift and floaters.
To assess tracking accuracy and system efficiency, we
compared the average ATE and frame rates as detailed
in
Table
II.
DenseSplat
delivers
competitive
tracking
performance while achieving the highest mapping frame
rate among frame-to-frame systems. Additionally, our map
division strategy results in the lowest memory consumption
Fig. 5.
NVS comparison on real-world ScanNet dataset [71]. DenseSplat
shows superior geometry accuracy and hole filling.
during runtime compared to the baseline methods.
Evaluations on ScanNet Dataset We present quantitative
evaluations of our method alongside the baseline approaches
using 6 scenes from the ScanNet dataset [71]. As shown in
Table III, we evaluate rendering quality by comparing PSNR,
SSIM, and LPIPS metrics across training views. In real-world
environments, challenges on noisy input such as depth map
errors and blurred images introduce significant difficulties
for map initialization and optimization. Additionally, recent
Gaussian-based methods like SplaTAM [19] and MonoGS
[20], which do not incorporate loop-closure, experience con-
siderable loop-induced map drift that remarkably reduces
reconstruction quality. In contrast, our method utilizes the
robust NeRF prior and integrates both the bundle adjustment

<!-- page 8 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
8
Fig. 6. Visualization of NVS of the Replica dataset [68]. The rendered views are shown along with plots for the center of primitives, depicted as blue dots
of the same size across all methods. For our method, we explicitly present the NVS at keyframe intervals (kf) at 4 and 20. Looking at objects like floors and
tables, DenseSplat demonstrates more evenly distributed primitives compared to baseline methods.
TABLE III
QUANTITATIVE COMPARISON OF OUR METHOD AND THE BASELINES IN
TRAINING VIEW RENDERING ON THE SCANNET DATASET [71]. OUR
METHOD DEMONSTRATES SOTA
PERFORMANCES.
Methods
Metrics 0000 0059 0106 0169 0181 0207
Avg.
PSNR↑17.81 19.60 19.23 20.55 16.76 17.95 18.65
SplaTAM [19]
SSIM↑0.602 0.796 0.741 0.785 0.683 0.705 0.719
LPIPS↓0.467 0.290 0.322 0.260 0.420 0.346 0.351
PSNR↑18.62 15.56 14.97 18.07 14.98 18.52 16.79
RTG-SLAM [22]
SSIM↑0.756 0.682 0.726 0.772 0.750 0.773 0.743
LPIPS↓0.468 0.531 0.480 0.451 0.492 0.459 0.480
PSNR↑19.74 20.01 18.70 20.23 14.45 19.96 18.85
Photo-SLAM [21]
SSIM↑0.761 0.799 0.760 0.781 0.698 0.765 0.761
LPIPS↓0.412 0.284 0.297 0.288 0.521 0.315 0.353
PSNR↑20.72 20.06 21.20 22.32 22.29 22.80 21.56
Gauss-SLAM [55]
SSIM↑0.702 0.728 0.785 0.764 0.774 0.769 0.754
LPIPS↓0.568 0.496 0.460 0.458 0.544 0.505 0.505
PSNR↑19.76 19.25 20.18 20.57 20.25 20.62 20.11
MonoGS [20]
SSIM↑0.772 0.767 0.785 0.790 0.788 0.798 0.783
LPIPS↓0.387 0.289 0.272 0.256 0.282 0.295 0.297
PSNR↑25.31 24.74 25.33 24.98 23.61 25.29 24.88
Ours
SSIM↑0.832 0.875 0.866 0.862 0.845 0.847 0.855
LPIPS↓0.206 0.211 0.195 0.215 0.250 0.198 0.212
(BA) and subsequent BA-induced map refinement modules,
achieving state-of-the-art performance. Table IV also presents
the tracking performance using ATE RMSE (cm) on the Scan-
Net dataset [71]. Our method demonstrates superior tracking
accuracy compared to Gaussian baseline methods [19], [20]
that do not incorporate the BA process. When comparing with
Photo-SLAM [21] and RTG-SLAM [22], which incorporate
point-based tracking systems derived from ORB-SLAM [31],
[33], DenseSplat achieves lower ATE errors, benefiting from
its robust tracking based on the NeRF model.
Figure 5 shows the novel-view synthesis of representative
TABLE IV
THE PERFORMANCE OF ATE RMSE (CM) ON 6 SCENES FROM THE
SCANNET [71] DATASET. FOR OUR METHOD, WE PROVIDE RESULTS BOTH
WITH AND WITHOUT BUNDLE ADJUSTMENT (BA).
Methods
0000
0059
0106
0169
0181
0207
Avg.
ORB
Photo-SLAM [21]
7.62
7.94
9.36
10.01
22.97
7.23
10.85
RTG-SLAM [22]
8.04
6.82
9.22
10.15
24.36
9.25
11.31
ESLAM [50]
7.54
8.52
7.39
8.17
9.13
5.61
7.73
Co-SLAM [51]
7.13
11.14
9.36
5.90
11.81
7.14
8.75
Frame-to-frame
Loopy-SLAM [53]
4.28
7.59
8.37
7.56
10.68
7.95
7.70
SplaTAM [19]
12.83
10.10
17.72
12.08
11.10
7.47
11.88
MonoGS [20]
15.94
6.41
19.44
10.44
12.23
10.46
12.48
Ours (w/o BA)
7.94
10.85
7.62
6.91
9.20
8.33
8.48
Ours
7.25
9.60
7.44
6.58
8.71
7.21
7.80
scenes from the ScanNet dataset [71]. DenseSplat offers
superior geometric accuracy, for instance, accurately capturing
the details of the bicycle in scene 0000, which often suffers
from scene drift due to trajectory loops. Compared to Photo-
SLAM [21], our method demonstrates comparable tracking
performance while providing more fine-grained and complete
map reconstruction.
Evaluations on TUM RGB-D Dataset We also conducted
experiments on the TUM RGB-D dataset [70] and compared
the ATE RMSE (cm) with both feature-based and frame-
to-frame tracking systems in Table V. Our system achieves
performance comparable to current state-of-the-art methods.
B. Evaluation of Scene Densification
Figure 6 illustrates the novel-view synthesis of Dens-
eSplat and Gaussian-based SLAM systems on the Replica
dataset [68]. The obstructed views typical of indoor environ-

<!-- page 9 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
9
Fig. 7.
Visualization of hole-filling in novel-view synthesis for the scene Room 0, Office 2, and Office 3 from the Replica dataset [68]. Employing the
NeRF prior, our method effectively interpolates the unobserved geometry obscured by obstacles in the room, contrasting with Gaussian-based approaches that
exhibit significant gaps, drastically compromising the completeness of the reconstruction.
TABLE V
THE PERFORMANCE OF ATE RMSE (CM) ON THE TUM RGB-D DATASET
[70]. FOR OUR METHOD, WE PROVIDE RESULTS BOTH WITH AND
WITHOUT BUNDLE ADJUSTMENT (BA).
Methods
fr1 desk
fr2 xyz
fr3 office
Avg.
ORB
Photo-SLAM [21]
2.61
0.35
1.00
1.32
RTG-SLAM [22]
1.66
0.38
1.13
1.06
ESLAM [50]
2.30
1.10
2.40
2.00
Co-SLAM [51]
2.70
1.90
2.60
2.40
Frame-to-frame
Loopy-SLAM [53]
3.79
1.62
3.41
2.94
SplaTAM [19]
3.35
1.24
5.16
3.25
MonoGS [20]
1.52
1.58
1.65
1.58
Ours (w/o BA)
2.02
0.97
2.48
1.82
Ours
2.02
0.86
1.86
1.58
ments pose remarkable challenges for Gaussian SLAM sys-
tems. For instance, SplaTAM [19] and MonoGS [20] exhibit
notable deficiencies with holes in their reconstructed Gaussian
maps. Although Photo-SLAM [21] mitigates these gaps using
a pyramid feature extraction strategy, it results in sparser map
representations, which diminish rendering quality. In contrast,
DenseSplat employs the robust NeRF prior for Gaussian
initialization, effectively filling these gaps by interpolating un-
observed geometry. Our method also ensures a more uniform
distribution of Gaussian primitives across the surface of the
objects using our point sampling strategy, leading to better
geometric alignment. Furthermore, Figure 6 also compares the
rendering results of our method at a dense keyframe interval
of 4, typical of current Gaussian SLAM systems, with a much
sparser interval of 20. Despite the map being slightly less
populated, DenseSplat continues to demonstrate robust recon-
struction and interpolation capabilities. This effectiveness in
sparse conditions significantly enhances real-time processing
TABLE VI
THE MESH EVALUATION OF OUR METHOD AND THE GAUSSIAN SLAM
BASELINES ON THE REPLICA DATASET [68]. THE RESULTS ARE
AVERAGED OVER 8 SCENES.
Methods
Accuracy↓
Completion↓
Comp. Ratio ↑
NeRF
NICE-SLAM [49]
2.85
3.02
89.34
Co-SLAM [51]
2.10
2.08
93.44
ESLAM [50]
0.97
1.05
98.60
3DGS
SplaTAM [19]
2.74
4.02
84.89
MonoGS [20]
3.16
4.45
81.52
Photo-SLAM [21]
2.53
3.75
85.67
Ours
2.18
2.01
94.64
and, more importantly, relaxes the stringent requirement for
dense multi-view observations previously essential in Gaussian
SLAM systems. Figure 7 provides a broader view of the
Replica scenes, showcasing the superior map completeness
of our method compared to baseline approaches. The latter
often display significant gaps due to common occlusions
encountered in indoor environments.
Moreover, to quantitatively evaluate the reconstruction
completeness, we transform Gaussian maps into meshes
using TSDF-fusion [72] and compare them with ground-
truth meshes. As demonstrated in Table VI, DenseSplat
significantly outperforms recent Gaussian SLAM systems in
accuracy and scene completion ratio, which often struggle
with gaps that result in underrepresented scenes.
V. ABLATION STUDY
In this section, we conduct extensive ablation studies on
each hyperparameter and core component of our system
in terms of rendering quality, scene accuracy, and system

<!-- page 10 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
10
Fig. 8. Visualization of the ablation study on Gaussian primitive initialization approaches for Room0 of the Replica dataset [68]. For direct backprojection,
we backproject the RGB-D frames and downsample by ×1 to initialize primitives. For native NeRF-based sampling, we uniformly sample NeRF using a
grid size of 0.01. For our geometry-aware sampling, we use τgrid = 0.001 and the same sampling grid size for initialization. Our strategy achieves superior
geometric accuracy compared to direct backprojection, which exhibits substantial holes, and the naive NeRF-based method, which produces many redundant
floating primitives.
TABLE VII
ABLATION STUDY OF EACH COMPONENT OF DENSESPLAT ON ROOM0 OF
THE REPLICA DATASET [68].
Ablation Method
PSNR
[dB]↑
Render.
FPS↑
Map.
FPS↑
Num. of
G [K]↓
Memory
[GB]↓
w/o pruning
32.01
19.96
6.94
97.98
10.84
w/o BA loss
36.66
23.87
8.76
40.98
6.09
w/o multi-scale GS
33.64
20.09
7.99
55.91
9.67
w/o submap
36.60
22.46
7.01
39.40
9.86
Ours
37.11
23.99
8.08
37.52
6.68
TABLE VIII
ABLATION STUDY ON THE INITIALIZATION OF PRIMITIVES G,
CONDUCTED ON ROOM0 FROM THE REPLICA DATASET [68]. FOR THE
BACKPROJECTION-BASED INITIALIZATION METHODS SHOWN IN THE
MIDDLE ROWS, WE DIRECTLY BACKPROJECT PIXELS FROM THE RGB-D
INPUT FRAME, USING DOWNSAMPLING RATIOS OF ×1, ×8, ×16.
Initialization Method
PSNR
[dB]↑
Render.
FPS↑
Map.
FPS↑
Num.
G [K]↓
Memory
[GB]↓
Random initialized G
18.19
117.47
10.12
190.64
17.13
Backprojected G (down ×1)
36.26
39.96
8.33
512.04
23.49
Backprojected G (down ×8)
33.27
57.13
11.56
339.17
20.39
Backprojected G (down ×16)
31.69
92.17
15.77
205.82
18.96
Naive NeRF sampling
34.12
16.54
6.64
289.36
21.31
Geometry-aware sampling (ours)
37.11
23.99
8.08
37.52
6.68
efficiency. The overall ablation analysis is presented in
Table VII. By taking the tradeoff of rendering and mapping
efficiency, our method capitalizes on the benefits of high-
quality rendering using the NeRF model.
Ablation of NeRF-based Initialization To evaluate the effec-
tiveness of our geometry-aware sampling strategy, Table VIII
presents an ablation study on different primitive initialization
strategies. We pay particular attention to the method of direct
backprojection from RGB-D streams, which is commonly
applied in previous Gaussian-based SLAM systems [19], [20],
[55]. Specifically, the baseline approaches include random
primitive initialization, direct backprojection of RGB-D frames
downsampled by ×1, ×8, and ×16, and a naive NeRF-based
sampling approach that uniformly samples NeRF without
removing redundant points. The ablation results shows that,
by actively selecting surface transition points, our geometry-
aware point sampling strategy achieves optimal rendering
quality while consuming the least memory.
Furthermore, Figure 8 shows novel-view synthesis results
TABLE IX
ABLATION STUDY OF THRESHOLDING HYPERPARAMETERS, CONDUCTED
ON ROOM0 FROM THE REPLICA DATASET [68]. NOTE THAT τgrid = 0 IS
EQUIVALENT TO NAIVE NERF-BASED INITIALIZATION.
Sampling Strategy
PSNR
[dB]↑
Render.
FPS↑
Map.
FPS↑
Num.
G [K]↓
Memory
[GB]↓
τgrid = 0
34.12
16.54
6.64
289.36
21.31
τgrid = 0.001 (ours)
37.11
23.99
8.08
37.52
6.68
τgrid = 0.005
36.58
23.07
7.61
55.97
7.74
τgrid = 0.01
35.56
20.15
6.70
78.63
9.87
τprune = 0
31.97
20.12
6.21
299.36
21.67
τprune = 0.001 (ours)
37.11
23.99
8.08
37.52
6.68
τprune = 0.01
35.19
31.91
11.54
21.23
5.08
τscale = 0.01
35.15
17.77
5.34
160.77
11.25
τscale = 0.1
37.20
20.94
6.89
70.61
9.34
τscale = 1 (ours)
37.11
23.99
8.08
37.52
6.68
for Room0 using direct backprojection, naive NeRF-based
sampling, and our adaptive strategy. Direct backprojection
leaves substantial holes due to insufficient or occluded
views, while naive NeRF-based sampling introduces floating
primitives due to the lack of filtering. Our method delivers
a
complete
reconstruction,
effectively
filling
gaps
and
maintaining geometric accuracy.
Ablation of Thresholding Hyperparameters We investigate
three thresholding hyperparameters in Table IX: τgrid for
NeRF sampling, τprune for primitive pruning, and τscale for
scale regularization. Setting τgrid = 0 is equivalent to naive
NeRF sampling. We employ a small sampling threshold
τgrid
= 0.001 to effectively remove erroneous sampling
points from the NeRF prior, improving geometry accuracy
and system efficiency. The parameter τprune controls the
removal of primitives with negligible radiance contribution;
a small threshold brings certain improvement in quality
and efficiency, while a larger value remarkably accelerates
rendering but degrades performance by pruning too many
primitives. The parameter τscale penalizes excessively large
Gaussian primitives to suppress view-dependent artifacts,
especially
in
sparse-view
settings.
Smaller
thresholds
encourage finer primitives at the cost of increased memory
usage, whereas τscale = 1 offers a balanced trade-off between
primitive granularity and overall efficiency.
Ablation of Sampling Size Although precise control over

<!-- page 11 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
11
Fig. 9.
Ablation study on sampling grid size, showing rendered images of
Room0 from the Replica dataset [68], with primitive’s centers µ shaded in
blue. (a) Sampling size of 0.01 (b) Sampling size of 0.001.
TABLE X
ABLATION STUDY OF SAMPLING GRID SIZE DURING NERF SAMPLING,
CONDUCTED ON ROOM0 FROM THE REPLICA DATASET [68].
Hyperparameters
PSNR
[dB]↑
Render.
FPS↑
Map.
FPS↑
Num.
G [K]↓
Memory
[GB]↓
grid size = 0.001
36.28
21.15
6.93
69.66
9.27
grid size = 0.01 (ours)
37.11
23.99
8.08
37.52
6.68
grid size = 0.1
35.44
27.26
9.88
25.91
5.37
the number of Gaussian primitives is difficult due to the
densification process [4], adjusting the sampling grid size of
the NeRF model can roughly control the granularity of the
scene representation. We perform an ablation study on the
sampling grid size, as shown in Table X. The corresponding
visualization in Figure 9 shows that, compared to our choice
of 1 cm, increasing the sampling grid size by a factor of
10 produces a sparser primitive distribution and reduces
rendering quality to some extent.
Ablation of Keyframe Interval As discussed in Section IV-B,
the NeRF model provides exceptional interpolation capabilities
and robust initialization, enabling the integration of a sparse
keyframe list, which significantly reduces the computation
time. However, excessively large intervals eventually result
in insufficient supervision, thereby lowering reconstruction
quality. Figure 10 examines the trend of mapping FPS and
rendering PSNR in relation to keyframe intervals, showing
a decline beyond an interval of 40. Consequently, we balance
this trade-off by using a keyframe interval of 20 in our system.
Fig. 10. Ablation study on keyframe intervals, presenting the mapping FPS
and rendering PSNR as functions of keyframe intervals on the Room0 of the
Replica dataset [68].
Ablation of Bundle Adjustment Figure 11 visualizes the
rendering results with and without BA and the refinement
using BA loss. In contrast to the baseline method that suffers
from substantial scene drift, incorporating loop closure in
Fig. 11. Ablation study on BA, showing the rendered scenes of Apartment1
from the Replica dataset [68]. (a) baseline method employing naive frame-
to-frame tracking. (b) no loop closure and BA on the tracking system. (c) no
BA loss defined in Equation (9) and Equation (10). (d) Full implementation
of DenseSplat.
Fig. 12. Visualization of hole-filling failures in novel-view synthesis for the
scene Apartment 2 from the Replica dataset [68]. The camera trajectory is
highlighted in red on the ground-truth map. Our method and Gaussian-based
approaches struggle to completely fill the notably large gaps. In contrast,
NeRF-based systems like Co-SLAM [51] manage to fill these gaps to some
extent, though the results are not sufficiently smooth.
tracking and BA-driven refinement in mapping successfully
tackles this challenge.
Analysis of an Interpolation Failure Case When the uncov-
ered regions from viewpoints become extensively large, our
method still faces challenges in completely filling holes. As
illustrated in Figure 12, in the case of Apartment 2 from the
Replica dataset [68], the limitations become apparent when the
camera trajectory does not sufficiently cover the scene, leaving
nearly half of the room uncaptured. This situation makes
scene interpolation particularly challenging. The NeRF-based

<!-- page 12 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
12
Fig. 13. Visulization of the VR/AR demo on Office 3 of the Replica dataset
[68]. We introduce the reconstructed LEGO object from the NeRF Synthetic
Dataset [1] into the real-time reconstructed map and show the novel-view
synthesis result.
model, such as Co-SLAM [51], manages to fill some gaps but
introduces erroneous surfaces with significant artifacts. Our
method outperforms other Gaussian-based methods by a large
margin, using sparse primitives to fill gaps; however, it still
leaves certain areas underrepresented. In scenarios where un-
observed geometry primarily results from insufficient camera
coverage, inferring these unobserved regions aligns more with
scene extrapolation. Our method faces certain limitations as
the NeRF prior is less reliable for extrapolation, and addressing
these challenges will be a key direction for future research.
VI. VR/AR APPLICATIONS
Neural dense SLAM is well-suited for VR/AR applications
[17], where the essential synthesis of novel views enables
seamless and dynamic perspectives that adapt to user move-
ments. Despite the substantial advantages offered by Gaussian-
based systems, such as producing high-fidelity maps and
efficient real-time rendering capabilities, they often display
significant holes and gaps when confronted with limited
viewpoint supervision, which can severely impact the user
experience. As illustrated in Figure 13, we place the LEGO
object reconstructed from the NeRF Synthetic Dataset [1] into
the dense Gaussian map generated by SLAM systems. In com-
parison to previous methods like MonoGS [20], our method
remarkably improves scene completeness by interpolating
missing geometry, thereby enhancing the user’s immersion and
interaction within the virtual environment.
VII. CONCLUSION
We propose DenseSplat, the first visual dense SLAM
system that seamlessly integrates the strengths of NeRF
and 3DGS for robust tracking and mapping. DenseSplat
targets practical challenges including obstructed views and
the impracticality of maintaining dense keyframes, driven by
hardware limitations and computational costs. This strategic
integration enhances the system’s ability to interpolate missing
geometries and robustly optimize Gaussian primitives with
fewer keyframes, leading to fine-grained scene reconstructions
and extraordinary real-time performance. Future research
of this study could focus on implementing the system in
practical mobile applications or multi-agent collaboration
systems and conducting further experiments in real-world
environments.
Limitation DenseSplat also has its limitations. As discussed in
Section IV-B, its scene interpolation capabilities are dependent
on the NeRF model, and thus inherit NeRF’s limitations.
When the missing areas become excessively large and the
NeRF model cannot adequately capture the geometry, both our
method and NeRF struggle to extrapolate the scene geometry,
leaving some regions underrepresented on the reconstructed
map. Although recent efforts using generative priors show po-
tential, extrapolation under realistic and large-scale conditions
remains an open challenge. Additionally, because DenseSplat
employs explicit Gaussian primitives for scene representation,
storing the high-fidelity map requires more memory compared
to NeRF models, which use neural implicit representations.
Moreover, while DenseSplat supports submap systems to
minimize memory usage during computation, there is room
for a more advanced submap strategy that could potentially
be expanded in real-world multi-agent systems. Addressing
these limitations will be the focus of future research.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[2] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Doso-
vitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields
for unconstrained photo collections,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2021, pp. 7210–
7219.
[3] S. Liu, L. Gu, Z. Cui, X. Chu, and T. Harada, “I2-nerf: Learning
neural radiance fields under physically-grounded media interactions,” in
Advances in Neural Information Processing Systems (NeurIPS), 2025.
[4] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Transactions on
Graphics (TOG), vol. 42, no. 4, pp. 139–1, 2023.
[5] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler,
“Wildgaussians: 3d gaussian splatting in the wild,” Advances in Neural
Information Processing Systems, 2024.
[6] S. Liu, X. Chen, H. Chen, Q. Xu, and M. Li, “Deraings: Gaussian
splatting for enhanced scene reconstruction in rainy environments,”
arXiv preprint arXiv:2408.11540, 2024.
[7] T. Schops, T. Sattler, and M. Pollefeys, “Bad slam: Bundle adjusted
direct rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2019, pp. 134–144.
[8] J. Huang, S.-S. Huang, H. Song, and S.-M. Hu, “Di-fusion: Online
implicit 3d reconstruction with deep priors,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 8932–8941.
[9] M. R¨unz and L. Agapito, “Co-fusion: Real-time segmentation, tracking
and fusion of multiple objects,” in IEEE International Conference on
Robotics and Automation (ICRA).
IEEE, 2017, pp. 4471–4478.
[10] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison,
“Codeslam—learning a compact, optimisable representation for dense
visual slam,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2018, pp. 2560–2568.
[11] R. Craig and R. C. Beavis, “Tandem: matching proteins with tandem
mass spectra,” Bioinformatics, vol. 20, no. 9, pp. 1466–1467, 2004.
[12] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular,
stereo, and rgb-d cameras,” Advances in Neural Information Processing
Systems, vol. 34, pp. 16 558–16 569, 2021.
[13] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance
fields from one or few images,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp.
4578–4587.

<!-- page 13 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
13
[14] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, 2021, pp. 5855–5864.
[15] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised
nerf: Fewer views and faster training for free,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 12 882–12 891.
[16] Y.-C. Guo, D. Kang, L. Bao, Y. He, and S.-H. Zhang, “Nerfren:
Neural radiance fields with reflections,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
18 409–18 418.
[17] H. Zhai, G. Huang, Q. Hu, G. Li, H. Bao, and G. Zhang, “ NIS-
SLAM: Neural Implicit Semantic RGB-D SLAM for 3D Consistent
Scene Understanding ,” IEEE Transactions on Visualization & Computer
Graphics, vol. 30, no. 11, pp. 7129–7139, 2024.
[18] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “Gs-
slam: Dense visual slam with 3d gaussian splatting,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 19 595–19 604.
[19] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “Splatam: Splat track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 21 357–21 366.
[20] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting
slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024, pp. 18 039–18 048.
[21] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Real-
time simultaneous localization and photorealistic mapping for monocular
stereo and rgb-d cameras,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 21 584–21 593.
[22] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, “Rtg-
slam: Real-time 3d reconstruction at scale using gaussian splatting,” in
ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
[23] S. Liu, H. Zhou, L. Li, Y. Liu, T. Deng, Y. Zhou, and M. Li, “Struc-
ture gaussian slam with manhattan world hypothesis,” arXiv preprint
arXiv:2405.20031, 2024.
[24] T. Deng, Y. Chen, L. Zhang, J. Yang, S. Yuan, J. Liu, D. Wang, H. Wang,
and W. Chen, “Compact 3d gaussian splatting for dense visual slam,”
arXiv preprint arXiv:2403.11247, 2024.
[25] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, “Sgs-
slam: Semantic gaussian splatting for neural dense slam,” in European
Conference on Computer Vision.
Springer, 2025, pp. 163–179.
[26] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics (TOG), vol. 36, no. 4, 2017.
[27] S. Liu, C. Bao, Z. Cui, Y. Liu, X. Chu, L. Gu, M. V. Conde, R. Umagami,
T. Hashimoto, Z. Hu et al., “Realx3d: A physically-degraded 3d
benchmark for multi-view visual restoration and reconstruction,” arXiv
preprint arXiv:2512.23437, 2025.
[28] M. Niemeyer, F. Manhardt, M.-J. Rakotosaona, M. Oechsle, D. Duck-
worth, R. Gosula, K. Tateno, J. Bates, D. Kaeser, and F. Tombari,
“Radsplat: Radiance field-informed gaussian splatting for robust real-
time rendering with 900+ fps,” arXiv preprint arXiv:2403.13806, 2024.
[29] S. Ha, J. Yeon, and H. Yu, “Rgbd gs-icp slam,” in European Conference
on Computer Vision.
Springer, 2025, pp. 180–197.
[30] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: a versatile
and accurate monocular slam system,” IEEE Transactions on Robotics,
vol. 31, no. 5, pp. 1147–1163, 2015.
[31] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam
system for monocular, stereo, and rgb-d cameras,” IEEE Transactions
on Robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
[32] J. Zhang, M. Gui, Q. Wang, R. Liu, J. Xu, and S. Chen, “ Hierarchical
Topic Model Based Object Association for Semantic SLAM ,” IEEE
Transactions on Visualization & Computer Graphics, vol. 25, no. 11,
pp. 3052–3062, 2019.
[33] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D.
Tard´os, “Orb-slam3: An accurate open-source library for visual, visual–
inertial, and multimap slam,” IEEE Transactions on Robotics, vol. 37,
no. 6, pp. 1874–1890, 2021.
[34] Z.-J. Du, S.-S. Huang, T.-J. Mu, Q. Zhao, R. R. Martin, and K. Xu,
“Accurate dynamic slam using crf-based long-term consistency,” IEEE
Transactions on Visualization and Computer Graphics, vol. 28, no. 4,
pp. 1745–1757, 2022.
[35] C.-M. Chung, Y.-C. Tseng, Y.-C. Hsu, X.-Q. Shi, Y.-H. Hua, J.-F. Yeh,
W.-C. Chen, Y.-T. Chen, and W. H. Hsu, “Orbeez-slam: A real-time
monocular visual slam with orb features and nerf-realized mapping,”
in IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2023, pp. 9400–9406.
[36] X. Pan, G. Huang, Z. Zhang, J. Li, H. Bao, and G. Zhang, “ Robust
Collaborative Visual-Inertial SLAM for Mobile Augmented Reality ,”
IEEE Transactions on Visualization & Computer Graphics, vol. 30,
no. 11, pp. 7354–7363, 2024.
[37] Z. Yan, M. Ye, and L. Ren, “ Dense Visual SLAM with Probabilistic
Surfel Map ,” IEEE Transactions on Visualization & Computer Graph-
ics, vol. 23, no. 11, pp. 2389–2398, 2017.
[38] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global
optimization for consistent 3d instant reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 3727–3737.
[39] M. Li, J. He, Y. Wang, and H. Wang, “End-to-end rgb-d slam with
multi-mlps dense neural implicit representations,” IEEE Robotics and
Automation Letters, 2023.
[40] S.-S. Huang, H. Chen, J. Huang, H. Fu, and S.-M. Hu, “ Real-Time
Globally Consistent 3D Reconstruction With Semantic Priors ,” IEEE
Transactions on Visualization & Computer Graphics, vol. 29, no. 04,
pp. 1977–1991, 2023.
[41] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and
W. Chen, “Plgslam: Progressive neural scene represenation with local to
global bundle adjustment,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 19 657–19 666.
[42] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and
M. Pollefeys, “Nicer-slam: Neural implicit scene encoding for rgb slam,”
in International Conference on 3D Vision (3DV).
IEEE, 2024, pp. 42–
52.
[43] H. Zhou, Z. Guo, Y. Ren, S. Liu, L. Zhang, K. Zhang, and M. Li,
“Mod-slam: Monocular dense mapping for unbounded 3d scene recon-
struction,” IEEE Robotics and Automation Letters, vol. 10, no. 1, pp.
484–491, 2025.
[44] T. Deng, Y. Wang, H. Xie, H. Wang, J. Wang, D. Wang, and W. Chen,
“Neslam: Neural implicit mapping and self-supervised feature tracking
with depth completion and denoising,” arXiv preprint arXiv:2403.20034,
2024.
[45] W. Xie, G. Chu, Q. Qian, Y. Yu, H. Li, D. Chen, S. Zhai, N. Wang,
H. Bao, and G. Zhang, “ Depth Completion with Multiple Balanced
Bases and Confidence for Dense Monocular SLAM ,” IEEE Transactions
on Visualization & Computer Graphics, no. 01, pp. 1–12, 2025.
[46] S. Liu, T. Deng, H. Zhou, L. Li, H. Wang, D. Wang, and M. Li, “Mg-
slam: Structure gaussian splatting slam with manhattan world hypothe-
sis,” IEEE Transactions on Automation Science and Engineering, 2025.
[47] T. Deng, S. Liu, X. Wang, Y. Liu, D. Wang, and W. Chen, “Prosgnerf:
Progressive dynamic neural scene graph with frequency modulated auto-
encoder in urban scenes,” arXiv preprint arXiv:2312.09076, 2023.
[48] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit mapping and
positioning in real-time,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 6229–6238.
[49] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 786–12 796.
[50] M. M. Johari, C. Carta, and F. Fleuret, “Eslam: Efficient dense slam
system based on hybrid representation of signed distance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 17 408–17 419.
[51] H. Wang, J. Wang, and L. Agapito, “Co-slam: Joint coordinate and
sparse parametric encodings for neural real-time slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 293–13 302.
[52] E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald, “Point-slam:
Dense neural point cloud-based slam,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 18 433–18 444.
[53] L. Liso, E. Sandstr¨om, V. Yugay, L. Van Gool, and M. R. Oswald,
“Loopy-slam: Dense neural slam with loop closures,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 20 363–20 373.
[54] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and
Z. Cui, “Cg-slam: Efficient dense rgb-d slam in a consistent uncertainty-
aware 3d gaussian field,” in European Conference on Computer Vision.
Springer, 2025, pp. 93–112.
[55] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam:
Photo-realistic dense slam with gaussian splatting,” arXiv preprint
arXiv:2312.10070, 2023.

<!-- page 14 -->
IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS
14
[56] L. Zhu, Y. Li, E. Sandstr¨om, K. Schindler, and I. Armeni, “Loop-
splat: Loop closure by registering 3d gaussian splats,” arXiv preprint
arXiv:2408.10154, 2024.
[57] C. Bregler, A. Hertzmann, and H. Biermann, “Recovering non-rigid 3d
shape from image streams,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, vol. 2.
IEEE, 2000, pp.
690–696.
[58] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2016, pp. 4104–4113.
[59] J. Jung, J. Han, H. An, J. Kang, S. Park, and S. Kim, “Relaxing ac-
curate initialization constraint for 3d gaussian splatting,” arXiv preprint
arXiv:2403.09413, 2024.
[60] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 5470–5479.
[61] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-
nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 10 318–10 327.
[62] C. Yang, P. Li, Z. Zhou, S. Yuan, B. Liu, X. Yang, W. Qiu, and W. Shen,
“Nerfvs: Neural radiance fields for free view synthesis via geometry
scaffolds,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 16 549–16 558.
[63] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions on
Graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[64] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Surface splatting,”
in Proceedings of the 28th annual conference on Computer graphics
and interactive techniques, 2001, pp. 371–378.
[65] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Zip-nerf: Anti-aliased grid-based neural radiance fields,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 19 697–19 705.
[66] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-splatting:
Alias-free 3d gaussian splatting,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
19 447–19 456.
[67] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[68] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel,
R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital
replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.
[69] Y. Tang, J. Zhang, Z. Yu, H. Wang, and K. Xu, “Mips-fusion: Multi-
implicit-submaps for scalable and robust online neural rgb-d reconstruc-
tion,” ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1–16,
2023.
[70] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A
benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ
International Conference on Intelligent Robots and Systems, 2012, pp.
573–580.
[71] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor
scenes,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2017, pp. 5828–5839.
[72] B. Curless and M. Levoy, “A volumetric method for building complex
models from range images,” in Proceedings of the 23rd annual confer-
ence on Computer graphics and interactive techniques, 1996, pp. 303–
312.
Mingrui Li is currently pursuing the Ph.D. degree
with the School of Information and Communica-
tion Engineering, Dalian University of Technology,
Dalian, China. His research interests include 3D re-
construction, simultaneous localization and mapping
(SLAM), and computer vision.
Shuhong Liu is currently pursuing the Ph.D. degree
in Department of Mechano-informatics, Information
Science and Technology with the University of
Tokyo, Tokyo, Japan. Before that, he received his
bachelor’s degree in Department of Electrical and
Computer Engineering at University of Waterloo,
Ontario, Canada, and the master’s degree in Creative
Informatics, Information Science and Technology,
the University of Tokyo, Tokyo, Japan. His research
interests include 3D computer vision, visual SLAM,
and computational photography.
Tianchen Deng is currently pursuing the Ph.D. de-
gree in control science and engineering with Shang-
hai Jiao Tong University, Shanghai, China. His main
research interests include 3D Reconstruction, long-
term visual simultaneous localization and mapping
(SLAM), and vision-based localization.
Hongyu Wang (Member, IEEE) received his B.S.
degree in electronic engineering from the Jilin Uni-
versity of Technology, Changchun, China, in 1990,
the M.S. degree in electronic engineering from the
Graduate School, Chinese Academy of Sciences,
Beijing, China, in 1993, and the Ph.D. degree in
precision instrument and optoelectronics engineering
from Tianjin University, Tianjin, China, in 1997. He
is currently a Professor with the Dalian University
of Technology, Dalian, China. His research interests
include image processing, computer vision, 3-D re-
construction, and simultaneous localization and mapping (SLAM).
