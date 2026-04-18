<!-- page 1 -->
GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels
Yongxin Su1
Lin Chen2
Kaiting Zhang1
Zhongliang Zhao1,B
Chenfeng Hou1
Ziping Yu1
1 Beihang University
2 Northwestern Polytechnical University
https://gaus-slam.github.io
SOTA: 1.91 cm
Ours:  0.42 cm
Rendering Quality (PSNR[dB])  
Geometry Accuracy (Depth L1[cm])
Replica Performance
Tracking Performance (ATE-RMSE[cm])
SOTA: 28.14
Ours:  30.32
Ours: 0.62
SOTA: 1.64
Better
Figure 1. GauS-SLAM is a dense SLAM system using 2D Gaussian surfels[8], capable of simultaneously achieving high-precision localiza-
tion and high-fidelity reconstruction. As shown in the left figure, GauS-SLAM exhibits millimeter-level tracking accuracy on a challenging
real-world scenario(b20a261fdf in ScanNet++[33] dataset), significantly outperforming the SOTA approach, SplaTAM[12]. The right
figure demonstrates GauS-SLAM’s SOTA performance on the Replica[22] dataset, achieving an absolute trajectory error(ATE-RMSE) of
0.06cm and 40.25 dB in rendering quality.
Abstract
We propose GauS-SLAM, a dense RGB-D SLAM system
that leverages 2D Gaussian surfels to achieve robust track-
ing and high-fidelity mapping.
Our investigations re-
veal that Gaussian-based scene representations exhibit ge-
ometry distortion under novel viewpoints, which signifi-
cantly degrades the accuracy of Gaussian-based tracking
methods.
These geometry inconsistencies arise primar-
ily from the depth modeling of Gaussian primitives and
the mutual interference between surfaces during the depth
blending. To address these, we propose a 2D Gaussian-
based incremental reconstruction strategy coupled with a
Surface-aware Depth Rendering mechanism, which signifi-
cantly enhances geometry accuracy and multi-view consis-
tency. Additionally, the proposed local map design dynam-
ically isolates visible surfaces during tracking, mitigating
misalignment caused by occluded regions in global maps
while maintaining computational efficiency with increasing
Gaussian density. Extensive experiments across multiple
datasets demonstrate that GauS-SLAM outperforms compa-
rable methods, delivering superior tracking precision and
rendering fidelity. The project page will be made available
at https://gaus-slam.github.io.
1. Introduction
Over the past decade, dense visual simultaneous local-
ization and mapping (SLAM) has remained a fundamen-
tal research area in computer vision. Recent advances in
map representation have increasingly focused on integrat-
ing neural models with traditional 3D features—such as
points, voxels, and surfels—enabling more flexible and ac-
curate map construction. Despite these innovations, current
methods still face considerable challenges in areas like pose
optimization, convergence difficulties, and catastrophic for-
getting during continual learning.
Explicit representation using 3D Gaussian Splatting[13]
(3DGS) has shown promising results in both 3D reconstruc-
tion and dense SLAM. Pioneering works [12, 14, 31, 35]
propose tracking and mapping pipelines based on Gaus-
sian representation. However, these Gaussian-based track-
1
arXiv:2505.01934v1  [cs.CV]  3 May 2025

<!-- page 2 -->
ing approaches often suffer from inaccurate pose estima-
tion and convergence problems. In contrast, some advances
[2, 3, 7, 9, 39] address these issues by decoupling the track-
ing from the Gaussian model and leveraging mature odom-
etry methods [16, 26]. While this decoupled design im-
proves real-time performance, it inherently lacks the mutual
reinforcement between reconstruction and localization that
a coupled system could offer.
In this paper, we focus on two critical challenges en-
countered in coupled Gaussian-based SLAM frameworks,
as illustrated in Fig. 2. The first issue is geometry distor-
tion. In most Gaussian-based tracking methods, the camera
transformation is estimated by aligning the observation with
the rendering result from the current viewpoint. During this
process, perspective-induced geometry distortion leads to
misalignment between the frame and the Gaussian model,
consequently degrading the tracking accuracy.
We attribute geometry distortion to two primary factors.
First, inherent inconsistencies exist in the Gaussian-based
depth representation model, where the center depth model
of 3D Gaussian primitives exhibits multi-view inconsistent
depth estimations (as visualized in Fig. 2(a1)), whereas
2D Gaussian surfels effectively address this inconsistency
through intersection depth model [8]. The second arises
from the mutual interference between different surfaces dur-
ing depth blending. As illustrated in Fig. 2(a2), when ren-
dering the backrest of the chair under reconstruction, the
distant floor with greater depth results in ill-blended depth.
To resolve depth blending ambiguities, we propose an in-
cremental reconstruction strategy based on Gaussian sur-
fels [8], coupled with a Surface-aware Depth Rendering
scheme, which significantly enhances the geometry accu-
racy and view consistency of the Gaussian scene.
The second challenge lies in the outlier removal dur-
ing the alignment between frames and the Gaussian model.
As demonstrated by SplaTAM[12], outlier elimination is
crucial, and their approach achieves this by masking re-
gions with low accumulated opacity. However, as shown in
Fig.2(b), interference regions with high accumulated opac-
ity remain challenging to mask, especially when the camera
moves around the object. Our approach confines camera
tracking to a small local map, thereby isolating these in-
terference regions from the global map. Additionally, by
periodically resetting the local map, we ensure that camera
tracking always operates within a subset of Gaussian prim-
itives, avoiding the degradation of tracking efficiency as the
number of Gaussians increasing.
We propose GauS-SLAM, a dense SLAM system that
leverages 2D Gaussian surfels within a tightly coupled
front-end/back-end framework to address these challenges,
achieving superior localization accuracy and view synthesis
quality on RGB-D datasets. Our contributions are as fol-
lows:
Geometry distortion
(a1)                                 (a2)                                          (b)
ill-blended depth
opacity
under 
construction
interference area
t-1
t
Misalignment
t+1
t
alignment
inconsistent depth under t+1
depth under 
view t
Figure 2. Two challenges in Gaussian-based tracking methods.
(a1) illustrates geometry distortions caused by center depth model
of the 3D Gaussian. (a2) shows ill-blended depth arising from
depth rendering between different surfaces. (b) demonstrates that,
during the alignment, certain interference area exhibit high accu-
mulated opacity making them challenging to be masked out as out-
liers.
• We propose a 2D Gaussian-based incremental reconstruc-
tion strategy and Surface-aware Depth Rendering mech-
anism that effectively mitigates geometry distortions and
improves tracking accuracy.
• We propose a dense SLAM system with a front-end/back-
end architecture, and incorporates a local map design that
ensures tracking accuracy and efficiency.
• We conduct extensive experiments that demonstrate the
superiority of our approach, both in tracking accuracy and
in reconstruction quality, compared to SOTA methods.
2. Related Works
Neural based dense SLAM
Leveraging the powerful
modeling capability of deep neural networks, [10, 28, 32]
utilize implicit representation of 3D scenes through neu-
ral voxel and neural grid frameworks.
iSDF[17] inte-
grates neural networks with traditional Signed Distance
Functions, enabling the adaptive adjustment of detail lev-
els. iMap[25] demonstrates that the MLP-based neural im-
plicit representation can be effectively utilized for localiza-
tion and reconstruction in dense SLAM. [11, 41] employ
NeRF[15] to represent 3D scenes, and introduce hierar-
chical multi-feature grids for coarse-to-fine optimization of
camera poses. NeRF-SLAM [19] integrates the end-to-end
DROID-SLAM[26] with NeRF, achieving accurate pose es-
timation and efficient reconstruction. Point-SLAM[20] rep-
resents scenes as neural point clouds, enhancing the abil-
ity to capture details near scene surfaces. Considering that
rendering implicit neural radiance fields requires sampling
along rays, the slow rendering speed limits its applicability
in practical scenarios.
2

<!-- page 3 -->
Tracking
Local map
Surface-aware depth rendering
Frontend
Backend
Reference Keyframe (RKF)
Keyframe
Frame
Gaussian Surfel 
Database
Percentage of 
increment > 
The number of Gaussian
primitives in local map > 
RGB-D data
is
？
is
？
gt
n
α-blending
×
GT surface
depth
Occluded surface
Ours
accumulated 
opacity 
t
R,
Mapping
Reset Local 
map
Send  to backend
Merging
Bundle
Adjustment
Random
Submaps
current
RKF
 Depth adjustment
Submap
Stored as 
a submap
Random Optimization
(
）
Local map
Global map
Pose of RKF
Local map
1
Figure 3. Overview of GauS-SLAM. This framework consists of a front-end that performs tracking and mapping using a single local map,
and a back-end responsible for merging the local map into the global map and submap-based global optimization.
3DGS based dense SLAM
3D Gaussian primitives offer
a concise and flexible ellipsoid representation compared to
Point-NeRF[29]. For camera pose estimation, some 3DGS-
based approaches [3, 14, 31] implement a renderer that is
differentiable with respect to camera pose. SplaTAM[12]
and Gaussian-SLAM[35] consider that the gradient of the
camera pose can be approximated as the gradient of the
Gaussian pose relative to the camera pose.
Other dense
SLAM systems [2, 9, 18, 39] incorporate traditional feature-
based visual odometry to balance real-time performance
with precision. DROID-SLAM[26] and Splat-SLAM[21]
integrate end-to-end systems for global camera pose opti-
mization. GS-ICP[6] utilizes point-to-Gaussian ICP regis-
tration, achieving ultrafast and accurate localization.
However, the geometry consistency of Gaussian repre-
sentation has emerged as a critical research focus in the
field of 3D scene reconstruction. Some researches such as
2DGS[8], GOF[34], and RaDe-GS[37] attempt to flatten the
Gaussian ellipsoid and employ equivalent planes along with
unbiased depth estimates to improve the multi-view geom-
etry consistency for Gaussian representation. Unlike previ-
ous works that focus on reconstruction quality of Gaussian
model, we investigate the impact of geometry consistency
on the camera tracking. and propose a novel framework
to integrate 2D Gaussian primitives into dense SLAM sys-
tems. And we explore incorporating 2D Gaussians and re-
search on geometry consistency from the field of 3D recon-
struction into Gaussian-based dense SLAM.
3. Method
3.1. Gaussian Surfel-based Representation
Compared to 3D Gaussians, 2D Gaussian surfels offer su-
perior surface modeling capabilities and enhanced geom-
etry accuracy. Theoretically, 2D Gaussian primitives are
attached to the tangent planes of the scene surface, which
can be represented by the central point µ of the 2D Gaus-
sian and two tangential vectors (eu and ev) of the space. A
point p = (x, y, z, 1)T in the plane can be described as:
p =
 eu
ev
0
µ
0
0
0
1



u
v
1
1

=
 Σ
µ
0
1

p′
(1)
where p′ = (u, v, 1, 1)T is the homogeneous coordinate
of P in the space. And Σ ∈R3×3 is represented as the
3

<!-- page 4 -->
geometry of the 2D plane, which can be decomposed into
rotation transformation R and scaling transformation S as
described by the following formulation.
Σ = RS
(2)
The scaling transformation S has a scaling factor of 0 along
the third dimension. Then, the value of the 2D Gaussian at
point p can be evaluated by the following formulation:
G(p′) = exp
u2 + v2
2

(3)
Following the representation method of 3DGS[13], a scene
can be represented as a set of 2D Gaussian primitives with
geometry Σi, central points µi, opacity oi, and color ci:
G = {Gi : (Σi, µi, oi, ci)|i = 1, ..., n}
(4)
Given a ray sampled from the observation perspective and a
set of 2D Gaussian primitives, the intersection p′
i of the ray
with i-th primitive can be solved. The color correspond-
ing to the ray can be obtained by accumulating the Gaus-
sian values at intersection points sequentially using the α-
blending technique, which can be formulated as:
αi = oiGi(p′
i)
wi = αi
i−1
Y
j=1
(1 −αi)
(5)
I(r) =
X
i=1
ciwi
A(r) =
X
i=1
wi
(6)
where r is the sampled ray corresponding to a pixel, wi
is denoted the blending weight of the i-th Gaussian. We
also render the accumulated opacity map A, which repre-
sents the sum of the blending weights along each ray. When
the accumulated opacity approaches 1, it indicates that the
scene has been well-optimized.
3.2. Surface-aware Depth Rendering
Unbiased depth.
In 2DGS[8], depth rendering no longer
relies on EWA splatting[42], a method that approximates
Gaussian primitives as ellipses on the projection plane. In-
stead, it directly computes the intersection depth (unbiased
depth) between the ray and the 2D Gaussian primitives.
Compared to the 3D Gaussian-based SLAM, our innovative
use of 2D Gaussian primitives achieves an unbiased depth
that more accurately represents the geometry of the Gaus-
sian primitive.
Depth adjustment.
In α-blending process, when a ray
passes through multiple surfaces, surfaces located behind,
although having a small weight, may still interfere with the
depth estimation of the foreground surfaces due to their sig-
nificant depth differences.
This interference is challeng-
ing to eliminate with a simple threshold. To address this,
we propose a depth adjustment approach, which assigns a
weight βi to i-th Gaussian in the α-blending process. This
weight is used to adjust the contribution of i-th Gaussian’s
unbiased depth di to the overall depth composition. The
surface-aware weighted depth d′
i of i-th Gaussian is com-
puted by the following equation:
d′
i = βidi + (1 −βi)dm
(7)
The median depth dm is defined as the depth corresponding
to the m-th Gaussian when the accumulated opacity first
exceeds 0.5(Pm
i=0 wi > 0.5) during ray traversal through
the sequence of Gaussians. Specifically, if the ray has not
reached the m-th Gaussian, we set βi = 1, as the depth of i-
th Gaussian does not negatively impact the blending depth.
Otherwise, βi is computed based on the distance between
di and dm, as well as the variance of distances, as described
by the following formula.
σi =
v
u
u
t
i−1
X
j=0
wi(d′
j −dm)2
(8)
βi = Exp(−(di −dm)2
Bσ2
i
), i > m
(9)
where B serves as a hyperparameter that controls the sensi-
tivity of the weight with respect to both distance and vari-
ance. As shown in Fig. 3, when the distance between the
Gaussian and the median depth increases, the weight βi de-
creases, leading to a reduced influence of the i-th Gaussian
on the depth rendering.
Depth Normalization.
During the α-blending process,
small differences in the accumulated weights along the rays
from different views can lead to significant underestimation
of the rendered depth. To address this, we normalize the
weights of all Gaussian depths during depth map rendering,
which can be formulated as:
D(r) =
Pn
i=1 d′
iwi
A(r)
(10)
3.3. Camera Tracking
GauS-SLAM employs a frame-to-model tracking approach.
Specifically, given a set of 2D Gaussian primitives Gw =
{Gi : (Σi, µi, oi, ci)|i = 1, ..., n} in the scene and the ini-
tial pose {R, t} of the tracking frame, the camera pose is
iteratively optimized by refining R and t to minimize the
discrepancy between the rendered and real images. Sim-
ilar to Gaussian-SLAM [35], we treat the optimization of
the camera pose as an equivalent optimization of the rela-
tive pose of the Gaussian primitives within the scene under
a fixed camera viewpoint. Specifically, the set of Gaussian
4

<!-- page 5 -->
primitives is transformed into the camera coordinate sys-
tem, which can be formalized as:
Gc = {Gi : (RΣi, Rµi + t, oi, ci)|i = 1, ..., n}
(11)
The following loss function will be used to jointly optimize
the {R, t}, such that the rendered depth and color maps are
aligned with the ground truth.
Ltrack = (A > 0.9)

L1(D, ˆD) + λ1L1(I, ˆI)

(12)
To mitigate the influence of new observed scene and out-
liers, we apply the loss function to pixels with accumulated
opacity exceeding 0.9.
3.4. Incremental Mapping
In mapping process, the densification strategy based on
cloning and splitting in 3DGS[13] typically requires sub-
stantial iteration counts and extensive multi-view con-
straints to achieve satisfactory scene coverage. To accel-
erate reconstruction efficiency, we propose a Surfel Attach-
ment initialization approach inspired by [12]. Specifically,
the 2D Gaussian surfels will be directly positioned at the
unprojection of pixels where the accumulated opacity is less
than 0.6. And the initial scale S and orientation R are de-
termined by the following equations:
S = diag(dgt
f , dgt
f , 0)
(13)
R = (e1, e2, ngt)
e1 ⊥e2 ⊥ngt
(14)
ngt = ∇xDgt × ∇yDgt
|∇xDgt × ∇yDgt|
(15)
where Dgt, ngt is represented as the ground-truth depth and
normal of the pixel. In regions lacking ground-truth depth,
Gaussians are placed in areas where the accumulated opac-
ity falls between 0.4 and 0.6. Since these regions have been
partially reconstructed, the rendered depth can be used as
the ground-truth depth to initialization Gaussians. This pro-
cess facilitates the expansion of the Gaussian model along
object boundaries, which we refer to as Edge Growth.
During the mapping process, the front-end randomly se-
lects a frame from the local map and utilizes the following
loss function to optimize the Gaussian model for local re-
construction.
Lmap = L1(D, ˆD) + λ1L1(I, ˆI) + λ2Lreg
(16)
The term Lreg reduces the depth uncertainty along the rays
by minimizing the weighted MSE between all fixed depth
of each Gaussian on the ray and the median depth dm.
Lreg =
X
r
X
i=1
wi(d′
i −dm)2
(17)
3.5. GauS-SLAM System
Front-end.
In the front-end, all optimization processes
are performed within a local map.
The first frame of
the local map serves as the reference keyframe (RKF).
When processing a new frame, the front-end first performs
camera tracking to estimate its pose relative to the RKF
TF
RKF {R, t} . It then evaluates whether the frame qual-
ifies as a keyframe (KF) based on the proportion of newly
observed scene exceeding a predefined threshold τk. The
incremental mapping is performed on KFs. If the number
of Gaussian primitives in the local map exceeds a specified
threshold τl, the front-end send the frames and local Gaus-
sian map to the back-end and a new local map is reinitial-
ized to continue tracking and mapping. At this point, the
current frame is marked as a new RKF within the new local
map.
Back-end.
The back-end of the system is primarily re-
sponsible for merging local maps and optimizing the global
map. Upon receiving a local map, the backend stores the
frames from the local map as sub-maps in the database
and integrates the local Gaussian map into the global map.
Specifically, the Gaussian primitives in the local map are
first reset to 0.01 in terms of opacity and then added to
the global map according to their RKF poses.
Subse-
quently, the current submap and its co-visible submaps
are jointly selected for local mapping. To determine co-
visibility between submaps, we utilize the visual features
extracted from the first and last frames of each submap us-
ing NetVLAD[1].
After the mapping process, Gaussian
primitives with opacity below 0.05 are pruned. This step
effectively removes overlapping parts between the local and
global maps, thereby preventing the continuous accumula-
tion of Gaussian primitives.
To reduce the accumulation of trajectory errors, Bundle
Adjustment(BA) will be applied to optimize the poses of the
RKFs TRKF
w
{R, t} involved in the co-visible submaps, as
well as the global map. During the BA process, frames are
randomly selected from co-visible submaps and the opti-
mization is performed through the minimization of the fol-
lowing formula.
Lba = L1(D, ˆD) + λ1L1(I, ˆI)
(18)
When the backend is not busy, a frame is stochastically
selected from the submaps in the database to refine the
global map, which we refer to as the Random Optimiza-
tion. This process effectively mitigates the forgetting issue
and enhances the global consistency of the Gaussian scene.
After the reconstruction is completed by the front-end and
back-end, Random Optimization continues to run for an ad-
ditional period to reduce floating Gaussians, ensuring that
the global map is evenly optimized. We refer to this pro-
5

<!-- page 6 -->
cess as Final Refinement, and experiments demonstrate that
it significantly improves the rendering quality.
4. Experiments
4.1. Experiment Setup
Implementation Details.
The parameter B in the depth
adjustment is set to 4. To accelerate the convergence of pose
estimation during tracking, the parameters β1 and β2 of the
Adam optimizer are set to 0.7 and 0.99, respectively. In the
frontend, a keyframe is generated when the proportion of
newly observed scene exceeds τk = 1%. And, a new local
map is created when the number of Gaussian primitives in
the local map exceeds τl = 1.5 · HW. In the loss function,
the parameters λ1 and λ2 are set to 0.5 and 0.1, respectively.
To mesh scenes, we employed TSDF Fusion [36] with voxel
size 1 cm to integrate the color maps and depth maps, which
is similar to [20, 35]. All experiments in this paper are con-
ducted on the Intel Core i9-14900K processor along with an
NVIDIA GeForce A6000 GPU.
Datasets & Evaluation Metrics.
We evaluate our ap-
proach on both synthetic and real-world datasets, in-
cluding the Replica[22] synthetic dataset, and the TUM-
RGBD[24], ScanNet[4], and ScanNet++[33] real-world
datasets. Two challenging sequences S1(b20a261fdf),
S2(8b5caf3398) are sampled from ScanNet++[33] for
evaluation.
EVO[5] toolkit is employed to calculate the
ATE-RMSE[23] metric to measure the trajectory accuracy.
PSNR, SSIM[27] and LPIPS[38] are utilized for evaluating
rendering quality. Meanwhile, we also provide the Depth
L1 as in [41] and the F1-Score to evaluate the geometry
quality of reconstruction results.
4.2. Comparison with SOTA Baselines
Tracking Performance.
The tracking performance com-
parison of selected sequences on the four datasets is pre-
sented in Tab. 1 and Tab. 2. Our proposed GauS-SLAM
achieves millimeter-level localization accuracy, establish-
ing new SOTA performance on both Replica[22] and
ScanNet++[33] datasets. Specifically, on the Replica[22]
dataset, our method demonstrates superior performance
with an ATE-RMSE[23] of 0.06 cm,
representing a
62.5% improvement over the previous SOTA method GS-
ICP[6] and an 83% enhancement compared to our baseline
SplaTAM[12]. Despite the presence of challenging factors
such as exposure variations and motion blur in the TUM-
RGBD[24] and ScanNet[4] datasets, GauS-SLAM main-
tains competitive performance.
Notably, it even outper-
forms SLAM methods with loop closure correction on some
sequences in ScanNet[33].
Method
PSNR
SSIM
LPIPS
ATE
Depth L1
F1-Score
[db]↑
↑
↓
[cm]↓
[cm]↓
[%]↑
ESLAM[11]
27.80
0.921
0.245
0.63
2.08
78.2
Point-SLAM[20]
35.17
0.975
0.124
0.52
0.44
89.7
MonoGS[14]
37.50
0.960
0.070
0.58
0.95
78.6
SplaTAM[12]
34.11
0.978
0.104
0.36
0.72
86.1
Gaussian SLAM[35]
42.08
0.996
0.018
0.31
0.68
88.9
GS-ICP[6]
38.83
0.975
0.041
0.16
-
-
GauS-SLAM(Ours)
40.25
0.991
0.027
0.06
0.43
90.5
Table 1. Comparison of Tracking and Reconstruction Perfor-
mance on the Replica Dataset. The best results are highlighted
as first , second , and third . Same methods we employed to
extract meshes and compute metrics for both MonoGS[14] and
SplaTAM[12]. Other results are from the respective papers.
Dataset
TUM-RGBD[24]
ScanNet[4]
ScanNet++[33]
fr2
fr3
0059
0169
S1
S2
ORB-SLAM2[16]
0.40
1.00
14.25
8.72
✗
✗
Point-SLAM[20]
1.31
3.48
7.81
22.16
✗
✗
MonoGS*[14]
1.77
1.49
32.10
10.70
7.00
3.66
SplaTAM[12]
1.24
5.16
10.10
12.10
1.91
0.61
Gaussian-SLAM*[35]
1.39
5.31
12.80
16.30
1.37
2.28
LoopSplat*[40]
1.30
3.53
7.10
10.60
1.14
3.16
GauS-SLAM(Ours)
1.34
1.46
7.14
7.45
0.42
0.47
✗indicates a large trajectory error, even in the first 250 frames
Table 2. Comparison of Tracking Performance(ATE-RMSE
↓[cm]). The methods marked with an ”*” were evaluated only
on the first 250 frames of the ScanNet++[33] dataset, as the large
camera disparity in later frames often led to tracking failures.
Room 0
Room 2
Office 2
MonoGS[14]
SplaTAM[12]
GauS-SLAM(Ours)
Figure 5. Comparison of mesh results on Replica[22]. Com-
pared to isotropic 3D Gaussians, Gaussian surfels produce
smoother mesh reconstructions.
Rendering and Reconstruction Performance.
In the
Tab. 1, we present the rendering and reconstruction perfor-
mance of GauS-SLAM on the Replica [22] dataset. Al-
though the rendering quality of 2DGS [8] has been em-
pirically demonstrated to be inferior to that of 3DGS [13],
GauS-SLAM surpasses the majority of 3D Gaussian-based
methods. Notably, it outperforms our baseline algorithm,
SplaTAM [12], by 6 dB in PSNR. This improvement is
attributed to our novel local map-based design, which en-
6

<!-- page 7 -->
Office 2
Office 4
Ground-truth
Point-SLAM[20]
SplaTAM[21]
MonoGS[14]
GauS-SLAM(Ours)
Figure 4. The comparison of Rendering performance on Replica[22]. We present rendered color maps and depth error maps from 2
viewpoints to comparatively evaluate the rendering quality and geometry accuracy of different approaches.
Frame 40
Frame 55
3DGS[13]
2DGS[8]
GauS-SLAM(Ours)
Figure 6. In the geometry consistency experiment, the error maps
of the depth rendering results on the Frame 40 and Frame 55
of Room 0 in the Replica dataset.
ables more accurate initialization of Gaussian primitives.
A more detailed comparison is presented in Fig. 4.
By
employing the Surface-aware Depth Rendering approach,
our method achieves superior performance in both Depth
L1 and F1-Score compared to other Gaussian-based algo-
rithms. It is noteworthy that the isotropic Gaussian prim-
itives [12, 14] tend to produce uneven mesh surfaces, as
depicted in Fig. 5, whereas the 2D Gaussian surfels yield
significantly smoother results.
Geometry consistency
To evaluate the geometry consis-
tency of the rendering method, we design the following ex-
periment. First, we fully train the model on the first four
frames of the Replica Room0 dataset using ground-truth
poses. We then compute the average L1 error of the ren-
dered depth maps across the first 60 viewpoints as the met-
Methods
Tracking
Mapping
ATE
PSNR
/Frame
/Frame
[cm]↓
[dB]↑
Point-SLAM
0.47
2.57
0.49
33.10
SplaTAM
2.07
3.73
0.31
32.56
GauS-SLAM
1.04
0.65
0.05
38.20
GauS-SLAM-S
0.42
0.16
0.10
37.01
Table 3. Runtime on Replica/Room0 using a A6000. The av-
erage per-frame time for tracking and mapping are computed by
dividing the total processing time of the tracking and mapping pro-
cedures by the total number of frames.
ric for geometry consistency assessment, which is utilized
in ablation studies. We present the error maps for Frame
40 and Frame 55 in Fig. 6. While 2DGS [8] demon-
strates higher view consistency, significant depth errors are
also observed at object boundary regions. This phenomenon
occurs because depth values from different surfaces collec-
tively influence the final rendering depth.
Our proposed
surface-aware depth rendering strategy effectively mitigates
the impact of occluded surfaces on the rendering results,
thereby enhancing geometry consistency.
Runtime Comparison
Tab. 3 presents the average per-
frame time for tracking and mapping process of GauS on
the Room 0 sequence(resolution 1200×680). Compared to
our baseline SplaTAM [12], GauS-SLAM We demonstrate
significant improvements not only in rendering quality and
tracking accuracy but also achieve a more than threefold
7

<!-- page 8 -->
enhancement in time efficiency. Specifically, we developed
a more efficient model, GauS-SLAM-S, which reduces the
number of tracking iterations from 40 to 25 and mapping
iterations from 40 to 30, while decreasing the keyframe
threshold τk to 5%.
4.3. Ablation Study
In this section, we conduct ablation studies on the depth
rendering and SLAM components of our proposed method.
By systematically removing components, we demonstrate
the superior rendering quality and tracking accuracy.
Depth rendering ablation
We conducted ablation stud-
ies on improvements to depth rendering, including unbiased
depth, depth adjustment, depth normalization, and regular-
ization loss. The effectiveness of these strategies was eval-
uated using three metrics: geometry consistency, trajectory
error, and rendering quality. The results are presented in
Tab. 4. Among these, unbiased depth and depth normaliza-
tion had the most significant impact on the results. Remov-
ing them led to notable degradation. Depth adjustment ef-
fectively reduces the mutual influence among different sur-
faces, thereby improving tracking performance.
Ablation on SLAM components
We conducted ablation
studies on the SLAM system to analyze the impact of key
components, including keyframes, local mapping, random
optimization, and final refinement.
We employed ATE-
RMSE [23], PSNR, and the average processing time per
frame as metrics to evaluate the impact of these modules
on system accuracy and efficiency. Notably, we utilized
Room0 from Replica [22] and fr3/office from TUM
[24], the latter of which represents a typical scenario where
the camera moves around an object. The experimental re-
sults are presented in Tab. 5.
In Experiment E, we ablate keyframes and performed
mapping for each frame. This approach reduces tracking er-
rors to some extent, but at the cost of significantly increased
computational time. In Experiment F, we replace the front-
end and the back-end based on local maps with the track-
ing and mapping framework proposed by SplaTAM [12].
The results indicate that this modification greatly reduced
system efficiency, with a noticeable decline in accuracy in
the fr3/office sequence. This suggests that the design
of the local map plays a crucial role in tracking with cam-
eras that perform object-circling movements. Experiments
G and H conducted ablation studies on the system’s back-
end, specifically on random optimization and final refine-
ment. The results show that random optimization helps im-
prove tracking accuracy, while final optimization enhances
rendering quality.
Methods
Geo. Con
ATE
PSNR
[mm]↓
[mm]↓
[dB]↑
A. w/o Unbiased Depth
1.94
2.10
36.06
B. w/o Depth Adjustment
1.75
0.85
38.10
C. w/o Depth Norm.
2.51
1.92
35.98
D. w/o Regulation Loss
1.01
0.63
38.25
Full Model
1.01
0.60
38.04
Table 4. Ablation Study on Depth Rendering. In Experiment A,
we exclusively substituted 2D Gaussian surfels in rendering model
with 3D Gaussian primitives.
Methods
Sequence
ATE
PSNR
Time
[mm]↓
[dB]↑
[s]↓
E. w/o
Keyframe
Room 0
0.52
38.28
2.13
fr3/office
14.53
25.03
1.72
F. w/o
LocalMap
Room 0
0.49
38.25
6.77
fr3/office
52.91
24.16
5.58
G. w/o Random
Optimization
Room 0
0.70
37.78
1.63
fr3/office
14.37
25.03
1.62
H. w/o Final
Refinement
Room 0
0.54
37.48
1.73
fr3/office
14.30
24.34
1.63
Full Model
Room 0
0.60
38.04
1.73
fr3/office
14.29
25.06
1.62
Table 5. Ablation Study on GauS-SLAM System. In Exper-
iment F, we implemented an alternating execution paradigm for
tracking and mapping following the SplaTAM[12] framework
5. Conclusion
In this paper, we address two critical challenges in camera
tracking within Gaussian representation: geometric distor-
tions in multi-view scenarios and outlier rejection during
the frame-to-model alignment process. To address these is-
sues, we propose a Surface-aware Depth Rendering strategy
based on 2DGS[8] and design a SLAM system that inte-
grates keyframes and local maps. Our experimental results
demonstrate that the proposed GauS-SLAM outperforms
the baseline methods in both tracking and rendering on four
benchmark datasets. In particular, it achieves SOTA track-
ing performance in the Replica[22] and ScanNet++[33]
datasets, underscoring the efficacy of 2D Gaussian in cam-
era tracking tasks.
Limitation and Future Work
GauS-SLAM, similar
to other Gaussian-based algorithms, exhibits significant
sensitivity to motion blur and exposure variations.
This
limitation leads to suboptimal tracking performance on
benchmark datasets such as TUM-RGBD[24] and ScanNet
[4]. For future work, we will focus on enhancing the ro-
bustness to these factors causing multi-view inconsistency.
8

<!-- page 9 -->
References
[1] Relja Arandjelovi´c, Petr Gronat, Akihiko Torii, Tomas Pa-
jdla, and Josef Sivic. Netvlad: Cnn architecture for weakly
supervised place recognition. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 40(6):1437–1451, 2018.
5
[2] Lizhi Bai, Chunqi Tian, Jun Yang, Siyu Zhang, Masanori
Suganuma, and Takayuki Okatani. Rp-slam: Real-time pho-
torealistic slam with efficient 3d gaussian splatting, 2024. 2,
3
[3] Lin Chen, Boni Hu, Jvboxi Wang, Shuhui Bu, Guang-
ming Wang, Pengcheng Han, and Jian Chen. G2-mapping:
General gaussian mapping for monocular, rgb-d, and lidar-
inertial-visual systems. IEEE Transactions on Automation
Science and Engineering, pages 1–1, 2025. 2, 3
[4] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proc. Computer Vision and Pattern Recognition (CVPR),
IEEE, 2017. 6, 8, 1, 2, 3
[5] Michael Grupp.
evo:
Python package for the evalua-
tion of odometry and slam.
https://github.com/
MichaelGrupp/evo, 2017. 6
[6] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam, 2024. 3, 6
[7] Christian
Homeyer,
Leon
Begiristain,
and
Christoph
Schn¨orr. Droid-splat: Combining end-to-end slam with 3d
gaussian splatting. ArXiv, abs/2411.17660, 2024. 2, 3
[8] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 1, 2, 3, 4, 6, 7,
8
[9] Huajian Huang, Longwei Li, Cheng Hui, and Sai-Kit Yeung.
Photo-slam: Real-time simultaneous localization and photo-
realistic mapping for monocular, stereo, and rgb-d cameras.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024. 2, 3
[10] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and Shi-
Min Hu. Di-fusion: Online implicit 3d reconstruction with
deep priors. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2021. 2
[11] M. M. Johari, C. Carta, and F. Fleuret.
ESLAM: Effi-
cient dense slam system based on hybrid representation of
signed distance fields. In Proceedings of the IEEE interna-
tional conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2023. 2, 6, 3, 4
[12] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat, track and map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024.
1, 2, 3, 5, 6, 7, 8, 4
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 4, 5, 6, 7
[14] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison. Gaussian splatting slam. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024. 1, 3, 6, 7, 2, 4
[15] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2
[16] Ra´ul Mur-Artal and Juan D. Tard´os.
ORB-SLAM2: an
open-source SLAM system for monocular, stereo and RGB-
D cameras.
IEEE Transactions on Robotics, 33(5):1255–
1262, 2017. 2, 6
[17] Joseph Ortiz, Alexander Clegg, Jing Dong, Edgar Sucar,
David Novotny, Michael Zollhoefer, and Mustafa Mukadam.
isdf: Real-time neural signed distance fields for robot per-
ception. In Robotics: Science and Systems, 2022. 2
[18] Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin Yang,
Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d re-
construction at scale using gaussian splatting. ACM Trans-
actions on Graphics, 2024. 3
[19] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. arXiv preprint arXiv:2210.13641, 2022. 2
[20] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R. Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2023. 2, 6, 7, 3, 4
[21] Erik Sandstr¨om, Keisuke Tateno, Michael Oechsle, Michael
Niemeyer, Luc Van Gool, Martin R Oswald, and Federico
Tombari. Splat-slam: Globally optimized rgb-only slam with
3d gaussians. arXiv preprint arXiv:2405.16544, 2024. 3, 7
[22] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal,
Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan,
Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang
Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler
Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva,
Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael
Goesele, Steven Lovegrove, and Richard Newcombe. The
Replica dataset: A digital replica of indoor spaces. arXiv
preprint arXiv:1906.05797, 2019. 1, 6, 7, 8, 2, 4
[23] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the eval-
uation of rgb-d slam systems. In 2012 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems, pages
573–580, 2012. 6, 8
[24] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cre-
mers. A benchmark for the evaluation of rgb-d slam systems.
In Proc. of the International Conference on Intelligent Robot
Systems (IROS), 2012. 6, 8, 1, 2, 3, 4
[25] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davi-
son. imap: Implicit mapping and positioning in real-time.
In 2021 IEEE/CVF International Conference on Computer
Vision (ICCV), pages 6209–6218, 2021. 2
9

<!-- page 10 -->
[26] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual
SLAM for Monocular, Stereo, and RGB-D Cameras. Ad-
vances in neural information processing systems, 2021. 2,
3
[27] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 6
[28] Silvan Weder, Johannes L. Schonberger, Marc Pollefeys, and
Martin R. Oswald.
Neuralfusion: Online depth fusion in
latent space. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
3162–3172, 2021. 2
[29] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin
Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf:
Point-based neural radiance fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5438–5448, 2022. 3
[30] Ziheng Xu, Qingfeng Li, Chen Chen, Xuefeng Liu, and Jian-
wei Niu. Glc-slam: Gaussian splatting slam with efficient
loop closure, 2024. 2, 3
[31] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In CVPR, 2024. 1, 3
[32] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian
Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and
mapping with voxel-based neural implicit representation. In
2022 IEEE International Symposium on Mixed and Aug-
mented Reality (ISMAR), pages 499–507, 2022. 2
[33] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d
indoor scenes. In Proceedings of the International Confer-
ence on Computer Vision (ICCV), 2023. 1, 6, 8, 2, 3, 5
[34] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics, 2024. 3
[35] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting, 2023. 1, 3, 4, 6
[36] Andy Zeng, Shuran Song, Matthias Nießner, Matthew
Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3dmatch:
Learning local geometric descriptors from rgb-d reconstruc-
tions. In CVPR, 2017. 6
[37] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang,
Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in
gaussian splatting, 2024. 3
[38] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. CoRR, abs/1801.03924,
2018. 6
[39] Wei Zhang, Qing Cheng, David Skuddis, Niclas Zeller,
Daniel Cremers, and Norbert Haala. Hi-slam2: Geometry-
aware gaussian slam for fast monocular scene reconstruction,
2024. 2, 3
[40] Liyuan Zhu, Yue Li, Erik Sandstrom, Shengyu Huang, Kon-
rad Schindler, and Iro Armeni. Loopsplat: Loop closure by
registering 3d gaussian splats, 2024. 6, 2, 3
[41] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022. 2, 6
[42] M. Zwicker, H. Pfister, J. van Baar, and M. Gross.
Ewa
splatting. IEEE Transactions on Visualization and Computer
Graphics, 8(3):223–238, 2002. 4
10

<!-- page 11 -->
GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels
Supplementary Material
Abstract
This supplementary material encompasses an evaluation
video of GauS-SLAM on the S2(8b5caf3398) sequence of
ScanNet++. Furthermore, we present comprehensive ex-
perimental results and visualizations across multiple bench-
mark datasets, including Replica, ScanNet, TUM-RGBD,
and ScanNet++. In addition, we provide an extended anal-
ysis and discussion on the ablation studies on the local map
and the runtime evaluation.
6. Video
We submit a video GauS SLAM S2.mp4.
This video
demonstrates the experimental results of GauS-SLAM in a
challenging real-world scenario S2. It records the back-
end operation of GauS-SLAM, where the deep blue lines
represent the camera trajectory, the blue frustums denote
keyframes, and the larger orange frustums indicate the ref-
erence keyframes of each submap. Throughout the video,
GauS-SLAM continuously integrates local maps received
from the front-end, optimizing the reconstruction and re-
moving redundant Gaussian surfels after each fusion. De-
spite the large disparity between adjacent frames, GauS-
SLAM maintains stable tracking and reconstruction. Ad-
ditionally, the video highlights the effectiveness of Edge
Growth in reconstructing depth-missing regions, such as the
window area.
7. Implementation details
Hyperparameters
Tab. 6 presents the principal hyperpa-
rameters used in our experiments on Replica[22], TUM-
RGBD[24], ScanNet[4] and ScanNet++[33].
The learn-
ing rates for camera rotation lr and translation lt during
tracking are dynamically adjusted using an exponential de-
cay strategy, which is consistently applied in both front-
end and back-end optimization processes. Notably, these
learning rates are adaptively modulated based on the cam-
era’s motion velocity, resulting in dataset-specific varia-
tions. Furthermore, the table enumerates the number of it-
erations for tracking(itert) and mapping(iterm) across dif-
ferent datasets. In scenarios with large disparity between
adjacent frames, an increased number of tracking iterations
is implemented to ensure the convergence of pose optimiza-
tion.
Pose Initialization
Similar to the majority of SLAM al-
gorithms, the constant velocity model is employed for ini-
Params
Replica
TUM-RGBD
ScanNet
ScanNet++
lr
0.0004
0.0008
0.0008
0.01
lt
0.002
0.004
0.004
0.04
itert
40
120
100
120
iterm
40
40
40
60
Table 6. Per-dataset Hyperparameters.
tializing the pose of each frame. Specifically, the initial pose
T ′
i of the i-th frame can be derived using the following for-
mulation.
T′
i = (Ti−1T−1
i−2) · Ti−1
(19)
where Ti−1 denotes the predicted pose of the preceding
frame.
Exposure compensation
Exposure variation poses a sig-
nificant challenge in 3D reconstruction using Gaussian
Splatting, particularly in the incremental reconstruction
task. To address this issue, we employ a simple linear com-
pensation method by introducing learnable compensation
coefficients a, b for each frame. During rendering, the color
map is corrected according to the following formulation:
I′ = aI + b
(20)
Re-tracking
In the front-end tracking process, significant
disparity can result in the loss of tracking for specific frames
in the local map. This phenomenon is particularly observed
in the S1 sequence of the ScanNet++[33] dataset. To ad-
dress this issue, we propose a re-tracking strategy. When
the rendered depth error significantly exceeds the average
level, the front-end will flag it as a lost frame. Subsequently,
the local map is reset, and the lost frame is designated as the
reference keyframe for a new local map, enabling the con-
tinuation of the front-end. Upon receiving a lost frame, our
tracking approach will extend beyond local map by leverag-
ing a broader set of submaps, thereby enhancing the robust-
ness of the tracking process
8. Novel View Synthesis
We evaluate novel view synthesis (NVS) performance on
sequences S1 and S2 of the ScanNet++[33] dataset, with
results presented in Tab. 7. Our method demonstrates su-
perior performance over SplaTAM[12] method in both ren-
dering quality and depth map accuracy, which can be at-
tributed to our improved pose estimation and depth render-
ing method. Fig. 7 provides a comparative visualization
1

<!-- page 12 -->
Training view
Novel view
S1
S2
Ground-truth
SplaTAM[12]
GauS-SLAM(Ours)
Ground-truth
SplaTAM[12]
GauS-SLAM(Ours)
Figure 7. NVS on ScanNet++[33].
Methods
Metrics
Novel View
Training View
S1
S2
Avg.
S1
S2
Avg.
Point-SLAM[20]
PSNR [dB] ↑
12.10
11.73
11.91
14.62
14.30
14.46
SSIM ↑
0.31
0.26
0.28
0.35
0.41
0.38
LPIPS ↓
0.62
0.74
0.68
0.68
0.62
0.65
SplaTAM[12]
PSNR [dB] ↑
23.99
24.84
24.41
27.82
28.14
27.98
SSIM ↑
0.88
0.87
0.88
0.94
0.94
0.94
LPIPS ↓
0.21
0.26
0.24
0.12
0.13
0.12
Depth L1 [cm] ↓
1.91
2.23
2.07
0.93
1.64
1.28
GauS-SLAM(Ours)
PSNR [dB] ↑
25.48
25.76
25.62
29.21
30.07
29.64
SSIM ↑
0.89
0.87
0.88
0.95
0.95
0.95
LPIPS ↓
0.25
0.23
0.24
0.16
0.11
0.13
Depth L1 [cm] ↓
1.15
2.23
1.69
0.52
0.62
0.57
Table 7. Novel & Train View Rendering Performance on Scan-
Net++[33].
of rendering results from training and novel views, where
our proposed Edge Growth strategy effectively handles re-
gions with missing ground truth depth, particularly in win-
dow areas, demonstrating enhanced reconstruction capabil-
ity. Owing to the enhanced accuracy of our depth maps,
the extracted mesh exhibits superior quality, particularly in
planar regions such as floors and walls, as shown in Fig. 11.
9. Additional experiments
Experiments on Replica[22]
Tab. 11 and Tab. 12 present
comprehensive evaluations of tracking accuracy, render-
ing quality, and reconstruction performance across all se-
quences in the Replica dataset.
Our proposed GauS-
SLAM framework demonstrates state-of-the-art perfor-
mance, achieving superior results on nearly every sequence
compared to existing methods.
Methods
000
059
106
169
181
207
Avg.
Point-SLAM[20]
10.2
7.8
8.6
22.1
14.7
9.5
12.1
SplaTAM[12]
12.8
10.1
17.7
12.1
11.1
7.5
11.8
MonoGS [14]
9.8
32.1
8.9
10.7
21.8
7.9
15.2
GLC-SLAM[30]
12.9
7.9
6.3
10.5
11.0
6.3
9.2
LoopSplat [40]
6.2
7.1
7.4
10.6
8.5
6.6
7.7
GauS-SLAM(Ours)
9.3
7.1
7.0
7.4
17.5
6.2
11.5
Table 8. Comparison of tracking performance on ScanNet[4].
Note that the GLC-SLAM[30] and LoopSplat[40] have loop clo-
sure correction. Results are taken from [30, 40].
Experiments on TUM-RGBD[24].
Tab. 9 present an
evaluation of tracking performance and rendering quality
on TUM-RGBD[24]. The dataset’s comparatively inferior
depth map quality poses significant challenges for RGBD-
based GauS-SLAM. While our proposed Edge Growth
method can compensate for missing depth values, the ab-
sence of multiview constraints often leads to incorrect
growth of Gaussians, directly impacting tracking perfor-
mance. On the other hand, MonoGS[14] and SplaTAM[12]
employ isotropic 3D Gaussians, which help to maintain
scene smoothness to some extent but result in the loss of fine
details. As shown in Fig. 10, although our approach demon-
strates a strong capability in modeling details, the recon-
structed surfaces exhibit persistent high-frequency noise,
making them appear visually cluttered.
Experiments on ScanNet[4].
Tab. 8 present an evalu-
ation of tracking performance on the ScanNet[4].
The
2

<!-- page 13 -->
Dataset
Metric
fr1/desk
fr2/xyz
fr3/office
Avg.
Point-SLAM[20]
PSNR↑
13.91
17.23
18.11
16.41
SSIM↑
0.626
0.707
0.756
0.696
LPIPS↓
0.540
0.589
0.445
0.525
ATE-RMSE[cm]↓
4.34
1.31
3.48
3.04
SplaTAM[12]
PSNR↑
22.07
24.55
21.62
22.74
SSIM↑
0.851
0.930
0.816
0.865
LPIPS↓
0.231
0.102
0.205
0.180
ATE-RMSE[cm]↓
3.35
1.24
5.16
3.25
MonoGS[14]
PSNR↑
23.73
24.75
24.96
24.28
SSIM↑
0.786
0.836
0.840
0.820
LPIPS↓
0.240
0.209
0.207
0.218
ATE-RMSE[cm]↓
1.47
1.77
1.49
1.57
GauS-SLAM(Ours)
PSNR↑
23.73
27.72
24.92
25.45
SSIM↑
0.902
0.958
0.908
0.922
LPIPS↓
0.231
0.082
0.198
0.170
ATE-RMSE[cm]↓
1.82
1.34
1.47
1.54
Table 9. Comparison of tracking and rendering performance
on TUM-RGBD[24].
Sequence
S1
S2
S3
S4
S5
Avg.
SplaTAM[12]
1.50
0.57
0.31
443.10
1.58
89.41
MonoGS[14]
7.00
3.66
6.37
3.28
44.09
12.88
Gaussian SLAM[35]
1.37
2.82
6.80
3.51
0.88
3.08
LoopSplat [40]
1.14
3.16
3.16
1.68
0.91
2.05
GauS-SLAM(Ours)
0.41
0.38
0.16
0.28
0.34
0.31
Table 10.
Tracking performance on ScanNet++[33](ATE
RMSE↓[cm]). Results are taken from [40].
ScanNet[4] dataset presents significant challenges due to
prevalent motion blur and exposure variations, which can
severely compromise view consistency and lead to trajec-
tory drift in SLAM systems. Despite these inherent difficul-
ties, GauS-SLAM framework demonstrates superior track-
ing accuracy on specific sequences, outperforming even
state-of-the-art methods incorporating loop closure correc-
tion mechanisms.
Experiments on ScanNet++[33]
To ensure equitable
benchmarking, we evaluate the first 250 frames for se-
quences: (S1) b20a261fdf, (S2) 8b5caf3398, (S3)
fb05e13ad1, (S4) 2e74812d00, (S5) 281bc17764.
With tracking results detailed in Tab. 10. Due to the slight
motion blur and the high-precision depth measurements,
GauS-SLAM achieves millimeter-level tracking accuracy,
reducing the trajectory error by approximately 84% com-
pared to the state-of-the-art method, LoopSplat[40].
Runtime Analysis.
As shown in Fig. 8, SplaTAM[12]
experiences a decrease in the efficiency of tracking and
mapping due to the continuous accumulation of Gaussian
primitives in the global map.
In contrast, GauS-SLAM
periodically reset the local map, preventing degradation in
tracking and mapping efficiency.
Ablation on Local map
In the ablation study F presented
in Tab. 5, we investigate the impact of the structure of the
Method
R0
R1
R2
O0
O1
O2
O3
O4
Avg.
NeRF-based
ESLAM[11]
0.71
0.70
0.52
0.57
0.55
0.58
0.72
0.63
0.63
Point-SLAM[20]
0.61
0.41
0.37
0.38
0.48
0.54
0.69
0.72
0.52
Gaussian-based (decoupled)
RP-SLAM[2]
0.43
0.38
0.53
0.36
0.56
0.43
0.25
0.23
0.40
DROID-Splat[7]
0.34
0.13
0.27
0.25
0.42
0.32
0.52
0.40
0.33
HI-SLAM2[39]
0.23
0.22
0.19
0.23
0.27
0.25
0.37
0.33
0.26
Gaussian-based (coupled)
MonoGS[14]
0.44
0.32
0.31
0.44
0.52
0.23
0.17
2.25
0.58
GS-SLAM[31]
0.48
0.53
0.33
0.52
0.41
0.59
0.46
0.7
0.50
SplaTAM[12]
0.31
0.40
0.29
0.47
0.27
0.29
0.32
0.55
0.36
Gaussian-SLAM[35]
0.29
0.29
0.22
0.37
0.23
0.41
0.30
0.35
0.31
GLC-SLAM[30]
0.20
0.19
0.13
0.31
0.13
0.32
0.21
0.33
0.23
RTG-SLAM[18]
0.15
0.14
0.22
0.26
0.25
0.21
0.19
0.12
0.19
GS-ICP[6]
0.15
0.16
0.11
0.18
0.12
0.17
0.16
0.21
0.16
GauS-SLAM(Ours)
0.06
0.08
0.08
0.06
0.03
0.09
0.05
0.05
0.06
Table 11.
Comparison of tracking performance on Replica
(ATE RMSE↓[cm]). Results are tacken from respective papers.
0
200
400
600
800
1000
1200
1400
1600
1800
2000
Frame Index
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
Time Cost(s)
SplaTAM Avg Mapping Time: 3.87s
SplaTAM Avg Tracking Time: 2.14s
GauS-SLAM Avg Mapping Time: 1.14s
GauS-SLAM Avg Tracking Time: 1.04s
Runtime Performance Analysis
SplaTAM Mapping
SplaTAM Tracking
GauS-SLAM Mapping
GauS-SLAM Tracking
Figure 8. The comparison of system efficiency over time. No-
tably, GauS-SLAM performs mapping operations exclusively on
keyframes, and the mapping time of non-keyframe corresponds to
the recent keyframe.
Start
End
Figure 9. The comparison of trajectories with and without local
map.. The local tracking method effectively avoids the influence
of outlier regions in the blue area.
scene without local mapping by conducting experiments on
the fr3/office sequence in TUM-RGBD[24]. In this
sequence, the camera follows a circular trajectory around a
desk, as illustrated in Fig. 9. The tracking error in the blue
area is notably larger compared to the initial portion, which
we attribute to the frequent occurrence of interference re-
gions depicted in Fig. 2(b).
3

<!-- page 14 -->
SplaTAM[12]
MonoGS[14]
GauS-SLAM(Ours)
Ground-truth
Figure 10. Rendering performance on TUM-RGBD[24].
Method
Metric
R0
R1
R2
O0
O1
O2
O3
O4
Avg.
ESLAM[11]
PSNR↑
25.25
25.31
28.09
30.33
27.04
27.99
29.27
29.15
27.80
SSIM↑
0.874
0.245
0.935
0.934
0.910
0.942
0.953
0.948
0.921
LPIPS↓
0.315
0.296
0.245
0.213
0.254
0.238
0.186
0.210
0.245
Depth L1 [cm]↓
0.970
1.070
1.280
0.860
1.260
1.710
1.430
1.060
1.180
F1 [%]↑
81.00
82.20
83.90
78.40
75.50
77.10
75.50
79.10
79.10
Point-SLAM[20]
PSNR↑
32.40
34.08
35.50
38.26
39.16
33.99
33.48
33.49
35.17
SSIM↑
0.974
0.977
0.982
0.983
0.986
0.960
0.960
0.979
0.975
LPIPS↓
0.113
0.116
0.111
0.100
0.118
0.156
0.132
0.142
0.124
Depth L1 [cm]↓
0.530
0.220
0.460
0.300
0.570
0.490
0.510
0.460
0.440
F1 [%]↑
86.90
92.31
90.78
93.77
91.62
88.98
88.22
85.55
89.77
MonoGS[14]
PSNR↑
34.83
36.43
37.49
39.95
42.09
36.24
36.70
36.07
37.50
SSIM↑
0.954
0.959
0.965
0.971
0.977
0.964
0.963
0.957
0.960
LPIPS↓
0.068
0.076
0.075
0.072
0.055
0.078
0.065
0.099
0.070
Depth L1 [cm]↓
0.793
0.561
0.914
0.702
1.099
0.951
0.968
1.661
0.956
F1 [%]↑
79.92
83.60
81.26
86.51
82.18
78.20
79.86
58.01
78.69
SplaTAM[12]
PSNR↑
32.86
33.89
35.25
38.26
39.17
31.97
29.70
31.81
34.11
SSIM↑
0.980
0.970
0.980
0.980
0.980
0.980
0.95
0.950
0.970
LPIPS↓
0.070
0.100
0.080
0.090
0.090
0.100
0.120
0.150
0.100
Depth L1 [cm]↓
0.425
0.364
0.516
0.414
0.646
0.980
1.234
0.609
0.648
F1 [%]↑
89.95
89.13
88.86
92.32
90.01
86.10
79.17
80.96
87.06
GauS-SLAM(Ours)
PSNR↑
38.04
39.89
40.25
43.44
41.22
39.31
39.55
40.34
40.25
SSIM↑
0.989
0.991
0.993
0.994
0.986
0.993
0.994
0.992
0.991
LPIPS↓
0.020
0.030
0.023
0.022
0.051
0.023
0.019
0.029
0.027
Depth L1 [cm]↓
0.343
0.198
0.451
0.287
0.365
0.828
0.736
0.259
0.433
F1 [%]↑
91.46
92.09
91.11
93.59
89.96
89.14
88.90
88.47
90.59
Table 12. Rendering and reconstruction performance on Replica[22].
4

<!-- page 15 -->
SplaTAM[12]
GauS-SLAM(Ours)
Ground-truth
Figure 11. The comparison of mesh results in ScanNet++[33].
5
