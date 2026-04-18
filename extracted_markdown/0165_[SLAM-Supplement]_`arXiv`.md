<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
1
Gaussian-LIC2: LiDAR-Inertial-Camera
Gaussian Splatting SLAM
Xiaolei Lang1,†, Jiajun Lv1,†, Kai Tang1, Laijian Li1, Jianxin Huang1, Lina Liu1, Yong Liu1,∗, Xingxing Zuo2,∗
Abstract—This paper presents the first photo-realistic LiDAR-
Inertial-Camera Gaussian Splatting SLAM system that simulta-
neously addresses visual quality, geometric accuracy, and real-
time performance. The proposed method performs robust and
accurate pose estimation within a continuous-time trajectory
optimization framework, while incrementally reconstructing a
3D Gaussian map using camera and LiDAR data, all in real
time. The resulting map enables high-quality, real-time novel
view rendering of both RGB images and depth maps. To
effectively address under-reconstruction in regions not covered
by the LiDAR, we employ a lightweight zero-shot depth model
that synergistically combines RGB appearance cues with sparse
LiDAR measurements to generate dense depth maps. The depth
completion enables reliable Gaussian initialization in LiDAR-
blind areas, significantly improving system applicability for
sparse LiDAR sensors. To enhance geometric accuracy, we use
sparse but precise LiDAR depths to supervise Gaussian map
optimization and accelerate it with carefully designed CUDA-
accelerated strategies. Furthermore, we explore how the incre-
mentally reconstructed Gaussian map can improve the robustness
of odometry. By tightly incorporating photometric constraints
from the Gaussian map into the continuous-time factor graph
optimization, we demonstrate improved pose estimation under
LiDAR degradation scenarios. We also showcase downstream
applications via extending our elaborate system, including video
frame interpolation and fast 3D mesh extraction. To support
rigorous evaluation, we construct a dedicated LiDAR-Inertial-
Camera dataset featuring ground-truth poses, depth maps, and
extrapolated trajectories for assessing out-of-sequence novel view
synthesis. Extensive experiments on both public and self-collected
datasets demonstrate the superiority and versatility of our system
across LiDAR sensors with varying sampling densities. Both the
dataset and code will be made publicly available on project page
https://xingxingzuo.github.io/gaussian lic2.
Index Terms—LiDAR-Inertial-Camera SLAM, multi-sensor
fusion, photo-realistic dense mapping, 3D Gaussian Splatting.
I. INTRODUCTION
S
IMULTANEOUS
localization
and
mapping
(SLAM)
serves as a fundamental technology that facilitates spatial
perception in both mixed reality systems and robotic appli-
cations. Remarkably, recent advances in radiance field rep-
resentations, particularly Neural Radiance Fields (NeRF) [1]
and 3D Gaussian Splatting (3DGS) [2], have pioneered a new
paradigm in SLAM, namely radiance-field-based SLAM [3].
Powered by differentiable photo-realistic rendering, radiance
field-based SLAM systems aim to provide both accurate pose
1Institute of Cyber-Systems and Control, Zhejiang University, China.
2Department of Robotics, Mohamed Bin Zayed University of Artificial
Intelligence (MBZUAI).
†Contributed equally. ∗Corresponding authors.
This work is supported by Zhejiang Provincial Natural Science Founda-
tion of China under Grant No.LQN25F030006.
Figure 1: Overview of the system outputs (ordered left to right, top
to bottom): (1) reconstructed map comprised of 3D Gaussians, (2)
RGB image rendered from a novel viewpoint, (3) sparse LiDAR point
cloud map, (4) depth map rendered from the novel viewpoint, and
(5) 3D mesh extracted from the Gaussian map.
estimation and photo-realistic 3D maps in real time, equipping
robots with richer 3D scene understanding. These systems can
benefit a wide range of tasks, including path planning [4–6],
active mapping [7, 8], and 3D mesh reconstruction [9].
Initially, NeRF-based SLAM systems [10–14] leverage
multi-layer perceptrons (MLPs) to represent the entire scene
and generate high-quality dense maps with low memory con-
sumption. Nonetheless, such implicit representations necessi-
tate computationally intensive volumetric rendering based on
sampling in 3D space, undermining the real-time capability,
which is essential for robot applications. The emergence of
3DGS has shifted this landscape. It features fast rendering
along with superior visual quality and demonstrates greater
potential for real-time use. Equipped with RGB-D or RGB
sensors, most 3DGS-based SLAM systems [15–19] focus on
indoor environments and outperform NeRF-based approaches,
but unfortunately struggle in challenging conditions such as
violent ego-motion, varying illumination, and lack of visual
textures, commonly arising in unbounded outdoor scenes. Sev-
eral works address the issue by fusing LiDAR and IMU [20–
24], among which the effectiveness of LiDAR-Inertial-Camera
fusion has been extensively validated in traditional SLAM
arXiv:2507.04004v2  [cs.RO]  9 Jul 2025

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
2
methods [25–35]. Notably, the precise geometric priors pro-
vided by LiDAR contribute to both pose tracking and Gaussian
mapping, and the significantly reduced cost of LiDARs nowa-
days has made them much more accessible for integration.
Although
existing
LiDAR-Inertial-Camera
3DGS-based
SLAM systems [20, 21, 23, 24, 36, 37] can achieve robust state
estimation and dense mapping with high visual quality, several
issues remain unresolved or insufficiently addressed. First,
current methods predominantly rely on the dense LiDAR with
the FoV closely aligned with that of the camera, initializing
Gaussians solely from LiDAR points [22–24, 36]. This can
lead to under-reconstruction in LiDAR blind spots, especially
when the sparse LiDAR is used. Although adaptive density
control (ADC) can alleviate this limitation by cloning and
splitting Gaussians based on the magnitude of back-propagated
gradients and scales of Gaussians
[2], it struggles in the
incremental SLAM systems involving optimization of Gaus-
sians with different maturity. ADC also requires aggregating
gradients over multiple iterations of optimization, which leads
to a lack of timeliness for real-time SLAM systems. Second,
despite leveraging LiDAR data with precise geometric infor-
mation, existing SLAM methods overemphasize the visual
quality of the map while neglecting its geometric reconstruc-
tion quality [21, 24, 36], which limits its applicability in
geometry-critical tasks, like obstacle avoidance. Pursuing both
high-quality RGB and depth rendering while maintaining real-
time performance remains a great challenge. Furthermore,
most existing methods [23, 24, 37] focus merely on render-
ing quality from training views, overlooking the novel view
synthesis capability in sequence or even out of sequence. Ad-
mittedly, LiDAR-Inertial-Camera SLAM datasets that support
RGB and depth rendering evaluation across both in-sequence
and out-of-sequence novel views are scarce.
By combining 3D Gaussian Splatting with LiDAR-Inertial-
Camera fusion, this paper addresses the aforementioned chal-
lenges and proposes a real-time photo-realistic SLAM system,
dubbed Gaussian-LIC2. The system achieves robust and ac-
curate pose estimation while constructing photo-realistic 3D
Gaussian maps that encompass both high-fidelity visual and
geometric information. Our main contributions are as follows:
• We propose the first LiDAR-Inertial-Camera Gaussian
Splatting SLAM system that jointly takes care of visual
quality, geometric accuracy, and real-time performance.
It is capable of robustly and precisely estimating poses
while constructing a photo-realistic and geometrically
accurate 3D Gaussian map, all in real time.
• We propose integrating LiDAR and visual data through
a fast, lightweight, generalizable sparse depth comple-
tion network to predict depths for pixels not covered
by the LiDAR, enabling more comprehensive Gaussian
initialization and mitigating under-reconstruction. During
training, we fully leverage LiDAR-provided depth for
supervision and accelerate the process with a series of
meticulously designed C++ and CUDA implementations.
• We explore tightly fusing photometric constraints from
the incrementally built Gaussian map with LiDAR-
Inertial data in a continuous-time framework, successfully
helping the odometry overcome the LiDAR degradation.
Besides, we have extended our system to enable Gaussian
map utilization for downstream applications such as video
frame interpolation and rapid mesh generation.
• We curate a specialized LiDAR-Inertial-Camera dataset
that provides ground-truth poses and depth maps, along
with carefully designed capturing trajectories, enabling
the evaluation of out-of-sequence novel view synthesis.
We conduct extensive experiments on public and self-
collected datasets, demonstrating the superiority of our
approach and its adaptability to various types of LiDARs.
II. RELATED WORKS
A. Photo-Realistic Reconstruction with Radiance Field
Map representation deeply affects architecture designs and
potential downstream applications of SLAM systems. Sparse
SLAM approaches [38, 39] excel in pose estimation but
yield only sparse keypoint maps. On the contrary, dense
SLAM methods produce dense maps beneficial for scene
understanding. For instance, DTAM [40], REMODE [41],
DSO [42], DROID-SLAM [43], etc, can achieve accurate
camera pose tracking and reconstruct dense point cloud maps.
KinectFusion [44], ElasticFusion [45], and SurfelMeshing [46]
model 3D scenes using truncated signed distance function
(TSDF), surfel, and mesh, all of which are commonly utilized
in the SLAM field. However, it is challenging to recover photo-
realistic camera views from these map representations. Fortu-
nately, the emergence of NeRF [1] has brought a promising
solution. As a novel radiance field representation, NeRF im-
plicitly models the geometry and texture of the scene through
MLPs and achieves differentiable photo-realistic rendering
even at novel views. Plenoxels [47] and instant-ngp [48]
further accelerate the training and inference of NeRF by
introducing explicit feature grids. To improve the geometric
accuracy of NeRF representation and extract a precise 3D
mesh from it, VolSDF [49], NeuS [50], and NeuS2 [51] replace
the density field with signed distance function (SDF).
However, NeRF still falls slightly short in real-time recon-
struction due to the computationally intensive ray-based vol-
ume rendering. In contrast, 3D Gaussian Splatting (3DGS)[2]
explicitly represents scenes using view-dependent anisotropic
Gaussians and introduces a tile-based rasterization strategy
for splats, achieving faster rendering speed while maintain-
ing superior visual quality. Built upon 3DGS, a number of
follow-up works have been proposed to further enhance its
performance and flexibility. Scaffold-GS[52] incorporates tiny
MLPs to predict the properties of neural Gaussians, enabling
more compact and expressive representations. Meanwhile,
SuGaR [53] and GOF [54] focus on efficient mesh extraction
from 3DGS representations, facilitating downstream tasks such
as geometry processing and simulation. 2DGS [55], Gaus-
sianSurfels [56], and PGSR [57] flatten the Gaussians to
accurately conform to the scene surface, while RaDe-GS [58]
introduces an enhanced depth rasterizing approach without the
reformulation of the Gaussian primitives. NeuSG [59] and
GSDF [60] try to combine neural SDF with 3DGS.
Original NeRF and 3DGS methods typically rely solely
on image data for photo-realistic reconstruction. However,

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
3
the incorporation of LiDAR, which has become increasingly
affordable and accessible, can significantly enhance the per-
formance of both NeRF and 3DGS, particularly in unbounded
outdoor environments. For NeRF, LiDAR data can guide ray
sampling around surfaces and provide accurate depth super-
vision during optimization. URF[61] and EmerNeRF[62] are
NeRF variants tailored for autonomous driving scenarios and
achieve outstanding rendering quality with the aid of LiDAR.
Bootstrapped by a LiDAR SLAM system, SiLVR[63] con-
structs multiple NeRF submaps efficiently. M2Mapping[64]
further unifies surface reconstruction and photo-realistic ren-
dering in LiDAR-Camera systems through an SDF-based
NeRF formulation. In the context of 3DGS, LiDAR data
can be leveraged not only for geometric supervision but also
for the accurate and efficient initialization of 3D Gaussians.
PVG [65], DrivingGaussian [66], StreetGaussians [67], and
TCLC-GS [68] are the first to introduce LiDAR into the
3DGS framework, effectively modeling both static scenes and
dynamic objects in autonomous driving environments. LIV-
GaussMap [69] and LetsGo [70] both utilize the point cloud
and poses from LiDAR-based SLAM to initialize 3DGS. Fo-
cusing more on geometric quality, LI-GS [71] employs 2DGS
as the map representation to enhance surface alignment, while
GS-SDF [72] incorporates LiDAR into SDF-based 3DGS.
The aforementioned works are all per-scene optimization
frameworks. Interestingly, a series of feed-forward mod-
els [73–76] have emerged and achieved generalizable photo-
realistic reconstruction in an end-to-end manner. However,
their accuracy still lag behind that of per-scene optimization
methods, and their applicability is limited to a small number
of high-quality images with minimal viewpoint variation.
B. Incremental Visual SLAM Systems with Radiance Field
Radiance-field-based reconstruction discussed in the last
section, whether per-scene optimized or feed-forward, is essen-
tially an offline process with all the collected data accessible.
In comparison, radiance-field-based SLAM systems incremen-
tally perform photo-realistic reconstruction with sequentially
input sensor data, utilizing radiance file map representations.
Given sequential RGB-D inputs, iMAP [10] is a pio-
neering work built upon the implicit neural representation
to achieve watertight online reconstruction. Following this,
NICE-SLAM [11] is able to scale up to larger indoor scenes
by combining MLP representation with hierarchical feature
grids. Vox-Fusion [12] further adopts the octree to dynamically
expand the volumetric map, eliminating the need for pre-
allocated grids. Leveraging hash grids, tri-planes, and neu-
ral point cloud as their respective implicit representations,
Co-SLAM [13], ESLAM [14], and Point-SLAM [77] get
enhancement in both localization and mapping. Moreover,
H2-Mapping [78] handles the forgetting issue by a novel
coverage-maximizing keyframe selection strategy. In terms
of 3DGS-based SLAM, GS-SLAM [15], SplaTAM [16], and
Gaussian-SLAM [17] demonstrate the advantages of 3DGS
over existing map representations in SLAM systems for online
photo-realistic mapping. By forcing binary opacity for each
Gaussian, RTG-SLAM [79] achieves real-time performance
indoors with the compact scene representation. GSFusion [80]
jointly constructs the Gaussian map with a TSDF map and uses
a quadtree data structure to reduce the number of Gaussians.
MM3DGS-SLAM [20] loosely fuses RGB-D and IMU mea-
surements to enable more robust and precise pose estimation.
A range of studies have also explored operating solely on
monocular RGB images, among which NICER-SLAM [81]
and MonoGS [18] are the representative radiance-field-map-
centric approaches. The former fully makes use of the monocu-
lar geometric cues for supervision, and the latter introduces the
isotropic regularization to address ambiguities in incremental
reconstruction. However, decoupled tracking and mapping
methods, which adopt the state-of-the-art visual odometry
for pose tracking and radiance field optimization for photo-
realistic mapping in parallel, usually demonstrate much more
robustness. Orbeez-SLAM [82] and Photo-SLAM [19] are
both built upon the ORB-SLAM-based pose tracking [39].
NeRF-SLAM [83] and IG-SLAM [84] utilize the dense depth
maps estimated from the tracking front-end DROID-SLAM
as additional information to supervise the training of instant-
ngp and 3DGS. NeRF-VO [85] and MGS-SLAM [86] instead
employ the sparse DPVO [87] as a faster tracker with network-
predicted dense depth maps for supervision.
C. LiDAR-Integrated SLAM Systems with Radiance Field
Radiance-field-based Visual SLAM has achieved great per-
formance in confined indoor scenes, and fusing IMU data can
further improve the robustness. But they are still challenged by
extreme violent motions, drastic lighting changes, and texture
deficiency. Studies in both radiance-field-based reconstruc-
tion [61–72] and conventional SLAM [25–35] have validated
the superiority of introducing LiDARs. For radiance-field-
based SLAM, the advantages of integrating LiDAR include,
at the very least, enhanced robustness and accuracy in pose
estimation, precise geometric supervision, facilitating efficient
sampling for NeRF and accurate initialization for 3DGS.
Powered by the LiDAR data, SHINE-Mapping [88] is
an incremental mapping framework that uses octree-based
hierarchical neural SDF to perform large-scale 3D reconstruc-
tion in a memory-efficient way. Meanwhile, NF-Atlas [89]
organizes multiple neural submaps by a pose graph, and
N3-Mapping [90] applies a voxel-oriented sliding window
mechanism to alleviate the forgetting issue with a bounded
memory footprint. Instead of purely geometric mapping, HGS-
Mapping [91] incrementally builds a dense photo-realistic
map with hybrid Gaussians in urban scenarios. All these
mapping systems assume a priori ground-truth poses, whereas
full SLAM systems that estimate poses simultaneously are
inherently more complicated. NeRF-LOAM [92] is a typical
NeRF-based LiDAR odometry and mapping method, optimiz-
ing poses and voxel embeddings concurrently. Furthermore,
LONER [93] proposes a novel information-theoretic loss func-
tion to attain real-time performance. Benefiting from a semi-
explicit representation, PIN-SLAM [94] utilizes a neural point
cloud map representation with elasticity for globally consistent
mapping. Splat-LOAM [95] exploits explicit 2DGS primitives
with spherical projection for localization and LiDAR mapping.
Going beyond merely recovering the geometric structure
of the scene, Rapid-Mapping [9] utilizes NeRF to represent

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
4
and render detailed scene textures. MM-Gaussian [36], LiV-
GS [22], and PINGS [96] perform coupled pose tracking and
dense mapping within a unified 3DGS optimization framework
using LiDAR-Camera data. While these methods demonstrate
impressive performance, they are not capable of real-time
operation, despite the inherent efficiency of 3DGS. In contrast,
decoupled tracking and Gaussian mapping approaches have
shown promising results. Notably, Gaussian-LIC [21] is the
first real-time LiDAR-Inertial-Camera 3DGS-based SLAM
system, combining continuous-time LiDAR-Inertial-Camera
odometry for tracking with 3DGS for mapping, achieving
high-fidelity photo-realistic reconstruction in real time. Sub-
sequently, GS-LIVM [37], LVI-GS [23], and GS-LIVO [24]
have also been built atop classical LiDAR-Inertial-Camera
odometry. Among them, GS-LIVM [37] and GS-LIVO [24]
adopt voxel-based map structures and initialize Gaussian co-
variances. However, all these methods heavily rely on dense
LiDAR sensors with the FoV aligned with that of the camera,
and typically initialize Gaussians solely from LiDAR points.
These approaches lead to under-reconstruction in LiDAR blind
spots and suboptimal performance when using sparse LiDAR
sensors. Notably, Gaussian-LIC incorporates online triangula-
tion of visual features to compensate for regions not covered
by the LiDAR. However, it struggles in textureless scenarios
and is sensitive to parallax between views, which affects
triangulation reliability. Furthermore, Gaussian-LIC primarily
emphasizes visual rendering quality while overlooking geo-
metric accuracy, such as dense depth rendering. The potential
of leveraging the 3D Gaussian map to enhance pose tracking
performance also remains underexplored in Gaussian-LIC.
In this paper, we propose a meticulous LiDAR-Inertial-
Camera Gaussian Splatting SLAM system. In contrast to prior
methods, our approach jointly considers visual fidelity, geo-
metric precision, and real-time performance. It simultaneously
estimates poses and constructs a photo-realistic Gaussian map
in real time, enabling high-quality RGB and depth rendering.
III. SYSTEM OVERVIEW
Fig. 2 depicts the overview of our proposed Gaussian-
LIC2, which consists of two main modules: a continuous-
time tightly-coupled LiDAR-Inertial-Camera Odometry and an
incremental photo-realistic mapping back-end with 3DGS.
In the rest sections, we first present the formulation of
continuous-time trajectory in Sec. IV and introduce the pre-
liminaries of 3DGS in Sec. V. Next, in Sec. VI, we design
a tightly-coupled LiDAR-Inertial-Camera odometry system as
the front-end which supports two optional camera factors
tightly fused within a continuous-time factor graph, including
constraints from the Gaussian map. We then utilize an efficient
but generalizable depth model to fully initialize Gaussians and
prepare mapping data for the back-end in Sec. VII. Finally, we
perform photo-realistic Gaussian mapping with depth regular-
ization and CUDA-related acceleration in Sec. VIII.
IV. CONTINUOUS-TIME TRAJECTORY FORMULATION
Two non-uniform cumulative B-splines, parameterizing the
3D rotation and the 3D translation, can jointly represent a
continuous-time trajectory. The 6-DoF poses at time t ∈
[ti, ti+1) of a continuous-time trajectory are denoted by:
R(t) = F1 (Ri−3, · · · , Ri, ti, ti+1, t) ,
(1)
p(t) = F2 (pi−3, · · · , pi, ti, ti+1, t) ,
(2)
where Rn ∈SO(3) and pn ∈R3 denote control points (n ∈
{i −3, · · · , i}). ti and ti+1 represent two adjacent knots. The
functions F1 and F2 derive poses from control points, knots,
and querying time. Refer to [34, 35, 97] for more details.
The continuous-time trajectory of IMU in the world frame
{W} is denoted as W
I T(t) =
W
I R(t), W pI(t)

. Given known
extrinsics between LiDAR/camera and IMU, we can handily
get LiDAR trajectory W
L T(t) and camera trajectory W
C T(t).
V. 3D GAUSSIAN SPLATTING REPRESENTATION
Due to the strengths of 3DGS illustrated in Sec. II, we
reconstruct a richly detailed photo-realistic map using a set
of anisotropic 3D Gaussians. Each Gaussian is characterized
by spatial position µ ∈R3, scale S ∈R3, rotation R ∈R3×3,
opacity o ∈R, and three-degree spherical harmonics coeffi-
cients SH ∈R3×16 to encode view-dependent appearance of
the scene [2]. Note that we do not compromise map quality for
speed by employing isotropic and view-independent Gaussians
as prior works [16, 20]. Instead, we retain the original expres-
sive parameterization [2] and adopt the acceleration strategies
described in Sec. X-D to pursue both quality and efficiency.
Representing the Gaussian’s ellipsoidal shape, the covari-
ance of each Gaussian is parameterized as Σ = RSST RT .
Given a camera pose C
W T = {C
W R, CpW }, which maps a 3D
point W p from the world frame {W} to the camera frame
{C}, a 3D Gaussian N(µ, Σ) can be splatted onto the image
screen, resulting in a corresponding 2D Gaussian N(µ′, Σ′):
µ′ = πc

ˆµ
e⊤
3 ˆµ

, ˆµ = C
W R µ + CpW ,
(3)
Σ′ = J C
W R Σ C
W R
T JT ,
(4)
where ei is a 3 × 1 vector with its i-th element to be 1
and the other elements to be 0. Thus, e⊤
3 ˆµ gives the depth
d of the 3D Gaussian in the camera frame. The function πc(·)
projects a 3D point on the normalized image plane to a pixel.
J ∈R2×3 represents the Jacobian of the affine approximation
to the perspective projection [2]. The projected 2D Gaussian
contributes to the image at pixel ρ =
u
v⊤with a weight:
α = o exp

−1
2(µ′ −ρ)T (Σ′)−1(µ′ −ρ)

.
(5)
By arranging all the successfully splatted 3D Gaussians in
depth order, the color, depth, and opacity at pixel ρ can be
efficiently rendered using front-to-back α-blending:
C(ρ) =
n
X
i=1
ciαi
i−1
Y
j=1
(1 −αj),
(6)
D(ρ) =
n
X
i=1
diαi
i−1
Y
j=1
(1 −αj),
(7)
O(ρ) =
n
X
i=1
αi
i−1
Y
j=1
(1 −αj),
(8)

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
5
IMU
Camera
LiDAR
[Option1
]
Reprojection Constraint
from LiDAR Map
KNN Search via ikd-Tree
Point-to-Map
Residual [L]
Frame-to-Map
Residual [C]
Inertial 
Residual [I]
Posed 
RGB & Depth
🚀Sped-up
Backward
🚀Sped-up
Render
or
Continuous-Time
Factor Graph
LiDAR Point
Visual Point
𝒏𝝅
Zero-Shot & Fast
Depth Completion
❄
Existing Gaussian Map
Depth-Regularized
Map Optimization
Map Expansion
Initialized to GS
[Option2
]
Photometric Constraint
from Gaussian Map
Figure 2: Pipeline of our real-time photo-realistic LiDAR-Inertial-Camera SLAM system, which represents the map using 3D Gaussians.
where c denotes the view-dependent color of the 3D Gaussian
derived from spherical harmonics SHb
∈R16 (as basis
functions) and the corresponding coefficients SH (as weights),
while SHb is computed from the viewing direction between
the Gaussian position µ and the camera position CpW [2].
To investigate the constraint of the Gaussian map on pose
estimation, we analytically compute the Jacobian of the 3DGS
representation w.r.t camera pose on the manifold according
to MonoGS [18], thereby avoiding the overhead of automatic
differentiation. In contrast to MonoGS that models appearance
with view-independent color, we further include the Jacobian
of high-order spherical harmonics w.r.t the camera pose.
VI. LIDAR-INERTIAL-CAMERA FUSION
We draw on the insights of classic multi-sensor fused
odometry to achieve real-time, robust, and accurate pose
estimation, paving the way for subsequent photo-realistic map-
ping. Specifically, our odometry system is inherited from and
further developed based on Coco-LIC [35], a continuous-time
tightly-coupled LiDAR-Inertial-Camera odometry using non-
uniform B-spline. By leveraging the continuous-time trajectory
representation, which inherently supports pose querying at any
timestamp corresponding to sensor measurements, we seam-
lessly and tightly fuse asynchronous, high-frequency LiDAR-
Inertial-Camera data without introducing interpolation errors.
This results in improved stability and precision [34, 35].
A. Trajectory Extension
Our LiDAR-Inertial-Camera system is initialized from a
stationary state, using a buffer of IMU measurements to
initialize IMU biases and the gravity-aligned orientation [98].
The system adopts an active sliding window of the latest 0.1
seconds for optimization, consistent with Coco-LIC [35].
Starting from time instant tκ−1, we estimate the trajectory in
[tκ−1, tκ) once the LiDAR-Inertial-Camera data in the sliding
window time interval is ready, where tκ = tκ−1 + 0.1. The
accumulated data contains all LiDAR raw points Lκ, all IMU
raw data Iκ, and the latest image frame Fκ in [tκ−1, tκ). For
simplicity, we omit other image frames captured within this
interval, which we found does not affect performance. As in
Coco-LIC, we first adaptively initialize a variable number of
control points based on the motion intensity inferred from the
IMU data, and then insert these control points into the sliding
window to extend the trajectory. The following states are then
optimized within the sliding window:
X κ = {Φ(tκ−1, tκ), xκ
Ib},
xκ
Ib = {bκ−1
g
, bκ−1
a
, bκ
g, bκ
a},
(9)
where Φ(tκ−1, tκ) denotes all control points within the inter-
val [tκ−1, tκ), parameterizing the continuous-time trajectory
of the IMU in the world frame. xκ
Ib denotes the IMU bias,
which includes the gyroscope bias bg and the accelerometer
bias ba. The IMU biases during [tκ−1, tκ) are assumed to be
constant as bκ−1
g
and bκ−1
a
. They are under Gaussian random
walk and evolve to bκ
g and aκ
g at tκ.
B. Continuous-Time Factor Graph
1) LiDAR Factor: Different from Coco-LIC with LiDAR
feature extraction and kd-Tree-organized map, we here directly
register downsampled LiDAR raw points to the LiDAR map
organized by the ikd-Tree for enhanced efficiency and accu-
racy [99]. Given a raw LiDAR point Lp ∈Lk measured at
time t, we first transform it to the world frame with the queried
pose from the continuous-time trajectory, and then search for
its five nearest neighbors in the LiDAR map to fit a 3D plane
for a point-to-plane residual:
rL = W n⊤
π
W ˆp + W dπ, W ˆp = W
L R(t)Lp + W pL(t), (10)
where W nπ and W dπ denote the unit normal vector and the
distance of the plane to the origin, respectively.
2) Inertial Factor: We define the following inertial factors:
rI =
Iω(t) −Iωm + bκ−1
g
Ia(t) −Iam + bκ−1
a

, rIb =
bκ
g −bκ−1
g
bκ
a −bκ−1
a

,
(11)
where the former is the IMU factor and the latter is the bias
factor based on the random walk process. Iωm, Iam are the
raw measurements of angular velocity and linear acceleration
of the IMU data at time t in Iκ. Iω(t) and Ia(t) are the
corresponding predicted values computed from the derivatives
of the continuous-time trajectory in Eq.(1) and Eq.(2) [35].

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
6
3) Camera Factor (Option 1, Default): Our odometry sys-
tem incorporates two types of camera factors, the first of which
is based on reprojection constraints using the reconstructed
LiDAR point map. Specifically, similar to Coco-LIC [35], we
maintain a subset of global LiDAR points stored in voxels
and associate them with image pixels by projection, KLT
sparse optical flow [100], and RASAC-based outlier removal.
Consider a global LiDAR point W p associated with the pixel
ρ =
u
v⊤in the image frame at the timestamp t. The
reprojection error for this LiDAR point is defined as:
rC1 = πc

C ˆp
e⊤
3 C ˆp

−
u
v

,
C ˆp = W
C T−1(t)W p .
(12)
4) Camera Factor (Option 2): The second type of camera
factor fully leverages the incrementally constructed Gaussian
map. Thanks to the continuous-time trajectory formulation,
we can obtain the camera pose W
C T at any time instant within
the interval [tκ−1, tκ). Given the timestamp t at which the
raw RGB image C (treated as ground truth) of frame Fκ is
captured, we render an RGB image ˆC(G, W
C T(t)) from the
Gaussian map G using Eq. (6). The rendered image is expected
to closely match the raw image, enabling optimization of the
camera pose by minimizing the rendering loss:
L = 1
2
 ˆC −C

1 + 1
2LD−SSIM( ˆC, C),
(13)
where LD−SSIM is a D-SSIM term [2]. Low-image-gradient
or low-opacity pixels are penalized [18], which is omitted
here for simplicity. Besides, in unbounded outdoor scenes
with highly variable illumination and under conditions of
fluctuating camera exposure, a standalone RGB L1 loss is
susceptible to these noises, which may corrupt gradient direc-
tions, leading to erroneous camera pose optimization. To this
end, we additionally introduce the D-SSIM loss to enhance the
optimization robustness by accounting for structural similarity.
We perform Nt iterations using the Adam optimizer, during
which the Gaussian map G is kept fixed while the camera pose
is iteratively optimized. The resulting optimized camera pose
from Eq. (13) is denoted as W
C eT = {W
C eR, W epC}, and below
is the derived photometric constraint from the Gaussian map:
rC2 = Log

W
C T(t)W
C eT −1
.
(14)
C. LiDAR-Inertial-Camera Factor Graph Optimization
We jointly fuse LiDAR-Inertial-Camera data in the factor
graph and formulate the following nonlinear least-squares
problem to efficiently optimize states X κ:
arg min
X κ
 X
∥rL∥2
ΣL +
X
∥rI∥2
ΣI +
X
∥rIb∥2
ΣIb +
X
∥rCx∥2
ΣC +
X
∥rprior∥2
Σprior

,
(15)
which is solved via the Levenberg-Marquardt algorithm in
Ceres Solver [101] and accelerated through the analytical
derivatives. The camera factor rCx can be chosen as either rC1
or rC2. The former is a reprojection constraint from the LiDAR
map, which is somewhat handcrafted but very lightweight,
while the latter is a photometric constraint from the Gaussian
map, which consumes modest GPU computing resources but is
more direct and natural. A comparative evaluation of the two
factors will be presented in Sec. X. rprior is the prior factor
from marginalization [35]. ΣL, ΣI, ΣIb, ΣC, Σprior are the
corresponding measurement covariances.
VII. DATA PREPARATION FOR GAUSSIAN MAPPING
A. Mapping Data Grouping
After finishing the optimization of the latest trajectory seg-
ment in [tκ−1, tκ), we ultimately obtain the accurate camera
pose for the latest image frame Fκ. All LiDAR raw points
within this time interval, namely Lκ, can be easily transformed
into the world frame based on the optimized continuous-time
trajectory. The posed image, together with the transformed
LiDAR points, is jointly regarded as a hybrid frame, and we
treat every fifth hybrid frame as a keyframe for photo-realistic
mapping. If the current hybrid frame is selected as a keyframe,
we merge all LiDAR points from the latest five hybrid frames
into a single point cloud. This point cloud is then projected
onto the current image plane to generate a sparse depth map.
Next, we downsample the point cloud by randomly retaining
one out of every Np points. The downsampled point cloud
is also projected onto the current image plane for coloring.
Colorzied points will be used for initializing 3D Gaussians,
which will be discussed in Sec. VIII-A2. At this stage, the
keyframe is fully constructed, consisting of a posed image, a
sparse depth map, and a set of colorized LiDAR points.
B. Instant Depth Completion
LiDARs are able to provide precise geometric priors for
Gaussian initialization. However, due to the mismatch in FoV
between the LiDAR and the camera, relying solely on LiDAR
points would lead to under-reconstruction in LiDAR blind
areas or when using the sparse LiDAR. Although the adaptive
density control (ADC) from vanilla 3DGS [2] offers a partial
relief, its effectiveness is limited in SLAM. The incremen-
tal insertion of Gaussians leads to inconsistent convergence
among different batches of Gaussians, which may misguide
gradient-based operations like cloning and splitting. Moreover,
the reliance on multi-iteration gradient accumulation hinders
the real-time performance of the system. Gaussian-LIC [21]
leverages online triangulated visual features to compensate
for regions unobserved by LiDAR. However, it struggles in
textureless areas and is sensitive to inter-frame viewpoint
variations. Therefore, further investigation is warranted to
explore how visual cues can be more effectively leveraged
for efficient Gaussian initialization. The central challenge lies
in obtaining accurate depth for image pixels.
Why should it be depth completion? With the ad-
vancement of computer vision research, numerous methods
have emerged for recovering dense depth from images. First,
monocular relative depth estimation [102, 103] has made
significant progress in predicting depth from a single RGB
image. However, it inherently suffers from scale ambiguity.
While least-squares alignment with metric LiDAR depth can
be applied, it often fails to recover accurate metric scale
across all image pixels—particularly in regions with poor

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
7
(a)
(c)
(b)
(d)
(e)
(f)
(g)
(h)
Figure 3: Details on the LiDAR depth completion: (a) Projection of
LiDAR points of the latest 5 frames. (b) Colorized LiDAR points of
the latest 5 frames. It is worth noting that the area highlighted by the
red arrow has been never scanned by the LiDAR during the entire data
acquisition. (c) Completed dense depth map. (d) Dense colored point
cloud from the completed depth map. (e) The image is divided into
30 × 30 patches, and the red dot denotes the selected pixels. (f) The
red star indicates the 3D visual points obtained by back-projecting
the selected pixels to compensate for the LiDAR blind area. (g) and
(h) respectively show the rendering results of our method without
and with depth completion.
relative depth estimation or lacking LiDAR coverage. Second,
monocular metric depth estimation [104, 105] directly predicts
scale-aware depth. However, its accuracy remains modest and
lags far behind LiDAR-based methods, especially in large-
scale outdoor environments. Third, multi-view stereo (MVS)
approaches [106, 107] estimate metric depth from multiple
posed images. Although MVS can produce accurate depth in
favorable conditions, it is sensitive to parallax and exhibits lim-
ited generalization, particularly in large-scale outdoor scenes.
In contrast, depth completion, which fuses sparse LiDAR data
with a single RGB image, offers a practical and robust solution
for recovering metric dense depth [108–110] in large-scale
scenarios. It avoids the issue of scale ambiguity and can
achieve LiDAR-comparable depth estimation accuracy.
In this paper, SPNet [110] is selected as the depth completer
due to its efficiency, compactness, and strong generalization
ability. We use it directly off the shelf, without any additional
fine-tuning. As outlined in Alg. 1 and illustrated in Fig. 3, we
feed the sparse depth map Ds and the RGB image C of the
keyframe in Sec. VII-A into SPNet to generate a completed
dense depth map Dc. We then compute the mean depth change
in the known regions before and after completion. If the depth
change exceeds a threshold ϵ1, the completion is considered a
failure (rarely occurring). After successful completion, pixels
in Dc with negative depth values or high depth gradient
magnitudes are discarded, producing the filtered dense depth
Algorithm 1: LiDAR Blind-Area Compensation
Input: Sparse depth map Ds, RGB image C
Output: Supplemented colored point cloud P
1 P ←∅, A ←∅
// Initialize outputs
2 M ←ValidMask(Ds)
3 Dc ←SPNet(C, Ds, M)
4 δ ←Mean(|(Dc −Ds)[M]|)
5 if δ < ϵ1 then
6
Dcf ←Filter(Dc)
7
Divide Ds into 30 × 30 grid patches
8
foreach patch R in grid do
9
if IsEmpty(Ds[R]) then
10
ρ ←MinValidDepthPixel(Dcf [R])
11
if Depth(ρ) < ϵ2 then
12
A ←A ∪{ρ}
13
P ←BackProject(A, I)
14 return P
map Dcf . Subsequently, we divide the input sparse LiDAR
depth map into patches of size 30 × 30 and iterate over each
of them. For patches without any valid LiDAR depth, we select
the pixel with the smallest completed depth within the patch
of Dcf and store it in a container A if its depth is less than
ϵ2. All pixels in A are back-projected to form a supplemented
colorized point cloud P, compensating for LiDAR-unobserved
regions. Finally, P is transformed to the world frame and
merged with the colored LiDAR points of the keyframe.
VIII. REAL-TIME PHOTO-REALISTIC MAPPING
A mapping back-end thread runs in parallel with the
tracking front-end thread and continuously receives sequential
keyframes from the tracker to perform real-time photo-realistic
reconstruction. Each keyframe contains an estimated camera
pose, an undistorted image, a sparse LiDAR depth map, and
a set of colored LiDAR points augmented with supplemented
visual points derived from the completed dense depth map.
A. Gaussian Map Management
Once a keyframe is received, the mapping thread will
initialize or expand the Gaussian map and optimize it.
1) Initialization of Gaussian Map: The Gaussian map is
initialized using both the colorized LiDAR points and the
supplemented visual points from the first keyframe. Specif-
ically, for each point, a new Gaussian is instantiated at its
3D location, with the zeroth degree of SH initialized using
its RGB color, opacity set to 0.1, and rotation initialized as
the identity matrix. To mitigate aliasing artifacts as discussed
in [111], we adapt the scale of each Gaussian based on its
distance to the image plane [16, 21]. The scale is modeled as
S = d
f e, where e is a 3 × 1 vector of ones, d is the depth of
the 3D point in the camera frame, and f is the focal length.
2) Expansion of Gaussian Map: Each incoming keyframe
typically captures new geometric and appearance information.
However, the colorized LiDAR points and supplemented visual
points from consecutive keyframes often contain redundant 3D

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
8
points, resulting in excessive points for Gaussian initialization.
To ensure high efficiency, we selectively initialize Gaussians
using selected 3D points. Similar to previous works [16, 21],
we first render an opacity map O from the perspective of the
current keyframe using Eq. (8). A binary mask Mo = O <
τ (τ is a constant threshold) is then constructed to identify
potentially newly observed image regions. Only the 3D points
projected into these image regions are used to initialize new
Gaussians, thereby expanding the Gaussian map.
3) Depth-Regularized Map Optimization: After the initial-
ization or expansion, we randomly sample K keyframes out of
all keyframes to optimize the Gaussian map, preventing catas-
trophic forgetting and ensuring global geometric consistency.
The selected keyframes are shuffled and sequentially used to
update the map by minimizing the rendering loss:
L = Lc + ξLd,
(16)
Lc = (1 −λ)
 ˆC −C

1 + λLD−SSIM( ˆC, C),
(17)
Ld =

 ˆD
ˆO
−Ds
!
· Md

1
, Md = Ds > 0,
(18)
where Lc denotes the RGB rendering loss computed by the
rendered image ˆC based on Eq.(6) and the raw image C, while
Ld is the depth rendering loss involved with the rendered depth
map ˆD via Eq.(7) and the sparse LiDAR depth map Ds. Note
that we utilize the rendered opacity map ˆO to normalize the
rendered depth map in incremental mapping.
B. CUDA-Related Acceleration Strategies
To improve real-time performance, previous works either
forcibly limit the number of Gaussians or simplify the Gaus-
sian representation [16, 79], which might work well indoor but
often degrades performance in unbounded and complex out-
door scenarios. On the contrary, similar to [21], we concentrate
on speeding up the CUDA-based operations, particularly the
forward and backward of the 3DGS rasterizer, so as to ensure
real-time performance without compromising the quality.
1) Fast Tile-based Culling: During the forward pass, 3D
Gaussians are splatted onto the image, resulting in elliptical 2D
Gaussians. To determine the tiles affected by each Gaussian,
the elliptical 2D Gaussian is further dilated into a circle whose
radius equals the major axis length [2]. However, such an
approximation results in overly inflated tiles influenced by
the Gaussian, particularly for highly anisotropic Gaussians,
which are common in incremental mapping systems such as
SLAM. To address this issue, we adopt a fast tile-based culling
strategy, as illustrated in Fig. 4(a). For each 2D Gaussian
N(µ′, Σ′) and the tiles affected by it, we identify the pixel
ρ within every tile where the 2D Gaussian yields the highest
contribution (with the max weight α in Eq.(5)), as in [112]. If
the weight α at the pixel ρ is smaller than the threshold
1
255,
we regard the tile as weakly affected and cull it. In this way,
the number of Gaussians per tile can be significantly reduced,
accelerating the subsequent forward and backward pass.
2) Per-Gaussian Backpropagation: During the backward
pass, gradients flow from RGB pixels back to Gaussians,
0
1
2
31
for iter in range(256 + 31)
0
1
0
2
1
0
31
30
29
0
255
254
253
224
255
254
225
255
255
226
RGB
Depth
Opacity
GS
Pixel
…
16 x 16 
tile 
…
a bucket of Gaussians
(num: 32)
(b2)
(b1)
(a)
✂
remained tiles
Figure 4: Acceleration strategies in the forward and backward pass
when optimizing the Gaussian map: (a) Culling the tiles weakly
affected by the splatted Gaussians. (b1) Every 32 Gaussians within a
tile (16 × 16 pixels) are grouped into a bucket. (b2) All buckets are
processed in parallel for (256+31) iterations (row by row). In each
iteration, all Gaussians within the bucket accumulate gradients from
different pixels for optimization with respect to the rendered RGB,
depth, and opacity maps—without incurring atomic collisions.
which is the most time-consuming stage during the map-
ping [2]. Each pixel typically receives contributions from a
different number of Gaussians, and the runtime is ultimately
dominated by the pixel associated with the largest number of
Gaussians. As a result, the remaining CUDA threads must wait
idly, leading to significant inefficiency. Also, atomic gradient
addition conflicts arise when multiple pixels backpropagate to
the same Gaussian. To address these issues, we shift from
pixel-wise parallelism to Gaussian-wise parallelism following
[113], as illustrated in Fig. 4(b1) and (b2). To be specific, we
divide the depth-sorted Gaussians within each tile into buckets
of 32 Gaussians each. Then, all Gaussians across all buckets
are processed in parallel. For a given bucket, all 32 Gaussians
simultaneously iterate over pixels within the tile to accumulate
gradients. As a result, the upper bound of the backward time
is largely determined by the number of pixels, leading to more
consistent computational load and reduced collisions. In this
way, compared to [21, 113], In addition to the RGB channels,
we also incorporate per-Gaussian gradient backpropagation for
the depth map and the opacity map.
3) Additional Strategies: a) Sparse Adam.
The original
3DGS [2] uses the Adam optimizer to update all Gaussians,
including those not involved in the current rendering, by
applying zero gradients. The computational burden becomes
progressively unsustainable as the map size increases. There-
fore, we adopt sparse Adam [113] to update only the valid
Gaussians that participate in the rendering.
b) Separated
SH. In the original 3DGS framework, the low-order and high-
order SH coefficients are concatenated before the forward
and backward passes per iteration. However, the concatenation
operation is time-consuming, especially when using three-
degree SH in our method. To address this, we handle the
low-order and high-order coefficients separately, eliminating

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
9
the concatenation. c) Efficient SSIM. We employ the highly
efficient and differentiable SSIM implemented in [113], where
separable Gaussian convolutions lead to reduced computation.
d) Warp-Level Predicate Masking.
The warp serves as a
bridge between threads and blocks, which is crucial for parallel
performance in GPU. Threads are typically operated in groups
of 32 as a warp, and performance can degrade if threads within
a warp diverge due to conditional branches. For example, in the
forward pass of 3DGS, Gaussians handled by different threads
may be deemed invalid—due to invalid depth, non-invertible
2D covariance, or insufficient weights—causing threads to exit
under varying conditions, leading to divergence. To tackle this
problem, we use flags to unify the return decisions in the
final stage, avoiding divergence. Besides, we implement early
stopping for warps in which all threads are flagged to return,
improving efficiency. e) CPU to GPU. When optimizing the
map, ground-truth images and sparse LiDAR depth maps need
to be transferred from the CPU to the GPU, which incurs a
certain amount of overhead. However, preloading them onto
the GPU is not allowed due to limited memory. Thus, we store
them in pinned memory and utilize non-blocking transfer to
move data from the CPU to the GPU. This allows the transfer
to be executed in parallel with GPU computation, improving
computing asynchrony and overall throughput.
IX. DATASETS FOR EVALUATION
This section describes the datasets used for evaluation,
encompassing both public and self-collected ones. The pub-
lic datasets, including R3LIVE [30], FAST-LIVO [32, 33],
MCD [114], and M2DGR [115], are mainly employed for
localization and in-sequence novel view rendering evaluation.
Our self-collected dataset supports both in-sequence and out-
of-sequence novel view rendering evaluation, and enables
quantitative evaluation of rendered depth. Tab. VI summarizes
the duration and length of each sequence.
A. Public Dataset
The R3LIVE dataset [30] and FAST-LIVO dataset [32,
33] are both collected within the campuses using a handheld
device equipped with a Livox Avia LiDAR at 10 Hz and its
built-in IMU at 200 Hz, and a 15 Hz RGB camera. Closely
aligned with the camera’s FoV and performing non-repetitive
scanning, the projected points of the adopted solid-state Li-
DAR are distributed across the image, while still suffering
from non-negligible blind spots as shown in Fig. 3. Six outdoor
sequences from the R3LIVE dataset that are well-suited for
mapping are selected for evaluation. The FAST-LIVO dataset
used in this work includes all sequences from FAST-LIVO and
the first two sequences from FAST-LIVO2, whose high-quality
images make the dataset particularly suitable for mapping
tasks. Notably, the R3LIVE and FAST-LIVO datasets also
provide challenging sequences with LiDAR or visual degen-
eration, such as degenerate seq 00-01, LiDAR Degenerate,
and Visual Challenge. Although the ground-truth trajectory
is unavailable, the start and end poses coincide, allowing
localization evaluation via start-to-end drift. When stereo pairs
are available, only the left image (640 × 512) is used.
Livox Mid-360
Livox Avia
RealSense D455
Hikvision
MV-CA013-21UC
Figure 5: Two types of LIC sensor rig for self-collected dataset.
The MCD dataset [114] provides sequences across seasons
and continents, captured in large-scale campuses. We here pick
the sequences (tuhh day 02-04) with the repetitive mechanical
spinning Ouster OS1-64 LiDAR at 10Hz, a 30 Hz RGB camera
(640 × 480) and a 400 Hz IMU. The sparsity of spinning
LiDAR poses greater challenges for LiDAR-based Gaussian
mapping. Due to the extensive scale of the environment, we
truncate the sequences to the first 200 seconds (∼300 m). The
dataset provides highly accurate ground-truth trajectories for
quantitative evaluation of localization.
The M2DGR dataset [115], collected by a ground robot
during both day and night using a Velodyne VLP-32C LiDAR
at 10 Hz, an IMU at 100 Hz, and a 10 Hz RGB camera (640×
480), presents a more challenging case due to the extremely
sparse LiDAR measurements. A set of sequences (room 01-
03) under favorable lighting are selected for evaluation.
B. Self-Collected Dataset
To further evaluate out-of-sequence novel view synthesis
and quantitatively assess the accuracy of the rendered depth,
we collect additional sequences leveraging two handheld de-
vices in Fig. 5, enriching LiDAR modality diversity. The first
setup integrates a Livox Mid-360 LiDAR sampled at 10 Hz
paired with a 30 Hz RealSense D455 RGB camera (640×480).
Compared to Avia, Mid-360 offers a wider horizontal FoV
but shorter maximum ranging distance, presenting unique
reconstruction challenges. Meanwhile, the second one follows
the LIV handheld 1 configuration, combining a Livox Avia
LiDAR at 10 Hz with a Hikvision MV-CA013-21UC RGB
camera at 15 Hz (640 × 512). Both handheld systems utilize
the built-in 200 Hz IMU from the LiDAR, with the LiDAR
and the IMU factory-synchronized and well-calibrated. The
spatial and temporal extrinsics between the camera and the
IMU are carefully calibrated through Kalibr [116].
For each sequence, we capture a long and smooth trajectory
with loop closures, ensuring the scene is revisited for out-
of-sequence evaluation and making the captured data well-
suited for mapping purposes. Due to unreliable GPS between
buildings, GPS-based pose estimation is often unreliable.
We generate continuous-time ground-truth trajectories using
a LiDAR-Inertial-Camera SLAM framework that incorporates
1https://github.com/xuankuzcr/LIV handhold

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
10
continuous-time loop closure [117] to enhance accuracy. Note
that our captured scenes are not excessively large (i.e., not at
an urban scale), and the estimated ground-truth trajectories are
sufficiently accurate for evaluation purposes. For generating
ground-truth depth maps, we query the camera pose from
the ground-truth trajectory and project the temporally closest
LiDAR frame onto the image. Each collected trajectory is
divided into two segments, used respectively for in-sequence
and out-of-sequence evaluations, as illustrated in Fig. 9.
X. EXPERIMENT RESULTS
A. Implementation Details
We implement the proposed system fully in C++ and
CUDA. The tracking and mapping modules run in parallel
and communicate via ROS, with the mapping built upon
the LibTorch framework. The depth completion network SP-
Net [110] is deployed with TensorRT for efficient inference.
For odometry, we reuse the ikd-Tree parameters from FAST-
LIO2 [99], set the camera pose optimization step Nt in
Sec. VI-B4 to 30. For mapping data preparation, we set Np in
Sec. VII-A to 10 (Avia, Mid-360), 5 (OS1-64), and 1 (VLP-
32C). ϵ1 and ϵ2 in Sec. VII-B are set to 0.1 m and 50 m,
respectively. For map expansion, we set τ in Sec. VIII-A2 to
0.99. As for map optimization, we set the loss weighting λ to
0.2 and ξ to 0.005, and the number K of selected keyframes
to 100. All the learning rates for Gaussian attributes follow
vanilla [2] but do not decay with schedulers. We keep the same
hyperparameters in all sequences with the same sensor setup
to ensure a fair and comprehensive evaluation. Experiments
are run on a desktop PC with an NVIDIA RTX 4090 GPU
(24 GB VRAM), Intel i9-13900KF CPU, and 64 GB RAM.
B. Baselines
This paper focuses on developing an elaborated radiance-
filed-based SLAM system capable of robust and accurate pose
estimation while constructing fine-grained photo-realistic 3D
maps in real time. Therefore, our comparative analysis specif-
ically targets authentic SLAM systems, explicitly excluding
pure mapping systems that require a priori ground-truth poses.
We first evaluate localization performance against state-
of-the-art traditional LiDAR-Inertial-Camera SLAM systems,
including optimization-based LVI-SAM [28] and filter-based
R3LIVE [30] and FAST-LIVO2 [33]. We also include two
neural SLAM methods: PIN-SLAM [94], a neural LiDAR
SLAM approach utilizing point-based NeRF for geometry-
only reconstruction, and DBA-Fusion [118], which integrates
IMU data into DROID-SLAM [43], currently the most ad-
vanced neural visual SLAM framework. We run PIN-SLAM
by feeding in undistorted LiDAR point clouds.
Furthermore, to evaluate both the localization and photo-
realistic mapping performances of the entire system, we
compare against photo-realistic radiance-field-based SLAM,
including both NeRF-based and 3DGS-based. 1) NeRF-based:
As no related LiDAR-Camera system is publicly available,
we adapt the state-of-the-art RGB-D NeRF-based method Co-
SLAM [13] with pseudo RGB-D input by merging LiDAR
scans. When rendering the full image, we use SPNet [110] to
complete the LiDAR depth maps for depth-guided sampling.
2) 3DGS-based: We compare with the state-of-the-art RGB-
only method MonoGS [18], which elegantly derives pose
gradients. Given the scarcity of available LiDAR-Camera
3DGS-based methods, we run the state-of-the-art RGB-D
3DGS-based approach SplaTAM [16] using pseudo RGB-
D images. We also compare against the RGB-D inertial
method, MM3DGS-SLAM [20], which extends SplaTAM by
incorporating IMU data, and include a comparison with the
LiDAR-Inertial-Camera Gaussian Splatting SLAM framework
Gaussian-LIC [21], where the sky modeling is disabled to
facilitate depth rendering. For fair comparison, all methods
are evaluated without loop closure and post-processing.
C. Experiment-1: Evaluation of Localization
We validate the robustness and accuracy of our method
tracking in both challenging degenerate and large-scale envi-
ronments. In particular, we compare two types of camera fac-
tors presented in Sec. VI-B: option 1 (LiDAR map reprojection
constraint, Ours-1) and option 2 (Gaussian map photometric
constraint, Ours-2). Also, we investigate the effectiveness
of the camera factor option 2 when the depth supervision
(Eq. (18)) is disabled, denoted as Ours-2-w/o-d.
1) Challenging Degenerate Sequences: Sequences degen-
erate seq 00-01, LiDAR Degenerate, and Visual Challenge,
provided in the R3LIVE and FAST-LIVO datasets, exhibit
severe degradation, such as the solid-state LiDAR with small
FoV facing the ground or walls, and the camera facing
textureless surfaces or undergoing aggressive motions (Vi-
sual Challenge). Tab. I reports the start-to-end drift (and
rotation) errors. Methods that rely solely on the rendering loss
from radiance field maps for localization, including MonoGS,
Co-SLAM, SplaTAM, and MM3DGS-SLAM, achieve excel-
lent localization accuracy in confined indoor scenarios with
moderate view changes. However, they tend to fail in large-
scale outdoor scenes with significant viewpoint variation, as
the camera can easily enter regions where the map optimiza-
tion has not yet converged. MonoGS is the only one that lacks
an absolute scale, we align its estimated trajectory with that
of Ours-1 for obtaining the absolute scale. MM3DGS-SLAM,
aided by IMU, completes the sequence degenerate seq 00,
but still fails on the rest. Based on a point-based NeRF pre-
sentation, PIN-SLAM optimizes the poses via point-to-model
SDF loss [94], which is similar to the point-to-plane metric
in ICP. It builds a purely geometric map and achieves robust
pose optimization even in challenging lighting conditions, but
suffers in LiDAR degradation scenarios where environmen-
tal structures fail to provide sufficient constraints for pose
optimization. Moreover, relying solely on the LiDAR data
and constant velocity assumaption for initial pose, PIN-SLAM
crashes on the sequence Visual Challenge where aggressive
motions happen. As a learning-based visual-inertial odometry
system, DBA-Fusion exhibits greater robustness to LiDAR
degradation, and its tight fusion with IMU data facilitates
handling of aggressive motions. Nevertheless, excessive view
changes in the sequence Visual Challenge adversely affect the
accuracy of the predicted dense optical flow thus significantly
deteriorates the pose tracking in DBA-Fusion.

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
11
Table I: Localization Evaluation: The start-to-end drift error (translation | rotation) on challenging sequences. The best are in bold and the
second best are underscored. × means failure. Note that, except for the sequence LiDAR Degenerate, which is an indoor sequence, all the
others are outdoor sequences. Hybrid means using both the point cloud map and the 3DGS map for localization.
Method
Sensor
Map Type
(for Loc)
Sequence
degenerate seq 00
degenerate seq 01
LiDAR Degenerate
Visual Challenge
MonoGS [18]
C
3DGS
14.54m
|
67.40°
×
×
×
Co-SLAM [13]
L + C
NeRF
12.37m
|
30.50°
×
×
×
SplaTAM [16]
L + C
3DGS
17.96m
|
16.75°
×
×
×
MM3DGS-SLAM [20]
L + I + C
3DGS
9.14m
|
14.02°
×
×
×
PIN-SLAM [94]
L
NeRF
9.07m
|
30.05°
13.53m
|
29.29°
4.13m
|
3.71°
×
DBA-Fusion [118]
I + C
PointCloud
0.53m
|
8.98°
1.87m
|
1.93°
0.46m
|
4.66°
3.19m
|
6.21°
LVI-SAM [28]
L + I + C
PointCloud
0.08m
|
6.60°
0.11m
|
2.67°
×
×
R3LIVE [30]
L + I + C
PointCloud
0.04m
|
0.41°
0.11m
|
0.55°
8.52m
|
2.99°
0.21m
|
0.69°
FAST-LIVO2 [33]
L + I + C
PointCloud
5.07m
|
8.38°
2.41m
|
6.20°
0.02m
|
2.40°
0.02m
|
0.14°
Gaussian-LIC [21]
L + I + C
PointCloud
0.04m
|
0.54°
0.06m
|
0.62°
0.05m
|
2.63°
0.07m
|
0.30°
Ours-1
L + I + C
PointCloud
0.04m
|
0.58°
0.04m
|
0.55°
0.05m
|
2.58°
0.06m
|
0.23°
Ours-2
L + I + C
Hybrid
0.03m
|
0.43°
0.05m
|
0.59°
0.05m
|
2.50°
0.02m
|
0.11°
Ours-2-w/o-d
L + I + C
Hybrid
7.12m
|
6.22°
0.45m
|
0.80°
3.27m
|
5.92°
0.10m
|
0.89°
Table II: Localization Evaluation: The RMSE (m) of APE results on
large-scale outdoor sequences in MCD dataset [114].
tuhh day 02
tuhh day 03
tuhh day 04
MonoGS [18]
29.53
42.54
25.4
Co-SLAM [13]
47.26
×
×
SplaTAM [16]
×
34.49
×
MM3DGS-SLAM [20]
17.53
20.14
26.33
PIN-SLAM [94]
0.16
0.11
0.38
DBA-Fusion [118]
1.86
1.25
2.71
LVI-SAM [28]
0.14
0.13
0.18
R3LIVE [30]
0.10
0.13
0.15
FAST-LIVO2 [33]
0.08
0.11
0.13
Gaussian-LIC [35]
0.08
0.09
0.10
Ours-1
0.08
0.08
0.09
Ours-2
0.08
0.09
0.09
Among all the comparisons, LiDAR-Inertial-Camera fusion-
based methods with each module meticulously designed show-
case more robust and accurate performance. Rather than
separately fusing LiDAR-inertial and visual-inertial data like
LVI-SAM, R3LIVE, and FAST-LIVO2, our method maintains
a single unified system that jointly and tightly integrates
LiDAR-Inertial-Camera data within a continuous-time factor
graph, achieving superior overall performance. Fig. 6 shows
the trajectory and colored LiDAR map produced by Ours-
2, where the odometry successfully overcomes severe LiDAR
degradation challenges and returns to the origin with minor
drift. A set of subfigures in Fig. 6 presents rendered images
from three different viewpoints corresponding to poses pre-
dicted by the IMU, refined by Gaussian-map-based optimiza-
tion (W
C eT in Eq. (14)), and further refined through LiDAR-
Inertial-Camera factor graph optimization (Sec. VI-C). The
increasing similarity between the rendered and raw images
visually demonstrates the progressive refinement of the esti-
mated pose. Interestingly, Ours-2 outperforms Ours-1 on the
Visual Challenge sequence, which features textureless white
walls. This improvement can be attributed to the photometric
constraints derived from the Gaussian map, which enhance
Ours-2’s robustness in low-texture environments compared
to the optical-flow-based Ours-1. Without depth supervision,
Ours-2-w/o-d shows reduced accuracy. The depth regulariza-
tion prevents the Gaussian map from overfitting to training
(a)
(b)
(c)
(d)
SSIM: 0.326
SSIM: 0.455
SSIM: 0.489
Figure 6: The trajectory and colored LiDAR point cloud map output
by the continuous-time odometry tightly fused with the Gaussian
map. Even undergoing severe degradation when the solid-state Li-
DAR faces the plain ground, the odometry performs pose estimation
accurately with minor start-to-end drift error, benefiting from the
photometric constraints provided by the Gaussian map. (a)-(d) are
the raw image of the current view and rendered images from poses
predicted by the IMU, refined by Gaussian-map-based optimization,
and further optimized through LiDAR-Inertial-Camera factor opti-
mization, respectively. Color discrepancies between the ground-truth
and rendered images reflect variable illumination and varying camera
exposure, highlighting the importance of the D-SSIM loss in Eq.(13).
views, which enables more robust novel view synthesis with
fewer artifacts, as discussed in Sec. X-D. This is critical
since localization based on Gaussian maps inherently requires
continuous novel view synthesis from varying perspectives.
2) Large-Scale Outdoor Sequences: Tab. II displays the
localization error across different methods in the large-scale
scenes without sensor degradation. Rendering-based methods
including MonoGS, Co-SLAM, SplaTAM, and MM3DGS-
SLAM, exhibit reduced accuracy in pose estimation. Although
MM3DGS-SLAM benefits from IMU-based pose initializa-
tion, its loosely coupled integration without accounting for

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
12
Table III: Evaluation of RGB rendering (in-sequence novel view) on public datasets.
Sequence
Rendering Performance (PSNR↑SSIM↑LPIPS↓)
MonoGS [18]
Co-SLAM [13]
MM3DGS-SLAM [20]
Gaussian-LIC [21]
Ours
R3LIVE (Avia)
degenerate seq 00
14.59 | 0.460 | 0.624
14.42 | 0.286 | 0.810
19.35 | 0.698 | 0.212
21.94 | 0.777 | 0.171
21.99 | 0.782 | 0.162
degenerate seq 01
14.66 | 0.438 | 0.692
14.88 | 0.267 | 0.822
18.57 | 0.621 | 0.277
22.80 | 0.787 | 0.169
22.94 | 0.790 | 0.159
hku campus 00
15.41 | 0.464 | 0.675
15.50 | 0.302 | 0.794
18.70 | 0.621 | 0.319
24.38 | 0.787 | 0.162
24.34 | 0.787 | 0.156
hku campus 01
8.39 | 0.065 | 0.873
12.92 | 0.294 | 0.796
16.97 | 0.593 | 0.271
20.55 | 0.671 | 0.275
20.52 | 0.674 | 0.236
hku park 00
13.24 | 0.319 | 0.733
14.02 | 0.218 | 0.790
14.98 | 0.363 | 0.396
17.09 | 0.466 | 0.354
17.17 | 0.469 | 0.340
hku park 01
12.22 | 0.289 | 0.790
13.65 | 0.207 | 0.815
15.79 | 0.392 | 0.409
19.45 | 0.529 | 0.340
19.46 | 0.529 | 0.325
FAST-LIVO (Avia)
LiDAR Degenerate
25.44 | 0.779 | 0.645
23.59 | 0.622 | 0.496
28.58 | 0.828 | 0.157
30.09 | 0.829 | 0.155
30.36 | 0.831 | 0.144
Visual Challenge
15.36 | 0.563 | 0.674
15.96 | 0.365 | 0.711
17.62 | 0.712 | 0.240
23.26 | 0.821 | 0.162
23.44 | 0.822 | 0.157
hku1
13.44 | 0.385 | 0.787
15.98 | 0.269 | 0.738
21.76 | 0.713 | 0.162
23.82 | 0.757 | 0.153
23.74 | 0.758 | 0.149
hku2
21.24 | 0.559 | 0.691
20.07 | 0.398 | 0.631
26.29 | 0.754 | 0.156
29.11 | 0.798 | 0.153
29.08 | 0.798 | 0.148
Retail Street
18.74 | 0.574 | 0.537
17.32 | 0.375 | 0.669
21.55 | 0.694 | 0.162
24.15 | 0.770 | 0.128
24.37 | 0.775 | 0.121
CBD Building 01
17.11 | 0.630 | 0.640
18.16 | 0.508 | 0.644
22.13 | 0.812 | 0.126
25.16 | 0.851 | 0.104
25.20 | 0.851 | 0.103
MCD (OS1-64)
tuhh day 02
8.72 | 0.158 | 0.893
12.65 | 0.295 | 0.766
11.15 | 0.478 | 0.350
19.99 | 0.621 | 0.312
20.35 | 0.626 | 0.262
tuhh day 03
8.09 | 0.136 | 0.896
12.48 | 0.407 | 0.684
11.36 | 0.542 | 0.301
21.09 | 0.666 | 0.273
21.38 | 0.672 | 0.229
tuhh day 04
11.73 | 0.250 | 0.805
13.02 | 0.413 | 0.626
13.86 | 0.384 | 0.398
19.27 | 0.528 | 0.329
19.27 | 0.528 | 0.310
M2DGR (VLP-32C)
room 01
14.92 | 0.573 | 0.643
15.01 | 0.317 | 0.880
12.27 | 0.554 | 0.458
17.04 | 0.697 | 0.365
17.45 | 0.721 | 0.282
room 02
13.57 | 0.515 | 0.697
14.94 | 0.318 | 0.877
11.44 | 0.565 | 0.465
17.32 | 0.705 | 0.375
17.72 | 0.725 | 0.291
room 03
16.15 | 0.627 | 0.587
14.88 | 0.370 | 0.841
12.50 | 0.573 | 0.475
17.19 | 0.701 | 0.401
17.38 | 0.709 | 0.334
IMU bias limits accuracy. In contrast, PIN-SLAM and DBA-
Fusion perform better, with PIN-SLAM in particular approach-
ing the performance of traditional multi-sensor-based methods.
Attributed to the continuous-time trajectory representation,
which effectively handles LiDAR distortion and efficiently
fuses high-rate IMU data, our method achieves the best lo-
calization accuracy. In non-degenerate scenarios, visual infor-
mation has a relatively minor impact on localization accuracy,
resulting in similar performance between Ours-1 and Ours-2.
D. Experiment-2: Evaluation of Mapping
1) Evaluation Protocols: We assess the performance of
photo-realistic mapping by evaluating the quality of rendered
images generated from the Gaussian map. To this end, we
adopt several widely used metrics, including Peak Signal-
to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
and Learned Perceptual Image Patch Similarity (LPIPS)[2].
Consistent with MonoGS and SplaTAM, we use AlexNet[119]
as the backbone network for LPIPS evaluation. When ground-
truth depth maps are available, we also evaluate geometric
accuracy using Depth-L1 error computed over valid depth
regions. It should be noted that we adopt Ours-1 in the
mapping evaluation and the following experiments.
Many existing methods [23, 24, 37] report rendering per-
formance at their respective training views. However, such
evaluations may be biased due to overfitting and do not reliably
reflect the quality of photo-realistic mapping, particularly at
novel views. To ensure a comprehensive and fair evaluation,
we assess rendering quality on both in-sequence and out-of-
sequence novel views, explicitly excluding training views.
In our evaluations, all compared methods are constrained
to use the same training views, selected using the keyframing
strategy described in Sec. VII-A, while the remaining non-
keyframes are used for evaluating in-sequence novel view
rendering. To eliminate the influence of pose estimation errors
on mapping performance, all methods are provided with our
estimated poses for evaluation on public datasets. For our self-
collected dataset where ground-truth poses are available, both
our method and the baselines use ground-truth poses to avoid
trajectory alignment errors, which could significantly affect the
evaluation of out-of-sequence novel view rendering.
2) In-Sequence Novel View Synthesis: Tab. III and Tab. IV
show quantitative evaluation of the rendering results of in-
sequence novel views across both the pulic datasets and our
self-collected dataset. Fig. 7 and Fig. 8 present the novel
view renderings of both RGB and depth in sequence. The
RGB-only method MonoGS initializes Gaussians at a preset
constant depth with random noise and subsequently inserts
new Gaussians based on the rendered depth statistics. In
small-scale indoor scenarios, the optimization of the inserted
Gaussians may gradually converge. However, it becomes much
more challenging in larger scenes, where more and more
Gaussians with incorrect positions accumulate, resulting in nu-
merous floaters and poor visual quality. Incorporating LiDARs
with accurate geometric priors can substantially alleviate this
problem. For example, Co-SLAM leverages the LiDAR depth
for ray sampling and supervision of the neural implicit map
optimization, and its rendered results show relatively clear
structures. It is capable of rendering beyond the LiDAR FoV
based on the optimized neural representation, as shown in
Fig. 7 and Fig. 8. However, the renderings of Co-SLAM appear
noisy with severe artifacts, even though we specially utilize
completed depth maps to guide the sampling during full-image
rendering. In contrast, the LiDAR-incorporated 3DGS-based
methods MM3DGS-SLAM and Gaussian-LIC exhibit better

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
13
GT-RGB
MonoGS [18]
Co-SLAM [13]
MM3DGS-SLAM [20]
Gaussian-LIC [21]
Ours
GT-RGB
MonoGS
Co-SLAM
MM3DGS-SLAM
Gaussian-LIC
Ours
Figure 7: Qualitative results of RGB rendering (in-sequence novel view) on public datasets. Regions not observed by the LiDAR throughout
the entire process are indicated by the red arrow, floaters are highlighted with green boxes, and key details are marked with yellow boxes.

<!-- page 14 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
14
GT-RGB
LiDAR-Depth
Co-SLAM [13]
MM3DGS-SLAM [20]
Gaussian-LIC [21]
Ours
GT-RGB
LiDAR-Depth
Co-SLAM
MM3DGS-SLAM
Gaussian-LIC
Ours
Figure 8: Qualitative results of depth rendering (in-sequence novel view) on public datasets. Sparse LiDAR depth at the viewpoints is also
shown in the second column.

<!-- page 15 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
15
Table IV: Evaluation of RGB and depth rendering (in-sequence & out-of-sequence novel view) on self-collected datasets. Depth-L1 (m).
LiDAR
Sequence
Method
In-Sequence Novel View
Out-of-Sequence Novel View
PSNR↑
SSIM↑
LPIPS↓
Depth-L1↓
PSNR↑
SSIM↑
LPIPS↓
Depth-L1↓
Mid-360
Liberal Arts Group 01
MM3DGS-SLAM [20]
22.89
0.721
0.298
0.65
21.05
0.653
0.360
0.65
Gaussian-LIC [21]
25.12
0.759
0.297
0.92
22.08
0.670
0.350
0.81
Ours
25.35
0.761
0.278
0.29
22.10
0.671
0.332
0.30
Liberal Arts Group 02
MM3DGS-SLAM [20]
17.96
0.574
0.271
0.67
16.98
0.444
0.351
0.72
Gaussian-LIC [21]
23.02
0.703
0.210
0.66
19.95
0.520
0.275
0.71
Ours
23.42
0.713
0.186
0.30
20.05
0.528
0.256
0.37
Liberal Arts Group 03
MM3DGS-SLAM [20]
17.74
0.493
0.385
0.96
13.87
0.347
0.478
1.22
Gaussian-LIC [21]
22.59
0.656
0.236
0.89
18.28
0.458
0.351
1.25
Ours
22.79
0.662
0.222
0.44
18.28
0.459
0.345
0.67
Medical Building 01
MM3DGS-SLAM [20]
16.45
0.642
0.266
1.24
16.43
0.574
0.321
1.34
Gaussian-LIC [21]
22.77
0.737
0.199
1.28
19.40
0.632
0.268
1.64
Ours
22.89
0.741
0.188
0.62
19.54
0.633
0.257
0.84
Medical Building 02
MM3DGS-SLAM [20]
16.29
0.685
0.217
0.91
19.62
0.679
0.222
0.75
Gaussian-LIC [21]
23.15
0.758
0.177
1.53
21.33
0.685
0.216
1.35
Ours
23.11
0.760
0.174
0.54
21.57
0.691
0.213
0.53
Avia
Lecture Hall
MM3DGS-SLAM [20]
19.54
0.671
0.198
0.74
18.73
0.624
0.236
0.75
Gaussian-LIC [21]
21.60
0.710
0.178
0.68
20.32
0.669
0.207
0.73
Ours
21.78
0.719
0.167
0.39
20.40
0.672
0.205
0.45
Robot Center
MM3DGS-SLAM [20]
18.83
0.631
0.190
0.59
16.63
0.558
0.260
0.47
Gaussian-LIC [21]
22.89
0.693
0.206
0.92
18.81
0.600
0.264
0.79
Ours
22.95
0.699
0.188
0.29
18.80
0.602
0.257
0.25
Bell Tower 01
MM3DGS-SLAM [20]
18.13
0.600
0.300
0.67
16.40
0.592
0.299
0.74
Gaussian-LIC [21]
25.38
0.717
0.282
0.92
22.87
0.676
0.324
0.91
Ours
25.53
0.720
0.256
0.27
22.87
0.676
0.298
0.35
Bell Tower 02
MM3DGS-SLAM [20]
16.49
0.661
0.278
0.87
17.01
0.620
0.320
1.14
Gaussian-LIC [21]
26.42
0.751
0.285
1.18
23.68
0.716
0.321
1.77
Ours
26.85
0.756
0.242
0.29
24.17
0.708
0.276
0.41
Bell Tower 03
MM3DGS-SLAM [20]
16.61
0.651
0.256
0.90
14.08
0.603
0.294
1.02
Gaussian-LIC [21]
27.29
0.748
0.296
1.47
25.02
0.735
0.278
1.68
Ours
27.63
0.756
0.244
0.33
25.43
0.738
0.238
0.42
rendering performance. They reliably initialize Gaussians from
precise LiDAR points. Nonetheless, MM3DGS-SLAM repre-
sents the scene as isotropic and view-independent Gaussians to
improve running speed, but unfortunately still fails to achieve
real-time performance while sacrificing the visual quality.
Note that MM3DGS-SLAM uses around three times as many
Gaussians as our method to model the scene. Gaussian-LIC
underutilizes precise LiDAR depth and ignores to optimize
the geometric quality of the Gaussian map, leading to poor-
quality rendered depth. In addition, both methods struggle to
accurately reconstruct areas unobserved by the LiDAR.
Our method attains the best in-sequence novel view ren-
dering performance, both quantitatively and qualitatively. The
system renders sharper RGB images with fewer artifacts at
novel views, beyond only overfitting to the training views. It
successfully reconstructs regions never scanned by the LiDAR,
such as the ceiling beyond the reach of the VLP-32C, as
demonstrated in the last row of Fig. 7. Moreover, despite
relying on sparse LiDAR depth, our method is able to produce
high-quality depth maps across different LiDAR modalities.
It is worth noting that severe rolling shutter distortion in
the images of the M2DGR dataset affects the quality of the
rendered depth, as illustrated in the last row of Fig. 8.
3) Out-of-Sequence Novel View Synthesis:
We further
evaluate out-of-sequence novel view rendering on our self-
collected dataset, which presents a greater challenge than in-
sequence rendering. For this assessment, views are sampled
at 10 Hz along the out-of-sequence trajectory. As shown in
Tab. IV, our method achieves the best results. Fig. 9 illustrates
the camera trajectories and the corresponding novel view
rendering results along the out-of-sequence path. Compared
to our closest baseline, Gaussian-LIC, our Gaussian map is
more effectively regularized and optimized, leading to sharper
and higher-fidelity novel view synthesis with fewer artifacts.
E. Experiment-3: Offline Gaussian Mapping
As a SLAM system, our method incrementally estimates
poses and constructs a photo-realistic map from sequential
sensor data, enabling real-time perception for robotic appli-
cations. In contrast, offline methods disregard incremental or
online processing capabilities and typically perform heavy,
computationally intensive optimization only after all sensor
data has been collected. Existing offline Gaussian Mapping
methods typically fall into two paradigms, namely per-scene
optimization and generalizable feed-forward models.
1) Per-Scene Optimization: We first compare our method
to
the
state-of-the-art
LiDAR-based
3DGS
framework,
LetsGo [70], on the Lecture Hall sequence, which spans 105
seconds. To ensure a fair comparison, LetsGo is provided
with the same sparse LiDAR depth maps and our estimated
poses. Additionally, it uses the same set of training views
as our method and is trained for 30,000 iterations. Tab. V
presents the results, where we report the total time from the

<!-- page 16 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
16
GT
MM3DGS-SLAM [20] Gaussian-LIC [21]
Ours
GT
Gaussian-LIC
Ours
MM3DGS-SLAM
(a)
(b)
(c)
(d)
Figure 9: Qualitative results of RGB and depth rendering (out-of-
sequence novel view) on our self-collected dataset. (a-d) The green
path represents the trajectory for collecting training views, the red
path shows the out-of-sequence trajectory for evaluation, and the
yellow stars indicate the selected out-of-sequence novel views. The
sky regions in rendered depth maps are masked in black.
beginning of data acquisition to the completion of reconstruc-
tion. While LetsGo performs time-consuming optimization
after receiving the full dataset, our method completes the
reconstruction within the duration of data acquisition and
Table V: Comparison with LiDAR-based offline Gaussian mapping
method on the sequence Lecture Hall (span: 105 seconds). The total
time consumption reported here is the duration of data acquisition
time + reconstruction time.
PSNR↑
SSIM↑
LPIPS↓
Depth-L1 (m)↓
Time (s)↓
LetsGo [70]
21.81
0.708
0.170
0.37
105 + 347
Ours
21.78
0.719
0.167
0.39
105
Ours-W-Refine
22.28
0.730
0.148
0.37
105 + 24
GT
Ours
MVSplat
Figure 10: Comparison with the feed-forward method, MVSplat [76].
Image renderings from both MVSplat and ours are shown.
still achieves comparable performance. Furthermore, after an
additional 10,000 iterations of refinement (requiring only 24
seconds), our method significantly outperforms LetsGo. This
demonstrates that our approach not only supports real-time
reconstruction but can also achieve higher accuracy when post-
acquisition optimization is permitted.
2) Feed-Forward Model: We also compare our method with
the state-of-the-art feed-forward Gaussian mapping approach,
MVSplat [76], which predicts a Gaussian map from a small
set of posed images (two by default). For achieving good
performance with MVSplat, we carefully select a pair of
images that provides sufficient parallax and maximal scene
coverage, using poses estimated by our method. Fig. 10 shows
the rendering results on the degenerate seq 00 and hku1 se-
quences. While MVSplat is capable of predicting pixel-aligned
Gaussians within seconds, it often produces blurry renderings
and suffers from parallax sensitivity. In contrast, our method
consistently generates sharp, photo-realistic images.
F. Runtime Analysis
We evaluate the real-time performance of our method across
all datasets. Following the definition in Gaussian-LIC [21], a
system is considered to be real-time capable if it completes
processing within the duration of sensor data acquisition,
without extensive post-processing. During runtime analysis,
we evaluate our implementation using Ours-1 as the tracking
module for pose estimation. The system achieves an average
pose estimation frequency of 10 Hz, while depth completion
inference requires only 10–20 ms per frame. Therefore, we pri-
marily focus on the time consumption in the mapping thread.
As shown in Tab. VI, the most time-consuming components
in the mapping thread are primarily the forward and backward
pass of the rasterizer as well as the Adam optimizer up-

<!-- page 17 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
17
Table VI: Runtime Analysis (Time unit: s). Fwd.: forward time; Bwd.:
backward time; Adam: optimizer update time; Total: total mapping
time; Dur.: rosbag duration; Len.: trajectory length (m).
Fwd.
Bwd.
Adam
Total
Dur.
Len.
R3LIVE
degenerate seq 00
9
16
9
38
87
42
degenerate seq 01
11
15
8
38
86
61
hku campus seq 00
69
69
45
195
202
172
hku campus seq 01
154
73
48
285
304
337
hku park 00
62
87
55
218
228
212
hku park 01
138
112
73
340
362
354
FAST-LIVO
LiDAR Degenerate
7
12
6
29
78
38
Visual Challenge
18
29
14
71
162
78
hku1
20
28
15
71
128
64
hku2
13
21
11
51
105
59
Retail Street
15
26
15
64
135
66
CBD Building 01
12
19
10
48
119
33
MCD
tuhh day 02
74
57
35
179
200
306
tuhh day 03
70
56
34
173
200
276
tuhh day 04
43
46
27
127
187
298
M2DGR
room01
9
17
11
40
75
27
room02
8
16
10
38
89
46
room03
21
43
28
100
195
71
Self-collected
Liberal Arts Group 01
8
9
3
24
101
42
Liberal Arts Group 02
23
35
16
88
225
59
Liberal Arts Group 03
39
50
25
132
285
102
Medical Building 01
9
10
4
28
113
53
Medical Building 02
10
12
4
32
120
38
Lecture Hall
9
15
6
36
105
45
Robot Center
18
25
13
63
136
53
Bell Tower 01
36
30
15
90
150
67
Bell Tower 02
13
18
9
45
103
39
Bell Tower 03
18
19
9
52
120
48
Figure 11: Application – Video Frame Interpolation: The images in
the middle are the interpolated frames at the intermediate timestamps
of the left and right images.
date. Thanks to carefully designed CUDA-related acceleration
strategies for these components described in Sec. VIII-B, the
time for finishing mapping remains within the bag duration,
showcasing the real-time capability of our method.
XI. APPLICATIONS
A. Video Frame Interpolation
The continuous-time trajectory enables pose querying at any
valid timestamp, while the Gaussian map allows rendering
from arbitrary viewpoints. Interestingly, combining these ca-
pabilities facilitates spatiotemporal interpolation, which can be
applied to video interpolation. As in Fig. 11, after obtaining
the continuous-time trajectory and Gaussian map from our
system, we successfully double the frame rate of the sequence
Visual Challenge from 15 Hz to 30 Hz by rendering interme-
diate frames with poses queried at the middle time instant.
B. Rapid 3D Mesh Extraction
Compared to previous LiDAR-Inertial-Camera 3DGS-based
SLAM methods, our approach balances visual quality, geomet-
ric accuracy, and computational efficiency. In this application,
we showcase how our method can be adapted for rapid mesh-
ing. We increase the depth regularization weight ξ (see (18))
to be 2, and run our system in real time within the duration
of data acquisition. Subsequently, based on the reconstructed
Gaussian map, we render RGB images and depth maps from
all viewpoints. These rendered depth maps are fused using
TSDF fusion [120] (with a voxel size of 0.05m) to reconstruct
a 3D-consistent map. An accurate mesh is then extracted from
the TSDF volume using the Marching Cubes algorithm [121].
Fig. 12 illustrates the resulting mesh, textured with RGB colors
or colorized by surface normal directions.
XII. CONCLUSIONS AND FUTURE WORK
In this paper, We propose a novel LiDAR-Inertial-Camera
Gaussian Splatting SLAM system that jointly considers vi-
sual quality, geometric accuracy, and real-time performance.
Our method enables accurate pose estimation and photo-
realistic map construction in real time, supporting high-
quality RGB and depth rendering. Markedly, we incorporate
a zero-shot depth completion model that fuses RGB and
sparse LiDAR data to generate dense depth maps, which
facilitate initialization of Gaussians in large-scale scenarios.
The training of the Gaussian map is efficiently supervised
by our curated sparse LiDAR depth and accelerated with
meticulously designed CUDA-related strategies. Meanwhile,
we explore tightly fusing the visual photometric constraints
derived from the Gaussian map with the LiDAR-inertial data
within the continuous-time framework, effectively overcom-
ing the LiDAR degradation. We also extend our system to
support downstream tasks such as video interpolation and
rapid mesh generation. Finally, we introduce a dedicated self-
collected LiDAR-Inertial-Camera dataset for benchmarking
photometric and geometric mapping in large-scale scenarios,
with ground-truth poses and depth maps, as well as out-of-
sequence trajectories. Extensive experiments shows that our
system outperforms existing methods in various aspects.
Our method enhances the real-time perception capabilities
of mobile robotic systems in large-scale scenarios. However,
there are still several limitations, which we aim to address in
future work. (1) Compactness: We currently do not impose
constraints on map size, which may result in large memory
usage. It is worth investigating to reduce the number of
Gaussians while maintaining the map quality. (2) Geometric
Accuracy: There remains a trade-off between visual quality
and geometric accuracy. We aim to further improve geometric
precision without sacrificing visual fidelity. (3) Applicability
in Giga Scene: In extremely large-scale environments, our
system may accumulate pose drift over time. We plan to
introduce loop closures to mitigate this issue. (4) Integration

<!-- page 18 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
18
Figure 12: Application – 3D Mesh Extraction: Textured and normal-colorized meshes generated from our reconstructed Gaussian map.
with Foundation Models: Leveraging foundation models for
feed-forward Gaussian prediction, combined with generative
models, presents a promising direction for improving novel
view synthesis performance in out-of-sequence scenarios.
REFERENCES
[1]
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng. “Nerf: Representing scenes as neural radiance fields for view
synthesis”. In: Communications of the ACM 65.1 (2021), pp. 99–106.
[2]
B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis. “3D Gaussian
Splatting for Real-Time Radiance Field Rendering”. In: ACM Transactions
on Graphics 42.4 (2023).
[3]
F. Tosi, Y. Zhang, Z. Gong, E. Sandstr¨om, S. Mattoccia, M. R. Oswald, and
M. Poggi. “How NeRFs and 3D Gaussian Splatting are Reshaping SLAM: a
Survey”. In: arXiv preprint arXiv:2402.13255 (2024).
[4]
R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao. “Gs-planner:
A gaussian-splatting-based planning framework for active high-fidelity recon-
struction”. In: 2024 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS). IEEE. 2024, pp. 11202–11209.
[5]
T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami,
P. Dames, and M. Schwager. “Splat-nav: Safe real-time robot navigation in
gaussian splatting maps”. In: IEEE Transactions on Robotics (2025).
[6]
X. Lei, M. Wang, W. Zhou, and H. Li. “Gaussnav: Gaussian splatting for
visual navigation”. In: IEEE Transactions on Pattern Analysis and Machine
Intelligence (2025).
[7]
W. Jiang, B. Lei, K. Ashton, and K. Daniilidis. “Ag-slam: Active gaussian
splatting slam”. In: (2024).
[8]
L. Chen, H. Zhan, K. Chen, X. Xu, Q. Yan, C. Cai, and Y. Xu. “ActiveG-
AMER: Active GAussian Mapping through Efficient Rendering”. In: arXiv
preprint arXiv:2501.06897 (2025).
[9]
H. Zhang, Y. Zou, Z. Yan, and H. Cheng. “Rapid-Mapping: LiDAR-Visual
Implicit Neural Representations for Real-Time Dense Mapping”. In: IEEE
Robotics and Automation Letters (2024).
[10]
E. Sucar, S. Liu, J. Ortiz, and A. J. Davison. “iMAP: Implicit mapping
and positioning in real-time”. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. 2021, pp. 6229–6238.
[11]
Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and
M. Pollefeys. “Nice-slam: Neural implicit scalable encoding for slam”. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 2022, pp. 12786–12796.
[12]
X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang. “Vox-Fusion:
Dense tracking and mapping with voxel-based neural implicit representation”.
In: 2022 IEEE International Symposium on Mixed and Augmented Reality
(ISMAR). IEEE. 2022, pp. 499–507.
[13]
H. Wang, J. Wang, and L. Agapito. “Co-SLAM: Joint Coordinate and Sparse
Parametric Encodings for Neural Real-Time SLAM”. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023,
pp. 13293–13302.
[14]
M. M. Johari, C. Carta, and F. Fleuret. “Eslam: Efficient dense slam system
based on hybrid representation of signed distance fields”. In: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
2023, pp. 17408–17419.
[15]
C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li. “Gs-
slam: Dense visual slam with 3d gaussian splatting”. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024,
pp. 19595–19604.
[16]
N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D.
Ramanan, and J. Luiten. “SplaTAM: Splat Track & Map 3D Gaussians for
Dense RGB-D SLAM”. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 2024, pp. 21357–21366.
[17]
V. Yugay, Y. Li, T. Gevers, and M. R. Oswald. “Gaussian-slam: Photo-realistic
dense slam with gaussian splatting”. In: arXiv preprint arXiv:2312.10070
(2023).

<!-- page 19 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
19
[18]
H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison. “Gaussian splatting
slam”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 2024, pp. 18039–18048.
[19]
H. Huang, L. Li, H. Cheng, and S.-K. Yeung. “Photo-SLAM: Real-time
Simultaneous Localization and Photorealistic Mapping for Monocular Stereo
and RGB-D Cameras”. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 2024, pp. 21584–21593.
[20]
L. C. Sun, N. P. Bhatt, J. C. Liu, Z. Fan, Z. Wang, T. E. Humphreys, and
U. Topcu. “Mm3dgs slam: Multi-modal 3d gaussian splatting for slam using
vision, depth, and inertial measurements”. In: 2024 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS). IEEE. 2024, pp. 10159–
10166.
[21]
X. Lang, L. Li, C. Wu, C. Zhao, L. Liu, Y. Liu, J. Lv, and X. Zuo. “Gaussian-
LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-
Inertial-Camera Fusion”. In: arXiv preprint arXiv:2404.06926 (2024).
[22]
R. Xiao, W. Liu, Y. Chen, and L. Hu. “LiV-GS: LiDAR-Vision Integration for
3D Gaussian Splatting SLAM in Outdoor Environments”. In: IEEE Robotics
and Automation Letters (2024).
[23]
H. Zhao, W. Guan, and P. Lu. “LVI-GS: Tightly-coupled LiDAR-Visual-
Inertial SLAM using 3D Gaussian Splatting”. In: IEEE Transactions on
Instrumentation and Measurement (2025).
[24]
S. Hong, C. Zheng, Y. Shen, C. Li, F. Zhang, T. Qin, and S. Shen. “GS-
LIVO: Real-Time LiDAR, Inertial, and Visual Multi-sensor Fused Odometry
with Gaussian Mapping”. In: arXiv preprint arXiv:2501.08672 (2025).
[25]
J. Zhang and S. Singh. “Laser–visual–inertial odometry and mapping with
high robustness and low drift”. In: Journal of field robotics 35.8 (2018),
pp. 1242–1264.
[26]
X. Zuo, P. Geneva, W. Lee, Y. Liu, and G. Huang. “Lic-fusion: Lidar-inertial-
camera odometry”. In: 2019 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS). IEEE. 2019, pp. 5848–5854.
[27]
X. Zuo, Y. Yang, P. Geneva, J. Lv, Y. Liu, G. Huang, and M. Pollefeys. “Lic-
fusion 2.0: Lidar-inertial-camera odometry with sliding-window plane-feature
tracking”. In: 2020 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS). IEEE. 2020, pp. 5112–5119.
[28]
T. Shan, B. Englot, C. Ratti, and D. Rus. “Lvi-sam: Tightly-coupled lidar-
visual-inertial odometry via smoothing and mapping”. In: 2021 IEEE in-
ternational conference on robotics and automation (ICRA). IEEE. 2021,
pp. 5692–5698.
[29]
J. Lin, C. Zheng, W. Xu, and F. Zhang. “R2LIVE: A Robust, Real-Time,
LiDAR-Inertial-Visual Tightly-Coupled State Estimator and Mapping”. In:
IEEE Robotics and Automation Letters 6.4 (2021), pp. 7469–7476.
[30]
J. Lin and F. Zhang. “R 3 LIVE: A Robust, Real-time, RGB-colored, LiDAR-
Inertial-Visual tightly-coupled state Estimation and mapping package”. In:
2022 International Conference on Robotics and Automation (ICRA). IEEE.
2022, pp. 10672–10678.
[31]
J. Lin and F. Zhang. “R3LIVE++: A Robust, Real-time, Radiance Reconstruc-
tion Package with a Tightly-coupled LiDAR-Inertial-Visual State Estimator”.
In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
[32]
C. Zheng, Q. Zhu, W. Xu, X. Liu, Q. Guo, and F. Zhang. “Fast-livo: Fast
and tightly-coupled sparse-direct lidar-inertial-visual odometry”. In: 2022
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
IEEE. 2022, pp. 4003–4009.
[33]
C. Zheng, W. Xu, Z. Zou, T. Hua, C. Yuan, D. He, B. Zhou, Z. Liu, J. Lin,
F. Zhu, et al. “Fast-livo2: Fast, direct lidar-inertial-visual odometry”. In: IEEE
Transactions on Robotics (2024).
[34]
J. Lv, X. Lang, J. Xu, M. Wang, Y. Liu, and X. Zuo. “Continuous-time fixed-
lag smoothing for lidar-inertial-camera slam”. In: IEEE/ASME Transactions
on Mechatronics 28.4 (2023), pp. 2259–2270.
[35]
X. Lang, C. Chen, K. Tang, Y. Ma, J. Lv, Y. Liu, and X. Zuo. “Coco-
lic: continuous-time tightly-coupled lidar-inertial-camera odometry using non-
uniform b-spline”. In: IEEE Robotics and Automation Letters 8.11 (2023),
pp. 7074–7081.
[36]
C. Wu, Y. Duan, X. Zhang, Y. Sheng, J. Ji, and Y. Zhang. “MM-Gaussian:
3D Gaussian-based multi-modal fusion for localization and reconstruction
in unbounded scenes”. In: 2024 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS). IEEE. 2024, pp. 12287–12293.
[37]
Y. Xie, Z. Huang, J. Wu, and J. Ma. “GS-LIVM: Real-Time Photo-Realistic
LiDAR-Inertial-Visual Mapping with Gaussian Splatting”. In: arXiv preprint
arXiv:2410.17084 (2024).
[38]
T. Qin, P. Li, and S. Shen. “Vins-mono: A robust and versatile monocular
visual-inertial state estimator”. In: IEEE transactions on robotics 34.4 (2018),
pp. 1004–1020.
[39]
C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. Montiel, and J. D. Tard´os.
“Orb-slam3: An accurate open-source library for visual, visual–inertial, and
multimap slam”. In: IEEE transactions on robotics 37.6 (2021), pp. 1874–
1890.
[40]
R. A. Newcombe, S. J. Lovegrove, and A. J. Davison. “DTAM: Dense tracking
and mapping in real-time”. In: 2011 international conference on computer
vision. IEEE. 2011, pp. 2320–2327.
[41]
M. Pizzoli, C. Forster, and D. Scaramuzza. “REMODE: Probabilistic, monoc-
ular dense reconstruction in real time”. In: 2014 IEEE international conference
on robotics and automation (ICRA). IEEE. 2014, pp. 2609–2616.
[42]
J. Engel, V. Koltun, and D. Cremers. “Direct sparse odometry”. In: IEEE
transactions on pattern analysis and machine intelligence 40.3 (2017),
pp. 611–625.
[43]
Z. Teed and J. Deng. “Droid-slam: Deep visual slam for monocular, stereo,
and rgb-d cameras”. In: Advances in neural information processing systems
34 (2021), pp. 16558–16569.
[44]
R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison,
P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon. “Kinectfusion: Real-
time dense surface mapping and tracking”. In: 2011 10th IEEE international
symposium on mixed and augmented reality. Ieee. 2011, pp. 127–136.
[45]
T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison.
“ElasticFusion: Dense SLAM without a pose graph.” In: Robotics: science and
systems. Vol. 11. Rome. 2015, p. 3.
[46]
T. Sch¨ops, T. Sattler, and M. Pollefeys. “Surfelmeshing: Online surfel-based
mesh reconstruction”. In: IEEE transactions on pattern analysis and machine
intelligence 42.10 (2019), pp. 2494–2507.
[47]
S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa.
“Plenoxels: Radiance fields without neural networks”. In: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 2022,
pp. 5501–5510.
[48]
T. M¨uller, A. Evans, C. Schied, and A. Keller. “Instant neural graphics
primitives with a multiresolution hash encoding”. In: ACM transactions on
graphics (TOG) 41.4 (2022), pp. 1–15.
[49]
L. Yariv, J. Gu, Y. Kasten, and Y. Lipman. “Volume rendering of neural
implicit surfaces”. In: Advances in Neural Information Processing Systems
34 (2021), pp. 4805–4815.
[50]
P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang. “Neus:
Learning neural implicit surfaces by volume rendering for multi-view recon-
struction”. In: arXiv preprint arXiv:2106.10689 (2021).
[51]
Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and L. Liu.
“Neus2: Fast learning of neural implicit surfaces for multi-view recon-
struction”. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision. 2023, pp. 3295–3306.
[52]
T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai. “Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering”. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024,
pp. 20654–20664.
[53]
A. Gu´edon and V. Lepetit. “Sugar: Surface-aligned gaussian splatting for
efficient 3d mesh reconstruction and high-quality mesh rendering”. In: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 2024, pp. 5354–5363.
[54]
Z. Yu, T. Sattler, and A. Geiger. “Gaussian opacity fields: Efficient adaptive
surface reconstruction in unbounded scenes”. In: ACM Transactions on
Graphics (TOG) 43.6 (2024), pp. 1–13.
[55]
B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao. “2d gaussian splatting for
geometrically accurate radiance fields”. In: ACM SIGGRAPH 2024 conference
papers. 2024, pp. 1–11.
[56]
P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu. “High-quality surface
reconstruction using gaussian surfels”. In: ACM SIGGRAPH 2024 Conference
Papers. 2024, pp. 1–11.
[57]
D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao,
and G. Zhang. “Pgsr: Planar-based gaussian splatting for efficient and high-
fidelity surface reconstruction”. In: IEEE Transactions on Visualization and
Computer Graphics (2024).
[58]
B. Zhang, C. Fang, R. Shrestha, Y. Liang, X. Long, and P. Tan. “Rade-gs:
Rasterizing depth in gaussian splatting”. In: arXiv preprint arXiv:2406.01467
(2024).
[59]
H. Chen, C. Li, and G. H. Lee. “Neusg: Neural implicit surface reconstruction
with 3d gaussian splatting guidance”. In: arXiv preprint arXiv:2312.00846
(2023).
[60]
M. Yu, T. Lu, L. Xu, L. Jiang, Y. Xiangli, and B. Dai. “Gsdf: 3dgs
meets sdf for improved rendering and reconstruction”. In: arXiv preprint
arXiv:2403.16964 (2024).
[61]
K. Rematas, A. Liu, P. P. Srinivasan, J. T. Barron, A. Tagliasacchi, T.
Funkhouser, and V. Ferrari. “Urban radiance fields”. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022,
pp. 12932–12942.
[62]
J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu,
S. Fidler, M. Pavone, et al. “Emernerf: Emergent spatial-temporal scene
decomposition via self-supervision”. In: arXiv preprint arXiv:2311.02077
(2023).
[63]
Y. Tao, Y. Bhalgat, L. F. T. Fu, M. Mattamala, N. Chebrolu, and M. Fallon.
“SiLVR: Scalable LiDAR-visual reconstruction with neural radiance fields for
robotic inspection”. In: 2024 IEEE International Conference on Robotics and
Automation (ICRA). IEEE. 2024, pp. 17983–17989.
[64]
J. Liu, C. Zheng, Y. Wan, B. Wang, Y. Cai, and F. Zhang. “Neural Surface
Reconstruction and Rendering for LiDAR-Visual Systems”. In: arXiv preprint
arXiv:2409.05310 (2024).
[65]
Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang. “Periodic vibration gaussian:
Dynamic urban scene reconstruction and real-time rendering”. In: arXiv
preprint arXiv:2311.18561 (2023).
[66]
X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang. “Driving-
gaussian: Composite gaussian splatting for surrounding dynamic autonomous
driving scenes”. In: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2024, pp. 21634–21643.
[67]
Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and
S. Peng. “Street gaussians: Modeling dynamic urban scenes with gaussian
splatting”. In: European Conference on Computer Vision. Springer. 2024,
pp. 156–173.

<!-- page 20 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, JULY 2025
20
[68]
C. Zhao, S. Sun, R. Wang, Y. Guo, J.-J. Wan, Z. Huang, X. Huang, Y. V. Chen,
and L. Ren. “TCLC-GS: Tightly Coupled LiDAR-Camera Gaussian Splatting
for Autonomous Driving”. In: arXiv preprint arXiv:2404.02410 (2024).
[69]
S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen. “LIV-GaussMap: LiDAR-
inertial-visual fusion for real-time 3D radiance field map rendering”. In: IEEE
Robotics and Automation Letters (2024).
[70]
J. Cui, J. Cao, F. Zhao, Z. He, Y. Chen, Y. Zhong, L. Xu, Y. Shi, Y. Zhang, and
J. Yu. “Letsgo: Large-scale garage modeling and rendering via lidar-assisted
gaussian primitives”. In: ACM Transactions on Graphics (TOG) 43.6 (2024),
pp. 1–18.
[71]
C. Jiang, R. Gao, K. Shao, Y. Wang, R. Xiong, and Y. Zhang. “Li-gs: Gaussian
splatting with lidar incorporated for accurate large-scale reconstruction”. In:
IEEE Robotics and Automation Letters (2024).
[72]
J. Liu, Y. Wan, B. Wang, C. Zheng, J. Lin, and F. Zhang. “GS-SDF: LiDAR-
Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent
Rendering and Reconstruction”. In: arXiv preprint arXiv:2503.10170 (2025).
[73]
A. Yu, V. Ye, M. Tancik, and A. Kanazawa. “pixelnerf: Neural radiance fields
from one or few images”. In: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 2021, pp. 4578–4587.
[74]
A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su. “Mvsnerf:
Fast generalizable radiance field reconstruction from multi-view stereo”. In:
Proceedings of the IEEE/CVF international conference on computer vision.
2021, pp. 14124–14133.
[75]
D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann. “pixelsplat: 3d
gaussian splats from image pairs for scalable generalizable 3d reconstruction”.
In: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 2024, pp. 19457–19467.
[76]
Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham,
and J. Cai. “Mvsplat: Efficient 3d gaussian splatting from sparse multi-
view images”. In: European Conference on Computer Vision. Springer. 2024,
pp. 370–386.
[77]
E. Sandstr¨om, Y. Li, L. Van Gool, and M. R. Oswald. “Point-slam: Dense neu-
ral point cloud-based slam”. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. 2023, pp. 18433–18444.
[78]
C. Jiang, H. Zhang, P. Liu, Z. Yu, H. Cheng, B. Zhou, and S. Shen. “H {2}-
mapping: Real-time dense mapping using hierarchical hybrid representation”.
In: IEEE Robotics and Automation Letters 8.10 (2023), pp. 6787–6794.
[79]
Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou. “Rtg-
slam: Real-time 3d reconstruction at scale using gaussian splatting”. In: ACM
SIGGRAPH 2024 Conference Papers. 2024, pp. 1–11.
[80]
J. Wei and S. Leutenegger. “Gsfusion: Online rgb-d mapping where gaussian
splatting meets tsdf fusion”. In: IEEE Robotics and Automation Letters (2024).
[81]
Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M.
Pollefeys. “Nicer-slam: Neural implicit scene encoding for rgb slam”. In:
2024 International Conference on 3D Vision (3DV). IEEE. 2024, pp. 42–52.
[82]
C.-M. Chung, Y.-C. Tseng, Y.-C. Hsu, X.-Q. Shi, Y.-H. Hua, J.-F. Yeh, W.-C.
Chen, Y.-T. Chen, and W. H. Hsu. “Orbeez-slam: A real-time monocular
visual slam with orb features and nerf-realized mapping”. In: 2023 IEEE
International Conference on Robotics and Automation (ICRA). IEEE. 2023,
pp. 9400–9406.
[83]
A. Rosinol, J. J. Leonard, and L. Carlone. “Nerf-slam: Real-time dense
monocular slam with neural radiance fields”. In: 2023 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS). IEEE. 2023, pp. 3437–
3444.
[84]
F. A. Sarikamis and A. A. Alatan. “Ig-slam: Instant gaussian slam”. In: arXiv
preprint arXiv:2408.01126 (2024).
[85]
J. Naumann, B. Xu, S. Leutenegger, and X. Zuo. “NeRF-VO: Real-time
sparse visual odometry with neural radiance fields”. In: IEEE Robotics and
Automation Letters (2024).
[86]
P. Zhu, Y. Zhuang, B. Chen, L. Li, C. Wu, and Z. Liu. “Mgs-slam: Monocular
sparse tracking and gaussian mapping with depth smooth regularization”. In:
IEEE Robotics and Automation Letters (2024).
[87]
Z. Teed, L. Lipson, and J. Deng. “Deep patch visual odometry”. In: Advances
in Neural Information Processing Systems 36 (2023), pp. 39033–39051.
[88]
X. Zhong, Y. Pan, J. Behley, and C. Stachniss. “Shine-mapping: Large-scale
3d mapping using sparse hierarchical implicit neural representations”. In: 2023
IEEE International Conference on Robotics and Automation (ICRA). IEEE.
2023, pp. 8371–8377.
[89]
X. Yu, Y. Liu, S. Mao, S. Zhou, R. Xiong, Y. Liao, and Y. Wang. “Nf-atlas:
Multi-volume neural feature fields for large scale lidar mapping”. In: IEEE
Robotics and Automation Letters 8.9 (2023), pp. 5870–5877.
[90]
S. Song, J. Zhao, K. Huang, J. Lin, C. Ye, and T. Feng. “N3-Mapping: Normal
Guided Neural Non-Projective Signed Distance Fields for Large-scale 3D
Mapping”. In: IEEE Robotics and Automation Letters (2024).
[91]
K. Wu, K. Zhang, Z. Zhang, M. Tie, S. Yuan, J. Zhao, Z. Gan, and W. Ding.
“HGS-mapping: Online dense mapping using hybrid Gaussian representation
in urban scenes”. In: IEEE Robotics and Automation Letters (2024).
[92]
J. Deng, Q. Wu, X. Chen, S. Xia, Z. Sun, G. Liu, W. Yu, and L. Pei.
“Nerf-loam: Neural implicit representation for large-scale incremental lidar
odometry and mapping”. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. 2023, pp. 8218–8227.
[93]
S. Isaacson, P.-C. Kung, M. Ramanagopal, R. Vasudevan, and K. A. Skinner.
“Loner: Lidar only neural representations for real-time slam”. In: IEEE
Robotics and Automation Letters 8.12 (2023), pp. 8042–8049.
[94]
Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stachniss.
“PIN-SLAM: LiDAR SLAM using a point-based implicit neural representa-
tion for achieving global map consistency”. In: IEEE Transactions on Robotics
(2024).
[95]
E. Giacomini, L. Di Giammarino, L. De Rebotti, G. Grisetti, and M. R.
Oswald. “Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping”.
In: arXiv preprint arXiv:2503.17491 (2025).
[96]
Y. Pan, X. Zhong, L. Jin, L. Wiesmann, M. Popovi´c, J. Behley, and C.
Stachniss. “PINGS: Gaussian Splatting Meets Distance Fields within a Point-
Based Implicit Neural Map”. In: arXiv preprint arXiv:2502.05752 (2025).
[97]
X. Lang, J. Lv, J. Huang, Y. Ma, Y. Liu, and X. Zuo. “Ctrl-VIO: Continuous-
time visual-inertial odometry for rolling shutter cameras”. In: IEEE Robotics
and Automation Letters 7.4 (2022), pp. 11537–11544.
[98]
P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, and G. Huang. “Openvins: A
research platform for visual-inertial estimation”. In: 2020 IEEE International
Conference on Robotics and Automation (ICRA). IEEE. 2020, pp. 4666–4672.
[99]
W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang. “Fast-lio2: Fast direct lidar-inertial
odometry”. In: IEEE Transactions on Robotics 38.4 (2022), pp. 2053–2073.
[100]
B. D. Lucas and T. Kanade. “An iterative image registration technique with an
application to stereo vision”. In: IJCAI’81: 7th international joint conference
on Artificial intelligence. Vol. 2. 1981, pp. 674–679.
[101]
S. Agarwal, K. Mierle, and Others. Ceres Solver. http://ceres-solver.org.
[102]
L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. “Depth
anything v2”. In: Advances in Neural Information Processing Systems 37
(2024), pp. 21875–21911.
[103]
R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang. “Moge:
Unlocking accurate monocular geometry estimation for open-domain images
with optimal training supervision”. In: arXiv preprint arXiv:2410.19115
(2024).
[104]
M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C.
Shen, and S. Shen. “Metric3d v2: A versatile monocular geometric foundation
model for zero-shot metric depth and surface normal estimation”. In: IEEE
Transactions on Pattern Analysis and Machine Intelligence (2024).
[105]
A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R. Richter,
and V. Koltun. “Depth pro: Sharp monocular metric depth in less than a
second”. In: arXiv preprint arXiv:2410.02073 (2024).
[106]
C. Cao, X. Ren, and Y. Fu. “MVSFormer: Multi-view stereo by learning
robust image features and temperature-based depth”. In: arXiv preprint
arXiv:2208.02541 (2022).
[107]
H. Xu, J. Zhang, J. Cai, H. Rezatofighi, F. Yu, D. Tao, and A. Geiger.
“Unifying flow, stereo and depth estimation”. In: IEEE Transactions on
Pattern Analysis and Machine Intelligence 45.11 (2023), pp. 13941–13958.
[108]
Z. Liu, K. L. Cheng, Q. Wang, S. Wang, H. Ouyang, B. Tan, K. Zhu, Y.
Shen, Q. Chen, and P. Luo. “DepthLab: From Partial to Complete”. In: arXiv
preprint arXiv:2412.18153 (2024).
[109]
H. Lin, S. Peng, J. Chen, S. Peng, J. Sun, M. Liu, H. Bao, J. Feng, X. Zhou,
and B. Kang. “Prompting Depth Anything for 4K Resolution Accurate Metric
Depth Estimation”. In: arXiv preprint arXiv:2412.14015 (2024).
[110]
H. Wang, M. Yang, X. Zheng, and G. Hua. “Scale Propagation Network for
Generalizable Depth Completion”. In: IEEE Transactions on Pattern Analysis
and Machine Intelligence (2024).
[111]
Z. Yan, W. F. Low, Y. Chen, and G. H. Lee. “Multi-scale 3d gaussian splatting
for anti-aliased rendering”. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 2024, pp. 20923–20931.
[112]
L. Radl, M. Steiner, M. Parger, A. Weinrauch, B. Kerbl, and M. Steinberger.
“Stopthepop: Sorted gaussian splatting for view-consistent real-time render-
ing”. In: ACM Transactions on Graphics (TOG) 43.4 (2024), pp. 1–17.
[113]
S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco, and F. De
La Torre. “Taming 3dgs: High-quality radiance fields with limited resources”.
In: SIGGRAPH Asia 2024 Conference Papers. 2024, pp. 1–11.
[114]
T.-M. Nguyen, S. Yuan, T. H. Nguyen, P. Yin, H. Cao, L. Xie, M. Wozniak,
P. Jensfelt, M. Thiel, J. Ziegenbein, et al. “Mcd: Diverse large-scale multi-
campus dataset for robot perception”. In: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition. 2024, pp. 22304–22313.
[115]
J. Yin, A. Li, T. Li, W. Yu, and D. Zou. “M2dgr: A multi-sensor and multi-
scenario slam dataset for ground robots”. In: IEEE Robotics and Automation
Letters 7.2 (2021), pp. 2266–2273.
[116]
P. Furgale, J. Rehder, and R. Siegwart. “Unified temporal and spatial calibra-
tion for multi-sensor systems”. In: 2013 IEEE/RSJ International Conference
on Intelligent Robots and Systems. IEEE. 2013, pp. 1280–1286.
[117]
J. Lv, K. Hu, J. Xu, Y. Liu, X. Ma, and X. Zuo. “CLINS: Continuous-
time trajectory estimation for LiDAR-inertial system”. In: 2021 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS). IEEE.
2021, pp. 6657–6663.
[118]
Y. Zhou, X. Li, S. Li, X. Wang, S. Feng, and Y. Tan. “DBA-fusion: tightly
integrating deep dense visual bundle adjustment with multiple sensors for
large-scale localization and mapping”. In: IEEE Robotics and Automation
Letters (2024).
[119]
A. Krizhevsky, I. Sutskever, and G. E. Hinton. “Imagenet classification with
deep convolutional neural networks”. In: Advances in neural information
processing systems 25 (2012).
[120]
B. Curless and M. Levoy. “A volumetric method for building complex models
from range images”. In: Proceedings of the 23rd annual conference on
Computer graphics and interactive techniques. 1996, pp. 303–312.
[121]
W. E. Lorensen and H. E. Cline. “Marching cubes: A high resolution 3D
surface construction algorithm”. In: Seminal graphics: pioneering efforts that
shaped the field. 1998, pp. 347–353.
