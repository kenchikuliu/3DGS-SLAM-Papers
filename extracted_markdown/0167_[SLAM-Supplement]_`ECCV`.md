<!-- page 1 -->
TCLC-GS: Tightly Coupled LiDAR-Camera
Gaussian Splatting for Autonomous Driving
Cheng Zhao1⋆, Su Sun2⋆, Ruoyu Wang1, Yuliang Guo1, Jun-Jun Wan3,
Zhou Huang3, Xinyu Huang1, Yingjie Victor Chen2, and Liu Ren1
1 Bosch Research North America & Bosch Center for Artificial Intelligence (BCAI)
{cheng.zhao, ruoyu.wang, yuliang.guo, xingyu.huang, liu.ren}@us.bosch.com
2 Purdue University
3 XC Cross Domain Computing, Bosch
{sun931, victorchen}@purdue.edu {jun-jun.wan, zhou.huang}@bosch.com
Abstract. Most 3D Gaussian Splatting (3D-GS) based methods for ur-
ban scenes initialize 3D Gaussians directly with 3D LiDAR points, which
not only underutilizes LiDAR data capabilities but also overlooks the
potential advantages of fusing LiDAR with camera data. In this pa-
per, we design a novel tightly coupled LiDAR-Camera Gaussian Splat-
ting (TCLC-GS) to fully leverage the combined strengths of both LiDAR
and camera sensors, enabling rapid, high-quality 3D reconstruction and
novel view RGB/depth synthesis. TCLC-GS designs a hybrid explicit
(colorized 3D mesh) and implicit (hierarchical octree feature) 3D repre-
sentation derived from LiDAR-camera data, to enrich the properties of
3D Gaussians for splatting. 3D Gaussian’s properties are not only initial-
ized in alignment with the 3D mesh which provides more completed 3D
shape and color information, but are also endowed with broader contex-
tual information through retrieved octree implicit features. During the
Gaussian Splatting optimization process, the 3D mesh offers dense depth
information as supervision, which enhances the training process by learn-
ing of a robust geometry. Comprehensive evaluations conducted on the
Waymo Open Dataset and nuScenes Dataset validate our method’s state-
of-the-art (SOTA) performance. Utilizing a single NVIDIA RTX 3090 Ti,
our method demonstrates fast training and achieves real-time RGB and
depth rendering at 90 FPS in resolution of 1920×1280 (Waymo), and
120 FPS in resolution of 1600×900 (nuScenes) in urban scenarios.
Keywords: LiDAR-Camera · Gaussian Splatting · Real-time Rendering
· Sensor Fusion · Autonomous Driving
1
Introduction
Urban-level reconstruction and rendering present significant challenges due to
the vast scale of the unbounded environments and the sparse nature of the
captured data. Fortunately, in autonomous vehicle settings, data from various
modalities captured by multiple sensors are typically available. However, fully
⋆Equally contributed as co-first author.
arXiv:2404.02410v2  [cs.CV]  12 Jul 2024

<!-- page 2 -->
2
C. Zhao et al.
Fig. 1: Left: Original 3D-GS [11] based methods directly initialize 3D Gaussians
by 3D LiDAR points; Right: Our TCLC-GS enriches the geometry and appearance
attributes of 3D Gaussians by explicit (colorized 3D mesh) and implicit (hierarchical
octree feature) representations.
leveraging different modality data from multi-sensors for precise modeling and
real-time rendering in urban scenes remains an open question in the field.
Neural Radiance Fields (NeRF) [17] based solutions are effective in recon-
structing urban environments when a sufficient number of images captured from
diverse viewpoints are available. NeRF-W [16] and MipNeRF-360 [2] focused on
utilizing NeRF for reconstructing street scenes within large-scale, unbounded
scenarios. A further extension, Block-NeRF [24], adapts NeRF for city-level sce-
narios, constructing block-wise radiance fields from individual NeRF models to
form a complete scene. However, the quality of modeling and rendering signif-
icantly deteriorates when relying solely on images captured from very sparse
viewpoints. In addition, a significant limitation of these methods is their de-
pendence on intensive volumetric sampling in free space, leading to excessive
consumption of computational resources in areas with no content.
The point-based urban radiance field methods [20,21] integrate LiDAR point
clouds to provide supervision, thereby facilitating geometry learning. These ap-
proaches take advantage of explicit 3D geometry derived from LiDAR data to
represent the radiance field, enhancing rendering efficiency. Mesh-based render-
ing technique [13, 15] employs learned neural descriptors on the 3D mesh to
achieve both accurate reconstruction and fast rasterization. Fusing these two
modalities offers a viable solution to address the viewpoint sparsity in large-
scale and complex scenes. However, these methods face significant limitations
due to their slow training and rendering speeds, which are further compounded
by the critical requirement in real-time applications.
In contrast to Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3D-
GS) [11], utilizes a more extensive 3D Gaussian representation for scenes, which
not only achieves faster training but also facilitates real-time rendering. The ini-

<!-- page 3 -->
TCLC-GS
3
tial 3D-GS method sought to initialize Gaussians using the points from structure-
from-motion (SfM). However, this approach faces challenges in unbounded ur-
ban scenes within autonomous driving contexts, particularly when viewpoints
are sparse. This sparsity can cause SfM techniques to fail in accurately and com-
pletely recovering scene geometries. To facilitate better 3D Gaussian initializa-
tion, pioneering research [5,28,31] has introduced LiDAR priors into the 3D-GS
process, enabling more accurate geometries and ensuring rendering consistency
of multiple surrounding views. However, directly initializing 3D Gaussians’ po-
sitions with LiDAR points does not thoroughly exploit the rich 3D geometric
information embedded within 3D points, such as depth and geometric features.
In this paper, we proposed a novel Tightly Coupled LiDAR-Camera Gaus-
sian Splatting (TCLC-GS) for accurate modeling and real-time rendering in
surrounding autonomous driving scenes. Contrasting with the intuitive method
of directly initializing 3D Gaussians using LiDAR points (Fig. 1. Left), TCLC-
GS (Fig. 1. Right), offers a more cohesive solution, effectively leveraging the
combined strengths of both LiDAR and camera sensors. The TCLC-GS’s key
idea is a hybrid 3D representation combining explicit (colorized 3D mesh) and
implicit (hierarchical octree feature) derived from LiDAR-camera data, to en-
hance both geometry and appearance properties of 3D Gaussians. To be specific,
we first learn and store implicit features in an octree-based hierarchical struc-
ture through encoding LiDAR geometries and image colors. Then we initialize
3D Gaussians in alignment with a colorized 3D mesh decoded from the implicit
feature volume. The 3D mesh enhances continuity/completeness, increases den-
sity, and adds color details compared to the original LiDAR points. Meanwhile,
we enhance the learning of appearance descriptions for each 3D Gaussian by in-
corporating implicit features retrieved from the octree. We further render dense
depths from the explicit meshes to supervise the GS optimization process, en-
hancing the training robustness compared to using sparse LiDAR depths. By
this way, LiDAR and camera data are tightly integrated within the initialization
and optimization phases of 3D Gaussians.
The novel features of TCLC-GS can be summarized as follows: 1) Hybrid 3D
representation provides a explicit (colorized 3D mesh) and implicit (hierarchical
octree feature) representation to guide the properties initialization and optimiza-
tion of 3D Gaussians; 2) The geometry attribute of 3D Gaussian is initialized to
align with the 3D mesh which offers completed 3D shape and color information,
and the appearance attribute of 3D Gaussian is enriched with retrieved octree
implicit features which provides more extensive context information; 3) Besides
RGB supervision, the dense depths rendered from the 3D mesh offer supplemen-
tary supervision in GS optimizations. Our solution improves the quality of 3D
reconstruction and rendering in urban driving scenarios, without compromising
the efficiency of 3D-GS. It enables us to fast and accurately reconstruct an urban
street scene, while also achieving real-time RGB and depth rendering capabili-
ties at around 90 FPS for a resolution of 1920×1280, and around 120 FPS for a
resolution of 1600 × 900 using a single NVIDIA GeForce RTX 3090 Ti.

<!-- page 4 -->
4
C. Zhao et al.
2
Related Work
NeRF-based techniques [2,16,24] have demonstrated significant effectiveness in
large-scale urban scenarios for autonomous driving. NeRF-W [16] integrates
frame-specific codes within its rendering pipeline, effectively managing photo-
metric variations and transient objects. Mip-NeRF 360 [2], an advancement of
Mip-NeRF [1], adapts to unbounded scenes by compressing the entire space into
a bounded area, thereby enhancing the representativeness of position encoding.
Block-NeRF [24] tackles the challenge of modeling large-scale outdoor scenes by
partitioning the target urban scene into several blocks, with each segment repre-
sented by an individual NeRF network, boosting the overall modeling capability.
Point-based rendering techniques [13, 15, 20, 21] are characterized by their
use of learned neural descriptors on point clouds, coupled with differentiable
rasterization via a neural renderer. Urban Radiance Field [21] employs LiDAR
point clouds for supervision to facilitate geometry learning. The Neural Point
Light Field (NPLF) [20] method leverages explicit 3D reconstructions derived
from LiDAR data to efficiently represent the radiance field in rendering processes.
DNMP [15] employs learned neural descriptors on point clouds and marries this
approach with differentiable rasterization, facilitated through a neural renderer.
NeuRas [13] uses a scaffold mesh as its input and optimizes a neural texture field
to achieve rapid rasterization.
3D Gaussian Splatting (3D-GS) [11] establishes a set of anisotropic Gaussians
within a 3D world and further performs adaptive density control to achieve
real-time rendering results based on point cloud input. Most recent research [5,
28, 31] has expanded upon the original 3D-GS by integrating temporal/time
cues to model the dynamic objects in the urban environments. However, these
methods only take advantage of 3D LiDAR points to initialize the position of 3D
Gaussians, which does not entirely fulfill the potential for an efficient LiDAR-
camera fusion solution.
3
Methodology
Overview: The pipeline for our TCLC-GS is detailed in Fig. 2. The TCLC-
GS framework is composed of two primary learning components: 1) the octree
implicit feature with SDF and RGB decoders (Fig. 2 upper part), and 2) 3D
Gaussians with depth and RGB splattings (Fig. 2 lower part). The LiDAR and
camera data are tightly integrated in a uniform framework.
Problem Definition: In an urban street scene, given a sequence of sur-
rounding images and LiDAR data collected from a vehicle-mounted system, our
goal is to develop a model to reconstruct the environment and generate photo-
realistic images and depths from novel viewpoints. A set of surrounding images
I are captured by multiple surrounding cameras with corresponding intrinsic
matrices I and extrinsic matrices E. A set of 3D points P captured by LiDAR
with corresponding extrinsic matrices E′. Meantime, the vehicle trajectory T
is provided or calculated by multi-sensor-based odometry. Our model R aims

<!-- page 5 -->
TCLC-GS
5
Fig. 2: The pipeline of TCLC-GS: We first merge all the LiDAR sweeps together,
and then build a hierarchical octree implicit feature grid using the sampled 3D point
within the truncation region along the LiDAR rays. These octree implicit features are
trained with SDF and RGB decoders, supervised by sparse LiDAR range measurements
and surrounding image projected RGB colors. Subsequently, we obtain the optimized
octree implicit representations and colorized 3D mesh of the global scene. The geom-
etry attributes of 3D Gaussians are initialized by the 3D mesh while the appearance
attributes of 3D Gaussians are enriched by the mesh-vertex-retrieved octree implicit
features. The 3D Gaussians are optimized through depth and RGB splatting with dense
depth and color supervision. Different from the sparse depth supervision derived from
LiDAR, our dense depth supervision is rendered from the 3D mesh utilizing the Ray
Tracing method.
to achieve precise 3D reconstruction and synthesize novel viewpoints at novel
camera pose [En, In] by rendering ˆI, ˆD = R(En, In).
Hierarchical Octree Implicit Feature: To encapsulate fine-grained ge-
ometric details and contextual structure information of the global 3D scene,
we learn and store implicit features through learnable octree-based hierarchical
feature grids F similar to NGLOD [23] from a sequence of LiDAR data. The
implicit features can be further decoded into signed distance values (SDFs) and
RGB colors through a shallow dual-branch MLP decoder.
We first merge a sequence of LiDAR sweeps using vehicle trajectory T to
obtain the global point cloud of the scene. We construct a L level octree using
the point clouds, and store latent features at eight corners per octree node in
each tree level. Multi-resolution features are stored across different octree levels,
where the octree feature at level i is denoted as Fi ∈F. The fine-grained features
at the lower levels of the octree capture detailed geometry, whereas the coarse
features at the higher levels encode broader contextual information. To optimize
the implicit features, we sample 3D query points p ∈R3 within the truncation
region of beam endpoints along the LiDAR ray. For a query point p, we compute
its feature vector Fi(p) by trilinear interpolating its corresponding features at
octree node corners. We feed both position encoding and multi-resolution octree

<!-- page 6 -->
6
C. Zhao et al.
features retrieved by a query point p into the dual branch MLP decoder, i.e.,
SDF decoder Ds to estimate the SDF ˆsp, and RGB decoder Dc to estimate the
RGB color ˆcp,
  \ h at {s}_{ p} = D_ {s} ( f(p), F_ i(p)), \ h at  { c}_{p} = D_{c}(f(p), F_i(p)), i = 1,2,...L \label {eq:1} 
(1)
where f refers to the position encoding function.
We train Ds using binary cross entropy (BCE) loss Lbce as,
  \math c al {L } _{bce}(p) =  S( s _p) \c d ot lo g (S(\hat {s}_{p})) + (1-S(s_p)) \cdot log(1-S(\hat {s}_{p})), \label {eq:2} 
(2)
where S(x) = 1/(1 + ex/σ) is sigmoid function with a flatness hyperparame-
ter σ. The ground truth sp is calculated by the signed distance between the
sampled point to the beam endpoint along the LiDAR ray. We further employ
two regularization terms together with the BCE loss: eikonal term Leik [8] and
smoothness term Lsmooth [19],
  \labe l  { e q:3} \mathc al {L}_{eik}(p) = (1 - || \nabla D_{s}(f(p), F_i(p)) ||)^2 , 
(3)
  \label { e q:4} \mathc al {L} _ {smooth } (p)  = | |  \nabla D_{s}(f(p), F_i(p)) - \nabla D_{s}(f(p+\epsilon ), F_i(p+\epsilon )) ||^2, 
(4)
where ϵ is a small perturbation.
We train Dc using L1(p) = |ˆcp −cp| loss. For each LiDAR point p, we obtain
its RGB ground truth cp by projecting the point to the camera image plane using
calibration matrices E′, E, I. A LiDAR point might be projected onto multiple
pixels across multiple surrounding images. We select the RGB of pixel from the
image plane with the smallest Euclidean distance to the point as ground truth.
The global objective function L(p) of octree implicit feature learning is de-
fined as,
  \l a bel {eq : 6} \mathcal  {L}_(p) = \mathca l  {L}_{bce}(p) + \lambda _{eik} \mathcal {L}_{eik}(p) + \lambda _{smooth} \mathcal {L}_{smooth}(p) + \lambda _{RGB} \mathcal {L}_{1}(p), 
(5)
where λeik, λsmooth and λRGB are scale factors. We randomly initialize the
corner features when creating the octree and optimizing them during training.
Through encoding both LiDAR geometries and image color into an octree rep-
resentation of the entire scene, we enrich the of 3D Gaussian’s properties with
high-dimensional implicit features, as detailed in the subsequent steps.
3D Mesh and Dense Depth Generation: After learning and storing the
octree implicit features, which includes both geometry and RGB information,
we can generate a colorized 3D mesh. Specifically, we first uniformly sample 3D
points within the scene’s 3D spatial space. Each of these points is then used to
retrieve octree implicit features, which are then fed into a dual-branch decoder.
Subsequently, we generate a colorized 3D mesh M in the form of a triangle
mesh by marching cubes [14] based on the decoder predicted SDFs and colors.
Furthermore, from this 3D mesh M, we generate a set of dense depth images D
along the vehicle trajectory by Ray Tracing [6] as,
  \label {eq:7} \ ma th cal {D} = RayTracing(\mathcal {M} | \mathcal {T}, E, I, E'). 
(6)

<!-- page 7 -->
TCLC-GS
7
Fig. 3: Visualization of our colorized 3D mesh and dense depths. Row 1:
rendered dense surrounding depth images given the camera pose within the 3D mesh;
Row 2: generated colorized 3D mesh based on the octree implicit representation.
We present visualization examples of the colorized 3D mesh and dense depths
generated by our method in Fig. 3.
Through representing the scene as a mesh, the learning of 3D Gaussians is
enhanced with a more robust geometry and appearance prior. In detail, the
mesh fully exploits the geometries from LiDAR points in terms of continu-
ity/completion and density. Compared with sparse raw LiDAR points, the mesh
representations are able to reconstruct missed geometries due to sparse scans and
viewpoints, and provides scalable and adaptive geometry priors. Meanwhile, the
colorized 3D mesh provides RGB information for any 3D point positions sam-
pled on its surface. This is a significant improvement over the intuitive method
of projecting LiDAR points onto surrounding images, which typically assigns
colors to only about one-third of the LiDAR points due to limited sensor overlap
between camera and LiDAR. In contrast to the sparse depth obtained from Li-
DAR points, the dense depth rendered from the 3D mesh provides more robust
supervision and alleviate the over-fitting during training.
LiDAR-Camera Gaussian Splatting: 3D-GS [11] employs a collection
of 3D Gaussians to represent a scene. It uses a tile-based rasterization process,
enabling real-time alpha blending of multiple Gaussians. Different from directly
initializing Gaussian by LiDAR points, our method initials 3D Gaussians GS on
the 3D mesh M. We follow the rule of Gaussians bound to the triangles [9] to
keep the 3D Gaussians flat and aligned with the mesh triangles. Each vertex v
in the mesh provides position µ, color c, and the vertex-retrieved octree implicit
feature F(v) to initialize the 3D Gaussians’ attributes as,
  \ mathca l {GS } = \ { \mu (v), s(v), q(v), c(v) | \mathcal {M}; o(F_i(v)), sh(c(v), F_i(v)), F_i(v) | \mathcal {M}, \mathcal {F} \}, i=1,..L. \label {eq:8} o
(7)
The 3D Gaussians’ attributes include mean position µ ∈R3, anisotropic covari-
ance Σ ∈R3×3 represented by a scale s ∈R3 (diagonal matrix) and rotation q
(unit quaternion), opacity o and spherical harmonics coefficients sh. The vertices
and colors of 3D mesh M serve as geometry and appearance priors for µ, s, q, c
and o, sh properties of 3D Gaussians. Additionally, the octree implicit features
are also incorporated to improve appearance descriptions o, sh of 3D Gaussians.

<!-- page 8 -->
8
C. Zhao et al.
Please note that the newly added properties Fi(v) are dynamically controlled
through cloning, and pruning alongside other attributes by adaptive density con-
trol during optimization. Additionally, to enhance training robustness, we use
very shallow MLPs to encode the c and Fi(v) before their integration into the
3D Gaussians.
We utilize a differentiable 3D Gaussian splatting renderer to project each 3D
Gaussian onto the 2D image plane, resulting in a collection of 2D Gaussians.
The covariance matrix Σ′ in camera coordinates is computed by,
  \ Sigma  '  = JE \Sigma E^TJ^T, \label {eq:9} 
(8)
where E refers to the world-to-camera matrix and J refers to Jacobian of the
perspective projection I.
By sorting the Gaussians according to their depth within the camera space,
we can effectively query the attributes of each 2D Gaussian. This step facilitates
the following volume rendering process to estimate the color C, depth D and
accumulated opacity O of each pixel as,
  
C
 
= \
sum _{i = 1
}
^
{N}
 T_i \a l p
h
a
 _i
 c_i,  D  
= \
s
um 
_{ i =1}^{N} T_i \alpha _i z_i, O = \sum _{i=1}^{N} T_i \alpha _i, T_i = \prod _{j=1}^{i-1}(1 - \alpha _j), \label {eq:10} 
(9)
where z denotes the distance from the image plane to the Gaussian point center.
α is calculated by αi = oi · exp(−1
2(x −µi)T Σ
′−1(x −µi))
Given a specific view (E, I) in a scene containing N 3D Gaussians, both the
image ˆI and depth ˆD are rendered by the differentiable rendering function R
as,
  \ ha t  {\math c a l {I }}, \ha t {\mathcal {D}} = \mathcal {R}( \{\mathcal {GS}_i , i=1,2,..N\} | E, I ). \label {eq:11} 
(10)
Note all depths used in this paper are inverse depths. The parameters of GS
will be optimized during training by the objective function L as following,
  \b e gin {split}  \m a thcal {L} = ( 1-\ l ambda _{ssi m}) 
\mathcal {L}_{1}( \ma t hcal {I}, \hat {\mathcal {I}}) + \lambda _{ssim} \mathcal {L}_{ssim}(\mathcal {I}, \hat {\mathcal {I}}) + \lambda _{depth} \mathcal {L}_{1}(\mathcal {D}, \hat {\mathcal {D}}) + \\ \lambda _{smooth} \mathcal {L}_{smooth}(\mathcal {I}, \hat {\mathcal {D}}) + \lambda _{sky} \mathcal {L}_{sky}, \end {split} \label {eq:12} 
(11)
where λssim, λdepth, λsmooth and λsky are scale factors. I refers to the origi-
nal RGB image, and D refers to the dense depth rendered from 3D mesh by
Ray Tracing. Lssim refers to Structural Similarity Index Measure (SSIM) term
between the original RGB and rendered RGB images, and Lsmooth refers to
inverse depth smooth [7] term between the original RGB image and rendered
depth. Lsky refers to the sky opacity loss [29] defined as BCE loss,
  \m a thcal { L } _{sky} = \mathcal {L}_{bce}(O, 1-M_{sky}) \label {eq:13} 
(12)
where Msky is the sky mask from a pretrained segmentation model SegFormer [26].
The Lsky drives the opacity of sky pixels toward zero to be transparent, which
pushes the 3D Gaussian positions of the sky far away. The Lsky significantly re-
duces the sky artifacts, enabling the depth synthesis with clear sky areas. These
sky masks are only necessary during training and not during inference.

<!-- page 9 -->
TCLC-GS
9
Fig. 4: Visual comparison of image and depth synthesis from novel front-left, front,
and front-right surrounding views on the Waymo dataset. Row 1: 3D-GS images; Row
2: TCLC-GS images; Row 3: GT images; Row 4: 3D-GS depths; Row 5: TCLC-GS
depths; Row 6: GT depth of LiDAR points projected on images.
The enhancement of Gaussian optimization is achieved by improving the ini-
tialization of 3D Gaussians’ geometry and appearance, enriching the appearance
descriptions of 3D Gaussians, and integrating dense depth supervision. In order
to mitigate the scale ambiguity due to the long duration of the driving scenario,
we employ the incremental optimization strategy [31], which involves incremen-
tally adding 3D Gaussians by uniformly segmenting the scenario into a series of
bins based on the LiDAR depth range. For each bin, we further employ position-
aware point adaptive control [5] during Gaussian optimization, which utilizes
smaller points for nearby positions and larger points for distant locations in the
unbounded scene.
4
Experiments
Datasets: Our experimental evaluations are conducted on two of the most
widely-used datasets in autonomous driving research: the Waymo Open Dataset
[22] and the nuScenes Dataset [4]. The Waymo Open Dataset offers urban driv-
ing scenarios, with each scenario lasting 20 seconds and recorded using five high-
resolution LiDARs and five cameras oriented towards the front and sides. For our
experiments, we selected six challenging recording sequences from this dataset,
utilizing surrounding views captured by three cameras and corresponding data

<!-- page 10 -->
10
C. Zhao et al.
Sequence
3D-GS [11]
PSNR↑/SSIM↑/LPIPS↓
TCLC-GS
PSNR↑/SSIM↑/LPIPS↓
Segment-1024795...
26.19/0.84/0.25
28.28/0.89/0.16
Segment-1071392...
27.35/0.84/0.25
28.68/0.88/0.17
Segment-1103765...
26.48/0.67/0.44
28.54/0.71/0.42
Segment-1346990...
25.08/0.84/0.26
27.72/0.88/0.16
Segment-1433374...
26.20/0.85/0.25
27.30/0.87/0.21
Segment-1466335...
26.87/0.86/0.25
28.13/0.90/0.18
Average
26.36/0.82/0.28
28.11/0.86/0.22
Table 1: Performance comparison of image synthesis from novel views between the
proposed method and baseline on the Waymo dataset.
Sequence
3D-GS [11]
AbsRel↓/RMSE↓/RMSElog↓
TCLC-GS
AbsRel↓/RMSE↓/RMSElog↓
Segment-1024795...
0.37/6.66/1.10
0.03/0.77/0.05
Segment-1071392...
0.22/7.80/0.34
0.03/3.10/0.08
Segment-1103765...
0.55/16.37/1.36
0.02/1.18/0.04
Segment-1346990...
0.34/7.31/0.77
0.02/1.01/0.04
Segment-1433374...
0.48/12.75/1.37
0.03/1.56/0.05
Segment-1466335...
0.56/14.62/1.43
0.03/1.60/0.06
Average
0.42/10.92/1.06
0.03/1.54/0.05
Table 2: Performance comparison of depth synthesis from novel views between the
proposed method and baseline on the Waymo dataset.
from five LiDAR sweeps. Each sequence consists of approximately 100 frames,
and we use a random one of every tenth frame as a test frame and the remaining
for training. The nuScenes dataset, a large-scale public resource for autonomous
driving research, includes data from an array of sensors: six cameras, one Li-
DAR, five RADARs, GPS, and IMU. In our experiments, we use the keyframes
from six challenging scenes, which include surrounding views from six cameras
and corresponding LiDAR sweeps. In each sequence, which comprises roughly 40
frames, we randomly select one of every fifth frame as a test frame and utilize the
rest for training purposes. To comprehensively evaluate and compare the detail
synthesis capabilities, we train and assess our methods and all baselines using
full-resolution images, i.e., 1920×1280 for the Waymo dataset and 1600×900 for
the nuScenes dataset.
Metrics: Following the previous research [5,15,20,28,31], our image synthe-
sis evaluation employs three widely-used benchmark metrics, i.e., peak signal-to-
noise ratio (PSNR), structural similarity index measure (SSIM), and the learned
perceptual image patch similarity (LPIPS) in novel views. Similarly as the previ-
ous research [3,30], we choose three widely-used benchmark metrics, i.e., Abso-
lute Relative Difference (AbsRel), Root Mean Squared Error (RMSE) and Root
Mean Squared Error in the Logarithmic Scale (RMSElog) for depth synthesis
evaluation from novel views. The ground truth RGB images are provided directly
from the datasets, while sparse depth images, used as depth ground truth for

<!-- page 11 -->
TCLC-GS
11
Fig. 5: Visual comparison of image synthesis from novel views on nuScenes dataset.
Row 1: 3D-GS; Row 2: TCLC-GS; Row 3: GT.
Sequence
3D-GS [11]
PSNR↑/SSIM↑/LPIPS↓
TCLC-GS
PSNR↑/SSIM↑/LPIPS↓
Scene-0008
25.32/0.82/0.26
26.41/0.85/0.22
Scene-0051
25.84/0.85/0.25
26.78/0.86/0.23
Scene-0058
24.71/0.81/0.27
26.45/0.88/0.20
Scene-0062
25.91/0.87/0.22
27.47/0.89/0.17
Scene-0129
26.87/0.88/0.21
28.65/0.90/0.16
Scene-0382
23.96/0.82/0.32
25.76/0.84/0.26
Average
25.44/0.84/0.26
26.92/0.87/0.21
Table 3: Performance comparison of image synthesis from novel views between the
proposed method and baseline on the nuScenes dataset.
depth evaluation, are obtained by projecting LiDAR points onto the surrounding
images according to the calibration information.
Baselines: We selected 3D-GS [11], which utilizes LiDAR points to directly
initialize 3D Gaussians, as our primary baseline for comparison. We also selected
NeRF [17], NeRF-W [16], Instant-NGP [18], Point-NeRF [27], NPLF [20], Mip-
NeRF [1], Mip-NeRF 360 [2], DNMP [15], as the supplementary baselines for
further comparison.
Evaluation on Waymo Open Dataset: We evaluated the proposed method
by comparing it with the baseline on the Waymo Open dataset. The performance
comparison of image and depth synthesis from novel views, relative to the main
baseline, is detailed in Table 1 and Table 2 separately. Additional comparisons
of image synthesis from novel views against a broader range of baselines are
depicted in Table 5. As indicated in Table 1, our method outperforms 3D-GS
across individual scenes in terms of PSNR, SSIM, and LPIPS metrics for novel
image synthesis. Furthermore, Table 5 shows that our method exceeds a broader
array of baselines in average performance across these same metrics (PSNR,
SSIM, and LPIPS) for novel image synthesis. Similarly, Table 2 demonstrates

<!-- page 12 -->
12
C. Zhao et al.
Fig. 6: Visual comparison of depth synthesis from novel views on nuScenes dataset.
Row 1: 3D-GS; Row 2: TCLC-GS; Row 3: GT of LiDAR points projected on images.
Sequence
3D-GS [11]
AbsRel↓/RMSE↓/RMSElog↓
TCLC-GS
AbsRel↓/RMSE↓/RMSElog↓
Scene-0008
0.31/9.97/0.47
0.07/4.27/0.12
Scene-0051
0.47/8.68/0.60
0.05/2.21/0.07
Scene-0058
0.35/10.48/0.55
0.06/4.23/0.11
Scene-0062
0.32/8.22/0.39
0.04/4.03/0.08
Scene-0129
0.29/8.01/0.44
0.05/2.83/0.09
Scene-0382
0.32/10.91/0.51
0.05/4.58/0.10
Average
0.34/9.38/0.49
0.05/3.69/0.10
Table 4: Performance comparison of depth synthesis from novel views between the
proposed method and baseline on the nuScenes dataset.
that our approach significantly surpasses 3D-GS in the depth synthesis metrics of
AbsRel, RSME, and RSMElog. The significant improvement in depth synthesis
performance can be attributed to the robust supervision provided by the ren-
dered dense depths derived from the generated accurate 3D mesh. For a visual
perspective, the comparison results of image synthesis from novel surrounding
views are illustrated in Fig. 4. We can see that TCLC-GS renders more clear
and accurate RGB images than the 3D-GS, especially on roadside objects and
in distant areas viewed from the front, front-left, and front-right perspectives.
Similarly, the visual comparison results for depth synthesis from novel surround-
ing views are showcased in Fig. 4. Here, TCLC-GS is observed to render denser
and sharper depths compared to 3D-GS, especially in areas further away in the
front, front-left, and front-right views.
Evaluation on nuScenes Dataset: We conducted further comprehensive
evaluations comparing our method with the baseline on the nuScenes dataset.
The results of image and depth synthesis from novel viewpoints, compared with
the primary baseline 3D-GS, are detailed in Table 3 and Table 4 separately.
According to Table 3, our method outperforms 3D-GS in the metrics of PSNR,

<!-- page 13 -->
TCLC-GS
13
Method
PSNR↑SSIM↑LPIPS↓
NeRF [17]
26.24
0.87
0.47
NeRF-W [16]
26.92
0.89
0.42
Instant-NGP [18]
26.77
0.88
0.40
Point-NeRF [27]
26.26
0.87
0.45
NPLF [20]
25.62
0.88
0.45
Mip-NeRF [1]
26.96
0.88
0.45
Mip-NeRF 360 [2]
27.43
0.89
0.39
DNMP [15]
27.62
0.89
0.38
3D-GS [11]
26.36
0.82
0.28
TCLC-GS
28.11
0.86
0.22
Table 5: Performance comparison of im-
age synthesis from novel view between the
proposed method and additional compre-
hensive baselines on the Waymo dataset.
Method
PSNR↑SSIM↑LPIPS↓
Mip-NeRF [1]
18.10
0.63
0.46
Mip-NeRF 360 [2]
23.45
0.74
0.35
S-NeRF [12]
25.62
0.77
0.27
3D-GS [11]
25.44
0.84
0.26
TCLC-GS
26.92
0.87
0.21
Table 6: Performance comparison of im-
age synthesis from novel views between
the proposed method and additional base-
lines on the nuScenes dataset.
Method
PSNR↑
SSIM↑
LPIPS↓
AbsRel↓
RMSE↓
RMSElog↓
TCLC-GS w/o 3D mesh
26.36
0.82
0.28
0.42
10.92
1.06
TCLC-GS w/o colorized 3D mesh
27.61
0.85
0.22
0.03
1.55
0.05
TCLC-GS w/o octree implicit feature
27.81
0.85
0.23
0.04
1.63
0.05
TCLC-GS w/o dense depth supervision
27.96
0.86
0.22
0.37
9.80
0.35
TCLC-GS full
28.11
0.86
0.22
0.03
1.54
0.05
Table 7: Ablation study of the proposed method on the Waymo dataset.
SSIM, and LPIPS for novel image synthesis. Likewise, as indicated in Table 4,
our approach significantly excels beyond 3D-GS in the depth synthesis metrics
of AbsRel, RSME, and RSMElog. The lower depth performance of our method
observed on the nuScenes dataset, compared to the Waymo dataset, is due to
the nuScenes dataset’s reliance on relatively sparse 32-line LiDAR data. In con-
trast, the Waymo dataset includes much denser LiDAR data, which contributes
to a more accurate generation of 3D mesh and, subsequently, better rendered
dense depth. In Table 6, we present a performance comparison between TCLC-
GS and additional baselines on the nuScenes dataset, showcasing the average
performance on PSNR, SSIM, and LPIPS metrics. Our method surpasses these
baselines in all mentioned metrics, demonstrating its superior performance in
image synthesis. Visual comparison results depicted in Figure 5 and 6 illustrate
that our method not only produces clearer synthesized RGB images with fewer
artifacts and less blurring but also generates sharper and denser synthesized
depth images compared to the 3D-GS baseline.
Ablation Study: To demonstrate the individual effectiveness of each com-
ponent in our method, we conducted ablation studies using the Waymo dataset.
Since our contributions mainly include colorized 3D mesh, octree implicit repre-
sentation, and dense depth supervision, our ablation study analyses the impact
of our designs from these aspects. We train five different variations: 1) TCLC-GS
without 3D mesh, initializing 3D Gaussian using original 3D LiDAR points; 2)
TCLC-GS without colorized 3D mesh, initializing 3D Gaussian using 3D mesh
without color information; 3) TCLC-GS without octree implicit representation,

<!-- page 14 -->
14
C. Zhao et al.
Method
FPS
SUDS [25]
∼0.01
StreetSurf [10]
∼0.10
S-NeRF [12]
∼0.02
3D-GS [11]
∼200
TCLC-GS
∼90
Table 8: The comparison running-time
analysis on the Waymo dataset.
Method
FPS
Instant-NGP [18]
∼0.23
Mip-NeRF360 [2]
∼0.08
Urban-NeRF [21]
∼0.03
SUDS [25]
∼0.02
S-NeRF [12]
∼0.04
3D-GS [11]
∼350
TCLC-GS
∼120
Table 9: The comparison running-time
analysis on the nuScenes dataset.
initializing 3D Gaussian using colorized 3D mesh without octree representation;
4) TCLC-GS without dense depth supervision, training TCLC-GS only using
RGB supervision; 5) TCLC-GS full method. 6) TCLC-GS + sky refine The av-
erage values of the evaluation metrics across six testing scenes from the Waymo
dataset are given in Table 8. These ablation results indirectly validate the effec-
tiveness of colorized 3D mesh, octree implicit representation, and dense depth
supervision, underscoring their contributions to the overall performance.
Running-time Analysis: We present a running time performance compar-
ison between our TCLC-GS and various baselines on the Waymo and nuScenes
datasets, detailed in Table 8 and Table 9, respectively. Utilizing a single NVIDIA
GeForce RTX 3090 Ti, TCLC-GS attains real-time rendering speeds with high-
resolution images, achieving around 90 FPS with an image resolution of 1920×1280
on the Waymo dataset, and around 120 FPS with an image resolution of 1600×900
on the nuScenes dataset. Compared to NeRF-based methods such as SUDS [25]
and S-NeRF [12], TCLC-GS significantly outperforms the speed of these meth-
ods, achieving real-time running performance. When compared to 3D-GS [11],
TCLC-GS offers higher accuracy in both RGB and depth rendering, and still
maintains the performance efficiency.
5
Conclusion
In this paper, we proposed a novel Tightly Coupled LiDAR-Camera Gaussian
Splatting (TCLC-GS) that synergizes the strengths of LiDAR and surrounding
cameras for fast modeling and real-time rendering in urban driving scenarios. The
key idea of TCLC-GS is a hybrid 3D representation combining explicit (colorized
3D mesh) and implicit (hierarchical octree feature) information derived from the
LiDAR-camera data, which enriches both geometry and appearance properties
of 3D Gaussians. The optimization of Gaussian Splatting is further enhanced by
incorporating rendered dense depth data within the 3D mesh. The experimen-
tal evaluations demonstrate that our model surpasses SOTA performance while
maintaining the real-time efficiency of Gaussian Splatting on Waymo Open and
nuScenes datasets.

<!-- page 15 -->
TCLC-GS
15
In this Appendix, we firstly provide more details of our TCLC-GS implemen-
tation, including network architecture, and hyper-parameters of octree feature
grid and 3D Gaussians in Appendix 6. Secondly, more comparison visualizations
with the baseline are presented in Appendix 7. Lastly, three video demos are
attached in Appendix 8.
6
Implementation Details
Network Architecture: Figure 7 illustrates the network architecture of the
SDF and RGB decoders in octree training, and feature encoder for Gaussian
splatting. For building the hierarchical octree feature and colorized 3D mesh,
the SDF and RGB decoders employ a shallow multilayer perceptron (MLP)
consisting of three fully connected layers. Each of these layers is followed by a
ReLU activation, except for the final prediction layer. A Sigmoid layer is inserted
after the final layer in the RGB decoder. In terms of the optimization of 3D
Gaussians, the feature encoder adopts a shared two layers MLP following a dual-
brunch fully connected layer. Each of MLP is followed by a ReLU activation,
except for the final prediction layer. A Sigmoid layer is inserted after the final
layer in the opacity prediction.
32
1
Latent Codes: 
3X8=24
Postion
Encodings: 51
SDF
32
+
128
Latent Codes: 
3X8=24
Postion
Encodings: 51
128
+
FC & ReLU
FC
SDF Decoder
RGB Decoder
3
Sigmoid
RGB
128
1
Sigmoid
128
Feature Encoder
3X(3+1)**2
SH
Octree Feature: 
3X8=24
Color Feature: 
3X1=3
+
Opacity
Fig. 7: Left: network architecture of SDF and RGB decoders in octree training. Right:
network architecture of feature encoder for Gaussian splatting.
Octree Feature Grid Hyper-parameters: We construct the octree with
a height of 12, storing the latent features in the last three layers. The leaf voxel
resolution in our configuration is set to 0.2m, and the latent code assigned to
each corner of the octree node is set to 8 dimension.
3D Gaussian Hyper-parameters: We set the position learning rate to
vary from 1.6e-5 to 1.6e-6, set opacity learning rate as 0.05, scale learning rate
as 0.01, feature learning rate a 2.5e-3, and rotation learning rate as 0.001. We

<!-- page 16 -->
16
C. Zhao et al.
configure the interval of densification and opacity reset as 500 and 3000 respec-
tively. We set the densify grad threshold as 2e-4. Additionally, the Spherical
Harmonics (SH) degree for each 3D Gaussian is established at 3.
7
More Visualization Results
Visualization Results on the Waymo Open dataset: We present visual
comparison results of image and depth synthesis on the Waymo dataset in Fig. 8
and Fig. 10. It can be seen that TCLC-GS produces clearer images compared to
3D-GS [11]. Additionally, the depth images generated by TCLC-GS are denser
and sharper than those produced by 3D-GS.
Visualization Results on the nuScenes dataset: We present visual com-
parison results of image and depth synthesis on the nuScenes dataset in Fig. 9
and Fig. 11 separately. It’s observable that the images generated by TCLC-GS
exhibit fewer blurring areas compared to those from 3D-GS [11]. The depth im-
ages rendered by TCLC-GS are denser compared to those produced by 3D-GS.
8
Video Demo
We provide Video Demo 1, Demo 2 and Demo 3 in the website 3. Demos 1 and
2 illustrate the rendered images and depths by our TCLC-GS, alongside the
ground truth (GT) data, in two different urban scenes. In these videos, the first
row displays the images rendered by TCLC-GS, the second row shows the GT
images, the third row presents the dense depth rendered by TCLC-GS, and the
fourth row illustrates the GT sparse depth obtained by projecting LiDAR data
onto the surrounding images. Demo 3 showcases a more challenging rendering
scenario. This video presents image and depth rendering from a new generated
ego-car trajectory which is 0.5 meter higher or 0.5 meter lower than the original
ego-car trajectory. The first and third rows display the rendered RGB and depth
images from a trajectory 0.5m lower than the original, while the second and
fourth rows show the rendered RGB and depth images from a trajectory 0.5m
higher than the original. This demonstration highlights our method’s potential to
address the domain gap in data reuse caused by different hardware car settings,
such as collecting data from a sedan but deploying it on an SUV.
3 https://github.com/BoschRHI3NA/Video-Demo-ECCV24/tree/main

<!-- page 17 -->
TCLC-GS
17
Fig. 8: Visual comparison of image and depth synthesis from novel front-left, front,
and front-right surrounding views on the Waymo dataset. Row 1: 3D-GS images; Row
2: TCLC-GS images; Row 3: GT images; Row 4: 3D-GS depths; Row 5: TCLC-GS
depths; Row 6: GT depth of LiDAR points projected on images.
Fig. 9: Visual comparison of image synthesis from novel views on the nuScenes dataset.
Row 1: 3D-GS; Row 2: TCLC-GS; Row 3: GT.
References
1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance

<!-- page 18 -->
18
C. Zhao et al.
Fig. 10: Visual comparison of image and depth synthesis from novel front-left, front,
and front-right surrounding views on the Waymo dataset. Row 1: 3D-GS images; Row
2: TCLC-GS images; Row 3: GT images; Row 4: 3D-GS depths; Row 5: TCLC-GS
depths; Row 6: GT depth of LiDAR points projected on images.
Fig. 11: Visual comparison of depth synthesis from novel views on the nuScenes
dataset. Row 1: 3D-GS; Row 2: TCLC-GS; Row 3: GT of LiDAR points projected
on images.
fields. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision. pp. 5855–5864 (2021)

<!-- page 19 -->
TCLC-GS
19
2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-
nerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470–
5479 (2022)
3. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: Zoedepth: Zero-shot transfer
by combining relative and metric depth. arXiv preprint arXiv:2302.12288 (2023)
4. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A.,
Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous
driving. In: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. pp. 11621–11631 (2020)
5. Chen, Y., Gu, C., Jiang, J., Zhu, X., Zhang, L.: Periodic vibration gaussian:
Dynamic urban scene reconstruction and real-time rendering. arXiv preprint
arXiv:2311.18561 (2023)
6. Glassner, A.S.: An introduction to ray tracing. Morgan Kaufmann (1989)
7. Godard, C., Mac Aodha, O., Brostow, G.J.: Unsupervised monocular depth esti-
mation with left-right consistency. In: CVPR (2017)
8. Gropp, A., Yariv, L., Haim, N., Atzmon, M., Lipman, Y.: Implicit geometric reg-
ularization for learning shapes. In: Proceedings of Machine Learning and Systems
2020, pp. 3569–3579 (2020)
9. Guédon, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient
3d mesh reconstruction and high-quality mesh rendering. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)
10. Guo, J., Deng, N., Li, X., Bai, Y., Shi, B., Wang, C., Ding, C., Wang, D., Li, Y.:
Streetsurf: Extending multi-view implicit surface reconstruction to street views.
arXiv preprint arXiv:2306.04988 (2023)
11. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions on Graphics 42(4) (2023)
12. Li, Q., Peng, Z.M., Feng, L., Liu, Z., Duan, C., Mo, W., Zhou, B.: Scenarionet:
Open-source platform for large-scale traffic scenario simulation and modeling. Ad-
vances in neural information processing systems 36 (2024)
13. Liu, J.Y., Chen, Y., Yang, Z., Wang, J., Manivasagam, S., Urtasun, R.: Real-time
neural rasterization for large scenes. In: Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision. pp. 8416–8427 (2023)
14. Lorensen, W.E., Cline, H.E.: Marching cubes: A high resolution 3d surface con-
struction algorithm. ACM siggraph computer graphics 21(4), 163–169 (1987)
15. Lu, F., Xu, Y., Chen, G., Li, H., Lin, K.Y., Jiang, C.: Urban radiance field represen-
tation with deformable neural mesh primitives. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision. pp. 465–476 (2023)
16. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duck-
worth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo col-
lections. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 7210–7219 (2021)
17. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commu-
nications of the ACM 65(1), 99–106 (2021)
18. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with
a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1–
15 (2022)
19. Ortiz, J., Clegg, A., Dong, J., Sucar, E., Novotny, D., Zollhoefer, M., Mukadam,
M.: isdf: Real-time neural signed distance fields for robot perception. In: Robotics:
Science and Systems (2022)

<!-- page 20 -->
20
C. Zhao et al.
20. Ost, J., Laradji, I., Newell, A., Bahat, Y., Heide, F.: Neural point light fields.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 18419–18429 (2022)
21. Rematas, K., Liu, A., Srinivasan, P.P., Barron, J.T., Tagliasacchi, A., Funkhouser,
T., Ferrari, V.: Urban radiance fields. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 12932–12942 (2022)
22. Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo,
J., Zhou, Y., Chai, Y., Caine, B., et al.: Scalability in perception for autonomous
driving: Waymo open dataset. In: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. pp. 2446–2454 (2020)
23. Takikawa, T., Litalien, J., Yin, K., Kreis, K., Loop, C., Nowrouzezahrai, D., Ja-
cobson, A., McGuire, M., Fidler, S.: Neural geometric level of detail: Real-time
rendering with implicit 3D shapes (2021)
24. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Bar-
ron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 8248–8258 (2022)
25. Turki, H., Zhang, J.Y., Ferroni, F., Ramanan, D.: Suds: Scalable urban dynamic
scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 12375–12385 (2023)
26. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer:
Simple and efficient design for semantic segmentation with transformers. In: Neural
Information Processing Systems (NeurIPS) (2021)
27. Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.: Point-nerf:
Point-based neural radiance fields. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 5438–5448 (2022)
28. Yan, Y., Lin, H., Zhou, C., Wang, W., Sun, H., Zhan, K., Lang, X., Zhou, X.,
Peng, S.: Street gaussians for modeling dynamic urban scenes. arXiv preprint
arXiv:2401.01339 (2024)
29. Yang, J., Ivanovic, B., Litany, O., Weng, X., Kim, S.W., Li, B., Che, T., Xu, D.,
Fidler, S., Pavone, M., et al.: Emernerf: Emergent spatial-temporal scene decom-
position via self-supervision. arXiv preprint arXiv:2311.02077 (2023)
30. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth anything: Un-
leashing the power of large-scale unlabeled data. arXiv preprint arXiv:2401.10891
(2024)
31. Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., Yang, M.H.: Drivinggaussian:
Composite gaussian splatting for surrounding dynamic autonomous driving scenes.
arXiv preprint arXiv:2312.07920 (2023)
