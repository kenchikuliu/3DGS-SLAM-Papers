<!-- page 1 -->
GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF
for Geometrically Consistent Rendering and Reconstruction
Jianheng Liu, Yunfei Wan, Bowen Wang, Chunran Zheng, Jiarong Lin, and Fu Zhang
Abstract— Digital twins are fundamental to the development
of autonomous driving and embodied artificial intelligence.
However, achieving high-granularity surface reconstruction and
high-fidelity rendering remains a challenge. Gaussian splatting
offers efficient photorealistic rendering but struggles with geo-
metric inconsistencies due to fragmented primitives and sparse
observational data in robotics applications. Existing regulariza-
tion methods, which rely on render-derived constraints, often
fail in complex environments. Moreover, effectively integrating
sparse LiDAR data with Gaussian splatting remains challeng-
ing. We propose a unified LiDAR-visual system that synergizes
Gaussian splatting with a neural signed distance field. The
accurate LiDAR point clouds enable a trained neural signed
distance field to offer a manifold geometry field. This motivates
us to offer an SDF-based Gaussian initialization for physically
grounded primitive placement and a comprehensive geometric
regularization for geometrically consistent rendering and re-
construction. Experiments demonstrate superior reconstruction
accuracy and rendering quality across diverse trajectories. To
benefit the community, the codes are released at https:
//github.com/hku-mars/GS-SDF.
I. INTRODUCTION
3D surface reconstructions and photorealistic renderings
are essential for a wide range of applications, like aug-
mented reality [1], and embodied artificial intelligence [2].
The increasing accessibility of cameras and cost-effective
LiDAR sensors on robots has enabled the collection of rich
multimodal data. LiDAR-visual SLAM technologies [3]–[5]
are widely applied in 3D reconstruction tasks. However, these
methods typically produce only discrete colorized raw point
clouds, which lack complete surface structure and rendering
ability for view-dependent novel view synthesis. In practical
digital-twin applications [6], high-quality watertight surface
reconstructions and photorealistic rendering are critical.
Neural Radiance Fields (NeRFs) [7] have gained atten-
tion for their ability to generate high-fidelity renderings.
However, their computational intensity and inefficiency in
real-time applications limit their practicality. 3D Gaussian
Splatting (3DGS) [8] has emerged as a compelling alterna-
tive to NeRFs for efficient, high-quality rendering. Unlike
NeRF, which requires laborious volume rendering, 3DGS
employs explicit ellipsoidal Gaussian primitives for efficient
rasterization. However, the fragmented structure of these
primitives compromises structural continuity and often mis-
aligns with underlying geometry. This misalignment hinders
geometrically consistent rendering, particularly in robotics
applications where free-view trajectories yield insufficient
J. Liu, Y. W, B. W, C. Z, J. L, and F. Zhang are with the De-
partment of Mechanical Engineering, University of Hong Kong. Email:
jianheng@connect.hku.hk, fuzhang@hku.hk
(a) Data Collection Trajectory and Input Point Clouds
(b) Surface Reconstruction
(c) Geometrically Consistent Gaussian Splats
(d) Novel View Synthesis
Fig. 1.
We show the input raw colorized point clouds (FAST-LIVO2
SYSU Scene) collected by a real-world LiDAR-visual sensor system. We
demonstrate the proposed system, GS-SDF, with the surface reconstruction
results (color indicates the normal direction), Gaussian splats’ geometric
distribution, and novel view synthesis results.
observations. Furthermore, under image-only supervision
with imperfect camera poses, 3DGS frequently overfits to
photometric cues, producing floating artifacts. It motivates
us to introduce geometric regularization into the 3DGS
framework.
Geometric inconsistencies in 3DGS manifest as render-
ing distortions. Current regularization strategies focus on
rendering-depth-derived constraints, such as rendering nor-
mal consistency from 2D Gaussian splatting (2DGS) [9] or
multi-view alignment from PGSR [10]. They focus on object-
centric scenes with abundant multi-view regularization and
limited generalization to complex environments. Depth cam-
eras and LiDARs providing direct structural priors inspire us
arXiv:2503.10170v2  [cs.RO]  29 Jul 2025

<!-- page 2 -->
to incorporate these sensors into the 3DGS framework for
geometric regularization. And depth cameras with limited
precision are confined to indoor applications [11]. While
LiDAR provides sparse point clouds, it remains challenging
to integrate with NeRF [12] or 3DGS [13] frameworks
designed for dense, image-scale geometry.
The combination of LiDAR and neural signed distance
fields (NSDF) has shown promise in high-granularity sur-
face reconstructions [14]. The NSDF, providing a manifold
geometry field, motivates us to explore a more compre-
hensive geometric regularization for Gaussian splats and
the importance of the Gaussian initialization to the results.
Surface reconstruction from Gaussian splats is complicated
by their discontinuous nature. The truncated Signed Distance
Field (TSDF) fusion [15] is commonly applied for 3DGS-
based surface reconstructions through depth images rendered
from Gaussian splatting, which demonstrates unsatisfactory
inaccuracies in surface reconstruction for generalized scenes.
To address the above limitations, we propose the combi-
nation of 2D Gaussian splatting and neural signed distance
field for LiDAR-visual systems. We aim to reconstruct both
the appearance and surface of generalized scenes using
posed images and low-cost LiDAR data from any casual
trajectories. We emphasize the important role of the Gaussian
initialization in returning a good structure of the Gaussian
splats. We explore the combination of the NSDF and Gaus-
sian splats to leverage the structural and photometric hints
for high-granularity surface reconstruction and geometrically
consistent rendering, as shown in Fig. 1. Our contributions
are as follows:
1) A unified LiDAR-visual system achieving geomet-
rically consistent photorealistic rendering and high-
granularity surface reconstruction.
2) A physically grounded Gaussian initialization based
on the neural signed distance field to boost training
convergence and reduce floating artifacts.
3) A thorough shape regularization gives bidirectional su-
pervision between the neural signed distance field and
Gaussian splats for geometrically consistent rendering
and reconstruction.
Extensive experiments validate our approach across diverse
scenarios, demonstrating superior reconstruction accuracy
and rendering quality.
II. RELATED WORKS
A. Geometrically Consistent Novel View Synthesis
NeRFs [7] and 3DGS [8] have been widely employed for
novel view synthesis, owing to their remarkable photoreal-
istic rendering capabilities. However, neither of them can
guarantee geometric consistency, frequently leading to the
emergence of floating artifacts [16]. To tackle this problem,
geometric regularizations and structural priors derived from
depth sensors have been incorporated into both NeRF and
3DGS. 2DGS [9] projects Gaussian ellipsoids into surfels
and enforces normal-depth alignment geometric regular-
ization. PGSR [10] provides further guarantees for multi-
view rendering consistency regularization. RGBD cameras
that provide aligned color and depth information are well
practiced in both NeRFs [17], [18] and 3DGS [11] to align
rendered depth images with sensor-acquired ones. However,
the low measurement accuracy of depth cameras limited their
application in indoor scene settings. LiDAR sensors provide
accurate depth information, but their sparse nature compli-
cates integration with NeRFs [19] or 3DGS [20] frameworks
designed for dense, image-scale geometry. M2Mapping [12]
enforces geometric consistency to NeRF through the com-
bination of NSDF and ray marching, albeit at a significant
computational expense. LIV-GaussMap [20] stabilizes splat
orientations using planar priors derived from LiDAR-inertial
odometry. And LI-GS [13] utilizes Gaussian Mixture Models
for constraining the Gaussian splats’ distribution.
B. Surface Reconstruction
Surface reconstructions predominantly use point clouds
with direct structural information. The TSDF fusion [15] is
well practiced for surface reconstructions [21]. While the
quest for continuous surface reconstruction motivates the
exploration of implicit representations, like Poisson functions
[22] or NSDFs [23], [24]. In terms of the exploration
of Gaussian splats for surface reconstruction, their discon-
tinuous nature complicates surface reconstruction. Recent
approaches circumvent this limitation through intermediate
volumetric representations. SuGaR [25] computes an approx-
imate density field based on the opacities of neighboring
Gaussians, and GOF [26] defines per-point opacity as the
minimal opacity across training views. However, both density
and opacity fields suffer from computational inefficiency
and ambiguous surface-level sets for surface reconstruction.
In contrast, the SDF facilitates precise surface extraction
through defined zero-level sets. The TSDF fusion driven by
the rendered depth images is favored by 3DGS-based surface
reconstructions [9], [10], but yields insufficient accuracy.
Image-driven NSDFs combined with 3DGS [27], [28] are
still under exploration for achieving considerable accuracy
in surface reconstructions.
III. METHODOLOGY
Fig. 2.
The overall pipeline of the proposed system, GS-SDF, for
geometrically consistent rendering and reconstruction.
Given posed images and point clouds from LiDAR-
inertial-visual odometry (e.g., FAST-LIVO2 [5]), we propose
a framework to jointly reconstruct high-granularity scene

<!-- page 3 -->
geometry and photorealistic appearance. As shown in Fig. 2,
our pipeline consists of three stages: A neural signed distance
field (NSDF) is first trained using point clouds to establish
a manifold geometry field (Sec. III-B). Gaussian splats are
initialized from the NSDF (Sec. III-C), and a thorough SDF-
aided shape regularization for both Gaussian splats and the
NSDF (Sec. III-D) achieves a geometric-consistent rendering
and reconstruction.
A. 2D Gaussian Splatting
2DGS [9] represents 3D scenes using planar Gaussian
disks. Each disk is characterized by a center 𝒑∈R3,
orthogonal tangent vectors {𝒕𝑢, 𝒕𝑣∈R3} that define its local
plane, scale factors {𝑠𝑢, 𝑠𝑣∈R}, opacity 𝛼∈R, and view-
dependent color 𝒄∈R3 encoded using spherical harmonics.
The disk’s normal vector is computed as the cross product
of its tangent vectors: 𝒏= 𝒕𝑢× 𝒕𝑣. A point 𝒖= [𝑢, 𝑣]⊤in the
tangent plane is mapped to 3D space via the transformation
𝒑(𝑢, 𝑣) = 𝒑+ 𝑠𝑢𝑢𝒕𝑢+ 𝑠𝑣𝑣𝒕𝑣, with the Gaussian weight kernel
defined as G(𝒖) = exp

−𝑢2+𝑣2
2

. For rendering, Gaussians
are transformed into camera coordinates and depth-sorted.
The final pixel color 𝒄is computed through alpha blending:
𝑪=
∑︁
𝑖=1
𝒄𝑖𝛼𝑖G𝑖(𝒖)
𝑖−1
Ö
𝑗=1
 1 −𝛼𝑗G𝑗(𝒖) =
∑︁
𝑖=1
𝒄𝑖𝑤𝑖.
(1)
The Gaussian attributes are optimized using a combined
rendering loss function of L1 loss and structural dissimilarity
(DSSIM) metrics [8]:
L𝑐= 0.8𝐿1(𝑪, ¯𝑪) + 0.2𝐿DSSIM(𝑪, ¯𝑪),
(2)
where ¯𝑪denotes the ground truth pixel color.
B. Neural Signed Distance Field
We utilize hash encoding [29] and multi-layer perceptrons
(MLPs) to construct a neural network for shaping a scalable
signed distance field: (𝑠, 𝛽) = 𝑓(𝒙). A 3D point 𝒙∈R3 is
mapped to a signed distance value 𝑠∈R and a scale factor
𝛽∈R [12]. For a LiDAR measurement comprising an origin
𝐿𝒐, a ray direction 𝐿𝒅and a distance 𝑡, we uniformly sample
points along the ray path {𝑡𝑖: 𝐿𝒙𝑖= 𝐿𝒐+ 𝑡𝑖𝐿𝒅}. The NSDF
gives prediction at each sampled point: (𝑠𝑖, 𝛽𝑖) = 𝑓(𝐿𝒙𝑖).
The ray distance ¯𝑠𝑖= 𝑡−𝑡𝑖derived from the sampled distance
is converted to an occupancy value ¯𝑜𝑖= Φ(−¯𝑠𝑖, 𝛽𝑖) using
the sigmoid function Φ(𝑣, ℎ) = (1 + exp(−𝑣/ℎ))−1 and the
predicted scale factor 𝛽𝑖. The NSDF is trained using binary
cross-entropy loss [24]:
L𝑠𝑑𝑓= −
∑︁
𝑖=1

𝑜𝑖log( ¯𝑜𝑖) + (1 −𝑜𝑖) log(1 −¯𝑜𝑖)

,
(3)
where 𝑜𝑖= Φ(−𝑠𝑖, 𝛽𝑖) is the predicted occupancy value.
The Eikonal regularization [30] is employed to ensure the
gradient of the SDF maintains unit magnitude:
L𝑒=
∑︁
𝑖=1
(∥∇𝑓(𝒙𝑖)∥2 −1)2 .
(4)
C. Initialization of Gaussian Splats
In this section, we address the crucial role that Gaussian
initialization plays in optimizations and propose a careful
Gaussian initialization with the help of a coarse neural signed
distance field trained using an input point cloud. We provide
a thorough initialization of Gaussians, which boosts the
convergence efficiency and suppresses floating artifacts.
1) SDF-aided Geometry Initialization: The neural signed
distance field, providing a manifold geometry field, can well
stabilize the initial geometry of Gaussians. We extract a
surface mesh from the NSDF using marching cubes [31],
and utilize its vertices as initial Gaussian positions 𝒑. This
approach provides spatially accurate and denser positional
initialization compared to structure-from-motion points, and
reduces the impact of sensor outliers compared to the sam-
pling initialization from input point clouds. The marching
cube resolution directly determines the initial scale {𝑠𝑢, 𝑠𝑣}
of the Gaussian splats. We leverage the SDF’s geometric
hints and Gram-Schmidt orthogonalization for Gaussian ori-
entation initialization, using gradient direction 𝒏=
∇𝑓
∥∇𝑓∥
as Gaussian normals and principal curvature direction 𝒃=
∇2 𝑓
∥∇2 𝑓∥to define the second rotation axis 𝒕𝑢=
𝒃−(𝒏𝑇𝒃)𝒏
∥𝒃−(𝒏𝑇𝒃)𝒏∥,
and the first rotation axis 𝒕𝑣= 𝒏× 𝒕𝑢for the final rotation:
𝑹= [𝒕𝑢, 𝒕𝑣, 𝒏]. The opacity is derived from the SDF value
and Gaussian kernel: 𝛼= exp(−𝑠2
𝛽).
2) Sky Initialization: For the infinite background, we
uniformly tile the opaque Gaussian over a map-sized sphere.
It provides a canvas for infinite objects to avoid floating
artifacts derived from the foreground Gaussians.
3) Color Initialization: Next, we maintain fixed Gaus-
sians’ structural attributes (position, rotation, scale, and opac-
ity) while training the Gaussians on all training images in a
single round for color initialization. The color initialization
helps to solidify the Gaussians’ structure from being deviated
in early training stages.
D. Geometric Regularization for Rendering
The rasterization-based rendering of Gaussian splatting
can lead to occlusion artifacts where foreground elements
inappropriately obscure background content. Geometric reg-
ularization is designed to hold the rendering consistency of
different views to avoid such artifacts. Considering that the
LiDAR structure priors are not always available everywhere,
we incorporate both the rendering-based regularization in 2D
space and the SDF-aided shape regularization in 3D space
to guarantee a geometrically consistent rendering.
1) Rendering Consistency Regularization: Following the
rendering regularization in 2DGS [9], we obtain the rendered
normal images from splats’ normals: 𝑵= Í
𝑖=1 𝒏𝑖𝑤𝑖. And
the finite differential normal images from the rendered depth
images:
ˆ𝑵(𝑥, 𝑦) =
∇𝑥𝒑𝑑×∇𝑦𝒑𝑑
∥∇𝑥𝒑𝑑×∇𝑦𝒑𝑑∥, where 𝒑𝑑is the point
derived from the rendered depth images at piexl (𝑥, 𝑦).
The rendering consistency regularization aligns the rendered
normal with the finite differential normal as follows:
L𝑟= (1 −𝑵𝑇ˆ𝑵).
(5)

<!-- page 4 -->
2) SDF-aided Shape Regularization: With the NSDF pro-
viding a manifold geometry field, we can regularize the
Gaussian splats’ shape to align with the physical surface.
Consider that the splats represent a local plane of the surface,
simply pulling the splat center 𝒑to the zero-level sets
is insufficient [28], as shown in Fig. 3(a). We propose
the shape regularization to enable comprehensive structural
regularization across the entire splat surface. For each splat,
we sample a point 𝒖𝑠= [𝑢𝑠, 𝑣𝑠]𝑇∼N (0, 1) on its disk
surface:
𝒑𝑠= 𝒑+ 𝒕𝑢𝑠𝑢𝑢𝑠+ 𝒕𝑣𝑠𝑣𝑣𝑠.
(6)
For 𝑖-th sampled point 𝒑𝑠𝑖, we infer its signed distance value
𝑓( 𝒑𝑠𝑖) to align splats with NSDF’s zero-level sets:
L𝑠=
∑︁
𝑖=1
1
2𝑊𝑖G(𝒖𝑠𝑖) 𝑓( 𝒑𝑠𝑖)2,
(7)
where 𝑊𝑖= Í
𝑘𝑤𝑘represents the accumulated weight of
the 𝑖-th splat from Eq. 1 in a image. Considering that the
SDF gradient maintains unit magnitude ∥
𝜕𝑓(𝒑𝑠𝑖)
𝜕𝒑𝑠𝑖
∥= 1, we
adopt the 𝐿2 loss function to enable adaptive refinement with
proper gradients: 𝜕L𝑠
𝜕𝒑𝑠𝑖= 𝑊𝑖G(𝒖𝑠𝑖) 𝑓( 𝒑𝑠𝑖)
𝜕𝑓(𝒑𝑠𝑖)
𝜕𝒑𝑠𝑖
.
(a) Center Regularization
(b) Shape Regularization
Fig. 3.
Illustration of different geometric regularizations. The blue splats
represent the initial splat, and the orange splats represent the optimized
splat. The dashed lines represent the optimization directions.
As illustrated in Fig. 3(b), we take one sampling for each
Gaussian splat in the shape regularization. This regularization
provides a concise all-in-one shape alignment to enable
gradient-based supervision of all structural attributes through
the following partial derivatives:
𝜕𝒑𝑠
𝜕𝒑
= 𝑰,
𝜕𝒑𝑠
𝜕𝒕𝑢
= 𝑠𝑢𝑢𝑠,
𝜕𝒑𝑠
𝜕𝒕𝑣= 𝑠𝑣𝑣𝑠, 𝜕𝒑𝑠
𝑠𝑢= 𝒕𝑢𝑢𝑠, 𝜕𝒑𝑠
𝑠𝑣= 𝒕𝑣𝑢𝑠.
E. Optimization
The overall optimization function is a combination of
the supervision from point clouds, images, and geometric
regularizations:
L = L𝑠𝑑𝑓+ 𝜆𝑒L𝑒+ L𝑐+ 𝜆𝑟L𝑟+ 𝜆𝑠L𝑠,
(8)
where 𝜆𝑒= 0.1, 𝜆𝑟= 0.01, and 𝜆𝑠= 0.005 are hyperparam-
eters that balance the loss terms.
IV. EXPERIMENTS
A. Implementation Details
We represent our neural signed distance field follow-
ing a similar architecture to InstantNGP [29], utilizing a
combination of multi-resolution hash encoding and a tiny
MLP decoder. Given any position 𝑥, the hash encoding
concatenates each resolution’s interpolation features to form
a feature vector of size 32. A geometry MLP with 64-
width and 3 hidden layers decodes the encoding feature
to obtain the SDF value and scale. We sample 4096 rays
for NSDF training with 4 surface samples and 4 free space
samples on each ray. We first take 10000 iterations for NSDF
training and leverage the trained NSDF for our initialization
as described in Sec. III-C. Then, we follow the setting of
2DGS [9] to take 30000 iterations for the training. The
implementation is based on LibTorch and CUDA, and all
experiments are conducted on a platform equipped with an
Intel i7-13700K CPU and an NVIDIA RTX 4090 GPU.
(a) VDB-Fusion
(b) iSDF
(c) SHINE-Mapping
(d) 2DGS
(e) PGSR
(f) H2Mapping
(g) M2Mapping
(h) Ours
(i) Ground Truth
Fig. 4.
Reconstructed mesh on the Replica dataset’s Office-2 and colors
indicate the direction of the surface normal. Our method and M2Mapping
can capture precise geometric details in slim objects.
(a) InstantNGP
(b) 3DGS
(c) 2DGS
(d) PGSR
(e) MonoGS
(f) H2Mapping
(g) M2Mapping
(h) Ours
(i) Ground Truth
Fig. 5.
Extrapolation rendering results on the Replica dataset’s room-0.
Our method can preserve great structure and texture details, while the NeRF-
based method, M2Mapping, retains better consistency on the left wall.
B. Experiments Settings
1) Baselines: We compare our method with other pre-
vailing approaches. For pure surface reconstruction methods,
we include the voxel-based method VDBFusion [21], and
the neural-network-based method SHINE-Mapping [24]. For
novel view synthesis methods, we include the NeRF-based
method: InstantNGP [29], and the 3DGS-based methods

<!-- page 5 -->
TABLE I
QUANTITATIVE SURFACE RECONSTRUCTION, INTERPOLATION(I) AND EXTRAPOLATION(E) RENDERING RESULTS ON THE REPLICA DATASET.
Metrics
Methods
Office-0
Office-1
Office-2
Office-3
Office-4
Room-0
Room-1
Room-2
Avg.
C-L1[cm]↓
VDBFusion
0.618
0.595
0.627
0.685
0.633
0.637
0.558
0.656
0.626
iSDF
2.286
4.032
3.144
2.352
2.120
1.760
1.712
2.604
2.501
SHINE-Mapping
0.753
0.663
0.851
0.965
0.770
0.650
0.844
0.826
0.790
H2Mapping
0.557
0.529
0.585
0.644
0.616
0.568
0.523
0.616
0.580
M2Mapping
0.494
0.476
0.501
0.517
0.531
0.486
0.455
0.532
0.499
Ours
0.500
0.487
0.502
0.525
0.540
0.489
0.460
0.542
0.506
F-Score[%]↑
VDBFusion
97.129
97.107
97.007
96.221
96.993
97.871
98.311
96.117
97.095
iSDF
75.829
55.429
71.232
77.274
81.281
78.253
84.801
76.815
75.114
SHINE-Mapping
94.397
95.606
92.150
87.855
94.427
97.010
91.307
92.601
93.169
H2Mapping
98.430
98.279
97.935
97.077
97.391
98.850
99.003
96.904
97.983
M2Mapping
98.970
98.771
98.657
98.415
98.158
99.238
99.496
97.677
98.673
Ours
98.986
98.628
98.671
98.290
97.990
99.242
99.468
97.582
98.607
SSIM(I)↑
InstantNGP
0.981
0.980
0.963
0.960
0.966
0.964
0.964
0.967
0.968
3DGS
0.987
0.980
0.980
0.976
0.980
0.977
0.982
0.980
0.980
2DGS
0.978
0.958
0.967
0.964
0.969
0.966
0.971
0.969
0.968
PGSR
0.981
0.967
0.966
0.963
0.966
0.898
0.973
0.972
0.961
H2Mapping
0.963
0.960
0.931
0.929
0.941
0.914
0.929
0.935
0.938
MonoGS
0.985
0.982
0.973
0.970
0.976
0.971
0.976
0.977
0.975
M2Mapping
0.985
0.983
0.973
0.970
0.977
0.968
0.975
0.977
0.976
Ours
0.986
0.971
0.979
0.976
0.980
0.978
0.980
0.979
0.979
PSNR(I)↑
InstantNGP
43.538
44.340
38.273
37.603
39.772
37.926
38.859
39.568
39.984
3DGS
43.932
43.237
39.182
38.564
41.366
39.081
41.288
41.431
41.010
2DGS
41.854
40.796
38.138
37.704
40.144
37.484
39.905
39.748
39.472
PGSR
43.018
42.303
38.405
37.634
40.143
38.165
40.246
40.415
40.041
H2Mapping
38.307
38.705
32.748
33.021
34.308
31.660
33.466
32.809
34.378
MonoGS
43.648
43.690
37.695
37.539
40.224
37.779
39.563
40.134
40.034
M2Mapping
44.369
44.935
39.652
38.874
41.318
38.541
40.775
40.705
41.146
Ours
44.383
43.389
39.851
39.245
41.989
39.303
41.500
41.802
41.433
SSIM(E)↑
InstantNGP
0.972
0.961
0.934
0.938
0.952
0.918
0.936
0.941
0.944
3DGS
0.936
0.897
0.924
0.917
0.925
0.881
0.915
0.919
0.914
2DGS
0.944
0.829
0.922
0.920
0.940
0.901
0.924
0.922
0.913
PGSR
0.949
0.870
0.932
0.931
0.938
0.970
0.924
0.930
0.930
H2Mapping
0.957
0.955
0.932
0.925
0.937
0.866
0.916
0.917
0.926
MonoGS
0.974
0.961
0.945
0.942
0.950
0.912
0.942
0.946
0.947
M2Mapping
0.980
0.976
0.960
0.964
0.970
0.955
0.963
0.965
0.967
Ours
0.973
0.952
0.956
0.952
0.958
0.945
0.949
0.954
0.955
PSNR(E)↑
InstantNGP
39.874
39.120
31.274
32.135
34.458
32.587
33.024
32.266
34.341
3DGS
31.220
29.959
27.411
26.442
28.324
27.541
28.429
27.139
28.307
2DGS
30.847
25.546
27.248
25.499
29.270
28.149
28.694
26.846
27.762
PGSR
32.525
29.544
28.243
27.032
28.944
27.801
28.977
27.603
28.834
H2Mapping
36.740
37.841
31.427
31.144
31.988
28.815
31.192
30.603
32.468
MonoGS
39.197
38.818
29.740
29.664
31.632
29.949
31.126
30.621
32.593
M2Mapping
41.965
42.215
35.056
37.465
38.667
36.427
37.294
36.722
38.226
Ours
39.102
38.174
30.769
31.503
33.211
32.253
32.226
32.566
33.725
initialized with the LiDAR point clouds: 3D Gaussian Splat-
ting [8], 2D Gaussian Splatting [9], and PGSR [10]. For
depth-aided methods, we include RGBD-based methods:
H2Mapping [18] and MonoGS [11], and LiDAR-augmented
NeRF-based method M2Mapping [12].
2) Metrics: To evaluate the quality of the geometry, we
recover the NSDF to a triangular mesh using marching cubes
[31] and calculate the Chamfer Distance (C-L1, cm) and F-
Score (< 2 cm, %) with the ground truth. For rendering
quality evaluation, we use the Structural Similarity Index
(SSIM) and Peak Signal-to-Noise Ratio (PSNR) to compare
the rendered image with the ground truth image.
C. Comparative Study
1) Replica Datasets: The Replica datasets [6] provide
room-scale indoor simulation RGBD sensor data. We follow
the M2Mapping [12] to emphasize the issues of extrapolation
rendering consistency with extrapolation evaluation datasets
generated by uniformly sampling positions and orientations
in each scene from Replica. The quantitative results, compar-
ing surface reconstruction, interpolation, and extrapolation
rendering, are shown in Tab. I. Our method shows compet-
itive surface reconstruction results with the state of the art,
and the best performance in terms of interpolation rendering,
but fails to compete with the extrapolation rendering results
of the NeRF-based M2Mapping. We validate that our method

<!-- page 6 -->
(a) Campus
(b) Sculture
(c) Culture
(d) Drive
(i) Station
(j) CBD
Fig. 6.
We show our surface reconstruction (color indicates the normal direction) and rendering results on the FAST-LIVO2 Datasets collected by
real-world LiDAR-visual sensor systems under different trajectories, where from top to bottom are collected point clouds, reconstructed meshes, and novel
view synthesis by Gaussian splatting.
is capable of capturing more precise geometric details in
slim objects, as shown in Fig. 4. In the extrapolation views
shown in Fig. 5, our method can retain more high-frequency
textures (sharper spots) while NeRF-based methods show
more natural transitions (smoother lighting).
2) FAST-LIVO2 Datasets: For real-world scenes, we eval-
uate generalized types of trajectory datasets collected with a
camera and a LiDAR and use the localization results from
FAST-LIVO2 [5] as the input poses, as shown in the top
row of Fig. 1 and Fig. 6. We adopt a train-test split for
interpolation rendering evaluation, where every 8th photo is
used for the test sets and the rest for the train sets [8], and the
quantitative results are shown in Tab. II. Our method yields
the best rendering performance overall, and also returns high-
granularity surface reconstructions and considerable extrap-
olation rendering results across generalized scenes, as shown
in Fig. 1 and Fig. 6.
We compare the qualitative rendering results in a free
trajectory (Drive), as shown in Fig. 8, where our method out-
performs the others in retaining detailed textures of the floor
TABLE II
QUANTITATIVE RESULTS ON THE FAST-LIVO2 DATASETS.
Metrics
Methods
Campus
Sculpture
Culture
Drive
Station
SYSU
CBD
Avg.
SSIM↑
InstantNGP
0.789
0.698
0.670
0.697
0.780
0.792
0.779
0.744
3DGS
0.849
0.769
0.726
0.780
0.853
0.789
0.825
0.799
2DGS
0.839
0.730
0.611
0.741
0.798
0.808
0.760
0.755
PGSR
0.836
0.745
0.595
0.762
0.828
0.806
0.735
0.758
M2Mapping
0.834
0.729
0.727
0.764
0.809
0.789
0.850
0.786
Ours
0.858
0.774
0.797
0.788
0.861
0.841
0.892
0.830
PSNR↑
InstantNGP
28.880
22.356
21.563
24.145
24.111
22.517
22.468
23.720
3DGS
31.310
24.128
21.764
25.837
26.859
28.798
22.078
25.825
2DGS
30.611
22.654
19.218
24.847
24.756
27.517
22.022
24.518
PGSR
30.648
23.302
18.438
25.382
25.504
27.563
20.662
24.500
M2Mapping
30.681
23.453
24.695
25.941
25.859
28.011
26.655
26.471
Ours
31.697
24.196
25.074
25.657
27.065
28.532
26.696
26.988
with the help of geometric regularization. It is noted that both
the LiDAR-augmented NeRF-based method (M2Mapping)
and 3DGS-based method (Ours) give physically grounded
rendering results, while the NeRF-based method shows more
complete rendering at views lacking observations, and the
3DGS-based methods show more detailed rendering overall.
We also compared the reconstruction results over the diverse
types of methods, as shown in Fig. 7. It is shown that LiDAR-
based methods (VDBFusion, SHINE-Mapping, M2Mapping,
Ours) offer far more detailed geometry than the vision-only

<!-- page 7 -->
(a) VDBFusion
(b) SHINE-Mapping
(c) 2DGS
(d) PGSR
(e) M2Mapping
(f) Ours
Fig. 7.
The surface reconstructions in the FAST-LIVO2 Campus dataset.
(a) InstantNGP
(b) 3DGS
(c) 2DGS
(d) PGSR
(e) M2Mapping
(f) Ground Truth
(g) RR
(h) RR+CR
(i) Ours (RR+SR)
Fig. 8.
The qualitative comparative results in the FAST-LIVO2 Drive
dataset. We compare different geometric regularization strategies represented
in Rendering Regularization (RR), Center Regularization (CR), and Shape
Regularization (SR).
methods (2DGS, PGSR).
D. Ablation Study
1) Initialization: In this section, we only apply the ren-
der regularization (Eq. 5) as a pure image supervision to
address the importance of the proposed Gaussian initial-
izations (Sec. III-C) in converging to a good structure of
Gaussians. As shown in Fig. 9, the original initialization,
with random sampling position from the input point clouds
and random attributes initialization, falls into unreasonable
structures (Fig. 9(b)). The proposed SDF-aided physically-
grounded initialization gives a structure that is more in line
with the real scenarios. A color initialization assigns a proper
appearance to the Gaussians, which is essential to stabilize
the structure of the Gaussians. As shown in Fig. 10, the
initialized Gaussian splats with proper structure but wrong
color are deviated during early training.
2) Geometric regularization: We compare the geometric
regularization methods between the render regularization
[9], center regularization and shape regularization (Sec. III-
D.2), as shown in Fig. 8(g)-8(i). The render regularization
(a) Random Initialization (0 iteration)
(b) Random Initialization (30000 iteration)
(c) SDF-aided Initialization (0 iteration)
(d) SDF-aided Initialization (30000 iteration)
(e) SDF-aided Initialization with Shape Reg. (30000 iteration)
Fig. 9.
The ablation study of the SDF-aided Gaussian initialization and
shape regularization in the FAST-LIVO2 Station scene. From left to right
are the rendered color, depth, and normal images.
(a) 0 iter.
(b) 2000 iter.
(c) 30000 iter.
Fig. 10.
The ablation study of the color initialization in the Replica Office0
dataset. The first row shows the results without color initialization, and the
second row shows the results with color initialization.
(PSNR: 27.24, SSIM: 0.805, Fig. 8(g)) improves the ge-
ometric consistency to an extent, but still produces false
structure on the flat surfaces under scenes with less multi-
view constraints. The center regularization (PSNR: 28.60,
SSIM: 0.861, Fig. 8(h)) shows its limited improvement in
aligning Gaussians with the SDF field, still producing blur
rendering on the ground. And our proposed shape regular-
ization (PSNR: 28.66, SSIM: 0.866, Fig. 8(i)) shows better
alignment with the surface to retain the details of the bricks.
As shown in Fig. 9(e)’s rendered normal image, the shape

<!-- page 8 -->
regularization gives a more detailed and physically grounded
structure of Gaussians for sharper rendering results, while the
evaluation metrics do not show a significant improvement.
E. Efficiency Analysis
The training time for the Replica Room-2 scene takes
16.6 minutes, and the average rendering time for a 1200x680
image takes about 9.9 milliseconds and reaches 101.1 frames
per second (FPS). We evaluate the efficiency of the com-
pared NeRF-based method, M2Mapping, which takes 53.9
milliseconds for rendering a 1200x680 image (18.4 FPS).
As our method trains both the NSDF and 3DGS, it takes
more training time than the other 3DGS-based methods, like
2DGS (9.1 minutes / 103.2 FPS), but the inference time is
comparable to the other methods and shows a significant
improvement over the NeRF-based methods.
V. CONCLUSION
In this paper, we addressed the geometric inconsistency
challenges of Gaussian splatting in robotics applications
by proposing a unified LiDAR-visual system that syner-
gizes Gaussian splatting with neural signed distance fields.
We leverage the NSDF to provide a physically grounded
Gaussian initialization and effective shape regularization
for geometrically consistent rendering and reconstruction.
Extensive experiments demonstrate our method’s superior
reconstruction accuracy and rendering quality across diverse
trajectories. However, comparative analysis reveals the lim-
itations in extrapolative novel view synthesis capabilities
contrasted with NeRF-based frameworks. Therefore, we aim
to tackle this limitation in future work by exploring advanced
neural rendering techniques.
REFERENCES
[1] Y. Wang, Z. Su, N. Zhang, R. Xing, D. Liu, T. H. Luan, and X. Shen,
“A survey on metaverse: Fundamentals, security, and privacy,” IEEE
Communications Surveys & Tutorials, vol. 25, no. 1, pp. 319–352,
2022.
[2] T. Wang, X. Mao, C. Zhu, R. Xu, R. Lyu, P. Li, X. Chen, W. Zhang,
K. Chen, T. Xue, et al., “Embodiedscan: A holistic multi-modal
3d perception suite towards embodied ai,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 19 757–19 767.
[3] T. Shan, B. Englot, C. Ratti, and R. Daniela, “Lvi-sam: Tightly-
coupled lidar-visual-inertial odometry via smoothing and mapping,” in
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2021, pp. 5692–5698.
[4] J. Lin and F. Zhang, “R 3 live: A robust, real-time, rgb-colored, lidar-
inertial-visual tightly-coupled state estimation and mapping package,”
in 2022 International Conference on Robotics and Automation (ICRA).
IEEE, 2022, pp. 10 672–10 678.
[5] C. Zheng, W. Xu, Z. Zou, T. Hua, C. Yuan, D. He, B. Zhou, Z. Liu,
J. Lin, F. Zhu, et al., “Fast-livo2: Fast, direct lidar-inertial-visual
odometry,” arXiv preprint arXiv:2408.14035, 2024.
[6] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J.
Engel, R. Mur-Artal, C. Ren, S. Verma, et al., “The replica dataset:
A digital replica of indoor spaces,” arXiv preprint arXiv:1906.05797,
2019.
[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp.
99–106, 2021.
[8] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, pp. 1–14, 2023.
[9] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splat-
ting for geometrically accurate radiance fields,” in ACM SIGGRAPH
2024 conference papers, 2024, pp. 1–11.
[10] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang,
H. Liu, H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction,” arXiv preprint
arXiv:2406.06521, 2024.
[11] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian
Splatting SLAM,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024.
[12] J. Liu, C. Zheng, Y. Wan, B. Wang, Y. Cai, and F. Zhang, “Neural
surface reconstruction and rendering for lidar-visual systems,” arXiv
preprint arXiv:2409.05310, 2024.
[13] C. Jiang, R. Gao, K. Shao, Y. Wang, R. Xiong, and Y. Zhang, “Li-
gs: Gaussian splatting with lidar incorporated for accurate large-scale
reconstruction,” IEEE Robotics and Automation Letters, 2024.
[14] J. Liu and H. Chen, “Towards large-scale incremental dense mapping
using robot-centric implicit neural representation,” in 2024 IEEE
International Conference on Robotics and Automation (ICRA), 2024,
pp. 4045–4051.
[15] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim,
A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon,
“Kinectfusion: Real-time dense surface mapping and tracking,” in
2011 10th IEEE International Symposium on Mixed and Augmented
Reality.
Ieee, 2011, pp. 127–136.
[16] Y. Tao, M. ´A. Mu˜noz-Ba˜n´on, L. Zhang, J. Wang, L. F. T. Fu, and
M. Fallon, “The oxford spires dataset: Benchmarking large-scale lidar-
visual localisation, reconstruction and radiance field methods,” arXiv
preprint arXiv:2411.10546, 2024.
[17] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit map-
ping and positioning in real-time,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 6229–6238.
[18] C. Jiang, H. Zhang, P. Liu, Z. Yu, H. Cheng, B. Zhou, and S. Shen,
“H {2}-mapping: Real-time dense mapping using hierarchical hybrid
representation,” IEEE Robotics and Automation Letters, 2023.
[19] Y. Tao, Y. Bhalgat, L. F. T. Fu, M. Mattamala, N. Chebrolu, and
M. Fallon, “Silvr: Scalable lidar-visual reconstruction with neural ra-
diance fields for robotic inspection,” in IEEE International Conference
on Robotics and Automation (ICRA), 2024.
[20] S. Hong, J. He, X. Zheng, H. Wang, H. Fang, K. Liu, C. Zheng, and
S. Shen, “Liv-gaussmap: Lidar-inertial-visual fusion for real-time 3d
radiance field map rendering,” arXiv preprint arXiv:2401.14857, 2024.
[21] I. Vizzo, T. Guadagnino, J. Behley, and C. Stachniss, “Vdbfusion:
Flexible and efficient tsdf integration of range sensor data,” Sensors,
vol. 22, no. 3, p. 1296, 2022.
[22] M. Kazhdan, M. Bolitho, and H. Hoppe, “Poisson surface recon-
struction,” in Proceedings of the fourth Eurographics symposium on
Geometry processing, vol. 7, no. 4, 2006.
[23] J. Ortiz, A. Clegg, J. Dong, E. Sucar, D. Novotny, M. Zollhoefer, and
M. Mukadam, “isdf: Real-time neural signed distance fields for robot
perception,” arXiv preprint arXiv:2204.02296, 2022.
[24] X. Zhong, Y. Pan, J. Behley, and C. Stachniss, “Shine-mapping:
Large-scale 3d mapping using sparse hierarchical implicit neural
representations,” arXiv preprint arXiv:2210.02299, 2022.
[25] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5354–5363.
[26] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient
adaptive surface reconstruction in unbounded scenes,” ACM Transac-
tions on Graphics (TOG), vol. 43, no. 6, pp. 1–13, 2024.
[27] X. Lyu, Y.-T. Sun, Y.-H. Huang, X. Wu, Z. Yang, Y. Chen, J. Pang,
and X. Qi, “3dgsr: Implicit surface reconstruction with 3d gaussian
splatting,” arXiv preprint arXiv:2404.00409, 2024.
[28] M. Yu, T. Lu, L. Xu, L. Jiang, Y. Xiangli, and B. Dai, “Gsdf: 3dgs
meets sdf for improved rendering and reconstruction,” arXiv preprint
arXiv:2403.16964, 2024.
[29] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graph-
ics primitives with a multiresolution hash encoding,” arXiv preprint
arXiv:2201.05989, 2022.
[30] H. Yang, Y. Sun, G. Sundaramoorthi, and A. Yezzi, “Steik: Stabilizing
the optimization of neural signed distance functions and finer shape
representation,” in Thirty-seventh Conference on Neural Information
Processing Systems, 2023.
[31] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution
3d surface construction algorithm,” ACM Siggraph Computer Graph-
ics, vol. 21, no. 4, pp. 163–169, 1987.
