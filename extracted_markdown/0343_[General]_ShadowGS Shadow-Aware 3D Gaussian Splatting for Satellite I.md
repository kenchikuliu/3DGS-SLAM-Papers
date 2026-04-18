<!-- page 1 -->
ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery
Feng Luo,Hongbo Pan*,Xiang Yang,Baoyu Jiang,Fengqing Liu,Tao Huang
Central South University
245007016@csu.edu.cn, hongbopan@csu.edu.cn
Abstract
3D Gaussian Splatting (3DGS) has emerged as a novel
paradigm for 3D reconstruction from satellite imagery.
However, in multi-temporal satellite images, prevalent
shadows exhibit significant inconsistencies due to varying
illumination conditions. To address this, we propose Shad-
owGS, a novel framework based on 3DGS. It leverages
a physics-based rendering equation from remote sensing,
combined with an efficient ray marching technique, to pre-
cisely model geometrically consistent shadows while main-
taining efficient rendering. Additionally, it effectively disen-
tangles different illumination components and apparent at-
tributes in the scene. Furthermore, we introduce a shadow
consistency constraint that significantly enhances the geo-
metric accuracy of 3D reconstruction. We also incorpo-
rate a novel shadow map prior to improve performance
with sparse-view inputs.
Extensive experiments demon-
strate that ShadowGS outperforms current state-of-the-art
methods in shadow decoupling accuracy, 3D reconstruction
precision, and novel view synthesis quality, with only a few
minutes of training. ShadowGS exhibits robust performance
across various settings, including RGB, pansharpened, and
sparse-view satellite inputs.
1. Introduction
High-resolution optical satellites capture large-scale, sub-
meter imagery of the Earth’s surface from orbital altitudes.
Compared to close-range or UAV platforms, satellites pro-
vide extensive spatial coverage at lower acquisition costs,
making them highly valuable for large-scale 3D reconstruc-
tion [40], digital twins, and smart city applications [43].
With the ongoing launch of new-generation sub-meter satel-
lites, imagery availability has increased significantly, fur-
ther establishing satellite-based 3D reconstruction as an im-
portant research area in computer vision.
*Corresponding author.
Figure 1. ShadowGS reconstructs 3D geometry with consistent
shadow modeling from multi-temporal satellite imagery. The top
row displays reconstructed DSMs while the bottom row shows
shadow decomposition results. Compared to EO-NeRF [46] and
EOGS [1], our method produces superior reconstruction quality
with sharper edges, richer details, smoother surfaces, and shadows
that align precisely with scene geometry.
However, due to orbital constraints, satellite images are
often acquired at different times and from limited view-
points, and frequently contain shadows caused by occluded
sunlight. Although shadows can provide valuable 3D cues,
their strong inconsistency across multi-temporal images
poses significant challenges for reconstruction. Traditional
multi-view stereo (MVS) methods [9, 15, 20, 21] gener-
ally assume simultaneous image capture and struggle with
strong appearance variations across time.
Neural Radi-
ance Fields (NeRF) [48] have shown promise for multi-
temporal satellite imagery, yet limitations remain. For in-
stance, S-NeRF [11] and SatNeRF [45] use MLPs to model
shadow features related to sun position but fail to incorpo-
rate geometric context, leading to inaccurate shadow sep-
aration. EO-NeRF [46] models geometry-dependent shad-
ows but lacks strict constraints between shadow and geom-
etry, causing substantial shadow information to be entan-
gled within geometric representations. Furthermore, NeRF-
based methods typically suffer from high computational
1
arXiv:2601.00939v1  [cs.CV]  4 Jan 2026

<!-- page 2 -->
cost and slow inference.
3D Gaussian Splatting (3DGS) [29], with its efficient
rasterization-based rendering, has recently emerged as a
promising alternative for satellite 3D reconstruction. How-
ever, rasterization is inherently local and struggles to model
global effects like shadows.
Existing 3DGS adaptations
have attempted to handle multi-temporal shadow inconsis-
tencies in different ways. SatGS [2] uses an MLP to esti-
mate solar visibility per Gaussian but ignores geometric re-
lationships, while EOGS [1] introduces shadow mapping to
model geometry-aware shadows, though it remains an ap-
proximation and tends to produce aliasing artifacts.
To address these issues, we propose ShadowGS, a
3DGS-based framework that disentangles geometry (eleva-
tion and normals), appearance (albedo), and illumination
(direct sunlight, skylight, and near-surface reflection) from
multi-temporal satellite images, enabling rendering under
arbitrary views and lighting. Specifically, ShadowGS as-
signs two sets of spherical harmonic (SH) coefficients to
each Gaussian to represent albedo and near-surface reflec-
tion, while a global low-order SH models skylight.
For
geometrically consistent shadows, we cast rays from each
Gaussian toward the sun and employ hardware-accelerated
ray marching to determine occlusions and compute solar
visibility. A physics-based rendering equation is then ap-
plied after rasterization to compose the final pixel color
from albedo and multiple illumination components. To im-
prove geometry, we derive Gaussian normals and depth
contributions via ray-Gaussian intersection and enforce a
depth-normal consistency constraint.
We further intro-
duce a shadow consistency constraint, which requires that
when the camera view aligns with the sun direction, object
shadows should be fully self-occluded—i.e., the rendered
shadow map should be entirely lit. This encourages Gaus-
sians to align closely with true surfaces and converge to
higher opacity. In sparse-view settings, we integrate a pre-
trained shadow detection network [63] to provide shadow
map priors that guide optimization under limited inputs.
Experiments on the DFC2019 [6, 31] and IARPA [5]
datasets show that ShadowGS accurately models geometry-
aware shadows and outperforms existing methods in
shadow disentanglement, 3D reconstruction accuracy, and
novel view synthesis—all within minutes of training. Our
main contributions are:
• An efficient ray-marching-based shadow computation
method for satellite imagery that models geometry-
consistent shadows while maintaining high rendering ef-
ficiency.
• A remote-sensing physics-based rendering equation that
effectively disentangles illumination and appearance.
• A shadow consistency constraint that significantly im-
proves reconstruction quality.
• Integration of a shadow map prior to enhance perfor-
mance under sparse-view conditions.
2. Related Work
2.1. Shadow in Remote Sensing
Shadows result from the occlusion of light propagation, re-
vealing interactions among light sources, scene geometry,
and object spatial relationships [23]. They are prevalent
in remote sensing imagery and present dual characteristics:
while often degrading image information and hampering vi-
sual interpretation, object classification, and quantitative in-
version, they simultaneously offer valuable geometric cues.
Prior work has leveraged shadows for building height esti-
mation [27, 37], scene geometry recovery [28, 51], illumi-
nation direction estimation [30], and camera calibration [7].
Accurate shadow detection and removal are therefore
crucial. Early methods primarily relied on handcrafted fea-
tures and traditional machine learning [16, 19]. Recent ad-
vances in deep learning have substantially improved perfor-
mance in both detection and removal [18, 22, 34, 63]. In
remote sensing, the introduction of the AISD dataset [41]
spurred the development of specialized detectors [38, 42].
Notably, SEO [47] recently released a large-scale, high-
resolution dataset containing multi-temporal and multi-
view WorldView-3 imagery, along with geo-registered
shadow masks and aligned LiDAR DSMs. Shadow detec-
tion networks trained on such data have been used to super-
vise EO-NeRF [46], demonstrating the potential of shadow
priors in enhancing radiance field methods.
2.2. NeRF for Satellite Images
NeRF[48] model scenes as continuous volumetric represen-
tations using fully-connected networks. By mapping 3D
coordinates and viewing directions to volume density and
view-dependent color, and employing volume rendering,
NeRF can optimize scene representations from images with
known poses while handling complex appearance changes.
In remote sensing, NeRF has been adapted to address il-
lumination inconsistencies, shadows, and transient objects
in multi-temporal data. S-NeRF [11] first introduced NeRF
to satellite photogrammetry, using a lighting model to de-
couple albedo and irradiance. Sat-NeRF [45] incorporated
rational polynomial camera models and transient embed-
dings to handle dynamic elements. EO-NeRF [46] directly
rendered shadows by integrating geometry and solar posi-
tion, leveraging UTM coordinates and multi-parameter joint
optimization to improve accuracy. On the application side,
SpS-NeRF [61] combined traditional MVS [20] depth pri-
ors to enhance sparse-view rendering; Sat-Mesh[52] used
signed distance functions for high-quality surface recon-
struction; Season-NeRF [13] introduced temporal encod-
ing for seasonal feature rendering; and Snake-NeRF [4] ex-
tended NeRF to large-scale satellite 3D reconstruction. Fur-
2

<!-- page 3 -->
Figure 2. The overall pipeline of ShadowGS.
ther efficiency improvements have been achieved by meth-
ods like RS-NeRF [50], SatNGP [3], and SatensoRF [62]
through various acceleration strategies.
2.3. 3DGS for Satellite Images
3DGS [29] initializes Gaussian scenes from SfM point
clouds [53], explicitly representing scenes with anisotropic
3D Gaussians and rendering via differentiable rasterization.
Combining the strengths of neural fields and point-based
rendering, 3DGS achieves high-fidelity, real-time render-
ing and has attracted significant attention. Recent exten-
sions include geometry reconstruction [8, 24, 56, 57], ren-
dering quality improvement [36, 55], sparse-view general-
ization [33, 54, 58], and inverse rendering/relighting [14,
17, 26, 35]. EVER [44] and 3DGRT [49] further integrated
ray tracing into 3DGS, enabling complex camera models
and accurate shadow computation.
In satellite imagery, 3DGS has been specialized for re-
mote sensing applications: EOGS [1] first adapted 3DGS to
satellite data using an affine camera model for RPC fitting
and implemented shadow mapping for shadow rendering;
SatGS [2] incorporated appearance embedding and uncer-
tainty modeling to handle seasonal variations and transient
objects; Skysplat [25] proposed a feedforward 3DGS frame-
work for rapid reconstruction from sparse multi-temporal
images; and Skyfall-GS [32] combined 3DGS with diffu-
sion models and curriculum learning to synthesize naviga-
ble 3D cityscapes with geometric consistency and visual re-
alism.
3. Method
In this section, we introduce ShadowGS, a novel frame-
work based on 3D Gaussian Splatting, designed to de-
couple geometric properties (e.g., normal, depth), appear-
ance attributes (e.g., albedo), and illumination components
(e.g., direct sunlight, skylight, and near-surface reflection)
from multi-temporal satellite image collections. The over-
all pipeline is illustrated in Fig. 2.
3DGS Basics: Each Gaussian ellipsoid is parameterized by
a center position µ ∈R3, a scaling factor s ∈R3, and a ro-
tation quaternion q ∈R4. Its spatial influence is defined by
a covariance matrix Σ ∈R3×3, constructed from a scaling
matrix S (derived from s) and a rotation matrix R (obtained
from q) as Σ = RSST RT . The 3D Gaussian distribution
G(x) is formulated as:
G(x) = e−(x−µ)T Σ−1(x−µ)
(1)
3DGS employs EWA splatting [64] to project 3D Gaus-
sians onto the 2D image plane. The projected 2D covariance
matrix Σ′ is given by:
Σ′ = JWΣWT JT
(2)
where W denotes the viewing transformation matrix from
world to camera coordinates, and J is the Jacobian of the
projective transformation.
In addition to geometry, each Gaussian stores an opacity
value o and a set of learnable spherical harmonics (SH) co-
efficients that model the view-dependent appearance c. The
color C of a pixel is computed via alpha blending:
C =
X
i
Tiαici,
Ti =
i−1
Y
j=1
(1 −αj)
(3)
Here, αi is the pixel translucency of the i-th Gaussian, de-
termined by the opacity of the i-th Gaussian and the pixel’s
position.
During training, all Gaussian parameters are optimized
via a photometric loss Lc:
Lc = (1 −λssim)Lcolor + λssimLD-SSIM
(4)
where Lcolor and LD-SSIM denote the color reconstruction
loss and structural similarity loss, respectively, and λssim
controls the balance between them.
3

<!-- page 4 -->
3.1. Camera Model and Geometry
Camera Model: The standard 3DGS framework is built
on the pinhole camera model, whereas satellite imagery
typically adopts the Rational Polynomial Camera (RPC)
model to map image coordinates to geographic locations.
As shown in [59], the pinhole model introduces only minor
error when approximating the RPC model within a local re-
gion. Therefore, ShadowGS fits the RPC model using a
pinhole camera to align with the existing 3DGS pipeline.
Specifically, ShadowGS first refines the original RPC pa-
rameters via bundle adjustment [45] to generate a sparse
point cloud for initializing the 3D Gaussians. A pinhole
model is then used in the local tangent plane coordinate sys-
tem to approximate the optimized RPC model. The average
reprojection error for RPC model fitting across all scenes in
ShadowGS remains below 0.5 pixels.
Geometric Representation: Radiance-based methods of-
ten suffer from geometry–radiance ambiguity [60], which
complicates the accurate recovery of geometry and appear-
ance in real scenes. Numerous relighting and inverse ren-
dering techniques [14, 17, 26, 35] address this by explicitly
defining depth and normal attributes in 3DGS and intro-
ducing geometric regularization. Following RadeGS [57],
we adopt an explicit ray–Gaussian intersection strategy to
determine the Gaussian’s normal direction and its depth
contribution per pixel. Specifically, for a 3D Gaussian G,
let (uc, vc) denote the center of its 2D projection. For a
pixel (u, v), the intersection between the camera ray and the
Gaussian forms a 1D Gaussian distribution, whose peak de-
fines the ray–Gaussian intersection point. The correspond-
ing depth d represents the depth contribution of Gaussian G
to pixel (u, v) and is given by:
d = zc + zc
tc
m
uc −u
vc −v

,
ˆm = vT Σ′−1
vT Σ′−1v
(5)
Here, zc and tc represent the depth values of the Gaussian
center and the distance from the Gaussian center to the cam-
era center, respectively, and the vector m is a 1 × 2 vector
formed by omitting the third row of ˆm, v = (0, 0, 1)T .
This formulation implies that the intersection between the
3D Gaussian and the camera ray defines a surface, where
each pixel corresponds to a different depth value. The nor-
mal vector n of this surface is defined as the Gaussian’s
normal vector:
n = JT (−m
−1)T
(6)
The normalized depth D and normal maps N are ren-
dered via alpha blending:
D
N

=
X
i
Tiαi
P
i Tiαi
di
ni

(7)
To further enhance geometric detail,
we apply a
depth–normal consistency loss Ln [24]:
Ln = (1 −NT ˜N)
(8)
where ˜N denotes the surface normal derived from the ren-
dered depth map D via finite differences.
3.2. Physics-based Rendering Equation
Ray-marching
shadow:
Following
the
hardware-
accelerated ray-tracing pipeline for 3DGS introduced in
[49], we model shadows using an efficient ray-marching
strategy. In satellite scenes, the sun is considered as the
sole directional light source. We leverage the ray tracer to
evaluate the solar visibility of each Gaussian and combine
it with the standard 3DGS rasterizer for pixel-accurate
shadow rendering.
Specifically, all Gaussians are organized into a stretched
icosahedron bounding volume hierarchy (BVH). The
bounding boxes are adaptively scaled according to each
Gaussian’s opacity and geometry by applying the following
transformation to the icosahedron vertices a [49]:
a ←a
p
2 log(o/omin)SRT + µ
(9)
To ensure the bounding volume fully covers the effective
region of each Gaussian, a transparency threshold omin =
0.001 is applied. For each Gaussian center, taken as the ray
origin µ0, a ray is cast along the solar direction r. Using
a fixed step size, intersections with other Gaussians are de-
tected. The intersection point τ with the i-th Gaussian is
defined as the peak of the 1D Gaussian distribution G1D
formed by the ray–Gaussian intersection (consistent with
Section 3.1), and is computed as:
τ = (µ0 −µi)T Σ−1
i r
rT Σ−1
i r
(10)
The response value ˜α of the intersecting Gaussian along
the ray is:
˜α = oG1D(τ)
(11)
The solar visibility Ssun of the current Gaussian is then
given by:
Ssun =
k
Y
i=1
(1 −˜αi)
(12)
where k is the total number of intersecting Gaussians and
˜αi is the response value of the k-th intersected Gaussian.
Remote Sensing Physics-based Rendering Equation: To
effectively decouple illumination components and appear-
ance attributes in satellite imagery, we model skylight us-
ing a set of globally shared spherical harmonics (SH). This
representation captures spatially uniform skylight radiance
lsky, with higher-order SH terms disabled to restrict learn-
ing to low-frequency features. Each Gaussian is assigned
4

<!-- page 5 -->
two sets of SH coefficients: one encoding albedo f, and the
other representing reflected radiance ln from nearby sur-
faces.
This allows each primitive to model independent
material properties and local light interactions. As in stan-
dard 3DGS, we progressively enable higher-order SH terms
to represent high-frequency details in appearance and near-
surface reflections. We render albedo and illumination com-
ponent maps via alpha blending:




S
Lsky
Ln
F



=
X
i
Tiαi




Ssun
lsky
ln
f




i
(13)
The total incident radiance at a surface point is computed
as:
Ltotal = S + (1 −S) · (Lsky + Ln)
(14)
This formulation ensures that sunlit regions are domi-
nated by direct solar radiance S, while shadowed areas are
illuminated by skylight Lsky and reflections Ln from nearby
surfaces. The final rendered color is obtained as:
C = F · Ltotal
(15)
3.3. Shadow Consistency Constraint
The discrete 3D Gaussian representation in 3DGS exhibits
greater irregularity compared to NeRF’s continuous neu-
ral representation, often leading to insufficient optimiza-
tion constraints.
To address this limitation, we intro-
duce a shadow consistency constraint that leverages unique
shadow formation characteristics in satellite imaging.
As shown in Fig. 3, when the satellite is at position A,
shadows cast by objects under sunlight are visible in the
captured image. However, at position B where the satellite
view direction aligns with the sun direction, these shadows
become self-occluded by the object’s own geometry, result-
ing in shadow-free imaging. This phenomenon occurs due
to the parallel light characteristics of both solar illumination
and satellite viewing rays. We formalize this constraint as
follows. Given a virtual camera viewpoint and a collinear
sun direction, we render the corresponding shadow map Sv
using the method in Section 3.2. The shadow consistency
loss is defined as:
LS1 = ∥Sv −1∥1
(16)
In practice, we implement this constraint under two con-
figurations: (1) fixing the camera viewpoint while aligning
the sun direction with the view direction, and (2) simulta-
neously adjusting both sun and camera directions to be per-
pendicular to the scene surface. In both cases, the constraint
drives the rendered shadow map toward a fully illuminated
state under collinear light-view conditions. This constraint
encourages surface Gaussians to achieve higher opacity and
promotes better alignment with the underlying geometric
surfaces, thereby enhancing reconstruction quality.
Figure 3. Shadow consistency constraint. Shadows are visible
when the satellite’s viewing direction differs from the solar direc-
tion (position A), but become self-occluded by the object’s geom-
etry when the two directions align (position B).
3.4. Shadow Prior for Sparse View
Sparse input views pose significant challenges for both
3DGS and NeRF frameworks, often leading to overfitting
and degraded performance. Existing approaches typically
rely on depth priors or diffusion models [33, 39, 54, 58],
yet these face limitations: depth-based methods struggle in
textureless regions, while diffusion-based techniques may
introduce erroneous pseudo-views.
ShadowGS addresses this by leveraging geometry-
correlated shadow information as a robust supervision sig-
nal [47].
We integrate FDRNet [63], a self-supervised
shadow detection network that demonstrates low false-
negative rates in satellite imagery. Under sparse-view con-
ditions, FDRNet-provided shadow priors effectively guide
the optimization of global geometry and illumination pa-
rameters, promoting stable convergence.
Specifically, we minimize the discrepancy between the
rendered shadow map S and the FDRNet-extracted shadow
mask ˆS using a binary cross-entropy loss:
LS3 = −(S log2( ˆS) + (1 −S) log2(1 −ˆS))
(17)
To address false positives in vegetation areas, we employ
the Normalized Difference Vegetation Index (NDVI) with
multi-spectral data, or the Difference Enhanced Vegetation
Index (DEVI) for RGB imagery, excluding detected vegeta-
tion regions from shadow supervision. Furthermore, we dis-
continue the shadow prior loss after densification concludes
to prevent interference from remaining false positives.
3.5. Training Strategy and Total Loss
Total Loss: In addition to the aforementioned losses, we in-
corporate a binary cross-entropy loss LS2 —consistent with
EOGS [1]—to encourage projected shadows to converge to-
ward binary states (fully transparent or opaque):
LS2 = −(S log2(S) + (1 −S) log2(1 −S))
(18)
5

<!-- page 6 -->
AOI
DFC2019 (RGB)
IARPA(Pansharpened)
PSNR(dB)↑/ MAE(m)↓
PSNR(dB)↑/ MAE(m)↓
JAX 004
JAX 068
JAX 165
JAX 214
JAX 260
OMA 288
OMA 315
Mean
IARPA 001
IARPA 002
IARPA 003
Mean
S2P[10]
–
/ 2.88
–
/ 1.64
–
/ 4.45
–
/ 2.82
–
/ 2.02
–
/ 3.21
–
/ 1.97
–
/ 2.71
–
/ 3.00
–
/ 4.65
–
/ 2.37
–
/ 3.34
Sat-NGP[3]
20.33 / 1.47
18.14 / 1.43
23.14 / 2.20
17.99 / 1.99
18.73 / 2.25
15.38 / 5.08
14.99 / 1.72
18.39 / 2.31
–
/
–
–
/
–
–
/
–
–
/ –
EO-NeRF[46]
20.16 / 1.41
17.80 / 1.56
23.46 / 3.39
16.76 / 2.76
18.57 / 1.82
17.79 / 4.87
15.51 / 1.79
18.58 / 2.51
22.17 / 1.56
21.06 / 1.87
19.17 / 2.34
20.80 / 1.92
EOGS[1]
22.56 / 2.06
21.44 / 2.10
17.88 / 4.21
18.54 / 3.62
20.60 / 3.90
15.43 / 19.45
15.84 / 6.22
18.90 / 5.94
22.35 / 1.58
23.43 / 1.99
24.65 / 1.28
23.48 / 1.62
Ours
24.11 / 1.61
24.10 / 0.98
25.20 / 1.55
22.20 / 1.58
23.04 / 1.32
17.92 / 2.79
16.23 / 1.99
21.83 / 1.69
24.95 / 1.44
24.30 / 1.85
24.74 / 1.52
24.66 / 1.60
Table 1. Quantitative comparison of novel view synthesis and 3D reconstruction results across 10 AOIs under multi-view input. Best
results are bolded.
AOI
BER(%)↓/ ACC(%)↑
Mean
JAX 004
JAX 068
JAX 165
JAX 214
JAX 260
OMA 288
OMA 315
S-EO[47]
48.11 / 85.55
22.98 / 92.92
–
/
–
22.41 / 89.54
40.52 / 86.06
–
/
–
–
/
–
–
/
–
FSDNet[22]
41.81 / 83.15
26.67 / 90.90
16.29 / 89.15
23.79 / 86.53
41.16 / 72.73
31.83 / 79.95
29.08 / 91.88
30.09 / 84.90
FDRNet[63]
31.97 / 78.04
20.81 / 91.20
15.09 / 85.92
18.55 / 84.84
32.10 / 82.07
29.42 / 80.50
26.07 / 90.73
24.86 / 84.76
EO-NeRF[46]
25.50 / 86.67
26.55 / 91.91
40.97 / 79.59
38.58 / 83.08
33.80 / 87.59
40.31 / 79.19
49.88 / 88.92
36.51 / 85.28
EOGS[1]
35.68 / 88.19
15.88 / 92.76
17.44 / 89.24
19.80 / 88.88
28.68 / 88.32
51.19 / 72.20
47.81 / 89.16
30.93 / 86.97
Ours
22.43 / 86.54
12.37 / 91.95
10.26 / 91.23
11.84 / 90.72
18.89 / 87.88
24.92 / 85.39
24.09 / 92.19
17.83 / 89.41
Table 2. Quantitative comparison of shadow detection performance across 7 AOIs from DFC2019 under multi-view input. Best results are
shown in bold.
The complete loss function is defined as:
L = λcLc + λnLn + λS1LS1 + λS2LS2 + λS3LS3 (19)
Here, λ are experimentally determined weighting coeffi-
cients. For the DFC2019 dataset [6, 31], we set λc = 10,
λn = 0.5, λS1 = 0.2, λS2 = 0.3, λS3 = 1. For IARPA [5],
we reduce λn to 0.1 for optimal performance. The shadow
prior loss LS3 is activated only under sparse-view condi-
tions.
Training Strategy: We adapt the original 3DGS training
procedure to accommodate ShadowGS’s extended parame-
ter set and optimization objectives. Key modifications in-
clude:
• Disabling opacity reset throughout training
• Activating all loss terms from the first iteration
• Setting total iterations to 5000, with densification halted
after iteration 3000
• Performing densification every 300 iterations during the
densification phase
These adjustments ensure stable convergence given the
increased number of appearance and illumination parame-
ters. Reducing the frequency of densification allows Gaus-
sian primitives to be more fully optimized before further
densification occurs. This strategy also helps prevent the
proliferation of redundant Gaussians, thereby alleviating
computational overhead.
4. Experiment
Datasets: We train and evaluate ShadowGS on datasets
from the 2019 IEEE GRSS Data Fusion Competition
(DFC2019) [6, 31] and the 2016 IARPA Multi-View Stereo
3D Mapping Challenge (IARPA2016) [5]. The DFC2019
dataset covers two urban regions—Jacksonville (JAX)
and Omaha (OMA)—using RGB imagery, whereas the
IARPA2016 dataset includes a single urban area, Buenos
Aires, and provides pan-sharpened imagery.
All images
were captured by the WorldView-3 satellite at a ground
sampling distance of 0.3 m/pixel and are provided with
RPC parameters, acquisition timestamps, and solar angles.
LiDAR-derived Digital Surface Models (DSMs) are avail-
able as reference, with resolutions of 0.5 m for DFC2019
and 0.3 m for IARPA2016. Additionally, using solar angles,
RPCs, and LiDAR DSMs, we generate accurate shadow
masks for DFC2019 following SEO [47] to enable quan-
titative evaluation of shadow disentanglement.
Experimental Setup: We evaluate under two input set-
tings: multi-view and sparse-view.
• Multi-View Input: We use 10 areas of interest (AOIs): 5
from JAX with diverse terrain features, 2 from OMA with
notable seasonal variations, and 3 from IARPA based
on pan-sharpened images.
Each AOI contains 10–40
multi-temporal images with varying viewpoints. Approx-
imately 15% of images are held out as the test set, with
the rest used for training.
• Sparse-View Input: We use 5 AOIs from JAX, each con-
taining 4 images captured under different viewing angles,
times, and illumination. Three images are used as input,
and the remaining one is reserved for testing.
Evaluation Metrics: We adopt the following metrics:
• Novel View Synthesis:
Peak Signal-to-Noise Ratio
(PSNR) between rendered and real images.
• 3D Reconstruction: Height Mean Absolute Error (MAE)
between the reconstructed DSM and the LiDAR reference
6

<!-- page 7 -->
Figure 4. Geometric reconstruction visualization on the JAX 068
dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
DSM.
• Shadow Disentanglement: Balanced Error Rate (BER)
and Accuracy (ACC) between the binarized rendered
shadow map and the reference shadow mask.
Implementation Details: All experiments are conducted
on a Ubuntu 22.04 server with a single NVIDIA RTX 4090
GPU (24 GB). Our implementation builds on the public re-
lease of RadeGS [57]. Training time for each AOI is ap-
proximately 10 minutes. For the classic multi-view stereo
pipeline S2P [10, 12], in the multi-view experiments, we
generated the DSM following the method provided by Sat-
NeRF [45]; under sparse-view conditions, we generated the
DSM by fusing pairwise image matching results.
4.1. Multi-View Experimental Results
Tab. 1 provides a quantitative comparison between Shad-
owGS and state-of-the-art methods under multi-view in-
put. ShadowGS achieves higher DSM accuracy across most
Figure 5. Albedo Decomposition and Shadow Modeling Results.
(a) Input image; (b-c) Albedo and shadow maps decoupled by
ShadowGS; (d-f) Shadow maps under different solar elevation an-
gles; (g-i) Shadow maps under different solar azimuth angles.
AOIs and consistently outperforms existing approaches in
novel view synthesis. On the DFC2019 dataset, ShadowGS
reduces the average height MAE by approximately 0.62 me-
ters and improves the average PSNR by about 2.93 dB com-
pared to the previous best method. On the IARPA dataset, it
attains a similar height MAE to EOGS [1] while increasing
PSNR by roughly 1.18 dB.
For shadow disentanglement, Tab. 2 reports quantitative
results on 7 AOIs from DFC2019. ShadowGS reduces the
balanced error rate by nearly half while also improving de-
tection accuracy.
We also compare against two natural-
scene shadow detection methods—FDRNet [63] and FSD-
Net [22]—as well as the SEO [47] shadow detection net-
work. ShadowGS outperforms all three in shadow detection
performance.
Fig. 4 visualizes geometric reconstruction results. Shad-
owGS reconstructs sharper edges for structural objects such
as buildings and yields smoother surfaces in low-texture re-
gions. The overall geometry aligns more closely with Li-
DAR reference data. Additional reconstruction visualiza-
tions are included in the Supplementary material.
Fig. 5 illustrates ShadowGS’s ability to recover albedo
and model geometry-aware shadows. ShadowGS success-
fully restores albedo information in shadowed regions and
7

<!-- page 8 -->
AOI
PSNR(dB)↑/ MAE(m)↓
Mean
JAX 004
JAX 068
JAX 165
JAX 214
JAX 260
S2P[10]
–
/ 3.14
–
/ 1.58
–
/ 5.47
–
/ 3.42
–
/ 3.20
–
/ 3.36
EO-NeRF[46]
20.83 / 1.51
18.24 / 5.81
12.53 / 9.85
11.93 / 8.40
17.44 / 3.38
16.19 / 5.79
EOGS[1]
22.46 / 2.64
14.02 / 6.63
10.11 / 12.90
7.13 / 17.40
13.33 / 4.98
13.41 / 8.91
Ours
21.91 / 3.24
18.64 / 1.89
18.68 / 3.97
17.90 / 4.98
20.10 / 2.44
19.45 / 3.30
Ours + Shadow
22.30 / 2.08
18.66 / 1.69
19.10 / 3.36
18.25 / 3.94
20.27 / 2.26
19.72 / 2.67
Table 3. Quantitative comparison on 5 JAX AOIs under sparse-view input. Best results are shown in bold. ”Ours + Shadow” indicates
supervision with shadow map from FDRNet[63].
Depth-Normal
Render Equation
Shadow Consistency
MAE(m) ↓
PSNR(dB) ↑
4.00
13.91
✓
2.50
17.07
✓
✓
2.11
23.00
✓
✓
✓
1.41
23.73
Table 4. Ablation study on 5 JAX AOIs. Results are averaged over all 5 AOIs.
renders shadows consistent with scene geometry under
varying sun angles.
4.2. Sparse-View Experimental Results
Tab. 3 compares ShadowGS with existing methods under
sparse-view input.
Without using shadow priors, Shad-
owGS already significantly outperforms EO-NeRF [46] and
EOGS [1] in both height MAE and novel view PSNR across
all AOIs, and slightly surpasses the classic MVS pipeline
S2P [10, 12] in reconstruction quality. When shadow map
priors are incorporated, ShadowGS further improves DSM
accuracy and synthesis performance under sparse views.
4.3. Ablation Studies
We conduct ablation experiments to evaluate the contribu-
tion of each major component in ShadowGS to geometric
reconstruction and novel view synthesis. Tab. 4 summarizes
the results, where each row corresponds to a different model
configuration and columns indicate the activation of the fol-
lowing components: depth–normal consistency constraint,
rendering equation, and shadow consistency constraint.
Experiments are conducted under multi-view settings,
with the last two columns reporting the average height MAE
and PSNR across five JAX AOIs. Results show that adding
the depth–normal constraint reduces MAE by 1.50 meters
and increases PSNR by 3.16 dB. Enabling the rendering
equation further improves PSNR by 5.93 dB and reduces
MAE by 0.39 meters. Incorporating the shadow consistency
constraint brings an additional MAE reduction of 0.70 me-
ters and a PSNR gain of 0.73 dB. In summary, all compo-
nents contribute positively to both reconstruction and ren-
dering quality.
5. Conclusion
We propose ShadowGS, a novel 3DGS-based framework
that decouples geometry, appearance, and illumination
from multi-temporal satellite images—including RGB, pan-
sharpened, and sparse-view inputs.
By introducing a
remote-sensing physics-based rendering equation combined
with efficient ray marching, ShadowGS accurately mod-
els shadow variations across multi-temporal observations,
effectively disentangles illumination and appearance, and
achieves high-quality geometric reconstruction.
A limitation of ShadowGS is that it does not currently
account for content inconsistencies in multi-temporal im-
agery caused by seasonal or land-cover changes, which may
affect performance in dynamically varying scenes. Future
work could incorporate seasonal appearance modeling into
the rendering equation to improve robustness in such sce-
narios.
8

<!-- page 9 -->
References
[1] Luca Savant Aira, Gabriele Facciolo, and Thibaud Ehret.
Gaussian splatting for efficient satellite image photogram-
metry. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 5959–5969, 2025. 1, 2, 3, 5,
6, 7, 8
[2] Nan Bai, Anran Yang, Hao Chen, and Chun Du.
Satgs:
Remote sensing novel view synthesis using multi-temporal
satellite images with appearance-adaptive 3dgs.
Remote
Sensing, 17(9):1609, 2025. 2, 3
[3] Camille Billouard, Dawa Derksen, Emmanuelle Sarrazin,
and Bruno Vallet.
Sat-ngp: Unleashing neural graphics
primitives for fast relightable transient-free 3d reconstruction
from satellite imagery. In IGARSS 2024-2024 IEEE Inter-
national Geoscience and Remote Sensing Symposium, pages
8749–8753. IEEE, 2024. 3, 6
[4] Camille Billouard, Dawa Derksen, Alexandre Constantin,
and Bruno Vallet. Tile and slide: A new framework for scal-
ing nerf from local to global 3d earth observation.
arXiv
preprint arXiv:2507.01631, 2025. 2
[5] Marc Bosch, Zachary Kurtz, Shea Hagstrom, and Myron
Brown. A multiple view stereo benchmark for satellite im-
agery. In 2016 IEEE Applied Imagery Pattern Recognition
Workshop (AIPR), pages 1–9. IEEE, 2016. 2, 6
[6] Marc Bosch, Kevin Foster, Gordon Christie, Sean Wang,
Gregory D Hager, and Myron Brown. Semantic stereo for
incidental satellite images. In 2019 IEEE Winter Conference
on Applications of Computer Vision (WACV), pages 1524–
1532. IEEE, 2019. 2, 6
[7] Xiaochun Cao and Hassan Foroosh. Camera calibration and
light source orientation from solar shadows. Computer Vi-
sion and Image Understanding, 105(1):60–72, 2007. 2
[8] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for ef-
ficient and high-fidelity surface reconstruction. IEEE Trans-
actions on Visualization and Computer Graphics, 2024. 3
[9] Pablo d’Angelo and Georg Kuschk. Dense multi-view stereo
from satellite imagery.
In 2012 IEEE international geo-
science and remote sensing symposium, pages 6944–6947.
IEEE, 2012. 1
[10] Carlo De Franchis, Enric Meinhardt-Llopis, Julien Michel,
Jean-Michel Morel, and Gabriele Facciolo. An automatic
and modular stereo pipeline for pushbroom images. In ISPRS
Annals of the Photogrammetry, Remote Sensing and Spatial
Information Sciences, 2014. 6, 7, 8
[11] Dawa Derksen and Dario Izzo.
Shadow neural radiance
fields for multi-view satellite photogrammetry. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 1152–1161, 2021. 1, 2
[12] Gabriele Facciolo, Carlo De Franchis, and Enric Meinhardt-
Llopis. Automatic 3d reconstruction from multi-date satellite
images. In Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition Workshops, pages 57–
66, 2017. 7, 8
[13] Michael Gableman and Avinash Kak. Incorporating season
and solar specificity into renderings made by a nerf archi-
tecture using satellite images. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 46(6):4348–4365, 2024.
2
[14] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Re-
alistic point cloud relighting with brdf decomposition and
ray tracing. In European Conference on Computer Vision,
pages 73–89. Springer, 2024. 3, 4
[15] Ke Gong and Dieter Fritsch. Dsm generation from high reso-
lution multi-view stereo satellite imagery. Photogrammetric
Engineering & Remote Sensing, 85(5):379–387, 2019. 1
[16] Maciej Gryka, Michael Terry, and Gabriel J Brostow. Learn-
ing to remove soft shadows. ACM Transactions on Graphics
(TOG), 34(5):1–15, 2015. 2
[17] Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao, and Li
Zhang. Irgs: Inter-reflective gaussian splatting with 2d gaus-
sian ray tracing. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 10943–10952, 2025.
3, 4
[18] Lanqing Guo, Siyu Huang, Ding Liu, Hao Cheng, and Bihan
Wen. Shadowformer: Global context helps shadow removal.
In Proceedings of the AAAI conference on artificial intelli-
gence, pages 710–718, 2023. 2
[19] Ruiqi Guo, Qieyun Dai, and Derek Hoiem. Single-image
shadow detection and removal using paired regions. In CVPR
2011, pages 2033–2040. IEEE, 2011. 2
[20] Heiko Hirschmuller. Accurate and efficient stereo processing
by semi-global matching and mutual information. In 2005
IEEE computer society conference on computer vision and
pattern recognition (CVPR’05), pages 807–814. IEEE, 2005.
1, 2
[21] Heiko Hirschmuller. Stereo processing by semiglobal match-
ing and mutual information. IEEE Transactions on pattern
analysis and machine intelligence, 30(2):328–341, 2007. 1
[22] Xiaowei Hu, Tianyu Wang, Chi-Wing Fu, Yitong Jiang,
Qiong Wang, and Pheng-Ann Heng. Revisiting shadow de-
tection: A new benchmark dataset for complex world. IEEE
Transactions on Image Processing, 30:1925–1934, 2021. 2,
6, 7
[23] Xiaowei Hu, Zhenghao Xing, Tianyu Wang, Chi-Wing Fu,
and Pheng-Ann Heng.
Unveiling deep shadows: A sur-
vey and benchmark on image and video shadow detection,
removal, and generation in the deep learning era.
arXiv
preprint arXiv:2409.02108, 2024. 2
[24] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 3, 4
[25] Xuejun Huang, Xinyi Liu, Yi Wan, Zhi Zheng, Bin
Zhang,
Mingtao Xiong,
Yingying Pei,
and Yongjun
Zhang.
Skysplat:
Generalizable 3d gaussian splatting
from multi-temporal sparse satellite images. arXiv preprint
arXiv:2508.09479, 2025. 3
[26] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xi-
aoxiao Long, Wenping Wang, and Yuexin Ma. Gaussian-
shader: 3d gaussian splatting with shading functions for re-
flective surfaces. In Proceedings of the IEEE/CVF Confer-
9

<!-- page 10 -->
ence on Computer Vision and Pattern Recognition, pages
5322–5332, 2024. 3, 4
[27] Nada Kadhim and Monjur Mourshed. A shadow-overlapping
algorithm for estimating building heights from vhr satellite
images. IEEE Geoscience and remote sensing letters, 15(1):
8–12, 2017. 2
[28] Kevin Karsch, Varsha Hedau, David Forsyth, and Derek
Hoiem. Rendering synthetic objects into legacy photographs.
ACM Transactions on graphics (TOG), 30(6):1–12, 2011. 2
[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[30] Jean-Franc¸ois Lalonde, Alexei A Efros, and Srinivasa G
Narasimhan. Estimating the natural illumination conditions
from a single outdoor image. International Journal of Com-
puter Vision, 98(2):123–145, 2012. 2
[31] Bertrand Le Saux, Naoto Yokoya, Ronny Hansch, Myron
Brown, and Greg Hager. 2019 data fusion contest [technical
committees]. IEEE Geoscience and Remote Sensing Maga-
zine, 7(1):103–105, 2019. 2, 6
[32] Jie-Ying Lee, Yi-Ruei Liu, Shr-Ruei Tsai, Wei-Cheng
Chang,
Chung-Ho Wu,
Jiewen Chan,
Zhenjun Zhao,
Chieh Hubert Lin, and Yu-Lun Liu. Skyfall-gs: Synthesiz-
ing immersive 3d urban scenes from satellite imagery. arXiv
preprint arXiv:2510.15869, 2025. 3
[33] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normaliza-
tion. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 20775–20785,
2024. 3, 5
[34] Zhuohao Li, Guoyang Xie, Guannan Jiang, and Zhichao Lu.
Shadowmaskformer: Mask augmented patch embedding for
shadow removal.
IEEE Transactions on Artificial Intelli-
gence, 2025. 2
[35] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia.
Gs-ir: 3d gaussian splatting for inverse rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21644–21653, 2024. 3, 4
[36] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng,
and Kui Jia.
Analytic-splatting: Anti-aliased 3d gaussian
splatting via analytic integration. In European conference on
computer vision, pages 281–297. Springer, 2024. 3
[37] Gregoris Liasis and Stavros Stavrou. Satellite images analy-
sis for shadow detection and building height estimation. IS-
PRS Journal of Photogrammetry and Remote Sensing, 119:
437–450, 2016. 2
[38] Dongyang Liu, Junping Zhang, Kun Liu, and Ye Zhang.
Aerial remote sensing image cascaded road detection net-
work based on edge sensing module and attention module.
IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2022.
2
[39] Xinhang Liu, Jiaben Chen, Shiu-Hong Kao, Yu-Wing Tai,
and Chi-Keung Tang.
Deceptive-nerf/3dgs:
Diffusion-
generated pseudo-observations for high-quality sparse-view
reconstruction. In European Conference on Computer Vi-
sion, pages 337–355. Springer, 2024. 5
[40] Haitao Luo, Jinming Zhang, Xiongfei Liu, Lili Zhang, and
Junyi Liu. Large-scale 3d reconstruction from multi-view
imagery: A comprehensive review. Remote Sensing, 16(5):
773, 2024. 1
[41] Shuang Luo, Huifang Li, and Huanfeng Shen. Deeply su-
pervised convolutional neural network for shadow detection
based on a novel aerial shadow imagery dataset. ISPRS Jour-
nal of Photogrammetry and remote sensing, 167:443–457,
2020. 2
[42] Shuang Luo, Huifang Li, Ruzhao Zhu, Yuting Gong, and
Huanfeng Shen. Espfnet: An edge-aware spatial pyramid
fusion network for salient shadow detection in aerial re-
mote sensing images. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, 14:4633–
4646, 2021. 2
[43] Masoud Mahdianpari, Jean Elizabeth Granger, Fariba Mo-
hammadimanesh, Sherry Warren, Thomas Puestow, Bahram
Salehi, and Brian Brisco. Smart solutions for smart cities:
Urban wetland mapping using very-high resolution satellite
imagery and airborne lidar data in the city of st. john’s, nl,
canada. Journal of environmental management, 280:111676,
2021. 1
[44] Alexander Mai, Peter Hedman, George Kopanas, Dor
Verbin, David Futschik, Qiangeng Xu, Falko Kuester,
Jonathan T Barron, and Yinda Zhang.
Ever: Exact volu-
metric ellipsoid rendering for real-time view synthesis. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 4930–4939, 2025. 3
[45] Roger Mar´ı, Gabriele Facciolo, and Thibaud Ehret. Sat-nerf:
Learning multi-view satellite photogrammetry with transient
objects and shadow modeling using rpc cameras. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 1311–1321, 2022. 1, 2, 4, 7
[46] Roger Mar´ı, Gabriele Facciolo, and Thibaud Ehret. Multi-
date earth observation nerf: The detail is in the shadows. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 2035–2045, 2023. 1, 2,
6, 8
[47] El´ıas Masquil, Roger Mar´ı, Thibaud Ehret, Enric Meinhardt-
Llopis, Pablo Mus´e, and Gabriele Facciolo. S-eo: A large-
scale dataset for geometry-aware shadow detection in remote
sensing applications. In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 2383–2393,
2025. 2, 5, 6, 7
[48] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[49] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Ric-
cardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja
Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray trac-
ing: Fast tracing of particle scenes. ACM Transactions on
Graphics (TOG), 43(6):1–19, 2024. 3, 4
[50] Muyao Niu, Tong Chen, Yifan Zhan, Zhuoxiao Li, Xiang Ji,
and Yinqiang Zheng. Rs-nerf: Neural radiance fields from
rolling shutter images.
In European Conference on Com-
puter Vision, pages 163–180. Springer, 2024. 3
10

<!-- page 11 -->
[51] Takahiro Okabe, Imari Sato, and Yoichi Sato.
Attached
shadow coding: Estimating surface normals from shadows
under unknown reflectance and lighting conditions. In 2009
IEEE 12th International Conference on Computer Vision,
pages 1693–1700. IEEE, 2009. 2
[52] Yingjie Qu and Fei Deng. Sat-mesh: Learning neural im-
plicit surfaces for multi-view satellite reconstruction. Remote
Sensing, 15(17):4297, 2023. 2
[53] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 3
[54] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay,
Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-
time 360 {\deg} sparse view synthesis using gaussian splat-
ting. arXiv preprint arXiv:2312.00206, 2023. 3, 5
[55] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 3
[56] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (ToG),
43(6):1–13, 2024. 3
[57] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang,
Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in
gaussian splatting. arXiv preprint arXiv:2406.01467, 2024.
3, 4, 7
[58] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu,
Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian
splatting via co-regularization. In European Conference on
Computer Vision, pages 335–352. Springer, 2024. 3, 5
[59] Kai Zhang, Noah Snavely, and Jin Sun. Leveraging vision re-
construction pipelines for satellite imagery. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion Workshops, pages 0–0, 2019. 4
[60] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen
Koltun. Nerf++: Analyzing and improving neural radiance
fields. arXiv preprint arXiv:2010.07492, 2020. 4
[61] Lulin Zhang and Ewelina Rupnik.
Sparsesat-nerf: Dense
depth supervised neural radiance fields for sparse satellite
images. arXiv preprint arXiv:2309.00277, 2023. 2
[62] Tongtong Zhang, Yu Zhou, Yuanxiang Li, and Xian Wei.
Satensorf: Fast satellite tensorial radiance field for multidate
satellite imagery of large size. IEEE Transactions on Geo-
science and Remote Sensing, 62:1–15, 2024. 3
[63] Lei Zhu, Ke Xu, Zhanghan Ke, and Rynson WH Lau. Miti-
gating intensity bias in shadow detection via feature decom-
position and reweighting. In Proceedings of the IEEE/CVF
international conference on computer vision, pages 4702–
4711, 2021. 2, 5, 6, 7, 8
[64] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross. Ewa splatting. IEEE Transactions on Visual-
ization and Computer Graphics, 8(3):223–238, 2002. 3
11

<!-- page 12 -->
ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery
Supplementary Material
A. Additional Experimental Results under
Multi-view Input
A.1. Geometric Reconstruction Visualization
We provide additional 3D reconstruction visualizations un-
der multi-view input in Figures 6 to 14.
Figure 6. Geometric reconstruction visualization on the JAX 004
dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
Figure 7. Geometric reconstruction visualization on the JAX 165
dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
12

<!-- page 13 -->
Figure 8. Geometric reconstruction visualization on the JAX 214
dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
Figure 9. Geometric reconstruction visualization on the JAX 260
dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
13

<!-- page 14 -->
Figure 10. Geometric reconstruction visualization on the OMA
288 dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
Figure 11. Geometric reconstruction visualization on the OMA
315 dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
14

<!-- page 15 -->
Figure 12. Geometric reconstruction visualization on the IARPA
001 dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
Figure 13. Geometric reconstruction visualization on the IARPA
002 dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
15

<!-- page 16 -->
Figure 14. Geometric reconstruction visualization on the IARPA
003 dataset. The fourth column shows the error map between each
method’s reconstructed DSM and the ground truth (GT), where
red indicates overestimation and blue indicates underestimation of
height values.
A.2. Novel View Synthesis Results
We provide comparative results of novel view synthesis in
Figures 15-17.
Figure 15. Visualization of novel view synthesis on 5 AOIs in the
JAX region.
Figure 16. Visualization of novel view synthesis on 2 AOIs in the
OMA region.
16

<!-- page 17 -->
Figure 17. Visualization of novel view synthesis on 3 AOIs in the
IARPA region.
A.3. Shadow and Albedo Decomposition Results
We provide visualization results of shadow and albedo de-
composition in Figures 18-22.
Figure 18. Shadow decomposition and albedo visualization on the
JAX 068 dataset.
Figure 19. Shadow decomposition and albedo visualization on the
JAX 165 dataset.
Figure 20. Shadow decomposition and albedo visualization on the
JAX 214 dataset.
17

<!-- page 18 -->
Figure 21. Shadow decomposition and albedo visualization on the
OMA 315 dataset.
Figure 22. Shadow decomposition and albedo visualization on the
IARPA 003 dataset.
B. Other Experiments
We evaluated the effect of incorporating FDRNet shadow
map priors on reconstruction accuracy across different num-
bers of input views. As shown in Tab. 5, the priors con-
sistently improve performance when using limited views
(≤7). However, their effectiveness diminishes with more
input views, eventually leading to performance degrada-
tion. This occurs because the inevitable false detections in
shadow priors and inconsistencies among multiple priors in-
troduce noise that adversely affects the reconstruction.
MAE (m)
3 views
5 views
7 views
9 views
All views
w/o Shadow mask
3.30
2.40
1.96
1.64
1.41
Shadow mask
2.67
2.28
1.89
1.70
1.47
Table 5. Impact of FDRNet shadow map priors on mean height
MAE(m) across 5 AOIs in the JAX region under varying numbers
of input views.
18
