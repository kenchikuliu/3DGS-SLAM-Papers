<!-- page 1 -->
Material-informed Gaussian Splatting for 3D World
Reconstruction in a Digital Twin
Andy Huynh1,2∗, Jo˜ao Malheiro Silva1∗, Holger Caesar2, and Tong Duy Son1
Abstract—3D reconstruction for Digital Twins often relies on
LiDAR-based methods, which provide accurate geometry but
lack the semantics and textures naturally captured by cameras.
Traditional LiDAR-camera fusion approaches require complex
calibration and still struggle with certain materials like glass,
which are visible in images but poorly represented in point
clouds. We propose a camera-only pipeline that reconstructs
scenes using 3D Gaussian Splatting from multi-view images,
extracts semantic material masks via vision models, converts
Gaussian representations to mesh surfaces with projected ma-
terial labels, and assigns physics-based material properties for
accurate sensor simulation in modern graphics engines and simu-
lators. This approach combines photorealistic reconstruction with
physics-based material assignment, providing sensor simulation
fidelity comparable to LiDAR-camera fusion while eliminating
hardware complexity and calibration requirements. We validate
our camera-only method using an internal dataset from an
instrumented test vehicle, leveraging LiDAR as ground truth for
reflectivity validation alongside image similarity metrics.
Index Terms—3D Reconstruction, Computer Vision, Digital
Twin, Gaussian Splatting, Material Segmentation, Physics-Based
Rendering, Sensor Simulation
I. INTRODUCTION
Digital Twins advance sensor technologies by enabling safe
validation of high-risk scenarios and precise environmental
control prior to real-world deployment. Modern Advanced
Driver Assistance Systems (ADAS) and Artificial Intelligence
(AI) systems rely on sophisticated algorithms and complex
multimodal sensor setups, often including LiDAR sensors and
cameras. Achieving accurate translation of real-world sce-
narios to simulation—where virtual sensors exhibit identical
behavior to their physical counterparts—requires 3D world
reconstruction that captures spatial and geometric features to
ensure realistic simulation and synthetic sensor data consistent
with reality.
Previous LiDAR-based approaches [36, 17] achieve accu-
rate geometry but lack camera-captured texture information.
Moreover, LiDAR struggles with transparent and reflective
∗Equal contribution.
1Siemens Digital Industries Software, 3001 Leuven, Belgium. Email:
{silva.joao, son.tong}@siemens.com
2Dept. of Cognitive Robotics, Delft University of Technology, 2628 CD
Delft, The Netherlands. Email: {a.huynh, H.Caesar}@tudelft.nl
Funded by the European Union. Views and opinions expressed are however
those of the author(s) only and do not necessarily reflect those of the European
Union or the European Health and Digital Executive Agency (HADEA).
Neither the European Union nor the granting authority can be held responsible
for them. This work is part of the ROBUSTIFAI (grant agreement No.
101212818) and SYNERGIES (grant agreement No. 101146542) projects,
both funded by Horizon Europe. The work also benefited from VLAIO-funded
projects SATISFY.AI and BECAREFUL, which provided the instrumented test
vehicle.
materials like glass and metal, readily captured in camera
images. While incorporating material properties into LiDAR-
based reconstruction has been shown to improve sensor sim-
ulation accuracy [19], such approaches still require LiDAR
hardware and do not address texture acquisition. Combining
LiDAR with cameras to capture both geometry and appearance
requires complex sensor synchronization and calibration pro-
cedures, yet still produces limited texture quality under sparse-
view conditions, as traditional reconstruction methods struggle
to generate photorealistic results.
Recent advancements in novel view synthesis have in-
troduced 3D Gaussian Splatting [12], an efficient technique
providing fast, differentiable rendering and high-quality re-
constructions from sparse input views. Recent works like
MILo [8] extract geometric surfaces from Gaussian Splatting
using only multi-view images, enabling hybrid representations
combining photorealistic visualization with physics-based sim-
ulation in graphics engines. These camera-only methods pro-
vide an alternative to LiDAR hardware, avoiding intricate
sensor synchronization and demanding calibration procedures
while delivering superior texture quality.
Building on these observations, we propose a camera-only
pipeline that combines photorealistic Gaussian Splatting with
physics-based material assignment (Fig. 1). This approach
enables accurate sensor simulation by integrating semantic
material information into visually realistic reconstructions.
Our main contributions are:
• (1) An automated method for projecting 2D semantic
material masks onto 3D mesh surfaces through Gaus-
sian Splatting-based reconstruction, enabling accurate
physics-based LiDAR reflectivity simulation.
• (2) A modular, camera-only pipeline integrating Gaussian
Splatting for photorealistic geometry reconstruction with
automated mesh-based material assignment.
• (3) Comprehensive evaluation demonstrating sensor sim-
ulation accuracy comparable to LiDAR-camera fusion,
validated through LiDAR reflectivity and rendering qual-
ity analysis.
While our pipeline requires only camera input, we validate
it using LiDAR ground truth from an instrumented test vehicle.
II. RELATED WORK
A. Novel View Synthesis
Traditional 3D reconstruction methods use explicit repre-
sentations. LiDAR-based methods [11] excel at geometric
accuracy but struggle with fine visual details. Camera-based
arXiv:2511.20348v3  [cs.CV]  3 Feb 2026

<!-- page 2 -->
Internal Dataset
Fig. 1. Overview of our camera-only reconstruction pipeline. From RGB images, we: (A) extract semantic material masks, (B) reconstruct the 3D scene, (C)
project material labels onto mesh surfaces, (D) assign physics-based materials, and (E) validate through sensor simulation.
alternatives [23] offer richer visual information but produce
noisier geometry and face challenges with transparent and
reflective surfaces [22]. Neural Radiance Fields (NeRF) [18]
enabled photorealistic novel view synthesis from camera-only
input but suffers from high computational overhead and lacks
discrete surfaces needed for graphics engines.
3D Gaussian Splatting [12] addressed these limitations
through an explicit point-based representation enabling real-
time rendering with photorealistic quality. Subsequent re-
finements have improved geometric accuracy, transparency
handling, and shading [10, 38, 30]. However, these methods
remain primarily oriented toward small, object-centric captures
and struggle with large-scale outdoor scenes.
B. Large-Scale Scene Reconstruction
To scale Gaussian Splatting to large-scale outdoor envi-
ronments, recent methods have adopted scene partitioning
strategies [25], dividing scenes into spatially consistent blocks
for parallel reconstruction [16, 13]. Many employ Level-of-
Detail (LoD) techniques, allocating higher resolution to nearby
regions to improve rendering speed. Alternative strategies
include temporal segmentation [5] and distance-based decom-
position [24]. However, these methods often require man-
ual parameter tuning and lack robust surface representations
needed for simulation engines.
C. Surface Extraction from Gaussian Primitives
While Gaussian Splatting enables high-quality rendering,
simulation engines require explicit mesh representations.
Methods such as [7] use Poisson reconstruction, while Signed
Distance Field (SDF)-based approaches [4] require dense
multi-view data. Recent work [8] integrates mesh extraction
directly into training via Delaunay triangulation, enabling
joint optimization of appearance and geometry. However,
these methods are primarily designed for controlled, object-
centric scenarios and lack robustness for large-scale outdoor
environments.
D. Material Extraction
Beyond extracting accurate geometry, material properties
are essential for realistic simulation. Image-based methods
classify materials through semantic segmentation [2]. Multi-
modal approaches [15] and large-scale datasets [26] have
improved accuracy. While LiDAR-based methods [19, 28]
measure physical properties directly, camera-only approaches
capture materials that LiDAR struggles with, such as transpar-
ent and reflective surfaces. However, transferring 2D material
segmentation to 3D reconstructed surfaces remains challenging
for camera-only Gaussian Splatting pipelines, where material
labels must be accurately projected from multi-view obser-
vations onto extracted mesh geometry. Our work addresses
this gap by introducing an automated projection method that
enables physics-based sensor simulation.
E. Physics-Based Materials for Rendering
Once material labels are assigned to 3D surfaces, they must
be represented in a format compatible with physics-based
rendering engines. Modern graphics engines simulate surface
interactions through the Bidirectional Reflectance Distribution
Function (BRDF) [14]. Recent work integrates material prop-
erties directly into 3D Gaussians [34, 33] focusing on ren-
dering, while mesh-compatible approaches [31] enable seam-

<!-- page 3 -->
Fig. 2. Shape-aware material refinement. From left to right: input RGB image, RMSNet predictions, FastSAM object boundaries, and final result after majority
voting. The combination produces consistent material labels with sharp edges. Example from our internal dataset.
less integration with standard simulation engines. However,
bridging the gap between semantic material labels from 2D
segmentation and physically accurate material parameters for
physics-based sensor simulation in Digital Twin environments
remains an open challenge.
III. METHOD
Our pipeline reconstructs large-scale outdoor scenes from
multi-view RGB images and assigns physics-based materials
for accurate sensor simulation. The approach is decoupled,
separating geometric reconstruction from material assignment,
and employs a hybrid representation, combining photorealistic
Gaussian Splatting with explicit mesh geometry. As illustrated
in Fig. 1, the pipeline consists of five stages: monocular mate-
rial extraction (Sec. III-A), Gaussian Splatting reconstruction
(Sec. III-B), per-pixel material projection from 2D to 3D
(Sec. III-C), physics-based material assignment (Sec. III-D),
and simulation validation (Sec. III-E).
A. Monocular Material Extraction
We extract per-pixel material labels from RGB images using
a two-stage approach: texture-based material segmentation
followed by shape-aware refinement. For the first stage, we
adopt RMSNet [2] due to its strong performance on au-
tonomous driving datasets and its ability to operate on RGB
images alone, without requiring LiDAR or infrared sensors.
It distinguishes classes such as asphalt, concrete, glass, and
metal.
Material regions often exhibit fragmented distributions lack-
ing prominent shape cues. We employ FastSAM [39] to iden-
tify instances, producing pixel-wise boundaries that we post-
process to remove overlaps before overlaying with RMSNet
predictions.
For each region, we apply majority voting: counting ma-
terial class labels within its boundary and assigning the most
frequent label to all enclosed pixels. This strategy ensures con-
sistent material assignment while preserving sharp boundaries,
as illustrated in Fig. 2. The result is a set of n shape-aware
material masks, where each pixel inherits its class according
to the FastSAM-derived segmentation.
B. Large-Scale Gaussian Splatting
Our 3D reconstruction combines photorealistic rendering
with explicit geometry through a hybrid approach. We first ap-
ply COLMAP [23] to generate a sparse point cloud from multi-
view RGB images, providing GPS-derived camera poses, pre-
calibrated intrinsics, and extrinsics as priors. We mask the
moving ego-vehicle in all input images to avoid artifacts.
From the COLMAP output, we train two complementary
models. For photorealistic visualization, we use H3DGS [13],
selected for its superior visual quality on autonomous driving
datasets over alternatives like CityGaussianV2 [16]. H3DGS
constructs a global scaffold and subdivides the scene into
chunks. However, H3DGS’s hierarchical LoD representation
uses a proprietary .hier format, and its anti-aliasing mech-
anism [35] is not supported by standard 3DGS rendering
tools that expect the conventional .ply format. To maximize
compatibility with existing 3DGS viewers and downstream
processing tools, we therefore disable these features and
perform a custom merging of the chunks.
For mesh geometry and semantic labeling, we employ
MiLO [8], which provides geometrically accurate mesh ex-
traction required for physics-based simulation. Trained on the
same COLMAP sparse point cloud, MiLO produces both a 3D
Gaussian model and an explicit mesh. The MiLO Gaussian
model initializes our semantic Gaussian Splatting for material
projection (Sec. III-C), while the mesh provides the geometric
basis for physics-based simulation (Sec. III-E). Both H3DGS
and MiLO are trained with monocular depth supervision [32]
following their respective default configurations, and both
reconstructions share the same coordinate system, ensuring
alignment between appearance and geometry.
C. Per-Pixel Material Projection
This module projects the 2D material masks from Sec. III-A
onto the MiLO Gaussian representation, then transfers the la-
bels to the mesh. Unlike previous pipelines relying on camera-

<!-- page 4 -->
Fig. 3.
Qualitative comparison of novel view rendering. From left to right: ground truth RGB, our adapted H3DGS model, 3DGS, CityGaussianV2, and
3DGUT. Our model achieves competitive visual quality comparable to state-of-the-art baselines.
to-LiDAR transformations, our Gaussian model is natively
aligned in world coordinates. We employ differentiable rasteri-
zation to render Gaussians to 2D, maintaining correspondence
between 3D Gaussian positions and 2D pixel coordinates.
When multiple Gaussians project to the same pixel, depth
sorting during alpha blending assigns the label of the closest
Gaussian.
Inconsistencies across views, such as static objects receiving
different labels, are resolved using SegAnyGS [3], which
enforces consistent assignment of our material masks to the
Gaussians. We apply majority voting across overlaid masks
to ensure label consistency. Each material-labeled Gaussian is
assigned to its nearest mesh triangle using K-Nearest Neigh-
bors (KNN), and the triangle inherits the Gaussian’s label,
producing a per-triangle material mesh as the final output.
D. Physics-Based Material Assignment
We assign physics-based material properties to the mesh
using the Principled BSDF shader following the Disney BSDF
standard [1], ensuring compatibility with graphics engines
such as Blender, Unreal Engine, and simulation platforms like
Simcenter Prescan. The Principled BSDF is parameterized by
physically meaningful properties including base color, specu-
lar, metallic factor, roughness, opacity, and surface normals.
From
the
20
material
classes
detected
by
RMSNet
(Sec. III-A), we selected a subset commonly encountered
in urban driving environments: glass, brick and ceramics,
concrete, asphalt, vegetation (e.g., grass and leaves), metals,
plastics, gravel, tree trunks, and rubber. For each category,
PBR textures were sourced from the Matsynth dataset [27],
which provides over 4,000 physically-based materials across
14 categories, and matched to Prescan’s laboratory-tested
material database to ensure validated reflectivity properties for
LiDAR simulation. Within each category, textures were bal-
anced across key parameters (base color, roughness, metallic
factor, clear coat). For each material class from Sec. III-A, we
apply the selected PBR textures to the corresponding mesh
triangles, producing a fully textured mesh directly usable by
standard graphics engines and physics-based sensor simulation
frameworks.
E. Simulation Validation
To validate our material-informed reconstruction pipeline,
we integrate the reconstructed scenes into Simcenter Prescan
to recreate realistic traffic scenarios with physics-based sen-
sor simulation. The material-labeled mesh from Sec. III-D
is imported into Prescan, where laboratory-measured PBR
material properties are assigned to each triangle based on its
material label. This enables accurate interaction with physics-
based sensors, particularly LiDAR, where surface reflectivity
depends on material composition.
We replicate the original sensor configuration from our
instrumented test vehicle, including LiDAR specifications and
mounting positions. The ego-vehicle trajectory is extracted and
imported via the Simcenter Prescan MATLAB API, ensuring
the virtual sensor follows the exact path and poses as during
real-world data capture. The simulation generates synthetic
LiDAR point clouds with material-specific reflectivity values
for each point.
We validate our camera-only reconstruction by comparing
these simulated LiDAR responses against real-world LiDAR
measurements from the instrumented test vehicle. This demon-
strates that our pipeline produces 3D models suitable for
physics-based sensor simulation, providing a practical alter-
native to traditional LiDAR-based reconstruction workflows
while achieving comparable sensor simulation accuracy.
IV. EXPERIMENTS
We evaluate our camera-only reconstruction pipeline using
our internal dataset. To assess whether camera-only recon-
struction achieves sensor simulation accuracy comparable to
LiDAR-based methods, we establish a LiDAR-camera fusion
baseline that combines LiDAR-based mesh reconstruction
(NKSR [11]) with camera-derived material labels. This base-
line provides geometric accuracy through direct depth mea-
surements while incorporating semantic material information
from camera images.
We conduct two complementary evaluations with different
baselines: (1) visual quality, where we compare our Gaussian
Splatting reconstruction against state-of-the-art camera-only

<!-- page 5 -->
TABLE I
IMAGE SIMILARITY METRICS (PSNR ↑, SSIM ↑, LPIPS ↓) EVALUATED ON NOVEL VIEWS FOR ALL BASELINES.
Adapted H3DGS (Ours)
3DGS
CityGaussianV2
3DGUT
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Scene 1
16.88
0.299
0.451
19.13
0.476
0.438
20.36
0.486
0.480
19.97
0.465
0.517
Scene 2
17.86
0.358
0.432
19.81
0.525
0.474
20.79
0.526
0.522
21.16
0.505
0.512
Scene 3
18.35
0.375
0.437
19.48
0.515
0.492
20.42
0.524
0.540
21.10
0.502
0.557
Scene 4
19.03
0.504
0.439
20.88
0.659
0.450
21.15
0.676
0.483
21.85
0.666
0.526
Scene 5
19.21
0.503
0.427
21.04
0.650
0.446
21.63
0.690
0.499
22.04
0.662
0.520
Average
18.27
0.41
0.44
20.07
0.57
0.46
20.87
0.58
0.50
21.22
0.56
0.53
Best performance highlighted in red, second-best in yellow.
methods (3DGS [12], 3DGUT [30], CityGaussianV2 [16])
using image similarity metrics (PSNR, SSIM, LPIPS), and (2)
sensor simulation accuracy, where we compare our camera-
only approach against the LiDAR-camera fusion baseline
by evaluating synthetic LiDAR reflectivity against real-world
ground truth.
A. Dataset
We collected data using an instrumented test vehicle
equipped with an Ouster OS1-128 LiDAR [20] (128 channels,
20 Hz, 100 Hz IMU), six surround-view cameras with Sony
IMX490 sensors (2880×1860 resolution, one rear with fisheye
lens for 360° coverage), and a Septentrio AsteRx GNSS/RTK
system (20 Hz). The LiDAR provides hardware triggers to
synchronize all cameras at 20 Hz. Five static road scenes were
captured near our company facilities in Leuven, Belgium, each
approximately eight seconds at 20-30 km/h. Data collection on
company-controlled roads avoided GDPR compliance issues
and ensured controlled validation conditions.
We focus on static scene reconstruction as this aligns
with Digital Twin simulation workflows. Modern simulation
platforms (e.g., Simcenter Prescan, CARLA) provide realistic
dynamic actor models with controllable behaviors, making the
challenge the creation of photorealistic static environments
(roads, buildings, vegetation, infrastructure). Our method au-
tomates this process, providing realistic backgrounds for simu-
lation scenarios. Dynamic actors in our recordings are masked
out during reconstruction. While our detection pipeline enables
trajectory replication for re-insertion as simulation assets, this
falls outside the scope of this paper.
Scenes primarily consist of company parking lots and
nearby access roads, capturing typical automotive testing en-
vironments with diverse material challenges including asphalt,
concrete, glass facades, metallic surfaces, extensive vegetation
(trees, hedges, grass), road markings, and signage. Recordings
under varying weather conditions (clear to partly cloudy) cap-
ture natural lighting variations. We reserve 50 novel viewpoints
per scene for evaluation.
B. Baseline
Visual quality: We compare our Gaussian Splatting recon-
struction against three state-of-the-art camera-only methods:
3DGS [12], 3DGUT [30], and CityGaussianV2 [16]. All
methods are trained on the same five scenes and evaluated
on 50 uniformly sampled novel viewpoints per scene using
PSNR, SSIM, and LPIPS.
Sensor simulation accuracy: We compare our camera-only
reconstruction against a LiDAR-camera fusion baseline by
simulating both environments in Simcenter Prescan with iden-
tical PBR material properties and ego-vehicle trajectories.
The LiDAR-based baseline uses NKSR [11] for mesh recon-
struction from LiDAR point clouds, combined with the same
camera-derived material labels as our camera-only approach.
This controlled setup minimizes external variables, allowing
direct comparison of geometric reconstruction quality and its
impact on reflectivity accuracy.
C. Implementation Details
Scene reconstruction: We train all Gaussian Splatting meth-
ods using their default hyperparameters as specified in their
respective repositories. For both H3DGS and MiLO, we
enable monocular depth supervision using Depth Anything
V2 [32]. Material assignment follows the pipeline described
in Sec. III-A–III-D, projecting semantic labels onto the re-
constructed geometry and mapping them to PBR material
properties.
LiDAR simulation: We configure the Simcenter Prescan Li-
DAR sensor according to the specifications of the Ouster OS1-
128 [20] used in our dataset. The physics-based sensor returns
both Cartesian coordinates of detected points and a power
value representing reflected signal strength. We normalize this
power output for range and incidence angle to obtain accurate
surface reflectivity values, following established calibration
procedures [9].
Hardware: All experiments were conducted on a Dell Preci-
sion 7680 equipped with 64GB RAM and an NVIDIA RTX
A4000 ADA Laptop GPU. Sensor validation was performed
using Simcenter Prescan 2411, with scenario control via the
Prescan MATLAB API (R2023b).
D. Metrics
We employ complementary metrics for visual quality and
sensor simulation accuracy:

<!-- page 6 -->
Ground Truth
LiDAR-camera fusion
Camera-only
30
40
50
Qualitative evaluation of reflectivity synthetic lidar with respect to GT
0
10
20
w
.
p
d
f
-
x
c
h
a
n
g
e
.
c
w
.
p
d
f
-
x
c
h
a
n
g
e
.
c
Fig. 4.
From left to right: real-world LiDAR ground truth from our instrumented vehicle, LiDAR-camera fusion baseline simulated in Prescan, and our
camera-only reconstruction simulated in Prescan.
Visual quality: PSNR [6] (pixel-level fidelity), SSIM [29]
(structural similarity), and LPIPS [37] (perceptual quality).
Higher is better for PSNR and SSIM; lower is better for LPIPS.
LiDAR simulation accuracy: Mean Absolute Error (MAE)
and median error quantify reflectivity prediction error between
synthetic and real-world measurements, with median providing
robustness against outliers.
E. Visual Quality Evaluation
We evaluate our adapted H3DGS model against three state-
of-the-art Gaussian Splatting methods by comparing novel
view renderings across all five scenes.
Qualitative analysis. Figure 3 demonstrates that our adapted
model achieves competitive visual quality with minimal devi-
ation from ground truth. The model effectively fills masked
ego-vehicle regions and produces cleaner reconstructions by
filtering sensor noise in the ground truth. However, limitations
emerge when reconstructing highly reflective and transparent
surfaces, occasionally resulting in misalignments with ground
truth. Rendering quality is highest for front-facing cameras,
while side cameras exhibit reduced quality due to motion
blur from the vehicle’s movement. Scenes with dense object
distributions and occlusions also reduce reconstruction quality.
Quantitative analysis. Table I presents PSNR, SSIM, and
LPIPS metrics across all methods and scenes. Our adapted
H3DGS model is used solely for photorealistic visualization,
while MiLO provides geometry for simulation (Sec. III-B).
The model achieves lower PSNR (18.27 vs. 20.07–21.22)
and SSIM (0.41 vs. 0.56–0.58) compared to baselines, but
the best LPIPS score (0.44 vs. 0.46–0.53), outperforming all
methods on four out of five scenes. Since LPIPS correlates
strongly with human perceptual similarity [37], this suggests
our reconstruction maintains perceptual quality despite lower
pixel-level accuracy.
The degradation stems from disabling H3DGS’s hierarchical
optimization and anti-aliasing mechanism [35]. As noted by
the H3DGS authors, hierarchies operating at different scales
require correct anti-aliasing [13]. Anti-aliasing alone improves
3DGS quality by +4.56 dB PSNR and +0.070 SSIM [35];
our observed degradation (-1.80 dB PSNR, -0.15 SSIM)
is consistent with these modifications. The superior LPIPS
score confirms these changes preserve perceptual quality while
ensuring compatibility with standard rendering tools.
F. LiDAR Simulation Accuracy
We evaluate our camera-only Gaussian Splatting reconstruc-
tion against the LiDAR-camera fusion baseline by measuring
reflectivity error relative to real-world ground truth.
TABLE II
LIDAR REFLECTIVITY PREDICTION ERROR (LOWER IS
BETTER). REFLECTIVITY VALUES NORMALIZED TO 0–255
RANGE.
Camera-only (Ours)
LiDAR-based
MAE↓
Median↓
MAE↓
Median↓
Scene 1
10.91
7.16
11.73
7.84
Scene 2
9.37
5.54
10.17
6.70
Scene 3
11.03
6.91
10.47
6.87
Scene 4
9.32
5.77
9.41
6.02
Scene 5
8.98
6.71
8.76
6.36
Average
10.05
6.48
10.14
6.78
MAE: Mean Absolute Error. Our camera-only method
achieves comparable accuracy to the LiDAR-based base-
line.
Qualitative analysis. Figure 4 presents a visual comparison
of LiDAR reflectivity predictions between our camera-only
method, the LiDAR-camera fusion baseline, and real-world
ground truth. Both methods consistently overestimate vegeta-
tion reflectivity due to: (1) temporal inconsistency from wind-
induced vegetation movement violating static scene assump-
tions, (2) high intra-class material variance (leaves, bark, grass
share one label but have different reflectivities), (3) idealized
PBR materials not capturing weathered outdoor conditions,

<!-- page 7 -->
and (4) geometric complexity affecting surface normal estima-
tion. These limitations impact both reconstruction approaches
equally, indicating bottlenecks in the static scene assump-
tion and material parameterization rather than reconstruction
modality. Despite higher geometric noise in the camera-only
method, both approaches produce comparable reflectivity pat-
terns, validating our pipeline for sensor simulation.
Fig. 5. Distribution of absolute reflectivity errors for camera-only (ours) and
LiDAR-based reconstruction methods. Our camera-only approach achieves
comparable median error with slightly higher variability.
Quantitative analysis. Table II and Figure 5 present re-
flectivity error metrics across all five scenes. Our camera-
only method achieves comparable reflectivity accuracy to
the LiDAR-camera fusion baseline, with mean absolute error
(MAE: 10.05 vs. 10.14) and median error (6.48 vs. 6.78)
showing minimal differences between the two approaches.
This demonstrates that camera-only reconstruction can achieve
sensor simulation accuracy comparable to LiDAR-based meth-
ods without requiring LiDAR hardware or complex sensor syn-
chronization. The higher within-scene variability in camera-
only results (visible in Figure 5) reflects the increased geo-
metric noise inherent to MiLO’s camera-based mesh extrac-
tion, which lacks the direct depth measurements provided by
LiDAR. However, the comparable MAE and median errors
demonstrate that this geometric uncertainty does not signifi-
cantly degrade reflectivity simulation accuracy.
G. Ablation Study
We evaluate the impact of incorporating segmentation masks
to improve material classification accuracy. As described in
Sec. III-A, we augment RMSNet’s material predictions with
shape cues from segmentation models.
Experimental setup. We quantitatively assess segmentation
quality using the MCubes multi-modal dataset [15], which
contains 500 material-annotated images. We compare three
configurations: (1) RMSNet alone, (2) RMSNet + FastSAM,
and (3) RMSNet + SAM2 [21].
Results. Table III shows that adding segmentation masks
improves both accuracy (from 0.63 to 0.65–0.67) and mIoU
(from 0.28 to 0.29–0.31) over RMSNet alone. While SAM2
achieves the highest scores (0.67 accuracy, 0.31 mIoU),
we select FastSAM (0.65 accuracy, 0.29 mIoU) due to its
significantly lower computational cost and faster inference
TABLE III
MATERIAL SEGMENTATION PERFORMANCE ON
MCUBES DATASET [15].
Method
Accuracy
mIoU
RMSNet
0.63
0.28
RMSNet + FastSAM (Ours)
0.65
0.29
RMSNet + SAM2
0.67
0.31
speed—critical for processing large-scale multi-view datasets.
Although SAM2 offers superior segmentation quality, the 2%
accuracy improvement does not justify the increased compu-
tational overhead for our application.
Analysis. The 3.2% accuracy improvement (0.63 to 0.65)
demonstrates that shape cues effectively enhance material
classification consistency. However, segmentation masks can
occasionally reduce per-pixel accuracy when incorrect predic-
tions propagate across larger regions, potentially overwriting
fine details such as road markings. The 2% accuracy gap
between FastSAM and SAM2 (0.65 vs. 0.67) represents an
acceptable tradeoff for the practical benefits of automatic mask
extraction in large-scale scenarios.
V. CONCLUSION
We present a camera-only pipeline that combines photo-
realistic Gaussian Splatting with physics-based material as-
signment for Digital Twin reconstruction. Our approach lever-
ages Gaussian Splatting to bridge 2D camera observations
to comprehensive 3D scene representations, including photo-
realistic visuals, semantic information, mesh geometry, and
material properties for sensor simulation. Through compre-
hensive evaluation on real-world urban driving scenes using
LiDAR ground truth for validation, we demonstrate that our
material projection approach achieves reflectivity prediction
accuracy comparable to LiDAR-based methods while main-
taining photorealistic rendering quality, offering a practical
alternative for ADAS development when LiDAR is unavailable
or impractical.
Future work. Future directions include improving geometric
reconstruction quality from camera inputs, extending to dy-
namic scenes, and exploring learned material representations
to reduce dependency on predefined PBR databases. Specific
improvements for vegetation modeling could include: (1)
environment-specific material calibration capturing weathering
and seasonal effects, (2) hierarchical material classification
distinguishing foliage types, bark, and ground cover, and
(3) refined geometric alignment between multi-modal recon-
structions to improve surface normal accuracy for reflectivity
calculations. We have developed an internal tool to convert our
data to the nuScenes format, enabling future benchmarking
against public datasets to demonstrate broader generalization
across diverse urban environments.
REFERENCES
[1] B.
Burley
and
Walt
Disney
Animation
Studios.
Physically-based shading at disney. In ACM SIGGRAPH,
2012.

<!-- page 8 -->
[2] S. Cai, R. Wakaki, S. Nobuhara, and K. Nishino. Rgb
road scene material segmentation. In ACCV, 2022.
[3] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen,
and Q. Tian. Segment any 3d gaussians, 2025.
[4] H. Chen, C. Li, Y. Wang, and G. H. Lee. Neusg: Neural
implicit surface reconstruction with 3d gaussian splatting
guidance, 2025.
[5] X. Cui, W. Ye, Y. Wang, G. Zhang, W. Zhou, and H. Li.
Streetsurfgs: Scalable urban street surface reconstruction
with planar-based gaussian splatting, 2024.
[6] R. C. Gonzalez and R. E. Woods. Digital Image Pro-
cessing. Pearson, 3rd edition, 2008.
[7] A. Gu´edon and V. Lepetit.
Sugar: Surface-aligned
gaussian splatting for efficient 3d mesh reconstruction
and high-quality mesh rendering. In CVPR, 2024.
[8] A. Gu´edon, D. Gomez, N. Maruani, B. Gong, G. Dret-
takis, and M. Ovsjanikov. Milo: Mesh-in-the-loop gaus-
sian splatting for detailed and efficient surface recon-
struction, 2025.
[9] N. Hermidas and M. Phillips. Physics based lidar power
calibration. Technical report, Siemens Industry Software
Netherlands B.V., May 2021.
[10] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao. 2d
gaussian splatting for geometrically accurate radiance
fields. In SIGGRAPH, July 2024.
[11] J. Huang, Z. Gojcic, M. Atzmon, O. Litany, S. Fidler,
and F. Williams. Neural kernel surface reconstruction.
In CVPR, 2023.
[12] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis.
3d gaussian splatting for real-time radiance field render-
ing, 2023.
[13] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer,
A. Lanvin, and G. Drettakis.
A hierarchical 3d gaus-
sian representation for real-time rendering of very large
datasets. ACM TOG, 43(4), July 2024.
[14] M. Kurt. A survey of bsdf measurements and represen-
tations. J. Sci. Eng., 20(58), 2018.
[15] Y. Liang, R. Wakaki, S. Nobuhara, and K. Nishino.
Multimodal material segmentation. In CVPR, June 2022.
[16] Y. Liu, C. Luo, Z. Mao, J. Peng, and Z. Zhang. City-
gaussianv2: Efficient and geometrically accurate recon-
struction for large-scale scenes, 2025.
[17] S.
Manivasagam,
S.
Wang,
K.
Wong,
W.
Zeng,
M. Sazanovich, S. Tan, B. Yang, W.-C. Ma, and R. Ur-
tasun. Lidarsim: Realistic lidar simulation by leveraging
the real world, 2020.
[18] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron,
R. Ramamoorthi, and R. Ng. Nerf: Representing scenes
as neural radiance fields for view synthesis, 2020.
[19] Stefan Muckenhuber, Hannes Holzer, and Zrinka Bockaj.
Automotive lidar modelling approach based on material
properties and lidar capabilities. Sensors, 20(11):3309,
2020.
[20] Ouster, Inc. Os1 lidar sensor - specifications, 2024.
[21] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma,
H. Khedr, R. R¨adle, C. Rolland, L. Gustafson, E. Mintun,
J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick,
P. Doll´ar, and C. Feichtenhofer. Sam 2: Segment anything
in images and videos, 2024.
[22] F. Remondino, A. Karami, Z. Yan, G. Mazzacca,
S. Rigon, and R. Qin. A critical analysis of nerf-based
3d reconstruction. Remote Sens., 15(14), 2023.
[23] J. L. Sch¨onberger and J.-M. Frahm.
Structure-from-
motion revisited. In CVPR, 2016.
[24] X. Shi, L. Chen, P. Wei, X. Wu, T. Jiang, Y. Luo, and
L. Xie. Dhgs: Decoupled hybrid gaussian splatting for
driving scene, 2024.
[25] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall,
P. P. Srinivasan, J. T. Barron, and H. Kretzschmar. Block-
nerf: Scalable large scene neural view synthesis, 2022.
[26] P. Upchurch and R. Niu. A dense material segmentation
dataset for indoor and outdoor scene parsing, 2022.
[27] G. Vecchio and V. Deschaintre. Matsynth: A modern pbr
materials dataset. In CVPR. IEEE, June 2024.
[28] K. Viswanath, P. Jiang, and S. Saripalli. Reflectivity is
all you need!: Advancing lidar semantic segmentation,
2024.
[29] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simon-
celli. Image quality assessment: from error visibility to
structural similarity. IEEE TIP, 13(4), 2004.
[30] Q. Wu, J. Martinez Esturo, A. Mirzaei, N. Moenne-
Loccoz, and Z. Gojcic. 3dgut: Enabling distorted cameras
and secondary rays in gaussian splatting. In CVPR, 2025.
[31] B. Xiong, J. Liu, J. Hu, C. Wu, J. Wu, X. Liu, C. Zhao,
E. Ding, and Z. Lian.
Texgaussian: Generating high-
quality pbr material via octree-based 3d gaussian splat-
ting. In CVPR, June 2025.
[32] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao,
Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth
anything v2. arXiv:2406.09414, 2024.
[33] Y. Yao, Z. Zeng, C. Gu, X. Zhu, and L. Zhang. Reflective
gaussian splatting, 2025.
[34] J. Ye, L. Zhu, R. Zhang, Z. Hu, Y. Yin, L. Li, L. Yu, and
Q. Liao. Large material gaussian model for relightable
3d generation, 2025.
[35] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger.
Mip-splatting: Alias-free 3d gaussian splatting, 2023.
[36] J. E. Zaalberg. Reducing the sim-to-real gap: Lidar-based
3d static environment reconstruction. Master’s thesis, TU
Delft, 2024.
[37] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and
O. Wang. The unreasonable effectiveness of deep fea-
tures as a perceptual metric, 2018.
[38] Z. Zhang, B. Huang, H. Jiang, L. Zhou, X. Xiang, and
S. Shen. Quadratic gaussian splatting for efficient and
detailed surface reconstruction, 2024.
[39] X. Zhao, W. Ding, Y. An, Y. Du, T. Yu, M. Li, M. Tang,
and J. Wang. Fast segment anything, 2023.
