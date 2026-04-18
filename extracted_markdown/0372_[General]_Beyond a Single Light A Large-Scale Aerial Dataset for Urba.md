<!-- page 1 -->
Beyond a Single Light: A Large-Scale Aerial Dataset for Urban Scene
Reconstruction Under Varying Illumination
Zhuoxiao Li1
Wenzong Ma1
Taoyu Wu2
Jinjing Zhu1
Zhenchao Qi1
Shuai Zhang1
Jing OU1
Yinrui Ren1
Weiqing Qi1
Guobin Shen1
Hui Xiong1
Wufan Zhao1
1HKUST(GZ), 2University of Liverpool
Figure 1. We present SkyLume, the first comprehensive real-world UAV dataset centered on illumination variation. It provides 6K-
resolution five-direction imagery from three daily captures along identical RTK-guided flight paths, paired with LiDAR-derived ground
truth including precise meshes and per-frame depth maps and normal maps under unified 6-DoF poses. The benchmark enables rigorous
evaluation of 3D reconstruction, novel view synthesis. Moreover, leveraging the three time-of-day captures, SkyLume is the first real-
world UAV dataset to enable rigorous city-scale evaluation of inverse-rendering quality.
Abstract
Recent advances in Neural Radiance Fields and 3D
Gaussian Splatting have demonstrated strong potential for
large-scale UAV-based 3D reconstruction tasks by fitting
the appearance of images. However, real-world large-scale
captures are often based on multi-temporal data capture,
where illumination inconsistencies across different times of
day can significantly lead to color artifacts, geometric inac-
curacies, and inconsistent appearance. Due to the lack of
UAV datasets that systematically capture the same areas un-
der varying illumination conditions, this challenge remains
largely underexplored. To fill this gap, we introduce Sky-
Lume, a large-scale, real-world UAV dataset specifically
designed for studying illumination robust 3D reconstruc-
tion in urban scene modeling: (1) We collect data from 10
urban regions data comprising more than 100k high resolu-
tion UAV images (four oblique views and nadir), where each
region is captured at three periods of the day to systemat-
ically isolate illumination changes. (2) To support precise
evaluation of geometry and appearance, we provide per-
scene LiDAR scans and accurate 3D ground-truth for as-
sessing depth, surface normals, and reconstruction qual-
ity under varying illumination. (3) For the inverse render-
ing task, we introduce the Temporal Consistency Coefficient
(TCC), a metric that measuress cross-time albedo stability
and directly evaluates the robustness of the disentanglement
of light and material. We aim for this resource to serve as a
foundation that advances research and real-world evalua-
tion in large-scale inverse rendering, geometry reconstruc-
tion, and novel view synthesis.
1
arXiv:2512.14200v1  [cs.CV]  16 Dec 2025

<!-- page 2 -->
1. Introduction
UAV oblique photogrammetry has rapidly become a prac-
tical tool for city-scale modeling and visualization [64].
As 3D vision technologies mature, structure-from-motion
(SfM) [34] and multi-view stereo (MVS) [35] have scaled
through improved matching and learned regularization,
while neural radiance fields (NeRF) [28] and 3D Gaussian
Splatting (3DGS) [12] advance continuous-view rendering,
achieving higher fidelity and faster training.
These ad-
vancements improve geometric completeness, lighting re-
alism, and interactive performance, bringing urban digital
twins closer to reality [2, 3, 21, 25, 27, 38]. However, in
UAV-based 3D vision, particularly at the city scale, sig-
nificant challenges arise due to multi-temporal data cap-
ture. Flights are typically conducted at different times of
day under varying lighting conditions, which complicates
the construction of globally consistent models.
Current
pipelines often integrate observed lighting into textures or
radiance fields but fail to maintain a consistent appearance
over time [37, 53, 62]. This raises a central question: Can
current methods preserve geometric fidelity, photometric
consistency, and novel-view realism under illumination
shifts across different flight sessions?
To address this question, it is essential to assess the im-
pact of varying illumination conditions on the same scene,
as this evaluation is crucial for advancing and refining ex-
isting methodologies. Illumination is a critical factor be-
yond aesthetics [37, 48], as shadow and exposure vari-
ations alter keypoint statistics and multi-view correspon-
dences, which can bias depth and normal estimates [14, 31].
However, existing UAV datasets do not sufficiently address
this need. Many datasets provide either synthetic illumi-
nation diversity based on simplified light transport mod-
els from game engines [16, 63], or real-world data that
seldom revisits the same regions at different times of the
day [9, 22, 42, 49].
The lack of consistent illumination
changes across different time slots and geographic regions
limits the ability to test the stability of existing methods un-
der real-world dynamic lighting conditions. Furthermore,
the existing datasets either fail to include adequate ground-
truth geometry [9, 29, 38] or lack multi-illumination con-
ditions necessary to jointly evaluate geometry and appear-
ance under illumination shifts [23, 49, 50]. As a result, few
works have directly addressed this challenge due to the ab-
sence of datasets and evaluation protocols needed to jointly
study illumination-robust geometry, novel-view synthesis,
and inverse rendering from an aerial perspective.
To this end, we present Skylume, a large-scale, real-
world UAV oblique dataset designed for illumination-robust
3D modeling. Our dataset covers 10 urban regions with
five-direction imagery (four oblique views plus nadir) cap-
tured in the morning, at noon, and in the evening. Each
region includes per-scene LiDAR scans and high-precision
3D ground-truth geometry, and all time slots are precisely
aligned within a unified coordinate system. We also pro-
vide high-quality ground-truth annotations for depth, sur-
face normals, reconstruction quality, and novel-view syn-
thesis evaluation. Additionally, we introduce an inverse-
rendering track with the Temporal Consistency Coefficient
(TCC) metric to quantify cross-time stability. By design,
Skylume transforms multi-temporal capture into a testbed:
methods are challenged not only to render what was ob-
served, but also to harmonize appearance and preserve ge-
ometry across illumination changes throughout the day.
Our benchmark study highlights the value of Skylume
for illumination-robust urban scene modeling. Evaluating
representative 3DGS-based methods under that maintain-
ing stability under varying illumination at scale remains
a significant challenge: (1) Illumination shifts within the
same region induce substantial variations in rendering qual-
ity, particularly for weakly textured fac¸ades, glass bound-
aries, and water-adjacent surfaces, leading to divergent; (2)
Geometry is often corrupted by lighting effects, with cast
shadows and moving penumbras frequently overfitted as
solid structures, thereby biasing depth and surface normal
estimates and introducing spurious geometry; and (3) In-
verse rendering methods at city-scale remain fragile, as the
estimated albedo retains time-dependent shadows and ex-
posure variations, failing to achieve cross-time consistency.
Together, these findings highlight the need for methods that
explicitly separate light from material properties, incorpo-
rate advanced techniques for handling dynamic illumina-
tion, and ensure robust cross-time consistency in large-scale
3D reconstruction tasks.
The contributions are summarized as follows:
• We present Skylume, a multi-temporal UAV oblique
dataset targeting illumination diversity for geometry,
NVS, and inverse rendering. To our knowledge, skyLume
is the first real-world aerial dataset covering 10 urban re-
gions with more than 100K images captured under dif-
ferent illumination conditions, paired with high-precision
3D ground-truth geometry and 6-DoF poses.
• We provide standardized splits and evaluation protocols
for cross-time rendering and geometry evaluation, and
introduce the Temporal Consistency Coefficient (TCC)
metric to quantify cross-time albedo stability.
• We benchmark representative 3DGS variants across vary-
ing illumination conditions and reveal three key chal-
lenges in rendering quality, geometry extraction, and
albedo inconsistency under illumination changes.
2. Related Work
2.1. 3D Gaussian Splatting and Beyond
3D Gaussian Splatting (3DGS) [12] represents a scene as an
explicit set of anisotropic Gaussians and enables real-time,
2

<!-- page 3 -->
Table 1. Comparison of aerial datasets for 3D reconstruction. SkyLume is the only real-world dataset that supports rigorous evaluation
of 3D reconstruction, novel view synthesis, and inverse rendering, while prior corpora miss one or more of these axes.
Aerial Dataset
Real-World
Lidar
Camera Type
Light
Depth/Normal
Resolution
ISPRS-Bencnmark [29]
✓
✓(terrestrial)
Oblique
✗
✗
6000×4000
UrbanScene3D [22]
✓(part)
✓
Oblique
✗
✗
5490×3651
Mill 19 [38]
✓
✗
Oblique
✗
✗
4608×3456
Horizon-GS [9]
✓(part)
✗
Oblique+Nadir
✗
✗
1600×1066
GauU-Scene [49]
✓
✓
Oblique+Nadir
✗
✗
5468×3636
UAVScenes [42]
✓
✓
Nadir
✗
Depth
N/A
Matrix City [16]
✗
✗(from depth)
Oblique+Nadir
✓
Depth+Normal
1920×1080
SkyLume (ours)
✓
✓
Oblique+Nadir
✓
Depth+Normal
6252×4168
visibility-aware rendering with competitive fidelity. Since
its introduction, a rapidly growing literature has extended
3DGS in terms of its applications, core algorithms, and
compatibility: (a) Rendering-centric 3DGS aim for sta-
ble and real-time NVS. Key advances include anti-aliasing
[57], improving densification [5, 13, 17, 26], improving vis-
ibility via deferred or reflective [48, 53, 62], appearance
decoupling for view-dependent effects [48, 52], and LoD
or region-aware scheduling to keep memory bounded [26,
32, 36]. On top of these foundations, large-scene variants
mainly add partitioning and distributed optimization (with
hierarchical/LoD training) [2, 18, 21, 24, 25, 59], boosting
throughput while preserving fidelity. (b) Geometry-aware
3DGS steers Gaussians toward surfaces to obtain editable,
watertight geometry.
Core ideas include surface-aligned
Gaussians with fast mesh extraction and joint refinement
[7], and surfel-like primitives that stabilize normals and
coverage [1, 8, 46]. Hybrids with depth/normal-aware train-
ing [39, 58, 60] further improve mesh quality and down-
stream editability. (c) Inverse-rendering 3DGS augments
Gaussians with explicit shading, materials, and illumination
to enable relighting and editing. Representative directions
include (i) joint recovery of geometry, BRDF, and environ-
ment lighting [19], and (ii) deferred/reflective pipelines that
better handle speculars and visibility [10, 55, 62]; some
further integrate differentiable ray tracing and environment
Gaussians for real-time, view-dependent effects [48, 53].
Despite their strong performance, these methods are pri-
marily designed for single-session captures and lack the
ability to ensure multi-temporal consistency. This limita-
tion is precisely the gap our benchmark addresses.
2.2. UAV Datasets for 3D Reconstruction
UAV datasets for 3D modeling fall into two buckets: (a)
Synthetic city-scale corpora offer full ground truth, accu-
rate poses, and controllable illumination/weather, which are
ideal for ablations but carry domain gaps to real flights
[6, 16, 40, 63].
These methods are built on game en-
gines, provides aerial–street views with ground-truth cam-
eras and flexible light/weather control for city-scale recon-
struction [9, 33, 43].
(b) Real-world captured corpora
better reflect deployment but rarely include cross-time re-
visits under controlled capture, complicating evaluation of
illumination robustness. Some mix large synthetic cities
with a limited set of real scenes, where capture conditions
within a scene can vary [9, 23, 29, 38].
Some covers
large urban areas with RGB and LiDAR and emphasizes
reconstruction under observed lighting [49].
Some con-
tributes frame-wise semantic labels for images and LiDAR,
accurate 6-DoF poses, and reconstructed maps, broaden-
ing multi-modal evaluation but focusing primarily on per-
ception rather than systematic multi-temporal oblique pho-
togrammetry [4, 15, 30, 42, 47, 51].
As shown in Tab. 1, the aim of SkyLume is to address
these above discussed gaps with a large-scale, real-world
UAV oblique benchmark. It supports reproducible proto-
cols for cross-time material/illumination consistency and
fac¸ade-rich reconstruction at scale.
3. The SkyLume Dataset
The Skylume dataset is designed to address the limitations
of existing datasets by providing comprehensive and high-
quality data to support robust performance under varying il-
lumination conditions. Fig. 2 illustrates the overall pipeline.
Section 3.1 details the data acquisition setup, while Sec-
tion 3.2 provides an overview of the data collection process.
Section 3.3 outlines the data processing methods, and Sec-
tion 3.4 presents the resulting high-quality data. Detailed
per scene statics can be found at supplementary material.
3.1. Equipment
As shown in Fig. 2, we employ a survey-grade setup de-
signed for stable flights, high-resolution multi-view im-
agery, and accurate point clouds for reliable alignment and
evaluation: (a) DJI Matrice 350 RTK: A UAV equipped
with RTK-grade positioning and long endurance, enabling
repeatable flights across different times of day. (b) CHC-
NAV C30 Aerial Oblique Camera: A 130 MP camera fea-
turing four 45◦oblique views and one 90◦nadir view, cap-
tured synchronously to enhance fac¸ade coverage and mini-
mize drift. (c) DJI Zenmuse L2 LiDAR: A frame-scanning
LiDAR with 5 cm horizontal and 4 cm vertical accuracy
at 150 m, ensuring precise alignment. Detailed equipment
specifications can be found in the supplementary material.
3

<!-- page 4 -->
Figure 2. Dataset collection and processing pipeline. (a) A survey-grade UAV stack flies the same RTK-guided route at three times
of day to capture five-direction 6K imagery and LiDAR. (b) A unified LiDAR-guided SfM registration includes three periods and refines
poses by point-rendering LiDAR into the cameras. (c) A LiDAR-guided MVS to produce an high-quality aligned ground-truth geometry.
(d) We release per-period split SfM packages and per-frame LiDAR depth, mesh depth, mesh normals, and solar geometry.
Figure 3. Geometry post-processing. We first build geometry
ground truth via LiDAR-guided MVS. For reflective area such as
river and lake, we manually repair water surfaces to correct MVS
failures and ensure geometric continuity.
3.2. Data Collection
As illustrated in Fig. 2, data is collected at three distinct
times of day to capture varying illumination conditions:
early morning (07:00–09:00), midday (11:00–13:00), and
late afternoon (16:00–18:00). To ensure consistency across
these lighting conditions, the same waypoint trajectory is
repeated for each time slot, maintaining comparable view-
points and coverage.
The flight plan follows standard photogrammetric prac-
tices.
The forward overlap is set to 80%, and the side
overlap is 60%.
Flights are conducted at an altitude of
120 m above ground level, with an airspeed of 8 m/s.
The CHCNAV C30 camera is triggered at 1 Hz, captur-
ing five synchronized views to maximize fac¸ade and roof
coverage while minimizing viewpoint and exposure drift.
RTK corrections are applied throughout the flight, and post-
differential positioning is recorded in the EXIF metadata to
provide precise WGS 84 pose estimates for all images. Li-
DAR data is acquired separately on the following day, since
it is an active measurement modality, remains unaffected by
illumination variations.
3.3. Data Preprocessing
Multi-temporal SfM Alignment.
To facilitate accurate
multi-modal alignment, all data is projected into a com-
mon coordinate system. The LiDAR data is georeferenced
in WGS 84 / UTM, while RGB images contain RTK-aided
metadata. We use RealityScan to register the LiDAR point
cloud and generate fixed camera poses on the LiDAR point
cloud. As shown in Fig. 2, these fixed poses are then used
as alignment anchors to render pseudo-RGB LiDAR views.
Subsequently, the camera poses of all RGB images across
the different time slots are forced to align with the Li-
DAR poses. A joint structure-from-motion (SfM) process
is then performed across all three time slots, incorporating
both the RGB images and the pseudo-RGB LiDAR renders.
The RTK pose priors serve as soft constraints to initialize
the camera poses, which are refined during global bundle
adjustment. In areas with sparse features where residual
drift may occur, we correct for any misalignment by aug-
menting the solution with a small set of manually selected
ground control points (GCPs).
A detailed report on the
alignment process, including error metrics and validation,
can be found in the Supplemental Materials.
LiDAR-guided meshing. Given the aligned multi-temporal
poses, we perform surface reconstruction in RealityScan by
4

<!-- page 5 -->
Figure 4. Visualization of the ground-truth models. We visualize post-processed meshes for six representative medium-scale scenes.
fusing the three-time-slot RGB imagery with the LiDAR
data. The LiDAR provides a metric scaffold that regularizes
depth in weak-texture and shadowed regions. As shown in
Fig. 3, the final output is a high-fidelity dense reconstruc-
tion, which outperforms reconstruction using RGB images
alone.
Geometry completion. Rivers and ponds pose challenges
for MVS due to view-dependent specularity and translu-
cency. To address this, we estimate a physically plausible
flat surface at the shoreline using LiDAR data. For each
river or pond, we select four shoreline samples and com-
pute the mean elevation zmean (with a sample spread of ≤2
cm). Surface normals are regularized to be near horizontal,
and the patch is re-meshed (see Fig. 3) to ensure continuity
with the surrounding geometry.
Fig. 4 demonstrates the geometry ground-truth results af-
ter refinement and completion.
3.4. Output Data
To ensure seamless integration with existing pipelines and
facilitate reproducible evaluation, we export each recon-
struction region in the COLMAP format.
SfM Results for Three Time Slots. From the global cam-
era extrinsics, we isolate the image name corresponding to
each of the three time slots, creating three distinct subsets
of camera poses. Using these time-specific extrinsics, we
then match and extract the corresponding camera intrinsics
and sparse point cloud from the global SfM solution.
Per-frame Depth and Normals. We provide two distinct
modalities for depth and normal data: (a) Mesh Depth and
Normals are derived by projecting the mesh into each cam-
era view. The resulting depth and normal maps are dense,
with resolutions matching those of the original images. (b)
LiDAR Depth is obtained by re-projecting the LiDAR point
cloud into the camera frames after occupancy filtering. Al-
though this modality is sparser, it offers higher metric re-
Figure 5. Temporal Consistency Coefficient (TCC) evaluation.
For inverse rendering, we render albedo from identical test view-
points, and compute TCC-Albedo across periods. For Geome-
try, We evaluate each mesh against ground-truth, and compute
TCC-Geometry by averaging pairwise consistency between period
meshes.
liability for accurate depth estimation, as the removal of
background objects prevents erroneous points from being
projected.
Per-frame solar geometry. Using the capture timestamp
and geolocation of each image, we provide solar elevation
and solar azimuth using the NOAA Solar Geometry Cal-
culator. We hope these annotations can support future de-
shadowing, inverse rendering, and relighting studies.
4. Benchmark Experiments
In this section, we evaluate three tracks on six small- and
medium-scale scenes (see Fig. 4): inverse rendering meth-
ods GS-IR [19], Ref-Gaussian [53], and Ref-GS [62]; ge-
ometry methods 2DGS [8], PGSR [1], GOF [58], and City-
gaussianV2 [25]; and novel view synthesis methods 3DGS
[11], Abs-GS [56], Mip-SPlatting [57], and Octree-GS [32].
To better evaluate the performance of each method in large-
scale scenes, we conducted benchmarks on an NVIDIA
RTX A800 80G GPU, applying identical adjustments to the
training pipeline and parameter settings for each method.
5

<!-- page 6 -->
Table 2. TCC metric across three inverse-rendering baselines. We report the distribution of sub-metrics (TCC-LPIPS, TCC-SSIM,
TCC-MAE) and the combined TCC Overall with Mean, Min, Max, and Std. All scores lie in [0,1], higher is better except Std.
Mean↑
Min↑
Max↑
Std↓
methods
GS-IR [19] Ref-Gaussian [53] Ref-GS [62] GS-IR [19] Ref-Gaussian [53] Ref-GS GS-IR Ref-Gaussian Ref-GS [62] GS-IR [19] Ref-Gaussian [53] Ref-GS [62]
TCC-LPIPS
0.826
0.866
0.874
0.759
0.778
0.755
0.867
0.919
0.979
0.020
0.027
0.029
TCC-SSIM
0.905
0.883
0.928
0.864
0.832
0.878
0.936
0.928
0.985
0.014
0.019
0.017
TCC-MAE
0.700
0.513
0.766
0.212
0.262
0.587
0.800
0.653
0.977
0.074
0.078
0.063
TCC Overall
0.721
0.658
0.775
0.563
0.575
0.622
0.765
0.735
0.913
0.033
0.026
0.036
Figure 6. Albedo and TCC visualization across two time slots. Note that Period 2 is rendered from the viewpoints of Period 1 to fix
camera pose, so differences arise solely from illumination.
For detailed implementation information, please refer to the
supplementary material.
4.1. Benchmark Metrics
Illumination robustness. In real-world conditions, it is not
feasible to capture accurate ground-truth albedo. Therefore,
demonstrated in Fig. 5, we propose Temporal Consistency
Coefficient (TCC) to evaluate inverse rendering robustness
by rendering albedo from the same test viewpoints across
three time slots since all images are registered by a uni-
fied SfM solution. From the global trajectory, we select
K fixed test viewpoints {vk}K
k=1 using nearest pose match-
ing and uniform spatial coverage. To evaluate the consis-
tency of albedo across different illumination conditions, we
use a combination of four metrics: MAE and RMSE assess
pixel-level accuracy and color consistency, helping to en-
sure the albedo maintains precision across varying light-
ing. SSIM measures structural similarity, evaluating how
well the albedo preserves the underlying surface patterns.
LPIPS, which compares deep features of the images, cap-
tures perceptual differences and ensures that the albedo
maintains visual plausibility across time slots, especially
under complex lighting conditions.
At each test viewpoint, we render the three albedo
maps {A(k)
t
}3
t=1.
For each viewpoint vk, we compute
the combined score TCC(k)
comb ∈[0, 1] as: TCC(k)
comb =
α TCC(k)
MAE+β TCC(k)
RMSE+θ TCC(k)
SSIM+γ TCC(k)
LPIPS, where
α = 0.2, β = 0.2, θ = 0.2, and γ = 0.4. The final
TCC score is the arithmetic mean over the K viewpoints:
TCCoverall = 1
K
PK
k=1 TCC(k)
comb.
Geometry robustness. Demonstrated in Fig. 5, we evaluate
robustness of reconstructed geometry under changing illu-
mination in two ways. First, we measure absolute accuracy
against per scene GT geometry in Fig. 4. Second, we mea-
sure cross-time consistency between meshes reconstructed
from the three time slots.
Novel view synthesis. We additionally evaluate novel view
synthesis using standard image quality metrics, reporting
PSNR, SSIM [44], and LPIPS [61].
4.2. Cross-Time Albedo Evaluation
We assess temporal albedo consistency across three inverse-
rendering baselines using our TCC metric, which re-
wards cross-time stability rather than single-slot fidelity.
Tab. 2 shows that albedo estimates drift under illumination
changes even at matched viewpoints, with large dispersion
across scenes in the Min, Max, and Std columns. The per-
ceptual branch TCC-MAE is frequently the limiting factor
while TCC-SSIM remains comparatively stable, indicating
that structure is preserved more often than appearance.
Fig. 6 corroborates these trends on two regions with strong
lighting shifts where all methods retain shadow imprinting.
As a result, the proposed TCC turns multi-temporal robust-
ness into a measurable target, enabling fair and reproducible
comparison of illumination-robust inverse rendering.
4.3. Cross-Time Geometry Evaluation
We evaluate geometry under illumination change by report-
ing precision, recall, and F-1 against GT for each time slot,
and by measuring cross-time consistency via the average
6

<!-- page 7 -->
Table 3. TCC metric across four geometry baselines. For each method we report precision (Pre), recall (Rec), and F-1 at three distance
tolerances τ ∈{0.25, 0.5, 0.75} m for Period 0 and Period 1. The rightmost block gives temporal consistency as the average pairwise F-1
between the period meshes at the same distance tolerances.
Period 1
Period 2
TCC-Geometry
τ (meters)
0.25
0.5
0.75
0.25
0.5
0.75
0.25
0.5
0.75
metrics
Pre↑
Rec↑
F-1 ↑
Pre↑
Rec↑
F-1
Pre↑
Rec↑
F-1↑
Pre↑
Rec↑
F-1
Pre↑
Rec↑
F-1 ↑
Pre↑
Rec↑
F-1↑
F-1↑
F-1↑
F-1↑
2DGS [8]
0.203 0.259 0.227
0.462 0.562 0.508
0.663 0.750 0.704
0.199 0.278 0.232 0.486 0.639 0.552 0.672 0.811 0.735 0.570 0.675 0.750
PGSR [1]
0.136 0.171 0.152
0.382 0.485 0.428 0.680 0.764 0.719
0.150 0.198 0.171 0.368 0.489 0.420
0.671 0.778 0.720
0.554 0.662 0.749
GOF [58]
0.106 0.101 0.103
0.417 0.336 0.372 0.775 0.594 0.672 0.120 0.113 0.116 0.455 0.357 0.400
0.810 0.599 0.689 0.431 0.610 0.712
CityGaussianV2 [25] 0.315 0.270 0.291
0.618 0.603 0.610 0.782 0.835 0.808
0.256 0.245 0.250
0.563 0.569 0.566
0.797 0.746 0.770 0.547 0.719 0.790
Figure 7. Geometry visualization. Top: Geometry for Gym and Residence, red and yellow frames are the zoomed-in surface normals.
Bottom: Single Period 1 mesh comparison. Under the sunlit Period 1, all methods exhibit holes and breakups.
pairwise consistency F-1 score between the the meshes in
Tab. 3. Two lighting regimes are highlighted in Fig. 7: a
sunlit case with strong, moving shadows and color shifts,
and an overcast case dominated by diffuse irradiance. Ge-
ometry extracted in diffuse conditions is the most stable,
while direct sunlight degrades both absolute accuracy and
cross-time agreement. Methods that using alpha-blending
depth for TSDF-fusion [45] (2DGS [8] and PGSR [1]) pre-
serve mesh topology more consistently across slots but lack
geometry details. Approaches that rely on opacity (GOF
[58] and CityGaussianV2 [25]) can reach high per-slot F-
1 score under favorable lighting but are brittle under hard
shadows. Fig. 7 reveals holes and breakups in sunlight-
swept regions such as lawn and water-body.
4.4. Novel View Synthesis Evaluation
Under strong lighting conditions, we evaluate four NVS-
oriented methods and three geometry-oriented methods.
As summarized in Tab. 4, the trend is consistent across
Figure 8. Additional Visualization of the Impact of Strong Il-
lumination. Cast shadows break multi-view photometric consis-
tency, yielding shadow-as-geometry artifacts. In reflective planes,
view-dependent appearance leads to blurred textures and detail
loss in NVS.
scenes: geometry-oriented pipelines incur a noticeable loss
in photometric fidelity and therefore lag in rendering qual-
7

<!-- page 8 -->
Table 4. Novel view synthesis under Period 1 (sunlit) reconstruction. We report SSIM, PSNR, and LPIPS across six UAV scenes for
four NVS-oriented methods and three geometry-capable pipelines. ”-” indicates unavailable results due to GPU out-of-memory.
GYM
iPark
Tec School
Staff residence
Campus
High School
metrics
SSIM↑PSNR↑LPIPS↓
SSIM↑PSNR↑LPIPS↓
SSIM↑PSNR↑LPIPS↓
SSIM↑PSNR↑LPIPS↓
SSIM↑PSNR↑LPIPS↓
SSIM↑PSNR↑LPIPS↓
3DGS
0.672
21.144
0.318
0.707
23.250
0.355
0.631
21.652
0.384
0.670
23.099
0.353
0.617
21.507
0.409
0.626
20.570
0.389
Abs-GS
0.681
21.183
0.305
0.711
22.958
0.335
0.639
21.523
0.375
0.676
23.125
0.352
0.633
21.477
0.390
0.634
20.413
0.382
Mip-Splatting
0.669
21.131
0.325
0.701
22.708
0.362
0.621
21.364
0.398
0.667
23.006
0.360
0.619
20.015
0.427
-
-
-
Octree-GS
0.648
20.931
0.336
0.697
22.681
0.358
0.607
21.765
0.360
0.659
23.094
0.366
0.624
21.432
0.425
0.624
20.618
0.380
GOF
0.626
20.830
0.360
0.681
23.172
0.382
0.619
21.224
0.401
0.619
22.513
0.401
0.505
20.047
0.524
-
-
-
2DGS
0.583
20.350
0.425
0.648
22.722
0.440
0.549
21.103
0.489
0.585
21.891
0.458
0.507
20.518
0.539
0.497
19.531
0.539
PGSR
0.580
20.110
0.424
0.622
21.953
0.467
0.537
20.926
0.500
0.591
22.094
0.426
0.514
20.572
0.530
0.490
19.398
0.547
Figure 9. Visualization of novel view synthesis under Period 1 (sunlit). We visualize renderings from shadow-dominated regions to
highlight some baselines lost details.
ity. Within the NVS group, Abs-GS [56] is the most reliable
under UAV oblique viewpoints, and Octree-GS [32] recov-
ers fine structures. By contrast, 2DGS [8] produces blurred
renderings in all four representative scenes, as illustrated in
Fig. 9. In Fig. 8, we can observe that strong sunlight intro-
duces shadow boundaries and exposure drift, which amplify
cross-view color inconsistencies, degrade local detail, and
widen metric differences.
5. Conclusion and Future Work
In this paper, we introduce SkyLume, a large-scale UAV
dataset that couples high-resolution imagery captured under
three different illumination conditions. The release includes
unified SfM resolution, per-frame depth and normals, and
solar geometry. Central to the dataset is the Temporal Con-
sistency Coefficient metric, which evaluates cross-time sta-
bility rather than single-slot fidelity. Experiments findings
point to concrete directions: future research should inte-
grate cross-time, shadow-aware priors, and jointly reason
about illumination, material, and geometry at city scale.
Future Work. We plan to further expand SkyLume with
additional scenes, weather conditions, and sensing modali-
ties to enhance robustness and support tasks such as relight-
ing and editing. In addition, we plan to add multimodal an-
notations, such as semantic segmentation and infrared im-
agery, to enable cross-modal fusion for illumination-robust
reconstruction.
Our goal is to transform multi-temporal
UAV capture from a confounding factor into a powerful tool
for stable inverse rendering, reliable geometry reconstruc-
tion, and high-fidelity novel view synthesis in real urban
environments.
8

<!-- page 9 -->
References
[1] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for
efficient and high-fidelity surface reconstruction. 2024. 3, 5,
7
[2] Junyi Chen, Weicai Ye, Yifan Wang, Danpeng Chen, Di
Huang, Wanli Ouyang, Guofeng Zhang, Yu Qiao, and Tong
He. Gigags: Scaling up planar-based 3d gaussians for large
scene surface reconstruction, 2024. 2, 3
[3] Yu Chen and Gim Hee Lee.
Dogaussian:
Distributed-
oriented gaussian splatting for large-scale 3d reconstruction
via gaussian consensus, 2024. 2
[4] Devansh Dhrafani, Yifei Liu, Andrew Jong, Ukcheol Shin,
Yao He, Tyler Harp, Yaoyu Hu, Jean Oh, and Sebastian
Scherer. Firestereo: Forest infrared stereo dataset for uas
depth perception in visually degraded environments. arXiv
preprint arXiv:2409.07715, 2024. 3
[5] Guangchi Fang and Bing Wang. Mini-splatting: Represent-
ing scenes with a constrained number of gaussians, 2024. 3
[6] Michael Fonder and Marc Van Droogenbroeck. Mid-air: A
multi-modal dataset for extremely low altitude drone flights.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition workshops, pages 0–0, 2019.
3
[7] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh recon-
struction and high-quality mesh rendering. arXiv preprint
arXiv:2311.12775, 2023. 3
[8] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 3, 5, 7, 8
[9] Lihan Jiang, Kerui Ren, Mulin Yu, Linning Xu, Junting
Dong, Tao Lu, Feng Zhao, Dahua Lin, and Bo Dai. Horizon-
gs: Unified 3d gaussian splatting for large-scale aerial-to-
ground scenes. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 26789–26799, 2025.
2, 3
[10] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xi-
aoxiao Long, Wenping Wang, and Yuexin Ma. Gaussian-
shader: 3d gaussian splatting with shading functions for re-
flective surfaces. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
5322–5332, 2024. 3
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 5
[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[13] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. In Advances in Neural
Information Processing Systems (NeurIPS), 2024. Spotlight
Presentation. 3
[14] Fabian Langguth, Kalyan Sunkavalli, Sunil Hadap, and
Michael Goesele.
Shading-aware multi-view stereo.
In
European Conference on Computer Vision, pages 469–485.
Springer, 2016. 2
[15] Wen Li, Shangshu Yu, Cheng Wang, Guosheng Hu, Siqi
Shen, and Chenglu Wen.
SGLoc: Scene geometry en-
coding for outdoor LiDAR localization. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9286–9295, 2023. 3
[16] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhen-
zhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale
city dataset for city-scale neural rendering and beyond. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 3205–3215, 2023. 2, 3
[17] Zhuoxiao Li, Shanliang Yao, Yijie Chu, Angel F. Garcia-
Fernandez, Yong Yue, Eng Gee Lim, and Xiaohui Zhu.
Mvg-splatting: Multi-view guided gaussian splatting with
adaptive quantile-based geometric consistency densification,
2024. 3
[18] Zhuoxiao Li, Shanliang Yao, Taoyu Wu, Yong Yue, Wu-
fan Zhao, Rongjun Qin, ´Angel F. Garc´ıa-Fern´andez, Andrew
Levers, Jason Ralph, and Xiaohui Zhu. Ulsr-gs: Urban large-
scale surface reconstruction gaussian splatting with multi-
view geometric consistency. ISPRS Journal of Photogram-
metry and Remote Sensing, 230:861–880, 2025. 3
[19] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia.
Gs-ir: 3d gaussian splatting for inverse rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21644–21653, 2024. 3, 5, 6
[20] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen,
Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth
anything 3: Recovering the visual space from any views.
arXiv preprint arXiv:2511.10647, 2025. 5
[21] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiy-
ong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu,
Youliang Yan, and Wenming Yang. Vastgaussian: Vast 3d
gaussians for large scene reconstruction. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5166–5175, 2024. 2, 3
[22] Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and
Hui Huang. Capturing, reconstructing, and simulating: the
urbanscene3d dataset. In European Conference on Computer
Vision, pages 93–109. Springer, 2022. 2, 3
[23] Yilin Liu, Fuyou Xue, and Hui Huang. Urbanscene3d: A
large scale urban scene dataset and simulator. 2021. 2, 3
[24] Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Naiyan
Wang, Junran Peng, and Zhaoxiang Zhang.
Citygaus-
sian: Real-time high-quality large-scale scene rendering with
gaussians. arXiv preprint arXiv:2404.01133, 2024. 3
[25] Yang Liu, Chuanchen Luo, Zhongkai Mao, Junran Peng, and
Zhaoxiang Zhang. Citygaussianv2: Efficient and geometri-
cally accurate reconstruction for large-scale scenes. In ICLR,
2025. 2, 3, 5, 7
[26] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
9

<!-- page 10 -->
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3
[27] Zhenxing Mi and Dan Xu. Switch-nerf: Learning scene de-
composition with mixture of experts for large-scale neural
radiance fields.
In International Conference on Learning
Representations (ICLR), 2023. 2
[28] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[29] F. Nex, M. Gerke, F. Remondino, H.-J. Przybilla, M.
B¨aumker, and A. Zurhorst.
Isprs benchmark for multi-
platform photogrammetry. ISPRS Annals of the Photogram-
metry, Remote Sensing and Spatial Information Sciences, II-
3/W4:135–142, 2015. 2, 3
[30] Thien-Minh Nguyen, Shenghai Yuan, Muqing Cao, Yang
Lyu, Thien H Nguyen, and Lihua Xie. Ntu viral: A visual-
inertial-ranging-lidar dataset, from an aerial vehicle view-
point. The International Journal of Robotics Research, 41
(3):270–280, 2022. 3
[31] Julien Philip, Micha¨el Gharbi, Tinghui Zhou, Alexei A
Efros, and George Drettakis. Multi-view relighting using a
geometry-aware network. ACM Trans. Graph., 38(4):78–1,
2019. 2
[32] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 3, 5, 8
[33] Giulia Rizzoli, Francesco Barbato, Matteo Caligiuri, and
Pietro Zanuttigh. Syndrone-multi-modal uav dataset for ur-
ban scenarios.
In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 2210–2220,
2023. 3
[34] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 2
[35] Steven M Seitz, Brian Curless, James Diebel, Daniel
Scharstein, and Richard Szeliski. A comparison and evalua-
tion of multi-view stereo reconstruction algorithms. In 2006
IEEE computer society conference on computer vision and
pattern recognition (CVPR’06), pages 519–528. IEEE, 2006.
2
[36] Yunji Seo,
Young Sun Choi,
Hyun Seung Son,
and
Youngjung Uh. Flod: Integrating flexible level of detail into
3d gaussian splatting for customizable rendering, 2024. 3
[37] Shuang Song and Rongjun Qin. A general albedo recovery
approach for aerial photogrammetric images through inverse
rendering. ISPRS Journal of Photogrammetry and Remote
Sensing, 218:101–119, 2024. 2
[38] Haithem Turki,
Deva Ramanan,
and Mahadev Satya-
narayanan.
Mega-nerf:
Scalable construction of large-
scale nerfs for virtual fly-throughs.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 12922–12931, 2022. 2, 3
[39] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto
Seiskari, Esa Rahtu, and Juho Kannala. Dn-splatter: Depth
and normal priors for gaussian splatting and meshing, 2024.
3
[40] Khiem Vuong, Anurag Ghosh, Deva Ramanan, Srinivasa
Narasimhan, and Shubham Tulsiani.
Aerialmegadepth:
Learning aerial-ground reconstruction and view synthesis. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, 2025. 3
[41] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng
Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong
Yang. Moge-2: Accurate monocular geometry with metric
scale and sharp details, 2025. 5
[42] Sijie Wang, Siqi Li, Yawei Zhang, Shangshu Yu, Shenghai
Yuan, Rui She, Quanjiang Guo, JinXuan Zheng, Ong Kang
Howe, Leonrich Chandra, et al. Uavscenes: A multi-modal
dataset for uavs. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 28946–28958,
2025. 2, 3
[43] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu,
Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Se-
bastian Scherer. Tartanair: A dataset to push the limits of
visual slam. 2020. 3
[44] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 6
[45] Diana Werner, Ayoub Al-Hamadi, and Philipp Werner. Trun-
cated signed distance function: experiments on voxel size.
In Image Analysis and Recognition: 11th International Con-
ference, ICIAR 2014, Vilamoura, Portugal, October 22-24,
2014, Proceedings, Part II 11, pages 357–364. Springer,
2014. 7
[46] Yaniv Wolf, Amit Bracha, and Ron Kimmel.
Surface re-
construction from gaussian splatting via novel stereo views.
arXiv preprint arXiv:2404.01810, 2024. 3
[47] Rouwan Wu, Xiaoya Cheng, Juelin Zhu, Yuxiang Liu, Mao-
jun Zhang, and Shen Yan. Uavd4l: A large-scale dataset for
uav 6-dof localization. In 2024 International Conference on
3D Vision (3DV), pages 1574–1583. IEEE, 2024. 3
[48] Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yu-
jun Shen, Sida Peng, Hujun Bao, and Xiaowei Zhou. En-
vgs: Modeling view-dependent appearance with environ-
ment gaussian. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 5742–5751, 2025. 2,
3
[49] Butian Xiong, Nanjun Zheng, Junhua Liu, and Zhen Li.
Gauu-scene v2: Assessing the reliability of image-based
metrics with expansive lidar image dataset using 3dgs and
nerf. arXiv preprint arXiv:2404.04880, 2024. 2, 3
[50] Butian Xiong, Nanjun Zheng, Junhua Liu, and Zhen Li.
Gauu-scene v2: Assessing the reliability of image-based
metrics with expansive lidar image dataset using 3dgs and
nerf. CoRR, 2024. 2
[51] Qi Yan, Jianhao Zheng, Simon Reding, Shanci Li, and Ior-
dan Doytchinov. Crossloc: Scalable aerial localization as-
sisted by multimodal synthetic data.
In Proceedings of
10

<!-- page 11 -->
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17358–17368, 2022. 3
[52] Ziyi Yang, Xinyu Gao, Yangtian Sun, Yihua Huang, Xi-
aoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and Xi-
aogang Jin. Spec-gaussian: Anisotropic view-dependent ap-
pearance for 3d gaussian splatting, 2024. 3
[53] Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, and
Li Zhang.
Reflective gaussian splatting.
arXiv preprint
arXiv:2412.19282, 2024. 2, 3, 5, 6
[54] Chongjie Ye,
Lingteng Qiu,
Xiaodong Gu,
Qi Zuo,
Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and
Xiaoguang Han. Stablenormal: Reducing diffusion variance
for stable and sharp normal. ACM Transactions on Graphics
(TOG), 2024. 5
[55] Keyang Ye, Qiming Hou, and Kun Zhou. 3d gaussian splat-
ting with deferred reflection. In ACM SIGGRAPH 2024 Con-
ference Papers, pages 1–10, 2024. 3
[56] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong
Dou. Absgs: Recovering fine details for 3d gaussian splat-
ting, 2024. 5, 8
[57] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 19447–19456,
2024. 3, 5
[58] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient high-quality compact surface recon-
struction in unbounded scenes. arXiv:2404.10772, 2024. 3,
5, 7
[59] Gim Hee Lee Yu Chen. Dogaussian: Distributed-oriented
gaussian splatting for large-scale 3d reconstruction via gaus-
sian consensus. In arXiv, 2024. 3
[60] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang,
Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in
gaussian splatting. arXiv preprint arXiv:2406.01467, 2024.
3
[61] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[62] Youjia Zhang, Anpei Chen, Yumin Wan, Zikai Song, Jun-
qing Yu, Yawei Luo, and Wei Yang.
Ref-gs: Directional
factorization for 2d gaussian splatting. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
26483–26492, 2025. 2, 3, 5, 6
[63] Fuqiang Zhao, Yijing Guo, Siyuan Yang, Xi Chen, Luo
Wang, Lan Xu, Yingliang Zhang, Yujiao Shi, and Jingyi Yu.
Aerialgo: Walking-through city view generation from aerial
perspectives. arXiv preprint arXiv:2412.00157, 2024. 2, 3
[64] Tihui Zhou, Linbing Lv, Jinhu Liu, and J Wan. Application
of uav oblique photography in real scene 3d modeling. The
International Archives of the Photogrammetry, Remote Sens-
ing and Spatial Information Sciences, 43:413–418, 2021. 2
11

<!-- page 12 -->
Beyond a Single Light: A Large-Scale Aerial Dataset for Urban Scene
Reconstruction Under Varying Illumination
Supplementary Material
6. Dataset Details
6.1. Equipment Details
Table 5. Specifications of the UAV platform, oblique camera, and
LiDAR payload used in our data acquisition.
Device
Type
Key parameters
DJI M350 RTK
UAV platform
1. Max flight time: 55 min (no payload);
2. Max payload: 2.7 kg;
3. GNSS: GPS/BeiDou;
4. Sensing: 6-direction.
CHCNAV C30
Oblique camera
1. Total resolution: 130 MP (26 MP × 5);
2. Image size: 6252 × 4168 (3:2);
3. Lenses: 1×∼nadir 90◦+ 4×∼oblique 45◦;
4. Focal lengths: 25 mm / 35 mm;
5. Minimum capture interval: 0.8 s;
6. Weight: 605 g;
7. Dimensions: 110 × 108 × 85 mm.
DJI L2
LiDAR
1. LiDAR range: 450 m (50% reflectivity);
2. Point rate: 1.2M pts/s (multi-return);
3. Accuracy: 5 cm horizontal / 4 cm vertical;
4. Ranging accuracy: 2 cm @ 150 m;
5. FOV (non-repetitive): 70◦× 75◦;
6. Max returns: 5;
7. RGB sensor: 4/3” CMOS, 20 MP;
8. RGB FOV: 84◦;
9. Weight: 905 g;
UAV Platform (DJI M350 RTK).
We use a DJI Matrice
350 RTK as the carrier platform. It is an industrial-rank
UAV with a maximum flight time of about 55 minutes with-
out payload. The built-in RTK module provides centimeter-
level positioning when used with network RTK or a base
station. These properties enable us to replicate nearly iden-
tical flight trajectories and camera poses at three times of
day, effectively controlling viewpoint and geometric vari-
ation so that differences across slots predominantly reflect
illumination rather than changes in scene structure.
Oblique
Imaging
Sensor
(CHCNAV
C30).
High-
resolution multi-view imagery is provided by the CHCNAV
C30 oblique camera system.
The C30 integrates five
synchronized lenses sharing a common APS-C sensor
footprint of 23.5 × 15.6 mm. Each view has a resolution of
6252×4168 pixels (about 26 MP), yielding a total effective
resolution of 130 MP per exposure. One lens is nadir (90◦)
with a 25 mm focal length, and four lenses are oblique
(45◦) with 35 mm focal length, forming a cross-shaped
layout that simultaneously captures fac¸ades and roofs. In
our flights, the minimum capture interval of 1.0 s allows us
to maintain high forward and side overlap at typical survey
Figure 10. The DJI M350 RTK UAV platform equipped with a
CHCNAV C30 oblique camera before take-off.
speeds, while the fixed aperture around f/5.6 and ISO in
the range 800-1600 provide robust exposure under varying
illumination.
These key parameters ensure fine ground
sampling, rich parallax on vertical structures, and reduced
drift in following SfM/MVS and reconstruction.
LiDAR Ground-truth Sensor (DJI Zenmuse L2).
To
obtain accurate 3D geometry for evaluation, we mount a DJI
Zenmuse L2 payload. The L2 integrates a frame LiDAR, a
high-accuracy IMU, and a 4/3” 20 MP RGB mapping cam-
era on a 3-axis stabilized gimbal. The ranging accuracy is
about 2 cm at 150 m, and under our survey configuration
(relative altitude ≈120 m, RTK FIX, and point-cloud accu-
racy optimization enabled in DJI Terra), the resulting point
clouds achieve approximately 5 cm horizontal and 4 cm ver-
tical accuracy with respect to check points. These key Li-
DAR characteristics (centimeter-level ranging accuracy and
high-density multi-return point clouds) enable us to con-
struct a high-fidelity reference surface via LiDAR-guided
MVS reconstruction, as illustrated in Fig. 3. For highly re-
flective or specular regions such as rivers and lakes, we fur-
ther apply manual water-surface editing on top of the L2-
based model to correct MVS failures and enforce geometric
continuity.
6.2. Data Collection Details
Per-scene statistics.
Tab. 6 summarizes the per-scene
statistics of our dataset.
We cover ten urban regions
grouped into three scales: compact campus-level blocks
1

<!-- page 13 -->
Figure 11. Challenging cases across our dataset. We show representative crops for four typical difficulty patterns: (a) water bodies with
strong specular reflections and weak texture, (b) glass fac¸ades with transparency and mirror-like reflections, (c) complex architectures with
intricate geometry and self-occlusions, and (d) high-density urban blocks with tightly packed buildings.
Table 6. Per-scene dataset statistics. For each scene we report the total number of multi-view UAV images over three time slots, nominal
flight height, building density, illumination configuration, and the presence of large water areas or glass fac¸ades. Scenes are grouped into
three scales: small, medium, and large, covering campus blocks to city-block-level neighborhoods.
Scene
Name
Total Images
Flight Height
Building Density
Illumination Type
Water Area
Glass Facade
Small Scale
Gym
6185
120.366
Low
Direct Sunlight (Morning)
Direct Sunlight (Noon)
Overcast (Afternoon)
No
No
Staff Residence
7920
130.018
Medium
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Overcast (Afternoon)
Yes
No
iPark
5355
115.241
Medium
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Overcast (Afternoon)
Yes
Yes
Medium Scale
Tec School
7185
108.481
Low
Direct Sunlight (Morning)
Direct Sunlight (Noon)
Direct Sunlight(Afternoon)
Yes
No
Buildings
10455
129.891
High
Direct Sunlight (Morning)
Direct Sunlight (Noon)
Overcast (Afternoon)
No
Yes
High School
10065
108.015
Medium
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Direct Sunlight(Afternoon)
Yes
No
Main Campus
10410
130.28
Medium
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Direct Sunlight(Afternoon)
No
No
Large Scale
Estate
18630
109.024
High
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Direct Sunlight(Afternoon)
No
No
Town
12435
149.26
High
Direct Sunlight (Morning)
Partly Cloudy (Noon)
Overcast (Afternoon)
Yes
Yes
Med School
20700
119.25
Medium
Direct Sunlight (Morning)
Direct Sunlight (Noon)
Overcast (Afternoon)
Yes
No
(Small Scale), mid-sized institutional and residential dis-
tricts (Medium Scale), and city-block-level neighborhoods
(Large Scale).
For each scene we report the total num-
ber of multi-view images aggregated over the three time
slots, the approximate flight height, a categorical building-
density label (Low/Medium/High), the illumination config-
uration, and two binary attributes indicating the presence of
large water bodies (Water Area) and extensive glass fac¸ades
2

<!-- page 14 -->
Figure 12. Reconstruction challenges under partly cloudy illumination. In partly cloudy scenes, moving clouds create strongly varying
local illumination, so the same region appears sunlit in some views and shadowed in others. This breaks photometric consistency across
viewpoints and leads to color inconsistencies.
(Glass Facade).
Across all scenes, the total number of images per scene
ranges from roughly 5k to over 20k, with nominal flight
heights concentrated around 110-150 m to keep the ground
sampling distance nearly consistent. Examples are illus-
trated in Fig. 11.
The building-density labels span two
low-density scenes, five medium-density scenes, and three
high-density scenes, covering open campuses, mixed-use
blocks, and dense residential areas. The illumination types
combine three time-of-day slots (morning, noon, afternoon)
with varying sky conditions (direct sunlight, partly cloudy
(see Fig. 12), overcast), yielding sequences that range from
purely clear-sky (e.g., Tec School) to mixed clear/overcast
setups (e.g., Gym, Buildings, Town). The Water Area and
Glass Facade columns highlight challenging scenes where
strong reflections or transparency are prominent: six scenes
contain sizable rivers or lakes, three scenes exhibit large
glass fac¸ades, and two scenes (iPark and Town) feature both.
In addition, the Med School scene in particular contains a
large-scale water body combined with complex surround-
ing structures, making it one of the most challenging cases
in our benchmark. These attributes allow users to define tar-
geted subsets focusing on specific difficulties such as spec-
ular water, glass buildings, complex architecture, or densely
built environments.
Alignment statistics.
Fig. 13 demonstrates an example
of the LiDAR-as-anchor for the SfM alignment for two
time slots on Residence Scene. And Table 7 summarizes
the global SfM alignment quality and camera position un-
certainty. The reconstruction is strongly constrained, with
Table 7. Summary of Mean SfM alignment statistics and camera
position uncertainty of 10 Scenes.
SfM alignment report
Average Projections
13 921 072
Average track length
3.3
Maximum non-compressible error [px]
2.00
Median reprojection error [px]
0.70
Mean reprojection error [px]
0.79
Relative camera position uncertainty [m]
X
Y
Z
Mean
≤0.001
≤0.002
≤0.001
Standard deviation
≤0.001
≤0.001
≤0.001
Maximum
≤0.008
≤0.019
≤0.014
Minimum
≤0.001
≤0.001
≤0.001
roughly 1.4 × 107 feature projections and an average track
length of 3.3 observations per 3D point. The median and
mean reprojection errors are 0.70 px and 0.79 px, both well
below one pixel, indicating accurate camera calibration and
bundle adjustment. The relative camera position uncertainty
is on the order of millimeters in all three axes on average,
with maximum uncertainties below 2 cm, demonstrating a
geometrically stable camera network and a reliable multi-
temporal registration.
3

<!-- page 15 -->
Figure 13. LiDAR-Guided SfM Pipeline.
7. Implementation Details
7.1. Parameter Settings
For all 3DGS-based baselines evaluated under our single-
GPU setting, we adopt a unified training schedule for fair-
ness and stable reconstruction. Unless otherwise specified,
each model is trained for 90,000 iterations in total. The
densification interval is set to 300 iterations, the opacity re-
set interval is set to 9,000 iterations, and densification is
disabled after 60,000 iterations. This allows the scene to
be sufficiently populated and refined in the early and mid
stages of training, while preventing unbounded growth of
Gaussians and keeping memory usage under control on a
single RTX A800 80 GB GPU.
7.2. Training Strategy on Single GPU
All methods in our benchmark are trained on a single RTX
A800 80 GB GPU. Directly using the original training
schedules leads to unfair behavior on large-scale scenes:
standard 3DGS-style pipelines will reduce opacity many
times and then prune low-opacity Gaussians at the next den-
sification step (100 iteration). On a single GPU, however,
a full sweep over all training views takes many iterations.
As a consequence, many Gaussians are pruned before they
have been sufficiently updated by all views, which results in
over-pruning and loss of fine structures, especially in large
or sparsely observed regions.
To ensure both stability and fairness across methods un-
der this resource constraint, we adopt the unified single-
GPU training strategy summarized in Alg. 1. The key mod-
ifications are highlighted in green in the pseudocode. In-
tuitively, this schedule enforces a simple but important con-
straint: every Gaussian must “see” all views at least once af-
ter its opacity is reset before it can be judged as low-opacity
and removed. This prevents premature pruning caused by
partial-view updates, reduces the risk of deleting valid ge-
ometry in large scenes, and yields more stable training tra-
jectories across all compared methods on a single 80 GB
GPU. As shown in the comparison Fig 14, the modified
schedule preserves thin structures and distant geometry sig-
nificantly better than the original pruning behavior.
Figure 14. Effect of the modified pruning schedule. Qualita-
tive comparison after the first opacity reset iteration (9000) and
pruning. left: Opacity reset iteration is 9,000, densify interval
is 300, and we visualized the result of 9,301 (9000+300+1) iter-
ation; Right: Opacity reset iteration is 9,000, image count and
densify interval is 1770, and we visualized the result of 10771
(9000+1770+1) iteration.
Algorithm 1 Optimized Training Strategy on Single GPU
w, h: width and height of the training images
M ←SfM Points
▷Positions
N ←Images Count
i ←0
▷Iteration Count
OpacityIter ←0
▷Record the Last Opacity Reset
if IsRefinementIteration(i) then
if i −OpacityIter > N then
for all Gaussians (µ, Σ, c, α) in (M, S, C, A) do
if α < ϵ or IsTooLarge(µ, Σ) then ▷Pruning
RemoveGaussian()
end if
if ∇pL > τp then
▷Densification
if ∥S∥> τS then
▷Over-reconstruction
SplitGaussian(µ, Σ, c, α)
else
▷Under-reconstruction
CloneGaussian(µ, Σ, c, α)
end if
if i ÷ opacity reset interval == 0 then
ResetOpacity()
OpacityIter ←i ▷Record the Last
Opacity Reset Iteration
end if
end if
end for
end if
end if
i ←i + 1
8. Benchmark Metrics
Novel view synthesis (NVS). Given a set of test images
{In}N
n=1 and the corresponding rendered images {ˆIn}N
n=1,
we evaluate NVS quality using PSNR, SSIM, and LPIPS.
4

<!-- page 16 -->
The peak signal-to-noise ratio (PSNR) is defined as:
PSNR(I, ˆI) = 10 log10
 
MAX2
MSE(I, ˆI)
!
.
(1)
Structural similarity (SSIM) between I and ˆI is defined
as
SSIM(I, ˆI) =
(2µIµˆI + C1)(2σI ˆI + C2)
(µ2
I + µ2
ˆI + C1)(σ2
I + σ2
ˆI + C2).
(2)
LPIPS is defined as:
LPIPS(I, ˆI) =
X
l
wl
ˆϕl(I) −ˆϕl(ˆI)

2
2 .
(3)
Geometry accuracy. We evaluate geometric accuracy by
comparing a predicted point set P (e.g., sampled from the
reconstructed mesh) with a ground-truth point set G. For a
point p ∈P, let
d(p, G) = min
g∈G ∥p −g∥2,
(4)
and analogously d(g, P) for g ∈G.
Given a distance
threshold τ, we define precision and recall as
Precision(τ) = |{p ∈P | d(p, G) < τ}|
|P|
,
(5)
Recall(τ) = |{g ∈G | d(g, P) < τ}|
|G|
.
(6)
The F1-score is the harmonic mean of precision and recall:
F1(τ) = 2 · Precision(τ) · Recall(τ)
Precision(τ) + Recall(τ) .
(7)
In all experiments we use a fixed threshold τ
=
0.25m, 0.5m, 0.75m for fair comparison across methods
and scenes.
TCC implementation details.
For completeness, we de-
tail how the four TCC components in Sec. 4 are computed.
For each fixed test viewpoint vk and T = 3 time slots t ∈
{1, 2, 3}, we render albedo images A(k)
t
∈[0, 1]H×W ×3.
We first form the temporal mean albedo
¯A(k)(x, c) = 1
T
T
X
t=1
A(k)
t
(x, c),
(8)
where x ∈Ωindexes pixels and c ∈{1, 2, 3} is the color
channel.
We compute MAE and RMSE between each slot and the
temporal mean and averaging over the three slots gives
MAE
(k) = 1
T
T
X
t=1
MAE(k)
t
, RMSE
(k) = 1
T
T
X
t=1
RMSE(k)
t
.
(9)
We map these errors to [0, 1] consistency scores via
TCC(k)
MAE = 1 −clip[0,1]
 10 MAE
(k)
,
(10)
TCC(k)
RMSE = 1 −clip[0,1]
 10 RMSE
(k)
,
(11)
where clip[0,1](x) = min(max(x, 0), 1), ensures that ex-
tremely large errors are saturated and the resulting scores
remain in the valid range [0, 1] without becoming negative.
Similarly, we compute SSIM and LPIPS between each
slot and the temporal mean:
s(k)
t
= SSIM
 A(k)
t
, ¯A(k)
,
ℓ(k)
t
= LPIPS
 A(k)
t
, ¯A(k)
, (12)
and average over time,
s(k) = 1
T
T
X
t=1
s(k)
t
,
ℓ
(k) = 1
T
T
X
t=1
ℓ(k)
t
.
(13)
The corresponding consistency scores are
TCC(k)
SSIM = s(k),
TCC(k)
LPIPS = 1 −clip[0,1]
 ℓ
(k)
,
(14)
since SSIM is already in [0, 1] while LPIPS is a distance that
we invert.
These four components are then combined as defined in
the main paper.
9. Additional Results
We first compare our ground-truth depth and normals
against state-of-the-art monocular depth [20, 41] and nor-
mal [41, 54] estimators trained on generic indoor or street-
view datasets.
As illustrated in Fig. 20, these methods
struggle significantly under UAV oblique viewpoints: they
systematically oversmooth fac¸ades and roof structures, blur
depth discontinuities at building edges, and fail to recover
thin elements such as railings, dormers, and small rooftop
equipment.
Fig. 16 shows bird’s-eye visualizations rendered with the
SuperSplat web viewer for several large-scale scenes. We
compare geometry-based methods 2DGS [8] and PGSR [1]
and observe that both methods exhibit under-reconstruction:
distant buildings are partially missing, roof geometry is
over-smoothed, and fine structures are either broken or en-
tirely absent. For NVS-based baselines [11, 25, 56], strong
illumination changes (e.g., moving cast shadows, partly
cloudy conditions) lead to view-dependent artifacts such as
ghosting along building contours and inconsistent shading
across adjacent views. Quantitative and qualitative TCC-
albedo of Ref-GS [62], Ref-Gaussian [53] and GS-IR [19]
are reported in Fig. 17, 18, and 19 highlighting the tem-
poral instability of albedo estimates under challenging illu-
mination, while geometry comparison are demonstrated in
Fig. 21.
5

<!-- page 17 -->
Figure 15. Additional geometry ground-truth results. We visualize the geometry ground truth of the Town scene, which features dense
high-rise blocks, narrow streets, and deeply occluded courtyards. Our LiDAR-guided MVS pipeline produces a metrically accurate and
topologically complete reference surface that preserves fine fac¸ade details and inner-structure layout, providing a reliable ground-truth
benchmark for evaluating large-scale urban reconstruction methods in highly cluttered environments.
Figure 16. Additional bird’s-eye visualizations rendered with the SuperSplat web viewer.
6

<!-- page 18 -->
Figure 17. Additional TCC-albedo visualization of Ref-GS.
7

<!-- page 19 -->
Figure 18. Additional TCC-albedo visualization of Ref-Gaussian.
8

<!-- page 20 -->
Figure 19. Additional TCC-albedo visualization of GS-IR.
9

<!-- page 21 -->
Figure 20. Qualitative comparison between SOTA monocular depth/normal estimators on SkyLume dataset.
Figure 21. Additional bird’s-eye geometry visualizations.
10
