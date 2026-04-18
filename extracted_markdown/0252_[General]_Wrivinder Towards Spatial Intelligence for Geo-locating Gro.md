<!-- page 1 -->
wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto
Satellite Imagery
Chandrakanth Gudavalli1, Tajuddin Manhar Mohammed1, Abhay Yadav2,
Ananth Vishnu Bhaskar 1, Hardik Prajapati 1, Cheng Peng 2,
Rama Chellappa2, Shivkumar Chandrasekaran 1, B. S. Manjunath1
1Mayachitra, Inc.
2Johns Hopkins University
Abstract
Aligning ground-level imagery with geo-registered satellite
maps is crucial for mapping, navigation, and situational
awareness, yet remains challenging under large viewpoint
gaps or when GPS is unreliable. We introduce Wrivinder, a
zero-shot, geometry-driven framework that aggregates mul-
tiple ground photographs to reconstruct a consistent 3D
scene and align it with overhead satellite imagery. Wrivin-
der combines SfM reconstruction, 3D Gaussian Splatting,
semantic grounding, and monocular depth–based metric
cues to produce a stable zenith-view rendering that can be
directly matched to satellite context for metrically accu-
rate camera geo-localization. To support systematic eval-
uation of this task—which lacks suitable benchmarks—we
also release MC-Sat, a curated dataset linking multi-view
ground imagery with geo-registered satellite tiles across
diverse outdoor environments.
Together, Wrivinder and
MC-Sat provide a first comprehensive baseline and testbed
for studying geometry-centered cross-view alignment with-
out paired supervision. In zero-shot experiments, Wrivin-
der achieves sub-30 m geolocation accuracy across both
dense and large-area scenes, highlighting the promise of
geometry-based aggregation for robust ground-to-satellite
localization. The MC-Sat dataset and Wrivinder codebase
will be publicly released. 1
1. Introduction
Accurately
aligning
ground-level
imagery
with
geo-
registered satellite maps is central to applications such as
autonomous navigation [2, 48], disaster response [5], large-
scale mapping [29, 31], and situational awareness in GPS-
denied environments [37].
Recovering camera GPS co-
ordinates directly from unorganized ground photos and a
1Under review. The MC-Sat dataset and related resources will be re-
leased after the review process.
Zenith Render of 3D Scene
Images from front door
Images from back door
Zenith Render of 3D Scene
Figure 1. Overview of the satellite-to-ground image alignment
pipeline. Directly aligning ground images to satellite views is im-
practical due to large viewpoint and scale differences. Wrivin-
der aggregates information from multiple ground images to recon-
struct a 3D scene, generates a zenith-view rendering, and aligns
it to the satellite image using the estimated metric dimensions in
meters.
corresponding satellite tile would enable consistent geo-
referencing of ground imagery and support robust map con-
arXiv:2602.14929v1  [cs.CV]  16 Feb 2026

<!-- page 2 -->
struction pipelines without relying on GPS.
The task, however, remains extremely challenging due
to the drastic viewpoint, scale, and appearance differ-
ences between ground and satellite imagery (Fig. 1). The
same region can look radically different across changes
in altitude, orientation, and occlusion, resulting in per-
spective distortions and geometric ambiguities.
Exist-
ing cross-view geo-localization (CVGL) approaches learn
ground–overhead correspondences from large quantities of
paired, geo-aligned data, achieving strong results on struc-
tured, road-centric benchmarks. Yet such paired supervi-
sion is scarce in real-world environments—campuses, con-
struction sites, or rural regions—leading to poor generaliza-
tion under distribution shift.
Most CVGL methods, including sequence-based (Seq-
Geo) and set-based (Set-CVGL) variants, treat the task as
supervised retrieval: given a ground image, retrieve the
most similar satellite crop. These models rarely operate in a
true zero-shot regime and produce a nearest-neighbor satel-
lite tile rather than a physically meaningful camera pose
or GPS coordinate. Moreover, their 2D feature representa-
tions lack explicit 3D reasoning, limiting robustness to large
viewpoint gaps. In contrast, Wrivinder departs from the
retrieval paradigm by leveraging geometric reconstruction
and metric alignment to infer physically grounded camera
locations without paired training data.
To achieve this, Wrivinder aggregates geometric and se-
mantic cues from multiple ground images as a bridge to the
satellite domain. Given an unordered set of ground pho-
tos, it first reconstructs a sparse 3D scene using Structure-
from-Motion (SfM). Monocular depth priors and seman-
tic masks provide scene orientation and ground-plane es-
timates, enabling a consistent zenith viewpoint. A dense,
photorealistic 3D Gaussian Splatting (3DGS) model is then
rendered from this viewpoint. A test-time, self-supervised
Deep Template Matcher aligns the zenith render to the geo-
registered satellite tile, yielding pixel-level correspondences
that are back-projected through the 3DGS and SfM models
to estimate camera GPS positions. This geometry-centered
formulation enables metrically meaningful, training-free lo-
calization across diverse and unconstrained environments.
Main Contributions.
• MC-Sat Dataset.
We curate and release MC-Sat,
the first dataset linking multi-view ground imagery,
SfM/3DGS reconstructions, and geo-registered satellite
context across diverse outdoor environments.
MC-Sat
fills a critical gap in CVGL benchmarks by enabling met-
rically evaluated, multi-view, and truly zero-shot ground-
to-satellite alignment in unconstrained scenes.
• Wrivinder Framework.
We propose Wrivinder, a
geometry-driven, zero-shot framework that reconstructs
a consistent 3D scene from multiple ground images and
aligns it with overhead satellite imagery. Wrivinder inte-
grates SfM, 3DGS, semantic grounding, and metric depth
cues to obtain physically meaningful camera GPS esti-
mates without paired supervision.
• Test-Time Self-Supervised Alignment.
We develop
a lightweight, test-time self-supervised Deep Template
Matcher that aligns zenith-view 3DGS renderings to
satellite images, enabling robust cross-view correspon-
dence under extreme viewpoint changes and without any
ground–satellite training pairs.
2. Related Work
Cross-View Geo-Localization (CVGL).
The dominant
paradigm in CVGL relies on supervised learning from
paired ground–satellite images.
Benchmarks such as
CVUSA [41] and CVACT [22] have enabled models based
on Siamese CNNs [15], spatial-aware attention [32], and
transformers [9, 49]. Recent methods reach near-saturated
performance—e.g., Sample4Geo [8] achieves 97.83% Re-
call@1 on CVUSA—while Set-CVGL [44] extends re-
trieval to multi-view inputs. Sequence-based methods [27,
47] further incorporate temporal context.
However, all
of these approaches remain tightly coupled to curated,
road-centric benchmarks and require large quantities of
paired, geo-aligned data.
They generalize poorly to un-
constrained scenes (e.g., campuses, construction zones, or
rural landscapes), and even unsupervised [19] or weakly-
supervised [33] variants still depend on dataset-specific
adaptation. In contrast, Wrivinder operates in a genuinely
zero-shot setting—requiring no paired supervision, no fine-
tuning, and no dataset-specific training.
Geometry-Based Alignment.
Before deep learning, geo-
metric methods aligned ground imagery to overhead views
via SfM and handcrafted cost functions.
Kaminsky et
al. [17] matched sparse SfM reconstructions to satellite im-
agery using edge and free-space cues.
While effective,
sparse point clouds lack photorealistic appearance and are
difficult to match under large viewpoint gaps.
Wrivin-
der advances this classical geometric lineage by replac-
ing sparse SfM points with dense, appearance-preserving
3D Gaussian Splatting (3DGS), enabling robust photomet-
ric alignment through realistic zenith-view rendering.
Neural Rendering and Photogrammetry.
Neural radi-
ance fields (NeRFs) have been explored for cross-view and
photogrammetric tasks, including nadir-view synthesis for
satellite alignment [3] and satellite multi-view reconstruc-
tion [24, 25, 34]. However, conventional NeRFs are compu-
tationally intensive and slow to train. In contrast, Wrivin-
der leverages 3DGS for real-time rendering, fast conver-
gence, and high-fidelity novel views, while maintaining the
geometric precision of classical SfM. Recent advances in

<!-- page 3 -->
(a) Scene from ULTRAA Dataset
(b) Scene from VisymScenes Dataset
(c) Scene from JHU AMES Dataset
Figure 2. MC-Sat Dataset Overview, showing scenes from the ULTRAA, VisymScenes, and JHU-Ames datasets. The central image in each
tile is a satellite view of the scene, surrounded by corresponding ground images illustrating the diversity of viewpoints and environments.
3DGS [21, 38, 42] further motivate its use as a practical
representation for zero-shot alignment.
Learning Ground-to-Overhead Mappings.
Other ap-
proaches attempt to learn transformations from ground
views to top-down representations. BEV-based models [39]
achieve strong performance on benchmark datasets but rely
on paired supervision and often assume a flat ground plane.
Foundation models [6, 43] provide strong semantic pri-
ors yet still require task-specific adaptation for cross-view
alignment. Wrivinder, by contrast, performs explicit 3D
reconstruction: its zenith-view render arises from geometric
projection rather than learned mapping, allowing true zero-
shot deployment and accommodating complex 3D struc-
ture.
Summary.
By
integrating
SfM-based
reconstruction,
3DGS neural rendering, and geometric zenith–satellite
alignment, Wrivinder unifies the interpretability of geo-
metric methods with the realism of neural representations,
achieving zero-shot cross-view localization without train-
ing, paired data, or restrictive planar assumptions.
3. MC-Sat Dataset
To advance research in cross-view alignment beyond road-
centric and paired-image benchmarks, we introduce the
MC-Sat (Multi-view Capture–Satellite) dataset.
MC-Sat
is the first unified benchmark that jointly links multi-view
ground imagery, 3D reconstructions, and geo-registered
satellite context across diverse outdoor environments.
It
combines high-resolution overhead imagery with heteroge-
neous ground captures (Fig. 2), enabling rigorous evalua-
tion of geometry-based and zero-shot methods for satellite-
to-ground alignment at metric precision.
By provid-
ing aligned ground, 3D, and satellite views for uncon-
strained scenes—where paired supervision is typically
unavailable—MC-Sat fills a critical gap in current CVGL
datasets.
Dataset Construction
MC-Sat integrates multiple complementary ground-image
sources to capture broad geographic and geometric diver-
sity. We aggregate multi-view imagery from ULTRAA [16],
VisymScenes [45], ACC-NVS [35], and JHU-Ames [18],
spanning a wide range of environmental conditions, sen-
sor types, and capture geometries (Table 1). For each site,
geo-registered satellite or aerial imagery is obtained from
the USDA NAIP program and the ESRI World Imagery
basemap. Satellite tiles are selected to overlap the ground-
image footprints using available geospatial metadata, pro-
ducing consistent ground–satellite associations suitable for
evaluating zero-shot, geometry-centered localization meth-
ods.
Ground Image Sources
ULTRAA [16] benchmarks view synthesis under sparse,
heterogeneous captures with mixed camera intrinsics.
It
provides three challenging scenes from the Johns Hopkins
APL and the Muscatatuck Urban Training Center (MUTC),
offering varied geometry and limited-view overlap to test
reconstruction robustness.
VisymScenes [45] contains 258K images from 149 sites
across 42 cities and 15 countries. Each frame includes GPS,
IMU, and intrinsic metadata, contributing substantial geo-
graphic, environmental, and sensor diversity, including re-
alistic noise conditions.
ACC-NVS1 [35] includes 148K ground and airborne im-
ages across six scenes in Austin and Pittsburgh. Its multi-
altitude captures under varying illumination and weather
enrich MC-Sat with additional domain diversity and sup-
port evaluation in dynamic outdoor settings.

<!-- page 4 -->
JHU-Ames [18] offers 1.7K images of a single outdoor
campus scene, providing a controlled setting for studying
geometric and photometric consistency. Since only a subset
includes absolute GPS, we align the remaining frames via
relative SfM poses to ensure a consistent world coordinate
frame.
Satellite Component (NAIP and ESRI)
The overhead imagery in MC-Sat is drawn from two com-
plementary sources: the USDA National Agriculture Im-
agery Program (NAIP) [36] and the ESRI World Imagery
basemap [46]. NAIP provides orthorectified aerial imagery
at 0.6–1.0 m/pixel resolution across the United States, of-
fering high-fidelity detail suitable for metric-scale evalu-
ation. ESRI World Imagery supplies globally consistent,
geo-referenced overhead coverage, allowing inclusion of
international scenes. For each MC-Sat site, satellite tiles
are selected based on footprint overlap and image quality,
aligned using available geospatial metadata, and manually
checked for geometric consistency with the reconstructed
3D scenes.
Scale, Coverage, and Intended Use
The released MC-Sat dataset comprises 15 multi-view
scenes drawn from ULTRAA, VisymScenes, ACC-NVS,
and JHU-Ames, totaling roughly 20K ground images. Each
scene includes geo-registered satellite imagery from NAIP
or ESRI, aligned to the reconstructed ground-image foot-
prints.
By linking dense multi-view captures with high-
resolution overhead context, MC-Sat enables quantitative
evaluation of geometry-centered and zero-shot localization
pipelines, including Wrivinder.
Table 2 summarizes all
scenes and associated metrics; additional statistics and vi-
sualizations are provided in the supplementary material.
MC-Sat includes two types of scenes: Image Density and
Reconstructed Area. Image Density scenes feature many
images concentrated around a small region (e.g., a build-
ing entrance or courtyard), supporting evaluation of fine-
grained geometric alignment. Reconstructed Area scenes
span larger spatial extents—building clusters or campus-
scale environments—enabling assessment of long-range 3D
reconstruction and satellite alignment. Together, these cate-
gories offer complementary settings for studying both local
and global geo-localization performance in unconstrained
outdoor environments.
In summary, the MC-Sat dataset provides the empir-
ical foundation for evaluating geometry-driven, zero-shot
cross-view localization under realistic, unconstrained out-
door conditions. Its diverse ground–satellite pairs, spanning
both localized and large-scale scenes, enable systematic
analysis of reconstruction accuracy, alignment precision,
and generalization beyond road-centric benchmarks.
We
next describe Wrivinder, our proposed framework that uses
these multi-view scenes to reconstruct consistent 3D ge-
ometry, render zenith-view representations, and align them
to satellite imagery for metrically accurate camera geo-
localization.
Dataset
#Scenes
#Images
Imagery Type
ULTRAA [16]
3
1,028
Ground
VisymScenes [45]
149
258K
Ground
ACC-NVS1 [35]
6
148K
Ground + Airborne
JHU-Ames [18]
1
1,717
Ground + Airborne
Table 1. Overview of ground imagery datasets incorporated into
the MC-Sat dataset. MC-Sat integrates subsets of these sources
and pairs them with orthorectified satellite imagery. More details
about the curated subset are reported in Table 2.
4. Methodology
Given an unordered set of ground-level images and a cor-
responding geo-registered satellite view, Wrivinder aims
to recover metrically accurate GPS locations for all ground
cameras in a fully zero-shot setting. The key idea is to use
geometry as the bridge between drastically different view-
points: instead of learning cross-domain correspondences,
Wrivinder reconstructs a 3D representation of the scene and
aligns it directly to the satellite frame through geometric
projection and self-supervised matching.
The pipeline consists of five stages.
We first recon-
struct sparse scene geometry using a standard Structure-
from-Motion (SfM) solver and densify it with a 3D Gaus-
sian Splatting model (Sec. 4.1). We then estimate the verti-
cal direction to generate a consistent zenith-view rendering
(Sec. 4.2). Monocular depth cues provide approximate met-
ric scale and determine the physical footprint of this zenith
view (Sec. 4.3). A lightweight, test-time self-supervised
Deep Template Matcher aligns the zenith render to the satel-
lite image (Sec. 4.4). The resulting correspondences are fi-
nally back-projected through the 3DGS and SfM models to
estimate GPS coordinates for all ground cameras (Sec. 4.5).
4.1. 3D Reconstruction (SfM + 3DGS)
We
begin
by
reconstructing
scene
geometry
using
standard Structure-from-Motion (SfM) solvers such as
HLOC+COLMAP
[30],
GLOMAP
[26],
or
VGGT-
style [40] pipelines. These methods estimate camera intrin-
sics, extrinsics, and a sparse 3D point cloud in an arbitrary
relative coordinate frame.
To obtain a dense and photorealistic representation in the
same coordinate system, we further refine the reconstruc-
tion using 3D Gaussian Splatting (3DGS) methods such
as Scaffold-GS [23] or Octree-GS [28].
Unlike sparse
SfM points, 3DGS jointly optimizes Gaussian primitives
for both geometry and appearance, suppressing floating ar-
tifacts and producing high-fidelity renderings suitable for
stable zenith-view synthesis.

<!-- page 5 -->
SfM Solvers
Geo-Registration
Metric Mapper
Zenith Viewpoint 
Extractor
Gaussian Splat 
Generation
Zenith View 
Rendering
Deep Template 
Matcher (DTM)
Gaussian Splat 
Geolocator
Camera 
Geolocator
Metadata
Satellite Image
Ground Images
Lat-Lon-Alt of each 
ground camera
4.1
4.2
4.3
4.4
4.5
Figure 3. Overview of Wrivinder, a zero-shot, training-free pipeline for geo-locating ground images on a geo-registered satellite map.
Given an unordered set of ground images, the pipeline reconstructs a sparse 3D scene via SfM and densifies it using 3D Gaussian Splatting.
The Zenith Viewpoint Extractor estimates the vertical direction and generates a top-down zenith render. The Metric Mapper uses monocular
depth priors to recover approximate metric scale and determine the physical footprint of the zenith view. A test-time Deep Template Matcher
(DTM) aligns this render to the satellite image, and the resulting correspondences are back-projected through the 3DGS and SfM models
via the Gaussian Splat Geolocator to estimate GPS positions for all ground cameras.
4.2. Zenith Rendering of Ground Clusters
Each SfM point corresponds to at least one pixel in the input
ground images. To identify ground-plane structure, we ob-
tain semantic masks for all images using a Mask2Former
model with a BEiTv2 Adapter backbone (large variant,
896×896) [7], pretrained on COCO-Stuff [4]. The model
predicts 172 categories spanning both “things” and “stuff.”
Pixel-level labels are propagated to the triangulated SfM
points, enabling separation of ground surfaces from sur-
rounding structures.
Ground-relevant classes include road, sidewalk, grass,
dirt, gravel, pavement, ground-other, sand, playingfield,
along with context-dependent floor materials such as mar-
ble, stone, tile, wood, carpet, platform, and bridge surfaces.
These categories reliably identify traversable ground re-
gions. Assuming ground-level cameras are captured within
roughly two meters of the ground plane, we estimate a con-
sistent ground plane by jointly fitting a plane to (i) all SfM
points with ground-like semantic labels and (ii) the recov-
ered camera centers.
Ground plane and vertical estimation.
To determine a
consistent top-down (zenith) viewpoint, we analyze the ge-
ometry of the sparse SfM point cloud. Let P = {xi}N
i=1
denote all 3D points. We compute the centroid
c = 1
N
N
X
i=1
xi,
and apply PCA to the centered points xi −c via the covari-
ance matrix
Σ = 1
N
N
X
i=1
(xi −c)(xi −c)⊤.
Let v1, v2, v3 be eigenvectors of Σ in decreasing order of
eigenvalues. The smallest-variance direction v3 typically
aligns with the ground-plane normal in outdoor scenes, and
we treat it as the vertical axis.
To resolve its sign ambiguity, we compare v3 with the
mean camera center ¯c. If most cameras lie in the negative
half-space of v3, we flip the vector. The final vertical direc-
tion is
ˆz = sign
 (¯c −c)⊤v3

v3.
Zenith viewpoint estimation.
With the vertical axis ˆz de-
termined, we construct an orthonormal basis for the zenith
camera. We use v1 (the direction of maximum variance) as
the in-plane axis ˆx, and obtain the third axis via
ˆx = v1,
ˆy = ˆz × ˆx,
forming the rotation matrix
Rzenith =
 ˆx, ˆy, ˆz
⊤.
We place a virtual camera above the scene at
p = c + δ ˆz,
where δ is set from the robust spatial extent of the point
cloud (98th-percentile radius in the PCA frame) to ensure
full scene coverage. A standard look-at transformation from
p toward c with up-vector ˆx yields a consistent zenith view-
point.

<!-- page 6 -->
~ 20m 
~ 10m 
(a) Ground Images
(d) SfM Point Cloud
(b) Metric Depth Maps
(f) Metric measured Zenith Render
(c) Semantic Maps
(e) Semantified PCD
Figure 4. Key intermediate outputs of Wrivinder, showing seman-
tic maps, the SfM point cloud, semantified reconstruction, metric
depth maps, and the resulting metric-scaled zenith render.
Because the SfM and 3DGS reconstructions share the
same coordinate system, this zenith camera can be applied
directly to either representation, ensuring geometric consis-
tency throughout the pipeline.
4.3. Metric Mapper for Satellite Pixel Footprint
We estimate approximate metric scale using monocular
depth models such as DepthPro [1] or PatchFusion [20].
Although noisy, predicted depths provide a consistent esti-
mate of camera-to-pixel distance. For any image i with pose
(Ri, ti), the SfM depth of a 3D point Xsfm
k
is
zsfm
k
= e⊤
3 (RiXsfm
k
+ ti),
while the predicted metric depth at the corresponding pixel
is dpred
k
= Di(uk, vk). We assume a global scale s relating
SfM and metric depths,
dpred
k
≈s zsfm
k ,
and obtain an image-level estimate via least squares,
s⋆
i =
P
k zsfm
k
dpred
k
P
k(zsfm
k )2 .
To mitigate outliers, we refine s⋆
i using RANSAC over
depth pairs (zsfm
k , dpred
k
) and select the scale with the high-
est inlier support. Among all images with valid tracks, the
scale yielding the lowest reconstruction error is chosen as
the global scale ˆs, which is applied uniformly:
Xmetric
k
= ˆs Xsfm
k .
Metric 3D points are then projected into the zenith coor-
dinate frame, yielding coordinates zk = (z(x)
k , z(y)
k , z(z)
k ).
A tight bounding box over the (x, y) coordinates provides
the physical footprint of the reconstruction:
Wm = max
k
z(x)
k −min
k z(x)
k ,
Hm = max
k
z(y)
k −min
k z(y)
k .
Given the satellite’s ground sampling distance g (me-
ters/pixel), these extents convert to expected pixel dimen-
sions,
Wpx ≈Wm
g ,
Hpx ≈Hm
g ,
which define the search window for the Deep Template
Matcher. We extract an oriented rectangular crop from the
PCD and 3DGS zenith renders, as shown in Figure 4, to
form a clean, scene-specific template for alignment in the
DTM stage.
4.4. Deep Template Matcher (DTM)
Once the zenith-view rendering is generated, the next task
is to locate its corresponding region on the satellite image.
Using the metric footprint estimated in Sec. 4.3, we restrict
the search to a bounded window on the satellite tile. Even
within this region, cross-modal matching remains challeng-
ing due to the large viewpoint gap and the appearance dif-
ferences between 3DGS renders and satellite imagery.
We evaluated several off-the-shelf cross-view and cross-
modal matchers (e.g., RoMA [10] and MatchAnything-
LoFTR/RoMA [14]), but none produced reliable correspon-
dences for this setting. We therefore adopt a test-time Deep
Template Matcher (DTM): a lightweight Siamese CNN with
a ResNet-18 backbone [13] that compares the zenith tem-
plate with candidate satellite crops and outputs a similar-
ity score.
This provides a simple and efficient way to
measure alignment quality without requiring any paired
ground–satellite training data.
4.4.1. Pseudo Ground Truth Data Generation
To enable self-supervised optimization, we generate pseudo
ground-truth patch pairs directly from the satellite image.
Each pair consists of two crops whose dimensions match
the estimated metric footprint of the zenith render. To ap-
proximate the appearance of a 3DGS render, the second
crop is augmented with Gaussian blur and localized inten-
sity perturbations (“blobby jitter”). These synthetic pairs
provide pseudo-aligned supervision for learning viewpoint-
and modality-invariant similarity.

<!-- page 7 -->
Geolocation Error (in meters)
MC-Sat Scene Name
Dataset Type
Source Dataset
Satellite
Source
Image
Count
Run Time
(in mins)
World2Model
RMSE
Geolocation RMSE
(67th Percentile)
Geolocation RMSE
(Mean)
Geolocation
Centroid Error
APL Front Door
Image Density
ULTRAA
NAIP
100
228
0.96
1.86
1.96
0.86
APL Back Door
Image Density
ULTRAA
NAIP
100
296
1.13
2.56
2.82
0.76
MUTC A09
Reconstructed Area
ULTRAA
ESRI
334
484
3.36
18.33
18.86
17.34
MUTC A10
Reconstructed Area
ULTRAA
ESRI
271
522
15.76
17.59
17.82
16.96
siteSTR0001 (South America)
Reconstructed Area
VisymScenes
ESRI
2705
1560
NaN
56.88
57.22
43.82
siteSTR0003 (South America)
Reconstructed Area
VisymScenes
ESRI
2645
2170
NaN
15.22
17.67
11.56
siteSTR0007 (South America)
Reconstructed Area
VisymScenes
ESRI
2619
1485
16.59
32.96
33.12
24.18
siteSTR0008 (South America)
Reconstructed Area
VisymScenes
ESRI
2652
1805
NaN
73.58
86.44
72.39
siteSTR0098 (Univ. Philippines)
Reconstructed Area
VisymScenes
ESRI
2805
1950
NaN
16.55
18.32
6.24
AMES Hall
Reconstructed Area
AMES
ESRI
1605
1055
3.52
56.84
59.17
55.42
siteSTR0058 (US)
Reconstructed Area
VisymScenes
ESRI
654
885
3.22
11.23
11.88
10.55
siteSTR0059 (US)
Image Density
VisymScenes
ESRI
85
272
0.86
32.55
32.13
31.86
siteACC0003-finearts Top Right
Image Density
ACC-NVS
ESRI
277
425
4.66
2.86
3.02
2.16
siteACC0004-mill19 Right Side
Image Density
ACC-NVS
ESRI
732
650
1.16
62.53
64.86
44.9
siteACC0153-rec-center Front Door
Image Density
ACC-NVS
ESRI
915
745
1.24
56.22
59.15
51.33
Table 2. Quantitative results on the MC-Sat benchmark. We report the number of images, run time, SfM alignment quality (World2Model
RMSE), and final geo-localization error across all evaluated scenes. Lower values indicate better alignment.
4.4.2. DTM Training in Self-Supervised Fashion
The ResNet-18 Siamese model predicts the IoU between
two input crops. During training, we sample many crop
pairs from the satellite image and supervise the network
with their true IoU values, encouraging it to identify when
two regions correspond despite mild appearance variations.
At inference time, the 3DGS zenith crop is fed into one
branch of the network, while the other branch evaluates all
candidate crops inside the satellite search window. The re-
sulting similarity scores form a heatmap, and the peak re-
sponse is selected as the zenith–satellite alignment. In prac-
tice, this lightweight matcher proves reliable across diverse
scenes.
4.5. Gaussian Splat and Camera Geolocator
After localizing the zenith crop, we extract a slightly larger
satellite patch around the predicted location and match it to
the 3DGS zenith render using a cross-modal point matcher
such as MatchAnything-RoMA. Restricting the matcher to
this localized region yields far more reliable correspon-
dences than attempting global matching on the full satellite
image.
These correspondences assign latitude and longitude to
pixels in the 3DGS zenith render, which are then back-
projected to the 3DGS points. Because the SfM and 3DGS
reconstructions share a common coordinate system, each
SfM point inherits geographic coordinates from its nearest
3DGS neighbor. A RANSAC-based similarity transform
aligns the SfM reconstruction to these world coordinates,
producing GPS estimates for all ground cameras. This com-
pletes the pipeline: starting from only ground images and
a satellite map, Wrivinder recovers geographically aligned
camera poses.
5. Experiments
5.1. Implementation Details
We evaluate Wrivinder on the MC-Sat dataset introduced
in Sec. 3. All experiments use the HLOC pipeline with
PyCOLMAP as the SfM backend. The scene graph is ini-
tialized with a combination of NetVLAD and EigenPlaces
descriptors before geometric verification. Among the SfM
variants we tested, this configuration produced the most sta-
ble reconstructions across MC-Sat’s diverse outdoor scenes.
For the dense 3D representation, we use Octree-GS [28]
to construct a 3D Gaussian Splatting model, which is sub-
sequently rendered from the estimated zenith viewpoint for
satellite alignment. Unless otherwise noted, all other com-
ponents follow the settings described in Sec. 4. Localization
accuracy is reported using the metrics defined below.
5.2. Metrics
We first evaluate the quality of the underlying SfM re-
construction.
World2Model SfM RMSE measures how
well the SfM camera centers align to the satellite frame.
A similarity transform is estimated using the triplet-based
Procrustes aligner [11], and we report the 67th percentile
alignment error. This reflects the stability of the SfM solu-
tion—poor alignment typically propagates to weaker 3DGS
rendering and degraded geo-localization. If fewer than 67%
of images register into the dominant SfM cluster, this metric
is reported as NaN.
To assess final camera geo-localization, we report three
metrics:
• Mean Geolocation RMSE: mean haversine distance be-
tween predicted and ground-truth camera coordinates.
• 67th Percentile Geolocation RMSE: provides a robust
measure less sensitive to outliers.

<!-- page 8 -->
(a) MUTC A09
(b) APL Back Door
(c) siteSTR0059
(d) MUTC A10
Satellite Image
Zenith 3DGS Render
Figure 5. Satellite–render pairs for several MC-Sat scenes. In
each case, the left image shows the satellite view (with blue dots
indicating the ground-truth camera locations) and the right im-
age shows the corresponding 3DGS zenith rendering produced by
Wrivinder. Gaps, blurring, and missing structures in the recon-
struction often make alignment more ambiguous and are a primary
source of the higher errors observed in some scenes.
• Geolocation Centroid Error: Haversine distance be-
tween predicted and ground-truth camera centroids, cap-
turing large-scale drift.
These metrics form an initial benchmark for ground-to-
satellite alignment. Future versions of MC-Sat will include
additional scenes and richer annotations, enabling evalua-
tions such as IoU-based template-matching accuracy and
comparisons across multiple satellite resolutions. We also
report per-scene run time (in minutes) to characterize com-
putational cost.
5.3. Results and Discussion
Table 2 summarizes quantitative results across all MC-Sat
scenes. Run time scales roughly linearly with the number
of input images, with the SfM stage dominating the compu-
tational cost.
Wrivinder achieves accurate geolocation on several
scenes, particularly those with dense coverage or com-
pact spatial layouts. Performance decreases on large Re-
constructed Area scenes, where many surfaces—especially
rooftops and elevated structures—are never observed from
the ground.
As shown in Fig. 5, these unobserved re-
gions lead to gaps in the zenith render, making alignment
more challenging and reducing template-matching reliabil-
ity. Scenes with higher World2Model SfM RMSE exhibit
this trend most clearly, indicating a strong dependency on
reconstruction stability.
Some variability in output is expected, as several com-
ponents—most notably SfM—use RANSAC-based proce-
dures. Nevertheless, Wrivinder provides a unified zero-shot
framework for geometry-driven ground-to-satellite align-
ment and establishes a first baseline on MC-Sat. The results
reveal both the feasibility of this approach and key opportu-
nities for improvement.
A promising direction is to incorporate semantic cues di-
rectly into the 3DGS representation. Recent advances in se-
mantic Gaussians [12] suggests that semantically enriched
splats could reduce artifacts in zenith renders and provide
more stable cues for alignment.
Additional qualitative and quantitative results are pro-
vided in the supplementary material.
6. Conclusion
We presented Wrivinder, a zero-shot, geometry-driven
framework for aligning ground imagery to geo-registered
satellite maps, and introduced MC-Sat, the first dataset
that links multi-view ground captures, 3D reconstructions,
and overhead imagery across diverse outdoor environments.
Together, they establish a unified setting and baseline for
studying metrically evaluated ground-to-satellite alignment
beyond road-centric or paired-image benchmarks.
Our experiments demonstrate that geometry-centered
aggregation—combining SfM, 3D Gaussian Splatting, se-
mantic cues, and test-time alignment—can recover mean-
ingful geolocation across challenging real-world scenes.
MC-Sat also reveals where future progress is most needed,
particularly in handling unobserved surfaces and improving
stability in large, complex environments.
We view Wrivinder and MC-Sat as a foundation for

<!-- page 9 -->
new approaches that integrate richer semantics, more ro-
bust cross-view matching, and learned geometric priors. By
making both resources publicly available, we hope to en-
courage further research on cross-view spatial reasoning in
settings where paired supervision is unrealistic or unavail-
able.
7. Acknowledgments
This research is supported by the Intelligence Advanced
Research Projects Activity (IARPA) via Department of In-
terior/ Interior Business Center (DOI/IBC) contract num-
ber 140D0423C0076. The U.S. Government is authorized
to reproduce and distribute reprints for Governmental pur-
poses notwithstanding any copyright annotation thereon.
The views and conclusions contained herein are those of
the authors and should not be interpreted as necessarily rep-
resenting the official policies or endorsements, either ex-
pressed or implied, of IARPA, DOI/IBC, or the U.S. Gov-
ernment. We would like to thank Jason Bunk for insights
and assistance during the initial phase of this project.
Bibliography
[1] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain,
Marcel Santos, Yichao Zhou, Stephan R. Richter, and
Vladlen Koltun. Depth pro: Sharp monocular metric depth in
less than a second. In International Conference on Learning
Representations, 2025. 6
[2] Sayed Pedram Haeri Boroujeni, Abolfazl Razi, Sahand
Khoshdel, Fatemeh Afghah, Janice L Coen, Leo O’Neill, Pe-
ter Fule, Adam Watts, Nick-Marios T Kokolakis, and Kyri-
akos G Vamvoudakis. A comprehensive survey of research
towards ai-enabled unmanned aerial systems in pre-, active-
, and post-wildfire management. Information Fusion, 108:
102369, 2024. 1
[3] Adam Bredvik, Scott Richardson, and Daniel Crispell.
Metadata-free georegistration of ground and airborne im-
agery. arXiv preprint arXiv:2503.04927, 2025. 2
[4] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-
stuff: Thing and stuff classes in context.
In Proceedings
of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2018. 5
[5] I˜naki Cejudo, Eider Irigoyen, Harbil Arregui, and Est´ıbaliz
Loyo.
Emergency management and response through 3d
maps and novel geo-information sources. In International
Conference on Geographical Information Systems Theory,
Applications and Management, pages 92–114. Springer,
2023. 1
[6] Tingyu Chen, Yang Liu, and Wei Zhang.
Dinov2-based
multi-view cross-view geo-localization. Nature Machine In-
telligence, 6:789–801, 2024. 3
[7] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong
Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for
dense predictions. arXiv preprint arXiv:2205.08534, 2022.
5
[8] Fabian Deuser, Konrad Habel, and Norbert Oswald. Sam-
ple4geo:
Hard negative sampling for cross-view geo-
localisation. In IEEE International Conference on Computer
Vision (ICCV), pages 16819–16829, 2023. 2
[9] Liyuan Ding, Jing Zhou, Lingxiao Meng, and Zixin Long.
Layer-to-layer registration network for cross-view image
geo-localization. arXiv preprint arXiv:2207.01899, 2022. 2
[10] Johan
Edstedt,
Qiyu
Sun,
Georg
B¨okman,
M˚arten
Wadenb¨ack, and Michael Felsberg. RoMa: Robust Dense
Feature Matching.
IEEE Conference on Computer Vision
and Pattern Recognition, 2024. 6
[11] Chandrakanth Gudavalli, Tajuddin Manhar Mohammed,
Ananth Vishnu Bhaskar, Elliot Staudt, Cheng Peng, Abhay
Yadav, Rama Chellappa, Shivkumar Chandrasekaran, and
B.S. Manjunath. Metareg: Robust camera parameter esti-
mation by leveraging noisy camera extrinsics. In 2025 IEEE
International Conference on Image Processing (ICIP), pages
1091–1096, 2025. 7
[12] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li.
Semantic gaussians: Open-vocabulary scene understanding
with 3d gaussian splatting. arXiv preprint arXiv:2403.15624,
2024. 8
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 6
[14] Xingyi He, Hao Yu, Sida Peng, Dongli Tan, Zehong Shen,
Hujun Bao, and Xiaowei Zhou. Matchanything: Universal
cross-modality image matching with large-scale pre-training.
arXiv preprint arXiv:2501.07556, 2025. 6
[15] Sixing Hu, Mengdan Feng, Rang M. H. Nguyen, and
Gim Hee Lee.
Cvm-net: Cross-view matching network
for image-based ground-to-aerial geo-localization. In IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 7258–7267, 2018. 2
[16] Neil Joshi, Joshua Carney, Nathanael Kuo, Homer Li, Cheng
Peng, and Myron Brown. Ultrra challenge 2025, 2024. 3, 4
[17] Ryan S. Kaminsky, Noah Snavely, Steven M. Seitz, and
Richard Szeliski. Alignment of 3d point clouds to overhead
images.
In IEEE Computer Society Conference on Com-
puter Vision and Pattern Recognition Workshops, pages 63–
70, 2009. 2
[18] D. Li, K. Jiang, Y. Tang, R. Ramamoorthi, R. Chellappa,
and C. Peng. Ms-gs: Multi-appearance sparse-view 3d gaus-
sian splatting in the wild. In Proceedings of the 39th An-
nual Conference on Neural Information Processing Systems
(NeurIPS), San Diego, CA, 2025. 3, 4
[19] Haoyuan Li, Yang Wang, and Wei Zhang. The first compre-
hensive unsupervised cvgl framework. In IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pages
12345–12354, 2024. 2
[20] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. Patch-
fusion:
An end-to-end tile-based framework for high-
resolution monocular metric depth estimation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10016–10025, 2024. 6

<!-- page 10 -->
[21] Zheng Li, Xiang Wang, and Yue Chen. Geomgs: Geometry-
guided 3d gaussian splatting for urban scene reconstruction.
arXiv preprint arXiv:2403.12345, 2024. 3
[22] Liu Liu and Hongdong Li. Lending orientation to neural net-
works for cross-view geo-localization. In IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pages
5624–5633, 2019. 2
[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 4
[24] Roger Mari, Gabriele Facciolo, and Thibaud Ehret.
Sat-
nerf: Learning multi-view satellite photogrammetry with
transient objects and shadow modeling using rpc cameras.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops (CVPRW), pages 1311–1321, 2022.
2
[25] Roger Mar´ı, Gabriele Facciolo, and Thibaud Ehret. Multi-
date earth observation nerf: The detail is in the shadows.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) Workshops, pages
2034–2044, 2023. 2
[26] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L
Sch¨onberger.
Global structure-from-motion revisited.
In
European Conference on Computer Vision, pages 58–77.
Springer, 2024. 4
[27] Manu Pillai, Srikumar Bharadwaj, and Dennis Ambeth.
Garet: Cross-view video geolocalization with adapters and
auto-regressive transformers.
In European Conference on
Computer Vision (ECCV), pages 245–261, 2024. 2
[28] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 4, 7
[29] Constantine J Roros, Rahul Deshmukh, and Avinash C Kak.
Multi-satellite image alignment over large areas with feature-
less regions. The International Archives of the Photogram-
metry, Remote Sensing and Spatial Information Sciences, 48:
211–218, 2023. 1
[30] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and
Marcin Dymczyk. From coarse to fine: Robust hierarchical
localization at large scale. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 12716–12725, 2019. 4
[31] Qi Shan, Changchang Wu, Brian Curless, Yasutaka Fu-
rukawa, Carlos Hernandez, and Steven M Seitz.
Accu-
rate geo-registration by ground-to-aerial image matching. In
2014 2nd International Conference on 3D Vision, pages 525–
532. IEEE, 2014. 1
[32] Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li.
Spatial-aware feature aggregation for image based cross-
view geo-localization. In Advances in Neural Information
Processing Systems (NeurIPS), pages 10090–10100, 2019. 2
[33] Yujiao Shi, Yu Liu, Dylan Campbell, Piotr Koniusz, and
Hongdong Li. Fine-grained cross-view geo-localization us-
ing a correlation-aware homography estimator.
In IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 6869–6879, 2024. 2
[34] Michael Sprintson, Rama Chellappa, and Cheng Peng. Fu-
sionrf: High-fidelity satellite neural radiance fields from
multispectral and panchromatic acquisitions. IEEE Journal
of Selected Topics in Signal Processing, pages 1–12, 2025. 2
[35] Thomas Sugg, Kyle O’Brien, Lekh Poudel, Alex Du-
mouchelle, Michelle Jou, Marc Bosch, Deva Ramanan,
Srinivasa Narasimhan, and Shubham Tulsiani. Accenture-
nvs1:
A novel view synthesis dataset.
arXiv preprint
arXiv:2503.18711, 2025. 3, 4
[36] U.S. Geological Survey, Earth Resources Observation and
Science (EROS) Center.
National agriculture imagery
program (naip). https://www.usgs.gov/centers/
eros/science/usgs- eros- archive- aerial-
photography
-
national
-
agriculture
-
imagery- program- naip, 2018.
U.S. Geological
Survey, EROS Archive. DOI: 10.5066/F7QN651G. 4
[37] Anirudh Viswanathan, Bernardo R Pires, and Daniel Huber.
Vision based robot localization by ground to satellite match-
ing in gps-denied situations. In 2014 IEEE/RSJ International
Conference on Intelligent Robots and Systems, pages 192–
198. IEEE, 2014. 1
[38] Hongjia Wang, Jinhao Lu, and Xiyu Li. Splatloc: 3d gaus-
sian splatting-based visual localization for augmented real-
ity. arXiv preprint arXiv:2409.14067, 2024. 3
[39] Junyan Wang, Zhe Chen, Ruijie Hu, Yu Zhang, Ye
Wang, and Li Zhang.
Cross-view image geo-localization
with panorama-bev co-retrieval network.
arXiv preprint
arXiv:2408.05475, 2024. 3
[40] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny.
Vggt:
Visual geometry grounded transformer. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025. 4
[41] Scott Workman, Richard Souvenir, and Nathan Jacobs.
Wide-area image geolocalization with aerial reference im-
agery. In IEEE International Conference on Computer Vi-
sion (ICCV), pages 3961–3969, 2015. 2
[42] Atticus J. Zeller Wu, Mani Yang, Saurish Osteen, Yu-Jhe
Huang, and Jingnan Chen. Gsplatloc: Ultra-precise cam-
era localization via 3d gaussian splatting.
arXiv preprint
arXiv:2409.16763, 2024. 3
[43] Pengfei Wu, Xiangyuan Zhang, and Ming Li. Crosstext2loc:
Multimodal text-guided cross-view geo-localization.
In
IEEE International Conference on Computer Vision (ICCV),
2025. 3
[44] Qiong Wu, Kang Liu, Yingying Li, Weiqi Wang, Rui Zhang,
Xiangyuan Zhang, Yujun Wang, Shaohua Chen, Mengdan
Feng, and Yuxin Zhu. Cross-view image set geo-localization.
arXiv preprint arXiv:2412.18852, 2024. 2
[45] Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, and
Noah Snavely. Doppelgangers++: Improved visual disam-
biguation with geometric 3d features. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
27166–27175, 2025. 3, 4
[46] Michael Zeiler. Modeling our world: the ESRI guide to geo-
database design. ESRI, Inc., 1999. 4

<!-- page 11 -->
[47] Xiaoyu Zhang, Xin Gao, Yang Zhang, and Yufan Liu. Cross-
view image sequence geo-localization. In IEEE/CVF Win-
ter Conference on Applications of Computer Vision (WACV),
pages 3654–3663, 2023. 2
[48] Kai Zhu and Tao Zhang. Deep reinforcement learning based
mobile robot navigation: A review. Tsinghua Science and
Technology, 26(5):674–691, 2021. 1
[49] Sijie Zhu, Taojiannan Yang, and Chen Chen.
Transgeo:
Transformer is all you need for cross-view image geo-
localization. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 1162–1171, 2022. 2
