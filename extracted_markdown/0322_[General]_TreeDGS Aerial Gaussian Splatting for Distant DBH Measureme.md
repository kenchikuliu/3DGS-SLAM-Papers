<!-- page 1 -->
TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement
Belal Shaheen1, Minh-Hieu Nguyen1, Bach-Thuan Bui1, Shubham1, Tim Wu1, Michael Fairley1,
Matthew Zane1, Michael Wu1, James Tompkin2
1Coolant, San Francisco, CA 94111, USA
2Department of Computer Science, Brown University, Providence, RI 02912, USA
michael@coolant.earth
(c)
(b)
(a) 
Figure 1. TreeDGS accurately estimates DBH from distant RGB imagery: (a) High-fidelity TreeDGS reconstruction from distant UAV
RGB imagery as an optimized set of 3D Gaussians. (b) Surface points extracted via opacity-consistent sampling (built on RaDe-GS [52])
and segmented into stem vs. vegetation to isolate trunk geometry for DBH fitting. In (b), red points denote trunk/stem, green points
denote vegetation/canopy, and blue points denote ground/other non-trunk points. (c) DBH errors against field measurements, shown as
the distribution of DBH estimates relative to the field DBH distribution; TreeDGS + opacity-weighted circle fitting reduces error vs. UAV
LiDAR + cylinder fitting [33] (RMSE/MAE: 4.79/3.67 cm vs. 7.66/5.23 cm) at a ground sample distance (GSD) of approximately 1.84 cm.
Abstract
Aerial remote sensing efficiently surveys large areas, but
accurate direct object-level measurement remains difficult
in complex natural scenes. Advancements in 3D computer
vision, particularly radiance field representations such as
NeRF and 3D Gaussian splatting, can improve reconstruc-
tion fidelity from posed imagery. Nevertheless, direct aerial
measurement of important attributes like tree diameter at
breast height (DBH) remains challenging. Trunks in aerial
forest scans are distant and sparsely observed in image
views; at typical operating altitudes, stems may span only a
few pixels. With these constraints, conventional reconstruc-
tion methods have inaccurate breast-height trunk geometry.
TreeDGS is an aerial image reconstruction method that uses
3D Gaussian splatting as a continuous scene representation
for trunk measurement. After SfM–MVS initialization and
Gaussian optimization, we extract a dense point set from the
Gaussian field using RaDe-GS’s depth-aware cumulative-
opacity integration and associate each sample with a multi-
view opacity reliability score. Then, we isolate trunk points
and estimate DBH using opacity-weighted solid-circle fit-
ting.
Evaluated on 10 plots with field-measured DBH,
TreeDGS reaches 4.79 cm RMSE (about 2.6 pixels at this
GSD) and outperforms a LiDAR baseline (7.66 cm RMSE).
This shows that TreeDGS can enable accurate, low-cost
aerial DBH measurement (Figure 1).
1. Introduction
Aerial remote sensing has quickly become a cornerstone of
modern environmental monitoring for its ability to survey
large areas rapidly and cost efficiently
[1, 7, 37].
Yet,
despite major gains in sensor resolution, platform stabil-
arXiv:2601.12823v3  [cs.CV]  13 Mar 2026

<!-- page 2 -->
~13 pixels
"latitude": 33.47688150999893,
"longitude": -92.13269958706124,
"altitude": 69.975,
"width x height ": 5280 x 3956
"focal_length": 12.29,
"x_fov": 69.6,
"y_fov": 55.01532598689411,
"pitch": 24.899999999999977,
Figure 2. Pixel-limited trunk observations from distant aerial
imagery. At \sim 70 m altitude (\pr o tect \mathrm  {GSD}\approx 1.84 cm/px), a typical pine
stem can occupy only \sim 13 pixels across in a single RGB image,
making per-image diameter cues highly quantized, sensitive to oc-
clusion, and difficult to measure precisely.
ity, and reconstruction pipelines, extracting direct, object-
level measurements from aerial imagery remains difficult in
structurally complex natural scenes [8, 15]. Forested envi-
ronments are particularly challenging as heterogeneous ge-
ometry, self-similar textures, frequent occlusion, and strong
appearance variability across lighting conditions all weaken
the visual cues that aerial reconstruction methods rely
on [15].
A prominent example is tree diameter at breast height
(DBH). This describes an individual tree’s size and growth
stage.
DBH is an important forestry variable that is a
primary input to standard allometric models and report-
ing workflows to estimate wood volume, biomass, and car-
bon [2, 5, 17, 18]. In turn, these allometric models underpin
decision making used across forestry operations and pol-
icy, ranging from timber inventory and silvicultural plan-
ning to carbon offset quantification, wildfire risk mitigation,
and long-term ecosystem management [3, 42].
DBH estimation from above-canopy aerial sensing is
fundamentally limited by observability. In typical UAV im-
age surveys flown at operational altitudes, stems are not
large objects. In our data, captured at \sim 70 m above ground
(\pr o tect \mathrm  {GSD}\approx 1.84 cm/px), stems in the evaluation plots span
only \sim 15 pixels across per view on average (Table 1),
and they can be just a few pixels wide for smaller trees (Fig-
ure 2). This pixel scarcity is compounded by view scarcity:
the breast-height band (h_{ \ mathrm {BH}}\approx 1.37 m in our field protocol)
is frequently occluded by crowns, branches, and understory
vegetation, so each stem may be seen only in a handful of
oblique views through noisy canopy gaps. Thus, the core
geometric challenge of measurement is to recover enough
faithful breast-height trunk surface samples to support sta-
ble circle or cylinder fitting.
Active sensing can mitigate some of these limitations
by providing explicit range measurements rather than re-
lying on image texture. Terrestrial laser scanning (TLS),
for example, is well known to support robust DBH mea-
surements because it can produce dense, well-populated
stem cross-sections [21, 31, 39].
However, the mostly
downward-looking trajectories of airborne or UAV laser
scanning (ULS) do not typically yield equally dense returns
on the trunk at breast height: even with high point den-
sity, trunk hits near h_{\mathrm {BH}} can be sparse, uneven, and con-
taminated by surrounding vegetation, complicating per-tree
stem isolation and reliable fitting in cluttered stands [25,
26, 36]. These limitations help explain why UAV LiDAR
has seen broader operational adoption for canopy struc-
ture and terrain products than for direct DBH measure-
ment [6, 13, 32, 40, 49].
Nevertheless,
RGB-only UAV photogrammetry has
made substantial progress for other forest attributes, includ-
ing canopy height models and structural mapping [8, 15,
45, 46].
In contrast, DBH remains difficult from aerial
imagery alone, even under careful survey design with ag-
gressive overlap and strong oblique views [35].
In the
pixel- and view-limited regime described above, conven-
tional SfM–MVS pipelines often yield incomplete and frag-
mented breast-height trunk surfaces that are insufficient for
stable diameter fitting. Sparse SfM relies on repeatable key-
points and long feature tracks [41], but distant trunks pro-
vide weak texture and are frequently interrupted by foliage
occlusions, yielding unstable or missing correspondences.
View-dependent MVS densification [9] then tends to drop
out on trunks when visibility is intermittent and texture cues
are weak. Still, the partial geometry recovered by MVS can
serve as a useful coarse prior for later optimization.
Recent radiance field representations, such as Neural
Radiance Fields (NeRF) [34] and 3D Gaussian Splatting
(3DGS) [23], can improve reconstruction fidelity from
posed imagery and offer a promising path forward for
thin structures that are only weakly observed in individ-
ual frames. Early forestry applications have begun explor-
ing radiance fields for close-range terrestrial forest moni-
toring and tree-scale reconstruction [14, 24, 28, 43], but the
problem of direct DBH measurement from stand-off, above-
canopy UAV surveys remains unsolved. Moreover, trans-
lating a “fluffy” radiance field reconstruction into a reliable
diameter measurement requires an extraction step, such that
we can densely sample trees at breast height and prioritize
samples that are consistently supported across views.
We introduce a method—TreeDGS—that directly mea-
sures DBH from above-canopy UAV RGB imagery from
a 3D Gaussian splatting scene reconstruction under pixel-
and view-limited trunk observations. Starting from standard
SfM camera poses and an MVS initialization, we optimize a
RaDe-GS model [52] to obtain a continuous Gaussian field
whose covariances and opacities are refined by multi-view
photometric consistency. Then, we extract a dense point
set using depth-aware cumulative-opacity integration, re-
taining samples whose accumulated opacity indicates a re-
liably occupied surface under a fixed threshold. To miti-

<!-- page 3 -->
Table 1. Per-plot summary for the manually paired subset used in
per-tree evaluation. #Trees denotes the number of stems in each
plot with a verified field-to-reconstruction association; the “All”
row reports pooled statistics across all plots.
Plot
#Trees
DBH Mean ± SD
(cm)
DBH Range
(cm)
1
22
31.67 ± 5.11
25.6–43.3
2
27
28.23 ± 5.18
15.0–34.8
3
25
26.26 ± 7.08
7.9–44.3
4
16
24.52 ± 7.30
9.8–34.0
5
22
28.18 ± 4.13
17.8–34.3
6
23
27.02 ± 6.49
11.3–36.7
7
19
20.87 ± 7.60
10.2–38.4
8
17
29.25 ± 6.38
13.4–39.0
9
19
27.32 ± 8.61
11.8–55.7
10
20
29.53 ± 4.20
21.1–36.9
All
210
27.39 ± 6.73
7.9–55.7
gate spurious samples from foliage and partial occlusions,
we compute a per-point multi-view opacity support score
as a reliability signal and retain trunk-isolated points via
3D semantic segmentation. Finally, we estimate DBH with
opacity-weighted, slice-wise circle fitting, yielding stable
diameter estimates even when the trunk is only weakly ob-
served in individual images. We demonstrate real-world,
field-validated performance in a managed loblolly pine for-
est with dense understory and varied growth patterns (10
plots), achieving 4.79 cm RMSE (\sim 2.6 pixels at our GSD)
against ground-truth tape DBH and outperforming an ultra-
high-resolution UAV LiDAR baseline by 37.5% in RMSE.
2. Materials
We collected data in a managed Pinus taeda (loblolly pine)
stand in southeastern Arkansas, USA. We established ten
0.2-acre circular plots (plots 1–10; radius 16.05 m; Fig-
ure 3). At each plot center, we marked a circular plot bound-
ary with a 16.05 m radius and measured all stems within the
boundary. Then, we assigned a unique ID to each included
tree. For each tree, we recorded the following: (i) Bearing
and distance from the plot center; (ii) DBH measured with a
diameter tape at 1.37 m above ground level; and (iii) height
measured using a laser rangefinder.
We surveyed plot centers using ORS/NTRIP network
corrections.
For each plot, we centered a survey-grade
GNSS rover mounted on a survey pole and stabilized with
a bipod over the plot and occupied for 30 min in static
mode while receiving RTCM corrections via an NTRIP
caster.
Then, we derived plot-center coordinates using
a differential GNSS workflow, following forest-canopy
positioning best-practice recommendations described by
Strunk et al. [44]. Finally, we exported corrected plot-center
coordinates for co-registration with the aerial products.
UAV RGB imagery.
For the full set of plots, we col-
lected overlapping RGB imagery using a DJI Matrice 4E
(SZ DJI Technology Co., Ltd., Shenzhen, China) in two
campaigns. We captured imagery with high overlap and
Figure 3. Region of interest (ROI) and plot layout (10 sub-
plots). Each subplot corresponds to one 0.2-acre circular plot (ra-
dius 16.05 m). The circle indicates the field plot boundary used for
tree inclusion, while the dashed polygon outlines the ROI used to
clip and organize aerial products for per-plot processing.
mixed viewing geometry (including oblique views at 70 and
90 m altitude) to support SfM/MVS reconstruction and in-
crease the chance of observing trunk surfaces through gaps
in the canopy and understory. To improve absolute posi-
tioning consistency across flights, we corrected the onboard
UAV GNSS trajectory via post-processed kinematic (PPK)
processing using reference data from a nearby CORS sta-
tion (ARMO), and we used the corrected solution to up-
date image geotags before reconstruction. An example UAV
RGB flight trajectory is shown in Figure 4a.
(b)
(a)
Figure 4. RGB and LiDAR acquisition patterns. (a) We col-
lected RGB imagery with high overlap and mixed viewing angles
to support SfM/MVS and improve trunk visibility. (b) We cap-
tured LiDAR with dense grid (lawnmower) flight lines to obtain
uniform coverage across the plot network. Red polylines indicate
UAV flight trajectories; in (a) points are colored by RGB, while in
(b) LiDAR points are colored by height above ground (blue low to
yellow high).
UAV LiDAR. We acquired UAV laser scanning (ULS)
data with an Inertial Labs RESEPI payload (Inertial Labs,
Inc., Paeonian Springs, VA, USA) integrating a Hesai XT-
32 LiDAR (Hesai Technology, Shanghai, China). The XT-
32 is a 360^\circ mid-range scanner with a 0.05–120 m ranging
capability and up to 640k pts/s in single-return mode, with a
typical range precision of 0.5 cm (1σ) and a ranging accu-
racy of ±1 cm [11, 16]. Figure 4b summarizes the LiDAR
flight paths. The LiDAR survey used dense, back-and-forth
flight lines (a lawnmower grid) to provide uniform cover-
age and reduce directional bias in canopy and stem obser-
vations. We flew the UAV at 50 m above ground level,
yielding a point density of ≈1407 pts/m2.
The resulting georeferenced point clouds serve as a base-

<!-- page 4 -->
line geometry source for trunk isolation and DBH fitting.
Pairing workflow and quality control. Across all plots,
we measured 458 stems in the field. To create the per-tree
evaluation set, we paired each field-inventoried pine stem
to a reconstructed trunk instance using a standardized, spa-
tially driven workflow.
For each plot, we converted the
field-measured (bearing and distance) from the surveyed
plot center into georeferenced (x, y) coordinates. For the
reconstructed data (TreeDGS and LiDAR), we represented
each trunk instance by the centroid of its segmented trunk
points near breast height. First, we generated candidate as-
sociations using nearest-neighbor matching within the plot
boundary, and then we manually verified each pair in a
plot-level visualization that overlays the field points, recon-
structed instances, and high-resolution aerial context (or-
thomosaic/point cloud; Figure 5). Two authors performed
pairing; uncertain or ambiguous cases (e.g., clustered stems,
partially reconstructed trunks, or missing instances) were
discussed and, if unresolved, excluded from the paired set.
This resulted in 210 pair stems.
During fieldwork, we
also recorded a plot walkthrough video, which served as
an additional reference to confirm tree neighborhoods when
needed. Table 1 reports the per-plot pine stem counts and
DBH distribution statistics.
(a)
(b)
Figure 5. Example of field-to-reconstruction stem matching.
Plot-level overlay illustrating the manual pairing between field-
inventoried pine stems and reconstructed trunk instances (after
removing small non-pine trees).
Lines connect matched pairs.
The mean planimetric offset is 0.776 m for field-to-TreeDGS
matches (210 pairs across 10 plots) and 0.270 m for LiDAR-
to-TreeDGS matches. These offsets reflect combined field dis-
tance/bearing uncertainty and georegistration error, and they are
used only as a sanity check after visual verification.
3. Proposed Pipeline
3.1. Problem Statement
Given a set of N high-resolution UAV RGB images
{Ii}N
i=1, our goal is to estimate the diameter at breast height
(DBH) for each tree instance in the scene.
DBH is de-
fined as the trunk diameter at a fixed height above local
ground, and we follow the field protocol used in this study
(hBH ∈[1.37, 1.40] m).
Our approach is geometry-driven: we reconstruct cam-
era poses and an initial 3D structure with SfM, densify it
with OpenMVS [4], optimize a Gaussian splatting [23, 52]
scene representation, and then measure DBH from trunk-
only geometry using opacity-aware sampling and robust fit-
ting (Figure 6):
  \l a bel {e q:pipel i n e _ove r v i
ew} \{I_i\} \rightarrow (\{\mathbf {P}_i\}, \mathcal {X}_{\text {dense}}) \rightarrow \mathcal {G} \rightarrow \{\mathcal {T}_t\} \rightarrow \{\widehat {\mathrm {DBH}}_t\}, 
(1)
where Pi are the calibrated camera matrices, Xdense is a den-
sified point set from OpenMVS, G is the optimized Gaus-
sian field, and Tt denotes trunk points for tree instance t.
Sections 3.2–3.6 detail the SfM/OpenMVS reconstruction,
Gaussian optimization, trunk extraction through 3D seg-
mentation [50] with opacity cues, and the final DBH fit-
ting procedure.
3.2. Structure-from-Motion and Multi-View Stereo
We estimate camera poses and an initial scene structure us-
ing a Structure-from-Motion (SfM) pipeline [38] adapted
to high-resolution UAV forest imagery, where repeated
textures and partial occlusions can reduce matching re-
liability.
To improve correspondence quality, we use a
customized TopicFM [10] model trained on a mixture of
(i) MegaDepth [29], (ii) 100 synthetic UAV nadir forest
scenes, and (iii) 200 available real 3D models from the
Coolant Dataset.
Coarse-to-fine matching. Because our UAV imagery
is high resolution, we perform matching in a coarse-to-
fine manner [27]: we first obtain coarse correspondences
on downsampled images to establish robust global align-
ment, and we then refine matches locally at higher resolu-
tion. This reduces computation while maintaining accurate
pixel localization needed for stable pose estimation.
SfM reconstruction. SfM takes the matched correspon-
dences across overlapping image pairs and estimates the fol-
lowing: (i) camera poses {Pi} and (ii) a sparse 3D point
set. We use standard robust estimation and bundle adjust-
ment from GLOMAP [38] to refine the reconstruction. We
keep notation minimal and denote the resulting calibrated
cameras as:
  \ mathb f  {P}_i = \mathbf {K}_i [\mathbf {R}_i \mid \mathbf {t}_i]. 
(2)
These cameras are used both for Gaussian training (Sec-
tion 3.3) and for the multi-view opacity tests during surface
sampling (Section 3.4).
Coarse densification with OpenMVS [4]. After SfM,
we densify geometry using OpenMVS. OpenMVS pro-
duces depth information and a denser point set Xdense,
which provides stronger geometric support than sparse SfM
points in many well-observed regions (e.g., ground, canopy,
and partially visible trunk segments). We note that Xdense is
not assumed to be complete or coherent in the breast-height
band. In our setting, MVS trunk coverage can be sparse and

<!-- page 5 -->
slices
Weighting 
Opacity-Weighted DBH estimation
Gaussian Splats
Fitted Circle
(DBH)
RGB 
Images
SfM/MVS
Gaussian 
Splatting
Surface Sample 
3D Points
3D Trunk 
Segmentation
𝛼
ݓ
ܩ
ܶ
Figure 6. TreeDGS pipeline. RGB images are reconstructed with SfM/MVS and represented as 3D Gaussian splats G. We surface-
sample dense points using opacity α and weights w, segment trunks T, and estimate DBH by opacity-weighted circle fitting on multiple
trunk slices. Inset colors: α and w are visualized with a low-to-high colormap (cool colors = low values; warm colors = high values); in T,
trunk points are red and non-trunk points are green/yellow.
fragmented due to occlusion and weak texture. Instead, we
use Xdense only as a coarse prior to initialize the Gaussian
centers for faster and more stable optimization, following
common 3DGS practice [19, 23, 47].
3.3. Reconstruction with Gaussian Splats
Given calibrated cameras {Pi} and an MVS-densified ini-
tialization Xdense from Section 3.2, we optimize a 3D Gaus-
sian Splatting (3DGS) scene representation.
We adopt
RaDe-GS [52], which builds on the real-time 3DGS ren-
derer [23] while explicitly rasterizing depth in a way that is
also useful for our subsequent surface-consistent point sam-
pling.
Gaussian parameterization.
The scene is repre-
sented as a set of anisotropic 3D Gaussians Gk
=
(µk, sk, Rk, αk, ck), where µk ∈R3 is the Gaussian cen-
ter, sk ∈R3
+ are axis-aligned scales in the local frame,
Rk ∈SO(3) is the per-Gaussian orientation, αk ∈[0, 1]
is the opacity, and ck denotes appearance parameters (e.g.,
spherical-harmonic features as in [23]). We initialize {µk}
from Xdense and optimize all parameters following RaDe-
GS [52].
Note that RaDe-GS interleaves photometric
optimization with density control (splitting and pruning).
Therefore, the Gaussian set can be progressively densified
and refined over time, improving coverage in regions that
are only weakly supported by the MVS initialization.
RaDe-GS depth-plane formulation. A key difference
from vanilla 3DGS is that RaDe-GS associates each Gaus-
sian and view with a local ray-distance plane in screen
space. For a view v, let uk,v ∈R2 be the projected mean of
Gaussian k, and let tk,v be its ray-distance (range) in that
view (i.e., ∥Rvµk + tv∥2 under perspective projection).
RaDe-GS additionally provides a 2D slope gk,v ∈R2, so
that the Gaussian’s predicted ray-distance at a nearby pixel
u is approximated by
  \labe l  {eq : ra
degs_dep t h_plane} d_{k,v}(\mathbf {u}) \;=\; t_{k,v} \;+\; \mathbf {g}_{k,v}^{\top }(\mathbf {u}_{k,v}-\mathbf {u}). 
(3)
We use this depth-plane formulation in Section 3.4 to de-
fine a depth-aware, surface-consistent opacity integral at ar-
bitrary 3D query points.
Adaptive
training
with
RLGS.
We
integrate
RLGS [30] as an online controller that adapts selected
training hyper-parameters based on observed optimization
dynamics. In our setting, this is primarily used to stabilize
opacity and geometry quality for the downstream DBH
pipeline. After convergence, the optimized Gaussian field
is denoted by G and serves as the source representation for
dense point sampling.
3.4. Surface Sampling with Opacity
Our downstream stages (3D trunk segmentation and DBH
fitting) require a point cloud that is (i) dense enough to rep-
resent thin stems at stand-off distance and (ii) geometrically
trustworthy under heavy occlusion from foliage and under-
story. This is not automatic from a 3D Gaussian Splatting
(3DGS) reconstruction: the optimized field G is a collection
of overlapping volumetric primitives whose parameters are
trained through front-to-back alpha compositing rather than
an explicit surface loss [23, 52]. As a result, two seem-
ingly straightforward exports are unreliable in our UAV for-
est regime. First, exporting one point per Gaussian (the
means) produces a fragmented and extremely sparse cloud
which is insufficient for robust 3D segmentation and
measurement.
Second, back-projecting rendered depth
maps into 3D and fusing them via multi-view depth-
consistency preserves view-consistent sheet-like artifacts
from semi-transparent vegetation while eroding already-
scarce trunk evidence, especially when stems are only a few

<!-- page 6 -->
pixels wide or intermittently visible (see Figure 7). There-
fore, we sample densely in 3D, but we accept and score
samples using tests that are consistent with the same com-
positing model that governs 3DGS rendering and optimiza-
tion.
(a)
(b)
Figure 7. Comparison of back-projection points with multi-
view depth consistency and the proposed sample points.
(a) Depth-fusion baseline:
rendered depth points are back-
projected into 3D and kept only if they pass multi-view depth-
consistency checks.
This produces large sheet-like artifacts
near the ground/understory and incomplete stem surfaces (red
dashed boxes), despite appearing consistent across views.
(b)
Our opacity-guided surface sampling with depth-aware point-wise
compositing give a stem geometry suitable for DBH estimation.
Blue indicates lower elevations near ground, and red/orange indi-
cates higher canopy points.
Opacity-guided stochastic sampling.
Our goal is to
convert the optimized Gaussian field G into a point cloud
that is dense enough for trunk segmentation and DBH fit-
ting. For each Gaussian Gi = (µi, si, Ri, αi), we draw
candidate offsets ξij ∼N(0, I3) in the local frame and
map them to world coordinates. To focus sampling on re-
liable (high-opacity) regions, we apply Bernoulli thinning
controlled by αi:
  \ l ab e l 
{
eq : opa
c
i
ty_ s ampling} \begin {aligned} \mathbf {p}_{ij} &= \boldsymbol {\mu }_i \;+\; \mathbf {R}_i\left (\mathbf {s}_i \odot \boldsymbol {\xi }_{ij}\right ), \\ b_{ij} &\sim \mathrm {Bernoulli}(\alpha _i). \end {aligned} 
(4)
We keep pij if bij = 1, in expectation each Gaussian con-
tributes αiM points when drawing M candidates. Here,
M is the maximum number of candidate samples drawn per
Gaussian (before opacity-guided thinning) and directly con-
trols the point density and sampling cost. In all experiments
we use M = 100 (i.e., at most 100 candidate draws per
splat). Since candidates are retained with probability αi,
the expected number of retained samples per Gaussian is
αiM, which keeps the computational budget bounded while
still allocating higher sample density to high-opacity (more
reliable) regions.
Front-to-back compositing and accumulated alpha.
In 3DGS, solidity is not defined by an explicit surface. It
is defined implicitly by how Gaussians accumulate opac-
ity under the renderer’s front-to-back alpha compositing.
Therefore, any surface sampling rule that aims to match
what the model optimized should be expressed using the
same compositing mechanism.
For a view v and pixel
u, the renderer composites Gaussians front-to-back using
transmittance tracking. If αk,v(u) is the per-Gaussian al-
pha contribution at u, we define T0(u) = 1 and update
  \labe l  {eq:al pha_comp
ositi n g} \beg
i
n  {aligne
d
} w_{k,v}(\mathbf {u}) &= \alpha _{k,v}(\mathbf {u})\,T_{k-1}(\mathbf {u}), \\ T_k(\mathbf {u}) &= T_{k-1}(\mathbf {u})\bigl (1-\alpha _{k,v}(\mathbf {u})\bigr ). \end {aligned} 
(5)
so the accumulated alpha mask is Av(u) = P
k wk,v(u) =
1 −Tfinal(u). In our code this mask is read from the ren-
derer’s alpha channel and used as a visibility gate for sam-
pled points.
The per-pixel accumulated alpha mask Av(u) tells us
whether a ray intersects the reconstructed mass somewhere
along the ray, but it does not tell us whether a particular 3D
sample p lies on the visible surface, behind it, or in front
of it (all can project to the same pixel). For surface extrac-
tion we therefore need a depth-aware test that measures how
much opacity accumulates in front of p along the viewing
ray, using the same compositing model that governs train-
ing. RaDe-GS provides exactly this query by integrating
point-wise alphas with a depth-plane formulation.
Depth-aware point-wise opacity integration.
Be-
yond per-pixel alpha, RaDe-GS provides an integration
query at an arbitrary 3D point p: it returns the projected
pixel coordinate uv(p) and a cumulative opacity value
˜αv(p) that measures how much opacity is accumulated in
front of p along the viewing ray. Let rv(p) = ∥Rvp+tv∥2
denote the point ray-distance in view v. For a Gaussian k,
RaDe-GS evaluates a ray-space quadratic form using a pre-
computed inverse covariance over the 3D offset
  \label  
{
eq:r a degs_
poin t _of
fset} \Delta \mathbf {u}_{k,v}(\mathbf {p}) = \begin {bmatrix} \mathbf {u}_{k,v}-\mathbf {u}_v(\mathbf {p}) \\ t_{k,v} - \min \!\bigl (r_v(\mathbf {p}),\, d_{k,v}(\mathbf {u}_v(\mathbf {p}))\bigr ) \end {bmatrix}, 
r
(6)
and converts it to an alpha contribution via an exponential
falloff
  \alph a  _ {k,
v
} (
\mathbf {p}) 
\;\propto \
;
 \alpha _k \exp \!\left ( -\tfrac {1}{2}\Delta \mathbf {u}_{k,v}(\mathbf {p})^{\top } \boldsymbol {\Sigma }_{k,v}^{-1} \Delta \mathbf {u}_{k,v}(\mathbf {p}) \right ). 
(7)
The min(·) depth clamping in Equation (6) is critical for
surface consistency: once p lies behind the locally raster-
ized surface depth dk,v(·), the depth residual term saturates
rather than increases with rv(p), preventing spurious “vol-
umetric” accumulation behind the surface. Then, RaDe-GS
composites these point-wise alphas using the same front-to-
back transmittance update as in Equation (5), yielding the
cumulative opacity
  \lab e
l
 
{eq:poi ntwise_opa
city_in t egral} \b
e
g i n {alig
n
ed} \tilde {\alpha }_v(\mathbf {p}) &= \sum _k \alpha _{k,v}(\mathbf {p})\,T_{k-1,v}(\mathbf {p}), \\ T_{k,v}(\mathbf {p}) &= T_{k-1,v}(\mathbf {p})\bigl (1-\alpha _{k,v}(\mathbf {p})\bigr ). \end {aligned} 
(8)

<!-- page 7 -->
Multi-view surface consistency and reliability scores.
Since DBH fitting is highly sensitive to even small amounts
of geometric contamination, we aggregate evidence across
the calibrated views and attach a per-point reliability that
downstream steps can threshold or use as a weight. Con-
cretely, we (i) only allow a view to contribute if the point
projects to a foreground pixel in that view, and we (ii) sum-
marize the point-wise opacity integrals across contributing
views conservatively. Let P = {pij | bij = 1} be the can-
didate samples retained after the opacity-guided thinning in
Equation (4). For each p ∈P, we evaluate its support
over the calibrated views; a view v contributes only if p
projects inside the image and lands on a non-background
pixel in the rendered accumulated alpha mask Av. Specifi-
cally, with uv(p) and the projection of p, we define
  {m_ v (\mathbf {p } ) = \mathbf {1}\!\left (A_v(\mathbf {u}_v(\mathbf {p}))>0\right ).} 
(9)
Then, we store two per-point reliability signals:
  \la b
el 
{e q:opaci ty_reli
abil i
t
y
} \bar {\alpha }(\mathbf {p}) = \min _{v:\,m_v(\mathbf {p})=1}\tilde {\alpha }_v(\mathbf {p}), \qquad w(\mathbf {p})=\sum _v m_v(\mathbf {p}). (10)
Here, ¯α(p) is a conservative multi-view opacity esti-
mate, and w(p) counts the number of views in which p
projects onto a foreground pixel, i.e., Av(uv(p)) > 0.
In practice, for surface extraction, we keep points with
¯α(p) > τ and w(p) > 0. Especially for this work, we
set τ = 0 to keep all points while storing (¯α, w) for down-
stream weighting.
3.5. Semantic Trunk Extraction
The sampled point cloud from Section 3.4 contains trunks
mixed with foliage, understory vegetation, and occasional
floating samples caused by Gaussians with nonzero opac-
ity in free space. To isolate stem geometry for DBH mea-
surement, we used ForestFormer3D [50], a 3D semantic
segmentation model, to predict a class label for each sam-
pled point and to assign per-tree instance IDs when avail-
able. In our pipeline, the per-point opacity reliability ¯α is
kept as an auxiliary output feature alongside (x, y, z). We
retain only points predicted as trunk to form a trunk-only
cloud for each tree instance t, denoted by Tt = {(xk, ¯αk)}.
This separation step reduces contamination from branches
and understory, while the retained opacity values enable
reliability-weighted fitting downstream. The trunk-only in-
stances are then passed to the geometric measurement stage
described next.
3.6. Opacity-Weighted DBH Measurement
Given a trunk-only point set for tree instance t,
  \ begin  {align
ed}  
\m a thca l { T}_t  &= 
\le f t \ {(\mathbf {x}_k,\bar {\alpha }_k)\right \}_{k=1}^{N_t}, \\ \mathbf {x}_k &= (x_k,y_k,z_k)^\top \in \mathbb {R}^3,\quad \bar {\alpha }_k \in [0,1]. \end {aligned} 
(11)
our goal is to estimate the diameter at breast height
(DBH), i.e., the trunk diameter at hBH ≈1.3–1.4 m above
local ground. A key challenge is that (i) the sampled points
are volumetric (many points lie inside the trunk cross-
section, not only on the boundary), and (ii) residual non-
trunk points and floaters can persist even after segmenta-
tion.
We address both issues using an opacity-weighted
solid-circle RANSAC in each horizontal slice, followed by
a robust height-wise taper regression.
DBH is defined relative to local ground height. For each
tree instance, we estimate a ground elevation zg by query-
ing a digital terrain model (DTM) at the tree location (e.g.,
nearest-neighbor lookup at the mean trunk (x, y)), and we
convert all points to height above ground:
  z _g 
=
 
\m
at
h
rm 
{DTM }\!
\
l
ef t  ( \ tfrac {1}{N_t}\sum _{k=1}^{N_t}(x_k,y_k)\right ), \qquad h_k = z_k - z_g. (12)
We perform all subsequent slicing and DBH evaluation in
the (x, y) plane as a function of h.
Next, we construct a sequence of slice centers {hs} with
spacing ∆z and thickness H (a slab of height H), and we
collect 2D points in each slab:
  \ l abe
l  {e q: s l i c
in g
}
 \be gin  {al
ig ned }  h_ s  
&
=
 s\,\Delta z,\quad s=0,1,\dots , \\ \mathcal {S}_s &= \left \{(x_k,y_k,\bar {\alpha }_k)\;\middle |\; \left |h_k-h_s\right |\le \tfrac {H}{2}\right \}. \end {aligned} 
(13)
Slices with fewer than a small minimum number of
points nmin are discarded. This step produces a set of can-
didate diameters along the lower stem.
Unlike classical TLS stem fitting where points often lie
near the circumference, our sampling in Section 3.4 draws
points from Gaussian volumes; therefore, many slice points
legitimately fall inside the trunk cross-section. For this rea-
son we treat inliers as belonging to a filled disk rather than a
narrow ring around a circle. This choice makes the estima-
tor consistent with volumetric sampling and substantially
more stable under partial visibility.
Opacity-weighted solid-circle RANSAC in each slice.
For slice s, we have 2D points {(qk, wk)}n
k=1 with qk =
(xk, yk)⊤and opacity weights wk = ¯αk. Each RANSAC
hypothesis performs the following steps:
(i) Weighted sampling:
Draw three distinct indices
(k1, k2, k3) without replacement with Pr(k) ∝wk.
(ii) Circle fit from three points: Fit the circle through the
sampled points using the algebraic form x2 + y2 + ax +
by + d = 0. We solve for (a, b, d) after subtracting the
slice mean (for numerical stability), and we then recover
c = (−a
2, −b
2)⊤and r2 = ∥c∥2
2 −d (adding the mean back
to c).
(iii) Solid inliers and validity checks: A point is an in-
lier if it lies inside the disk: ∥qk −c∥2
2 ≤r2. We discard
hypotheses whose radius r is abnormally large or that yield
fewer than ρn inliers.

<!-- page 8 -->
(iv) Opacity-weighted scoring: For the remaining hy-
potheses, we score
  \label {eq:weighted_score} \begin {aligned} S(\mathbf {c},r) &= \frac {\sum _{k=1}^{n} w_k\, \mathbf {1}\!\left (\lVert \mathbf {q}_k-\mathbf {c}\rVert _2^2 \le r^2\right )}{r^{p}}, \\ &\quad p>0. \end {aligned} r
(14)
and select (ˆcs, ˆrs) = arg maxc,r S(c, r), reporting ˆds =
2ˆrs.
Height-wise taper regression and DBH prediction.
Even after per-slice robust fitting, some slices can be cor-
rupted by branch attachments, residual foliage, or incom-
plete sampling. Therefore, we fit a simple taper model on
the lower stem using RANSAC with a negative-slope prior:
  \la b el  {eq:
ta p er }  \hat {d}(h)=\beta _0 + \beta _1 h, \qquad {-\kappa \le \beta _1 < 0}, 
(15)
where RANSAC discards outlier slices using an absolute
residual threshold ϵ on diameter (
 ˆds −(β0 + β1hs)
 ≤ϵ)
and the slope constraint enforces physically plausible taper
(diameter should not increase with height over short stem
segments). The additional bound κ prevents unrealistically
steep taper caused by noisy slice diameters. In practice,
we fit Equation (15) over progressively larger height ranges
starting near the ground (to avoid upper-canopy contami-
nation) until a stable inlier set is found. Finally, we report
DBH as:
  \ l abel {eq
:db h _final } \ma t hrm {DBH} = \hat {d}(h_{\mathrm {BH}}), \qquad h_{\mathrm {BH}}\in [1.37,1.40]\;\text {m} \,. 
(16)
Role of opacity in robustness.
The opacity weights
wk = ¯αk originate from the multi-view rendering consis-
tency test in Section 3.4. Points that are only weakly sup-
ported (e.g., floaters, thin vegetation, or ambiguous geom-
etry) tend to have small ¯α and therefore (i) are sampled
less often during hypothesis generation and (ii) contribute
little to the inlier score in Equation (14). This coupling
between rendering consistency and geometric fitting is the
central mechanism behind our opacity-weighted measure-
ment mode.
4. Experiments
4.1. Experimental Settings
We evaluate TreeDGS on the 10 circular field plots de-
scribed in Section 2. Field DBH was measured with a di-
ameter tape at breast height (hBH = 1.37 m above ground,
Section 2).
We compare direct DBH measurement un-
der the following reconstruction sources and fitting strate-
gies: (i) UAV LiDAR: DBH estimation from the aerial
LiDAR point cloud using (a) cylinder fitting [33] and (b)
our slice-wise circle fitting (non-weighted and intensity-
weighted variants). (ii) TreeDGS: DBH estimation from
the point cloud sampled from the optimized 3D Gaussian
field (Section 3.4), using the same cylinder and circle fit-
ting variants, with opacity-based weighting available only
for TreeDGS. To ensure a fair comparison, trunk/instance
segmentation for both LiDAR and TreeDGS inputs is per-
formed using the same ForestFormer3D checkpoint [50]. If
ForestFormer3D misses a matched tree instance (e.g., due to
strong bending/occlusion) or if the downstream fitting can-
not return a DBH estimate, we count that tree as a failure
when computing the Success Rate (SR) in Table 2.
Cylinder fitting baseline. We follow DigiForests [33]
for cylinder fitting. A cylinder is fit to each trunk point
cloud using a RANSAC-initialized model. Each RANSAC
round samples five points, estimates the cylinder center and
axis from the covariance SVD, and keeps models whose
axis is within π/8 of vertical and radius ≤0.5 m.
The
RANSAC cylinder is then refined by least squares, updating
the cylinder center, axis, and radius to minimize a robust
(Geman–McClure) sum of squared point-to-cylinder dis-
tance residuals. Iterations stop when the parameter update
norm falls below ∥∆x∥< 10−5, where ∆x ∈R7 stacks the
center shift, rotation/axis update, and radius change. If the
least-squares refinement increases the inlier ratio, it replaces
the RANSAC estimate; otherwise, the RANSAC cylinder is
retained. Finally, the diameter at breast height is reported as
DBH = 2r from the selected cylinder radius.
Circle fitting variants.
For the proposed circle fit-
ting, we evaluate the following two reliability modes: non-
weighted (nw), which treats all points equally, and weighted
(w), which uses a per-point reliability score as a weight
(Equation (10)) in both hypothesis sampling and inlier scor-
ing (Equation (14)). For TreeDGS, this reliability is the
multi-view opacity consistency estimated by our renderer.
For LiDAR, we additionally report an intensity-weighted
analogue that uses normalized return intensity as a proxy
weight. In our UAV data, intensity-weighting helps reduce
the strong positive bias of non-weighted solid-circle fitting
(Table 2), but it still underperforms LiDAR cylinder fitting
and has lower SR, consistent with raw airborne LiDAR in-
tensity being speckled and unstable without careful radio-
metric correction and calibration (e.g., sensitivity to range,
incidence angle, and scanner settings) [12, 20, 22, 48, 51].
Figure 8 visualizes this gap: opacity yields a cleaner,
geometry-aligned confidence core around stem cross-
sections compared to noisy intensity. In addition, we omit-
ted SfM/MVS benchmarking because breast-height trunk
points were insufficient for stable segmentation and fitting
in this dataset (Figure 9).
Across all circle-fitting variants, we use slice thickness
H = 1.0 m and slice spacing ∆z = 0.1 m, discarding slices
with fewer than 5 points. For each slice, we run solid-circle
RANSAC (Equation (14)) with K = 2000 hypotheses,
a minimum inlier fraction ρmin = 0.1, and radius bounds
rmin = 0.02 m and rmax = 1.0 m. We select the radius

<!-- page 9 -->
(a)
(b)
(a-1)
(a-2)
(b-1)
(b-2)
a-1-tree
a-2-tree
b-1-tree
b-2-tree
Figure 8. LiDAR intensity vs. TreeDGS opacity as a reliabil-
ity signal at a same 5m slice height (Plot 2). (a) UAV LiDAR
point cloud colored by normalized return intensity. (b) TreeDGS
surface samples colored by opacity (0–1). Bottom: zoomed top-
down views of two example stems ((a-1,a-2) intensity; (b-1,b-2)
opacity). Compared to LiDAR intensity, the opacity values form a
clearer high-confidence core around the stem cross-sections. This
provides per-point confidence cue for DBH circle fitting. LiDAR
has 281.4K points and TreeDGS has 66.6K points.
regularization exponent p in Equation (14) via held-out val-
idation on a random selection of 10% of the matched stems
and then keep it fixed for all test experiments. We use the
same validation/test split IDs for all compared methods for
fair comparison. Concretely, from the 210 matched trees
in Table 1, we randomly reserve 21 trees (10%) for vali-
dation and report all numbers in Table 2 on the remaining
disjoint test set (N=189). This is why the per-plot denomi-
nators in Table 2 differ from Table 1. We obtain p = 0.6 for
TreeDGS opacity weighting and p = 0.85 for the LiDAR
intensity-weighted analogue (Figure 10).
The taper model in Equation (15) uses RANSAC with
residual threshold ϵ = 2 cm (max trials T = 1000, min
samples 3), requiring at least 10 inlier slices and enforc-
ing β1 < 0. Finally, DBH is reported as DBH = ˆd(hBH)
(Equation (16)) evaluated at hBH = 1.37 m.
(a) TreeDGS (Cir. w. fit.) 
(b) LiDAR (Cir. w. fit.)
Figure 10. Held-out validation for selecting the radius penalty
exponent p in Equation (14). We sweep p for weighted solid-
circle RANSAC and measure RMSE/MAE on the held-out vali-
dation split. (a) TreeDGS opacity-weighted circle fitting selects
p = 0.6.
(b) LiDAR intensity-weighted circle fitting selects
p = 0.85. We keep these values fixed when reporting all test
results in Table 2.
(a) SFM
(b) MVS
(c) TreeDGS
(d) LiDAR
Figure 9. Qualitative comparison of point density and trunk
completeness (Plot 1). (a) SfM points are sparse with limited
stem coverage.
(b) OpenMVS densification improves density
but remains incomplete on trunks under canopy occlusion. (c)
TreeDGS extracts a dense point set from the optimized Gaussians
via opacity-guided sampling (M = 100) and RaDe-GS depth-
aware cumulative-opacity tagging, yielding more continuous trunk
support near breast height. (d) UAV LiDAR provides strong struc-
ture but can still be stem-sparse/contaminated depending on scan
geometry and occlusion. Colors indicate height above ground.
We report four standard error measures between esti-
mated and field DBH: root mean squared error (RMSE), rel-
ative RMSE (RRMSE, normalized by the mean field DBH),
mean absolute error (MAE), and mean error (ME, bias).
All errors are reported in centimeters, and lower is bet-
ter for RMSE/RRMSE/MAE while ME closer to 0 indi-
cates lower systematic bias. We additionally report the Suc-
cess Rate (SR) as the fraction of trees in the test split for
which a method returns a valid DBH estimate (Table 2).
RMSE/RRMSE/MAE/ME are computed over the success-
ful subset, while SR captures failures (e.g., missing in-
stances or insufficient inlier slices for the taper RANSAC).
4.2. Results
Table 2 summarizes per-plot performance and includes a
pooled ‘All’ row. All results in Table 2 are reported on
the disjoint test split (N = 189) after reserving 21 of the
210 matched trees (10%) for validation/hyperparameter se-
lection (see Experimental settings). TreeDGS with opacity-
weighted circle fitting achieves the best overall accuracy
(pooled RMSE = 4.79 cm; MAE = 3.70 cm) with negli-
gible bias (ME = −0.38 cm), while maintaining a high
success rate (189/189).
Compared with the UAV Li-
DAR cylinder-fitting baseline (pooled RMSE = 7.66 cm),
TreeDGS reduces pooled RMSE by ≈37% while using
only RGB imagery.
Ablations also show that the fitting model must match
the reconstructed geometry.
Applying a classical cylin-

<!-- page 10 -->
Table 2.
Per-plot DBH accuracy and robustness (All–5).
RMSE/MAE/ME are in cm, RRMSE is in %, and SR is success
rate. Abbrev.: L/T = LiDAR/TreeDGS; Cyl = cylinder; CN/CW =
circle non-weighted/weighted.
Plot Meth. RMSE RRMSE MAE
ME
SR
All
L-Cyl
7.66
28.09
5.23
1.50 189/189
L-CN
16.61
60.96 14.81
14.67 187/189
L-CW
10.29
36.97
8.79
1.02 176/189
T-Cyl
13.29
48.69 10.25
−6.46 189/189
T-CN
9.61
35.22
6.97
5.79 189/189
T-CW
4.79
17.54
3.70
−0.38 189/189
1
L-Cyl
8.16
26.14
5.71
−0.04
19/19
L-CN
10.12
32.42
8.57
7.57
19/19
L-CW
9.94
32.04
8.04
2.01
18/19
T-Cyl
11.69
37.47 10.99 −10.99
19/19
T-CN
7.58
24.29
5.91
2.63
19/19
T-CW
4.94
15.83
3.72
−2.04
19/19
2
L-Cyl
5.36
18.84
4.34
0.38
22/22
L-CN
13.27
46.66 11.92
11.92
22/22
L-CW
8.60
30.25
7.12
−0.59
22/22
T-Cyl
13.43
47.21
9.84
−5.49
22/22
T-CN
8.43
29.65
7.26
7.09
22/22
T-CW
5.07
17.82
3.18
1.88
22/22
3
L-Cyl
12.92
49.55
7.75
4.47
20/20
L-CN
20.89
80.11 18.69
18.69
20/20
L-CW
11.85
45.44 10.32
3.55
20/20
T-Cyl
10.49
40.23
8.77
−7.35
20/20
T-CN
16.39
62.83
9.47
8.15
20/20
T-CW
5.93
22.73
4.35
−0.10
20/20
4
L-Cyl
6.63
27.06
5.81
2.94
16/16
L-CN
19.34
78.87 17.95
17.95
16/16
L-CW
9.59
35.77
7.78
−1.41
12/16
T-Cyl
25.24
102.93 16.57
3.84
16/16
T-CN
11.46
46.73
7.59
4.95
16/16
T-CW
5.74
23.40
4.53
−0.62
16/16
5
L-Cyl
6.20
21.87
4.27
2.37
21/21
L-CN
17.30
60.96 16.60
16.60
20/21
L-CW
11.61
40.92 10.99
3.26
20/21
T-Cyl
12.39
43.69 11.01
−7.98
21/21
T-CN
5.84
20.60
4.64
3.57
21/21
T-CW
4.38
15.46
4.01
−2.10
21/21
der fit [33] to points sampled from TreeDGS performs
poorly (13.29 cm RMSE; ME = −6.46 cm), consistent
with our discussion in Section 3.6: TreeDGS sampling
produces slice-wise cross-sections rather than a thin ring
of surface points.
Similarly, unweighted circle fitting
is substantially less accurate (9.61 cm RMSE), showing
that down-weighting low-confidence samples using multi-
view opacity is a better approach. On LiDAR, the same
trend holds: non-weighted solid-circle fitting strongly over-
estimates (ME = 14.67 cm), while intensity-weighted fit-
ting with a radius penalty substantially reduces this bias
(ME = 1.02 cm), though it still underperforms LiDAR
cylinder fitting.
TreeDGS (Cir. w. fit.) attains the lowest RMSE in nine
of ten plots (all except Plot 10) and achieves the best pooled
performance (All row). In Plot 2 it slightly outperforms
the LiDAR cylinder baseline (5.07 vs. 5.36 cm RMSE).
The largest gap appears in Plot 3, where LiDAR (cyl. fit.)
reaches 12.92 cm RMSE while TreeDGS (Cir. w. fit.) re-
mains at 5.93 cm. In Plot 10, LiDAR (cyl. fit.) remains the
best (3.79 cm RMSE).
Although a solid-disk inlier model can in principle en-
Table 2. Per-plot DBH accuracy and robustness (cont., plots
6–10).
Plot Meth. RMSE RRMSE MAE
ME
SR
6
L-Cyl
4.96
18.41
3.27
1.27 22/22
L-CN
14.03
52.15 12.86
12.86 22/22
L-CW
9.45
35.12
7.60
0.26 22/22
T-Cyl
8.64
32.10
7.91
−7.45 22/22
T-CN
8.91
33.10
8.23
8.23 22/22
T-CW
3.15
11.72
2.48
0.91 22/22
7
L-Cyl
9.16
44.08
7.23
1.92 17/17
L-CN
15.79
75.98 14.86
14.55 17/17
L-CW
7.79
33.96
6.31
−2.32 13/17
T-Cyl
13.66
65.77
8.15
−0.28 17/17
T-CN
12.01
57.81 10.57
9.68 17/17
T-CW
5.91
28.44
5.03
3.27 17/17
8
L-Cyl
9.65
33.11
7.80
2.61 16/16
L-CN
20.61
70.70 18.21
18.21 16/16
L-CW
11.07
36.67
9.77
1.11 15/16
T-Cyl
11.83
40.59 10.97
−7.64 16/16
T-CN
5.42
18.59
3.96
3.83 16/16
T-CW
3.68
12.63
3.14
−1.69 16/16
9
L-Cyl
6.18
23.13
4.20
−0.38 16/16
L-CN
16.27
62.05 15.09
15.09 15/16
L-CW
11.08
42.26 10.33
1.92 15/16
T-Cyl
8.73
32.66
7.73
−7.73 16/16
T-CN
8.50
31.81
6.85
5.17 16/16
T-CW
3.93
14.72
3.08
−1.26 16/16
10
L-Cyl
3.79
12.84
3.00
−0.30 20/20
L-CN
16.95
57.41 14.93
14.80 20/20
L-CW
10.73
35.95
9.23
0.84 19/20
T-Cyl
12.07
40.88 11.40 −11.15 20/20
T-CN
6.28
21.27
5.15
4.10 20/20
T-CW
4.42
14.95
3.73
−2.35 20/20
courage larger radii when volumetric interior points dom-
inate, in our data the dominant failure mode for non-
weighted fitting is positive bias driven by sparse ex-
terior outliers (foliage/floaters) inflating the disk radius.
The weighted objective in Equation (14) counteracts this
through (i) reliability weights (opacity/intensity) and (ii)
the explicit radius penalty r−p, which requires propor-
tionally more weighted inlier support to justify a larger
radius.
After optimizing p on held-out validation (Fig-
ure 10), the estimator transitions from the positive-bias
regime to near-unbiased behavior; the small pooled nega-
tive ME of −0.38 cm for TreeDGS (Cir. w. fit.) indicates
mild shrinkage when only a partial high-confidence arc of
the trunk boundary is visible and is within typical measure-
ment/registration noise.
Success Rate (SR) in Table 2 explicitly accounts for
cases where a method cannot return a DBH estimate.
For example, LiDAR intensity-weighted circle fitting fails
on 13 test trees (SR 176/189) because the taper RANSAC
requires at least 10 inlier slices; intensity weighting can
amplify slice-to-slice diameter variability and reduce the
number of inlier slices below this threshold. In contrast,
the non-weighted LiDAR circle fit fails on only two low-
density trees (SR 187/189), suggesting that most additional
failures are due to sensitivity to noisy intensity rather than
fundamental point scarcity.
Beyond reporting pooled RMSE/MAE, we add three
diagnostics to characterize robustness and failure modes.

<!-- page 11 -->
First, Figure 11 plots pooled per-tree signed error distri-
butions (estimate−field). The opacity-weighted solid-circle
variant yields the tightest distribution centered near zero,
consistent with its lowest pooled RMSE/MAE and near-
zero ME in Table 2. In contrast, the non-weighted solid-
circle fits show a pronounced positive shift and heavier right
tails (radius inflation), while cylinder fitting on TreeDGS
samples shifts negative (systematic underestimation). Sec-
ond, Figure 12 reports signed error versus ground-truth
DBH. Size-dependent bias is clearly visible for several
baselines: TreeDGS (Cyl. fit.) increasingly underestimates
as DBH grows (large negative mean error for the largest
stems), and LiDAR (Cir. nw. fit.) maintains strong posi-
tive bias across sizes. TreeDGS (Cir. w. fit.) shows sub-
stantially reduced size dependence: its binned mean error
stays within single-digit centimeters and close to zero for
the mid-range DBH values that dominate our plots (\sim 18–
38 cm), while becoming more negative in the largest-DBH
bin (which contains relatively few trees). Third, to quan-
tify the view-scarcity/occlusion bottleneck, Figure 13 re-
lates the absolute DBH error (TreeDGS (Cir. w. fit.)) to sim-
ple breast-height (BH) observability metrics computed from
the calibrated RGB views: the number of unoccluded BH
views, the mean projected trunk width in pixels, and their
product. On the 180/189 test trees with at least one unoc-
cluded BH view, we observe weak but consistently nega-
tive correlations, indicating that higher BH visibility is as-
sociated with lower error. Importantly, the largest outliers
(>10 cm) occur predominantly in the low-visibility regime
(few unoccluded views and/or small projected trunk width),
supporting our discussion that even with opacity reliability
weighting, accurate DBH requires at least a few informative
views around breast height. Figure 14 shows an example of
the BH-band view selection used for this analysis.
Figure 11. Aggregated DBH error distributions. Kernel den-
sity estimates of signed per-tree DBH error (estimate−field, cm)
pooled across all plots for each method. The vertical grey dashed
line marks zero error (perfect agreement with field DBH).
We additionally report a simple baseline that exports one
point per Gaussian by using the splat centers {µi}, similar
to trunk-focused usage in prior 3DGS tree pipelines [28].
In our stand-off UAV forest scenes, this produces an
extremely sparse and surface-incomplete point set (Fig-
Figure 12.
DBH error versus DBH size (aggregated across
plots). Signed per-tree error (estimate-field, cm) versus ground-
truth DBH on the test split, aggregated across all plots. Semi-
transparent dots represent individual trees. For each method, thick
solid segments show a robust binned error trend. Dashed segments
are only used to bridge DBH ranges with no observations, linking
the nearest available bins without implying data exist inside the
gap.
Figure 13. DBH error versus breast-height (BH) visibility in
the input RGB views. For each tree we compute (left) the num-
ber of unoccluded views in which the BH trunk band is visible,
(middle) the average projected trunk width in pixels over those
views, and (right) a combined visibility score (#views×width).
Each point is one tree from the test split (180 trees across 10
plots with at least one unoccluded BH view), colored by plot; blue
hexbin shading indicates point density. The y-axis is the absolute
DBH error for TreeDGS (Cir. w. fit.); red lines show least-squares
trends (Pearson r and p in legend). Mean unoccluded views per
tree = 8.2 and mean trunk width = 15.4 px.
ure 15), which breaks downstream instance segmentation
with ForestFormer3D [50] and prevents reliable DBH fit-
ting. These failures highlight that surface-densified sam-
pling is necessary in the stand-off UAV setting.
4.3. Ablation Study
4.3.1. Sensitivity Analysis of Fitting Parameters
In this section, we analyze the sensitivity of the DBH es-
timation stage to the main design parameters that govern
slice construction (Equation (13)) and robust aggregation
across height (Equation (15)). We vary one parameter at
a time, slice thickness H, minimum points per slice nmin,
taper RANSAC residual threshold ϵ, taper slope bound
κ, and whether solid-circle fitting uses opacity reliability
weights (wk = ¯αk) or uniform weights (wk ≡1) while
keeping all other settings fixed (∆z = 0.1 m, K = 2000,
ρmin = 0.1, r ∈[0.02, 1.0] m, and the tuned radius expo-

<!-- page 12 -->
Figure 14. Example BH-band visibility classification used in
Figure 13. We project a small trunk band around hBH into each
calibrated RGB view and classify the crop (red box) as unoccluded
or occluded. (Top): views counted as unoccluded where bark is
visible within the BH band. (Middle): views rejected as occluded
because foliage covers the BH band. (Bottom): corresponding
top-full-frame context, illustrating how small the BH band is rela-
tive to the canopy in stand-off UAV imagery.
(a)
(b)
Figure 15. Mean-only vs. proposed point sampling from Gaus-
sian Splats. (a) Exporting only Gaussian means (one point per
splat) yields an extremely sparse, fragmented cloud that is insuf-
ficient for downstream tasks such as 3D segmentation. (b) Our
opacity-guided surface sampling densifies geometry and preserves
stem surfaces and canopy structure. Point colors encode height
above ground (blue low to red/orange high).
nent p from Figure 10).
Table 3 shows that DBH accuracy is stable over a broad
range of values. Thin slabs increase error due to sparse sup-
port, whereas overly thick slabs can mix residual clutter;
Table 3.
Ablation on TreeDGS on the held-out test split
(N=189). Default values are shown in bold. SR is 189/189 for
all rows, so only MAE, RMSE, and ME (cm) are shown.
Param.
Val.
MAE RMSE
ME
Slice H
(m)
0.2
5.19
6.62 −3.85
0.5
3.86
5.23 −1.57
1.0
3.70
4.79 −0.38
1.5
3.89
6.42
0.41
2.0
4.03
5.94
0.88
nmin
3
3.94
6.35 −0.06
5
3.70
4.79 −0.38
10
3.66
4.82 −0.36
20
3.65
4.84 −0.31
ϵ
(cm)
1.0
4.15
6.63 −0.25
2.0
3.70
4.79 −0.38
3.0
3.53
4.77 −0.41
5.0
3.53
4.73 −0.27
κ
(cm/m)
0.1
3.78
4.94 −0.62
0.2
3.71
4.87 −0.43
0.3
3.70
4.79 −0.38
0.5
3.68
4.86 −0.10
unb.
5.01
7.93
2.26
wk
opacity
3.70
4.79 −0.38
uniform
6.97
9.61
5.79
H = 1.0 m performs best in our data and nearby choices
remain competitive.
Results are largely insensitive once
nmin ≥5 and ϵ ≥2 cm. Imposing a moderate taper-slope
bound improves robustness, and disabling this bound in-
creases both RMSE and positive bias. Finally, using opacity
as a continuous reliability weight yields a large improve-
ment compared to uniform weighting, confirming the im-
portance of the opacity signal.
4.3.2. ForestFormer3D Segmentation Quality
Because our field dataset does not provide the point-wise
trunk/instance ground truth, we quantify segmentation per-
formance on a small but representative subset (Plot 2) by
creating a hand-corrected reference annotation.
Starting
from the ForestFormer3D predictions, a single annotator
manually fixed trunk false negatives/positives, ground mis-
labels, and instance-ID assignment errors for all matched
trees in Plot 2 (\approx 14 labeling hours; Figure 16). Table 4
summarizes trunk semantic accuracy (F1 = 0.812, IoU
= 0.683).
To quantify the impact on DBH, we reran the fitting
pipeline on Plot 2 using the hand-corrected labels as an or-
acle segmentation. This reduces TreeDGS (Cir. w. fit.)
RMSE from 5.07 cm to 4.21 cm (MAE from 3.18 cm to
2.68 cm) on the same 22 matched trees (Table 5). This in-
dicates that segmentation errors contribute to DBH error,
but a substantial portion of the remaining error is due to re-
construction/visibility limitations rather than segmentation
alone.
4.3.3. Runtime Analysis
To assess practical utility, we report a representative runtime
breakdown for one plot (Plot 2, 365 RGB images) processed
end to end on a single workstation (AMD Ryzen Thread-

<!-- page 13 -->
(a)
(b)
Figure 16. ForestFormer3D segmentation on Plot 2 and our
hand-corrected reference.
(a) Off-the-shelf ForestFormer3D
trunk/ground predictions and instance IDs (trunk points in red).
(b) Hand-corrected labels used for evaluation, obtained by editing
the predictions to fix trunk false negatives/positives, ground mis-
labels, and instance-ID assignment errors. Creating this reference
annotation for Plot 2 required \approx 14 hours by a single annotator.
Yellow boxes highlight representative regions where manual ed-
its corrected misclassifications/instance IDs; trunk points are red,
while non-trunk points are blue/gray and ground points are green.
Table 4. ForestFormer3D segmentation accuracy on Plot 2.
Hand-corrected reference;
trunk semantic segmentation over
2,396,986 points.
Metric
Value
Notes
Precision
0.760
TP=107,746; FP=34,038
Recall
0.871
FN=15,974
F1
0.812
—
IoU
0.683
—
Accuracy
0.979
TN=2,239,228
Table 5.
Effect of segmentation quality on DBH accuracy
(Plot 2, 22 trees). TreeDGS (Cir. w. fit.) with ForestFormer3D
predictions versus hand-corrected oracle labels. MAE/RMSE/ME
are in cm and RRMSE is in %.
Segmentation
SR
MAE RMSE RRMSE
ME
ForestFormer3D 22/22
3.18
5.07
17.82
1.88
Oracle
22/22
2.68
4.21
14.81
1.49
Table 6. Runtime breakdown per plot (Plot 2, 365 images).
Feature matching and Gaussian optimization dominate runtime;
segmentation and DBH fitting are comparatively fast.
Stage
Time
Resource
Feature matching (TopicFM)
50 min
GPU
SfM + bundle adjustment (GLOMAP)
3 min
CPU
OpenMVS densification
5 min
CPU
Gaussian optimization (RaDe-GS)
20 min
GPU
ForestFormer3D segmentation
5 min
GPU
Slice fitting + taper aggregation
10 s
CPU
Total
\approx 83 min
—
ripper PRO 7995WX; 192 CPU threads; 1× NVIDIA RTX
5090, 32 GB). Reported times are wall-clock and depend on
image count, scene difficulty, and hardware, but they indi-
cate that the overall cost is dominated by SfM matching and
Gaussian optimization, while DBH fitting itself is negligi-
ble (Table 6)
Since plots are independent, the pipeline is trivially par-
allelizable across plots (one plot per GPU).
5. Discussion
Overall, our results suggest that TreeDGS can convert pixel-
limited UAV RGB imagery into a trunk-centric 3D repre-
sentation that is sufficiently coherent for plot-scale DBH es-
timation, narrowing the performance gap to a UAV LiDAR
baseline while using only low-cost imagery.
Compared
to conventional SfM/MVS densification, Gaussian opti-
mization combined with opacity-guided sampling produces
denser and more spatially consistent support near breast
height, which directly benefits downstream cross-sectional
fitting. The added breast-height observability analysis fur-
ther indicates that DBH errors increase primarily in the low-
visibility regime, reinforcing that view scarcity and occlu-
sion are key bottlenecks in aerial DBH measurement. In ad-
dition, our Plot 2 hand-correction study highlights that seg-
mentation quality contributes measurably to DBH accuracy,
but this does not fully account for residual error, implying
that reconstruction completeness and boundary visibility re-
main limiting factors even with improved labels. Finally,
the runtime breakdown shows that the full pipeline is prac-
tically feasible at the plot level, and that the proposed slice
fitting stage is computationally negligible compared to re-
construction and learning-based components. These find-
ings collectively support TreeDGS as a practical RGB-only
alternative (or complement) for rapid forest inventory under
appropriate visibility conditions. However, several limita-
tions remain.
Limitations.
TreeDGS has important limitations. First,
it relies on trunk visibility: the breast-height band must be
observed in at least a few frames (in practice, \sim 2+ infor-
mative views for SfM/MVS and Gaussian optimization to
constrain the stem surface (Figure 17a). When this band is
fully occluded in most views, the method cannot recover the
missing geometry, regardless of how densely the Gaussian
field is sampled; sampling density cannot compensate for
geometry that is never sufficiently observed in the input im-
agery.
Second, DBH estimation is sensitive to 3D semantic
segmentation quality. Because diameter is fit from trunk-
isolated points, false-positive trunk labels from nearby veg-
etation can bias the estimate (Figure 17c). Although multi-
view opacity weighting mitigates this effect, the pipeline
remains subject to the failure modes of the trunk classi-
fier [50].
This sensitivity can also contribute to poorer
LiDAR baseline performance in challenging plots: when
breast-height stem returns a small number of vegetation
false positives can disproportionately bias cylinder fits.
Third, our current DBH estimator makes a morpholog-
ical assumption that is well satisfied in the managed Pi-
nus taeda plantation used here but may not hold in more

<!-- page 14 -->
GT: 55.7 cm
Est. 25.3 cm
Tree ID: 9-37
GT: 10.2 cm
Est. 27.5 cm
Tree ID: 7-28  
GT: 29.8 cm
Est. 38.4 cm
Tree ID:7-20 
GT: 23.2 cm
Est. 22.5 cm
Tree ID: 7-1
GT: 33.3 cm
Est. 31.9 cm
Tree ID: 6-17
GT: 32.0 cm
Est. 34.9 cm
Tree ID: 6-27
Failed 
Case
Successful 
Case
(a)
(b)
(c)
(c)
(d)
(e)
Figure 17. Examples of successful and failed DBH fits at breast height. (a–c) Failure cases: (a) no unoccluded breast-height views,
(b) strongly curved/leaning stem causing biased horizontal slicing, and (c) instance segmentation that merges a target stem with a nearby
sapling. (d–f) Successful cases with clear breast-height visibility and accurate per-slice fits. Yellow circles show the fitted cross-sections
on representative slices. Green dots show the segmented trunk points; the estimated DBH (Est.) is colored red for failure cases and blue
for successful cases.
complex forests. Specifically, we assume that (i) the lower
stem cross-section can be reasonably approximated by a
circle (per-slice solid-circle RANSAC), (ii) horizontal slic-
ing is a good proxy for slicing orthogonal to the stem axis
(i.e., stems are approximately vertical/straight near breast
height), and (iii) diameter decreases monotonically with
height over the fitted lower-stem segment (linear taper with
a negative-slope prior). In natural or mixed-species forests,
stems can be leaning, curved (Figure 17b), buttressed,
fluted, forked, or multi-stem, and these cases can violate
the above assumptions and introduce bias. Importantly, this
limitation is in the geometric measurement model rather
than in TreeDGS sampling itself:
the opacity-weighted
sampling yields a denser, reliability-weighted point set that
could support more flexible trunk models. We view extend-
ing the measurement stage to such morphologies and vali-
dating on mixed-species natural forests as key future work.
Future Work.
A promising direction is to reduce depen-
dence on post-processing (sampling and segmentation) by
incorporating explicit trunk primitives into Gaussian opti-
mization. Jointly optimizing a small set of parameterized
stem elements (e.g., tapered cylinders) together with the
Gaussian field would enable DBH to be inferred directly
from model parameters and would provide a stronger in-
ductive bias under partial trunk visibility. Such a hybrid rep-
resentation may also support end-to-end supervision when
ground-truth field DBH is available.
6. Conclusions
We introduced TreeDGS, a UAV RGB-only pipeline that
repurposes 3D Gaussian splatting as an opacity-weighted
density sampling method for trunk measurement, rather
than only a view-synthesis renderer. Building on SfM/MVS
initialization, TreeDGS optimizes a Gaussian field and then
converts it into a measurement-ready stem point set via
depth-aware cumulative-opacity sampling and multi-view
reliability weighting, enabling robust circle fitting at breast
height. In doing so, our approach bridges a key gap left
by prior UAV LiDAR and UAV-SfM DBH pipelines that
operate on inherently sparse or view-limited point clouds,
and demonstrates that Gaussian splats can support accurate
trunk-level metrics from commodity aerial imagery.

<!-- page 15 -->
Acknowledgments
We thank PotlatchDeltic Corporation for supporting this re-
search by allowing us to conduct data collection and val-
idation on their managed forest stands.
We are grateful
to Jacob Strunk, Kit Hart, Nathaniel Naumann, Paurava
Thakore, Tim Sydor, and Bill Driscoll for coordinating lo-
gistics, sharing field context, and assisting with site access
and study planning.
Data Availability
The original contributions presented in this study are in-
cluded in the article. Further inquiries can be directed to
the corresponding author.
Conflict of Interest
Authors
Belal
Shaheen,
Minh-Hieu
Nguyen,
Bach-
Thuan
Bui,
Shubham,
Tim
Wu,
Michael
Fairley,
Matthew Zane,
and Michael Wu were employed by
Coolant.
James Tompkin is a scientific advisor to
Coolant.
The authors declare that these affiliations did
not influence the study design, data collection, analy-
sis, interpretation of results, or the decision to pub-
lish.
References
[1] Karen Anderson and Kevin J. Gaston.
Lightweight un-
manned aerial vehicles will revolutionize spatial ecology.
Frontiers in Ecology and the Environment, 11(3):138–146,
2013. 1
[2] Sandra Brown.
Estimating biomass and biomass change
of tropical forests: A primer. Technical Report 134, Food
and Agriculture Organization of the United Nations (FAO),
Rome, Italy, 1997. 2
[3] California Air Resources Board. Compliance offset proto-
col: U.s. forest projects. Technical report, California Envi-
ronmental Protection Agency, 2015. Adopted June 25, 2015.
2
[4] Dan Cernea. Openmvs: open multi-view stereo reconstruc-
tion library.
https://github.com/cdcseacave/
openMVS. Accessed: 2026-01-09. 4
[5] J´erˆome Chave, Maxime R´ejou-M´echain, Alberto B´urquez,
and et al. Improved allometric models to estimate the above-
ground biomass of tropical trees. Global Change Biology, 20
(10):3177–3190, 2014. 2
[6] Ziyue Chen, Bingbo Gao, and Bernard Devereux. State-of-
the-art: Dtm generation using airborne lidar data. Sensors,
17(1):150, 2017. 2
[7] Ismael Colomina and Pere Molina. Unmanned aerial systems
for photogrammetry and remote sensing: A review. ISPRS
Journal of Photogrammetry and Remote Sensing, 92:79–97,
2014. 1
[8] Jonathan P. Dandois and Erle C. Ellis. High spatial resolution
three-dimensional mapping of vegetation spectral dynamics
using computer vision. Remote Sensing of Environment, 136:
259–276, 2013. 2
[9] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and
robust multi-view stereopsis. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 32(8):1362–1376, 2010.
2
[10] Khang Truong Giang, Soohwan Song, and Sungho Jo. Top-
icfm+: Boosting accuracy and efficiency of topic-assisted
feature matching. IEEE Transactions on Image Processing,
2024. 4
[11] Hesai Technology. XT16/32/32M high-precision 360° mid-
range lidar: Specification details (xt32). https://www.
hesaitech.com/product/xt16-32-32m/.
Ac-
cessed 2026-01-15. 3
[12] Bernhard H¨ofle and Norbert Pfeifer. Correction of laser scan-
ning intensity data: Data and model-driven approaches. IS-
PRS Journal of Photogrammetry and Remote Sensing, 62(6):
415–433, 2007. 8
[13] Markus Hollaus, Wolfgang Wagner, Bernhard Maier, and
Klemens Schadauer. Airborne laser scanning of forest stem
volume in a mountainous environment. Sensors, 7(8):1559–
1577, 2007. 2
[14] Hongyu Huang, Guoji Tian, and Chongcheng Chen. Evalu-
ating the point cloud of individual trees generated from im-
ages based on neural radiance fields (nerf) method. Remote
Sensing, 16(6):967, 2024. 2
[15] Jakob Iglhaut, Carlos Cabo, Stefano Puliti, Livia Piermat-
tei, James O’Connor, and Jacqueline Rosette.
Structure
from motion photogrammetry in forestry: a review. Current
Forestry Reports, 5(3):155–168, 2019. 2
[16] Inertial Labs. RESEPI™Hesai XT-32 Datasheet (rev. 1.03),
2024. Accessed 2026-01-15. 3
[17] IPCC. 2006 ipcc guidelines for national greenhouse gas in-
ventories, volume 4: Agriculture, forestry and other land
use. Technical report, Intergovernmental Panel on Climate
Change (IPCC) / IGES, 2006. 2
[18] Jennifer C. Jenkins, David C. Chojnacky, Linda S. Heath,
and Richard A. Birdsey. National-scale biomass estimators
for united states tree species. Forest Science, 49(1):12–35,
2003. 2
[19] Jaewoo Jung, Jisang Han, Honggyu An, Jiwon Kang,
Seonghoon Park, and Seungryong Kim.
Relaxing accu-
rate initialization constraint for 3d gaussian splatting. arXiv
preprint arXiv:2403.09413, 2024. 5
[20] Sanna Kaasalainen, Ulla Pyysalo, Anssi Krooks, Ants Vain,
Antero Kukko, Juha Hyypp¨a, and Mikko Kaasalainen. Abso-
lute radiometric calibration of ALS intensity data: Effects on
accuracy and target classification. Sensors, 11(11):10586–
10602, 2011. 8
[21] Ville Kankare and et al.
Diameter distribution estimation
with laser scanning based multi-scan single-tree inventory.
ISPRS Journal of Photogrammetry and Remote Sensing,
2015. 2
[22] Alireza G. Kashani, Michael J. Olsen, Christopher E. Par-
rish, and Nicholas Wilson. A review of LiDAR radiometric
processing: From ad hoc intensity correction to rigorous ra-
diometric calibration. Sensors, 15(11):28099–28128, 2015.
8

<!-- page 16 -->
[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 4, 5
[24] Adam Korycki, Cory Yeaton, Gregory S Gilbert, Colleen
Josephson, and Steve McGuire. Nerf-accelerated ecologi-
cal monitoring in mixed-evergreen redwood forest. Forests,
16(1):173, 2025. 2
[25] Mikko Kukkonen, Matti Maltamo, Lauri Korhonen, and Pet-
teri Packalen. Evaluation of UAS LiDAR data for tree seg-
mentation and diameter estimation in boreal forests using
trunk- and crown-based methods. Canadian Journal of For-
est Research, 52(5):674–684, 2022. 2
[26] Karel Kuˇzelka, Martin Slav´ık, and Peter Surov´y. Very high
density point clouds from UAV laser scanning for automatic
tree stem detection and direct diameter measurement. Re-
mote Sensing, 12(8):1236, 2020. 2
[27] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 4
[28] Jiaqi Li, Qingqing Huang, Xin Wang, Benye Xi, Jie Duan,
Hang Yin, and Lingya Li. A method for the 3d reconstruction
of landscape trees in the leafless stage. Remote Sensing, 17
(8):1473, 2025. 2, 11
[29] Zhengqi Li and Noah Snavely. Megadepth: Learning single-
view depth prediction from internet photos.
In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 2041–2050, 2018. 4
[30] Zhan Li, Huangying Zhan, Changyang Li, Qingan Yan, and
Yi Xu.
Rlgs: Reinforcement learning-based adaptive hy-
perparameter tuning for gaussian splatting. arXiv preprint
arXiv:2508.04078, 2025. 5
[31] Xinlian Liang, Juha Hyypp¨a, and et al. Terrestrial laser scan-
ning in forest inventories. ISPRS Journal of Photogrammetry
and Remote Sensing, 115:63–77, 2016. 2
[32] Kevin Lim, Paul Treitz, Michael Wulder, Benoˆıt St-Onge,
and Martin Flood. Lidar remote sensing of forest structure.
Progress in Physical Geography, 27(1):88–106, 2003. 2
[33] Meher VR Malladi, Nived Chebrolu, Irene Scacchetti,
Luca Lobefaro,
Tiziano Guadagnino,
Benoˆıt Casseau,
Haedam Oh, Leonard Freißmuth, Markus Karppinen, Janine
Schweier, et al. Digiforests: a longitudinal lidar dataset for
forestry robotics. In 2025 IEEE International Conference on
Robotics and Automation (ICRA), pages 1459–1466. IEEE,
2025. 1, 8, 10
[34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing scenes as neural radiance fields for view syn-
thesis, 2020. 2
[35] Bruno Miguez Moreira, Gabriel Goyanes, Pedro Pina, Oleg
Vassilev, and Sandra Heleno. Assessment of the influence of
survey design and processing choices on the accuracy of tree
diameter at breast height (DBH) measurements using UAV-
based photogrammetry. Drones, 5(2):43, 2021. 2
[36] Romain Neuville, Jordan Steven Bates, and Franc¸ois Jonard.
Estimating forest structure from uav-mounted lidar point
cloud using machine learning. Remote Sensing, 13(3):352,
2021. 2
[37] Francesco Nex and Fabio Remondino. Uav for 3d mapping
applications: A review. Applied Geomatics, 6(1):1–15, 2014.
1
[38] Linfei Pan,
Daniel Barath,
Marc Pollefeys,
and Jo-
hannes Lutz Sch¨onberger.
Global Structure-from-Motion
Revisited.
In European Conference on Computer Vision
(ECCV), 2024. 4
[39] Pasi Raumonen, Mikko Kaasalainen, Markku ˚Akerblom,
Sanna Kaasalainen, Harri Kaartinen, Mikko Vastaranta,
Markus Holopainen, and Philip Lewis. Fast automatic preci-
sion tree models from terrestrial laser scanning data. Remote
Sensing, 5(2):491–520, 2013. 2
[40] Christian Salas, Liviu Ene, Timothy G. Gregoire, Erik Næs-
set, and Terje Gobakken. Modelling tree diameter from air-
borne laser scanning derived variables: A comparison of spa-
tial statistical models. Remote Sensing of Environment, 114
(6):1277–1285, 2010. 2
[41] Johannes L. Sch¨onberger and Jan-Michael Frahm. Structure-
from-Motion Revisited. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
2016. 2
[42] Joe H. Scott and Elizabeth D. Reinhardt. Assessing crown
fire potential by linking models of surface and crown fire be-
havior. Research Paper RMRS-RP-29, U.S. Department of
Agriculture, Forest Service, Rocky Mountain Research Sta-
tion, 2001. 2
[43] Belal Shaheen, Matthew David Zane, Bach-Thuan Bui,
Shubham, Tianyuan Huang, Manuel Merello, Ben Scheelk,
Steve Crooks, and Michael Wu.
Forestsplat:
Proof-of-
concept for a scalable and high-fidelity forestry mapping
tool using 3d gaussian splatting. Remote Sensing, 17(6):993,
2025. 2
[44] Jacob L. Strunk, Stephen E. Reutebuch, Robert J. Mc-
Gaughey, and Hans-Erik Andersen. An examination of gnss
positioning under dense conifer forest canopy in the pacific
northwest, usa. Remote Sensing Applications: Society and
Environment, 37(3):101428, 2025. 3
[45] Luke Wallace, Arko Lucieer, and et al. Assessment of forest
structure using two UAV techniques: A comparison of air-
borne laser scanning and structure from motion. Forests, 7
(3):62, 2016. 2
[46] Joanne C. White, Michael A. Wulder, Mikko Vastaranta,
Nicholas C. Coops, Douglas Pitt, and Murray Woods. The
utility of image-based point clouds for forest inventory: A
comparison with airborne laser scanning. Forests, 4(3):518–
536, 2013. 2
[47] Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, and Yan-
ning Zhang.
Sparse2dgs: Geometry-prioritized gaussian
splatting for surface reconstruction from sparse views. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR), 2025. 5
[48] Qiong Wu, Ruofei Zhong, Pinliang Dong, You Mo, and
Yunxiang Jin. Airborne LiDAR intensity correction based on
a new method for incidence angle correction for improving
land-cover classification. Remote Sensing, 13(3):511, 2021.
8
[49] Michael A. Wulder, Christopher W. Bater, Nicholas C.
Coops, Thomas Hilker, and Joanne C. White. The role of

<!-- page 17 -->
lidar in sustainable forest management. The Forestry Chron-
icle, 84(6):807–826, 2008. 2
[50] Binbin Xiang, Maciej Wielgosz, Stefano Puliti, Kamil Kr´al,
Martin Kr˚uˇcek, Azim Missarov, and Rasmus Astrup. Forest-
former3d: A unified framework for end-to-end segmenta-
tion of forest lidar 3d point clouds. In Proceedings of the
IEEE/CVF International Conference on Computer Vision
(ICCV), 2025. 4, 7, 8, 11, 13
[51] Wai Yeung Yan and Ahmed Shaker. Airborne LiDAR inten-
sity banding: Cause and solution. ISPRS Journal of Pho-
togrammetry and Remote Sensing, 142:301–310, 2018. 8
[52] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang,
Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in
gaussian splatting. arXiv preprint arXiv:2406.01467, 2024.
1, 2, 4, 5
