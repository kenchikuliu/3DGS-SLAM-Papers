<!-- page 1 -->
Broadening View Synthesis of Dynamic Scenes
from Constrained Monocular Videos
Le Jiang
Shaotong Zhu
Yedi Luo
Shayda Moezzi
Sarah Ostadabbas
Northeastern University, Boston, MA
{jiang.l, zhu.shaot, luo.ye, moezzi.s, s.ostadabbas}@northeastern.edu
Abstract
In dynamic Neural Radiance Fields (NeRF) systems,
state-of-the-art novel view synthesis methods often fail un-
der significant viewpoint deviations, producing unstable
and unrealistic renderings. To address this, we introduce
Expanded Dynamic NeRF (ExpanDyNeRF), a monocular
NeRF framework that leverages Gaussian splatting pri-
ors and a pseudo-ground-truth generation strategy to en-
able realistic synthesis under large-angle rotations.
Ex-
panDyNeRF optimizes density and color features to im-
prove scene reconstruction from challenging perspectives.
We also present the Synthetic Dynamic Multiview (SynDM)
dataset—the first synthetic multiview dataset for dynamic
scenes with explicit side-view supervision—created using
a custom GTA V-based rendering pipeline.
Quantitative
and qualitative results on SynDM and real-world datasets
demonstrate that ExpanDyNeRF significantly outperforms
existing dynamic NeRF methods in rendering fidelity under
extreme viewpoint shifts. Further details are provided in
the supplementary materials. Code is available at https:
//github.com/ostadabbas/ExpanDyNeRF.
1. Introduction
Novel view synthesis is essential in applications like mixed
reality [7, 31], medical supervision [30, 37], autonomous
driving [21, 40], and wildlife observation [20, 41]. Neural
Radiance Fields (NeRF) and their dynamic variants have
improved 3D reconstruction with high precision [24, 26,
32], speed [3, 4, 10], and style editing [1, 6].
Alterna-
tively, Gaussian splatting [27, 28, 34] offers an efficient and
flexible framework for high-quality rendering. While both
approaches produce sharp results from primary viewpoints,
renderings degrade with significant viewpoint shifts due to
the lack of diverse-view supervision during training, which
is a limitation of monocular settings (Fig. 1).
To overcome these limitations, we propose Expanded
Dynamic NeRF (ExpanDyNeRF), a dynamic NeRF frame-
work designed to expand reliable rendering to large-angle
novel views, even under monocular camera constraints.
This end-to-end pipeline, illustrated in Fig. 2, incorporates
a novel-view pseudo ground truth strategy that optimizes
model training from novel view by leveraging Gaussian pri-
ors [25], effectively refining dynamic object contours and
color consistency across frames.
One of the key challenges in novel view synthesis for
dynamic scenes lies in the lack of suitable datasets that of-
fer both dynamic motion and side-view supervision. Ex-
isting datasets either have dynamic camera motion without
side-view ground truth (e.g., NVIDIA [35]) or include ro-
tated views without camera motion (e.g., DyNeRF [11]).
This gap stems from the difficulty of capturing multi-view
dynamic scenes in the real world. To address this, we in-
troduce SynDM, a GTA V-based dataset with a novel dy-
namic camera dome that enables synchronized main-view
motion and side-view supervision. Our evaluation focuses
on SynDM, with only qualitative comparisons on existing
datasets. Our main contributions are:
• We identify and characterize the limitations of current
monocular dynamic NeRFs in rendering from signifi-
cantly deviated viewpoints, highlighting their inability to
preserve structure and appearance consistency under an-
gular shifts
• We propose ExpanDyNeRF, a novel dynamic NeRF ar-
chitecture that incorporates pseudo-novel view supervi-
sion using Gaussian splatting priors, enabling reliable
synthesis at large viewpoint deviations
• We introduce SynDM, the first synthetic dataset for dy-
namic monocular NeRFs with paired primary and rotated
views, captured via a custom GTA V pipeline to bench-
mark novel view synthesis under controlled angular per-
turbations
• We perform comprehensive experiments on the SynDM,
DyNeRF, and NVIDIA datasets, demonstrating improved
perceptual quality and geometric consistency over previ-
ous dynamic NeRF methods, especially when handling
large viewpoint deviations
1
arXiv:2512.14406v1  [cs.CV]  16 Dec 2025

<!-- page 2 -->
Figure 1. Comparison of novel-view rendering from constrained-pose monocular video. Leading dynamic NeRF and Gaussian splatting methods—DetRF
[39], RoDynRF [19], DecNeRF [36], and D3DGS [34]—all suffer from artifacts and depth errors at viewpoints distant from the training pose, whereas
ExpanDyNeRF (ours) maintains accurate shape and color consistency. The first column shows an example frame from the training video; the remaining
columns show novel-view renderings obtained by rotating that frame’s camera pose.
2. Related Work
NeRF-based Dynamic Novel View Synthesis. NeRF al-
gorithms have emerged as a powerful technique for high-
quality 3D scene reconstruction from a sparse set of images.
Original NeRF [15] leverages a fully connected deep neural
network to model the volumetric scene function. This func-
tion outputs the color and density for any given 3D point
and viewing direction, enabling the synthesis of novel views
through volume rendering techniques. NeRFs have been
particularly successful in static scenes, and recent advance-
ments have extended their application to dynamic 3D recon-
struction. Dynamic NeRF methods, such as HyperNeRF
[17], DetRF [39], and DecNeRF [36] incorporate temporal
components, allowing for the modeling of scenes with mov-
ing objects and varying illumination. These methods often
employ additional strategies like temporal consistency loss
and motion field modeling to handle the complexities of dy-
namic environments. Despite their success, dynamic NeRFs
face challenges with computational expense and the need
for densely sampled temporal data. Many dynamic NeRF
methods require substantial compute resources—e.g., Dyni-
bar [12] reports training times exceeding 1 day on 8 A100
GPUs for a single scene—highlighting the difficulty of scal-
ing dynamic NeRFs to large datasets or real-time applica-
tions.
Gaussian Splatting-based Dynamic Novel View Syn-
thesis. 3D Gaussian splatting [9] offers an efficient alterna-
tive to NeRF by representing scenes with Gaussian blobs,
enabling real-time, high-resolution rendering. Recent meth-
ods like 4DGS [27] and D3DGS [34] extend this approach
to dynamic scenes using monocular inputs, capturing non-
rigid motion via deformation fields. While NeRF provides
high-fidelity reconstructions and Gaussian splatting excels
in speed, both struggle with novel view synthesis from de-
viated angles in monocular settings. Our method combines
their strengths to overcome these limitations.
3. Method
We present ExpanDyNeRF, a monocular dynamic NeRF
framework for synthesizing novel views of 3D scenes under
large viewpoint deviations. To address the lack of ground
truth in monocular settings, we combine a two-branch dy-
namic NeRF with pseudo-supervision from 3D Gaussian
priors. Section 3.1 details the dynamic NeRF backbone,
Section 3.2 describes pseudo-novel view supervision using
3D Gaussian priors, and Section 3.3 introduces our SynDM
dataset with multi-view dynamic scenes.
3.1. ExpanDyNeRF Model Architecture
Our preliminary experiments indicate that NeRF provides
greater visual consistency than Gaussian Splatting, espe-
cially for distant elements like the sky (shown in Fig. 3).
Therefore, we adopt NeRF as the backbone of our model.
NeRF vs Gaussian Splatting Analysis: While both
methods appear visually consistent in primary views, sig-
nificant differences emerge in side view reconstruction. As
demonstrated in Fig. 3, 3DGS introduces substantial ar-
tifacts in side views, with the sky being incorrectly re-
constructed as nearby structures, obstructing distant back-
ground elements including mountains and bushes. In con-
trast, Instant-NGP (NeRF-based) retains the expected char-
acteristics of the sky as distant and uniform, achieving
higher fidelity and richer scene details. Accordingly, we
adopt a NeRF backbone for its stability under large view-
2

<!-- page 3 -->
Static
Encoder
𝛷𝑏
𝛷𝑓
𝑐𝑓
𝜎𝑓
3D Prior
Gaussian
Splatting
Prior
𝑃𝑛𝑣
𝑃𝑛𝑣
𝑟𝑖𝑔ℎ𝑡
𝑃𝑛𝑣
𝑙𝑒𝑓𝑡
𝐼𝑡
𝑐𝑏
𝜎𝑏
መ𝐼𝑡
መ𝐼0
⋯
⋯
መ𝐼𝑁
0°
+10°
+20°
−20°
−10°
Novel View Renderings for 𝑃𝑛𝑣
Novel View Feature Optimization
Backbone Dynamic NeRF
𝑃𝑡−1
⋯
Ray 𝑟
⋯
𝑃0 ⋯
⋯
𝑃𝑡𝑃𝑡+1
𝑃𝑁
𝛷𝑏: Background NeRF
𝑃: Primary Camera Pose
𝜎, 𝑐: Density, Color of Ray
𝛷𝑓: Foreground NeRF
𝑃𝑛𝑣: Novel Camera Pose
𝐼1
𝐼0
𝐼𝑁
Input 
Frames
ℒ𝑛𝑣
𝜎
ℒ𝑛𝑣
𝑐
Pseudo Ground Truth
Super 
Resolution
ℒ𝑠𝑟
Figure 2. The ExpanDyNeRF architecture is structured into two main components: (1) Backbone dynamic NeRF model that processes rays to extract
density (σ) and color (c) features from both background (Φb) and foreground (Φf) models, generating rendering predictions ˆIt from primary camera
positions, supervised by the super-resolution loss Lsr (2) Novel View Feature Optimization uses the Gaussian Splatting based method to generate a 3D
prior for each frame, facilitating the optimization of density and color features via pseudo ground truth for novel views. This includes updating σf and cf
using novel view loss metrics Lσ
nv and Lc
nv, respectively, enhancing feature representation across different perspectives.
3DGS
Instant-NGP
Primary 
View
Side 
View
Figure 3. Comparison between Instant-NGP [16] and 3DGS [9] from
primary and side views. While both methods appear consistent in primary
views (top), 3DGS introduces significant artifacts in side views (bottom
right), whereas Instant-NGP maintains better reconstruction quality.
point rotations, while using Gaussian Splatting only as a
prior generator rather than a rendering backbone. We do not
claim NeRF to be universally superior, but find this combi-
nation effective for the targeted setting.
Following the architecture in [39], we employ two inter-
twined neural networks: Φb for modeling the static back-
ground and Φf for the dynamic foreground. As illustrated
in Fig. 2, our framework is structured into two main compo-
nents: (1) a backbone dynamic NeRF model that processes
rays to extract density (σ) and color (c) features from both
background and foreground models, generating rendering
predictions ˆIt from primary camera positions, supervised
by the super-resolution loss Lsr; (2) novel view feature op-
timization that uses Gaussian splatting priors to generate 3D
representations for each frame, facilitating optimization of
density and color features via pseudo ground truth for novel
views. This includes updating σf and cf using novel view
loss metrics Lσ
nv and Lc
nv, respectively, enhancing feature
representation across different perspectives.
Preliminaries: We use N video frames It, t ∈[1, N],
to reconstruct a point cloud and estimate primary camera
poses P. For each pose Pt ∈P, we compute ray trajectories
to sample points x = (x, y, z) along ray r at time t in direc-
tion d. The sampling process is defined as F(Pt) →(x, d),
linking the camera orientation to the sampled points.
Static Background Representation: The static back-
ground module Φb takes all N frames and encodes static
scene components using a distribution-based representa-
tion.
This improves alignment between camera projec-
tions and the background geometry, allowing simplified
renderings of low-variance static features.
The module,
Φb(F(P), θ) →(cb, σb), predicts color cb and density σb
of spatial points from all poses in P, using a prior distribu-
tion (θ ∼PΘ(θ)).
Dynamic Foreground Representation: The dynamic
foreground module Φf captures temporal variation us-
ing a three-frame sliding window.
It integrates spa-
tial coordinates, viewing directions, and the timestamp t:
(Φf(F(Pt), t) →(cf, σf)). Temporal consistency is main-
tained by encoding time t directly and using optical flow to
estimate scene flow, predicting future states of dynamic ob-
jects. Continuity constraints are applied to maintain smooth
attribute transitions across frames, expressed as: Lcont =
P ∥σf(t + 1) −σf(t)∥2.
3

<!-- page 4 -->
−45°
45°
5°
Primary View
Deviated View
0° 15°
30°
45°
0°
Azimuth
𝑅𝑑𝑔
Figure 4. Pseudo ground truth generation demonstration. Novel view-
points (black cameras) are sampled around a dome centered on the fore-
ground object, with the primary view shown in red. Example renderings
and shape masks are shown for different viewing angles.
Primary View Reconstruction Loss: The system em-
ploys a reconstruction loss to optimize Φb and Φf by min-
imizing the discrepancies between the features ˆC(r) from
rendered images and C(r) from the ground truth images,
defined as Lrec = PN
i=1
P
r∈R ∥ˆC(r) −C(r)∥2
2.
This
loss ensures the renderings from the primary views closely
match the ground truth frames, setting up a baseline for the
following novel view optimization.
Super-Resolution Loss:
Inspired by SOTA super-
resolution methods [22, 23], we incorporate a super-
resolution loss (Lsr) to enhance image quality. Rendered
patches from ExpanDyNeRF are processed by a pre-trained
super-resolution model, which preserves fine textures while
increasing resolution.
The loss is computed by compar-
ing sampled patches from the predicted and corresponding
high-resolution reference. The formulation of Lsr is:
Lsr =
K
X
k=1
 ˆQk −Qk

1 +
K
X
k=1
X
l
λl
F l
vgg( ˆQk) −F l
vgg(Qk)

1 .
Here, ˆQk and Qk represent the super-resolution predic-
tion and reference patches, respectively, F l
vgg is a set of lay-
ers in a pretrained VGG-19 feature extractor, and λl is the
reciprocal of the number of neurons in layer l, combining
reconstruction and perceptual losses.
3.2. Pseudo Ground Truth Optimization Strategy
Through empirical experiments, we observed that fore-
ground objects appear blurrier than the background during
viewpoint rotation. This is due to affine distortion, where
nearby objects exhibit greater apparent motion than distant
ones. To address this, we prioritize optimizing foreground
representations.
Our method leverages FreeSplatter [25]
to generate high-quality 3D mesh priors, enabling pseudo-
ground truth supervision from novel viewpoints.
Pseudo Ground Truth Generation for Novel Views:
For each input frame It, we construct a 3D Gaussian prior
representing the foreground object in its local coordinate
frame.
Centered around this object, we define a dome-
shaped sampling space, with a radius Rd as shown in Fig. 4.
The radius Rd represents the distance from the primary
viewpoint to the object. The position of Pt on the dome is
denoted as (elevation = e, azimuth = 0, radius = Rd),
where e corresponds to the elevation angle of the primary
recording view.
Novel Viewpoint Sampling Strategy: We systematically
sample novel viewpoints by varying both azimuth and ele-
vation angles while maintaining the fixed radius Rd. Specif-
ically, we generate viewpoints spanning azimuth angles
from −45◦to 45◦in 5◦increments, at three elevation levels:
0◦, 15◦, and 30◦. The forward vector of all camera poses
P (d)
nv points towards the dome center, ensuring consistent
object framing across viewpoints. From these novel camera
poses, we render pseudo ground truth images that encode
the expected density and color distributions of the fore-
ground object. The resulting renderings provide both RGB
color and corresponding shape masks at different viewing
angles, creating comprehensive supervision that would be
impossible to obtain from real monocular capture.
Mapping Novel Views to the NeRF Coordinate System:
To apply pseudo ground-truth supervision at novel view-
points, we transform the sampled camera poses from the
Gaussian prior coordinate system to the NeRF coordinate
system using a rigid alignment matrix.
Let the primary
camera pose Pt in the foreground NeRF coordinate sys-
tem be P (n)
t
, and its corresponding pose in Gaussian prior
coordinate system be P (d)
t
. The transformation matrix T
that aligns the two coordinate systems is computed as:
T = P (n)
t
· (P (d)
t
)−1. We apply this transformation, T, to
each novel view camera pose, P (d)
nv , sampled in the Gaus-
sian prior coordinate system, to transfer all new camera po-
sitions to the foreground NeRF coordinate system:Pnv =
{P · T, ∀P ∈P (d)
nv }.
Novel View Loss: During each training iteration, two sym-
metric novel views are randomly sampled per frame from
Pnv. For each selected novel viewpoint, a set of rays Rnv
is sampled from the camera pose. Color and density predic-
tions in the foreground NeRF are obtained by evaluating the
network Φf on sampled ray (F(Pnv), t), producing outputs
(cf, σf). Specifically, F(Pnv) →(x, d) samples points
along a ray rnv ∈Rnv. We then integrate the predicted
color and density values along each ray rnv to produce the
corresponding pixel-wise predictions ˆCf(rnv) and ˆσf(rnv).
The corresponding novel view loss is defined as:
Lnv = Lc
nv + Lσ
nv
=
X
r∈Rnv
 ∥ˆC(rnv) −C(rnv)∥2
2 + ∥ˆσ(rnv) −σ(rnv)∥2
2

.
where C(rnv) and σ(rnv) denote the pseudo ground
truth color and density values for ray rnv, respectively. This
loss encourages the model to match the rendered appear-
ance and structure of the pseudo-supervised novel views,
4

<!-- page 5 -->
improving generalization to unseen angles. A further ex-
planation and visualization of ray sampling strategies are
detailed in Fig. 8 in the supplementary. To manage the risk
of exploding gradients early in training, we defer inclusion
of Lnv until after a fixed number of epochs. The final total
loss function is given by:
L = Lcont + Lrec + Lsr + Lnv.
3.3.
Synthetic
Dynamic
Multiview
(SynDM)
Dataset
To enable quantitative evaluation of novel view synthe-
sis under significant viewpoint deviations, we introduce
our SynDM dataset, built using the high-fidelity simulation
platform GTA V. The game offers rich dynamic environ-
ments and realistic rendering, making it an ideal founda-
tion. However, a core limitation of GTA V is its support for
only a single active viewport, which posing a challenge for
synchronized multi-view dynamic scene capture. We ex-
tend the GTAV-TeFS [14] framework–originally developed
for dual-camera capture –into a generalized multi-camera
pipeline to simultaneously support both monocular primary
camera capture and multi-view stereo camera collection in
GTA V’s dynamic environment. Traditionally, collecting
data from multiple camera views in a single-viewport en-
gine requires frame swapping, where each camera is ren-
dered sequentially. Under a 60 Hz refresh rate, this results
in a latency of at least 16.7 ms per camera swap. While
acceptable for static scenes, this approach quickly breaks
down in dynamic settings and as we add cameras, as the
accumulated latency introduces motion misalignment and
temporal artifacts. To address this we developed a custom
plugin that semi-freezes the game’s graphical state while
allowing the rendering and physics engine to continue run-
ning. This design enables us to cycle through camera views
in a controlled and consistent manner during a single logical
frame. With precise scheduling, we reduced the per-swap
latency from 16.7ms to just 0.2ms, making high-resolution,
low-latency multi-view capture of dynamic scenes possible.
Our dataset enables synthetic object tracking via syn-
chronized multi-view recordings, offering a robust ground
truth for evaluating dynamic NeRF-based novel view syn-
thesis. It consists of nine distinct scenes spanning three
categories–humans, vehicles, and animals (Fig. 5). Each
scene is captured using 22 cameras: 19 are distributed hori-
zontally around a reference point at 5◦intervals from −45◦
to 45◦, including a central anchor camera, while the remain-
ing three are elevated vertically at −45◦, 0◦, and 45◦. All
frames are rendered at a resolution of 1920×1080 with a 90◦
horizontal and 59◦vertical field of view.
Necessity
and
Advantage
of
Proposed
SynDM
Dataset As shown in Table 1, existing datasets lack crit-
ical features for evaluating dynamic novel view synthe-
sis under large viewpoint deviations.
Most importantly,
Table 1.
Comparison of dynamic 3D reconstruction datasets.
Columns: Multi-view (MV), Deviated View Ground Truth (De-
viated View GT), Unconstrained Scene (Unconst. Scene), Camera
Motion (Cam. Motion), and Background (Bkg.). SynDM is the
only dataset supporting all five attributes.
Dataset
MV Deviated View GT Unconst. Scene Cam. Motion Bkg.
DAVIS [18]
✗
✗
✓
✓
✓
iPhone [2]
✗
✗
✓
✓
✓
NeRFDS [33]
✗
✗
✓
✓
✓
NVIDIA [35]
✗
✗
✓
✓
✓
HyperNeRF [17]
✗
✗
✗
✓
✓
DyNeRF [11]
✓
✓
✗
✗
✓
ActorsHQ [8]
✓
✓
✗
✗
✗
Multi-face [29]
✓
✓
✗
✗
✗
SynDM (Ours)
✓
✓
✓
✓
✓
Figure 5.
Gallery of images from our SynDM dataset, showcasing a
variety of subjects. Animals are featured in the top row, humans in the
middle row, and vehicles in the bottom row.
no real-world dataset provides deviated view ground truth,
severely limiting quantitative assessment beyond primary
viewpoints. Our SynDM dataset uniquely combines all es-
sential features, including multi-view data, deviated GT,
full-scene representation, and camera motion, enabling
comprehensive evaluations previously impossible with ex-
isting datasets alone.
4. Experimental Results
We present a comprehensive evaluation of ExpanDyNeRF
against state-of-the-art dynamic novel view synthesis meth-
ods.
Our evaluation strategy progresses from controlled
synthetic environments to challenging real-world scenarios,
demonstrating robustness across diverse settings. More re-
sults can be find in our Supplementary Material.
4.1. Experimental Setup
Datasets.
We evaluate ExpanDyNeRF on three datasets
with complementary characteristics. Our SynDM Dataset
provides complete ground truth for quantitative analysis
across three scene categories (human, animals, vehicles)
spanning rural and urban environments within GTA V sim-
ulation.
For training, we use the first 24 frames from
each scene and evaluate on 12 novel views, uniformly sam-
5

<!-- page 6 -->
D4NeRF
RoDynRF
MonoNeRF
(Ours)
GT
4DGS
0°
+30°
−30°
ExpanDyNeRF
D3DGS
DetRF
DecNeRF
Figure 6. Comparison of dynamic NeRF models on SynDM dataset. Each column shows a method’s performance across different rotation angles, with
ground truth in blue. Red boxes highlight key differences, showing ExpanDyNeRF’s superior color and shape fidelity.
Table 2. Quantitative results on SynDM; best scores are bolded, second-best scores are in blue.
Method
Human
Animal
Vehicle
Average
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
D3DGS [34]
87.83
16.97
0.305
315.3
18.55
0.272
267.8
14.70
0.543
223.6
16.74
0.373
RoDynRF [13]
167.3
19.66
0.318
262.0
21.00
0.302
285.3
16.31
0.395
238.2
18.99
0.338
DecNeRF [36]
178.5
17.46
0.441
287.3
15.11
0.545
211.3
14.76
0.562
225.7
15.78
0.516
DetRF [39]
200.0
21.27
0.408
290.6
22.69
0.378
226.4
17.07
0.521
239.0
20.34
0.436
ExpanDyNeRF (Ours)
85.61
21.71
0.182
155.8
23.66
0.142
186.7
17.21
0.341
142.7
20.86
0.209
pled between −30◦to +30◦at 5◦intervals. The DyNeRF
Dataset [11] offers real-world dynamic scenes with multi-
view ground truth, enabling quantitative evaluation under
challenging viewing conditions. The NVIDIA Dataset [35]
provides real-world monocular sequences for generaliza-
tion assessment, though without deviated viewpoint ground
truth (see Table 1 for detailed comparison).
Implementation Details. Implementation Details. Ex-
panDyNeRF is trained with per-scene optimization on 2
A100 GPUs for 300k iterations (approximately 15 hours per
scene). The training cost is primarily dominated by NeRF
optimization with pseudo-novel view supervision, while
Gaussian prior generation is performed once per frame as
a preprocessing step and incurs marginal overhead. Con-
sistent with prior dynamic NeRF approaches, our method
prioritizes rendering fidelity under large viewpoint devia-
tions over training efficiency. Loss coefficients are set to:
λc
nv = 1.0, λσ
nv = 0.1, and λsr = 0.5. All other parameters
follow [38].
PSNR Limitation Analysis. Our experiments reveal im-
portant limitations of PSNR in evaluating perceptual qual-
ity. Despite ExpanDyNeRF producing sharper, more de-
tailed reconstructions, PSNR scores may paradoxically be
lower due to localized high-density errors highlighted in
pixel-wise error heatmaps. In contrast, baseline methods
with blurry results yield smoother transitions, leading to
lower MSE despite poorer visual quality. This occurs be-
cause blurry regions (e.g., object boundaries, fine textures)
blend into backgrounds, minimizing MSE contributions
(shown in Fig. 9 in the supplementary). This phenomenon
demonstrates why complementary perceptual metrics like
LPIPS and FID are essential for comprehensive quality as-
sessment, particularly when evaluating sharpness and clar-
ity improvements.
4.2. Evaluation on SynDM Dataset
The SynDM dataset enables comprehensive quantitative
evaluation with complete ground truth across diverse dy-
namic scenes.
We compare ExpanDyNeRF against four
state-of-the-art methods using PSNR, LPIPS, and FID met-
rics (Table 2).
Quantitative Results.
ExpanDyNeRF achieves the
highest PSNR score of 20.86 while producing the sharpest
renderings.
Although DetRF also reports competitive
PSNR despite producing heavily blurred images, this un-
derscores the importance of complementary perceptual met-
rics. Our model significantly outperforms competing meth-
ods in both LPIPS (38% lower than second-best) and FID
(36% lower than second-best), demonstrating superior per-
ceptual alignment and distributional consistency.
Qualitative Analysis.
Fig. 6 illustrates ExpanDyN-
eRF’s superior shape coherence and color stability in dy-
namic regions. Each baseline method exhibits distinct fail-
ure patterns: DecNeRF renders sharp details but suffers
from poor depth perception, resulting in flat, cardboard-
like appearances that fail to maintain 3D structure consis-
6

<!-- page 7 -->
Table 3. Quantitative results on DyNeRF dataset. We show Cof-
fee and Beef scenes which represent the two primary scenarios in
this dataset as shown in the first and second scenes of DyNeRF in
Fig. 1. Best scores are bolded, second-best scores are in blue.
Method
Coffee
Beef
FID↓
PSNR↑LPIPS↓
FID↓
PSNR↑LPIPS↓
D3DGS [34]
178.5
28.45
0.227
172.8
32.78
0.268
RoDynRF [13]
156.3
29.13
0.268
148.2
33.45
0.252
DecNeRF [36]
152.8
28.87
0.304
155.4
33.12
0.285
DetRF [39]
165.2
29.52
0.275
151.3
33.99
0.259
ExpanDyNeRF (Ours) 132.4
30.32
0.189
135.8
34.92
0.195
Table 4. Quantitative comparison on NVIDIA dynamic scenes.
We show Skate and Truck scenes which correspond to the first and
second scenes of NVIDIA dataset in Fig. 1. Best is bold; second-
best is blue.
Method
Skate
Truck
FID↓
PSNR↑LPIPS↓
FID↓
PSNR↑LPIPS↓
D3DGS [34]
142.8
25.67
0.258
95.7
26.39
0.239
RoDynRF [13]
103.5
27.89
0.087
78.4
29.13
0.063
DecNeRF [36]
112.9
26.83
0.134
85.6
27.56
0.115
DetRF [39]
84.34
29.45
0.072
67.69
31.75
0.041
ExpanDyNeRF (Ours) 90.83
28.91
0.079
69.37
30.60
0.034
tency.
D3DGS introduces depth inconsistencies causing
foreground objects to fracture under rotation, particularly
evident after 30◦viewpoint changes where object parts ap-
pear disconnected. RoDynRF struggles with consistent ob-
ject placement, often producing floating artifacts or mis-
aligned body parts. These failure modes stem from insuf-
ficient side-view supervision during training. Further visu-
alizations of results are provided in Fig. 10 and 11 in the
supplementary.
4.3. Evaluation on DyNeRF Dataset
The DyNeRF dataset provides real-world dynamic scenes
with multi-view ground truth, but presents challenges for
monocular training due to its stationary camera setup. Un-
like our SynDM dataset which provides natural camera
motion, DyNeRF’s fixed camera positions prevent direct
COLMAP reconstruction from monocular sequences. To
address this limitation, we construct synthetic monocular
sequences by selecting frames from central cameras (cam0,
cam4, cam5, cam6) across different timestamps, creating
the necessary camera motion for COLMAP initialization.
We then adopt a holdout validation strategy using these
central cameras for training while reserving the geomet-
rically challenging outer cameras (cam01 and cam10) as
test views—specifically selected as the most deviated view-
points from the training set’s central viewing positions.
Quantitative Results. Table 3 shows our method’s con-
sistent superiority across all baseline methods on repre-
sentative DyNeRF scenes.
ExpanDyNeRF achieves the
best performance across all metrics in both scenes. Our
method demonstrates particularly strong performance on
Coffee (FID: 132.4, PSNR: 30.32, LPIPS: 0.189) and Beef
sequences (FID: 135.8, PSNR: 34.92, LPIPS: 0.195), with
consistent improvements across all three evaluation metrics
in both representative scenarios.
Qualitative Analysis.
As shown in Fig. 1, our
method maintains superior performance on DyNeRF
scenes.
Leading dynamic NeRF and Gaussian splatting
methods—DetRF, RoDynRF, DecNeRF, and D3DGS—all
suffer from artifacts and depth errors at viewpoints dis-
tant from the training pose, whereas ExpanDyNeRF main-
tains accurate shape and color consistency. The improve-
ments are especially significant for the challenging view-
points, which demand accurate geometric reasoning due
to their significant displacement from training views, con-
firming that our pseudo-ground truth supervision strategy
successfully addresses fundamental geometric consistency
challenges in dynamic 4D scene reconstruction.
4.4. Generalization to NVIDIA Dataset
The NVIDIA dataset provides real-world monocular se-
quences captured with stationary cameras, offering insights
into our method’s generalization capabilities under differ-
ent capture conditions. However, this dataset presents lim-
itations for evaluating large viewpoint deviations: the 12
cameras are positioned in a compact matrix formation with
relatively small angular separations, lacking the significant
viewpoint variations needed for rigorous novel view synthe-
sis evaluation. Consequently, unlike DyNeRF and SynDM
datasets, NVIDIA lacks ground truth for challenging side-
view positions, limiting quantitative assessment to mod-
est viewpoint changes.
As a result, performance on the
NVIDIA dataset should be interpreted as a complementary
indicator of real-world robustness, rather than a definitive
evaluation of large-angle novel view synthesis.
Quantitative Analysis. Table 4 shows competitive per-
formance on representative NVIDIA scenes. While we do
not achieve the highest scores, this reflects the dataset’s
unique characteristics: 12 stationary cameras (using for
both train and test) positioned primarily in front of the
scene, favoring methods optimized for primary viewpoints.
Our approach demonstrates strong performance on Skate
(FID: 90.83, PSNR: 28.91, LPIPS: 0.079), and Truck (FID:
69.37, PSNR: 30.60, LPIPS: 0.034), consistently achieving
second-best results. DetRF achieves superior performance
on primary viewpoints due to its focus on depth estimation
from forward-facing cameras, while our method’s relatively
balanced performance across different viewpoints suggests
more stable behavior beyond primary training views. This
trade-off validates our design choice to prioritize robustness
over dataset-specific optimization.
Qualitative Results. Fig. 1 reveals notable limitations
in existing methods following camera rotation. We com-
7

<!-- page 8 -->
Ground Truth
𝑳𝒏𝒗
𝝈only
𝑳𝒏𝒗
𝝈+ 𝑳𝒏𝒗
𝒄
𝑳𝒏𝒗
𝒄
only
Baseline
𝑳𝒏𝒗
𝝈+ 𝑳𝒏𝒗
𝒄+ 𝑳𝒔𝒓
𝑳𝒔𝒓 only
Figure 7. The images above, rendered from a view rotated by -30 degrees, illustrate the impact of different loss functions on the quality of novel view
synthesis. The best performance is achieved when all loss functions (Lσ
nv + Lc
nv + Lsr) are applied simultaneously, highlighting the complementary role
each loss plays in enhancing rendering quality and achieving a closer match to the ground truth.
Table 5. Quantitative Evaluation of optimization strategies on the SynDM Dataset. Baseline is without any optimization, Lσ
nv only uses
density optimization, Lc
nv only applies color optimization, and Lσ
nv + Lc
nv is trained with both. Lσ
nv + Lc
nv + Lsr includes all modules
including super resolution.
Method
Human
Animal
Vehicle
Average
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
FID↓
PSNR↑
LPIPS↓
Baseline
200.0
21.27
0.408
290.6
22.69
0.378
226.4
17.07
0.521
239.0
20.34
0.436
Lσ
nv
158.2
21.56
0.399
237.8
23.57
0.374
186.7
16.69
0.524
194.2
20.61
0.432
Lc
nv
147.5
20.10
0.395
162.8
19.88
0.442
210.9
16.17
0.525
173.7
18.72
0.454
Lsr
147.8
20.85
0.212
194.6
21.30
0.183
176.6
16.62
0.331
173.0
19.59
0.242
Lσ
nv + Lc
nv
146.2
21.66
0.395
207.9
23.69
0.339
198.8
17.14
0.537
184.3
20.83
0.424
Lσ
nv + Lc
nv + Lsr
85.61
21.71
0.182
155.8
23.66
0.142
144.9
17.21
0.304
142.7
20.86
0.209
pare against leading dynamic NeRF and Gaussian splatting
methods including DetRF [39], RoDynRF [19], DecNeRF
[36], and D3DGS [34]. All baseline methods suffer from
artifacts and depth errors at viewpoints distant from train-
ing poses.
Specifically, RoDynRF struggles to maintain
structural consistency—rendered figures may retain correct
foot placement yet exhibit misaligned upper bodies that
appear ”stuck” to background elements like pillars in the
skate scene. DecNeRF exhibits cardboard-like foreground
appearance or complete disappearance in side views, re-
flecting insufficient depth estimation that leads to extremely
thin or missing foreground rendering from oblique angles.
D3DGS shows spatial fragmentation stemming from absent
side-view supervision, hindering accurate depth estimation.
In contrast, ExpanDyNeRF maintains accurate shape and
color consistency across the evaluated rotation angles, ef-
fectively addressing common failure modes such as frac-
tured object reconstructions and temporal inconsistency.
Further visualizations of results on the NVIDIA dataset are
provided in Figs. 12 and 13 in the supplementary.
4.5. Ablation Study
We conduct a comprehensive ablation study to validate the
contribution of each component in our optimization strat-
egy. The analysis examines the effects of our novel view
losses and super-resolution enhancement.
Component Analysis. Fig. 7 demonstrates the visual
impact of different loss combinations on novel view synthe-
sis quality. The baseline model without novel view super-
vision produces poorly defined shapes and noticeable blur-
ring, indicating the critical importance of side-view guid-
ance. Introducing only color-based novel view loss Lc
nv
reduces blurring but introduces white artifacts due to ab-
sent shape constraints provided by the density supervision.
Conversely, using only Lσ
nv helps preserve object shape but
results in faded or distorted colors. The synergistic effect
between Lσ
nv and Lc
nv is evident when combined, lead-
ing to improved structural clarity and temporal stability, al-
though fine textures remain underrepresented. The super-
resolution loss Lsr proves essential for texture enhancement
but requires the foundation of geometric consistency from
novel view losses. Using Lsr in isolation produces arti-
facts and inconsistencies, confirming that high-level seman-
tic supervision must precede low-level texture refinement.
Importantly, Lsr primarily improves perceptual sharpness
and does not introduce new geometric constraints. Without
the geometric consistency established by Lσ
nv and Lc
nv, it
tends to amplify existing structural errors rather than cor-
rect them.
Quantitative Validation. Table 5 confirms these visual
observations across all metrics. The best performance is
achieved when integrating Lσ
nv, Lc
nv, and super-resolution
loss Lsr. Notably, using Lsr in isolation fails to produce ac-
ceptable renderings, underscoring the essential role of novel
view supervision in mitigating artifacts during viewpoint ro-
tations. The full combination achieves optimal perceptual
quality, confirming our optimization strategy’s effectiveness
in reducing visual artifacts and enhancing overall realism.
8

<!-- page 9 -->
5. Discussion and Conclusion
Limitations. Despite outperforming existing models, Ex-
panDyNeRF has limitations, particularly in handling ex-
treme viewing angles (beyond 45 degrees) and unseen back-
ground generation. Also, ExpanDyNeRF requires per-scene
optimization, resulting in higher computational cost com-
pared to feed-forward or purely Gaussian-based methods.
Conclusion.
ExpanDyNeRF advances dynamic NeRF
by significantly improving novel view synthesis, partic-
ularly at wider viewing angles, by extending the range
of stable visualization.
Our SynDM dataset, based on
GTA V for dynamic multiview scenarios, provides a
strong foundation for evaluating dynamic scene recon-
structions from varied angles.
Our evaluations demon-
strate ExpanDyNeRF’s superior ability to render dynamic
scenes.
References
[1] Yaosen Chen, Qi Yuan, Zhiqiang Li, Yuegen Liu, Wei Wang,
Chaoping Xie, Xuming Wen, and Qien Yu. Upst-nerf: Uni-
versal photorealistic style transfer of neural radiance fields
for 3d scene. IEEE Transactions on Visualization and Com-
puter Graphics, 2024. 1
[2] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell,
and Angjoo Kanazawa. Monocular dynamic view synthesis:
A reality check. Advances in Neural Information Processing
Systems, 35:33768–33780, 2022. 5
[3] Xinyu Gao, Ziyi Yang, Yunlu Zhao, Yuxiang Sun, Xiao-
gang Jin, and Changqing Zou. A general implicit framework
for fast nerf composition and rendering. In Proceedings of
the AAAI Conference on Artificial Intelligence, pages 1833–
1841, 2024. 1
[4] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie
Shotton, and Julien Valentin. Fastnerf: High-fidelity neural
rendering at 200fps. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages 14346–
14355, 2021. 1
[5] Rafael C Gonzalez. Digital image processing. Pearson edu-
cation india, 2009. 1
[6] Jiatao
Gu,
Lingjie
Liu,
Peng
Wang,
and
Christian
Theobalt.
Stylenerf:
A style-based 3d-aware genera-
tor for high-resolution image synthesis.
arXiv preprint
arXiv:2110.08985, 2021. 1
[7] Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu,
Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, and
Mohammed Bennamoun. Ue4-nerf: Neural radiance field for
real-time rendering of large-scale scene. Advances in Neural
Information Processing Systems, 36, 2024. 1
[8] Mustafa Is¸ık, Martin R¨unz, Markos Georgopoulos, Taras
Khakhulin, Jonathan Starck, Lourdes Agapito, and Matthias
Nießner. Humanrf: High-fidelity neural radiance fields for
humans in motion. ACM Transactions on Graphics (TOG),
42(4):1–12, 2023. 5
[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3
[10] Byeonghyeon Lee, Howoong Lee, Usman Ali, and Eun-
byung Park. Sharp-nerf: Grid-based fast deblurring neural
radiance fields using sharpness prior. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 3709–3718, 2024. 1
[11] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 5521–5531, 2022. 1, 5,
6
[12] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker,
and Noah Snavely. Dynibar: Neural dynamic image-based
rendering.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4273–
4284, 2023. 2
[13] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu
Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Jo-
hannes Kopf, and Jia-Bin Huang. Robust dynamic radiance
fields. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 13–23, 2023. 6,
7
[14] Yedi Luo, Xiangyu Bai, Le Jiang, Aniket Gupta, Eric
Mortin, and Hanumant Singh Sarah Ostadabbas. Temporal-
controlled frame swap for generating high-fidelity stereo
driving data for autonomy analysis.
arXiv preprint
arXiv:2306.01704, 2023. 5
[15] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[16] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 3
[17] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2, 5
[18] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Ar-
bel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017
davis challenge on video object segmentation. arXiv preprint
arXiv:1704.00675, 2017. 5
[19] Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin,
David J Fleet, and Andrea Tagliasacchi.
Robustnerf: Ig-
noring distractors with robust losses.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20626–20636, 2023. 2, 8
[20] Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ig-
nacio Rocco, Natalia Neverova, Andrea Vedaldi, and David
Novotny.
Common pets in 3d: Dynamic new-view syn-
thesis of real-life deformable categories. In Proceedings of
9

<!-- page 10 -->
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 4881–4891, 2023. 1
[21] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron,
and Henrik Kretzschmar. Block-nerf: Scalable large scene
neural view synthesis. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
8248–8258, 2022. 1
[22] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang,
Yu-Wing Tai, and Shi-Min Hu. Nerf-sr: High quality neural
radiance fields using supersampling. In Proceedings of the
30th ACM International Conference on Multimedia, pages
6445–6454, 2022. 4
[23] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan.
Real-esrgan: Training real-world blind super-resolution with
pure synthetic data. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 1905–1914,
2021. 4
[24] Yifan Wang, Yi Gong, and Yuan Zeng. Hyb-nerf: A mul-
tiresolution hybrid encoding for neural radiance fields. In
2024 IEEE/CVF Winter Conference on Applications of Com-
puter Vision (WACV), pages 3677–3686. IEEE, 2024. 1
[25] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee.
Freesplat: Generalizable 3d gaussian splatting to-
wards free-view synthesis of indoor scenes. arXiv preprint
arXiv:2405.17958, 2024. 1, 4
[26] Zhongshu Wang, Lingzhi Li, Zhen Shen, Li Shen, and
Liefeng Bo.
4k-nerf: High fidelity neural radiance fields
at ultra high resolutions. arXiv preprint arXiv:2212.04701,
2022. 1
[27] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
arXiv preprint arXiv:2310.08528, 2023. 1, 2
[28] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-
Pei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d
gaussian splatting. arXiv preprint arXiv:2403.11134, 2024.
1
[29] Cheng-hsin Wuu, Ningyuan Zheng, Scott Ardisson, Rohan
Bali, Danielle Belko, Eric Brockmeyer, Lucas Evans, Tim-
othy Godisart, Hyowon Ha, Xuhua Huang, et al.
Multi-
face: A dataset for neural face rendering.
arXiv preprint
arXiv:2207.11243, 2022. 5
[30] Magdalena Wysocki, Mohammad Farid Azampour, Chris-
tine Eilers, Benjamin Busam, Mehrdad Salehi, and Nassir
Navab.
Ultra-nerf: neural radiance fields for ultrasound
imaging. In Medical Imaging with Deep Learning, pages
382–401. PMLR, 2024. 1
[31] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia,
Aayush Bansal, Changil Kim, Samuel Rota Bul`o, Lorenzo
Porzi, Peter Kontschieder, Aljaˇz Boˇziˇc, et al. Vr-nerf: High-
fidelity virtualized walkable spaces.
In SIGGRAPH Asia
2023 Conference Papers, pages 1–12, 2023. 1
[32] Yiwei Xu,
Tengfei Wang,
Zongqian Zhan,
and Xin
Wang.
Mega-nerf++:
An improved scalable nerfs for
high-resolution photogrammetric images. The International
Archives of the Photogrammetry, Remote Sensing and Spa-
tial Information Sciences, 48:769–776, 2024. 1
[33] Zhiwen Yan, Chen Li, and Gim Hee Lee. Nerf-ds: Neural ra-
diance fields for dynamic specular objects. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8285–8295, 2023. 5
[34] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 20331–20341, 2024. 1, 2, 6,
7, 8
[35] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park,
and Jan Kautz. Novel view synthesis of dynamic scenes with
globally coherent depths from a monocular camera. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 5336–5345, 2020. 1, 5, 6
[36] Meng You and Junhui Hou. Decoupling dynamic monocular
videos for dynamic view synthesis. IEEE Transactions on
Visualization and Computer Graphics, 2024. 2, 6, 7, 8
[37] Zhengming Yu, Wei Cheng, Xian Liu, Wayne Wu, and
Kwan-Yee Lin.
Monohuman:
Animatable human neu-
ral field from monocular video.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16943–16953, 2023. 1
[38] Boyu Zhang, Wenbo Xu, Zheng Zhu, and Guan Huang.
Detachable novel views synthesis of dynamic scenes using
distribution-driven neural radiance fields.
arXiv preprint
arXiv:2301.00411, 2023. 6
[39] Boyu Zhang, Zheng Zhu, and Wenbo Xu.
Detrf:
De-
tachable novel views synthesis of dynamic scenes using
backdrop-driven neural radiance fields. In Proceedings of
the AAAI Conference on Artificial Intelligence, pages 9860–
9868, 2025. 2, 3, 6, 7, 8
[40] Junge Zhang, Feihu Zhang, Shaochen Kuang, and Li Zhang.
Nerf-lidar: Generating realistic lidar point clouds with neural
radiance fields. In Proceedings of the AAAI Conference on
Artificial Intelligence, pages 7178–7186, 2024. 1
[41] Shangzhan Zhang, Sida Peng, Yinji ShenTu, Qing Shuai,
Tianrun Chen, Kaicheng Yu, Hujun Bao, and Xiaowei Zhou.
Dyn-e: Local appearance editing of dynamic neural radiance
fields. arXiv preprint arXiv:2307.12909, 2023. 1
10

<!-- page 11 -->
Broadening View Synthesis of Dynamic Scenes
from Constrained Monocular Videos
Supplementary Material
Overview of Supplementary Materials This supple-
mentary document provides extended experimental results,
ablations, and visual comparisons that complement the
main paper.
In particular, we include detailed ablation
studies on our ray sampling and padding strategies (Fig. 8)
and error heatmaps illustrating the limitations of standard
metrics such as PSNR and MSE (Fig. 9). Furthermore,
we include additional novel view synthesis visualizations
across both the SynDM and NVIDIA datasets in Figs.
10–13. These results further validate the effectiveness of
ExpanDyNeRF in capturing sharper, more consistent scene
details across diverse scenarios. For an interactive overview
with richer visualizations, readers are encouraged to view
the accompanying IO page, which showcases additional
qualitative examples and video results.
Ray Sampling Strategies We compared various ray
sampling strategies for novel view density and color
optimization in Equation 3.2.
Examples are shown in
Fig. 8. Global sampling over the whole frame yields results
in Panel (e) similar to the base output in Panel (a), due
to the small proportion of dynamic segments in the frame,
causing generalized and ineffective updates.
Alternate
strategies sample within the foreground object’s area shown
white in Panel (c), which may overlook updates outside this
zone. Panel (f) demonstrates that sampling from various
viewpoints for dynamic density updates can unintentionally
extend beyond the intended mask, causing non-dynamic
areas to obscure the background.
Panel (g) shows the
third strategy where the GaussianBlur [5] expands the
foreground boundary, creating a zero gray-scale edge.
Sampling within this blurred mask improves results, yet
areas adjacent to the person still see undue dynamic density
updates beyond the motion mask.
Our final strategies
focused on ray sampling within the padded area of the
mask’s bounding box (bounding boxes in Panel (c)), which
outperforms the other strategies. Experimentation showed
that while larger padding, like 10 pixels in Panel (h),
achieves comparable foreground optimization to smaller
padding, such as 2 pixels in Panel (d), it adversely affects
background clarity.
1

<!-- page 12 -->
𝑎
𝑏
𝑐
𝑑
𝑒
𝑓
𝑔
ℎ
Figure 8. This figure presents an ablation study on our ray sampling strategy. Panel (a) displays the base model’s output without optimization for
color and density. Panel (b) depicts the pseudo-ground-truth of novel views from the created 3D mesh. Panel (c) illustrates the density mask derived
from pseudo ground truth, where the yellow and blue boxes represent the bounding box with 2-pixel padding, and 10-pixel padding, respectively. Panel
(e) shows predictions from global ray sampling on the mask, while panel (f) shows predictions from ray sampling within the foreground object area
only. Panel (g) demonstrates the GaussianBlur strategy’s prediction. Panels (d) and (h) showcase predictions with 2-pixel and 10-pixel padding,
respectively. The comparison between panels (d) and (h) reveals that employing 2-pixel padding leads to enhanced quality in reconstructing novel
view details with minimum background distortion.
GT
Baseline
PSNR: 20.43
MSE: 588.95
ExpanDyNeRF
PSNR: 20.20
MSE: 620.98
GT
Baseline
PSNR: 21.77
MSE: 432.59
ExpanDyNeRF
PSNR: 20.67
MSE: 557.29
40000
30000
0
20000
20000
Pixel-wise 
Square Error
10000
Figure 9. Comparison of visual and quantitative results between the baseline and ExpanDyNeRF models, evaluated using PSNR and MSE metrics. The
“GT” column represents the ground truth images. The red and green boxes highlight critical regions of interest for analysis. The red box demonstrates areas
with sharp and detailed reconstruction by ExpanDyNeRF, whereas the baseline exhibits significant blur. Despite ExpanDyNeRF producing clearer outputs,
PSNR scores are lower due to localized high-density errors (highlighted in the pixel-wise error heatmap). In contrast, the baseline’s blurry results yield
smoother transitions, leading to lower MSE despite poorer visual quality. The green box further illustrates how blurry regions (e.g., human pants or chicken
body) blend into the background, minimizing MSE contributions. This demonstrates the limitation of PSNR in capturing perceptual quality, particularly
when evaluating sharpness and clarity.
2

<!-- page 13 -->
DecNeRF
DetRF
Figure 10. This figure presents comparison on the novel view synthesis performance of leading dynamic NeRF models and our ExpanDyNeRF training
on the animal data from our SynDM dataset.
3

<!-- page 14 -->
DecNeRF
DetRF
Figure 11. This figure presents comparison on the novel view synthesis performance of leading dynamic NeRF models and our ExpanDyNeRF training
on the human data from our SynDM dataset.
4

<!-- page 15 -->
DecNeRF
DeteRF
Figure 12. This figure presents comparison on the novel view synthesis performance of leading dynamic NeRF models and our ExpanDyNeRF training
on the truck data from the NVIDIA dataset.
5

<!-- page 16 -->
Decoupling
NeRF
DecNeRF
DetRF
Figure 13. This figure presents comparison on the novel view synthesis performance of leading dynamic NeRF models and our ExpanDyNeRF training
on the skating data from the NVIDIA dataset.
6
