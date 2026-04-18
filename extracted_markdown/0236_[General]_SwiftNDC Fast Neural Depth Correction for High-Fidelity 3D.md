<!-- page 1 -->
SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction
Kang Han1
Wei Xiang1*
Lu Yu1
Mathew Wyatt2
Gaowen Liu3
Ramana Rao Kompella3
1La Trobe University, Melbourne, VIC 3086, Australia
2Australian Institute of Marine Science, Perth, WA 6009, Australia
3Cisco Research, San Jose, CA, USA
{k.han, w.xiang, l.yu}@latrobe.edu.au, m.wyatt@aims.gov.au, {gaoliu, rkompell}@cisco.com
Abstract
Depth-guided 3D reconstruction has gained popularity as
a fast alternative to optimization-heavy approaches, yet ex-
isting methods still suffer from scale drift, multi-view in-
consistencies, and the need for substantial refinement to
achieve high-fidelity geometry.
Here, we propose Swift-
NDC, a fast and general framework built around a Neu-
ral Depth Correction field that produces cross-view con-
sistent depth maps. From these refined depths, we gener-
ate a dense point cloud through back-projection and robust
reprojection-error filtering, obtaining a clean and uniformly
distributed geometric initialization for downstream recon-
struction. This reliable dense geometry substantially accel-
erates 3D Gaussian Splatting (3DGS) for mesh reconstruc-
tion, enabling high-quality surfaces with significantly fewer
optimization iterations.
For novel-view synthesis, Swift-
NDC can also improve 3DGS rendering quality, highlight-
ing the benefits of strong geometric initialization. We con-
duct a comprehensive study across five datasets, including
two for mesh reconstruction, as well as three for novel-view
synthesis. SwiftNDC consistently reduces running time for
accurate mesh reconstruction and boosts rendering fidelity
for view synthesis, demonstrating the effectiveness of com-
bining neural depth refinement with robust geometric ini-
tialization for high-fidelity and efficient 3D reconstruction.
1. Introduction
High-quality 3D reconstruction from multi-view images un-
derpins a wide range of applications, including real-time
rendering, robotics, simulation, content creation, and dig-
ital preservation [10]. While recent radiance field methods
such as Neural Radiance Fields (NeRF) [24, 27, 30] and
3DGS [6, 14, 19] have demonstrated impressive geomet-
ric and photometric fidelity, they typically require extensive
*Corresponding author.
optimization iterations per scene to converge to accurate ge-
ometry. This high computational cost poses a significant
barrier for practical 3D reconstruction pipelines that must
operate at scale.
In contrast, feed-forward depth estimation networks [4,
5, 7, 9, 35] predict depth maps in seconds, offering an ap-
pealing route to fast scene reconstruction. However, even
state-of-the-art depth predictors exhibit residual scale drift,
local bias, and cross-view inconsistencies [18]. These small
errors accumulate during back-projection and fusion, lead-
ing to degraded surface accuracy and unreliable geometry
for downstream tasks such as mesh extraction or radiance
field optimization. Despite extensive progress in depth pre-
diction and refinement [7, 17, 23, 35, 38], obtaining dense
and multi-view consistent geometry at low computational
cost remains an unsolved challenge.
We bridge this gap with SwiftNDC, a unified frame-
work that combines neural depth refinement with robust
geometric filtering to produce high-quality dense geometry
suitable for both mesh reconstruction and radiance-based
representation learning. At the core of our approach is a
Neural Depth Correction field that refines per-pixel depths
from feed-forward networks using a lightweight and scene-
specific model, producing depth maps that are geometri-
cally aligned and consistent across views. These refined
depths are back-projected into a dense point cloud, which
we further clean through a multi-view reprojection-error fil-
tering process. The result is a reliable, uniformly distributed
dense geometry that can serve as a strong initialization for
downstream reconstruction.
This dense and accurate geometric initialization has two
key benefits. First, for surface mesh reconstruction, ini-
tializing 3DGS with our filtered dense geometry drasti-
cally reduces the required optimization iterations, achiev-
ing high-fidelity meshes in as few as 1k iterations on DTU
object-level surface reconstruction. Second, for novel view
synthesis, our initialization improves 3DGS rendering fi-
delity especially in weakly observed regions, highlight-
ing the importance of accurate and well-distributed geom-
1
arXiv:2602.22565v1  [cs.CV]  26 Feb 2026

<!-- page 2 -->
SfM sparse point cloud
GOF mesh / 27 mins / CD 0.73 mm 
Ours dense point cloud
Ours / 1 min / CD 0.61 mm 
SfM sparse point cloud
3DGS rendering / PSNR 28.05 dB
Ours dense point cloud
3DGS rendering / PSNR 29.65 dB
Figure 1.
Comparison of mesh reconstruction and novel view synthesis.
Standard 3DGS based methods (Gaussian Opacity Field
(GOF) [42]) initialized from a sparse SfM point cloud requires extensive optimization to achieve high-fidelity results. SwiftNDC pro-
duces reliable dense geometry that accelerates mesh reconstruction (quality measured in Chamfer Distance (CD)↓) and improves rendering
quality for novel view synthesis.
etry for radiance-field learning (as shown in examples in
Fig.1). These complementary advantages allow SwiftNDC
to bridge the gap between fast depth-based pipelines and
optimization-heavy radiance-field methods.
We conduct a comprehensive evaluation across five
datasets, covering both surface reconstruction and novel
view synthesis. SwiftNDC consistently reduces reconstruc-
tion time for mesh reconstruction while improving render-
ing quality for novel view synthesis. In summary, our con-
tributions are threefold:
• We introduce SwiftNDC, a framework built upon a Neu-
ral Depth Correction field that yields geometrically accu-
rate and cross-view consistent depth maps.
• We
propose
a
reliable
dense
geometry
initializa-
tion method, combining depth back-projection with
reprojection-error filtering to produce high-quality geom-
etry for downstream reconstruction.
• We present a comprehensive five-dataset evaluation,
showing that SwiftNDC reduces reconstruction time for
mesh reconstruction and improves rendering quality for
novel view synthesis.
2. Related work
2.1. Per-scene optimization pipelines
Neural implicit surface methods represent geometry as con-
tinuous fields, typically signed distance functions (SDFs)
or density-radiance volumes encoded by multilayer percep-
trons (MLPs) and optimized on a per-scene basis. Building
on the idea of learning a SDF by an MLP in DeepSDF [31],
NeRF [27] introduced differentiable volumetric rendering
of a density-radiance field, showing that photo-consistent
geometry could be recovered without explicit 3D supervi-
sion. Subsequent methods such as NeuS [36], VolSDF [41],
UniSurf [30] and Neuralangelo [24] refined this radiance
geometry marriage by explicitly enforcing surface consis-
tency while retaining view-dependent color. Although these
models achieve sub-millimeter accuracy, they typically re-
quire hours of per-scene optimization because of intensive
ray sampling and repeated network evaluations.
3DGS replaces volumetric representations with thou-
sands of anisotropic Gaussians optimized via differentiable
rasterization, dramatically reducing training time and accel-
erating rendering [19]. Because the original 3DGS formu-
lation optimizes only for photometric consistency, it lacks
an explicit surface constraint, so extracting meshes from
the raw Gaussian cloud yields noisy, incomplete geome-
try. To improve surface modeling, a line of variants has
emerged, including SuGaR [14], 2DGS [16], GOF [42],
and NSDF-Splat [43]. Although faster than neural implicit
pipelines, these approaches still require extensive scene spe-
cific optimization, partially due to very sparse SfM ini-
tilization and iterative densification. EDGS [21] attempts
to mitigate this cost through a dense Gaussian initializa-
tion obtained from triangulated 2D correspondences, but
its design is tailored exclusively for novel view synthesis.
In contrast, our approach produces accurate, multi-view-
consistent dense geometry from corrected depth maps, en-
abling both high-fidelity surface mesh reconstruction and
improved view synthesis.
2

<!-- page 3 -->
Mono/MV depths
Per-pixel 
corrected depths
Mesh reconstruction
Neural depth 
correction field
Dense
Geometry Init
Sampled dense 
point cloud
3DGS reconstruction
SfM sparse
point cloud
Figure 2. SwiftNDC pipeline. The method starts with both monocular and multi-view depth maps. A neural depth correction field,
supervised by sparse SfM points, refines these depths at pixel level. The corrected depths are then transformed into a dense and reliable
sampled point cloud, which serves as a strong geometric initialization for downstream mesh reconstruction and 3DGS-based view synthesis.
2.2. Feed-forward depth estimation
Classical PatchMatch-style stereo algorithms, including
COLMAP’s MVS module [32], propagate depth hypotheses
across images but need several minutes per scene and strug-
gle on texture-poor regions [33]. Learning-based networks
such as MVSNet [39], R-MVSNet [40], Cas-MVSNet [13],
TransMVSNet [9], and the more recent MVSFormers [4, 5]
replace hand-crafted costs with deep cost volumes and pre-
dict depth in seconds.
However, small pixel-wise scale
or bias errors still persist in MVS depth maps.
After
TSDF fusion [8] these residuals manifest as wavy sur-
faces, holes, and over-smoothed details [37]. Because con-
ventional feed-forward MVS optimizes each reference im-
age independently, it is forced to subsample to about 3–7
best-overlapping views to keep cost volume memory af-
fordable, sacrificing global geometric consistency and ul-
timately mesh fidelity.
Visual Geometry Grounded Transformer (VGGT) [35]
changes this landscape by ingesting all available views (up
to hundreds) in a single forward pass and jointly predict-
ing depths, cameras and point maps, thus sidestepping the
fixed K view bottleneck of current MVS networks. While
VGGT’s global multi-view reasoning yields depth maps
that are markedly more 3D consistent, we observe persis-
tent local mis-alignments and small metric biases that trans-
late into poor, fragmented meshes when fused directly (see
Fig. 3). These residual errors motivate the pixel-level depth-
correction strategy proposed in this work to refine VGGT
outputs into high-fidelity dense geometry.
2.3. Depth correction and test-time refinement
Early depth alignment methods such as MegaDepth [23]
and AdaBins [3] applied a single scale factor or bias per
image using sparse SfM depths, correcting gross metric er-
ror but not spatial variations. Recent work has moved to-
ward pixel-level refinement. SfM-TTR [17] fine-tunes an
entire depth network at test time from COLMAP points,
while Fink et al. [11] learn a residual MLP for the first
image but then perform an additional differentiable render-
ing stage that adds extensive computation. Sparse-to-dense
completion systems like RadarCam-Depth [22] and depth
from focus-based HybridDepth [12] predict dense correc-
tions with heavy CNN or Transformer backbones and bi-
ased on specific training datasets.
All of these methods
either operate on single images, require large network up-
dates, or involve complex multi-stage optimization.
SwiftNDC departs from prior work by learning a
lightweight neural depth correction field that refines per-
pixel depths using monocular and multi-view cues, super-
vised only by sparse SfM points.
Crucially, the refined
depths are converted into a dense point cloud and filtered
via multi-view reprojection error, yielding reliable geome-
try for downstream reconstruction. This initialization en-
ables high-fidelity meshes with fewer 3DGS iterations and
improves novel view synthesis quality.
3. Method
SwiftNDC efficiently converts an image set into multi-
view-consistent depth maps and a reliable dense point
cloud, completing this process in under one minute for
scenes with fewer than 60 views. Fig. 2 depicts the pro-
posed pipeline. Starting from COLMAP [32] camera poses
and sparse points, we obtain two dense depth estimates per
view: a multi-view depth from VGGT [35], which provides
strong global consistency, and a monocular depth from
Video Depth Anything (VDA) [7], which preserves fine de-
tail but is only up to scale. Both maps are coarsely aligned
to the sparse points through a per-view affine fit. At the core
of our approach is a pixel-level neural depth correction field
trained with sparse SfM supervision, which removes the re-
maining local depth biases in under one second per image.
The corrected depth maps are then back-projected to form
a dense point cloud and refined via reprojection-error fil-
tering to provide reliable geometry for downstream mesh
reconstruction and view synthesis.
3.1. Initial depth estimation and correction
Let I = {Ii}N
i=1 be the input images, with intrinsics Ki and
extrinsics (Ri, ti) from COLMAP, and let P = {xk}M
k=1
denote the sparse COLMAP point cloud. Our objective in
this subsection is to derive, for each view i, an initially
corrected depth map eDi that is aligned to P and contains
3

<!-- page 4 -->
Depth map
Error map
Error map
Fused point cloud
Reconstructed mesh
44 mm
8 mm
36 mm
CD 3.21 mm 
3.19 mm 
0.55 mm 
L1 36 mm
40 mm
6 mm
Error bar
Multi-view 
depth
Monocular 
depth
Our depth
Figure 3. Depth accuracy drives surface quality. The per-view–corrected monocular and multi-view depth maps look plausible, yet L1
distance comparison against GOF’s [42] geo-consistent depth renderings reveals ≈40 mm residual error. Fusing these depths into point
clouds exposes severe cross-view mis-alignment that propagates into rough, hole-ridden meshes with high Chamfer distance error. Our
SwiftNDC cuts the depth error to less than 10 mm, producing a point cloud that aligns tightly across views and a final surface that is
visually smooth and metrically accurate.
as much geometric detail as possible. This map seeds the
pixel-level refinement of Section 3.2.
We first feed the entire image set to VGGT, which jointly
reasons over all views and outputs a globally consistent
depth map
Dvggt
i
: Ωi →R+,
(1)
where Ωi ⊂R2 is the pixel grid. Processing every image
simultaneously eliminates most of the scale drift found in
earlier K-view MVS nets, but VGGT tends to oversmooth
thin structures and exhibits local bias.
To recover lost
high-frequency geometry, we independently run the VDA
monocular model on each frame, producing Dmono
i
. These
depths capture geometry details well but are only relative;
their per-view scales may differ. Importantly, VGGT and
VDA run in seconds for N < 60, so combining them incurs
negligible overhead.
Next, each 3D point xk visible in view i projects to
pik = π(Ki, Ri, ti; xk),
dcol
ik =
 Rixk + ti

z,
(2)
providing an accurate depth anchor. At the same pixel we
sample dvggt
ik
= Dvggt
i
(pik) and dmono
ik
= Dmono
i
(pik),
yielding the triplet set
Ti =

(dvggt
ik
, dmono
ik
, dcol
ik )
	Ki
k=1,
(3)
with Ki ≈103 −104. We align both depth maps to the
sparse point cloud by solving two independent ordinary-
least-squares problems:
min
s,b
Ki
X
k=1
 s dt
ik + b −dcol
ik
2,
t ∈{vggt, mono},
(4)
which admit analytic solutions in O(Ki) time. The affine-
corrected depths are
eDvggt
i
(p) = sv
i Dvggt
i
(p) + bv
i ,
(5)
eDmono
i
(p) = sm
i Dmono
i
(p) + bm
i .
(6)
These maps weave together VGGT’s global consistency
and monocular fine detail but still contain residual local er-
ror (see Fig. 3). We next introduce a sparse-guided, pixel-
level correction field that removes those errors and unlocks
high-fidelity geometry.
3.2. Neural depth correction
The per-view affine alignment rectifies gross scale drift, yet
centimetre-level, spatially varying errors remain inside each
depth map. When these residuals are integrated along view-
ing rays in a TSDF volume they amplify into surface ripples
and pinholes that defeat the goal of a reconstruction (see
Fig. 3). We therefore introduce a neural depth correction
4

<!-- page 5 -->
field that predicts a pixel-wise affine residual from nothing
more than the sparse, accurate COLMAP anchors.
For every sparse point xk observed in view i we collect
two depth samples and three view descriptors, forming
zik =
 dv
ik, dm
ik, uik, vik, ιi

∈R5,
(7)
where dv
ik = eDvggt
i
(pik) and dm
ik = eDmono
i
(pik) are the
affine-corrected VGGT and monocular depths, (uik, vik) ∈
[−1, 1]2 are normalized image-plane coordinates, and ιi =
i/(N −1) normalizes the view index. An MLP fθ with six
hidden layers of width 64 maps this five-vector with posi-
tional encoding φ to four coefficients,
 αv
ik, βv
ik, αm
ik, βm
ik

= fθ(φ(zik)),
(8)
and delivers the corrected depths
ˆdv
ik = eαv
ik dv
ik + βv
ik,
ˆdm
ik = eαm
ik dm
ik + βm
ik.
(9)
Exponentiating the learned scales guarantees positivity
without extra constraints. We train the network for view
i by minimizing the sparse L1 reprojection loss
L(θ) =
X
k∈Ti

| ˆdv
ik −dcol
ik | + | ˆdm
ik −dcol
ik |

,
(10)
where dcol
ik is the COLMAP depth.
Directly minimizing the above loss per view requires
about 5000 gradient steps per image, or several minutes
for a 60-view scene. We therefore amortize training across
views by introducing a global-then-local schedule. We first
learn a scene-level field by solving
min
θ
Lglobal(θ) =
N
X
i=1
X
k∈Ti

| ˆdv
ik −dcol
ik | + | ˆdm
ik −dcol
ik |

,
using AdamW [25] with an initial learning rate of 1 ×
10−3 and the cosine annealing warm restart scheduler [26]
(restart period T0 = 1000, total steps T = 5000). This
global optimization captures systematic biases shared by all
views. The converged weights θ⋆serve as a warm start for
a per-view refinement using Eq. 10 again with AdamW, a
restart period T0 = 250, and only 500 steps, typically fin-
ishing in less than one second. As shown in Fig. 3, this
per-view correction significantly reduces depth error and, in
turn, produces accurate mesh that cannot be achieved using
feed-forward depths alone.
3.3. Reliable dense geometry initialization
After obtaining corrected depths ˆDi, we construct a dense
point cloud by back-projecting each pixel in view i. For a
pixel piu, the corresponding 3D point is
xiu = Π(Ki, Ri, ti; piu),
(11)
SfM sparse point cloud
Ours dense point cloud
without reprojection filtering
Ours dense point cloud
with reprojection filtering
Figure 4. Effect of removing unreliable points using reprojection-
error filtering.
where Π is the back-projection operation. Although the cor-
rected depth maps are aligned, individual pixels may still
contain residual spatial errors. We therefore evaluate each
3D point’s reliability using a view-to-view reprojection cy-
cle. We reproject the 3D point into a neighboring view j:
pju = π(Kj, Rj, tj; xiu),
(12)
At this 2D location pju, we sample the corrected depth
ˆDj(pju), back-project it in view j to obtain xju, and then
reproject this 3D point back into view i to get the repro-
jected pixel location ˆpiu. We define the reprojection error
in pixel space as
ei→j(piu) = ∥ˆpiu −piu∥2 .
(13)
A point is considered reliable if its average reprojection
error across a small set of overlapping views is below a
threshold. We set this threshold as 1 pixel to ensure reli-
ability. The filtered set of points forms a clean and geomet-
rically consistent dense point cloud, as illustrated in Fig. 4.
We further apply a simple downsampling method to obtain a
uniformly distributed subset, which serves as the dense ge-
ometry initialization for 3DGS. This initialization yields a
high-quality geometric scaffold that can significantly accel-
erate 3DGS convergence for mesh reconstruction and im-
prove rendering quality for novel view synthesis.
4. Experiments
4.1. Experiment settings
Our neural depth correction field is implemented in Tiny
CUDA Neural Networks [28] for high arithmetic through-
put. Corrected depths can be directly fused into a TSDF
to extract surface mesh by Marching Cubes [44] or used to
produce reliable dense geometry initialization. For mesh
reconstruction, we adopt the DTU surface-reconstruction
benchmark [18] and the Tanks and Temples (TnT) dataset
5

<!-- page 6 -->
Table 1. Surface reconstruction accuracy on the DTU dataset [18], measured by Chamfer distance ↓. Ours-Depth directly reconstructs
meshes from the corrected depth maps, while Ours-3DGS further applies a lightweight 3DGS refinement (1k iterations, initialized from
our dense point samples) to enhance surface fidelity. Results are highlighted as 1st , 2nd , and 3rd .
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
Time
NeRF [27]
1.90
1.60
1.85
0.58
2.28
1.27
1.47
1.67
2.05
1.07
0.88
2.53
1.06
1.15
0.96
1.49
>12h
VolSDF [41]
1.14
1.26
0.81
0.49
1.25
0.70
0.72
1.29
1.18
0.70
0.66
1.08
0.42
0.61
0.55
0.86
>12h
NeuS [36]
1.00
1.37
0.93
0.43
1.10
0.65
0.57
1.48
1.09
0.83
0.52
1.20
0.35
0.49
0.54
0.84
>12h
3DGS [19]
2.14
1.53
2.08
1.68
3.49
2.21
1.43
2.07
2.22
1.75
1.79
2.55
1.53
1.52
1.50
1.96
20m
SuGaR [14]
1.47
1.33
1.13
0.61
2.25
1.71
1.15
1.63
1.62
1.07
0.79
2.45
0.98
0.88
0.79
1.33
107m
2DGS [16]
0.48
0.91
0.39
0.39
1.01
0.83
0.81
1.36
1.27
0.76
0.70
1.40
0.40
0.76
0.52
0.80
19m
GOF [42]
0.50
0.82
0.37
0.37
1.12
0.74
0.73
1.18
1.29
0.68
0.77
0.90
0.42
0.66
0.49
0.74
33m
PGSR [6]
0.34
0.58
0.29
0.29
0.78
0.58
0.54
1.01
0.73
0.51
0.49
0.69
0.31
0.37
0.38
0.53
30m
Ours depth
0.44
0.60
0.37
0.41
0.91
1.01
0.61
1.80
1.00
0.69
0.64
1.28
0.44
0.56
0.55
0.75
1m
Ours 3DGS
0.39
0.68
0.46
0.39
0.80
0.73
0.59
1.06
0.84
0.59
0.58
0.67
0.33
0.42
0.40
0.59
3m
Table 2. Quantitative results of F1 score ↑for surface reconstruc-
tion on Tanks and Temples dataset [20]. Our method attains com-
parable reconstruction quality to existing approaches while requir-
ing the shortest reconstruction time.
Neurlangelo
2DGS [16]
GOF [42]
PGSR [6]
Ours
Barn
0.70
0.36
0.51
0.66
0.62
Caterpillar
0.36
0.23
0.41
0.41
0.42
Courthouse
0.28
0.13
0.28
0.21
0.25
Ignatius
0.89
0.44
0.68
0.80
0.80
Meetingroom
0.32
0.16
0.28
0.29
0.31
Truck
0.48
0.26
0.58
0.60
0.60
Mean
0.50
0.30
0.46
0.50
0.50
Time
>128h
35m
2h
45m
26m
[20]. Thanks to our reliable dense geometry initialization,
our method requires only 1k 3DGS optimization steps to
achieve state-of-the-art mesh quality on the DTU bench-
mark. For large-scale scenes in TnT, we use 15k steps, ap-
proximately 50% of the iterations typically required for full
3DGS optimization. We compare against two families of
state-of-the-art baselines: neural implicit methods (NeRF
[27], VolSDF [41], and NeuS [36]) and 3DGS methods
(original 3DGS [19], SuGaR [14], 2DGS [16], GOF [42]
and PGSR [6]). All training timing results are measured on
a single NVIDIA L40S with 48 GB VRAM. Training times
are averaged across all the scenes.
For view synthesis, we evaluate on three widely used
multi-view benchmarks:
MipNeRF 360 [2], Tanks and
Temples [20], and Deep Blending [15]. We adopt the stan-
dard train–test splits and report PSNR, SSIM, and LPIPS
to assess rendering quality. Baseline comparison includes
NeRF-based methods [1, 2, 29] and recent 3DGS variants
[19, 34] using their recommended hyperparameters.
4.2. Mesh reconstruction
Table 1 reports CD surface error on the DTU scans together
with averaged per-scene optimization time.
The neural-
implicit baselines deliver competitive geometry (mean er-
rors 0.84–1.49 mm) but require more than twelve hours of
per-scene optimization, an order of magnitude longer than
most explicit approaches. Gaussian splatting slashes run-
time to the minutes range while typically improving accu-
racy. For instance, GOF achieves the mean error 0.74 mm in
33 minutes. In comparison, our method without any 3DGS
optimization already achieves a mean CD of 0.75 mm in
only 60 seconds, making it comparable to the best ex-
plicit methods while being 20–30 times faster. Viewed on
the speed-quality front, this marks the first point that lies
inside the sub-minute regime without conceding accuracy
to minutes-long pipelines. With an additional lightweight
3DGS refinement, our method further reduces error, achiev-
ing results comparable to PGSR (0.59 vs. 0.53 mm) while
still running an order of magnitude faster.
We present surface reconstruction results on the TnT
benchmark, evaluated using the F1 score in Table 2. Across
all six scenes, our method consistently matches or exceeds
the performance of recent 3DGS approaches while achiev-
ing a mean F1 of 0.50, on par with the two strongest
Neurlangelo and PGSR. Crucially, this accuracy is obtained
with the shortest reconstruction time, requiring only 26
minutes, substantially faster than PGSR (45 minutes), GOF
(2 hours), and orders of magnitude faster than Neurlangelo
(over 128 hours). Qualitative results in Fig. 5 further il-
lustrate that our method produces dense point clouds that
are both geometrically accurate and well distributed, clearly
improving over the sparse SfM input. This dense and reli-
able point geometry provides a strong initialization for sub-
sequent 3DGS mesh refinement, enabling our final meshes
to closely match the quality of PGSR despite much shorter
optimization schedules.
4.3. View synthesis
We summarize the novel view synthesis performance in Ta-
ble 3.
Across all three datasets and all evaluation met-
rics, our method consistently outperforms both NeRF-based
6

<!-- page 7 -->
SfM sparse point cloud
Ours dense point cloud
SfM - 3DGS rendering / mesh
Ours - 3DGS rendering / mesh
Ground truth view
Figure 5. Qualitative results for view synthesis and mesh reconstruction. Our dense initialization yields more complete geometry, especially
in sparsely observed regions, enabling higher-quality rendering or reduced mesh reconstruction time with similar quality.
Table 3. View synthesis quality comparison. Our method delivers consistent improvements across three datasets and all quality metrics.
MipNeRF 360 [2]
Tank & Temples [20]
Deep Blending[15]
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Instant-NGP [29]
26.75
0.75
0.30
21.92
0.75
0.31
-
MipNeRF [1]
27.60
0.81
0.25
-
21.54
0.78
0.37
MipNeRF 360 [2]
29.23
0.84
0.21
22.22
0.76
0.26
-
3DGS (SfM sparse) [19]
28.69
0.87
0.22
23.14
0.84
0.21
-
Splatfacto (SfM sparse) [34]
29.05
0.87
0.21
23.51
0.85
0.19
29.45
0.90
0.31
Ours (Sampled dense, splatfacto)
29.33
0.88
0.19
23.74
0.86
0.17
29.74
0.91
0.30
baselines and recent 3DGS pipelines built on sparse SfM
points. When integrated into Splatfacto [34], our sampled
dense initialization yields the best scores on every bench-
mark, improving PSNR and substantially reducing percep-
tual error (LPIPS). The qualitative results in Fig. 5 reveal
why: compared to the sparse SfM input, our dense point
clouds provide significantly more complete geometric cov-
erage, including regions where the original SfM reconstruc-
tion contains few or no points. This reliable and uniformly
distributed initialization enables the subsequent 3DGS op-
timization to better model scene structures that are weakly
observed in the training images and would otherwise remain
under-densified.
As a result, our method boosts render-
ing quality especially in challenging areas with occlusions,
grazing angles, or limited viewpoint diversity. These results
also suggest strong compatibility with existing 3DGS vari-
ants. Our dense sampling acts as a drop-in enhancement
that can be paired with different splatting pipelines to sys-
tematically improve view synthesis fidelity.
4.4. Ablation study
Table 4 dissects how each design choice affects accuracy
and runtime. The full neural depth correction attains the
mean error 0.75 mm in one minute, establishing the ref-
erence point.
Per-view correction.
Per-view corrected
monocular variant by the same SfM points reaches 3.15 mm
mean CD and the multi-view variant 2.50 mm, significantly
worse than the full model despite taking only a few sec-
onds. These results confirm that per-view correction cannot
resolve cross-view mis-alignment. Pixel-level correction is
7

<!-- page 8 -->
Table 4. Ablation study on the DTU dataset.
Model
Mean CD ↓
Time (m)
Full neural depth correction
0.75
1
Per-view cor. mo. depth
3.15
0.18
Per-view cor. mv. depth
2.50
0.13
Monocular depth head
0.91
1
Monocular depth only
1.00
0.94
Multi-view depth only
0.98
0.89
W/o two-stage training
0.81
12
3DGS opt w reliable geo.
0.59
3
3DGS opt w/o reliable geo.
0.82
3
therefore essential for high-fidelity reconstruction. Dual-
depth synergy. Correcting monocular depth (Monocular
depth head) rather than multi-view depth increases error to
0.91 mm, while using either modality in isolation (Monocu-
lar only, Multi-view only) pushes the error to approximately
1.0 mm. Combining the two cues therefore yields a 25 %
accuracy gain without significantly hurting speed.
Two-stage training schedule. If the neural depth cor-
rection field is trained separately for each image (W/o two-
stage training), the error remains competitive (0.81 mm) yet
optimization balloons to 12 minutes, twelve times slower
than the proposed amortized scheme. The global pass thus
captures most of the bias at once, allowing each fine-tune
to converge in less than one second. Runtime observa-
tions. Using a single depth branch trims 5–10 % off the
one-minute budget (0.89–0.94 min), but the accuracy drop
outweighs the marginal savings. Conversely, the two-stage
schedule slashes computation by an order of magnitude
with no loss in fidelity, demonstrating that where we spend
optimization budget matters more than how much we spend.
Reliable dense geometry initialization.
While a
lightweight 3DGS refinement can further reduce the mean
surface error, it is effective only when built upon our reli-
able dense geometry initialization. Without this module, the
error becomes even larger than that of directly fusing depth
maps, underscoring the crucial role of our outlier filtering
strategy in enabling fast and high-quality mesh reconstruc-
tion. To summarize, every modeling cue, including global
VGGT depth, local monocular detail, pixel-level correction
and reliable dense geometry initialization, contributes mea-
surably, and their benefits are maximally realized within our
two-stage optimization framework.
4.5. Reconstruction time breakdown
Table 5 shows that a complete SwiftNDC reconstruction fin-
ishes in just under a minute. More than half of this time
(about 31 s) is spent on the per-pixel refinement network,
which adjusts depth values with pixel-level precision. The
Table 5. Reconstruction time breakdown (seconds) on the DTU
scenes with 49 images.
Stage
Time (s)
VGGT depth inference
3.6
Monocular depth inference
6.6
Initial per-view correction
7.0
Global scale refinement
7.0
Per-pixel correction
30.9
3DGS optimization-1k iterations
110
TSDF fusion + Marching Cubes
1.6
Total
167
remaining steps are comparatively lightweight: depth infer-
ence from VGGT and the monocular model together take
roughly 10 seconds, while the two scale-correction passes
consume another 14 seconds. TSDF integration followed
by Marching Cubes completes in less than two seconds, in-
dicating that surface extraction is not a time bottleneck once
accurate depths are available.
4.6. Limitations
While SwiftNDC produces reliable geometry and improves
both the efficiency and the quality of 3DGS, it still de-
pends on accurate SfM poses and on the quality of the initial
monocular or MVS depth maps. Although SwiftNDC sig-
nificantly reduces the number of 3DGS iterations for mesh
reconstruction, it introduces additional preprocessing steps,
including depth correction and reprojection-based filtering,
that may become noticeable for extremely large image col-
lections (e.g., thousands of views). Finally, SwiftNDC uses
the corrected depths only to initialize 3DGS, and the two
stages are not jointly optimized. An end-to-end formulation
that couples depth correction with radiance-field optimiza-
tion could potentially yield even higher fidelity and repre-
sents an interesting direction for future work.
5. Conclusion
This work demonstrates that reliable dense geometry ini-
tialization greatly enhances fast mesh reconstruction and
high-quality view synthesis. By transforming sparse SfM
points into a dense, accurate, and well-distributed geomet-
ric prior, our method provides a strong foundation for down-
stream 3DGS without the need for long optimization sched-
ules. The proposed approach integrates global multi-view
depth, fine-grained monocular detail, and rigorous outlier
filtering into a unified dense initialization that is both robust
and highly efficient to compute.
This initialization alone yields competitive surfaces on
the DTU dataset. With further 3DGS refinement, it matches
the mesh quality of state-of-the-art methods at a fraction of
8

<!-- page 9 -->
their runtime on large-scale outdoor scenes. For view syn-
thesis, the dense geometry fills regions poorly covered by
SfM, enabling more consistent and robust rendering across
datasets. Overall, our approach demonstrates that fast, scal-
able 3D reconstruction is best achieved by coupling efficient
dense geometry with targeted Gaussian refinement, offering
a simple and compatible drop-in improvement for existing
3DGS methods.
Acknowledgments
This research was supported by the Australian Research
Council Discovery Project (Grant Number DP240103334).
References
[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-NeRF: A multiscale representation for anti-aliasing
neural radiance fields.
In Proceedings of the IEEE/CVF
international conference on computer vision, pages 5855–
5864, 2021. 6, 7
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman.
Mip-NeRF 360:
Un-
bounded anti-aliased neural radiance fields. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 6, 7
[3] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka.
AdaBins: Depth estimation using adaptive bins. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 4009–4018, 2021. 3
[4] Chenjie Cao, Xinlin Ren, and Yanwei Fu.
MVSFormer:
Multi-view stereo by learning robust image features and
temperature-based depth. Transactions on Machine Learn-
ing Research, 2022. 1, 3
[5] Chenjie Cao, Xinlin Ren, and Yanwei Fu. MVSFormer++:
Revealing the devil in transformer’s details for multi-view
stereo. In The Twelfth International Conference on Learning
Representations, 2024. 1, 3
[6] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang.
PGSR: Planar-based gaussian splat-
ting for efficient and high-fidelity surface reconstruction.
IEEE Transactions on Visualization and Computer Graph-
ics, 2024. 1, 6
[7] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zi-
long Huang, Jiashi Feng, and Bingyi Kang. Video depth any-
thing: Consistent depth estimation for super-long videos. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 22831–22840, 2025. 1, 3
[8] Brian Curless and Marc Levoy. A volumetric method for
building complex models from range images. In Proceedings
of the 23rd Annual Conference on Computer Graphics and
Interactive Techniques, pages 303–312, 1996. 3
[9] Yikang Ding, Wentao Yuan, Qingtian Zhu, Haotian Zhang,
Xiangyue Liu, Yuanjiang Wang, and Xiao Liu. TransMVS-
Net: Global context-aware multi-view stereo network with
transformers. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8585–
8594, 2022. 1, 3
[10] Anis Farshian, Markus G¨otz, Gabriele Cavallaro, Charlotte
Debus, Matthias Nießner, J´on Atli Benediktsson, and Achim
Streit.
Deep-learning-based 3D surface reconstruction—a
survey. Proceedings of the IEEE, 111(11):1464–1501, 2023.
1
[11] Laura Fink, Linus Franke, Joachim Keinert, and Marc Stam-
minger. Refinement of monocular depth maps via multi-view
differentiable rendering. arXiv preprint arXiv:2410.03861,
2024. 3
[12] Ashkan Ganj, Hang Su, and Tian Guo. HybridDepth: Ro-
bust metric depth fusion by leveraging depth from focus and
single-image priors. In 2025 IEEE/CVF Winter Conference
on Applications of Computer Vision (WACV), pages 973–
982. IEEE, 2025. 3
[13] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong
Tan, and Ping Tan. Cascade cost volume for high-resolution
multi-view stereo and stereo matching. In Proceedings of
the IEEE/CVF Conference on Computer Cision and Pattern
Recognition, pages 2495–2504, 2020. 3
[14] Antoine Gu´edon and Vincent Lepetit.
SuGaR: Surface-
aligned gaussian splatting for efficient 3D mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 1, 2, 6
[15] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. ACM Transactions
on Graphics (ToG), 37(6):1–15, 2018. 6, 7
[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2D Gaussian splatting for geometrically ac-
curate radiance fields. In SIGGRAPH’24: ACM SIGGRAPH
2024 Conference Papers. ACM, 2024. 2, 6
[17] Sergio Izquierdo and Javier Civera. SfM-TTR: Using struc-
ture from motion for test-time refinement of single-view
depth networks. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
21466–21476, 2023. 1, 3
[18] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola,
and Henrik Aanæs. Large scale multi-view stereopsis evalu-
ation. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 406–413, 2014.
1, 5, 6
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis. 3D Gaussian splatting for real-time
radiance field rendering.
ACM Transactions on Graphics
(TOG), 42(4):1–14, 2023. 1, 2, 6, 7
[20] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics, 36(4):1–13,
2017. 6, 7
[21] Dmytro Kotovenko, Olga Grebenkova, and Bj¨orn Ommer.
EDGS: Eliminating densification for efficient convergence of
3dgs. arXiv preprint arXiv:2504.13204, 2025. 2
[22] Han Li, Yukai Ma, Yaqing Gu, Kewei Hu, Yong Liu, and
Xingxing Zuo. RadarCam-depth: Radar-camera fusion for
9

<!-- page 10 -->
depth estimation with learned metric scale. In 2024 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 10665–10672. IEEE, 2024. 3
[23] Zhengqi Li and Noah Snavely. MegaDepth: Learning single-
view depth prediction from internet photos. In Proceedings
of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 2041–2050, 2018. 1, 3
[24] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Tay-
lor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin.
Neuralangelo: High-fidelity neural surface reconstruction. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 8456–8465, 2023. 1,
2
[25] Ilya Loshchilov and Frank Hutter.
Decoupled weight de-
cay regularization. In International Conference on Learning
Representations, 2017. 5
[26] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient
descent with warm restarts. In International Conference on
Learning Representations, 2022. 5
[27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
representing scenes as neural radiance fields for view syn-
thesis. In European Conference on Computer Vision, pages
405–421. Springer, 2020. 1, 2, 6
[28] Thomas M¨uller. tiny-cuda-nn, 2021. 5
[29] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 6, 7
[30] Michael Oechsle, Songyou Peng, and Andreas Geiger.
Unisurf:
Unifying neural implicit surfaces and radiance
fields for multi-view reconstruction.
In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 5589–5599, 2021. 1, 2
[31] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove.
DeepSDF: Learning
continuous signed distance functions for shape representa-
tion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 165–174, 2019.
2
[32] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
4104–4113, 2016. 3
[33] Johannes L Sch¨onberger,
Enliang Zheng,
Jan-Michael
Frahm, and Marc Pollefeys.
Pixelwise view selection for
unstructured multi-view stereo. In European Conference on
Computer Vision, pages 501–518. Springer, 2016. 3
[34] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li,
Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake
Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A
modular framework for neural radiance field development.
In ACM SIGGRAPH 2023 conference proceedings, pages 1–
12, 2023. 6, 7
[35] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. VGGT:
Visual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 1, 3
[36] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. NeuS: learning neural implicit
surfaces by volume rendering for multi-view reconstruction.
In Proceedings of the 35th International Conference on Neu-
ral Information Processing Systems, pages 27171–27183,
2021. 2, 6
[37] Yaniv Wolf, Amit Bracha, and Ron Kimmel. GS2Mesh: Sur-
face reconstruction from gaussian splatting via novel stereo
views. In European Conference on Computer Vision, pages
207–224. Springer, 2024. 3
[38] Zhiqiang Yan, Kun Wang, Xiang Li, Zhenyu Zhang, Jun Li,
and Jian Yang. DesNet: Decomposed scale-consistent net-
work for unsupervised depth completion. In Proceedings of
the AAAI Conference on Artificial Intelligence, pages 3109–
3117, 2023. 1
[39] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long
Quan.
MVSNet: Depth inference for unstructured multi-
view stereo. In Proceedings of the European Conference on
Computer Vision, pages 767–783, 2018. 3
[40] Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang,
and Long Quan.
Recurrent MVSNet for high-resolution
multi-view stereo depth inference.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5525–5534, 2019. 3
[41] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Vol-
ume rendering of neural implicit surfaces. Advances in Neu-
ral Information Processing Systems, 34:4805–4815, 2021. 2,
6
[42] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (TOG),
43(6):1–13, 2024. 2, 4, 6
[43] Wenyuan Zhang, Yu-Shen Liu, and Zhizhong Han.
Neu-
ral signed distance function inference through splatting 3D
gaussians pulled on zero-level set. In The Thirty-eighth An-
nual Conference on Neural Information Processing Systems,
2024. 2
[44] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D:
A modern library for 3d data processing.
arXiv preprint
arXiv:1801.09847, 2018. 5
10
