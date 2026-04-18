<!-- page 1 -->
3D Gaussian Splatting with Fisheye Images: Field of View Analysis and
Depth-Based Initialization
Ulas Gunes1, Matias Turkulainen 2, Mikhail Silaev1, Juho Kannala 2,3, Esa Rahtu 1
1Department of Computing Sciences, Tampere University, Korkeakoulunkatu 7, Tampere, Finland
2Department of Computer Science, Aalto University, Otakaari, Espoo, Finland
2Department of Computer Science and Engineering, University of Oulu, Pentti Kaiteran katu 1, Oulu, Finland
{ulas.gunes, mikhail.silaev, esa.rahtu}@tuni.fi, {matias.turkulainen, juho.kannala}@aalto.fi
Keywords:
Fisheye, 3D Gaussian Splatting, Depth Initialization
Abstract:
We present the first evaluation of 3D Gaussian Splatting methods on real fisheye imagery with fields of view
above 180°. Our study evaluates Fisheye-GS (Liao et al., 2024) and 3DGUT (Wu et al., 2025) on indoor and
outdoor scenes captured with 200° fisheye cameras, with the aim of assessing the practicality of wide-angle
reconstruction under severe distortion. By comparing reconstructions at 200°, 160°, and 120° field-of-view, we
show that both methods achieve their best results at 160°, which balances scene coverage with image quality,
while distortion at 200° degrades performance. To address the common failure of Structure-from-Motion
(SfM) initialization at such wide angles, we introduce a depth-based alternative using UniK3D (Universal
Camera Monocular 3D Estimation) (Piccinelli et al., 2025). This represents the first application of UniK3D to
fisheye imagery beyond 200°, despite the model not being trained on such data. With the number of predicted
points controlled to match SfM for fairness, UniK3D produces geometrically accurate reconstructions that
rival or surpass SfM, even in challenging scenes with fog, glare, or open sky. These results demonstrate the
feasibility of fisheye-based 3D Gaussian Splatting and provides a benchmark for future research on wide-angle
reconstruction from sparse and distorted inputs.
1
INTRODUCTION
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023)
has become a foundational technique for high-quality
3D scene reconstruction and real-time image-based
rendering, with wide adoption in both academia and
industry. While many extensions improve rendering,
editing, and optimization, most operate under narrow
field-of-view (FoV) perspective cameras due to their
simple geometry and mature calibration pipelines.
Fisheye cameras, by contrast, provide ultra-wide
FoVs that capture larger portions of a scene with
fewer images. This makes them attractive for tasks
in autonomous driving, robotics, and VR/AR, where
reducing sensor count and capture time is crucial.
Fewer viewpoints can also lower the computational
and memory requirements of reconstruction. How-
ever, fisheye-based 3D reconstruction remains rela-
tively unexplored due to the nonlinear projection and
strong radial distortion characteristics of such lenses.
Recent approaches such as Fisheye-GS (Liao
et al., 2024) and 3DGUT (Wu et al., 2025) extend
3DGS to fisheye images by modifying projection and
rendering to support non-linear camera models. Al-
though these methods are promising, their real-world
behavior across different fields of view and diverse
scene types has not been comprehensively evaluated.
Initialization poses an additional challenge. Stan-
dard Structure-from-Motion (SfM) pipelines assume
perspective projection and struggle with distortion-
heavy fisheye images. Monocular 3D estimation of-
fers a lightweight alternative, yet most models gener-
alize poorly to wide-angle data. UniK3D (Piccinelli
et al., 2025) supports arbitrary intrinsics, including
fisheyes, but it is trained mainly on synthetic wide-
FoV data, leaving its performance on real fisheye im-
agery largely untested.
In this work, we explore whether monocular 3D
estimation can reliably replace SfM for initializing
fisheye-based 3DGS pipelines. Specifically, we eval-
uate how well UniK3D predictions can substitute SfM
point clouds when used with splatting pipelines de-
signed for non-linear projection models. Our goal is
to assess both compatibility and practical reconstruc-
tion quality across a diverse set of real-world scenes.
arXiv:2508.06968v2  [cs.CV]  7 Mar 2026

<!-- page 2 -->
Contributions
We present the first evaluation of
the fisheye-adapted 3D Gaussian Splatting methods,
Fisheye-GS and 3DGUT, on real images with fields
of view exceeding 180° in both indoor and outdoor
scenes.
We provide the first empirical analysis of
UniK3D on real fisheye data with ultra-wide fields
of view and test its effectiveness as an alternative to
SfM-based initialization. We show that monocular es-
timates from only 2–3 fisheye images per scene can be
converted into point clouds that yield reconstruction
quality comparable to SfM. We further evaluate how
reducing the field of view (FoV) affects reconstruc-
tion quality and analyze the trade-off between periph-
eral distortion and scene coverage. Finally, we align
monocular point clouds to the COLMAP coordinate
frame, for compatibility with existing 3D Gaussian
Splatting pipelines.
2
RELATED WORK
2.1
Novel View Synthesis and 3D
Gaussian Splatting
Novel view synthesis aims to generate realistic im-
ages from unseen viewpoints. Neural Radiance Fields
(NeRF) (Mildenhall et al., 2020) model scenes as vol-
umetric radiance fields and achieve high fidelity, but
suffer from slow training and inference due to ray
marching. While non-pinhole camera models such
as fisheye cameras can be incorporated into Radiance
Field based works (Duckworth et al., 2024; Barron
et al., 2023), large-scale evaluations remain limited.
3D Gaussian Splatting (3DGS) (Kerbl et al.,
2023) addresses NeRF’s computational cost by rep-
resenting scenes with anisotropic Gaussians raster-
ized directly in image space, enabling real-time
rendering while preserving photorealism.
Typi-
cal pipelines assume narrow-FoV perspective im-
ages and rely on SfM-based initialization (commonly
COLMAP (Sch¨onberger and Frahm, 2016)).
Al-
though COLMAP supports fisheye models (Bradski,
2000), parameter estimation degrades under strong
distortion or very wide FoV, often requiring pre-
calibrated intrinsics. This restricts scalability when
dealing with uncalibrated fisheye data.
2.2
Fisheye Extensions of 3D Gaussian
Splatting
Recent work has begun adapting 3DGS to non-linear
projection models. Several methods propose modi-
fied projection functions or differentiable distortion
handling (Huang et al., 2024; Deng et al., 2025;
Nazarenus et al., 2024), but results are typically lim-
ited or demonstrated on synthetic data.
Only a few approaches explicitly render circular
fisheye images. Fisheye-GS (Liao et al., 2024) re-
places the perspective projection with an equidistant
model and renders directly in the fisheye domain,
assuming accurate calibration. 3DGUT (Wu et al.,
2025) replaces EWA splatting (Zwicker et al., 2002)
with an Unscented Transform (Gustafsson and Hen-
deby, 2012), enabling projection through arbitrary
non-linear camera models and modeling additional
effects such as reflections and rolling shutter. Both
methods are summarized in the Appendix.
2.3
Learning-Based Methods for
Monocular Fisheye Depth
Estimation
Transformer-based monocular depth models such as
Depth Anything (Yang et al., 2024a; Yang et al.,
2024b) and MoGe (Wang et al., 2024) perform well
on perspective data but degrade under strong distor-
tion. Fisheye-specific self-supervised methods (Ku-
mar et al., 2020; Kumar et al., 2021; Kumar et al.,
2023; Zhao et al., 2024) address this by incorporating
distortion-aware warping or unified projection mod-
els, but typically rely on fixed distortion assumptions
and controlled hardware setups.
UniK3D (Piccinelli et al., 2025) is a recent
transformer-based model supporting arbitrary intrin-
sics, including fisheye lenses, via a spherical scene
representation and angular supervision.
It predicts
depth and ray directions without requiring predefined
distortion models. We use UniK3D to obtain depth
maps and rays from 200◦fisheye images and fuse
them into dense point clouds that replace SfM initial-
ization for 3DGS. Although not trained on real fish-
eye images with extreme FoV, we evaluate whether
UniK3D’s predictions are sufficiently robust for re-
construction with Fisheye-GS and 3DGUT.
3
METHOD
3.1
Camera Calibration and Data
We use the FIORD dataset (Gunes et al., 2025), which
provides 5 indoor and 5 outdoor real-world scenes,
each with 200–400 high-resolution fisheye images
captured using the Insta360 One RS 1-Inch camera
(Insta360, 2024). Each dual-lens capture is split into
two 3264×3264 fisheye PNGs with a 200◦FoV. The

<!-- page 3 -->
GT
FGS - SfM
FGS - Depth
3DGUT - SfM
3DGUT - Depth
Figure 1: Example renderings from Hall and Building Out using SfM- and UniK3D-based initialization (FGS: Fisheye-GS).
Towards the periphery, depth-based results degrade for both methods due to projection ambiguity, while in central regions
they often recover sharper details.
dataset spans a variety of scales and challenging con-
ditions (fog, snow, glare, clutter), making it suitable
for evaluating wide-FoV reconstruction.
Both lenses are calibrated independently using
the Camera Calibration Toolbox for Generic Lenses
(Kannala and Brandt, 2006; Kannala et al., 2008),
and the resulting intrinsics and distortion parame-
ters are converted to OpenCV’s fisheye model for
use in COLMAP’s SfM pipeline. Calibration accu-
racy is verified by rectifying sample images and visu-
ally checking the projected geometry. FIORD is one
of the few real datasets providing 200◦fisheye im-
agery with COLMAP-compatible calibrations, mak-
ing it well suited for controlled evaluation. We use
a standard 90/10 train–test split and report PSNR,
SSIM, and LPIPS (Zhang et al., 2018) metrics on
held-out views.
3.2
Evaluated Gaussian Splatting
Methods for Fisheye Cameras
We evaluate two splatting pipelines designed for fish-
eye imagery: Fisheye-GS (Liao et al., 2024) and
3DGUT (Wu et al., 2025). Both extend 3DGS (Kerbl
et al., 2023) to non-linear camera models. We use
the official implementations and adapt them to our
calibrated fisheye inputs. Splatting requires camera
poses and an initial point cloud; we therefore bench-
mark both traditional SfM initialization and monocu-
lar depth-based initialization (UniK3D), which avoids
multi-view matching and leverages the large scene
coverage of a single fisheye.
Structure-from-Motion (SfM)
We run COLMAP
v3.9.1 (Sch¨onberger and Frahm, 2016) using the

<!-- page 4 -->
OpenCV fisheye model (Bradski, 2000).
Scenes
are processed with calibrated intrinsics (fx, fy, cx,
cy, k1–k4).
High-resolution feature extraction and
vocabulary-tree matching are enabled to improve cor-
respondences under wide distortion. After incremen-
tal registration and triangulation, bundle adjustment
refines poses and sparse geometry, which are exported
for downstream splatting.
Fisheye-GS Implementation
Fisheye-GS is evalu-
ated on the full 200◦dataset, including scenes with
reflections and low-texture regions where distortion
is strongest.
The method assumes an equidistant
projection, where radial displacement scales linearly
with the ray’s incidence angle. Input images are re-
projected using COLMAP intrinsics, but the imple-
mentation omits the k1 distortion term to maintain
the distortion-free equidistant assumption. Since the
equidistant model is defined only up to 180◦(θ = π),
our 200◦imagery introduces projection ambiguity
near the image boundary, allowing us to assess robust-
ness under extreme FoV.
3DGUT Implementation
3DGUT is evaluated on
the same 200◦fisheye data.
Unlike Fisheye-GS,
it directly applies the full non-linear fisheye model
without reprojection and uses the Unscented Trans-
form (Gustafsson and Hendeby, 2012) to propagate
sigma points through the projection function. This
avoids local linearization and improves behavior near
the periphery. We enable the optional MCMC train-
ing mode (Kheradmand et al., 2025), which up-
dates Gaussians via Langevin dynamics (Brosse et al.,
2018) and replaces heuristic densification. We use the
official implementation and adapt it to the intrinsics
of fisheye images.
3DGUT estimates its maximum field of view
(θmax) from image resolution, principal point, and
focal length using a perspective-inspired approxima-
tion. While suitable for perspective datasets, this as-
sumption can be inaccurate for ultra-wide fisheye im-
agery, providing an additional point of comparison in
our experiments.
3.3
Field of View Adjustment
Our fisheye cameras capture 200° views, exceeding
the 180° limit of the equidistant projection assumed
by Fisheye-GS. Beyond this range, rays originate
from behind the optical axis and cannot be projected
consistently. While 3DGUT supports arbitrary cam-
era models, both methods show sensitivity to distor-
tion near the image boundary at wide angles.
To investigate the tradeoff between peripheral dis-
tortion and scene coverage, we generate additional
image sets at 160° and 120°. Each 200° input is repro-
jected to the target FoV under the equidistant model
by adjusting the focal length, rescaling angular coor-
dinates, and discarding pixels beyond the desired an-
gular range. Each FoV variant is then used to train and
evaluate both Fisheye-GS and 3DGUT to assess how
reconstruction quality changes under reduced angular
input.
3.4
Gaussian Initialization: SfM vs
UniK3D
SfM pipelines often degrade on fisheye images due
to distortion and sparse feature matches. As an alter-
native, we initialize Gaussian Splatting using monoc-
ular 3D estimates from UniK3D (Piccinelli et al.,
2025), which supports arbitrary intrinsics—including
fisheyes, unlike most trained depth estimators. This
makes UniK3D well suited for our wide-FoV set-
ting, where standard monocular models fail to pro-
duce consistent geometry.
We use only 2 fisheye views per scene, leverag-
ing the wide FoV to capture most of the environment
in a few frames while avoiding inconsistencies that
arise when fusing many monocular predictions. De-
spite the minimal input, the fused point clouds pro-
vide sufficient structure for Gaussian initialization.
UniK3D outputs points in the local camera frame.
We convert them to the COLMAP world frame us-
ing estimated extrinsics, fuse points across the se-
lected views, and apply a similarity transform to
match COLMAP’s scale. Because UniK3D produces
dense geometry, we downsample its point clouds us-
ing voxel-based sampling to achieve approximate par-
ity with SfM point counts while preserving spatial
coverage. Exact matching is impossible without al-
tering structure, but approximate equality is sufficient
for a fair comparison. The number of initialization
points for both methods is reported in Table 2.
To verify alignment quality, we compute 2D–3D
correspondences between UniK3D predictions and
SfM points and report reprojection errors as a quan-
titative check.
This confirms that UniK3D’s point
clouds are sufficiently consistent with COLMAP
for use in both Fisheye-GS and 3DGUT. Although
UniK3D has not been trained on real fisheye images
with extreme FoVs (e.g., 200◦), our setup allows us to
directly evaluate its suitability as a geometry source
for 3D Gaussian initialization.

<!-- page 5 -->
Table 1: Performance of the Fisheye-GS and 3DGUT methods across scenes and FOVs. SfM based initialization is used.
Metrics: PSNR ↑/ SSIM ↑/ LPIPS ↓. The best average results are bolded.
Scene
Method
FOV = 200°
FOV = 160°
FOV = 120°
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Kitchen
Fisheye-GS
20.05
0.7238
0.3862
24.01
0.8190
0.2509
19.43
0.7713
0.2569
3D-GUT
22.16
0.8140
0.3242
23.29
0.8621
0.2131
20.33
0.8402
0.2128
Hall
Fisheye-GS
19.62
0.7008
0.4024
24.04
0.8445
0.2203
20.98
0.8314
0.2075
3D-GUT
20.91
0.7787
0.3594
22.67
0.8783
0.1946
21.32
0.8766
0.1812
Meeting
Fisheye-GS
17.75
0.6921
0.4540
20.07
0.7468
0.3363
19.19
0.6901
0.3368
3D-GUT
20.93
0.8171
0.3155
22.32
0.8653
0.2168
21.00
0.8331
0.2781
Building
Fisheye-GS
18.57
0.6505
0.4828
20.59
0.7477
0.3181
16.21
0.6991
0.3271
3D-GUT
17.54
0.6281
0.4811
19.46
0.7792
0.4150
16.99
0.7546
0.3885
Upstairs
Fisheye-GS
19.42
0.7231
0.4234
19.28
0.7112
0.3932
17.70
0.7004
0.3607
3D-GUT
16.64
0.5582
0.6955
20.61
0.8181
0.4083
17.27
0.7467
0.6198
Corridor
Fisheye-GS
19.86
0.6432
0.3648
20.58
0.7332
0.2527
15.75
0.6394
0.3179
3D-GUT
20.25
0.6261
0.3385
20.35
0.7422
0.2413
16.37
0.6641
0.2745
Building Out
Fisheye-GS
17.11
0.5664
0.4877
19.20
0.6539
0.3652
18.14
0.6596
0.3411
3D-GUT
18.54
0.6479
0.4461
18.22
0.7510
0.3002
17.66
0.7375
0.2286
Night
Fisheye-GS
33.18
0.9204
0.1838
28.75
0.8710
0.2383
32.62
0.9182
0.1678
3D-GUT
28.19
0.8910
0.1677
28.57
0.8940
0.2110
24.61
0.8667
0.2482
Bridge
Fisheye-GS
23.00
0.7529
0.3606
22.91
0.8311
0.2318
17.37
0.7835
0.2649
3D-GUT
22.06
0.7651
0.4150
21.39
0.8586
0.2059
18.21
0.8235
0.2591
Road
Fisheye-GS
18.21
0.6178
0.4552
19.71
0.7099
0.3357
15.02
0.6397
0.3746
3D-GUT
17.57
0.6405
0.4998
19.53
0.7121
0.3954
17.07
0.7211
0.3957
Average
Fisheye-GS
20.67
0.6991
0.4001
21.91
0.7668
0.2942
19.24
0.7333
0.2955
3D-GUT
20.48
0.7167
0.4043
21.64
0.8161
0.2802
19.08
0.7864
0.3087
4
RESULTS
4.1
Evaluation on Real Fisheye Images
Our primary evaluation uses the full 200◦fisheye in-
puts, where distortion is strongest. Table 1 reports
quantitative results across ten FIORD scenes, span-
ning compact indoor spaces (Kitchen, Hall, Meet-
ing), larger indoor areas (Building, Upstairs), small
outdoor regions (Corridor, Night, Building Out), and
large outdoor environments (Bridge, Road). This di-
versity allows us to analyze how scene scale and cap-
ture extent influence reconstruction quality.
Reconstruction quality depends strongly on scene
scale. In compact indoor scenes, 3DGUT clearly out-
performs Fisheye-GS, showing PSNR gains of 2–3
dB and consistently higher SSIM and lower LPIPS.
This reflects its strength in modeling non-linear dis-
tortion when the scene is bounded or semi-bounded.
In contrast, in larger indoor and outdoor environ-
ments, 3DGUT loses this advantage and sometimes
falls behind Fisheye-GS. These scenes involve longer
baselines, wider spatial coverage, and more input
views, which amplify the limitations of 3DGUT’s
FoV approximation. At 200◦, 3DGUT often produces
blurred reconstructions at the periphery. This behav-
ior aligns with its projection formulation, which esti-
mates the maximum field of view using a perspective-
inspired approximation that becomes inaccurate at
ultra-wide angles, leading to projection ambiguity
near the 200◦boundary. Outdoor scenes further chal-
lenge the method due to low texture (sky, snow, repet-
itive surfaces) and strong lighting variation (glare,
overexposure, nighttime), causing noticeable degra-
dation in the largest scenes (Upstairs, Road).
Overall, 3DGUT preserves peripheral detail well
in compact scenes but becomes unstable in larger-
scale settings.
Fisheye-GS, despite its simplified
equidistant model, offers more stable performance in
these cases due to its tight coupling with COLMAP’s
sparse initialization. These results match the intended
behavior of the two methods: 3DGUT performs best
in small distortion-heavy scenes, while Fisheye-GS
trades local fidelity for greater robustness in large
multi-view environments. Our experiments provide
the first empirical confirmation of these theoretical
trade-offs on real fisheye imagery at 200◦FoV.
4.2
Impact of Field of View Adjustment
We evaluate both methods under reduced FoVs of
160◦and 120◦to study the trade-off between periph-

<!-- page 6 -->
Table 2: Performance comparison of 3D Gaussian splatting renderings for SfM based and UniK3D based Gaussian initializa-
tion across scenes at FOV=200°. Rows stack SfM (top) over Depth (bottom). Metrics: PSNR ↑/ SSIM ↑/ LPIPS ↓. Init Pts
refers to the number of initial Gaussian points. The best average results are bolded.
Scene
Init
# Init Pts
Fisheye-GS
3D-GUT
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Kitchen
SfM
82K
20.05
0.7238
0.3862
22.16
0.8140
0.3242
Depth
85K
21.80
0.7751
0.2982
20.63
0.7782
0.4147
Hall
SfM
147K
19.62
0.7008
0.4024
20.91
0.7787
0.3594
Depth
149K
22.16
0.7824
0.3010
19.80
0.7732
0.4371
Meeting
SfM
55K
17.75
0.6921
0.4540
20.93
0.8171
0.3155
Depth
37K
17.63
0.6692
0.4648
20.48
0.8165
0.4009
Building
SfM
194K
18.57
0.6505
0.4828
17.54
0.6281
0.4811
Depth
189K
19.86
0.6647
0.4033
18.62
0.7230
0.4480
Upstairs
SfM
186K
19.42
0.7231
0.4234
16.64
0.5582
0.6955
Depth
195K
20.95
0.7387
0.3568
14.68
0.5286
0.7222
Corridor
SfM
219K
19.86
0.6432
0.3648
20.25
0.6261
0.3385
Depth
221K
21.90
0.7263
0.3072
20.49
0.6244
0.3951
Building Out
SfM
140K
17.11
0.5664
0.4877
18.54
0.6479
0.4461
Depth
150K
18.31
0.6125
0.3940
18.46
0.6430
0.5140
Night
SfM
20K
33.18
0.9204
0.1838
28.19
0.8910
0.1677
Depth
22K
35.98
0.9427
0.1177
29.00
0.8960
0.2341
Bridge
SfM
235K
23.00
0.7529
0.3606
22.06
0.7651
0.4150
Depth
238K
24.87
0.7901
0.2935
22.02
0.7624
0.4203
Road
SfM
352K
18.21
0.6178
0.4552
17.57
0.6405
0.4998
Depth
352K
18.36
0.6244
0.4440
16.12
0.5747
0.6081
Average
SfM
163K
20.68
0.6991
0.4001
20.48
0.7167
0.4043
Depth
164K
22.18
0.7326
0.3381
20.03
0.7120
0.4595
eral distortion and scene coverage while testing how
these two elements affect the rendering results. As
shown in Table 1, 160◦FoV consistently provides the
best compromise.
For Fisheye-GS, trimming the FoV from 200◦to
160◦yields clear improvements in SSIM and LPIPS
across most scenes (Kitchen, Hall, Meeting, Build-
ing), confirming its sensitivity to peripheral distor-
tion.
Even when PSNR or SSIM remain slightly
higher at 200◦, perceptual quality still improves at
160◦due to lower LPIPS. The only exception is the
Night scene, where 200◦shows marginally higher
PSNR/SSIM because dark peripheral regions inflate
pixel-wise scores after trimming. This does not trans-
late into perceptual gains, and visual quality remains
comparable. At 120◦, performance generally drops
again: distortion is reduced, but excessive cropping
removes important scene content, lowering PSNR and
reducing contextual consistency.
3DGUT shows relatively stable PSNR across
FoVs, but SSIM and LPIPS consistently improve
at 160◦, indicating perceptually closer reconstruc-
tions even when pixel-wise differences remain simi-
lar. This suggests that 160◦FoV reduces distortion ef-
fects without sacrificing necessary coverage. Overall,
160◦provides the most balanced FoV for both meth-
ods: wide enough to maintain scene context, while
sufficiently narrow to limit peripheral distortion.
4.3
Impact of Depth-Based
Initialization
Table 2 reports results using monocular depth-based
initialization versus SfM for Fisheye-GS and 3DGUT.
To avoid bias from denser UniK3D monocular predic-
tions, we applied uniform sampling to the UniK3D
outputs so that the point counts were comparable to
the sparse SfM reconstructions.
This ensures that
differences in the performances obtained by our two
methods are attributable to initialization quality (geo-
metric accuracy of the scene) rather than point count.
For Fisheye-GS, depth-based initialization per-
forms competitively across scenes: most metrics re-
main close to SfM, with slight improvements in some
cases (e.g., Kitchen, Hall, Upstairs, Bridge).
The
main benefit is more accurate geometric coverage
from fewer input views, obtained in a fraction of the
time: UniK3D produces usable geometry in about 10
seconds, compared to roughly an hour for a full SfM
reconstruction. This faster initialization helps capture
fine structure reducing preprocessing cost.
For 3DGUT, depth-based initialization generally
yields lower performance than SfM in several scenes
(e.g., Upstairs, Road), with only mixed gains in
others (e.g., Meeting, Building Out).
The largest
performance drops occur near the periphery, where
strong distortion and projection ambiguity at ultra-

<!-- page 7 -->
wide FoVs amplify errors. Figure 1 illustrates this
trend: depth initialization degrades peripheral regions
due to projection ambiguity, while central regions
may even achieve sharper detail. This spatial trade-off
explains the overall metric drop for 3DGUT, despite
localized gains in accuracy.
Taken together, these findings show that depth-
based initialization is a practical substitute for SfM in
fisheye Gaussian Splatting. While its benefits at 200°
are muted by the limitations of 3DGUT’s projection,
the approach offers substantial savings in preprocess-
ing time and remains effective in distortion-heavy or
challenging indoor scenes where SfM often generates
floaters or is costly to compute.
5
Conclusion
We presented the first systematic evaluation of
fisheye-adapted 3D Gaussian Splatting on real im-
agery with fields of view beyond 180◦. Across the
FIORD dataset, 3DGUT showed advantages in com-
pact environments due to its non-linear distortion han-
dling, while Fisheye-GS proved more stable in large
scenes through its simplified distortion model and
SfM coupling. FoV reduction experiments reinforced
these trends: narrowing the FoV reduces peripheral
distortion, with 160◦offering a strong balance be-
tween coverage and geometric stability, whereas 120◦
removes too much context. We also evaluated depth-
based initialization using UniK3D. Despite lacking
training on real fisheye images, UniK3D generalized
well and enabled Fisheye-GS reconstructions compa-
rable to SfM while reducing preprocessing time sub-
stantially. However, for 3DGUT, peripheral distortion
limited reconstruction quality.
Overall, our results show that fisheye Gaussian
Splatting is feasible without heavy preprocessing and
that monocular depth can serve as a practical alterna-
tive to SfM in distortion-heavy scenes. Future work
may explore regressing Gaussian parameters directly
from monocular 3D estimators and extending these
approaches to larger-scale scenes.
ACKNOWLEDGEMENTS
We acknowledge the financial support of the In-
telligent Work Machines Doctoral Education Pi-
lot Program (IWM VN/3137/2024-OKM-4) and the
Academy of Finland projects 353139 and 362409. We
also acknowledge CSC – IT Center for Science, Fin-
land, for computational resources.
REFERENCES
Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P.,
and Hedman, P. (2023). Zip-nerf: Anti-aliased grid-
based neural radiance fields. ICCV.
Bradski, G. (2000). The OpenCV Library. Dr. Dobb’s Jour-
nal of Software Tools.
Brosse, N., Durmus, A., and Moulines, E. (2018).
The
promises and pitfalls of stochastic gradient langevin
dynamics.
Deng, Y., Xian, W., Yang, G., Guibas, L., Wetzstein,
G., Marschner, S., and Debevec, P. (2025).
Self-
calibrating gaussian splatting for large field of view
reconstruction.
Duckworth, D., Hedman, P., Reiser, C., Zhizhin, P., Thib-
ert, J.-F., Luˇci´c, M., Szeliski, R., and Barron, J. T.
(2024).
Smerf: Streamable memory efficient radi-
ance fields for real-time large-scene exploration. ACM
Trans. Graph., 43(4).
Gunes, U., Turkulainen, M., Ren, X., Solin, A., Kannala,
J., and Rahtu, E. (2025).
Fiord: A fisheye indoor-
outdoor dataset with lidar ground truth for 3d scene
reconstruction and benchmarking. In SCIA.
Gustafsson, F. and Hendeby, G. (2012). Some relations be-
tween extended and unscented kalman filters. Signal
Processing, IEEE Transactions on, 60:545 – 555.
Huang, L., Bai, J., Guo, J., Li, Y., and Guo, Y. (2024). On
the error analysis of 3d gaussian splatting and an opti-
mal projection strategy.
Insta360 (2024). Insta360 One RS 1-Inch 360 Edition.
Kannala, J. and Brandt, S. (2006). A generic camera model
and calibration method for conventional, wide-angle,
and fish-eye lenses.
IEEE Transactions on Pattern
Analysis and Machine Intelligence, 28(8):1335–1340.
Kannala, J., Heikkil¨a, J., and Brandt, S. S. (2008). Geomet-
ric camera calibration.
Wiley encyclopedia of com-
puter science and engineering, 13(6):1–20.
Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis,
G. (2023). 3d gaussian splatting for real-time radi-
ance field rendering. ACM Transactions on Graphics,
42(4).
Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng,
J., Isack, H., Kar, A., Tagliasacchi, A., and Yi, K. M.
(2025). 3d gaussian splatting as markov chain monte
carlo.
Kumar, V. R., Hiremath, S. A., Bach, M., Milz, S., Witt,
C., Pinard, C., Yogamani, S., and M¨ader, P. (2020).
Fisheyedistancenet: Self-supervised scale-aware dis-
tance estimation using monocular fisheye camera for
autonomous driving. In 2020 IEEE International Con-
ference on Robotics and Automation (ICRA), pages
574–581.
Kumar,
V. R.,
Yogamani,
S.,
Bach,
M.,
Witt,
C.,
Milz, S., and Mader, P. (2023).
Unrectdepthnet:
Self-supervised monocular depth estimation using a
generic framework for handling common camera dis-
tortion models.
Kumar, V. R., Yogamani, S. K., Milz, S., and M¨ader, P.
(2021). Fisheyedistancenet++: Self-supervised fish-
eye distance estimation with self-attention, robust loss

<!-- page 8 -->
function and camera view generalization.
In Au-
tonomous Vehicles and Machines, pages 1–11.
Liao, Z., Chen, S., Fu, R., Wang, Y., Su, Z., Luo, H., Ma,
L., Xu, L., Dai, B., Li, H., Pei, Z., and Zhang, X.
(2024). Fisheye-gs: Lightweight and extensible gaus-
sian splatting module for fisheye cameras.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. (2020). Nerf: Repre-
senting scenes as neural radiance fields for view syn-
thesis. In ECCV.
Nazarenus, J., Kou, S., Zhang, F.-L., and Koch, R. (2024).
Arbitrary optics for gaussian splatting using space
warping. Journal of Imaging, 10(12).
Piccinelli, L., Sakaridis, C., Segu, M., Yang, Y.-H., Li, S.,
Abbeloos, W., and Van Gool, L. (2025).
UniK3D:
Universal camera monocular 3d estimation.
In
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR).
Sch¨onberger, J. L. and Frahm, J.-M. (2016).
Structure-
from-motion revisited. In Conference on Computer
Vision and Pattern Recognition (CVPR).
Wang, R., Xu, S., Dai, C., Xiang, J., Deng, Y., Tong, X., and
Yang, J. (2024). Moge: Unlocking accurate monocu-
lar geometry estimation for open-domain images with
optimal training supervision.
Wu, Q., Martinez Esturo, J., Mirzaei, A., Moenne-Loccoz,
N., and Gojcic, Z. (2025).
3dgut: Enabling dis-
torted cameras and secondary rays in gaussian splat-
ting.
Conference on Computer Vision and Pattern
Recognition (CVPR).
Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao,
H. (2024a). Depth anything: Unleashing the power of
large-scale unlabeled data. In CVPR.
Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J.,
and Zhao, H. (2024b). Depth anything v2.
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang,
O. (2018).
The unreasonable effectiveness of deep
features as a perceptual metric.
Zhao, G., Liu, Y., Qi, W., Ma, F., Liu, M., and Ma, J. (2024).
Fisheyedepth: A real scale self-supervised depth esti-
mation model for fisheye camera.
Zwicker, M., Pfister, H., van Baar, J., and Gross, M. (2002).
Ewa splatting.
IEEE Transactions on Visualization
and Computer Graphics, 8(3):223–238.
APPENDIX
Preliminaries
Fisheye-GS
adapts the 3D Gaussian Splatting
pipeline to operate directly on distorted fisheye im-
agery by replacing perspective projection with an
equidistant model. In this formulation, the radial dis-
tance from the optical center to the projected point
on the image plane is proportional to the angle of in-
cidence θ between the incoming ray and the optical
axis:
r = f ·θ
(1)
where θ is defined for a 3D point pc = (xc,yc,zc)⊤in
camera coordinates as:
θ = arctan
 p
x2c +y2c
zc
!
(2)
To support gradient-based optimization, Fisheye-GS
derives the Jacobian of its projection function and in-
tegrates it into the rasterization backend. This enables
accurate backpropagation while preserving spatial fi-
delity in wide-angle views. Aside from the projec-
tion module, the remaining pipeline—Gaussian ini-
tialization, filtering, and rendering—follows the orig-
inal 3DGS implementation.
Although the equidistant model provides a reason-
able approximation up to θ = π (180◦), it becomes
inaccurate when the field of view exceeds this range.
In our setup, the cameras capture up to 200◦, mean-
ing rays with θ > π originate from behind the optical
axis. Such rays cannot be consistently projected onto
a forward-facing image plane, leading to unavoidable
errors near the image periphery.
3D Gaussian Unscented Transform (3DGUT)
takes a different approach to handling non-linear cam-
era models by replacing the Elliptical Weighted Aver-
age (EWA) splatting (Zwicker et al., 2002) used in
standard 3DGS with the Unscented Transform (UT)
(Gustafsson and Hendeby, 2012). Rather than pro-
jecting a single Gaussian mean, each 3D Gaussian
is represented as a set of deterministically sampled
sigma points that approximate the original distribu-
tion:
{xi} = UT(µ,Σ)
(3)
Each sigma point xi is projected individually through
the camera model, which avoids local linearization
and captures non-linear effects such as fisheye distor-
tion more faithfully. This results in a more accurate
rendering of Gaussian contributions, particularly near
the image periphery where standard approximations
break down.
