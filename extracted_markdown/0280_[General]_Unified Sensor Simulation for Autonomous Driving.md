<!-- page 1 -->
Unified Sensor Simulation for Autonomous Driving
Nikolay Patakin1 , Arsenii Shirokov1 , Anton Konushin1 , Dmitry Senushkin1
1Lomonosov Moscow State University
Abstract
In this work, we introduce XSIM, a sensor simula-
tion framework for autonomous driving. XSIM ex-
tends 3DGUT splatting with a generalized rolling-
shutter modeling tailored for autonomous driving
applications. Our framework provides a unified and
flexible formulation for appearance and geomet-
ric sensor modeling, enabling rendering of com-
plex sensor distortions in dynamic environments.
We identify spherical cameras, such as LiDARs,
as a critical edge case for existing 3DGUT splat-
ting due to cyclic projection and time discontinu-
ities at azimuth boundaries leading to incorrect par-
ticle projection. To address this issue, we propose
a phase modeling mechanism that explicitly ac-
counts temporal and shape discontinuities of Gaus-
sians projected by the Unscented Transform at az-
imuth borders. In addition, we introduce an ex-
tended 3D Gaussian representation that incorpo-
rates two distinct opacity parameters to resolve
mismatches between geometry and color distribu-
tions.
As a result, our framework provides en-
hanced scene representations with improved geo-
metric consistency and photorealistic appearance.
We evaluate our framework extensively on multi-
ple autonomous driving datasets, including Waymo
Open Dataset, Argoverse 2, and PandaSet.
Our
framework consistently outperforms strong recent
baselines and achieves state-of-the-art performance
across all datasets.
The source code is publicly
available at https://github.com/whesense/XSIM.
1
Introduction
Development of safe and reliable autonomous vehicles ne-
cessitates access to large-scale and diverse datasets for
both training and evaluation.
However, acquiring diverse
real-world data remains prohibitively expensive and labor-
intensive.
Simulation based on real data has emerged as
a promising alternative, enabling the creation of augmented
datasets in a cost and time efficient manner for various
downstream applications [Yuan et al., 2024; Adamkiewicz
et al., 2022; Ljungbergh et al., 2025]. Recently, 3D Gaus-
sian Splatting (3DGS) [Kerbl et al., 2023; Wu et al., 2025;
Figure 1: Synthetic example of LiDAR rendering. In regions near
the azimuth discontinuity border, the standard 3DGUT projection
results in partially missing and distorted range image renders. Our
phase modeling approach alleviates this issue and also effectively
handles surfaces observed twice due to the rolling shutter.
Hess et al., 2024] has introduced a photorealistic rendering
engine lowering sim-to-real gap and offering an effective bal-
ance among visual fidelity and computational efficiency.
While existing simulators predominantly rely on EWA
splatting [Zwicker et al., 2001], the recently proposed
3DGUT framework [Wu et al., 2025] offers benefits for sen-
sor simulation in autonomous driving scenarios. Real-world
driving data is typically captured using cameras with highly
nonlinear and dynamic effects, such as rolling shutter and op-
tical distortions. These effects are difficult to model accu-
rately with EWA splatting due to its reliance on linearized
particle projection during rasterization, resulting in special-
ized rendering procedures for each camera type. In contrast,
3DGUT projects Gaussians through arbitrary camera models
using the Unscented Transform (UT), effectively emulating
ray-based rendering within a rasterization framework and en-
arXiv:2602.05617v1  [cs.CV]  5 Feb 2026

<!-- page 2 -->
XSIM, Ours
SplatAD
OmniRE
Figure 2: Novel view synthesis (Lane shift 3m) Qualitative comparison on Waymo Open Dataset demonstrates that XSIM
provides scene representation which can be consistently rendered from novel ego-vehicle trajectories.
abling accurate modeling of complex distortions. Building
on this capability, we introduce a sensor simulation frame-
work that adapts 3DGUT splatting to autonomous driving
data. We further extend the framework with a generalized
rolling-shutter modeling, enabling accurate simulation of Li-
DAR and arbitrary camera sensors in unified manner. How-
ever, we observe that UT-based projection for spherical cam-
eras introduces challenges for Gaussian particles spanning the
azimuth boundary. Naive processing of such particles leads
to missing projections or distorted positions and shapes due
to cyclic azimuth parameterization and temporal discontinu-
ities at azimuth edges. To address this issue, we propose a
phase modeling mechanism that explicitly accounts for these
effects, enabling accurate rendering of spherical cameras with
rolling shutter.
Autonomous driving scenarios require jointly modeling ge-
ometric and appearance sensors. Accurate appearance model-
ing typically necessitates multiple transparent Gaussian par-
ticles to capture complex lighting effects, whereas geomet-
ric sensors provide precise surface measurements that can be
faithfully represented using a single opaque Gaussian. To
address this distribution mismatch, we extend scene param-
eterization with two distinct opacity parameters per Gaus-
sian, jointly optimized for color and geometry distributions.
Such parameterization alleviates potential mismatches within
a unified representation and results in enhanced quality.
Eventually, our main contributions are three-fold:
• We introduce XSIM – the sensor simulation framework
for autonomous driving extending 3DGUT splatting and
enabling rendering LiDAR and camera sensors in a uni-
fied manner with generalized rolling shutter modeling.
• We propose a phase modeling mechanism for spherical
rasterization that explicitly accounts temporal and shape
discontinuities of Gaussians projected by the Unscented
Transform at azimuth borders.
• We introduce an extended 3D Gaussian representation
that incorporates two distinct opacity parameters per
Gaussian, jointly optimized for color and geometry dis-
tributions, mitigating distribution mismatches within a
unified representation.
2
Related Work
3DGS and NeRF for automotive scenes.
In automotive
applications Neural Radiance Field (NeRF)
[Mildenhall
et al., 2020; Barron et al., 2023; M¨uller et al., 2022;
Wu et al., 2023; Xie et al., 2023] and 3D Gaussian Splat-
ting (3DGS) [Kerbl et al., 2023; Wu et al., 2025; Hess et
al., 2024; Zhang et al., 2024] gathered significant attention
since it provide flexible tools for photorealistic sensor model-
ing. Initial research efforts [Ost et al., 2021; Fu et al., 2022;
Kundu et al., 2022; Kerbl et al., 2023; Moenne-Loccoz et
al., 2024] emphasized the reconstruction of scene seman-
tic and appearance guided by RGB camera images. Later
works enriched scene representation by decoupled model-
ing of dynamic actors and static layout leveraging either ex-
plicit scene graphs [Yang et al., 2023; Tonderski et al., 2024;
Yang et al., 2024; Hess et al., 2024; Zhou et al., 2024c;
Yan et al., 2024; Zhou et al., 2024b; Zhou et al., 2024a;
Jiang et al., 2025] or time-dependent implicit representa-
tions [Go et al., 2025; Chen et al., 2023; Peng et al., 2025].
Recent approaches further extended prior methods to encom-
pass accurate sensor simulations including modeling of com-
plex camera distortions [Xie et al., 2023; Yang et al., 2023;
Tonderski et al., 2024; Chen et al., 2025a; Zhang et al., 2024;

<!-- page 3 -->
Ground-truth image
XSIM, Ours
SplatAD
OmniRE
Figure 3: Depth map rendering.
Qualitative comparison of depth map rendering on Waymo Open Dataset. Compared to
previous methods, our framework provides smooth geometric representations with high level of details.
Wu et al., 2025; Moenne-Loccoz et al., 2024] and Li-
DAR [Huang et al., 2023; Wu et al., 2024; Yang et al., 2023;
Tonderski et al., 2024; Chen et al., 2025a; Hess et al., 2024;
Zhou et al., 2025; Chen et al., 2025b]. 3DGS provides a
better trade-off between photorealism, physical accuracy and
computational efficiency. However, the conventional 3DGS
based on EWA [Zwicker et al., 2001] formulation introduces
challenges for automotive scenarios limiting the ability to
render accurately complex sensors [Huang et al., 2025]. The
promising alternative 3DGUT [Wu et al., 2025] provides flex-
ibility to render arbitrary cameras without strong approxi-
mation. In this work we introduced the sensor simulation
framework based on 3DGUT [Wu et al., 2025].
In con-
trast to previous 3DGS-based methods [Hess et al., 2024;
Chen et al., 2025b], our framework provides a unified render-
ing formulation for rolling-shutter camera and LiDAR sen-
sors, resulting in improved cross-sensor consistency.
3
Method
We introduce XSIM – the sensor simulation framework for
autonomous driving. Our framework is based on 3DGUT
formulation that we overview in Section 3.1. In Section 3.2
we introduce extended 3D Gaussian representation to allevi-
ate geometry and color distribution mismatches. Finally, we
describe our rolling shutter sensor modeling approach along
with the proposed phase mode mechanism in Section 3.3.
3.1
Gaussian Splatting Preliminary
Scene representation
3DGS [Kerbl et al., 2023; Zwicker
et al., 2001; Wu et al., 2025] represents an arbitrary scene
as a set of transparent Gaussian particles. Each particle is
parameterized by its mean position µ ∈R3 and a shape en-
coded by a covariance matrix Σ ∈R3×3. The contribution of
a particle is defined by a Gaussian kernel function:
ρ(x) = σ exp

−1
2(x −µ)⊤Σ−1(x −µ)

(1)
Covariances Σ = RSS⊤R⊤in practice are represented and
optimized via decoupled scaling vector s ∈R3 and rota-
tion quaternion q ∈R4. Each particle is associated with
opacity value σ ∈R, diffuse color cd ∈R3, and view-
dependent appearance feature vector cs ∈Rf. While orig-
inal 3DGS [Kerbl et al., 2023] uses spherical harmonics for
encoding view-dependent appearance and evaluate them into
color before volumetric integration, recent autonomous driv-
ing simulation frameworks [Tonderski et al., 2024; Hess et
al., 2024] render both cd and cs features into an image. We
follow their approach, and use small trainable network con-
sisting of few convolution layers to decode RGB color.
Dynamic
scenes
are
typically
represented
as
the
graph [Chen et al., 2025b] with static and dynamic ac-
tor nodes,
consisting of individual Gaussian particles.
Dynamic actors are associated with their bounding boxes
and an optimizable trajectory (sequence of SE3 poses). As
pedestrians and cyclists represent a vulnerable road users
category, we follow [Chen et al., 2025b] and represent
humans as SMPL [Loper et al., 2015] bodies to improve
reconstruction quality and provide better controllability.
Unscented transform.
A key stage that enables efficient
rendering in Gaussian splatting is tiling, in which particles
are assigned to screen-space tiles based on the 2D projection
of their shape. The EWA splatting formulation relies on a
linearized projection approximation based on a single point,
which makes rendering highly non-linear cameras (e.g. with
rolling shutter or optical distortions) challenging. Alterna-
tively, 3DGUT projects particles onto an arbitrary camera via
Unscented Transform (UT). Given the mean and covariance
of a particle in 3D, UT constructs a set of sigma points, which
are individually projected through the camera model. The re-
sulting 2D Gaussian conics are then approximated by weight-
ing projected points. As a result, UT enables tiling for a wide
range of complex camera models without requiring any mod-
ifications. Finally, particles assigned to each tile are sorted by
depth to ensure correct ordering during volumetric rendering.

<!-- page 4 -->
LiDAR sensor
Time
Sensor ray direction
Start of capture
End of capture
a)
b)
Figure 4: LiDAR discontinuities occurring near azimuth border. a)
Even when sensor is stationary and covers exactly 360 degrees, time
discontinuity due to the rolling shutter may lead to objects observed
twice. b) Ego movement in combination with rolling shutter leads
to spatial discontinuities
Volumetric integration.
The final stage of rendering is ras-
terization, which performs volumetric integration. In EWA
splatting, 2D Gaussian conics are used explicitly to compute
particle responses during volumetric rendering, introducing
projection approximation errors into the rendering process.
In contrast, 3DGUT uses 2D conics only for tiling, while
volumetric rendering is performed by computing particle re-
sponses directly in 3D space. Specifically, for a given camera
ray with origin o ∈R3 and direction d ∈R3, the particle is
evaluated at the point of maximum response along the ray:
xmax = o + τmaxd,
τmax = d⊤Σ−1(µ −o)
d⊤Σ−1d
(2)
Given the maximum response point, the overall volumetric
integration follows standard formulation with αi = ρi(xmax):
Ti =
Y
j<i
(1 −αj),
c =
X
i
ciαiTi
(3)
Similarly, for depth and range image rendering we use the
same equation as above, but replace particle colors ci with
maximum response τmax distances along the ray.
3.2
Extended 3D Gaussian Representation
While using geometric guidance is generally beneficial for
3DGS scene reconstuction, in some cases the two sens-
ing modalities may impose different requirements on opac-
ity modeling. Accurate geometric modelling requires rep-
resenting surfaces with precisely located opaque gaussians,
as guided by LiDAR measurements.
In contrast, appear-
ance modelling must account for view-dependent effects such
as specular reflections and translucency, which often require
multiple semi-transparent Gaussians along a viewing ray.
Furthermore, surface opacity can be wavelength-dependent,
as materials such as glass exhibit different transparency prop-
erties for LiDAR and visible light. To accommodate these
effects, we augment each Gaussian with separate opacity pa-
rameters: σc for camera and σL for LiDAR rendering, which
are jointly optimized within the unified representation. These
two opacity values are then regularized during optimization
to ensure consistency. We experimentally demonstrate that
this extended representation effectively resolves distribution
mismatches and improves both camera and LiDAR modelling
for the novel-view synthesis problem.
Ground-truth
Ours, With LO
Ours, Without LO
Figure 5: Modeling LiDAR opacity (LO) separately resolves geom-
etry and color distributions mismatch, and increases quality of ap-
pearance modeling for translucent surfaces and specular reflections.
3.3
Sensor Modeling
Autonomous driving perception and planning algorithms pri-
marily rely on two sensor modalities: cameras and LiDARs.
Both sensors typically operate in rolling shutter mode, in
which sensor readings are acquired sequentially over time
in a row-by-row fashion.
As highlited by multiple previ-
ous works [Tonderski et al., 2024; Hess et al., 2024], due to
high velocities experienced in autonomous driving scenarios,
modeling rolling shutter is essential for accurate reconstruc-
tion. Whereas previous work [Hess et al., 2024] addressed
rolling-shutter effects using separate, sensor-specific models,
we describe generalized rolling-shutter modeling approach.
We also identify spherical cameras (i.e. LiDARs) as an edge
case of 3DGUT approach and propose phase modeling mech-
anism to mitigate arising rendering issues.
General rolling shutter modeling.
Due to the rolling-
shutter mechanism, sensor readings are not captured instanta-
neously, and the observation time varies across pixels. Con-
sequently, each image-space point (u, v) is associated with
a distinct capture time τ(u, v). While in practice the rolling
shutter time is linear and aligned with either horizontal or ver-
tical image axis, we can express it as:
τ(u, v) = τstart + uτu + vτv
(4)
where τu, τv define the scan direction and speed, and denote
middle of exposure time as τmid = τ(0.5, 0.5).
As time progresses during image acquisition, both the sen-
sor and the scene evolve. We assume that, over the duration
of capture, the camera and all dynamic actors move with con-
stant linear and angular velocities. Under this assumption, the
position of a 3D world point xw ∈R3 at time η is given by:
xw(η) = xw(τmid) + (va + wa × r)η
(5)
where va ∈R3 and wa ∈R3 denote linear and angular ve-
locities of the dynamic actor, respectively, and r ∈R3 is
radius vector defined by point position in actor coordinates.
Similarly, the camera pose evolves over time according to
constant-velocity motion. Let q(η) ∈R4 and t(η) ∈R3
denote the camera orientation and position at time η. Camera
movement is then modeled as:
q(η) = ewcη/2 ⊗q(τmid),
t(η) = t(τmid) + ηvc
(6)

<!-- page 5 -->
Ground-truth projection
Esimated 2D Gaussian
Sigma-point projections
Visible range
Negative shift range
Positive shift range
x
x
x
x
Unscented transform
x
x
x
x
x
x
x
x
x
x
x
x
x
x
Ours, with phase modeling
Figure 6: Single Gaussian particle spanning the azimuth boundaries (φ = ±π) of rolling-shutter spherical camera projects into two separate
2D gaussians with different 2D covariances. Unscented Transform provides unimodal posterior approximation of particle projection, leading
to a overly large projections with incorrect shapes. Our phase modeling mechanism enables bimodal 2D gaussian projection by considering
two additional projections shifted by ±π from visible range. By performing projections shifted by half period, we handle particles which
sigma points fall into multiple azimuth periods.
where vc and wc are the camera’s linear and angular veloci-
ties, and ⊗denotes the Hamilton product. Given the arbitrary
static camera projection function π(·) : R3 −→[0, 1]2 and
known point time η projection of world point xw ∈R3 into
image coordinates (u, v) has the closed form:
xc(η) = q−1(η) ⊗(xw(η) −t(η)) ⊗q(η),
u(η), v(η) = π(xc(η)),
(7)
However, since the observation time η of the point is typi-
cally not known prior to projection, the projection problem
formulates as estimating world time η which is consistent
with image space point observation time:
η = τ(u(η), v(η))
(8)
Although this dependency is nonlinear and does not admit
a closed-form solution, it can be efficiently solved using iter-
ative methods. Following [Sun et al., 2020], we use the New-
ton–Raphson method and observe that only a few iterations
are sufficient in practice.
Phase modeling.
Our framework builds upon 3DGUT
splatting, which leverages the Unscented Transform (UT) to
approximate the projection of a 3D Gaussian. The UT per-
forms a posterior approximation under the assumption that
the projection of a 3D Gaussian can be adequately repre-
sented by a single 2D Gaussian. While this assumption holds
in many practical settings, it breaks down in scenarios involv-
ing spherical rolling-shutter sensors. In particular, spinning
spherical LiDAR sensors may observe a single Gaussian par-
ticle multiple times when it spans the azimuthal boundary
or ignore it depending on the world dynamics (see Fig. 4).
Moreover, this can lead to a same particle projecting into two
separate shapes near boundaries with different covariances
and depths (see ground-truth projections in Fig. 6). In such
cases, the projected distribution becomes multimodal, violat-
ing the assumptions inherent to the UT.
To address this limitation, we introduce a phase modeling
mechanism for spherical camera rendering. Under the spher-
ical projection, the azimuth is a periodic function defined as:
ϕ = atan2(y, x) + 2πk,
k ∈Z
(9)
While conventional static spherical mappings retain only
the principal solution (k = 0), we explicitly account for phase
wrapping by considering k ∈{−1, 0, +1}. Since additional
projections may differ in covariance and depth to camera, in
addition to the standard central projection, we perform auxil-
iary projections shifted by ±π (see Fig. 6). For each interval,
we constrain projections of sigma points into it by initializing
τ in Eq. 8 with an interval middle of exposure time, and shift
projected point azimuths by 2πk if they fall outside of inter-
val. 2D Gaussian conics and depth are individually estimated
for each interval based on projected sigma points, and valid
projections are passed to the next tiling stage.
Compared
to original 3DGUT projection, our mechanism results in ac-
curate tiling with no false tile-particle intersections and less
depth sorting errors for particles near azimuth boundaries.
3.4
Camera and LiDAR supervision
Our scene representation consisting of multiple object nodes
is optimized simultaneously from driving logs by randomly
sampling images and closest by time LiDAR sweeps at each
iteration. We supervise it using combination of losses:
L = λL1 + (1 −λ)LSSIM
|
{z
}
camera guidance
+ Ldepth
| {z }
LiDAR
+Lopacity + Lreg
(10)
Following common practice we set λ = 0.2. LiDAR guid-
ance loss is L1 loss between rendered and ground-truth ray
lengths. To ensure consistency between optimized opacities
we impose Lopacity = P
i |σc,i −σL,i| regularization. Details
on other regularization terms Lreg are listed in appendix.
4
Experiments
Implementation details.
We implement camera and Li-
DAR rendering using custom CUDA kernels.
A unified
rendering pipeline allows sharing rasterization forward and
backward passes across all camera models.
To handle
non-uniform LiDAR beam angles during tiling, particle as-
signment iterates over elevation tile boundaries [Hess et
al., 2024].
Human modeling follows OmniRE [Chen et
al., 2025b], using their deformable and SMPL-based scene
nodes.
We adopt the specular Gaussian configuration and
CNN post-processing from [Hess et al., 2024; Tonderski et

<!-- page 6 -->
Dataset
Method
Conference
Reconstruction
Novel-view synthesis
PSNR↑
SSIM↑
LPIPS↓
CD↓
PSNR↑
SSIM↑
LPIPS↓
CD↓
Waymo
(12 scenes)
PVG
Arxiv23
25,02
0,8005
0,4408
82,11
24,29
0,7864
0,4451
68,27
StreetGS
ECCV24
25,45
0,8123
0,3111
16,16
24,41
0,7827
0,3198
16,87
OmniRE
ICLR25
25,94
0,8159
0,3049
15,68
24,60
0,7765
0,3207
13,76
HUGS
CVPR24
26,90
0,8513
0,3351
44,58
25,95
0,8267
0,3337
37,60
EmerNerf
ICLR24
27,15
0,8056
0,4620
0,71
26,12
0,7962
0,4575
2,54
SplatAD
CVPR25
27,74
0,8650
0,2807
0,82
27,06
0,8492
0,2807
0,82
XSIM, Ours
–
30,75
0,9030
0,2228
0,08
29,80
0,8904
0,2236
0,18
Argoverse
(10 scenes)
PVG
Arxiv23
23,78
0,7164
0,4840
31,36
22,93
0,7031
0,4908
27,20
StreetGS
ECCV24
23,85
0,7223
0,3824
21,62
22,37
0,6975
0,3806
21,01
OmniRE
ICLR25
23,97
0,7230
0,3822
21,87
22,44
0,6975
0,3815
22,02
UniSim
CVPR23
23,04
0,6697
0,3986
25,58
23,06
0,6694
0,3962
26,24
NeuRad
CVPR24
26,46
0,7271
0,3045
2,43
26,49
0,7271
0,3044
2,63
SplatAD
CVPR25
28,71
0,8333
0,2653
2,78
28,40
0,8258
0,2706
2,68
XSIM, Ours
–
29,44
0,8431
0,2563
0,57
29,44
0,8423
0,2514
1,26
Pandaset
(10 scenes)
PVG
Arxiv23
23,62
0,7066
0,4405
101,33
22,81
0,6885
0,4537
121,54
StreetGS
ECCV24
23,70
0,7192
0,3206
18,68
22,53
0,6866
0,3249
19,90
OmniRE
ICLR25
23,73
0,7196
0,3246
21,02
22,58
0,6884
0,3262
18,99
UniSim
CVPR23
23,62
0,6953
0,3291
10,46
23,45
0,6910
0,3300
9,68
NeuRad
CVPR24
26,54
0,7675
0,2386
1,45
26,05
0,7589
0,2418
1,65
SplatAD
CVPR25
28,69
0,8759
0,1853
1,54
26,77
0,8044
0,1904
1,69
XSIM, Ours
–
29,05
0,8839
0,1872
0,20
27,00
0,8055
0,1944
1,23
Table 1: Quantitative results on three datasets under scene reconstruction and novel-view synthesis scenarios. We report RGB image quality
metrics (PSNR, SSIM, LPIPS) and LiDAR reconstruction accuracy measured by Chamfer Distance (CD). Our framework achieves state-of-
the-art performance across all datasets and scenarios, with particularly large error reductions for LiDAR rendering. On PandaSet, LPIPS
remains competitive and ranks second, with a minor gap to the best-performing method.
al., 2024]. Hyperparameters largely follow SplatAD [Hess et
al., 2024], while Gaussian splitting and densification use the
3DGUT [Wu et al., 2025] strategy with minor modifications.
Full hyperparameter details are provided in the appendix.
Datasets.
To evaluate our framework we use three popular
datasets – Waymo Open Dataset [Sun et al., 2020], Argov-
erse2 [Wilson et al., 2021] and PandaSet [Xiao et al., 2021].
Following OmniRE [Chen et al., 2025b], we use 12 scenes
from Waymo, featuring ego-vehicle motion, dynamic and di-
verse classes of vehicles and pedestrians. As for Argoverse2
and PandaSet we adopt SplatAD [Hess et al., 2024] partition
without modifications. Both training and evaluation are per-
formed using full-resolution images and point clouds.
Baselines.
For experimental evaluation we choose a wide
range of baselines, featuring both recent NeRF-based meth-
ods (UniSim [Yang et al., 2023], NeuRAD [Tonderski et
al., 2024], EmerNerf [Yang et al., 2024]) and 3DGS-based
PVG [Chen et al., 2023], StreetGaussians [Yan et al., 2024],
OmniRe [Chen et al., 2025b], HUGS [Zhou et al., 2024b;
Zhou et al., 2024a] and SplatAD [Hess et al., 2024]. For
PVG, StreetGaussians and OmniRe we use implementation
based on drivestudio. As UniSim has no official reposi-
tory, we use reimplementation by neurad-studio.
Evaluation metrics.
For rendered image quality assess-
ment we use standard novel-view synthesis metrics – PSNR
↑, SSIM ↑, and LPIPS↓. To measure LiDAR simulation qual-
ity we compute Chamfer Distance (CD↓) metric between ren-
dered and ground-truth point clouds.
4.1
Image rendering
Scene Reconstruction.
Following common practice, we
evaluate the upper bound of reconstruction and modeling ca-
pacity of frameworks by reconstructing scenes using all avail-
able RGB images and LiDAR sweeps. Quantitative results
on three datasets are reported in Table 1. On the Waymo
dataset, XSIM achieves substantial improvements over the
previous state of the art, SplatAD, with gains of +3.01 PSNR
and +3.8% SSIM, while reducing LPIPS by 20.6%. Across
all three datasets, XSIM consistently demonstrates superior
RGB image reconstruction quality compared to prior base-
lines. On PandaSet, performance remains competitive across
all metrics, with LPIPS showing a minor performance drop.
Novel-view synthesis.
As our framework targets simula-
tion of novel scenarios and trajectories, we evaluate render-

<!-- page 7 -->
Ground-truth
XSIM, Ours
SplatAD
OmniRE
Figure 7: LiDAR Rendering. Comparison of LiDAR point clouds rendered by different methods. Previous approaches exhibit distorted
scan-line patterns and incomplete geometry, including vulnerable road users such as pedestrians.
ing quality on views unseen during reconstruction (Table 1).
We train on every second sensor frame and evaluate on the
remaining frames. In this setting, XSIM achieves gains of
+2.74 PSNR on Waymo and +1.04 PSNR on Argoverse over
previous state-of-the-art method. We further assess extrapola-
tion capability in Figure 2 by rendering ego-vehicle cameras
under a lateral shift of 3 meters. Compared to prior methods,
our framework produces more consistent renderings.
4.2
LiDAR and depth rendering
We evaluate LiDAR rendering quality on three datasets in
both scene reconstruction and novel-view synthesis settings
using Chamfer Distance (CD) (Table 1).
XSIM achieves
state-of-the-art results on all datasets, with an x8.8 CD error
reduction on Waymo reconstruction and an x4.5 error reduc-
tion on Waymo NVS. As shown in Figure 7, XSIM better
preserves characteristic LiDAR ring patterns and accurately
renders pedestrians. We further assess geometric quality via
RGB-camera depth rendering (Figure 3), where our method
produces smooth, dense depth maps.
4.3
Ablation study
We illustrate the effect of our phase modeling mechanism
via synthetic example (fig. 1) with a single car mesh cap-
tured by rolling shutter LiDAR camera. As the LiDAR moves
along the vehicle, it becomes visible twice near the azimuthal
boundaries of the range image due to rolling shutter. While
3DGUT default projection produces artifacts at the bound-
aries, our framework with phase modeling precisely recon-
structs and renders the scene. We further ablate our frame-
work components in a novel-view synthesis setting on six
scenes from Waymo dataset by individually disabling features
(Tab. 2). Camera rolling-shutter modeling (c vs. a) leads
to significant gains in PSNR. Removing the separate LiDAR
opacity parameter (c vs. a) degrades both image and LiDAR
metrics, with qualitative effects shown in Figure 5. Model-
ing LiDAR rolling shutter (d vs. c) with our phase modeling
mechanism (e vs. c) further improves image and LiDAR ren-
dering demonstrating effectiveness of our contributions.
Component
PSNR↑SSIM↑LPIPS↓CD↓
(a) XSIM, full
30,03
0,8945
0,2122
0,21
(b) – Camera rolling shutter
28,95
0,8761
0,2407
0,21
(c) – Lidar opacity
29,55
0,8888
0,2189
0,25
(d)
– Lidar rolling shutter
29,05
0,8816
0,2296
0,31
(e)
– Phase modelling
29,32
0,8824
0,2245
0,28
Table 2: Ablation studies on novel-view synthesis on Waymo dataset
(half of split). Each row corresponds to disabling of a single com-
ponent relative to the parent configuration.
5
Conclusion
In this paper, we presented XSIM, a unified sensor simula-
tion framework for autonomous driving that extends 3DGUT
splatting with generalized rolling-shutter modeling, enabling
modeling of complex sensors in dynamic environments in
a unified manner. Our phase modeling mechanism enables
robust rendering for spherical rolling-shutter sensors by ex-
plicitly accounting for azimuthal discontinuities. Extensive
experiments on multiple autonomous driving benchmarks
demonstrate that our approach consistently improves geomet-
ric accuracy and photorealism, outperforming strong previous
baselines.

<!-- page 8 -->
References
[Adamkiewicz et al., 2022] Michal Adamkiewicz, Timothy
Chen, Adam Caccavale, Rachel Gardner, Preston Cul-
bertson, Jeannette Bohg, and Mac Schwager.
Vision-
Only Robot Navigation in a Neural Radiance World.
IEEE Robotics and Automation Letters (RA-L), 7(2):4606–
4613, April 2022. website: https://mikh3x4.github.io/nerf-
navigation/.
[Barron et al., 2023] Jonathan T. Barron, Ben Mildenhall,
Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-
nerf: Anti-aliased grid-based neural radiance fields. ICCV,
2023.
[Chen et al., 2023] Yurui Chen, Chun Gu, Junzhe Jiang, Xi-
atian Zhu, and Li Zhang. Periodic vibration gaussian: Dy-
namic urban scene reconstruction and real-time rendering.
arXiv:2311.18561, 2023.
[Chen et al., 2025a] Yun Chen, Matthew Haines, Jingkang
Wang, Krzysztof Baron-Lis, Sivabalan Manivasagam,
Ze Yang, and Raquel Urtasun. Salf: Sparse local fields
for multi-sensor rendering in real-time.
arXiv preprint
arxiv:2507.18713, 2025.
[Chen et al., 2025b] Ziyu Chen, Jiawei Yang, Jiahui Huang,
Riccardo de Lutio,
Janick Martinez Esturo,
Boris
Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco
Pavone, Li Song, and Yue Wang. Omnire: Omni urban
scene reconstruction. In The Thirteenth International Con-
ference on Learning Representations, 2025.
[Fu et al., 2022] Xiao Fu, Shangzhan Zhang, Tianrun Chen,
Yichong Lu, Lanyun Zhu, Xiaowei Zhou, Andreas Geiger,
and Yiyi Liao. Panoptic nerf: 3d-to-2d label transfer for
panoptic urban scene segmentation. In International Con-
ference on 3D Vision (3DV), 2022.
[Go et al., 2025] Hyojun Go, Byeongjun Park, Jiho Jang,
Jin-Young Kim, Soonwoo Kwon, and Changick Kim.
Splatflow: Multi-view rectified flow model for 3d gaus-
sian splatting synthesis. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 21524–21536, June 2025.
[Hess et al., 2024] Georg Hess, Carl Lindstr¨om, Maryam
Fatemi, Christoffer Petersson, and Lennart Svensson.
Splatad: Real-time lidar and camera rendering with 3d
gaussian splatting for autonomous driving. arXiv preprint
arXiv:2411.16816, 2024.
[Huang et al., 2023] Shengyu Huang,
Zan Gojcic,
Zian
Wang, Francis Williams, Yoni Kasten, Sanja Fidler, Kon-
rad Schindler, and Or Litany. Neural lidar fields for novel
view synthesis. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), pages
18236–18246, October 2023.
[Huang et al., 2025] Letian Huang, Jiayang Bai, Jie Guo,
Yuanqi Li, and Yanwen Guo.
On the error analysis of
3d gaussian splatting and an optimal projection strategy.
In Computer Vision – ECCV 2024, pages 247–263, Cham,
2025. Springer Nature Switzerland.
[Jiang et al., 2025] Junzhe Jiang, Nan Song, Jingyu Li, Xi-
atian Zhu, and Li Zhang.
Realengine: Simulating au-
tonomous driving in realistic context.
arXiv preprint
arXiv:2505.16902, 2025.
[Kerbl et al., 2023] Bernhard
Kerbl,
Georgios
Kopanas,
Thomas Leimk¨uhler, and George Drettakis.
3d gaus-
sian splatting for real-time radiance field rendering. ACM
Transactions on Graphics, 42(4), July 2023.
[Kundu et al., 2022] Abhijit Kundu, Kyle Genova, Xiaoqi
Yin, Alireza Fathi, Caroline Pantofaru, Leonidas J.
Guibas, Andrea Tagliasacchi, Frank Dellaert, and Thomas
Funkhouser. Panoptic neural fields: A semantic object-
aware neural scene representation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 12871–12881, June 2022.
[Ljungbergh et al., 2025] William Ljungbergh, Adam Ton-
derski, Joakim Johnander, Holger Caesar, Kalle ˚Ast¨om,
Michael Felsberg, and Christoffer Petersson. Neuroncap:
Photorealistic closed-loop safety testing for autonomous
driving.
In European Conference on Computer Vision,
pages 161–177. Springer, 2025.
[Loper et al., 2015] Matthew Loper,
Naureen Mahmood,
Javier Romero, Gerard Pons-Moll, and Michael J. Black.
Smpl: a skinned multi-person linear model. ACM Trans.
Graph., 34(6), October 2015.
[Mildenhall et al., 2020] Ben Mildenhall, Pratul P. Srini-
vasan, Matthew Tancik, Jonathan T. Barron, Ravi Ra-
mamoorthi, and Ren Ng. Nerf: Representing scenes as
neural radiance fields for view synthesis. In ECCV, 2020.
[Moenne-Loccoz et al., 2024] Nicolas
Moenne-Loccoz,
Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Mar-
tinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp,
and Zan Gojcic.
3d gaussian ray tracing: Fast tracing
of particle scenes. ACM Transactions on Graphics and
SIGGRAPH Asia, 2024.
[M¨uller et al., 2022] Thomas M¨uller, Alex Evans, Christoph
Schied, and Alexander Keller.
Instant neural graphics
primitives with a multiresolution hash encoding.
ACM
Trans. Graph., 41(4):102:1–102:15, July 2022.
[Ost et al., 2021] Julian Ost, Fahim Mannan, Nils Thuerey,
Julian Knodt, and Felix Heide.
Neural scene graphs
for dynamic scenes.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 2856–2865, June 2021.
[Peng et al., 2025] Chensheng Peng, Chengwei Zhang, Yix-
iao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng,
Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan.
Desire-gs:
4d street gaussians for static-dynamic de-
composition and surface reconstruction for urban driving
scenes. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
6782–6791, June 2025.
[Sun et al., 2020] Pei Sun, Henrik Kretzschmar, Xerxes
Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui,

<!-- page 9 -->
James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vi-
jay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Alek-
sei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao,
Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen,
and Dragomir Anguelov. Scalability in perception for au-
tonomous driving: Waymo open dataset. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), June 2020.
[Tonderski et al., 2024] Adam Tonderski, Carl Lindstr¨om,
Georg Hess, William Ljungbergh, Lennart Svensson, and
Christoffer Petersson. Neurad: Neural rendering for au-
tonomous driving.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 14895–14904, 2024.
[Wilson et al., 2021] Benjamin Wilson, William Qi, Tanmay
Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khan-
delwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett,
Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and
James Hays. Argoverse 2: Next generation datasets for
self-driving perception and forecasting.
In Proceedings
of the Neural Information Processing Systems Track on
Datasets and Benchmarks (NeurIPS Datasets and Bench-
marks 2021), 2021.
[Wu et al., 2023] Zirui Wu, Tianyu Liu, Liyi Luo, Zhide
Zhong, Jianteng Chen, Hongmin Xiao, Chao Hou, Haozhe
Lou, Yuantao Chen, Runyi Yang, Yuxin Huang, Xiaoyu
Ye, Zike Yan, Yongliang Shi, Yiyi Liao, and Hao Zhao.
Mars: An instance-aware, modular and realistic simula-
tor for autonomous driving. In Lu Fang, Jian Pei, Guang-
tao Zhai, and Ruiping Wang, editors, CICAI (1), volume
14473 of Lecture Notes in Computer Science, pages 3–15.
Springer, 2023.
[Wu et al., 2024] Hanfeng
Wu,
Xingxing
Zuo,
Stefan
Leutenegger, Or Litany, Konrad Schindler, and Shengyu
Huang. Dynamic lidar re-simulation using compositional
neural fields. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR),
pages 19988–19998, June 2024.
[Wu et al., 2025] Qi Wu, Janick Martinez Esturo, Ashkan
Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3dgut:
Enabling distorted cameras and secondary rays in gaus-
sian splatting. Conference on Computer Vision and Pattern
Recognition (CVPR), 2025.
[Xiao et al., 2021] Pengchuan Xiao, Zhenlei Shao, Steven
Hao, Zishuo Zhang, Xiaolin Chai, Judy Jiao, Zesong Li,
Jian Wu, Kai Sun, Kun Jiang, Yunlong Wang, and Diange
Yang.
Pandaset: Advanced sensor suite dataset for au-
tonomous driving. In 2021 IEEE International Intelligent
Transportation Systems Conference (ITSC), pages 3095–
3101, 2021.
[Xie et al., 2023] Ziyang Xie, Junge Zhang, Wenye Li, Feihu
Zhang, and Li Zhang. S-neRF: Neural radiance fields for
street views. In The Eleventh International Conference on
Learning Representations, 2023.
[Yan et al., 2024] Yunzhi Yan, Haotong Lin, Chenxu Zhou,
Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang,
Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling
dynamic urban scenes with gaussian splatting. In ECCV,
2024.
[Yang et al., 2023] Ze Yang, Yun Chen, Jingkang Wang, Siv-
abalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang,
and Raquel Urtasun. Unisim: A neural closed-loop sen-
sor simulator. In CVPR, 2023.
[Yang et al., 2024] Jiawei Yang, Boris Ivanovic, Or Litany,
Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che,
Danfei Xu, Sanja Fidler, Marco Pavone, and Yue Wang.
EmerneRF: Emergent spatial-temporal scene decomposi-
tion via self-supervision.
In The Twelfth International
Conference on Learning Representations, 2024.
[Yuan et al., 2024] Tianyuan Yuan, Yucheng Mao, Jiawei
Yang, Yicheng Liu, Yue Wang, and Hang Zhao. Presight:
Enhancing autonomous vehicle perception with city-scale
nerf priors. In Computer Vision – ECCV 2024: 18th Euro-
pean Conference, Milan, Italy, September 29–October 4,
2024, Proceedings, Part LXXVII, page 323–339, 2024.
[Zhang et al., 2024] Baowen Zhang, Chuan Fang, Rakesh
Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan.
Rade-gs: Rasterizing depth in gaussian splatting, 2024.
[Zhou et al., 2024a] Hongyu Zhou, Longzhong Lin, Jiabao
Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue
Wang, Andreas Geiger, and Yiyi Liao.
Hugsim:
A
real-time, photo-realistic and closed-loop simulator for
autonomous driving.
arXiv preprint arXiv:2412.01718,
2024.
[Zhou et al., 2024b] Hongyu Zhou, Jiahao Shao, Lu Xu,
Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang,
Andreas Geiger, and Yiyi Liao. Hugs: Holistic urban 3d
scene understanding via gaussian splatting. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 21336–21345, June 2024.
[Zhou et al., 2024c] Xiaoyu Zhou, Zhiwei Lin, Xiaojun
Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang.
Drivinggaussian: Composite gaussian splatting for sur-
rounding dynamic autonomous driving scenes.
In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 21634–21643,
June 2024.
[Zhou et al., 2025] Chenxu Zhou, Lvchang Fu, Sida Peng,
Yunzhi Yan, Zhanhua Zhang, Yong Chen, Jiazhi Xia, and
Xiaowei Zhou.
LiDAR-RT: Gaussian-based ray tracing
for dynamic lidar re-simulation.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025.
[Zwicker et al., 2001] Matthias Zwicker, Hanspeter Pfister,
Jeroen van Baar, and Markus Gross. Ewa volume splat-
ting. In Proceedings of the Conference on Visualization
’01, VIS ’01, page 29–36, 2001.

<!-- page 10 -->
Appendices
A
Generic rolling-shutter camera projection
We provide full listing of our generalized rolling shutter pro-
jection in Algorithm 1. Given the static camera projection
function π(x) 7→(u, v), rolling-shutter time function τ(u, v),
sensor velocities vc, wc, torque-corrected actor point veloc-
ity v′
a, middle of exposure camera SE3 camera pose q0, t0,
Algorithm 1 allows to project world-space point xw directly
into a rolling-shutter image. Due to unknown point obser-
vation time η, this algorithm iteratively finds solution for
η = τ(u(η), v(η)) equation. Here we use Newton-Raphson
method for finding equation root, that requires computing Ja-
cobian of discrepancy function ∆η = η −τ(u(η), v(η)).
For linear rolling shutter time function, Jacobian is defined
as:
d(∆η)
dη
= 1 −τu
du
dη −τv
dv
dη
(11)
In its turn, screen-space coordinate derivatives with respect
to time can be computed as:
du
dη = dπu
dxc
dxc
dη
dv
dη = dπv
dxc
dxc
dη
(12)
where xc – point position in camera coordinate system. Here,
dxc
dη does not depend on the camera model and is derived once
and computed analytically in implementation. Jacobian of
camera
dπ
dxc here is the same as Jacobian used in EWA splat-
ting for approximating projection. For instance, given a per-
spective camera projection function πperspective(x) = (fx x
z +
cx, fy
y
z + cy) Jacobian is defined as:
dπperspective
dxc
=
 
fx
z
0
−fxx
z2
0
fy
z
−fyy
z2
!
(13)
For LiDAR rendering, spherical projection is defined as:
πspherical(x, y, z) = (atan2(y, x), arcsin
y
p
x2 + y2 + z2 )
(14)
and has Jacobian matrix (provided by [Hess et al., 2024]):
dπspherical
dxc
=


−
y
x2+y2
x
x2+y2
0
−
xz
r2√
x2+y2
−
yz
r2√
x2+y2
√
x2+y2
r2


(15)
Note, that in general differentiability of projection function
π(x) is not required. In practice, Newton-Raphson method
can be swapped with fixed-point iteration method by assum-
ing d∆η
dη = 1. While both methods typically converge to pre-
cise solutions, we observe that in practice Newton-Raphson
method converges 1-2 iterations faster than fixed-point itera-
tion.
B
Phase modeling
Complete listing of Unscented Transform with phase mod-
eling mechanism for spherical rolling-shutter camera is pro-
vided in Algorithm 2.
For simplicity of description, here
Algorithm 1 πrolling: Rolling-shutter point projection
Input: π(x), xw, η0
Parameters: τ(u, v), vc, wc, v′
a, q0, t0, ∆ηthr, N
Output: u, v, d, η, isValid
1: Let η = η0.
2: Let i = 0
3: while (∆η > ∆ηthr) or (i ≤N) do
4:
// According to Equation 7 of the main paper
5:
u, v, d = π(q−1(η)) ⊗(xw(η) −t(η)) ⊗q(η))
6:
// (u, v) ∈[0; 1]2 – screen space point coordinates
7:
// d – projected point depth
8:
∆η = η −τ(u, v)
9:
d(∆η)
dη
= 1 −τu du
dη −τv dv
dη
10:
// Newton-Raphson iteration:
11:
η = η −∆η/ d(∆η)
dη
12:
++i
13: end while
14: isValid = (d > 0) and (∆η < ∆ηthr)
15: return u, v, d, η, isValid
we assume that projection function return values in radians,
rather than normalized coordinates. As UT(·) function we de-
note standard Unscented Transform, which constructs sigma
points, projects them via given function π(·), and returns
µ2D = (µφ, µθ), particle 2D extent (φext, θext) based on
estimated 2D covariance, depth d, and flag I that specifies
if projection is sucessful and projected particle extent inter-
sects with visible image range. By considering additional
projections πnegative and πpositive, our mechanism handles cases
where particle 2D projection becomes bimodal.
C
Framework details
Initialization
We initialize the Gaussian scene representa-
tion using LiDAR sweeps, object bounding box annotations,
and camera data available in the driving logs. For each Li-
DAR sweep, we use the corresponding bounding box annota-
tions to separate points into static background and individual
dynamic actors. Points are colored by projecting them onto
the corresponding camera images. Points that are not visible
in any camera and therefore lack color are colored using the
three nearest neighbors with known color. For dynamic ac-
tors known to be symmetric (e.g., vehicles), we additionally
symmetrize points along the longitudinal axis. Background
and actor points are then randomly downsampled to meet
node-specific thresholds. Additional points are allocated us-
ing inverse-distance sphere sampling to cover regions of the
scene outside LiDAR coverage.
Scene nodes
Our framework is fully modular, consisting of
multiple nodes that form the scene representation, render it,
and compute loss functions given a camera input. We begin
by estimating dynamic object SE(3) poses and velocities at
the sensor’s mid-exposure time. Object poses are initialized
from dataset annotations and refined using learnable additive
corrections. Poses and velocities at arbitrary times are ob-
tained by interpolating the trajectory. Following [Chen et al.,
2025b], the scene Gaussian representation comprises static,

<!-- page 11 -->
Algorithm 2 Unscented transform with phase modeling
Input: µ3D, Σ3D, πrolling(x)
Parameters: τ(u, v), vc, wc, v′
a, q0, t0
Output: µ2D ∈RK×3, φext ∈RK, θext ∈RK, d ∈RK
1: // Define visible range projection function
2: // Initialize solver with τmid time.
3: πcentral := πrolling(x, τmid)
4: // Perform standard UT
5: µC
φ, µC
θ , φC
ext, θC
ext, dC, IC = UT(µ3D, Σ3D, πcentral)
6: // If projection does not intersect with visible range
7: if not IC then
8:
return ∅
9: end if
10: // Even if central projection is valid, particle still can be
seen twice
11: // Define negative shift projection shifted by −π
12: πnegative(x) := πrolling(x, τstart) −2π[πrolling(x, τstart) ≥0]
13: µL
φ, µL
θ , φL
ext, θL
ext, dL, IL = UT(µ3D, Σ3D, πnegative)
14: // Define positive shift projection shifted by +π
15: πpositive(x) := πrolling(x, τend) + 2π[πrolling(x, τstart) < 0]
16: µR
φ, µR
θ , φR
ext, θR
ext, dR, IR = UT(µ3D, Σ3D, πpositive)
17: // If both projections valid, return both
18: if IL and IR and φL
ext < π and φR
ext < π then
19:
return (µL
2D, µR
2D), (φL
ext, φR
ext), (θL
ext, θR
ext), (dL, dR)
20: end if
21: // Only one projection is valid, return it
22: if IL and φL
ext < π then
23:
return µL
φ, µL
θ , φL
ext, θL
ext, dL
24: end if
25: if IR and φR
ext < π then
26:
return µR
φ, µR
θ , φR
ext, θR
ext, dR
27: end if
28: return µC
φ, µC
θ , φC
ext, θC
ext, dC
rigid, deformable, and SMPL scene nodes. The SMPL node
represents humans detected by a pretrained pose and shape
estimation network, while the deformable node is used for un-
detected far-range pedestrians and cyclists. The deformable
node employs a learnable MLP with instance embeddings to
deform actor-frame Gaussians according to sensor time. The
concatenated world-space Gaussians are then rendered us-
ing our 3DGUT-based procedure. The resulting color feature
maps are transformed into RGB renderings by a small CNN
post-processor [Hess et al., 2024]. Finally, the RGB render-
ing is alpha-composited with a learnable environment cube
map to represent the sky.
Loss functions
Our scene representation consisting of mul-
tiple object nodes is optimized simultaneously from driving
logs by randomly sampling images and closest by time Li-
DAR sweeps at each iteration. We supervise it using combi-
nation of losses:
L = λL1 + (1 −λ)LSSIM
|
{z
}
camera guidance
+ Ldepth
| {z }
LiDAR
+Lopacity + Lreg
(16)
Specifically, Lreg loss function is defined as 0.01Lmask +
0.01Lpose + LSMPL. Following OmniRE [Chen et al., 2025b]
we use sky segmentation masks and use Lmask to penalize
rendered image alpha for pixels assigned by segmentation
mask to sky. Lpose loss function is a L2 penalty on actor tra-
jectory adjustments that prevents drift of actors inside their
coordinate system. LSMPL is also derived from OmniRE, con-
sisting of multiple regularizations that constrain Gaussians
onto SMPL body shape.
Gaussian strategy
We use densification and splitting strat-
egy proposed by 3DGUT [Wu et al., 2025] with minimal
modifications. While vanilla 3DGS [Kerbl et al., 2023] uses
2D position gradients as a criteria for particle splitting and
cloning, 3DGUT framework uses particle positions in 3D di-
rectly and does not produce 2D positional gradients directly.
Instead, 3D position gradients norm multiplied by distance
to camera is used as a direct criteria replacement. When su-
pervised by RGB camera rendering, high 3D positional gra-
dients are typically correspond to underrepresented scene re-
gions, causing densification by design. This intuition falls
apart in case of LiDAR supervision, which produces strong
3D positional gradients along the rays, causing excessive par-
ticle densification. We solve this issue by accumulating gradi-
ents for criteria only based on RGB cameras supervision. To
unify densification strategy with original 3DGS [Kerbl et al.,
2023], we also add the same densification criteria based on
2D scales. To enforce proper scene decomposition, we prune
dynamic nodes particles which fall outside of their bound-
ing boxes. We also modify opacity-based pruning criteria to
be based on maximum value out of lidar and camera particle
opacities.
Optimization
We optimize our scene representation si-
multaneously with an Adam optimizer for 40000 iterations.
Gaussian particles learning rates match SplatAD [Hess et al.,
2024], while most other scene node-specific learning rates are
derived from [Chen et al., 2025b]. We detail specific learn-
ing rate values in Table 3. All learning rates are scheduled
with 500 iterations warm-up and exponential decay if multi-
ple LR-s are specified.
Parameter
Initial LR
Final LR
Positions
1.6e-4
1.6e-6
Scale
5e-3
Rotation
1e-3
Camera opacity
0.05
LiDAR opacity
0.05
Diffuse color
2.5e-3
Specular features
2.5e-3
Actors quaternion corr-s
1e-5
5e-6
Actors translation corr-s
5e-4
1e-4
Deformable actor embeds
1e-3
1e-4
Deformable MLP
8e-3
8e-4
Post-process CNN
1e-3
Env light texture
0.01
Table 3: Optimization learning rates

<!-- page 12 -->
D
Datasets
This section lists details related to datasets we use in our
experimental evaluation.
Overall, we use three datasets –
Waymo Open Dataset [Sun et al., 2020], Argoverse 2 [Wilson
et al., 2021], PandaSet [Xiao et al., 2021].
Waymo Open Dataset.
Dataset features ≈19 seconds
long sequences with camera rig consisting of five RGB
cameras (front, front left, front right, left and right).
We
use full resolution (1920x1080 for front and 1920x886
for side cameras) for training and evaluation.
We choose
12 scenes from the split used by OmniRE [Chen et al.,
2025b] that feature ego-vehicle movement, diverse range
of dynamic objects (vehicles, buses, heavy trucks, con-
struction vehicles) and vulnerable road users (pedestrians,
cyclicts).
Specifically,
we use the following scenes:
10231929575853664160 1160 000 1180 000
(16), 10391312872392849784 4099 400 4119 400
(21), 12027892938363296829 4086 280 4106 280
(94), 12251442326766052580 1840 000 1860 000
(102), 13254498462985394788 980 000 1000 000
(149), 1382515516588059826 780 000 800 000
(172), 16801666784196221098 2480 000 2500 000
(323), 17388121177218499911 2520 000 2540 000
(344), 1918764220984209654 5680 000 5700 000
(402), 4487677815262010875 4940 000 4960 000
(552), 454855130179746819 4580 000 4600 000
(555), 9653249092275997647 980 000 1000 000
(788)
For ablations we used half of this scenes list: 21, 94, 344,
552, 555, 788. For modeling humans as SMPL [Loper et al.,
2015] bodies, we use poses and shape parameters provided
by [Chen et al., 2025b].
Argoverse 2.
Dataset consists of ≈15.5 seconds long
sequences with seven RGB cameras (2048x1550 resolu-
tion for six cameras and frontal camera with 1550x2048
resolution).
Following
[Hess et al., 2024] we crop bot-
tom 250 pixels of three cameras that contain ego-vehicle.
We reuse the same 10 sequences split used by previ-
ous works
[Tonderski et al., 2024; Hess et al., 2024]:
05fa5048-f355-3274-b565-c0ddc547b315,
0b86f508-5df9-4a46-bc59-5b9536dbde9f,
185d3943-dd15-397a-8b2e-69cd86628fb7,
25e5c600-36fe-3245-9cc0-40ef91620c22,
27be7d34-ecb4-377b-8477-ccfd7cf4d0bc,
280269f9-6111-311d-b351-ce9f63f88c81,
2f2321d2-7912-3567-a789-25e46a145bda,
3bffdcff-c3a7-38b6-a0f2-64196d130958,
44adf4c4-6064-362f-94d3-323ed42cfda9,
5589de60-1727-3e3f-9423-33437fc5da4b
PandaSet.
Provides sequences of ≈8 seconds each with
an RGB camera rig consisting of six cameras.
All cam-
eras have 1920x1080 resolution. We also crop bottom 260
pixels from the back camera to remove ego-vehicle. As in
previous works [Tonderski et al., 2024; Hess et al., 2024],
we use 10 sequences: 1, 11, 16, 53, 63,84,106,
123,158.
Synthetic example for projection comparison (main
paper).
To illustrate the effect of our phase model-
ing mechanism, we constructed a synthetic scene with
a single vehicle mesh (VW Golf MK4 by Jay-Artist,
https://www.blendswap.com/blend/3976).
We
simulate
rolling-shutter LiDAR by rendering ground-truth range im-
ages using a custom ray tracer. We sample ground-truth Li-
DAR images with resolution 1024x256 (uniform elevation
beams from −20◦to 20◦) emulating moving along static ve-
hicle at 30-60km/h speeds with vehicle seen near azimuth
discontinuity boundaries. We optimize 3DGS using our ren-
dering method with phase modelling mechanism and demon-
strate the same representation rendered with standard un-
scented transform projection. Optimization with standard UT
from scratch produces poor result due to excessive particles
splitting caused by high positional gradients of boundary par-
ticles.
E
Qualitative comparisons
In this section we provide additional qualitative comparisons
with previous methods.
While previous methods provide
noisy geometry reconstructions illustrated in Figure 8, XSIM
precisely reconstructs geometry without artifacts Figure 9.
Consistency between reconstructed appearance and geome-
try representations allow XSIM to produce clean renders with
low distortions even for laterally shifted trajectories, as illus-
trated in Figure 10. Additionally, we demonstrate comparison
of RGB camera renders on Pandaset dataset in Figure 11.

<!-- page 13 -->
Ground-truth
XSIM, Ours
SplatAD
OmniRE
Argoverse
Argoverse
Argoverse
Pandaset
Pandaset
Figure 8: Qualitative comparison of LiDAR point cloud rendering with previous methods on Argoverse and Pandaset datasets.

<!-- page 14 -->
Ground-truth
XSIM, Ours
SplatAD
OmniRE
Figure 9: Qualitative comparison of rendered depth maps with previous methods on Waymo dataset.

<!-- page 15 -->
XSIM, Ours
SplatAD
OmniRE
Figure 10: Lane shift 3m visualizations on Waymo dataset. XSIM provides consistent appearance renderings without floating and blurring
artifacts on both static and dynamic objects.

<!-- page 16 -->
Ground-truth
XSIM, Ours
SplatAD
OmniRE
Figure 11: Pandaset dataset. RGB camera rendering comparison. Our general rolling-shitter modeling enables accurate actor reconstructions
with precise appearance and geometry under dynamic complex lighting conditions.
