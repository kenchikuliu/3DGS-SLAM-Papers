<!-- page 1 -->
EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene
Reconstruction and Editing
Xijie Yang1,2
Mulin Yu2
Changjian Jiang1
Kerui Ren3,2
Tao Lu2
Jiangmiao Pang2
Dahua Lin4,2
Bo Dai5,6*
Linning Xu4,2
1Zhejiang University
2Shanghai Artificial Intelligence Laboratory
3Shanghai Jiao Tong University
4The Chinese University of Hong Kong
5The University of Hong Kong
6Feeling AI
Figure 1.
Scene editing on 2D Gaussian primitives of a reconstructed real indoor scene, F-CLASSROOM, including: a) relighting the
ceiling, b) inserting colorful emissive balls, c) duplicating a chair with modified material, d) adding a diffuse ball, and e) importing a lamp
from another scene. Path-traced rendering after editing produces coherent global illumination (reflections, interreflections, and shadows)
in contrast to direct radiant scene composition.
Abstract
Recent reconstruction methods based on radiance field such
as NeRF and 3DGS reproduce indoor scenes with high vi-
sual fidelity, but break down under scene editing due to
baked illumination and the lack of explicit light transport.
In contrast, physically based inverse rendering relies on
mesh representations and path tracing, which enforce cor-
rect light transport but place strong requirements on ge-
ometric fidelity, becoming a practical bottleneck for real
indoor scenes. In this work, we propose Emission-Aware
Gaussians and Path Tracing (EAG-PT), aiming for physi-
cally based light transport with a unified 2D Gaussian rep-
resentation. Our design is based on three cores: (1) us-
ing 2D Gaussians as a unified scene representation and
transport-friendly geometry proxy that avoids reconstructed
mesh, (2) explicitly separating emissive and non-emissive
components during reconstruction for further scene editing,
∗corresponding author
and (3) decoupling reconstruction from final rendering by
using efficient single-bounce optimization and high-quality
multi-bounce path tracing after scene editing. Experiments
on synthetic and real indoor scenes show that EAG-PT pro-
duces more natural and physically consistent renders after
editing than radiant scene reconstructions, while preserving
finer geometric detail and avoiding mesh-induced artifacts
compared to mesh-based inverse path tracing. These re-
sults suggest promising directions for future use in interior
design, XR content creation, and embodied AI.
1. INTRODUCTION
Given multi-view captures of an indoor scene, modern 3D
reconstruction methods such as Neural Radiance Fields
(NeRF) [34] and 3D Gaussian Splatting (3DGS) [23] can re-
cover scene representations that achieve high-fidelity novel-
view synthesis. Compared to implicit neural representations
and traditional mesh, the explicit Gaussian primitives used
1
arXiv:2601.23065v1  [cs.GR]  30 Jan 2026

<!-- page 2 -->
in 3DGS provide direct access to geometry and appearance
parameters, making them attractive 3D representation for
interactive scene manipulation and editing. However, de-
spite their representational flexibility, radiance-field-based
reconstructions fail to produce physically consistent ren-
ders after scene editing. Modifying light sources, materi-
als, or object layout does not yield corresponding changes
in illumination or shadowing. This limitation stems from
a shared modeling assumption: the whole scene is treated
as uniformly radiant, with illumination implicitly encoded
in outgoing radiance. While sufficient for reproducing the
appearance at capture time, this formulation does not model
light transport and therefore breaks under changes to scene
configuration.
Prior efforts partially address this limitation by intro-
ducing limited reflection modeling [27, 41, 55, 70]. These
methods add one or a small number of light bounces to im-
prove view-dependent effects, yet they continue to rely on
radiance cached from the original scene and do not explic-
itly reconstruct physical light sources. As a result, indirect
illumination remains tied to the capture-time lighting con-
figuration, and physically correct global illumination after
editing remains out of reach.
At the other end of the spectrum, physically based in-
verse rendering [28, 53] has long relied on mesh represen-
tations and path tracing to model light transport explicitly.
Figure 2. Renders of F-CLASSROOM before and after editing. Ra-
diant Scene: Most radiance field reconstruction works [23, 34, 35]
regard the whole scene as radiant, which cannot produce light
changes and shadow effects after scene editing. Radiant Reflec-
tion: Some reflection modeling works [27, 70] add a single bounce
to produce more realistic results, while still suffering from the in-
correct radiance after scene editing. Radiant Emission: We ex-
plicitly separate light sources from the radiant scene, and use path
tracing to bounce light in the scene to derive photo-realistic ren-
ders.
While physically grounded, mesh-based inverse path trac-
ing places strong requirements on geometric fidelity, which
become a practical bottleneck for real indoor scenes with
cluttered layouts and fine-scale structures. Errors in recon-
structed meshes directly propagate through visibility, shad-
ing, and multi-bounce illumination, often dominating the
final rendering quality. Recent work such as UGP [69] ex-
plores path tracing on Gaussian primitives, but focuses on
forward rendering and does not address inverse reconstruc-
tion.
To address these limitations, we propose Emission-
Aware Gaussians and Path Tracing (EAG-PT), a physically
grounded reconstruction framework that enables consistent
indoor scene editing without mesh conversion. EAG-PT is
built around four tightly coupled components, each target-
ing a core challenge in physically based scene reconstruc-
tion:
• Emission-aware scene decomposition. We explicitly sep-
arate emissive light sources from non-emissive geometry
using 2D emission masks, fixing previous inappropriate
radiant scene modeling for further editing.
• Inverse recovery of radiance and material properties. We
recover emitter radiance and spatially varying surface re-
flectance for non-emissive components via differentiable
rendering from multi-view observations.
• Physically based light transport after editing. We apply
path tracing to edited scenes to re-evaluate multi-bounce
global illumination, avoiding reliance on obsolete radi-
ance cache at capture time.
• Unified 2D Gaussian representation. We adopt 2D Gaus-
sians as a single scene representation that supports ray
intersection, light transport, material modeling, and radi-
ance caching, enabling efficient reconstruction and ren-
dering.
Our method produces physically consistent renders with
realistic global illumination after scene editing, as illus-
trated in Fig. 2. Experiments on both synthetic and real
scenes demonstrate clear improvements over prior radiance-
field-based approaches for indoor scene reconstruction and
editing.
On real-world indoor scenes, comparisons with
mesh-based inverse path tracing show that our unified Gaus-
sian representation preserves finer geometric detail and
yields more stable and visually natural results. Together,
these results indicate that EAG-PT enables practical indoor
scene editing, with broad applicability to interior design,
XR content creation, and physically grounded asset prepa-
ration for embodied AI.
2. RELATED WORK
Below, we briefly review prior work on multi-view 3D re-
construction and inverse rendering for scenes, with a focus
on indoor environments. We organize existing approaches
2

<!-- page 3 -->
by their treatment of light transport into three categories: ra-
diant scene reconstruction, reflection modeling, and meth-
ods based on path tracing.
2.1. Radiant Scene Reconstruction
Most multi-view 3D reconstruction methods implicitly treat
the scene as radiant. Classical pipelines [22, 30, 45, 46, 49]
reconstruct point clouds or meshes, attach them with view-
independent colors, and render via rasterization. NeRF [34]
and its extensions [3, 37], inspired by volume rendering [7],
instead represent the scene as a continuous radiance field
optimized by ray marching and alpha blending, achieving
significantly improved novel-view synthesis. 3DGS [23]
and its variants [17, 31, 42] further model the scene as ra-
diant 3D or 2D Gaussian primitives rendered by rasteriza-
tion (splatting [72]) with alpha blending for high efficiency,
while recent works [6, 14, 35, 51] adopt ray tracing on
Gaussian primitives to alleviate some limitations of rasteri-
zation.
Despite differences in representation and rendering,
these approaches share a key limitation: illumination is
baked into appearance.
As a result, they faithfully re-
produce captured views but fundamentally cannot support
physically plausible scene editing, such as modifying light
sources or object layout. Our work departs from this formu-
lation by explicitly separating emission from reflection and
modeling multi-bounce light transport.
2.2. Reflection Modeling
Radiant methods model view-dependent effects using
learned angular conditioning, such as MLP in NeRF [32,
34, 54] and spherical harmonics in 3DGS [23, 37]. While
effective for specular effects, they still lack explicit light
transport, limiting physically consistent scene editing.
Reflection-aware approaches separate scenes into a re-
flective base and an emitting environment, which works
well for object-centric or local scenes [10, 13, 15, 16, 29,
55, 67, 68].
For indoor environments, where emission
and reflection are tightly coupled, recent methods jointly
optimize materials and illumination within a single re-
gion [27, 41, 70]. However, these methods typically treat
much of the scene as radiant and do not explicitly recon-
struct physical light sources.
In contrast, we represent indoor scenes using 2D Gaus-
sians [15, 17, 55, 67] and explicitly partition them into
emissive and non-emissive components, enabling physi-
cally based light transport. The well-defined distance and
normal of 2D Gaussians provide more stable geometry for
light transport than volumetric 3D Gaussians [23, 35, 41].
2.3. Path Tracing
Path tracing [21] is the standard tool for simulating phys-
ically correct global illumination.
Most prior indoor in-
verse rendering methods rely on meshes for geometry, ei-
ther recovering materials and lighting directly in the mesh
domain or parameterizing them with neural networks [2, 19,
25, 28, 38, 53, 60, 62, 65]. In these approaches, rendering
quality is fundamentally constrained by mesh fidelity. For
real-world indoor scenes, reconstructed meshes often fail
to capture fine-scale geometry and are frequently converted
from other representations such as SDFs or Gaussian prim-
itives [17, 61, 63, 64], introducing additional degradation.
By contrast, radiance field and Gaussian-based recon-
structions offer more faithful geometric representations but
are rarely integrated with physically based light transport.
Existing efforts include I2SDF [70] that adds a single
bounce on NeRF and ESR-NeRF [20] that considers emis-
sion modeling on objects, but they ignore path tracing for
global illumination; UGP [69] adopts path tracing on Gaus-
sian primitives for forward rendering yet does not address
inverse reconstruction.
We aim to bridge this gap by enabling physically based
light transport directly on Gaussian scene representations.
EAG-PT reconstructs emissive components and recovers
SVBRDF properties for non-emissive geometry, and ap-
plies multi-bounce path tracing after scene editing without
mesh conversion. This formulation preserves fine-scale de-
tail while maintaining physically consistent global illumi-
nation.
3. METHOD
Our goal is to reconstruct a static indoor scene from multi-
view images captured in linear color space with known cam-
era poses, and to enable physically correct scene editing
and photo-realistic path-traced rendering.
The input im-
ages densely cover the indoor environment, ensuring direct
observation of light emitters. As illustrated in Fig. 3, our
pipeline proceeds in two stages. Radiant scene reconstruc-
tion in Stage 0 first lifts multi-view observations into a 3D
representation of 2D Gaussians, followed by material re-
covery in Stage 1, which estimates albedo for non-emissive
regions. After scene editing, path tracing is adopted on de-
rived scene for photo-realistic renders.
We begin in Sec. 3.1 by introducing 2D Gaussians as the
scene representation in our method, along with the associ-
ated tracing and bouncing used for rendering. We adopt 2D
Gaussians as they provide a favorable trade-off between ge-
ometric accuracy and appearance quality. In Sec. 3.2, we
describe radiant scene reconstruction, where the scene is
initialized using differentiable rendering with tracing only,
without any light bouncing. Building on this initial radi-
ance field, Sec. 3.3 introduces the rendering equation and
details material recovery for non-emissive regions via dif-
ferentiable rendering with a single bounce into the radiance
cache. Finally, Sec. 3.4 presents multi-bounce path tracing
for physically based forward rendering, together with the
3

<!-- page 4 -->
Figure 3. Pipeline of Emission-Aware Gaussians and Path Tracing. Given multi-view linear captures of an indoor scene with corresponding
emitter masks and estimated normals, the radiant scene is first reconstructed in Stage 0 to get radiance, separate emitters, and derive
geometry, based on 2D Gaussians and ray tracing. The material of the non-emitters is then recovered in Stage 1 through light bouncing and
differentiable rendering. With properties of emitters, non-emitters, and scene geometry, path tracing that bounces light around the scene is
adopted for photo-realistic renders on various scene editing scenarios.
corresponding light baking techniques.
3.1. 2D Gaussian Ray Tracing and Bouncing
2D Gaussian Ray Tracing
[17] represents scenes using
2D Gaussians, which can be interpreted as small ellipti-
cal surface elements embedded in 3D space, enabling more
geometrically compatible reconstruction over 3DGS. While
subsequent works [15, 55] introduce limited ray tracing af-
ter splatting to query radiance from object surfaces or the
environment, our method relies exclusively on ray tracing
over 2D Gaussians. We therefore reformulate 2D Gaus-
sians directly as traceable primitives. Each 2D Gaussian
is centered at a 3D position ⃗p and has finite spatial ex-
tent, with higher influence near its center and smoothly de-
caying weight toward its boundary. It is parameterized by
anisotropic in-plane scale su,v, orientation represented by a
quaternion ˆq, and opacity σ. Given a ray with origin ⃗O and
direction ˆω, the intersection point in 3D space is given by
⃗x = ⃗O+tˆω, where t denotes the ray distance. The response
of 2D Gaussian at the intersection is:
g(⃗x) = exp

−1
2
h (⃗x −⃗p) · ˆtu
su
2 +
 (⃗x −⃗p) · ˆtv
sv
2i
,
(1)
where ˆtu,v are unit vectors along the short and long axes
derived from ˆq.
For a scene composed of 2D Gaussians, a ray keeps trac-
ing forward and sequentially intersects with ng 2D Gaus-
sians.
At each intersection, a per-Gaussian micro-level
quantity vi is accumulated via alpha compositing to pro-
duce a macro-level quantity V , until the accumulated trans-
parency T falls below a preset threshold:
V =
ng
X
i=1
wi ·vi =
ng
X
i=1
Ti−1 ·σigi ·vi, Ti =
iY
j=1
(1−σjgj).
(2)
In our formulation, the quantities include:
v ∈{±ˆn, t, r, e, ρ}, V ∈{N, D, R, E, P}.
(3)
For each 2D Gaussian (and a given ray), ±ˆn = ±ˆtu × ˆtv
(N) denotes the surface normal oriented toward the camera
center, t (D) is the intersection distance, r (R) is the linear
radiance, e (E) is the emissive term in [0, 1], and ρ (P) is
the diffuse albedo. The accumulated normal and distance
are normalized as ˜N = N/||N||, ˜D = D/A, where A =
1 −Tn.
Light Bouncing
After tracing, to bounce the ray, the ef-
fective macro-level intersection point is defined as ⃗X =
⃗O + ˜Dˆω for the new origin.
And the new direction
ˆω′ is sampled from the upper hemisphere Ω+ defined by
˜N, with sampling probability p(ˆω′).
The new ray then
proceeds and collects new set of tracing results V ′
∈
{ ˜N ′, ˜D′, R′, E′, P ′}, and ⃗X′ = ⃗X + ˜D′ˆω′, after which
the process either terminates or starts another bounce. Note
that, light bounces are not applied when reconstructing ra-
diant scene (Sec. 3.2), while a single bounce is added for
4

<!-- page 5 -->
material recovery (Sec. 3.3) and multiple bounces are re-
quired for path tracing (Sec. 3.4). For simplicity, we omit
intersection point and only use direction ˆω to represent a ray
in subsequent equations.
3.2. Radiant Scene Reconstruction
Given multi-view inputs and an initial coarse RGB point
cloud of an indoor scene, we first lift 2D observations into
a 3D radiant scene represented by a collection of 2D Gaus-
sians.
Radiance Loss.
At Stage 0, the scene is radiant, which
means outgoing radiance L0
o(ˆωo) of a pixel is radiance
R(ˆωo) of the ray. Following prior work on radiant scene re-
construction, we employ the standard color reconstruction
loss Lc [17, 23, 35] for radiance. To improve numerical sta-
bility when optimizing linear radiance values, we apply a
perceptual quantization curve PQ(·) [48, 56], resulting in
L0
c = Lc

PQ
 L0
o(ˆωo)

, PQ
 Lo,gt

.
Geometry Loss.
By only applying radiance loss, the re-
constructed radiant scene looks photo-realistic, but gener-
ally with poor geometry [31, 41], which is insufficient for
precise light bounce.
To improve geometric fidelity, we
follow [17, 28, 55] and supervise both surface orientation
and depth variation. Specifically, render normal ˜N and dis-
tance normal ˜Nd (the gradient of ˜D) are directly supervised
by mono normal maps estimated from sRGB images us-
ing StableNormal [59]: Ln = ||1 −˜N · Nmono||1, Ld =
||1 −˜Nd · Nmono||1.
Empirically, we observe that even
state-of-the-art monocular depth estimators lack the accu-
racy and cross-view consistency required for reliable su-
pervision, whereas monocular normal estimation provides
more stable and geometrically meaningful guidance.
Emission Loss.
To explicitly distinguish physical light
sources from reflective surfaces, we incorporate 2D emis-
sion masks for scene editing and path tracing. Given an
emission mask M, we supervise the emissive component E
via Le = ||E −M||1. Details on the construction of 2D
emission masks are provided in Appendix 7.
Final Loss.
We jointly optimize the parameters of all 2D
Gaussians using differentiable 2D Gaussian ray tracing, by
minimizing the weighted sum of the above losses given
weights λc, λn, λd, λe:
min
⃗p,s,ˆq,σ,r,e L0 = λcL0
c + λnLn + λdLd + λeLe.
(4)
After this stage, a radiant scene is reconstructed, with
accurate geometry for light bouncing, separation of emit-
ters and non-emitters, true radiance of emitters, and radi-
ance cache of non-emitters. This representation serves as
the foundation for material recovery with a single bounce
(Sec. 3.3) and multi-bounce path tracing (Sec. 3.4).
3.3. Material Recovery via Single Bounce
With the reconstructed radiant scene obtained in Sec. 3.2,
we perform single-bounce differentiable rendering to re-
cover material properties of the 2D Gaussians (Stage 1 in
Fig. 3). We first review the rendering equation under our
assumptions, and then describe the material recovery pro-
cedure enabled by a single bounce and radiance cache.
The Rendering Equation
The rendering equation [21]
depicts light transport inside 3D space at each surface
point. Since emission typically dominates reflection, we
follow [28, 53] and assume that emissive surfaces do not
reflect incoming light, enabling a clean separation between
emission and reflection:
Lo(ˆωo) = Le(ˆωo) + Lr(ˆωo)
= Le(ˆωo) if E(ˆωo) > τE else Lr(ˆωo),
(5)
where ˆωo is the viewing direction, Lo the outgoing radiance,
Le the radiance from emitters, and Lr the reflected radi-
ance. τE = 0.1 is used to keep a smooth transition between
emitters and non-emitters. For emitters, Le(ˆωo) = R(ˆωo).
For reflection of non-emitters:
Lr(ˆωo) =
Z
Ω+ Li(ˆωi) f(ˆωi, ˆωo) (ˆωi · ˜N) dˆωi,
(6)
where ˆωi is the incident direction, Li the incident radiance,
and f the BRDF. This formulation is recursive, as Li itself
depends on radiance reflected from other surfaces.
Material Recovery
For material recovery, we follow the
idea of using radiance cache (or irradiance cache) from
[27, 36, 52, 70]. By approximating the incident radiance
Li using the accumulated radiance R obtained from the ra-
diant scene reconstruction stage, we remove the need for
iterative multi-bounce simulation during material recovery.
The reflected radiance is estimated via Monte Carlo integra-
tion with nspp samples per pixel:
L1
r(ˆωo) ≈
1
nspp
nspp
X
R(ˆωi)f(ˆωi, ˆωo) (ˆωi · ˜N)
p(ˆωi)
, ˆωi ∼Ω+.
(7)
In this work, we assume a diffuse BRDF f(ˆωi, ˆωo) = P/π
and optimize the diffuse albedo ρ of each 2D Gaussian by
min
ρ L1 = λcLc

PQ
 L1
o(ˆωo)

, PQ
 Lo,gt

.
(8)
5

<!-- page 6 -->
Notice that radiance r is only optimized in Stage 0 (Sec. 3.2)
and kept fixed in this stage to avoid the diffuse-emission
ambiguity, as pointed out in [53, 60].
3.4. Path Tracing for Edited Scenes
Path Tracing
After scene editing, previous works [27, 35,
41, 70] that do not use path tracing, still use L0
o(ˆωo) or
L1
o(ˆωo) with obsolete radiance cache R at capturing time
for final rendering. However, scene editing (e.g. changing
light color, inserting object, etc.) should change R, and us-
age of obsolete radiance cache produces unnatural render-
ing results.
In our method, we only keep accurate radiance of emit-
ters, drop obsolete radiance cache of non-emitters, and
adopt path tracing to derive final renders after scene edit-
ing. This should correctly solve the recursion in Eqs. 5,6.
Each ray shoots from the camera center, intersects with 2D
Gaussians at ⃗
X1, and bounces around inside the scene at
⃗X2, · · · , ⃗Xb, until ⃗Xb is emissive. The ray is discarded if
bounce count exceeds the given bounce limit τB, and valid
paths are averaged to reduce sampling noise:
Lpt
r (ˆωo) ≈
1
nspp
nspp
X
Le(ˆωi,b)
b
Y
k=1
f(ˆωi,k, ˆωo,k) (ˆωi,k · ¯Nk)
p(ˆωi,k)
,
ˆωi,k ∼Ω+
k ,
Le(ˆωi,b) = R(ˆωi,b)
if
 E(ˆωi,b) > τE and b ≤τB

else 0.
(9)
Lpt
o (ˆωo) is finally sent to a denoiser [8] for better visual
quality.
Light Baking
While path tracing produces physically ac-
curate renderings after scene editing, its computational cost
precludes real-time performance, and low sample counts of-
ten result in visible noise or blur. To enable efficient visu-
alization of edited scenes, we adopt a light baking strategy
inspired by commercial game engines [47]. Specifically,
we re-bake the radiance obtained from path tracing into the
2D Gaussian representation by directly optimizing the per-
Gaussian radiance r:
min
r
Llb = λcLc

PQ
 L0
o(ˆωo)

, PQ
 Lpt
o (ˆωo)

.
(10)
This transfers global illumination effects into the radiant
scene representation, enabling interactive real-time render-
ing of edited scenes and, in practice, slightly reducing resid-
ual blur.
4. EXPERIMENTS
4.1. Datasets
Since path tracing is performed in linear radiance space,
our method requires multi-view indoor captures with cali-
brated linear radiance. For real scenes, this is typically ob-
tained via exposure bracketing followed by HDR merging.
We primarily evaluate our method on real-world datasets.
Specifically, we use indoor scenes from FIPT [53] (F-) and
VR-NeRF Eyeful Tower [56] (E-). For F-, each scene con-
tains several hundred views at a resolution of 360 × 540,
with 1/8 of the views reserved for testing. For E-, each
scene includes thousands of views captured by a camera rig,
downsampled to 540 × 360. For completeness, we also in-
clude two synthetic scenes from [4], directly exported from
Blender (B-) with ground-truth relighting results obtained
by inserting a light ball. As existing real-world datasets do
not provide relighting ground truths, we capture an addi-
tional scene, LECTUREROOM, with controlled relighting for
comprehensive validation. Further details on data acquisi-
tion are provided in Appendix 8.
4.2. Implementation Details
Drawing inspiration from [11, 15, 35, 55], we implement
differentiable 2D Gaussian ray tracing with ray bouncing,
including Stage 0, Stage 1, and path tracing, from scratch
using PyTorch and OptiX [39]. This hybrid implementation
is significantly faster than a PyTorch-only baseline in prac-
tice. We fix the number of 2D Gaussians to reflect scene
complexity: 200k for B-, 500k for F- and LECTUREROOM,
and 1M for E-. In Stage 0, we reconstruct the radiant scene
using 30k iterations with λc = 1.0, λn = 0.5, λd = 0.05,
and λe = 0.1. In Stage 1, diffuse albedo is optimized for
400 iterations with nspp = 256.
For rendering, we use
nspp = 1024 for both single-bounce (1-bounce) and path
tracing to reduce noise, with a bounce limit of τB = 7
for path tracing. During light baking, path-traced results
from the test set are used to optimize per-Gaussian radiance
for 3k iterations. All experiments are conducted on a sin-
gle NVIDIA RTX 4090 GPU. The code will be released as
open-source to support reproducibility and further research.
4.3. Results
4.3.1. Results on Synthetic Scenes
We report quantitative comparisons of different render-
ing strategies before (ORIGINAL) and after (RELIGHTING)
inserting an illuminated ball on the synthetic scenes B-
KITCHEN and B-LIVINGROOM, for which relighting ground
truth is available (Table 1). We evaluate FLIP [1] on linear-
radiance images, and PSNR and LPIPS [66] on sRGB im-
ages obtained by converting linear radiance to sRGB and
clipping to [0, 1].
Qualitative comparisons are shown in
Fig. 4.
On the original scenes, radiant scene reconstruction
without light bounces (0-bounce) achieves the best per-
formance, as it directly recovers the radiance at capture
time.
Introducing additional bounces (1-bounce or path
tracing) degrades reconstruction accuracy. After scene edit-
6

<!-- page 7 -->
Figure 4. Relighting results with an inserted illuminated ball on synthetic scenes. Insets show FLIP error maps w.r.t. the relighting ground
truth. While 0-bounce and 1-bounce renderings fail to reproduce global illumination after editing, our path tracing reproduces the target
global illumination.
Table 1. Results on synthetic scenes before and after relighting. Path tracing achieves the highest relighting quality on synthetic scenes,
and light baking preserves this quality while greatly reducing render time.
Scene
B-KITCHEN
B-LIVINGROOM
Setting
ORIGINAL
RELIGHTING
ORIGINAL
RELIGHTING
Method
PSNR↑
LPIPS↓
FLIP↓
PSNR↑
LPIPS↓
FLIP↓
Time↓
PSNR↑
LPIPS↓
FLIP↓
PSNR↑
LPIPS↓
FLIP↓
Time↓
0-Bounce
37.57
0.0680
0.0732
16.22
0.1098
0.3593
0.015
37.83
0.0587
0.0646
11.59
0.1687
0.5147
0.013
1-Bounce
32.05
0.0733
0.1064
19.78
0.0896
0.3176
27.9
33.02
0.0724
0.0982
15.47
0.1243
0.4474
22.4
Ours (Path Tracing)
26.57
0.0829
0.1759
28.70
0.0825
0.1839
188
27.03
0.0829
0.1843
29.30
0.0983
0.2047
155
Ours (Light Baking)
-
-
-
28.89
0.0968
0.1854
0.015
-
-
-
29.39
0.1026
0.2040
0.013
ing, 0-bounce rendering fails to model physically correct
light transport and produces visually inconsistent results.
Incorporating a single bounce provides limited improve-
ment but remains insufficient to achieve global illumina-
tion.
Only path tracing with separation of emitters and
non-emitters and multiple light bounces, consistently yields
photo-realistic relighting results.
While path tracing produces the highest visual fidelity,
it incurs rendering costs on the order of hundreds of sec-
onds per frame. To address this limitation for content dis-
tribution and interactive applications, we re-bake the path-
traced results into 2D Gaussians and render using 0-bounce.
As shown in Table 1, this light baking strategy preserves
rendering quality while dramatically improving rendering
speed, enabling real-time navigation in edited scenes.
4.3.2. Results on Real Scenes
Path tracing results on the captured real-world scene LEC-
TUREROOM, compared against relighting ground truth, are
shown in Fig. 5.
Our method faithfully reconstructs the
scene under the fully illuminated condition and produces
visually consistent relighting results when half of the light
sources are turned off.
We further demonstrate a range
of scene editing operations on real scenes, including F-
CLASSROOM (Fig. 1) and Eyeful Tower scenes (Fig. 6).
These edits include modifying lighting and material proper-
ties, inserting illuminated balls, and importing non-emissive
objects. Across all scenarios, path tracing produces coher-
ent renderings with natural reflections, physically plausible
shadows, interreflections, and consistent global illumina-
tion. We encourage readers to refer to the supplementary
video for direct visual comparisons between counterintu-
itive 0-bounce rendering (radiant scene composition) and
physically accurate path tracing. In addition to static scene
edits, the supplementary video also presents dynamic edit-
ing results, including moving illuminated balls in F- scenes.
4.3.3. Comparison with FIPT
We compare our method with the state-of-the-art inverse
path tracing approach FIPT [53] on the real scenes F-
CLASSROOM and F-CONFERENCE. FIPT operates on re-
constructed meshes; we evaluate two mesh variants: (i) the
MonoSDF mesh [63] provided by the FIPT authors, trained
for approximately one day per scene, and (ii) a TSDF mesh
(resolution 512) reconstructed from multi-view depth maps
derived from our representation, which exhibits comparable
geometry while requiring only several minutes to generate.
The optimization times of the two methods are comparable
on an RTX 4090 (ours: 19+32=51 min and 17+33=50 min;
FIPT: 86 min and 61 min). Quantitative novel-view path
tracing results with nspp = 1024 are reported in Table 2,
with qualitative comparisons shown in Fig. 7.
EAG-PT consistently achieves higher rendering qual-
ity than FIPT. Mesh representations struggle to capture
fine geometric details such as chair legs and lamp arms
in real-world scenes, even at high resolution.
In con-
7

<!-- page 8 -->
Figure 5. Relighting results on the captured real scene LECTUREROOM. For each relighting condition that turns off half lights, our path
tracing closely reproduces the spatially varying indoor illumination compared with ground-truth relit photograph.
Figure 6. Path-traced results for various scene-editing operations on Eyeful Tower scenes. After editing, our EAG-PT yields plausible ren-
ders: imported non-emissive objects integrate naturally into the scene (a,e), inserted luminous balls cast consistent reflections and shadows
(b,d), and modified emitters or materials produce the expected changes in atmosphere (c,f). Continuous renderings and comparisons to the
counterintuitive 0-bounce (radiant scene composition) baseline are provided in the supplementary video.
8

<!-- page 9 -->
Figure 7. Path-traced novel views on real scenes compared with mesh-based FIPT. Zoomed regions highlight that our 2D Gaussians better
preserve thin structures and avoid mesh triangulation artifacts, yielding more detailed and natural renderings.
Table 2.
Our Gaussian-based path tracing achieves consistently
better novel-view quality than mesh-based FIPT on real scenes.
Scene
F-CLASSROOM
F-CONFERENCE
Method
PSNR↑
LPIPS↓
FLIP↓
PSNR↑
LPIPS↓
FLIP↓
0-Bounce
32.09
0.1432
0.1329
29.41
0.1809
0.1326
1-Bounce
30.37
0.1849
0.1658
28.57
0.1834
0.1490
FIPT (Exported)
23.57
0.3042
0.2801
21.54
0.3948
0.2856
FIPT (MonoSDF)
26.38
0.2031
0.2265
22.29
0.2729
0.2622
Ours (Path Tracing)
28.65
0.1998
0.2117
26.44
0.1960
0.2066
trast, 2D Gaussians naturally model such structures using
anisotropic primitives. Moreover, mesh-based path tracing
often exhibits visible triangulation artifacts at edges and
corners, whereas our Gaussian-based representation pro-
duces smoother and more realistic results. Moreover, our
method adopts 2D Gaussians as the unified representation,
which is much simpler than FIPT that combines triangle
mesh, voxel grid, MLP material, and image-based shading.
Except for the mesh that occupies around 150 MB storage
per scene, FIPT also uses around 500 MB to store recovered
material (and additional 23 GB storage for image-based
shading during training). While our method only produces
a single 33 MB ply file that stores 500k 2D Gaussians con-
taining all properties, which is only 5% compared to FIPT.
4.4. Ablations
Normal Supervision and Consistency
Correct light
bouncing in our method depends on good geometry (per-
pixel normal and distance). As shown in Fig. 8, normal su-
pervision and normal consistency together produce smooth
surfaces with the help of estimated normal maps. The nor-
mal supervision prevents extruding Gaussians, and normal
consistency avoids cavities in the scene (though better quan-
titative numbers are achieved without applying normal con-
sistency in Table 3).
Table 3. Ablation study on F-CLASSROOM comparing rendering
quality and speed across different configurations.
Method
PSNR↑
LPIPS↓
FLIP↓
Time↓
w/o normal supervision
28.29
0.2122
0.2158
160
w/o normal consistency
29.20
0.1952
0.1978
125
inaccurate emission mask
23.58
0.2162
0.3232
132
bounce limit 7 →3
24.94
0.2092
0.3277
55
bounce limit 7 →11
28.90
0.1989
0.2029
187
nspp 1024 →256
28.57
0.2283
0.2131
31
nspp 1024 →4096
28.67
0.1846
0.2113
496
lower Gaussian count
28.39
0.2187
0.2152
110
Full
28.65
0.1998
0.2117
123
Emission Masks
Emission masks play a significant role
in separating emitters and non-emitters. As shown in Ta-
ble 3 and Fig. 8, shrunken emission masks not only cause in-
accurate emitters reconstruction, but also damage the over-
all rendering quality.
Material Recovery
Results in Fig. 9 and Fig. 10 show the
effectiveness of material recovery during Stage 1. Based on
1-bounce and differentiable rendering, we can recover the
diffuse material well, comparing with albedo ground truths
on synthetic scene B-KITCHEN.
Another finding is that,
when using the same durations to optimize material, higher
nspps (nspp 256 iter 400) can yield better results than lower
nspps (nspp 64 iter 1600) due to lower sampling noise, which
is different from the strategy of I2SDF [70] that sets nspp to
16 and optimizes for 100k iterations over several days.
Path Tracing
We set the bounce limit to 7 and nspp to
1024 in our path-traced renderings to balance visual quality
and computation time. As shown in Fig. 8, using a smaller
bounce limit (e.g. 3) introduces significant bias and pro-
duces overly dark images, while reducing nspp (e.g. 256)
leads to loss of detail and noticeably blurrier results. In-
9

<!-- page 10 -->
Figure 8. Ablation study on F-CLASSROOM. The comparisons show that accurate normals, proper emission masks, sufficient bounce limit
and samples per pixel, and a denoiser are all necessary to avoid artifacts and to achieve our final high-quality path-traced result.
Figure 9. Albedo recovery on B-KITCHEN. Low samples per pixel make it difficult to accurately recover the ceiling albedo because of
high sampling noise.
3
4
5
6
7
8
22
24
26
28
Albedo PSNR
Optimization Duration (log2)
nspp 16
nspp 64
nspp 256
nspp 1024
Figure 10.
Albedo PSNR during material recovery on B-
KITCHEN: for a fixed optimization budget, higher nspp yields
higher PSNR, showing that reducing sampling noise is more ef-
fective than increasing iterations at low nspp.
creasing either the bounce limit or nspp further improves im-
age quality but also increases rendering time, as reported in
Table 3. In addition, we conduct an experiment demonstrat-
ing that path tracing with fewer 2D Gaussians can be faster;
details are provided in Appendix 9. Finally, unlike radi-
ant scene rendering, path-traced results require a denoiser
to obtain clean images, as illustrated in Fig. 8.
5. CONCLUSION
In this work, we propose Emission-Aware Gaussians and
Path Tracing (EAG-PT) to introduce correct light transport
into previous radiance field reconstruction work. The core
of our method lies in the separation of emitters and non-
emitters, differentiable rendering that recovers radiance and
material, and multi-bounce path tracing for final rendering.
Based on the unified representation, 2D Gaussians, that sup-
ports ray tracing and light bouncing, we formulate indoor
scene reconstruction and rendering into an integral frame-
work. This derives much more natural renders in recon-
structed indoor scenes after editing, and even better visual
quality than mesh-based baseline, which is promising for
practical and realistic real-to-sim reconstruction.
However, there are limitations in our method. The emis-
sion masks for real-world indoor scenes are complicated
and depend on labeling: An automatic way to obtain them
should reduce data-processing duration. Our current mate-
rial model assumes diffuse reflectance; extending the frame-
work to more expressive BRDFs would further enhance
realism.
Though we already implement path tracing on
GPU, the rendering speed is still unsatisfactory: combin-
ing with level-of-detail, multiple importance sampling, and
real-time global illumination techniques should improve the
efficiency during iterative scene editing. We leave address-
ing these limitations and exploring possible improvements
to future work.
10

<!-- page 11 -->
References
[1] Pontus Andersson, Jim Nilsson, Tomas Akenine-Möller,
Magnus Oskarsson, Kalle Åström, and Mark D. Fairchild.
Flip: A difference evaluator for alternating images. Proc.
ACM Comput. Graph. Interact. Tech., 3(2), 2020. 6
[2] Dejan Azinovic,
Tzu-Mao Li, Anton Kaplanyan, and
Matthias Nießner. Inverse path tracing for joint material and
lighting estimation. In 2019 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 2442–
2451, 2019. 3
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 3
[4] Benedikt
Bitterli.
Rendering
resources,
2016.
https://benedikt-bitterli.me/resources/. 6
[5] Brent Burley. Physically based shading at disney. In SIG-
GRAPH 2012 Course: Practical Physically Based Shading
in Film and Game Production, 2012. Course Notes. 15
[6] Krzysztof Byrski, Marcin Mazur, Jacek Tabor, Tadeusz
Dziarmaga, Marcin K ˛adziołka, Dawid Baran, and Prze-
mysław Spurek.
Raysplats: Ray tracing based gaussian
splatting, 2025. 3
[7] Adam Celarek, George Kopanas, George Drettakis, Michael
Wimmer, and Bernhard Kerbl. Does 3d gaussian splatting
need accurate volumetric rendering? In Computer Graphics
Forum, page e70032. Wiley Online Library, 2025. 3
[8] Chakravarty R. Alla Chaitanya,
Anton S. Kaplanyan,
Christoph Schied, Marco Salvi, Aaron Lefohn, Derek
Nowrouzezahrai, and Timo Aila. Interactive reconstruction
of monte carlo image sequences using a recurrent denoising
autoencoder. ACM Trans. Graph., 36(4), 2017. 6
[9] Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic,
Simon Green, Piotr Didyk, and Adrian Jarabo. Don’t splat
your gaussians: Volumetric ray-traced primitives for mod-
eling and rendering scattering and emissive media.
ACM
Trans. Graph., 44(1), 2025. 14
[10] Yuxin Dai, Qi Wang, Jingsen Zhu, Dianbing Xi, Yuchi Huo,
Chen Qian, and Ying He.
Inverse rendering using multi-
bounce path tracing and reservoir sampling. arXiv preprint
arXiv:2406.16360, 2024. 3
[11] eliphatfs.
Torchoptix.
https : / / github . com /
eliphatfs/torchoptix, 2024. Accessed: 2024-09-
13. 6
[12] Duan Gao, Guojun Chen, Yue Dong, Pieter Peers, Kun Xu,
and Xin Tong. Deferred neural lighting: free-viewpoint re-
lighting from unstructured photographs. ACM Trans. Graph.,
39(6), 2020. 15
[13] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Re-
alistic point cloud relighting with brdf decomposition and
ray tracing. In Computer Vision – ECCV 2024, pages 73–89,
Cham, 2025. Springer Nature Switzerland. 3
[14] Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi,
and Andrea Tagliasacchi. Radiant foam: Real-time differen-
tiable ray tracing, 2025. 3
[15] Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao, and Li
Zhang. Irgs: Inter-reflective gaussian splatting with 2d gaus-
sian ray tracing. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 10943–10952, 2025.
3, 4, 6
[16] Jon Hasselgren, Nikolai Hofmann, and Jacob Munkberg.
Shape, light, and material decomposition from images us-
ing monte carlo rendering and denoising.
In Proceedings
of the 36th International Conference on Neural Information
Processing Systems, Red Hook, NY, USA, 2022. Curran As-
sociates Inc. 3
[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In ACM SIGGRAPH 2024 Conference
Papers, New York, NY, USA, 2024. Association for Com-
puting Machinery. 3, 4, 5
[18] Zhening Huang, Xiaoyang Wu, Fangcheng Zhong, Heng-
shuang Zhao, Matthias Nießner, and Joan Lasenby.
Lite-
reality: Graphics-ready 3d scene reconstruction from rgb-d
scans. arXiv preprint arXiv:2507.02861, 2025. 14
[19] Wenzel Jakob, Sébastien Speierer, Nicolas Roussel, Merlin
Nimier-David, Delio Vicini, Tizian Zeltner, Baptiste Nicolet,
Miguel Crespo, Vincent Leroy, and Ziyi Zhang. Mitsuba 3
renderer, 2022. https://mitsuba-renderer.org. 3
[20] Jinseo Jeong, Junseo Koo, Qimeng Zhang, and Gunhee Kim.
Esr-nerf: Emissive source reconstruction using ldr multi-
view images. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4598–
4609, 2024. 3
[21] James T. Kajiya. The rendering equation. SIGGRAPH Com-
put. Graph., 20(4):143–150, 1986. 3, 5
[22] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe.
Poisson surface reconstruction. In Proceedings of the Fourth
Eurographics Symposium on Geometry Processing, page
61–70, Goslar, DEU, 2006. Eurographics Association. 3
[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4), 2023.
1, 2, 3, 5
[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 14
[25] Peter Kocsis, Lukas Höllein, and Matthias Nießner. Intrin-
sic image fusion for multi-view 3d material reconstruction.
arXiv preprint arXiv:2512.13157, 2025. 3
[26] Zhiyi Kuang, Yanchao Yang, Siyan Dong, Jiayue Ma,
Hongbo Fu, and Youyi Zheng. Olat gaussians for generic re-
lightable appearance acquisition. In SIGGRAPH Asia 2024
Conference Papers, New York, NY, USA, 2024. Association
for Computing Machinery. 15
[27] Zhen Li, Lingli Wang, Mofang Cheng, Cihui Pan, and Ji-
aqi Yang. Multi-view inverse rendering for large-scale real-
world indoor scenes. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
12499–12509, 2023. 2, 3, 5, 6
11

<!-- page 12 -->
[28] Chih-Hao Lin, Jia-Bin Huang, Zhengqin Li, Zhao Dong,
Christian Richardt, Tuotuo Li, Michael Zollhöfer, Johannes
Kopf, Shenlong Wang, and Changil Kim. Iris: Inverse ren-
dering of indoor scenes from low dynamic range images. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 465–474, 2025. 2, 3, 5, 14
[29] Jingwang Ling, Ruihan Yu, Feng Xu, Chun Du, and Shuang
Zhao. Nerf as a non-distant environment emitter in physics-
based inverse rendering.
In ACM SIGGRAPH 2024 Con-
ference Papers, New York, NY, USA, 2024. Association for
Computing Machinery. 3, 14, 16
[30] Agisoft LLC. Agisoft metashape (version 2.2) [software],
2025.
Available at https://www.agisoft.com/
downloads/installer/. 3
[31] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3, 5
[32] Li Ma, Vasu Agrawal, Haithem Turki, Changil Kim,
Chen Gao, Pedro Sander, Michael Zollhöfer, and Christian
Richardt. Specnerf: Gaussian directional encoding for spec-
ular reflections. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21188–
21198, 2024. 3
[33] Nadav Magar, Amir Hertz, Eric Tabellion, Yael Pritch, Alex
Rav-Acha, Ariel Shamir, and Yedid Hoshen.
Lightlab:
Controlling light sources in images with diffusion models.
In Proceedings of the Special Interest Group on Computer
Graphics and Interactive Techniques Conference Conference
Papers, New York, NY, USA, 2025. Association for Comput-
ing Machinery. 14
[34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
representing scenes as neural radiance fields for view synthe-
sis. Commun. ACM, 65(1):99–106, 2021. 1, 2, 3
[35] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Ric-
cardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja
Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray trac-
ing: Fast tracing of particle scenes. ACM Trans. Graph., 43
(6), 2024. 2, 3, 5, 6, 14, 16
[36] Thomas Müller, Fabrice Rousselle, Jan Novák, and Alexan-
der Keller. Real-time neural radiance caching for path trac-
ing. ACM Trans. Graph., 40(4), 2021. 5
[37] Thomas Müller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Trans. Graph., 41(4), 2022. 3
[38] Merlin Nimier-David, Zhao Dong, Wenzel Jakob, and Anton
Kaplanyan. Material and Lighting Reconstruction for Com-
plex Indoor Scenes with Texture-space Differentiable Ren-
dering. In Eurographics Symposium on Rendering - DL-only
Track. The Eurographics Association, 2021. 3
[39] Steven G. Parker, James Bigler, Andreas Dietrich, Heiko
Friedrich, Jared Hoberock, David Luebke, David McAllis-
ter, Morgan McGuire, Keith Morley, Austin Robison, and
Martin Stich. Optix: a general purpose ray tracing engine.
ACM Trans. Graph., 29(4), 2010. 6
[40] Weikun Peng, Sota Taira, Chris Careaga, and Ya˘gız Ak-
soy.
Interactive object insertion with differentiable ren-
dering.
In Proceedings of the Special Interest Group on
Computer Graphics and Interactive Techniques Conference
Posters, New York, NY, USA, 2025. Association for Com-
puting Machinery. 14
[41] Yohan Poirier-Ginter, Jeffrey Hu, Jean-François Lalonde,
and George Drettakis. Editable physically-based reflections
in raytraced gaussian radiance fields. In SIGGRAPH Asia
2025 - 18th ACM SIGGRAPH Conference and Exhibition
on Computer Graphics and Interactive Techniques in Asia,
Hong Kong, Hong Kong SAR China, 2025. 2, 3, 5, 6, 14, 15
[42] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 3
[43] Kerui Ren, Jiayang Bai, Linning Xu, Lihan Jiang, Jiangmiao
Pang, Mulin Yu, and Bo Dai. Mv-colight: Efficient object
compositing with consistent lighting and shadow generation.
arXiv preprint arXiv:2505.21483, 2025. 14
[44] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kun-
chang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen,
Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang,
Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam:
Assembling open-world models for diverse visual tasks,
2024. 16
[45] Johannes L. Schönberger, Enliang Zheng, Jan-Michael
Frahm, and Marc Pollefeys.
Pixelwise view selection for
unstructured multi-view stereo. In Computer Vision – ECCV
2016, pages 501–518, Cham, 2016. Springer International
Publishing. 3
[46] Johannes L. Schönberger and Jan-Michael Frahm. Structure-
from-motion revisited. In 2016 IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 4104–
4113, 2016. 3
[47] Dario Seyb, Peter-Pike Sloan, Ari Silvennoinen, Michał
Iwanicki, and Wojciech Jarosz. The design and evolution
of the uberbake light baking system. ACM Trans. Graph., 39
(4), 2020. 6
[48] SMPTE. High dynamic range electro-optical transfer func-
tion of mastering reference displays. SMPTE Standard ST
2084:2014, Society of Motion Picture and Television Engi-
neers, 2014. 5
[49] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The replica dataset: A digital
replica of indoor spaces. arXiv preprint arXiv:1906.05797,
2019. 3
[50] SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize,
Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang,
Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei
Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang,
Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari,
Matt Feiszli, and Jitendra Malik. Sam 3d: 3dfy anything in
images, 2025. 16
[51] Rafał Tobiasz, Grzegorz Wilczy´nski, Marcin Mazur, Sła-
womir Tadeja, and Przemysław Spurek. Meshsplats: Mesh-
12

<!-- page 13 -->
based rendering with gaussian splatting initialization, 2025.
3
[52] Gregory J. Ward, Francis M. Rubinstein, and Robert D.
Clear. A ray tracing solution for diffuse interreflection. In
Proceedings of the 15th Annual Conference on Computer
Graphics and Interactive Techniques, page 85–92, New
York, NY, USA, 1988. Association for Computing Machin-
ery. 5
[53] Liwen Wu, Rui Zhu, Mustafa B. Yaldiz, Yinhao Zhu, Hong
Cai, Janarbek Matai, Fatih Porikli, Tzu-Mao Li, Manmo-
han Chandraker, and Ravi Ramamoorthi. Factorized inverse
path tracing for efficient and accurate material-lighting es-
timation. In 2023 IEEE/CVF International Conference on
Computer Vision (ICCV), pages 3825–3835, 2023. 2, 3, 5, 6,
7, 15
[54] Xiuchao Wu, Jiamin Xu, Zihan Zhu, Hujun Bao, Qixing
Huang, James Tompkin, and Weiwei Xu. Scalable neural
indoor scene rendering. ACM Trans. Graph., 41(4), 2022. 3
[55] Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yu-
jun Shen, Sida Peng, Hujun Bao, and Xiaowei Zhou. En-
vgs: Modeling view-dependent appearance with environ-
ment gaussian, 2024. 2, 3, 4, 5, 6
[56] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia,
Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo
Porzi, Peter Kontschieder, Aljaž Božiˇc, Dahua Lin, Michael
Zollhöfer, and Christian Richardt. Vr-nerf: High-fidelity vir-
tualized walkable spaces. In SIGGRAPH Asia 2023 Con-
ference Papers, New York, NY, USA, 2023. Association for
Computing Machinery. 5, 6, 15
[57] Kai Yan, Fujun Luan, Miloš Hašan, Thibault Groueix,
Valentin Deschaintre, and Shuang Zhao. Psdr-room: Single
photo to scene using differentiable rendering. In SIGGRAPH
Asia 2023 Conference Papers, New York, NY, USA, 2023.
Association for Computing Machinery. 14
[58] Xijie Yang, Linning Xu, Lihan Jiang, Dahua Lin, and Bo
Dai. Virtualized 3d gaussians: Flexible cluster-based level-
of-detail system for real-time rendering of composed scenes.
In Proceedings of the Special Interest Group on Computer
Graphics and Interactive Techniques Conference Conference
Papers, New York, NY, USA, 2025. Association for Comput-
ing Machinery. 16
[59] Chongjie Ye,
Lingteng Qiu,
Xiaodong Gu,
Qi Zuo,
Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and
Xiaoguang Han. Stablenormal: Reducing diffusion variance
for stable and sharp normal.
ACM Trans. Graph., 43(6),
2024. 5
[60] Bohan Yu, Siqi Yang, Xuanning Cui, Siyan Dong, Baoquan
Chen, and Boxin Shi. Milo: Multi-bounce inverse rendering
for indoor scene with light-emitting objects. IEEE Transac-
tions on Pattern Analysis and Machine Intelligence, 45(8):
10129–10142, 2023. 3, 6, 15
[61] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xian-
gli, and Bo Dai. Gsdf: 3dgs meets sdf for improved neural
rendering and reconstruction. In Advances in Neural Infor-
mation Processing Systems, pages 129507–129530. Curran
Associates, Inc., 2024. 3
[62] Yizhou Yu, Paul Debevec, Jitendra Malik, and Tim Hawkins.
Inverse global illumination:
recovering reflectance mod-
els of real scenes from photographs.
In Proceedings of
the 26th Annual Conference on Computer Graphics and
Interactive Techniques, page 215–224, USA, 1999. ACM
Press/Addison-Wesley Publishing Co. 3
[63] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sat-
tler, and Andreas Geiger. Monosdf: exploring monocular
geometric cues for neural implicit surface reconstruction. In
Proceedings of the 36th International Conference on Neural
Information Processing Systems, Red Hook, NY, USA, 2022.
Curran Associates Inc. 3, 7
[64] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Trans. Graph., 43(6), 2024. 3
[65] Edward Zhang, Michael F. Cohen, and Brian Curless. Emp-
tying, refurnishing, and relighting indoor spaces.
ACM
Trans. Graph., 35(6), 2016. 3
[66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6
[67] Wenyuan Zhang, Jimin Tang, Weiqi Zhang, Yi Fang, Yu-
Shen Liu, and Zhizhong Han.
Materialrefgs: Reflective
gaussian splatting with multi-view consistent material infer-
ence. arXiv preprint arXiv:2510.11387, 2025. 3
[68] Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul
Debevec, William T. Freeman, and Jonathan T. Barron. Ner-
factor: neural factorization of shape and reflectance under an
unknown illumination. ACM Trans. Graph., 40(6), 2021. 3
[69] Yang Zhou, Songyin Wu, and Ling-Qi Yan. Unified gaussian
primitives for scene representation and rendering, 2024. 2,
3, 14
[70] Jingsen Zhu, Yuchi Huo, Qi Ye, Fujun Luan, Jifan Li, Dian-
bing Xi, Lisha Wang, Rui Tang, Wei Hua, Hujun Bao, and
Rui Wang. I2-sdf: Intrinsic indoor scene reconstruction and
editing via raytracing in neural sdfs.
In 2023 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 12489–12498, 2023. 2, 3, 5, 6, 9
[71] Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li,
Tianzhu Zhang, Jiangmiao Pang, and Bo Dai.
Objectgs:
Object-aware scene reconstruction and scene understanding
via gaussian splatting, 2025. 16
[72] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and
Markus Gross. Ewa volume splatting. In Proceedings of the
Conference on Visualization ’01, page 29–36, USA, 2001.
IEEE Computer Society. 3
13

<!-- page 14 -->
EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene
Reconstruction and Editing
Supplementary Material
The appendices provide supplementary material to sup-
port and extend the main paper. Additional related work
is discussed in Appendix 6. Details on deriving emission
masks are presented in Appendix 7.
The image capture
pipeline for acquiring linear radiance is described in Ap-
pendix 8. Further analysis of our method is provided in
Appendix 9, and potential downstream applications are il-
lustrated in Appendix 10.
6. Additional Related Work
Scene Editing in 3DGRT. Methods such as 3DGRT [35] can
accomplish scene editing and re-rendering to some extent.
It is able to insert reflective mesh objects into the scene and
achieve realistic renders. Yet, the reflection only happens on
the newly inserted mesh objects, which is a one-way bounce
between the object and scene, instead of bouncing around
inside the reconstructed scene. The way 3DGRT creates
shadows is to dim the color of the corresponding radiant
3D Gaussians, similar to the harmonization method used in
MV-CoLight [43], which is a heuristic approximation in-
stead of a physically-correct calculation.
Path Tracing in EGR. Though EGR [41] also uses the
phrase path tracing in their paper, they ignore true light
sources in the scene and only refer to the multi-bounce
light transport. The main target of EGR is to reconstruct
the radiance field for reflection (like environment model-
ing in PBIR-NeRF [29]), and they do not optimize material
through inverse rendering. While our main target is to re-
cover the SVBRDF properties, and to support editing and
re-rendering of the whole scene.
Volumetric Path Tracing. There are some works using
volumetric path tracing based on 3D Gaussians. DSurG [9]
focuses on modeling radiant semi-transparent media, which
is not related with our goal. We only use the volumetric
representation for the scene, while still use the general path
tracing to bounce light when "hitting" the surface. UGP [69]
tends to use a modified 3D Gaussians as a general scene
representation. However, such representation is rather com-
plicated for indoor scene reconstruction, and it is mainly de-
signed for appearance modeling and forward rendering; the
exploration of inverse rendering on such new presentation
is still preliminary.
Single Image Relighting. We are happy to see lots of 2D
relighting methods [33] that achieve photo-realistic scene
editing results. However, it is difficult for these methods to
achieve 3D consistency results. For example, changing the
light in one image does not affect another image in the same
Figure 11.
Exceptions of emission masks.
(a) Reflection is
brighter than emitter. (b) Human-defined emitters. (c) Emitter
with reflection.
scene. [40, 57] start from single image yet adopt explicit
3D representation to solve 3D consistency issue, though a
single image only has small scene coverage. This does not
satisfy interactive roaming in 3D scene.
Asset Retrieval. Other systems [18, 57] retrieve and ad-
just CG assets from input images, but the resulting scenes
are typically less photo-realistic than the captured data.
7. Emission Mask Derivation
Our method relies on 2D emission masks to separate emit-
ters from non-emitters. For images in linear radiance, emis-
sion masks can typically be obtained by simple threshold-
ing, since emitters exhibit high radiance. For most scenes,
including B-, F-, and LECTUREROOM, we classify a pixel as
emissive if its radiance exceeds a scene-dependent threshold
τR ∈{1.0, 1.5, 2.0}. This method is similar to IRIS [28],
which applies a threshold of 0.99 to SDR images.
However, thresholding alone is insufficient in several
cases. First, strong reflections can be brighter than the ac-
tual light sources, causing reflective surfaces to be misla-
beled as emitters, while genuinely dimmer emitters may fall
below the threshold, as illustrated in Fig. 11 (a). Second,
emission may be defined semantically: for example, bot-
tles inside the vending machine can be treated as emitters
when they are unimportant for subsequent scene editing, as
shown in Fig. 11 (b). To handle such cases, we manually re-
fine emission masks with SAM [24] for efficient annotation.
The final emission mask is the union of the thresholded and
manually labeled regions, M = Mthreshold ∪MSAM.
Besides, the use of emission masks assumes, as in
Sec. 3.3, that emitters do not reflect light.
Real indoor
scenes can violate this assumption: for instance, the glass
door in E-OFFICE1B should be treated as an emitter but is
14

<!-- page 15 -->
also highly reflective, as shown in Fig. 11 (c). Currently,
our method does not explicitly model such mixed emitter-
reflector materials.
8. Real-World Scene Capture
Our method operates on calibrated multi-view images in
linear radiance space.
Following VR-NeRF [56] and
FIPT [53], we capture the indoor scene LECTUREROOM
with an APS-C camera (Sony ZV-E10M2) mounted on a tri-
pod. The camera is operated in full manual mode with aper-
ture fixed to f/8.0, ISO to 100, and focal length to 16 mm
to obtain a wide field of view and a stable radiometric re-
sponse. Before capture, we set the white balance using a
gray card placed in the scene and then fix it for the en-
tire sequence to ensure consistent color calibration across
all views. The lens is switched to manual focus with the
focus distance set to 1.0 m, and kept unchanged during cap-
ture to avoid per-view focus variations. To avoid clipping in
shadows and highlights, we determine a bracketing range of
exposure times in the scene such that the shortest exposure
resolves the brightest emitters and the longest exposure re-
solves the darkest shadows. We then interpolate within this
range to acquire multiple exposures per viewpoint. We ro-
tate the camera, adjust the tripod height, and translate the
tripod to obtain dense multi-view coverage of the scene.
After capture, for each viewpoint, the captured RAW im-
ages are converted to linear radiance in the range [0, 1] and
merged into a single linear image. We then apply lens undis-
tortion and vignetting correction to every image. We run
COLMAP on sRGB images converted from linear images
to recover intrinsics and extrinsics. The recovered poses are
subsequently rotated so that the scene floor aligns with the
+z axis of the world coordinate system, which simplifies
downstream editing and rendering. This pipeline is suffi-
cient to produce input data for our reconstruction method
from generic indoor scenes.
However, capturing a scene under a single lighting con-
dition is insufficient for evaluating real-scene relighting per-
formance. While one-light-at-a-time (OLAT) datasets have
been studied for real-world objects [12, 26], there are very
few real-world indoor multi-view datasets with multiple
controllable light configurations.
FIPT [53] shows sev-
eral relit images in their paper but does not release the full
dataset. We regard multi-condition, multi-view indoor data
as an important resource for physically based indoor recon-
struction and relighting research.
As an initial step, we
capture LECTUREROOM under three distinct lighting con-
figurations: all lights on, only the front-half lights on, and
only the back-half lights on. At each camera position, we
keep the camera rigidly fixed and switch the scene among
these three light conditions; for each condition, we record a
multi-exposure bracket as described above. For LECTURE-
ROOM, we capture 100 distinct camera positions, each with
3 lighting conditions. We show representative ground-truth
images in Fig. 5, together with path-traced renderings pro-
duced by our method, for evaluating the relighting capabil-
ity of our method on real scene. We will release this multi-
condition LECTUREROOM dataset to the research commu-
nity.
Looking forward, extending this capture process to a
broader variety of indoor environments and to a richer set
of lighting conditions would provide valuable benchmarks
for disentangling geometry, materials, and illumination, and
for advancing inverse rendering and indoor scene relighting
methods.
9. Discussion
Modeling
In our current formulation, Gaussians are at-
tached with diffuse albedos. While this already yields high-
quality reconstructions and renderings, it still falls short of
the richness of real-world materials. For example, highly
specular objects such as the metallic range hood and mi-
crowave oven in Fig. 10 are not faithfully reproduced. In
real scenes (e.g., Fig. 2 and Fig. 5), reflections on the white-
board and ceiling are also difficult to capture. Extending
EAG-PT with a compact parametric model, such as a sim-
plified Disney BRDF [5], should further improve realism
and enable a broader range of appearance effects.
Moreover, the indoor scenes considered in this work
are predominantly confined and dominated by artificial il-
lumination, whereas real indoor environments often re-
ceive significant external lighting, as in E-OFFICEVIEW1,
E-OFFICEVIEW2, and E-RIVERVIEW.
Incorporating an
environment map with explicit window modeling, as in
MILO [60], is a promising direction to extend EAG-PT to
more diverse and open indoor configurations.
Insufficient Capture
EAG-PT assumes multi-view cap-
ture that observes most of the scene, as described in Sec. 3.
In practice, however, it is impossible to sample all points in
3D space. Occluded or rarely seen regions can acquire an
incorrect radiance cache in Stage 0, which in turn degrades
albedo recovery in Stage 1. As presented in Fig. 3, albedo of
the ceiling above the projector appears darker than expected
due to insufficient observations. Integrating radiance regu-
larization and optimization strategies as in [41, 53], or more
generally coupling EAG-PT with priors for unobserved re-
gions, could mitigate such artifacts and improve robustness
under incomplete coverage.
Path Tracing
While our path tracer produces photo-
realistic results after editing, the current rendering speed
remains far from real-time. The two dominant bottlenecks
are repeated ray–Gaussian intersections and the large num-
ber of required samples.
15

<!-- page 16 -->
Figure 12. Ray-Gaussian intersection count visualization on two
versions of reconstructed F-CLASSROOMs with different Gaussian
count.
We re-trained a variant F-CLASSROOM with fewer 2D
Gaussians (200k instead of 500k), which reduced the av-
erage intersection count per ray from 79 to 63 (visualized
in Fig. 12) and improved rendering speed by approximately
11%. Instead of retraining separate checkpoints, applying a
level-of-detail hierarchy to the original reconstruction [58]
is an attractive alternative for accelerating rendering. In ad-
dition, more compact Gaussian primitives [35] could further
reduce intersection counts and improve performance.
On the sampling side, our current implementation only
uses cosine-weighted sampling for secondary rays.
Al-
though this reduces noise compared to uniform sampling,
it is still insufficient to obtain clean images at low sample
counts. Adopting multiple importance sampling and direct
light sampling targeted at emissive Gaussian primitives, as
in PBIR-NeRF [29], should substantially reduce variance
and enable fewer samples per pixel for smoother path trac-
ing during scene editing.
Instance-Level Reconstruction
Most prior work on in-
door inverse rendering, including ours, reconstructs the en-
tire scene as a single undifferentiated instance, without ex-
plicit object- or semantic-level structure. Consequently, we
currently rely on box selection of Gaussians for editing,
which often leaves extruding Gaussians at object bound-
aries, as in Fig. 4. Similar problems arise when removing
objects: newly exposed regions correspond to previously
unseen surfaces, degrading realism and geometric consis-
tency.
Incorporating instance-level segmentation [44, 71] into
EAG-PT would enable object-aware reconstruction and
editing, reducing boundary artifacts and simplifying user
interaction. Coupling such segmentation with generative
3D completion models [50] is another promising avenue
to plausibly hallucinate newly visible regions and improve
both reconstruction quality and editing flexibility in com-
plex indoor scenes.
10. Possible Applications
While our experiments focus on reconstruction and editing
quality, we briefly outline two downstream applications that
could benefit from EAG-PT. These use cases are illustrative
only; we do not conduct task-level evaluations.
Interior design and virtual prototyping
A homeowner
first captures their existing indoor space, and the captured
data are provided to an interior designer. The designer re-
constructs the scene with EAG-PT and iteratively explores
multiple editing proposals by modifying furniture layout,
materials, and lighting. Because EAG-PT preserves realis-
tic, physically consistent illumination after editing, the de-
signer can then light-bake the edited scenes to obtain 3D
assets that can be previewed in real time (e.g., on a desktop
viewer or XR device), enabling the client to compare design
alternatives before committing to a final furnishing plan.
Embodied AI post-training
Prior to deploying a robot in
a specific real indoor environment, we reconstruct that en-
vironment with EAG-PT and generate a family of edited
variants that simulate plausible changes in layout, object
placement, and lighting conditions. Training or fine-tuning
the robot’s perception and control policies in these photo-
realistic, emission-aware variants can reduce the sim-to-
real gap, by exposing the policy to realistic illumination
effects (shadows, interreflections, bright emissive sources)
and moderate structural changes in advance. This may pro-
vide improved robustness to lighting and layout changes en-
countered during deployment.
16
