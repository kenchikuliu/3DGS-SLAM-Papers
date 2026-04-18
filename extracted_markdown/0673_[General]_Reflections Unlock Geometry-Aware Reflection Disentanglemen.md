<!-- page 1 -->
Reflections Unlock: Geometry-Aware Reflection
Disentanglement in 3D Gaussian Splatting for Photorealistic
Scenes Rendering
Jiayi Song1†, Zihan Ye1†, Qingyuan Zhou1†, Weidong Yang1, Ben Fei2*,
Jingyi Xu1, Ying He3, Wanli Ouyang2
1School of Computer Science, Fudan University, Shanghai, China.
2Department of Information Engineering, The Chinese University of Hong Kong, Hong
Kong, China.
3College of Computing and Data Science, Nanyang Technological University, Singapore.
*Corresponding author(s). E-mail(s): benfei@cuhk.edu.hk;
Contributing authors: 22307130359@m.fudan.edu.cn; 22300240027@m.fudan.edu.cn;
zhouqy23@m.fudan.edu.cn; wdyang@fudan.edu.cn; jy22@m.fudan.edu.cn;
yhe@ntu.edu.sg; wlouyang@ie.cuhk.edu.hk;
†These authors contributed equally to this work.
Abstract
Accurately rendering scenes with reflective surfaces remains a significant challenge in novel view
synthesis, as existing methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting
(3DGS) often misinterpret reflections as physical geometry, resulting in degraded reconstructions.
Previous methods rely on incomplete and non-generalizable geometric constraints, leading to mis-
alignment between the positions of Gaussian splats and the actual scene geometry. When dealing
with real-world scenes containing complex geometry, the accumulation of Gaussians further exac-
erbates surface artifacts and results in blurred reconstructions. To address these limitations, in
this work, we propose Ref-Unlock, a novel geometry-aware reflection modeling framework based
on 3D Gaussian Splatting, which explicitly disentangles transmitted and reflected components to
better capture complex reflections and enhance geometric consistency in real-world scenes. Our
approach employs a dual-branch representation with high-order spherical harmonics to capture high-
frequency reflective details, alongside a reflection removal module providing pseudo reflection-free
supervision to guide clean decomposition. Additionally, we incorporate pseudo-depth maps and a
geometry-aware bilateral smoothness constraint to enhance 3D geometric consistency and stability
in decomposition. Extensive experiments demonstrate that Ref-Unlock significantly outperforms clas-
sical GS-based reflection methods and achieves competitive results with NeRF-based models, while
enabling flexible vision foundation models (VFMs) driven reflection editing. Our method thus offers
an efficient and generalizable solution for realistic rendering of reflective scenes. Our code is available
at https://ref-unlock.github.io/.
Keywords: Novel view synthesis, real-time rendering, reflection modeling, 3D Gaussian Splatting.
1
arXiv:2507.06103v1  [cs.CV]  8 Jul 2025

<!-- page 2 -->
1 Introduction
Novel view synthesis from multi-view images has
emerged as a pivotal research direction in com-
puter vision and graphics, driven by its wide
applications in virtual reality (Wang et al, 2024),
augmented reality (Zuo et al, 2025), and 3D
reconstruction (Liu et al, 2025). The advent
of Neural Radiance Fields (NeRF)(Reiser et al,
2021) has marked a significant breakthrough in
scene representation by enabling photorealistic
image synthesis with view-dependent effects. This
approach models scenes implicitly using a multi-
layer perceptron that translates 3D positions and
camera directions into color and density, which
are then composited using volumetric rendering
techniques(Jain et al, 2024). Although NeRF-
based methods (Deng et al, 2022; Barron et al,
2021; Pumarola et al, 2021) achieves impressive
visual fidelity, its practical use is limited due to
high computational demands and the extensive
time required for model optimization. In con-
trast, 3D Gaussian Splatting (3DGS) (Kerbl et al,
2023) offers an explicit scene representation using
anisotropic 3D Gaussians. It initializes Gaus-
sians from a Structure-from-Motion (SfM) (Schon-
berger and Frahm, 2016) point cloud and projects
them onto the 2D image plane during rendering,
with colors computed from Spherical Harmonics
(SH) parameters and images synthesized via dif-
ferentiable rasterization. By leveraging fast train-
ing and rendering strategies, 3DGS achieves real-
time performance and delivers rendering results
comparable to those of NeRF.
However, NeRF and 3DGS both struggle
to faithfully reconstruct environments containing
reflective materials—such as mirrors, transpar-
ent glass, or display screens (Fei et al, 2024;
Zhou et al, 2025). This challenge stems from
their lack of explicit mechanisms to model strong
specular reflections, which hampers their ability
to handle such surfaces effectively. As a result,
such approaches frequently mistake reflections
for actual scene geometry, leading to significant
degradation in rendering accuracy and visual
coherence. Furthermore, since reflections change
with the viewpoint, this misrepresentation leads
to inconsistencies across multiple views (Yao et al,
2024). These inconsistencies, in turn, introduce
conflicts during optimization, ultimately resulting
Fig. 1 Novel view synthesis with Ref-Unlock, ful-
filling better reflections modeling than other meth-
ods. The top row compares rendering quality of various
methods (GS-IR (Liang et al, 2024), 2DGS (Huang et al,
2024), GShader (Jiang et al, 2024), 3DGS (Kerbl et al,
2023), Ours) against the ground truth (GT), with PSNR
scores provided for quantitative evaluation. The bottom
row shows the VFMs-driven reflection editing with varying
reflection strength scales using our method.
in blurred and degraded reconstructions (Verbin
et al, 2022).
To address this issue, NeRF-based meth-
ods like Mirror-NeRF (Zeng et al, 2023) and
TraM-NeRF (Holland et al, 2023) are specifically
devised for mirror reflections. In contrast, methods
like Ref-NeRF (Verbin et al, 2022) and NeR-
FReN (Guo et al, 2022) extend the NeRF frame-
work to handle more diverse types of reflections,
including those on non-ideal surfaces. Although
notable progress has been made, these approaches
continue to suffer from the persistent issue of
time-consuming optimization, which remains an
open problem (Lin et al, 2024). To overcome this,
GS-based methods have emerged as a promising
alternative. Among these, Physically Based Ren-
dering (PBR) methods, including Bidirectional
Reflectance Distribution Function (BRDF)-based
approaches, focus on modeling reflections based
on physical principles, such as how light interacts
with surface geometry and material properties.
Some approaches enhance appearance modeling
through PBR-based shading and color decompo-
sition (Jiang et al, 2024), while others integrate
normals, materials, and lighting to enable realis-
tic relighting (Gao et al, 2023; Liang et al, 2024).
Additionally, advanced shading strategies such
2

<!-- page 3 -->
as deferred rendering and reflection-guided nor-
mal optimization have been proposed to improve
specular realism (Ye et al, 2024b).
While
recent
GS-based
PBR
methods
achieve
impressive
results,
their
geometric
constraints—particularly regarding depth esti-
mation—lack applicability across various scenes.
This often leads to misalignment between the
positions of Gaussian primitives and the underly-
ing scene geometry (Huang et al, 2024). In scenes
with complex geometry, such as locally intricate
surfaces—such as locally intricate surfaces like
the curtains reflected in the mirror (Fig. 1),
the accumulation of misaligned Gaussians fur-
ther amplifies surface artifacts and results in
noticeable blurring. As shown in Fig. 1, existing
methods such as GS-IR (Liang et al, 2024) and
GShader (Jiang et al, 2024) yield suboptimal
performance compared to 3DGS (Kerbl et al,
2023) on the RFFR dataset (Guo et al, 2022).
Similarly, models like R3DG (Gao et al, 2023)
and 3DGS-DR (Ye et al, 2024b) also encounter
difficulties in rendering these scenes consistently.
These methods typically perform well on curated
datasets but exhibit limited applicability to more
diverse and real-world environments that contain
complex and fine-grained geometric structures.
To enhance generalization, some approaches (Ye
et al, 2024b; Yao et al, 2024) require case-specific
parameter tuning, while others (Liang et al, 2024;
Gao et al, 2023) adopt staged training strate-
gies—typically stabilizing geometry first before
refining material and lighting properties.
In
contrast,
SH-only
methods
like
3DGS
(Kerbl
et
al,
2023)
typically
exhibit
stronger robustness and generalization due to
simplified color modeling. By capturing spherical
variations in color and lighting, these methods
reduce the reliance on scene-specific properties
such as surface normals or lighting distributions.
This simplicity allows them to operate effec-
tively across diverse scenes without extensive
parameter tuning or staged training. Moreover,
the SH representation naturally aligns with the
“massive small particles and rapid blending”
strategy of Gaussian Splatting (Kerbl et al, 2023),
enabling efficient local approximations with low
computational
overhead
(Sloan
et
al,
2002).
To address the challenge of rendering reflective
scenes, we propose enhancements within the
SH-based shading framework, which offers better
generalization than complex PBR methods. Our
objective is to balance generalization with the
physically detailed color modeling compared with
PBR-based approaches.
In
this
study,
we
propose
Ref-Unlock,
a
geometry-aware
disentanglement
framework
designed to improve the fidelity of 3DGS by
explicitly modeling and separating physical reflec-
tions from true scene geometry. Specifically, our
approach separates the scene representation into
two components: one modeling transmitted light
and the other capturing reflected light, with
each branch having dedicated color and opacity
attributes. To explicitly quantify reflection inten-
sity, we introduce a reflection map. To better
capture high-frequency view-dependent effects,
we employ the high-degree spherical harmonics
for both reflected and transmitted color compo-
nents, enabling finer modeling of reflective details.
However, unsupervised scene decomposition may
lead to incomplete separation. To mitigate this,
we integrate a Reflection Removal Module (RRM)
based on DSRNet (Hu and Guo, 2023), gener-
ating pseudo reflection-free images that guide
the learning of the transmitted component and
promote the disentanglement between reflection
and transmission. In addition, since reflections
are inherently view-dependent and closely tied to
geometry, we employ Depth Anything v2 (Yang
et al, 2024) to provide pseudo-depth maps as
additional supervision for the depth parame-
ter. To further regularize the geometry and the
decomposition process, we introduce a geometry-
aware bilateral smoothness constraint (Guo et al,
2022), which jointly considers both depth varia-
tions and color differences. This ensures coherent
depth prediction and more accurate reflection-
transmission separation. Through the combined
design of reflection-specific modeling, reflection-
guided
disentanglement,
and
geometry-aware
regularization, Ref-Unlock achieves efficient and
faithful reconstruction of reflective scenes without
the need for additional Gaussians. Moreover, the
explicit reflection representation enables flexible
editing operations such as reflection manipulation
shown in Fig. 1.
Our contributions can be concluded as follows:
• We propose a dual-branch representation that
separates reflected and transmitted compo-
nents, together with higher-degree spherical
3

<!-- page 4 -->
harmonics for both branches, enabling more
accurate modeling of high-frequency reflections.
• We integrate a reflection removal module to pro-
vide pseudo reflection-free supervision, facilitat-
ing the disentanglement between reflection and
transmission without requiring manual masks.
• We leverage pseudo-depth supervision from
Depth Anything v2 and a bilateral smooth-
ness constraint that jointly regularizes depth
and color, improving geometric consistency and
enhancing the stability of scene decomposition.
• Our method achieves faithful rendering of reflec-
tive scenes while maintaining high general-
ization to diverse environments, and enables
additional scene reflection editing capabilities
through the explicitly modeled reflection.
2 Related Work
2.1 Neural Scene Representation for
Novel View Synthesis
Novel View Synthesis (NVS) aims to generate
realistic images from unseen viewpoints by uti-
lizing multiple calibrated photos that encode the
scene’s 3D layout and visual details (Wang et al,
2023; Chan et al, 2023). NeRF (Mildenhall et al,
2021) represents a significant advance by employ-
ing neural networks to predict color and density
values at sampled 3D locations along camera rays.
These predictions are then combined through vol-
ume rendering to produce novel views. However, it
requires extensive sampling and numerous neural
network evaluations, which make the process com-
putationally heavy and slow, limiting its practical-
ity for real-time applications (Xu et al, 2022). This
challenge has led to the exploration of point-based
rendering methods, which offer a better balance
between rendering quality and computational effi-
ciency (Kopanas et al, 2021; R¨uckert et al, 2022).
One noteworthy method that has attracted con-
siderable interest is 3DGS (Kerbl et al, 2023; Fei
et al, 2024), which models scenes using anisotropic
Gaussian functions in 3D space. Each Gaussian
element encodes opacity and utilizes Spherical
Harmonic coefficients to capture view-dependent
color variations. Unlike traditional ray tracing
approaches, 3DGS leverages a point-based raster-
ization pipeline accelerated by modern hardware,
enabling both rapid and visually detailed render-
ing.
While NeRF and 3DGS represent two dis-
tinct
paradigms
of
neural
scene
representa-
tion—implicit
and
explicit,
respectively—they
both share the goal of modeling scenes with
high fidelity and view-consistency. However, both
of them face limitations when applied to com-
plex scenes involving diverse material properties,
intricate geometry, occlusions, or other challeng-
ing visual phenomena (Gao et al, 2022; Zhu
et al, 2024; Fei et al, 2024). This has moti-
vated a growing number of studies focused on
extending these methods to improve rendering
quality and physical realism in challenging sce-
narios. For instance, Mip-NeRF (Barron et al,
2021) improves NeRF by representing scenes at
multiple continuous scales, reducing aliasing and
enhancing detail while achieving faster rendering.
NeRFusion (Zhang et al, 2022) integrates NeRF
with TSDF-based fusion to enable efficient large-
scale reconstruction and photo-realistic rendering,
addressing the scalability limitations of standard
NeRF in complex indoor scenes. SuGaR (Gu´edon
and Lepetit, 2024) imposes regularizations to
keep Gaussian points aligned with the surface,
while GSDF (Yu et al, 2024) and Gaussian-
Rooms (Xiang et al, 2024) integrate 3DGS with
implicit SDF fields for mutual supervision. Among
these efforts, one particularly important direction
involves addressing novel view synthesis in scenes
with reflective surfaces, where view-dependent
appearance varies dramatically due to mirror-like
effects, screen contents, or transparent materials.
2.2 NeRF-Based Approaches for
Reflective Scene Rendering
Reflection effects present significant challenges in
neural scene representations due to their strong
view-dependent
properties
and
complex
light
interactions. Recent studies have extended NeRF
to better capture reflection phenomena, aiming
to enhance rendering fidelity in scenes containing
reflective surfaces. As a pioneering work, Ref-
NeRF (Verbin et al, 2022) effectively decomposes
light into diffuse and specular components by
leveraging a reflection-dependent radiance field,
enabling accurate modeling of reflections based on
the viewing direction. NeRFReN (Guo et al, 2022)
divides the scene into several subregions, allowing
for distinct treatment of various reflection types
4

<!-- page 5 -->
such as diffuse and specular reflections. This sub-
division approach enhances the model’s efficiency
in rendering intricate reflections and improves its
capability to represent a wide range of reflective
effects. SpecNeRF (Ma et al, 2023) proposes a
learnable Gaussian-based directional encoding to
better capture view-dependent variations, espe-
cially in regions near light sources. Following these
developments, NeRRF (Chen et al, 2023) incorpo-
rates the reflection equation and Snell’s law into
NeRF’s ray tracing, unifying refraction and reflec-
tion via the Fresnel term. NeRF-Casting (Verbin
et al, 2024) replaces MLP queries with reflec-
tion rays cast into NeRF geometry, sampling and
decoding features via a small MLP to ensure
consistent reflections and simplify view-dependent
representations. Despite these improvements, the
consistently high rendering time remains a major
barrier to the widespread adoption of NeRF-based
techniques. To address this challenge of reflec-
tion modeling in NeRF, in this paper, we propose
Ref-Unlock for the modeling of reflective scenes
within the 3DGS framework based on reflection
decoupling.
2.3 Reflection Modeling in 3D
Gaussian Splatting
Leveraging
the
efficient
rendering
and
high-
quality output of 3DGS, recent work has investi-
gated reflection modeling using the 3DGS frame-
work (Jiang et al, 2024; Gao et al, 2023; Malarz
et al, 2023; Liu et al, 2024). Some approaches (Liu
et al, 2024; Meng et al, 2024) have been developed
to particularly tackle reflections from mirror sur-
faces MirrorGaussian (Liu et al, 2024) leverages
mirror symmetry in point-cloud representations
for efficient real-time rendering of mirrored envi-
ronments, whereas Mirror-3DGS (Meng et al,
2024) utilizes a two-step training strategy to iden-
tify mirror-related components and estimate mir-
ror geometry and camera poses. Although these
methods perform well in mirrored scenes, they
rely on specifically crafted mirror masks, which
limits their generality and practicality in diverse
real-world scenarios. In contrast, some methods
based on physically based rendering (PBR) (Gao
et al, 2023; Liang et al, 2024; Ye et al, 2024b) are
capable of handling more general reflective scenes
across a wider range of conditions. GShader (Jiang
et al, 2024) applies simplified BRDF shading
functions on 3D Gaussians to enhance neural ren-
dering in scenes with reflective surfaces, focusing
on decomposing color attributes into components
such as diffuse, specular, and residual color for
modeling reflective surfaces. R3DG (Gao et al,
2023) extends 3DGS with embedded normals,
BRDF parameters, and incident lighting, and
uses point-based ray tracing with bounding vol-
ume hierarchies (BVH) for efficient visibility han-
dling, enabling realistic relighting. GS-IR (Liang
et al, 2024) extends 3DGS with staged optimiza-
tion to jointly recover geometry, material, and
lighting under unknown illumination, incorporat-
ing normal estimation and occlusion modeling
within a full PBR pipeline. 3DGS-DR (Ye et al,
2024b) leverages deferred shading with per-pixel
reflection gradients to optimize surface normals
during 3DGS for improved specular reflection
rendering. GeoSplating (Ye et al, 2024a) com-
bines 3DGS with explicit geometry for inverse
rendering, enabling accurate normal estimation
and differentiable physically based shading, but
it only handles single-bounce specular lighting,
with inter-reflections and baked shadows causing
relighting inaccuracies. However, it is worth noting
that these designs are often optimized for particu-
lar datasets or illumination setups, which restricts
their adaptability to a broader range of real-world
scenes.
Therefore,
our
Ref-Unlock
aims
to
han-
dle diverse reflective phenomena in real-world
scenes—such as mirrors, glass, and glossy sur-
faces—without
relying
on
manually
labeled
masks, while remaining adaptable to varying light-
ing conditions.
3 Ref-Unlock
3.1 Preliminary
3D Gaussian Splatting offers an explicit formula-
tion of radiance fields by representing the scene
as a collection of anisotropic 3D Gaussian prim-
itives (Kerbl et al, 2023; Fei et al, 2024). Each
primitive is defined as a Gaussian function G(X)
over 3D space, where X ∈R3 denotes the spatial
point being evaluated:
G(X) = exp

−1
2(X −µ)T Σ−1(X −µ)

,
(1)
5

<!-- page 6 -->
Fig. 2 Overview of Ref-Unlock. Given input images, the scene is modeled as a composition of reflected and transmitted
components, each modeled with color and opacity parameters. The final rendered image is synthesized by combining the
reflection and transmission branches, relying on the reflection map.
Here, µ ∈R3 is the mean position, and the covari-
ance matrix Σ characterizes the Gaussian’s spatial
extent and orientation. To facilitate optimization,
Σ is factorized into a rotation matrix R and a scale
matrix S as:
Σ = RSST RT ,
(2)
Practically, R is parameterized via a quaternion
r ∈R4, and S via a scale vector s ∈R3. Each
Gaussian also carries additional attributes: an
opacity value α ∈R and view-dependent appear-
ance coefficients C ∈Rk expressed using Spherical
Harmonics, where k = (l + 1)2 and l is the
maximum SH degree.
Rendering in 3DGS is performed via a dif-
ferentiable point-based splatting pipeline. Each
Gaussian is first projected onto the 2D image
plane. To efficiently render the image, the screen
is divided into small tiles (e.g., 16 × 16 pixels),
and for each tile, all Gaussians whose projections
intersect with it are collected and sorted along the
camera depth axis. At each pixel, the final color Cp
is computed by compositing the colors and opaci-
ties of the contributing Gaussians in front-to-back
order using alpha blending:
Cp =
X
i∈Ncov
ciαi
i−1
Y
j=1
(1 −αj),
(3)
Here, Ncov denotes the set of Gaussians that influ-
ence the pixel, αi is the effective opacity of the i-th
Gaussian at the projected pixel location and ci is
the color computed from its SH coefficients. This
compositing process ensures that closer Gaussians
have a stronger influence, while more distant ones
are progressively attenuated, yielding a smooth
and realistic final image.
3.2 Overview
While 3DGS is effective for modeling general
scenes, it struggles to model complex physical phe-
nomena such as surface reflections. Specifically,
3DGS fails to explicitly model reflection compo-
nents from transparent or reflective surfaces (e.g.,
glass or mirrors), leading to rendering inaccura-
cies in scenes involving such materials. This is
because reflections are inherently view-dependent
and modulated by material properties, modeling
their appearance using a single set of spherical
harmonics is insufficient for accurate representa-
tion. To address this limitation and improve ren-
dering quality in reflective scenes, our Ref-Unlock
extends 3DGS by introducing reflection-specific
parameters that explicitly model and disentangle
reflected and transmitted light components.
Firstly, as shown in Fig. 2, building upon the
original 3DGS parameters—mean position µ ∈
R3, covariance Σ, and depth D ∈R—our method
introduces a unified parameter set that disentan-
gles color and opacity into two branches: reflected
(cref, αref) and transmitted (ctrans, αtrans), along
6

<!-- page 7 -->
with a reflection map c
Mref ∈[0, 1] that explicitly
represents the proportion of reflection computed
from reflection confidence βref, specifically men-
tioned in Sec. 3.3. This design allows for a more
comprehensive and disentangled representation of
reflective scenes.
Moreover,
as
discussed
in
Sec.
3.3.1,
we
increase the spherical harmonics degree from 3 to 5
for both reflected color Cref and transmitted color
Ctrans to better capture high-frequency reflections
and enhance the model’s applicability to diverse
scenes. This higher-degree representation facili-
tates more accurate modeling of complex and
high-frequency view-dependent lighting effects. To
enhance the disentanglement of transmission com-
ponents, we also employ an auxiliary reflection
removal model (Hu and Guo, 2023) (Sec. 3.3.2)
that generates pseudo-reflection-free images Iclean,
which serves as supervision for the transmitted
image bItrans during training.
Furthermore,
considering
that
the
view-
dependent projection from 3D to 2D and ras-
terization process are closely related to depth
parameters D, to better model reflections that
leverage view-dependent geometric properties nat-
urally, we employ Depth Anything v2 (Yang et al,
2024) to obtain pseudo-depth maps of the scene
(detailed in Sec. 3.4), which supervises the depth
parameters and constrain geometric features.
3.3 Ref-Unlock with Spherical
Harmonics Decomposition
To achieve more comprehensive modeling of reflec-
tive scenes, we decompose the original color and
opacity parameters in 3DGS into two distinct
branches. Additionally, we introduce a learnable
reflection confidence βref ∈[0, 1] to represent the
reflection proportion. Our Ref-Unlock representa-
tions are defined as follows:
Ref-Unlock = {µ, Σ, D, cref, αref, ctrans, αtrans, βref}.
(4)
Once the Gaussians are projected onto the
2D image plane, we employ a unified rendering
process to compute both the transmitted color
bCp
trans and the reflected color bCp
ref for each pixel,
using their respective parameters and following
the compositing formulation in Eq. 3.
Further, the pixel-wise reflection mask c
M p
ref is
computed through an accumulation process sim-
ilar to that used for color composition. Here, we
use a learnable parameter reflection confidence
βref to describe it. The reflection confidence mea-
sures the probability that a Gaussian primitive
primarily represents the reflected or transmitted
part. This continuous weighting scheme enables
differentiable separation of radiance fields while
maintaining physical plausibility. The equation of
pixel reflection map c
M p
ref is as follows:
c
M p
ref =
X
i∈Ncov
βi
refαi
trans
i−1
Y
j=1

1 −βj
ref

,
(5)
where Ncov represents the set of overlapping Gaus-
sians sorted by depth, βi
ref denotes the reflection
confidence at the i-th Gaussian, and αi
trans corre-
sponds to the transmission opacity. The product
term Qi−1
j=1(1 −βj
ref) physically represents the
cumulative attenuation of reflection contribution
from preceding Gaussians along the ray path.
This formulation originates from the need to
disentangle reflection and transmission effects in a
physically plausible manner, where the reflection
map c
Mref serves as the fundamental building block
that generates the per-Gaussian reflection confi-
dence parameters βi
ref. The derivation follows an
energy conservation principle (Hecht, 2002; Pharr
et al, 2016). ensuring that:
c
Mref + c
Mtrans = 1.
(6)
The exponential falloff in the product term accu-
rately models the occlusion effects between mul-
tiple reflective surfaces, while the linear combi-
nation with transmission opacity αi
trans maintains
the material-dependent relationship.
The final pixel color bIp is obtained by fus-
ing the transmitted and reflected components,
weighted by the corresponding reflection map:
bIp = c
M p
trans ∗bCp
trans + c
M p
ref ∗bCp
ref.
(7)
3.3.1 High-degree Spherical
Harmonics for Reflections
The adoption of high-degree spherical harmon-
ics is motivated by the necessity to accurately
capture high-frequency lighting effects, includ-
ing sharp specular reflections and fine shadowing
details. Low-degree SH (e.g., 2 or 3 degree) is
7

<!-- page 8 -->
Fig. 3 The results of our reflection removal mod-
ule.
limited to encoding smooth, low-frequency signals
such as diffuse lighting (Green, 2003). In contrast,
high-frequency reflections exhibit rapid angular
variations, necessitating higher-degree basis func-
tions for accurate reconstruction. By employing
higher-degree spherical harmonics (e.g., 5 degree
or above), the basis functions achieve broader
frequency bandwidth, thereby enabling accurate
representation of directional light transport with
high angular fidelity. Mathematically, the approx-
imation error of SH projections decreases expo-
nentially with increasing degree (Dai, 2013):
∥L(θ, φ) −
n
X
l=0
l
X
m=−l
qm
l Y m
l (θ, φ)∥2 ≤Ae−bn, (8)
where A and b are constants dependent on the
reflection sharpness, L(θ, φ) denotes theb target
function defined over angular coordinates (θ, φ),
Y m
l (θ, φ) are the spherical harmonic basis func-
tions of degree l ≥0 and order m satisfying −l ≤
m ≤l, which can be computed using the recur-
rence relations of Legendre polynomials P m
l (x)
as shown in Eq. 9, 10, and 11, where Km
l
and
N m
l
are all normalization factors, and qm
l
are the
projection coefficients learned as parameters dur-
ing the training process. The index n represents
the maximum degree of harmonics used in the
approximation.
Y m
l
(θ, φ) =





√
2Km
l
cos(mφ)P m
l (cos θ)
for m > 0
√
2Km
l
sin(−mφ)P −m
l
(cos θ)
for m < 0
K0
l P m
l (cos θ)
for m = 0
,
(9)
Km
l
=
1
N m
l
=
s
2l + 1
4π
(l −|m|)!
(l + |m|)! ,
(10)





(l −m)P m
l (x) = x(2l −1)P m
l−1(x) −(l + m −1)P m
l−2(x)
P m
m (x) = (−1)m(2m −1)!!(1 −x2)m/2
P m
m+1(x) = x(2m + 1)P m
m (x)
.
(11)
Specular reflections, particularly those result-
ing from sharp or glossy surfaces, exhibit rapid
variations in radiance across small angular neigh-
borhoods (Ramamoorthi and Hanrahan, 2001).
Low-degree SH expansions tend to oversmooth
such variations, failing to capture fine details and
introducing artifacts or blurriness in the rendered
image (Green, 2003). Using higher-degree harmon-
ics, such as those up to the 5 degree (resulting
in 36 coefficients), can provide a better represen-
tation of high-frequency content in the reflection
domain (Sloan et al, 2002; Green, 2003).
Specifically, the angular resolution of spheri-
cal harmonics increases with the max degree n,
with each additional degree enabling the repre-
sentation of finer angular variations. A 5th-degree
SH expansion can effectively resolve structures
with angular width on the degree of approximately
∼
180◦
5+1
=
30◦, making it suited to captur-
ing the concentrated energy of glossy reflections
that fall within small solid angles. Mathemat-
ically, higher-degree SH expansions are capable
of approximating spherical functions with higher-
frequency components, as reflected by the expo-
nential decay in projection error shown in Eq. 8.
In practical terms, this means that important
specular features—such as tight highlights and
view-dependent color shifts—can be encoded with
significantly lower error, resulting in more realis-
tic and sharper appearance in the final rendered
images.
3.3.2 Reflection Removal Module
The transmitted component should exclusively
represent the realistic (non-reflective) parts of the
scene. Most existing methods (Guo et al, 2022;
Meng et al, 2024; Liu et al, 2024) rely on reflection
masks to prevent interference between reflected
and
transmitted
content.
However,
obtaining
accurate reflection masks is challenging, partic-
ularly in scenes with semi-reflective overlaps,
which also limits the generalization ability of
existing methods. To address this, we propose a
mask-free reflection removal module to decompose
the semi-reflection superpositions in the input
images. Specifically, our approach builds upon
DSRNet (Hu and Guo, 2023), an efficient single-
image reflection removal model. After the initial-
ization of Gaussian primitives, the input image I
is processed by the reflection removal module to
8

<!-- page 9 -->
generate the pseudo-reflection-free image Iclean.
Then, the rendered image bI and the transmitted
part bItrans = c
Mtrans ∗bCtrans are optimized by the
photometric loss as follows:
LbI = λL1(I, bI) + (1 −λ)LD−SSIM(I, bI),
(12)
LbItrans = λL1(Iclean, bItrans)
+ (1 −λ)LD−SSIM(Iclean, bItrans),
(13)
where λ is the balance coefficient, L1 calculates
the absolute error between inputs while LD−SSIM
refers to the differentiable structural similarity
index measure.
As shown in Fig. 3, the reflection removal
module is able to effectively detect the reflection
superpositions and detach them from the original
images accordingly. By imposing the alignment
with the pseudo-reflection-free image, we guide the
transmitted part to focus only on the reconstruc-
tion of the realistic scene, thus achieving a better
decomposition result.
Although RRM is capable of effectively sepa-
rating reflection and transmission components in
general scenarios, it still has several limitations.
Specifically, the pseudo-reflection-free regions pro-
duced by RRM often exhibit blurring, leading
to the loss of fine details such as textures and
sharp edges in the generated clean images. This
degradation can adversely affect the geometric
fidelity of the reconstructed scene. However, in our
method, RRM serves only as auxiliary supervi-
sion rather than the primary reconstruction signal.
By integrating high-degree spherical harmonics
(Sec. 3.3.1) and geometric constraints (Sec. 3.4),
our approach preserves high-frequency details and
significantly reduces blurring artifacts.
Moreover, in some scenes, RRM struggles to
accurately determine whether objects visible in
specular reflections belong to the reflected or
transmitted components. While this issue does not
substantially degrade rendering quality, it devi-
ates from the physical principles underlying the
model’s design. We will further discuss the limita-
tions of current one-shot approaches and explore
the feasibility of future zero-shot solutions in
Sec. 6.
3.4 Geometry-Aware Constraints
Modeling reflections in the rendering process
presents significant challenges, not only because
of the complexity of distinguishing their appear-
ance from the underlying geometry but also due
to the geometric ambiguities they inherently intro-
duce. Since the projection from 3D space to 2D
image planes and the rasterization process are
closely coupled with depth parameter, inaccu-
rate depth estimation near reflective surfaces often
results in distorted geometry and photometric
inconsistencies.
To address these issues, we employ Depth Any-
thing v2 (Yang et al, 2024) to generate pseudo-
depth maps, which provide supervision signals for
the rendered depth. This supervision is formulated
as follows:
Ldepth =
1
H × W
H
X
i=1
W
X
j=1
Dpseudo
i,j
−bDi,j
 , (14)
where Dpseudo is the pseudo-depth predicted by
Depth Anything and bD is the rendered depth.
Although incorporating pseudo-depth super-
vision significantly enhances our model’s depth
perception, depth estimation models (Yang et al,
2024) based on texture continuity and illumi-
nation cues may still assign depth values that
contradict physical reality, such as assigning depth
to reflected objects appearing in a mirror. This
may affect the actual placement of Gaussians and
introduce noise during the optimization of their
positions. Meanwhile, as observed in the ablation
study (Fig. 9), the depth maps generated using
only pseudo-depth supervision often exhibit a lack
of smooth depth transitions within certain regions
(e.g., curtains in a mirror).
To mitigate the impact of physically implau-
sible pseudo-depth and insufficient depth smooth-
ness, we introduce a depth smoothness constraint
inspired by bilateral filtering. This constraint
enforces local geometric consistency guided by
photometric similarity, helping to reduce the
depth gap between reflected objects and real-
world content while promoting more gradual and
coherent depth transitions. The loss function is as
follows:
Lbi =
X
pi∈bI
X
pj∈b
Ni
f(pi, pj)|| bDpi −bDpj||1,
(15)
f(pi, pj) = exp(−||bIpi
trans −bIpj
trans||1
γ
),
(16)
9

<!-- page 10 -->
where the weight function f(pi, pj) measures
photometric similarity, and
b
Ni denotes the 8-
neighborhood of pixel pi. Unlike NeRFReN (Guo
et al, 2022), which relies on ground-truth color
for weight computation, our approach directly uti-
lizes the rendered transmitted components bItrans,
making the constraint more tightly coupled with
the learned geometry. This local regularization
strategy effectively complements the global super-
vision provided by Depth Anything: while the
pseudo-depth maps offer coarse yet semantically
meaningful structural information, our bilateral
filtering constraint simultaneously addresses local
inconsistencies while preserving important geo-
metric edges. The synergistic combination of these
components significantly enhances both depth
estimation accuracy and scene rendering consis-
tency.
3.5 Loss Function
To further stabilize the decomposition, we also
add a smoothness constraint on the reflection map
to reduce noise and mutual interference between
the two parts:
Lref =
X
pi∈bI
X
pj∈b
Ni
||c
M pi
ref −c
M pj
ref||1,
(17)
where c
M p
ref is the rendered reflection map of
the pixel. By introducing this smoothness con-
straint, interference between the two components
is mitigated, leading to a more coherent and
well-separated decomposition. We optimize the
attributes of Gaussians with both the photomet-
ric loss and the aforementioned constraints. The
photometric loss Lrgb is the combination of the LbI
and LbItrans in Eq. 12 and Eq. 13:
Lrgb = λbILbI + SbItrans ∗(1 −λbI)LbItrans,
(18)
where λbI is the balance coefficient between bI
and bItrans, while SbItrans represents optimization
strength of transmitted branch.
To encourage stable training and accelerate
convergence, we also introduce an L1 loss during
the initial training iterations, guiding the geome-
try of the transmitted component to align closely
with the ground truth:
Linit = SbItrans ∗L1(I, bItrans).
(19)
The overall loss can be formulated as follows:
Loverall = Lrgb + λinitLinit + λdepthLdepth
+ λbiLbi + λrefLref,
(20)
where λinit, λdepth, λbi and λref are the coeffi-
cients of each loss term.
4 Experiments
In this section, we systematically evaluate the pro-
posed Ref-Unlock through comprehensive experi-
ments and analyses to demonstrate its effective-
ness and versatility.
4.1 Datasets
We conduct experiments on two representative
datasets that capture diverse reflection character-
istics in both real-world and synthetic scenes: a)
RFFR Dataset (Guo et al, 2022): A real-world
dataset specifically curated for reflection-aware
rendering. It consists of indoor scenes with var-
ious reflective materials and surfaces, capturing
complex real-world reflections under diverse light-
ing conditions. b) Shiny Blender Dataset (Verbin
et al, 2022): A synthetic dataset containing a
range of objects with highly reflective surfaces.
4.2 Baselines and Metrics
We categorize the baselines into two main classes:
NeRF-based and GS-based methods. For fair-
ness, the majority of comparisons are performed
intra-class. Additionally, inter-class comparisons
are provided to elucidate the fundamental repre-
sentational differences.
NeRF-based
baselines
include
specialized
methods such as NeRFReN (Guo et al, 2022) and
Ref-NeRF (Verbin et al, 2022), explicitly designed
for reflective surfaces or non-Lambertian light-
ing scenarios. Comparative evaluation is extended
to original NeRF (Mildenhall et al, 2021) and
its dynamic extension D-NeRF (Pumarola et al,
2021), which establish critical performance refer-
ences for neural rendering benchmarks.
For GS-based methods, our evaluation includes
vanilla 3DGS (Kerbl et al, 2023), geometry-aware
10

<!-- page 11 -->
Table 1 View Synthesis Comparison Results on RFFR Dataset.
Method
art1
art2
art3
bookcase
tv
mirror
Avg.
PSNR ↑
NeRF (Mildenhall et al, 2021)
34.686
40.816
40.304
29.655
32.863
32.825
35.191
NeRFReN (Guo et al, 2022)
36.004
40.877
40.676
30.369
33.306
33.446
35.780
GS-IR (Liang et al, 2024)
28.072
32.646
27.964
27.406
28.854
25.139
28.347
GShader (Jiang et al, 2024)
28.711
27.198
30.677
26.930
30.404
26.521
28.407
2D-GS (Huang et al, 2024)
26.664
33.423
31.537
28.269
30.453
26.174
29.420
3D-GS (Kerbl et al, 2023)
28.251
34.426
38.732
28.259
31.775
28.555
31.666
Ours
35.158
38.405
39.747
30.149
33.289
29.441
34.365
SSIM ↑
NeRF (Mildenhall et al, 2021)
0.9643
0.9610
0.9591
0.9233
0.9551
0.9464
0.9515
NeRFReN (Guo et al, 2022)
0.9663
0.9610
0.9607
0.9232
0.9536
0.9483
0.9522
GS-IR (Liang et al, 2024)
0.9333
0.9221
0.8966
0.8854
0.9297
0.8762
0.9072
GShader (Jiang et al, 2024)
0.9313
0.8833
0.9180
0.8922
0.9426
0.9020
0.9116
2D-GS (Huang et al, 2024)
0.9256
0.9273
0.9297
0.9099
0.9496
0.9076
0.9250
3D-GS (Kerbl et al, 2023)
0.9334
0.9369
0.9565
0.9143
0.9501
0.9276
0.9365
Ours
0.9684
0.9535
0.9593
0.9192
0.9575
0.9351
0.9488
LPIPS ↓
NeRF (Mildenhall et al, 2021)
0.1859
0.2191
0.2217
0.2135
0.2241
0.1893
0.2089
NeRFReN (Guo et al, 2022)
0.1828
0.2281
0.2316
0.2284
0.2306
0.1897
0.2152
GS-IR (Liang et al, 2024)
0.2631
0.3292
0.3246
0.3015
0.3007
0.3227
0.3070
GShader (Jiang et al, 2024)
0.2361
0.3896
0.3348
0.3010
0.2543
0.2889
0.3008
2D-GS (Huang et al, 2024)
0.2671
0.3169
0.3270
0.2886
0.2458
0.2711
0.2861
3D-GS (Kerbl et al, 2023)
0.2581
0.3304
0.2681
0.2631
0.2385
0.2373
0.2659
Ours
0.1621
0.2318
0.2287
0.2211
0.2286
0.1944
0.2111
2DGS (Huang et al, 2024), and specialized meth-
ods for reflective scene modeling and physically
based
rendering (PBR)
integration—including
GShader (Jiang et al, 2024), R3DG (Gao et al,
2023), and GS-IR (Liang et al, 2024). These
methods leverage the efficiency and expressiveness
of Gaussian primitives for scene representation,
with some incorporating reflection-aware compo-
nents or shader-level enhancements to improve
rendering accuracy and visual fidelity.
The quantitative evaluation of NVS employs
three metrics: PSNR, SSIM, and LPIPS. PSNR
measures
pixel-level
fidelity,
whereas
SSIM
assesses structural similarity. In contrast, LPIPS
evaluates perceptual similarity using deep fea-
tures,
offering
better
alignment
with
human
visual perception. Additionally, qualitative visual
comparisons are provided to facilitate an intuitive
evaluation of rendering quality.
4.3 Implementation Details
All experiments are conducted on NVIDIA A100
GPU(s). For the baseline methods, we follow
either the officially released settings on the corre-
sponding datasets or use the default hyperparam-
eters when no configuration is available for the
particular dataset. Our model is optimized for a
total of 30,000 iterations. The key hyperparam-
eters are set as follows: λbI = 0.8, SbItrans = 10,
λinit = 0.01, λdepth = 30, and both λbi and λref are
set to 0.001. The degree of the spherical harmonics
is set to 5.
4.4 Comparisons
Ref-Unlock is evaluated on the real-world RFFR
Dataset (Guo et al, 2022) and synthetic Shiny
Blender Dataset (Verbin et al, 2022). Exten-
sive quantitative and visual comparisons against
11

<!-- page 12 -->
Table 2 View Synthesis Comparison Results on ShniyBlender Dataset.
Method
car
coffee
helmet
teapot
toaster
Avg.
PSNR ↑
D-NeRF (Pumarola et al, 2021)
23.305
26.584
24.562
41.628
21.067
28.146
Ref-NeRF (Verbin et al, 2022)
29.268
33.617
29.268
44.676
24.893
32.345
GS-IR (Liang et al, 2024)
25.589
30.850
24.882
37.536
18.862
27.544
R3DG (Gao et al, 2023)
25.943
30.341
25.170
43.646
18.776
28.775
3D-GS (Kerbl et al, 2023)
25.998
31.224
28.120
40.642
19.916
29.180
Ours
26.823
31.276
27.890
42.908
21.389
30.057
SSIM ↑
D-NeRF (Pumarola et al, 2021)
0.8817
0.9312
0.8931
0.9923
0.8603
0.9117
Ref-NeRF (Verbin et al, 2022)
0.9514
0.9711
0.9514
0.9951
0.9052
0.9548
GS-IR (Liang et al, 2024)
0.8947
0.9532
0.9018
0.9893
0.7569
0.8992
R3DG (Gao et al, 2023)
0.9233
0.9646
0.9314
0.9954
0.8582
0.9346
3D-GS (Kerbl et al, 2023)
0.9250
0.9687
0.9482
0.9950
0.8828
0.9439
Ours
0.9216
0.9654
0.9428
0.9946
0.8910
0.9431
LPIPS ↓
D-NeRF (Pumarola et al, 2021)
0.0765
0.1538
0.1621
0.1010
0.2009
0.1389
Ref-NeRF (Verbin et al, 2022)
0.0989
0.0849
0.0989
0.0166
0.1278
0.0854
GS-IR (Liang et al, 2024)
0.0814
0.1090
0.1616
0.0222
0.2388
0.1226
R3DG (Gao et al, 2023)
0.0567
0.0882
0.1238
0.0120
0.1699
0.0901
3D-GS (Kerbl et al, 2023)
0.0550
0.0898
0.0875
0.0138
0.1431
0.0779
Ours
0.0544
0.0869
0.1013
0.0136
0.1272
0.0767
state-of-the-art GS-based and NeRF-based base-
lines validate the effectiveness and performance
advantages of the proposed method.
4.4.1 Comparisons on RFFR Dataset
Quantitative Results. As shown in the Table 1,
Ref-Unlock achieves the best performance among
all GS-based methods in terms of PSNR, SSIM,
and LPIPS. Compared to NeRF-based methods,
although the average PSNR and SSIM of our
method do not surpass the top-performing NeRF-
based
models,
Ref-Unlock
outperforms
NeR-
FReN—the method specifically proposed for this
dataset—in terms of LPIPS, indicating better
perceptual quality. Moreover, on scenes such as
art1, bookcase, and tv, our method achieves higher
PSNR than NeRF, and on art1 and tv, it also out-
performs all NeRF-based methods in SSIM. These
results highlight the effectiveness of our approach
in handling reflective scenes.
Ref-Unlock performs favorably against those
GS-based methods on RFFR. While NeRF-based
methods still retain a slight numerical advantage
in some metrics, our Ref-Unlock offers comparable
visual quality with significantly higher efficiency.
This trade-off makes Ref-Unlock a compelling
choice.
Qualitative Analysis. We provide qualitative
comparisons on all methods in Fig. 4. Across all
tested scenes, our method achieves superior per-
formance compared to all GS-based approaches,
demonstrating its robustness and effectiveness
in handling reflective scenarios. Moreover, our
approach achieves performance on par with NeRF-
based techniques in handling semi-reflective scenes
(e.g., art2 and art3) as well as scenes featur-
ing specular reflections (e.g., mirror and tv). The
results highlight the strong reconstruction abil-
ity and photorealism of our method, especially
in faithfully reproducing reflective surfaces and
subtle lighting effects.
In addition, we provide a detailed visual com-
parison with 3DGS, the best-performing GS-based
baseline in terms of quantitative metrics, to fur-
ther demonstrate the advantages of our method
in handling reflective regions. As shown in Fig. 5,
in semi-reflective scenes, our method is capable
12

<!-- page 13 -->
Fig. 4 Visual comparisons between NeRF-based methods, GS-based methods and our Ref-Unlock. Our
method presents a more detailed and realistic rendering than GS-based methods in all cases. Compared to NeRF-based
methods, our method shows comparable results in scenes with semi-reflections like art2 and art3 and delivers almost
indistinguishable rendering quality in scenes containing specular reflections, like tv and mirror.
of accurately reconstructing reflective highlights
on the closet surface (art2) as well as objects
such as picture frames within the reflected regions
(art3). Moreover, in specular reflection scenes, our
method provides a much better reconstruction of
reflected content, such as the sofa reflected in tv
and the curtains reflected in mirror. In contrast,
3DGS suffers from blurred edges and artifacts in
these regions.
4.4.2 Comparisons on ShinyBlender
Dataset
Quantitative Results. Table 2 shows that our
Ref-Unlock achieves the best or second-best per-
formance in terms of PSNR and LPIPS among
GS-based methods and D-NeRF. Notably, Ref-
Unlock outperforms all methods except Ref-
NeRF on scenes with macroscopic reflections, i.e.,
toaster, coffee, car, where reflective effects are
prominent. For scenes like teapot, where reflec-
tions are subtler, our method performs on par
with or even surpasses other 3DGS and D-
NeRF approaches. In terms of SSIM, Ref-Unlock
13

<!-- page 14 -->
Fig. 5 Detailed visual comparisons between 3DGS and our method. For a comprehensive comparison, we further
show visual results of 3DGS and our method with zoom-in details on RFFR scenes.
Fig. 6 Visual comparisons between ReFNeRF, D-NeRF, GS-IR, 3DGS, R3DG and our method, Our method
delivers visual results that not only rival those of 3DGS but, in many cases, also closely match the high visual fidelity of
Ref-NeRF.
achieves the highest perceptual quality on the
toaster scene among GS-based methods. While it
lags slightly behind Ref-NeRF and 3DGS variants
on a few metrics, Ref-Unlock still ranks second
in average performance overall. More importantly,
our method offers significantly faster rendering
speed, making it a more efficient and practical
choice for real-time or large-scale applications.
Qualitative Analysis. Furthermore, we present
a comprehensive visual comparison of all six eval-
uated methods in Fig. 6. We observe that for
the teapot—a scene with relatively weak reflectiv-
ity—only our method and Ref-NeRF accurately
reconstruct the subtle white highlights within the
enlarged sub-image. For the remaining four highly
specular objects, our method achieves visualiza-
tion results that are comparable to or even surpass
those of 3DGS and, in many instances, outper-
form Ref-NeRF in quality. Notably, it achieves this
high-fidelity rendering efficiently—without requir-
ing object masks—while maintaining competitive
training times. In each column of Fig. 6, the
enlarged sub-images highlight regions for detailed
comparison of reflections across different meth-
ods. Our method achieves highly accurate render-
ing, particularly in preserving fine details within
reflective surfaces. Specifically, for highly specular
objects such as the helmet in Fig. 6, our approach
not only faithfully captures the reflected content
but also preserves the smoothness of the object’s
material after rendering, resulting in more real-
istic and visually compelling results compared to
existing approaches.
14

<!-- page 15 -->
Fig. 7 Reflection Disentanglement. We present the reflection maps and the separated reflection and transmission
images for three scenes: art2, tv, and art3.
4.5 Reflection Disentanglement
To thoroughly assess the reflection handling per-
formance of Ref-Unlock, we conducted dedicated
experiments on separating reflected content for
novel view generation. As indicated in Fig. 7,
the reflected components are clearly separated
from the transmitted components and represented
separately by reflection images bIref and transmis-
sion images bItrans. The final rendered images are
obtained by adding the reflection images and the
transmission images together. During this process,
the learned reflection map captures the spatially
varying reflection intensity, where higher values
(white) correspond to reflective surfaces and lower
values (black) denote non-reflective regions. As
shown in Eq. 7, the reflection image is obtained
through the element-wise product of the reflec-
tion map c
M p
ref and reflection color bCp
ref, while the
transmission image is computed analogously using
the transmission map c
M p
trans and transmission
color bCp
trans.
In contrast to NeRFReN (Guo et al, 2022),
which
requires
manually
provided
reflection
masks as auxiliary input, our approach gener-
ates all necessary outputs—such as transmitted
and reflected components—automatically during
inference, without relying on extra assumptions
or supervision. As shown in Fig. 7, Ref-Unlock
produces a perceptually plausible separation of
transmission and reflection, while still preserving
high-fidelity novel view rendering.
While our experimental results already show-
case strong performance in separating reflections,
it is worth emphasizing that our method pri-
marily aims to reconstruct stable virtual images
via the reflected component, whereas the low-
frequency specular highlights are handled as view-
dependent effects within the transmitted branch.
This design choice, however, may deviate from the
physical laws of reflection in certain cases. For
example, when dealing with mirror-like reflections,
some low-frequency objects reflected in the mirror
may be incorrectly attributed to the transmis-
sion component rather than the reflection compo-
nent. While this approximation is not fully phys-
ically accurate, its impact on final view synthesis
quality remains negligible due to our reflection-
transmission composition strategy. Future work
could explore more physically precise disentangle-
ment methods.
4.6 Reflection Manipulation
Given the inherent advantage of our method in
separating reflection and transmission regions, we
can easily enable editing of the reflection intensity
across the scene. Considering that user require-
ments may vary—and that global edits may
15

<!-- page 16 -->
Fig. 8 Reflection Manipulation. By adjusting the reflection coefficient (0.1, 0.5, 1.0, 1.5 and 2.0 from left to right)
on the reflection branch, we can arbitrarily diminish or augment the brightness of the reflected content in our text-prompted
regions.
not always be desirable—we further introduce
a text-prompt-based region selection mechanism,
allowing for more flexible and targeted reflection
adjustments.
Our method allows users to define the target
editing region via a text prompt. Using the EVF-
SAM model (Zhang et al, 2024), we generate a
segmentation mask for the specified region. The
reflection edit is then applied by scaling the reflec-
tion branch output within the masked area by
a user-defined coefficient—values greater than 1
amplify reflections, while values below 1 attenuate
them. Regions outside the mask remain unmod-
ified, ensuring the rest of the scene retains its
original appearance.
As a result, we achieve editable scene reflec-
tions. As shown in Fig. 8, our editing strat-
egy effectively adjusts the reflection intensity in
the scene. For example, the curtains reflected
in mirror and the sofa reflected in tv become
progressively brighter as the reflection coefficient
increases.
Importantly, the edited scenes remain free of
color artifacts or visual inconsistencies, which fur-
ther demonstrates the effectiveness of our model in
accurately separating reflection and transmission
components.
5 Ablation Studies
We perform all ablation experiments on the RFFR
dataset to assess the contribution of our model
components and the influence of different hyper-
parameter settings. The reported results reflect
the average performance across the entire RFFR
dataset.
5.1 Ablation Study on Model Design
To assess the impact of different design choices,
we trained four additional model variants to eval-
uate the effectiveness of the reflection removal
module (RRM) constraint, pseudo depth con-
straint, and depth and reflection map smoothness,
respectively. The specific configurations of each
model, along with their corresponding quantita-
tive metrics and visual comparisons, are presented
16

<!-- page 17 -->
Table 3 Ablation Studies on Model Design.
Model
Pseudo Depth
RRM
Depth Smoothness
Ref-map Loss
SSIM ↑
PSNR ↑
LPIPS ↓
A
✓
✓
✓
0.9475
34.432
0.2113
B
✓
✓
✓
0.9343
32.179
0.2360
C
✓
✓
✓
0.9488
34.049
0.2133
D
✓
✓
✓
0.9483
34.215
0.2124
Ours
✓
✓
✓
✓
0.9488
34.365
0.2111
Fig. 9 Effectiveness of Model Design. The combination of all model designs produces the best rendering results.
Fig.
10 Comparison
of
different
transmission
strengths. We compare the visual results of models with
transmission strength = 1 and 10 separately.
in Table 3 and Fig. 9. The ablation study demon-
strates that our fully designed model consistently
outperforms the variants, achieving the highest
performance.
Specifically, we observe that removing the
Reflection Removal Module (RRM, Model A)
results in slightly higher PSNR but lower SSIM
and LPIPS compared to our full model. More
importantly, visual comparisons (left column of
the second row in Fig. 9) suggest that the model
without RRM fails to physically disentangle reflec-
tion and transmission components. This indicates
that relying solely on learning the reflection map
parameters is insufficient for achieving effective
separation - explicit external constraints, such
as those introduced by RRM, are essential for
accurate reflection disentanglement.
Moreover, removing the pseudo-depth con-
straint (Model B) leads to a substantial decline
across all three evaluation metrics. This val-
idates the pivotal role of the geometry-aware
constraint in enabling accurate 3D scene represen-
tation with Gaussian spheres and contributing to
high-quality rendering. As demonstrated in Fig. 9,
the depth maps also show noticeable improve-
ments—featuring fewer abrupt transitions and
more consistent depth estimates, particularly in
planar regions like walls.
The ablation study further reveals that remov-
ing either the depth smoothness constraint or the
17

<!-- page 18 -->
Table 4 Transmission Strength Ablation Results on RFFR Dataset.
SbItrans
art1
art2
art3
bookcase
tv
mirror
Avg.
PSNR ↑
1
35.210
37.720
39.895
30.525
34.035
29.957
34.557
10 (default)
35.158
38.405
39.747
30.149
33.289
29.441
34.365
20
33.992
37.013
39.399
29.920
32.563
29.407
33.715
SSIM ↑
1
0.9651
0.9538
0.9593
0.9250
0.9587
0.9299
0.9486
10 (default)
0.9684
0.9535
0.9593
0.9192
0.9575
0.9351
0.9488
20
0.9653
0.9510
0.9588
0.9120
0.9541
0.9350
0.9460
LPIPS ↓
1
0.1788
0.2508
0.2485
0.2289
0.2282
0.2148
0.2250
10 (default)
0.1621
0.2318
0.2287
0.2211
0.2286
0.1944
0.2111
20
0.1555
0.2434
0.2247
0.2213
0.2323
0.1909
0.2113
Table 5 Pseudo Depth Weights Ablation Results on RFFR Dataset.
λdepth
art1
art2
art3
bookcase
tv
mirror
Avg.
PSNR ↑
1
34.208
34.871
39.158
29.362
32.742
29.069
33.235
30 (default)
35.158
38.405
39.747
30.149
33.289
29.441
34.365
50
34.686
38.938
39.614
30.207
33.063
29.507
34.336
SSIM ↑
1
0.9657
0.9351
0.9556
0.9097
0.9547
0.9340
0.9425
30 (default)
0.9684
0.9535
0.9593
0.9192
0.9575
0.9351
0.9488
50
0.9673
0.9542
0.9593
0.9199
0.9563
0.9365
0.9489
LPIPS ↓
1
0.1706
0.2602
0.2372
0.2333
0.2293
0.2021
0.2221
30 (default)
0.1621
0.2318
0.2287
0.2211
0.2286
0.1944
0.2111
50
0.1615
0.2363
0.2381
0.2190
0.2295
0.1954
0.2133
reflection map smoothness constraint (Models C
and D) results in consistent performance degrada-
tion across all evaluation metrics. Notably, despite
their relatively small weight contributions (only
0.001) in the total loss function, these constraints
prove essential for optimal performance. In the
visual comparisons (middle column of the second
row in Fig. 9), the depth smoothness constraint
leads to smoother and more natural depth transi-
tions. Additionally, the reflection map smoothness
constraint significantly enhances boundary delin-
eation between reflection and transmission regions
through noise suppression and improved map
robustness, as evidenced in the right column of the
second row in Fig. 9.
5.2 Ablation Study on Transmission
Strength
We
conduct
three
sets
of
experiments
with
SbItrans = 1, 10, and 20 to decide the optimal
parameter for the transmission strength.
As shown in the Table 4, the quantitative
performance of the model exhibits a rise-then-
fall trend as the transmission strength increases.
Notably, when the transmission strength is set to
10, the model achieves the best SSIM and LPIPS
18

<!-- page 19 -->
Table 6 SH Degree Ablation Results on RFFR Dataset.
SH Degree n
art1
art2
art3
bookcase
tv
mirror
Avg.
PSNR ↑
3
34.763
38.701
39.652
30.116
33.048
29.220
34.250
4
34.226
39.121
39.789
30.062
33.077
29.387
34.277
5 (default)
35.158
38.405
39.747
30.149
33.289
29.441
34.365
SSIM ↑
3
0.9674
0.9544
0.9593
0.9214
0.9579
0.9368
0.9495
4
0.9662
0.9554
0.9601
0.9201
0.9569
0.9363
0.9492
5 (default)
0.9684
0.9535
0.9593
0.9192
0.9575
0.9351
0.9488
LPIPS ↓
3
0.1657
0.2377
0.2331
0.2217
0.2309
0.1940
0.2138
4
0.1634
0.2325
0.2300
0.2214
0.2323
0.1945
0.2124
5 (default)
0.1621
0.2318
0.2287
0.2211
0.2286
0.1944
0.2111
scores, and we therefore select it as the optimal
parameter.
Further analysis of the other two settings
reveals that, although the model with transmis-
sion strength equal to 1 attains the highest PSNR,
the visual results in Fig. 10 suggest inferior disen-
tanglement quality. Specifically, compared to the
setting of 10, the reflection map under strength 1
fails to accurately distinguish between reflection
and transmission regions-especially in the mirror
scene, where the reflected curtain is poorly cap-
tured. This results in an under-reconstruction of
fine details, manifested as noticeable blurriness in
the curtain area.
For the model with a larger transmission
strength (20), although the separation between
reflection and transmission components improves
compared to the setting of 1, the improvement
over strength 10 is marginal. However, due to
the increased weight, the transmission component
dominates the overall loss, causing the optimiza-
tion to shift its focus disproportionately toward
the transmission branch. This imbalance nega-
tively impacts the reconstruction quality of other
components, such as reflection and depth, ulti-
mately leading to a drop in overall performance.
5.3 Ablation Study on Weight of
Pseudo Depth
Likewise, we performed three experiments to
explore the impact of different weights assigned to
the pseudo depth constraint. The weight λdepth is
set to 1, 30 and 50, respectively.
As shown in the Table 5, when the pseudo-
depth loss weight is set to 30, this setting yields the
highest performance with respect to both PSNR
and LPIPS, and we adopt this as the final weight.
Overall, the performance also follows a rise-then-
fall trend as the depth weight increases. This
behavior essentially reflects a trade-off in model
optimization. When the weight is too low, the
depth constraint is too weak, making it difficult
for the model to accurately perceive and repre-
sent the scene geometry. When the weight is too
high, it dominates the optimization and impairs
the learning of other components, leading to a
drop in overall performance. Therefore, a weight
of 30 serves as a balanced setting that provides
sufficient geometric supervision while maintaining
overall model effectiveness.
5.4 Ablation Study on Degree of SH
Additionally, we also conduct an ablation study on
the degree of SH to verify whether using higher-
degree SH contributes to improved reconstruction
quality of reflective scenes in our framework.
As shown in the Table 6, increasing the SH
degree consistently improves the model perfor-
mance in terms of PSNR and LPIPS, indicating
enhanced reconstruction accuracy and perceptual
quality.
Interestingly, we observe a slight decrease in
SSIM despite improvements in PSNR and LPIPS
when using higher-degree SH. This discrepancy
can be attributed to the limitations of the SSIM
metric itself (Nilsson and Akenine-M¨oller, 2020)
19

<!-- page 20 -->
rather than an actual degradation in reconstruc-
tion quality. Specifically, SSIM is sensitive to local
structural consistency but fails to capture percep-
tual realism in high-frequency reflective regions.
Since reflective surfaces often contain visually
accurate but structurally dissimilar content (Li
et al, 2025) (e.g., mirror reflections, specular high-
lights), SSIM may penalize these regions even
when they are faithfully reconstructed. In con-
trast, LPIPS, which aligns more closely with
human perception, reflects the improved visual
fidelity achieved through high-degree SH model-
ing.
Based on the overall evaluation results, we
select the highest tested SH degree of 5 as the final
configuration for our framework. Given the sub-
stantially increased computational cost associated
with higher SH degrees, and the fact that 5-degree
SH already yields high-fidelity reconstruction of
reflective details.
6 Limitations and Future
Works
Although our model demonstrates strong capa-
bility in separating reflection and transmission
components, it currently relies heavily on the
Reflection Removal Module for effective disentan-
glement. However, the underlying RRM (DSR-
Net) has inherent limitations. In certain scenarios,
DSRNet fails to accurately extract the reflection
layer and may introduce ambiguities by misclassi-
fying reflected content (e.g., mirror reflections) as
part of the transmission. Although our Ref-Unlock
could correct some of these errors during opti-
mization, such misalignment with physical reality
may still misguide the training process, leading to
entangled or incorrect decomposition results.
Moreover, while we employ pseudo-depth as a
geometric constraint to enhance 3D scene repre-
sentation, our approach remains limited to depth
supervision. Key geometric attributes—such as
surface normals and curvature—are not yet inte-
grated into the learning framework. Incorporating
these features could provide richer structural pri-
ors, potentially improving both the fidelity of view
synthesis and the interpretability of reflection-
transmission separation.
These limitations suggest several promising
directions for future research: (1) developing more
physically accurate and robust reflection modeling
approaches, (2) enhancing geometric understand-
ing through integration of surface normals and
additional differential geometry constraints, and
(3) designing illumination models that explicitly
disentangle various lighting components, including
inter-reflections, specular highlights, and ambient
illumination.
7 Conclusions
In this paper, we introduce Ref-Unlock, a new
framework grounded in 3D Gaussian Splatting
that incorporates geometric awareness to enable
high-fidelity rendering of reflective scenes. Our
Ref-Unlock framework introduces a novel strat-
egy that separates the scene into transmitted
and reflected components, and synthesizes the
final output by integrating the renderings from
both branches. Accordingly, we correspondingly
increased the degree of the spherical harmonics
used to represent color (radiance) in each branch,
employing higher-degree spherical harmonics to
better capture high-frequency reflections. To facil-
itate the comprehensive decomposition of the
two components, we have developed a reflection
removal module to ensure the alignment of the
transmitted part with scenes that are free of reflec-
tions. Further, we constrain the model-generated
depth using a pseudo-depth map to provide the
model with enhanced 3D geometric awareness,
thereby further improving the accuracy of view-
dependent 3D scene reconstruction. At the same
time, we incorporate a bilateral depth smooth-
ness constraint to ensure that depth gradients in
the scene change gradually and continuously, bet-
ter reflecting the depth variation in real-world
scenes. Our method significantly outperforms the
original 3DGS and some classic GS-based reflec-
tion techniques on scenes containing reflections,
while delivering results comparable to NeRF-
based methods. Finally, the ability to control
and manipulate reflections highlights the poten-
tial of our approach for a wide range of practical
applications.
Declarations
• Conflict of interest: Authors have no compet-
ing interests to declare that are relevant to the
content of this article.
20

<!-- page 21 -->
• Data and code is available at our Github Repo
https://ref-unlock.github.io/
Data Availability
Figures 4, 5, 7, 8, 9, 10 and Tables 1, 3, 4, 5, 6 are
experimental results on the publicly available data
shared in Guo et al (2022). Figure 6 and Table 2
are experimental results on the publicly available
data shared in Verbin et al (2022). All the inter-
mediate experimental results and data shown in
this paper can be accessed from the first author
by request [https://ref-unlock.github.io/].
References
Barron JT, Mildenhall B, Tancik M, et al (2021)
Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields. In: Proceedings
of the IEEE/CVF International Conference on
Computer Vision, pp 5855–5864
Chan ER, Nagano K, Chan MA, et al (2023) Gen-
erative novel view synthesis with 3d-aware diffu-
sion models. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision,
pp 4217–4229
Chen X, Liu J, Zhao H, et al (2023) Nerrf: 3d
reconstruction and view synthesis for transpar-
ent and specular objects with neural refractive-
reflective fields. arXiv preprint arXiv:230913039
Dai F (2013) Approximation theory and harmonic
analysis on spheres and balls. Springer
Deng K, Liu A, Zhu JY, et al (2022) Depth-
supervised nerf: Fewer views and faster training
for free. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern
Recognition, pp 12882–12891
Fei B, Xu J, Zhang R, et al (2024) 3d gaussian
splatting as new era: A survey. IEEE Transac-
tions on Visualization and Computer Graphics
Gao J, Gu C, Lin Y, et al (2023) Relightable
3d gaussian: Real-time point cloud relighting
with brdf decomposition and ray tracing. arXiv
preprint arXiv:231116043
Gao K, Gao Y, He H, et al (2022) Nerf: Neu-
ral radiance field in 3d vision, a comprehensive
review. arXiv preprint arXiv:221000379
Green R (2003) Spherical harmonic lighting: The
gritty details. Tech. rep., Sony Computer Enter-
tainment America
Gu´edon A, Lepetit V (2024) Sugar: Surface-
aligned gaussian splatting for efficient 3d mesh
reconstruction and high-quality mesh rendering.
In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition,
pp 5354–5363
Guo YC, Kang D, Bao L, et al (2022) Ner-
fren: Neural radiance fields with reflections. In:
Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp
18409–18418
Hecht E (2002) Optics. Addison Wesley
Holland LV, Bliersbach R, M¨uller JU, et al (2023)
Tram-nerf: Tracing mirror and near-perfect
specular reflections through neural radiance
fields. 2310.10650
Hu Q, Guo X (2023) Single image reflection sep-
aration via component synergy. In: Proceedings
of the IEEE/CVF International Conference on
Computer Vision, pp 13138–13147
Huang B, Yu Z, Chen A, et al (2024) 2d gaussian
splatting for geometrically accurate radiance
fields. In: ACM SIGGRAPH 2024 conference
papers, pp 1–11
Jain N, Kumar S, Van Gool L (2024) Learn-
ing robust multi-scale representation for neural
radiance fields from unposed images. Interna-
tional Journal of Computer Vision 132(4):1310–
1335
Jiang Y, Tu J, Liu Y, et al (2024) Gaussianshader:
3d gaussian splatting with shading functions
for reflective surfaces. In: Proceedings of the
IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp 5322–5332
Kerbl B, Kopanas G, Leimk¨uhler T, et al (2023)
3d gaussian splatting for real-time radiance
21

<!-- page 22 -->
field rendering. ACM Transactions on Graphics
42(4):1–14
Kopanas G, Philip J, Leimk¨uhler T, et al (2021)
Point-based neural rendering with per-view
optimization. In: Computer Graphics Forum,
Wiley Online Library, pp 29–43
Li F, Ma J, Liang HN, et al (2025) A compre-
hensive survey of specularity detection: state-of-
the-art techniques and breakthroughs. Artificial
Intelligence Review 58(7):1–53
Liang Z, Zhang Q, Feng Y, et al (2024) Gs-ir:
3d gaussian splatting for inverse rendering. In:
Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp
21644–21653
Lin CY, Fu Q, Merth T, et al (2024) Fastsr-nerf:
improving nerf efficiency on consumer devices
with a simple super-resolution pipeline. In: Pro-
ceedings of the IEEE/CVF Winter Conference
on Applications of Computer Vision, pp 6036–
6045
Liu J, Tang X, Cheng F, et al (2024) Mirror-
gaussian: Reflecting 3d gaussians for recon-
structing
mirror
reflections.
arXiv
preprint
arXiv:240511921
Liu X, Li S, Gao Y (2025) Image matting and
3d reconstruction in one loop. International
Journal of Computer Vision pp 1–21
Ma L, Agrawal V, Turki H, et al (2023) Specn-
erf: Gaussian directional encoding for specular
reflections. arXiv preprint arXiv:231213102
Malarz D, Smolak W, Tabor J, et al (2023) Gaus-
sian splatting with nerf-based color and opacity.
arXiv preprint arXiv:231213729
Meng J, Li H, Wu Y, et al (2024) Mirror-3dgs:
Incorporating mirror reflections into 3d gaus-
sian splatting. arXiv preprint arXiv:240401168
Mildenhall B, Srinivasan PP, Tancik M, et al
(2021) Nerf: Representing scenes as neural radi-
ance fields for view synthesis. Communications
of the ACM 65(1):99–106
Nilsson J, Akenine-M¨oller T (2020) Understand-
ing ssim. arXiv preprint arXiv:200613846
Pharr M, Jakob W, Humphreys G (2016) Physi-
cally Based Rendering: From Theory to Imple-
mentation. Morgan Kaufmann
Pumarola A, Corona E, Pons-Moll G, et al (2021)
D-nerf: Neural radiance fields for dynamic
scenes. In: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recog-
nition, pp 10318–10327
Ramamoorthi R, Hanrahan P (2001) An efficient
representation for irradiance environment maps.
In: Proceedings of the 28th annual conference on
Computer graphics and interactive techniques,
pp 497–500
Reiser C, Peng S, Liao Y, et al (2021) Kilo-
nerf: Speeding up neural radiance fields with
thousands of tiny mlps. In: Proceedings of the
IEEE/CVF international conference on com-
puter vision, pp 14335–14345
R¨uckert D, Franke L, Stamminger M (2022) Adop:
Approximate differentiable one-pixel point ren-
dering. ACM Transactions on Graphics (ToG)
41(4):1–14
Schonberger JL, Frahm JM (2016) Structure-
from-motion revisited. In: Proceedings of the
IEEE conference on computer vision and pat-
tern recognition, pp 4104–4113
Sloan PP, Kautz J, Snyder J (2002) Precomputed
radiance transfer for real-time rendering in
dynamic, low-frequency lighting environments.
In: ACM Transactions on Graphics (TOG),
ACM, pp 527–536
Verbin D, Hedman P, Mildenhall B, et al (2022)
Ref-nerf: Structured view-dependent appear-
ance
for
neural
radiance
fields.
In:
2022
IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), IEEE, pp
5481–5490
Verbin D, Srinivasan PP, Hedman P, et al (2024)
Nerf-casting: Improved view-dependent appear-
ance with consistent reflections. In: SIGGRAPH
Asia 2024 Conference Papers, pp 1–10
22

<!-- page 23 -->
Wang G, Chen Z, Loy CC, et al (2023) Sparsenerf:
Distilling depth ranking for few-shot novel view
synthesis. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision,
pp 9065–9076
Wang J, Lyu Z, Fei B, et al (2024) Slide: A uni-
fied mesh and texture generation framework
with enhanced geometric control and multi-view
consistency. International Journal of Computer
Vision pp 1–24
Xiang H, Li X, Lai X, et al (2024) Gaussian-
room: Improving 3d gaussian splatting with sdf
guidance and monocular cues for indoor scene
reconstruction. arXiv preprint arXiv:240519671
Xu Y, Peng S, Yang C, et al (2022) 3d-aware
image synthesis via learning structural and tex-
tural representations. In: Proceedings of the
IEEE/CVF conference on computer vision and
pattern recognition, pp 18430–18439
Yang L, Kang B, Huang Z, et al (2024) Depth
anything v2. Advances in Neural Information
Processing Systems 37:21875–21911
Yao Y, Zeng Z, Gu C, et al (2024) Reflective gaus-
sian splatting. arXiv preprint arXiv:241219282
Ye K, Gao C, Li G, et al (2024a) Geosplatting:
Towards geometry guided gaussian splatting
for physically-based inverse rendering. arXiv
preprint arXiv:241024204
Ye K, Hou Q, Zhou K (2024b) 3d gaussian
splatting with deferred reflection. In: ACM SIG-
GRAPH 2024 Conference Papers, pp 1–10
Yu M, Lu T, Xu L, et al (2024) Gsdf: 3dgs meets
sdf for improved rendering and reconstruction.
arXiv preprint arXiv:240316964
Zeng J, Bao C, Chen R, et al (2023) Mirror-
nerf: Learning neural radiance fields for mirrors
with whitted-style ray tracing. In: Proceedings
of the 31st ACM International Conference on
Multimedia, pp 4606–4615
Zhang X, Bi S, Sunkavalli K, et al (2022) Ner-
fusion: Fusing radiance fields for large-scale
scene reconstruction. In: Proceedings of the
IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp 5449–5458
Zhang Y, Cheng T, Zhu L, et al (2024) Evf-
sam: Early vision-language fusion for text-
prompted
segment
anything
model.
arXiv
preprint arXiv:240620076
Zhou
Q,
Gong
Y,
Yang
W,
et
al
(2025)
Mgsr: 2d/3d mutual-boosted gaussian splat-
ting
for
high-fidelity
surface
reconstruction
under various light conditions. arXiv preprint
arXiv:250305182
Zhu H, Zhang Z, Zhao J, et al (2024) Scene
reconstruction techniques for autonomous driv-
ing: a review of 3d gaussian splatting. Artificial
Intelligence Review 58(1):30
Zuo X, Samangouei P, Zhou Y, et al (2025)
Fmgs: Foundation model embedded 3d gaus-
sian splatting for holistic 3d scene understand-
ing. International Journal of Computer Vision
133(2):611–627
23
