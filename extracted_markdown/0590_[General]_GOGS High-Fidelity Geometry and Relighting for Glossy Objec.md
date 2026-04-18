<!-- page 1 -->
GOGS: High-Fidelity Geometry and Relighting for Glossy Objects
via Gaussian Surfels
Xingyuan Yang
Min Wei
Chengdu University of Information Technology
yangcyanx@gmail.com
weimin@cuit.edu.cn
Figure 1. Our method achieves accurate geometry recovery and photorealistic rendering under novel illumination. Our rendering closely
matches the GT with realistic specular effects, outperforming competitors suffering from material errors or blurred specular. Normal maps
accurately capture inter-reflection geometry (e.g. cat plate reflections), unlike competitors that fail under specular interference.
Abstract
Inverse rendering of glossy objects from RGB imagery
remains fundamentally limited by inherent ambiguity. Al-
though NeRF-based methods achieve high-fidelity recon-
struction via dense-ray sampling, their computational cost
is prohibitive.
Recent 3D Gaussian Splatting achieves
high reconstruction efficiency but exhibits limitations un-
der specular reflections. Multi-view inconsistencies intro-
duce high-frequency surface noise and structural artifacts,
while simplified rendering equations obscure material prop-
erties, leading to implausible relighting results. To address
these issues, we propose GOGS, a novel two-stage frame-
work based on 2D Gaussian surfels. First, we establish
robust surface reconstruction through physics-based ren-
dering with split-sum approximation, enhanced by geomet-
ric priors from foundation models.
Second, we perform
material decomposition by leveraging Monte Carlo impor-
tance sampling of the full rendering equation, modeling in-
direct illumination via differentiable 2D Gaussian ray trac-
ing and refining high-frequency specular details through
spherical mipmap-based directional encoding that captures
anisotropic highlights. Extensive experiments demonstrate
state-of-the-art performance in geometry reconstruction,
material separation, and photorealistic relighting under
novel illuminations, outperforming existing inverse render-
ing approaches.
1. Introduction
Inverse rendering of glossy objects poses formidable chal-
lenges in computer vision and graphics. These surfaces ex-
hibit strong view-dependent effects, such as specular high-
lights, environmental reflections, and inter-reflections, that
inherently undermine multi-view consistency. This funda-
mental conflict severely impedes the accurate recovery of
geometry and material properties from multi-view imagery.
Although methods [1–10] based on Neural Radiance Fields
(NeRF) demonstrate promise in modeling complex light
transport, their reliance on dense ray sampling compromises
computational efficiency for practical deployment.
Recent advances in 3D Gaussian Splatting (3DGS) [11]
deliver efficient scene reconstruction through explicit rep-
resentations and real-time rendering.
However, subse-
quent methods exhibit various limitations:
certain ap-
proaches [12–17] exhibit limitations for glossy objects,
while others achieve plausible geometry reconstruction for
glossy surfaces, yet either fail to perform material decompo-
sition [18–21] or persistently yield blurred decomposition
results despite enabling inverse rendering [22–27]. Specifi-
cally, geometry reconstruction is severely compromised by
view-dependent specular reflections that violate multi-view
consistency. This manifests itself as both high-frequency
surface noise and structural artifacts, particularly in inter-
reflection regions.
Simultaneously, material decomposi-
arXiv:2508.14563v1  [cs.CV]  20 Aug 2025

<!-- page 2 -->
tion is impaired by a simplified rendering equation and
neglected indirect illumination, leading to blurred albedo,
metallic, and roughness estimates that yield physically im-
plausible relighting results.
We propose GOGS, a novel geometry prior-guided
framework based on 2D Gaussian surfels that resolves these
limitations through a two-stage pipeline. First, we establish
robust surface reconstruction through physics-based render-
ing with split-sum approximation, enhanced by geometric
priors from foundation models to explicitly enforce curva-
ture continuity and mitigate ambiguities under specular in-
terference. Subsequently, we perform physically accurate
material decomposition on this optimized geometry by eval-
uating the full rendering equation via Monte Carlo impor-
tance sampling, where visibility and indirect radiance are
computed through differentiable 2D Gaussian ray tracing.
To address fidelity limit in specular rendering, we further
introduce a spherical mipmap-based specular compensation
mechanism that adaptively refines high-frequency details.
Extensive experiments demonstrate state-of-the-art per-
formance in both geometry accuracy and material decom-
position. Our contributions are as follows:
• A robust geometric reconstruction method for glossy ob-
jects that mitigates geometry ambiguities by leveraging
geometric priors and split-sum approximation;
• Physically-based material decomposition via the full ren-
dering equation evaluation with Monte Carlo importance
sampling;
• An adaptive specular compensation mechanism that di-
rectionally refines high-frequency details, mitigating fi-
delity limits in specular rendering.
2. Related Work
Novel View Synthesis.
Novel View Synthesis gener-
ates unseen scene perspectives from limited input images.
Neural Radiance Fields (NeRF) [1] pioneered photoreal-
istic synthesis through volumetric rendering with implicit
neural representations.
Subsequent work advanced three
directions:
geometry enhancement [2], rendering qual-
ity [3, 4], and accelerated training and rendering via hy-
brid representations such as hash grids and voxels [5, 6].
3D Gaussian Splatting (3DGS) [11] revolutionized NVS
with anisotropic 3D Gaussians and tile-based rasterization,
achieving state-of-the-art quality and speed.
Follow-ups
further improved geometric fidelity through surface regular-
ization [12, 16], anti-aliasing [13], and texture and gradient
optimization [14, 15]. Addressing 3DGS’s geometric in-
consistencies, 2D Gaussian Splatting (2DGS) [17] projects
disks onto explicit surfaces with local smoothing for view-
consistent reconstruction. Our work adopts 2D Gaussian
primitives to improve geometric accuracy.
Inverse Rendering of Glossy Objects. Inverse rendering
estimates scene geometry, materials, and lighting from im-
ages but suffers from material-light ambiguities and com-
plex light interactions on glossy surfaces. NeRF-based ap-
proaches employ neural radiance fields with ray marching
but incur computational bottlenecks from dense sampling.
Ref-NeRF [7] simplifies view-dependent effects via direc-
tional encoding, yet omits material-light decomposition.
TensoIR [10] uses tensor factorization for geometry and ray
marching for indirect illumination, while ENVIDR [9] em-
ploys surface-based modeling of glossy reflections via a de-
composed neural renderer. NeRO [8] reconstructs geometry
using split-sum approximation without masks, utilizing in-
tegrated directional encoding for illumination, then recov-
ers BRDF and lighting via Monte Carlo sampling. Despite
flexibility, these implicit methods incur significant overhead
from dense ray sampling.
Recent 3DGS-based methods significantly advance in-
verse rendering. Pioneering works [22, 23] apply simplified
per-primitive rendering to Gaussians but face geometric and
material fidelity limitations. Subsequent methods demon-
strate divergent strengths: 3DGS-DR [25] optimizes high-
frequency details via deferred shading; Ref-Gaussian [26]
uses split-sum approximation for glossy surfaces, inspir-
ing our geometry reconstruction.
Parallel developments
integrate specialized priors such as microfacet segmenta-
tion [28], foundation model supervision [29], and diffusion
models [30] to reduce geometry ambiguities, motivating our
geometry-aware initialization. R3DG [24] models the full
rendering equation, extended by IRGS [27] with differen-
tiable 2D Gaussian ray tracing [31] for precise visibility and
indirect illumination, a technique adopted in our pipeline.
Finally, spherical mipmap encoding and directional factor-
ization [21, 32] enhance specular effects, guiding our spec-
ular compensation.
3. Preliminary
3.1. 2D Gaussian Splatting
2D Gaussian Splatting (2DGS) [17] models scenes using
oriented planar disks derived from projected 3D Gaussian
distributions, establishing view-consistent geometry with
explicit surface normals defined as the direction of steep-
est density change. Each primitive is parameterized by: a
center p ∈R3, orthogonal tangent vectors tu and tv defin-
ing disk orientation, and scaling factors su, sv controlling
axial variances. The combined rotation and scaling trans-
formations are represented by a covariance matrix Σ.
Unlike volumetric rendering, 2DGS employs explicit
ray-splat intersections to evaluate Gaussian contributions
directly on 2D disks, enabling perspective-correct splatting
while mitigating perspective distortion. The Gaussian func-
tion is defined as:
G(u) = exp

−1
2(u2 + v2)

,
(1)

<!-- page 3 -->
Figure 2. Overview of : Stage I (Sec. 4.1) reconstructs geometry with 2DGS using split-sum shading, supervised by foundation model
priors with geometric curvature losses. Stage II (Sec. 4.2) performs inverse rendering on fixed geometry via Monte Carlo sampling,
utilizing 2D Gaussian ray tracing for visibility and indirect illumination and spherical mipmap-based compensation for high-frequency
specular details. The pipeline’s decoupled geometry-material optimization enables high-fidelity relighting.
where u = (u, v) denotes the local UV coordinates of the
ray-disk intersection.
Rendering uses front-to-back alpha blending of depth-
sorted Gaussians with degeneracy handling:
c(r) =
N
X
i=1
ciαi ˆGi(u)
i−1
Y
j=1

1 −αj ˆGj(u)

,
(2)
where N is the number of overlapping Gaussians, αi the
opacity, and ci the view-dependent color of the i-th Gaus-
sian.
3.2. Physically Based Deferred Shading
In physically-based Gaussian splatting rendering, shading-
rasterization order critically impacts fidelity.
We adopt
deferred shading to decouple this process:
a geometry
pass stores attributes in G-buffers, followed by a per-pixel
lighting pass.
Building on 3DGS-DR’s pioneering de-
ferred shading integration [25], our method extends it to
physically-based rendering using Disney BRDF [33]. Each
2D Gaussian carries: albedo λ ∈[0, 1]3; metallic m ∈
[0, 1]; roughness r ∈[0, 1]; surface normal n = tu × tv;
position xi; and feature vector ki that aggregates into the
feature map K utilized in the second stage (Sec. 4.2) to en-
hance specular effects.
We render attributes into G-buffers using alpha-blending
from Eq. 2, replacing color ci with f i:
F =
N
X
i=1
f iαi ˆGi(u(r))
i−1
Y
j=1

1 −αj ˆGj(u(r))

,
(3)
where
f i
=
[λi, mi, ri, ni, xi, ki]⊤
contains
per-
Gaussian properties, yielding aggregated attributes F =
[Λ, M, R, N, D, K]⊤.
Using aggregated G-buffer at-
tributes, we compute outgoing radiance Lo at x with normal
n via the rendering equation [34]:
Lo (ωo, x) =
Z
Ω
f (ωo, ωi, x) Li (ωi, x) (ωi · n)dωi,
(4)
where f is the Disney BRDF, consisting of a diffuse term
fd and a specular term fs:
fd = Λ
π (1 −M)(1 −Fd),
(5)
fs (ωi, ωo, x) =
DFG
4 (ωi · n) (ωo · n).
(6)
Here, D, F, and G represent the microsurface normal distri-
bution, Fresnel reflectance, and geometry attenuation terms,
respectively, with the diffuse Fresnel factor Fd enforcing
energy conservation by deducting specular-reflected energy.
These terms are computed using the stored roughness R and
metallic M values, following the energy-conserving prop-
erties of the Disney BRDF model [33].

<!-- page 4 -->
4. Method
We present a novel inverse rendering framework for re-
constructing reflective objects from multi-view RGB im-
ages under unknown illumination. As illustrated in Fig. 2,
our approach decomposes the problem into two sequential
stages: First, geometry reconstruction leverages deferred
Gaussian splatting with split-sum approximation for effi-
cient physically-based rendering while incorporating geo-
metric priors from foundation models to mitigate geometry
ambiguities (Sec. 4.1). Second, with object geometry fixed,
we refine material properties through Monte Carlo impor-
tance sampling of the full rendering equation to optimize
BRDF parameters, enhanced by a spherical mipmap-based
specular compensation mechanism for complex specular ef-
fects (Sec. 4.2).
4.1. Stage I: Geometry reconstruction
Split-sum Approximation. Within deferred shading us-
ing aggregated per-pixel material attributes, we compute the
specular term via split-sum approximation [35] to mitigate
its computational cost. This decomposes the rendering in-
tegral into:
Ls(ωo) ≈
Z
Ω
fs(·)(ωi·n)dωi
|
{z
}
BRDF factor
·
Z
Ω
Li(·)D(·)(ωi·n)dωi
|
{z
}
Lighting factor
.
(7)
The BRDF factor, dependent on material roughness R and
normal angle (ωi ·n), is precomputed into a 2D lookup tex-
ture. The lighting factor integrates environmental radiance
over the specular lobe using trilinear interpolation in pre-
filtered Mipmap cubemaps, indexed by reflected direction
ωr = 2(n · ωo)n −ωo and roughness R. While com-
putationally efficient, this approximation inherently lim-
its high-frequency specular modeling, particularly on low-
roughness surfaces, necessitating refinement with the full
rendering equation.
Supervision from Foundation Models. To mitigate ge-
ometry ambiguities in specular surface reconstruction, we
leverage robust geometric priors from large-scale vision
foundation models [36, 37]. These models generate monoc-
ular depth ˜D and normal ˜N estimates invariant to view-
dependent effects, providing reliable supervision even for
highly reflective surfaces where multi-view consistency
fails.
We incorporate these priors as regularization during
Gaussian optimization. The predicted normals ˜N supervise
rendered normals ˆN via a dual loss:
Lgeo-n = ∥ˆN −˜N∥1 + λ
 
1 −
ˆN · ˜N
∥ˆN∥∥˜N∥
!
,
(8)
where λ balances magnitude and angular alignment. For
depth supervision, we employ a scale-invariant formula-
tion [38] with predicted normals ˜D and rendered normals
ˆD:
Lgeo-d = min
ω,b
X
p
h
(ω ˆD + b) −˜D
i2
,
(9)
with ω and b optimized per-view via least squares.
As
demonstrated in Fig. 3, this geometric regularization explic-
itly enhances curvature continuity while suppressing sur-
face noise from specular interference.
Training Loss. The complete training objective integrates
physical rendering losses with our geometric priors:
L1 =Lc + λnLn + λoLo + λsmoothLsmooth
+ λgeo-nLgeo-n + λgeo-dLgeo-d.
(10)
Following 2DGS [17], we utilize its core components in-
cluding the RGB reconstruction loss Lc
=
∥crender −
Cgt∥1 and the normal alignment loss defined as Ln =
1 −˜N
TN. Lo is a binary cross-entropy loss constrain-
ing the geometry using the provided object mask M de-
fined as Lo = −M log O −(1 −M) log(1 −O), where
O = PN
i=1 Tiαi denotes the accumulated opacity map.
To provide joint regularization for both normals and depth,
we incorporate an edge-aware smoothness term Lsmooth =
∥∇N∥exp (−∥∇Cgt∥). Finally, our novel geometric regu-
larizers Lgeo-n and Lgeo-d, defined in Eq. 8 and Eq. 9, enforce
alignment with foundation model priors.
Figure 3. Per-scene qualitative comparisons of normals.
4.2. Stage II: Inverse Rendering
Building upon the reconstructed geometry from Stage I 4.1,
we now refine the initial BRDF estimation through phys-
ically accurate inverse rendering. This stage employs the
full rendering equation to precisely optimize material pa-
rameters, including metalness m, albedo a, and roughness
ρ, while leveraging Monte Carlo importance sampling for
efficient incident radiance computation and rendering equa-
tion evaluation.
Sampling Strategy. We implement two physically-based
importance sampling strategies for optimal variance reduc-
tion. For the diffuse component, we follow the Lambertian

<!-- page 5 -->
Table 1. Quantitative comparison of average novel view synthesis metrics on synthesized test views. Deeper red indicates
better performance. Ours(geo) corresponds to results from our first-stage geometric reconstruction (Sec. 4.1), while Ours(ir)
corresponds to those from our second-stage inverse rendering (Sec. 4.2).
Datasets
Shiny Synthetic[7]
Glossy Synthetic[8]
Shiny Real[7]
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Ref-NeRF[7]
33.13
0.961
0.080
25.65
0.905
0.112
23.62
0.646
0.239
ENVIDR[9]
33.46
0.979
0.046
29.06
0.947
0.060
23.00
0.606
0.332
2DGS[17]
29.58
0.946
0.084
26.07
0.918
0.088
24.15
0.661
0.292
GShader[22]
31.97
0.958
0.067
27.11
0.922
0.145
23.46
0.647
0.257
R3DG[24]
27.77
0.926
0.112
24.13
0.892
0.106
21.98
0.619
0.349
3DGS-DR[25]
34.00
0.972
0.059
28.73
0.948
0.058
24.37
0.678
0.232
Ref-Gaussian[26]
34.66
0.972
0.055
30.77
0.962
0.045
24.71
0.691
0.263
IRGS[27]
28.39
0.932
0.110
24.40
0.892
0.109
21.38
0.425
0.486
Ours(geo)
35.03
0.975
0.055
30.83
0.962
0.048
26.56
0.783
0.178
Ours(ir)
32.21
0.959
0.078
29.38
0.947
0.060
24.47
0.687
0.299
distribution with sampling probability:
pd(ωi) = cos θ
π
,
(11)
while for the specular component, we employ GGX normal
distribution-based sampling [39]:
ps(ωi) = D(h) cos θ
4(ωo · h) .
(12)
To minimize variance in transitional regions between dif-
fuse and specular responses, we integrate both sampling
strategies through a balance heuristic [40]. Defining strat-
egy proportions πd = Nd/(Nd + Ns) and πs = Ns/(Nd +
Ns), the weight for a sample ωi from strategy k ∈{d, s} is
computed as:
wk(ωi) =
πkpk(ωi)
P
m∈{d,s} πmpm(ωi).
(13)
The final radiance Lphys
o
estimate combines both contribu-
tions:
Ldiffuse
o
= 1
Nd
Nd
X
i=1
fdLi(n · ωi)wd
pd
,
(14)
Lspecular
o
= 1
Ns
Ns
X
j=1
fsLj(n · ωj)ws
ps
,
(15)
Lpbr
o
= Ldiffuse
o
+ Lspecular
o
.
(16)
This formulation adaptively balances sample contributions
based on local material properties, significantly reducing
noise while maintaining physical accuracy.
Light parametrization. The incident radiance Li(ωi, x) at
surface point x along direction ωi (Eq. 4) decomposes into
direct radiance from distant sources and indirect radiance
from scene surfaces:
Li (ωi, x) = V (ωi, x) Ldir (ωi) + Lind (ωi, x) ,
(17)
where direct radiance Ldir is parameterized via an environ-
ment cubemap, while visibility V and indirect radiance Lind
are computed through IRGS’s differentiable 2D Gaussian
ray tracing [27].
Specular Compensation. To mitigate high-frequency ar-
tifacts from Monte Carlo sampling variance, specifically
material parameter noise and specular-diffuse separation er-
rors, we introduce a compensation mechanism. Drawing in-
spiration from the directional factorization paradigm [21],
we leverage the blended feature map K stored in the G-
buffer (Eq. 3) and a directionally encoded feature h queried
via spherical Mip-grid M:
h = M (θr(N, ωo), ϕr(N, ωo), R) ,
(18)
where (θr, ϕr) are spherical coordinates of ωr, computed
using the blended normal N from the G-buffer and view-
ing direction ωo, with R representing the optimized surface
roughness. The compensation radiance Lc is synthesized
by:
Lc = fc (K, h) .
(19)
This term refines physical shading through additive blend-
ing:
Lfinal
o
= Lpbr
o
+ Lc.
(20)
It is noteworthy that Lc is explicitly disabled during re-
lighting with novel illumination and operates exclusively
within the inverse rendering optimization loop to enhance
reconstruction fidelity for difficult-to-model specular phe-
nomena.
Training Loss. We define the total loss L2 as:
L2 = Lc + λsmoothLsmooth + λlightLlight.
(21)
We retaining the RGB reconstruction loss Lc and smooth-
ness regularization Lsmooth from the first stage. The edge-
aware smoothness regularization Lsmooth is now notably ap-
plied to rendered material properties including albedo (Λ),
roughness (R), and metallic (M) maps to enforce coher-
ent surface characteristics. The lighting regularization term
Llight imposes a neutral white prior on diffuse illumina-

<!-- page 6 -->
Figure 4. Qualitative decomposition results of our model.
Figure 5. Qualitative comparisons of NVS renderings and
normal maps, focusing on geometric accuracy and indirect
illumination in inter-reflection regions.
tion Ldiffuse =
1
Nr
PNr
i=1 L (ωi, x) by minimizing RGB
deviations from their mean value, formulated as Llight =
P
c

Lc
diffuse −1
3
P
c′ Lc′
diffuse

where c ∈{R, G, B}. Fol-
lowing IRGS [27] for computational efficiency, we restrict
rendering equation evaluations to a subset of pixels per iter-
ation, sampling ⌊Nrays/Nr⌋pixels through a maximum ray
budget Nrays to enable high-quality estimation with large
Nr.
Our
physically-based
inverse
rendering
framework
achieves precise material decomposition through joint
BRDF-geometry optimization.
The recovered materials
maintain intrinsic physical consistency, enabling faithful re-
lighting and high-fidelity rendering under novel illumina-
tion.
5. Experiment
Datasets and metrics. We utilize two synthetic datasets,
Shiny Blender [7] and Glossy Synthetic [8], along with the
real-world Shiny Real dataset [7] to evaluating novel view
synthesis of reflective objects. For quantitative assessment,
we employ three standard image quality metrics: PSNR,
SSIM [41], and LPIPS [42], complemented by Mean An-
gular Error (MAE) for normal estimation accuracy.
Table 2. Quantitative comparison on normal maps
Metrics
Model
GShader[22]
3DGS-DR[25]
Ref-Gaussian[26]
Ours
MAE ↓
4.74
9.49
2.28
2.16
SSIM ↑
0.853
0.824
0.924
0.924
LPIPS ↓
0.108
0.172
0.073
0.072
Implementation details. All experiments run on a single
NVIDIA RTX 3090 GPU. Our training pipeline follows the
two-stage process in Sec 4. The first stage trains for 50,000
iterations with hyperparameters: λn = 0.05, λd = 0.05,
λsmooth = 0.01, λgeo-n = 0.005, and λgeo-d = 0.005. We
use Marigold fine-tuned for normal and depth estimation
due to its superior object-level prediction.
The environ-
mental lighting employs a 6 × 128 × 128 mipmap cube-
map with 3 RGB channels.
The second stage performs
20,000 iterations of inverse rendering with λsmooth = 2.0
and λlight = 0.01. The learning rates are set to 0.0075 for
albedo, 0.005 for roughness, 0.005 for metallic, and 0.01
for cubemap. Ray tracing uses IRGS [27] hyperparameters
while other settings follow 2DGS [17]. The output at this
stage enables direct relighting with ray samples increased to
2048 for improved quality. For enhanced novel view syn-
thesis, we perform up to 80,000 specular compensation it-
erations with geometry, materials and lighting frozen. Only
the spherical encoding components are updated, including
an 8 × 512 × 512 mipmap with 16 feature channels and
a shallow MLP that has two 256-unit hidden layers. The
total training without specular compensation takes approxi-
mately 80 minutes: 40 minutes for Stage I 4.1 and 40 min-
utes for Stage II 4.2. Specular compensation requires an
additional hour.
5.1. Comparison
Novel View Synthesis. Tab. 1 shows quantitative compar-
isons against SOTA across multiple datasets, including av-
erage metric scores per dataset. Our first-stage (Sec. 4.1)
consistently outperforms SOTA across all metrics, with no-

<!-- page 7 -->
Table 3. Quantitative comparisons of relighting results in terms of PSNR↑on the Glossy Synthetic dataset [8].
Datasets
angel
bell
cat
horse
luyu
potion
tbell
teapot
avg.
corridor
GShader[22]
21.83
22.98
16.42
26.42
16.74
14.99
18.74
20.35
19.81
Ref-Gaussian[26]
20.97
23.02
20.57
25.45
20.34
20.22
21.51
22.59
21.83
IRGS[27]
20.31
21.16
22.40
22.25
24.65
24.53
20.71
20.05
22.01
Ours
23.92
25.76
27.32
25.80
25.11
27.42
24.61
26.05
25.75
golf
GShader[22]
21.04
21.40
14.84
25.01
14.85
12.65
16.80
18.50
18.14
Ref-Gaussian[26]
22.32
23.05
21.57
25.87
19.41
19.85
19.96
21.54
21.69
IRGS[27]
20.58
20.89
21.96
22.24
22.64
22.52
18.90
18.56
21.04
Ours
26.37
26.99
27.35
26.28
24.85
26.14
23.91
26.19
26.01
neon
GShader[22]
21.16
21.41
16.74
24.12
16.87
15.59
18.94
19.07
19.24
Ref-Gaussian[26]
21.20
21.85
20.56
23.32
20.41
20.23
21.65
21.64
21.36
IRGS[27]
20.30
19.75
21.33
21.29
20.27
20.69
18.71
18.64
20.12
Ours
21.72
21.36
25.31
24.52
21.65
23.66
21.87
22.93
22.88
Figure 6. Qualitative comparisons of the estimated environ-
ment maps.
table gains on Glossy Synthetic [8] due to geometric pri-
ors.
Although second-stage results (Sec. 4.2) exhibit a
modest decrease in NVS performance, they yield signifi-
cantly more accurate material decomposition, as validated
by subsequent relighting experiments.
Fig. 4 illustrates
the output decomposition: geometry (via normals), mate-
rial properties, visibility, indirect illumination, and specular
compensation. Our normals are smoother and more accu-
rate, while material decomposition adheres better to physi-
cal constraints, effectively disentangling geometry and ma-
terials for photorealistic rendering.
Fig. 6 compares the
estimated environment maps. Our method effectively dis-
entangles illumination from scene properties. Versus Ref-
Gaussian [26] and GShader [22], we obtain more refined
maps with sharper details, fewer artifacts, and closer to
ground truth.
Geometry Reconstruction. Tab. 2 quantitatively validates
our superior normal map quality, attributable to first-stage
geometry optimization. Fig. 3 and Fig. 5 further illustrate
our advantage. The qualitative comparison in Fig. 3 demon-
strates our method’s comprehensive detail capture, such as
Figure 7. Qualitative comparison of relighting results on the
Glossy Synthetic dataset [8].
the base of a cat, cup bottom, and horse back. Fig. 5 com-
pares our approach to competitors on glossy objects. While
competitors achieve adequate novel view synthesis, they fail
to reconstruct accurate geometry or precise indirect illumi-
nation in inter-reflection regions. In contrast, our method
leverages geometric priors to recover precise inter-reflection
geometry and generates physically consistent indirect illu-
mination via ray tracing, producing realistic light bounces
(red boxes) absent in competitors. Our solution consistently
delivers accurate normals and photorealistic indirect illumi-
nation, critical for high-quality novel view synthesis in re-
flective scenes.
Relighting. Tab. 3 shows our method achieves superior re-
lighting quality on the Glossy Synthetic dataset [8], outper-
forming Gaussian-based approaches. Qualitative results in
Fig. 7 demonstrate our inverse rendering produces realistic
specular highlights where competitors fail: GShader [22]
exhibits geometric and material artifacts, IRGS [27] is lim-
ited to diffuse objects, and Ref-Gaussian’s simplified ren-
dering equation [26] yields inaccurate material decompo-
sition despite strong geometry. Fig. 8 compares our ”cat”
reconstruction with GS-based methods [22, 26], showcas-
ing high-fidelity material estimation. Additional relighting

<!-- page 8 -->
Figure 8. Qualitative comparison of normal, material, and lighting estimation, and relighting results (using the Corri-
dor, Golf, and Neon light maps) on the “cat” scene of the Glossy Synthetic Dataset [8].
Figure 9. Relighting results on the Glossy Synthetic [8] and
Shiny Blender dataset [7].
results for low-metallic objects (Fig. 9) further validate the
versatility of our method across diverse material types.
5.2. Ablation Study
In Tab. 4, we conduct an ablation study to demonstrate the
contributions of our method. First, we ablate the geometric
prior supervision in the first stage and observe a minor per-
formance drop—this is because the geometric prior super-
vision primarily acts on inter-reflection regions, which have
limited impact on overall performance. Second, we ablate
the importance sampling in the inverse rendering stage and
Table 4. Ablation study on the Glossy Synthetic dataset [8].
PSNR↑
SSIM↑
LPIPS↓
w/o geometry prior
29.36
0.947
0.061
w/o important sampling
28.09
0.930
0.080
w/o specular compensation
28.58
0.939
0.067
Ours
29.38
0.947
0.061
notice a substantial performance drop, demonstrating the
necessity of importance sampling for smooth object recon-
struction and accurate estimation of materials and illumina-
tion. Finally, we ablate the specular compensation and once
again observe a performance drop, which highlights its ben-
efit to photorealistic reconstruction of glossy objects.
6. Conclusion
We present GOGS, a geometry prior-guided Gaussian
surfel framework addressing geometry-material ambigui-
ties in inverse rendering of glossy objects. Our two-stage
approach first establishes robust geometry reconstruction
under specular interference by integrating foundation
model priors with physics-based rendering using split-sum
approximation.
Subsequently, we perform physically-
based material decomposition via Monte Carlo importance
sampling of the full rendering equation, leveraging dif-
ferentiable 2D Gaussian ray tracing.
Additionally, a
spherical mipmap-based directional encoding mechanism
adaptively compensates for high-frequency specular de-
tails.
Extensive experiments demonstrate state-of-the-art
performance
in
geometry
accuracy,
material
separa-
tion, and photorealistic relighting under novel illumination.

<!-- page 9 -->
References
[1] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.
Nerf: Representing scenes as neural radiance fields
for view synthesis.
Communications of the ACM,
65(1):99–106, 2021. 1, 2
[2] Peng Wang,
Lingjie Liu,
Yuan Liu,
Christian
Theobalt, Taku Komura, and Wenping Wang. Neus:
Learning neural implicit surfaces by volume render-
ing for multi-view reconstruction.
arXiv preprint
arXiv:2106.10689, 2021. 2
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik,
Peter Hedman, Ricardo Martin-Brualla, and Pratul P
Srinivasan. Mip-nerf: A multiscale representation for
anti-aliasing neural radiance fields. In Proceedings of
the IEEE/CVF international conference on computer
vision, pages 5855–5864, 2021. 2
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin,
Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360:
Unbounded anti-aliased neural radiance fields. In Pro-
ceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 5470–5479, 2022.
2
[5] Thomas M¨uller, Alex Evans, Christoph Schied, and
Alexander Keller. Instant neural graphics primitives
with a multiresolution hash encoding. ACM transac-
tions on graphics (TOG), 41(4):1–15, 2022. 2
[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu,
and Hao Su. Tensorf: Tensorial radiance fields. In
European conference on computer vision, pages 333–
350. Springer, 2022. 2
[7] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd
Zickler, Jonathan T Barron, and Pratul P Srinivasan.
Ref-nerf: Structured view-dependent appearance for
neural radiance fields. In 2022 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR),
pages 5481–5490. IEEE, 2022. 2, 5, 6, 8, 13
[8] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long,
Jiepeng Wang, Lingjie Liu, Taku Komura, and Wen-
ping Wang. Nero: Neural geometry and brdf recon-
struction of reflective objects from multiview images.
ACM Transactions on Graphics (ToG), 42(4):1–22,
2023. 2, 5, 6, 7, 8, 13
[9] Ruofan Liang, Huiting Chen, Chunlin Li, Fan Chen,
Selvakumar Panneer, and Nandita Vijaykumar.
En-
vidr: Implicit differentiable renderer with neural en-
vironment lighting. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages
79–89, 2023. 2, 5, 13
[10] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang,
Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu,
and Hao Su.
Tensoir: Tensorial inverse rendering.
In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 165–174,
2023. 1, 2
[11] Bernhard
Kerbl,
Georgios
Kopanas,
Thomas
Leimk¨uhler, and George Drettakis.
3d gaussian
splatting for real-time radiance field rendering. ACM
Trans. Graph., 42(4):139–1, 2023. 1, 2
[12] Antoine Gu´edon and Vincent Lepetit. Sugar: Surface-
aligned gaussian splatting for efficient 3d mesh recon-
struction and high-quality mesh rendering.
In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 5354–5363,
2024. 1, 2
[13] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sat-
tler, and Andreas Geiger. Mip-splatting: Alias-free 3d
gaussian splatting. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recogni-
tion, pages 19447–19456, 2024. 2
[14] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaus-
sian opacity fields: Efficient adaptive surface recon-
struction in unbounded scenes. ACM Transactions on
Graphics (ToG), 43(6):1–13, 2024. 2
[15] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Wei-
jian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hu-
jun Bao, and Guofeng Zhang.
Pgsr: Planar-based
gaussian splatting for efficient and high-fidelity sur-
face reconstruction. IEEE Transactions on Visualiza-
tion and Computer Graphics, 2024. 2
[16] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli,
Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering.
In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20654–
20664, 2024. 2
[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas
Geiger, and Shenghua Gao. 2d gaussian splatting for
geometrically accurate radiance fields. In ACM SIG-
GRAPH 2024 conference papers, pages 1–11, 2024.
1, 2, 4, 5, 6
[18] Zhe Jun Tang and Tat-Jen Cham. 3igs: Factorised ten-
sorial illumination for 3d gaussian splatting. In Euro-
pean Conference on Computer Vision, pages 143–159.
Springer, 2024. 1
[19] Ziyi Yang, Xinyu Gao, Yang-Tian Sun, Yihua Huang,
Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan
Qi, and Xiaogang Jin.
Spec-gaussian: Anisotropic
view-dependent appearance for 3d gaussian splatting.
Advances in Neural Information Processing Systems,
37:61192–61216, 2024.
[20] Zoubin Bi, Yixin Zeng, Chong Zeng, Fan Pei, Xiang
Feng, Kun Zhou, and Hongzhi Wu. Gs3: Efficient re-
lighting with triple gaussian splatting. In SIGGRAPH
Asia 2024 Conference Papers, pages 1–12, 2024.

<!-- page 10 -->
[21] Youjia Zhang, Anpei Chen, Yumin Wan, Zikai Song,
Junqing Yu, Yawei Luo, and Wei Yang. Ref-gs: Direc-
tional factorization for 2d gaussian splatting. In Pro-
ceedings of the Computer Vision and Pattern Recog-
nition Conference, pages 26483–26492, 2025. 1, 2,
5
[22] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao,
Xiaoxiao Long, Wenping Wang, and Yuexin Ma.
Gaussianshader: 3d gaussian splatting with shading
functions for reflective surfaces. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 5322–5332, 2024. 1, 2, 5, 6,
7, 12, 13
[23] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and
Kui Jia. Gs-ir: 3d gaussian splatting for inverse ren-
dering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
21644–21653, 2024. 2
[24] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu,
Xun Cao, Li Zhang, and Yao Yao.
Relightable 3d
gaussians: Realistic point cloud relighting with brdf
decomposition and ray tracing.
In European Con-
ference on Computer Vision, pages 73–89. Springer,
2024. 2, 5
[25] Keyang Ye, Qiming Hou, and Kun Zhou. 3d gaussian
splatting with deferred reflection. In ACM SIGGRAPH
2024 Conference Papers, pages 1–10, 2024. 2, 3, 5, 6,
13
[26] Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, and
Li Zhang. Reflective gaussian splatting. arXiv preprint
arXiv:2412.19282, 2024. 2, 5, 6, 7, 12, 13
[27] Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao,
and Li Zhang.
Irgs: Inter-reflective gaussian splat-
ting with 2d gaussian ray tracing. In Proceedings of
the Computer Vision and Pattern Recognition Confer-
ence, pages 10943–10952, 2025. 1, 2, 5, 6, 7, 13
[28] Shuichang Lai, Letian Huang, Jie Guo, Kai Cheng,
Bowen Pan, Xiaoxiao Long, Jiangjing Lyu, Chengfei
Lv, and Yanwen Guo. Glossygs: Inverse rendering
of glossy objects with 3d gaussian splatting.
IEEE
Transactions on Visualization and Computer Graph-
ics, 2025. 2
[29] Jinguang Tong, Xuesong Li, Fahira Afzal Maken,
Sundaram Muthu, Lars Petersson, Chuong Nguyen,
and Hongdong Li. Gs-2dgs: Geometrically supervised
2dgs for reflective object reconstruction. In Proceed-
ings of the Computer Vision and Pattern Recognition
Conference, pages 21547–21557, 2025. 2
[30] Kang Du, Zhihao Liang, and Zeyu Wang. Gs-id: Illu-
mination decomposition on gaussian splatting via dif-
fusion prior and parametric light source optimization.
arXiv preprint arXiv:2408.08524, 2024. 2
[31] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel,
Riccardode Lutio, JanickMartinez Esturo, Gavriel
State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic.
3d gaussian ray tracing: Fast tracing of particle scenes.
Oct 2024. 2
[32] Georgios Kouros, Minye Wu, and Tinne Tuyte-
laars.
Rgs-dr: Reflective gaussian surfels with de-
ferred rendering for shiny objects.
arXiv preprint
arXiv:2504.18468, 2025. 2
[33] Brent Burley and Walt Disney Animation Studios.
Physically-based shading at disney. In Acm siggraph,
volume 2012, pages 1–7. vol. 2012, 2012. 3
[34] James T. Kajiya.
The rendering equation, page
157–164. Jul 1998. 3
[35] Jacob Munkberg, Jon Hasselgren, Tianchang Shen,
Jun Gao, Wenzheng Chen, Alex Evans, Thomas
M¨uller, and Sanja Fidler.
Extracting triangular 3d
models, materials, and lighting from images. In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 8280–8290,
2022. 4
[36] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando
Metzger, RodrigoCaye Daudt, and Konrad Schindler.
Repurposing diffusion-based image generators for
monocular depth estimation. Dec 2023. 4, 11
[37] Gonzalo Martin Garcia, Karim Abou Zeid, Chris-
tian Schmidt, Daan De Geus, Alexander Hermans,
and Bastian Leibe.
Fine-tuning image-conditional
diffusion models is easier than you think.
In 2025
IEEE/CVF Winter Conference on Applications of
Computer Vision (WACV), pages 753–762. IEEE,
2025. 4, 11
[38] Ren´e Ranftl, Katrin Lasinger, David Hafner, Kon-
rad Schindler, and Vladlen Koltun. Towards robust
monocular depth estimation: Mixing datasets for zero-
shot cross-dataset transfer. IEEE transactions on pat-
tern analysis and machine intelligence, 44(3):1623–
1637, 2020. 4
[39] Robert L Cook and Kenneth E. Torrance.
A re-
flectance model for computer graphics. ACM Trans-
actions on Graphics (ToG), 1(1):7–24, 1982. 5
[40] Eric Veach.
Robust Monte Carlo methods for light
transport simulation. Stanford University, 1998. 5
[41] Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simon-
celli. Image quality assessment: From error visibility
to structural similarity. IEEE Transactions on Image
Processing, page 600–612, Apr 2004. 6, 13
[42] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli
Shechtman, and Oliver Wang. The unreasonable ef-
fectiveness of deep features as a perceptual metric. In
2018 IEEE/CVF Conference on Computer Vision and
Pattern Recognition, Jun 2018. 6, 13

<!-- page 11 -->
GOGS: High-Fidelity Geometry and Relighting for Glossy Objects
via Gaussian Surfels
Supplementary Material
This supplementary document complements the main
paper by providing extended qualitative and quantitative
analyses to further demonstrate the capabilities of our pro-
posed GOGS framework. Specifically, it expands on the
experimental results presented in Sec. 5 of the main pa-
per, including comprehensive visualizations of decomposed
model outputs across diverse reflective objects, comparative
assessments of recovered environment illumination maps,
and extensive relighting scenarios under novel illumination
conditions. Additionally, we discuss current methodologi-
cal limitations and future work.
A. Foundation Models
To mitigate geometric ambiguities under specular surfaces,
we leverage foundation models [36, 37] trained on large-
scale datasets for monocular depth and normal estimation.
These models generate robust geometric predictions by in-
ferring scene geometry from single-view images through
learned priors, bypassing the need for multi-view consis-
tency. As demonstrated in Fig. 10, the predicted depth and
normal maps accurately capture surface details across di-
verse materials including highly reflective regions. The pre-
dictions remain robust to complex specular interference and
provide reliable pseudo-ground truth for geometric supervi-
sion. This capability stems from foundation models’ ex-
posure to massive real-world variations during pretraining,
enabling them to disambiguate lighting-surface ambiguities
that degrade traditional multi-view reconstruction.
It is worth noting that two key factors introduce non-
Figure 10. Foundation models generate precise depth and
normal predictions resilient to specular interference.
determinism. First, depth and normal priors from external
foundation models exhibit inherent randomness during in-
ference, leading to minor variations in predicted geometric
cues across repeated runs; second, the depth loss (Eq. 9),
which solves for scale ω and shift b via least squares opti-
mization—is highly sensitive to minimal distributional dis-
crepancies between rendered depth and priors, and such dis-
crepancies can yield non-unique solutions for ω and b, caus-
ing fluctuations in alignment parameters during each itera-
tion. These small variations alter the direction of gradient
updates, which amplify over time and ultimately result in
differences in the final geometric reconstructions.
B. Comparison
Model Outputs.
In Fig. 12, we present detailed output
decompositions of our model applied to reflective objects.
From left to right, we show the ground truth (GT) for com-
parison, our model’s novel view synthesis (NVS) results,
pseudo-colored surface normals (for enhanced geometric
clarity), albedo (diffuse color), metallic, roughness, visi-
bility maps (encoding direct illumination access), indirect
illumination, and specular compensation (SpecComp) out-
puts. These results demonstrate our method’s ability to ef-
fectively disentangle geometry (normals), material proper-
ties (albedo, metallic, roughness), and illumination (visi-
Figure 11. Estimated environment maps for reflective object
scenes. Our method consistently produces sharper, more
artifact-free maps with clearer environmental details com-
pared to competing approaches.

<!-- page 12 -->
Figure 12. Outputs of our model including ground truth (GT), novel view synthesis (NVS), surface normals (visualized with
pseudo-colors), albedo, metallic, roughness, visibility, indirect illumination, and specular compensation (SpecComp).
bility, indirect, specular compensation) during inverse ren-
dering.
By separating these components, our model en-
ables high-quality reconstruction, photorealistic novel view
synthesis, and high-fidelity relighting—critical for handling
reflective surfaces where traditional methods often exhibit
limitations. The disentangled illumination and material out-
puts, in particular, allow precise adjustment of lighting con-
ditions without compromising surface appearance, a key ad-
vantage for applications like product visualization and vir-
tual reality.
Environment Maps. In Fig. 11, we present a comparison
of estimated environment maps for various reflective object
scenes (left column), including a teapot, horse, car, and hel-
met. From left to right after the scene column, we show
results from our model, GShader [22], Ref-Gaussian(Ref-
GS) [26]. Our method effectively disentangles illumina-
tion from scene geometry and material properties, resulting
in sharper environment maps with fewer artifacts—such as
the crisp ceiling patterns in the teapot scene, distinct win-
dow frames in the horse scene—compared to GShader’s
slightly blurred outputs [22] and Ref-Gaussian’s muted de-
tails [26].
Notably, these methods exhibit severe noise
and color distortion in most cases (e.g., the car and hel-
met scenes), while our model maintains consistent clarity
across all scenes. These results demonstrate our method’s
superior ability to recover high-fidelity environment illumi-
nation, which is critical for realistic rendering and relighting
of reflective surfaces—an essential capability for applica-
tions like product visualization and virtual reality.
Relighting. In Fig. 13, we demonstrate GOGS’s relight-
ing capabilities across surfaces with diverse reflectance
properties.
For each exemplar object, we preserve the
original rendering under default illumination (leftmost col-
umn), followed by relighting under three novel illumina-
tion conditions (subsequent columns). Our method consis-
tently disentangles illumination from intrinsic scene prop-
erties, maintaining geometric fidelity (e.g., fine structural
details) and material characteristics across lighting varia-
tions. Relighted outputs exhibit physically accurate reflec-
tions: highly specular surfaces precisely mirror environ-

<!-- page 13 -->
Table 5. Quantitative relighting comparisons on three environment maps (corridor, golf, neon) from Glossy Synthetic [8]
using three metrics (PSNR ↑, SSIM ↑[41], LPIPS ↓[42]), with best results per metric highlighted in red.
corridor
golf
neon
avg.
corridor
golf
neon
avg.
corridor
golf
neon
avg.
PSNR ↑
SSIM↑
LPIPS↓
GShader[22]
19.81
18.14
19.24
19.06
0.862
0.876
0.860
0.866
0.101
0.102
0.104
0.103
Ref-GS[26]
21.83
21.69
21.36
21.63
0.900
0.920
0.894
0.905
0.075
0.077
0.082
0.078
IRGS[27]
22.01
21.04
20.12
21.06
0.874
0.840
0.832
0.849
0.131
0.164
0.151
0.149
Ours
25.75
26.01
22.88
24.89
0.925
0.914
0.909
0.916
0.070
0.092
0.084
0.082
Figure 13. Our GOGS framework generates high-fidelity
relighting results across diverse objects, consistently pre-
serving geometric fidelity and material properties while
producing sharp, physically plausible specular reflections
under novel illumination conditions.
mental features, while matte surfaces retain color/texture
consistency.These results validate our model’s capability
for high-fidelity relighting of arbitrary materials under var-
ied illuminations.
Tab. 5 summarizes the relighting performance of four
methods (GShader [22], Ref-GS [26], IRGS [27], and
ours) on three representative environment maps (corridor,
golf, neon) using three widely adopted metrics: PSNR↑,
SSIM↑[41], and LPIPS↓[42].
For each metric, we re-
port results on individual environment maps and their av-
erage (avg.), with the best results highlighted in red. Con-
sistently, our method achieves the highest average PSNR
and SSIM across all environment maps, outperforming the
second-best method by significant margins. These results
validate the effectiveness of our method in disentangling
illumination from scene geometry and material properties,
which is critical for high-fidelity relighting applications like
product visualization and virtual reality. These results val-
idate our model’s capability for high-fidelity relighting of
arbitrary materials under varied illuminations.
Table 6. Quantitative comparison of training time efficiency on the
Shiny Blender dataset [7].
Model
Efficiency
evaluation
ENVIDR[9]
3DGS-DR[25]
Ref-GS[26]
IRGS[27]
Ours
Training
time (h)↓
5.84
0.35
0.58
0.70
1.33
C. Limitation
Our framework has certain limitations requiring further in-
vestigation. While optimized for specular reflections, our
Monte Carlo importance sampling exhibits suboptimal per-
formance on complex mixed-material surfaces due to dif-
fering BRDF characteristics. To address this, future work
will focus on adaptive importance sampling techniques ca-
pable of handling complex mixed materials and general-
izing across material roughness.
Additionally, computa-
tional costs from our ray tracing implementation and spher-
ical encoding currently preclude real-time applications, de-
spite offering significant speed improvements over NeRF-
based methods. Tab. 6 quantifies this trade-off: our model
achieves a training time of 1.33 hours without specular
compensation—far faster than NeRF-based ENVIDR (5.84
hours)—but slower than lightweight models like 3DGS-DR
(0.35 hours) and GShader (0.48 hours). This aligns with our
observation that while we outperform NeRF-based meth-
ods, real-time capability remains a challenge. To address
this, we plan to accelerate rendering by baking incident
radiance through precomputed radiance transfer schemes
while maintaining physical accuracy.Finally, while our 2D
Gaussian ray tracing approximates visibility and indirect
illumination, it remains insufficient for capturing high-
fidelity multi-bounce specular inter-reflections. Real-world
light transport in glossy surfaces involves theoretically infi-
nite path bounces—a phenomenon prohibitively expensive
to simulate using path-tracing in inverse rendering due to its
computational burden. To address this limitation, we will
explore novel neural network-based approximation tech-
niques for modeling complex light paths.
