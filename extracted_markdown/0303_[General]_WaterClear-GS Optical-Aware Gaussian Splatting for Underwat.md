<!-- page 1 -->
1
WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater
Reconstruction and Restoration
Xinrui Zhang, Yufeng Wang, Shuangkang Fang, Zesheng Wang, Dacheng Qi, Wenrui Ding
xr zhang@buaa.edu.cn, Beihang University
Abstract—Underwater
3D
reconstruction
and
appearance
restoration are hindered by the complex optical properties of
water, such as wavelength-dependent attenuation and scattering.
Existing Neural Radiance Fields (NeRF)-based methods struggle
with slow rendering speeds and suboptimal color restoration,
while 3D Gaussian Splatting (3DGS) inherently lacks the capa-
bility to model complex volumetric scattering effects. To address
these issues, we introduce WaterClear-GS, the first pure 3DGS-
based framework that explicitly integrates underwater optical
properties of local attenuation and scattering into Gaussian
primitives, eliminating the need for an auxiliary medium net-
work. Our method employs a dual-branch optimization strategy
to ensure underwater photometric consistency while naturally
recovering water-free appearances. This strategy is enhanced
by depth-guided geometry regularization and perception-driven
image loss, together with exposure constraints, spatially-adaptive
regularization, and physically guided spectral regularization,
which collectively enforce local 3D coherence and maintain
natural visual perception. Experiments on standard benchmarks
and our newly collected dataset demonstrate that WaterClear-GS
achieves outstanding performance on both novel view synthesis
(NVS) and underwater image restoration (UIR) tasks, while
maintaining real-time rendering. The code will be available at
this https URL.
Index Terms—Underwater 3D reconstruction, underwater
restoration, novel view synthesis, Gaussian Splatting
I. INTRODUCTION
U
NDERWATER 3D reconstruction presents unique op-
portunities for marine exploration [1], [2], autonomous
underwater navigation [3], [4], and archaeological research [5],
[6]. However, the complex optical properties of the underwater
environment pose significant challenges. Underwater imaging,
in contrast to clear-air environments, suffers from severe
degradation due to wavelength-dependent light attenuation and
scattering effects caused by water molecules and suspended
particles. These phenomena result in pronounced color distor-
tion, diminished contrast, and suppression of fine structural
details, as shown in Figure 1. While traditional single-image
underwater restoration methods [7], [8], [9], [10], [11], [12]
can leverage statistical priors or data-driven learning to miti-
gate degradation effects, their direct application to underwater
3D scenes inevitably compromises multi-view consistency due
to the inherent frame-by-frame processing paradigm.
Neural Radiance Fields (NeRF) [13] has revolutionized 3D
scene representation, enabling high-quality rendering from
multi-view images. Recent underwater 3D reconstruction
methods have sought to integrate the physics-based underwater
imaging models into the NeRF framework [14], [15], [16].
However, these approaches encounter several fundamental lim-
itations in underwater scattering scenarios, such as artifacts in
Camera I
Object J
Moving obstacles
Suspended particles
Object distance Z
     Attenuation 
  Backscatter
160+ FPS
Fig. 1. Illustration of underwater imaging and our results. Top: Light from
objects is affected by significant attenuation and scattering, which intensify
with distance. Bottom: Our method enables high-quality rendering at over 160
FPS, effectively removes scattering effects, and restores true colors.
volumetric rendering, expensive ray sampling procedures, and
slow convergence in underwater scenes due to the additional
modeling complexity of water medium effects. To overcome
these limitations, recent efforts have transitioned from NeRF
to 3D Gaussian Splatting (3DGS) [17], which offers superior
rendering efficiency. Pioneering underwater 3DGS variants
[18], [19], [20], [21] have attempted to model water medium
effects through auxiliary neural networks or volumetric fields.
However, these solutions introduce additional shortcomings:
(1) compromised training and rendering efficiency of 3DGS,
and (2) more critically, suboptimal color restoration quality
that ultimately restricts their real-world applicability.
To this end, we present WaterClear-GS, the first underwater
3DGS framework that intrinsically embeds optical physics
into Gaussian primitives. Unlike hybrid approaches that patch
3DGS with heavy neural networks, our method introduces a
purely explicit optical-aware Gaussian modeling. Each prim-
itive intrinsically encodes the wavelength-specific attenuation
and backscatter coefficients, enabling direct learning of under-
water optical effects without any auxiliary neural components.
We further propose a dual-branch optimization strategy that
arXiv:2601.19753v1  [cs.CV]  27 Jan 2026

<!-- page 2 -->
2
simultaneously optimizes underwater and water-free rendering
paths, facilitating better color restoration while maintaining
geometric consistency. Our method advances the underwater
3D reconstruction by achieving both high-quality geometric
reconstruction and realistic color restoration within a unified
3DGS framework, simultaneously addressing computational
efficiency for real-time underwater applications.
The main contributions of our work are as follows:
• We propose WaterClear-GS, the first pure explicit 3DGS
method that integrates underwater optical-aware mod-
eling into Gaussian primitives, achieving simultaneous
underwater scene reconstruction and restoration without
relying on separate neural networks for medium.
• We design a dual-branch framework that jointly opti-
mizes underwater 3D reconstruction and water-free color
restoration, enhanced by geometry and optics regulariza-
tion to ensure physical plausibility and visual quality.
• We introduce a new underwater dataset, ShipWreck,
featuring complex shipwreck scenes to provide a more
challenging and diverse test case than current coral
scenes.
• Experiments demonstrate that the proposed WaterClear-
GS achieves state-of-the-art results in both NVS and UIR
tasks while maintaining real-time rendering capabilities.
II. RELATED WORK
A. Novel View Synthesis
Novel view synthesis (NVS) aims to render photorealis-
tic images from arbitrary unseen viewpoints given a set of
posed input images. A major breakthrough was introduced
by NeRF [13], which represents scenes as a continuous
volumetric function parameterized by a multi-layer perceptron
(MLP). Colors and densities are predicted by querying the
MLP at sampled points along camera rays and integrated via
volumetric rendering. Numerous NeRF variants have improved
efficiency, quality, or generalization [22], [23], [24], [25], [26],
[27], [28], [29], [30], [31]. Examples include fast training
through tensor decomposition [23], sparse voxel grids [24],
multi-resolution hash grids [22], and factorized scene represen-
tations [28], [32]. Other methods address dynamic scenes [33],
[34], [35] or large-scale environments [36], [37].
Recently, 3D Gaussian Splatting [17] has emerged as a com-
pelling alternative to NeRF-based approaches. It represents
the scene using a set of explicit 3D Gaussians with learnable
properties such as position, color, scale, and opacity, achiev-
ing state-of-the-art rendering quality while enabling real-time
performance. Subsequent works have expanded its applica-
bility to various challenging domains, including sparse-view
reconstruction [38], [39], [40], [41], dynamic or deformable
scenes [42], [43], and large-scale environments [44], [45],
[46], [47]. Further extensions improve shading and appearance
modeling [48], [49] or geometric accuracy [50], [51].
B. Underwater Scene Representation and Restoration
Underwater images suffer from severe degradation caused
by wavelength-dependent attenuation and scattering, present-
ing great challenges to vision tasks. Traditional single-image
methods [52], [8], [7], [53] typically rely on hand-crafted pri-
ors, such as the dark channel or white balance assumptions, but
these priors break down under strong color casts or spatially
varying illumination in underwater environments. Akkaynak
and Treibitz [54] introduced the revised underwater image
formation model, which separates direct transmission from
backscatter and explicitly accounts for wavelength-dependent
attenuation, forming the foundation of many subsequent meth-
ods. Although extensive progress has been made in single-
image underwater enhancement and restoration [9], [10], [55],
[11], [12], they inevitably introduce inconsistencies when ap-
plied independently to each view in 3D scenes. This limitation
highlights the need for underwater scene modeling approaches
that jointly reason about the imaging physics and 3D structure.
With the rise of neural rendering, several recent works [16],
[15], [56], [57], [14], [18], [19], [20], [21] attempt to in-
corporate underwater physics into NeRF or 3DGS for multi-
view reconstruction. WaterNeRF [56] jointly estimates scene
density, color, and attenuation parameters within the NeRF
framework. WaterHE-NeRF [57] predicts illuminance atten-
uation alongside radiance and takes the histogram-equalized
image as a pseudo GT value to guide color restoration.
SeaThru-NeRF [14] explicitly models scattering media and
decomposes the rendering process into direct and backscatter
components. However, these NeRF-based methods generally
suffer from high training costs and often yield suboptimal
reconstruction quality. WaterSplatting [18] extends 3DGS by
representing the water medium as an additional volumetric
field, while SeaSplat [19] predicts attenuation and scattering
maps using two separate learned medium models. Aquatic-
GS [20] introduces neural water fields that predict medium
parameters conditioned on viewpoint. RestorGS [21] employs
an additional color MLP and a CNN to predict color map-
ping and illumination map, respectively, though it increases
architectural complexity.
Although these approaches improve underwater reconstruc-
tion, most rely on implicit networks or additional volumetric
fields to estimate medium properties. Due to the difficulty of
consistently estimating per-view medium parameters, they of-
ten incur higher memory usage, unstable optimization behav-
ior, and limited color restoration accuracy. Moreover, modeling
the water medium as a separate component often complicates
the rendering pipeline and reduces runtime efficiency. In
contrast, our method directly embeds wavelength-dependent
medium properties into the Gaussian primitives themselves,
enabling the rendering pipeline to operate entirely within a
unified, CUDA-efficient 3DGS framework. This eliminates the
need for separate implicit fields or additional MLP estimators,
resulting in faster convergence, real-time rendering perfor-
mance, and significantly improved color fidelity in underwater
reconstruction and restoration.
III. METHODOLOGY
A. Preliminaries
1) 3D Gaussian Splatting: 3DGS represents a scene using
a large number of anisotropic 3D Gaussian primitives, each
defined by a center µ ∈R3, a covariance matrix Σ, an

<!-- page 3 -->
3
Optical-Aware 
Gaussian Modeling
Extended Gaussian
Gaussian Splatting
Dual-Branch Rendering 
& Optimization
Camera
Projection
Adaptive 
Density Control
Differentiable 
Rasterization
Ls
SfM Points
Operation Flow
Gradient Flow
Init.
Limg
Ld
Captured Images
Water
Clear
Le
uc
Intrinsic Gaussian
c)
R
S
,
,
,
,
(


)
B
c,
R
S
,
B
D
,
,
,
,
,
(




GT
Lp
Dpred
Depth Est.
Dpseudo
Exposure
Channel
Neighbors
Fig. 2. Overview of the WaterClear-GS framework. Our method extends each Gaussian with water optical parameters βD, βB and B. The dual-branch
design simultaneously renders underwater images by applying these parameters and clear images by zeroing them out. Depth-guided enhancement guides the
geometry optimization, while exposure constraint balances the dynamic range of restored color and spatially-adaptive regularization, together with spectral
regularization, ensures physical plausibility of medium properties. Our framework ensures high-quality reconstruction and realistic color restoration.
opacity α ∈[0, 1], and a set of spherical harmonic (SH)
coefficients C ∈Rk for modeling view-dependent appearance.
The Gaussian density is given by:
G(x) = exp

−1
2(x −µ)T Σ−1(x −µ)

.
(1)
The covariance matrix Σ is decomposed into a scaling matrix
S (represented by a vector s ∈R3), and a rotation matrix
(derived from a quaternion q ∈R4) as:
Σ = RSST RT .
(2)
During rendering, each 3D Gaussian is projected to a 2D
ellipse on the image plane, and only the Gaussians overlapping
a pixel contribute to its color. The final color is accumulated
through α-blending:
C =
N
X
i=1
ciαi
Y
j<i
(1 −αj),
(3)
where N is the number of Gaussians, and the depth is rendered
using the same transmittance weights:
D =
N
X
i=1
ziαi
Y
j<i
(1 −αj).
(4)
Through iterative optimization with photometric supervision
and adaptive density control which clones, splits, or prunes
Gaussians based on gradient magnitude, the model progres-
sively refines the number of Gaussians and their attributes until
they can effectively represent the entire 3D scene.
2) Underwater Image Formation Model: Unlike imaging
in clear air, image formation in scattering media is affected
by the medium in two aspects. On one hand, the signal
reflected from objects undergoes distance- and wavelength-
dependent attenuation as it propagates through the medium. On
the other hand, the in-scattering of ambient light by suspended
particles introduces a view-dependent backscatter component
along the line of sight (LOS). This backscatter is independent
of scene content but accumulates with distance, reducing
visibility and contrast while causing severe color distortion.
Following the revised underwater imaging model of Akkaynak
and Treibitz [54], the image I captured at distance z can be
expressed as:
I = J · e−βD(vD) z
|
{z
}
direct component
+ B∞·
 1 −e−βB(vB) z
|
{z
}
backscatter component
,
(5)
where J is the true radiance of the object, and βD(vD)
and βB(vB) denote the wavelength-dependent attenuation and
backscatter coefficients. The vectors vD and vB encode the
dependence of the coefficients on multiple wavelength-related
factors, including object reflectance, ambient illumination,
water-body scattering properties, and the camera’s spectral
response. Both terms are inherently channel-dependent, ex-
plaining the nonlinear interaction between distance, wave-
length, and color distortion observed in underwater images.
The backscatter term B∞represents the water color from
backscattering at infinity (also called veiling light or ambient
light), which is typically non-uniform due to the directionality
of sunlight and other factors [58].
B. WaterClear-GS Overview
WaterClear-GS is built upon 3DGS framework but extends
it with a physically grounded formulation tailored for un-
derwater imaging. As illustrated in Figure 2, WaterClear-GS
augments each Gaussian primitive with wavelength-dependent
attenuation, backscatter, and veiling light parameters, enabling
the representation itself to capture local water-medium effects
without relying on an auxiliary implicit network. To jointly
achieve accurate underwater reconstruction and faithful color
restoration, we introduce a dual-branch rendering strategy
that synthesizes both underwater and water-free views from
the same set of Gaussians, enforcing geometric consistency
while progressively recovering true object appearance. This

<!-- page 4 -->
4
section introduces our optical-aware Gaussian modeling ap-
proach, followed by the dual-branch optimization strategy and
regularization mechanisms.
1) Optical-Aware Gaussian Modeling: To enable Gaus-
sian primitives to inherently model underwater attenuation
and backscatter, we associate each Gaussian with local,
wavelength-dependent optical parameters and apply them to
modulate its color before rasterization. For the i-th Gaussian,
let co
i ∈R3 be the original color predicted from spherical har-
monics (SH), and let di = (µi −c)z denote the distance from
the camera center c. We define channel-wise direct attenuation,
backscatter, and veiling-light coefficients βD
i , βB
i , Bi ∈R3,
and compute the underwater-corrected color as:
cu
i = T D
i
· co
i + (1 −T B
i ) · Bi,
(6)
where T D
i
= exp(−βD
i di) and T B
i
= exp(−βB
i di) are
wavelength-dependent transmittances applied element-wise
over RGB. During α-blending, the underwater color cu
i sim-
ply replaces the original SH-predicted color co
i , yielding the
rendered pixel:
Cu =
N
X
i=1
cu
i αi
Y
j<i
(1 −αj).
(7)
The optical parameters βD
i , βB
i , Bi are fully differentiable and
optimized jointly with all other Gaussian attributes. Assigning
these parameters at the primitive level enables the model to
capture spatially varying water-body properties, avoiding the
computational overhead and inaccuracies of auxiliary MLPs
while encouraging convergence toward physically consistent
attenuation–scattering behavior.
2) Dual-Branch Optimization Strategy:
To enable both
faithful underwater reconstruction and accurate color recovery,
we design a dual-branch optimization strategy that jointly
supervises the scene under two rendering conditions: water
and clear. The water branch renders images using the full
underwater formation model. By fitting the input underwater
images, this branch provides geometric supervision and con-
strains the learned attenuation and backscatter parameters to
be consistent with real observations. The clear branch renders
the same Gaussian primitives after removing all water-medium
terms, effectively simulating an ideal water-free environment.
This branch encourages the model to recover intrinsic object
colors that are not directly available from supervision. Cru-
cially, it acts as a regularization that forces the disentanglement
of scene appearance from water effects, preventing degen-
erate solutions where the model might incorrectly attribute
scattering effects to the object’s surface texture. Sharing the
same set of Gaussian primitives, the two rendering branches
collaboratively optimize the representation, acting as mutual
constraints and facilitators to achieve a balanced result.
C. Optimization for Reconstruction
1) Depth-Guided Geometric Regularization: In underwater
imaging, depth information plays a crucial role in accurately
modeling scene geometry. The effects of light attenuation and
scattering, governed by the learned optical parameters, are ex-
ponentially dependent on the propagation distance. Moreover,
conventional Gaussian-based 3D representations often suffer
from floating artifacts and geometric instability, especially
in water bodies where texture cues are weak and depth
ambiguity is severe. To address this, we introduce a depth-
guided geometric regularization mechanism to stabilize the 3D
representation learning.
We leverage the powerful monocular depth estimation
model DepthAnythingV2 [59] to generate pseudo-depth maps.
Typical pixel-wise L1 losses on depth map may lead to scale
and shift ambiguities. To alleviate this issue and emphasize
structural consistency rather than absolute depth values, we
employ a Pearson correlation coefficient (PCC) based loss to
maximize the similarity between the predicted depth Dpred
and the pseudo-depth Dpseudo:
Ld = 1 −Cov(Dpred, Dpseudo)
σ(Dpred) σ(Dpseudo).
(8)
This correlation-based formulation focuses on the relative
depth relationships across the scene. Since our goal is to
leverage the robust 2D structural priors from the foundation
model to guide 3D geometry rather than distilling absolute
metric depth, this scale-invariant loss effectively prevents the
propagation of scale/shift errors from the monocular estimator.
2) Perception-Driven Image Loss: Underwater scenes ex-
hibit strong wavelength-dependent attenuation and scatter-
ing, resulting in highly unbalanced illumination and color-
dependent degradation. Conventional image loss used by
3DGS treats all pixels uniformly, causing optimization to be
dominated by bright regions while neglecting low-intensity
areas. This leads to a visually imbalanced reconstruction,
where foreground objects are rendered with relatively high
fidelity, but background scenes and shadowed areas suffer from
blurring, distortion, or a complete loss of detail.
Inspired by RawNeRF [60], we perform a perception-driven
transformation ψ(·) that adaptively rescales image intensities
to balance gradient magnitudes across illumination levels.
Define a gradient supervised mapping on the estimated image
ˆy and the ground truth image y:
ψ(y) =
y
sg(ˆy) + ϵ,
(9)
where ϵ=10−3 ensures numerical stability and sg(·) denotes
the stop-gradient operator. Using sg(ˆy) rather than sg(y) al-
lows the model to adaptively emphasize different regions based
on its own predictions. This transformation ψ(·) behaves like a
differentiable inverse-intensity tone mapping that redistributes
gradient energy across the dynamic range, ensuring that darker
regions receive sufficient supervision while maintaining nu-
merical stability in bright areas.
The photometric reconstruction loss is evaluated in the
perceptual domain:
LW-L2 = 1
N ∥ψ(ˆy) −ψ(y)∥2
2,
(10)
where N denotes the total number of pixels. Similarly, the
structural similarity loss is computed between the transformed
images:
LW-DSSIM = 1 −SSIM
 ψ(ˆy), ψ(y)

,
(11)

<!-- page 5 -->
5
which effectively measures perceptual similarity in a tone-
mapped domain, enhancing sensitivity to structural errors in
low-intensity regions. Then the final image reconstruction loss
combines the two terms:
Limg = (1 −λSSIM) LW-L2 + λSSIM LW-DSSIM.
(12)
D. Optimization for Restoration
The absence of ground-truth supervision introduces inherent
ambiguity between the recovered colors and the estimated
attenuation–scattering parameters, leading to multiple visually
plausible but physically inconsistent solutions. To constrain the
solution space, we employ three complementary components:
an exposure constraint to avoid over-amplification of high-
lights, a spatially-adaptive regularization to stabilize neighbor-
ing Gaussians, and a physically guided spectral prior to enforce
wavelength-consistent ordering of the optical coefficients.
1) Exposure Constraint: Underwater scenes often contain
uneven illumination, and the interplay between attenuation
estimation and color recovery can make the restored results
susceptible to overexposure. This issue is further amplified by
the perception-driven reconstruction loss, which up-weights
dark regions and may inadvertently encourage the network to
brighten the output to reduce the weighted error. To main-
tain a visually natural dynamic range consistent with human
perception, we introduce an exposure regularization term that
penalizes over-bright regions in the restored image:
Le = 1
|Ω|
X
p∈Ω
n
max(Iclear(p) −τ, 0)
o
,
(13)
where Iclear represents the water-free rendered image, Ω
denotes the set of image pixels, τ represent the brightness
thresholds. This constraint prevents overexposure and encour-
ages balanced illumination across the restored scene.
2) Spatially-Adaptive Regularization: To model more plau-
sible non-uniform optical properties in underwater environ-
ments, we propose a spatially-adaptive regularization method
based on local consistency. This approach applies distance-
weighted neighborhood smoothing constraints to key water
optical parameters (βD, βB, B). Specifically, for each visi-
ble Gaussian from the current viewpoint, we perform local
smoothing within a spherical neighborhood of radius r, con-
straining parameter consistency with neighboring Gaussians
through the following loss function:
Ls =
1
|V |
X
i∈V
P
j∈Ni ∥θi −θj∥2
2 · wij
P
j∈Ni wij
,
(14)
where V represents the set of valid Gaussians with sufficient
neighbors (≥nmin), Ni denotes the neighborhood of Gaussian
i, θ represents the water optical parameters to be smoothed,
and wij = 1/(dij +ε) is a weight based on the Euclidean dis-
tance dij. This distance-weighted local smoothing mechanism
only requires local consistency rather than global uniformity,
preserving the ability of optical parameters to vary with depth
and water quality.
3) Physically Guided Spectral Regularization: To encode
physically plausible spectral behavior of underwater light
propagation, we introduce a lightweight physics-inspired reg-
ularizer that enforces soft spectral ordering across RGB wave-
lengths among the learned per-Gaussian optical parameters.
For each Gaussian, the optical parameters are wavelength-
dependent: the attenuation βD(i)
=
[βD
r , βD
g , βD
b ]i, the
backscatter βB(i) = [βB
r , βB
g , βB
b ]i, and the veiling light
B(i) = [Br, Bg, Bb]i are each defined with separate values at
red, green, and blue wavelengths. We define their combined
spectral differences as:
∆i = [ ∆βD(i), ∆βB(i), ∆B(i) ],
(15)





∆βD(i) = [ βD
g (i) −βD
r (i), βD
b (i) −βD
g (i) ],
∆βB(i) = [ βB
r (i) −βB
g (i), βB
g (i) −βB
b (i) ],
∆B(i) = [ Br(i) −Bg(i), Bg(i) −Bb(i) ].
(16)
The physical prior loss softly penalizes violations of the
expected spectral ordering using a smooth Softplus function:
Lp = Ei

Softplus(∆i + δ)

,
(17)
where δ is a small tolerance margin. It softly enforces the
underwater spectral priors: red attenuates fastest, blue scatters
most, and bluish veiling light, while remaining continuous and
differentiable, which stabilizes training and avoids gradient
discontinuities at spectral boundaries.
The total loss function for the WaterClear-GS is defined as:
Ltotal = Limg + λdLd + λsLs + λpLp + λeLe.
(18)
IV. EXPERIMENTS
A. Experimental Setup
1) Public Datasets: SeaThru-NeRF [14] is a real-world
underwater dataset captured by a Nikon D850 SLR camera, in-
cluding Panama, Curasao, IUI3-RedSea and JapaneseGardens-
RedSea. We also employ the D3 and D5 datasets from
SeaThru [61], captured by a Sony α7R Mk III camera and
a Nikon D810 camera, respectively. Each dataset is divided
into training and testing subsets as follows: Panama (15/3),
Curasao (18/3), JapaneseGardens-RedSea (17/3), IUI3-RedSea
(25/4), D3 (59/9), and D5 (37/6). All images are captured
under natural underwater illumination. For D3 and D5, we
apply gamma correction to the original linear-space PNGs fol-
lowing SeaThru [61]. The Curasao, D3 and D5 scenes contain
complete and visible color charts, making them suitable for
evaluating image color restoration performance.
2) Self-Collected Dataset: Noticing the limitations of ex-
isting datasets in terms of scene diversity, we construct a new
challenging real-world underwater dataset named ShipWreck.
Images are collected from publicly available online videos,
which contain two shipwreck scenes with complex structures,
metal surfaces, and varying water conditions. Each scene
includes high-resolution images (1920×1080) captured from
diverse viewpoints, with 90 and 36 images, respectively. Train-
ing and testing subsets are divided as: ShipWreck-1 (78/12),
ShipWreck-2 (31/5). Camera intrinsics and poses are estimated
using COLMAP [62] for initialization.

<!-- page 6 -->
6
TABLE I
QUANTITATIVE RESULTS OF THE NVS TASK ON THE SEATHRU-NERF DATASET. HIGHLIGHTS THE TOP THREE:
FIRST ,
SECOND , AND
THIRD .
Method
Panama
Curasao
IUI3-RedSea
J.G.-RedSea
Avg.
Avg.
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Time↓
FPS↑
3DGS [17]
29.59
0.919
0.181
30.92
0.936
0.183
24.26
0.896
0.245
22.08
0.851
0.207
3.6min 241.14
STN [14]
27.82
0.838
0.266
30.29
0.881
0.251
25.97
0.787
0.316
21.78
0.769
0.294
10.5h
0.13
Seasplat [19]
29.86
0.935
0.147
29.82
0.925
0.165
28.43
0.900
0.220
23.24
0.874
0.180
29.8min 50.55
WS [18]
31.64
0.943
0.104
32.67
0.950
0.140
29.76
0.909
0.203
24.20
0.896
0.136
9.5min
73.28
Ours
32.27
0.950
0.112
33.49
0.956
0.136
30.04
0.916
0.200
24.35
0.900
0.147
15.7min 160.48
TABLE II
QUANTITATIVE COMPARISON OF THE NVS TASK ON THE SEATHRU
DATASET. HIGHLIGHTS THE TOP THREE:
FIRST ,
SECOND , AND
THIRD .
Method
D3
D5
Avg.
Avg.
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓Time↓
FPS↑
3DGS [17]
27.99
0.844
0.289
29.94
0.903
0.281
3.5min 228.67
STN [14]
20.34
0.562
0.555
24.34
0.801
0.453
10h
0.12
Seasplat [19]
28.27
0.851
0.256
28.98
0.904
0.263
27.3min 67.35
WS [18]
28.03
0.852
0.219
30.62
0.910
0.247
13min
76.14
Ours
28.48
0.860
0.214
30.79
0.919
0.245
16.7min 176.63
TABLE III
QUANTITATIVE COMPARISON OF THE NVS TASK ON THE SHIPWRECK
DATASET. HIGHLIGHTS THE TOP THREE:
FIRST ,
SECOND , AND
THIRD .
Method
ShipWreck-1
ShipWreck-2
Avg.
Avg.
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
Time↓
FPS↑
3DGS [17]
27.39
0.852
0.196
24.80
0.855
0.191
8.6min 191.57
STN [14]
20.00
0.530
0.580
24.01
0.767
0.315
12h
0.10
Seasplat [19]
27.62
0.853
c0.187
25.18
0.863
0.189
33.8min
61.12
WS [18]
26.74
0.840
0.191
24.34
0.852
0.187
12.5min
78.35
Ours
27.71
0.858
0.177
25.31
0.865
0.182
13.5min 160.16
3) Baselines:
On the task of NVS, we compare our
method with vanilla 3DGS [17] and several state-of-the-
art underwater methods: 1) NeRF-based methods, SeaThru-
NeRF (STN) [14]; and 2) 3DGS-based methods, including
WaterSplatting (WS) [18] and SeaSplat [19]. On the task of
UIR, we only compare ours with these underwater methods.
4) Metrics: For the NVS task, we evaluate the Peak Signal-
to-Noise Ratio (PSNR), Structural Similarity Index Measure
(SSIM) [63], and Learned Perceptual Image Patch Similarity
(LPIPS) [64] between the rendered and ground-truth under-
water images on the test datasets. We also report the training
time and rendering FPS to evaluate model efficiency. For the
UIR task, obtaining ground-truth references is challenging.
Following prior underwater enhancement studies [58], [65],
[66], we adopt the CIEDE2000 color difference (△E00) [67]
and the average angular error ¯ψ (in degrees) [68] to assess
color restoration quality on datasets containing standard color
charts. △E00 quantifies perceptual color differences in the
CIELAB color space, which is derived from RGB through
a perceptually uniform nonlinear transformation. ¯ψ measures
color accuracy as the angle between restored and reference
color vectors in RGB space. It is computed over 12 color
chart patches to assess restoration fidelity in our experiments.
5) Implementation Details: Our method is based on the
pipeline of 3DGS method. Each dataset is trained for 15,000
iterations. Densification strategy is adjusted to perform every
500 iterations. For the image reconstruction loss, λSSIM =
0.2. For the depth-guided PCC loss, λd is set to 0.2. For
exposure constraint Le, the exposure threshold τ is set to 0.9,
with corresponding weights λe = 0.1. Spatial regularization
Ls and physical prior regularization Lp are performed every
10 iterations. λs and λp are set to 0.1 and smooth radius is set
to 0.05 with at least 2 neighbors. Owing to the wavelength-
dependent attenuation and scattering properties of the water
medium, the parameters βD, βB and B are each modeled as
three-component vectors that correspond to the RGB channels.
The learning rates are set to 0.0001 for βD and βB, and
0.001 for B. We do not employ additional spherical-harmonics
(SH) coefficients; only RGB values are used as the base
color representation for each Gaussian. All experiments are
conducted on a single NVIDIA RTX 4090 GPU.
B. Novel View Synthesis Results
We first report the quantitative results on the NVS task
in Table I, Table II and Table III. The results demonstrate
that our method consistently achieves superior performance,
attaining the highest PSNR and SSIM across all datasets,
despite slightly suboptimal LPIPS scores in a few cases.
While WaterSplatting benefits from the inherent smoothness of
volumetric fields, our Gaussian-based representations preserve
sharper appearance boundaries and color transitions, which
LPIPS tends to penalize. Notably, our approach strikes a
favorable balance between quality and efficiency. While NeRF-
based underwater models SeaThru-NeRF require hours of
training per scene, and 3DGS variants augmented with addi-
tional medium networks (e.g., WaterSplatting) incur noticeable
overhead, our method retains the highest FPS advantage of
the 3DGS pipeline. Specifically, WaterClear-GS achieves over
160 FPS on average, which is 2× faster than WaterSplatting
and Seasplat, and orders of magnitude faster than SeaThru-
NeRF. This efficiency stems from embedding underwater op-
tical effects directly into Gaussian rendering without auxiliary
volumetric fields or MLP estimators. This means that our
method can support more real-time applications.
Figure 3 presents qualitative comparisons of the NVS re-
sults. On the SeaThru-NeRF dataset, 3DGS exhibits severe
artifacts and blurred distant structures, as the original formu-
lation does not account for medium and thus struggles under
strong degradation. For the D3 and D5 scenes with lower
exposure and more turbid water conditions, SeaThru-NeRF
fails to reconstruct valid geometry, resulting in incomplete
structures and heavily distorted colors. Seasplat performs more
robustly overall, yet still produces localized distortions, such

<!-- page 7 -->
7
Panama
GT 
3DGS 
SeaSplat
SeaThru-NeRF
Ours
WaterSplatting
Curasao
IUI3
JG
D3
D5
S-1
S-2
Fig. 3. Qualitative results of the NVS task. We present rendered underwater images and their depth maps. The pseudo-depth maps are used as a reference.
Zoomed-in regions (highlighted with red bounding boxes) illustrate detailed differences. Our method consistently preserves geometric structures and fine-level
details in most cases.

<!-- page 8 -->
8
Panama
Curasao
IUI3
JG
D3
D5
GT (Underwater)
SeaThru-NeRF
SeaSplat
WaterSplatting
Ours
S-1
S-2
Fig. 4. Qualitative results of the UIR task. Details are zoomed in and highlighted with red and yellow bounding boxes. In contrast to other methods, which
often produce grayish, underexposed, or severely color-shifted results, our method restores images with more natural visual quality.

<!-- page 9 -->
9
TABLE IV
QUANTITATIVE RESULTS OF THE UIR TASK. HIGHLIGHTS THE TOP TWO
IN EACH METRIC:
FIRST
AND
SECOND .
Scene
Curasao
D3
D5
Method
△E00 ↓
¯ψ ↓
△E00 ↓
¯ψ ↓
△E00 ↓
¯ψ ↓
STN [14]
16.67
19.77
40.89
32.54
40.76
31.42
Seasplat [19]
15.78
18.61
27.98
27.53
41.37
27.06
WS [18]
19.98
22.98
30.82
28.09
36.09
25.76
Ours
12.67
10.71
27.82
27.20
30.37
24.87
as inconsistent textures in the Curasao, J.G.-RedSea, and
ShipWreck-1 scenes. WaterSplatting achieves sharper details
but occasionally misattributes water-medium effects to object
surfaces, leading to incorrect occlusions and unnatural color
biases, particularly visible in the D5 and ShipWreck datasets.
Depth maps further reveal the shortcomings of existing meth-
ods. Under the influence of water medium and ambiguous
attenuation, these methods tend to inject floating artifacts or
irregular blobs near the camera to compensate for depth uncer-
tainty, degrading both geometry and appearance. In contrast,
WaterClear-GS produces more accurate depth distributions and
preserves fine-grained structural details, owing to its optical
modeling and geometry-aware regularization. As a result, our
method delivers more robust and visually consistent recon-
structions with cleaner boundaries and more reliable depth
maps across diverse underwater conditions.
C. Underwater Images Restoration Results
Figure 4 shows the qualitative comparison of underwater
image restoration. SeaThru-NeRF often fails to reconstruct
several scenes due to strong degradation and insufficient
multi-view constraints; even when reconstruction succeeds,
the restored images show only marginal improvements with
limited recovery of color and details. Seasplat produces vi-
sually enhanced outputs but frequently introduces unnatural
color shifts and oversaturated regions, particularly in the D5
and ShipWreck-1 scenes. WaterSplatting also tends to generate
outputs with noticeably reduced brightness and produces hazy,
low-contrast renderings across all datasets, largely because it
overestimates the contribution of scattering medium. In con-
trast, WaterClear-GS yields visually compelling restorations
with natural color saturation, enhanced contrast, and sharper
fine-scale structures. Thanks to the wavelength-dependent
optical modeling embedded in each Gaussian, our method
restores color tones—especially in the severely attenuated red
channel—more faithfully than competing approaches. This
leads to restorations that better align with human visual
perception and more accurately reflect the true appearance of
underwater scenes.
To quantitatively evaluate the color restoration quality of
each model, we compute the color difference △E00 and the
average angular error ¯ψ (in degrees) between the restored
and ground-truth colors on three scenes that include complete
standard color charts. As reported in Table IV, our method
consistently achieves the best performance, with the lowest
△E00 and ¯ψ across all scenes. On the Curasao scene, our
approach attains the lowest △E00 and ¯ψ, improving over the
second-best method by 3.11 and 7.90, respectively. A similar
advantage is observed on the D5 scene, where our method
ranks first on both metrics and maintains a clear margin
over competing approaches. These results demonstrate that
WaterClear-GS not only minimizes perceptual color error but
also preserves chromatic angular consistency more accurately,
leading to faithful and stable restoration under a wide range
of underwater degradation conditions.
D. Analysis on the Optical-Aware Gaussian Modeling
1) Physical Interpretation: We visualize the learned atten-
uation βD, backscatter βB, and veiling light B across all
Gaussians as Figure 5a, 5b, and 5c. These parameters exhibit
clear physical structure: all RGB channels follow the expected
spectral ordering (βD
r
> βD
g
> βD
b , βB
b
> βB
g
> βB
r ,
Bb > Bg > Br), confirming the interpretability of our optical-
aware formulation. Importantly, these parameters are defined
per Gaussian and represent the local optical properties of the
water medium; they are therefore not inherently functions of
the camera–point distance. The effect of distance is already
handled by the rendering equation through the exponential
transmission terms. Nevertheless, when plotting the learned
parameters against depth, smooth depth-related trends emerge:
βD tends to increase with distance, while βB and B rise in
mid-range regions and decrease for very distant Gaussians.
Note that the depth-related trends observed are not inherent
distance-dependent definitions, but dataset-driven correlations
that naturally emerge in forward-facing underwater captures,
where distant regions typically exhibit stronger attenuation,
backscatter, and ambient veiling. But in surround capture
setups, such depth-related patterns may weaken or disappear.
2) Effect of Multichannel Optical Parameters: To further
validate the necessity of modeling wavelength-dependent op-
tical effects, we replace the three-channel optical parameters
βD, βB, B ∈R3 with single-channel scalars shared across
RGB. As shown in Figure 6, this modification leads to
noticeable hue shifts in restored images. And distant regions
become unrealistically blue due to the inability to attenuate
short-wavelength components relative to red and green. This
confirms that single-channel optical modeling fails to capture
the spectral behavior of underwater light propagation. This
experiment highlights that multichannel optical modeling is
essential for accurate color restoration, particularly for recov-
ering long-wavelength components.
3) Impact of Color Representation: In contrast to in-air
scenes where high-order spherical harmonics (SHs) are essen-
tial for modeling view-dependent specularities, the underwater
medium acts as a natural low-pass filter on angular radiance.
Scattering suppresses high-frequency view-dependent signals,
making high-order SHs redundant. Moreover, our formulation
explicitly embeds wavelength-dependent optical parameters
into each Gaussian, allowing the model to explain spectral
and view variations physically rather than through a purely
expressive SH basis. As a result, relying on high-order SHs
risks introducing redundant degrees of freedom that may inter-
fere with parameter interpretability and overfit sparse views in
underwater datasets. We conduct experiments under different

<!-- page 10 -->
10
(a) βD w/ Lp
(b) βB w/ Lp
(c) B w/ Lp
(d) βD w/o Lp
(e) βB w/o Lp
(f) B w/o Lp
Fig. 5. Visualization of learned optical parameters with/without Lp on the IUI3-RedSea scene.
GT (Underwater)
1-channel
Ours (3-channel)
Fig. 6. Effect of single-channel vs. RGB-channel optical modeling.
SH degrees, as shown in Table V. Interestingly, lowering SH
degree even slightly improves PSNR (+0.14). We attribute this
to the reduced overfitting caused by excessive SH parameters
and the improved stability of color estimation under our physi-
cally grounded formulation. Moreover, reducing the SH degree
significantly decreases the overall model size: color parameters
drop from 48 at degree-3 to 3 per Gaussian at degree-0, leading
to a substantial savings in storage and computation. Even with
comparable numbers of Gaussians (differs 1%), using degree-
0 SH reduces memory consumption by ∼63% and improves
rendering speed by ∼20%. We therefore adopt degree-0
(RGB only) in all experiments, achieving a favorable balance
between accuracy, compactness, and real-time performance.
E. Ablation Study
To validate the effectiveness of each component in our
proposed WaterClear-GS, we conduct a series of ablation
studies on the SeaThru-NeRF dataset. As shown in Table VI,
the full model achieves the best performance on both the NVS
and UIR tasks, demonstrating the complementary benefits of
all components.
1) Effectiveness of the Depth-Guided Geometric Regular-
ization: Figure 7 and Table VII compare different forms of
TABLE V
QUANTITATIVE RESULTS OF DIFFERENT SH DEGREE ON THE J.G-REDSEA
SCENE. STORAGE IS BASED ON THE SIZE OF THE EXPORTED PLY FILE,
WHICH IS RELATIVE TO THE NUMBER OF GAUSSIANS AND THE NUMBER
OF ATTRIBUTE PARAMETERS PER GAUSSIAN. BOLD DENOTES THE BEST
PERFORMANCE.
SH Degree Gaussians(K) Storage(MB)↓PSNR↑Time(min)↓
FPS↑
3
1528
414
24.21
17
127.58
2
1526
291
24.27
16
134.84
1
1542
206
24.31
15.5
144.71
Ours (0)
1540
153
24.35
15
152.72
TABLE VI
ABLATION RESULTS ON THE SEATHRU-NERF DATASET. UIR METRICS
ARE MEASURED ON CURASAO SCENE. BOLD DENOTES THE BEST
PERFORMANCE, AND UNDERLINE DENOTES THE SECOND PLACE.
Method
NVS
UIR
PSNR↑SSIM↑LPIPS↓△E00 ↓
¯ψ ↓
Baseline (3DGS)
26.71
0.901
0.204
-
-
w/o Ld
28.75
0.926
0.154
12.93
11.78
w/o weighted Limg
29.06
0.899
0.246
13.12
11.09
w/o Le
29.84
0.930
0.151
15.61
14.69
w/o Ls
29.68
0.929
0.151
13.77
14.14
w/o Lp
29.98
0.930
0.150
15.56
15.48
Ours (Full Model)
30.04
0.931
0.149
12.67
10.71
depth regularization. Without using any depth prior, the recon-
struction exhibits noticeable artifacts and floating noise, as the
model lacks geometric guidance to suppress medium-induced
ambiguities. Incorporating an L1 loss on depth provides partial
improvement by encouraging coarse alignment. An L2 depth
loss enforces overly strict pixel-wise matching, leading to
severe ’holes’ and discontinuities in the predicted depth due
to the inevitable scale and shift inconsistencies present in
monocular depth maps. In contrast, with our PCC-based depth
regularization, the model achieves the highest reconstruction
accuracy while producing smooth and geometrically coherent
depth maps. We attribute this robustness to the scale- and shift-

<!-- page 11 -->
11
w/o depth loss
L1-loss
L2-loss
Ours (PCC loss)
Fig. 7. Visualization of depth maps under different depth loss functions.
The proposed PCC depth loss produces noticeably smoother and more
spatially consistent depth estimates compared to alternative losses.
TABLE VII
RESULTS OF DIFFERENT Ld ON THE SEATHRU-NERF DATASET. BOLD
DENOTES THE BEST PERFORMANCE.
Method
PSNR↑
SSIM↑
LPIPS↓
w/o depth loss
28.75
0.926
0.154
w L1
29.51
0.929
0.152
w L2
29.84
0.930
0.151
Ours (PCC loss)
30.04
0.931
0.149
TABLE VIII
RESULTS OF DIFFERENT Limg ON THE SEATHRU-NERF DATASET. BOLD
DENOTES THE BEST PERFORMANCE, AND UNDERLINE DENOTES THE
SECOND PLACE.
Method
PSNR↑
SSIM↑
LPIPS↓
L1+DSSIM
29.06
0.899
0.246
W-L1+DSSIM
29.34
0.918
0.184
L1+W-DSSIM
29.19
0.919
0.183
W-L1+W-DSSIM
29.85
0.929
0.151
L2+DSSIM
28.28
0.880
0.286
W-L2+DSSIM
29.14
0.903
0.237
L2+W-DSSIM
28.61
0.912
0.200
Ours (W-L2+W-DSSIM)
30.04
0.931
0.149
invariant nature of PCC, which makes it better suited for super-
vising monocular depth priors that may not be geometrically
consistent with the reconstructed 3D scene.
2) Effectiveness of the Perception-Driven Image Loss:
Figure 10 and Table VIII report the impact of different
configurations of photometric and structural losses. Replacing
standard L1, L2, and SSIM with our perception-driven variants
(W-L1, W-L2, and W-DSSIM) consistently improves recon-
struction quality. For example, W-L1+DSSIM increases PSNR
from 29.06 to 29.34 and reduces LPIPS from 0.246 to 0.184,
showing that inverse-intensity reweighting better supervises
underexposed regions. Introducing W-DSSIM provides addi-
tional structural improvement (29.19 PSNR and 0.183 LPIPS).
While standard L2 performs poorly (28.28 PSNR and 0.286
LPIPS), its weighted form greatly alleviates over-penalization
of bright regions, achieving 29.14 PSNR and 0.237 LPIPS.
The best performance is obtained by combining both weighted
terms (W-L2+W-DSSIM), which reaches 30.04 PSNR, 0.931
SSIM, and 0.149 LPIPS. These results confirm that the pro-
posed perception-driven transformation redistributes gradient
emphasis toward underexposed regions and leads to noticeably
improved reconstruction quality.
w/o  Le
w/  Le
Fig. 8. Visualization of the contribution of the exposure constraint. With
the introduction of Le, the model effectively suppresses over-exposure and
produces visually more natural and balanced restorations. Details are zoomed
in and highlighted with red and yellow bounding boxes.
w/o  Ls
w/  Ls
Fig. 9. Visualization of the contribution of the spatial constraint. With the
introduction of Ls, the model produces restored results with fewer spatially
inconsistent colors (highlighted with the red boxes).
3) Effectiveness of the Exposure Constraint: As shown in
Figure 8, removing the exposure constraint leads to noticeable
overexposure in bright or near-white regions. In scenes such as
IUI3-RedSea and ShipWreck-1, objects with inherently high
albedo produce strong highlights in the water-free rendering,
causing unnatural colors and loss of fine texture. By imposing
our exposure constraint, the model effectively suppresses such
over-amplification, yielding more faithful color reproduction
and preserving surface details. This demonstrates that the
exposure constraint provides a useful regularization signal
that prevents the network from overcompensating in highly
illuminated regions and improves perceptual realism in the
restored images.
4) Effectiveness of the Spatially-Adaptive Regularization:
As evidenced by the visual results in Figure 9, removing
spatial regularization leads to noticeable inconsistency in
the estimated water-related parameters, represented as color
discontinuities and local jumps in homogeneous background
regions. Our spatially-adaptive regularization enforces coher-
ence among neighboring Gaussians by encouraging smooth
variations in the learned optical coefficients, which improves
the overall visual continuity of the reconstructed scene. This
produces more stable and spatially consistent color restoration,
particularly in large water-body areas where the underlying
medium properties should vary gradually.
5) Effectiveness of the Physically Guided Spectral Regu-
larization: Figure 5 visualizes the learned optical parame-
ters with and without applying the proposed physical prior.

<!-- page 12 -->
12
GT
W-L1+DSSIM
L1+DSSIM
L1+W-DSSIM
W-L1+W-DSSIM
W-L2+DSSIM
L2+DSSIM
L2+W-DSSIM
W-L2+W-DSSIM
Fig. 10. Qualitative comparison of different image losses.
Without this regularization, the model is still able to capture
coarse spectral tendencies, but the separation between RGB
channels becomes weaker and violations of expected physical
ordering occur frequently. This indicates that the raw data term
alone is insufficient to reliably disambiguate the wavelength-
dependent behavior of attenuation, scattering, and veiling light.
By enforcing soft spectral ordering, our physical prior substan-
tially stabilizes the optimization and yields optical parameter
fields that are more consistent, interpretable, and aligned with
underwater light-transport principles. This not only improves
the physical plausibility of the learned parameters but also
enhances the robustness of the reconstruction, demonstrating
the effectiveness of incorporating physics-aware regularization
into Gaussian-based modeling.
V. CONCLUSION AND DISCUSSION
We presented WaterClear-GS, a purely Gaussian-based un-
derwater reconstruction and restoration framework that em-
beds a physically grounded underwater formation model into
3D Gaussian primitives. By jointly learning wavelength-
dependent attenuation, backscatter, and veiling light within the
Gaussian representation, combined with geometric and optical
regularization, our method achieves high-quality novel view
synthesis and underwater image restoration while preserving
the real-time rendering capability (160+ FPS).
Current limitations include handling only static scenes with-
out addressing dynamic elements or temporal lighting vari-
ations. Meanwhile, both existing methods and ours struggle
in extreme underwater environments, such as those with very
high turbidity, which makes accurate color recovery exceed-
ingly difficult. Improving performance in these challenging
scenarios is a key objective for future research in this field.
REFERENCES
[1] L. Cai, N. E. McGuire, R. Hanlon, T. A. Mooney, and Y. Girdhar,
“Semi-supervised visual tracking of marine animals using autonomous
underwater vehicles,” International Journal of Computer Vision, vol.
131, no. 6, pp. 1406–1427, 2023. 1
[2] B. Joshi, M. Xanthidis, M. Roznere, N. J. Burgdorfer, P. Mordohai,
A. Q. Li, and I. Rekleitis, “Underwater exploration and mapping,” in
2022 IEEE/OES Autonomous Underwater Vehicles Symposium (AUV).
IEEE, 2022, pp. 1–7. 1
[3] E. Galceran, R. Campos, N. Palomeras, D. Ribas, M. Carreras, and
P. Ridao, “Coverage path planning with real-time replanning and surface
reconstruction for inspection of three-dimensional underwater structures
using autonomous underwater vehicles,” Journal of Field Robotics,
vol. 32, no. 7, pp. 952–983, 2015. 1
[4] J. J. Leonard and A. Bahr, “Autonomous underwater vehicle navigation,”
Springer handbook of ocean engineering, pp. 341–358, 2016. 1
[5] M. Johnson-Roberson, M. Bryson, A. Friedman, O. Pizarro, G. Troni,
P. Ozog, and J. C. Henderson, “High-resolution underwater robotic
vision-based mapping and three-dimensional reconstruction for archae-
ology,” Journal of Field Robotics, vol. 34, no. 4, pp. 625–643, 2017.
1
[6] T. Missiaen, D. Sakellariou, and N. C. Flemming, “Survey strategies
and techniques in underwater geoarchaeological research: An overview
with emphasis on prehistoric sites,” Under the sea: Archaeology and
palaeolandscapes of the continental shelf, pp. 21–37, 2017. 1
[7] Ancuti, Codruta O and Ancuti, Cosmin and De Vleeschouwer,
Christophe and Bekaert, Philippe, “Color balance and fusion for un-
derwater image enhancement,” IEEE Transactions on image processing,
vol. 27, no. 1, pp. 379–393, 2017. 1, 2
[8] C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, and P. Bekaert, “Color
balance and fusion for underwater image enhancement,” IEEE Trans-
actions on image processing, vol. 27, no. 1, pp. 379–393, 2017.
1,
2
[9] Q. Qi, Y. Zhang, F. Tian, Q. M. J. Wu, K. Li, X. Luan, and D. Song,
“Underwater image co-enhancement with correlation feature matching
and joint learning,” IEEE Transactions on Circuits and Systems for Video
Technology, vol. 32, no. 3, pp. 1133–1147, 2022. 1, 2
[10] Y. Kang, Q. Jiang, C. Li, W. Ren, H. Liu, and P. Wang, “A perception-
aware decomposition and fusion framework for underwater image
enhancement,” IEEE Transactions on Circuits and Systems for Video
Technology, vol. 33, no. 3, pp. 988–1002, 2023. 1, 2
[11] G. Hou, N. Li, P. Zhuang, K. Li, H. Sun, and C. Li, “Non-uniform illu-
mination underwater image restoration via illumination channel sparsity
prior,” IEEE Transactions on Circuits and Systems for Video Technology,
vol. 34, no. 2, pp. 799–814, 2024. 1, 2
[12] W. Zhang, L. Zhou, P. Zhuang, G. Li, X. Pan, W. Zhao, and C. Li,
“Underwater image enhancement via weighted wavelet visual perception
fusion,” IEEE Transactions on Circuits and Systems for Video Technol-
ogy, vol. 34, no. 4, pp. 2469–2483, 2024. 1, 2
[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021. 1, 2
[14] D. Levy, A. Peleg, N. Pearl, D. Rosenbaum, D. Akkaynak, S. Korman,
and T. Treibitz, “Seathru-nerf: Neural radiance fields in scattering

<!-- page 13 -->
13
media,” in Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, 2023, pp. 56–65. 1, 2, 5, 6, 9
[15] T. Zhang and M. Johnson-Roberson, “Beyond nerf underwater: Learning
neural reflectance fields for true color correction of marine imagery,”
IEEE Robotics and Automation Letters, vol. 8, no. 10, pp. 6467–6474,
2023. 1, 2
[16] Y. Tang, C. Zhu, R. Wan, C. Xu, and B. Shi, “Neural underwater
scene representation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 11 780–11 789. 1,
2
[17] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023. 1, 2, 6
[18] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, “WaterSplatting:
Fast underwater 3D scene reconstruction using gaussian splatting,” 3DV,
2025. 1, 2, 6, 9
[19] D. Yang, J. J. Leonard, and Y. Girdhar, “Seasplat: Representing under-
water scenes with 3d gaussian splatting and a physically grounded image
formation model,” arXiv preprint arXiv:2409.17345, 2024. 1, 2, 6, 9
[20] S. Liu, J. Lu, Z. Gu, J. Li, and Y. Deng, “Aquatic-gs: A hybrid 3d
representation for underwater scenes,” arXiv preprint arXiv:2411.00239,
2024. 1, 2
[21] Y. Qiao, M. Shao, L. Meng, and K. Xu, “Restorgs: Depth-aware
gaussian splatting for efficient 3d scene restoration,” in Proceedings of
the Computer Vision and Pattern Recognition Conference, 2025, pp.
11 177–11 186. 1, 2
[22] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022. 2
[23] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in European conference on computer vision.
Springer, 2022,
pp. 333–350. 2
[24] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022, pp. 5501–5510. 2
[25] S. Fang, Y. Wang, Y. Yang, W. Xu, H. Wang, W. Ding, and S. Zhou,
“Pvd-al: Progressive volume distillation with active learning for effi-
cient conversion between different nerf architectures,” arXiv preprint
arXiv:2304.04012, vol. 3, 2023. 2
[26] S. Fang, W. Xu, H. Wang, Y. Yang, Y. Wang, and S. Zhou, “One is
all: Bridging the gap between neural radiance fields architectures with
progressive volume distillation,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 37, no. 1, 2023, pp. 597–605. 2
[27] S. Fang, D. Qi, W. Xu, Y. Wang, Z. Zhang, X. Zhang, H. Zhang, Z. Shao,
and W. Ding, “Efficient implicit sdf and color reconstruction via shared
feature field,” in Proceedings of the Asian Conference on Computer
Vision, 2024, pp. 3499–3516. 2
[28] X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman, and
J. T. Barron, “Nerfactor: Neural factorization of shape and reflectance
under an unknown illumination,” ACM Transactions on Graphics (ToG),
vol. 40, no. 6, pp. 1–18, 2021. 2
[29] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in Proceedings of the IEEE/CVF inter-
national conference on computer vision, 2021, pp. 5855–5864. 2
[30] Y. Wang, S. Fang, H. Zhang, H. Li, Z. Zhang, X. Zeng, and W. Ding,
“Uav-enerf: Text-driven uav scene editing with neural radiance fields,”
IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–
14, 2024. 2
[31] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing
and improving neural radiance fields,” arXiv preprint arXiv:2010.07492,
2020. 2
[32] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa,
“K-planes: Explicit radiance fields in space, time, and appearance,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), June 2023, pp. 12 479–12 488. 2
[33] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-
nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2021,
pp. 10 318–10 327. 2
[34] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman,
R. Martin-Brualla, and S. M. Seitz, “Hypernerf: A higher-dimensional
representation for topologically varying neural radiance fields,” ACM
Trans. Graph., vol. 40, no. 6, dec 2021. 2
[35] A. Lin, Y. Xiang, J. Li, and M. Prasad, “Dynamic appearance particle
neural radiance field,” IEEE Transactions on Circuits and Systems for
Video Technology, 2025. 2
[36] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan,
J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large scene
neural view synthesis,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2022, pp. 8248–8258. 2
[37] H. Turki, D. Ramanan, and M. Satyanarayanan, “Mega-nerf: Scalable
construction of large-scale nerfs for virtual fly-throughs,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 12 922–12 931. 2
[38] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, “Dngaus-
sian: Optimizing sparse-view 3d gaussian radiance fields with global-
local depth normalization,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2024, pp. 20 775–20 785. 2
[39] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-
J. Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from
sparse multi-view images,” in European Conference on Computer Vision.
Springer, 2024, pp. 370–386. 2
[40] Y. Niu, X. Li, and Y. Wang, “Stereo-gaussian: Enhanced sparse view
gaussian splatting with one stereopair for light-field 3d display,” IEEE
Transactions on Circuits and Systems for Video Technology, pp. 1–1,
2025. 2
[41] S. Fang, I. Shen, T. Igarashi, Y. Wang, Z. Wang, Y. Yang, W. Ding,
S. Zhou et al., “Nerf is a valuable assistant for 3d gaussian splatting,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2025, pp. 26 230–26 240. 2
[42] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow: 4d reconstruction
with dynamic 3d gaussian particle,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 136–21 145. 2
[43] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 20 675–20 685. 2
[44] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan
et al., “Vastgaussian: Vast 3d gaussians for large scene reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5166–5175. 2
[45] J. Chen, W. Ye, Y. Wang, D. Chen, D. Huang, W. Ouyang, G. Zhang,
Y. Qiao, and T. He, “Gigags: Scaling up planar-based 3d gaussians for
large scene surface reconstruction,” arXiv preprint arXiv:2409.06685,
2024. 2
[46] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, “Citygaussian:
Real-time high-quality large-scale scene rendering with gaussians,” in
European Conference on Computer Vision.
Springer, 2024, pp. 265–
282. 2
[47] X. Cui, W. Ye, Y. Wang, G. Zhang, W. Zhou, T. He, and H. Li,
“Streetsurfgs: Scalable urban street surface reconstruction with planar-
based gaussian splatting,” IEEE Transactions on Circuits and Systems
for Video Technology, vol. 35, no. 9, pp. 8780–8793, 2025. 2
[48] J. Gao, C. Gu, Y. Lin, H. Zhu, X. Cao, L. Zhang, and Y. Yao,
“Relightable 3d gaussian: Real-time point cloud relighting with brdf
decomposition and ray tracing,” arXiv:2311.16043, 2023. 2
[49] Y. Jiang, J. Tu, Y. Liu, X. Gao, X. Long, W. Wang, and Y. Ma, “Gaus-
sianshader: 3d gaussian splatting with shading functions for reflective
surfaces,” arXiv preprint arXiv:2311.17977, 2023. 2
[50] Z. Liu, J. Su, G. Cai, Y. Chen, B. Zeng, and Z. Wang, “Georgs:
Geometric regularization for real-time novel view synthesis from sparse
inputs,” IEEE Transactions on Circuits and Systems for Video Technol-
ogy, vol. 34, no. 12, pp. 13 113–13 126, 2024. 2
[51] L. Fan, Y. Yang, M. Li, H. Li, and Z. Zhang, “Trim 3d gaussian splatting
for accurate geometry representation,” arXiv preprint arXiv:2406.07499,
2024. 2
[52] K. He, J. Sun, and X. Tang, “Single image haze removal using dark
channel prior,” IEEE transactions on pattern analysis and machine
intelligence, vol. 33, no. 12, pp. 2341–2353, 2010. 2
[53] N. Carlevaris-Bianco, A. Mohan, and R. M. Eustice, “Initial results in
underwater single image dehazing,” in Oceans 2010 Mts/IEEE Seattle.
IEEE, 2010, pp. 1–8. 2
[54] D. Akkaynak and T. Treibitz, “A revised underwater image formation
model,” in Proceedings of the IEEE conference on computer vision and
pattern recognition, 2018, pp. 6723–6732. 2, 3
[55] J. Xie, G. Hou, G. Wang, and Z. Pan, “A variational framework for
underwater image dehazing and deblurring,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 32, no. 6, pp. 3514–
3526, 2022. 2

<!-- page 14 -->
14
[56] A. V. Sethuraman, M. S. Ramanagopal, and K. A. Skinner, “Watern-
erf: Neural radiance fields for underwater scenes,” in OCEANS 2023-
MTS/IEEE US Gulf Coast.
IEEE, 2023, pp. 1–7. 2
[57] J. Zhou, T. Liang, D. Zhang, S. Liu, J. Wang, and E. Q. Wu, “Waterhe-
nerf: Water-ray matching neural radiance fields for underwater scene
reconstruction,” Information Fusion, vol. 115, p. 102770, 2025. 2
[58] Y. Bekerman, S. Avidan, and T. Treibitz, “Unveiling optical properties
in underwater images,” in 2020 IEEE International Conference on
Computational Photography (ICCP).
IEEE, 2020, pp. 1–12. 3, 6
[59] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao,
“Depth anything v2,” arXiv:2406.09414, 2024. 4
[60] B. Mildenhall, P. Hedman, R. Martin-Brualla, P. P. Srinivasan, and
J. T. Barron, “Nerf in the dark: High dynamic range view synthesis
from noisy raw images,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), June 2022, pp.
16 190–16 199. 4
[61] D. Akkaynak and T. Treibitz, “Sea-thru: A method for removing water
from underwater images,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2019, pp. 1682–1691. 5
[62] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113. 5
[63] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004. 6
[64] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595. 6
[65] Z. Wang, L. Shen, M. Xu, M. Yu, K. Wang, and Y. Lin, “Domain
adaptation for underwater image enhancement,” IEEE Transactions on
Image Processing, vol. 32, pp. 1442–1457, 2023. 6
[66] L. Peng, C. Zhu, and L. Bian, “U-shape transformer for underwater
image enhancement,” IEEE transactions on image processing, vol. 32,
pp. 3066–3079, 2023. 6
[67] G. Sharma, W. Wu, and E. N. Dalal, “The ciede2000 color-difference
formula: Implementation notes, supplementary test data, and mathemat-
ical observations,” Color Research & Application: Endorsed by Inter-
Society Color Council, The Colour Group (Great Britain), Canadian
Society for Color, Color Science Association of Japan, Dutch Society
for the Study of Color, The Swedish Colour Centre Foundation, Colour
Society of Australia, Centre Franc¸ais de la Couleur, vol. 30, no. 1, pp.
21–30, 2005. 6
[68] C. Boittiaux, R. Marxer, C. Dune, A. Arnaubec, M. Ferrera, and
V. Hugel, “Sucre: Leveraging scene structure for underwater color
restoration,” in 2024 International Conference on 3D Vision (3DV).
IEEE, 2024, pp. 1488–1497. 6
