<!-- page 1 -->
UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian
Splatting
Wenpeng Xing1, Jie Chen2, Zaifeng Yang3, Changting Lin4, Jianfeng Dong5, Chaochao Chen1,
Xun Zhou6, Meng Han1
1Zhejiang University
2Hong Kong Baptist University, Hong Kong SAR
3A*STAR, Singapore
4Binjiang Institute of
Zhejiang University
5Zhejiang Gongshang University
6Harbin Institute of Technology
Abstract
Underwater 3D scene reconstruction faces severe chal-
lenges from light absorption, scattering, and turbidity,
which degrade geometry and color fidelity in traditional
methods like Neural Radiance Fields (NeRF). While
NeRF extensions such as SeaThru-NeRF incorporate
physics-based models, their MLP reliance limits effi-
ciency and spatial resolution in hazy environments. We
introduce UW-3DGS, a novel framework adapting 3D
Gaussian Splatting (3DGS) for robust underwater re-
construction. Key innovations include: (1) a plug-and-
play learnable underwater image formation module us-
ing voxel-based regression for spatially varying atten-
uation and backscatter; and (2) a Physics-Aware Un-
certainty Pruning (PAUP) branch that adaptively re-
moves noisy floating Gaussians via uncertainty scor-
ing, ensuring artifact-free geometry. The pipeline oper-
ates in training and rendering stages. During training,
noisy Gaussians are optimized end-to-end with under-
water parameters, guided by PAUP pruning and scatter-
ing modeling. In rendering, refined Gaussians produce
clean Unattenuated Radiance Images (URIs) free from
media effects, while learned physics enable realistic
Underwater Images (UWIs) with accurate light trans-
port. Experiments on SeaThru-NeRF and UWBundle
datasets show superior performance, achieving PSNR
of 27.604, SSIM of 0.868, and LPIPS of 0.104 on
SeaThru-NeRF, with
65% reduction in floating arti-
facts.
Introduction
Accurate 3D scene reconstruction is fundamental to ap-
plications ranging from immersive virtual environments
to marine exploration and underwater archaeology. How-
ever, underwater imaging remains challenging due to depth-
dependent light absorption, scattering, and turbidity, which
degrade color fidelity and geometry. Traditional methods
and neural volumetric models like NeRF (Mildenhall et al.
2020) struggle in such conditions, as they assume clear me-
dia and cannot disentangle complex underwater light trans-
port.
Recent extensions such as SeaThru-NeRF (Levy et al.
2023) incorporate underwater image formation models, but
their reliance on MLPs limits spatial resolution and hampers
accurate geometry recovery in scattering-dominated scenes.
To address these limitations, we propose UW-3DGS,
a novel framework that adapts 3D Gaussian Splatting
(3DGS) (Kerbl et al. 2023) for underwater 3D reconstruc-
tion. Our method integrates two key innovations: (1) a
learnable underwater image formation module that simu-
lates wavelength-dependent attenuation and backscatter via
voxel-based parameter regression, and (2) a Physics-Aware
Uncertainty Pruning (PAUP) Branch that removes floating
Gaussians based on uncertainty scores, enhancing geomet-
ric fidelity.
UW-3DGS operates in two stages: during the Training
Stage, noisy 3D Gaussians are jointly optimized with under-
water parameters using end-to-end supervision from real un-
derwater images. The PAUP branch prunes unreliable Gaus-
sians, while the image formation module learns spatially
varying scattering effects. In the Rendering Stage, the re-
fined 3D Gaussians, optimized through the training process,
are directly rasterized to produce clean, water-independent
Unattenuated Radiance Images (URIs), capturing the intrin-
sic scene radiance free from scattering effects. Meanwhile,
the learned physics parameters from the learnable under-
water image formation module are applied to these Gaus-
sians to generate realistic Underwater Images (UWIs), incor-
porating accurate light attenuation and backscatter as sim-
ulated by the module. This dual-output capability, driven
by the module’s spatially adaptive modeling, supports high-
fidelity novel view synthesis and facilitates downstream vi-
sual tasks such as marine mapping, ecological analysis, and
autonomous underwater navigation.
Extensive experiments on real-world datasets demonstrate
UW-3DGS’s superior performance. On the SeaThru-NeRF
dataset, it achieves a PSNR of 27.604, SSIM of 0.868, and
LPIPS of 0.104. Our method reduces floating artifacts by
65% and preserves fine-grained structures such as coral tex-
tures and seabed contours, outperforming prior approaches
in both geometric accuracy and visual realism.
Contributions include:
• Pioneering integration of learnable underwater physics
into 3DGS, enabling exceptional URI and UWI quality.
• Novel PAUP Branch for uncertainty-driven pruning,
yielding artifact-free underwater geometry.
• Demonstrated advancements in reconstruction accuracy
on challenging underwater datasets, advancing practical
arXiv:2508.06169v1  [cs.CV]  8 Aug 2025

<!-- page 2 -->
utility.
Related Work
Neural Radiance Fields (NeRF) (Mildenhall et al. 2020)
have revolutionized 3D scene reconstruction and novel view
synthesis, demonstrating exceptional fidelity. Their versatil-
ity extends to 2D image enhancement tasks, including de-
noising (Pearl, Treibitz, and Korman 2022), deblurring (Ma
et al. 2022), super-resolution (Wang et al. 2022), and low-
light enhancement (Mildenhall et al. 2022), as well as
robotics applications such as Simultaneous Localization and
Mapping (SLAM) (Rosinol, Leonard, and Carlone 2023;
Yan et al. 2023) and robotic grasping (Kerr et al. 2023).
Participating Media and Underwater NeRF
Adaptations of NeRF to participating media, especially
underwater environments, have gained traction to address
light scattering and absorption challenges. Early works like
SeaThru-NeRF (Levy et al. 2023) and WaterNeRF (Sethura-
man, Ramanagopal, and Skinner 2023) incorporate physics-
based image formation models into NeRF’s volumet-
ric framework, using MLPs to simulate attenuation and
backscattering. WaterHE-NeRF (Zhou et al. 2023) leverages
histogram equalization for pseudo-ground truth supervision,
while Dehaze-NeRF (Chen et al. 2023) applies atmospheric
scattering models to hazy scenes. Recent advancements ex-
tend these foundations. NeuroPump (Guo et al. 2024) in-
troduces self-supervised geometric and color rectification to
”pump out” water effects in NeRF reconstructions. AquaN-
eRF (Gough et al. 2025) proposes an MLP-based scheme for
distractor-aware rendering. Despite these innovations, MLP
reliance often results in prolonged training times, and vali-
dations are frequently limited to controlled settings. Emerg-
ing methods increasingly incorporate 3DGS for efficiency,
as discussed below.
Underwater 3D Gaussian Splatting
Building on 3DGS’s efficiency, recent works adapt it for
underwater reconstruction to mitigate scattering-induced ar-
tifacts. SeaSplat (Yang, Leonard, and Girdhar 2024) en-
ables real-time rendering by combining 3DGS with a
physically grounded image formation model, disentangling
medium effects from scene radiance. WaterSplatting (Li
et al. 2024) fuses volumetric rendering with 3DGS, incorpo-
rating distractor-aware mechanisms for enhanced clarity in
turbid waters. Further developments include UW-GS (Wang
et al. 2025), a distractor-aware variant with physics-based
density control, and RUSplatting (Jiang et al. 2025), which
bolsters robustness for sparse-view scenarios through im-
proved Gaussian optimization. Water-Adapted 3DGS (Fan
et al. 2025) introduces complexity-adaptive point distri-
bution and depth-based multi-scale rendering for precise
scene recovery. For dynamic environments, UDR-GS (Du
et al. 2024) extends to 4D Gaussians, addressing temporal
variations in underwater light propagation. These methods
demonstrate improved scalability for open-ocean applica-
tions, though challenges in handling extreme turbidity and
real-time deployment persist. In contrast, our UW-3DGS
distinguishes itself by introducing a physics-aware uncer-
tainty pruning branch to adaptively suppress floating Gaus-
sians and a plug-and-play learnable underwater image for-
mation module with voxel-based regression, enabling supe-
rior media-free reconstruction and end-to-end optimization.
Light Propagation in Scattering Media
Fundamental research on light propagation in scattering me-
dia underpins these advancements. SeaThru models (Akkay-
nak and Treibitz 2018, 2019; Akkaynak et al. 2017) empha-
size wavelength-dependent parameters in underwater op-
tics. Recent reviews by Yang et al. (Yang et al. 2019)
cover monocular restoration techniques, while Sharma et
al. (Sharma, Kumar, and Singla 2021) survey deep learning-
based defogging. For a thorough overview, consult the ref-
erenced surveys.
Preliminaries
3D Gaussian Splatting (3DGS)
UW-3DGS builds upon 3DGS (Kerbl et al. 2023), which
represents scenes as sets of anisotropic Gaussians {Gi |
i ∈[1, N]} for efficient tile-based rasterization and real-
time rendering. This representation is particularly promising
for underwater scenes, where traditional volumetric methods
like NeRF struggle with scattering-induced artifacts.
Gaussians are initialized from sparse point clouds gen-
erated by Structure-from-Motion (SfM) tools such as
COLMAP (Sch¨onberger and Frahm 2016). Each Gaussian
Gi includes view-dependent color ci (modeled via spheri-
cal harmonics) and opacity αi. The position and shape are
defined by mean µi
W and covariance Σi
W in world space,
decomposed as:
ΣW = RSSTRT,
(1)
where S is the scaling matrix and R is the rotation matrix.
During rasterization, 3D Gaussians are projected to 2D
via:
µI = π(T CW µW ) , ΣI = JWΣW WT JT ,
(2)
where π(·) is the projection operation, J is the Jacobian of
the affine approximation, T CW ∈SE(3) is the camera
pose, and W is the viewing transformation.
Pixel colors C are computed through alpha blending:
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(3)
accumulating color contributions modulated by opacity and
transmittance T = Qi−1
j=1(1 −αj). This fully differentiable
formulation enables gradient-based optimization of Gaus-
sian parameters, enhancing scene representation and image
fidelity.
Underwater Image Formation Model
To simulate light propagation in scattering media, UW-
3DGS adopts a physically grounded underwater image for-
mation model, enabling realistic rendering and geometry-
aware restoration.

<!-- page 3 -->
Rasterization
Base Rendering 
Branch
Depth
Preliminary Unattenuated 
Radiance Image
Learnable 
Underwater 
Image 
Formation 
Physics-Aware 
Uncertainty 
Pruning
Underwater 
Environment Parameters 
Learning
Loss
Enhanced Unattenuated 
Radiance Image
Underwater Image
Gradient
Rasterization
Gradient
Input
Gradient
Gradient
Gradient
Input
Input
Rendering
Image 
Rendering
Loss
All 3D Gaussians
Noisy floating 3D 
Gaussians simulating 
water
��
�
��
�
��
�
��
�,�
��
�,�
(�, �, �)
Modeling
(1) Training Stage
3D Gaussians
Learnable 
Underwater 
Image 
Formation 
Underwater 
Image
Unattenuated 
Radiance 
Image
(2) Rendering Stage
Rasterization
Figure 1: Architecture of UW-3DGS. In the training Stage, the Base Rendering Branch generates the preliminary Unattenuated
Radiance Image (URI) and depth, the PAUP Branch prunes floating Gaussians using PAPSL, and the Learnable Underwater
Image Formation Module applies scattering effects to produce the underwater image, all optimized end-to-end. In the Rendering
Stage, refined Gaussians yield clean URIs, while learnable underwater image formation module enable realistic UWIs with
accurate attenuation and backscatter.
Early models (Jaffe 1990; Schechner and Karpel 2005)
express observed intensity I(x) at pixel x as:
I(x) = D(x) + B(x),
(4)
where D(x) is the attenuated direct signal from the scene
point, and B(x) is the backscattered light by water parti-
cles. Color degradation primarily stems from wavelength-
dependent attenuation in D(x), while B(x) reduces contrast
via a veiling effect.
We adopt a revised formulation (Akkaynak and Treibitz
2018) for precise physical modeling:
I = J · e−βD·z
|
{z
}
direct transmission
+ B∞· (1 −e−βB·z)
|
{z
}
backscattering
,
(5)
where J is the intrinsic scene radiance, βD and βB are at-
tenuation coefficients for direct and backscatter signals, z is
the scene depth, and B∞is the far-field veiling light. The
first term models exponential decay due to absorption and
scattering, while the second captures accumulated backscat-
ter increasing with depth.
Approach
Directly applying 3D Gaussian Splatting (3DGS) (Kerbl
et al. 2023) to underwater imagery yields noisy, floating
Gaussians due to unmodeled light absorption and scattering,
causing distorted geometry, inaccurate colors, and loss of
details like seabed contours—limiting marine exploration,
robotics, and ecological applications.
We propose UW-3DGS, an end-to-end framework for un-
derwater 3D reconstruction that disentangles scattering ef-
fects from intrinsic scene properties, enhancing geometry
accuracy and novel view synthesis. It comprises three in-
tegrated components:
1. Base Rendering Branch: An adapted 3DGS pipeline
generating initial depth maps and unattenuated radiance
images.
2. Physics-Aware Uncertainty Pruning (PAUP): An aux-
iliary branch pruning noisy Gaussians using voxel-wise
uncertainty for improved consistency.
3. Learnable Underwater Image Formation Module: A
physics-based model simulating light propagation with
spatially varying parameters via voxel regression guided
by PAUP.
The framework operates in two stages (Figure. 1):
1. Training Stage. Starting with noisy 3D Gaussians, the
Base Rendering Branch produces initial radiance images
and depth maps, while PAUP prunes unreliable Gaus-
sians based on uncertainty. These feed into the Learnable
Underwater Image Formation Module to model attenua-
tion and scattering with learned parameters. End-to-end
training minimizes rendering errors against ground truth
using gradient-based losses.
2. Rendering Stage. Post-training, refined Gaussians and
learned parameters render high-fidelity unattenuated ra-
diance images and underwater views, preserving details
efficiently for novel view synthesis, visualization, and
navigation.

<!-- page 4 -->
Base Rendering Branch
The
Base
Rendering
Branch
adapts
the
3DGS
pipeline (Kerbl et al. 2023) to generate initial represen-
tations, including the preliminary Unattenuated Radiance
Image (URI) ˆIUR and depth map z, which serve as inputs
for the PAUP and underwater image formation modules.
Unlike standard 3DGS, we incorporate underwater-specific
optimizations to mitigate scattering-induced artifacts.
Rendering follows 3DGS rasterization (Eq. (2)), produc-
ing:
ˆIUR =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj).
(6)
Depth is computed with uncertainty weighting from the
PAUP branch:
z =
X
i∈N
ziαi(1 −U i)
i−1
Y
j=1
(1 −αj),
(7)
where zi is the Gaussian’s depth and U i is the uncertainty
component from PAUP, prioritizing low-uncertainty contri-
butions to mitigate scattering-induced depth errors.
Physics-Aware Uncertainty Pruning Branch
To suppress noisy floating Gaussians and enhance recon-
struction quality, we introduce the Physics-Aware Uncer-
tainty Pruning (PAUP) Branch, operating in parallel with the
Base Rendering Branch. This branch uses a physics-aware
uncertainty score (PUS) to guide adaptive pruning and pro-
vide uncertainty feedback for parameter regression in the
underwater module.
For the i-th Gaussian Gi, PUS is computed as:
PUSi = wu · U i + wp · P i,
(8)
where wu, wp are learnable weights initialized to 0.5. The
uncertainty component U i captures rendering instability:
U i = wα · Varviews(αi
eff,k) + wc · Varviews(ci(vk)),
(9)
with wα = 0.4, wc = 0.6. The effective opacity is:
αi
eff,k = αi ·
i−1
Y
j=1
(1 −αj),
(10)
and variances are computed over K = 5 neighboring
views:
Varviews(αi
eff,k) = 1
K
K
X
k=1
 αi
eff,k −¯αi
eff
2 ,
(11)
Varviews(ci(vk)) = 1
K
K
X
k=1
∥ci(vk) −¯ci∥2
2.
(12)
The physics component P i ensures consistency with the
underwater model:
P i = |zi −ˆzi| +
αi · (1 −e−ˆβD(xi)·zi)
 ,
(13)
where ˆzi is the depth predicted by the Base Rendering
Branch for Gaussian i, computed as the distance from the
camera to the Gaussian’s center.
PUS is fed into a lightweight MLP ϕ (2-layer, 32 hidden
units) to predict pruning probability:
mi = σ(ϕ(PUSi)),
(14)
where σ is the sigmoid function. Pruning employs
Gumbel-Softmax (Jang, Gu, and Poole 2016):
{Gi
Pruned}NPruned
i=1
= {Gi | mi < τadapt},
(15)
with τadapt as the 95% of mi, updated per iteration.
Pruned Gaussians produce IEnhan.
UR
, reducing scattering ar-
tifacts compared to ˆIUR. The Physics-Aware Pruning Super-
vision Loss (PAPSL) is defined in Section .
Learnable Underwater Image Formation Module
The underwater image formation module simulates light
propagation in scattering media, building on Eq. (5). It uti-
lizes the preliminary Unattenuated Radiance Image (URI)
ˆIUR from the Base Rendering Branch as input to synthe-
size the underwater image. Integrated with the refinement
from the PAUP branch, it replaces intrinsic radiance J with
the enhanced Unattenuated Radiance Image (URI) IEnhan.
UR
,
yielding:
I = IEnhan.
UR
· exp(−βD(vD) · z)
+ B∞·
 1 −exp(−βB(vB) · z)

,
(16)
where unknowns include attenuation coefficients βB, βD,
directional dependencies vB, vD, depth z, and veiling light
B∞. To enable spatial variability while improving effi-
ciency, we adopt a tensor-decomposed voxel grid for regres-
sion, guided by PUS from the PAUP branch.
Unlike prior MLP-based approaches (Levy et al. 2023;
Sethuraman, Ramanagopal, and Skinner 2023), our method
leverages low-rank tensor decomposition inspired by Ten-
soRF (Chen et al. 2022), reducing memory and query costs.
We adopt:
• Veiling Light: B∞is a learnable RGB vector ˆB∞∈R3.
• Attenuation Coefficients: βB and βD are regressed us-
ing voxel grids V D, V B ∈RG×G×G×3 (G = 64), via
vector-matrix (VM) decomposition:
V D ≈
R
X
r=1
uD
r MD
r (vD
r ◦wD
r ),
(17)
where R
= 16, uD
r
∈RG, MD
r
∈RG×G, and
vD
r , wD
r
∈RG. Parameters are queried as ˆβD(x) =
Query(V D, x), ˆβB(x) = Query(V B, x) via trilinear in-
terpolation.
• Depth Estimation: Depth z is sourced from Eq. (7).

<!-- page 5 -->
• Directional Dependencies: We assume isotropic media,
omitting vB, vD, to focus on spatial variations.
The final rendering equation is:
IUW = IEnhan.
UR
· exp(−ˆβD(x) · z)
+ ˆB∞· (1 −exp(−ˆβB(x) · z)).
(18)
Loss Function
To optimize UW-3DGS, we define a total loss that integrates
all components for end-to-end training:
Ltotal = Lbase + λPAPSLLPAPSL + λβLβ + λzLz,
(19)
where λPAPSL = 0.1, λβ = 0.05, and λz = 0.05 balance
the contributions of each term. Below, we detail each loss
function, its purpose, and its components.
Image Rendering Loss (LIMG)
The image rendering loss
ensures that the rendered underwater image IUW matches the
ground-truth underwater image IGT:
LIMG = (1−λ) ∥IUW −IGT∥1+λLD-SSIM(IUW, IGT), (20)
where λ = 0.2 balances the L1 loss (pixel-wise intensity dif-
ference) and the differentiable SSIM loss (LD-SSIM), which
captures structural similarity. The L1 term is computed over
all pixels in the image.
Physics-Aware Pruning Supervision Loss (LPAPSL)
The
PAUP branch is optimized with:
LPAPSL =
IUR −IEnhan.
UR

1 + λs
X
i
(1 −mi) + λw∥ϕ∥2
2,
(21)
where λs = 0.01, λw = 0.001. The first term (L1 over
pixels) encourages similarity between unpruned and pruned
Unattenuated Radiance Images, reducing scattering arti-
facts. The second term (sum over Gaussians i) promotes
pruning by penalizing high pruning probabilities mi. The
third term regularizes the MLP ϕ to prevent overfitting.
Attenuation Regression Loss (Lβ)
The attenuation coef-
ficients are regressed with the loss function:
Lβ =
X
x
PUS(x) · ∥ˆβ(x) −βprior∥2
2+
λr
R
X
r=1
 ∥uD
r ∥2
2 + ∥vD
r ∥2
2 + ∥wD
r ∥2
2 + ∥MD
r ∥2
F

,
(22)
where λr = 0.001, and the first sum is over all voxel
positions x. βprior is an empirical mean attenuation coeffi-
cient. The PUS term weights the loss to prioritize scattering-
dominated regions. A symmetric term applies to ˆβB. The
second term regularizes the Vector-Matrix (VM) decompo-
sition components for the voxel grid V D, penalizing the L2
norm of the vectors (uD
r , vD
r , wD
r ) and the Frobenius norm
of the matrices (MD
r ).
Depth Refinement Loss (Lz)
The depth refinement loss
is:
Lz =
X
i
(1 −U i) · |z −ˆzi|,
(23)
where the sum is over Gaussians i, z is the rendered depth
from Eq. (7), and ˆzi is the predicted depth for Gaussian i
(distance from the camera to its center). The (1 −U i) term
prioritizes low-uncertainty Gaussians to refine depth esti-
mates.
Implementation Details
UW-3DGS is implemented in PyTorch with CUDA, based
on 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023), and
trained on a single NVIDIA Tesla V100 GPU (32 GB).
Training runs for 40,000 iterations using the Adam opti-
mizer (Kingma and Ba 2014) with a batch size of one image.
Key configurations include:
• Hyperparameters: Learning rates are 0.05 (opacity),
0.005 (scaling), 0.001 (rotation), and 0.001 (ˆβB, ˆβD,
ˆB∞). Loss weights: λPAPSL = 0.1, λβ = λz = 0.05,
λ = 0.2 (LIMG), λβ
= 0.01 (Lbase), λs
= 0.01,
λw = λr = 0.001. Voxel grid resolution is G = 64, rank
R = 16. PAUP parameters: wu = wp = 0.5, wα = 0.4,
wc = 0.6, K = 5. MLP ϕ: 2 layers, 32 units.
• Training Setup: Gaussian densification starts at iteration
500 (rate 0.01), with opacity resets every 3,000 iterations.
The PAUP branch activates at iteration 500, with prun-
ing threshold τadapt as the median of pruning probabili-
ties mi, updated per iteration. Spherical harmonics are
truncated at order three.
• Preprocessing: SeaThru-NeRF images (Levy et al. 2023)
are white-balanced; UWBundle images (Skinner, Ru-
land, and Johnson-Roberson 2017) use raw data. Ini-
tial Gaussians come from COLMAP (Sch¨onberger and
Frahm 2016). Images are resized to 1024×1024. βprior is
set to [0.1, 0.15, 0.2] (RGB) from dataset statistics.
Experiments
This section introduces the experiment settings and results.
All experimental results are obtained through our rerunning.
Datasets
We evaluate UW-3DGS on UWBundle (Skinner, Ruland,
and Johnson-Roberson 2017) and SeaThru-NeRF (Levy
et al. 2023) datasets, covering synthetic and real-world
underwater scenarios. UWBundle has 36 synthetic im-
ages of a submerged rock platform, captured in a lab
with a lawnmower trajectory. SeaThru-NeRF includes 58
white-balanced images from the Pacific (Panama), Red
Sea (Israel), and Caribbean (Curac¸ao), with challenges
like variable water properties. Official training/testing splits
from (Mildenhall et al. 2019; Levy et al. 2023) ensure fair
comparisons.

<!-- page 6 -->
GT 
GT 
TensoRF
SeaThru-NeRF
3DGS
Ours
Figure 2: Visualization of rendered underwater images.
Evaluation Rubrics
We assess UW-3DGS on three aspects: rendering quality of
underwater (UWI) and no-water (URI) images, training ef-
ficiency, and 3D reconstruction quality without water. Ren-
dering quality is measured against ground-truth images us-
ing PSNR (↑), SSIM (↑), and LPIPS (↓) (Zhang et al. 2018).
Competing Methods
We compare UW-3DGS with key methods for underwater
3D reconstruction, focusing on rendering quality and effi-
ciency. These methods include 3DGS (Kerbl et al. 2023),
TensoRF (Chen et al. 2022), SeaThru-NeRF (Levy et al.
2023), WaterSplatting(Li et al. 2024), SeaSplat (Yang,
Leonard, and Girdhar 2024).
Underwater Image Rendering Quality Comparison
Table 1 presents the quantitative evaluation of novel view
synthesis quality for underwater images (UWIs) on the
SeaThru-NeRF dataset. Among all variants, our method per-
forms favorably on UWI rendering tasks, achieving the best
SSIM among compared methods. These results confirm that
modeling underwater light attenuation and scattering signif-
icantly improves photorealistic rendering under aquatic con-
ditions.
Unattenuated Radiance Image Rendering Quality
Comparisons
The visualization results in Figure 3 demonstrate the effec-
tiveness of our method trained with the Physics-Aware Prun-
ing Supervision Loss (PAPSL), producing clear Unattenu-
ated Radiance Images (URI) of the seabed by directly ren-
dering 3D Gaussians. Compared to standard 3DGS, which
Table 1: Quantitative comparisons of underwater image ren-
dering quality averaged on the SeaThru-NeRF dataset.
Method
Metric
PSNR↑
SSIM↑
LPIPS↓
3DGS
26.113
0.861
0.216
TensoRF
24.307
0.787
0.285
SeaThru-NeRF
25.768
0.806
-
WaterSplatting
29.687
0.830
0.120
SeaSplat
27.108
0.835
0.183
Ours
27.604
0.868
0.134
exhibits noisy floating Gaussians and blurred topography,
UW-3DGS yields sharper geometric details, such as well-
defined coral formations and marine flora, with minimal ar-
tifacts, highlighting PAPSL’s role in suppressing scattering-
induced noise. Furthermore, Figure 3 shows URI com-
parisons between UW-3DGS and SeaThru-NeRF on the
SeaThru-NeRF dataset. Our method reconstructs underwa-
ter scenes with superior clarity, preserving intricate seabed
contours and reducing volumetric haze, leading to more ac-
curate and visually coherent results. This underscores UW-
3DGS’s advantage in disentangling scattering effects, result-
ing in better overall underwater reconstruction quality, in-
cluding enhanced depth accuracy and artifact-free geometry,
essential for applications like marine exploration.
To validate the contributions of UW-3DGS’s key com-
ponents—Base Rendering Branch, Physics-Aware Uncer-
tainty Pruning (PAUP) Branch, and Learnable Underwater
Image Formation Module—we conduct ablation studies on
the SeaThru-NeRF dataset. We evaluate variants by remov-

<!-- page 7 -->
(a) 3DGS
(b) Ours - w/ LPAPSL
Figure 3: Unattenuated Radiance Images (URI) of seabed, directly rendered from 3D Gaussians trained with the Physics-Aware
Pruning Supervision Loss (PAPSL), demonstrating effective suppression of floating artifacts.
Table 2: Ablation study results on SeaThru-NeRF dataset (averaged across scenes).
Variant
PSNR↑
SSIM↑
LPIPS↓
Train Time (min)
Big Float Gauss Ratio (%) ↓
w/o PAUP
25.374
0.837
0.266
42
8.2
w/o Underwater Module
24.912
0.812
0.298
28
6.5
w/o Lβ
25.754
0.851
0.247
46
5.8
w/o Lz
26.028
0.859
0.266
47
7.4
Full UW-3DGS
27.604
0.868
0.134
48
1.3
ing or modifying components and assess impacts on render-
ing quality (PSNR, SSIM, LPIPS), training time, and ge-
ometric fidelity (measured by floating Gaussian ratio, i.e.,
percentage of pruned Gaussians). Details will be discussed
in the subsequent section.
Ablation Study
We test the following variants:
• w/o PAUP: Disables the PAUP Branch and LPAPSL, rely-
ing only on base rendering and underwater module.
• w/o Underwater Module: Removes the learnable under-
water model (Eq. (18)) and associated losses (Lβ, Lz),
using standard 3DGS rendering.
• w/o Lβ: Omits attenuation regression loss, using fixed
βprior instead of learned coefficients.
• w/o Lz: Disables depth refinement loss, using un-
weighted depth computation.
• Full UW-3DGS: Complete method with all components.
Results are in Table 2, the full UW-3DGS model achieves
the highest rendering quality (PSNR: 27.604, SSIM: 0.868,
LPIPS: 0.134) and the lowest big floating Gaussian ra-
tio (1.3%), demonstrating superior geometric fidelity and
artifact reduction. Ablations reveal that removing PAUP
markedly increases the ratio to 8.2% and degrades PSNR
by over 2 dB, underscoring its role in pruning noisy Gaus-
sians. Similarly, omitting the underwater module or specific
losses elevates artifacts and lowers performance, confirming
the synergistic necessity of all components for balanced ef-
ficiency and fidelity.
Conclusions
We propose UW-3DGS, an efficient framework for under-
water 3D scene reconstruction that integrates a physically
grounded image formation model into the 3D Gaussian
Splatting pipeline. This enables simultaneous geometry re-
covery and color restoration, producing high-fidelity ren-
derings of both underwater and media-free appearances. To
address scattering artifacts, we introduce a Physics-Aware
Uncertainty Pruning Branch, which refines noisy Gaussians
and yields clean, physically consistent reconstructions. UW-
3DGS excels in media-free rendering by generating clear
radiance images and depth maps, preserving fine details
such as coral textures and seabed structures—critical for
marine ecology and robotic perception. Experiments on the
SeaThru-NeRF dataset demonstrate superior rendering qual-
ity and geometric accuracy. UW-3DGS offers a promising
solution for underwater exploration, marine robotics, and
environmental monitoring.
Limitations
(1) The fixed voxel grid resolution with tensor decompo-
sition may inadequately capture fine spatial variations in
large-scale scenes, especially in environments with signif-
icant depth changes. (2) The PAUP branch’s uncertainty
computation relies on variance over neighboring views, po-
tentially reducing robustness in sparse viewpoint scenarios
common in underwater data collection.
References
Akkaynak, D.; and Treibitz, T. 2018. A revised underwater
image formation model. In Proceedings of the IEEE con-

<!-- page 8 -->
ference on computer vision and pattern recognition, 6723–
6732.
Akkaynak, D.; and Treibitz, T. 2019. Sea-thru: A method for
removing water from underwater images. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, 1682–1691.
Akkaynak, D.; Treibitz, T.; Shlesinger, T.; Loya, Y.; Tamir,
R.; and Iluz, D. 2017. What is the space of attenuation co-
efficients in underwater computer vision? In Proceedings of
the IEEE conference on computer vision and pattern recog-
nition, 4931–4940.
Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Ten-
soRF: Tensorial Radiance Fields. In ECCV.
Chen, W.-T.; Yifan, W.; Kuo, S.-Y.; and Wetzstein, G. 2023.
Dehazenerf: Multiple image haze removal and 3d shape re-
construction using neural radiance fields.
arXiv preprint
arXiv:2303.11364.
Du, Y.; Zhang, Z.; Zhang, P.; Sun, F.; and Lv, X. 2024.
Udr-gs: Enhancing underwater dynamic scene reconstruc-
tion with depth regularization. Symmetry, 16(8): 1010.
Fan, X.; Wang, X.; Ni, H.; Xin, Y.; and Shi, P. 2025. Water-
Adapted 3D Gaussian Splatting for precise underwater scene
reconstruction. Frontiers in Marine Science, 12: 1573612.
Gough, L.; Azzarelli, A.; Zhang, F.; and Anantrasirichai, N.
2025. AquaNeRF: Neural radiance fields in underwater me-
dia with distractor removal.
In 2025 IEEE International
Symposium on Circuits and Systems (ISCAS), 1–5. IEEE.
Guo, Y.; Liao, H.; Ling, H.; and Huang, B. 2024. NeuroP-
ump: Simultaneous Geometric and Color Rectification for
Underwater Images. arXiv preprint arXiv:2412.15890.
Jaffe, J. S. 1990. Computer modeling and the design of opti-
mal underwater imaging systems. IEEE Journal of Oceanic
Engineering, 15(2): 101–111.
Jang, E.; Gu, S.; and Poole, B. 2016.
Categorical
reparameterization with gumbel-softmax.
arXiv preprint
arXiv:1611.01144.
Jiang, Z.; Wang, H.; Huang, G.; Seymour, B.; and
Anantrasirichai, N. 2025. RUSplatting: Robust 3D Gaus-
sian Splatting for Sparse-View Underwater Scene Recon-
struction. arXiv preprint arXiv:2505.15737.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian Splatting for Real-Time Radiance Field
Rendering. ACM Transactions on Graphics, 42(4).
Kerr, J.; Fu, L.; Huang, H.; Avigal, Y.; Tancik, M.; Ich-
nowski, J.; Kanazawa, A.; and Goldberg, K. 2023.
Evo-
nerf: Evolving nerf for sequential robot grasping of trans-
parent objects. In Conference on Robot Learning, 353–367.
PMLR.
Kingma, D. P.; and Ba, J. 2014.
Adam: A method for
stochastic optimization. arXiv preprint:1412.6980.
Levy, D.; Peleg, A.; Pearl, N.; Rosenbaum, D.; Akkaynak,
D.; Korman, S.; and Treibitz, T. 2023. SeaThru-NeRF: Neu-
ral Radiance Fields in Scattering Media. In CVPR, 56–65.
Li, H.; Song, W.; Xu, T.; Elsig, A.; and Kulhanek, J. 2024.
Watersplatting: Fast underwater 3d scene reconstruction us-
ing gaussian splatting. arXiv preprint arXiv:2408.08206.
Ma, L.; Li, X.; Liao, J.; Zhang, Q.; Wang, X.; Wang, J.; and
Sander, P. V. 2022. Deblur-nerf: Neural radiance fields from
blurry images. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 12861–12870.
Mildenhall, B.; Hedman, P.; Martin-Brualla, R.; Srinivasan,
P. P.; and Barron, J. T. 2022. Nerf in the dark: High dynamic
range view synthesis from noisy raw images. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 16190–16199.
Mildenhall, B.; Srinivasan, P. P.; Ortiz-Cayon, R.; Kalantari,
N. K.; Ramamoorthi, R.; Ng, R.; and Kar, A. 2019. Local
Light Field Fusion: Practical View Synthesis with Prescrip-
tive Sampling Guidelines. ACM Transactions on Graphics,
38(4): 1–14.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing
Scenes as Neural Radiance Fields for View Synthesis. In
ECCV, 405–421.
Pearl, N.; Treibitz, T.; and Korman, S. 2022. Nan: Noise-
aware nerfs for burst-denoising.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 12672–12681.
Rosinol, A.; Leonard, J. J.; and Carlone, L. 2023. Nerf-slam:
Real-time dense monocular slam with neural radiance fields.
In IROS, 3437–3444.
Schechner, Y. Y.; and Karpel, N. 2005. Recovery of under-
water visibility and structure by polarization analysis. IEEE
Journal of oceanic engineering, 30(3): 570–587.
Sch¨onberger, J. L.; and Frahm, J.-M. 2016. Structure-from-
Motion Revisited. In CVPR, 4104–4113.
Sethuraman, A. V.; Ramanagopal, M. S.; and Skinner, K. A.
2023.
Waternerf: Neural radiance fields for underwater
scenes. In OCEANS 2023-MTS/IEEE US Gulf Coast, 1–7.
IEEE.
Sharma, N.; Kumar, V.; and Singla, S. K. 2021. Single image
defogging using deep learning techniques: past, present and
future. Archives of Computational Methods in Engineering,
28: 4449–4469.
Skinner, K. A.; Ruland, E. I.; and Johnson-Roberson, M.
2017. Automatic Color Correction for 3D Reconstruction
of Underwater Scenes. In IEEE International Conference
on Robotics and Automation.
Wang, C.; Wu, X.; Guo, Y.-C.; Zhang, S.-H.; Tai, Y.-W.; and
Hu, S.-M. 2022. Nerf-sr: High quality neural radiance fields
using supersampling. In Proceedings of the 30th ACM In-
ternational Conference on Multimedia, 6445–6454.
Wang, H.; Anantrasirichai, N.; Zhang, F.; and Bull, D.
2025.
UW-GS: Distractor-aware 3d gaussian splatting
for enhanced underwater scene reconstruction.
In 2025
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), 3280–3289. IEEE.
Yan, C.; Qu, D.; Wang, D.; Xu, D.; Wang, Z.; Zhao, B.; and
Li, X. 2023. Gs-slam: Dense visual slam with 3d gaussian
splatting. arXiv preprint arXiv:2311.11700.
Yang, D.; Leonard, J. J.; and Girdhar, Y. 2024. Seasplat:
Representing underwater scenes with 3d gaussian splatting

<!-- page 9 -->
and a physically grounded image formation model. arXiv
preprint arXiv:2409.17345.
Yang, M.; Hu, J.; Li, C.; Rohde, G.; Du, Y.; and Hu, K. 2019.
An in-depth survey of underwater image enhancement and
restoration. IEEE Access, 7: 123638–123657.
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In CVPR, 586–595.
Zhou, J.; Liang, T.; He, Z.; Zhang, D.; Zhang, W.; Fu, X.;
and Li, C. 2023. WaterHE-NeRF: Water-ray Tracing Neu-
ral Radiance Fields for Underwater Scene Reconstruction.
arXiv preprint arXiv:2312.06946.
