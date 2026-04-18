<!-- page 1 -->
ASAP-Textured Gaussians: Enhancing Textured Gaussians with Adaptive
Sampling and Anisotropic Parameterization
Meng Wei1
Cheng Zhang1
Jianmin Zheng2
Hamid Rezatofighi1
Jianfei Cai1
1Monash University
2Nanyang Technological University
Abstract
Recent advances have equipped 3D Gaussian Splatting
with texture parameterizations to capture spatially vary-
ing attributes, improving the performance of both appear-
ance modeling and downstream tasks. However, the added
texture parameters introduce significant memory efficiency
challenges.
Rather than proposing new texture formula-
tions, we take a step back to examine the characteristics
of existing textured Gaussian methods and identify two key
limitations in common: (1) Textures are typically defined
in canonical space, leading to inefficient sampling that
wastes textures’ capacity on low-contribution regions; and
(2) texture parameterization is uniformly assigned across
all Gaussians, regardless of their visual complexity, result-
ing in over-parameterization.
In this work, we address
these issues through two simple yet effective strategies:
adaptive sampling based on the Gaussian density distri-
bution and error-driven anisotropic parameterization that
allocates texture resources according to rendering error.
Our proposed ASAP-Textured Gaussians, short for Adap-
tive Sampling and Anisotropic Parameterization, signifi-
cantly improve the quality–efficiency trade-off, achieving
high-fidelity rendering with far fewer texture parameters.
1. Introduction
3D Gaussian Splatting (3DGS) [10] has emerged as an ef-
ficient and effective representation for 3D scenes, offering
real-time performance with photorealistic rendering qual-
ity. Originally developed for novel view synthesis [31] and
3D reconstruction [9], it has since been extended to various
downstream tasks by associating each Gaussian with task-
specific attributes. Such extensions have enabled 3DGS to
function as a general-purpose 3D backbone across appli-
cations, including dynamic scene modeling [22], physical
simulation [23], and 3D scene understanding [13].
Recent works have explored enriching the expressive
capacity of Gaussians by embedding spatially varying at-
tributes through local texture parameterizations. These tex-
tured representations have been applied to encode appear-
ance [17, 25, 26], geometry [5, 19], and materials [28], sig-
nificantly broadening the representational scope of 3DGS.
In particular, textured Gaussians [5, 19, 28] for appearance
have proven effective in capturing high-frequency details
and structural variations through RGBA textures, surpass-
ing the limitations of uniform color and ellipsoidal shape.
However, while textured Gaussians improve image fi-
delity, they also introduce new memory efficiency chal-
lenges: Attaching a dedicated texture to each Gaussian in-
troduces significant memory overhead and complicates the
balance of rendering quality and resource usage. This raises
a central question: Can we improve the utilization of texture
resources in textured Gaussian Splatting without sacrific-
ing rendering quality? Addressing this question is critical
for advancing the scalability and generalizability of textured
Gaussians.
In this work, rather than proposing new texture formu-
lations, we take a step back to analyze the properties of
existing textured Gaussian representations and identify two
fundamental limitations in common. 1) First, most meth-
ods treat the Gaussian’s canonical space as its texture space,
which leads to inefficient sampling—many texture samples
are assigned to regions with negligible visual contribution.
2) Second, texture parameterization is typically fixed and
uniformly distributed across all Gaussians, regardless of
each primitive’s visual complexity or importance. This uni-
formity can lead to over-parameterization in simple areas,
while under-representing regions that require more detail,
ultimately reducing efficiency.
To address these limitations, we introduce two simple
yet effective techniques: adaptive sampling and error-driven
anisotropic parameterization. Adaptive sampling warps the
texture coordinate space according to the Gaussian’s den-
sity distribution, concentrating samples where the rendered
contribution is higher. Error-driven anisotropic parameteri-
zation allocates resolution to each texture based on its esti-
mated rendering error, allowing both resolution and aspect
ratio to adapt to content complexity. We refer to our repre-
1
arXiv:2512.14039v1  [cs.CV]  16 Dec 2025

<!-- page 2 -->
sentation as ASAP-Textured Gaussians, short for Adaptive
Sampling and Anisotropic Parameterization. By integrating
these two techniques, our method preserves the expressive
power of textured Gaussians while significantly improving
parameter efficiency.
ASAP-Textured Gaussians support
flexible quality–efficiency trade-offs, achieving comparable
or even superior rendering quality with far fewer texture pa-
rameters than prior approaches.
Our contributions can be summarized as follows:
• We propose an adaptive sampling strategy that warps tex-
ture coordinates based on the Gaussian density, improv-
ing sampling efficiency for texture utilization.
• We introduce an error-driven anisotropic parameteriza-
tion method that allocates texture resolution and aspect
ratio based on rendering error, offering efficient texture
resource allocation.
• Experiments on multiple datasets demonstrate that
ASAP-Textured Gaussians achieve comparable or even
superior rendering quality with significantly fewer texture
parameters compared to existing methods.
2. Related Work
Neural
Radiance
Fields.
Neural
radiance
fields
(NeRF) [14] have established a differentiable volume
rendering paradigm for photorealistic novel view synthesis,
inspiring various advancements in efficiency [10, 15, 29],
quality [2, 4], and generalizability [7, 30]. Among these, 3D
Gaussian Splatting (3DGS) [10] has achieved high-fidelity
synthesis at interactive rates through an efficient splat-
based rasterization. It models 3D scenes as a collection
of 3D Gaussians and gradually optimizes their geometric
and appearance parameters via differentiable rendering.
Built upon 3DGS, subsequent works further enhance
robustness and scalability through stable training [11], anti-
aliasing [31], and geometry-aware reconstruction [6, 9].
Recent works have also leveraged 3DGS as a versatile
3D backbone and augmented per-Gaussian attributes for
semantics [13, 24, 27], dynamics [22], and physics [18, 23],
extending the vanilla Gaussian set to general-purpose 3D
scene representation.
Textured
Gaussian
Splatting.
To
enrich
the
per-
primitive expressiveness beyond spatially uniform at-
tributes, recent works augment Gaussian primitives with lo-
cal texture parameterizations. Early attempts, such as MLP-
based texture mappings [26], demonstrate the feasibility of
spatially varying appearances but struggle to scale to com-
plex, scene-level data. Subsequent approaches build upon
the 2D Gaussian Splatting (2DGS) framework [9], which
defines a local tangent plane for each Gaussian, allowing
textures to be attached and sampled via UV coordinates
aligned to the Gaussian’s principal axes. Examples include
GSTex [17], which adopts learned UV mappings, and Gaus-
sian Billboards [21], which employ fixed square textures
for simplicity. SuperGaussians [25] further explore alterna-
tive kernel functions for texture representation, improving
fidelity at the cost of higher complexity. TextureSplat [28]
extends textures to model specular effects and employs tex-
ture atlases for acceleration. Recent attempts to generalize
the geometry beyond ellipsoids—such as BBSplat [19] and
Textured Gaussians [5]—achieve strong visual quality but
require substantially larger memory footprints. Although
these methods demonstrate the potential of textured repre-
sentations, how to solve the memory efficiency challenge is
an open-ended problem. Our work addresses this limitation
through adaptive sampling and anisotropic texture param-
eterization, improving texture efficiency without compro-
mising quality. Our design is orthogonal to compression-
based methods such as vector quantization or ZIP-based
packing [19], and can be combined with them for further
memory savings.
3. Preliminaries
2D Gaussian Splatting (2DGS).
2DGS [9] represents
scenes using flattened 2D Gaussians, enabling accurate sur-
face reconstruction and providing a natural foundation for
texture definition. Each 2D Gaussian is parameterized as
{µ, r, s, o, c}, where µ ∈R3 denotes the spatial mean,
r ∈R4 the rotation quaternion, s ∈R2 the scaling factor,
o the opacity, and c the color. The 2D Gaussian is defined
as a local tangent plane in 3D space, spanned by the first
two orthogonal basis vectors {tu, tv} from the correspond-
ing rotation matrix, while the last basis vector represents the
plane’s normal vector n.
To render an image, the 2D Gaussians are first rasterized
onto the image plane and sorted based on their depth values.
Then, for each image coordinate p ∈R2, the corresponding
ray-splat intersection u(p) is computed in the local canon-
ical space of the 2D Gaussian. Finally, the pixel color is
obtained by alpha blending all 2D Gaussians covering the
current pixel, from front to back:
C(p) =
X
i=1
cioiGi(u(p))
i−1
Y
j=1
(1 −ojGj(u(p))) ,
(1)
where i denotes the 2D Gaussian index in the depth-sorted
order, and G the standard 2D Gaussian function.
Mappings in 2DGS.
Considering the 2D Gaussian with
the geometric parameters {µ, tu, tv, s}, where {tu, tv} de-
notes the orthogonal principal axes, and s = (su, sv) the
corresponding scales, any world-space points P on the lo-
cal plane of the 2D Gaussian can be expressed as:
P(u) = µ + sutuu + svtvv,
(2)
2

<!-- page 3 -->
𝒑
Image Space
World Space
𝝁
𝑠𝑢𝒕𝑢
𝑠𝑣𝒕𝒗
Canonical Space
(a) 2D Gaussian Splatting
(b) Adaptively Warped Textures
(c) Anisotropic Texture Growth
Previous: Over/Under Sampling
Ours: Adaptive Sampling
Squared, Fixed Textures →Over Parameterization
Anisotropic, Grown Textures →Error-driven Parameterization
Texture Space
∅
World Space
Initialization
After Optimization
𝑷𝒖
𝒖→𝒖
𝒖→𝚽𝒖
Figure 1. Method Overview. We introduce two key techniques to enhance the texture representation for (a) 2D Gaussian Splatting. (b)
Adaptively Warped Textures: We introduce warping operations to align textures with the 2D Gaussian distribution, enabling more efficient
usage of texture resources. Grid lines here correspond to the texture space coordinates visualized in the canonical space. (c) Anisotropic
Texture Growth: We propose an error-driven texture resolution growth strategy that dynamically adjusts the texture resolution based on the
learning status, allowing details to be captured where necessary while avoiding over-allocation of resources.
where u = (u, v) denotes the local canonical coordinates
on the 2D Gaussian’s tangent plane. Compactly, the map-
ping from the local canonical space to the world space could
be represented as:
P(u) = H(u, v, 1, 1)⊤,
(3)
where H =
sutu
svtv
0
µ
0
0
0
1

∈R4×4
(4)
is the homogeneous transformation matrix from the canon-
ical space to the world space.
The mapping between the camera space and the local
canonical space of the 2D Gaussian is thus:
r = (rxz, ryz, z, z)⊤= W H(u, v, 1, 1)⊤,
(5)
where r denotes a homogeneous ray exmitting from the
pixel (rx, ry) and intersecting with the splat at the depth z,
W ∈R4×4 is the world-to-camera transformation matrix.
Inversely, u = (W H)−1r, where M = (W H)−1 is usu-
ally pre-computed for acceleration and named as the “ray-
transform” matrix. For more details, please refer to [9].
4. Method
Our ASAP-Textured Gaussian builds upon the 2DGS
framework and adopts the widely used texture-map repre-
sentation for its simplicity and flexibility. We improve the
efficiency and scalability of textured 2DGS by addressing
two fundamental limitations: redundant sampling within
each primitive and over-parameterization across the scene.
To this end, we introduce two complementary techniques:
(1) adaptive sampling (Sec. 4.1), which aligns sampling
density with each Gaussian’s mass distribution for intra-
primitive efficiency, and (2) anisotropic texture resolution
growth (Sec. 4.2), which progressively adjusts texture size
and aspect ratio based on gradient statistics for scalability.
An overview of our method is shown in Fig. 1.
4.1. Adaptive Sampling for Textured 2D Gaussians
As discussed in Sec. 3, the 2DGS framework defines the
ray–splat intersections u(p) = (u, v) in the canonical
space of each Gaussian. This formulation naturally pro-
vides a convenient uv-parameterization for associating tex-
tures with individual primitives. Existing textured-Gaussian
methods adopt this parameterization by either attaching
explicit texture maps c(u, v) [5, 19] or learning MLPs
K(u, v) [25, 26] to model spatially-varying appearances.
While straightforward, these approaches implicitly assume
that the canonical space is equivalent to the texture space.
However, this assumption overlooks the differences be-
tween spaces, caused by the non-uniform opacity distribu-
tion of the 2D Gaussian. As illustrated in Fig. 1 (b), directly
using the canonical space as the texture space produces
uniformly distributed texture samples, whereas their con-
tribution to rendering decays exponentially from the center.
Consequently, regions near the Gaussian tails—where opac-
ity and color weights are negligible—still consume a large
fraction of textures, while areas near the center—which
contribute significantly to rendering—desire higher sam-
pling densities to capture fine details. Such an imbalance
results in inefficient usage of textures, motivating the need
for warping functions that better align texture sampling den-
sities with the 2D Gaussian’s support.
Warping to align with the 2D Gaussian distribution.
To address the inefficient usage of textures, we introduce
warping functions that map the canonical coordinates u =
3

<!-- page 4 -->
(u, v) to a mass-aware texture domain ˜u = (˜u, ˜v), such that
the texture sampling density follows the Gaussian’s local
probability density. Formally, we want to define the warp-
ing function, Φ(u) = ˜u, according to the cumulative distri-
bution function (CDF) of the 2D Gaussian.
We consider two practical variants of warping functions:
(1) Axis-wise CDF warping, which separately warps each
axis, based on the 1D Gaussian CDF; and (2) Radial CDF
warping, which warps the radial distance from the Gaus-
sian center, respecting the radially symmetric nature of the
canonical Gaussian.
The axis-wise CDF warping Φaxis is defined as:
Φaxis(a) = 1
2

1 + erf
 a
√
2

,
(6)
where erf(·) is the error function of the 1-D Gaussian dis-
tribution, measuring the cumulative probability mass.
The radial CDF warping Φradial is defined as:
r =
p
u2 + v2,
(7)
˜r = 1 −e−r2
2 ,
(8)
Φradial(a) = ˜r
ra,
(9)
where r specifies the radial distance of u from the center
and ˜r is the warped radial distance using the Rayleigh CDF.
Both warping functions redistribute texel sampling den-
sities according to the Gaussian’s mass distribution (either
marginally or radially), better matching the opacity fall-off.
The final color contribution is thus c(˜u), avoiding wasted
texture capacity in low-contribution regions and enabling
detail representation around the center.
Discussions.
The proposed warping serves as a bridge be-
tween the canonical and texture spaces and is agnostic to
the appearance representation c(˜u), supporting both ex-
plicit texture maps and learned kernels or MLPs. While
the warping is well-defined for RGB textures, it remains
practical for RGBA settings as well. Through interpreting
the Gaussian distribution as a prior on opacity, the warping
functions thus effectively perform importance-based sam-
pling: the texel density is concentrated in regions of higher
expected contribution, guided by the Gaussian mass. Over-
all, the warping function provides a principled mechanism
to allocate texture capacity within each Gaussian, thereby
improving texture utilization regardless of underlying ap-
pearance models.
4.2. Anisotropic Texture Resolution Growth
While warping improves sampling efficiency within each
Gaussian, existing textured-Gaussian methods still suffer
from fixed per-Gaussian texture parameterizations across
the entire scene. As illustrated in Fig. 1 (c), such uniform
parameterizations disregard the spatially varying complex-
ity of scene appearance: regions with smooth color vari-
ations or small geometric support receive the same texel
budget as highly detailed or large surfaces, resulting in re-
dundant memory and unnecessary computation.
To address this limitation, we propose an error-driven
anisotropic texture resolution growth strategy that progres-
sively allocates texture capacity across Gaussians during
training. Inspired by the adaptive density control in [10],
we monitor the magnitude of texture gradients within each
Gaussian as an indicator of representational adequacy, and
adaptively adjust its texture size and aspect ratio to match
local appearance complexity. We follow [5, 10] and use the
photometric loss L between rendered and ground-truth im-
ages as the supervision signal, defined as:
L = (1 −λSSIM)L1 + λSSIMLSSIM.
(10)
Adaptive axis growth.
To enable anisotropic resolution
adaptation, we analyze the gradient statistics of textures
along each axis. Intuitively, large gradients along one axis
indicate insufficient texture resolution in that direction, mo-
tivating additional texels to be allocated. Formally, for each
Gaussian with the texture size (Tu, Tv), we accumulate the
row- and column-wise gradients of texel values (either col-
ors or features) with respect to the rendering loss L:
gu[i] =
Tv−1
X
j=0

∂L
∂f(i, j)
 ,
i = 0, . . . , Tu −1,
(11)
gv[j] =
Tu−1
X
i=0

∂L
∂f(i, j)
 ,
j = 0, . . . , Tv −1,
(12)
where f(i, j) denotes the texel value at coordinate (i, j).
The aggregated vectors gu ∈RTu and gv ∈RTv quantify
the per-axis update pressure, serving as indicators for direc-
tional texture growth in subsequent steps.
To ensure robustness, the per-axis gradients are accumu-
lated across multiple training iterations and views. For each
Gaussian g, we record its visibility count ng, accounting
for observation frequencies. After accumulation, the direc-
tional gradient pressures are averaged over both the visibil-
ity count and the per-axis texel size:
¯gu =
1
ngTv
Tu−1
X
i=0
gu[i],
¯gv =
1
ngTu
Tv−1
X
j=0
gv[j].
(13)
This averaging mitigates bias toward axes that currently
contain more texels or Gaussians that appear in more views.
Implementation details of growth decision.
A Gaussian
expands when the corresponding averaged gradient mag-
nitude ¯gu or ¯gv exceeds a predefined gradient threshold
4

<!-- page 5 -->
τtex. Our growth checks are performed periodically—every
Ntex iterations—to suppress transient gradient fluctuations,
and the maximum number of growth steps is limited by
Nmax to control memory usage and allow stable fine-tuning
thereafter.
Whenever a texture expands, it is resampled
to the new resolution using bilinear interpolation, ensuring
smooth parameter transfer and continuity without introduc-
ing rendering artifacts.
Texture Initialization.
To maximize the benefits of adap-
tive growth, we initialize Gaussians without textures and
accumulate gradients over base appearance attributes to de-
termine whether texture activation is necessary. Once acti-
vated, the texture is initialized with a small anisotropic grid,
(1×2) or (2×1), where the aspect ratio is determined by the
Gaussian’s relative axis scales. Some concurrent work [1]
explores texture resolution adaptation based on geometric
size and lacks a statistical measure of representational ad-
equacy along each axis. In contrast, our gradient-driven
criterion is more precise and appearance-aware: it directly
reflects the rendering error propagated through each texel.
For instance, Gaussians with nearly uniform colors may re-
main low-resolution or untextured under our approach, even
when spatially elongated, while visually complex regions
naturally trigger anisotropic texture expansion.
CUDA-Efficient Anisotropic Texture Pipeline
A key
advantage of 3DGS lies in its high-performance CUDA ras-
terization pipeline. To ensure that our anisotropic textures
remain fully compatible with this parallel rendering frame-
work, we design an implementation that preserves GPU-
friendly data access patterns and avoids divergence across
kernels. Specifically, we determine a global maximum tex-
ture resolution and employ a differentiable bilinear resam-
pling scheme that maps each Gaussian’s stored anisotropic
texture to a uniform texture grid passed to the CUDA ker-
nels. This design maintains efficient parallelism and pre-
vents irregularities caused by texture size variability. Dur-
ing inference, the resampling is performed once and does
not alter the standard textured-Gaussian pipeline.
As a
result, our anisotropic parameterization introduces only a
minimal computational overhead during training, as verified
in our experiments. Moreover, complementary acceleration
techniques such as texture ATLAS representations [16, 28]
can be integrated for additional speedups and compression.
In summary, our ASAP-Textured Gaussians enhance
texture efficiency from both intra- and cross-primitive per-
spectives.
The proposed adaptive warping aligns sam-
pling density with each Gaussian’s mass distribution to
eliminate redundant texel usage, while the error-driven
anisotropic texture growth adaptively allocates texture ca-
pacity across Gaussians according to gradient-derived error
statistics. Together, these components form a unified frame-
work that achieves high-fidelity rendering with substantially
improved texture efficiency and scalability.
5. Experiments
5.1. Experimental Setup
Datasets and Metrics.
Following the common practice
in novel view synthesis [2, 10], we conducted experiments
on three widely-used datasets, including the Mip-NeRF 360
dataset [3] (7 scenes), the Tanks and Temples dataset [12] (2
scenes), and the Deep Blending dataset [8] (2 scenes). The
data preprocessing and train/test splits follow the original
settings in 3DGS [10]. We evaluated the rendering quality
using three standard metrics, including the Peak Signal-to-
Noise Ratio (PSNR), the Structural Similarity Index Mea-
sure (SSIM) [20], and the Learned Perceptual Image Patch
Similarity (LPIPS) [32].
We also report the number of
Gaussians and the model size (in MB) to evaluate the ef-
ficiency of different methods.
Baselines.
We compared our method with four closely re-
lated 3DGS-based neural rendering methods: 1) 2D Gaus-
sian Splatting (2DGS) [9], the backbone of many texture-
based methods; 2) Textured Gaussians (TexGau) [5], the
state-of-the-art approach for textured Gaussian representa-
tion that achieves a balanced trade-off between efficiency
and quality; 3) BBSplat [19] and 4) SuperGaussian [25]
(SuperGau), two recent methods that improve the expres-
siveness of textured Gaussians. All methods were retrained
using their publicly available implementations. For 2DGS,
we retrained the model without geometric regularization for
improved rendering quality and applied the MCMC strat-
egy [11] to control the exact Gaussian count, consistent with
TexGau and BBSplat. We used 2DGS† to denote this mod-
ified version. For TexGau, its released implementation is
built on the 2DGS framework rather than 3DGS as stated in
the paper; we denote this version as TexGau†. To isolate the
effect of textured Gaussians, we retrained BBSplat without
its sky-box strategy from the supplementary material, de-
noted as BBSplat†.
Implementation details.
Following TexGau, we used
identical initialization, training schedule, learning rates, and
optimizers for a fair comparison.
Specifically, both our
ASAP-Textured Gaussians and TexGau shared exactly the
same initialization of 30K-step pre-trained 2DGS and were
fine-tuned for another 30K steps with textures. All other
baselines were also trained for 60K steps to ensure fairness.
All methods used the same order of Spherical Harmonics
for view-dependent color modeling. The gradient thresh-
olds for our texture growth were set to τbase = 4×10−6 and
τtex = 2×10−7. Gradients were accumulated over 100 steps
before evaluating the growth condition, which was applied
5

<!-- page 6 -->
Figure 2. Rendering Quality vs. Model Size. Rendering quality (PSNR, SSIM, LPIPS) plotted against model size (MB) on three datasets.
For clarity, we zoom in to exclude outliers with exceptionally large memory footprints or poor performance. Across all datasets, our ASAP-
Textured Gaussians consistently achieve superior trade-offs between rendering quality and model size compared to prior approaches.
Table 1. Quantitative comparisons. Our method achieves comparable or even better rendering quality with fewer parameters than baseline
textured Gaussians methods. BBsplat uses significantly larger memory and is included here only for reference. The model size here is
measured as the parameter size of the entire model.
# Gaussians
Mem
Mip-NeRF 360
Tanks & Temples
DeepBlending
Method
MB (%)
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
# 100K
2DGS†
22.51 (100%)
26.71
0.775
0.283
22.22
0.779
0.241
28.66
0.882
0.253
BBSplat† (ref.)
391 (+1638%)
28.05
0.834
0.192
23.43
0.838
0.150
28.92
0.895
0.203
SuperGau
33.6 (+49%)
23.78
0.726
0.361
20.51
0.738
0.293
26.67
0.870
0.286
TexGau†
46.9 (+108%)
27.09
0.787
0.267
22.58
0.785
0.233
29.12
0.890
0.235
Ours
34.6 (+54%)
27.11
0.785
0.259
22.64
0.793
0.216
29.12
0.890
0.227
# 200K
2DGS†
45.0 (100%)
27.50
0.804
0.235
22.67
0.800
0.202
29.13
0.891
0.226
BBSplat† (ref.)
781 (+1636%)
28.37
0.844
0.181
23.48
0.841
0.141
29.24
0.897
0.192
SuperGau
67.1 (+49%)
25.65
0.777
0.268
22.22
0.795
0.210
28.82
0.889
0.224
TexGau†
93.6 (+108%)
27.81
0.813
0.223
23.03
0.807
0.196
29.44
0.894
0.216
Ours
60.7 (+35%)
27.79
0.810
0.220
22.97
0.809
0.186
29.40
0.893
0.208
# 500K
2DGS†
113 (100%)
28.18
0.828
0.190
23.09
0.817
0.169
29.22
0.896
0.202
SuperGau
168 (+48%)
27.66
0.814
0.198
23.30
0.825
0.160
29.41
0.897
0.188
TexGau†
235 (+108%)
28.46
0.834
0.182
23.38
0.820
0.164
29.56
0.898
0.193
Ours
128 (+14%)
28.44
0.833
0.182
23.28
0.822
0.158
29.51
0.896
0.192
periodically until 15K steps. Thanks to our efficient texture
parameterization, we could use detailed texture resolutions
up to 8 × 8 per Gaussian without notable memory over-
head. Unless otherwise stated, all experiments shared iden-
tical settings and were conducted on NVIDIA A100 GPUs.
Our code and trained models will be released to support fu-
ture research.
6

<!-- page 7 -->
Figure 3. Qualitative Results. Our method preserves rendering quality with far fewer parameters on a varied number of Gaussians across
three datasets. With a similar model size, our methods achieve better rendering quality than baseline methods.
Table 2. Ablation studies of the warping operations. Experiments conducted with the equal number of Gaussians on the MipNeRF-360
dataset. Each pair of experiments uses the same initialization of 2D Gaussians and is trained for the same number of steps and time. The
only differences within each pair lie in the adopted warping functions. The introduction of warping functions consistently improves the
rendering quality under varied texture resolutions and number of Gaussians.
# 500K
# 200K
# 100K
Method
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
TexGau† (4x4)
28.46
0.834
0.182
27.81
0.813
0.223
27.09
0.787
0.267
+ axis warp
28.56 (+)
0.839 (+)
0.171 (-)
27.91 (+)
0.819 (+)
0.206 (-)
27.21 (+)
0.795 (+)
0.244 (-)
TexGau† (6x6)
28.58
0.838
0.173
27.98
0.819
0.209
27.30
0.795
0.248
+ axis warp
28.65 (+)
0.841 (+)
0.162 (-)
28.07 (+)
0.824 (+)
0.191 (-)
27.45 (+)
0.804 (+)
0.223 (-)
5.2. Overall Results
We compare our ASAP-Textured Gaussians against base-
line methods under the same number of Gaussians, rang-
ing from 100K to 500K. Model size is reported as the total
number of parameters before any quantization or ZIP-based
compression [19]. We use the base 2DGS model size as
the reference and report the relative overhead in percentage.
Unless stated otherwise, axis-wise CDF warping is used for
its simplicity; a detailed comparison with radial CDF warp-
ing is provided in Sec. 5.3.
As shown in Tab. 1, our method consistently achieves
comparable or better rendering quality while requiring
fewer parameters, as reflected in the memory footprint
(MB). BBSplat introduces more than an order of magnitude
additional texture parameters (over 15×), so we report it
only as a reference. SuperGau exhibits a similar model size
to ours but performs significantly worse when the Gaussian
count is small (100K and 200K). The quality–size trade-offs
in Fig. 2 further highlight this trend: points in the upper-
right corner indicate better trade-offs. Our ASAP-Textured
Gaussians achieve a good trade-off across all datasets.
From Tab. 1, we observe that under a small Gaussian
budget (e.g., 100K), the scene is under-represented, lead-
ing to higher modeling error. Consequently, our gradient-
based, error-driven allocation assigns more texture re-
sources, which results in a slightly higher proportion of
added parameters in this low-budget regime. As the num-
ber of Gaussians increases, the overall rendering quality in-
creases, and our adaptive mechanism becomes more selec-
tive, as reflected in the decreasing percentage of added pa-
rameters. Fig. 3 shows qualitative comparisons on represen-
tative scenes from the three datasets. Our method achieves
comparable results when using the same number of Gaus-
sians, but with fewer parameters. When the model size is
similar, our method could synthesize visually appealing re-
sults, indicating our methods maintain a good trade-off be-
tween image quality and memory efficiency. For more re-
sults, please refer to the Supplementary.
We also report the additional texture map training time of
the baseline and our method on the MipNeRF-360 dataset.
As Tab. 5 illustrates, our implementation is GPU-friendly,
introducing small training overhead.
5.3. Ablation studies.
Comparisons between axis-wise and radial CDF warp-
ing functions.
Tab. 3 shows that the approximate axis-
7

<!-- page 8 -->
Table 3. Comparisons between axis-wise and radial CDF warp-
ing functions. Experiments conducted with an equal number of
Gaussians (500K) on Mip-NeRF 360.
PSNR ↑
SSIM ↑
LPIPS ↓
TexGau† (4x4)
28.464
0.8345
0.1818
w/ axis warp
28.558
0.8388
0.1708
w/ radial warp
28.563
0.8390
0.1729
TexGau† (6x6)
28.580
0.838
0.173
w/ axis warp
28.650
0.8414
0.1616
w/ radial warp
28.633
0.8413
0.1641
Table 4. Performance of warping operations on RGB textures.
Experiments on Mip-NeRF 360 with varying Gaussian counts
show that our warping operations consistently improve rendering
quality, regardless of the underlying texture representation.
PSNR ↑
SSIM ↑
LPIPS ↓
TexGau† (# 200K)
27.586
0.8071
0.2258
+ axis warp
27.642 (+)
0.8093 (+)
0.2189 (-)
TexGau† (# 100K)
26.845
0.7799
0.2712
+ axis warp
26.923 (+)
0.7840 (+)
0.2594 (-)
Table 5. Texture training time on MipNeRF-360. Our method
remains compatible with fast GPU training and adds only minor
overhead.
# 100K
# 200K
# 500K
TexGau†
40m
43m
47m
Ours
42m
46m
52m
Table 6. Rendering FPS on MipNeRF-360. Experiments con-
ducted with identical RGB textures to isolate alpha-blending dif-
ferences. Warping introduces negligible cost, preserving real-time
performance.
# 100K
# 200K
# 500K
TexGau†
125.7
105.9
76.8
+ axis warp
120.5
104.4
75.4
+ radial warp
121.7
102.7
74.1
wise CDF warping yields performance improvements com-
parable to the precise radial CDF warping. As reported in
Table 6, both warping functions introduce negligible over-
head at inference time and preserve real-time rendering per-
formance. We adopt axis-wise CDF warping for the remain-
ing experiments, considering its implementation simplicity.
Varied texture resolution and number of Gaussians.
We further evaluate the effectiveness of the proposed warp-
ing functions under different texture resolutions and Gaus-
sian counts on the Mip-NeRF 360 dataset. All settings are
kept identical across experiments except for the warping
function, ensuring a fair comparison. As shown in Tab. 2,
incorporating the axis-wise warping function consistently
improves reconstruction quality across all texture resolu-
tions and Gaussian numbers. A notable observation is that,
with warping, the performance of lower-resolution textures
can match or even surpass that of higher-resolution textures
without warping. This highlights a key limitation of pre-
vious approaches: to achieve comparable fidelity near the
high-contribution center of each Gaussian, they rely on uni-
formly allocating excessively large textures, which results
in redundant sampling and unnecessary memory cost. In
contrast, warping concentrates sampling density where it
matters most, enabling significantly more efficient use of
texture capacity.
RGB vs RGBA textures.
We also evaluate the impact of
our warping operations on RGB textures across different
numbers of Gaussians. As shown in Tab. 4, the warping op-
eration consistently improves the rendering quality, regard-
less of the underlying textures, demonstrating its robustness
and potential generalizability to other parameterizations.
Limitations and Future Work.
While our method
achieves strong performance, several practical aspects
present opportunities for further improvement. First, the
anisotropic texture parameterization introduces a modest
amount of training overhead, though this has no impact on
inference speed. Our implementation also trades a small
amount of runtime memory for GPU compatibility; incor-
porating texture ATLAS representations [16, 28] could offer
additional speedups and memory savings. Second, although
our error-driven texture growth strategy is effective, inte-
grating Monte Carlo–based adaptive sampling [11] presents
a promising direction for further enhancing both efficiency
and reconstruction quality.
6. Conclusion.
In this work, we present ASAP-Textured Gaussians, a novel
approach to enhance texture representation in 2D Gaus-
sian Splatting through adaptive sampling and anisotropic
parameterization.
By introducing warping functions that
align textures with the underlying 2D Gaussian distribu-
tions, we enable efficient usage of texture resources. Ad-
ditionally, our error-driven texture growth strategy dynami-
cally adjusts texture resolutions based on learning status, al-
lowing for detailed capture where necessary while avoiding
over-allocation. Extensive experiments on multiple datasets
demonstrate that our method achieves superior rendering
quality with reduced model size compared to existing ap-
proaches.
8

<!-- page 9 -->
References
[1] Anonymous. Aˆ2TG: Adaptive anisotropic textured gaus-
sians for efficient 3d scene representation. In Submitted to
The Fourteenth International Conference on Learning Rep-
resentations, 2025. under review. 5
[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, pages 5855–5864,
2021. 2, 5
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5470–5479, 2022. 5
[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-
based neural radiance fields. ICCV, 2023. 2
[5] Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao,
Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes
Kopf, Gordon Wetzstein, and Changil Kim. Textured gaus-
sians for enhanced 3d scene appearance modeling. In CVPR,
2025. 1, 2, 3, 4, 5
[6] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for ef-
ficient and high-fidelity surface reconstruction. IEEE Trans-
actions on Visualization and Computer Graphics, 2024. 2
[7] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision. Springer, 2024. 2
[8] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. ACM Transactions
on Graphics (Proc. SIGGRAPH Asia), 37(6):257:1–257:15,
2018. 5
[9] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery, 2024. 1, 2, 3, 5
[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):1–14, 2023. 1, 2, 4, 5
[11] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. Advances in Neural In-
formation Processing Systems, 37:80965–80986, 2024. 2, 5,
8
[12] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics, 36(4), 2017.
5
[13] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin
Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo
Gevers, et al. Scenesplat: Gaussian splatting-based scene
understanding with vision-language pretraining. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision (ICCV), 2025. 1, 2
[14] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[15] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 2
[16] Budirijanto Purnomo, Jonathan D Cohen, and Subodh Ku-
mar. Seamless texture atlases. In Proceedings of the 2004
Eurographics/ACM SIGGRAPH symposium on Geometry
processing, pages 65–74, 2004. 5, 8
[17] Victor Rong, Jingxiang Chen, Sherwin Bahmani, Kiriakos N
Kutulakos, and David B Lindell. Gstex: Per-primitive tex-
turing of 2d gaussian splatting for decoupled appearance and
geometry modeling. In 2025 IEEE/CVF Winter Conference
on Applications of Computer Vision (WACV), pages 3508–
3518. IEEE, 2025. 1, 2
[18] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen
Zhao, Haocheng Feng, Jingtuo Liu, Liangjun Zhang, Jian
Zhang, Bin Zhou, et al.
Gir: 3d gaussian inverse ren-
dering for relightable scene factorization.
arXiv preprint
arXiv:2312.05133, 2023. 2
[19] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio
Del Bue. Billboard splatting (bbsplat): Learnable textured
primitives for novel view synthesis. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 25029–25039, 2025. 1, 2, 3, 5, 7
[20] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 5
[21] Sebastian Weiss and Derek Bradley.
Gaussian billboards:
Expressive 2d gaussian splatting with textures, 2024. 2
[22] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene render-
ing. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 20310–
20320, 2024. 1, 2
[23] Tianyi Xie, Zeshun Zong, Yuxin Qiu, Xuan Li, Yutao Feng,
Yin Yang, and Chenfanfu Jiang.
Physgaussian: Physics-
integrated 3d gaussians for generative dynamics. In CVPR,
2024. 1, 2
[24] Yusen Xie,
Zhenmin Huang,
Jianhao Jiao,
Dimitrios
Kanoulas, and Jun Ma. UniGS: Unified Geometry-Aware
Gaussian Splatting for Multimodal Rendering.
arXiv
preprint arXiv:2510.12174, 2025. 2
[25] Rui Xu, Wenyue Chen, Jiepeng Wang, Yuan Liu, Peng
Wang, Lin Gao, Shiqing Xin, Taku Komura, Xin Li, and
9

<!-- page 10 -->
Wenping Wang. Supergaussians: Enhancing gaussian splat-
ting using primitives with spatially varying colors, 2024. 1,
2, 3, 5
[26] Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, and
Song-Hai Zhang. Texture-gs: Disentangling the geometry
and texture for 3d gaussian splatting editing. In European
Conference on Computer Vision, pages 37–53. Springer,
2024. 1, 2, 3
[27] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In European conference on computer vision, pages 162–179.
Springer, 2024. 2
[28] Mae Younes and Adnane Boukhayma. Texturesplat: Per-
primitive texture mapping for reflective gaussian splatting,
2025. 1, 2, 5, 8
[29] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng,
and Angjoo Kanazawa. Plenoctrees for real-time rendering
of neural radiance fields. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 5752–
5761, 2021. 2
[30] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 4578–4587, 2021. 2
[31] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2024. 1, 2
[32] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 5
10

<!-- page 11 -->
ASAP-Textured Gaussians: Enhancing Textured Gaussians with Adaptive
Sampling and Anisotropic Parameterization
Supplementary Material
The supplementary material is organized as follows:
• Sec. A presents detailed derivations of the two warping
variants.
• Sec. B reports additional experiments under fixed mem-
ory budgets.
• Sec. C includes further qualitative results.
A. Warping function derivations
The goal of warping is to map canonical texture coordinates
u = (u, v) into a mass-aware texture domain ˜u = (˜u, ˜v),
such that a uniform sampling pattern in the warped domain
corresponds to a Gaussian-weighted sampling pattern in the
canonical domain. Formally, we desire the sampling density
in canonical space to satisfy P(u) ∝G(u), where texels
should be denser near regions of higher Gaussian mass and
sparser in low-mass regions.
For clarity, we first consider the 1D case with a scalar
coordinate u and warp ˜u = ϕ(u). The change of variables
theorem gives
P(u) du = ˜P(˜u) d˜u
⇒
P(u) = ˜P(˜u)

d˜u
du
 .
Given the uniform sampling in the warped domain, such
that ˜P(˜u) = const, and our goal P(u) ∝G(u), we obtain
d˜u
du ∝G(u).
(A.1)
Integrating this differential relationship produces the cu-
mulative distribution function (CDF) based warping func-
tion:
˜u = ϕ(u) =
Z u
−∞
G(t)dt.
(A.2)
Formally, this yields the closed form:
˜u = ϕ(u) = 1
2

1 + erf
 u
√
2

,
(A.3)
which corresponds to our axis-wise warping in Eq. (6).
Radial CDF Warping.
One interesting property of the
(normalized) canonical space of the 2D Gaussian is its ra-
dially symmetric structure. Formally, the local density de-
pends only on the radius r = ∥u∥from the center:
∀∥u1∥= r = ∥u2∥,
G(u1) = G(u2) = G′(r), (A.4)
where G′(r) ∝exp(−r2/2) denotes the radial profile of
the isotropic standard Gaussian.
Similar to the 1D axis-wise case, we now work with the
radial marginal density of r = ∥u∥. In polar coordinates,
the area element is dx dy = r dr dθ, so the induced radial
density is
P(r) ∝r G′(r).
(A.5)
With the constant density in the warped radial coordinate,
˜P(˜r) = const, the 1D change-of-variables relation
P(r) dr ∝˜P(˜r) d˜r
(A.6)
gives
r G′(r) dr ∝d˜r
⇒
d˜r
dr ∝r G′(r),
(A.7)
whose integral yields the Rayleigh CDF
˜r = ϕ(r) =
Z r
0
τ G′(τ) dτ = 1 −exp

−r2
2

.
(A.8)
This constitutes our radial CDF warping in Eq. (9).
B. Additional experiments under fixed mem-
ory budgets.
In this experiment, we control the number of Gaussians
to ensure comparable overall memory consumption across
methods. Although the Gaussian count affects both geom-
etry fidelity and texture memory—and thus does not isolate
texture efficiency—our method still achieves strong ren-
dering quality under this controlled setting, as illustrated
in Tab. B.1.
C. Qualitative results.
We include more qualitative results in Fig. C.1.
1

<!-- page 12 -->
Table B.1. Quantitative comparisons under fixed memory budgets. We control the number of Gaussians to match the overall memory
usage across methods. Under this fixed memory budget, our approach in general delivers higher rendering quality than texture-based
Gaussian baselines, highlighting the effectiveness of our design.
Mem
Mip-NeRF 360
Tanks & Temples
DeepBlending
Method
# Gaussians
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
≈34.6MB
2DGS†
154K
27.05
0.791
0.257
22.48
0.793
0.217
28.95
0.889
0.230
SuperGau
103K
24.04
0.731
0.351
20.59
0.740
0.291
26.74
0.873
0.282
TexGau†
74K
26.64
0.772
0.292
22.34
0.776
0.251
28.93
0.886
0.249
Ours
100K
27.11
0.785
0.259
22.64
0.793
0.216
29.12
0.890
0.227
≈60.7MB
2DGS†
270K
27.75
0.811
0.223
22.85
0.806
0.191
29.25
0.892
0.214
SuperGau
181K
25.44
0.772
0.283
21.87
0.788
0.221
28.65
0.888
0.232
TexGau†
130K
27.31
0.797
0.250
22.76
0.796
0.218
29.29
0.892
0.229
Ours
200K
27.79
0.810
0.220
22.97
0.809
0.186
29.40
0.893
0.208
≈128MB
2DGS†
570K
28.26
0.832
0.185
23.15
0.819
0.164
29.21
0.894
0.198
SuperGau
384K
27.17
0.806
0.212
23.21
0.824
0.166
29.41
0.896
0.195
TexGau†
273K
28.09
0.822
0.207
23.17
0.814
0.183
29.47
0.897
0.204
Ours
500K
28.44
0.833
0.182
23.28
0.822
0.158
29.51
0.896
0.192
Figure C.1. More Qualitative Results. Our method preserves rendering quality with far fewer parameters on a varied number of Gaussians
across three datasets. With a similar model size, our methods achieve better rendering quality than baseline methods.
2
