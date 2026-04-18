<!-- page 1 -->
1
Unifying Color and Lightness Correction with
View-Adaptive Curve Adjustment for Robust 3D
Novel View Synthesis
Ziteng Cui, Shuhong Liu, Xiaoyu Dong, Xuangeng Chu, Lin Gu , Ming-Hsuan Yang, Tatsuya Harada
Abstract—High-quality image acquisition in real-world envi-
ronments remains challenging due to complex illumination varia-
tions and inherent limitations of camera imaging pipelines. These
issues are exacerbated in multi-view capture, where differences in
lighting, sensor responses, and image signal processor (ISP) con-
figurations introduce photometric and chromatic inconsistencies
that violate the assumptions of photometric consistency under-
lying modern 3D novel view synthesis (NVS) methods, including
Neural Radiance Fields (NeRF) and 3D Gaussian Splatting
(3DGS), leading to degraded reconstruction and rendering qual-
ity. We propose Luminance-GS++, a 3DGS-based framework for
robust NVS under diverse illumination conditions. Our method
combines a globally view-adaptive lightness adjustment with a
local pixel-wise residual refinement for precise color correction.
We further design unsupervised objectives that jointly enforce
lightness correction and multi-view geometric and photometric
consistency. Extensive experiments demonstrate state-of-the-art
performance across challenging scenarios, including low-light,
overexposure, and complex luminance and chromatic variations.
Unlike prior approaches that modify the underlying represen-
tation, our method preserves the explicit 3DGS formulation,
improving reconstruction fidelity while maintaining real-time
rendering efficiency.
Index Terms—Novel View Synthesis, Computational Photogra-
phy, Color Correction, Image Enhancement.
I. INTRODUCTION
I
N recent years, Neural Radiance Fields (NeRF) and 3D
Gaussian Splatting (3DGS)-based methods [1]–[9] have
substantially advanced realistic 3D scene reconstruction and
photorealistic novel view synthesis. These approaches learn
continuous scene representations from multi-view observa-
tions, enabling accurate geometry recovery and modeling of
view-dependent appearance, and have become foundational
techniques across applications such as AR/VR, computational
photography, medical imaging, and autonomous driving. How-
ever, existing methods rely heavily on implicit photometric
consistency assumptions that rarely hold in real-world capture.
Variations in illumination intensity (e.g., low-light and expo-
sure differences), illumination color (e.g., color temperature
shifts), and camera-dependent processing (e.g., white bal-
ance and ISP pipelines) introduce systematic photometric and
Ziteng Cui, Shuhong Liu, Xiaoyu Dong and Xuangeng Chu is with the
University of Tokyo, Japan.
Lin Gu is with the Tohoku University, Japan.
Ming-Hsuan Yang is with the University of California at Merced and
Google DeepMind, US.
Tatsuya Harada is with the University of Tokyo and RIKEN AIP, Japan.
 Corresponding Author: Lin Gu (lingu.edu@gmail.com)
chromatic inconsistencies, leading to degraded reconstruction
fidelity and unstable rendering. These failures primarily arise
from lightness distortions induced by environmental conditions
and sensor limitations [10], [11], along with the limited
capability of current NeRF- and 3DGS-based frameworks to
model illumination variations explicitly [12], [13].
Numerous approaches have been proposed for lightness
correction and perceptual image enhancement. Image enhance-
ment methods [14]–[18] primarily address low-light conditions
by increasing brightness while suppressing sensor noise. Ex-
posure correction methods [19]–[21] aim to recover balanced
luminance from underexposed or overexposed inputs, while
white balance (WB) correction techniques compensate for
illumination color shifts in either RAW [22], [23] or sRGB
domains [24]–[26]. Despite their success in single-image
restoration, these methods are not well suited for multi-view
3D reconstruction. Prior work [27]–[29] shows that indepen-
dently preprocessing views disrupts radiometric consistency,
violating key assumptions of NVS optimization. The resulting
inter-view photometric discrepancies often manifest as floaters,
ghosting artifacts, and geometric instability [30].
To improve NVS robustness under challenging lighting
conditions, NeRF-W [31] and subsequent works [32]–[35] in-
troduce view-dependent appearance embeddings to disentangle
illumination effects from scene geometry. While effective at
mitigating view inconsistencies and reducing floaters, these
approaches are inherently constrained by the lighting con-
ditions observed in the training views, limiting their ability
to generalize to novel illumination scenarios during rendering
(e.g., low-light enhancement or color correction).
To achieve robust lightness correction in multi-view 3D
scenes, several NeRF-based approaches [11], [27]–[29] in-
corporate illumination-specific modeling within the volume
rendering pipeline, enabling unsupervised lightness adjust-
ment while preserving multi-view consistency. LL-NeRF [27]
decomposes the NeRF color MLP into view-dependent and
view-independent components for targeted processing. Aleth-
NeRF [28] introduces a concealing-field branch that models
low-light effects as occlusions, allowing adaptive enhancement
or suppression during inference. AME-NeRF [29] formulates
a bilevel optimization framework to normalize brightness
while maintaining structural consistency. For multi-view color
correction, Zhang et al. [38] emulate digital camera processing
by integrating ISP-inspired kernel mappings and color matrices
into NeRF’s implicit representation.
However, these solutions do not readily extend to ex-
arXiv:2602.18322v1  [cs.CV]  20 Feb 2026

<!-- page 2 -->
2
Fig. 1. (a) An overview of Luminance-GS++, designed to address multi-view images affected by real-world illumination and color degradations via joint
global and local adjustments, enabling robust 3DGS reconstruction. (b) Performance comparison between Luminance-GS++ and state-of-the-art methods [28],
[34], [36], [37] under diverse illumination and color conditions.
plicit representations such as 3DGS [6], which relies on
rasterization-based rendering and explicitly stores appearance
within a 3D Gaussian point cloud, eliminating the need for
implicit neural fields and MLP-based architectures. This ex-
plicit formulation provides substantial advantages over NeRF-
based methods in both training and rendering efficiency [6],
[7], [39], [40]. For example, compared to state-of-the-art NeRF
approaches such as [5], 3DGS achieves over 1000× faster
rendering and nearly 480× faster training while maintaining
comparable visual quality [6]. These advances raise a funda-
mental question: how can we leverage the efficiency of explicit
3DGS representations while ensuring robust reconstruction
under diverse and challenging illumination conditions?
Our preliminary work, Luminance-GS [36], introduces a
pseudo-label–based image enhancement strategy that jointly
optimizes tonal curves with 3D Gaussian attributes. Specifi-
cally, the method applies per-view color matrix mapping and
view-adaptive curve adjustments via differentiable tone curves
to transform inputs captured under diverse lighting conditions
(e.g., low-light, overexposure, or exposure variations) into
consistently well-lit pseudo-enhanced images. These color
transformations are jointly optimized with the Gaussian rep-
resentation during training, while unsupervised objectives en-
force multi-view alignment and radiometric consistency across
viewpoints.
However, Luminance-GS assumes white ambient illumina-
tion and does not explicitly evaluate reconstruction robust-
ness under varying color temperatures or mixed degradations.
Moreover, it relies on a global curve-adjustment strategy for
pseudo-label generation, applying view-level tone operations
that lack local pixel-wise refinement capabilities (e.g., pixel-
level white balance correction or handling specular residuals).
To address these limitations, we propose the upgraded
Luminance-GS++, which introduces an additional residual
branch for pixel-level local adjustments, improving decom-
position fidelity (see Fig. 1(a)). We further redesign the loss
functions to strengthen unsupervised multi-view color correc-
tion and conduct extensive new experiments to validate the
effectiveness of the proposed approach. The main contribu-
tions of our work are summarized as follows:
• We propose Luminance-GS++, a framework that im-
proves the robustness of novel view synthesis (NVS) un-
der lightness and color degradations. Without modifying
the explicit 3DGS representation, our method introduces
a pseudo-label generation strategy that integrates global
adjustment and local refinement, jointly optimized with
the 3DGS model.
• We design unsupervised objectives for multi-view light-
ness and color correction that preserve radiometric con-
sistency, enabling Luminance-GS++ to generalize across
diverse degradation scenarios while maintaining stable
3D reconstruction.
• We perform extensive evaluations on benchmark datasets
under challenging real-world illumination and synthetic
lighting variations. Extensive eperimental results demon-
strate consistent improvements over state-of-the-art 3D
NVS methods.
II. RELATED WORK
A. 3D Gaussian Splatting and Its Applications
In recent years, Neural Radiance Fields (NeRF) [1] have
significantly advanced novel view synthesis (NVS). However,

<!-- page 3 -->
3
NeRF-based methods [1], [3]–[5], [41], [42] are computa-
tionally expensive, as the rendering requires dense sampling
and multiple network evaluations per pixel, limiting real-time
applicability. Most recently, 3D Gaussian Splatting (3DGS) [6]
has emerged as a highly efficient alternative, enabling real-
time NVS while maintaining rendering quality comparable
to state-of-the-art NeRF-based approaches [5]. Its efficiency
arises from an explicit scene representation based on learnable
anisotropic 3D Gaussians and differentiable splatting via tile-
based rasterization.
Building on this foundation, numerous works have extended
3DGS to improve rendering quality [37], [43]–[45], train-
ing efficiency [40], [46], [47], generalization capability [7],
[9], [48], [49], and real-world applicability [13], [50]–[55].
Representative examples include 2DGS [44], which replaces
volumetric Gaussians with oriented 2D disks to better cap-
ture surface structure; LightGaussian [40], which introduces
a pruning–recovery pipeline for efficient compression; GS-
MCMC [37], which models Gaussians as samples from a
scene distribution optimized via stochastic gradient Langevin
dynamics; and MVSplat [49], which leverages Transformer-
based multi-view features with cost-volume construction for
feed-forward reconstruction. Application-driven extensions
further demonstrate the versatility of 3DGS, including X-
Gaussian [50] for sparse-view CT reconstruction, Driving-
Gaussian [52] for dynamic driving scenes, and Gaussian Wave
Splatting [55] for occlusion-aware holographic rendering.
B. 3D Reconstruction under Degradation
1) Robust 3D Reconstruction: Despite significant progress
in NeRF- and 3DGS-based reconstruction, most existing meth-
ods assume ideal imaging conditions. In real-world scenarios
involving degradations such as low resolution [56], [57],
underwater environments [58], [59], fog [60], [61], rain [62],
noise [63], and motion blur [64], [65], reconstruction perfor-
mance often degrades significantly [66], [67]. In this work,
we focus on degradations induced by illumination and color
variations (e.g., environmental lighting changes, ISO settings,
exposure variations, color temperature shifts, and white bal-
ance errors), which are pervasive yet relatively underexplored
challenges in real-world multi-view capture.
2) Lightness Degradations: For real-world lightness degra-
dations, methods such as NeRF-W [31] and its follow-up
works [32]–[35], [68] address reconstruction under inconsis-
tent illumination and occlusions by disentangling appearance
variations from scene geometry. This line of research primarily
aims to recover uniform lighting from multi-view inputs ex-
hibiting non-uniform illumination. For example, NeRF-W [31]
introduces learnable appearance and transient embeddings
per view, while GS-W [34] models intrinsic and dynamic
appearance components for individual 3D Gaussian primitives.
However, these approaches are largely limited to reconstruct-
ing lighting conditions observed in the training data, restricting
adaptation to novel illumination scenarios (e.g., low-light
enhancement).
For low-light environments, RAW-NeRF [10] leverages
HDR RAW inputs combined with an ISP model for im-
proved reconstruction. Subsequent works [69]–[72] extend
this paradigm to 3DGS for improved efficiency. Nevertheless,
reliance on RAW data introduces practical limitations, includ-
ing acquisition constraints, storage overhead, and prolonged
training times.
Several studies have instead explored NVS under challeng-
ing lighting using sRGB inputs [11], [27]–[29], [61], [73]–
[76]. For instance, Aleth-NeRF [28] introduces a concealing
field to regulate image lightness, enabling both low-light
enhancement and overexposure suppression. BilaRF [11] em-
ploys 3D bilateral grids to model view-specific camera effects
for refined tonal adjustment. Thermal-NeRF [73] fuses thermal
and sRGB data for improved reconstruction under low-light
conditions. In contrast, our Luminance-GS++ adopts a curve-
based pseudo-enhancement framework that generalizes across
diverse lighting conditions without requiring modifications to
training strategies or hyperparameters.
3) Color Distortions: Real-world color distortions caused
by varying color temperatures and camera white balance errors
remain relatively underexplored. Early works [77]–[81] fo-
cused primarily on multi-camera color consistency rather than
addressing challenges in 3D NVS. URF [82] applies a per-
image latent-decoded 3×3 matrix for white balance correction,
but its global formulation limits flexibility under complex color
variations. Zhang et al. [38] propose a physically grounded
NeRF-based color correction framework incorporating nonlin-
ear modules and histogram-based consistency losses; however,
the histogram loss does not explicitly enforce 3D consistency,
and reliance on implicit NeRF representations results in slower
training. Our approach instead leverages the efficiency of
explicit 3DGS [6] to achieve robust and efficient reconstruction
under color distortions.
C. Curve Adjustment in Image Processing
Curve-based adjustment modifies tonal ranges by map-
ping pixel intensities through predefined or learned tone
curves, commonly used in commercial tools such as Adobe
Lightroom®. Early image enhancement methods relied on
handcrafted tone-mapping functions (e.g., power-law curves
and S-curves) that required manual tuning to achieve desirable
perceptual results [83]–[86].
More recently, data-driven approaches learn tone curves
from large-scale datasets for global or local image en-
hancement [14], [87]–[95]. For example, CURL [89] learns
tone adjustments across multiple color spaces, while Zero-
DCE [14] predicts pixel-wise high-order curves using unsu-
pervised objectives. NamedCurves [93] further decomposes
images via color naming systems and applies B´ezier curves
per category. Unlike these 2D methods, our approach performs
view-dependent curve optimization across multi-view inputs,
enabling consistent brightness and color correction while pre-
serving 3D consistency for stable 3DGS training.
III. PROPOSED METHOD
We begin with a brief overview of the vanilla 3D Gaussian
Splatting (3DGS) framework [6] (Sec. III-A). We then describe
the key training enhancements that led to Luminance-GS++

<!-- page 4 -->
4
Fig. 2. Progression from 3DGS [6] to Luminance-GS [36] and Luminance-
GS++, highlighting joint optimization of input images and 3DGS attributes
through global adjustment and local refinement. Multi-view rendering exam-
ples illustrate the resulting improvements.
(Sec. III-B), followed by a detailed presentation of the pro-
posed architecture and pipeline, which integrate global curve
adjustment with local pixel-wise refinement (Sec. III-C).
A. 3D Gaussian Splatting Revisited
3DGS [6] is an explicit primitive-based representation that
models a scene as a collection of anisotropic 3D Gaussians
{G1, ..., GM}. Each Gaussian Gi is parameterized by a center
position µi ∈R3, a covariance matrix Σi ∈R3×3, view-
dependent color ci, and opacity oi ∈[0, 1]. The covariance
matrix encodes spatial extent and orientation and is decom-
posed into a diagonal scaling matrix Si = diag(σi1, σi2, σi3)
and a rotation matrix Ri ∈SO(3):
Σi = RiS2
i R⊤
i .
(1)
This
formulation
represents
each
Gaussian
as
a
fully
anisotropic ellipsoid in 3D space. View-dependent appearance
is modeled using low-order spherical harmonics, enabling
efficient approximation of angular radiance variation and sup-
porting real-time rendering.
During rendering, Gaussians are projected onto the image
plane using camera parameters and accumulated via differ-
entiable elliptical surface splatting [96]. Projection produces
Fig. 3.
A low-light “chair” scene from the LOM dataset [28]. Our global
adjustment enhances overall illumination, while our local refinement improves
fine details and the quality of pseudo label Cout.
a 2D elliptical footprint determined by the transformed co-
variance Σscreen
i
= JiΣiJ⊤
i , where Ji denotes the projection
Jacobian evaluated at µi. Each pixel x receives contributions
from overlapping Gaussians, sorted by increasing depth. The
rendered color is computed using front-to-back compositing:
ˆC(x) =
N
X
i=1
cio′
i
i−1
Y
j=1
(1 −o′
j),
(2)
where o′
i denotes the visibility-weighted opacity modulated by
the projected Gaussian footprint and pixel distance. This for-
mulation resembles volumetric alpha compositing and enables
smooth blending and partial transparency [1].
All Gaussian parameters are optimized using a mixed re-
construction objective L3DGS that measures the discrepancy
between the rendered image ˆC and ground truth image C:
L3DGS( ˆC, C) = λLDSSIM( ˆC, C) + (1 −λ)L1( ˆC, C),
(3)
where LDSSIM denotes DSSIM loss, L1 is pixel-wise L1 loss,
and λ ∈[0, 1] balances the two terms. The differentiable
rendering pipeline enables joint optimization of all Gaussian
parameters for high-fidelity multi-view reconstruction.
Despite its efficiency, 3DGS performs direct color fit-
ting, where spherical harmonics provide only an approxi-
mate shading representation without explicit decomposition of
reflectance and illumination [13], [70]. Consequently, under
illumination degradations, the model entangles material and
lighting effects, producing floaters and color shifts in rendered
outputs (see Fig. 2(a)).
B. From 3DGS to Luminance-GS++
Figure 4 provides an overview of the proposed Luminance-
GS++ framework. Starting from SfM points [97] (Fig. 4 Up),
we initialize a set of anisotropic 3D Gaussians {G1, . . . , GM}
with color attributes ci to reconstruct multi-view images
{Cin
1 , . . . , Cin
N } captured under challenging illumination con-
ditions. In addition to the standard 3DGS representation, we
introduce learnable color adjustment parameters ai and bi,
which transform the original color attributes into adjusted
colors using a linear mapping [35]:
cout
i
= ai · ci + bi.
(4)

<!-- page 5 -->
5
The transformed colors cout
i
are used to render pseudo-
enhanced images Cout alongside the original colors ci. During
training, Luminance-GS++ jointly predicts both the original
and pseudo-enhanced renderings, extending Eq. 2 as:
ˆCin(x) =
N
X
i=1
cio′
i
i−1
Y
j=1
(1 −o′
j),
ˆCout(x) =
N
X
i=1
cout
i
o′
i
i−1
Y
j=1
(1 −o′
j).
(5)
During inference, the rendering relies solely on cout
i
, preserv-
ing the real-time performance of the original 3DGS pipeline.
The pseudo-enhanced image generation process (Fig. 4
Down) aims to address real-world degradations including
illumination variations (e.g., low-light, overexposure) and
chromatic distortions (e.g., varying color temperatures). Since
3DGS training implicitly assumes consistent lighting, our
objectives are twofold: (1) normalize illumination toward well-
exposed conditions, and (2) enforce cross-view consistency to
stabilize multi-view optimization.
Figure 2(a) illustrates direct 3DGS reconstruction on a
scene [5], where illumination and color degradations intro-
duce view-inconsistent appearances, preventing reliable sur-
face alignment [30], [35]. As a result, Gaussians may stretch
or drift in space, producing elongated or floating artifacts.
Our preliminary work, Luminance-GS [36], applies global
tonal curve adjustment by mapping the color matrix per-view
and view-adaptive curves (Fig. 2(b)). While effective for nor-
malizing exposure and illumination, global curve operations
alone cannot fully resolve multi-view color inconsistencies,
leading to residual artifacts. In addition, curve-based map-
pings apply identical transformations to pixels with identical
values [90], limiting spatial adaptivity.
To overcome these limitations, Luminance-GS++ introduces
local pixel-wise refinement following global adjustment. A
learned residual map enables fine-grained correction of spa-
tially varying color and illumination discrepancies, improving
pseudo-enhanced image quality and reconstruction stability
(Fig. 2(c)). Even under relatively uniform lighting conditions
(e.g., low-light scenes), local refinement further enhances
detail fidelity, as demonstrated in Fig. 3.
C. Module Details and Pipeline
This section describes the generation of pseudo-enhanced
images Cout from multi-view inputs Cin and details the
proposed global adjustment and local refinement modules.
1) Global Adjustment: The global adjustment module fol-
lows the design in Luminance-GS [36], consisting of per-view
color matrix mapping and view-adaptive curve adjustments.
Specifically, to compensate for multi-view lightness varia-
tions caused by illumination differences and camera exposure
settings [34], [35], we introduce a learnable per-view color
transformation matrix Mk. Before curve adjustment, each
input image Cin
k∈(1,N) is projected into a view-dependent color
space via Mk. Figure 6(a) shows an illustration of this color
matrix mapping.
As indicated by the blue arrow in Fig. 4, given an input im-
age Cin
k (r, g, b) in RGB color space I(R, G, B), we optimize a
per-view invertible 3×3 matrix Mk, initialized as the identity
matrix and jointly optimized with the 3DGS parameters. The
mapping is defined as:
Cin
k · Mk =

Cin
k (r), Cin
k (g), Cin
k (b)



a11
a12
a13
a21
a22
a23
a31
a32
a33

,
=

Cin
k (r′), Cin
k (g′), Cin
k (b′)

,
(6)
where each element Mk(aij) is learnable. The transformed
image Cin
k (r′, g′, b′) represents the projection into a view-
dependent coordinate system for subsequent tone adjustment.
For tone mapping, we learn a shared global curve Lg and
a per-view curve bias Lb
k, yielding the final tone curve Lk =
Lg + Lb
k. The global curve enforces consistent brightness and
color tone across views, while the view-specific bias provides
flexibility under non-uniform illumination.
In practice, Lg is implemented as a 1D lookup table of
length 256. To prevent overfitting, Lb
k is generated via a view-
adaptive attention module inspired by [20], [98]. As illustrated
in Fig. 6(b), the image Cin
k is encoded using convolutional and
linear layers to produce key and value representations, while
the camera matrix serves as the query. Cross-attention [99]
followed by a feed-forward network outputs a 1 × 256 curve
bias.
The final tone curve Lk is applied to the mapped image
Cin
k · Mk, and the result is transformed back to RGB space
using M−1
k , producing the pseudo-enhanced image:
Cout
k
= Lk(Cin
k · Mk) · M−1
k .
Despite its effectiveness, global curve adjustment alone cannot
fully resolve viewpoint-dependent color variations (Fig. 2(b)),
and lacks pixel-level adaptability for fine detail correction
(Fig. 3). To address these limitations, we introduce a local
refinement branch described next.
2) Local Refinement: The bottom part of Fig. 4 illustrates
the structure of the local refinement residual branch R. By
adding an additional pixel-wise residual map Cres
k
to the
enhanced pseudo-labeled images, we aim to improve the
model’s robustness to color variations while further refining
fine-grained details. For each input view Cin
k , the image is
passed through the residual branch to generate its correspond-
ing residual map Cres
k
.
The
residual
branch
R
consists
of
three
ConvNeXt
blocks [100] connected in series, followed by an activation
layer. Each ConvNeXt block consists of a 7×7 convolution
followed by layer normalization (LN), a 1×1 convolution, a
GeLU activation, and another 1×1 convolution. The output is
subsequently fused with the block input through a residual
connection. When each view input image Cin
k is processed by
the three ConvNeXt blocks, it is subsequently passed through
an activation layer to obtain the final residual map Cres
k
.
The purpose of the final activation layer is to constrain the
output values of the residual map. Specifically, we employ
a clipping operation, where the range is set to [−0.1, 0.1] for
lightness variation scenarios and [−0.5, 0.5] for color variation

<!-- page 6 -->
6
Fig. 4. An overview of the Luminance-GS++ pipeline. Up: Joint optimization of 3D Gaussians with dual color attributes ci and cout
i
to render input images
Cin and pseudo-enhanced outputs Cout. Down: Pseudo-enhancement is achieved through global adjustment (view-adaptive color matrix mapping and curve
adjustment; stages (I)–(III)) followed by local refinement using a pixel-wise residual map.
scenarios. The final pseudo-enhanced image Cout
k
is processed
through global adjustment and local refinement, as expressed
below:
Cout
k
= Lk(Cin
k · Mk) · M−1
k
+ R(Cin
k ).
(7)
In the next section, we introduce the useful loss constraints
that guide the pseudo-enhanced labels Cout
k
and the overall
Luminance-GS++ training.
IV. LOSS FUNCTIONS
The overall objective consists of three components: image
intensity losses, image color loss, and curve regularization
terms. The image-level losses supervise pseudo-label gener-
ation and the optimization of 3DGS-related attributes, while
the curve-level constraints preserve the structural properties of
the tone curve L.
A. Image Intensity Losses
We introduce the image-level objectives based on the dual-
rendering formulation in Eq. 5. Luminance-GS++ employs two
sets of color attributes, ci and cout
i
, to jointly reconstruct the
input multi-view images Cin
k∈(1,N) and the pseudo-enhanced
images Cout
k∈(1,N). Accordingly, the overall regression loss
extends Eq. 3 as:
Lreg = L3DGS( ˆCin, Cin) + L3DGS( ˆCout, Cout).
(8)
To preserve structural consistency during enhancement, we
further introduce a spatial consistency loss Lspa inspired by
Zero-DCE [14] and Aleth-NeRF [28]. This term enforces con-
sistency of local intensity relationships between the pseudo-
enhanced image ˆCout and the input image Cin by matching
relative differences between neighboring pixels:
Lspa = 1
K
K
X
x=1
X
y∈Ω(x)

| ˆCout(x) −ˆCout(y)|
−
0.5
Cin(x)
|Cin(x) −Cin(y)|
2
,
(9)
where x denotes a pixel location and y ∈Ω(x) represents
its neighboring pixels. Unlike the fixed weighting in Aleth-
NeRF [28], which can result in degraded novel view rendering
quality (Fig. 2 (bottom)), we introduce an adaptive factor
0.5
Cin(x), where Cin(x) denotes the mean intensity of the current
input view. This adaptive weighting improves robustness under
varying illumination conditions.

<!-- page 7 -->
7
Fig. 5.
An illustration of our spatial consistency loss function Lspa, and
comparisons with the spatial consistency loss in Aleth-NeRF [28].
B. Image Color Loss
Multi-view color variation introduces additional challenges,
as color temperature and camera white balance (WB) settings
may either remain consistent across views or vary significantly
(see Fig. 7, Up). Traditional single-view WB approaches rely
on heuristic priors (e.g., gray-world [101], white-patch [102])
or end-to-end learning strategies [24], [25], but typically
operate on individual images without enforcing cross-view
consistency.
Inspired by the Shade-of-Gray (SoG) algorithm, which gen-
eralizes classical WB methods through a tunable Minkowski
parameter S, we introduce a color correction loss Lcc:
Lcc = 1
N
X
k∈(1,N)
 ˆCoutSk
k
(p)−ˆCoutSk
k
(q)

1
Sk −0.1·δk, (10)
where (p, q) ∈{(R, G), (G, B), (R, B)}, and
δk = mean
 
1 −min( ˆCout
k
)
mean( ˆCout
k
)
!
.
(11)
For each view k, we learn an adaptive Minkowski parameter
Sk to enable view-specific color correction. The parameter
Sk is predicted using a view-adaptive parameter generator
(Fig. 6(c)), which shares the architecture of the view-adaptive
curve generator. Additionally, a color saturation regularization
term δk is introduced to prevent over-desaturation during
optimization. Fig. 7 illustrates the effectiveness of Lcc through
visual comparisons and color distribution analysis.
C. Curve Losses
Beyond image-level objectives, we introduce curve-level
constraints on L to regulate both its values and structural
properties. These constraints stabilize pseudo-enhanced image
generation and improve overall training robustness.
To avoid excessive deviation of the learned curve, we
impose shape regularization using two canonical tone priors:
a power curve Lpo and an S-curve Ls. These priors guide
the learned curve toward plausible tonal transformations while
reducing overfitting:
Lpo : y = (x + ϵ)Gk,
ϵ = 10−4,
0 ≤x ≤1,
Ls : y =





Ak −Ak

1 −
x
Ak
Bk ,
0 ≤x ≤Ak,
Ak + (1 −Ak)

x−Ak
1−Ak
Bk ,
Ak < x ≤1,
(12)
where {Gk, Ak, Bk} are view-adaptive learnable parameters
predicted by the view-adaptive parameter generator (Fig. 6(c)).
To further constrain curve values, we use the cumulative
distribution function (CDF) Lcdf derived from histogram
equalization of the input image Cin(x) as an initialization
target. The curve loss is defined as:
Lcurve = ω∥L −Lcdf∥2 + 0.5∥L −(Lpo · Ls)∥2,
(13)
where ω is set to 1.0 during the first 3,000 iterations and
reduced to 0.1 thereafter to gradually relax the initialization
prior.
Additionally, we impose a total variation (TV) regularization
term to encourage smoothness:
Ltv =
1
255
254
X
i=0
|L(i + 1) −L(i)|2.
(14)
D. Overall Objective
The overall training objective of Luminance-GS++ is:
Ltotal = Lreg + Lspa + Ltv + 10 Lcurve + η Lcc,
(15)
where η is a weighting factor set to 0.1 when color degradation
is present and 0.005 otherwise.
V. EXPERIMENTS
This section presents the experiments validating the effec-
tiveness of Luminance-GS++. We first introduce the datasets
and the baseline settings, followed by the experimental results
under lightness-degradation, color-degradation, and mixed-
degradation scenarios. Finally, we provide comprehensive ab-
lation studies to examine the contribution of each proposed
component.

<!-- page 8 -->
8
Fig. 6. (a) Per-view color matrix mapping with learnable transformation Mk. (b) Architecture of the view-adaptive curve generator, which predicts curve
bias Lb
k from input image Cin
k
and camera pose. (c) Structure of the view-adaptive parameter generator.
TABLE I
EXPERIMENTAL RESULTS (PSNR ↑, SSIM ↑, LPIPS ↓) ON THE LOW-LIGHT SUBSET OF LOM DATASET [28]. WE COMPARE WITH VARIOUS
ENHANCEMENT METHODS [14]–[16], [103], [104] AND NERF- AND 3DGS-BASED METHODS [28], [29], [74]. RED
DENOTES THE BEST RESULT,
WHILE
YELLOW
DENOTES THE SECOND-BEST.
Method
buu
chair
sofa
bike
shrub
mean
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
3DGS [6]
7.53/ 0.299/ 0.442
6.06/ 0.151/ 0.742
6.31/ 0.216/ 0.723
6.37/ 0.077/ 0.781
8.15/ 0.044/ 0.620
6.88/ 0.157/ 0.662
Image Enhancement Methods + 3DGS
3DGS + Z-DCE [14]
18.02/ 0.834/ 0.303
12.55/ 0.725/ 0.478
14.66/ 0.822/ 0.460
10.26/ 0.509/ 0.491
12.93/ 0.468/ 0.309
13.64/ 0.672/ 0.408
Z-DCE [14] + 3DGS
17.83/ 0.874/ 0.350
12.47/ 0.762/ 0.399
13.86/ 0.841/ 0.308
10.37/ 0.544/ 0.441
12.74/ 0.487/ 0.248
13.45/ 0.702/ 0.349
3DGS + SCI [15]
13.80/ 0.845/ 0.339
19.70/ 0.812/ 0.455
19.63/ 0.851/ 0.455
12.86/ 0.621/0.463
16.14/ 0.600/ 0.442
15.22/ 0.748/ 0.430
SCI [15] + 3DGS
7.68/ 0.690/ 0.523
11.69/ 0.794/ 0.419
10.02/ 0.770/ 0.365
13.55/ 0.667/ 0.390
15.72/ 0.538/ 0.339
11.73/ 0.692/ 0.407
3DGS + NeRCo [16]
16.64/ 0.765/ 0.401
19.24/ 0.759/ 0.466
16.77/ 0.834/ 0.399
16.33/ 0.700/ 0.427
17.07/ 0.503/ 0.411
17.21/ 0.712/ 0.421
NeRCo [16] + 3DGS
16.69/ 0.802/ 0.330
19.11/ 0.773/ 0.376
18.04/ 0.868/ 0.381
16.16/ 0.703/ 0.397
17.97 / 0.502/ 0.399
17.59/ 0.727/ 0.345
Video Enhancement Methods + 3DGS
LLVE [103] + 3DGS
19.67/ 0.868/ 0.253
15.29/ 0.805/ 0.424
17.18/ 0.858/ 0.379
14.01/ 0.677/ 0.452
15.98/ 0.430/ 0.488
16.43/ 0.728/ 0.399
SGZ [104] + 3DGS
19.21/ 0.832/ 0.270
12.30/ 0.755/ 0.377
14.54/ 0.815/ 0.329
10.61/ 0.563/ 0.375
14.04/ 0.565/ 0.416
14.14/ 0.706/ 0.353
NeRF- and 3DGS-based Enhancement Methods
AME-NeRF [29]
19.89/ 0.854/ 0.312
17.05/ 0.751/ 0.381
17.93/ 0.847/ 0.378
18.14/ 0.732/ 0.437
15.23/ 0.462/ 0.518
17.65/ 0.729/ 0.405
Aleth-NeRF [28]
20.22 / 0.859/ 0.315
20.93 / 0.818/ 0.468
19.52/ 0.857/ 0.354
20.46 / 0.727/ 0.499
18.24 / 0.511/ 0.448
19.87 / 0.754/ 0.417
Gaussian-DK [74]
15.95/ 0.753/ 0.247
14.86/ 0.693/ 0.318
14.39/ 0.782/ 0.350
17.83/ 0.687/ 0.394
12.47/ 0.138/ 0.565
15.10/ 0.610/ 0.375
Our Proposed Method
Luminance-GS [36]
18.09/ 0.877 / 0.193
19.82/ 0.835 / 0.367
20.12 / 0.871 / 0.259
18.27/ 0.749 / 0.411
15.40/ 0.666 / 0.241
18.34/ 0.799 / 0.294
Luminance-GS++
24.75 / 0.924 / 0.164
21.95 / 0.860 / 0.309
23.01 / 0.910 / 0.242
19.54 / 0.764 / 0.385
14.45/ 0.656 / 0.262
20.74 / 0.823 / 0.272
A. Experimental Settings
1) Datasets:
Lightness Degradation: To evaluate novel view synthesis
(NVS) under lightness degradation, we follow the experimen-
tal protocol of our conference version [36] and consider three
scenarios: low-light, overexposure, and varying exposure.
For low-light and overexposure settings, we use the LOM
benchmark [28], which contains five scenes (“buu,” “chair,”
“sofa,” “bike,” and “shrub”) captured using a DJI Osmo Ac-
tion 3 camera. Each scene includes 25–65 multi-view images
with three illumination levels (low-light, normal-light, and
overexposure), obtained by varying exposure time and ISO
settings. Following [28], low-light and overexposed views
are used for training, while normal-light images serve as
evaluation targets.
For lightness variation experiments, we adopt the un-
bounded MipNeRF360 dataset [5], consistent with Luminance-
GS [36]. We use seven scenes (“bicycle,” “bonsai,” “counter,”
“garden,” “kitchen,” “room,” and “stump”). Following [36],
varying-exposure inputs are synthesized by applying exposure
scaling [19] and gamma perturbations to increase task diffi-
culty. The synthesis process is defined as:
Cin(x) = (Cori(x) · K)γ,
K ∈(0.05, 0.8) ∪(1.25, 3.0),
γ ∈(0.8, 2.5),
(16)
where Cori(x) denotes the original MipNeRF360 images
and Cin(x) denotes the synthesized varying-exposure inputs.
Example results are shown in Fig. 11.
Color Degradation: We evaluate robustness to color degra-
dation on the Mip-NeRF360 dataset [5] by synthesizing re-
alistic illuminant variations. Following the Planckian Jitter
framework [105], we model the illuminant color ρT ∈R3
by sampling a black-body radiator temperature T according
to Planck’s Law [106] and mapping the resulting spectrum
to sRGB space. Color degradation is applied using a Von

<!-- page 9 -->
9
Fig. 7.
Effect of varying color temperatures (cool, warm, and mixed) on
NVS. We compare the results with and without the color correction loss Lcc
and visualize the corresponding color distributions of rendered novel views
using scatter plots.
Kries transform combined with brightness and contrast per-
turbations. The degraded image Cin(x) is defined as:
Cin(x) = cB · cC · (Cori(x) ⊙ρT )
+ (1 −cC) · µ
 cB · (Cori(x) ⊙ρT )

,
(17)
where ⊙denotes element-wise multiplication, cB and cC
respectively represent brightness and contrast factors [105],
and µ(·) denotes the mean operator.
Following [105], we evaluate three illumination conditions:
(1) Cool lighting (8000K ≤T
≤9500K) for “bonsai,”
“counter,” and “garden”; (2) Warm lighting (1800K ≤T ≤
2500K) for “kitchen,” “room,” and “stump”; and (3) Mixed
lighting for “bicycle,” where views are sampled across a wide
temperature range (1800K–9500K).
Mixed Degradation: For this setting, we use the MipNeRF-
360 dataset [5], where lightness and color degradations are
jointly applied to simulate challenging real-world conditions.
Such combined degradations across multiple views signifi-
cantly increase the difficulty of consistent 3D reconstruction.
The degraded input is defined as:
Cin(x) =(colordeg(Cori(x)) · K)γ,
K ∈(0.05, 0.8) ∪(1.25, 3.0),
T ∈(2500K, 8000K), γ ∈(0.8, 2.5),
(18)
Fig. 8. Chromaticity consistency visualization in CIELAB color space. Pixel
chromaticity coordinates (a∗, b∗) from rendered views are plotted for a static
region. Luminance-GS++ produces a compact and concentrated distribution,
indicating improved cross-view color consistency and stable photometric
appearance.
where colordeg(·) denotes the color degradation process de-
scribed previously. Synthesized examples are shown in Fig. 11.
2) Comparison Methods:
We compare against two categories of baselines: (1) state-
of-the-art 2D image/video restoration methods [14]–[16],
[103], [104] applied as preprocessing for 3DGS, and (2) recent
NeRF- and 3DGS-based methods designed for robust 3D
reconstruction under challenging conditions [28], [29], [74].
Image Restoration Methods target single-view enhancement
and therefore do not explicitly enforce coherence or consis-
tency in multi-view 3D settings. We include three state-of-
the-art image enhancement methods (Z-DCE [14], SCI [15],
and NerCo [16]) and three exposure correction methods
(MSEC [19], IAT [20], and MSLT [21]). Since single-image
approaches cannot directly produce multi-view-consistent out-
puts, we integrate them with 3DGS [6] using two evaluation
pipelines: ①applying restoration as a preprocessing prior to
3DGS reconstruction (denoted as “ ”+3DGS in Tab. I); and ②
performing restoration as a post-processing after novel view
generation (denoted as 3DGS+“ ” in Tab. I).
Video Restoration Methods emphasize temporal consistency
across frames, leading to improved spatial and temporal co-
herence compared to single-image approaches. We include two
state-of-the-art video enhancement methods, LLVE [103] and
SGZ [104], for comparison. Since these methods are designed
to enhance existing views rather than synthesize novel ones,
we integrate them with 3DGS [6] to ensure a fair comparison
within the NVS pipeline.
Novel View Synthesis (NVS) Methods. We compare with a
range of state-of-the-art NVS approaches: ①Aleth-NeRF [28],
a NeRF-based method for low-light and overexposed scenes
based on the “Concealing Fields” hypothesis; ②AME-
NeRF [29], which addresses low-light reconstruction using
a dual-level optimization framework; ③Gaussian-DK [74],

<!-- page 10 -->
10
TABLE II
EXPERIMENTAL RESULTS (PSNR ↑, SSIM ↑, LPIPS ↓) ON THE OVEREXPOSURE SCENES OF LOM [28] . WE COMPARE WITH EXPOSURE CORRECTION
METHODS [19]–[21] AND NERF-BASED METHODS [28]. RED
DENOTES THE BEST RESULT WHILE
YELLOW
DENOTES THE SECOND-BEST.
Method
“buu”
“chair”
“sofa”
“bike”
“shrub”
mean
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
3DGS [6]
6.96/ 0.674/ 0.609
11.14/ 0.790/ 0.362
10.17/ 0.790/ 0.369
9.58/ 0.730/ 0.323
10.34/ 0.646/ 0.299
9.64/ 0.726/ 0.392
Exposure Correction Methods + 3DGS
3DGS + MSEC [19]
16.03/ 0.806/ 0.517
20.81/ 0.851/ 0.408
20.65/ 0.862/ 0.397
22.10/ 0.826/ 0.305
18.21/ 0.678/ 0.289
19.56/ 0.805/ 0.382
MSEC [19] + 3DGS
15.08/ 0.804/ 0.440
16.63/ 0.797/ 0.416
20.09/ 0.828/ 0.335
17.57/ 0.739/ 0.368
16.61/ 0.666/ 0.255
17.20/ 0.767/ 0.363
3DGS + IAT [20]
15.34/ 0.804/ 0.522
21.96/ 0.833/ 0.292
20.23/ 0.872/ 0.402
22.36/ 0.832/ 0.291
21.24 / 0.765/ 0.226
20.23/ 0.821/ 0.347
IAT [20] + 3DGS
15.86/ 0.803/ 0.387
18.61/ 0.830/ 0.367
17.42/ 0.833/ 0.348
19.17/ 0.801/ 0.235
16.74/ 0.731/ 0.219
17.56/ 0.800/ 0.311
3DGS + MSLT [21]
15.34/ 0.798/ 0.473
21.69/ 0.823/ 0.304
23.05 / 0.830/ 0.317
23.37/ 0.830/ 0.317
18.89 / 0.779/ 0.214
20.39/ 0.815/ 0.345
MSLT [21] + 3DGS
16.35/ 0.805/ 0.333
20.93/ 0.828/ 0.275
21.65 / 0.847/ 0.259
24.03/ 0.841/ 0.244
18.29/ 0.797 / 0.199
20.25/ 0.824/ 0.262
NeRF-based Exposure Correction Method
Aleth-NeRF [28]
16.78/ 0.805/ 0.611
20.08/ 0.820/ 0.499
17.85/ 0.852/ 0.458
19.85/ 0.773/ 0.392
15.91/ 0.477/ 0.483
18.09/ 0.745/ 0.488
Our Proposed Method
Luminance-GS [36]
19.67 / 0.811 / 0.311
22.63 / 0.856 / 0.207
21.16/ 0.878 / 0.204
24.05 / 0.851 / 0.216
16.04/ 0.780 / 0.173
20.71 / 0.835 / 0.222
Luminance-GS++
19.86 / 0.878 / 0.304
24.61 / 0.888 / 0.237
21.54/ 0.889 / 0.216
25.13 / 0.854 / 0.192
15.92/ 0.762/ 0.166
21.41 / 0.854 / 0.223
TABLE III
TRAINING TIME COMPARISONS ON THE LOM [28] AND
MIP-NERF360 [5] DATASETS.
Methods
Aleth-NeRF AME-NeRF 3DGS Gaussian-DK Luminance-GS Luminance-GS++
LOM
(GPU hours)
14.08
11.28
0.08
0.58
0.13
0.23
Methods
NeRF-W
3DGS
GS-W
GS-MCMC
Luminance-GS Luminance-GS++
Mip-NeRF360
(GPU hours)
32.67
0.10
0.58
0.14
0.20
0.28
Fig. 9.
Efficiency comparison: FPS and peak GPU memory (GB) on the
LOM [28] and Mip-NeRF 360 [5] datasets. Methods closer to the upper-left
corner indicate better efficiency.
which integrates 3DGS with learnable tone mapping for low-
light view synthesis; ④NeRF-W [31], which introduces
learnable appearance embeddings and transient modeling for
view synthesis; ⑤GS-W [34], a 3DGS-based extension
of NeRF-W with dynamic appearance modeling; ⑥GS-
MCMC [37], which reformulates 3DGS densification and
pruning as MCMC-based state transitions; and ⑦Luminance-
GS [36], the conference version of our method.
B. Lightness Degradation Experiments
Tables I and II present quantitative comparisons on the
LOM dataset [28] under low-light and overexposure con-
ditions. In low-light scenarios (Tab. I), image enhancement
methods generally achieve limited performance, primarily due
to the lack of multi-view consistency and weak zero-shot
generalization. Among them, NeRCo [16] outperforms Z-
DCE [14] and SCI [15]. Video enhancement methods [103],
[104] achieve slightly improved LPIPS scores compared to
image enhancement and NeRF-based approaches, suggesting
that temporal modeling provides better cross-view coherence.
For NVS-specific methods, prior works such as AME-
NeRF [29], Aleth-NeRF [28], and Gaussian-DK [74] often
exhibit color shifts or geometric artifacts (see Fig. 10). In
contrast, Luminance-GS++ achieves the best overall per-
formance (highlighted in red in Table I) across nearly all
scene-specific metrics and the overall mean. Compared with
Luminance-GS [36], our method improves mean PSNR by
13.1% (18.34 dB →20.74 dB), demonstrating the effectiveness
of the proposed local refinement branch. Qualitative results
(Fig. 10) further confirm improved restoration of fine-grained
details.
A similar trend is observed under overexposure (Tab. II).
Aleth-NeRF [28] and exposure correction methods [19]–[21]
exhibit inferior LPIPS performance, although MSLT [21]
achieves relatively stronger PSNR and SSIM among exposure-
based methods. Luminance-GS++ consistently achieves the
best overall results and surpasses Luminance-GS across all
evaluation metrics.
For lightness variation experiments (Tab. IV), we compare
against 3DGS [6], in-the-wild methods (NeRF-W [31], GS-
W [34]), and the robust NVS method GS-MCMC [37]. NeRF-
W and GS-W generally produce suboptimal results; for exam-
ple, NeRF-W introduces blurry artifacts in 360-degree scenes

<!-- page 11 -->
11
Fig. 10. Experimental results on the low-light and overexposure subsets of LOM dataset [28]. The first column shows the results of the basic 3DGS [6],
while the second and third columns present the results of various comparison methods [14], [19], [20], [28], [29], [74]. The last two columns show the results
of our Luminance-GS [36] and Luminance-GS++. Some regions are zoomed in to better observe the details.
(Fig. 11). Although GS-MCMC achieves competitive LPIPS
scores, it suffers from abrupt lighting transitions and incon-
sistencies. In contrast, Luminance-GS++ consistently outper-
forms all baselines across nearly all scenes, demonstrating
strong robustness to lightness variations. As illustrated in
the “bicycle” scene (Fig. 11), our method maintains spatially
uniform illumination, particularly in low-texture regions such
as the ground.
C. Color Degradation Experiments
For this setting, we conduct comprehensive validations
across cool lighting, warm lighting, and mixed lighting con-
ditions (Tab. IV). Overall, Luminance-GS++ consistently out-
performs the baseline methods, achieving either the best or
second-best performance across most evaluation metrics.
Beyond standard quantitative metrics, we further analyze
chromaticity consistency across rendered views. As shown
in Fig. 8, rendered images are projected into the CIELab
color space, where discrepancies in pixel-wise chromaticity
coordinates (a∗, b∗) are measured across viewpoints. Existing
baselines exhibit dispersed chromaticity distributions, indi-
cating inconsistent color alignment. In contrast, Luminance-
GS++ produces compact and concentrated chromaticity clus-
ters, demonstrating improved cross-view color constancy. Ad-
ditional qualitative comparisons in Fig. 8 and Fig. 11 further
highlight the effectiveness of our method in preserving con-
sistent color tones under challenging illumination variations.
D. Mixed Degradation Experiments
Quantitative comparisons under mixed degradation are re-
ported in Tab. IV. Luminance-GS++ consistently achieves
strong performance, obtaining the best results across most
evaluation metrics and scenarios. Qualitative comparisons are
shown in Fig. 11.

<!-- page 12 -->
12
Fig. 11. Qualitative comparisons of NVS results under lightness, color, and mixed degradation. The last column presents our Luminance-GS++, while the
remaining columns correspond to competing methods [6], [31], [34], [36], [37]. Zoomed-in regions are provided to highlight details.

<!-- page 13 -->
13
TABLE IV
EXPERIMENTAL RESULTS (PSNR ↑, SSIM ↑, LPIPS ↓) ON THE MIPNERF-360 [5] DATASET UNDER THREE DEGRADATION SETTINGS: LIGHTNESS
VARIATION, COLOR DEGRADATION, AND MIXED DEGRADATION. IN THE COLOR DEGRADATION SETTING,
DENOTES COOL COLOR TEMPERATURE,
WHILE
DENOTES WARM COLOR TEMPERATURE.
Lightness Variation
3DGS [6]
NeRF-W [31]
GS-W [34]
GS-MCMC [37]
Luminance-GS [36]
Luminance-GS++
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
bicycle
18.63 /0.539/0.409
13.82/0.233/0.631
18.33/0.618/0.377
18.47/0.591/0.343
18.52/ 0.647 / 0.334
20.93 / 0.678 / 0.302
bonsai
12.50/0.299/0.560
9.64/0.232/0.725
15.46/0.541/ 0.421
13.28/0.371/0.470
15.61 / 0.560 /0.438
16.13 / 0.612 / 0.340
counter
14.01/0.491/0.376
10.81/0.411/0.758
15.79/0.618/0.344
15.64/0.584/ 0.249
16.29/ 0.631 / 0.313
18.01 / 0.677 / 0.289
garden
20.39/0.755/0.205
10.71/0.222/0.678
20.49 /0.765/0.222
19.24/0.768/ 0.186
20.93 / 0.786 /0.193
19.76/ 0.790 / 0.179
kitchen
22.26 /0.832/0.177
13.79/0.502/0.601
20.77/0.802/0.167
20.87/ 0.840 / 0.109
23.09 /0.834/0.144
21.81/ 0.845 / 0.099
room
14.22/0.549/ 0.299
12.89/0.432/0.534
17.19 /0.558/0.378
15.12/0.605/0.326
16.77/ 0.656 /0.320
17.02 / 0.675 / 0.298
stump
16.61/ 0.532 /0.388
13.98/0.403/0.712
14.65/0.503/0.429
18.86 /0.499/ 0.379
15.63/ 0.532 /0.388
18.88 / 0.591 / 0.368
mean
16.95/0.571/0.345
12.23/0.348/0.663
17.52/0.629/0.334
17.35/0.608/ 0.295
18.12 / 0.664 /0.304
18.93 / 0.696 / 0.268
Color Degradation
3DGS [6]
NeRF-W [31]
GS-W [34]
GS-MCMC [37]
Luminance-GS [36]
Luminance-GS++
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
bicycle
19.57 / 0.700 /0.357
11.21/0.273/0.812
16.39/0.356/0.837
18.26/0.629/0.463
14.45/0.665/ 0.328
19.67 / 0.730 / 0.284
bonsai
18.43 / 0.808 / 0.169
13.98/0.398/0.771
15.33/0.495/0.670
17.12/0.785/0.224
13.15/0.666/0.212
17.45 / 0.821 / 0.157
counter
17.21/0.704/0.239
14.51/0.309/0.728
13.95/0.428/0.637
18.00 / 0.722 /0.331
13.62/0.673/ 0.225
17.22 / 0.756 / 0.234
garden
18.74/ 0.828 / 0.221
17.52/0.299/0.835
17.20/0.365/0.832
18.78 / 0.823 /0.261
16.67/0.810/ 0.226
20.24 / 0.828 /0.234
kitchen
13.41/ 0.739 / 0.305
15.68/0.417/0.706
16.37 /0.422/0.595
13.68/0.721/0.382
12.00/0.638/0.406
17.92 / 0.749 / 0.350
room
16.88/ 0.745 /0.335
14.18/0.392/0.788
13.68/0.548/0.680
16.98 /0.733/0.310
13.11/0.668/ 0.252
17.19 / 0.758 / 0.288
stump
18.56 /0.606/0.422
15.23/0.322/0.801
17.66/0.338/0.728
18.53/ 0.611 / 0.383
10.64/0.538/0.448
19.19 / 0.617 / 0.374
mean
17.97 / 0.747 / 0.295
14.62/0.344/0.777
15.80/0.422/0.711
17.33/0.718/0.336
13.38/0.665/0.300
18.41 / 0.751 / 0.274
Mixed Degradation
3DGS [6]
NeRF-W [31]
GS-W [34]
GS-MCMC [37]
Luminance-GS [36]
Luminance-GS++
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
PSNR/ SSIM/ LPIPS
bicycle
14.12/0.350/0.631
9.65/0.176/0.654
16.13 /0.355/0.896
13.60/0.401/0.522
16.03/ 0.542 / 0.469
19.56 / 0.577 / 0.466
bonsai
16.32 /0.632/0.280
10.19/0.198/0.732
14.79/0.476/0.415
14.92/0.545/0.334
15.45/ 0.665 / 0.233
18.16 / 0.728 / 0.228
counter
16.13 /0.617/0.355
9.88/0.375/0.722
14.27/0.438/0.557
14.58/0.527/0.320
16.01/ 0.649 / 0.315
16.94 / 0.660 / 0.304
garden
15.55/ 0.606 / 0.374
13.98/0.216/0.694
15.87/0.364/0.561
14.12/0.561/0.399
16.14 /0.585/0.402
16.46 / 0.647 / 0.389
kitchen
17.87/ 0.731 /0.280
7.52/0.508/0.675
15.47/0.644/0.393
18.07/0.722/ 0.261
18.22 /0.688/0.295
18.14 / 0.738 / 0.257
room
15.60/ 0.665 /0.331
11.19/0.398/0.677
13.63/0.551/0.497
15.67 /0.607/0.335
15.67 /0.661/ 0.314
17.10 / 0.705 / 0.317
stump
15.48/0.328/0.562
10.29/0.378/0.722
17.40 /0.340/0.709
14.51/0.377/0.577
15.30/ 0.500 / 0.427
18.67 / 0.554 / 0.453
mean
15.87/0.561/0.402
10.39/0.321/0.697
15.37/0.453/0.575
15.07/0.534/0.393
16.12 / 0.613 / 0.350
17.86 / 0.658 / 0.352
TABLE V
ABLATION ANALYSIS OF EACH MODULE, INCLUDING THE GLOBAL CURVE Lg, CURVE BIAS Lb
k, PER-VIEW MATRIX Mk, AND THE RESIDUAL BRANCH R.
Lg
Lb
k
Mk
R
LOM [28]
(low-light) buu
LOM [28]
(overexposure) buu
MipNeRF360 [5]
(lightness variation) bicycle
MipNeRF360 [5]
(color degradation) “bicycle”
MipNeRF360 [5]
(mixed degradation) “bicycle”
✓
17.96/ 0.797/ 0.210
19.03/ 0.788/ 0.345
14.08/ 0.493/ 0.432
13.76/ 0.601/ 0.412
12.87/ 0.471/ 0.579
✓
17.79/ 0.654/ 0.375
16.45/ 0.713/ 0.462
17.03/ 0.584/ 0.397
13.65/ 0.599/ 0.427
15.42/ 0.488/ 0.556
✓
✓
18.03/ 0.858/ 0.199
19.35/ 0.803/ 0.346
17.01/ 0.592/ 0.388
14.05/ 0.629/ 0.355
15.78/ 0.507/ 0.548
✓
✓
✓
18.11/ 0.879/ 0.192
19.69/ 0.812/ 0.313
16.94/ 0.606/ 0.384
14.01/ 0.650/ 0.309
15.96/ 0.502/ 0.530
✓
10.86/ 0.691/ 0.409
17.86/ 0.752/ 0.413
19.51/ 0.631/ 0.369
21.07/ 0.719/ 0.289
18.27/ 0.531/ 0.508
✓
✓
✓
24.53/ 0.911/ 0.165
19.77/ 0.878/ 0.312
19.96/ 0.648/ 0.366
19.82/ 0.724/ 0.286
18.74/ 0.548/ 0.494
✓
✓
✓
19.21/ 0.901/ 0.179
17.95/ 0.842/ 0.354
20.41/ 0.663/ 0.323
19.54/ 0.724/ 0.291
19.12/ 0.563/ 0.468
✓
✓
✓
✓
24.75/ 0.924/ 0.164
19.86/ 0.878/ 0.304
20.93/ 0.678/ 0.302
19.67/ 0.730/ 0.284
19.56/ 0.577/ 0.466
Under challenging mixed-degradation conditions, the in-
the-wild method GS-W [34] struggles to identify a stable
reference appearance, leading to inconsistent and visually
disordered renderings. GS-MCMC [37] produces noticeable
non-uniform illumination artifacts, indicating limited robust-
ness to jointly varying lightness and color degradations. In
contrast, Luminance-GS++ generates visually coherent results
with spatially consistent brightness and well-balanced color
reproduction across views.
E. Efficiency Evaluation
We evaluate computational efficiency by comparing model
training time (GPU hours), as summarized in Tab. III. The re-
sults demonstrate that Luminance-GS++ reduces training costs
by orders of magnitude compared to NeRF-based methods
(e.g., Aleth-NeRF [28] and NeRF-W [31]). Despite introduc-
ing additional components to enhance reconstruction quality,
our approach maintains training efficiency comparable to other
3DGS-based variants.
Furthermore, we evaluate runtime efficiency in terms of
rendering FPS and GPU memory consumption. As illustrated

<!-- page 14 -->
14
TABLE VI
ABLATION ANALYSIS OF DIFFERENT LOSS FUNCTIONS.
w/o Lspa
w/o Lcurve
w/o Lcc
Full
LOM (low-light) buu
13.05/ 0.809/ 0.322
14.48/ 0.789/ 0.459
24.69/ 0.918/ 0.162
24.75/ 0.924/ 0.164
MipNeRF360
(color degradation) bicycle
21.11/ 0.710/ 0.324
19.57/ 0.612/ 0.449
20.35/ 0.728/ 0.295
19.67/ 0.730/ 0.284
MipNeRF360
(mixed degradation) bicycle
17.17/ 0.466/ 0.569
15.08/ 0.387/ 0.657
19.39/ 0.550/ 0.496
19.56/ 0.577/ 0.466
TABLE VII
ABLATION ANALYSIS OF DIFFERENT RESIDUAL BRANCH BLOCKS.
LOM (low-light)
LOM (overexposure)
MipNeRF-360 (mixed degradation)
ResNet [107] Block
20.77/ 0.812/ 0.279
21.33/ 0.851/ 0.226
17.88/ 0.651/ 0.367
PVT [108] Block
19.98/ 0.811/ 0.301
20.82/ 0.836/ 0.233
17.12/ 0.631/ 0.389
ConvNext [100] Block
20.74/ 0.823/ 0.272
21.41/ 0.854/ 0.223
17.86/ 0.658/ 0.352
in Fig. 9, Luminance-GS++ occupies the upper-left region of
the performance plot, indicating a favorable trade-off between
rendering speed and memory efficiency. Our method achieves
real-time rendering performance comparable to the original
3DGS while requiring substantially less GPU memory than
NeRF-based approaches, as well as Gaussian-DK [74] and
GS-W [34].
F. Ablation Studies
We conduct comprehensive ablation studies to systemati-
cally evaluate the contribution of each proposed component
and loss function across diverse scenarios.
We first analyze the impact of individual modules in
Tab. VII. The full model consistently achieves the best per-
formance, confirming the necessity of each component. In
particular, the residual branch R is critical for handling com-
plex degradations; for example, under “mixed degradation,”
incorporating R improves PSNR from 15.96 dB to 19.56 dB.
However, R relies on structural guidance from the global
curve Lg. When applied in isolation on the low-light dataset,
performance collapses to 10.86 dB, whereas the full config-
uration restores performance to 24.75 dB. This indicates that
Lg provides a stable luminance baseline, while R refines high-
frequency details and color deviations.
We further evaluate the contribution of each loss component
in Tab. VI. All loss terms contribute positively to performance,
with the spatial consistency loss Lspa playing a particularly
important role under illumination degradations. Qualitative
comparisons in Fig. 12 further highlight the benefits of in-
dividual losses: the curve loss Lcurve improves overall visual
quality, while the color correction loss Lcc enhances cross-
view color consistency. Additional examples are provided in
Fig. 7.
Finally, we analyze architectural choices within the resid-
ual branch R by replacing the ConvNeXt [100] block with
alternatives including ResNet [107] and PVT [108] (Tab. VII).
ConvNeXt yields the best performance, suggesting that strong
local modeling capability is essential for effective residual
correction.
VI. CONCLUSION
We present Luminance-GS++, a framework that improves
the robustness of 3D Gaussian Splatting (3DGS) under chal-
lenging illumination and color variations. Without modifying
Fig. 12. Visualization of the effectiveness of different loss components.
the explicit representation of 3DGS, our method formulates
illumination correction as a pseudo-label optimization problem
that integrates global adjustment and local refinement. We also
introduce effective unsupervised objectives to jointly constrain
pseudo-label generation and 3D Gaussian attribute learning.
Compared with existing NeRF- and 3DGS-based approaches,
Luminance-GS++ demonstrates stronger illumination general-
izability and achieves state-of-the-art performance across both
real-world and synthetic benchmarks.
Looking forward, several directions may further improve
reconstruction quality and efficiency. First, enhancing the qual-
ity of pseudo-labels remains critical, as improved supervision
directly benefits rendering performance. Recent advances in
generative models, particularly diffusion-based approaches,
offer promising opportunities for high-fidelity pseudo-label
generation; however, integrating such models without incur-
ring significant computational overhead remains an open chal-
lenge. Second, combining our framework with efficient feed-
forward NVS methods [49], [109] could improve scalability
by leveraging large-scale pretraining while avoiding costly per-
scene optimization.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” in ECCV, 2020. 1, 2, 3, 4
[2] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing
and improving neural radiance fields,” arXiv:2010.07492, 2020. 1
[3] R. Li, H. Gao, M. Tancik, and A. Kanazawa, “Nerfacc: Efficient sam-
pling accelerates nerfs,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), October 2023, pp. 18 537–
18 546. 1, 3
[4] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Zip-nerf: Anti-aliased grid-based neural radiance fields,” ICCV, 2023.
1, 3
[5] ——, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,”
CVPR, 2022. 1, 2, 3, 5, 8, 9, 10, 13
[6] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, July 2023. 1, 2, 3, 4, 8, 9, 10, 11, 12, 13
[7] D. Charatan, S. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat:
3d gaussian splats from image pairs for scalable generalizable 3d
reconstruction,” in arXiv, 2023. 1, 2, 3
[8] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), June 2024, pp. 10 349–10 358. 1
[9] T. Liu, G. Wang, S. Hu, L. Shen, X. Ye, Y. Zang, Z. Cao, W. Li, and
Z. Liu, “Mvsgaussian: Fast generalizable gaussian splatting reconstruc-
tion from multi-view stereo,” in European Conference on Computer
Vision.
Springer, 2025, pp. 37–53. 1, 3

<!-- page 15 -->
15
[10] B. Mildenhall, P. Hedman, R. Martin-Brualla, P. P. Srinivasan, and J. T.
Barron, “NeRF in the dark: High dynamic range view synthesis from
noisy raw images,” arXiv, 2021. 1, 3
[11] Y. Wang, C. Wang, B. Gong, and T. Xue, “Bilateral guided radiance
field processing,” ACM Transactions on Graphics (TOG), vol. 43, no. 4,
pp. 1–13, 2024. 1, 3
[12] V. Rudnev, M. Elgharib, W. Smith, L. Liu, V. Golyanik, and
C. Theobalt, “Nerf for outdoor scene relighting,” in European Con-
ference on Computer Vision (ECCV), 2022. 1
[13] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “Gs-ir: 3d gaussian
splatting for inverse rendering,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 644–21 653. 1, 3, 4
[14] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong,
“Zero-reference deep curve estimation for low-light image enhance-
ment,” in Proceedings of the IEEE conference on computer vision and
pattern recognition (CVPR), June 2020, pp. 1780–1789. 1, 3, 6, 8, 9,
10, 11
[15] L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward fast, flexible, and
robust low-light image enhancement,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
5637–5646. 1, 8, 9, 10
[16] S. Yang, M. Ding, Y. Wu, Z. Li, and J. Zhang, “Implicit neural represen-
tation for cooperative low-light image enhancement,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV),
October 2023, pp. 12 918–12 927. 1, 8, 9, 10
[17] W. Wang, H. Yang, J. Fu, and J. Liu, “Zero-reference low-light
enhancement via physical quadruple priors,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2024, pp. 26 057–26 066. 1
[18] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma, “Diff-retinex++: Retinex-
driven reinforced diffusion model for low-light image enhancement,”
IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 47, no. 8, pp. 6823–6841, 2025. 1
[19] M. Afifi, K. G. Derpanis, B. Ommer, and M. S. Brown, “Learn-
ing multi-scale photo exposure correction,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2021, pp. 9157–9167. 1, 8, 9, 10, 11
[20] Z. Cui, K. Li, L. Gu, S. Su, P. Gao, Z. Jiang, Y. Qiao, and T. Harada,
“You only need 90k parameters to adapt light: a light weight trans-
former for image enhancement and exposure correction,” in 33rd
British Machine Vision Conference 2022, BMVC 2022, London, UK,
November 21-24, 2022.
BMVA Press, 2022. 1, 5, 9, 10, 11
[21] Y. Zhou, C. Li, J. Liang, T. Xu, X. Liu, and J. Xu, “4k-resolution
photo exposure correction at 125 fps with 8k parameters,” in Winter
Conference on Applications of Computer Vision (WACV), 2024. 1, 9,
10
[22] J. T. Barron, “Convolutional color constancy,” ICCV, 2015. 1
[23] Y. Hu, B. Wang, and S. Lin, “Fc 4: Fully convolutional color con-
stancy with confidence-weighted pooling,” in Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, 2017, pp.
4085–4094. 1
[24] M. Afifi and M. S. Brown, “Deep white-balance editing,” in Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2020. 1, 7
[25] C. Li, X. Kang, and A. Ming, “Wbflow: Few-shot white balance for
srgb images via reversible neural flows,” in Proceedings of the Thirty-
Second International Joint Conference on Artificial Intelligence, ser.
IJCAI ’23, 2023. 1, 7
[26] Y. Cheng, Z. Cui, S. Su, L. Gu, and Z. Zhang, “Perception-inspired
color space design for photo white balance editing,” 2025. 1
[27] W. Haoyuan, X. Xiaogang, X. Ke, and W. L. Rynson, “Lighting up nerf
via unsupervised decomposition and enhancement,” in ICCV, 2023. 1,
3
[28] Z. Cui, L. Gu, X. Sun, X. Ma, Y. Qiao, and T. Harada, “Aleth-
nerf: Illumination adaptive nerf with concealing field assumption,” in
Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13
[29] Y. Zou, X. Li, Z. Jiang, and J. Liu, “Enhancing neural radiance fields
with adaptive multi-exposure fusion: A bilevel optimization approach
for novel view synthesis,” in Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7882–7890. 1, 3, 8, 9,
10, 11
[30] S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas,
A. Jacobson, D. Fleet, and A. Tagliasacchi, “Spotlesssplats: Ignoring
distractors in 3d gaussian splatting,” ACM Trans. Graph., vol. 44, no. 2,
Apr. 2025. 1, 5
[31] R. Martin-Brualla, N. Radwan, M. S. M. Sajjadi, J. T. Barron, A. Doso-
vitskiy, and D. Duckworth, “NeRF in the Wild: Neural Radiance Fields
for Unconstrained Photo Collections,” in CVPR, 2021. 1, 3, 10, 12, 13
[32] X. Chen, Q. Zhang, X. Li, Y. Chen, Y. Feng, X. Wang, and J. Wang,
“Hallucinated neural radiance fields in the wild,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 12 943–12 952. 1, 3
[33] Y. Yang, S. Zhang, Z. Huang, Y. Zhang, and M. Tan, “Cross-ray neural
radiance fields for novel-view synthesis from unconstrained image
collections,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 15 901–15 911. 1, 3
[34] D. Zhang, C. Wang, W. Wang, P. Li, M. Qin, and H. Wang, “Gaussian
in the wild: 3d gaussian splatting for unconstrained image collections,”
ECCV, 2024. 1, 2, 3, 5, 10, 12, 13, 14
[35] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler,
“WildGaussians: 3D gaussian splatting in the wild,” arXiv, 2024. 1, 3,
4, 5
[36] Z. Cui, X. Chu, and T. Harada, “Luminance-gs: Adapting 3d gaussian
splatting to challenging lighting conditions with view-adaptive curve
adjustment,” in CVPR, 2025. 2, 4, 5, 8, 10, 11, 12, 13
[37] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, J. Tseng, H. Isack,
A. Kar, A. Tagliasacchi, and K. M. Yi, “3d gaussian splatting as markov
chain monte carlo,” arXiv preprint arXiv:2404.09591, 2024. 2, 3, 10,
12, 13
[38] Q. Zhang, Y. Feng, and H. Li, “Physically plausible color correction
for neural radiance fields,” in ECCV 2024, 2024, pp. 169–187. 1, 3
[39] Y. Cai, Z. Xiao, Y. Liang, Y. Zhang, X. Yang, Y. Liu, and A. Yuille,
“Hdr-gs: Efficient high dynamic range novel view synthesis at 1000x
speed via gaussian splatting,” arXiv preprint arXiv:2405.15125, 2024.
2
[40] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
2023. 2, 3
[41] I. Chugunov, A. Joshi, K. Murthy, F. Bleibel, and F. Heide, “Neural
light spheres for Implicit Image Stitching and View Synthesis,” in
Proceedings of the ACM SIGGRAPH Asia 2024.
ACM, 2024. 3
[42] C. Zhu, R. Wan, and B. Shi, “Neural transmitted radiance fields,”
in Advances in Neural Information Processing Systems, S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds.,
vol. 35.
Curran Associates, Inc., 2022, pp. 38 994–39 006. 3
[43] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting
for efficient 3d mesh reconstruction and high-quality mesh rendering,”
CVPR, 2024. 3
[44] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian
splatting for geometrically accurate radiance fields,” in SIGGRAPH
2024 Conference Papers. Association for Computing Machinery, 2024.
3
[45] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang,
“Colmap-free 3d gaussian splatting,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June
2024, pp. 20 796–20 805. 3
[46] H. Zhao, H. Weng, D. Lu, A. Li, J. Li, A. Panda, and S. Xie,
“On scaling up 3d gaussian splatting training,” in The Thirteenth
International Conference on Learning Representations, 2025. 3
[47] Z. Zhang, T. Song, Y. Lee, L. Yang, C. Peng, R. Chellappa, and D. Fan,
“LP-3DGS: Learning to prune 3d gaussian splatting,” in The Thirty-
eighth Annual Conference on Neural Information Processing Systems,
2024. 3
[48] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, “Splatter image: Ultra-
fast single-view 3d reconstruction,” in The IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024. 3
[49] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J.
Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images,” arXiv preprint arXiv:2403.14627, 2024. 3, 14
[50] Y. Cai, Y. Liang, J. Wang, A. Wang, Y. Zhang, X. Yang, Z. Zhou, and
A. Yuille, “Radiative gaussian splatting for efficient x-ray novel view
synthesis,” arXiv preprint arXiv:2403.04116, 2024. 3
[51] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, “Sgs-
slam: Semantic gaussian splatting for neural dense slam,” in European
Conference on Computer Vision.
Springer, 2024, pp. 163–179. 3
[52] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang,
“Drivinggaussian: Composite gaussian splatting for surrounding dy-
namic autonomous driving scenes,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 634–21 643. 3

<!-- page 16 -->
16
[53] M. Li, S. Liu, T. Deng, and H. Wang, “Densesplat: Densifying gaussian
splatting slam with neural radiance prior,” IEEE Transactions on
Visualization and Computer Graphics, pp. 1–14, 2025. 3
[54] S. Liu, T. Deng, H. Zhou, L. Li, H. Wang, D. Wang, and M. Li,
“Mg-slam: Structure gaussian splatting slam with manhattan world hy-
pothesis,” IEEE Transactions on Automation Science and Engineering,
2025. 3
[55] S. Choi, B. Chao, J. Yang, M. Gopakumar, and G. Wetzstein, “Gaussian
wave splatting for computer-generated holography,” ACM Transactions
on Graphics (TOG), vol. 44, no. 4, p. 1–13, 2025. 3
[56] C. Wang, X. Wu, Y.-C. Guo, S.-H. Zhang, Y.-W. Tai, and S.-M. Hu,
“Nerf-sr: High-quality neural radiance fields using supersampling,”
arXiv, 2021. 3
[57] Y. Wan, Y. Cheng, M. Shao, and W. Zuo, “S2gaussian: Sparse-view
super-resolution 3d gaussian splatting,” CVPR, 2025. 3
[58] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, “WaterSplatting: Fast
underwater 3D scene reconstruction using gaussian splatting,” 3DV,
2025. 3
[59] D. Levy, A. Peleg, N. Pearl, D. Rosenbaum, D. Akkaynak, S. Korman,
and T. Treibitz, “Seathru-nerf: Neural radiance fields in scattering
media,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 56–65. 3
[60] R. Andrea, B. Mario, W. Stefanie, S. Alessandro, S. Dominik, and
H. Felix, “Scatternerf: Seeing through fog with physically-based in-
verse neural rendering,” 2023. 3
[61] S. Liu, L. Gu, Z. Cui, X. Chu, and T. Harada, “I2-nerf: Learning
neural radiance fields under physically-grounded media interactions,”
in Advances in Neural Information Processing Systems, 2025. 3
[62] S. Liu, X. Chen, H. Chen, Q. Xu, and M. Li, “Deraings: Gaussian
splatting for enhanced scene reconstruction in rainy environments,”
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
no. 5, pp. 5558–5566, Apr. 2025. 3
[63] N. Pearl, T. Treibitz, and S. Korman, “Nan: Noise-aware nerfs for burst-
denoising,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), June 2022, pp. 12 672–12 681.
3
[64] L. Ma, X. Li, J. Liao, Q. Zhang, X. Wang, J. Wang, and P. V.
Sander, “Deblur-nerf: Neural radiance fields from blurry images,” arXiv
preprint arXiv:2111.14292, 2021. 3
[65] B. Lee, H. Lee, X. Sun, U. Ali, and E. Park, “Deblurring 3d gaussian
splatting,” 2024. 3
[66] S. Liu, C. Bao, Z. Cui, Y. Liu, X. Chu, L. Gu, M. V. Conde,
R. Umagami, T. Hashimoto, Z. Hu et al., “Realx3d: A physically-
degraded 3d benchmark for multi-view visual restoration and recon-
struction,” arXiv preprint arXiv:2512.23437, 2025. 3
[67] X. Lin, S. Luo, X. Shan, X. Zhou, C. Ren, L. Qi, M.-H. Yang,
and N. Vasconcelos, “HQGS: High-quality novel view synthesis with
gaussian splatting in degraded scenes,” in The Thirteenth International
Conference on Learning Representations, 2025. 3
[68] H. Dahmani, M. Bennehar, N. Piasco, L. Rold˜ao, and D. V. Tsishkou,
“Swag: Splatting in the wild images with appearance-conditioned
gaussians,” ArXiv, vol. abs/2403.10427, 2024. 3
[69] T. Zhang, K. Huang, W. Zhi, and M. Johnson-Roberson, “Darkgs:
Learning neural illumination and 3d gaussians relighting for robotic
exploration in the dark,” 2024 International Conference on Intelligent
Robots and Systems (IROS), 2024. 3
[70] X. Jin, P. Jiao, Z.-P. Duan, X. Yang, C.-L. Guo, B. Ren, and C.-Y.
Li, “Lighting every darkness with 3dgs: Fast training and real-time
rendering for hdr view synthesis,” in arxiv preprint, 2024. 3, 4
[71] Z. Li, Y. Wang, A. Kot, and B. Wen, “From chaos to clarity: 3dgs in
the dark,” 2024. 3
[72] S. Singh, A. Garg, and K. Mitra, “Hdrsplat: Gaussian splatting for
high dynamic range 3d scene reconstruction from raw images,” BMVC,
2024. 3
[73] J. Xu, M. Liao, K. R. Prabhakar, and V. M. Patel, “Leveraging thermal
modality to enhance reconstruction in low-light conditions,” arXiv
preprint arXiv:2403.14053, 2024. 3
[74] S. Ye, Z.-H. Dong, Y. Hu, Y.-H. Wen, and Y.-J. Liu, “Gaussian in the
dark: Real-time view synthesis from inconsistent dark images using
gaussian splatting,” arXiv preprint arXiv:2408.09130, 2024. 3, 8, 9,
10, 11, 14
[75] Z. Qu, K. Xu, G. P. Hancke, and R. W. H. Lau, “Lush-neRF: Lighting
up and sharpening neRFs for low-light scenes,” in The Thirty-eighth
Annual Conference on Neural Information Processing Systems, 2024.
3
[76] H. Zhou, W. Dong, and J. Chen, “Lita-gs: Illumination-agnostic novel
view synthesis via reference-free 3d gaussian splatting and physical
priors,” in Proceedings of the Computer Vision and Pattern Recognition
Conference, 2025, pp. 21 580–21 589. 3
[77] M. A. Brown and D. G. Lowe, “Automatic panoramic image stitching
using invariant features,” International Journal of Computer Vision,
vol. 74, pp. 59–73, 2007. 3
[78] C. Ding and Z. Ma, “Multi-camera color correction via hybrid his-
togram matching,” IEEE Transactions on Circuits and Systems for
Video Technology, vol. 31, no. 9, pp. 3327–3337, 2021. 3
[79] M. Xia, J. Yao, and Z. Gao, “A closed-form solution for multi-
view color correction with gradient preservation,” ISPRS Journal of
Photogrammetry and Remote Sensing, vol. 157, pp. 188–200, 2019. 3
[80] L. Gu, A. A. Robles-Kelly, and J. Zhou, “Efficient estimation of
reflectance parameters from imaging spectroscopy,” IEEE Transactions
on Image Processing, vol. 22, no. 9, pp. 3648–3663, 2013. 3
[81] L. Gu, C. P. Huynh, and A. Robles-Kelly, “Segmentation and estima-
tion of spatially varying illumination,” IEEE Transactions on Image
Processing, vol. 23, no. 8, pp. 3478–3489, 2014. 3
[82] K. Rematas, A. Liu, P. P. Srinivasan, J. T. Barron, A. Tagliasacchi,
T. Funkhouser, and V. Ferrari, “Urban radiance fields,” CVPR, 2022. 3
[83] E. Reinhard, M. Stark, P. Shirley, and J. Ferwerda, “Photographic tone
reproduction for digital images,” ACM Trans. Graph., vol. 21, no. 3,
p. 267–276, Jul. 2002. 3
[84] S. B. Kang, A. Kapoor, and D. Lischinski, “Personalization of image
enhancement,” in 2010 IEEE Computer Society Conference on Com-
puter Vision and Pattern Recognition, 2010, pp. 1799–1806. 3
[85] L. Yuan and J. Sun, “Automatic exposure correction of consumer
photographs,” in European Conference on Computer Vision, 2012. 3
[86] V. Bychkovsky, S. Paris, E. Chan, and F. Durand, “Learning photo-
graphic global tonal adjustment with a database of input / output image
pairs,” in CVPR 2011, 2011, pp. 97–104. 3
[87] W. Wang, Z. Xu, H. Huang, and J. Liu, “Self-aligned concave curve:
Illumination enhancement for unsupervised adaptation,” in Proceedings
of the ACM International Conference on Multimedia, October 2022. 3
[88] T. Jiang, C. Wang, X. Li, R. Li, H. Fan, and S. Liu, “Meflut:
Unsupervised 1d lookup tables for multi-exposure image fusion,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 10 542–10 551. 3
[89] S. Moran, S. McDonagh, and G. Slabaugh, “Curl: Neural curve layers
for global image enhancement,” in 2020 25th International Conference
on Pattern Recognition (ICPR), 2021, pp. 9796–9803. 3
[90] H.-U. Kim, Y. J. Koh, and C.-S. Kim, “Global and local enhancement
networks for paired and unpaired image enhancement,” in Computer
Vision – ECCV 2020, A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm,
Eds.
Cham: Springer International Publishing, 2020, pp. 339–354. 3,
5
[91] C.-K. T. Chao, J. Klein, J. Tan, J. Echevarria, and Y. Gingold,
“Colorfulcurves: Palette-aware lightness control and color editing via
sparse optimization,” ACM Trans. Graph., vol. 42, no. 4, Jul. 2023. 3
[92] S. Yang, X. Zhang, Y. Wang, J. Yu, Y. Wang, and J. Zhang, “Difflle:
Diffusion-guided domain calibration for unsupervised low-light image
enhancement,” 2023. 3
[93] D. Serrano-Lozano, L. Herranz, M. S. Brown, and J. Vazquez-Corral,
“Colorcurves: Learned image enhancement via color naming,” ECCV,
2024. 3
[94] H. Lee, K. Kang, J. Ok, and S. Cho, “Cliptone: Unsupervised learning
for text-based image tone adjustment,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
2942–2951. 3
[95] Z. Cui, L. Gu, and T. Harada, “Discovering an image-adaptive
coordinate
system
for
photography
processing,”
arXiv
preprint
arXiv:2501.06448, 2025. 3
[96] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Surface splatting,”
in Proceedings of the 28th annual conference on Computer graphics
and interactive techniques, 2001, pp. 371–378. 4
[97] J. L. Sch¨onberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixel-
wise view selection for unstructured multi-view stereo,” in European
Conference on Computer Vision (ECCV), 2016. 4
[98] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” in
Computer Vision – ECCV 2020, A. Vedaldi, H. Bischof, T. Brox, and
J.-M. Frahm, Eds. Cham: Springer International Publishing, 2020, pp.
213–229. 5
[99] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,

<!-- page 17 -->
17
J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Trans-
formers for image recognition at scale,” in International Conference on
Learning Representations, 2021. 5
[100] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie,
“A convnet for the 2020s,” Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2022. 5, 14
[101] G. Buchsbaum, “A spatial processor model for object colour percep-
tion,” Journal of the Franklin Institute, vol. 310, no. 1, pp. 1–26, 1980.
7
[102] D. H. Brainard and B. A. Wandell, “Analysis of the retinex theory of
color vision,” J. Opt. Soc. Am. A, vol. 3, no. 10, pp. 1651–1661, Oct
1986. 7
[103] F. Zhang, Y. Li, S. You, and Y. Fu, “Learning temporal consistency for
low light video enhancement from single images,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2021, pp. 4967–4976. 8, 9, 10
[104] S. Zheng and G. Gupta, “Semantic-guided zero-shot learning for low-
light image/video enhancement,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2022, pp. 581–
590. 8, 9, 10
[105] S. Zini, A. Gomez-Villa, M. Buzzelli, B. Twardowski, A. D. Bagdanov,
and J. van de weijer, “Planckian jitter: countering the color-crippling
effects of color jitter on self-supervised training,” in The Eleventh
International Conference on Learning Representations, 2023. 8, 9
[106] D. G. Andrews, An Introduction to Atmospheric Physics.
Cambridge;
New York: Cambridge University Press, 2000. 8
[107] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” arXiv preprint arXiv:1512.03385, 2015. 14
[108] W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu,
P. Luo, and L. Shao, “Pyramid vision transformer: A versatile backbone
for dense prediction without convolutions,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2021, pp.
568–578. 14
[109] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang,
F. Zhao et al., “Anysplat: Feed-forward 3d gaussian splatting from
unconstrained views,” ACM Transactions on Graphics (TOG), vol. 44,
no. 6, pp. 1–16, 2025. 14
