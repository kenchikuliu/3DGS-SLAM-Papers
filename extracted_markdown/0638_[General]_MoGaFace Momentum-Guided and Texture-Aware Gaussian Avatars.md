<!-- page 1 -->
MoGaFace: Momentum-Guided and Texture-Aware Gaussian Avatars for
Consistent Facial Geometry
Yujian Liu1, 2, Linlang Cao1, Chuang Chen1, Fanyu Geng 1,
Dongxu Shen3, Peng Cao4, Shidang Xu2*, Xiaoli Liu1*
1AiShiWeiLai AI Research, Beijing, China
2South China University of Technology, Guangzhou, China
3Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China
4Northeastern University, Shenyang, China
Abstract
Existing 3D head avatar reconstruction methods adopt a two-
stage process, relying on tracked FLAME meshes derived
from facial landmarks, followed by Gaussian-based render-
ing. However, misalignment between the estimated mesh and
target images often leads to suboptimal rendering quality and
loss of fine visual details. In this paper, we present Mo-
GaFace, a novel 3D head avatar modeling framework that
continuously refines facial geometry and texture attributes
throughout the Gaussian rendering process. To address the
misalignment between estimated FLAME meshes and target
images, we introduce the Momentum-Guided Consistent Ge-
ometry module, which incorporates a momentum-updated ex-
pression bank and an expression-aware correction mechanism
to ensure temporal and multi-view consistency. Additionally,
we propose Latent Texture Attention, which encodes compact
multi-view features into head-aware representations, enabling
geometry-aware texture refinement via integration into Gaus-
sians. Extensive experiments show that MoGaFace achieves
high-fidelity head avatar reconstruction and significantly im-
proves novel-view synthesis quality, even under inaccurate
mesh initialization and unconstrained real-world scenarios.
Introduction
Creating animatable head avatars has long been a challenge
in computer vision (Beeler et al. 2010; Thies et al. 2016; Xu
et al. 2023). Photorealistic and dynamic rendering from ar-
bitrary viewpoints is essential for applications such as gam-
ing, film production, immersive telepresence, and AR/VR.
Equally important is the ability to control these avatars and
ensure they generalize well to novel poses and expressions.
Neural Radiance Fields (NeRF) (Mildenhall et al. 2021)
and its variants (Barron et al. 2021; Chen et al. 2022; M¨uller
et al. 2022) have achieved remarkable results in reconstruct-
ing static scenes. Subsequent works have extended NeRF to
dynamic scenarios, such as modeling arbitrary motions and
generating human-specific head or body avatars (Peng et al.
2024; Liu et al. 2025b). Despite synthesizing high-quality
novel views (Lombardi et al. 2021; Gao et al. 2022), NeRF-
based methods suffer from limited controllability and gen-
eralization due to implicit representations (Yu et al. 2024;
*Corresponding author.
Copyright © 2022, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: MoGaFace introduces a novel approach that
jointly optimizes facial expression coefficients and texture
attributes, enabling high-fidelity facial texture rendering
based on precise alignment with the FLAME geometry.
Zhang, Chen, and Wang 2023), and their costly volumetric
rendering prevents real-time applications (Li et al. 2024).
Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al.
2023) has gained widespread attention for achieving high-
quality rendering and real-time performance in novel view
synthesis. Unlike NeRF, 3DGS models surfaces with inde-
pendent 3D Gaussians defined by opacity, anisotropic co-
variance, and spherical harmonics (Tang and Cham 2024;
Kwon et al. 2024; Xie et al. 2025). By optimizing these dis-
crete geometric primitives throughout the scene, 3DGS not
only surpasses NeRF in rendering fidelity but also supports
real-time applications (Liu et al. 2024b). Building on this,
recent studies (Qian et al. 2024; Wang et al. 2025b; Aneja
et al. 2024; Wang et al. 2025b) have explored the integration
of 3DGS with FLAME (Li et al. 2017), aiming to leverage
the parametric nature of FLAME to enable explicit and tem-
porally consistent control over facial expressions and head
poses. However, current FLAME-based 3DGS approaches
commonly follow a two-stage pipeline, where unsupervised
2D-to-3D mesh estimation is first performed, followed by
mesh-to-image gaussian splatting, as shown in Fig.1(top).
The second stage generally relies on the quality of mesh
estimation through multi-view tracking, which limits scala-
bility in lightweight scenarios where camera viewpoints are
unavailable (Zheng et al. 2024; Wei et al. 2025). Moreover,
mesh-based texture representations lack image-space con-
arXiv:2508.01218v1  [cs.CV]  2 Aug 2025

<!-- page 2 -->
straints, making photorealistic view synthesis and control-
lable expression modeling challenging to unify.
To address these limitations, we propose MoGaFace, a
3D head avatar framework that jointly refines geometry
and texture for realistic and expressive avatar synthesis
(Fig.1(bottom). We first propose a Momentum-Guided Con-
sistent Geometry module to align 2D images with 3D fa-
cial geometry across temporal and multi-view settings. For
temporal consistency, an Expression-Aware Dynamic Cor-
rection mechanism is introduced, which leverages multi-
view images to refine FLAME expressions, aiming to bet-
ter align facial geometry with image-space details at each
time step. However, performing splatting independently for
each view can lead to inconsistent FLAME across views at
the same time. To ensure multi-view consistency, we intro-
duce a Momentum-updated Expression Bank that averages
expression corrections across views at the same timestamp
using a momentum strategy (He et al. 2020), enforcing con-
sistent FLAME geometry during training and inference. Fi-
nally, to establish a stronger constraint from image to mesh,
we propose Latent Texture Attention, which encodes com-
pact multi-view features into a head-aware representation.
This allows for enhanced texture information to be inte-
grated into the 3D Gaussians by supplementing geometry
with texture-related cues. Experimental results demonstrate
that MoGaFace not only improves the clarity and texture
quality of novel-view synthesis but also enables high-fidelity
3D head avatar reconstruction without relying on known
camera parameters, significantly broadening its applicability
in real-world scenarios. In summary, the main contributions
of our work are as follows:
• We propose MoGaFace, a novel 3D head avatar modeling
framework that jointly refines expression coefficients and
texture attributes during the rendering process, enabling
realistic and expressive avatar synthesis.
• We introduce the Momentum-Guided Consistent Geom-
etry module, a key component that significantly enhances
the consistency and accuracy of FLAME expression
modeling. By refining expressions from multi-view im-
ages and propagating corrections via momentum-based
updates, it ensures coherent geometry across both time
and viewpoints.
• We propose Latent Texture Attention, which encodes
compact multi-view features into head-aware represen-
tations, allowing texture-related cues to be effectively in-
tegrated into 3D Gaussians for enhanced texture quality.
• Extensive experiments demonstrate that MoGaFace
achieves high-fidelity 3D head avatar reconstruction and
significantly improves the quality of novel-view synthe-
sis, outperforming existing state-of-the-art methods on
multiple benchmarks.
Related Work
Monocular
3D
Gaussian
Splatting.
Early
3D
head–avatar reconstruction methods employed implicit
neural radiance fields (NeRF) (Mildenhall et al. 2021),
fitting dense MLPs in 4D space to achieve view-consistent
volume rendering. Recent studies turn to monocular 3D
Gaussian Splatting (Fei et al. 2024; Qiu et al. 2025; Ma et al.
2024), which discretises the radiance field into anisotropic
3D Gaussians and rasterises them for real-time synthesis.
MonoGaussianAvatar (Chen et al. 2024a) couples learnable
Gaussians with a deformation field, directly optimising
Gaussian parameters from a single-view video. Gaussian
Blendshapes (Ma et al. 2024) transfers the classic mesh-
blendshape paradigm to Gaussian space, linearly mixing
Gaussian blendshapes for real-time driving. GaussianHead
(Wang et al. 2025a) introduces a learnable Gaussian deriva-
tion mechanism to alleviate texture sparsity. Monocular
reconstruction
is
fundamentally
flawed
(Arampatzakis
et al. 2023), as a single 2D view cannot fully recover 3D
structure, leading existing monocular Gaussian splatting
methods to lack both depth information and multi-view
constraints (Liu et al. 2025a).
Multi-view 3D Gaussian Splatting.
In multi-camera or
turn-table settings, multi-view 3D Gaussian Splatting fully
exploits stereo cues, delivering faithful appearance and ac-
curate geometry (Chen et al. 2024b; Liu et al. 2024a).
Gaussian Avatars (Qian et al. 2024) attaches Gaussians to
FLAME triangles and proposes a binding-inheritance strat-
egy. Gaussian Head Avatar (Xu et al. 2024) extends this
framework by introducing a learnable deformation field
to capture complex expressions, designed as the first 3D
Gaussian-based avatar model for high-resolution reconstruc-
tion. TensorialGaussianAvatar (Wang et al. 2025b) improves
memory and runtime efficiency by encoding static appear-
ance into tri-planes and representing dynamic opacity with
1D feature lines. All the above methods adopt a two-
stage rendering pipeline, rely heavily on precisely estimated
meshes, and lack sufficient texture detail.
Method
This section introduces the proposed MoGaFace (Fig. 2),
which consists of two key modules. The Momentum-Guided
Consistent Geometry module integrates dynamic expression
correction with a momentum-updated expression bank to en-
force consistent, corrected FLAME meshes across views at
the same timestamp during rendering. The Latent-Guided
Texture Attention module adaptively encodes mesh textures
by leveraging multi-view latent features, enhancing fine-
grained visual details. The final part of the section outlines
the loss functions and training strategy used to enable high-
fidelity dynamic head reconstruction.
Preliminaries
3D
Gaussian
Splatting.
3D
Gaussian
Splatting
(3DGS)(Kerbl et al. 2023) is a recent explicit render-
ing technique that enables real-time novel view synthesis
from multi-view images and known camera parameters. A
scene is represented as a set of 3D Gaussian splats, each
parameterized by position µ ∈R3, scale s ∈R3, rotation
quaternion q ∈R4, opacity α ∈R, and view-dependent
color
encoded
via
spherical
harmonics
coefficients
SH ∈R(k+1)2×3. These Gaussians attributes are projected
into the image plane using camera intrinsics and extrinsics,
and rendered via a differentiable alpha-blending rasterizer.

<!-- page 3 -->
Figure 2: Framework: Given multi-view images, MoGaFace initializes FLAME via tracking, refines expressions through a
Momentum-Guided Geometry module for consistent and accurate fitting, and embeds 3D Gaussian textures using a Latent
Texture Attention module that exploits multi-view texture cues.
The final pixel colors are computed by depth-aware sorting
and blending of the projected Gaussians.
FLAME-based Avatar Modeling. To extend 3D Gaussian
Splatting to human head modeling with controllable expres-
sions and poses, representative methods such as Gaussian
Avatars(Qian et al. 2024) introduce a binding mechanism
between each Gaussian primitive and the FLAME mesh, a
parametric 3D facial model. Given multi-view facial videos
and known camera parameters, a common approach to ob-
taining per-frame FLAME parameters in head avatar mod-
eling is to employ landmark-based multi-view tracking al-
gorithms, such as VHAP. These methods typically rely on
detected 2D facial landmarks across views, combined with
camera intrinsics and extrinsics, to optimize the FLAME
model through multi-view landmark reprojection loss. In
this setting, the FLAME model is parameterized by four
components: global rigid motion ∈R6, joint-based articu-
lation ∈R15, shape ∈R300, and expression ∈R100. Non-
rigid facial deformations are captured by the expression pa-
rameters, which control a linear combination of blendshapes
derived from 4D facial scan data. These parameters allow
the model to generate a wide range of identity-independent
expressions, such as smiling, blinking, and mouth move-
ments. While these parameters provide a strong initializa-
tion, they often fail to capture fine-grained expression dy-
namics when projected into image space—particularly in
the absence of accurate camera parameters or under chal-
lenging conditions such as complex lighting, partial occlu-
sion, or identity-expression entanglement. These limitations
significantly hinder the applicability of such methods in
lightweight or mobile deployment scenarios, where precise
calibration is often unavailable.
Momentum-Guided Consistent Geometry
By integrating Expression-Aware Dynamic Correction and
Momentum-Guided Expression Representation, our Mo-
GaFace effectively enforces facial geometry consistency by
dynamically refining per-view expression parameters and
maintaining a shared, coherent FLAME mesh across views
at the same timestamp.
Expression-Aware Dynamic Correction To address the
misalignment between the predicted mesh and the input
images, we introduce a Dynamic Facial Expression-Aware
Correction method, which refines the expression parame-
ters ψ(t) during the rendering process by leveraging multi-
view image supervision at each time step t. As illustrated
in Fig. 2(top), MoGaFace extracts multi-view geometry-
constrained features from multiple facial images at the same
time during the rendering process, enabling end-to-end re-
gression of expression correction terms. Specifically, for
each frame t, the i-th view image I(t)
i
is passed through
a trainable convolutional encoder to extract geometry-
constrained features. The resulting encoded featureF (t)
i
is
denoted as:
F (t)
i
= Encoder(I(t)
i ),
(1)
Considering that FLAME models the entire head, a sin-
gle view is often insufficient to provide complete geomet-
ric constraints for accurate expression estimation. There-
fore, for each frame t, all views in the training set
(I(t)
0 , I(t)
1 , . . . , I(t)
N ) are independently passed through a fea-
ture encoder to extract geometry-constrained features. The
resulting per-view features are then fused using a cross-
attention module to aggregate multi-view information and
capture comprehensive geometry-aware cues. Finally, the
fused feature is subsequently used to regress a correction

<!-- page 4 -->
term that refines the initial FLAME expression parame-
ters, enabling dynamic and view-consistent expression re-
finement throughout the rendering process.
∆ψ(t) = MLP((CrossAttn(F (t)
0 , F (t)
1 , . . . , F (t)
N )))
(2)
Momentum-Guided
Expression
Representation.
In
multi-view 3DGS, the training process typically treats
each view image as an independent supervision unit for
rendering. However, the FLAME model represents the full
head geometry and assumes that all views at the same
timestamp share a set of pose and expression parameters.
This view-consistency constraint is often ignored when
expressions are changed independently from each image,
leading to inconsistencies across views of the same frame.
Inspired by self-supervised learning memory bank (Wu
et al. 2018) and momentum update mechanisms(He et al.
2020), MoGaFace introduces a Momentum-updated Expres-
sion Bank to enforce expression consistency across different
views at the same time step. The bank serves as a global
memory that maintains a set of shared expression correction
terms for each timestamp, ensuring that the expression re-
finement is view-consistent. Specifically, for each frame t,
we maintain a memory entry ∆ψ(t)
bank ∈RN×100, which is
progressively updated with the correction terms ∆ψ(t)
i
re-
gressed from individual views i ∈{0, . . . , N −1}. The
update is performed using exponential moving average as
follows:
∆ψ(t)
bank ←m · ∆ψ(t)
bank + (1 −m) · ∆ψ(t)
i
(3)
where m ∈[0, 1) is the momentum coefficient that controls
the smoothness of the update and is linearly decayed over
training iterations. During training, the banked correction
term ∆ψ(t)
bank is used to supervise and regularize the per-view
regressed expressions. At each timestamp t, the mean of the
N is obtained by averaging the correction terms across the
N per-view corrections is adopted as the FLAME expression
adjustment, thereby enforcing inter-view consistency with-
out requiring an explicit cross-view loss.
Latent-Guided Texture Attention
In this framework, each triangle in the FLAME mesh serves
as an attachment base for a 3D Gaussian primitive. Dur-
ing training, both geometric attributes (position, scale, ro-
tation) and appearance attributes (opacity) of each Gaussian
are optimized. To ensure geometric consistency, Gaussians
are densely anchored to the FLAME mesh via barycentric
coordinates within its associated triangles, analogous to UV-
based surface parameterization. Unlike traditional UV map-
ping that relies on high-quality mesh scans, this approach
uses only real RGB images as weak supervision, which
poses challenges for recovering fine-grained texture details.
To address this, recent advances in single-view UV es-
timation (Zielonka, Bolkart, and Thies 2022,?; Feng et al.
2021) are incorporated, along with the insight that texture at-
tributes can be represented through implicit feature encoding
to preserve local texture coherence. A triplane-based scheme
is applied to the corrected FLAME mesh, where each ver-
tex position is projected onto three orthogonal 2D feature
planes to obtain localized texture descriptors. The triplane
representation H consists of three feature planes:
H = {Hxy, Hxz, Hyz} ∈Rnf ×nf ×nd1
(4)
with spatial resolution nf × nf and feature dimension nd1.
For any point p in canonical space, its feature is computed
by projecting it onto the axis-aligned planes:
h(p) = Hxy(pxy) ⊕Hxz(pxz) ⊕Hyz(pyz)
(5)
where ⊕indicates feature concatenation, and pxy, pxz, pyz
are the 2D projections of point p onto each respective plane.
Inspired by single-view UV estimation, where image fea-
tures are shown to encode global pose, lighting, and fa-
cial details, we introduce an attention mechanism to modu-
late Gaussian attributes. After computing h(p), expression-
aware features F ∈Rd, extracted from the Dynamic Facial
Expression-Aware Correction module, are fused to enhance
the texture encoding with global and appearance-aware cues.
For each Gaussian i, the fusion of the texture descriptor
h(pi) and image feature F is formulated as:
vi = Attn(F, h(pi)),
(6)
where Attn(·) is a lightweight convolutional attention mod-
ule. The output vi adaptively regulates the expression-aware
attributes of the Gaussian, enabling context-dependent mod-
ulation over geometric and appearance spaces.
Following the attention modulation, the resulting atten-
tion vi is further decoded through a multi-head MLP to pro-
duce residual adjustments over key Gaussian attribute sub-
spaces. These include position offset ∆µi ∈R3, anisotropic
scale adjustment ∆si
∈R3, rotation quaternion offset
∆ri ∈R4, and opacity shift ∆αi ∈R. The prediction pro-
cess is formulated as:
[∆µi, ∆si, ∆ri, ∆αi] = MLPhead(vi)
(7)
Each of these components is then used to modulate the cor-
responding Gaussian attributes from their canonical values
Training
Loss Function.The training pipeline supervises the ren-
dered images using a combination of an L1 loss and a D-
SSIM loss. The overall image loss is defined as
Lrgb = (1 −λ)L1 + λLD-SSIM,
(8)
with the weighting factor λ = 0.2. To improve geometric
consistency during facial animation, particularly under dy-
namic facial expressions, two regularization terms are intro-
duced to guide the behavior of the 3D Gaussian primitives.
The position loss encourages each Gaussian to remain close
the centroid of its associated triangle on the FLAME mesh,
reducing misalignment and deformation. It is defined as
Lposition = ∥max(µ, ϵposition)∥2 ,
(9)
where µ denotes the positional offset relative to the triangle
center, and ϵposition = 1 defines a tolerance margin. In par-
allel, a scaling loss is applied to regularize Gaussian size.
Excessively small Gaussians may lead to unstable render-
ing due to high-frequency ray-splat intersections, while large

<!-- page 5 -->
ones may blur fine details. To prevent such artifacts, the scal-
ing loss penalizes values below a specified threshold:
Lscaling = ∥max(s, ϵscaling)∥2 ,
(10)
where s is the scale, and ϵscaling = 0.6 sets the minimum
allowable size. The final training objective combines the im-
age loss with both geometric regularization terms as follows:
L = Lrgb + λpositionLposition + λscalingLscaling,
(11)
where λposition = 0.01 and λscaling = 1. The position and
scaling losses are applied only to visible Gaussians, focus-
ing supervision on perceptually relevant regions while pre-
serving occluded areas
Implementation Details. For parameter optimization, the
Adam optimizer is used, and the same hyperparameters are
applied across all subjects. The learning rate is set to 5e-
3 for Gaussian positions and 1.7e-2 for their scales. For
the FLAME parameters, including translation, joint rotation,
and facial expressions, learning rates of 1e-6, 1e-5, and 1e-
3 are employed for different stages of training. The model
is trained for a total of 1000,000 iterations. During training,
the learning rate for Gaussian positions decays exponentially
and reaches 1% of its original value by the final iteration.
Additional mechanisms are employed to improve conver-
gence and visual quality. Starting from the 6,000th iteration,
a binding interaction loss is activated every 2,000 steps to
enforce consistency between splats and mesh. Furthermore,
every 100,000 iterations, the opacities of the Gaussians are
reset to avoid degenerate transparency. A photo-metric head
tracker is also utilized to supervise the global FLAME pa-
rameters, including shape coefficients (β), global transla-
tion (t), pose (θ), expression (ψ), and vertex offset (∆v)
in the canonical FLAME space. These details collectively
ensure stable, accurate, and high-fidelity dynamic avatar re-
construction during both training and animation.
Experimental Results
Implementations
Dataset.
In
our
experiments,
we
utilize
12
sub-
ject
sequences,
including
10
from
the
NeRSemble
dataset(Kirschstein et al. 2023) and 2 recorded by our-
selves using five unsynchronized cameras. The NeRSemble
recordings consist of 16 camera views covering both frontal
and lateral angles. For each subject, we select 21 video
sequences and downsample the frames to a resolution of
802 × 550. Participants were instructed to either perform
10 predefined facial expressions or recite 10 segments of
neutral speech. The final ”FREE” sequence, containing
spontaneous expressions, is used for self-reenactment task.
For our own recordings (approximately 5 min 10 s each),
each frame is cropped and resized to 512 × 512 with a
green screen background. The data for each subject is split
into training and self-reenactment subsets at a 4:1 ratio.
Evaluation Metrics. We demonstrate the superiority of
our method using several commonly adopted metrics.To
evaluate image quality, we employ full-reference metrics,
including Peak Signal-to-Noise Ratio (PSNR) (Hore and
Ziou 2010), Learned Perceptual Image Patch Similarity
(LPIPS) (Zhang et al. 2018) and Structural Similarity Index
Measure (SSIM) (Wang et al. 2004).
Comparison with Baselines. For quantitative evaluation,
we compare our method with existing state-of-the-art
(SOTA) approaches: Gaussian Avatars (Qian et al. 2024)
binds 3D Gaussian splats to FLAME mesh triangles with
learnable displacements, enabling controllable head synthe-
sis. However, slight misalignments between mesh and im-
ages may weaken local details. Gaussian Head Avatar (Xu
et al. 2024) adds an expression-driven MLP deforma-
tion field on neutral Gaussians to render ultra-high-fidelity
avatars, but relies on dense multi-view capture and complex
initialization. TensorialGaussianAvatar (Wang et al. 2025b)
uses static triplanes and a lightweight 1D feature line to en-
code dynamic textures, but relies on multi-view mesh esti-
mation and struggles to align full head textures due to the
use of a frozen mesh and limited 1D representation. Mono-
GaussianAvatar (Chen et al. 2024a) employs a Gaussian de-
formation field to reconstruct dynamic avatars from a single-
view video, lowering capture requirements; yet the monoc-
ular setup lacks geometric constraints, leading to holes and
artifacts around teeth and hair during large head motions.
GaussianBlendshapes (Ma et al. 2024) encapsulates high-
frequency details into expression-related Gaussian blend-
shapes, supporting linear mixing and real-time synthesis.
However, pure linear interpolation limits generalization to
highly complex or non-linear expressions.
Implementation Details. We evaluate the quality of head
avatars with three settings: (1). novel-view synthesis: driv-
ing an avatar with head poses and expressions from train-
ing sequences and rendering from a held-out viewpoint. (2).
self-reenactment: driving an avatar with unseen poses and
expressions from a held-out sequence of the same subject
and rendering all camera views. (3). cross-identity reenact-
ment: an avatar is driven by the expressions and motions of
a different individual.
Evaluation under Camera-Aware Settings
Quantitative Comparisons. As shown in the Tab.1, our
MoGaFace consistently achieves high PSNR, outperforming
SOTA methods by an average of 3.54 in novel view synthe-
sis, and achieving an average improvement of 2.56 in self-
reenactment tasks. These results demonstrate the strong 3D
reconstruction capability of our method and its superior fi-
delity at the pixel level. This improvement can be attributed
to our dynamic modulation of expression features, which
effectively mitigates the fine-grained visual artifacts previ-
ously highlighted by GaussianAvatar (Qian et al. 2024). Fur-
thermore, the average gain in SSIM (+0.05) and reduction
in LPIPS (−0.05) further demonstrate MoGaFace’s ability
to preserve structural coherence and perceptual quality, val-
idating its effectiveness in maintaining consistency between
3D facial geometry and 2D renderings under multi-view
conditions.
Qualitative Analysis. Fig. 3 and Fig.4 present qualita-
tive comparisons to further evaluate the visual quality
and structural accuracy of MoGaFace. In monocular meth-
ods (Fig.4(left), such as MonoGaussianAvatar (MonoGA)
and GaussiansBlendShape (GBS), limited visual cues from

<!-- page 6 -->
Method
Novel View Synthesis
Self-Reenactment
Self-Reenactment (Novel View)
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
MonoGaussianAvatar
-
-
-
-
-
-
21.78
0.871
0.130
GaussianBlendShapes
-
-
-
-
-
-
24.56
0.864
0.088
GaussianAvatars
29.71
0.939
0.079
23.58
0.891
0.093
23.01
0.900
0.100
GaussianHeadAvatar
26.27
0.807
0.185
19.43
0.794
0.188
19.74
0.792
0.184
TensorialGaussianAvatar
26.33
0.923
0.100
23.40
0.900
0.123
23.28
0.886
0.118
MoGaFace
30.98
0.946
0.068
24.70
0.900
0.084
24.75
0.902
0.090
Table 1: Quantitative comparisons with state-of-the-art methods.
Figure 3: Qualitative visualization with baseline methods on novel view synthesis.
single-view inputs lead to the loss of fine-grained details,
especially around the eyes and mouth, and often cause ren-
dering failures under large facial motions. In multi-view
methods (Fig.3 and Fig.4(right), the additional spatial con-
text provided by multiple views enables improved recon-
struction quality. However, GaussianAvatar (GA) and Gaus-
sianHeadAvatar (GHA) suffer from inconsistencies between
the 3D geometry and the resulting 2D images, leading to
rendering artifacts and blank regions. Moreover, they fail
to recover high-fidelity textures, particularly in low-texture
areas. TensorialGaussianAvatar (TGA) attempts to embed
textures using 1D blendshape features, but the sparsity of
these features and their reliance on imprecise mesh ge-
ometry result in only shallow and less realistic texture re-
constructions. In contrast, MoGaFace effectively preserves
both geometric and appearance consistency across views,
benefiting from a multi-view-consistent expression correc-
tion strategy and latent-guided texture rendering, which to-
gether enhance cross-view fidelity and perceptual realism.
Fig. 5 illustrates the performance of MoGaFace in the cross-
reenactment task, highlighting its ability to generate distinct
details while preserving identity.
Method
Self-Reenactment
PSNR↑
SSIM↑
LPIPS↓
GaussianAvatars
26.83
0.938
0.034
GaussianHeadAvatar
21.29
0.829
0.095
TensorialGaussianAvatar
27.11
0.941
0.055
MoGaFace
28.47
0.943
0.031
Table 2: Quantitative evaluation with multi-view baselines
under camera-free settings.
Evaluation under Camera-Free Settings
To further evaluate the robustness and practical applicabil-
ity of MoGaFace, we conduct experiments under camera-
free settings using two self-recorded portrait videos with-
out ground-truth camera extrinsics. This setup better reflects
real-world scenarios such as mobile or in-the-wild applica-
tions, where multi-view calibration is often infeasible and
camera poses must be estimated from weakly constrained
multi-view inputs to enable scalable deployment. As both
Gaussian Splatting and FLAME tracking rely on known
camera parameters, we first estimate them using a monoc-

<!-- page 7 -->
Figure 4: Qualitative visualization with monocular and multi-view baselines on self-reenactment task.
Figure 5: Cross-identity reenactment. Expressions and poses from the source (right) are transferred to the target (left).
Figure 6: Qualitative visualization with multi-view baselines
under camera-free settings.
ular method(Zielonka, Bolkart, and Thies 2022), which in-
evitably introduces errors. All baselines are evaluated using
these predicted parameters for both mesh fitting and Gaus-
sian rendering.
Quantitative Comparisons. The quantitative results under
this setting are presented in Tab. 2. Inaccurate camera pose
estimation often causes geometric distortions in the pre-
dicted FLAME mesh, resulting in misalignment between 3D
geometry and 2D images. As a result, existing multi-view
reconstruction methods experience a noticeable drop in per-
formance, with significantly lower scores across all evalua-
tion metrics. In contrast, MoGaFace incorporates a correc-
tion strategy that dynamically refines the FLAME mesh us-
ing view-aware cues, ensuring improved 3D–2D consistency
even without ground-truth camera parameters. Furthermore,
by leveraging latent image features for guided texture re-
Method
Novel View Synthesis
Self-Reenactment
PSNR↑
LPIPS↓
PSNR
LPIPS
Freeze
27.49
0.121
24.35
0.134
Single-view
28.71
0.080
24.42
0.105
Multi-view
30.31
0.079
24.56
0.102
Table 3: Quantitative results of expression correction under
different view settings.
finement, MoGaFace enhances the quality of high-frequency
details such as facial contours and skin textures. As a result,
it achieves strong performance—PSNR up by 3.39, SSIM
by 0.04, and LPIPS down by 0.03—showing robustness to
mesh noise and adaptability to camera-free scenarios.
Qualitative Analysis. The qualitative visualization in Fig.
6 further illustrate the advantages of MoGaFace. The base-
line model with frozen mesh during Gaussian rendering suf-
fers from large-scale facial and body artifacts due to cam-
era parameters errors, and also exhibits clothing boundary
leakage, distorted Adam’s apple, and poor shading detail. In
contrast, MoGaFace produces sharper and more structurally
consistent renderings, demonstrating its effectiveness in re-
covering high-fidelity geometry and texture under camera-
free conditions.
Ablation Study
We conduct ablation experiments on subject #074 to
evaluate the contributions of the three core components

<!-- page 8 -->
Method
Novel View Synthesis
Self-Reenactment
PSNR↑
LPIPS↓
PSNR
LPIPS
Multi-view-O
30.31
0.082
24.56
0.102
Multi-view-M
30.43
0.091
24.60
0.114
Multi-view-T
31.17
0.088
24.63
0.104
Table 4: Quantitative results of momentum-guided expres-
sion representation and latent-guided texture attention under
multi-view settings.
Figure 7: Qualitative visualization of momentum-guided ex-
pression representation and latent-guided texture attention
under multi-view settings.
of our method: Expression-Aware Dynamic Correction,
Momentum-Guided Expression Representation, and Latent-
Guided Texture Attention. Each module is evaluated on
novel view synthesis and self-reenactment tasks to assess
its individual contribution. SSIM results are included in the
supplementary due to space limitations.
Expression-Aware Dynamic Correction. The 2D image
features provide essential cues to guide the correction of the
FLAME mesh geometry during rendering. However, since
FLAME expression coefficients govern the entire head ge-
ometry, the limited information from a single image is of-
ten insufficient to achieve accurate expression correction. As
shown in Table 3, using multi-view input for dynamic cor-
rection achieves the best performance, with PSNR improved
by 2.82 and LPIPS reduced by 0.039 on novel view syn-
thesis, and PSNR improved by 0.21 and LPIPS reduced by
0.032 on self-reenactment compared to the Freeze baseline.
In contrast, single-view input results in a noticeable drop,
though it still outperforms the Freeze setting, where FLAME
expression parameters remain fixed during rendering.
Momentum-Guided Expression Representation. The ab-
lation on Expression-Aware Dynamic Correction verifies the
effectiveness of multi-view inputs for expression refinement.
However, independently fitted FLAME meshes across views
per frame cause inconsistent 3D geometry. To address this,
MoGaFace introduces Momentum-Guided Expression Rep-
resentation for temporal and cross-view consistency.
On top of multi-view dynamic correction (Multi-view-
O), we integrate the Momentum-Guided Expression Repre-
sentation, referred to as Multi-view-M, to enforce a shared
FLAME mesh across views at each frame. As shown in Ta-
ble 4, this yields a PSNR gain of 0.12 in novel view synthesis
and 0.04 in self-reenactment, while LPIPS slightly increases
due to momentum averaging, leading to minor detail loss
(e.g., wrinkles, teeth), as shown in Fig. 7.
Latent-Guided Texture Attention. To improve mesh tex-
ture quality, MoGaFace employs The Latent-Guided Texture
Attention module to modulate Gaussian properties. To val-
idate its effectiveness, this module is added to the Multi-
view-Only setting (Multi-view-O), forming Multi-view-T
for fine-grained texture enhancement. As shown in Tab. 4,
this module improves PSNR by 0.86 in novel view synthesis
and 0.07 in self-reenactment. Fig.7 further illustrates qual-
itative improvements in fine details such as teeth and eye
wrinkles.
Conclusion
In this work, we present MoGaFace, a novel 3D head
avatar modeling framework that enables high-fidelity and
expressive avatar reconstruction under both camera-aware
and camera-free conditions. Unlike prior methods that rely
on fixed mesh estimation and suffer from cross-view in-
consistencies, MoGaFace integrates multi-view image cues
throughout the rendering process to dynamically refine
facial expressions and textures. We introduce three key
components to address the limitations of prior 3DGS-
based avatar systems: (1) Expression-Aware Dynamic Cor-
rection, which leverages multi-view information to im-
prove the alignment between 2D image and 3D mesh dur-
ing rendering; (2) Momentum-Guided Expression Repre-
sentation, which ensures temporal and cross-view consis-
tency of FLAME expression parameters by maintaining
a momentum-updated expression bank; and (3) Latent-
Guided Texture Attention, which adaptively incorporates
view-aware features to enhance fine-grained detail synthe-
sis. Extensive experiments under both standard and uncali-
brated settings demonstrate that MoGaFace achieves supe-
rior performance in terms of geometric consistency, percep-
tual quality, and generalization to unseen views and expres-
sions. Moreover, its ability to operate without ground-truth
camera parameters significantly enhances its applicability
to mobile and in-the-wild scenarios. We believe MoGaFace
offers a promising step toward robust and expressive head
avatar generation in real-world applications.
References
Aneja, S.; Sevastopolsky, A.; Kirschstein, T.; Thies, J.; Dai,
A.; and Nießner, M. 2024. Gaussianspeech: Audio-driven
gaussian avatars. arXiv preprint arXiv:2411.18675.
Arampatzakis, V.; Pavlidis, G.; Mitianoudis, N.; and Papa-
markos, N. 2023. Monocular depth estimation: A thorough
review. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 46(4): 2396–2414.
Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.;
Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf:
A multiscale representation for anti-aliasing neural radiance

<!-- page 9 -->
fields. In Proceedings of the IEEE/CVF international con-
ference on computer vision, 5855–5864.
Beeler, T.; Bickel, B.; Beardsley, P.; Sumner, B.; and Gross,
M. 2010. High-quality single-shot capture of facial geome-
try. In ACM SIGGRAPH 2010 papers, 1–9.
Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Ten-
sorf: Tensorial radiance fields. In European conference on
computer vision, 333–350. Springer.
Chen, Y.; Wang, L.; Li, Q.; Xiao, H.; Zhang, S.; Yao, H.; and
Liu, Y. 2024a.
Monogaussianavatar: Monocular gaussian
point-based head avatar. In ACM SIGGRAPH 2024 Confer-
ence Papers, 1–9.
Chen, Y.; Xu, H.; Zheng, C.; Zhuang, B.; Pollefeys, M.;
Geiger, A.; Cham, T.-J.; and Cai, J. 2024b. Mvsplat: Ef-
ficient 3d gaussian splatting from sparse multi-view im-
ages. In European Conference on Computer Vision, 370–
386. Springer.
Fei, B.; Xu, J.; Zhang, R.; Zhou, Q.; Yang, W.; and He, Y.
2024. 3d gaussian splatting as new era: A survey. IEEE
Transactions on Visualization and Computer Graphics.
Feng, Y.; Feng, H.; Black, M. J.; and Bolkart, T. 2021.
Learning an animatable detailed 3D face model from in-the-
wild images. ACM Transactions on Graphics (ToG), 40(4):
1–13.
Gao, X.; Zhong, C.; Xiang, J.; Hong, Y.; Guo, Y.; and
Zhang, J. 2022. Reconstructing personalized semantic fa-
cial nerf models from monocular video. ACM Transactions
on Graphics (TOG), 41(6): 1–12.
He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020.
Momentum contrast for unsupervised visual representation
learning. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 9729–9738.
Hore, A.; and Ziou, D. 2010. Image quality metrics: PSNR
vs. SSIM. In 2010 20th international conference on pattern
recognition, 2366–2369. IEEE.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.
Kirschstein, T.; Qian, S.; Giebenhain, S.; Walter, T.; and
Nießner, M. 2023. Nersemble: Multi-view radiance field re-
construction of human heads. ACM Transactions on Graph-
ics (TOG), 42(4): 1–14.
Kwon, Y.; Fang, B.; Lu, Y.; Dong, H.; Zhang, C.; Carrasco,
F. V.; Mosella-Montoro, A.; Xu, J.; Takagi, S.; Kim, D.; et al.
2024. Generalizable human gaussians for sparse view syn-
thesis. In European Conference on Computer Vision, 451–
468. Springer.
Li, J.; Zhang, J.; Bai, X.; Zheng, J.; Ning, X.; Zhou, J.; and
Gu, L. 2024. Talkinggaussian: Structure-persistent 3d talk-
ing head synthesis via gaussian splatting. In European Con-
ference on Computer Vision, 127–145. Springer.
Li, T.; Bolkart, T.; Black, M. J.; Li, H.; and Romero, J. 2017.
Learning a model of facial shape and expression from 4D
scans. ACM Trans. Graph., 36(6): 194–1.
Liu, H.; Liu, B.; Hu, Q.; Du, P.; Li, J.; Bao, Y.; and Wang, F.
2025a. A review on 3D Gaussian splatting for sparse view
reconstruction. Artificial Intelligence Review, 58(7): 215.
Liu, T.; Wang, G.; Hu, S.; Shen, L.; Ye, X.; Zang, Y.; Cao, Z.;
Li, W.; and Liu, Z. 2024a. Mvsgaussian: Fast generalizable
gaussian splatting reconstruction from multi-view stereo. In
European Conference on Computer Vision, 37–53. Springer.
Liu, Y.; Luo, C.; Fan, L.; Wang, N.; Peng, J.; and Zhang,
Z. 2024b. Citygaussian: Real-time high-quality large-scale
scene rendering with gaussians. In European Conference on
Computer Vision, 265–282. Springer.
Liu, Y.; Xu, S.; Guo, J.; Wang, D.; Wang, Z.; Tan, X.;
and Liu, X. 2025b. SyncAnimation: A Real-Time End-to-
End Framework for Audio-Driven Human Pose and Talking
Head Animation. arXiv preprint arXiv:2501.14646.
Lombardi, S.; Simon, T.; Schwartz, G.; Zollhoefer, M.;
Sheikh, Y.; and Saragih, J. 2021.
Mixture of volumetric
primitives for efficient neural rendering. ACM Transactions
on Graphics (ToG), 40(4): 1–13.
Ma, S.; Weng, Y.; Shao, T.; and Zhou, K. 2024. 3d gaussian
blendshapes for head avatar animation. In ACM SIGGRAPH
2024 Conference Papers, 1–10.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2021.
Nerf: Representing
scenes as neural radiance fields for view synthesis. Com-
munications of the ACM, 65(1): 99–106.
M¨uller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-
stant neural graphics primitives with a multiresolution hash
encoding. ACM transactions on graphics (TOG), 41(4): 1–
15.
Peng, Z.; Hu, W.; Shi, Y.; Zhu, X.; Zhang, X.; Zhao, H.; He,
J.; Liu, H.; and Fan, Z. 2024. Synctalk: The devil is in the
synchronization for talking head synthesis. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, 666–676.
Qian, S.; Kirschstein, T.; Schoneveld, L.; Davoli, D.;
Giebenhain, S.; and Nießner, M. 2024.
Gaussianavatars:
Photorealistic head avatars with rigged 3d gaussians. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 20299–20309.
Qiu, L.; Zhu, S.; Zuo, Q.; Gu, X.; Dong, Y.; Zhang, J.; Xu,
C.; Li, Z.; Yuan, W.; Bo, L.; et al. 2025. Anigs: Animatable
gaussian avatar from a single image with inconsistent gaus-
sian reconstruction. In Proceedings of the Computer Vision
and Pattern Recognition Conference, 21148–21158.
Tang, Z. J.; and Cham, T.-J. 2024. 3igs: Factorised tensorial
illumination for 3d gaussian splatting. In European Confer-
ence on Computer Vision, 143–159. Springer.
Thies, J.; Zollhofer, M.; Stamminger, M.; Theobalt, C.; and
Nießner, M. 2016. Face2face: Real-time face capture and
reenactment of rgb videos. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, 2387–
2395.
Wang, J.; Xie, J.-C.; Li, X.; Xu, F.; Pun, C.-M.; and Gao,
H. 2025a.
Gaussianhead: High-fidelity head avatars with
learnable gaussian derivation. IEEE Transactions on Visual-
ization and Computer Graphics.

<!-- page 10 -->
Wang, Y.; Wang, X.; Yi, R.; Fan, Y.; Hu, J.; Zhu, J.; and Ma,
L. 2025b. 3D Gaussian Head Avatars with Expressive Dy-
namic Appearances by Compact Tensorial Representations.
In Proceedings of the Computer Vision and Pattern Recog-
nition Conference, 21117–21126.
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P.
2004.
Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image process-
ing, 13(4): 600–612.
Wei, Y.; Zhang, J.; Zhang, X.; Shao, L.; and Lu, S. 2025.
PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose
Co-Regularizations. arXiv preprint arXiv:2507.13891.
Wu, Z.; Xiong, Y.; Yu, S. X.; and Lin, D. 2018. Unsuper-
vised feature learning via non-parametric instance discrimi-
nation. In Proceedings of the IEEE conference on computer
vision and pattern recognition, 3733–3742.
Xie, T.; Chen, X.; Xu, Z.; Xie, Y.; Jin, Y.; Shen, Y.; Peng,
S.; Bao, H.; and Zhou, X. 2025. Envgs: Modeling view-
dependent appearance with environment gaussian. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, 5742–5751.
Xu, Y.; Chen, B.; Li, Z.; Zhang, H.; Wang, L.; Zheng,
Z.; and Liu, Y. 2024.
Gaussian head avatar: Ultra high-
fidelity head avatar via dynamic gaussians. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, 1931–1941.
Xu, Y.; Zhang, H.; Wang, L.; Zhao, X.; Huang, H.; Qi, G.;
and Liu, Y. 2023. Latentavatar: Learning latent expression
code for expressive neural head avatar. In ACM SIGGRAPH
2023 Conference Proceedings, 1–10.
Yu, H.; Qu, Z.; Yu, Q.; Chen, J.; Jiang, Z.; Chen, Z.;
Zhang, S.; Xu, J.; Wu, F.; Lv, C.; et al. 2024.
Gaus-
siantalker: Speaker-specific talking head synthesis via 3d
gaussian splatting. In Proceedings of the 32nd ACM Inter-
national Conference on Multimedia, 3548–3557.
Zhang, R.; Chen, J.; and Wang, Q. 2023. Explicifying neural
implicit fields for efficient dynamic human avatar modeling
via a neural explicit surface. In Proceedings of the 31st ACM
International Conference on Multimedia, 1955–1963.
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In Proceedings of the IEEE conference
on computer vision and pattern recognition, 586–595.
Zheng, S.; Zhou, B.; Shao, R.; Liu, B.; Zhang, S.; Nie, L.;
and Liu, Y. 2024. Gps-gaussian: Generalizable pixel-wise 3d
gaussian splatting for real-time human novel view synthesis.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 19680–19690.
Zielonka, W.; Bolkart, T.; and Thies, J. 2022. Towards met-
rical reconstruction of human faces. In European conference
on computer vision, 250–269. Springer.
