<!-- page 1 -->
NeAR: Coupled Neural Asset–Renderer Stack
Hong Li1,2∗
Chongjie Ye3,11∗
Houyuan Chen4
Weiqing Xiao5
Ziyang Yan6
Lixing Xiao7
Zhaoxi Chen8
Jianfeng Xiang9
Shaocong Xu2
Xuhui Liu1
Yikai Wang10
Baochang Zhang1†
Xiaoguang Han11,3
Jiaolong Yang
Hao Zhao12†
1BUAA
2BAAI
3FNii, CUHKSZ
4HKUST
5NJU
6UniTn
7ZJU
8NTU
9THU
10BNU
11SSE, CUHKSZ
12AIR, THU
Project Page: near-project.github.io
Single Image
2D Intrinsic
Decomposition
Single Image & Mesh
Texture
Generation
2D Intrinsic
Decomposition
Voxelize & 
3D Generation
Relit Image
LH-SLAT (Neural Assets)
Neural Renderer
Novel Light
Novel View
(a) RGB       X
(b) Diffusion Renderer
(c) Hunyuan3D 2.1
(d) Ours (NEAR)
Physical Renderer
Novel Light
Novel View
PBR Materials
Metallic
roughness
Base Color
Albedo
Roughness
Normal
Irradiance
Neural Renderer
Novel Light
Novel View
Base Color
Metallic
Roughness
Normal
Neural Renderer
Novel Light
Novel View
Depth
Input
Intermediate Representation & Rendering Paradigm
Output
Figure 1. Overview of NeAR vs. Existing Single Image Relighting Frameworks. (a-b) Existing 2D methods lack explicit 3D awareness;
specifically, (a) struggles to disentangle specular highlights, while both fail to guarantee multi-view consistency during relighting. (c) State-
of-the-art 3D generation methods decouple asset authoring from rendering, relying on ill-posed PBR decomposition that often results in
material inaccuracies and baked-in artifacts. In contrast, (d) NeAR (Ours) employs a Coupled Neural Asset–Renderer Stack. By
utilizing the LH-SLAT representation, we simultaneously achieve photorealistic relighting and consistent novel-view synthesis.
Abstract
Neural asset authoring and neural rendering have tradi-
tionally evolved as disjoint paradigms: one generates digi-
tal assets for fixed graphics pipelines, while the other maps
conventional assets to images. However, treating them as
independent entities limits the potential for end-to-end opti-
mization in fidelity and consistency. In this paper, we bridge
this gap with NeAR, a Coupled Neural Asset–Renderer
Stack. We argue that co-designing the asset representation
and the renderer creates a robust ”contract” for superior
generation. On the asset side, we introduce the Lighting-
Homogenized SLAT (LH-SLAT). Leveraging a rectified-
flow model, NeAR lifts casually lit single images into a
canonical, illumination-invariant latent space, effectively
suppressing baked-in shadows and highlights. On the ren-
derer side, we design a lighting-aware neural decoder tai-
lored to interpret these homogenized latents. Conditioned
on HDR environment maps and camera views, it synthesizes
*Equal contribution. †Corresponding authors.
relightable 3D Gaussian splats in real-time without per-
object optimization. We validate NeAR on four tasks: (1) G-
buffer-based forward rendering, (2) random-lit reconstruc-
tion, (3) unknown-lit relighting, and (4) novel-view relight-
ing. Extensive experiments demonstrate that our coupled
stack outperforms state-of-the-art baselines in both quanti-
tative metrics and perceptual quality. We hope this coupled
asset-renderer perspective inspires future graphics stacks
that view neural assets and renderers as co-designed com-
ponents instead of independent entities.
1. Introduction
Images are determined by the interaction of light with
scene geometry, materials, and lighting. Classical computer
graphics separates this process into asset authoring, where
artists define scene properties, and rendering, where a phys-
ically based renderer simulates light transport. While effec-
tive, this separation requires substantial manual effort, com-
putationally expensive simulations, and makes inverse re-
construction from real-world images or video challenging.
arXiv:2511.18600v3  [cs.CV]  29 Mar 2026

<!-- page 2 -->
Ours
Trellis
GT
Input Image
Light 
Condition
HY3D-2.1
25
27
29
31
33
Forward Rendering
Diff. Renderer
Ours
22
23
24
25
26
Reconstruction
DiLightNet
Ours
21
21.5
22
22.5
23
Relighting
Diff. Renderer
Ours
21
21.5
22
22.5
23
23.5
24
Novel View
Hy3D-2.1
Ours
Ours 
wo/Neural Render
+ 4.4 dB
+ 1.23 dB
+ 0.48 dB
+ 0.68 dB
Metric (PSNR) 
Figure 2. Comparison of NeAR and Decoupled Paradigms. Left: Visual results under target illumination. Cols. 3–5 are rendered via
Blender to evaluate asset quality. Insets (right of cols. 4&5) display PBR maps (top-down: Base Color, Metallic, Roughness). Baselines
suffer from baked-in lighting (Trellis) or material ambiguity (HY3D-2.1). Notably, HY3D-2.1 wrongly assigns high metallic values to the
bread (see Metallic map, Row 1) and exhibits inconsistent highlights on the robot (Row 3). While our intermediate PBR decomposition
(col. 5) corrects materials, it struggles with complex effects like transparency (Helmet, Row 2) under standard rendering. Our full Neural
Renderer (col. 6) resolves this, yielding photorealistic results closest to GT. Right: Quantitative results on the Glossy Synthetic dataset.
NeAR achieves the highest PSNR across all four tasks, demonstrating the superiority of our coupled stack.
Recent advances in neural graphics [3, 6, 16, 20, 28, 44, 45,
48, 52, 54, 55, 63] address these limitations from two com-
plementary directions: neural asset authoring uses genera-
tive models [3, 5, 6, 20, 44, 45, 48, 55, 63] to synthesize full
3D assets for traditional pipelines, reducing manual effort,
while neural renderers map these assets—often converted
into intermediate representations such as depth, normals, or
shading buffers—directly to images [23, 52, 54], providing
a data-driven alternative to analytic rendering and enabling
more robust inverse inference. Fig. 1 shows a comparison
between our method and previous single-image relighting
frameworks.
Despite recent progress in generating 3D assets with
PBR materials [3, 5, 6, 20, 44, 45, 48, 55, 63], a fundamen-
tal limitation remains: asset generation and neural rendering
are typically developed in isolation, with assets created as-
suming a fixed renderer and renderers trained on static asset
distributions. This separation becomes problematic when
errors in asset decomposition—such as misidentified albedo
or incorrect normal maps—propagate through the rendering
pipeline. Because rendering is a nonlinear process, small er-
rors in asset decomposition compound into visible artifacts
like baked-in shadows or lighting inconsistencies. Fig. 2
demonstrates this issue: existing methods rendered with tra-
ditional physically-based renderers (e.g., Blender) exhibit
lighting artifacts and fail to achieve faithful relighting.
To this end, we propose NeAR, a Coupled Neural Asset–
Renderer Stack for single-image relightable 3D genera-
tion. Our key insight is to co-design the asset represen-
tation and rendering process to enable relighting directly
through a shared, lighting-homogenized latent space. On
the asset side, we introduce a Lighting-Homogenized Struc-
tured 3D Latent (LH-SLAT). Unlike standard assets that rely
on fragile explicit decomposition, our model lifts the ca-
sually lit input into a canonical latent form.
As visual-
ized in Fig. 3, this process transforms a shadow-affected
representation (Shaded-SLAT) into a clean, homogenized
state, effectively suppressing baked-in shadows and unsta-
ble highlights while preserving geometric cues. On the ren-
derer side, we design a lighting-aware neural renderer.
Conditioned on a lighting tokenizer, this renderer learns
to interpret the homogenized latents and synthesize view-
dependent appearance under arbitrary HDR environments
via differentiable 3D Gaussian splatting. By unifying the
representation, NeAR generates assets that naturally sup-
port real-time, high-quality relighting and novel-view syn-
thesis with consistent materials across views.
We validate NeAR across four downstream tasks: (1)
G-buffer–based forward rendering, (2) random-lit single-
image reconstruction, (3) unknown-lit single-image relight-
ing, and (4) novel-view relighting. On benchmarks includ-
ing Digital Twin Category, Aria Digital Twin, and Obja-
verse, NeAR achieves state-of-the-art or improved perfor-
mance over recent neural relighting baselines in both quan-
titative metrics and perceptual quality, while running at real-
time frame rates without per-object optimization.
Our contributions can be summarized as follows:
1. Coupled neural asset–renderer stack. We introduce
NeAR, an learnable graphics stack where the neural as-
set representation and neural renderer are co-designed
for single-image relightable 3D asset generation.
2. Lighting-homogenized structured neural asset. We

<!-- page 3 -->
Render
Shadow
AO
Base Color
Random light
Normalized Light
Relighting
LH-SLAT
Shaded-SLAT
GT
Figure 3. Lighting homogenization as the bridge between as-
sets and renderer. We visualize the intrinsic components (Base
Color, Ambient Occlusion), rendering results under random and
uniform lighting, shadow maps, as well as relighting outputs gen-
erated respectively by Shaded SLAT and LH-SLAT. By mapping
casually lit images to a canonical illumination space, LH-SLAT ef-
fectively suppresses baked-in shadows and unstable specularities
while preserving geometry-consistent diffuse cues. This stable la-
tent space serves as the robust ”contract” for our lighting-aware
neural renderer to enable controllable relighting.
propose a Lighting-Homogenized Structured 3D Latent
(LH-SLAT) that suppresses shadows and unstable high-
lights while preserving geometry-consistent diffuse cues
in a compact, view-agnostic 3D latent.
3. Lighting tokenizer and lighting-aware neural 3D
Gaussian renderer. We design a lighting tokenizer and
a lighting-aware neural 3D Gaussian renderer that map
LH-SLAT, environment illumination, and view embed-
dings into a relightable 3D Gaussian field rendered via
differentiable Gaussian splatting.
4. Extensive evaluation and real-time performance. We
demonstrate on multiple datasets and tasks that NeAR
delivers state-of-the-art or better quality with strong gen-
eralization and consistent multi-view rendering, while
enabling real-time feed-forward inference.
2. Related Works
2.1. Image relighting and inverse rendering
Image relighting and inverse rendering lie at the inter-
section of geometry, material estimation, and light trans-
port, and have been studied from both physics- and data-
driven perspectives [16]. Classical methods (e.g., SIRFS)
recover interpretable PBR maps (albedo, roughness, nor-
mals) via optimization with hand-crafted priors [1]. While
interpretable and editable, these approaches are highly ill-
posed in real scenes: shadows, inter-reflections, and view-
dependent highlights bias material estimation, leading to
baked-in artifacts under re-rendering.
Recent learning-based approaches fall into two cate-
gories. The first focuses on physically structured decompo-
sition [10, 23, 63], which yields interpretable assets but of-
ten requires multi-view data or costly per-object optimiza-
tion to resolve ambiguities. The second targets diffusion-
based 2D relighting [11, 28, 52, 56]. Methods such as Di-
LightNet and IC-Light leverage diffusion priors for high-
fidelity relighting with fine control, but are computationally
expensive, stochastic, and limited to 2D, lacking multi-view
consistency for 3D applications.
We take a middle path: rather than brittle PBR inversion
or black-box diffusion, we homogenize illumination into a
canonical form (LH-SLAT) and synthesize a relightable 3D
field feed-forward, improving stability and controllability.
2.2. Generative 3D Priors and Representations
Diffusion priors and score-distillation sampling (SDS) have
catalyzed rapid progress in text-to-3D and image-to-3D
generation [34, 38, 40, 46, 59–61]. While SDS-based meth-
ods transfer 2D generative knowledge to 3D effectively,
they suffer from slow iterative optimization. Consequently,
recent works have shifted toward feed-forward 3D recon-
struction models trained on large-scale 3D datasets [14, 45,
55]. Specifically, Trellis [45] utilizes Structured 3D Latents
(SLAT) to compress complex geometry and appearance into
sparse tokens, enabling efficient decoding.
Concurrently, 3D Gaussian Splatting (3DGS) [18] has
emerged as a rasterization-friendly representation support-
ing real-time differentiable rendering. While current feed-
forward models (like LRM or Trellis) excel at geometry,
they typically bake lighting into the texture, limiting down-
stream utility. Our method builds upon the efficiency of
SLAT and 3DGS but fundamentally redesigns the genera-
tion process. We introduce a lighting-homogenized variant
of SLAT and a custom neural decoder, replacing static tex-
ture prediction with a relightable neural field.
2.3. Relightable 3D asset synthesis
Producing relightable 3D assets requires models to repre-
sent both intrinsic surface properties and lighting-dependent
transport (shadows, speculars, interreflections). Prior works
condition NeRFs, Gaussian splats or meshes on lighting in-
puts to enable relighting-aware outputs [2, 12, 16, 21, 33,
36, 47, 51, 62]. Many approaches either use volumetric neu-
ral renderers that are costly at inference, or attempt to esti-
mate PBR maps without lighting supervision, which leads
to poor disentanglement [26, 35, 39].
Some models ex-
plore large inverse-rendering architectures to predict PBR
properties from sparse views, but computational cost and
optimization per-object remain bottlenecks [22, 58]. Re-
cent works [10, 41] employ diffusion models to generate
multi-view material maps or multi-view relighted images,
followed by 3D reconstruction. However, the absence of
explicit 3D constraints in the generation stage makes it dif-
ficult to guarantee consistency across views.
In contrast, our homogenize-then-synthesize strategy
pipeline explicitly removes unstable, scene-specific illumi-
nation before decoding. This mitigates ill-posed PBR in-

<!-- page 4 -->
version, enabling a feed-forward decoder to produce re-
lightable 3DGS with real-time consistency.
NeAR thus
combines the stability of interpretable pipelines with the fi-
delity of neural rendering.
3. Method
3.1. Preliminary
3D Gaussian Splatting (3DGS). 3DGS [18] represents
scenes with anisotropic Gaussians, rendered via splatting
and α-blending: C = P
i∈N ciαi
Q
j<i(1−αj). Crucially,
standard 3DGS models color ci using Spherical Harmonics
(SH), which inherently bakes static lighting into the repre-
sentation. To enable relighting, we forego SH and predict
color dynamically conditioned on target illumination.
Structured 3D Latents (SLAT). Following Trellis [45],
we use SLAT to encode 3D assets efficiently.
A SLAT
Z = {(zk, pk)}K
k=1 consists of K active feature tokens,
where each token zk ∈RD is associated with a coordinate
pk in a sparse voxel grid. This representation focuses ca-
pacity on surface regions (K ≪N 3) and supports diverse
decoding heads. However, standard SLATs blindly encode
input appearance—including shadows and highlights. Our
goal is to transform Z into a lighting-homogenized form,
canonicalizing the appearance to a uniform illumination
while preserving geometry.
3.2. Overview of NeAR
The challenge in single-image 3D relighting lies in dis-
entangling lighting from intrinsic object properties, since
shadows, highlights, and interreflections are inherently en-
tangled with geometry. To avoid unstable PBR inversion
and black-box neural generation, we propose a homogenize-
then-synthesize framework that functions as a coupled
stack. NeAR first extracts a Lighting-Homogenized SLAT
(LH-SLAT) from the input image to neutralize lighting ef-
fects, then decodes a relightable 3DGS. Our framework
consists of two stages:
Stage 1: Light Homogenization-SLAT Generation.
We first utilize the pre-trained flow model fs to map the
arbitrarily lit input Iin into an initial shaded SLAT Zs. Op-
erating within this sparse voxel space, we employ a LoRA-
adapted model fθ to steer the latent representation from Zs
toward a Lighting-Homogenized SLAT (LH-SLAT) Zlh:
  Z _ {\text  {lh } } = f_{\the ta }(Z_s, I_\text {in}) = f_{\theta }(f_{s}(I_{\text {in}}), I_\text {in}). \label {eq:stage1} 
(1)
Specifically, Zlh suppresses the baked-in shadows and
highlights inherent in Zs, establishing a stable light-
homogenized space. This representation preserves essential
geometry-material-light interactions, yielding a unified and
generalizable foundation for the relighting task.
Stage 2: Relightable Neural 3DGS Synthesis. Lever-
aging the homogenized representation Zlh, a feed-forward
decoder D synthesizes a relightable Gaussian field G. This
process is conditioned on the target view vtarget and the tar-
get illumination Ltarget, encoded via El:
  \
m
athc al {G} =  \mathcal {
D
}\big (Z_{\text {lh}}, \mathbf {v}_{\text {target}}, \mathcal {E}_l(L_{\text {target}})\big ). \label {eq:stage2} 
(2)
Finally, the relighted image is rendered using a differen-
tiable GS rasterizer M:
  I_{\t e xt { target}} = \mathcal {M}(\mathcal {G}, \mathbf {v}_{\text {target}}). \label {eq:render} 
(3)
In the following, we describe Stage 1 (Sec. 3.3) and
Stage 2 (Sec. 3.5) in detail.
3.3. Light Homogenization & LH-SLAT Rec.
The first stage generates a Lighting-Homogenized Struc-
tured 3D Latent (LH-SLAT) Zlh from a single input im-
age Iin. This representation serves as a stable, illumination-
invariant substrate for downstream synthesis.
Lighting Homogenization. We define the homogenized
light Eh as a uniform, white ambient environment illumina-
tion. Extracting SLAT features under such lighting captures
intrinsic geometric and material cues uncorrupted by tran-
sient lighting effects, serving as a robust basis for relighting.
LH-SLAT Reconstruction.
To train fθ, we prepare
paired data (Iin, Zlh) via multi-step rendering of 3D assets
under homogenized lighting. As shown in Fig. 4 top left
corner, we first generate the ground-truth homogenized la-
tents Zlh: (1) for each 3D asset, we render N views under
our defined homogenized illumination; (2) we extract dense
2D visual features using a pre-trained DINOv2 model; (3)
these features are back-projected into a sparse 3D voxel
grid; (4) finally, this sparse grid is compressed by a pre-
trained SLAT VAE encoder to obtain Zlh. Second, to create
the corresponding input Iin, we render M additional images
of the same asset under diverse, random lighting conditions
and camera poses.
Optionally, for highly reflective materials, we extract
Basecolor SLAT Zbc from multi-view basecolor renderings,
concatenating with Zlh to retain base color information.
3.4. LH-SLAT Generation
As shown in Fig. 4 top right corner, we use a rectified flow
model fθ to generate the lighting-homogenized SLAT Zlh
from the input image Iin. The rectified flow model is trained
to learn the mapping between the arbitrarily lit image and
the corresponding latent representation under our homog-
enized lighting conditions. Specifically, we utilize a pre-
trained SLAT rectified flow model fs to generate the shad-
owed SLAT Zs from the input image Iin, and subsequently
fine-tune fs using LoRA [15] in the sparse voxel space [45]
to achieve lighting homogenization. The loss function for
training is the conditional flow matching loss Lstage1:
  \math c al {L}_{stage 1} =\ma th b b { E}_{t
,\boldsymbol {z}_0,\boldsymbol {\epsilon }}\|\boldsymbol {v}_\theta (\boldsymbol {z}, Z_s, I_{in}, t)-(\boldsymbol {\epsilon }-\boldsymbol {z}_0)\|^2_2, \label {eq:lora} 
(4)

<!-- page 5 -->
Relightable Neural 3D Gaussian Synthesis (Stage 2)
Lighting-
dependent 
feature 𝒉𝒆
Intrinsic  
Aware 
Decoder 
Input
Lighting 
Tokenizer
𝑬
Lighting-
dependent
feature 𝒉𝒗
Lighting
Aware 
Decoder 𝓓𝑬
𝓓𝑰
𝓔𝒍
offset 𝒐
basecolor 𝒃
roughness 𝜸
metallic 𝒎
scale 𝒔
alpha 𝜶
rotate 𝒓
feature 𝒇
shadow 𝝈
scale 𝒔(
View Emb. 𝒆𝒗
Intrinsic Feat. 𝒉
𝒁𝒍𝒉
𝓜GS Rasterizer
Channel-wise Concat
Voxel-wise Add
Trainable Weights
Pretrained Weights
𝓓
Neural 3D Gaussian Splatting (Sec 3.5.3)
Linear
Relu
Linear
Elu
{pos.,𝒐, 𝜎, 𝑐, 𝒔+, 𝛼, 𝒓}
{pos.,𝒐, 𝑏, 𝛾, 𝑚, 𝒔, 𝛼, 𝒓}
feature 𝒇
Pos.
Emb.
Color 𝒄
𝓜
𝐼!"#$%!
&'#
, 𝐼(
𝐼), 𝐼*, 𝐼+
×𝟑
3DGS 𝓖
Light Homogenization-SLAT Generation (Stage 1)
Sparse
Flow
Transformer
Image  𝑰𝐢𝐧
Cond.
𝒇𝒔
Input
𝒇𝜽
LORA
LH-SLAT
𝒇𝒔
Input
SLAT 𝒁𝒔
Normalized Light  𝑬𝒉
Light Homogenization & LH-SLAT Reconstruction (Sec 3.3)
Voxelize
Multiview
Average
LH-SLAT 𝒁𝒍𝒉
3D assets
Sparse
VAE
Encoder
ℰ
DINOv2
LH Structure Features 
𝑬𝒉
Mesh
normal 𝒏
scale 𝒔G
Vox.
Figure 4. Pipeline of NeAR as a coupled neural asset-renderer stack. Top (Inference Stage): An end-to-end inference pipeline. Given
a single image and a geometry prior (e.g., mesh from HY3D), Stage 1 utilizes a rectified-flow backbone with LoRA adaptation to predict
the Lighting-Homogenized SLAT (LH-SLAT). This latent acts as a bridge, which is then consumed by the Stage 2 lighting-aware neural
renderer to synthesize relightable 3DGS under novel illumination and viewpoints. Bottom-Left (Data Prep): Offline construction of
ground-truth LH-SLATs by rendering assets under homogenized illumination and encoding them via a sparse VAE. Bottom-Right (GS
Decoding & Rendering): Detailed architecture of the 3DGS decoding head, which predicts Gaussian attributes from lighting-dependent
features, followed by a differentiable rasterizer M that renders the final HDR image, shadow and PBR auxiliary maps.
𝑬𝒍𝒅𝒓
𝑬𝒍𝒐𝒈
𝑬𝒅𝒊𝒓
𝑬
ConvNext 
Backbone
Pos.
Emb.
×𝟑
Spatial
Cross MHA
Multi 
Scale
Self-MHA
Coord.
FFN
Lighting Cond. 𝑪𝑳𝓔𝒍
Linear
Pos.
Emb.
Linear
Self W-MHA
FFN
Pos.
Emb.
Linear
Linear
Input
Feature
Position
𝓓𝑰
×𝟏𝟐
Cross MHA
FFN
Intrinsic Feat. 𝒉
Register Tokens
Cross MHA
FFN
Pos.
Emb.
Linear
Linear
Input
𝒉+ 𝒆𝒗
Pos.
𝓓𝑬
Cross MHA
FFN
Lighting-dependent
feature 𝒉"
×𝟔
𝑪𝑳
KV
Lighting Tokenizer
Intrinsic  Aware Decoder 
Lighting Aware Decoder
Register Tokens
Figure 5. Architectures of Lighting Tokenizer, IAD, and LAD.
where z(t) = (1 −t)z0 + tϵ is the linear interpolation be-
tween the data sample z0 and noise ϵ, and vθ approximates
the time-dependent vector field. If the optional basecolor
SLAT zbc is used, it is concatenated with zlh to provide ad-
ditional color information to the subsequent stage.
3.5. Relightable Neural 3D Gaussian Synthesis
The second stage synthesizes a relightable 3D Gaussian
Splatting (3DGS) field G from LH-SLAT, conditioned on
target illumination and viewpoint. Unlike optimization ap-
proaches [2, 12], we employ an efficient feed-forward de-
coder with two sequential modules: the Intrinsic Aware De-
coder (IAD) and the Lighting Aware Decoder (LAD).
3.5.1. Intrinsic Aware Decoder (IAD)
The IAD, denoted as DI, aims to process LH-SLAT Zlh and
generate a view-independent and illumination-invariant in-
trinsic feature h = {(hi, pi)}L
i=1, where hi ∈R768. This
sparse feature field h effectively decodes the underlying ge-
ometric structure and material properties of the scene. To
achieve this, IAD employs a Transformer architecture akin
to TRELLIS [45], leveraging stacked self-shifted window
attention blocks to exploit the inherent locality of struc-
tured 3D latent sequences. To further enhance the model’s
comprehension of global structural relationships and light-
ing context, a register cross-attention layer is incorporated
into each block. Specifically, 16 learnable register tokens
are appended to each SLAT sequence to capture global con-
text and suppress high-frequency noise [7, 19]. These reg-
ister tokens are injected into the decoder via global cross-
attention, facilitating information exchange with all latent
variable tokens and enabling a coherent and globally con-
sistent intrinsic representation h.
3.5.2. Lighting Aware Decoder (LAD)
The LAD, denoted as DE, synthesizes the final lighting-
dependent features by injecting view embeddings and envi-
ronmental lighting conditions, as shown in Fig. 4.
Observe View Embedding. To explicitly model spec-
ular highlights that vary with viewing angles, we abandon
the commonly used spherical harmonics and instead inject
the observed view information into the learning process of
LAD from the outset to enhance the model’s perception of
specular highlights. Along the camera ray to each voxel

<!-- page 6 -->
Table 1. Quantitative comparison against state-of-the-art methods across four sub-tasks.
ADT [32]
DTC [9]
Objaverse data [8]
Glossy Synthetic dataset [24]
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
G-Buffers Forward Rendering
DiffusionRenderer [23]
0.0802
24.41
0.9172
0.0560
27.16
0.9354
0.0616
27.09
0.9288
0.0707
25.46
0.9126
Ours
0.0488
29.15
0.9484
0.0458
31.59
0.9586
0.0490
32.23
0.9627
0.0475
30.47
0.9594
Random-lit Single-image Reconstruction
RGB↔X [54]
0.1605
15.15
0.8445
0.1349
15.48
0.8624
0.1199
16.09
0.8801
0.1271
14.29
0.8612
DiLightNet [52]
0.0949
21.11
0.8947
0.0650
23.53
0.9147
0.0507
25.65
0.9300
0.0523
24.09
0.9213
DiffusionRenderer [23]
0.0767
22.50
0.9105
0.0579
23.70
0.9234
0.0516
24.81
0.9285
0.0547
23.40
0.9163
Ours
0.0754
22.89
0.9116
0.0532
24.68
0.9246
0.0394
26.53
0.9305
0.0368
25.32
0.9274
Unknown-lit Single-image Relighting
DiLightNet [52]
0.1037
20.59
0.8813
0.0729
22.63
0.8913
0.0657
23.87
0.9011
0.0622
22.40
0.9059
NeuralGrafferer [16]
0.2675
14.31
0.7839
0.2548
14.22
0.7943
0.2108
14.68
0.8238
0.1767
15.67
0.8200
DiffusionRenderer [23]
0.0916
21.91
0.8960
0.0691
22.99
0.9078
0.0609
23.75
0.9169
0.0632
22.13
0.9062
Ours
0.0915
21.95
0.8972
0.0642
23.47
0.9177
0.0557
24.38
0.9264
0.0465
22.61
0.9246
Novel-view Relighting
3DTopia-XL [5]
0.1754
17.24
0.8013
0.1051
21.56
0.8674
0.0769
23.22
0.8989
0.0857
20.89
0.8807
Stable-Fast-3D [3]
0.1028
19.43
0.8881
0.0616
22.07
0.9154
0.0666
22.26
0.9112
0.0747
20.17
0.8943
MeshGen [6]
0.0939
20.15
0.8879
0.0661
22.87
0.9101
0.0509
24.15
0.9306
0.0637
21.43
0.9071
Hunyuan3D-2.1 [63]
0.0727
22.30
0.9017
0.0481
24.89
0.9255
0.0479
25.47
0.9328
0.0533
22.26
0.9119
Ours
0.0693
22.87
0.9023
0.0475
25.53
0.9298
0.0486
25.97
0.9392
0.0502
22.94
0.9147
pi in the world coordinate system, we record the distance
x = {(li, pi)}L
i=1, where li ∈R, and the ray direction
dw = {(dwi, pi)}L
i=1. We then transform dw to the cam-
era coordinate system using the extrinsic matrix, denoted as
d = {(di, pi)}L
i=1, where di ∈R3. We apply NeRF posi-
tional encoding and learnable positional encoding to d and
l voxel-wise, respectively, obtaining the view embedding:
  \ bold sym b ol {
e } ^v =  \{
\b oldsy
mbol
 { e }^d,\boldsymbol {e}^l\} =\{(\boldsymbol {e}^{d}_i, \boldsymbol {p}_i),(\boldsymbol {e}^{l}_i, \boldsymbol {p}_i)\}_{i=1}^{L},\quad \boldsymbol {e}_i \in \mathbb {R}^{768}. 
Then, we add ed and el voxel-wise to h to obtain hv, which
serves as the input to LAD.
Lighting Tokenizer. We encode the high dynamic range
(HDR) environment map E into compact lighting condi-
tions using an HDRI encoder El. Following [13, 16, 23], we
decompose E into a tone-mapped LDR image Eldr, a nor-
malized log-intensity map Elog = log(E + 1)/Emax, and
a camera-space direction encoding Edir ∈RH×W ×3. Un-
like prior works that compress the entire map via VAE, we
employ a ConvNeXt backbone to extract multi-scale visual
features from Eldr and Elog. Crucially, rather than directly
compressing Edir, we first encode it via NeRF-style posi-
tional embedding [30] and fuse it with visual features using
Spatial Cross Attention. This mechanism acts as a learn-
able positional encoding, modulating visual features with
explicit directional cues. The resulting multi-scale features
are concatenated, processed with positional encoding, and
passed through self-attention blocks to yield the Lighting
Condition Tokens CL ∈R4096×768. This design explicitly
embeds directional information, facilitating editable light-
ing directions when switching views.
LAD Architecture.
LAD consists of stacked cross-
attention blocks that inject lighting condition CL into in-
trinsic features hv, enabling lighting awareness. Similar to
IAD, each block includes a register cross-attention layer to
enhance global illumination perception. The output is the
lighting-aware sparse feature he.
3.5.3. Neural 3D Gaussian Splatting
We regress the 3DGS parameters using both the intrinsic
feature h and the lighting-aware feature he:
  \l
a b el {e
q:3 d gs} \
b e gi
n  {a
l i gn
e d } 
& \ {(
\ b ol
d sym
bol {
h}^v
_i, 
\b oldsy
mbo l  {p} _
i ) \}_
{ i =1
} ^{L
} \ri
ghtarrow \{ \{(\boldsymbol {o}^k_i, \boldsymbol {b}^k_i, \gamma ^k_i, \boldsymbol {m}^k_i, \boldsymbol {s}^k_i, \alpha ^k_i, \boldsymbol {r}^k_i) \}_{k=1}^{K} \}_{i=1}^{L}, \\ &\{(\boldsymbol {h}^e_i,\boldsymbol {p}_i)\}_{i=1}^{L} \rightarrow \{ \{(\boldsymbol {f}^k_i, \hat {s}^k_i, \sigma ^k_i) \}_{k=1}^{K} \}_{i=1}^{L} \end {aligned} 
(5)
the intrinsic feature hi is decoded into K Gaussian param-
eters: position offset o, base color b, roughness γ, metal-
lic m, scale s, rotation r, and opacity α (activated via
tanh to support negative density [64]). Simultaneously, the
lighting-dependent feature he
i predicts the 48-dim color fea-
ture f, lighting-specific scale ˆs, and shadow σ. The Gaus-
sian centers are defined as xk
i = pi + tanh(ok
i ), with nor-
mals derived from the shortest axis of ˆs. Finally, we employ
a simple shallow MLP network that combines the positional
encoding of the normal vector and the color feature f. This
network uses ReLU activation functions in its intermedi-
ate layers and an ELU activation function in its final layer
to predict the radiance values for each Gaussian. Through
the rasterization operation M, we obtain the 2D HDR pre-
diction Ihdr
target. We also render 2D base color, roughness,
metallic, shadow images Ib, Ir, Im, Is.
Loss Function.
We supervise the training via
an HDR reconstruction loss Lhdr, which comprises L1,
LPIPS [57], D-SSIM and regularization terms.
Follow-
ing [53], to prevent high-intensity regions from dominat-

<!-- page 7 -->
Diff. Renderer
Ours
GT
G-buffer
Figure 6.
Visual comparison of Diffusion Renderer (with G-
buffer) and our LH-SLAT method for image relighting.
ing the L1 optimization, we apply a logarithmic transfor-
mation to the HDR images. For perceptual metrics (LPIPS
and D-SSIM), we operate on tone-mapped images using
clamp(log2(I), 0, 1). Additionally, we impose auxiliary L1
supervision on material properties maps (base color, rough-
ness, metallic), denoted as Lpbr, and shadows Lshadow. The
total objective is formulated as follows:
  \smal l  \ma t hcal {L} _ {stage2} = \mathcal {L}_{hdr} + \lambda _{pbr} \mathcal {L}_{pbr} + \lambda _{shadow} \mathcal {L}_{shadow}. \label {eq:loss} 
(6)
4. Experiments
4.1. Implementation Details
Please refer to the Supplementary Material for comprehen-
sive implementation details.
Training Data.
Our training dataset comprises 87K
3D assets with physically-based rendering (PBR) textures,
curated from the Objaverse-XL dataset.
These assets
are illuminated using 2K High Dynamic Range Images
(HDRIs), each at 4K resolution, used as environment maps.
We normalized the assets to fit within a bounding box
of [−0.5, 0.5].
The first training stage involves render-
ing 150 viewpoints under normalized lighting to extract
illumination-invariant structural latent representations. For
input images under unknown illumination, camera poses are
sampled with yaw within ±45 degrees and pitch from -10 to
45 degrees, oriented towards the object’s center, and with
field of view (FOV) and radius following [45]. Unknown il-
lumination is modeled with (1) six area lights uniformly dis-
tributed on a sphere, (2) 1-3 area lights randomly sampled
within the camera’s hemisphere, or (3) a random, Z-axis-
rotated environment map. Area light intensities are sampled
uniformly between 300 and 700 (units), distances between
5 and 8 units. In the second stage, we re-light objects using
randomly rotated environment maps as supervision, with a
Diff. Renderer
Ours
RGB     X
Input & GT
DiLightNet
Figure 7. Visual comparison of image reconstruction.
DiLightNet
Diff. Renderer
Input
Ours
Neural Graffer
GT Relit
Figure 8. Visual comparison of relighting.
fixed FOV of 40◦. We randomly and uniformly sample 12
camera viewpoints on a sphere of radius 2.0, where each
viewpoint is rendered under 16 different illumination con-
ditions. All data generation across both stages utilizes the
Blender EEVEE Next engine [42] with raytracing enabled.
Task Definitions And Baselines.
We evaluate our
method on two fundamental tasks: single-view forward
rendering and novel view relighting from single-image to
Relightable 3D. We evaluate the consistency between the
rendered outputs and the ground truth reference images.
The former involves single-view forward rendering with
input G-buffers (such as normals, material, and depth in-
formation), image reconstruction from a single-image un-
der random lighting, and relighting of a single image un-
der unknown lighting. For single-view forward rendering,
we compare against recent state-of-the-art neural render-
ing methods RGB↔X [54], neural-gaffer [16], DiLightNet
[52], and Diffusion-render [23]. For novel view relight-
ing, we compare against recent open-source methods that
support single-image to 3D generation with PBR materials,
including Huyuan3D-2.1 [63] (HY3D 2.1), MeshGen [6],
3DTopia-XL [5], and SF3D [3]. The schematic diagram for
the four subtasks is illustrated in Fig. 14. We additionally
present qualitative results for PBR material estimation in
comparison with HY3D 2.1.
Evaluation Metric.
We use PSNR, SSIM [43] and
LPIPS [57] to measure the quality of the rendering.
Evaluation Datasets. We construct a test set by ran-
domly selecting 800 unseen objects from our training data.
To validate generalization capability, we evaluate on out-of-
domain datasets: Aria Digital Twin (ADT) [32] and Digital
Twin Catalog (DTC) [9], which feature high-fidelity photo-
realistic models with sub-millimeter accuracy. We also in-

<!-- page 8 -->
Table 2. Ablation study on the number of blocks for DI and DE.
Num
PSNR
SSIM
LPIPS
DE Param.
FPS
12 + 1
31.56
0.9608
0.0508
12.65M
48
12 + 3
32.35
0.9635
0.0474
31.55M
38
12 + 6
32.54
0.9649
0.0442
59.8M
30
12 + 9
32.56
0.9645
0.0439
88.23M
23
0 + 18
29.43
0.9245
0.0624
173.25M
10
Table 3. Ablation study on decoder input SLAT types.
SLAT types
PSNR
SSIM
LPIPS
shaded
28.95
0.9281
0.0813
base color
30.38
0.9541
0.0564
LH
32.02
0.9631
0.0494
LH + base color
32.54
0.9649
0.0442
corporate the Glossy Synthetic dataset [24] and additional
assets from BlenderKit1, modifying rendering nodes to uti-
lize the Principled BSDF shader2.
4.2. Single-view Forward Rendering
G-buffers Forward Rendering. As shown in Fig. 6, we
compare against Diffusion Renderer using ground truth
G-buffers and LH-Slat , bypassing the single-image-to-
intermediate representation step. Our method demonstrates
superior accuracy in shadow and highlight distribution (e.g.,
the toy’s specular highlight and the sculpture’s shadow de-
tail), likely due to our explicit 3D structural information.
Furthermore, we accurately capture material reflections of
ambient light, as illustrated by the stainless steel. Quan-
titatively, our method significantly outperforms baselines
across four datasets in Tab. 1.
Random-lit Single-image Reconstruction. As shown
in Figs.7,16 and 17, our method achieves higher recon-
struction fidelity than the baselines, producing more visu-
ally consistent and geometrically faithful results. Specif-
ically, Diffusion Renderer and RGB-X misestimate mate-
rials, while DiLightNet exhibits color shifts. The quanti-
tative evaluations in Tab. 1 further confirm the superiority
of our approach: we achieve better performance across all
metrics, demonstrating improved accuracy and robustness
under varying conditions.
Unknown-lit Single-image Relighting.
Our method
achieves more accurate highlights and color in relit im-
ages with unknown lighting, compared to other methods,
as shown in Fig. 8, 18 and 19. For example, observe the
highlights on the speaker cones (first row) and the teapot
color (second row). Tab. 1 quantitatively demonstrates the
superiority of our method.
Novel-view Relighting. We benchmark our full pipeline
(single-image to relightable 3D) against state-of-the-art
generation methods. While other methods typically recon-
1https://www.blenderkit.com/
2https://www.blender.org/
Cross MHA
FFN
Input
𝒉+ 𝒆𝒗
Pos.
𝓓𝑬
Cross MHA
FFN
×𝟔
𝑪𝑳
KV
(c)
FFN
Input
𝒉
Pos.
𝓓𝑬
Cross MHA
×𝟔
𝑪𝑳
KV
(b)
𝒆𝒗
FFN
Cross MHA
···
···
(e)
(d)
𝒉
𝒆𝒗
𝒉𝒗
𝒉
𝒉𝒗
𝒉𝒆
SH
𝒉𝒆
f
𝒄
𝒄
MLP
(f)
(g)
𝒢
(a)
Cross MHA
FFN
Input
𝒉
Pos.
𝓓𝑬
Cross MHA
FFN
×𝟔
𝑪𝑳
KV
···
···
···
···
Figure 9. Different designs for the feedforward network D.
Table 4. Performance Comparison of Different Architectures.
Arch
PSNR
SSIM
LPIPS
a + e + f
29.82
0.9472
0.0642
a + e + g
30.66
0.9524
0.0515
a + d + g
31.96
0.9597
0.0492
b + d + g
32.43
0.9628
0.0472
c + d + g (ours)
32.54
0.9649
0.0442
Base Color
Metallic
Roughness
Input
Base Color
Metallic
Roughness
Ours
HY3D 2.1
GT
Figure 10. Visual comparison of PBR material estimation with
HY3D2.1 and our method.
struct a mesh and rely on Blender for relighting, we di-
rectly generate a relightable 3D Gaussian field. As shown
in Figs. 2 and 20, our method achieves more realistic light-
ing–material interactions than image-based textured mesh
methods [3, 5, 6, 63]. Quantitative results in Tab. 1 demon-
strate improvements over existing 3D generation baselines.
PBR Materials Estimation. Fig. 10 demonstrates that
our method surpasses the open-source SOTA model, HY3D
2.1, in material recovery. Hunyuan3D relies on multi-view
diffusion, which often introduces view-inconsistent arti-
facts (e.g., blurred edges on the wooden cup). In contrast,
our LH-SLAT preserves 3D consistency and retains cru-
cial light-material interaction cues. For instance, HY3D 2.1
misclassifies wood as metal, resulting in erroneous metallic
artifacts on the eggs, whereas our method correctly recovers

<!-- page 9 -->
Light
Style Image
Base Color
Metallic
Roughness
Shadow
Relit Render
Geometry
Figure 11. Texture Style Transfer. Given the geometry and a target-style image as guidance, our method can generate semantically
consistent stylized textures and support photorealistic neural relighting.
Figure 12. Relighting results from real-world single images.
the material properties.
4.3. Applications
Texture Style Transfer. As shown in Fig. 11, given the
target-style image in the second column and an arbitrary
geometry, we convert the geometry as coordinates, while
deriving the LH-SLAT from the stylized image using the
flow models fs and fθ. This enables semantically consis-
tent style transfer and material estimation. For instance, the
dragon’s mouth in the second row corresponds to the beak
style in the reference image, and the third-row horse met-
alness map matches the metallic appearance implied by the
target style. Moreover, benefiting from LH-SLAT, we can
produce photorealistic neural relighting in the fourth col-
umn under a given illumination condition, for example, the
green reflections from the tree and metallic highlights in the
third row.
Real-World Images. As shown in Fig. 12, to demon-
strate generalization to real-world data, our method suc-
cessfully removes baked-in lighting from both internet im-
ages (rows 1–2) and real mobile photos (row 3), enabling
effective relighting under novel views. It further exhibits
strong robustness in handling specular highlights and cap-
turing lighting directionality (rows 2–3).
4.4. Ablation Study.
We perform ablation studies on our test set, investigating
the Variants of D and input SLAT types.
Variants of D. Tab. 2 indicates that increasing the depth
of DE improves quality but reduces inference speed; we
therefore select 6 layers to strike a balance between effi-
ciency and performance. Relying solely on the LAD DE
leads to a significant decline in relighting performance, con-
sistent with [53]. Furthermore, Fig. 9 demonstrates that in-
jecting camera view information to identify which lighting
tokens should be attended to, prior to lighting baking, sig-
nificantly enhances relighting results compared to baking
global lighting first. This design allows for more effective
capture of geometric and lighting variations, boosting the
performance of DE (Tab. 4).
Input Types. We analyze the effect of different input la-
tent representations on the decoder D in Tab. 3. LH-SLAT,
which encodes rich and consistent lighting interaction in-
formation, outperforms both Base Color SLAT and Shaded
SLAT (Zs). The use of Zs complicates relighting due to the
entanglement of unknown lighting. However, Base Color
SLAT serves as a valuable complement to LH-SLAT; their

<!-- page 10 -->
combination yields the best performance.
5. Conclusion
We propose a compact multi-stage framework for re-
lightable 3D generation, enabling consistent high-fidelity
reconstruction and realistic relighting. Experiments show
improved quantitative and perceptual results over strong
baselines, and ablations confirm each component’s contri-
bution.
Although evaluated on controlled captures with
moderate compute, the approach suggests clear directions
for in-the-wild and dynamic scenes and for efficiency and
generalization improvements. We hope this work advances
practical neural relighting and reconstruction.
Acknowledgments:
This
research
was
supported
by the Beijing Natural Science Foundation (L244043),
the
Zhejiang
Provincial
Natural
Science
Foundation
(LD24F020007).
References
[1] Jonathan T Barron and Jitendra Malik. Intrinsic scene prop-
erties from a single rgb-d image. In CVPR, 2013. 3
[2] Zoubin Bi, Yixin Zeng, Chong Zeng, Fan Pei, Xiang Feng,
Kun Zhou, and Hongzhi Wu. Gs3: Efficient relighting with
triple gaussian splatting. In SIGGRAPH Asia, 2024. 3, 5
[3] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun
Jampani. Sf3d: Stable fast 3d mesh reconstruction with uv-
unwrapping and illumination disentanglement.
In CVPR,
2025. 2, 6, 7, 8, 3
[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. TVCG,
2024. 1
[5] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao,
Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie,
Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-
quality 3d asset generation via primitive diffusion. In CVPR,
2024. 2, 6, 7, 8, 1, 3
[6] Zilong Chen, Yikai Wang, Wenqiang Sun, Feng Wang, Yi-
wen Chen, and Huaping Liu. Meshgen: Generating pbr tex-
tured mesh with render-enhanced auto-encoder and genera-
tive data augmentation. In CVPR, 2025. 2, 6, 7, 8, 1, 3
[7] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr
Bojanowski. Vision transformers need registers. In ICLR,
2024. 5
[8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
A universe of annotated 3d objects. In CVPR, 2023. 6
[9] Zhao Dong, Ka Chen, Zhaoyang Lv, Hong-Xing Yu, Yunzhi
Zhang, Cheng Zhang, Yufeng Zhu, Stephen Tian, Zhengqin
Li, Geordie Moffatt, et al. Digital twin catalog: A large-scale
photorealistic 3d object digital twin dataset. In CVPR, 2025.
6, 7
[10] Andreas Engelhardt, Mark Boss, Vikram Voleti, Chun-Han
Yao, Hendrik Lensch, and Varun Jampani. Svim3d: Stable
video material diffusion for single image 3d generation. In
ICCV, 2025. 3
[11] Fr´ed´eric Fortier-Chouinard, Zitian Zhang, Louis-Etienne
Messier, Mathieu Garon, Anand Bhattad, and Jean-Franc¸ois
Lalonde. Spotlight: Shadow-guided object relighting via dif-
fusion. arXiv:2411.18665, 2024. 3
[12] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Re-
alistic point cloud relighting with brdf decomposition and
ray tracing. In ECCV, 2024. 3, 5
[13] Kai He, Ruofan Liang, Jacob Munkberg, Jon Hasselgren,
Nandita Vijaykumar, Alexander Keller, Sanja Fidler, Igor
Gilitschenski, Zan Gojcic, and Zian Wang.
Unirelight:
Learning joint decomposition and synthesis for video relight-
ing. In NeurIPS, 2025. 6
[14] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao
Tan. Lrm: Large reconstruction model for single image to
3d. In ICLR, 2024. 3
[15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-rank adaptation of large language models. In
ICCV, 2022. 4, 1
[16] Haian Jin, Yuan Li, Fujun Luan, Yuanbo Xiangli, Sai Bi,
Kai Zhang, Zexiang Xu, Jin Sun, and Noah Snavely. Neu-
ral gaffer: Relighting any object via diffusion. In NeurIPS,
2024. 2, 3, 6, 7
[17] Damjan Kalajdzievski. A rank stabilization scaling factor for
fine-tuning with lora. arXiv:2312.03732, 2023. 1
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. TOG, 2023. 3, 4
[19] Hong Li, Houyuan Chen, Chongjie Ye, Zhaoxi Chen, Bo-
han Li, Shaocong Xu, Xianda Guo, Xuhui Liu, Yikai Wang,
Baochang Zhang, Satoshi Ikehata, Boxin Shi, Anyi Rao, and
Hao Zhao. Light of normals: Unified feature representation
for universal photometric stereo. In ICLR, 2026. 5
[20] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan
Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding
Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape
synthesis using large-scale rectified flow models. TPAMI,
2025. 2
[21] Zhong Li, Liangchen Song, Zhang Chen, Xiangyu Du, Lele
Chen, Junsong Yuan, and Yi Xu. Relit-neulf: Efficient re-
lighting and novel view synthesis via neural 4d light field. In
ACM MM, 2023. 3
[22] Zhengqin Li, Dilin Wang, Ka Chen, Zhaoyang Lv, Thu
Nguyen-Phuoc, Milim Lee, Jia-Bin Huang, Lei Xiao, Cheng
Zhang, Yufeng Zhu, et al. Lirm: Large inverse rendering
model for progressive reconstruction of shape, materials and
view-dependent radiance fields. In CVPR, 2025. 3
[23] Ruofan Liang, Zan Gojcic, Huan Ling, Jacob Munkberg, Jon
Hasselgren, Zhi-Hao Lin, Jun Gao, Alexander Keller, Nan-
dita Vijaykumar, Sanja Fidler, et al. Diffusionrenderer: Neu-
ral inverse and forward rendering with video diffusion mod-
els. In CVPR, 2025. 2, 3, 6, 7, 1

<!-- page 11 -->
[24] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng
Wang, Lingjie Liu, Taku Komura, and Wenping Wang. Nero:
Neural geometry and brdf reconstruction of reflective objects
from multiview images. TOG, 2023. 6, 8
[25] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feicht-
enhofer, Trevor Darrell, and Saining Xie. A convnet for the
2020s. In CVPR, 2022. 3
[26] Zexiang Liu, Yangguang Li, Youtian Lin, Xin Yu, Sida Peng,
Yan-Pei Cao, Xiaojuan Qi, Xiaoshui Huang, Ding Liang, and
Wanli Ouyang. Unidream: Unifying diffusion priors for re-
lightable text-to-3d generation. In ECCV, 2024. 3
[27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. In ICLR, 2019. 1
[28] Nadav Magar, Amir Hertz, Eric Tabellion, Yael Pritch, Alex
Rav-Acha, Ariel Shamir, and Yedid Hoshen. Lightlab: Con-
trolling light sources in images with diffusion models. In
SIGGRAPH, 2025. 2, 3
[29] Sourab Mangrulkar,
Sylvain Gugger,
Lysandre Debut,
Younes Belkada,
Sayak Paul,
and Benjamin Bossan.
PEFT: State-of-the-art parameter-efficient fine-tuning meth-
ods. https://github.com/huggingface/peft,
2022. 1
[30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. CACM, 2021. 6
[31] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
In TMLR, 2023. 4
[32] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Pe-
ters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard
Newcombe, and Yuheng Carl Ren. Aria digital twin: A new
benchmark dataset for egocentric 3d machine perception. In
CVPR, 2023. 6, 7
[33] Y. Poirier-Ginter, A. Gauthier, J. Phillip, J.-F. Lalonde, and
G. Drettakis. A diffusion approach to radiance field relight-
ing using multi-illumination synthesis. EGSR, 2024. 3
[34] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,
2023. 3
[35] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mu-
tian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng
Bo, and Xiaoguang Han.
Richdreamer: A generalizable
normal-depth diffusion model for detail richness in text-to-
3d. In CVPR, 2024. 3
[36] Fabio Remondino, Ali Karami, Ziyang Yan, Gabriele Maz-
zacca, Simone Rigon, and Rongjun Qin. A critical analysis
of nerf-based 3d reconstruction. Remote Sensing, 2023. 3
[37] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
Pradeep Ramani, and Tri Dao.
Flashattention-3:
Fast
and accurate attention with asynchrony and low-precision.
arXiv:2407.08608, 2024. 1
[38] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li,
and Xiao Yang. Mvdream: Multi-view diffusion for 3d gen-
eration. In ICLR, 2024. 3
[39] Dongseok Shim, Yichun Shi, Kejie Li, H Jin Kim, and Peng
Wang. Mvlight: Relightable text-to-3d generation via light-
conditioned multi-view diffusion. arXiv:2411.11475, 2024.
3
[40] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for ef-
ficient 3d content creation. In ICLR, 2024. 3
[41] Jiapeng Tang, Matthew Lavine, Dor Verbin, Stephan J
Garbin, Matthias Nießner, Ricardo Martin Brualla, Pratul P
Srinivasan, and Philipp Henzler. Rogr: Relightable 3d ob-
jects using generative relighting. In NeurIPS, 2025. 3
[42] Blender Development Team. Eevee release notes for blender
4.2.
https://developer.blender.org/docs/
release_notes/4.2/eevee/, 2025. 7
[43] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. TIP, 2004. 7
[44] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi
Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable
image-to-3d generation via 3d latent diffusion transformer.
In NeurIPS, 2024. 2
[45] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng
Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong
Yang. Structured 3d latents for scalable and versatile 3d gen-
eration. In CVPR, 2025. 2, 3, 4, 5, 7, 1
[46] Ziyang Yan, Nazanin Padkan, Paweł Trybała, Elisa Mari-
arosaria Farella, and Fabio Remondino.
Learning-based
3d reconstruction methods for non-collaborative surfaces—a
metrological evaluation. Metrology, 2025. 3
[47] Ziyang Yan, Lei Li, Yihua Shao, Siyu Chen, Zongkai Wu,
Jenq-Neng Hwang, Hao Zhao, and Fabio Remondino. 3dsce-
needitor: Controllable 3d scene editing with gaussian splat-
ting. In WACV, 2026. 3
[48] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xi-
aoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han.
Hi3dgen: High-fidelity 3d geometry generation from images
via normal bridging. In ICCV, 2025. 2
[49] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen,
Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey
Hu, Matthew Tancik, and Angjoo Kanazawa.
gsplat: An
open-source library for gaussian splatting. JMLR, 2025. 1
[50] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. TOG, 2024. 1
[51] Chong Zeng, Guojun Chen, Yue Dong, Pieter Peers, Hongzhi
Wu, and Xin Tong. Relighting neural radiance fields with
shadow and highlight hints. In SIGGRAPH, 2023. 3
[52] Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong,
Hongzhi Wu, and Xin Tong. Dilightnet: Fine-grained light-
ing control for diffusion-based image generation. In SIG-
GRAPH, 2024. 2, 3, 6, 7
[53] Chong Zeng, Yue Dong, Pieter Peers, Hongzhi Wu, and Xin
Tong. Renderformer: Transformer-based neural rendering
of triangle meshes with global illumination. In SIGGRAPH,
2025. 6, 9
[54] Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick
Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and

<!-- page 12 -->
Miloˇs Haˇsan. Rgb↔x: Image decomposition and synthe-
sis using material- and lighting-aware diffusion models. In
SIGGRAPH, 2024. 2, 6, 7, 1, 3
[55] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu,
Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu.
Clay: A controllable large-scale generative model for creat-
ing high-quality 3d assets. ACM Transactions on Graphics
(TOG), 43(4):1–20, 2024. 2, 3
[56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling
in-the-wild training for diffusion-based illumination harmo-
nization and editing by imposing consistent light transport.
In ICLR, 2025. 3
[57] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 6, 7
[58] Tianyuan Zhang, Zhengfei Kuang, Haian Jin, Zexiang Xu,
Sai Bi, Hao Tan, He Zhang, Yiwei Hu, Milos Hasan,
William T Freeman, et al. Relitlrm: Generative relightable
radiance for large reconstruction models. In ICLR, 2025. 3
[59] Xuying Zhang, Bo-Wen Yin, Yuming Chen, Zheng Lin, Yun-
heng Li, Qibin Hou, and Ming-Ming Cheng. Temo: Towards
text-driven 3d stylization for multi-object meshes. In CVPR,
2024. 3
[60] Xuying Zhang, Yutong Liu, Yangguang Li, Renrui Zhang,
Yufei Liu, Kai Wang, Wanli Ouyang, Zhiwei Xiong, Peng
Gao, Qibin Hou, and Ming-Ming Cheng. Tar3d: Creating
high-quality 3d assets via next-part prediction.
In ICCV,
2025.
[61] Xuying Zhang, Yupeng Zhou, Kai Wang, Yikai Wang, Zhen
Li, Shaohui Jiao, Daquan Zhou, Qibin Hou, and Ming-Ming
Cheng. Ar-1-to-3: Single image to consistent 3d object via
next-view prediction. In ICCV, 2025. 3
[62] Xiaoming Zhao, Pratul P. Srinivasan, Dor Verbin, Keunhong
Park, Ricardo Martin Brualla, and Philipp Henzler. IllumiN-
eRF: 3D Relighting Without Inverse Rendering. In NeurIPS,
2024. 3
[63] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao,
Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng
Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffu-
sion models for high resolution textured 3d assets generation.
arXiv:2501.12202, 2025. 2, 3, 6, 7, 8, 1
[64] Jialin Zhu, Jiangbei Yue, Feixiang He, and He Wang. 3d
student splatting and scooping. In CVPR, 2025. 6

<!-- page 13 -->
NeAR: Coupled Neural Asset–Renderer Stack
Supplementary Material
A. More Implementation Details
A.1. Different Single-Image Relighting Paradigms
To further elucidate the architectural advantages of NeAR,
we present a structured comparison of three representative
paradigms in Fig. 1.
2D-based Intrinsic Decomposition [23, 54]):
As shown
in Fig. 1(a-b), these methods decompose a single image into
2D PBR maps (e.g., albedo, roughness, normals). However,
lacking an underlying 3D representation, they struggle to
disentangle view-dependent specular highlights from base
color, and fail to support novel-view rendering and accurate
shadow modeling.
Single Image to 3D Textured Mesh [5, 6, 63]):
Typ-
ical 3D generative models (Fig. 1(c)) follow a decoupled
paradigm, where asset construction (geometry and PBR tex-
tures) is fully separated from rendering. While compatible
with standard graphics pipelines (e.g., Blender), they rely
on highly ill-posed PBR inversion, often leading to material
ambiguity (e.g., misclassifying metallic surfaces as diffuse).
Our Coupled NeAR Stack:
In contrast, NeAR adopts
a “homogenize-then-synthesize” paradigm.
By lift-
ing single-image inputs under arbitrary lighting into a
Lighting-Homogenized SLAT (LH-SLAT), we construct an
illumination-invariant neural asset that serves as a stable
rendering substrate, preserving cues of geometry, uniform
lighting, and material interactions. On top of this, a cou-
pled neural renderer learns to interpret these homogenized
latents, enabling the synthesis of complex light–material in-
teractions.
A.2. Implementation Details
Training Details.
We conduct all training experiments on
four NVIDIA H100 80GB HBM3 GPUs.
In the LH-SLAT Reconstruction & Generation phase
(§3.3), we fine-tune a rectified flow model equipped with
LoRA [15] to normalize shaded SLATs from arbitrary im-
ages into LH-SLATs. The goal of this phase is to learn a
mapping, fθ, that transforms light-dependent shaded SLAT
representations into light-homogenized counterparts.
To
achieve this efficiently while preserving the prior knowl-
edge of the original flow model fs [45], we initialize LoRA
using PEFT [29]. We configure LoRA with a rank of 512
and a scaling factor, further integrating rslora [17] to en-
hance training stability. LoRA adaptors are applied to the
query, key, value, and output projection modules within
the attention mechanism.
We optimize the model using
AdamW [27] with a learning rate of 1×10−4. This training
phase takes approximately two days to complete.
In the Relightable Neural 3DGS Synthesis phase
(§3.5), we employ the AdamW optimizer with a batch size
of 48. The learning rate is warmed up linearly to 1 × 10−4
over the first 5K steps, followed by a cosine decay sched-
ule. We perform end-to-end joint training on the IAD, LAD,
and the Lighting-Aware tokenizer (denoted as El). To ac-
celerate training, we leverage Flash-Attention 3 [37] and
gsplat [49]. The model is trained for 500K iterations across
all loss components, requiring approximately 10 days. Ad-
ditionally, we investigated the incorporation of geomet-
ric constraint losses, specifically normal and depth losses.
However, we observed that adding these regularizations de-
graded both convergence speed and rendering quality, sug-
gesting a trade-off between geometric constraints and ren-
dering fidelity within the 3DGS framework [4, 50].
Inference Details.
Given a single input image Iin with
unknown lighting and a target high dynamic range (HDR)
environment map E, our inference pipeline proceeds as fol-
lows.
Since our method decouples geometry generation
from relighting, we first reconstruct a 3D mesh m from Iin
using Hunyuan3D 2.1 (HY3D 2.1) [63] with default set-
tings. This mesh is then voxelized to provide coordinates
for the structurally sparse voxel feature SLAT.
Following Trellis [45], we utilize the pre-trained SLAT
flow model fs to generate an initial shaded SLAT Zs from
Iin. Note that Zs inherently contains arbitrary lighting in-
formation from the input image. To remove these light-
ing effects, we concatenate Zs with noise (matching Zs in
shape) along the channel dimension and feed the result into
our fine-tuned corrective model fθ to yield the Lighting-
Homogenized SLAT (LH-SLAT).
Subsequently, for the target lighting, we pre-process the
environment map E (as detailed in Sec. 3.5.2) and encode it
into a lighting condition embedding CL using the Lighting-
Aware tokenizer El. The IAD module then processes the
LH-SLAT to extract intrinsic features h. Simultaneously,
the LAD DE integrates the viewing direction encoding ev
and the lighting condition CL to predict the 3D Gaussian
attributes for the specific view and lighting (Eq. 5). Finally,
we render the relit HDR image Ihdr
target via gsplat [49]. To
align the visual output with standard rendering engines like
Blender, we apply AgX tone mapping3 to convert the HDR
result into a low dynamic range (LDR) image.
3https://github.com/iamNCJ/simple-ocio

<!-- page 14 -->
A.3. Network Architectures
Register Tokens.
Apart from the lighting-aware tok-
enizer El, and consistent with [45], our method primar-
ily employs Transformer networks.
As depicted in Fig.
4, the IAD DI comprises 3D shifted window multi-head
self-attention (3D-SW-MSA) and a feed-forward network
(FFN). Addressing the limitation of the naive 3D-SW-MSA
design in [45], which computes attention solely within local
windows and neglects inter-window information exchange,
we introduce learnable register tokens. These tokens inter-
act with all windows via 3D multi-head cross-attention (3D-
MCA), serving as a global information bridge to facilitate
the model’s learning of global context. The lighting-aware
decoder DE receives intrinsic features h, view encoding ev,
register tokens, and lighting encoding to generate lighting-
dependent features hv. Register tokens and lighting encod-
ing are injected into the network via 3D-MCA. Here, h and
ev are added in a voxel-wise manner to determine which
lighting encoding tokens should be attended to under the
current viewpoint. The ablation study on the interaction or-
der of viewpoint and lighting information is illustrated in
Fig. 9 and Tab 4.
Loss Functions.
For the relightable neural 3DGS synthe-
sis stage, we optimize the model using a composite objec-
tive function Ltotal. This objective is a weighted sum of three
primary reconstruction components—HDR reconstruction
(Lrecon), physically-based material supervision (Lpbr), and
shadow-casting (Lshadow)—along with regularization terms
for Gaussian primitives:
  \sma l l \mat h cal {L}_ { \text {total}}  = \mathc a l {L}_{\text {recon}} + \lambda _{\text {pbr}}\mathcal {L}_{\text {pbr}} + \lambda _{\text {shadow}}\mathcal {L}_{\text {shadow}} + \lambda _{\text {vol}}\mathcal {L}_{\text {vol}} + \lambda _{\alpha }\mathcal {L}_{\alpha }. 
(7)
In our experiments, we set the weighting hyperparameters
to λpbr = 0.3, λshadow = 0.5, λvol = 10, 000, and λα =
0.001.
Reconstruction Loss (Lrecon).
We formulate Lrecon to en-
sure high-fidelity HDR rendering. Before calculating per-
ceptual metrics, we apply AgX tone mapping to both the
rendered HDR image Ihdr
target and the ground-truth Ihdr
gt , yield-
ing their LDR counterparts ˆItarget and ˆIgt. The loss com-
bines an L1 distance in the logarithmic domain for HDR
consistency, along with SSIM and LPIPS losses on the
tonemapped LDR images for perceptual quality:
  \sma l
l \begin {a
ligned }  \m athcal {
L} _ {\t
e xt {r e con}} = & \qua d \mat
h cal {L}_{1}(\log ( I^{\text {hdr}}_{\text {target}} + 1), \log (I^{\text {hdr}}_{\text {gt}} + 1)) \\ & + 0.2 (1 - \text {SSIM}(\hat {I}_{\text {target}}, \hat {I}_{\text {gt}})) \\ & + 0.2 \text {LPIPS}(\hat {I}_{\text {target}}, \hat {I}_{\text {gt}}). \end {aligned} \label {eq:recon_loss} 
(8)
PBR and Shadow Supervision.
To guide the model to-
wards physically plausible decomposition, we impose di-
rect constraints on the intermediate PBR feature maps. The
𝑬𝒍𝒅𝒓
𝑬𝒍𝒐𝒈
𝑬𝒅𝒊𝒓
Spatial 
Cross MHA 
ConvNext 
𝑬𝒍𝒅𝒓
𝑬𝒍𝒐𝒈
𝑬𝒅𝒊𝒓
Env.  
Encoder 
(c) Ours 
(b) Diffusion Renderer 
𝑬𝒍𝒅𝒓
𝑬𝒍𝒐𝒈
(a) Neural Gaffer 
Image 
Encoder 
Video 
Encoder 
Figure 13. Compared to existing HDRI encoding methods, we
bind directional information with multi-scale features using posi-
tional encoding.
material loss Lpbr supervises the base color (Ib), roughness
(Ir), metallic (Im), and shading (Is) maps against their
ground truths:
  \s m all \m at
hca l  {L}_{ \t
ext  {pbr}}  =
 \m a thcal {L
}_{1}(I^b, I^b_{gt}) + \mathcal {L}_{1}(I^r, I^r_{gt}) + \mathcal {L}_{1}(I^m, I^m_{gt}) + \mathcal {L}_{1}(I^s, I^s_{gt}). \label {eq:pbr_loss} 
(9)
Similarly, Lshadow employs an L1 loss to ensure the geomet-
ric consistency of cast shadows under novel lighting condi-
tions.
Regularization.
To prevent the degeneration of Gaussian
primitives (e.g., becoming too large or too opaque) during
optimization [45], we incorporate a volumetric loss Lvol and
an opacity loss Lα:
  \s m
a
ll
 
\
beg
i
n
 {a
l
ig
n e
d
} 
\
m
ath
c
a
l {
L
}_{
\ t
ex t
 
{v
o
l
}} 
&
=
 \f
ra c  {
1 }{LK}\sum _{i=1}^{L}\sum _{k=1}^K \prod \boldsymbol {s}_i^k + \frac {1}{LK}\sum _{i=1}^{L}\sum _{k=1}^K \prod \boldsymbol {\hat s}_i^k, \\ \mathcal {L}_{\alpha } &= \frac {1}{LK}\sum _{i=1}^{L}\sum _{k=1}^K(1-\alpha _i^k)^2. \end {aligned} \label {eq:reg_losses} 
(10)
These terms are calculated across the L active voxels, with
each voxel predicting K Gaussian primitives. Specifically,
Lvol regularizes the scale components s from the IAD and
ˆs from the LAD simultaneously.
Lighting Tokenizer.
As illustrated in Fig. 13, the light-
ing tokenizer El is primarily designed to process and in-
ject lighting information into the network for relighting pur-
poses, while also effectively perceiving rotations in the am-
bient lighting. Similar to Neural Graffer and Diffusion Ren-
derer, as depicted in Fig. 13 (a), our approach leverages the
Ehdr and Elog components of the environment map E to
provide lighting color characteristics. Neural Graffer en-
codes the environment map into an image latent space via
a pre-trained image VAE, whereas Diffusion Renderer em-
ploys a video VAE model to compress Eldr, Elog, and Edir

<!-- page 15 -->
Neural Renderer
or
Blender       
Forward Rendering
or
LH-SLAT 𝒁𝒍𝒉
GT G-Buffers
Neural Renderer
Novel Light
Relit Image
Reconstruction / Relighting
Single Image
Neural Relightable Model
Original / Novel Light 
Relit Image
Input
Input
Input View
Input View
Novel View Synthesis
Novel Light
Relit Image
Novel View
LH-Slat
or
3D PBR Mesh
Figure 14. Schematic illustration of four distinct sub-tasks.
into a video latent space of consecutive frames, thereby ac-
commodating subsequent image or video diffusion model
training.
The lighting-aware tokenizer, El, is primarily designed to
process and inject lighting information into the network to
enable relighting while also effectively perceiving rotations
in the environment map. Similar to Neural Graffer and Dif-
fusion Renderer, as depicted in Fig. 13, our approach lever-
ages the Ehdr and Elog components of environmental illu-
mination to provide lighting color features. Neural Graffer
encodes the environment map into the image latent space
using a pretrained image VAE’s encoder. Diffusion Ren-
derer, in contrast, employs a video VAE model to compress
Eldr, Elog, and Edir into a video latent space of consecutive
frames, thus accommodating subsequent image or video
diffusion model training. As shown in Fig. 13(c), the design
of El aims to facilitate the injection of lighting information
from the LH-SLAT into relightable 3D Gaussian Splatting
(GS). This design addresses two key challenges: First, dif-
ferent materials require sensitivity to varying resolutions of
lighting information. For instance, highly rough surfaces re-
quire only low-resolution environment maps, whereas high-
metallicity surfaces with low roughness necessitate high-
resolution maps. To address this, we utilize ConvNext [25]
to extract multi-resolution features from the lighting pyra-
mid and employ a spatial attention mechanism to com-
pute attention scores and exchange information between
these resolutions. Second, the model should accurately per-
ceive rotations of the environment map.
Neural Graffer
requires deforming the environment map itself.
Our ap-
proach, similar to Diffusion Renderer, can rotate the illu-
mination by adjusting the environment light direction map,
Edir, requiring only the application of a rotation matrix to
the direction vector. However, Diffusion Renderer relies
on an additionally trained environment encoder (Env. En-
coder). Our method employs a direction-encoding-aware
spatial cross-multihead attention (Spatial Cross MHA) to
guide visual features at different resolutions using direc-
tional information. It combines multi-scale feature fusion
to preserve both detailed and global information and uti-
lizes a RoPE+RMSNorm transformer layer for efficient se-
quence modeling, refer to Fig. 4. This allows the complex
HDRI lighting information to be encoded into conditional
tokens suitable for cross-attention, providing high-quality
lighting conditions for the renderer. Abstractly, we model
the environment map as a set of light source tokens, each en-
coded with absolute direction vector positional information
and multi-scale features. Subsequently, the Lighting-Aware
Decoder (LAD), El, can efficiently determine the relevance
of each token to the current viewpoint by leveraging view-
point direction encoding.
A.4. Experiments Setup
Fig. 14 shows the quantitative evaluation setup for four sub-
tasks—forward rendering, reconstruction, relighting, and
novel view synthesis—as summarized in Tab. 1. It illus-
trates how methods, given different input modalities (e.g.,
LH-SLAT, G-buffers, single images, or 3D assets), are pro-
cessed under varying viewpoints and lighting via neural ren-
dering or relightable models to produce relit images.
B. More Results
B.1. Additional Comparisons
Qualitative Evaluation.
We provide comprehensive vi-
sual comparisons to further substantiate the effectiveness of
our method. Figures 16 and 17 illustrate additional results
for single-image reconstruction under diverse illumination
conditions. For single-image relighting with unknown in-
put lighting, we present extended comparisons in Figures 18
and 19. Notably, our method recovers significantly more
accurate shadows and specular highlights compared to ex-
isting 2D diffusion-based relighting models [16, 23, 52, 54],
which often struggle with physical consistency.
Comparison with 3D Generation Baselines.
We also
conduct detailed comparisons against state-of-the-art 3D
generation methods capable of producing PBR materi-
als [3, 5, 6, 63]. As shown in Fig. 20, our approach demon-
strates superior material disentanglement, yielding high-
lights and tonal values that align closely with the ground
truth. To ensure a fair comparison and isolate material qual-
ity from geometric failures, we provide the baselines with
a fixed frontal view and evaluate the rendered output from
the same perspective. This setup mitigates the potential for
geometric collapse or severe artifacts in baseline methods,
focusing the evaluation on rendering and relighting fidelity.
B.2. Additional Visualization Results
PBR Material and Shadow Decomposition.
Leverag-
ing a single input image and a target environment map,

<!-- page 16 -->
Ours
Input Image
Light 
Condition
Ours
Input Image
Light 
Condition
Figure 15. Failure Cases. Left: High-frequency details (e.g.,
text) are blurred due to voxel resolution constraints and VAE com-
pression loss. Right: Transparent objects exhibit checkerboard
artifacts. While 3DGS theoretically supports alpha blending, data
scarcity in current datasets leads to inaccurate density estimation
for refractive surfaces.
our pipeline enables high-fidelity, relightable 3D Gaussian
Splatting synthesis with support for multi-view rendering.
In Fig. 21, we visualize the decomposed PBR material maps
(Albedo, Roughness, Metallic) and the generated shadow
maps. These visualizations explicitly demonstrate the ef-
fectiveness of our physically-based supervision signals (dis-
cussed in Sec. 3.5.3) in achieving clean and plausible mate-
rial decomposition.
C. Discussion
C.1. Limitations and Future Work
Despite the robust performance of our framework in gener-
alized single-image relightable 3D Gaussian synthesis, sev-
eral challenges remain that define coordinates for future re-
search.
Fine-grained Detail Preservation.
As illustrated in
Fig. 15 (Left), the reconstruction of high-frequency tex-
tures—such as small text—is currently hindered by the fea-
ture compression pipeline.
The semantic features from
DINOv2 [31] undergo substantial downsampling to match
the resolution of the LH-SLAT voxel grid.
This bottle-
neck inevitably leads to the loss of intricate details. Fu-
ture iterations will explore multi-scale feature refinement
or sparse high-resolution voxel structures to better preserve
these fine-grained elements.
Complex Material Modeling.
Handling transparent or
highly refractive materials remains a significant challenge.
While the alpha-blending mechanism of 3DGS natively
supports semi-transparency, our model’s ability to represent
such surfaces is heavily dependent on the training data dis-
tribution. Due to the scarcity of high-quality transparent
objects in current large-scale datasets, the model occasion-
ally fails to densify Gaussians sufficiently, resulting in the
artifacts shown in Fig. 15 (Right). Incorporating special-
ized physics-based transmission losses or curated transpar-
ent object datasets could mitigate this issue.
C.2. Scalability and Generalization
A core strength of our proposed framework is its inherent
scalability across data and architecture, which ensures its
long-term viability as a foundation for 3D generative tasks.
Data Scalability.
Our modular, multi-stage design allows
for independent scaling of different components. Stage 1
(Lighting Homogenization) directly benefits from increas-
ing data volume and diversity, as it learns to suppress
complex baked-in illumination—a task that scales effec-
tively with broader data distributions. In contrast, Stage
2 (Lighting-aware Synthesis) is highly efficient due to its
feed-forward nature, primarily requiring diversity in mate-
rial properties and lighting conditions rather than sheer vol-
ume.
Architectural Scalability.
All core components of our
pipeline are based on Transformer architectures, which of-
fer a predictable path for capacity scaling. As demonstrated
in our ablation studies (see Tab. 2), increasing model capac-
ity consistently improves rendering quality, particularly for
complex specular effects. Furthermore, the LH-SLAT rep-
resentation is spatially scalable; increasing the voxel resolu-
tion allows the model to capture more complex geometries
and lighting interactions. This flexibility enables a practi-
cal trade-off between inference speed and rendering fidelity
depending on the target application.
Overall, the capacity of our model to generalize across
diverse object categories and lighting conditions validates
our core design philosophy. Our framework validates the ef-
fectiveness of jointly designing neural rendering and neural
asset stacks, providing a robust and extensible path toward
high-fidelity, relightable 3DGS synthesis from arbitrary sin-
gle images.

<!-- page 17 -->
Diff. Renderer
Ours
RGB     X
Input & GT
DiLightNet
Light Condition
Figure 16. Additional qualitative results for single-image reconstruction under random illumination.

<!-- page 18 -->
Diff. Renderer
Ours
RGB     X
Input & GT
DiLightNet
Light Condition
Figure 17. Additional qualitative results for single-image reconstruction under random illumination.

<!-- page 19 -->
DiLightNet
Diff. Renderer
Input
Ours
Neural Graffer
GT Relit
Light 
Condition
Figure 18. More visualization results of relighting and rendering from a single image under unknown illumination.

<!-- page 20 -->
DiLightNet
Diff. Renderer
Input
Ours
Neural Graffer
GT Relit
Light 
Condition
Figure 19. More visualization results of relighting and rendering from a single image under unknown illumination.

<!-- page 21 -->
MeshGen
Hunyuan3D 2.1
Input
Ours
3D Topia-XL
GT Relit
Light 
Condition
Stable fast 3D
Figure 20. Comparison of relighting renderings between our neural rendering method and 3D generation methods that can recover PBR
material properties. Our method achieves more stable and accurate rendering results.

<!-- page 22 -->
Base Color
Metallic
Input
Roughness
Relight
Shadow
Light 
Condition
Figure 21. Additional relighting results from a single image under target illumination, along with the PBR materials and shadows estimated
by our method.
