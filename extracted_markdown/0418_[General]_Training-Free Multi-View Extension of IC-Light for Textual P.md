<!-- page 1 -->
Highlights
GS-Light: Training-Free Multi-View Extension of IC-Light for Tex-
tual Position-Aware Scene Relighting
Jiangnan Ye, Jiedong Zhuang, Lianrui Mu, Wenjie Zheng, Jiaqi Hu, Xingze
Zou, Jing Wang, Haoji Hu
• GS-Light enables fast, position-aware multi-view relighting in 3DGS.
• Lighting priors extracted via LVLM + geometry & semantics.
• Cross-View Attention enforce multi-view consistency in relit outputs.
• Outperforms baselines with better quality in diverse scenes.
arXiv:2511.13684v1  [cs.CV]  17 Nov 2025

<!-- page 2 -->
GS-Light: Training-Free Multi-View Extension of
IC-Light for Textual Position-Aware Scene Relighting
Jiangnan Yea, Jiedong Zhuanga, Lianrui Mua, Wenjie Zhenga, Jiaqi Hua,
Xingze Zoua, Jing Wanga, Haoji Hua,∗
aCollege of Information Science and Electronic Engineering, Zhejiang University,
Hangzhou, 310027, Zhejiang, China
Abstract
We introduce GS-Light, an efficient, textual position-aware pipeline for text-
guided relighting of 3D scenes represented via Gaussian Splatting (3DGS).
GS-Light implements a training-free extension of a single-input diffusion
model to handle multi-view inputs. Given a user prompt that may specify
lighting direction, color, intensity, or reference objects, we employ a large
vision-language model (LVLM) to parse the prompt into lighting pri-
ors. Using off-the-shelf estimators for geometry and semantics (depth, sur-
face normals, and semantic segmentation), we fuse these lighting priors with
view-geometry constraints to compute illumination maps and generate ini-
tial latent codes for each view. These meticulously derived init latents guide
the diffusion model to generate relighting outputs that more accurately re-
flect user expectations, especially in terms of lighting direction. By feeding
multi-view rendered images, along with the init latents, into our multi-view
relighting model, we produce high-fidelity, artistically relit images. Finally,
we fine-tune the 3DGS scene with the relit appearance to obtain a fully relit
3D scene. We evaluate GS-Light on both indoor and outdoor scenes, com-
paring it to state-of-the-art baselines including per-view relighting, video
relighting, and scene editing methods. Using quantitative metrics (multi-
view consistency, imaging quality, aesthetic score, semantic similarity, etc.)
∗Corresponding author.
∗∗Email addresses: jiangnan_ye@zju.edu.cn (Jiangnan Ye), zhuangjiedong@zju.edu.cn
(Jiedong
Zhuang),
mulianrui@zju.edu.cn
(Lianrui
Mu),
wenjie_zheng@zju.edu.cn
(Wenjie Zheng),
jiaqi_hu@zju.edu.cn (Jiaqi Hu),
zeezou@zju.edu.cn (Xingze Zou),
j_wang@zju.edu.cn (Jing Wang), haoji_hu@zju.edu.cn (Haoji Hu)

<!-- page 3 -->
and qualitative assessment (user studies), GS-Light demonstrates consistent
improvements over baselines. Code and assets will be made available upon
publication.
Keywords:
Gaussian Splatting, Diffusion models, Textual relighting
Relit Video
Relit Gasussian Splatting
Source
Ours
RelightVid
Lumen
Ours
DGE
EditSplat
IN2N
IGS2GS
IGS2GS-IC
"bear, forest, sunlight filtering through trees, natural lighting, warm atmosphere, light from top"
"portrait, detailed face, sunshine from window, warm atmosphere, light from the left"
"field, outdoor, starry night, silver moonlight, tranquil tone"
"bicycle, sunset, orange glow, long shadows"
Figure 1: Qualitative comparison between our GS-Light and other prior work on re-
lighting videos or gaussian splatting scenes. Our GS-Light produces results with higher
fidelity and aesthetic, improved multi-view consistency, stronger semantic relevance, and
enhanced controllability of lighting direction through text prompts.
2

<!-- page 4 -->
1. Introduction
Relighting a 3D scene — changing its illumination while preserving its
geometry, material attributes, and view coherence — is a central problem in
computer graphics and vision, with applications spanning augmented/virtual
reality, film production, and content creation. Modern neural scene represen-
tation methods such as Neural Radiance Fields (NeRF [1]) and 3D Gaussian
Splatting (3DGS [2]) have enabled impressive novel view synthesis, and sep-
arately, generative 2D image editing / relighting models (especially diffusion-
based ones, e.g. IC-Light [3]) have delivered strong image-level lighting edits.
However, combining these two worlds to achieve multi-view consistent, tex-
tually controllable, and position-aware relighting remains nontrivial.
Key
challenges include inferring plausible scene lighting from text, ensuring con-
sistency over views and efficiently dealing with geometry/light interaction
(shadows, normals, occlusion).
With the advent of IC-Light, its powerful 2D image relighting capability
— covering illumination harmonization, identity preservation, and lighting
artistry — has been rapidly adopted in a wide range of AIGC-related down-
stream applications. However, since ICLight is a model fine-tuned from Sta-
ble Diffusion [4] v1.5, it inherently introduces 3D inconsistency in multi-view
editing tasks, posing challenges to lifting its prior knowledge to 3D scene
editing. In addition, our experiments reveal that current SD-based models
and multimodal models are generally insensitive to positional information
in textual prompts (e.g., left, right), which further hinders the alignment
between relighting results and user intent. Although some visual grounding
works (e.g., VPP-LLaVA [5]) strengthen training in this aspect and achieve
promising results, experiments show that they mainly learn patterns between
key location-related texts and the ground truth in the limited training data,
rather than developing a semantic understanding of abstract spatial informa-
tion. This is reflected in their poor generalization performance when tested
on out-of-domain data.
To address the aforementioned challenges, we propose GS-Light, a light-
weight iterative Gaussian Splatting optimization pipeline that bridges text-
guided 2D relighting diffusion models with 3D scene representations to achieve
position-aware, multi-view consistent relighting.
Our framework is
built upon two key components.
The first is the Position-Align Mod-
ule (PAM), which aligns positional semantic information between the in-
puts—comprising scene-rendered images and user editing instructions—and
3

<!-- page 5 -->
the relighting model, ensuring accurate interpretation of spatial cues. The
second is MV-ICLight, a multi-view relighting model based on IC-Light,
specifically designed to enforce cross-view consistency, enabling coherent and
faithful relighting across different viewpoints.
PAM is primarily responsible for the preprocessing of input data.
It
extracts information from both image and text modalities and further es-
timates the scene’s geometry and lighting, generating the position-aligned
illumination map, which is required by the subsequent module.
Given a
user prompt, we employ an LVLM (e.g., Qwen2.5-vl [6]) to extract lighting
priors such as direction, color temperature or hue, intensity, and optionally
reference objects from the given rendered images and input instructions, us-
ing a constrained question–answer template to ensure outputs are limited
to relighting-relevant parameters. For each view, we further estimate scene
geometry and semantics via off-the-shelf predictors, including depth (e.g.,
VGGT [7]), surface normals (e.g., StableNormal [8]), and semantic masks
(e.g., LangSAM [9]), which provide cues for shading, occlusion, shadow cast-
ing, and object identification. These priors are then fused with geometry
constraints to compute per-pixel illumination maps, using a simple Phong
diffuse illumination model [10], which are further initialized as the init la-
tents of the diffusion model denoising process, enabling fine-grained control
over lighting in the editing results.
MV-ICLight is a variant implementation of IC-Light’s cross-view at-
tention mechanism, enabling a single-image editing model to perform edits
consistently across multiple views. Inspired by DGE [11], we introduce an
improved epipolar constraint to realize a memory-efficient, training-free ex-
tension of the relighting model from single-view input to multi-view input.
Conditioned on the illumination maps and aligned latents, we then generate
per-view relit images that maintain illumination consistency across views.
Finally, these relit images are integrated into the 3D Gaussian Splatting rep-
resentation, producing relit scenes that remain faithful to user instructions
while ensuring multi-view coherence.
GS-Light is designed to achieve high-quality relighting of 3DGS scenes,
faithfully adhering to the user’s editing intent. In tests on both indoor and
outdoor datasets, our method demonstrates clear competitiveness against
various state-of-the-art approaches in terms of reconstruction consistency,
semantic editing similarity, and user subjective evaluation. Our contributions
are:
4

<!-- page 6 -->
• We propose GS-Light, the first efficient method that supports textual,
position-aware relighting over Gaussian Splatting scenes with multi-
view consistency. It takes around 3 minutes to generate per-scene priors
once and around 3 minutes for each scene’s relighting.
• We develop a lighting prior extraction scheme via questioning an LVLM,
combined with geometry & semantic estimators, to produce view-wise
illumination and latent initialization.
• We enforce viewwise consistency using advanced epipolar constraints,
ensuring coherence in relit views and latent space. It worth noting that
our extension framework is compatible with other diffusion models (e.g.,
DiT [12]), which makes sure the great scalability of our method.
• We validate GS-Light across a variety of scenes (indoor and outdoor),
demonstrating superior performance vs baselines in both objective and
perceptual metrics, while maintaining fast inference.
2. Related Work
2.1. 2D Illumination Harmonization / Relighting
Image relighting methods based on deep neural networks first became
mainstream. To enable image-based relighting, Light Stage [13] was intro-
duced to capture the reflectance functions of human faces, while [14] signif-
icantly reduced the number of required input images. SfSNet [15] leveraged
deep neural networks to model 3D faces and decompose material properties
for portrait relighting. [16], using a mass transport approach, achieved por-
trait illumination transfer. [17] constructed training pairs using One-Light-
At-a-Time (OLAT) scans, and [18] further divided the process into diffuse
rendering and non-diffuse residual stages.
Going a step further, Switch-
Light [19] decomposed source images into intrinsic components, and predicted
them with separate networks.
With the development of diffusion models [20, 4, 21, 12], fine-tuning pre-
trained diffusion models for image-to-image tasks such as editing [22, 23, 24],
style transfer [25, 26, 27, 28], geometry estimation [8, 29, 30], and relighting
has proven to be an effective approach. Relightful Harmonization [31] used
the background image as a condition to generate harmonized foreground illu-
mination. IC-Light [3], by imposing the principle of consistent light transport
5

<!-- page 7 -->
and leveraging a carefully curated large-scale dataset, fine-tuned Stable Dif-
fusion [4] v1.5 to achieve state-of-the-art performance in 2D image relighting.
2.2. 3DGS Editing / Stylization Models
One category of methods directly learns features in 3D space and con-
strains the loss of the target task through specific regularization terms. [32]
employs a texture-guided control mechanism to directly constrain the pa-
rameters of Gaussians. ARF [33] proposes the NNFM loss, which provides
better 3D consistency compared with the widely used Gram-matrix-based
loss [34] in style transfer tasks. Building on this, G-Style [35] introduces a
CLIP similarity term and a total variation term, while StyleGaussian [36]
embeds VGG features into a radiance field.
Another category of methods leverages the strong priors of 2D editing
models to assist in 3D scene editing, mainly addressing inconsistencies in
multi-view editing. Instruct-NeRF2NeRF [37] first introduces the Dataset
Update strategy to ensure convergence during scene optimization.
Based
on this, ViCA-NeRF [38] adopts multi-view image and depth warping &
mixup methods to provide 2D models with richer 3D information.
Con-
sistDreamer [39] proposes using 3D-consistent structured noise in 2D dif-
fusion denoising, which yields better multi-view consistency. InstantStyle-
Gaussian [40] transfers the Instruct-NeRF2NeRF [37] approach to the style
transfer task. ProEdit [41] employs a progressive editing strategy to reduce
the feasible space of text-aligned editing tasks, thereby mitigating multi-
view inconsistency. DGE [11] directly modifies the 2D model’s self-attention
into cross-view self-attention across keyframes and further applies epipolar-
constrained matching to propagate attention results into non-keyframe fea-
ture maps, significantly reducing memory demand during inference.
Ed-
itSplat [42] introduces multi-view fusion guidance, which aligns multi-view
information with text prompts and source images to ensure multi-view consis-
tency. By collecting a large scale illumination video dataset, RelightVid [43]
finetuned IC-Light to adapted to video relighting task, while Lumen [44]
training an end to end DiT-based model.
Our work builds on the improved epipolar-constrained matching scheme
of DGE [11] to achieve more consistent editing results under reduced memory
usage. To further exploit the powerful relighting capability of IC-Light, we
adapt IC-Light into the DGE framework without modifying or fine-tuning
its weights, enabling a high-quality, multi-view consistent relighting module
across multiple images, which we call MV-ICLight.
6

<!-- page 8 -->
These methods work well on the multiview images or videos editing or
relighting though, however, they can hardly tackle the challenge about the
unfaithful result related to the positional concept (e.g., light from right) input
by users.
2.3. 3DGS with Inverse Rendering / Physically Based Rendering
GI-GS [45], GUS-IR [46] take a set of pretrained 3D Gaussians with
normals and intrisics, then perform a differentiable PBR to model the di-
rect light and global illumination.
GS3 [47] presents spatial and angular
Gaussians with intrinsics and lighting/view directions, using mlps to pre-
dict shadow and global illumination. RNG [48] add an extra latent vector
that describes the reflectance for each Gaussian to model the surface with
soft boundaries like fur or fabric. GeoSplatting [49] proposes MGadapter to
differentiably construct a surface-grounded 3DGS from an optimizable mesh
guidance, enabling a precise light transport calculation.
These physically based rendering methods can achieve highly realistic re-
lighting effects; however, their computational cost — including both training
and rendering — is often very expensive. Moreover, these methods require
prior knowledge of the light source parameters, such as position, color, and
intensity. In our relighting task, however, such information must be inferred
from the user’s textual instructions, which is typically beyond the capability
of this class of methods.
2.4. Positional Alignment between Images and Text Prompts
LISA [50] and GLaMM [51] employ the SAM decoder to achieve segmen-
tation at the pixel level on reference images, whereas LLaVA Grounding [52]
extends the model with a dedicated grounding head to output bounding
boxes.
VPP-LLaVA [5] enhances MLLMs’ visual grounding by introduc-
ing visual position prompts and a curated 0.6M-sample dataset, achieving
state-of-the-art localization performance with strong zero-shot generaliza-
tion. WhatsUp [53] shows that despite strong performance on VQAv2 [54],
VL models struggle with basic spatial relations, and they introduce new
benchmarks to highlight this limitation.
3. Method
In this section, we present the details of GS-Light, our proposed pipeline
for text-driven relighting of 3D Gaussian Splatting (3DGS) scenes. Our goal
7

<!-- page 9 -->
estimate 
light position
render
Reference / Other Views
…
Position-Aligned Light Intensity
…
Relit Images
…
“detailed face, sunshine, 
indoor, warm atmosphere, 
light from left”
Text Prompt
fine-tuning
condition
init latents
multi-view relighting
Position-Align
Module
MV-ICLight
Pretrained GS Scene
Figure 2: Circulation pipeline of GS-Light. Starting from a pre-trained Gaussian Splatting
(GS) scene and a text prompt specifying the relighting instruction, we first render images
from all training views. One training view is selected as the reference view to align the
positional information in the prompt. Through our proposed Position-Align Module
(PAM), we generate position-aligned light intensity maps for all views. These intensity
maps are then provided as initialization latents to our Multi-View ICLight, producing
multi-view consistent relit images.
Finally, the relit images are used to fine-tune the
opacity and color parameters of the GS scene, forming a closed-loop tuning circulation.
Repeating this circulation multiple times ensures that the relit GS converges to a stable
and consistent result.
is to design a method that (i) faithfully adheres to user instructions expressed
in natural language, especially the light direction, (ii) ensures multi-view con-
sistency, and (iii) operates efficiently without requiring per-scene retraining.
To achieve this, GS-Light integrates a Position-Align Module (PAM) for
prompt–geometry alignment with an extended MV-ICLight for training-
free multi-view diffusion-based relighting.
An overview of the pipeline is
shown in Fig. 2.
3.1. Preliminaries
3.1.1. Diffusion Editors
Diffusion-based image editing models have achieved state-of-the-art per-
formance in tasks such as inpainting, illumination harmonization, and re-
lighting. A diffusion model [20, 4] learns the data distribution pdata(x) by
reversing a gradual noising process. Specifically, the forward process adds
8

<!-- page 10 -->
Gaussian noise to a clean image x0:
q(xt | x0) = N(xt; √¯αtx0, (1 −¯αt)I),
(1)
while the reverse process uses a neural network ϵθ(xt, t, c) to predict and re-
move noise under condition c, thereby generating or editing images consistent
with the guidance.
Editing with Diffusion. Diffusion-based editing methods can be broadly di-
vided into two categories. On the one hand, training-free approaches reuse a
pretrained diffusion model and perform modifications by initializing from a
noised version of an existing image x0, then denoising under a new condition
cedit:
x′
0 = Denoise(xt, cedit),
xt ∼q(xt | x0),
(2)
which enables preserving the structure of x0 while applying edits guided
by cedit. On the other hand, training-based approaches rely on finetuning
or retraining with triplet data (xsrc, cedit, xtgt) to directly learn the condi-
tional distribution p (xtgt|xsrc, cedit) for editing. In this work, we adopt the
IC-Light model, adhered to the latter paradigm, which is specifically de-
signed for illumination control and artistic lighting editing, and finetuned on
a large-scale and carefully curated dataset on the base of Stable Diffusion
v1.5 model. IC-Light introduces dedicated conditioning channels for illumi-
nation harmonization, enabling edits that respect global lighting style while
preserving content fidelity. However, being trained purely in the 2D image
domain, IC-Light lacks mechanisms to enforce cross-view consistency, which
limits its direct applicability to 3D scene editing.
3.1.2. 3D Gaussian Splatting
A 3D Gaussian distribution defines a probability density in 3D:
G(x) = e−1
2 (x−µ)⊤Σ−1(x−µ)
(3)
where µ ∈R3, Σ ∈R3×3 define the mean and covariance matrix.
3D Gaussian Splatting (3DGS) [2] represents a scene with a set of anisotropic
Gaussian primitives G = {(µi, Σi, ci, αi)}G
i=1 where µi, Σi determine each
Gaussian primitive’s distribution, with associated attributes color ci ∈[0, 1]3
and opacity αi ∈[0, 1]. During rendering, Gaussians are projected onto the
image plane and rasterized in tiles for parallel efficiency. For a pixel p, which
9

<!-- page 11 -->
is receiving contributions from a sorted set of Gaussians Gi along the view
direction, its color C is obtained via differentiable α-blending:
C(p) =
G
X
i=1
αiTici
(4)
where Ti = Q
j<i(1 −αj) is the accumulated transmittance from front to
back, G is the total number of Gaussians.
The entire process is fully differentiable. GS initializes primitives through
the Structure from Motion [55] (SfM) process and applies densification and
pruning during training based on gradient magnitudes in NDC coordinates
to control the number of Gaussians, making training and finetuning both
efficient and simple. GS-Light finetunes GS using relit images by freezing the
position and shape of each Gaussian and disabling the densification strategy,
which allows rapid optimization of scene appearance while avoiding memory
overhead issues.
3.2. Position-Align Module (PAM)
Input
"detailed face,
sunshine, indoor,
warm atmosphere"
"..., warm atmosphere,
light from left"
"..., warm atmosphere,
light from right"
Figure 3: IC-Light relighting results on different light direction instructions, which show
a weak or wrong response towards the position-related information.
WhatsUp [53] points out that existing multimodal large models have a
relatively weak semantic understanding of positional relationships in the tex-
tual modality. Experiments show that the same phenomenon also occurs in
text-to-image models, as illustrated in Fig. 3. The Position-Align Module is
responsible for bridging the semantic gap between text prompts and geomet-
ric cues, producing position-aware illumination maps that are later used to
initialize the diffusion process, which is shown in Fig. 4. PAM consists of 2
stages:
10

<!-- page 12 -->
VGGT & 
StableNormal
Lang-SAM
“detailed face, 
sunshine, indoor, 
warm atmosphere, 
light from left”
reference view
text prompt
Vision-Language Input
SYSTEM
Template:{
“Position”: #POS#,
“Object”: #OBJ#
}
USER
Tell me what light source 
position is relative to which 
objects in the edited image.
multi-model tokens
Preset VQA Template
inject
ASISTANT
{
“Position”: “left”,
“Object”: “face”
}
Qwen2.5-VL Answer
depths & normals
Rendered Views
segmentation
x
y
z
left offset
camera coordinate
init pos
light intensity
ref view
obj text
input flow
light offset flow
geometry flow
segmentation flow
output flow
Figure 4: Details of PAM. Given the rendered views and a text prompt, Qwen2.5-VL
is employed with a preset VQA template to parse the user’s intended lighting direction
and reference object. Pretrained models VGGT, StableNormal, and Lang-SAM are then
applied to estimate the initial light position and scene geometry.
By combining these
estimates with the parsed light-position offset, PAM produces light-intensity maps that
are spatially aligned with the input positional intent across all views.
3.2.1. Parsing Lighting Priors via LVLM
Given a user instruction, we employ a large vision-language model (LVLM)
such as Qwen2.5-vl [6] to extract structured lighting priors. The priors in-
clude lighting direction (e.g., light from the left) and reference objects (e.g.,
detailed face). To ensure robustness and avoid irrelevant outputs, we design
a constrained question–answer template that restricts the LVLM’s output
space to lighting-relevant descriptors only. This provides reliable, semanti-
cally grounded cues for subsequent relighting.
Given a 3DGS scene G, a set of training input views V = {vn}Nv
n=1, where
Nv is the number of training views, and the user-specified relighting instruc-
tion cedit, we begin by rendering the GS scene into each view to obtain multi-
view images I = {In}Nv
n=1 that contain scene information. We assume that
the illumination conditions and lighting direction described by the user are
referenced from a specific view vref. Without loss of generality, we consider
11

<!-- page 13 -->
vref ∈V.
To extract the user’s intended lighting direction and the corresponding
reference object, we employ a predefined Q&A template T that prompts the
LVLM to respond in a fixed format: Light is on the {DIRECTION} of the
{OBJECT}. With the fixed format, we can easily parse the direction and
object prompts with a simple regular expression matching.
pdir, pobj = QwenVL(Iref, ceidit, T )
(5)
Here, QwenVL refers to Qwen2.5-VL [6], a powerful open-source LVLM.
pdir ∈{left, right, top, bottom} is direction prompt and pobj is the reference
object prompt. By enumerating the possible cases of pdir, we can determine
the light source offset direction that matches the user’s intent.
Next, we also need to perform text–image alignment based on the ref-
erence object, in order to determine the initial position of the light source.
Using a pretrained segmentation foundation model, we can obtain the mask
of the reference object M obj:
M obj = LangSAM(Iref, pdir)
(6)
Here, LangSAM is a text-guided segmentation model that integrates
GroundingDINO [56] and SAM2.1 [9], and outputs high-quality object masks
based on the given prompt. With this mask, we can compute the pixel co-
ordinates of the reference object in the image pobj:
pobj =
1
|M obj|
X
p∈Mobj
p
(7)
which will be used for subsequent estimation of the light source position.
3.2.2. Geometric and Semantic Understanding & Latent Initialization
Geometry Estimation. To complement text-based priors, we estimate per-
view scene geometry and semantics using off-the-shelf models. Scale-aligned
depth maps D = {Dn}Nv
n=1 are obtained via VGGT [7], which is a general
framework designed for multi-view geometry perception with an end-to-end
transformer-based architecture.
Then surface normals N = {Nn}Nv
n=1 are
estimated via StableNormal [8], which predicts more smooth and proper
normals on the base of a powerful generative diffusion prior model, rather
than directly inferring from depth maps D.
12

<!-- page 14 -->
∆init
∆𝑝𝑝dir
init pos
estimated pos
object segmentation
ref camera coordinate
𝑧𝑧
𝑥𝑥
𝑦𝑦
ref view
Gaussians
LVLM Answer
Figure 5: The process of estimating light source position.
Semantic Understanding. As mentioned earlier, we can use LangSAM to
align the reference image Iref with the user’s instructions ceidit, thereby esti-
mating the reference object’s pixel coordinates pobj. Considering that light-
ing usually comes from above, we assume here that the initial light source
position is located diagonally above on the side of the reference object fac-
ing the camera (the exact initial position is not critical; what matters is the
change in the light source position before and after relighting). Following the
convention of 3DGS, we adopt the OpenCV camera coordinate system (with
the x, y, z axes of the camera pointing to the right, downward, and into the
screen, respectively):
pl =
pobj
dref

+ ∆init
(8)
where pl is the light position in camera coordinate, dref is the depth at pixel
coordinate pobj. ∆init is the relative offset to initialize the light position.
Based on the direction prompt pdir provided by the LVLM, we can add a
corresponding offset ∆pdir to the initial light source position so that it aligns
with the user’s description of the light source direction:
p′
l = pl + ∆pdir
(9)
To better understand the light position estimation process, we illustrate
it as Fig. 5.
Latent Initialization. With geometry and light source ready, we can the com-
pute the distribution of light intensity across the views by a modified Phong-
13

<!-- page 15 -->
like diffuse illuminating model [10]:
Id = max(−⟨lin, n⟩, 0)γ
(10)
where Id is the diffuse part reflected by the surface, which indicates the
light intensity. lin denotes incident direction and n denotes surface normal,
⟨·, ·⟩is inner product operation, γ is a hyperparameter to balance the bright-
ness distribution.
The illumination intensity maps {Id}Nv
n=1 serve as initialization signals in
the diffusion process.
Specifically, we encode the illumination maps into
latent space and inject them into IC-Light’s denoising steps as init latents ld
ld = VAE (Id)
(11)
where VAE is the image encoder of SD-1.5. This provides explicit geometry-
aware lighting conditioning, allowing the model to align its generation trajec-
tory with the desired lighting direction and intensity, significantly reducing
prompt–output mismatch.
3.3. MV-ICLight
Although IC-Light is renowned for its powerful image relighting capabil-
ity, like other 2D image editing models, it also struggles to ensure consistency
in the outputs when given multi-view inputs. (Insert Image). To extend IC-
Light from single-view to multi-view editing, we introduce MV-ICLight,
which enforces cross-view coherence during relighting.
3.3.1. Cross-View Attention
To enable multi-view editing, it is necessary for the model to aggregate
global information from multi-view inputs during inference, thereby estab-
lishing a mechanism for inter-view information exchange and supplementa-
tion. A natural idea is to replace the self-attention in the Diffusion U-Net
with cross-view attention, as shown in the upper part of Fig. 6. Specifically,
for a given set of query and key tokens {Q}F
f=1, {K}F
f=1 from the f-th frame,
each of Qf can perceive all tokens from other frames, instead of only the
same frame which is the case of traditional self-attention. The Cross-View
Attention Score can be formulated as:
CVAttn (Q, K, f) = Softmax
Qf · [K1, · · · , KF]
√
d

(12)
where d is the dimension of the query and key embeddings.
14

<!-- page 16 -->
…
…
…
…
…
tokenize
Key frames
Rest frames
𝑓𝑓views
𝑓𝑓× 𝑝𝑝× 𝐶𝐶
𝑝𝑝tokens
reshape
…
…
…
…
1 × 𝑓𝑓𝑝𝑝× 𝐶𝐶
self attention
& reshape
…
…
2 closest
key frames
max similarity along
epipolar line
interpolate
Input features
Output features
Self-attn batch
Frame of interest
Pixel of interest
Epipolar line
Correspondence
Figure 6: Schematic diagram of self attention in MV-ICLight. Upper part: the implemen-
tation mechanism of Cross-View Attention. Lower part: epipolar constrain for non-key
frames from DGE [11].
3.3.2. Multi-View Relighting with Advanced Epipolar Constraints
Directly extending self-attention across multiple frames will significantly
increase computational and memory overhead, since the cost of self-attention
grows quadratically with the number of input tokens. Inherited from DGE [11],
we subsample a few key frames from all training views to perform a full-size
inference and implement an epipolar-matching mechanism to fill up the fea-
ture maps of the rest frames, as shown in the lower part of Fig. 6. We keep the
notations from Sec. 3.2.1 and DGE. Given a set of key frames K ⊂V, where
V is the training set, a feature map Ψv′ corresponding to image Iv′, where
v′ /∈K, and features Ψk∗of the nearest keyframe Ik∗, the correspondence
map Mv′ is given by:
Mv′ [pu] =
argmin
pv,pTv ˆFpu=0
D (Ψv′ [pu] , Ψk∗[pv])
(13)
where D is the cosine distance, pu and pv is the pixel-wise index, k∗is
the key view closest to view v′, and unlike DGE, ˆF =
F
∥F∥+ϵ is the normalized
fundamental matrix corresponding to the two views v′ and k∗, where ϵ is a
small constant introduced to avoid division-by-zero overflow.
Since the fundamental matrix F is defined up to an arbitrary nonzero
15

<!-- page 17 -->
scale factor, we normalized it to have unit norm to ensure numerical sta-
bility during inference. Experiments demonstrate that, as shown in Tab. 6,
by using our normalized fundamental matrix ˆF, the failure of epipolar con-
straints in the shallow attention blocks of the UNet is largely avoided. As
a result, inference stability is greatly improved, and the generated results
exhibit significantly better multi-view consistency.
During inference, as shown in Eq. 11, the spatially aligned illumination
diffuse map is encoded into latents ld, which are then noised up to timestep
T ′(< Tmax), as the starting point for denoising, where Tmax is the total de-
noising steps number. The latents representation of the original images lI, is
also used as a condition to keep the geometry and details of results consistent
with original images, and concatenated with ld along the channel dimension:
l
H
8 × H
8 ×2C
i
= Concat(l
H
8 × H
8 ×C
d
, l
H
8 × H
8 ×C
I
)
(14)
where li is the init noised latents to perform partial DDIM [21] pipeline, H, W
is resolution of source images and C is the channel dimension of latents.
3.3.3. GS Tuning and Iterative Dataset Updating Strategy
Through MV-ICLight, the relit images across multiple views indeed achieve
better consistency compared to independently relighting each view, though
slight inconsistencies still remain. To address this, we adopt the Iterative
Dataset Update strategy from IN2N [37]: after every Kint steps, the GS
scene is rendered from all viewpoints and relit using MV-ICLight, and the
resulting images are directly updated as the training data for the correspond-
ing views. This process is repeated Kreap times. The GS scene eventually
converges to a multi-view consistent relighting result, where Kint and Kreap
are hyperparameters controlling the update interval and number of iterations
during fine-tuning.
To apply the multi-view relighting edits to the GS-represented scene, we
directly use the edited results as the training set to fine-tune the Gaussians:
G′ =
argmin
ci,αi∈Gi,∀Gi∈G
X
v∈V
L (Render(G, v), Iv
relit)
(15)
where G′ is the final relit GS scene, L is the loss function of 3DGS train-
ing, typically composed of an L1 loss and an SSIM loss, Render(G, v) is the
Gaussian rendering at viewpoint v, and Iv
relit is the relit image generated from
MV-ICLight.
16

<!-- page 18 -->
4. Experiments
In this section, we first present the implementation details of our method,
followed by both qualitative and quantitative comparisons with existing ap-
proaches. We then conduct an ablation study to further analyze the effec-
tiveness of our method.
Datasets. The relighting tasks are performed on several datasets of indoor
and outdoor scenes, including IN2N [37] dataset, MipNerf360 [57], Scan-
net++ [58], etc., where 3D Gaussian Splatting models (or dense multi-view
imagery) are available.
We use the GPT-5 [59] model to generate var-
ious scene-related relighting instructions, and each instruction is iterated
over possible lighting directions pdir ∈{left, right, top, bottom} to construct
the benchmark.
Specifically, we select all scenes from IN2N and MipN-
eRF360, along with the first 10 scenes from ScanNet++, resulting in a total
of 25 scenes. For each scene, we assign three randomly generated relighting
prompts and random lighting directions, leading to 75 relighting tasks in
total, which constitute our benchmark. We also conducted a user study to
collect participants’ preferences regarding the relighting results produced by
different models.
Implementation Details. Under our 2D-to-MV framework, the capability of
the 2D image editor largely determines the quality of the final 3D editing
results. For fairness, we adopt several prior works based on IP2P [23] for 3D
scene editing and IC-Light [3] for scene/video relighting as our baselines. To
control the memory consumption during inference, we use 50–70 input images
of 512×512 resolution for all datasets. We adopt classifier-free guidance with
a text guidance scale of 7.5, applied throughout the generation process. For
the Dataset Update Strategy, we set the update interval Kint to 500 and the
number of updates Kreap to 2, consistent with DGE [11]. During Gaussian
Splatting fine-tuning, we disable gradient computation for all parameters
except color and opacity, and also deactivate the densification strategy.
Evaluation. We conduct both qualitative and quantitative evaluations of our
method. For quantitative analysis, following DGE, we evaluate CLIP score
and CLIP directional score (noted as CLIP-T and CLIP-D) between ren-
dered images and target text prompt to measure the alignment of 3D relit
scenes and instruction. In addition, metrics such as PSNR, SSIM, and LPIPS
are commonly used to evaluate how well a Gaussian Splatting (GS) scene fits
17

<!-- page 19 -->
Table 1: Quantitative comparison on IN2N [37] dataset.
Best and second results are
highlighted in bold and with underline.
Method
CLIP-T↑CLIP-D↑
VBench
Subject
Background
Aesthetic
Image
Consistency↑Consistency↑
Quality↑
Quality↑
Relighting on Videos
RelightVid
0.2383
0.0611
0.7806
0.8624
0.5991
61.31
Lumen*
0.1754
-0.0336
0.7215
0.8336
0.4720
53.88
Ours
0.2580
0.1170
0.8320
0.8884
0.6317
62.22
Relighting on Gaussian Splatting
DGE
0.2369
0.0918
0.8614
0.9017
0.5619
60.65
EditSplat
0.1983
-0.0021
0.8730
0.9094
0.5203
53.94
IN2N
0.2222
0.0410
0.8757
0.9098
0.5425
51.08
IGS2GS†
0.2055
0.0111
0.8594
0.9268
0.4249
30.27
IGS2GS-IC‡
0.1800
-0.0102
0.8642
0.9330
0.3938
26.76
Ours
0.2580
0.1171
0.8748
0.9270
0.6043
57.80
* the generating resolution setting of benchmark is 512×512, however the Lumen’s recom-
mended resolution is 480×832, which may cause a performance decline.
† denotes Instruct-GS2GS, an improved version adapted from IN2N for GS training, using
InstructPix2Pix [23] as the 2D image editor.
‡ denotes replacing the 2D image editor InstructPix2Pix with IC-Light [3].
the corresponding real-world scene. These metrics can also partially reflect
the multi-view consistency of rendered images. However, in our experiments,
we observed that when fine-tuning a pretrained GS scene, these metrics tend
to favor models with smaller editing degrees, since less-edited supervision
images make it easier for the GS to converge toward high multi-view consis-
tency—closer to that of the original training images, which is unfair to mod-
els that produce more complex and visually refined results. Therefore, we
omit these metrics in method comparison when evaluating the final relight-
ing performance, and instead adopt VBench [60], an end-to-end video quality
assessment framework. We select the following indicators—Subject Con-
sistency, Background Consistency, Aesthetic Quality, and Imaging
Quality—to comprehensively evaluate the relighting results of different mod-
els. It is worth noting that since VBench is video-based, we need to provide
a continuous video as input. For GS-based relighting methods, we generate a
smooth camera trajectory by interpolating between the training viewpoints
for each scene in the benchmark, and then render videos along these trajec-
tories. For methods that perform relighting directly on videos, we take the
18

<!-- page 20 -->
Table 2: Quantitative comparison on MipNerf360 [57] and Scannet++ [58] dataset. Best
and second results are highlighted in bold and with underline.
Method
MipNerf360
Scannet++
CLIP-T↑CLIP-D↑
S.C.↑
B.C.↑
A.Q.↑
I.Q.↑CLIP-T↑CLIP-D↑
S.C.↑
B.C.↑
A.Q.↑
I.Q.↑
Relighting on Videos
RelightVid
0.2346
0.0578
0.7260
0.8558
0.5593
61.16
0.2377
0.0267
0.6885 0.8494 0.5688 67.01
Lumen
0.1865
-0.0198
0.6108
0.8007
0.4180
45.62
0.2011
-0.0624
0.5687
0.7900
0.4275
43.17
Ours
0.2403
0.0883
0.7386 0.8684 0.5834 66.93
0.2309
0.0658
0.6636
0.8478
0.5447
57.08
Relighting on Gaussian Splatting
DGE
0.2304
0.0810
0.8343
0.8944
0.5888
69.71
0.2452
0.0551
0.8550
0.9236
0.5213
39.28
EditSplat
0.2201
0.0363
0.8349
0.8898
0.5571
62.44
0.2291
0.0163
0.8767
0.9261
0.5544
43.36
IN2N
0.2264
0.0433
0.8674
0.9118
0.5210
49.21
0.2142
-0.0135
0.8810
0.9225
0.5309
46.73
IGS2GS
0.2160
0.0422
0.8633
0.9184
0.4769
39.86
0.2243
0.0195
0.8628
0.9238
0.5079
38.46
IGS2GS-IC
0.2064
0.0251
0.8754 0.9278
0.4473
28.86
0.1977
-0.0287
0.8715
0.9233
0.4628
24.82
Ours
0.2402
0.0881
0.8421
0.9050
0.6047 68.81
0.2308
0.0658
0.8629
0.9223
0.5792 50.62
relit frames corresponding to the training viewpoints, sort them by viewing
direction, concatenate them into a single video, and then evaluate it using
VBench.
4.1. Comparisons with Prior Work
Our baselines include state-of-the-art 3D editing methods such as IN2N [37],
DGE [11], and EditSplat [42], as well as video relighting approaches includ-
ing RelightVid [43] and Lumen [44]. In addition, we develop an IC-Light [3]-
based variant of IGS2GS, which adapts IN2N to Gaussian Splatting scenes,
replacing the original 2D image editor InstructPix2Pix [23].
We present qualitative comparisons between our method and baselines
on the IN2N, Mip-NeRF360, and ScanNet++ datasets to visually assess the
effectiveness of our framework. In Fig. 1, our approach produces multi-view
consistent edits that faithfully reflect the user-provided lighting condition,
while maintaining high-quality geometry and texture details across views.
As shown in Tab. 1 and Tab. 2, our method consistently outperforms prior
approaches across all datasets and metrics. Specifically, our model achieves
the highest scores while keeping the inference process in few minutes, indicat-
ing superior visual fidelity, cross-view consistency, semantic alignment and
time consuming.
Tab. 3 presents the preferences of 29 users for GS relighting methods
shown in Fig. 1, based on dimensions such as subjective preference, image-
text matching, and lighting/artistic effects.
The data indicate that our
method is favored by users.
In summary, compared with prior methods, our method generates more
19

<!-- page 21 -->
Table 3: User preference study for various gaussian splatting relighting methods.
We
collected 29 users’ rankings of relighting results from different models along three di-
mensions—subjective preference, artistic aesthetics, and lighting-control consistency—and
computed the average rank.
Method
Avg. Rank (#) ↓
Subjective Preference
Artistic Aesthetics
Lighting-Control Consistency
DGE
2.15
2.23
2.31
EditSplat
3.23
3.28
3.48
IN2N
4.15
4.15
4.09
IGS2GS
5.45
5.28
5.05
IGS2GS-IC
4.36
4.26
4.01
Ours
1.57
1.69
1.87
Table 4: Component-wise ablation on IN2N dataset. Each component is removed progres-
sively to analyze its contribution.
PAM
CV-Attn
IC-Light
CLIP-T↑
CLIP-D↑
S.C.↑
B.C.↑
A.Q.↑
I.Q.↑
✓
✓
✓
0.2580
0.1170
0.8748
0.9270
0.6043
57.80
×
✓
✓
0.2467
0.1115
0.8734
0.9243
0.6130
58.22
×
×
✓
0.2667
0.1188
0.8631
0.9160
0.5957
56.36
×
×
×
0.2369
0.0918
0.8614
0.9016
0.5619
60.65
realistic lighting and color adjustments, preserves fine-grained scene struc-
tures, and exhibits fewer artifacts in occluded or texture-rich regions.
4.2. Ablation Study
Next, we conduct an ablation study to evaluate the effectiveness of several
key components in our editing pipeline: IC-Light Relighting Model, Multi-
View Consistent Inference and Position-Align Module.
Tab. 4 shows the
component-wise quantitative ablation study and then qualitative result of
each component will be present.
Component-wise Ablation. We sequentially remove key components
from our full pipeline, position-alignment module (w/o PAM), cross-view
attention module (w/o CV-Attn), and IC-Light as the 2D image editor (w/o
IC-Light). Table 4 reports the corresponding metrics scores on the IN2N
dataset. Removing PAM and CV-Attn leads to a noticeable drop in semantic
alignment and multi-view consistency, which shows the effectiveness of our
proprosed modules. However, we observed an abnormal increase in the CLIP
score when only IC-Light is used. We speculate that this may be because
the CLIP score emphasizes the overall alignment between the image and the
20

<!-- page 22 -->
Table 5: Multi-view consistency ablation on Cross-view Attention. We evaluate PSNR,
SSIM, LPIPS for renderings of relit GS scene on IN2N dataset.
Method
PSNR↑
SSIM↑
LPIPS↓
w/o CV-Attn
13.12
0.4602
0.4253
w/ CV-Attn
18.88
0.6267
0.2549
text, while lacking training objectives related to lighting consistency. The
Imaging Quality metric exhibits relatively “random” behavior on this task,
which may be because it focuses more on image sharpness rather than lighting
effects—an aspect that our proposed module is not designed to address.
Relighting without Multi-View Consistency. To evaluate the ef-
fectiveness of our multi-view consistent relighting, we replace it with inde-
pendent per-view relighting for the same set of views and use those results
to fine-tune Gaussians. As shown in Tab. 5, incorporating multi-view con-
sistency significantly improves reconstruction metrics such as PSNR, SSIM
and LPIPS, demonstrating the advantage of maintaining cross-view coher-
ence. Furthermore, Fig. 7 illustrates intermediate 2D relighting results with
and without multi-view consistency. It can be clearly observed that con-
sistent relighting leads to visually coherent appearances across views, while
independent relighting causes noticeable discrepancies among them.
Effectiveness of Position-Align Module. We present the relighting
results of different methods when the input instructions include lighting posi-
tion information, as shown in Fig. 8. Experimental results demonstrate that,
with the help of the PAM module, our method can more accurately capture
the lighting direction described in the editing instruction, thereby producing
relighting results that are more faithful to the user’s intent.
Fundamental Matrix Normalization. In practice, we found that the
epipolar constraint frequently fails during UNet inference, which is typically
caused by numerical overflow. To further investigate the cause, we take the
face scene from IN2N as an example. Fig. 9 (a) shows the proportion of pixels
across different UNet layers (i.e., different resolutions) and the ratio of those
affected by overflow. We observe that nearly half of the pixels experience
numerical overflow, severely disrupting the process of finding the maximum
value along the epipolar line. As shown in Fig. 9 (b), regions with numerical
instability cause the search range of the epipolar constraint to shrink, leading
to suboptimal matches, and in more severe cases, the matching process may
21

<!-- page 23 -->
inputs / instruction
relit images
GS renders
MV-ICLight
"detailed face, sunshine, indoor,
warm atmosphere"
IC-Light
MV-ICLight
"detailed bear statue, marble,
twilight, golden autumn forest,
sunset glory"
IC-Light
MV-ICLight
"detailed clear face, cool tunes,
stage lighting, blue spotlight"
IC-Light
MV-ICLight
"office desk, indoor, night scene,
cool fluorescent light"
IC-Light
MV-ICLight
"bonsai tree, indoor, warm desk
lamp illumination, cozy
atmosphere"
IC-Light
Figure 7: Ablation study of multi-view consistency. For each scene, first row shows the
relit images and renders from fine-tuned GS scene with our MV-ICLight, while second
row shows the ones with vanilla IC-Light. Our MV-ICLight shows a great multi-view
consistency between sampled perspectives which contributes to divergency of GS finetun-
ing. On the other hand, vanilla IC-Light without any multi-view consistency constrain
generates diverse relit images, which results a degrade and blurry GS rendering.
22

<!-- page 24 -->
instruction
inputs / init latents
with PAM
w/o PAM
"detailed face, sunshine,
indoor, warm
atmosphere,"
"..., light from left"
"..., light from right"
"detailed bear statue,
marble, twilight, golden
autumn forest,"
"..., sunset glory from left
side"
"..., sunset glory from
right side"
"young man, detailed face,
natural lighting, outdoor,
warm,"
"..., light from the top"
"..., light from the
bottom"
Figure 8: Visualization of PAM component. For each scene, first row shows the input
images and common instruction, as well as the depths and normals estimated from pre-
trained models. Next, each line represents supplementary instructions for different light
directions, sequentially displaying the corresponding initial latents and the relighting re-
sults of whether the PAM component is used.
23

<!-- page 25 -->
Pixel of interest
Epipolar line
Overflow area
@322
(a)
(b) 
Inner : Pixel 
Number Proportion
Outer : Overflow Rate
Figure 9: (a) Statistics of overflow occurrences across different resolutions (UNet layers);
(b) Visualization of how overflow disrupts the epipolar constraints (at 322 resolution)
degenerate into global matching. Fortunately, we discovered that the overflow
issue originates from the lack of normalization in the fundamental matrix
computed by PyTorch, resulting in excessively large values (up to the order
of 106) in the matrix F. Tab. 6 reports the average number of UNet layers
experiencing overflow per inference before and after normalization, as well as
the relighting results in terms of PSNR, SSIM and LPIPS. The experiments
demonstrate that our improved normalized fundamental matrix significantly
enhances the numerical stability of the inference process and improves the
multi-view consistency of the editing results.
Table 6: Ablation on epipolar constraint normalization of fundamental matrix. We eval-
uate on face scene from IN2N dataset.
Method
Avg. Overflow Rate↓
PSNR↑
SSIM↑
LPIPS↓
w/o normalization
49.82%
20.81
0.7312
0.1621
w/ normalization
0.00%
21.55
0.7328
0.1599
These ablation studies collectively demonstrate the importance of our
proposed modules and design choices in achieving semantically faithful and
geometrically coherent 3D scene edits.
5. Limitations and Future Work
There are several limitations to GS-Light. First, our reliance on off-the-
shelf geometry / normal estimators means that if depth or normal maps are
24

<!-- page 26 -->
inaccurate (e.g. due to occlusions, reflective or transparent surfaces), the
lighting fusion and shadow estimation may fail or artifacts may arise. Sec-
ond, the projection back into the 3DGS representation may leave uncovered
regions (views or surfaces not well seen in images), leading to inconsistency
or blurring. Third, strongly specular or anisotropic materials are difficult to
handle in inference only pipelines without BRDF fitting.
Future work could include integrating lightweight material priors or BRDF
estimation; extending LVLM prior extraction to more complex lighting (mul-
tiple lights, colored ambient, etc.); better handling of occlusion and shadows
via differentiable visibility; possibly allowing optional per-scene fine-tuning
when higher fidelity is required.
6. Conclusion
We presented GS-Light, a training-efficient, text-guided, position-aware
method for scene relighting in Gaussian Splatting representations. By com-
bining prompt-derived lighting priors and view-consistency constraints, our
pipeline generates multi-view coherent relit images and relit 3D scenes, espe-
cially in lighting directions, outperforming several baselines while operating
purely at inference time. We believe GS-Light provides a useful step toward
more accessible, controllable, and consistent relighting for 3D content.
References
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, R. Ng, Nerf: Representing scenes as neural radiance fields for view
synthesis, Communications of the ACM 65 (1) (2021) 99–106.
[2] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3d gaussian splat-
ting for real-time radiance field rendering., ACM Trans. Graph. 42 (4)
(2023) 139–1.
[3] L. Zhang, A. Rao, M. Agrawala, Scaling in-the-wild training for
diffusion-based illumination harmonization and editing by imposing con-
sistent light transport, in: The Thirteenth International Conference on
Learning Representations, 2025.
[4] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, High-
resolution image synthesis with latent diffusion models, in: Proceedings
25

<!-- page 27 -->
of the IEEE/CVF conference on computer vision and pattern recogni-
tion, 2022, pp. 10684–10695.
[5] W. Tang, Y. Sun, Q. Gu, Z. Li, Visual position prompt for mllm based
visual grounding, arXiv preprint arXiv:2503.15426 (2025).
[6] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
S. Wang, J. Tang, et al., Qwen2. 5-vl technical report, arXiv preprint
arXiv:2502.13923 (2025).
[7] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny,
Vggt: Visual geometry grounded transformer, in: Proceedings of the
Computer Vision and Pattern Recognition Conference, 2025, pp. 5294–
5306.
[8] C. Ye, L. Qiu, X. Gu, Q. Zuo, Y. Wu, Z. Dong, L. Bo, Y. Xiu, X. Han,
Stablenormal: Reducing diffusion variance for stable and sharp normal,
ACM Transactions on Graphics (TOG) 43 (6) (2024) 1–18.
[9] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rä-
dle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Car-
ion, C.-Y. Wu, R. Girshick, P. Dollár, C. Feichtenhofer, Sam 2: Segment
anything in images and videos, arXiv preprint arXiv:2408.00714 (2024).
URL https://arxiv.org/abs/2408.00714
[10] B. T. Phong, Illumination for computer generated pictures, in: Seminal
graphics: pioneering efforts that shaped the field, 1998, pp. 95–101.
[11] M. Chen, I. Laina, A. Vedaldi, Dge: Direct gaussian 3d editing by consis-
tent multi-view editing, in: European Conference on Computer Vision,
Springer, 2024, pp. 74–92.
[12] W. Peebles, S. Xie, Scalable diffusion models with transformers, in: Pro-
ceedings of the IEEE/CVF international conference on computer vision,
2023, pp. 4195–4205.
[13] P. Debevec, T. Hawkins, C. Tchou, H.-P. Duiker, W. Sarokin, M. Sagar,
Acquiring the reflectance field of a human face, in:
Proceedings of
the 27th annual conference on Computer graphics and interactive tech-
niques, 2000, pp. 145–156.
26

<!-- page 28 -->
[14] Z. Xu, K. Sunkavalli, S. Hadap, R. Ramamoorthi, Deep image-based
relighting from optimal sparse samples, ACM Transactions on Graphics
(ToG) 37 (4) (2018) 1–13.
[15] S. Sengupta, A. Kanazawa, C. D. Castillo, D. W. Jacobs, Sfsnet: Learn-
ing shape, reflectance and illuminance of facesin the wild’, in: Proceed-
ings of the IEEE conference on computer vision and pattern recognition,
2018, pp. 6296–6305.
[16] Z. Shu, S. Hadap, E. Shechtman, K. Sunkavalli, S. Paris, D. Samaras,
Portrait lighting transfer using a mass transport approach, ACM Trans-
actions on Graphics (TOG) 36 (4) (2017) 1.
[17] T. Sun, J. T. Barron, Y.-T. Tsai, Z. Xu, X. Yu, G. Fyffe, C. Rhemann,
J. Busch, P. E. Debevec, R. Ramamoorthi, Single image portrait relight-
ing., ACM Trans. Graph. 38 (4) (2019) 79–1.
[18] T. Nestmeyer, J.-F. Lalonde, I. Matthews, A. Lehrmann, Learning
physics-guided face relighting under directional light, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, 2020, pp. 5124–5133.
[19] H. Kim, M. Jang, W. Yoon, J. Lee, D. Na, S. Woo, Switchlight: Co-
design of physics-driven architecture and pre-training framework for hu-
man portrait relighting, in: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 25096–25106.
[20] P. Dhariwal, A. Nichol, Diffusion models beat gans on image synthesis,
Advances in neural information processing systems 34 (2021) 8780–8794.
[21] J. Song, C. Meng, S. Ermon, Denoising diffusion implicit models, arXiv
preprint arXiv:2010.02502 (2020).
[22] G. Kim, T. Kwon, J. C. Ye, Diffusionclip: Text-guided diffusion mod-
els for robust image manipulation, in: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp. 2426–
2435.
[23] T. Brooks, A. Holynski, A. A. Efros, Instructpix2pix: Learning to follow
image editing instructions, in: Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2023, pp. 18392–18402.
27

<!-- page 29 -->
[24] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, D. Cohen-Or, Null-text
inversion for editing real images using guided diffusion models, in: Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2023, pp. 6038–6047.
[25] A. Hertz, A. Voynov, S. Fruchter, D. Cohen-Or, Style aligned image gen-
eration via shared attention, in: Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2024, pp. 4775–4785.
[26] T. Qi, S. Fang, Y. Wu, H. Xie, J. Liu, L. Chen, Q. He, Y. Zhang,
Deadiff: An efficient stylization diffusion model with disentangled rep-
resentations, in: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2024, pp. 8693–8702.
[27] H. Wang, M. Spinelli, Q. Wang, X. Bai, Z. Qin, A. Chen, Instantstyle:
Free lunch towards style-preserving in text-to-image generation, arXiv
preprint arXiv:2404.02733 (2024).
[28] H. Wang, P. Xing, R. Huang, H. Ai, Q. Wang, X. Bai, Instantstyle-plus:
Style transfer with content-preserving in text-to-image generation, arXiv
preprint arXiv:2407.00788 (2024).
[29] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, K. Schindler,
Repurposing diffusion-based image generators for monocular depth es-
timation, in: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2024, pp. 9492–9502.
[30] J. Li,
H. Tan,
K. Zhang,
Z. Xu,
F. Luan,
Y. Xu,
Y. Hong,
K. Sunkavalli, G. Shakhnarovich, S. Bi, Instant3d: Fast text-to-3d with
sparse-view generation and large reconstruction model, arXiv preprint
arXiv:2311.06214 (2023).
[31] M. Ren, W. Xiong, J. S. Yoon, Z. Shu, J. Zhang, H. Jung, G. Gerig,
H. Zhang, Relightful harmonization:
Lighting-aware portrait back-
ground replacement, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 6452–6462.
[32] Y. Mei, J. Xu, V. M. Patel, Reference-based controllable scene styliza-
tion with gaussian splatting, arXiv preprint arXiv:2407.07220 (2024).
28

<!-- page 30 -->
[33] K. Zhang, N. Kolkin, S. Bi, F. Luan, Z. Xu, E. Shechtman, N. Snavely,
Arf: Artistic radiance fields, in: European Conference on Computer
Vision, Springer, 2022, pp. 717–733.
[34] L. A. Gatys, A. S. Ecker, M. Bethge, Image style transfer using con-
volutional neural networks, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2016, pp. 2414–2423.
[35] Á. S. Kovács, P. Hermosilla, R. G. Raidou, G-style: Stylized gaussian
splatting, in: Computer Graphics Forum, Vol. 43, Wiley Online Library,
2024, p. e15259.
[36] K. Liu, F. Zhan, M. Xu, C. Theobalt, L. Shao, S. Lu, Stylegaussian:
Instant 3d style transfer with gaussian splatting, in: SIGGRAPH Asia
2024 Technical Communications, 2024, pp. 1–4.
[37] A. Haque, M. Tancik, A. A. Efros, A. Holynski, A. Kanazawa, Instruct-
nerf2nerf:
Editing 3d scenes with instructions, in:
Proceedings of
the IEEE/CVF international conference on computer vision, 2023, pp.
19740–19750.
[38] J. Dong, Y.-X. Wang, Vica-nerf: View-consistency-aware 3d editing of
neural radiance fields, Advances in Neural Information Processing Sys-
tems 36 (2023) 61466–61477.
[39] J.-K. Chen, S. R. Bulo, N. Müller, L. Porzi, P. Kontschieder, Y.-X.
Wang, Consistdreamer: 3d-consistent 2d diffusion for high-fidelity scene
editing, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 21071–21080.
[40] X.-Y. Yu, J.-X. Yu, L.-B. Zhou, Y. Wei, L.-L. Ou, Instantstylegaussian:
Efficient art style transfer with 3d gaussian splatting, arXiv preprint
arXiv:2408.04249 (2024).
[41] J.-K. Chen, Y.-X. Wang, Proedit: Simple progression is all you need for
high-quality 3d scene editing, Advances in Neural Information Process-
ing Systems 37 (2024) 4934–4955.
[42] D. I. Lee, H. Park, J. Seo, E. Park, H. Park, H. D. Baek, S. Shin, S. Kim,
S. Kim, Editsplat: Multi-view fusion and attention-guided optimization
29

<!-- page 31 -->
for view-consistent 3d scene editing with 3d gaussian splatting, in: Pro-
ceedings of the Computer Vision and Pattern Recognition Conference,
2025, pp. 11135–11145.
[43] Y. Fang, Z. Sun, S. Zhang, T. Wu, Y. Xu, P. Zhang, J. Wang, G. Wet-
zstein, D. Lin, Relightvid: Temporal-consistent diffusion model for video
relighting, arXiv preprint arXiv:2501.16330 (2025).
[44] J. Zeng, Y. Liu, Y. Feng, C. Miao, Z. Gao, J. Qu, J. Zhang,
B. Wang, K. Yuan, Lumen: Consistent video relighting and harmonious
background replacement with video generative models, arXiv preprint
arXiv:2508.12945 (2025).
[45] H. Chen, Z. Lin, J. Zhang, Gi-gs: Global illumination decomposition on
gaussian splatting for inverse rendering, arXiv preprint arXiv:2410.02619
(2024).
[46] Z. Liang, H. Li, K. Jia, K. Guo, Q. Zhang, Gus-ir:
Gaussian
splatting with unified shading for inverse rendering, arXiv preprint
arXiv:2411.07478 (2024).
[47] Z. Bi, Y. Zeng, C. Zeng, F. Pei, X. Feng, K. Zhou, H. Wu, Gs3: Ef-
ficient relighting with triple gaussian splatting, in: SIGGRAPH Asia
2024 Conference Papers, 2024, pp. 1–12.
[48] J. Fan, F. Luan, J. Yang, M. Hasan, B. Wang, Rng: Relightable neural
gaussians, in: Proceedings of the Computer Vision and Pattern Recog-
nition Conference, 2025, pp. 26525–26534.
[49] K. Ye, C. Gao, G. Li, W. Chen, B. Chen, Geosplating: Towards ge-
ometry guided gaussian splatling for physically-based inverse rendering
(2024).
[50] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, J. Jia, Lisa: Rea-
soning segmentation via large language model, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 9579–9589.
[51] H. Rasheed, M. Maaz, S. Shaji, A. Shaker, S. Khan, H. Cholakkal, R. M.
Anwer, E. Xing, M.-H. Yang, F. S. Khan, Glamm: Pixel grounding large
30

<!-- page 32 -->
multimodal model, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 13009–13018.
[52] H. Zhang, H. Li, F. Li, T. Ren, X. Zou, S. Liu, S. Huang, J. Gao,
Leizhang, C. Li, et al., Llava-grounding: Grounded visual chat with
large multimodal models, in: European Conference on Computer Vision,
Springer, 2024, pp. 19–35.
[53] A. Kamath, J. Hessel, K.-W. Chang, What’s" up" with vision-language
models?
investigating their struggle with spatial reasoning, arXiv
preprint arXiv:2310.19785 (2023).
[54] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, D. Parikh, Making
the v in vqa matter: Elevating the role of image understanding in visual
question answering, in: Proceedings of the IEEE conference on computer
vision and pattern recognition, 2017, pp. 6904–6913.
[55] J. L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[56] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li,
J. Yang, H. Su, et al., Grounding dino: Marrying dino with grounded
pre-training for open-set object detection, in: European conference on
computer vision, Springer, 2024, pp. 38–55.
[57] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman,
Mip-nerf 360: Unbounded anti-aliased neural radiance fields, in: Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[58] C. Yeshwanth, Y.-C. Liu, M. Nießner, A. Dai, Scannet++: A high-
fidelity dataset of 3d indoor scenes, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 12–22.
[59] OpenAI,
Gpt-5
system
card,
https://cdn.openai.com/
gpt-5-system-card.pdf (2025).
[60] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu,
Q. Jin, N. Chanpaisit, et al., Vbench: Comprehensive benchmark suite
31

<!-- page 33 -->
for video generative models, in: Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2024, pp. 21807–
21818.
32
