<!-- page 1 -->
1
DreamLifting: A Plug-in Module Lifting MV
Diffusion Models for 3D Asset Generation
Ze-Xin Yin, Jiaxiong Qiu, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, and Jin Xie
Abstract—The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an
autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking
textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end
PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of
geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design
with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate
knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion
priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge.
Then, a tamed variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D Gaussian Splatting (2DGS) with PBR
channels. Finally, we introduce a dedicated post-processing procedure to effectively extract high-quality, relightable mesh assets from
the resulting 2DGS. Extensive quantitative and qualitative experiments demonstrate the superior performance of LGAA with both text-
and image-conditioned MV diffusion models. Additionally, the modular design enables flexible incorporation of multiple diffusion priors,
and the knowledge-preserving scheme leads to efficient convergence trained on merely 69k multi-view instances. Our code, pre-trained
weights, and the dataset used will be publicly available via our project page: https://zx-yin.github.io/dreamlifting/.
Index Terms—3D asset generation, diffusion model, 2D gaussian splatting
✦
1
INTRODUCTION
M
ODERN graphics pipelines rely on high-quality, PBR-
ready 3D assets to achieve photorealistic rendering,
particularly in film, game production, autonomous driving,
virtual reality, etc. However, the creation of such assets by
human artists demands significant labor and expertise. This
raises a strong demand for AI-driven content generation
methods in the field of 3D asset creation.
Although recent advances in native 3D generation [1]–[8]
have produced highly detailed geometries, these methods
still fail to model vivid PBR textures. Early works, includ-
ing distillation-based approaches [9]–[15] and multi-view
generation–reconstruction pipelines [16]–[25], typically bake
textures as simple vertex colors, which do not support
relighting or photorealistic rendering. While these methods
can leverage multi-view material diffusion models to syn-
thesize PBR textures and project them onto UV maps, such
operations introduce additional issues such as misalignment
or blurring in the texture maps. 3DTopia-XL [26] compresses
SDF and PBR materials into a specially designed 3D repre-
sentation named PrimX and builds a latent diffusion model
upon PrimX, resulting in an end-to-end 3D asset generation
•
This manuscript has been submitted to IEEE Transactions on Visualiza-
tion and Computer Graphics (TVCG) for review.
•
Z.X. Yin and J. Yang are with PCA Lab, VCIP, College of Computer
Science, Nankai University.
•
J.Xie is with School of Intelligence Science and Technology, Nanjing
University, Suzhou, China.
•
J. Qiu, L. Liu, X. Wang, Z. Su are with Horizon Robotics.
•
W. Sui is with D-Robotics.
•
This work is partially done while Z.X. Yin’s research internship at Horizon
Robotics & D-Robotics.
•
J. Xie is the corresponding author (csjxie@nju.edu.cn).
framework. However, it is challenging to satisfy the need
for large quantities of high-quality, PBR-ready 3D assets
required to train 3DTopia-XL. Therefore, the community
requires new data-efficient paradigms for PBR-ready 3D
asset generation.
To explore a new paradigm for 3D asset generation, we
introduce a novel perspective on utilizing 2D MV diffusion
models. Previous works [16], [17], [27] demonstrate that MV
RGB diffusion models can synthesize multi-view consistent
images, from which 3D meshes can be extracted using
neural 3D reconstruction methods [28], [29], indicating the
implicit geometry priors encapsulated in the MV RGB dif-
fusion models. Furthermore, subsequent studies [30]–[33]
extend diffusion models to generate multi-view consistent
surface normal maps, depth maps, and PBR materials,
which inherently encode valuable PBR priors. These pio-
neering works underscore the encapsulated geometric and
PBR priors in MV diffusion models, making them well
suited for high-quality 3D asset creation. In this paper, we
propose to explore a novel scheme to exploit the encapsulated
priors for direct 3D asset generation without relying on large
reconstruction models (LRMs) or neural reconstruction methods.
To achieve this goal, we introduce the Lightweight
Gaussian Asset Adapter (LGAA) for end-to-end, high-
quality, PBR-ready 3D asset generation by leveraging MV
diffusion priors from a novel perspective. The LGAA fea-
tures a plug-in module that integrates seamlessly with pre-
trained multi-view diffusion. Specifically, we first design a
simple yet effective LGAA Wrapper module, which serves
as the core component for adaptively preserving and fusing
pre-trained knowledge for 3D asset generation. The network
layers of MV diffusion models encode robust knowledge
acquired from billions of images, and the LGAA Wrap-
arXiv:2509.07435v1  [cs.CV]  9 Sep 2025

<!-- page 2 -->
2
A wooden
owl, 3d
model.
input
normal
albedo
metallic
roughness
(a) Text- and image-conditioned generation results.
(b) Generated 3D assets under different HDRI maps.
(c) Multiple 3D assets generated by our pipeline in different environment maps.
Fig. 1: Our pipeline possesses the capability of generating diverse, PBR-ready 3D assets from either text prompts or image
conditions. The synthesized assets are fully relightable with accurate PBR materials; for example, the wooden owl instance
exhibits diffuse color changes under different environment maps, while the specular dog instance successfully reflects its
surroundings. These results highlight the usability of the generated 3D assets.
per clones and freezes these pre-trained layers, and injects
learnable, zero-initialized convolutional layers to adapt this
knowledge for 3D asset generation. To incorporate multiple
priors, including MV RGB diffusion priors and MV PBR
material diffusion priors, for geometry and PBR material
generation, we introduce the LGAA Switcher, which aligns
different priors in a layer-wise manner using learnable, zero-
initialized convolutional layers, which prevents conflicts
between priors during the early training stage and enables
the progressive and adaptive growth of alignment. Finally,
we propose the LGAA Decoder to generate pixel-aligned 2D
Gaussian Splatting (2DGS) [34]–[36] as the 3D representa-
tion that bridges 2D priors and 3D content creation. During
training, as our module predicts PBR-ready 2DGS in a feed-
forward manner, supervision can be performed solely using
rendered RGB images and corresponding G-buffer images.
The 2D supervision scheme facilitates the incorporation of
various image-based loss functions. Therefore, we introduce
an image-based differentiable deferred shading scheme to
closely align the generated PBR materials with their corre-
sponding image appearances during training, substantially
enhancing their realism.
The design of LGAA offers several advantages: (1) It
adaptively preserves expressive knowledge learned from
large-scale 2D datasets and utilizes it for 3D generation,
enabling efficient and high-quality 3D asset synthesis. Our
experiments show that preserving and adapting these pre-
trained priors facilitates better convergence with less train-
ing data. Notably, our method achieves impressive perfor-
mance while trained on only 69k high-quality multi-view
instances from G-Objaverse [30], whereas 3DTopia-XL is
trained on a subset of Objaverse [37] with 256k 3D instances.
(2) It maintains modularity and flexibility, allowing seamless
integration with different base models and enabling scalable
performance improvements when paired with more pow-
erful foundational diffusion models. Extensive experiments
confirm that integrating more powerful base models en-
hances generation quality and that flexibly combining PBR
diffusion priors significantly improves material synthesis.
For downstream applications, we implement a test-time
refinement process to convert the resulting Gaussian splats
into high-quality, UV-mapped 3D meshes, as shown in
Figure 1. At inference, the system generates Gaussian splats
with PBR materials in under three seconds, which are then
efficiently converted into production-ready assets by the
proposed process within 30 seconds.
Our contributions are as follows:
•
We introduce Lightweight Gaussian Asset Adapter
(LGAA), a modular and plug-in approach that lever-
ages MV diffusion models for direct Gaussian asset
synthesis without extensive, large-scale 3D data.
•
We present a simple yet effective scheme to exploit
2D diffusion priors for 3D asset generation, offer-
ing several advantages, including improved perfor-
mance without relying on large-scale 3D data and
scalable generation quality when integrated with
more powerful backbone models.
•
We propose a feed-forward framework with post-

<!-- page 3 -->
3
processing, enabling the rapid generation of high-
quality, PBR-ready 3D assets within 30 seconds.
2
RELATED WORKS
Optimization-based 3D generation with diffusion priors.
With the development of diffusion theory [38] and the emer-
gence of various neural 3D representations [28], [29], [34],
[39]–[41], 3D generation has achieved significant progress
in terms of quality and speed. Dreamfusion [9] proposes
Score Distillation Sampling (SDS) loss to distill 3D consistent
NeRF from text-to-image (T2I) diffusion priors given only
text prompts, which opens up a new era for zero-shot 3D
generation and follow-up efforts that improve distillation
quality [13], [42], [43], apply on different neural represen-
tations [11], [12], [14], [15], [44], perform scene-level gener-
ation [45], [46], and even extend to 4D generations [47]–
[49], occur like mushrooms after rain. However, due to
the lack of 3D priors in T2I diffusion models, distillation-
based methods suffer from the 3D inconsistency problem,
also known as the Janus problem; therefore, the community
paves the way to inject multi-view 3D priors into T2I models
by fine-tuning pre-trained models using rendered multiple
views from large-scale dataset [37], [50] to generate multi-
view consistency images [16]–[19], [30], [51]–[53], which
server as strong 2D and 3D combined priors and improve
the generation quality of distillation-based methods by a
large margin. However, this diagram suffers from time-
consuming per-prompt optimization.
Feed-forward 3D generation with diffusion priors. To
solve the limitations of distillation-based methods, in-
stant3D [20] proposes a new diagram to decompose the text-
to-3D generation task into text-to-MV images generation
and MV-to-3D generation tasks, the former phase is im-
plemented as a fine-tuned multi-view 2D diffusion model,
while the latter phase features a feed-forward network
mapping multi-view images to NeRF representation. The
two-stage diagram indirectly benefits from both 2D and
3D priors and demonstrates superiority against previous
methods regarding quality and speed. InstantMesh [23] and
CRM [54] extend the diagram to direct mesh generation,
while LGM [24], GRM [25], LaRa [55], and Turbo3D [56]
build the reconstruction models with Gaussian Splats. As
the models are typically large transformer-like networks
with millions of learnable parameters, the training process
either consumes tens to hundreds of high-end GPUs or
requires a large-scale dataset to converge, which is typ-
ically unaffordable in the academic community. Besides,
these methods all model the appearance of the generated
3D instances as simple vertex colors, failing to synthesize
PBR materials; therefore, the applications of generated 3D
models are limited.
Recently, diffusion-based native 3D generation methods
have achieved promising results. DiffSplat [57] explores
the reuse of pre-trained text-to-image diffusion models to
produce Gaussian splat images, achieving good rendering
quality, but they do not decouple PBR materials and ge-
ometry. TRELLIS [58] is able to model complex topology
via proposed structured latents, but the representation tends
to model oversaturated vertex colors. 3DShape2VecSet [59]
paves the way to generate high-quality shapes via a 3D
diffusion model, and following methods [3], [7], [8], [60],
[61] are able to generate much more detailed geometry. But
these methods all focus on geometry modeling without PBR
materials.
3D generation with PBR materials. Simultaneously recov-
ering geometry, materials, and illumination is a highly ill-
posed problem even with densely captured data [62]–[66];
therefore, most existing 3D generation works synthesize
meshes with simple vertex colors or even without textures,
which are incompatible with the modern graphics pipeline.
Several works [30], [67], [68] introduce PBR priors into
the optimization process to achieve material decomposi-
tion during generation, but these methods suffer from the
time-consuming per-prompt optimization. Meta 3D Asset-
Gen [69] proposes a feed-forward network to regress SDF
fields with coarse PBR materials and refine the textures
using a texture refiner network, while ARM [70] introduces
a feed-forward network to generate a differentiable mesh
representation and leverages a triplane-based PBR synthe-
sizer to regress the PBR materials. However, these two-stage
methods increase the complexity of the generation pipeline.
TexGaussian [71] proposes a pipeline to generate Gaussian
assets with PBR channels with 3D meshes as the conditions,
while TexGen [72] generates PBR materials for 3D meshes in
the UV space. However, these methods rely on high-quality
3D meshes as input. 3DTopia-XL [26] introduces a novel
3D representation with the ability to encode PBR material
channels, enabling text- / image-to-3D asset generation.
However, it requires large amounts of computing resources
and high-quality, PBR-ready 3D data for training.
Additionally, several recently released datasets are fa-
cilitating the development of 3D asset generation. Objects-
with-lighting [73] introduces a real-world dataset containing
object-level multi-view images captured under different en-
vironment maps; however, this dataset contains only eight
objects. DTC [74] presents a large-scale digital twin dataset
comprising 2,000 real-world objects, each with detailed
PBR materials, geometry, and associated multi-view DSLR
photographs captured under diverse environment maps.
MAGE [75] filters the Objaverse dataset, leading to a subset
containing 17k 3D instances, and renders corresponding G-
buffers and images from multiple views for each filtered
instance. IDArb [31] releases a large-scale dataset containing
5.7 million multi-view posed RGB images with PBR mate-
rials, significantly promoting the development of PBR asset
generation.
3
METHODS
An overview of our framework is illustrated in Figure 2,
which is designed for 3D asset generation via extracting
geometric and appearance priors from 2D diffusion models
without requiring access to large amounts of high-quality
3D data. First, we briefly review the 3D Gaussian Splat
representation and the multi-view diffusion models that
constitute our backbone in Section 3.1. Next, we detail the
architecture of LGAA and describe its integration with var-
ious multi-view diffusion models in Section 3.2. To facilitate
3D asset generation with PBR materials, we introduce an
image-based differentiable deferred rendering scheme that

<!-- page 4 -->
4
Overall pipeline
”treasure
chest”
or
...
...
MV diffusion
feature maps
LGAA-W
LGAA-S
LGAA-W
· · ·
LGAA-W
LGAA-S
LGAA-W
LGAA-D
Gaussian Assets
X
Y
Y ′
ZC
ZC
(a) LGAA Wrapper
Ya
Y ′
a
Y ′
g
Yg
(a) LGAA Switcher
ZC
ZC
(c) LGAA Decoder
Frozen VAE
Attn. layers
Gaussian head
Deferred shading
Deferred shading loss
Pixel-level loss
Generated 3D asset
Fig. 2: Overall of the 3D asset generation pipeline. We propose the Lightweight Gaussian Asset Adapter (LGAA), which
is composed of three components: (a) LGAA Wrapper (LGAA-W), (b) LGAA Switcher (LGAA-S), and (c) LGAA Decoder
(LGAA-D), where ZC indicates zero-initialized convolutional layers. In (a), X indicates input feature maps, Y are feature
maps from MV diffusion models, and Y ′ are output maps from the LGAA-W. In (b), Ya and Yg are feature maps in
the appearance and geometry branches, while Y ′
a and Y ′
g are output feature maps after information alignment. Our
LGAA takes feature maps from a pre-trained MV diffusion model, adapts priors with the proposed modules, and produces
Gaussian Splat assets with PBR materials. During the training procedure, we tie the G-buffer maps with the RGB images
via image-based deferred shading. In inference, we extract the 3D mesh with PBR material maps from the Gaussian Splat
assets with carefully designed post-processing.
links intrinsic components to the final rendered appearance
in Section 3.3. Finally, we outline a detailed procedure for
extracting 3D meshes with PBR materials from Gaussian
assets in Section 3.4.
3.1
Preliminaries: Gaussian Splats and Multi-view dif-
fusion models
Pixel-aligned
Gaussian
Splats. 3D Gaussian Splatting
(3DGS) [34] proposes to parameterize the 3D scene via
radiance fields in the form of a collection of Gaussian
primitives G = {gi}, where each primitive contains multiple
attributes recovered by differentiable rendering. 2D Gaus-
sian Splatting (2DGS) [35] further improves the 3DGS for
accurate geometry representation, each primitive of which
is parameterized by a 3D position µ ∈R3, a rotation vector
R ∈R3, a scaling vector S ∈R2, an opacity o ∈R, and a
view-dependent appearance SH ∈R(d+1)2×3 represented
by spherical harmonic of degree d. As an explicit repre-
sentation, 2DGS is an unstructured representation, which is
incompatible with traditional 2D neural networks. Inspired
by Splatter Image [36], we leverage 2DGS and organize
them in the form of multi-view splatter images, where each
pixel represents one Gaussian primitive; therefore, we can
easily leverage 2D neural networks to generate 2DGS with
accurate geometry.
Multi-view diffusion model. Multi-view (MV) diffusion
models are typically fine-tuned from stable diffusion [76]
to generate 3D consistent MV images. MVDream [19]
and DreamView [53] are fine-tuned to generate four or-
thogonal views around the object with elevation at the
range of [0°, 30°] through the denoising process. Image-
Dream [77] finetunes MVDream with carefully designed
pixel injection scheme to achieve image-to-multi-view gen-
eration. IDArb [31] is a multi-view-RGB-conditioned diffu-
sion model to decouple multi-view consistent, physically
based rendering materials. The above diffusion models con-
tain rich 2D and 3D priors, based on which we conduct our
experiments.
3.2
Lightweight Gaussian Asset Adapter
Current MV diffusion models are capable of generating
multi-view consistent RGB appearance and PBR materials,
demonstrating their encapsulation of geometric and PBR
material priors. To leverage these rich priors for direct 3D
asset generation, we propose our Lightweight Gaussian
Asset Adapter, which adaptively preserves priors from MV
diffusion models, adapts them for 3D asset generation,
and generates Gaussian splats with PBR attribute chan-
nels. As shown in Figure 2, LGAA comprises three funda-
mental components: LGAA Wrapper, LGAA Switcher, and
LGAA Decoder. Detailed descriptions of each component
are provided below.
LGAA Wrapper. Current MV diffusion models are trained
on billions of high-quality images, and thus their network

<!-- page 5 -->
5
layers encapsulate robust knowledge. Our LGAA Wrapper
is designed to exploit the encapsulated knowledge for our
task. Therefore, the module must preserve the pre-trained
layers while also being able to adapt their outputs. We
achieve this in a simple yet effective manner.
Following [78], we define a network block in diffusion
models as modular units commonly utilized in neural
network architectures, including ResNet blocks, convolu-
tion batch-normalization ReLU (conv-bn-relu) blocks, trans-
former blocks, among others. As illustrated in Figure 2(a),
the LGAA Wrapper consists of layers cloned and frozen
from the middle and upsampling blocks of the UNet ar-
chitecture. These layers typically receive an input feature
map Xin, a skip connection feature Xres (absent in the
middle layer), and additional conditional information C, to
produce an output feature map Y . Given such a network
block F(Xin, Xres, C; Θ) with pre-trained parameters Θ,
our LGAA Wrapper creates a parallel copy of this block,
adapts the information flow with zero-initialized 1 × 1
convolutional layer ZC(·), and operates as follows:
Y
=
F(Xin, Xres, C; Θ)
Y ′
=
F(Xin, Xres + ZC(Xres), C; Θ) + ZC(Y )
(1)
This design maximizes the preservation of pre-trained pri-
ors while enabling adaptability for 3D asset generation.
LGAA Switcher. LGAA Wrapper offers the flexibility to
integrate multiple priors, for example, we can create parallel
branches, each encapsulating knowledge from different MV
diffusion models to separately synthesize geometry (mean
positions µ of Gaussian primitives) and appearance (other
Gaussian Splat attributes). However, this creates the need to
align information flow from the parallel branches to ensure
consistent instance generation. To enhance consistency and
alignment between geometry and appearance features, we
propose a lightweight LGAA Switcher, implemented using
zero-initialized 1×1 convolutional (ZC) layers, as illustrated
in Figure 2(b). Specifically, given a geometry branch layer
Fg(·) and an appearance branch layer Fa(·), which produce
feature maps Yg and Ya, respectively, the LGAA Switcher
enables bidirectional information exchange between these
branches as follows:
Y ′
g
=
Yg + ZC(Ya)
Y ′
a
=
Ya + ZC(Yg)
(2)
This design avoids conflicts between parallel branches dur-
ing the early training stage, while enabling progressive
development of information alignment paths.
LGAA Decoder. We design the LGAA based on latent
diffusion models, which produce latent feature maps at a
lower resolution, resulting in a limited number of Gaussian
primitives insufficient for representing fine-grained struc-
tures. To address this limitation, we utilize the pre-trained
decoder of the variational autoencoder (VAE) to construct
our LGAA Decoder, as depicted in Figure 2. By decoding
to a higher spatial resolution, our approach enables the
generation of a greater number of Gaussian primitives,
thus capturing more detailed geometric and appearance
information. In practice, we freeze most layers of the pre-
trained decoder to stabilize the training, unlock the input
layers, and adjust the output layers to produce Gaussian
splat attributes. The Gaussian head decodes pixel-aligned
Gaussian splat images, where each pixel is a Gaussian
primitive with 3D position µ ∈R3, rotation vector R ∈R3,
scaling vector S ∈R2, opacity o ∈R, color c ∈R3, albedo
a ∈R3, metallic m ∈R, and roughness r ∈R.
Integration of LGAA with MV diffusion models. As
illustrated in Figure 2, our framework is capable of com-
bining multiple diffusion priors within a unified model,
thus effectively harnessing the complementary strengths
of different base models. To validate the efficacy of this
design, we perform extensive experiments on the chal-
lenging task of end-to-end 3D asset generation with PBR
materials. We integrate robust multi-view RGB priors such
as MVDream [19], ImageDream [77], and DreamView [53],
alongside the multi-view PBR prior IDArb [31], showcasing
the versatility and superior performance of our integrated
framework. For text-conditioned generation models, we
construct the LGAA Wrapper layers with the middle and
upsampling blocks from both text-to-MV diffusion models
and the IDArb model to form a parallel branch structure,
then we leverage the LGAA Switcher to enable infor-
mation alignment between different branches, ultimately
predicting the Gaussian splat assets with the LGAA De-
coder. For image-conditioned generation models, layers
from text-conditioned diffusion models encapsulated in the
LGAA Wrapper can be straightforwardly replaced by those
from image-conditioned diffusion models.
3.3
Deferred shading loss
Although we have direct access to the G-buffer information
for supervision, the simultaneous generation of geometry
and appearance for 3D assets remains a highly ill-posed
problem. Inspired by [79], [80], we leverage an image-based
differentiable rendering approach to link the rendered G-
buffer information with the final RGB appearance to reduce
the ambiguity. Specifically, we separate the rendering equa-
tion [81] as the diffuse (Ld) and specular (Ls) components:
Lo(x, ωo) = Ld + Ls
(3)
where Lo models the outgoing radiance of the surface point
x at direction ωo.
For the diffuse Ld component, we employ an image-
based lighting model and the split-sum approximation for
direct illumination, while occlusion and indirect illumi-
nation are precomputed and stored into view-dependent
images Io and Iirr:
Ld ≈(1 −Io)
Z
Ω
Li(x, ωi)(ωi, n)dωi + IoIirr
(4)
where Ωdenotes the upper hemisphere. The first term
models the direct illumination depending on the param-
eters cosθ = ωi · n and the roughness r, which can be
precomputed into a lookup map. The second term encapsu-
lates indirect illumination, recovered for each training view
through efficient preprocessing.
The specular term Ls follows the split-sum approxima-
tion from [79]:
Ls ≈
Z
Ω
DFG
4(n, ωi)(n, ωo)dωi
Z
Ω
DLi(ωi)(n, ωi)dωi
(5)

<!-- page 6 -->
6
LaRa
LGM
DiffSplat
TRELLIS
3DTopia-XL
MVD-v1.5-3D
MVD-v2.1-3D
DV-3D
FID↓
38.80
36.00
-*
27.08
64.96
40.76
36.38
33.75
FIDmat↓
-
-
-
-
47.19
53.62
41.47
42.63
IS↑
12.77 ± 0.39
13.53 ± 0.52
13.75 ± 0.51
13.40 ± 0.28
9.42 ± 0.12
12.88 ± 0.46
14.63 ± 0.51
13.91 ± 0.38
CLIP score↑
31.41
32.17
31.62
31.25
29.26
32.84
33.70
34.36
TABLE 1: Quantitative comparisons with different base models demonstrate that our module directly lifts different base
multi-view diffusion models for 3D asset generation. MVD-v1.5-3D, MVD-v2.1-3D, and DV-3D denote MVDream 1.5 with
LGAA, MVDream 2.1 with LGAA, and Dreamview with LGAA, respectively. ’*’ indicates that we omit the FID for DiffSplat
because this approach is trained on the complete G-Objaverse dataset and thus our evaluation subset is covered by their
training dataset.
‘’A wooden chest with a lock and black trim, 3d asset.‘’
”Star Wars Stormtrooper helmet, 3d asset.”
”A pink teddy bear with a zipper on its back, 3d asset.”
LaRa
LGM
DiffSplat
TRELLIS
Meshy-4*
Luma Genie*
3DTopia-XL
Ours
Albedo Metallic
Roughness
Fig. 3: Visual comparisons of text-conditioned 3D asset generation methods. For LGM and LaRa, we use MVDream 2.1 to
generate four input views. ’*’ refers to the non-publicly available commercial software.
where the first integral term models the specular BSDF un-
der white light, and the second term integrates the incoming
radiance using a pre-integrated cubemap.
3.4
Geometry and Texture Refinement
To improve the usability of the generated Gaussian as-
sets, we introduce a dedicated post-processing procedure
to convert the Gaussian assets into PBR-ready mesh assets.
Specifically, we first extract meshes from 2DGS via TSDF fu-
sion and refine them through continuous remeshing [82] to
acquire watertight meshes. Next, we initialize texture maps
for the 3D assets using the Gaussian assets and leverage a
differentiable renderer to further refine the PBR materials.
Geometry extraction and refinement. To extract meshes
from 2DGS, we render albedo and depth images along
circular camera paths at elevations of [10°, 15°, 20°] around
each instance, along with additional top and bottom views,
and we utilize the ScalableTSDFVolume from Open3D [83]
with a voxel size of 0.008 and a truncation threshold of 0.02
to perform TSDF Fusion to extract the initial mesh. Then,
we compute the convex hull of the initial mesh to fill all
the holes in the original mesh. Finally, we render normal
maps and alpha maps from the 2DGS around the instances
as the target views and perform 100 iterations of continuous
remeshing [82] to transform the convex hull into smooth,
high-quality watertight meshes.
Texture initialization and refinement. After mesh extrac-
tion, we generate UV maps using Blender’s Smart UV
Project [84]. The rendered albedo, metallic, and roughness
maps from the 2DGS are then unprojected onto the mesh
to initialize the PBR materials. Then, following [73], we use
the differentiable renderer [85], [86] to align four orthogonal
views with the images generated by multi-view diffusion
models, thereby enhancing the visual quality of the final
appearance. The entire procedure takes less than 30 seconds
on an NVIDIA GeForce RTX 4090 GPU.
3.5
Training and Inference
Training. During training, we sample batches consisting of
four orthogonal views and four random views, each com-
prising RGB, alpha mask, normal, depth, albedo, metallic,
and roughness maps, all scaled at a resolution of 256. We
train our model using bfloat16 precision and a gradient
accumulation step of 8, with each GPU processing two
batches, resulting in a total batch size of 128. We add random
grid distortion [24] to the four orthogonal views, assign
random background color to the RGB images, and then
process the views following MVDream, ImageDream, and
Dreamview with random noise level t ∈[0, 1000) to get the
input for the 2D diffusion models. Then, we render RGB,
albedo, alpha mask, metallic, roughness, depth, and normal
maps for all eight views at the same resolution. For RGB

<!-- page 7 -->
7
PSNR↑
SSIM↑
LPIPS↓
PSNRalbedo ↑
MSEmetallic ↓
MSEroughness ↓
LaRa
13.32
0.750
0.344
-
-
-
LGM
15.98
0.781
0.249
-
-
-
TRELLIS
14.64
0.761
0.270
-
-
-
3DTopia-XL
13.36
0.756
0.334
13.42
0.122
0.037
ImageDream w/ LGAA
17.04
0.788
0.227
17.62
0.049
0.015
TABLE 2: Quantitative metrics for image-conditioned generation results.
Input
LaRa
LGM
DiffSplat
TRELLIS
Meshy-4*
3DTopia-XL
Ours
Albedo Metallic
Roughness
Fig. 4: Visual comparisons of image-conditioned 3D asset generation methods. For LGM and LaRa, we use ImageDream to
generate four input views. ’*’ refers to the non-publicly available commercial software.
and albedo supervision, we apply MSE loss, SSIM loss [34],
and LPIPS loss [87]:
Lcolor
=
λ1LMSE(Icolor, IGT
color)+
λ2LSSIM(Icolor, IGT
color)+
λ3LLP IP S(Icolor, IGT
color)
(6)
where λ1 = 1, λ2 = 2, λ3 = 5. For alpha map, we use binary
cross-entropy loss:
Lalpha = LBCE(Ialpha, IGT
alpha)
(7)
For metallic and roughness, we apply MSE loss:
Lmaterial = LMSE(Imaterial, IGT
material)
(8)
And we also apply the depth distortion loss and normal
consistency loss [35]. The depth distortion loss is applied
with a weight of 2 × 104 during the initial three epochs to
facilitate convergence, subsequently decaying to 1 × 102.
Fine-tuning with deferred shading loss. After 20 epochs of
training with direct supervision using G-buffer information,
we conduct further fine-tuning steps on a dataset subset
using the deferred shading loss described in Section 3.3.
During this phase, all original training losses are retained,
and image-based rendering is employed to align the albedo,
metallic, roughness, and normal maps with Ishade. The
shaded images are then supervised using the loss defined
in equation 6. This additional fine-tuning stage encompasses
another 20 epochs. The training and fine-tuning span 3 days
on 1 node of 8 NVIDIA H20 GPUs.
Inference. We infer our model via DDIM sampling with 50
steps and a guidance scale of 7.5 for the MVDream-based
and DreamView-based version. For the ImageDream-based
model, we use a guidance scale of 5.0 as in [77]. Since
our LGAA receives feature maps from diffusion models
and generates Gaussian splats in a feed-forward manner,
only a single inference pass with our model is required
during the sampling process of the multi-view diffusion
models. Empirically, we find that utilizing feature maps
from diffusion models at a noise level of t = 150 yields
optimal results.
4
EXPERIMENTS
4.1
Experimental settings
Dataset. We conduct all experiments using the open-source
G-Objaverse dataset [30], a high-quality multi-view dataset
that includes G-buffer information. This dataset comprises
265K rendered 3D instances from Objaverse [37] and 779K
3D instances from the Objaverse-XL Alignment [50], each
instance featuring 38 views around the object. We filter
this dataset based on aesthetic scores, material quality, and
valid pixel coverage, resulting in a high-quality subset of
69k instances for training. Within this subset, we apply
the deferred shading function (described in Section 3.3) to
ground truth (GT) G-buffer maps to recover environment
maps, as well as view-dependent occlusion and indirect
illumination maps, for fine-tuning purposes. We compute
the PSNR between the shaded images, shaded with GT G-
buffer maps and recovered lighting information, and the GT
images, selecting instances with a PSNR greater than 35 dB
for fine-tuning. This results in a subset of 10K high-quality

<!-- page 8 -->
8
instances with recovered environmental and illumination
maps.
Baselines. We select MVDream 1.5, MVDream 2.1 [19],
and DreamView [53] as text-to-3D baselines, and Image-
Dream [77] as the image-to-3D baseline for our experiments.
Additionally, we incorporate knowledge from IDArb [31]
into our framework to enhance PBR material generation.
For evaluating 3D asset generation, we compare against
3DTopia-XL [26], a baseline supporting text- and image-
conditioned 3D asset generation with PBR materials. We
also compare our results with DiffSplat [57], LGM [24], and
LaRa [55]. DiffSplat [57] supports native 3D Gaussian splat
generation, whereas LGM and LaRa are reconstruction-
based Gaussian generation methods. Furthermore, we re-
port evaluation metrics against TRELLIS [58], a leading
native 3D generation method supporting text- and image-
conditioned Gaussian splat generation. Besides, we also
compare our results against Meshy-v4 [88] and LumaAI-
Genie [89], proprietary software solutions for 3D mesh
generation with PBR materials.
Text-conditioned evaluation metrics. Following the eval-
uation protocol outlined by MVDream [19], we randomly
select 1,000 unseen prompts from the G-Objaverse dataset,
with each instance represented by 12 evenly distributed
views that serve as ground truth images. We evaluate the
alignment between rendered images and text prompts using
the CLIP similarity score [90]. Furthermore, we employ the
Fr´echet Inception Distance (FID) [91] and Inception Score
(IS) [92] to assess the quality of rendered images from the
generated 3D assets. To evaluate the quality of PBR materi-
als, we render 12 views of albedo, metallic, and roughness
maps. The metallic (m) and roughness (r) maps are stored
in the format (o, r, m), with the o channel unused. Finally,
we report the FIDmat metric, which compares the generated
material maps against ground truth material maps.
Image-conditioned evaluation metrics. We also perform
inference on the same 1,000 unseen instances from the text-
conditioned evaluation, selecting 12 views per instance as
ground truth images, with the front view serving as the
input view. We report the PSNR, SSIM [93], and LPIPS [87]
metrics to evaluate the rendered images. For the evaluation
of PBR materials, we report PSNR for albedo maps and
report MSE for the metallic and roughness images.
4.2
Main results
Text-conditioned 3D asset generation. As shown in Table 1,
the proposed LGAA demonstrates compatibility with differ-
ent base models. Most importantly, the generation quality
improves as the capacity of the base models increases,
as evidenced by the FID and CLIP scores of LGAA with
different base models. This indicates that our scheme can
effectively exploit priors from base models for 3D genera-
tion, demonstrating its potential. Furthermore, our models
outperform the state-of-the-art method 3DTopia-XL [26] on
the challenging task of end-to-end PBR-ready 3D asset gen-
eration. Notably, 3DTopia-XL is trained on 16 nodes with 8
NVIDIA A100 GPUs each and 256k 3D instances, whereas
our models are trained on a single node with 8 NVIDIA H20
GPUs and only 69k multi-view images, which demonstrates
the efficiency and effectiveness of our scheme. Although our
method achieves a higher FID compared to the native 3D
generation approach TRELLIS [58]—primarily because our
task requires the simultaneous synthesis of geometry and
PBR materials, while TRELLIS focuses only on geometry
and simple vertex colors— our approach achieves a better
CLIP score than TRELLIS, indicating better alignment be-
tween the generation results and the prompts.
Qualitative comparisons are illustrated in Figure 3. Our
approach generates accurate geometry and fine-grained
PBR maps across different materials. In contrast, 3DTopia-
XL produces inaccurate geometry or blurry appearances,
highlighting the robustness and superiority of our approach.
Furthermore, TRELLIS does not support the decoupling of
PBR materials, and it suffers from out-of-domain issues, as
shown in the second row of Figure 3. In addition, other GS-
based generation or reconstruction methods fail to produce
relightable 3D assets. For example, LGM and LaRa recon-
struct coarse geometry with appearances entirely baked into
vertex colors, while DiffSplat produces Gaussian splats that
only support novel view synthesis.
Image-conditioned 3D asset generation. As shown in Ta-
ble 2 and Figure 4, our module achieves superior results
both quantitatively and qualitatively, demonstrating the
compatibility of our scheme with image-conditioned multi-
view diffusion models. Our method not only faithfully
generates complete geometry but also synthesizes accurate
PBR materials for various objects, ensuring photorealistic
rendering and relighting. In contrast, the state-of-the-art
3DTopia-XL method even struggles to generate accurate ge-
ometry, whereas our approach successfully separates differ-
ent materials with correct PBR attributes and recovers accu-
rate geometry for specular instances. Additionally, TRELLIS
bakes PBR materials into simple vertex colors, which tend
to be oversaturated, leading to inferior quantitative metrics.
For the reconstruction method LaRa, specular regions sig-
nificantly degrade geometry quality, while LGM tends to
reconstruct blurry vertex colors.
Relighting results. We place the generated 3D assets under
different HDRI maps to showcase the relighting results in
Figure 5. Specular materials accurately reflect the surround-
ing environment, as demonstrated by the helmet and barrel
instances, while diffuse materials exhibit only changes in
base color, as seen in the fruit and teddy bear example.
These results demonstrate the accuracy of the synthesized
PBR materials. Please refer to our supplementary video for
relighting results under rotating HDRI maps.
More results. We present detailed visualizations of the ge-
ometry and PBR materials for additional results in Figure 6,
along with the corresponding input prompts or images.
Our text-conditioned models align the generated results
well with the input prompts, while the image-conditioned
model faithfully reconstructs the input views and infers the
correct geometry of the unseen parts. More importantly, all
our models are able to synthesize accurate PBR materials
for various instances, demonstrating the robustness and
usability of our approach. Additional rendering results can
be found in our supplementary video.

<!-- page 9 -->
9
Input prompt
”A wooden chest
with a lock and
black trim, 3d
asset.”
Input prompt
”Star Wars
Stormtrooper
helmet, 3d
asset.”
Input prompt
”A pink teddy
bear with a
zipper on its
back, 3d asset.”
Input image
Input image
Input image
Fig. 5: Relighting results under different HDRI maps. The synthesized diffuse materials exhibit base color changes under
different environment maps, as shown in the fruit and teddy examples, while specular materials, as the barrel and helmet
instances, correctly reflect the surroundings.
w/o LGAA-W
w/o LGAA-S
w/o LGAA-D
w/o IDArb
w/o deferred shading
Full model
FID↓
54.52
38.33
37.39
40.89
36.73
36.38
FIDmat↓
74.31
47.48
42.37
51.72
43.52
41.47
IS↑
13.10 ± 0.54
14.46 ± 0.63
14.96 ± 0.47
14.52 ± 0.56
14.84 ± 0.57
14.63 ± 0.51
CLIP score↑
32.76
33.60
33.73
33.40
33.73
33.70
TABLE 3: Quantitative evaluation of the ablation study experiments.
4.3
Ablation Study
We conduct ablation studies using the MVDream 2.1 model
to investigate the design choices of the proposed module.
Furthermore, given the flexibility of our approach to in-
tegrate multiple sources of knowledge within an unified
framework for 3D asset generation, we perform an ablation
study to assess the impact of incorporating IDArb priors.
Finally, we evaluate the impact of the deferred shading loss.
LGAA module designs. We comprehensively investigate
the design choices to validate the effectiveness of the pro-
posed modules. LGAA-W adaptively preserves the pre-
trained priors and leverages the knowledge for 3D asset
generation, serving as a critical component. To assess the
significance of this design, we unlock and train the cloned
layers inside the LGAA-W to investigate the effectiveness
of this key design, which we refer to as ’w/o LGAA-W’.
Furthermore, LGAA-D adapts the pre-trained VAE Decoder
to function as a powerful upsampler for Gaussian asset gen-
eration; therefore, we replace the pre-trained VAE Decoder
layers with three layers of PixelShuffle, denoted as ’w/o
LGAA-D’. Additionally, we remove other components to
assess their contributions to our framework. As shown in
Table 3 and Figure 7, we draw the following conclusions:

<!-- page 10 -->
10
geometry albedo
metallic roughness
A slice of cake on
a plate.
A pink teddy bear
with zipper on its
back.
A blue and golden
wooden barrel.
A wooden chest
with gold coins.
A red and yellow
Iron Man helmet.
A large pink and
white vase.
A silver Manda-
lorian helmet.
A large clay jar
with a lid.
(a) Detailed visualizations of the more generated 3D assets with text conditions.
geometry
albedo
metallic
roughness
input
image
(b) Detailed visualizations of the more generated 3D assets with image conditions.
Fig. 6: We provide detailed visualizations of geometry and PBR materials from the generated 3D assets, along with the input
conditions. The results indicate that the proposed LGAA is capable of generating a diversity of high-quality PBR-ready 3D
assets with either text conditions or image conditions.
(1) Our LGAA Wrapper effectively preserves knowledge
and fuses priors from pre-trained models into a unified
framework for end-to-end 3D asset generation. In contrast,
training the previously frozen layers increases training costs
and disrupts the preserved priors, leading to degraded
results, as indicated in the first column of Table 3 and the
first row of Figure 7. (2) Our LGAA Switcher plays a crucial
role in aligning different priors in the unified framework.
As demonstrated in the second column of Table 3, the
absence of the Switcher results in significant misalignment
between the rendered RGB appearance metric (FID) and the
synthesized PBR metric (FIDmat). (3) The LGAA Decoder
serves as a powerful upsampler in latent space, effectively
refining latent maps to generate more Gaussian primitives
with fine-grained details. As shown in the third row of
Figure 7, replacing LGAA-D introduces grid artifacts in the
generated albedo.
LGAA scaling ability. The proposed scheme demonstrates
two types of scalability. First, it achieves improved results
by incorporating more powerful base diffusion models. As
shown in Table 1, LGAA achieves significantly better FID
when paired with DreamView compared to MVDream 1.5.
Second, our approach flexibly integrates multiple diffusion
priors within a unified framework for specific tasks. As
illustrated in Table 3 and Figure 7, when integrated with the
IDArb prior, our module adaptively leverages this knowl-
edge to enhance the quality of PBR material generation.
Deferred shading loss. Simultaneously generating geom-
etry and PBR materials is a highly ill-posed problem due
to inherent ambiguity. Therefore, we introduce image-based
deferred shading to couple G-buffer information with RGB
appearance. As shown in the fifth column of Table 3, the
deferred shading loss quantitatively improves the quality of
PBR materials, and the results in Figure 7 further demon-
strate visual improvements in the rendered PBR maps.
5
LIMITATIONS
Although our scheme enables the simultaneous generation
of geometry and PBR materials, it has the following limita-
tions: First, since our approach tames pre-trained MV diffu-
sion models for 3D asset generation by training additional
adapters while keeping the pre-trained models frozen, the
dataset used to train the adapters must adhere to the con-
ventions of the dataset on which the MV diffusion models
were originally trained. Second, our method is supervised
solely by pixel-level losses. As a result, the internal struc-
tures of instances lack appropriate regularization. We leave
the exploration of combining our approach with native 3D
generation schemes for improved structural modeling to
future work.

<!-- page 11 -->
11
Albedo
Metallic
Roughness
w/o LGAA-W
w/o LGAA-S
w/o LGAA-D
w/o IDArb
w/o DS
Full model
Fig. 7: Visual comparisons from the ablation studies. ’DS’ denotes deferred shading. The three instances, from left to right,
correspond to the descriptions: ”a black panther mask in the shape of a cat head”, ”an ornate wooden box with a lock”,
and ”a wooden crate”. The results highlight the contributions of each component to our framework.
6
CONCLUSION
In this paper, we explore an alternative scheme for lever-
aging 2D diffusion priors in the challenging task of end-
to-end 3D asset generation with physically based rendering
(PBR) materials. We propose a novel LGAA that modulates
pre-trained multi-view diffusion models to enable direct
3D asset generation, demonstrating superior generalization
capability and efficiency. Moreover, our framework exhibits
scalability and flexibility by adaptively integrating diverse
knowledge into a unified model, with its generative capa-
bilities further enhanced when paired with more powerful
base models. Finally, we establish a complete pipeline based
on the proposed method, which efficiently generates high-
quality 3D meshes with PBR materials using consumer-
grade GPUs in under 30 seconds. Extensive quantitative and
qualitative evaluations highlight the significant potential of
our proposed scheme.
REFERENCES
[1]
X. Yang, H. Shi, B. Zhang, F. Yang, J. Wang, H. Zhao, X. Liu,
X. Wang, Q. Lin, J. Yu et al., “Hunyuan3d 1.0: A unified frame-
work for text-to-3d and image-to-3d generation,” arXiv preprint
arXiv:2411.02293, 2024. 1
[2]
L. Zhang, Z. Wang, Q. Zhang, Q. Qiu, A. Pang, H. Jiang, W. Yang,
L. Xu, and J. Yu, “Clay: A controllable large-scale generative model
for creating high-quality 3d assets,” ACM Transactions on Graphics
(TOG), vol. 43, no. 4, pp. 1–20, 2024. 1
[3]
Z. Zhao, Z. Lai, Q. Lin, Y. Zhao, H. Liu, S. Yang, Y. Feng, M. Yang,
S. Zhang, X. Yang et al., “Hunyuan3d 2.0: Scaling diffusion models
for high resolution textured 3d assets generation,” arXiv preprint
arXiv:2501.12202, 2025. 1, 3
[4]
Y. Yang, Y.-C. Guo, Y. Huang, Z.-X. Zou, Z. Yu, Y. Li, Y.-P. Cao,
and X. Liu, “Holopart: Generative 3d part amodal segmentation,”
arXiv preprint arXiv:2504.07943, 2025. 1
[5]
X. He, Z.-X. Zou, C.-H. Chen, Y.-C. Guo, D. Liang, C. Yuan,
W. Ouyang, Y.-P. Cao, and Y. Li, “Sparseflex: High-resolution
and
arbitrary-topology
3d
shape
modeling,”
arXiv
preprint
arXiv:2503.21732, 2025. 1
[6]
Y. Li, Z.-X. Zou, Z. Liu, D. Wang, Y. Liang, Z. Yu, X. Liu, Y.-C.
Guo, D. Liang, W. Ouyang et al., “Triposg: High-fidelity 3d shape
synthesis using large-scale rectified flow models,” arXiv preprint
arXiv:2502.06608, 2025. 1
[7]
S. Wu, Y. Lin, F. Zhang, Y. Zeng, Y. Yang, Y. Bao, J. Qian,
S. Zhu, P. Torr, X. Cao, and Y. Yao, “Direct3d-s2: Gigascale 3d
generation made easy with spatial sparse attention,” arXiv preprint
arXiv:2505.17412, 2025. 1, 3
[8]
Z. Li, Y. Wang, H. Zheng, Y. Luo, and B. Wen, “Sparc3d: Sparse
representation and construction for high-resolution 3d shapes
modeling,” arXiv preprint arXiv:2505.14521, 2025. 1, 3
[9]
B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion:
Text-to-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988,
2022. 1, 3
[10] H. Wang, X. Du, J. Li, R. A. Yeh, and G. Shakhnarovich, “Score
jacobian chaining: Lifting pretrained 2d diffusion models for 3d
generation,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 12 619–12 629. 1
[11] R. Chen, Y. Chen, N. Jiao, and K. Jia, “Fantasia3d: Disentangling
geometry and appearance for high-quality text-to-3d content cre-
ation,” in Proceedings of the IEEE/CVF international conference on
computer vision, 2023, pp. 22 246–22 256. 1, 3
[12] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang,

<!-- page 12 -->
12
K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3d: High-
resolution text-to-3d content creation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 300–309. 1, 3
[13] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu,
“Prolificdreamer: High-fidelity and diverse text-to-3d generation
with variational score distillation,” Advances in Neural Information
Processing Systems, vol. 36, 2024. 1, 3
[14] Z. Chen, F. Wang, Y. Wang, and H. Liu, “Text-to-3d using gaussian
splatting,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 21 401–21 412. 1, 3
[15] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “Dreamgaussian:
Generative gaussian splatting for efficient 3d content creation,”
arXiv preprint arXiv:2309.16653, 2023. 1, 3
[16] R. Shi, H. Chen, Z. Zhang, M. Liu, C. Xu, X. Wei, L. Chen, C. Zeng,
and H. Su, “Zero123++: a single image to consistent multi-view
diffusion base model,” arXiv preprint arXiv:2310.15110, 2023. 1, 3
[17] X. Long, Y.-C. Guo, C. Lin, Y. Liu, Z. Dou, L. Liu, Y. Ma, S.-
H. Zhang, M. Habermann, C. Theobalt et al., “Wonder3d: Single
image to 3d using cross-domain diffusion,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 9970–9980. 1, 3
[18] Y. Liu, C. Lin, Z. Zeng, X. Long, L. Liu, T. Komura, and W. Wang,
“Syncdreamer: Generating multiview-consistent images from a
single-view image,” arXiv preprint arXiv:2309.03453, 2023. 1, 3
[19] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, “Mv-
dream: Multi-view diffusion for 3d generation,” arXiv preprint
arXiv:2308.16512, 2023. 1, 3, 4, 5, 8
[20] J. Li, H. Tan, K. Zhang, Z. Xu, F. Luan, Y. Xu, Y. Hong,
K. Sunkavalli, G. Shakhnarovich, and S. Bi, “Instant3d: Fast
text-to-3d with sparse-view generation and large reconstruction
model,” arXiv preprint arXiv:2311.06214, 2023. 1, 3
[21] Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, D. Liu, F. Liu,
K. Sunkavalli, T. Bui, and H. Tan, “Lrm: Large reconstruction
model for single image to 3d,” arXiv preprint arXiv:2311.04400,
2023. 1
[22] Y. Xu, H. Tan, F. Luan, S. Bi, P. Wang, J. Li, Z. Shi, K. Sunkavalli,
G. Wetzstein, Z. Xu et al., “Dmv3d: Denoising multi-view dif-
fusion using 3d large reconstruction model,” arXiv preprint
arXiv:2311.09217, 2023. 1
[23] J. Xu, W. Cheng, Y. Gao, X. Wang, S. Gao, and Y. Shan, “In-
stantmesh: Efficient 3d mesh generation from a single image
with sparse-view large reconstruction models,” arXiv preprint
arXiv:2404.07191, 2024. 1, 3
[24] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, “Lgm:
Large multi-view gaussian model for high-resolution 3d content
creation,” arXiv preprint arXiv:2402.05054, 2024. 1, 3, 6, 8
[25] Y. Xu, Z. Shi, W. Yifan, H. Chen, C. Yang, S. Peng, Y. Shen,
and G. Wetzstein, “Grm: Large gaussian reconstruction model
for efficient 3d reconstruction and generation,” arXiv preprint
arXiv:2403.14621, 2024. 1, 3
[26] Z. Chen, J. Tang, Y. Dong, Z. Cao, F. Hong, Y. Lan, T. Wang, H. Xie,
T. Wu, S. Saito et al., “3dtopia-xl: Scaling high-quality 3d asset gen-
eration via primitive diffusion,” arXiv preprint arXiv:2409.12957,
2024. 1, 3, 8
[27] Z. Huang, Y.-C. Guo, H. Wang, R. Yi, L. Ma, Y.-P. Cao, and
L. Sheng, “Mv-adapter: Multi-view consistent image generation
made easy,” arXiv preprint arXiv:2412.03632, 2024. 1
[28] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang,
“Neus: Learning neural implicit surfaces by volume rendering for
multi-view reconstruction,” arXiv preprint arXiv:2106.10689, 2021.
1, 3
[29] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ra-
mamoorthi, and R. Ng, “Nerf: Representing scenes as neural
radiance fields for view synthesis,” Communications of the ACM,
vol. 65, no. 1, pp. 99–106, 2021. 1, 3
[30] L. Qiu, G. Chen, X. Gu, Q. Zuo, M. Xu, Y. Wu, W. Yuan, Z. Dong,
L. Bo, and X. Han, “Richdreamer: A generalizable normal-depth
diffusion model for detail richness in text-to-3d,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 9914–9925. 1, 2, 3, 7
[31] Z. Li, T. Wu, J. Tan, M. Zhang, J. Wang, and D. Lin, “Idarb:
Intrinsic decomposition for arbitrary number of input views and
illuminations,” arXiv preprint arXiv:2412.12083, 2024. 1, 3, 4, 5, 8
[32] R. Liang, Z. Gojcic, H. Ling, J. Munkberg, J. Hasselgren, Z.-H. Lin,
J. Gao, A. Keller, N. Vijaykumar, S. Fidler et al., “Diffusionren-
derer: Neural inverse and forward rendering with video diffusion
models,” arXiv preprint arXiv:2501.18590, 2025. 1
[33] B. Sun, M. Jin, B. Yin, and Q. Hou, “Depth anything at any
condition,” arXiv preprint arXiv:2507.01634, 2025. 1
[34] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d
gaussian splatting for real-time radiance field rendering,” ACM
Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Avail-
able: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
2, 3, 4, 7
[35] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian
splatting for geometrically accurate radiance fields,” in ACM
SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11. 2, 4, 7
[36] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, “Splatter image:
Ultra-fast single-view 3d reconstruction,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 10 208–10 217. 2, 4
[37] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. Van-
derBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi,
“Objaverse: A universe of annotated 3d objects,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 13 142–13 153. 2, 3, 7
[38] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic
models,” Advances in neural information processing systems, vol. 33,
pp. 6840–6851, 2020. 3
[39] T.
Shen,
J.
Munkberg,
J.
Hasselgren,
K.
Yin,
Z.
Wang,
W. Chen, Z. Gojcic, S. Fidler, N. Sharp, and J. Gao, “Flexible
isosurface extraction for gradient-based mesh optimization,”
ACM Trans. Graph., vol. 42, no. 4, jul 2023. [Online]. Available:
https://doi.org/10.1145/3592430 3
[40] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep march-
ing tetrahedra: a hybrid representation for high-resolution 3d
shape synthesis,” Advances in Neural Information Processing Systems,
vol. 34, pp. 6087–6101, 2021. 3
[41] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and
A. Geiger, “Occupancy networks: Learning 3d reconstruction in
function space,” in Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, 2019, pp. 4460–4470. 3
[42] X. Yu, Y.-C. Guo, Y. Li, D. Liang, S.-H. Zhang, and X. Qi,
“Text-to-3d
with
classifier
score
distillation,”
arXiv
preprint
arXiv:2310.19415, 2023. 3
[43] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Lucid-
dreamer: Towards high-fidelity text-to-3d generation via interval
score matching,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 6517–6526. 3
[44] T. Yi, J. Fang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian,
and X. Wang, “Gaussiandreamer: Fast generation from text to
3d gaussian splatting with point cloud priors,” arXiv preprint
arXiv:2310.08529, 2023. 3
[45] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli,
and Z. Xu, “Gs-lrm: Large reconstruction model for 3d gaussian
splatting,” arXiv preprint arXiv:2404.19702, 2024. 3
[46] C. Fang, X. Hu, K. Luo, and P. Tan, “Ctrl-room: Controllable
text-to-3d room meshes generation with layout constraints,” arXiv
preprint arXiv:2310.03602, 2023. 3
[47] J. Ren, L. Pan, J. Tang, C. Zhang, A. Cao, G. Zeng, and
Z. Liu, “Dreamgaussian4d: Generative 4d gaussian splatting,”
arXiv preprint arXiv:2312.17142, 2023. 3
[48] U. Singer, S. Sheynin, A. Polyak, O. Ashual, I. Makarov, F. Kokki-
nos, N. Goyal, A. Vedaldi, D. Parikh, J. Johnson et al., “Text-to-4d
dynamic scene generation,” arXiv preprint arXiv:2301.11280, 2023.
3
[49] H. Ling, S. W. Kim, A. Torralba, S. Fidler, and K. Kreis, “Align your
gaussians: Text-to-4d with dynamic 3d gaussians and composed
diffusion models,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 8576–8588. 3
[50] M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati,
A. Fan, C. Laforte, V. Voleti, S. Y. Gadre et al., “Objaverse-xl:
A universe of 10m+ 3d objects,” Advances in Neural Information
Processing Systems, vol. 36, 2024. 3, 7
[51] X. Zhang, Y. Zhou, K. Wang, Y. Wang, Z. Li, S. Jiao, D. Zhou,
Q. Hou, and M.-M. Cheng, “Ar-1-to-3: Single image to consistent
3d object generation via next-view prediction,” arXiv preprint
arXiv:2503.12929, 2025. 3
[52] W. Li, R. Chen, X. Chen, and P. Tan, “Sweetdreamer: Aligning
geometric priors in 2d diffusion for consistent text-to-3d,” arXiv
preprint arXiv:2310.02596, 2023. 3

<!-- page 13 -->
13
[53] J. Yan, Y. Gao, Q. Yang, X. Wei, X. Xie, A. Wu, and W.-S. Zheng,
“Dreamview: Injecting view-specific text guidance into text-to-3d
generation,” in European Conference on Computer Vision.
Springer,
2024, pp. 358–374. 3, 4, 5, 8
[54] Z. Wang, Y. Wang, Y. Chen, C. Xiang, S. Chen, D. Yu, C. Li, H. Su,
and J. Zhu, “Crm: Single image to 3d textured mesh with con-
volutional reconstruction model,” arXiv preprint arXiv:2403.05034,
2024. 3
[55] A. Chen, H. Xu, S. Esposito, S. Tang, and A. Geiger, “Lara: Efficient
large-baseline radiance fields,” arXiv preprint arXiv:2407.04699,
2024. 3, 8
[56] H. Hu, T. Yin, F. Luan, Y. Hu, H. Tan, Z. Xu, S. Bi, S. Tulsiani,
and K. Zhang, “Turbo3d: Ultra-fast text-to-3d generation,” arXiv
preprint arXiv:2412.04470, 2024. 3
[57] C. Lin, P. Pan, B. Yang, Z. Li, and Y. Mu, “Diffsplat: Repurposing
image diffusion models for scalable gaussian splat generation,”
arXiv preprint arXiv:2501.16764, 2025. 3, 8
[58] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen,
X. Tong, and J. Yang, “Structured 3d latents for scalable and
versatile 3d generation,” in Proceedings of the Computer Vision and
Pattern Recognition Conference, 2025, pp. 21 469–21 480. 3, 8
[59] B. Zhang, J. Tang, M. Nießner, and P. Wonka, “3dshape2vecset: A
3d shape representation for neural fields and generative diffusion
models,” ACM Trans. Graph., vol. 42, no. 4, jul 2023. [Online].
Available: https://doi.org/10.1145/3592442 3
[60] W. Li, J. Liu, H. Yan, R. Chen, Y. Liang, X. Chen, P. Tan, and
X. Long, “Craftsman3d: High-fidelity mesh generation with 3d
native generation and interactive geometry refiner,” arXiv preprint
arXiv:2405.14979, 2024. 3
[61] R. Chen, J. Zhang, Y. Liang, G. Luo, W. Li, J. Liu, X. Li, X. Long,
J. Feng, and P. Tan, “Dora: Sampling and benchmarking for 3d
shape variational auto-encoders,” in Proceedings of the Computer
Vision and Pattern Recognition Conference, 2025, pp. 16 251–16 261. 3
[62] Y. Liu, P. Wang, C. Lin, X. Long, J. Wang, L. Liu, T. Komura,
and W. Wang, “Nero: Neural geometry and brdf reconstruction
of reflective objects from multiview images,” ACM Transactions on
Graphics (TOG), vol. 42, no. 4, pp. 1–22, 2023. 3
[63] Z.-L. Zhu, B. Wang, and J. Yang, “Gs-ror: 3d gaussian splat-
ting for reflective object relighting via sdf priors,” arXiv preprint
arXiv:2406.18544, 2024. 3
[64] X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman,
and J. T. Barron, “Nerfactor: Neural factorization of shape and
reflectance under an unknown illumination,” ACM Transactions on
Graphics (ToG), vol. 40, no. 6, pp. 1–18, 2021. 3
[65] J. Li, L. Wang, L. Zhang, and B. Wang, “Tensosdf: Roughness-
aware tensorial representation for robust geometry and material
reconstruction,” ACM Transactions on Graphics (TOG), vol. 43, no. 4,
pp. 1–13, 2024. 3
[66] Z.-L. Zhu, J. Yang, and B. Wang, “Gaussian splatting with dis-
cretized sdf for relightable assets,” in Proceedings of IEEE Interna-
tional Conference on Computer Vision (ICCV), 2025. 3
[67] Z. Liu, Y. Li, Y. Lin, X. Yu, S. Peng, Y.-P. Cao, X. Qi,
X. Huang, D. Liang, and W. Ouyang, “Unidream: Unifying dif-
fusion priors for relightable text-to-3d generation,” arXiv preprint
arXiv:2312.08754, 2023. 3
[68] X. Xu, Z. Lyu, X. Pan, and B. Dai, “Matlaber: Material-
aware text-to-3d via latent brdf auto-encoder,” arXiv preprint
arXiv:2308.09278, 2023. 3
[69] Y. Siddiqui, T. Monnier, F. Kokkinos, M. Kariya, Y. Kleiman,
E. Garreau, O. Gafni, N. Neverova, A. Vedaldi, R. Shapovalov
et al., “Meta 3d assetgen: Text-to-mesh generation with high-
quality geometry, texture, and pbr materials,” arXiv preprint
arXiv:2407.02445, 2024. 3
[70] X. Feng, C. Yu, Z. Bi, Y. Shang, F. Gao, H. Wu, K. Zhou, C. Jiang,
and Y. Yang, “Arm: Appearance reconstruction model for re-
lightable 3d generation,” arXiv preprint arXiv:2411.10825, 2024. 3
[71] B. Xiong, J. Liu, J. Hu, C. Wu, J. Wu, X. Liu, C. Zhao, E. Ding, and
Z. Lian, “Texgaussian: Generating high-quality pbr material via
octree-based 3d gaussian splatting,” in Proceedings of the Computer
Vision and Pattern Recognition Conference, 2025, pp. 551–561. 3
[72] X. Yu, Z. Yuan, Y.-C. Guo, Y.-T. Liu, J. Liu, Y. Li, Y.-P. Cao,
D. Liang, and X. Qi, “Texgen: a generative diffusion model for
mesh textures,” ACM Transactions on Graphics (TOG), vol. 43, no. 6,
pp. 1–14, 2024. 3
[73] B. Ummenhofer, S. Agrawal, R. Sep´ulveda, Y. Lao, K. Zhang,
T. Cheng, S. R. Richter, S. Wang, and G. Ros, “Objects with lighting:
A real-world dataset for evaluating reconstruction and rendering
for object relighting,” in 3DV.
IEEE, 2024. 3, 6
[74] Z. Dong, K. Chen, Z. Lv, H.-X. Yu, Y. Zhang, C. Zhang, Y. Zhu,
S. Tian, Z. Li, G. Moffatt et al., “Digital twin catalog: A large-scale
photorealistic 3d object digital twin dataset,” in Proceedings of the
Computer Vision and Pattern Recognition Conference, 2025, pp. 753–
763. 3
[75] H. Wang, Z. Wang, X. Long, C. Lin, G. Hancke, and R. W. Lau,
“Mage: Single image to material-aware 3d via the multi-view g-
buffer estimation model,” in Proceedings of the Computer Vision and
Pattern Recognition Conference, 2025, pp. 10 985–10 995. 3
[76] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer,
“High-resolution image synthesis with latent diffusion models,” in
Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 10 684–10 695. 4
[77] P. Wang and Y. Shi, “Imagedream: Image-prompt multi-view
diffusion for 3d generation,” arXiv preprint arXiv:2312.02201, 2023.
4, 5, 7, 8
[78] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control
to text-to-image diffusion models,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 3836–3847. 5
[79] J. Munkberg, J. Hasselgren, T. Shen, J. Gao, W. Chen, A. Evans,
T. M¨uller, and S. Fidler, “Extracting triangular 3d models, mate-
rials, and lighting from images,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
8280–8290. 5
[80] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “Gs-ir: 3d gaussian
splatting for inverse rendering,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 644–21 653. 5
[81] J. T. Kajiya, “The rendering equation,” in Proceedings of the 13th
annual conference on Computer graphics and interactive techniques,
1986, pp. 143–150. 5
[82] W. Palfinger, “Continuous remeshing for inverse rendering,” Com-
puter Animation and Virtual Worlds, vol. 33, no. 5, p. e2101, 2022.
6
[83] Q.-Y. Zhou, J. Park, and V. Koltun, “Open3D: A modern library for
3D data processing,” arXiv:1801.09847, 2018. 6
[84] B. O. Community, Blender - a 3D modelling and rendering package,
Blender Foundation, Stichting Blender Foundation, Amsterdam,
2018. [Online]. Available: http://www.blender.org 6
[85] W. Jakob, S. Speierer, N. Roussel, M. Nimier-David, D. Vicini,
T. Zeltner, B. Nicolet, M. Crespo, V. Leroy, and Z. Zhang, “Mitsuba
3 renderer,” 2022, https://mitsuba-renderer.org. 6
[86] W. Jakob, S. Speierer, N. Roussel, and D. Vicini, “Dr. jit: A just-in-
time compiler for differentiable rendering,” ACM Transactions on
Graphics (TOG), vol. 41, no. 4, pp. 1–19, 2022. 6
[87] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
“The unreasonable effectiveness of deep features as a perceptual
metric,” in Proceedings of the IEEE conference on computer vision and
pattern recognition, 2018, pp. 586–595. 7, 8
[88] M. Team, “Meshy - Free 3D AI Model Generator.” [Online].
Available: https://www.meshy.ai/ 8
[89] L.
Ai,
“Luma
AI
-
Genie.”
[Online].
Available:
https:
//lumalabs.ai/genie 8
[90] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agar-
wal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning
transferable visual models from natural language supervision,”
in International conference on machine learning.
PMLR, 2021, pp.
8748–8763. 8
[91] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochre-
iter, “Gans trained by a two time-scale update rule converge to a
local nash equilibrium,” Advances in neural information processing
systems, vol. 30, 2017. 8
[92] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford,
and X. Chen, “Improved techniques for training gans,” Advances
in neural information processing systems, vol. 29, 2016. 8
[93] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,”
IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612,
2004. 8

<!-- page 14 -->
14
Ze-Xin Yin received the bachelor‘s degree from
Xidian University, in 2017. He is currently work-
ing toward the PhD degree under the supervi-
sion of Prof. Jin Xie with the College of Com-
puter Science, Nankai University. His research
interests mainly focus on neural radiance fields
and 3D computer vision.
Jiaxiong Qiu received the bachelor‘s degree
from Dalian Maritime University, in 2017, the
master degree from the University of Electronic
Science and Technology of China, in 2020, and
the PhD degree from the College of Computer
Science, Nankai University, in 2024. His re-
search interests include computer vision, com-
puter graphics, robotics, and deep learning.
Liu Liu is currently a 3D Vision Researcher at
the Robot Lab of Horizon Robotics. He received
his Master’s degree from Southeast University in
2019, advised by Prof. Fujun Yang. His research
interests include 3D Vision and Embodied AI.
Xinjie Wang is a research engineer at Horizon
Robotics. He received his Master’s degree in
Mechanical Automation from Tongji University in
2020. His research interests include computer
vision and Embodied AI.
Wei Sui is the Algorithm Leader at D-Robotics,
responsible for the algorithms related to em-
bodied intelligence. Before that, Dr. Wei Sui
was leading the 3D Vision Team at Horizon
Robotics. His research interests include Struc-
ture from Motion (SFM), Simultaneous Local-
ization and Mapping (SLAM), Neural Radiance
Field (NeRF), 3D perception, etc. Dr. Sui re-
ceived his B.Eng. and Ph.D. degrees from Bei-
hang University and the National Laboratory
of Pattern Recognition (NLPR), Institute of Au-
tomation, Chinese Academy of Sciences (CASIA), Beijing, China, in
2011 and 2016, respectively. Dr. Wei Sui has published one research
monograph and more than ten papers in TIP, TVCG, ICRA, CVPR, etc.
In addition, he holds over 40 Chinese patents and 5 U.S. patents.
Zhizhong Su is currently the Director of Robot
Lab at Horizon Robotics. He received his M.S.
Degree from Indiana University Bloomington,
USA and B.E. Degree from Shanghai Jiao Tong
University, China. His research interests include
autonomous driving and robotics learning.
Jian Yang received the PhD degree from
Nanjing University of Science and Technology
(NJUST) in 2002, majoring in pattern recognition
and intelligence systems. From 2003 to 2007,
he was a Postdoctoral Fellow at the University
of Zaragoza, Hong Kong Polytechnic University
and New Jersey Institute of Technology, respec-
tively. From 2007 to present, he is a professor in
the School of Computer Science and Technology
of NJUST. He is the author of more than 300
scientific papers in pattern recognition and com-
puter vision. His papers have been cited over 56000 times in the Scholar
Google. His research interests include pattern recognition and computer
vision. Currently, he is/was an associate editor of Pattern Recognition,
Pattern Recognition Letters, IEEE Trans. Neural Networks and Learning
Systems, and Neurocomputing. He is a Fellow of IAPR.
Jin Xie received the Ph.D degree in comput-
ing from the Department of Computing, The
Hong Kong Polytechnic University, Hong Kong,
in 2012. He is currently a Professor with the
School of Intelligence Science and Technology,
Nanjing University, Nanjing, China. He was a
Research Scientist with New York University Abu
Dhabi from 2013 to 2017. Prior to joining Nanjing
University in 2023, he was a Professor with the
Department of Computer Science and Engineer-
ing, Nanjing University of Science and Technol-
ogy, Nanjing, China. His research interests include machine learning,
computer vision, computer graphics, and robotics. His current research
focus is on 3-D computer vision and its applications on autonomous
driving and robotic manipulation. Dr. Xie has authored/co-authored
more than 50 papers in well-known journals/conferences such as IEEE
TPAMI, IJCV, CVPR, ICCV, ECCV and NeurIPS. He has served as a re-
viewer for IEEE TPAMI, TIP, TNNLS, TMM, CVPR, ICCV and ECCV. He
was a special issue chair for Asian Conference on Pattern Recognition
2017 and a Guest Editor for Pattern Recognition. He was a recipient
of the best paper award for Asian Conference on Pattern Recognition
2021.
