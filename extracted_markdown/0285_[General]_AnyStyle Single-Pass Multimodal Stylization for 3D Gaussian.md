<!-- page 1 -->
AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting
Joanna Kaleta 1 2 Bartosz ´Swirta 1 Kacper Kania 1 Przemysław Spurek 3 4 Marek Kowalski 5
Abstract
The growing demand for rapid and scalable 3D
asset creation has driven interest in feed-forward
3D reconstruction methods, with 3D Gaussian
Splatting (3DGS) emerging as an effective scene
representation. While recent approaches have
demonstrated pose-free reconstruction from un-
posed image collections, integrating stylization
or appearance control into such pipelines remains
underexplored. Existing attempts largely rely on
image-based conditioning, which limits both con-
trollability and flexibility. In this work, we intro-
duce AnyStyle, a feed-forward 3D reconstruction
and stylization framework that enables pose-free,
zero-shot stylization through multimodal condi-
tioning. Our method supports both textual and
visual style inputs, allowing users to control the
scene appearance using natural language descrip-
tions or reference images. We propose a modular
stylization architecture that requires only mini-
mal architectural modifications and can be inte-
grated into existing feed-forward 3D reconstruc-
tion backbones. Experiments demonstrate that
AnyStyle improves style controllability over prior
feed-forward stylization methods while preserv-
ing high-quality geometric reconstruction. A user
study further confirms that AnyStyle achieves su-
perior stylization quality compared to an existing
state-of-the-art approach. Repository: https:
//github.com/joaxkal/AnyStyle.
1. Introduction
The rapid progress of feed-forward 3D reconstruction meth-
ods (Wang et al., 2023; Leroy et al., 2024; Ye et al., 2024;
Wang et al., 2025a; Jiang et al., 2025), has enabled the
creation of realistic 3D assets from sparse image inputs in
seconds, making them increasingly relevant for applications
1Warsaw
University
of
Technology
2Sano
Cen-
tre
for
Computational
Medicine
3Jagiellonian
Univer-
sity
4IDEAS
NCBR
5Microsoft.
Correspondence
to:
<joanna.kaleta.dokt@pw.edu.pl>.
Preprint. February 5, 2026.
Figure 1. Teaser. Given a set of unposed input images and a style
conditioning signal provided as either text or an image, our method
generates a stylized 3D scene represented with 3D Gaussian Splats
in a single forward pass. The reconstructed scene can be stylized
in under 0.1 second per input content image.
in teleconferencing, gaming, and film production. Building
upon this foundation, AnySplat introduced a pose-free re-
construction pipeline that generalizes across unconstrained
image collections, representing a significant step toward
scalable, data-driven 3D reconstruction.
More recently, Styl3R (Wang et al., 2025b) and Stylos (Liu
et al., 2025) have explored simultaneous 3D reconstruction
and artistic stylization, enabling users to transfer the visual
style of a reference image directly into the reconstructed 3D
scene. These models demonstrated that stylization can be
integrated into a single forward pass, eliminating the need
for iterative optimization required by earlier works (Howil
et al., 2025; Liu et al., 2024b; Kovács et al., 2024a).
However, these approaches remain limited in two aspects.
First, the style control is restricted to visual exemplars: a
single style image determines the overall aesthetic, leav-
ing little room for fine-grained or semantic manipulation.
Second, existing 3D stylization approaches (Wang et al.,
2025b; Liu et al., 2025) use architectural components that
require training the underlying 3D reconstruction method
from scratch. This design makes the application of those
methods to new architectures difficult and reduces modular-
ity.
1
arXiv:2602.04043v1  [cs.CV]  3 Feb 2026

<!-- page 2 -->
AnyStyle
We introduce AnyStyle, a multimodal, zero-shot styliza-
tion framework for pose-free 3D Gaussian Splatting that
overcomes those limitations. AnyStyle unifies textual and
visual style control through a shared CLIP-based embedding
space, enabling stylization to be guided by style images, text
prompts, or continuous interpolation between the two. This
multimodal conditioning provides both semantic flexibility
and fine-grained control: for instance, when a reference
image produces a nearly desired appearance, user can refine
the result by steering the style embedding toward textual
cues such as “expressive brushwork” or “softer color tones,”
without the need to search for an alternative style image.
Crucially, our approach enables practical and efficient styl-
ization through its architecture-agnostic design. Unlike
existing feed-forward methods (Liu et al., 2025; Wang et al.,
2025b) that require training both geometric and styliza-
tion branches jointly from scratch, our method initializes
from a pretrained 3D reconstruction model and only adds a
lightweight style injection module that is fine-tuned along-
side the backbone. Our style injection approach is inspired
by ControlNet (Zhang et al., 2023), which introduced zero-
initialized convolutions for conditional control in 2D diffu-
sion models. To the best of our knowledge, we are the first
to adapt this mechanism for stylization in feed-forward 3D
reconstruction.
Unlike attention-based approaches (Chung et al., 2024a;b;
Han et al., 2025) that modify internal Query, Key, and Value
computations, our method injects style through additive
conditioning via zero-initialized convolutions. This design
offers two key advantages: (1) modularity – the style in-
jection module can be added to any pretrained attention-
based model without altering its internal mechanics; and (2)
architectural generality – stylization is decoupled from
attention mechanism, enabling style control at any stage
of the network, including tokens that do not re-enter atten-
tion layers. This flexibility allows integration into diverse
transformer-based reconstruction models regardless of their
attention patterns or layer configurations. We further val-
idate this property in our ablation study by stylizing only
tokens routed directly to prediction head.
Experiments demonstrate that AnyStyle achieves superior
stylization quality compared to existing methods, with the
best ArtFID scores among feed-forward approaches and sta-
tistically significant preference in a user study. Our method
maintains this quality advantage while not requiring training
from scratch.
In summary, AnyStyle advances controllable 3D stylization
by offering:
• Novel architecture applying zero-convolution condition-
ing to style injection and enabling multimodal style con-
trol via CLIP embeddings that combine text and image
inputs,
Table 1. Comparison of properties of discussed 3D reconstruction
and stylization methods.
Method
Multimodal
Scene
Style
Pose
Fast
Style Input
Zero-Shot
Zero-Shot
Free
Inference
ARF, SNeRF, StylizedNeRF
✗
✗
✗
✗
✗
StyleRF
✗
✗
✓
✗
✗
SGSST, G-Style, StylizedGS
✗
✗
✗
✗
✓
StyleGaussian
✗
✗
✓
✗
✓
ClipGaussian
✓
✗
✗
✗
✓
Stylos, Styl3r
✗
✓
✓
✓
✓
AnyStyle
✓
✓
✓
✓
✓
• Semantic and continuous style refinement through prompt
manipulation or interpolation,
• Architecture-agnostic design, enabling integration into
any transformer-based model without modifying the orig-
inal architecture or requiring additional pretraining.
These contributions enable expressive, flexible, and intuitive
control over 3D stylization in a single feed-forward pass.
2. Related Work
Neural 3D Scene Representations. Neural Radiance Fields
(NeRFs) (Mildenhall et al., 2021) revolutionized 3D scene
reconstruction by representing scenes as continuous volu-
metric functions optimized from multi-view images. While
achieving photorealistic novel view synthesis, NeRFs are
difficult to integrate with classical graphics engines and
many variants are slow to render.
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) addresses
these limitations by representing scenes as collections of 3D
Gaussians with learnable parameters, enabling real-time ren-
dering while maintaining high visual quality. This explicit
representation has become the foundation for numerous
downstream applications, including style transfer.
Feed-Forward 3D Reconstruction. Traditional NeRF and
3DGS methods require per-scene optimization, taking tens
of minutes on a single GPU to reconstruct each scene. This
computational burden limits their practical applicability,
particularly for applications requiring real-time or near-real-
time processing. To address this limitation, recent methods
leverage learned structural priors (Wang et al., 2023; Leroy
et al., 2024; Wang et al., 2025a; Charatan et al., 2024; Chen
et al., 2024b; Szymanowicz et al., 2024) to reconstruct 3D
point maps in a single feed-forward pass. Building on these
principles, subsequent approaches directly predict parame-
ters of 3D Gaussians (Smart et al., 2024; Ye et al., 2024),
eliminating the iterative optimization process entirely. Most
notably, AnySplat (Jiang et al., 2025) extends VGGT (Wang
et al., 2025a) to reconstruct 3D Gaussians without reliance
on calibrated cameras, enabling reconstruction from casual
image captures. The feed-forward nature of these methods is
particularly advantageous for 3D stylization, as it enables ap-
plying artistic styles to novel scenes without scene-specific
2

<!-- page 3 -->
AnyStyle
Figure 2. Method overview. AnyStyle takes unposed content images of a scene together with an arbitrary style input (text or image)
and produces a stylized 3D Gaussian representation from which novel stylized views can be rendered. The architecture follows a
dual-branch design that decouples geometric reconstruction from appearance stylization. Content images are processed by a pretrained
frozen backbone to recover geometry and camera poses, while the style signal is embedded using CLIP and applied only within the style
branch to control scene appearance. A pretrained AnySplat model initializes the copied Aggregator ˜A and Gaussian Head ˜Hgs in the style
branch. These components are subsequently fine-tuned with CLIP-based conditioning via style injection. The outputs of the two branches
are combined by a Gaussian Adapter and rendered to produce the final stylized views.
fine-tuning. We leverage AnySplat’s architecture as the
foundation for our stylization framework.
Style Transfer: From 2D to 3D. Classical 2D style trans-
fer methods (Gatys et al., 2016) match statistics of VGG-
extracted features (e.g., Gram matrices) between content
and style images. This approach enabled the development
of faster feed-forward methods (Huang & Belongie, 2017;
Li et al., 2017; 2019; Park & Lee, 2019; Liu et al., 2021;
Wu et al., 2021; Deng et al., 2022; Zhang et al., 2024b;
Tang et al., 2023), and eventually text-guided domain adap-
tation (Gal et al., 2022) and stylization (Kwon & Ye, 2022;
Radford et al., 2021; Suresh et al., 2024) using CLIP for
fine-grained control through natural language. However,
naïve per-view stylization of 3D scenes produces severe
multi-view inconsistencies.
Early 3D methods trained NeRF models on stylized multi-
view images (Zhang et al., 2022; Nguyen-Phuoc et al., 2022;
Huang et al., 2022; Liu et al., 2023), requiring lengthy per-
scene optimization. The emergence of 3DGS enabled faster
approaches (Liu et al., 2024a; Galerne et al., 2025; Kovács
et al., 2024b; Zhang et al., 2024c; Yu et al., 2024), with some
incorporating text prompts (Howil et al., 2025). However,
all these methods rely on per-scene optimization, requiring
scene-specific reconstruction and, in most cases, additional
stylization optimization. An overview comparing 3D style
transfer methods is shown in Table 1.
Recent feed-forward alternatives (Wang et al., 2025b; Liu
et al., 2025) eliminate per-scene optimization by predicting
stylized 3D Gaussians in a single pass. However, they rely
solely on a single style image as input, which limits fine-
grained control over specific stylistic elements (e.g., color
palette vs. brush strokes). Moreover, they require heavy
training process in which both, geometric backbone and
stylization branch are trained jointly.
AnyStyle addresses these limitations through multi-modal
conditioning with both style images and text prompts, us-
ing CLIP-based style embedding, enabling zero-shot style
transfer in a single feed-forward pass with fine-grained artis-
tic control. Inspired by ControlNet, our approach allows
for extending any attention-based pre-trained model with
stylization branch.
3. Method
AnySplat Backbone. AnyStyle is constructed upon AnyS-
plat (Jiang et al., 2025), a feed-forward 3D reconstruction
framework that predicts a complete 3D Gaussian scene rep-
resentation from unposed multi-view images in a single for-
ward pass. Given a set of input images {Ii}N
i=1, each image
is processed by a frozen DINO (Caron et al., 2021) feature
extractor f(·) to produce local patch features: Fi = f(Ii)
for i = 1, . . . , N.
The Aggregator A is composed of multiple self-attention
blocks with alternating local and global self-attention and
processes the DINO patch features of the input. At each
transformer layer l ∈{1, . . . , L}, the Aggregator produces
a sequence of token representations, denoted as T(l). Token
representations from a selected subset of intermediate layers
S ⊂{1, . . . , L −1}, together with the final-layer tokens
T(L), are used to form a unified scene representation, which
3

<!-- page 4 -->
AnyStyle
we denote compactly as
Tagg = {T(l) | l ∈S ∪{L}} = A({Fi}N
i=1).
(1)
This merged token representation is then provided as input
to three specialized reconstruction heads. The Camera Head
Hcam estimates camera parameters for each input view. The
Depth Head Hdepth predicts depth and unprojects it to a set
of 3D point locations {µi}n
i=1 that define the centers of
the Gaussian primitives for each pixel in each input view.
The Gaussian Head Hgs predicts the remaining geometric
and appearance attributes, yielding a set of n anisotropic
3D Gaussian primitives representing scene geometry and
appearance:
Gi = (µi, ri, si, σi, ci) ,
(2)
where ri denotes the rotation, si the scale, σi the opacity,
and ci the spherical-harmonic color coefficients.
To reduce redundancy while preserving geometric and ap-
pearance fidelity, AnySplat employs a voxelization module
that clusters nearby Gaussians in 3D space and merges them
via confidence-weighted averaging. The resulting Gaussian
set is rendered using differentiable Gaussian splatting to
produce high-quality novel views.
AnyStyle Architecture Overview. The AnyStyle architec-
ture adopts a dual-branch design that explicitly decouples
geometric reconstruction from artistic stylization. As il-
lustrated in Figure 2, the framework consists of a Frozen
Backbone original weights and a dedicated Style Branch.
The backbone remains frozen throughout training in or-
der to preserve the structural integrity of the reconstructed
scene, including camera poses and the underlying 3D point
distribution. We use AnySplat as backbone, but thanks
to AnyStyle’s architecture we could switch to a different
backbone, e.g. DepthAnything 3 (Lin et al., 2025).
In parallel to the frozen backbone, the Style Branch com-
prises a copied Gaussian head ˜Hgs and a copied Aggregator
˜A module (see Injection in Style Branch paragraph in Sec-
tion 3), which are fine-tuned to adapt the appearance of the
scene. Stylization is driven by multimodal conditioning,
where a style input—either text or an image—is embedded
into a shared CLIP feature space. The core mechanism en-
abling appearance adaptation is the style injection operation,
which is implemented via a dedicated Style Injector module.
Finally, a Gaussian Adapter combines the geometric pa-
rameters µi, si, σi obtained from the frozen backbone with
the stylized features ˜ri, ˜ci produced by the Style Branch.
These features are predicted by the Style Branch to improve
expressiveness without degrading geometric accuracy (e.g.,
by shrinking Gaussians or making them overly transparent).
The resulting set of stylized 3D Gaussian primitives is ren-
dered using a differentiable renderer to produce the final
output views.
Multimodal Style Conditioning. To support both text- and
image-based style control, we embed all style inputs into a
shared CLIP feature space. Specifically, we employ Long-
CLIP (Zhang et al., 2024a), which improves the modeling
of long-range semantics and fine-grained stylistic attributes,
enabling richer and more expressive style representations
from text.
Given a style image SI or a text prompt ST , we extract a
Long-CLIP embedding
zs = ECLIP(SI) or zs = ECLIP(ST ),
zs ∈Rds.
(3)
Although Long-CLIP aligns text and image embeddings
within a shared latent space, a residual modality gap persists
in practice. To mitigate this effect, we alternate between text-
conditioned and image-conditioned training batches. This
training strategy encourages the network to learn a unified
style representation that generalizes across both modalities.
Zero-Convolution Style Injection. To enable appearance
stylization, we inject style information into feature represen-
tations processed by the network. Style conditioning can be
applied at selected intermediate layers within the Aggregator
or/and to the features at the output of the Aggregator.
For this purpose, we introduce a Style Injector module,
denoted as I(·, ·). For each style input (text or image), we
extract the style embedding zs and project it to the token di-
mension using a lightweight multi-layer perceptron (MLP).
The resulting projected style embedding is then injected into
the features via additive modulation.
Naively injecting style features directly into a representation
can destabilize training , (Zhang et al., 2023). To address
this issue, we adopt zero-initialized convolutional adapters,
inspired by (Zhang et al., 2023; Lu et al., 2025). At initializa-
tion, these adapters output zeros, ensuring that the network
behaves identically to the pretrained AnySplat model before
stylization training begins.
Formally, given a token f ∈Rdf and a style embedding
zs ∈Rds, the Style Injector produces a style-conditioned
representation defined as
˜f = I(f, zs) = f + ZeroConv(Proj(zs)),
(4)
where Proj : Rds →Rdf denotes a projection MLP and
ZeroConv(·) is a zero-initialized convolutional layer as in
(Zhang et al., 2023). ZeroConv(·) is a single layer, with 1x1
convolution, zero padding and chin = chout, that performs
weighted multiplication of input channels.
In AnyStyle, each injection location corresponds to a spe-
cific Aggregator layer and employs its own dedicated Style
Injector instance. This allows layer-wise adaptation to fea-
ture representations of different dimensionalities.
Style Injection in Style Branch.
In this paragraph we
4

<!-- page 5 -->
AnyStyle
Figure 3. Comparison between AnyStyle and existing 3D style transfer methods with different architectural designs: feed-forward
(purple), per-scene optimization (green), and hybrid approaches (blue). Our method achieves high-quality style transfer while faithfully
preserving fine details (top row) as well as overall scene structure. All compared methods are conditioned on a style image.
consider two location at which the Style Injector I(·, ·) is
applied.
(i) Head injection. Style injection is applied to the tokens
exiting ˜A, included in representation Tagg, that is fed into
the Gaussian Head:
˜t(l) = I(l)(t(l), zs),
∀t(l) ∈T(l), T(l) ∈Tagg. (5)
(ii) Aggregator injection Style information is injected into
token representations at selected intermediate layers, which
are processed further by the aggregator ˜A.
Let Tx
(l) denote the set of token representations produced
at ˜A layer l. We denote by P ⊂{1, . . . , L −1} the set
of selected intermediate layers at which expressive style
injection is applied. For each layer l ∈P, style injection is
applied to all tokens at that layer:
˜t(l)
x = I(l)(t(l)
x , zs),
∀t(l)
x ∈T(l)
x , l ∈P.
(6)
This allows style cues to influence token interactions within
the ˜A itself.
Training losses. Our training objective builds on losses
used in feed-forward 3D stylization and augments them with
CLIP-based alignment. While prior methods mainly rely on
VGG(Guan et al., 2019) perceptual content and style losses,
we find that combining them with a CLIP directional loss
provides complementary supervision: VGG losses preserve
structure and texture statistics, whereas CLIP encourages
semantic style alignment. Together, they enable stronger
stylistic transfer than either objective alone.
Content Loss. To preserve scene structure and avoid over-
stylization, we employ a content loss computed between the
rendered stylized input views ˆI and the original RGB input
views I in a deep feature space. Let ϕ(·) denote a pretrained
VGG19 network, and let ϕ(l)(·) denote the activation at
layer l. The content loss is defined as
Lcontent =
X
l∈Kcontent
ϕ(l)(ˆI) −ϕ(l)(I)

2
2 ,
(7)
where Kcontent = {relu3_1, relu4_1} denotes the set
of VGG layers used for content supervision. Following clas-
sical neural style transfer, deeper VGG layers are used for
content supervision to preserve high-level structure, while
multiple shallow-to-deep layers capture multi-scale texture
statistics for style (Huang & Belongie, 2017; Zhang et al.,
2022). In line with Styl3R (Wang et al., 2025b), we use both
relu3_1 and relu4_1 for the content loss, which empiri-
cally better preserves scene structure than single-layer super-
vision commonly used in earlier style transfer works (Zhang
et al., 2022; Galerne et al., 2025).
Style Loss. We adopt a style loss that matches channel-
wise feature statistics between the rendered image ˆI and a
reference style image SI. Style features are extracted from
the same ϕ at multiple layers. The style loss is defined as
Lstyle =
X
l∈Kstyle
µ

ϕ(l)(ˆI)

−µ

ϕ(l)(SI)
 2
2
+
σ

ϕ(l)(ˆI)

−σ

ϕ(l)(SI)
 2
2

,
(8)
where
µ(·)
and
σ(·)
denote
the
channel-wise
mean
and
standard
deviation,
and
Kstyle
=
{relu1_1, relu2_1, relu3_1, relu4_1}.
CLIP Loss.
To enhance style alignment between the
stylized output and the desired style, we follow the CLIP-
based style transfer works (Kwon & Ye, 2022; Howil et al.,
2025) and employ a CLIP directional loss. Let z˜I and zI
denote the CLIP embeddings of the stylized rendering ˜I
5

<!-- page 6 -->
AnyStyle
Table 2. Stylization quality, as measured by ArtScore and Art-
FID (abbreviated as Score and FID respectively), and stylization
time comparisons with recent 3D stylization models. Results for
per-scene methods were imported from (Liu et al., 2025). img
and txt denote inference with style image and style text prompt
respectively. We mark with red the best and second best scores.
Method
Train
Truck
M60
Garden
Time↓
Score↑
FID↓
Score↑
FID↓
Score↑
FID↓
Score↑
FID↓
StyleGaussian
0.78
52.79
5.76
44.93
8.63
47.48
9.38
41.14
165 m
G-Style
9.52
23.24
9.67
22.15
9.73
22.36
8.98
25.76
14.5 m
StylizedGS
4.95
40.79
6.45
42.07
7.39
53.24
6.99
65.15
13.3 m
SGSST
1.84
38.24
5.34
32.34
5.26
38.73
4.89
33.54
35.0 m
Styl3R
4.67
27.74
7.07
27.28
8.43
23.98
8.61
29.73
0.16 s
Stylos
9.50
26.40
9.70
28.71
9.37
27.44
9.34
28.06
0.05 s
AnyStyleimg
8.84
22.86
9.59
22.95
9.47
22.81
9.48
22.32
0.07s
AnyStyletxt
9.87
24.41
10.56
24.67
10.20
24.16
10.46
24.20
0.07s
AnyStyleHead,img
8.45
22.89
9.97
25.29
9.03
22.81
9.72
23.85
0.05s
AnyStyleHead,txt
8.69
24.18
10.10
27.89
9.54
24.34
10.07
25.56
0.05s
Table 3.
Ablation study.
Stylization quality is measured by
ArtScore and ArtFID (abbreviated as Score and FID respectively).
We present comparisons for different versions of the Head model.
Method
Train
Truck
M60
Garden
Score↑
FID↓
Score↑
FID↓
Score↑
FID↓
Score↑
FID↓
AnyStyleHead,img
base config
8.45
22.89
9.97
25.29
9.03
22.81
9.72
23.85
w/ all geom features
8.77
23.08
9.61
25.07
8.77
23.30
9.57
23.65
w/o CLIP losses
5.80
23.26
8.75
26.35
7.30
25.00
8.60
23.85
w/o Style loss
4.67
27.65
6.56
27.20
5.65
24.95
5.93
25.76
AnyStyleHead,txt
base config
8.69
24.18
10.10
27.89
9.54
24.34
10.07
25.56
w/o text in training
9.89
29.40
10.53
32.59
9.90
30.04
10.59
31.00
and the original rendering I, respectively. Let zPhoto denote
the embedding of a universal neutral prompt “Photo”. We
define the CLIP directional loss between an original image
I, a stylized image ˆI, and a style input S as
LCLIP(I,˜I, S) = 1 −cos

z˜I −zI, zs −zPhoto

,
(9)
In addition to the global loss, we apply the same CLIP loss
to local image patches sampled from the stylized rendering.
The patch-level CLIP loss enforces local style consistency
beyond global semantic alignment, which can overlook fine-
grained textures. As observed in (Kwon & Ye, 2022), ap-
plying directional CLIP loss to randomly sampled patches
improves the transfer of spatially invariant texture cues.The
patch-level CLIP loss is defined as
Lp
CLIP =
1
|Npatch|
X
i∈Npatch
LCLIP
 I, pi(˜I), S

,
(10)
where pi(ˆI) denotes the i-th patch, and Npatch denotes num-
ber of patches. Random perspective augmentations are ap-
plied to each patch prior to CLIP embedding extraction,
encouraging fine-grained and locally consistent stylization.
We define the final objective as:
L = λcLcontent + λsLstyle + λclipLCLIP + λp
clipLp
CLIP. (11)
Figure 4. Stylization using text prompts. We compare AnyStyle
with ClipGaussian (Howil et al., 2025), which requires per-scene
optimization (>20min). Despite using identical text prompts, Clip-
Gaussian introduces semantic artifacts from the style input.
4. Experiments
We provide additional results including qualitative results
on scenes from other dataset, training details, code, ex-
tended analysis, user study details and limitations in the
Supplementary Material. We encourage the reader to see
the supplementary video, images and analysis.
Datasets. During training, for content supervision we used
DL3DV-480P dataset (Ling et al., 2023). It provides wide
range of both indoor at outdoor scenes. For style supervision
we use the WikiArt dataset (WikiArt), which is extensively
utilized in style transfer tasks. For stylization with text
input, we generated descriptions for the style images with
Mini-CPM-V4.5 model (Yao et al., 2024).
For evaluation, following (Liu et al., 2025), we used 3 scenes
(Train, Truck, M60) from TnT dataset (Knapitsch et al.,
2017), and Garden from Mip-NeRF360 dataset (Barron
et al., 2022) as content data and 50 Wikiart style images
provided by (Liu et al., 2025), excluded from the training.
Baselines. We closely follow (Liu et al., 2025) evaluation
procedure, thus we can directly compare our method to
multiple other approaches, both feed-forward: Stylos (Liu
et al., 2025), Styl3R (Wang et al., 2025b) and based on
per-scene optimization: StyleGaussian (Liu et al., 2024a),
G-Style (Kovács et al., 2024b), StylizedGS (Zhang et al.,
2024c), SGSST (Galerne et al., 2025), ClipGaussian (Howil
et al., 2025). AnyStyle is tested in a zero-shot manner
without prior knowledge of either the scenes or styles.
Stylization – quantitative evaluation. We evaluate styl-
ization quality using ArtScore (Chen et al., 2024a), which
measures how closely an image resembles authentic artwork,
and ArtFID (Wright & Ommer, 2022), which measures
how well an image matches a particular style. We conduct
evaluations under both image-based and text-based style
6

<!-- page 7 -->
AnyStyle
Figure 5. Stylization with text prompts vs. images. We compare
renderings conditioned either on a reference test style image or
on a textual description generated by Mini-CPM-V4.5 for that
image. Our method achieves coherent and plausible stylization
across both modalities. Please note that due to the inherently
lower amount of information encoded in text prompts and more
ambiguous nature of natural language, text-based conditioning
cannot reproduce the rendered appearance exactly the same as
image-based conditioning.
conditioning. Quantitative comparisons with prior methods
are reported in Table 2.
AnyStyle achieves the best ArtFID across all scenes for
both image and text input compared to other feed-forward
methods. In terms of ArtScore, AnyStyletxt also achieves
the best results across all scenes, while AnyStyleimg beats
most existing methods. We observe that text-conditioned
models obtain higher ArtScore but slightly worse ArtFID
than image-conditioned counterparts. We attribute this to
the nature of the stylization guided by text prompts which
inherently provides less visual cues.
Stylization – qualitative evaluation.
Qualitative com-
parisons between our method and current state-of-the-art
models are illustrated in Figure 3 and Figure 4, while Fig-
ure 5 provides a visual analysis of image- versus text-based
style conditioning.
Style interpolation. Our method operates in a unified CLIP-
based embedding space, enabling flexible control over the
stylization process. Qualitative results are shown in Figure 7.
Beyond standard interpolation between two distinct styles,
our model also supports interpolation between closely re-
Figure 6. User study. According to responders,images generated
by AnyStyle better follow reference style. Content preservation is
comparable for both methods.
lated style descriptions that differ only in specific attributes.
This allows fine-grained adjustment of stylistic properties
without requiring the user to search for a new reference
image, which is typically necessary in existing image-based
style transfer methods.
Multi-view consistency.
We provide a qualitative eval-
uation of multi-view consistency in Figure 8, where our
method preserves a coherent appearance of scene details
across viewpoints. For additional video results, quantitative
evaluation and detailed discussion, we refer to Section E in
the Supplementary Material.
To further assess perceptual quality, we conducted an on-
line user study comparing our method with Styl3R (StylOS
code was unavailable at submission time). A total of 40
participants evaluated 20 cases, each with three questions
related to stylization quality and content preservation. They
rated the results on a 5-point Likert scale (Likert, 1932).
This resulted in 800 responses per question. The results
are shown in Figure 6. Statistical significance was eval-
uated using one-sample t-tests and Wilcoxon signed-rank
tests. The observed preference for our method was statisti-
cally significant; detailed test statistics are reported in the
Supplementary Material.
Ablation study. First, we perform an ablation study on two
architectural design choices: (i) the full model where style
injection is applied in both the Aggregator and the Head,
and both ˜Hgs and ˜A are fine-tuned; and (ii) a lightweight
setting, denoted as Head, where style injection is applied
only to tokens entering ˜Hgs and only ˜Hgs is fine-tuned.
As shown in Table 2, both variants achieve strong results
compared to existing, particularly feed-forward, methods.
The Head version typically yields slightly higher FID than
full model, indicating weaker alignment with the target
style images. This ablation demonstrates that our injection
mechanism remains effective even when applied solely to
tokens forwarded directly to prediction head.
We further ablate the Head variant. The results are presented
in Figure 9 and Table 3. CLIP-based losses alone provide
a weak supervisory signal, while style loss alone degrades
7

<!-- page 8 -->
AnyStyle
Figure 7. Style interpolation. Thanks to the unified CLIP em-
bedding space, our method enables smooth interpolation between
two style images, between an image and a text prompt, as well as
between two text prompts. Compared to image-only style transfer
methods, the latter capability provides greater control over the
stylization process and supports an iterative workflow for refining
specific stylistic attributes (highlighted in red).
Figure 8. Multi-view consistency of the stylization. By di-
rectly updating 3D Gaussian representation, our method ensures
multi-view consistency of stylization while preserving fine-grained
details across viewpoints. Floor dot (marked in green) and table
discoloration (marked in red) remain consistent across views.
Figure 9. Ablation study. The first three columns show stylization
conditioned on a reference style image. Full model produces
richer and more vivid colors, while the Head variant yields slightly
flatter appearances. Removing the style loss prevents accurate
style learning, whereas removing the CLIP directional loss reduces
color expressiveness. Finetuning all geometric features leads to
visual inconsistencies (highlighted in green), such as unintended
transparent artifacts mimicking cubist features. The last column
shows text-conditioned stylization using a prompt corresponding to
the reference style image; models trained with textual supervision
follow the target style more closely than those trained without it.
color alignment; removing either leads to consistent per-
formance drops across all metrics. Updating all Gaussian
parameters, although not harmful in terms of quantitative
metrics, introduces visible geometric artifacts and degrades
multi-view consistency. Finally, excluding textual condi-
tioning during training significantly degrades text-driven
stylization, leading to results that poorly follow the target
style. This is also reflected in a substantially higher ArtFID
for the w/o text variant.
5. Conclusions
We introduced AnyStyle, a feed-forward framework for
multimodal, zero-shot stylization of 3D Gaussian Splatting
scenes. Unlike prior approaches that rely solely on image-
based conditioning or require retraining pipelines from
scratch, AnyStyle enables stylization via an architecture-
agnostic style injection mechanism attachable to pretrained
3D reconstruction backbones with minimal modification.
Operating in a shared CLIP embedding space, the method
supports text- and image-driven control, as well as con-
tinuous interpolation between styles, enabling flexible and
semantically meaningful manipulation of scene appearance.
By decoupling geometry from appearance, AnyStyle pre-
serves structural consistency while allowing expressive styl-
ization. Our experiments demonstrate strong quantitative
and perceptual results, achieving superior stylization quality
among feed-forward methods.
8

<!-- page 9 -->
AnyStyle
6. Acknowledgements
J. Kaleta was supported by National Science Centre, Poland
grant no 2022/47/O/ST6/01407. P. Spurek was supported
by the project Effective Rendering of 3D Objects Using
Gaussian Splatting in an Augmented Reality Environment
(FENG.02.02-IP.05-0114/23), carried out under the First
Team programme of the Foundation for Polish Science
and co-financed by the European Union through the Eu-
ropean Funds for Smart Economy 2021–2027 (FENG). We
gratefully acknowledge Polish high-performance comput-
ing infrastructure PLGrid (HPC Center: ACK Cyfronet
AGH) for providing computer facilities and support within
computational grants no.
PLG/2025/018551 and no.
PLG/2024/017221.
References
Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P.,
and Hedman, P.
Mip-NeRF 360: Unbounded Anti-
Aliased Neural Radiance Fields. CVPR, 2022.
Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J.,
Bojanowski, P., and Joulin, A. Emerging properties in
self-supervised vision transformers. In Proceedings of the
IEEE/CVF international conference on computer vision,
pp. 9650–9660, 2021.
Charatan, D., Li, S. L., Tagliasacchi, A., and Sitzmann, V.
pixelSplat: 3D Gaussian Splats from Image Pairs for Scal-
able Generalizable 3D Reconstruction. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pp. 19457–19467, 2024.
Chen, J., An, J., Lyu, H., Kanan, C., and Luo, J. Learning
to Evaluate the Artness of AI-Generated Images. IEEE
Transactions on Multimedia, 26:10731–10740, 2024a.
doi: 10.1109/TMM.2024.3410672.
Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M.,
Geiger, A., Cham, T.-J., and Cai, J. MVSplat: Efficient
3D Gaussian Splatting from Sparse Multi-view Images.
In European Conference on Computer Vision, pp. 370–
386. Springer, 2024b.
Chung, J., Hyun, S., and Heo, J.-P. Style injection in dif-
fusion: A training-free approach for adapting large-scale
diffusion models for style transfer. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 8795–8805, 2024a.
Chung, J., Hyun, S., and Heo, J.-P. Style injection in dif-
fusion: A training-free approach for adapting large-scale
diffusion models for style transfer. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 8795–8805, June 2024b.
Deng, Y., Tang, F., Dong, W., Ma, C., Pan, X., Wang, L., and
Xu, C. StyTr2: Image Style Transfer With Transformers.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 11326–11336, 2022.
Gal, R., Patashnik, O., Maron, H., Bermano, A. H., Chechik,
G., and Cohen-Or, D. Stylegan-nada: Clip-guided domain
adaptation of image generators. ACM Transactions on
Graphics (TOG), 41(4):1–13, 2022.
Galerne, B., Wang, J., Raad, L., and Morel, J.-M. SGSST:
Scaling Gaussian Splatting Style Transfer. In Proceed-
ings of the Computer Vision and Pattern Recognition
Conference, pp. 26535–26544, 2025.
Gatys, L. A., Ecker, A. S., and Bethge, M. Image Style
Transfer Using Convolutional Neural Networks. In Pro-
ceedings of the IEEE conference on computer vision and
pattern recognition, pp. 2414–2423, 2016.
Guan, Q., Wang, Y., Ping, B., Li, D., Du, J., Qin, Y., Lu,
H., Wan, X., and Xiang, J. Deep convolutional neural
network vgg-16 model for differential diagnosing of pap-
illary thyroid carcinomas in cytological images: a pilot
study. Journal of Cancer, 10(20):4876, 2019.
Han, Z., Mao, C., Jiang, Z., Pan, Y., and Zhang, J. Style-
booth: Image style editing with multimodal instruction.
In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision (ICCV) Workshops, pp. 1947–
1957, 2025.
Howil, K., Borycki, P., Dziarmaga, T., Mazur, M., Spurek,
P., et al. CLIPGaussian: Universal and Multimodal Style
Transfer Based on Gaussian Splatting. arXiv preprint
arXiv:2505.22854, 2025.
Huang, X. and Belongie, S. Arbitrary Style Transfer in Real-
Time With Adaptive Instance Normalization. In Proceed-
ings of the IEEE international conference on computer
vision, pp. 1501–1510, 2017.
Huang, Y.-H., He, Y., Yuan, Y.-J., Lai, Y.-K., and Gao,
L. StylizedNeRF: Consistent 3D Scene Stylization as
Stylized NeRF via 2D-3D Mutual Learning. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 18342–18352, 2022.
Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X.,
Yu, M., Pang, J., Zhao, F., et al. AnySplat: Feed-forward
3D Gaussian Splatting from Unconstrained Views. ACM
Transactions on Graphics (TOG), 44(6):1–16, 2025.
Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G.
3D Gaussian Splatting for Real-Time Radiance Field Ren-
dering. ACM Trans. Graph., 42(4):139–1, 2023.
9

<!-- page 10 -->
AnyStyle
Knapitsch, A., Park, J., Zhou, Q.-Y., and Koltun, V. Tanks
and temples: Benchmarking large-scale scene reconstruc-
tion. ACM Transactions on Graphics (ToG), 36(4):1–13,
2017.
Kovács, Á. S., Hermosilla, P., and Raidou, R. G. G-style:
Stylized gaussian splatting. In Computer Graphics Forum,
volume 43, pp. e15259. Wiley Online Library, 2024a.
Kovács, Á. S., Hermosilla, P., and Raidou, R. G. G-Style:
Stylized Gaussian Splatting.
In Computer Graphics
Forum, volume 43, pp. e15259. Wiley Online Library,
2024b.
Kwon, G. and Ye, J. C. CLIPstyler: Image Style Transfer
with a Single Text Condition.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 18062–18071, 2022.
Leroy, V., Cabon, Y., and Revaud, J. Grounding Image
Matching in 3D with MASt3R. In European Conference
on Computer Vision, pp. 71–91. Springer, 2024.
Li, X., Liu, S., Kautz, J., and Yang, M.-H. Learning Linear
Transformations for Fast Image and Video Style Transfer.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 3809–3817, 2019.
Li, Y., Fang, C., Yang, J., Wang, Z., Lu, X., and Yang,
M.-H. Universal Style Transfer via Feature Transforms.
Advances in neural information processing systems, 30,
2017.
Likert, R. A technique for the measurement of attitudes.
Archives of Psychology, 140:55, 1932.
Lin, H., Chen, S., Liew, J. H., Chen, D. Y., Li, Z., Shi,
G., Feng, J., and Kang, B. Depth anything 3: recov-
ering the visual space from any views. arXiv preprint
arXiv:2511.10647, 2025.
Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K.,
Yu, L., Guo, Q., Yu, Z., Lu, Y., Li, X., Sun, X., Ashok,
R., Mukherjee, A., Kang, H., Kong, X., Hua, G., Zhang,
T., Benes, B., and Bera, A. DL3DV-10K: A Large-Scale
Scene Dataset for Deep Learning-based 3D Vision, 2023.
URL https://arxiv.org/abs/2312.16256.
Liu, H., Huang, J., Lu, M., Saripalli, S., and Jiang, P. Stylos:
Multi-View 3D Stylization with Single-Forward Gaussian
Splatting, 2025. URL https://arxiv.org/abs/
2509.26455.
Liu, K., Zhan, F., Chen, Y., Zhang, J., Yu, Y., El Saddik,
A., Lu, S., and Xing, E. P. StyleRF: Zero-Shot 3D Style
Transfer of Neural Radiance Fields. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 8338–8348, 2023.
Liu, K., Zhan, F., Xu, M., Theobalt, C., Shao, L., and Lu, S.
StyleGaussian: Instant 3D Style Transfer with Gaussian
Splatting. arXiv preprint arXiv:2403.07807, 2024a.
Liu, K., Zhan, F., Xu, M., Theobalt, C., Shao, L., and Lu,
S. Stylegaussian: Instant 3d style transfer with gaussian
splatting. In SIGGRAPH Asia 2024 Technical Communi-
cations, pp. 1–4. 2024b.
Liu, S., Lin, T., He, D., Li, F., Wang, M., Li, X., Sun, Z., Li,
Q., and Ding, E. AdaAttN: Revisit Attention Mechanism
in Arbitrary Neural Style Transfer. In Proceedings of the
IEEE/CVF international conference on computer vision,
pp. 6649–6658, 2021.
Lu, J., Huang, T., Li, P., Dou, Z., Lin, C., Cui, Z., Dong, Z.,
Yeung, S.-K., Wang, W., and Liu, Y. Align3r: Aligned
monocular depth estimation for dynamic videos. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pp. 22820–22830, 2025.
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R., and Ng, R. NeRF: Representing Scenes
as Neural Radiance Fields for View Synthesis. Communi-
cations of the ACM, 65(1):99–106, 2021.
Nguyen-Phuoc, T., Liu, F., and Xiao, L. SNeRF: stylized
neural implicit representations for 3D scenes. ACM Trans.
Graph., 41(4), July 2022. ISSN 0730-0301. doi: 10.
1145/3528223.3530107. URL https://doi.org/
10.1145/3528223.3530107.
Niklaus, S. and Liu, F. Softmax Splatting for Video Frame
Interpolation. In IEEE Conference on Computer Vision
and Pattern Recognition, 2020.
Park, D. Y. and Lee, K. H.
Arbitrary Style Transfer
With Style-Attentional Networks. In proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 5880–5888, 2019.
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.,
et al. Learning Transferable Visual Models From Natural
Language Supervision. In International conference on
machine learning, pp. 8748–8763. PmLR, 2021.
Smart, B., Zheng, C., Laina, I., and Prisacariu, V. A.
Splatt3R: Zero-shot Gaussian Splatting from Uncali-
brated Image Pairs. arXiv preprint arXiv:2408.13912,
2024.
Suresh, A. P., Jain, S., Noinongyao, P., Ganguly, A.,
Watchareeruetai, U., and Samacoits, A. Fastclipstyler:
Optimisation-free text-based image style transfer using
style representations. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision,
pp. 7316–7325, 2024.
10

<!-- page 11 -->
AnyStyle
Szymanowicz, S., Rupprecht, C., and Vedaldi, A. Splatter
Image: Ultra-Fast Single-View 3D Reconstruction. In
Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 10208–10217, 2024.
Tang, H., Liu, S., Lin, T., Huang, S., Li, F., He, D., and
Wang, X. Master: Meta Style Transformer for Control-
lable Zero-Shot and Few-Shot Artistic Style Transfer. In
Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 18329–18338, 2023.
Teed, Z. and Deng, J. Raft: Recurrent all-pairs field trans-
forms for optical flow. In Vedaldi, A., Bischof, H., Brox,
T., and Frahm, J.-M. (eds.), Computer Vision – ECCV
2020, pp. 402–419, Cham, 2020. Springer International
Publishing. ISBN 978-3-030-58536-5.
Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht,
C., and Novotny, D. VGGT: Visual Geometry Grounded
Transformer. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 5294–5306, 2025a.
Wang, P., Liu, X., and Liu, P. Styl3R: Instant 3D Stylized
Reconstruction for Arbitrary Scenes and Styles. arXiv
preprint arXiv:2505.21060, 2025b.
Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud,
J. DUSt3R: Geometric 3D Vision Made Easy. In CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 20697–20709, 2023.
WikiArt.
WikiArt: Visual Art Encyclopedia.
https:
//www.wikiart.org/. Accessed: 2025-10.
Wright, M. and Ommer, B. ArtFID: Quantitative Evaluation
of Neural Style Transfer. GCPR, 2022.
Wu, X., Hu, Z., Sheng, L., and Xu, D. StyleFormer: Real-
Time Arbitrary Style Transfer via Parametric Style Com-
position. In Proceedings of the IEEE/CVF international
conference on computer vision, pp. 14618–14627, 2021.
Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H.,
Cai, T., Li, H., Zhao, W., He, Z., et al. MiniCPM-V: A
GPT-4V Level MLLM on Your Phone. arXiv preprint
arXiv:2408.01800, 2024.
Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.-H.,
and Peng, S. No Pose, No Problem: Surprisingly Simple
3D Gaussian Splats from Sparse Unposed Images. arXiv
preprint arXiv:2410.24207, 2024.
Yu, X.-Y., Yu, J.-X., Zhou, L.-B., Wei, Y., and Ou, L.-L. In-
stantStyleGaussian: Efficient Art Style Transfer with 3D
Gaussian Splatting I. arXiv preprint arXiv:2408.04249,
2024.
Zhang, B., Zhang, P., Dong, X., Zang, Y., and Wang, J.
Long-CLIP: Unlocking the Long-Text Capability of CLIP.
arXiv preprint arXiv:2403.15378, 2024a.
Zhang, C., Xu, X., Wang, L., Dai, Z., and Yang, J. S2WAT:
Image Style Transfer via Hierarchical Vision Transformer
Using Strips Window Attention. In Proceedings of the
AAAI conference on artificial intelligence, volume 38, pp.
7024–7032, 2024b.
Zhang, D., Yuan, Y.-J., Chen, Z., Zhang, F.-L., He, Z., Shan,
S., and Gao, L. Stylizedgs: Controllable stylization for
3d gaussian splatting. arXiv preprint arXiv:2404.05220,
2024c.
Zhang, K., Kolkin, N., Bi, S., Luan, F., Xu, Z., Shechtman,
E., and Snavely, N. ARF: Artistic Radiance Fields. In
European Conference on Computer Vision, pp. 717–733.
Springer, 2022.
Zhang, L., Rao, A., and Agrawala, M. Adding conditional
control to text-to-image diffusion models. In Proceedings
of the IEEE/CVF International Conference on Computer
Vision (ICCV), pp. 3836–3847, October 2023.
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang,
O. The Unreasonable Effectiveness of Deep Features as
a Perceptual Metric. In CVPR, 2018.
11

<!-- page 12 -->
AnyStyle
A. Supplementary overview
This Supplementary Material is organized as follows:
• Extended qualitative results (Section B): additional stylized renders on RE10K, including both image-guided and
text-prompt stylization.
• User study (Section C): study design, questions, and statistical analysis of participant preferences.
• Training details (Sec. D): code, datasets, optimization setup, loss weights, and architecture choices.
• Multi-view consistency: Analysis and discussion (Sec. E): consistency evaluation protocol and limitations of the
warping-based metric.
• Extended quantitative results (Sec. F): additional metrics for stylized outputs.
• Limitations and failure cases (Sec. G): representative AnyStyle failures for image- and text-conditioned stylization.
B. Extended qualitative results
We present additional stylized renders on publicly available RE10K dataset https://google.github.io/
realestate10k/. Text vs image stylization are presented in Figure 10, Figure 11, Figure 12, Figure 13, Figure 14,
while examples of stylization with natural text are presented in Figure 15, Figure 16.
Figure 10. Additional qualitative results of our method on RE10K dataset. We present stylization with image and text prompt.
12

<!-- page 13 -->
AnyStyle
Figure 11. Additional qualitative results of our method on RE10K dataset. We present stylization with image and text prompt.
13

<!-- page 14 -->
AnyStyle
Figure 12. Additional qualitative results of our method on RE10K dataset. We present stylization with image and corresponding text
prompt.
14

<!-- page 15 -->
AnyStyle
Figure 13. Additional qualitative results of our method on RE10K dataset. We present stylization with image and corresponding text
prompt.
15

<!-- page 16 -->
AnyStyle
Figure 14. Additional qualitative results of our method on RE10K dataset. We present stylization with image and corresponding text
prompt.
16

<!-- page 17 -->
AnyStyle
Figure 15. Additional qualitative results of our method on RE10K dataset. We present stylization with natural-language text prompt.
Figure 16. Additional qualitative results of our method on RE10K dataset. We present stylization with natural-language text prompt.
17

<!-- page 18 -->
AnyStyle
C. User study
To evaluate the effectiveness of our approach, we conducted a user study involving 40 participants. Each participant was
presented with 20 comparative image sets, resulting in 60 individual responses per form and a total of 2400 evaluations
across the study. Each set displayed a style reference alongside three reference images and three corresponding generated
outputs for two anonymous methods, labeled Method A and Method B. To eliminate positional bias, the assignment of our
method was counterbalanced such that it appeared as Method A in half of the trials and Method B in the remaining half.
For each comparative image set, participants were asked to answer the following three questions:
• On a scale from 1 (A is much better) to 5 (B is much better), rate which method better applied style to images, where 3
indicates methods are comparable.
• On a scale from 1 (very difficult) to 5 (very easy), how hard it is to recognize original content on images generated by
Method A?
• On a scale from 1 (very difficult) to 5 (very easy), how hard it is to recognize original content on images generated by
Method B?
The results for first question were analyzed using a one-sample T-test and a one-sample Wilcoxon Signed-Rank test with a
significance level of α = 0.05. For the T-test, the null hypothesis stated that the distribution mean is equal to the population
mean (where the neutral value is 3). For the Wilcoxon Signed-Rank test, the null hypothesis was that the distribution of
S −3 is symmetric around µ = 0, where S is a set of responses. In both cases, the alternative hypothesis was that the
distribution mean is smaller, corresponding to the statement that our method performed better than the baseline. We obtained
p-values of 3.47 × 10−40 and 1.67 × 10−35 for the one-sample t-test and the Wilcoxon signed-rank test, respectively. Both
values are far below the significance threshold of α = 0.05, indicating that the observed preference for our method is
statistically significant.
Regarding content preservation, results for both methods were comparable so we have abandoned any statistical test.
D. Training details
For full implementation and reproduction details, please refer to the supplementary code at https://anonymous.
4open.science/r/AnyStyle-13E9/README.md.
Training sets. We train our model on the full DL3DV-480P dataset and evaluate on scenes from other datasets. For style
supervision we use Wikiart dataset and we exclude 50 test style images from StylOS (Liu et al., 2025) as well as the entire
Styl3R test set (Wang et al., 2025b) to avoid data leakage.
Model training. Model is trained for 90k iterations. We optimize the copied Gaussian Head and style injectors with a
learning rate of 1 × 10−4. The Aggregator learning rate is set to 0.3× the original backbone learning rate. We use a cosine
annealing learning rate schedule.
Loss weights and parameters. The training loss consists of a style loss (weight 1.0), a content loss (weight 0.05), a CLIP
directional loss (weight 2.0) and patch CLIP loss with 16 crops per image, patch loss weight 4.0 and crop size 64. Depth
consistency loss weight is set to 0.1.
Architecture choices. Tokens from the intermediate layers [4, 11, 17, 23] are passed to the Gaussian Head, and style
injection for Gaussian Head is performed for tokens from these layers. For the Aggregator, features are injected into the
tokens before they enter the following layers: [0, 4, 11, 17, 23].
18

<!-- page 19 -->
AnyStyle
E. Multi-view consistency: Analysis and Discussion.
Following most style transfer works (Liu et al., 2025; Zhang et al., 2024c), we measure multi-view consistency. We strictly
follow the procedure used in Stylos: we take the first 16 frames and compute a warp between two frames using optical
flow (Teed & Deng, 2020) and softmax splatting (Niklaus & Liu, 2020). We then compute masked RMSE and LPIPS (Zhang
et al., 2018) on the warped results to evaluate stylization consistency. For short-range consistency, we use frame pairs
(t, t−1), and for long-range consistency, we use frame pairs (t, t−7). Results are presented in Table 4.
Discussion on the metric. We observed that, on the test scenes, the metric performs slighly worse than expected for stylized
images and also performs poorly for the GT RGB images. For the GT images, one natural reason we identified is that
view-dependent effects can lead to large errors (e.g., specular surface lit by the sun changes appearance with viewpoint).
However, we also found a second reason: warping errors, examples of which are shown in Figure 17 and Figure 18. We
hypothesize that some stylizations can make warping easier (for example, fewer details or more distinct shapes may benefit
optical flow, Figure 17). At the same time, we observed that the more uniform colors introduced naturally by stylization can
lead to lower RMSE values even when similar warping errors occur in both GT and stylized sequences. In such cases, lower
RMSE does not correspond to actual consistency. For these reasons, the mean consistency metrics for GT images are
sometimes higher (worse) than for our stylized outputs (see Table 4). Overall, we conclude that lower metric values
may reflect (i) better consistency, (ii) improved warping alignment, or (iii) simply reduced color variation. We want to
emphasize that these conclusions are valid for both our and baseline methods. We leave the interpretation to the reader.
Figure 17. Visualisation of warping and consistency metric RMSE on Train scene. In both sub-figures, stylized images of a train (bottom
rows) achieve lower RMSE then GT RGB images (top rows).
19

<!-- page 20 -->
AnyStyle
Figure 18. Visualisation of warping and consistency metric RMSE on Garden scene. In both sub-figures, stylized images of a garden
(bottom rows) achieve lower RMSE then GT RGB images (top rows).
20

<!-- page 21 -->
AnyStyle
Table 4. Short-range and long-range consistency evaluation. Lower is better for all metrics.
Method
Train
Truck
M60
Garden
LPIPS↓
RMSE↓
LPIPS↓
RMSE↓
LPIPS↓
RMSE↓
LPIPS↓
RMSE↓
Short-range consistency
StyleGaussian
0.033
0.038
0.031
0.034
0.038
0.037
0.069
0.061
G-Style
0.042
0.052
0.032
0.035
0.038
0.034
0.066
0.070
StylizedGS
0.016
0.023
0.013
0.019
0.017
0.020
0.056
0.041
SGSST
0.038
0.047
0.039
0.047
0.044
0.049
0.084
0.090
Styl3R
0.056
0.038
0.049
0.033
0.064
0.042
0.085
0.061
Stylos
0.030
0.026
0.028
0.021
0.035
0.024
0.047
0.044
AnyStyleAgg,img
0.032
0.038
0.033
0.034
0.042
0.041
0.063
0.063
AnyStyleAgg,txt
0.029
0.032
0.028
0.028
0.036
0.033
0.055
0.051
AnyStyleHead,img
0.032
0.035
0.031
0.031
0.041
0.036
0.066
0.060
AnyStyleHead,txt
0.029
0.030
0.028
0.026
0.038
0.030
0.059
0.049
GT images
0.070
0.069
0.055
0.055
0.059
0.053
0.121
0.092
Long-range consistency
StyleGaussian
0.067
0.072
0.086
0.077
0.091
0.091
0.177
0.141
G-Style
0.098
0.120
0.095
0.093
0.104
0.095
0.180
0.175
StylizedGS
0.040
0.065
0.049
0.064
0.065
0.086
0.100
0.202
SGSST
0.087
0.108
0.119
0.120
0.130
0.128
0.221
0.222
Styl3R
0.109
0.090
0.136
0.112
0.160
0.138
0.185
0.171
Stylos
0.051
0.056
0.074
0.069
0.083
0.082
0.139
0.134
AnyStyleAgg,img
0.082
0.088
0.092
0.094
0.115
0.117
0.198
0.170
AnyStyleAgg,txt
0.074
0.075
0.080
0.078
0.098
0.098
0.182
0.145
AnyStyleHead,img
0.070
0.073
0.091
0.089
0.115
0.108
0.189
0.147
AnyStyleHead,txt
0.062
0.061
0.079
0.074
0.103
0.091
0.169
0.122
GT images
0.130
0.142
0.129
0.113
0.147
0.123
0.284
0.206
21

<!-- page 22 -->
AnyStyle
F. Extended quantitative results.
In Tables 5 and 6, following (Chung et al., 2024b; Liu et al., 2025) we provide extended quantitative evaluation of stylized
data. Notably, we consistently achieve strong (low) FID, LPIPS and LPIPS-Gray results.
Table 5. Following Stylos (Liu et al., 2025) and StyleID (Chung et al., 2024b), we additionally report stylization quality metrics: FID,
LPIPS, LPIPS-gray, CFSD, and color matching loss (HistoGAN loss), as supplementary to results in main paper.
Method
Train
Truck
FID↓
LPIPS↓
Gray↓
CFSD↓
CM Loss↓
FID↓
LPIPS↓
Gray↓
CFSD↓
CM Loss↓
StyleGaussian
34.59
0.483
0.377
0.190
0.418
28.53
0.522
0.367
0.131
0.382
G-Style
14.80
0.471
0.364
0.165
0.185
13.65
0.512
0.353
0.087
0.176
StylizedGS
22.90
0.707
0.648
0.216
0.396
22.65
0.779
0.713
0.084
0.417
SGSST
24.15
0.520
0.409
0.220
0.257
19.50
0.577
0.445
0.177
0.196
Styl3R
19.77
0.670
0.598
0.244
0.364
19.28
0.682
0.575
0.102
0.350
Stylos
15.30
0.620
0.529
0.223
0.241
16.67
0.625
0.471
0.084
0.237
AnyStyleAgg,img
13.81
0.543
0.457
0.256
0.359
13.61
0.570
0.433
0.112
0.377
AnyStyleAgg,txt
15.08
0.518
0.421
0.236
0.438
14.97
0.544
0.399
0.090
0.447
AnyStyleHead,img
13.73
0.551
0.452
0.249
0.364
14.94
0.586
0.437
0.103
0.390
AnyStyleHead,txt
14.80
0.529
0.421
0.238
0.442
16.74
0.572
0.418
0.091
0.465
Table 6. Following Stylos (Liu et al., 2025) and StyleID (Chung et al., 2024b), we additionally report stylization quality metrics: FID,
LPIPS, LPIPS-gray, CFSD, and color matching loss (HistoGAN loss), as supplementary to results in main paper.
Method
M60
Garden
FID↓
LPIPS↓
Gray↓
CFSD↓
CM Loss↓
FID↓
LPIPS↓
Gray↓
CFSD↓
CM Loss↓
StyleGaussian
30.54
0.506
0.413
0.138
0.467
25.23
0.569
0.454
0.189
0.480
G-Style
13.93
0.498
0.395
0.092
0.208
16.17
0.500
0.364
0.103
0.179
StylizedGS
28.33
0.815
0.728
0.102
0.443
33.73
0.876
0.827
0.074
0.570
SGSST
23.95
0.552
0.458
0.204
0.264
20.93
0.529
0.415
0.233
0.228
Styl3R
17.14
0.646
0.573
0.124
0.314
22.38
0.637
0.556
0.097
0.335
Stylos
16.61
0.558
0.457
0.098
0.252
16.26
0.625
0.509
0.080
0.242
AnyStyleAgg,img
13.88
0.533
0.456
0.131
0.373
13.43
0.546
0.467
0.114
0.375
AnyStyleAgg,txt
15.07
0.503
0.429
0.111
0.450
14.76
0.536
0.447
0.086
0.446
AnyStyleHead,img
13.79
0.543
0.452
0.120
0.369
14.44
0.545
0.452
0.100
0.380
AnyStyleHead,txt
15.00
0.521
0.427
0.108
0.452
15.60
0.539
0.440
0.085
0.465
22

<!-- page 23 -->
AnyStyle
G. Limitations & failure cases
While our method performs well across a wide range of styles for both image and text stylization, its performance may
degrade for inputs that are far outside the WikiArt distribution, where our train artistic style examples originate. In such
out-of-distribution cases, the stylization can present some visual properties of target style, but can be less visually appealing.
Figure 19, Figure 20 show representative failure cases, including an extreme example with neon-like line structures and a
non-artistic text prompt. A potential direction to reduce these failures - especially in strongly out-of-distribution cases is
test-time embedding optimization.
Figure 19. Example failure case when providing image as conditioning signal. Model fails to transfer the essence of the style. Instead it
applies only color pallette from style image.
Figure 20. Example failure case when providing text as conditioning signal. While overall aesthetics my be pleasing, model fails to create
realistic nighttime view.
23
