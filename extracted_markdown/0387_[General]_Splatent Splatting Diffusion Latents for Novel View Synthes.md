<!-- page 1 -->
Splatent: Splatting Diffusion Latents for Novel View Synthesis
Or Hirschorn†1,2, Omer Sela†1,2, Inbar Huberman-Spiegelglas1, Netalee Efrat1,
Eli Alshan1, Ianir Ideses1, Frederic Devernay1, Yochai Zvik1, Lior Fritz1
1Amazon Prime Video
2Tel-Aviv University
https://orhir.github.io/Splatent
LRF
Splatent (Ours)
Ground Truth
MVSplat360
MVSplat360 + Splatent (Ours)
Ground Truth
Latent Radiance 
Field Optimization
Feed-Forward Latent 
Radiance Field
Figure 1. Novel view synthesis from a latent-space radiance field. Splatent is a principled framework to enhance rendered novel views
from a radiance field in the latent space of diffusion VAEs. We demonstrate improvements in image quality in the setting of test-time latent
radiance field optimization, compared to LRF [61]. In addition, we show how Splatent can be connected within a latent-based feed-forward
model like MVSplat360 [9] to enhance the results and reduce hallucinations.
Abstract
Radiance field representations have recently been ex-
plored in the latent space of VAEs that are commonly used
by diffusion models. This direction offers efficient render-
ing and seamless integration with diffusion-based pipelines.
However, these methods face a fundamental limitation: The
VAE latent space lacks multi-view consistency, leading to
blurred textures and missing details during 3D reconstruc-
tion. Existing approaches attempt to address this by fine-
tuning the VAE, at the cost of reconstruction quality, or
by relying on pre-trained diffusion models to recover fine-
grained details, at the risk of some hallucinations.
We
present Splatent, a diffusion-based enhancement frame-
work designed to operate on top of 3D Gaussian Splatting
(3DGS) in the latent space of VAEs. Our key insight departs
from the conventional 3D-centric view: rather than recon-
†Work was done during an internship at Amazon
structing fine-grained details in 3D space, we recover them
in 2D from input views through multi-view attention mech-
anisms. This approach preserves the reconstruction qual-
ity of pretrained VAEs while achieving faithful detail recov-
ery. Evaluated across multiple benchmarks, Splatent estab-
lishes a new state-of-the-art for VAE latent radiance field
reconstruction. We further demonstrate that integrating our
method with existing feed-forward frameworks, consistently
improves detail preservation, opening new possibilities for
high-quality sparse-view 3D reconstruction.
1. Introduction
Radiance field representations such as NeRFs [32] and 3D
Gaussian splatting (3DGS) [21] have established new stan-
dards for photorealistic novel view synthesis. Concurrently,
diffusion models have advanced rapidly, with state-of-the-
art models typically operating in a compressed latent space
1
arXiv:2512.09923v1  [cs.CV]  10 Dec 2025

<!-- page 2 -->
obtained through a variational autoencoder (VAE) [40]. Re-
cently, some works explored the idea of radiance field rep-
resentations directly in this space [1, 9, 30, 35, 59, 61].
Radiance field representations in diffusion latent spaces
offer several advantages. Operating at compressed spatial
resolutions substantially accelerates both the optimization
time and rendering speeds of radiance fields [1, 30, 35, 59].
Moreover, directly predicting the 3D Gaussian features in
latent space enables end-to-end training of feed-forward 3D
reconstruction models [9, 12, 18, 42], as gradients propa-
gate straight to the 3D model without attenuation through
an encoder. Prior work showed that predicting RGB val-
ues, encoding them, and then feeding them into a generative
model yields inferior results [9]. Operating in latent space
avoids this bottleneck and allows diffusion priors to refine
the 3D representation directly.
However, directly rendering these latents from a la-
tent radiance field faces obstacles. We demonstrate, con-
sistent with concurrent observations [61], that VAE la-
tent spaces used in modern latent diffusion models encode
high-frequency details in fundamentally view-inconsistent
ways [42, 61]. These inconsistencies can severely degrade
3D reconstruction, resulting in blurred textures and miss-
ing fine details (see Fig. 3). Recent attempts to address
this either compromise decoder quality, as in VAE fine-
tuning [61], or result in hallucinated high-frequency details
with stacked video diffusion models [9], failing to faithfully
reconstruct the scene (see Figs. 1 and 5).
We introduce Splatent, a principled solution for high-
fidelity novel view synthesis in latent space radiance fields.
Our approach preserves high-frequency details, while oper-
ating in a frozen VAE setting. The core insight is to keep
the 3D representation in the low-frequency domain and re-
cover high-frequency details in 2D-space from reference in-
put views. More specifically, we first encode each input
view into latent space using a pre-trained VAE and optimize
a 3DGS model on these latents. We show that this process
suffers from poor geometric and photometric reconstruction
quality due to multi-view inconsistencies. To enhance ren-
dered latents from novel views, we leverage latent diffusion
with multi-view attention [4, 7, 14, 17, 45], which recov-
ers fine details by conditioning on nearby views. Impor-
tantly, our VAE remains frozen, preserving the reconstruc-
tion quality and generalization capacity of pretrained au-
toencoders, trained on billions of images [40]. We further
demonstrate that our proposed framework can enhance the
results from MVSplat360 [9], a feed-forward latent 3DGS
model capable of novel view synthesis with as few as 5 in-
put views. Our method consistently improves fine details
while mitigating hallucinations.
Extensive experiments across multiple datasets show
that our method outperforms previous latent-based radiance
field approaches in terms of image quality and consistency
with source views. To summarize, our contributions are as
follows:
• An in-depth analysis of 3D reconstruction in VAE la-
tent space. We demonstrate that the latent spaces of mul-
tiple popular VAEs lack multi-view consistency, limiting
their applicability for latent-based 3D reconstruction.
• A principled framework for latent 3D reconstruction.
We show that high-frequency detail preservation in latent
space highly benefits from 2D-space context, introducing
multi-view attention as a key mechanism.
• State-of-the-art latent radiance field reconstruction.
Splatent significantly outperforms existing latent-based
methods [61], achieving superior quality on dense, sparse
and cross-dataset generalization tasks.
• Integration with feed-forward 3D models. We show
compatibility with existing latent-based feed-forward
frameworks like MVSplat360 [9], demonstrating consis-
tent improvements in sparse-view scenarios with minimal
modification.
2. Related Work
2.1. Latent Radiance Fields
Radiance fields.
Neural Radiance Fields (NeRFs) [32]
revolutionized 3D reconstruction by representing scenes as
continuous volumetric functions. Subsequent work has ex-
plored various extensions including faster training [6, 33],
unbounded scenes [3], and dynamic content [36, 38]. 3D
Gaussian splatting [21] introduced an explicit point-based
representation with differentiable rasterization, enabling
real-time rendering. Recent methods have extended Gaus-
sian splatting to dynamic scenes [29], large-scale environ-
ments [58], and feed-forward prediction [5, 8].
VAE latent space radiance fields.
Recent work has ex-
plored operating in compressed latent spaces, typically used
by diffusion models, for 3D reconstruction and rendering.
This latent space offers computational efficiency, and na-
tively integrates into latent-diffusion pipelines. For NeRFs,
several methods [30, 35] train directly in the latent space
for 3D generation. 3DGS-based approaches have adopted
a similar approach. Feature-3DGS [62] distills 3D feature
fields from 2D foundation models using Gaussian platting
to render 3D-consistent features. By assigning learnable
feature parameters to each Gaussian, features can be ren-
dered for arbitrary views.
This approach generalizes to
any latent space, including the VAE latent space we fo-
cus on. Building on this, latentSplat [53] extended 3DGS
to address the uncertainty and generative nature of novel
view synthesis given only two input views. They assigned
each Gaussian with VAE mean and variance parameters, al-
lowing for sampling during the splatting procedure. Later,
MVSplat360 [9] suggested a feed-forward pipeline which
2

<!-- page 3 -->
Input Views and Poses
Enhanced Latent
Single-Step
Diffusion
Encoder 
Input Grid
Latent 
3D Gaussian 
Splatting
Ref. View (𝑧!"#)
Ref. View (𝑧!"#)
Ref. View (𝑧!"#)
Rendered 
View ( ̂𝑧)
Decoder
̂𝑧"#$%&#'
̂𝑧
𝑧!"#
$
𝑧!"#
%
𝑧!"#
&
ℇ
𝒟
Figure 2. Framework Overview. Given a set of input views with known camera parameters, each image is encoded into the VAE
latent space of a diffusion model. We then perform 3DGS optimization to reconstruct the underlying latent radiance field. Due to multi-
view inconsistencies in diffusion VAEs latent space, a rendered novel view latent lacks high frequency details. We tile this rendered
view together with reference views into a grid, and leverage a single-step diffusion model with self-attention mechanism that aggregates
information across all views. The enhanced latent image is finally decoded to receive the novel view image.
directly predicts and renders VAE features, and refines them
by using video diffusion for consistent novel view synthe-
sis. Recently, Latent Radiance Fields (LRF) [61] was the
first to observe that although useful, this latent space ex-
hibits non-3D consistent characteristics. They fine-tuned
the VAE, making it more 3D-consistent between different
views. However, fine-tuning the VAE introduces degraded
reconstructions and makes it harder to integrate into already
pre-trained diffusion models, that expect the former input
latent distribution.
Diffusion models for 3D generation.
Diffusion mod-
els [16, 47] have demonstrated remarkable generative capa-
bilities [39]. Recent advances have successfully integrated
diffusion models with 3D representations, enabling appli-
cations such as text-to-3D generation [25, 37, 49], novel
view synthesis [27, 52], and 3D reconstruction [22, 56].
Several methods leverage diffusion for multi-view consis-
tent generation. Difix3D [55] employs a fine-tuned single-
step diffusion model to enhance rendered RGB views using
reference non-distorted views. DiffusioNeRF [57] and Re-
conFusion [56] apply diffusion priors to ensure consistency
across multiple viewpoints.
To aggregate multi-view information in 3D tasks, atten-
tion mechanisms have proven highly effective. Transform-
ers have been successfully applied to multi-view stereo [10],
novel view synthesis [19], and feed-forward 3D predic-
tion [5, 48]. Cross-attention, in particular, enables effec-
tive information flow between views [43, 50], and recent
diffusion-based approaches leverage attention mechanisms
to enforce multi-view consistency [28, 44]. While exist-
ing methods typically perform diffusion in the VAE la-
tent space but subsequently model 3D in RGB space, our
approach conducts the entire pipeline—rendering and en-
hancement—entirely in latent space, utilizing multi-view
attention during diffusion to fuse rendered features with
nearby input views for detail recovery.
3. Preliminary
3D Gaussian splatting.
3D Gaussian splatting [21] rep-
resents a scene using a set of 3D Gaussians, which can be
rendered into an image from a given camera C. Each Gaus-
sian is parameterized as
G = (µ, Σ, α, fc),
(1)
where the spatial properties are defined by the mean µ ∈R3
(center position) and covariance Σ ∈R3×3 (shape and
orientation). Together, these determine the 3D extent and
anisotropic structure of each Gaussian. The scalar opacity
α ∈R controls blending during rendering, allowing soft
compositing of overlapping Gaussians. Each Gaussian is
also characterized with a color representation, fc, which can
be RGB color values or spherical harmonics [11, 21]. Dur-
ing rendering, these colors are alpha-composited via differ-
entiable splatting to produce an output image.
VAEs in latent diffusion models.
The typical approach
for diffusion-based generation involves initially transform-
ing RGB images into a compressed latent representa-
tion [39]. This transformation is achieved through an au-
toencoder that maps the pixel space to either continuous or
discrete latent codes. We focus on the continuous variant,
which is more widely adopted in practice. For a given im-
age I ∈RH×W ×3, an encoder network E maps the image
to a lower-dimensional latent code
z = E(I) ∈Rh×w×d,
(2)
where h =
H
f , w =
W
f , f indicates the downsampling
factor and d is the latent dimension. A decoder network D
then takes this latent and recovers the original image as
ˆI = D(z).
(3)
3

<!-- page 4 -->
latent
image
Rendered
Ours
Ground Truth
(a)
0
2
4
6
8
10
12
14
Radial Frequency
10
2
10
1
100
Relative Magnitude (log scale)
VAE Latents Frequency Profile
Rendered
Ours
Ground Truth
(b)
Figure 3. VAE latents spectral analysis. (a) Images in latent space and the corresponding image space (after decoding) (b) Magnitude
spectrum of the latent image (Rendered, Ours and Ground Truth), normalized to 1. In both visualizations, VAE latents contain both low- and
high-frequency components (green). During 3DGS optimization, inconsistent high frequencies average out, leaving only low-frequency
components (blue) and causing blurry decoded images. Our method produces latents whose spectrum closely matches that of the original
VAE latents, reconstructing high-frequency details (orange). Graphs show averages over more than 45K latent images from 140 scenes.
4. Method
Our approach for novel view synthesis consists of two main
stages. In the first stage, we extract a 3D Gaussian splatting
representation given some input images, directly in the VAE
latent space. However, as we will show, this step alone is
insufficient for achieving high-fidelity reconstructions due
to inconsistencies across multi-view latents. To address this
limitation, the second stage introduces a diffusion-based re-
finement mechanism that leverages attention to fuse ren-
dered latent features from 3DGS with source input view
latents during the denoising process. This diffusion-aware
fusion leads to more consistent and higher-quality results.
In the following sections, we detail the 3DGS formu-
lation operating in the latent space (Section 4.1), analyze
the reconstruction challenges inherent to this setup (Sec-
tion 4.2) and describe our diffusion-based refinement strat-
egy for rendered latents (Section 4.3). The overall pipeline
is illustrated in Fig. 2.
4.1. Latent 3D Gaussian Splatting
Similar to Feature-3DGS [62], we adapt 3DGS to operate
in a feature space rather than directly in image color space.
We use the VAE latent representation as our feature space,
as also done by [61]. Specifically, the 3D Gaussians param-
eters in Eq. 1 are extended with additional fz ∈Rd values.
By splatting the Gaussians, we can render views in the la-
tent space.
Given a set of input views {Ii}N
i=1 with known cam-
era parameters {Ci}N
i=1, each image is encoded into latent
space via Eq. 2, producing {zi}N
i=1. We then perform 3DGS
optimization to reconstruct the underlying latent radiance
field. A rendered novel view latent ˆz can be decoded back
into image space using Eq. 3. This same reconstruction can
also be obtained through a feed-forward approach [9]. In
this case, a feed-forward network directly predicts the la-
tent 3DGS representation from input views.
4.2. Multi-View Inconsistencies in VAE Latents
Our work is driven by the observation that existing VAE
models, such as Stable Diffusion VAE [39], produce latent
representations that lack 3D consistency, limiting their di-
rect use for 3D scene reconstruction and novel view syn-
thesis. This limitation arises from two related spectral de-
ficiencies. First, the latent spaces fail to maintain equivari-
ance under basic spatial transformations like scaling and ro-
tation [23]. Second, and more importantly, view-dependent
high-frequency components, essential for accurate decod-
ing, exhibit the most severe 3D inconsistencies across view-
points, unlike in RGB space [46].
When optimizing 3D Gaussian splatting in latent space,
this spectral inconsistency becomes particularly problem-
atic. As also observed by LRF [61], high-frequency com-
ponents are highly view-dependent and fail to agree across
training views. This leads to conflicting signals during the
optimization process, effectively cancelling out the high fre-
quencies. Consequently, the latent space retains only coarse
structure, while losing the fine details required for photore-
alistic rendering, leading to blurred outputs. This effect is
demonstrated in Fig. 3, where rendered latent features ex-
hibit significantly attenuated high-frequency content com-
pared to the original encoded features, as seen in the ref-
erence and ground-truth views.
We further show in the
Appendix that this phenomenon is widespread across other
VAE models. In contrast, our approach produces latents
that better preserve high-frequency components, as shown
in Fig. 3, resulting in spectra that more closely match the
original VAE latent representation.
4

<!-- page 5 -->
4.3. Diffusion-Based Latent Refinement
To address the lack of high-frequency details in rendered la-
tent features, we propose a diffusion-based refinement mod-
ule that enhances the rendered latent. Inspired by [55], our
method leverages reference views to recover missing de-
tails while preserving the geometry of the rendered latent.
Unlike their work, which focuses on correcting artifacts in
image-space renderings, we aim to reconstruct lost details
arising from 3D inconsistencies in VAE latents.
We base our model on a single-step diffusion model,
which achieves efficient performance during inference. To
enable effective cross-view information transfer, we condi-
tion the diffusion model on reference views by arranging
the inputs in a spatial grid, following [13, 20, 24, 34, 54].
This grid-based arrangement has been shown to be effec-
tive at object preservation across views. This approach of-
fers a clean solution for information sharing without requir-
ing diffusion model-specific architectural changes, enabling
our method to work with future diffusion models. The in-
put grid contains latents extracted from reference images,
with the degraded latent placed in the top-left corner (see
Fig. 2). Reference views are selected as the closest train-
ing views to the degraded latent in both position and orien-
tation. During denoising, the attention mechanism propa-
gates high-frequency details from the references to the ren-
dered latent, mitigating artifacts and improving reconstruc-
tion quality. Alternative strategies for injecting reference
views, along with their corresponding results, are presented
in the Appendix.
Formally, for a rendered latent ˆz, we combine V ad-
ditional reference latents {zi
ref}V
i=1, encoded from nearby
training views.
All latents are tiled into a grid ˆzgrid ∈
R(V +1)×M×d, where d is the number of latent channels and
M = h × w. The view axis is then merged into the spa-
tial dimension, resulting in z ∈R((V +1)·M)×d, and self-
attention is applied jointly across all views. The resulting
grid is passed to the diffusion model, which outputs a re-
fined latent grid, from which we take the top-left position
as the enhanced latent, ˆzrefined. This enhanced latent is then
decoded back to image space with Eq. 3
Training objective.
During training, we use rendered la-
tents ˆz from known training cameras, for which we also
have the corresponding encoded ground-truth latents zgt.
The refinement is supervised by comparing the enhanced
latent with the ground-truth
Lrecon = ∥ˆzrefined −zgt∥2
2.
(4)
To improve perceptual quality, we include LPIPS [60] and
RGB reconstruction losses on decoded images
LLPIPS = LPIPS
 D(ˆzrefined), D(zgt)

,
(5)
LRGB =
D(ˆzrefined) −D(zgt)
2
2,
(6)
where D denotes the VAE decoder. The total training loss
is
Ltotal = Lrecon + λLPIPSLLPIPS + λRGBLRGB.
(7)
This formulation emphasizes that the model enhances a de-
graded latent frame by leveraging both the 3DGS rendering
and nearby reference views through self-attention in the dif-
fusion model.
5. Experiments
Our method consists of a latent space 3DGS reconstruction
stage followed by diffusion-based refinement. Novel view
latents are extracted and fed into our diffusion model for
refinement. We evaluate quantitatively and qualitatively the
reconstructed novel views. In addition, we demonstrate the
effectiveness of Splatent as an enhancer in a feed-forward
latent 3DGS setting.
Baselines.
We compare to Feature-3DGS [62] using the
original implementation but without the encoder-decoder
channel compression modules. Since the VAE latent has
only 4 channels, channel compression is unnecessary. We
also compare to LRF [61], a method for 3D latent recon-
struction. We use the original implementation to train and
evaluate LRF with their default hyperparameters.
Metrics.
We evaluate our approach using standard met-
rics that assess reconstruction quality.
Specifically, we
employ Peak Signal-to-Noise Ratio (PSNR) and Structural
Similarity Index (SSIM) [51] to measure pixel-level accu-
racy and structural fidelity of the reconstructed scenes. Ad-
ditionally, we use Learned Perceptual Image Patch Similar-
ity (LPIPS) [60] and Fr´echet Inception Distance (FID) [15]
to evaluate perceptual quality and distribution similarity be-
tween rendered and ground truth images.
5.1. Datasets
DL3DV-10K [26].
A challenging dataset of real-world
scenes. It includes 10K scenes providing diverse indoor
and outdoor environments with varying lighting conditions
and complex geometries. The benchmark consists of 140
scenes. Importantly, the training and benchmark splits over-
lap, so we filter the benchmark scenes from our training set.
We use a 960 × 540 resolution for efficient optimization,
though our method is not limited to this resolution.
5

<!-- page 6 -->
Feature-3DGS
LRF
Splatent (Ours)
Ground Truth
Figure 4. Qualitative comparison. We compare Splatent to other latent radiance field methods on novel view synthesis reconstruction
quality. Feature-3DGS [62] exhibits considerable loss of detail, and LRF [61] improves upon this baseline but still fails to recover fine
details. In contrast, Splatent produces sharper and more faithful reconstructions. The scenes are taken from the DL3DV-10K dataset.
Table 1. Quantitative comparison. We compare our method using DL3DV-10K, LLFF and Mip-NeRF360 datasets. In the dense setting,
we use 30 input views (except for the LLFF dataset, for which we use 1/8 of views in each scene). In the sparse setting, we use 5 input
views. The rest of the views in each scene are used for evaluation. LRF and Splatent are trained only on DL3DV-10K. Best results in bold.
DL3DV-10K
LLFF
Mip-NeRF360
Method
PSNR↑
SSIM↑
LPIPS↓
FID↓
PSNR↑
SSIM↑
LPIPS↓
FID↓
PSNR↑
SSIM↑
LPIPS↓
FID↓
Dense
Feature-3DGS
16.37
0.545
0.704
263.45
16.23
0.520
0.644
257.20
14.85
0.417
0.739
294.82
LRF [61]
20.19
0.619
0.322
75.32
17.98
0.542
0.379
103.18
19.08
0.489
0.409
135.22
Splatent (Ours)
21.94
0.692
0.265
35.60
19.57
0.610
0.307
52.14
20.42
0.546
0.364
70.90
Sparse
Feature-3DGS
15.04
0.519
0.742
308.00
15.71
0.512
0.660
273.26
14.35
0.406
0.738
314.36
LRF [61]
15.34
0.488
0.494
204.36
17.86
0.529
0.387
102.57
14.25
0.329
0.603
310.76
Splatent (Ours)
17.44
0.573
0.429
86.12
18.53
0.592
0.362
76.62
16.701
0.450
0.501
127.93
Generalization datasets.
To demonstrate generalization
capabilities, we evaluate on two additional standard bench-
marks. The Mip-NeRF360 dataset [2] contains 9 scenes
(5 outdoor and 4 indoor) featuring complex central objects
with detailed backgrounds, captured under controlled con-
ditions with fixed camera exposure and minimal lighting
variation. The LLFF dataset [31] consists of 8 forward-
facing scenes, containing 20-62 images per scene.
5.2. Implementation details
3D Gaussian splatting.
We optimize latent 3DGS repre-
sentations for each scene with dense and sparse input view
configurations: 30 views in the dense setting (for LLFF, we
sample 1/8 of available views), and 5 in the sparse setting.
To ensure optimal spatial coverage, we sample camera po-
sitions using furthest point sampling.
Diffusion model.
We use the pre-trained KL-based VAE
(with compression ratio f = 8) from Latent Diffusion
Model [40] and employ a pre-trained Stable Diffusion
Turbo [41] for latent refinement, which enables fast, single-
pass inference. We set V = 3 reference views in the input
grid to our diffusion model, selected from the set used for
3DGS optimization. The remaining novel views in each
scene are used for supervision and evaluation.
We fine-
6

<!-- page 7 -->
MVSplat360
MVSplat360 + Splatent (Ours)
Ground Truth
Figure 5. Feed-Forward Qualitative comparison. We demonstrate how Splatent can enhance feed-forward latent radiance field methods
such as MVSplat360 [9]. While MVSplat360 often hallucinates (e.g., the window in the first example or the tree in the last example) and
lacks fine details, Splatent yields sharper and more faithful reconstructions.
tune the diffusion model on a subset of 400 scenes from
the DL3DV-10K train split using 8 NVIDIA H100 GPUs
for approximately 24 hours. We use the AdamW optimizer
at a base learning rate of 2 · 10−5, noise level τ = 300, and
loss weights λLPIPS = 2 and λRGB = 1. In the Appendix,
we show the robustness of our model to the noise level τ
along with results using different hyperparameters.
5.3. Results
Qualitative.
We show a qualitative comparison between
Feature-3DGS, LRF and our approach in Fig. 4. As shown,
decoded novel views from Feature-3DGS lack fine-grained
details and textures due to the multi-view inconsistencies of
the latent space. In comparison, LRF reconstructs the scene
more reliably, but trades latent space 3D consistency for re-
construction quality, resulting in loss of details. We inject
details from reference views, leading to highly detailed and
structurally grounded rendered images. Additional visual
results can be found in the Appendix.
Quantitative.
Table 1 presents our results across various
settings and datasets, including experiments with extremely
sparse input views (5 views) to demonstrate the robustness
Table 2.
Components ablation.
Impact of reference image
count. Multiple references reduce hallucinations and enhance de-
tails, with performance saturating at 3 views.
Configuration
PSNR ↑
SSIM ↑
LPIPS ↓
FID ↓
No reference image
19.47
0.626
0.389
83.66
1 reference image
21.61
0.683
0.276
38.04
5 reference images
21.96
0.692
0.263
35.16
Splatent (3 views)
21.94
0.692
0.265
35.60
of our approach. Our method consistently outperforms ex-
isting approaches across all metrics, with particularly strong
generalization on LLFF and Mip-NeRF360. This superior
performance, especially under sparsity, stems from lever-
aging three complementary sources of information: prior
knowledge from the diffusion model, fine-grained details
from reference views and rough geometric structure from
the rendered latents. This combination enables robust re-
construction where traditional methods fail due to insuffi-
cient geometric constraints, as the diffusion prior compen-
sates for missing information while the reference views en-
sure accurate details grounded in the observed geometry.
7

<!-- page 8 -->
5.4. Ablation Studies
We perform an ablation study on the DL3DV-10K bench-
mark using the dense setting to evaluate key components
in our method. Table 2 reports the impact of using multi-
ple reference images compared to a single view. Leverag-
ing multiple references improves reconstruction quality, re-
ducing hallucinations with better grounding of fine-grained
scene details. Increasing the number of reference views fur-
ther enhances performance but eventually saturates, as addi-
tional views contribute limited new information in the dense
setting. We also note that memory consumption grows with
the number of views. For this reason, our base model uses 3
reference views, striking a balance between quality and ef-
ficiency. Additional ablations, including visualizations, are
provided in the Appendix.
5.5. Feed-Forward Setting
We further demonstrate the effectiveness of our diffusion
enhancement model for improving rendered image quality
in a feed-forward latent-based 3DGS model. Specifically,
we integrate Splatent into MVSplat360 [9], a feed-forward
latent 3DGS method for sparse-view (5 input views) 360◦
novel view synthesis. MVSplat360 renders Gaussian fea-
tures directly into the latent space of a pre-trained Stable
Video Diffusion (SVD) model, where these features guide
the denoising process for video generation.
We integrate our approach by applying the single-step
diffusion on the rendered latents, conditioned on reference
latents extracted from the input views. This refinement oc-
curs prior to the final SVD generation step, recovering at-
tenuated high-frequency details before video synthesis. We
train the entire model in an end-to-end fashion, where both
our model and the SVD are fine-tuned.
The weights of
MVSplat360 are initialized with pre-trained weights, and
the combined network is trained for additional 25K steps.
Results.
We demonstrate the effectiveness of Splatent in
improving MVSplat360 results in Fig. 5.
While MVS-
plat360 leverages the generative prior of SVD to synthesize
plausible content, fine-grained details remain blurred and
are sometimes hallucinated. For example, the window in
the top row or the tree in the bottom row in Fig. 5 are not
faithful to the structure of the scene, despite being covered
by the input views. Our integration accurately and faithfully
recovers these details from reference views. This demon-
strates that Splatent addresses a complementary challenge:
MVSplat360 benefits from strong diffusion priors for con-
tent generation, while Splatent enforces fidelity and detail
preservation relative to the input images.
Moreover, Table 3 presents quantitative results showing
consistent improvements across all metrics. By integrating
our method, we achieve more perceptually faithful recon-
structions with enhanced fine details and improved pixel-
Table 3.
Feed-forward latent 3DGS. Quantitative results on
DL3DV-10K using 5 input views. Our method consistently im-
proves perceptual quality while maintaining geometric accuracy.
Method
PSNR↑
SSIM↑
LPIPS↓
FID↓
MVSplat360 [9]
16.691
0.514
0.431
13.462
MVSplat360 + Splatent
17.976
0.531
0.378
11.097
level accuracy. These improvements pave the way for high-
quality latent-based feed-forward methods and diffusion-
based 3D reconstruction frameworks.
6. Limitations
While our method shows significant improvements in latent
space radiance fields representations, some limitations re-
main. Our approach inherently faces a more challenging
problem than RGB-space Gaussian splatting, as we oper-
ate in a lossy latent space where information has been lost
due to the non-3D-consistent latent representation. This re-
quires recovering discarded details rather than refining ex-
isting high-frequency information, making our optimiza-
tion fundamentally more difficult than fixing artifacts in
RGB space where pixel-level details are preserved. In set-
tings where RGB-space Gaussian splatting produces satis-
factory results, it may therefore be the preferable approach.
Nonetheless, our method addresses critical scenarios where
RGB rendering fails, as detailed in previous works [9, 61],
particularly in pipelines requiring latent space optimiza-
tion for memory efficiency or compatibility with generative
models. Additionally, our performance is inherently limited
by the quality of the pre-trained VAE. Despite these limita-
tions, our approach establishes a new paradigm for latent
space 3D reconstruction, demonstrating that high-quality
novel view synthesis is achievable through diffusion-based
refinement in settings where latent rendering is necessary.
7. Conclusion
In this paper, we presented Splatent, a novel approach for
novel view synthesis from radiance fields, working in the
VAE latent space of diffusion models.
We analyzed the
multi-view inconsistencies of this latent space, and identi-
fied several deficiencies in existing latent-space methods.
Our key insight is that these inconsistencies can be ad-
dressed by injecting high-frequency details from reference
views through self-attention in a diffusion model. Exper-
imental results demonstrate that Splatent consistently out-
performs existing methods for latent novel view synthesis
across all metrics. We further show that Splatent can effec-
tively enhance feed-forward latent 3DGS models, leading
to more detailed and faithful renderings. Splatent paves the
way for efficient and high-quality 3D reconstruction in la-
tent space pipelines.
8

<!-- page 9 -->
References
[1] Tristan Aumentado-Armstrong, Ashkan Mirzaei, Marcus A
Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G
Derpanis, and Igor Gilitschenski.
Reconstructive latent-
space neural radiance fields for efficient 3d scene represen-
tations. arXiv preprint arXiv:2310.17880, 2023. 2
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In CVPR, 2022. 6
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In CVPR, pages 5470–
5479, 2022. 2
[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al. Stable video
diffusion: Scaling latent video diffusion models to large
datasets. In CVPR, 2024. 2
[5] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent
Sitzmann. pixelsplat: 3d gaussian splats from image pairs for
scalable generalizable 3d reconstruction. In CVPR, 2024. 2,
3
[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In ECCV, pages
333–350, 2022. 2
[7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang,
Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu,
Qifeng Chen, Xintao Wang, et al.
Videocrafter1: Open
diffusion models for high-quality video generation. arXiv
preprint arXiv:2310.19512, 2023. 2
[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In ECCV, 2024. 2
[9] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang,
Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360:
Feed-forward 360 scene synthesis from sparse views. In Ad-
vances in Neural Information Processing Systems (NeurIPS),
2024. 1, 2, 4, 7, 8
[10] Yikang Ding, Wentao Yuan, Qingtian Zhu, Haotian Zhang,
Xiangyue Liu, Yuanjiang Wang, and Xiao Liu. Transmvsnet:
Global context-aware multi-view stereo network with trans-
formers. In CVPR, pages 8585–8594, 2022. 3
[11] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In CVPR, pages
5501–5510, 2022. 3
[12] Hyojun Go, Dominik Narnhofer, Goutam Bhat, Prune
Truong, Federico Tombari, and Konrad Schindler. Vist3a:
Text-to-3d by stitching a multi-view reconstruction network
to a video generator. arXiv preprint arXiv:2510.13454, 2025.
2
[13] Zheng Gu, Shiyuan Yang, Jing Liao, Jing Huo, and Yang
Gao. Analogist: Out-of-the-box visual in-context learning
with image diffusion model.
ACM Trans. Graph., 43(4),
2024. 5
[14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and
Bo Dai. Animatediff: Animate your personalized text-to-
image diffusion models without specific tuning. In ICLR,
2024. 2
[15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a
two time-scale update rule converge to a local nash equilib-
rium. In NeurIPS, 2017. 5
[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. In Advances in Neural Informa-
tion Processing Systems (NeurIPS), pages 6840–6851, 2020.
3
[17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang,
Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben
Poole, Mohammad Norouzi, David J Fleet, et al. Imagen
video: High definition video generation with diffusion mod-
els. arXiv preprint arXiv:2210.02303, 2022. 2
[18] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi,
Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang
Xu. Lvsm: A large view synthesis model with minimal 3d
inductive bias. In ICLR, 2025. 2
[19] Mohammad Mahdi Johari, Yann Lepoittevin, and Franc¸ois
Fleuret. Geonerf: Generalizing nerf with geometry priors.
In CVPR, pages 18365–18375, 2022. 3
[20] Hao Kang, Stathi Fotiadis, Liming Jiang, Qing Yan, Yumin
Jia, Zichuan Liu, Min Jin Chong, and Xin Lu. Flux already
knows – activating subject-driven image generation without
training. arXiv preprint arXiv:2504.11478, 2025. 5
[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 2, 3
[22] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In ICCV, pages 19729–19739, 2023. 3
[23] Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris,
and Nikos Komodakis.
Eq-vae: Equivariance regularized
latent space for improved generative image modeling.
In
ICML, 2025. 4
[24] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li,
Peng Gao, Zhanyu Ma, and Ming-Ming Cheng.
Visual-
cloze: A universal image generation framework via visual
in-context learning. In ICCV, 2025. 5
[25] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa,
Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler,
Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution
text-to-3d content creation. In CVPR, pages 300–309, 2023.
3
[26] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al.
Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision.
In CVPR, pages 22160–22169,
2024. 5
[27] Ruoshi Liu, Jun Gao, Ben Mildenhall, Xiaohui Shen, Tsung-
Yi Lin, Sanja Fidler, and Jonathan T Barron. Zero-1-to-3:
Zero-shot one image to 3d object. In ICCV, pages 9298–
9309, 2023. 3
9

<!-- page 10 -->
[28] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie
Liu, Taku Komura, and Wenping Wang. Syncdreamer: Gen-
erating multiview-consistent images from a single-view im-
age. In ICLR, 2024. 3
[29] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 3DV, 2024. 2
[30] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and
Daniel Cohen-Or. Latent-nerf for shape-guided generation of
3d shapes and textures. In CVPR, pages 12663–12673, 2023.
2
[31] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Transac-
tions on Graphics (TOG), 38(4):1–14, 2019. 6
[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2
[33] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. In ACM Trans. Graph. (SIGGRAPH),
pages 1–15, 2022. 2
[34] Trevine Oorloff, Vishwanath Sindagi, Wele G. C. Ban-
dara, Ali Shafahi, Amin Ghiasi, Charan Prakash, and Reza
Ardekani. Stable diffusion models are secretly good at vi-
sual in-context learning. In ICCV, 2025. 5
[35] Jangho Park, Gihyun Kwon, and Jong Chul Ye.
Ed-nerf:
Efficient text-guided editing of 3d scene with latent space
nerf. In ICLR, 2024. 2
[36] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In ICCV, pages 5865–5874, 2021. 2
[37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,
2023. 3
[38] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance fields for
dynamic scenes. In CVPR, pages 10318–10327, 2021. 2
[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, pages 10684–
10695, 2022. 3, 4
[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, pages 10684–
10695, 2022. 2, 6
[41] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin
Rombach. Adversarial diffusion distillation. In ECCV, pages
87–103, 2024. 6
[42] Katja Schwarz, Norman M¨uller, and Peter Kontschieder.
Generative gaussian splatting: Generating 3d scenes with
video diffusion priors. In ICCV, 2025. 2
[43] Robin Shi, Honglin Xue, Gaurav Pandey, Jiaming Liang, and
Anh Nguyen. Geometry-free view synthesis: Transformers
and no 3d priors. In ICCV, pages 1559–1569, 2023. 3
[44] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li,
and Xiao Yang. Mvdream: Multi-view diffusion for 3d gen-
eration. In ICLR, 2024. 3
[45] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman.
Make-a-video: Text-to-video generation without text-video
data. In ICLR, 2023. 2
[46] Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Mena-
pace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliak-
sandr Siarohin. Improving the diffusability of autoencoders.
In ICML, 2025. 4
[47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. In ICLR, 2021. 3
[48] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea
Vedaldi.
Splatter image: Ultra-fast single-view 3d recon-
struction. In CVPR, 2024. 3
[49] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for ef-
ficient 3d content creation. In ICLR, 2024. 3
[50] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P
Srinivasan, and Howard Zhou. Ibrnet: Learning multi-view
image-based rendering. In CVPR, pages 4690–4699, 2021.
3
[51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 5
[52] Daniel Watson, Jonathan Ho, Mohammad Norouzi, and
William Chan. Novel view synthesis with diffusion models.
In ICLR, 2023. 3
[53] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele,
and Jan Eric Lenssen. latentsplat: Autoencoding variational
gaussians for fast generalizable 3d reconstruction. In ECCV,
pages 456–473, 2024. 2
[54] Daniel Winter, Asaf Shul, Matan Cohen, Dana Berman, Yael
Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectmate: A
recurrence prior for object insertion and subject-driven gen-
eration. In ICCV, pages 16281–16291, 2025. 5
[55] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Goj-
cic, and Huan Ling. Difix3d+: Improving 3d reconstructions
with single-step diffusion models. In CVPR, pages 26024–
26035, 2025. 3, 5
[56] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong
Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor
Verbin, Jonathan T Barron, Ben Poole, and Aleksander
Holynski. Reconfusion: 3d reconstruction with diffusion pri-
ors. In CVPR, pages 5095–5105, 2024. 3
[57] Jamie Wynn and Daniyar Turmukhambetov. Diffusionerf:
Regularizing neural radiance fields with denoising diffusion
models. In CVPR, pages 4180–4189, 2023. 3
[58] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang,
Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou,
and Sida Peng. Street gaussians for modeling dynamic ur-
ban scenes. In ECCV, 2024. 2
10

<!-- page 11 -->
[59] Junwu Zhang, Zhenyu Tang, Yatian Pang, Xinhua Cheng,
Peng Jin, Yida Wei, Xing Zhou, Munan Ning, and Li Yuan.
Repaint123: Fast and high-quality one image to 3d genera-
tion with progressive controllable repainting. In European
Conference on Computer Vision (ECCV), pages 303–320,
2024. 2
[60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 5
[61] Chaoyi Zhou, Xi Liu, Feng Luo, and Siyu Huang. Latent
radiance fields with 3d-aware 2d representations. In Inter-
national Conference on Learning Representations (ICLR),
2025. 1, 2, 3, 4, 5, 6, 8
[62] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields.
In
CVPR, pages 21676–21685, 2024. 2, 4, 5, 6
11
