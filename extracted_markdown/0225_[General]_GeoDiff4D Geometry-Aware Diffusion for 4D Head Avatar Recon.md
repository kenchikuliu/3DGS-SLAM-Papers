<!-- page 1 -->
GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction
Chao Xu1,†
Xiaochen Zhao1
Xiang Deng1
Jingxiang Sun1
Donglin Di2
Zhuo Su1,‡
Yebin Liu1,§
1Tsinghua University
2Li Auto
Figure 1. We present GeoDiff4D, a framework that reconstructs animatable 4D head avatars from a single portrait image through geometry-
aware diffusion. By jointly predicting portrait image frames and surface normals with a pose-free expression encoder, our method trains
3D Gaussians under dual supervision, achieving exceptional identity preservation and 3D consistency.
Abstract
Reconstructing photorealistic and animatable 4D head
avatars from a single portrait image remains a fundamental
challenge in computer vision. While diffusion models have
enabled remarkable progress in image and video generation
for avatar reconstruction, existing methods primarily rely
on 2D priors and struggle to achieve consistent 3D geome-
try. We propose a novel framework that leverages geometry-
aware diffusion to learn strong geometry priors for high-
fidelity head avatar reconstruction. Our approach jointly
synthesizes portrait images and corresponding surface nor-
mals, while a pose-free expression encoder captures im-
†Intern.
‡Project leader.
§Corresponding author.
plicit expression representations. Both synthesized images
and expression latents are incorporated into 3D Gaussian-
based avatars, enabling photorealistic rendering with ac-
curate geometry. Extensive experiments demonstrate that
our method substantially outperforms state-of-the-art ap-
proaches in visual quality, expression fidelity, and cross-
identity generalization, while supporting real-time render-
ing. Our project page: https://lyxcc127.github.io/geodiff4d/
1. Introduction
The reconstruction of animatable and expressive head
avatars from a single portrait image has attracted consid-
erable attention in computer vision and computer graphics,
1
arXiv:2602.24161v2  [cs.CV]  12 Mar 2026

<!-- page 2 -->
given its wide range of applications in education, filmmak-
ing, gaming, teleconferencing, and virtual reality. However,
as a highly under-constrained problem, the main challenge
in achieving photorealistic and faithful avatar reconstruc-
tion lies in effectively learning strong priors from large-
scale datasets to enhance both identity preservation and sub-
tle facial motion fidelity. In this paper, we present a gen-
eral framework that learns geometry priors from both multi-
view and monocular datasets to reconstruct high-fidelity,
photorealistic 4D head avatars from a single portrait image.
Some recent methods [10, 30, 61], benefiting from the
impressive capabilities of diffusion models, have achieved
high-quality and vivid 2D portrait animation. These ap-
proaches typically excel at preserving identity and transfer-
ring expressions. However, their inherent lack of 3D con-
sistency often leads to significant quality degradation un-
der novel views. Other methods [7, 8, 23] introduce ex-
plicit 3D representations to improve geometric consistency,
but at the cost of reduced identity preservation and limited
ability to capture and reproduce subtle facial expressions.
More recently, several approaches [43, 44] have combined
diffusion-based generative models with 3D Gaussian Splat-
ting (3DGS) [25], synthesizing person-specific data with di-
verse expressions and head poses via diffusion models to
facilitate head avatar reconstruction. However, three key
challenges still remain: (1) existing methods often rely on
facial landmarks, implicit motion representations, or 3D
Morphable Model (3DMM) parameters for expression con-
trol, which often struggle to achieve both 3D consistency
and expressiveness simultaneously; (2) the diffusion mod-
els primarily learn 2D priors, such as pixel-level correspon-
dences, without fully capturing the underlying 3D geome-
try; (3) these methods generally supervise the avatar recon-
struction only with the generated portrait images, resulting
in a weak connection between diffusion and reconstruction
stages, thus failing to fully exploit the diffusion models’ ca-
pability of knowledge distillation.
To overcome these challenges, we propose GeoDiff4D,
a geometry-aware diffusion framework for 4D avatar recon-
struction (Fig. 1). First, we recognize that expressive yet
geometrically consistent expression representations are cru-
cial for guiding the diffusion model to generate vivid and
multi-view consistent results. To this end, we introduce a
pose-free expression encoder that extracts implicit expres-
sion latents capable of capturing subtle facial details. We
enhance its 3D consistency through explicit head-pose con-
trol and a cross-view pairing strategy during training, en-
abling the encoder to maintain geometric coherence across
different viewpoints while preserving expressiveness. Sec-
ond, we develop a joint image–normal diffusion model that
predicts portrait images together with their corresponding
surface normals. By modeling their joint distribution, the
diffusion model learns to leverage the rich 3D geometric
cues encoded in normals—information absent in RGB ap-
pearance alone—thereby significantly enhancing geometric
consistency. Finally, we optimize the avatar model using the
generated images, normals, and expression latents together,
effectively transferring the geometric priors from the diffu-
sion model to enable high-quality 4D reconstruction.
Our contributions can be summarized as follows:
• We present the first video generation model that jointly
synthesizes portrait frames and surface normals, enabling
the diffusion model to learn 3D-aware priors rather than
purely 2D appearance statistics.
• We propose a pose-free expression encoder trained with
cross-view pairing that captures expressive facial dynam-
ics while ensuring robust view consistency.
• We propose a head avatar model that leverages the gen-
erated images, normals, and expression latents to jointly
optimize 3D Gaussian Splatting.
2. Related Work
2.1. Head Avatar Reconstruction
Previous methods [19, 31, 64] leverage 3D Morphable
Models (3DMMs) [16, 28, 50] to reconstruct human heads
with explicit control over facial expressions and head poses,
enabling efficient animation and rendering. However, their
performance is limited by the inherent fixed topology of
mesh structures. Other studies have integrated Neural Ra-
diance Fields (NeRFs) [33] with parametric head models
to achieve view-consistent and photorealistic reconstruc-
tions [1, 14, 63, 65]. While these approaches effectively
capture intricate details, they often suffer from slow ren-
dering speed and long training time. Recent works have
adopted 3D Gaussian Splatting (3DGS) [25], which repre-
sents scenes using explicit 3D Gaussian primitives and en-
ables real-time rendering through efficient tile-based raster-
ization. Some approaches leverage synchronized camera
arrays to achieve photorealistic, high-fidelity reconstruc-
tion [27, 36, 56]. Other methods require only a monoc-
ular video as input, providing a more practical alterna-
tive [6, 39, 53, 58], albeit with a trade-off between data
requirements and view consistency. However, these meth-
ods often struggle to produce competitive results when the
dataset lacks diversity in head poses and facial expressions.
More recently, 3D avatar reconstruction from a single or
a few portrait images has attracted significant attention. Ex-
isting methods can be broadly categorized into two groups.
The first group employs generalizable frameworks [7, 20,
23, 62], which achieve generalization across different iden-
tities by training on large-scale portrait datasets.
These
models learn strong priors that map identity embeddings
to specific 3D representations and typically animate avatars
using 3DMM parameters. The second group follows a two-
stage pipeline [43, 44], leveraging powerful portrait gener-
2

<!-- page 3 -->
Figure 2. Overall architecture. Our system takes a reference image, driving expressions, and head poses as input. Specifically, the reference
image is encoded into hierarchical identity embeddings using a pretrained VAE and UNet-based reference network. Driving expressions
are compressed into low-dimensional latents via a pose-free expression encoder. Both embeddings are injected into the diffusion model
through cross-attention, while head pose maps concatenated with noise serve as inputs. The model then jointly predicts portrait images
and surface normals. For 3D reconstruction, a UNet refines FLAME meshes using expression latents through cross-attention, and an MLP
captures Gaussian dynamics. Finally, the generated surface normals provide additional geometric supervision that further enhances the
reconstruction fidelity.
ation models to synthesize diverse portrait frames from a
given image, which are then used to optimize a head avatar.
2.2. Portrait Animation
Previous methods primarily focus on GANs [18], which
synthesize facial expressions and head poses by warping
and rendering source images. Some approaches incorpo-
rate explicit control signals, such as facial landmarks or
3DMM parameters, to improve control and disentanglement
between motion and appearance [3, 4, 15, 21, 40, 42, 51],
but they often struggle to capture fine-grained facial de-
tails. Other methods leverage implicit representations to re-
produce subtle facial expressions [10–12, 47, 49, 55], yet
they are prone to identity leakage. Moreover, GAN-based
approaches generally face challenges in generating high-
quality outputs and handling out-of-domain portraits due to
the inherent limitations of the architecture. Recently, Latent
Diffusion Models [37] have demonstrated strong genera-
tive capabilities in portrait synthesis. Several works explore
frameworks that combine dual U-Nets [2] with plug-and-
play motion modules [22] to achieve temporally coherent
motion and consistent appearance across frames [45, 54].
Other approaches introduce implicit expression representa-
tion to capture expressive facial motion and achieve fantas-
tic animation results [30, 61].
2.3. Joint Representation
Recent studies have explored combining joint representa-
tions with diffusion models across multiple tasks. Some
approaches employ generative diffusion models to simulta-
neously estimate geometric attributes such as depth and sur-
face normals from single images of static scenes [13, 59].
Others focus on jointly generating color images and cor-
responding modalities of 3D assets or human bodies, such
as motion maps or normal maps, by introducing cross-
domain attention mechanisms [5, 29]. Collectively, these
works demonstrate that incorporating joint representations
enables models to become geometry-aware, thereby en-
hancing the quality and structural coherence of generated
results. Inspired by these findings, we introduce surface
normals paired with portrait frames as joint representation
for human portrait synthesis to improve the 3D consistency
of the generated color images, while the predicted surface
normals further serve as supervision signals for optimizing
avatar reconstruction.
3. Method
GeoDiff4D consists of three key components. First, a pose-
free expression encoder extracts a 1D view-consistent ex-
pression latent from a single image, capturing facial dynam-
ics while disentangling head pose. Second, a video genera-
tion model conditioned on the reference image and expres-
3

<!-- page 4 -->
sion latent jointly synthesizes a sequence of portrait frames
and surface normals. Finally, an animatable 4D avatar is re-
constructed via 3D Gaussian Splatting, leveraging the gen-
erated images, normals, and expression latents to achieve
high-fidelity animation. The full architecture of GeoDiff4D
is illustrated in Figure 2.
3.1. Pose-Free Expression Encoder
We employ an implicit representation to capture fine-
grained facial details and propose a cross-view pairing train-
ing strategy that encourages the encoder to extract consis-
tent expression representations across different viewpoints.
Through joint optimization with the diffusion model, both
components learn complementary and view-consistent fea-
tures. Experiments further demonstrate the robustness and
cross-view consistency of our encoder (See Section. 5.4).
Implicit expression representation.
Following prior
work [61], we use an expression encoder Emot to encode
an image into a low-dimensional latent fmot that captures
facial details while discarding spatial appearance informa-
tion, encouraging disentanglement between expression and
identity. Instead of combining head pose and expression
into a single latent, we disentangle head pose from the im-
plicit representation by introducing an explicit head pose
control signal (See Section. 3.2). The resulting expression
latent then guides the diffusion model via cross-attention to
achieve more accurate and expressive synthesis.
Figure 3. Cross-view pairing training strategy. For each iden-
tity and timestep, frames from different viewpoints are paired with
consistent expressions but varying poses, enabling the encoder to
learn view-invariant representations.
Cross-View Pairing Training Strategy.
To leverage the
multi-view nature of our datasets, we introduce a cross-
view pairing training strategy for learning robust, view-
invariant expression representations. We sample frames of
the same identity and timestep from different viewpoints
to form cross-view pairs, where driving and target frames
share identical expressions but differ in viewpoint (Fig-
ure 3). This effectively mitigates head pose and identity
leakage, encouraging the model to focus on expression-
specific features. The strategy facilitates learning of a pose-
free expression encoder and enhances 3D awareness in the
diffusion model.
Loss and Augmentation.
The encoder is trained end-to-
end with the diffusion model, supervised solely by the dif-
fusion denoising loss without any auxiliary specific objec-
tive. To further promote spatial invariance in the learned
embeddings, we apply data augmentation exclusively to the
cropped facial driving images, including pixelwise augmen-
tations (brightness, contrast, saturation, noise, and blur) to
simulate diverse capture conditions, and spatial augmenta-
tions (scaling, rotation, and translation) to reduce sensitivity
to spatial layout.
3.2. Geometry-Aware Video Generation Model
We adapt a UNet-based latent diffusion framework to
jointly generate portrait frames and their corresponding sur-
face normals while achieving disentangled control over fa-
cial expression and head pose. Furthermore, we introduce a
synthetic dataset with ground-truth surface normal annota-
tions to enhance both the generalizability of the model and
the quality of generated surface normals.
Architecture.
Following X-NeMo [61],
we adopt a
UNet-based latent diffusion framework that integrates a
reference network R, temporal modules and an expres-
sion encoder E.
We adapt this framework for predict-
ing joint distribution of portrait images and surface nor-
mals.
Specifically, a pretrained auto-encoder V com-
presses the reference image Iref into low-dimension la-
tent codes for computational efficiency. The reference net-
work then extracts hierarchical identity features Fref from
these latents. Simultaneously, the pose-free expression en-
coder E (See 3.1) extracts implicit expression representa-
tions Fexp = {fexp}N
n=1 from the driving frames Iexp =
{iexp}N
n=1. A sequence of driving normal maps (See Sec-
tion. 3.2) Mdrv = {mdrv}N
n=1 are interpolated and con-
catenated with the latents, and the diffusion model jointly
denoises the noisy RGB and normal latents, progressively
generating Zrgb = {zrgb}N
n=1 and Znorm = {znorm}N
n=1.
The model is conditioned on Fref and Fexp via cross-
attention mechanisms. Finally, the auto-encoder decodes
Zrgb and Znorm back to RGB space, yielding the synthe-
sized frames Irgb = {irgb}N
n=1 and Inorm = {inorm}N
n=1.
In summary, the model learns the joint distribution:
P(Irgb, Inorm|Iref, Mref, Iexp, Mdrv)
(1)
4

<!-- page 5 -->
Joint Appearance-Normal Representations.
To en-
hance 3D awareness, we introduce surface normals as an ad-
ditional denoising target and modify the diffusion model to
jointly generate RGB images and its corresponding normal
maps. To obtain normals, we use an off-the-shelf normal
estimator [38], which predicts accurate and detailed results
with rich facial structures such as wrinkles and hair strands.
During training, we separately encode the target video
clips and their corresponding normal clips to obtain latent
representations Z with shape [B × D × C × T × H × W].
The latents from different domains are then concatenated
along the domain dimension D where C and T denote the
channel and temporal dimensions, respectively. To ensure
consistency across domains, identical noise is applied to the
latents at each timestep, and class label embedding is in-
troduced to distinguish domain-specific latents. To enable
effective cross-domain interactions, we replace vanilla 2D
self-attention with 3D Domain-Spatial attention modules,
allowing elements from both domains to effectively learn
inter-domain relationships.
Specifically, the domain and
batch dimensions are merged [(B × D) C × T × H × W]
for convolutional processing and separated again before the
attention modules. Latents from different domains are then
concatenated along the width dimension W, forming a ten-
sor of shape [B×C×T ×H ×(2W)] for attention. This de-
sign preserves domain independence in convolutional layers
while allowing controlled information exchange.
Head Pose Map Conditioning.
To enable explicit head
pose control, we introduce head pose maps Mtar
=
{mn
tar}N
n=1 which are essentially normal maps that indicate
target head poses during both training and inference. We
utilize an off-the-shelf FLAME tracker [35] to extract 3D
Morphable Models (3DMM) parameters including shape,
expression, translation, and rotation, which can be used
to reconstruct a corresponding mesh of a personalized 3D
mesh. The head pose maps are then generated by rasteriz-
ing the vertex normals of the mesh onto the 2D image plane
using the corresponding camera intrinsics and extrinsics. To
minimize the identity leakage, we set expression related pa-
rameters to zero when rendering the normal maps. To en-
sure spatial compatibility with the noisy latent input, we in-
terpolate the head pose maps M ∈RB×(V/F )×C×Hm×Wm
to match the latent resolution Z ∈RB×(V/F )×C×Hz×Wz ,
where V and F denotes view dimension and temporal di-
mension. The interpolated head pose maps are then con-
catenated with the latent features before feeding them into
the reference model and denoising model.
Synthetic Dataset.
Current public datasets with real hu-
man portraits lack high-quality surface normal annotations,
and pseudo normals obtained from off-the-shelf estimators
are inherently limited by estimator accuracy.
Synthetic
datasets, by contrast, provide precise ground-truth normals
and have proven effective for various face-related tasks.
We thus incorporate SynthHuman [38], which offers di-
verse identities with ground-truth normals, to supplement
our multi-view training data, enhancing model generaliz-
ability and generated normal quality.
To effectively combine synthetic and multi-view datasets
during joint training, we adopt a weighted random sampling
strategy with manually-set importance weights adjusted by
sample count.
This balances dataset diversity while en-
suring that multi-view data is sampled approximately 10×
more frequently per sample than synthetic data, reflecting
its greater volume and reliability, while smaller synthetic
datasets remain adequately represented.
3.3. 4D Reconstruction
Our 4D avatar reconstruction builds upon the approach
of GaussianAvatars [36], which is based on 3D Gaussian
Splatting [25] and learns a set of 3D Gaussian primitives
attached to the triangles of the FLAME mesh. Specifically,
given a reference image, an expression-driving video, and
head pose maps, we first generate a sequence of portrait
frames along with their associated surface normals. We then
optimize a set of 3D Gaussian primitives under the supervi-
sion of the generated portraits and surface normals. Fur-
thermore, a hierarchical refinement strategy is employed to
enhance geometric consistency and appearance fidelity.
Hierarchical Refinements.
We treat the generated por-
trait video as a monocular input and employ the off-
the-shelf tracker Pixel3DMM [17] to estimate the initial
FLAME parameters, including shape, expression, and head
pose. Attaching the triangles to the FLAME mesh forms
a two-fold pipeline:
the FLAME parameters drive the
FLAME mesh, and the FLAME mesh in turn animates the
3D Gaussians. However, monocular FLAME tracking re-
mains a challenging problem, and the inherent discrepancy
between the 3DMM template and real human facial geom-
etry often degrades the quality of 3DGS optimization. To
mitigate this issue, we introduce a hierarchical refinement
strategy that progressively captures fine-grained facial de-
tails and improves overall geometric fidelity.
First, we introduce learnable residuals for FLAME pa-
rameters to compensate for tracking errors. Second, fol-
lowing prior work [44], we remesh the FLAME head and
employ a U-Net to predict per-vertex deformations. Instead
of reorganizing a UV mesh by sampling adjacent pixels in
the UV map, we construct a face graph in 3D space and re-
organize the UV mesh according to original spatial relation-
ships by querying this graph. Furthermore, unlike previous
methods that define deformation maps in world space, we
use a position map of the FLAME mesh animated solely by
expression-related parameters in canonical space and con-
5

<!-- page 6 -->
dition the U-Net on the expression latent from our pose-
free expression encoder (See 3.1) via cross-attention. Fi-
nally, we employ a lightweight MLP to predict per-primitive
Gaussian attribute residuals, since shared attributes across
expressions cannot fully capture expression-dependent dy-
namics.
Surface Normal Regularization.
Smooth and accurate
surface normals are essential for reducing rendering arti-
facts in novel-view synthesis. While prior methods ben-
efit from multi-view frames with broad facial coverage
and diverse head poses, they still struggle to recover high-
quality normals. To address this limitation, we introduce
pseudo normals predicted by our video generation model
as strong supervisory signals for Gaussian Splatting. Fol-
lowing GaussianShader [24], we take the shortest axis of
each Gaussian primitive as its normal ˆn, extend the Gaus-
sian Splatting representation with dedicated normal chan-
nels, and attach ˆn to the color features during rendering.
We then regularize the rendered normals using a L1 loss
over the foreground region:
Ln = λnL1(ˆn, αn)
(2)
where ˆn and n denote the predicted and pseudo ground-
truth normals, α is the foreground mask, λn is the loss
weight and L1 is the L1 Loss.
4. Implementation
4.1. Video Generation Model
We train our model using a combination of multi-view
datasets [26, 34] and synthetic datasets [38]. All data are
processed to 512×512 resolution, including color images,
surface normals, and foreground masks. Additionally, we
employ an off-the-shelf 3DMM tracker [35] to estimate
3DMM parameters and render head pose maps for each
frame. Training follows a two-stage pipeline. In the first
stage, the model is trained without temporal modules with
a batch size of 32. In the second stage, 16-frame sequences
are incorporated for temporal learning with a batch size of
8, with padding to maintain consistent sequence lengths.
Both stages use AdamW [57] with a learning rate of 1e −5,
trained for 80K and 20K iterations respectively on 4 A800
GPUs, taking approximately 3-4 days in total. All three
datasets are used in both stages. Further discussion of com-
putational efficiency is provided in the supplementary ma-
terial.
4.2. Head Avatar Reconstruction
For data generation, we use our video generation model to
synthesize portrait videos with 12 views and approximately
200 frames, using a 25-step DDIM [41] schedule, taking
1 hour on a single NVIDIA H800 GPU. Head avatar re-
construction follows GaussianAvatars [36] training, running
100K steps in 3 hours on an RTX 3090.
5. Experiments
5.1. Experiment Setting
Metrics.
We employ three image-based metrics to eval-
uate the photometric quality of generated frames: PSNR,
SSIM [52], and LPIPS [60], assessing reconstruction fi-
delity and perceptual similarity.
Temporal coherence is
evaluated using JOD [32], while identity preservation is
measured via the cosine similarity of identity embeddings
(CSIM) [9]. To quantify the accuracy of head pose and ex-
pression transfer, we further report Average Keypoint Dis-
tance (AKD) and Average Expression Distance (AED).
Baselines.
We compare our method with several single-
view 4D head avatar reconstruction systems: Portrait4D-
v2 [10], GAGAvatar [7], LAM [23], and CAP4D [44]
(P3DMM [17] version). These methods represent recent ad-
vances encompassing both feed-forward and optimization-
based paradigms, providing a comprehensive benchmark
for evaluating realism, expression accuracy, and general-
ization. Comparisons with more methods (X-NeMo [61],
Wan-Animate [48], LivePortrait [21] and VoodooXP [46])
are provided in the supplementary material.
5.2. Self-Reenactment
We evaluate self-reenactment on ten unseen subjects from
NeRSemblev2 [26], using 2 sequences per subject with 4
of 16 camera views, yielding 80 driving clips. Reference
images are sampled from the same sequences but different
viewpoints, ensuring comprehensive evaluation across di-
verse head poses and expressions.
As shown in Table 1, our VGM achieves the best per-
formance across all image quality metrics, demonstrating
superior fidelity, identity preservation, and temporal consis-
tency. GeoDiff4D ranks second on most metrics, outper-
forming all baselines. Figure 4 shows our method excels in
image quality, identity similarity, and expression transfer.
Additionally, VGM generates normal maps with rich head
details and high consistency with RGB outputs.
5.3. Cross-Reenactment
We conduct cross-identity animation experiments to eval-
uate performance across diverse subjects.
Following
Sec. 5.2, reference images are sampled from 10 unseen
NeRSemblev2 identities and paired with driving sequences
from different subjects. To test generalization to extreme
expressions and poses, we additionally include an in-the-
wild collection comprising 10 reference identities (real hu-
mans and cartoon characters) and 6 driving sequences.
6

<!-- page 7 -->
Figure 4. Self-Reenactment results. We conduct this experiment on a subset of the NeRSemblev2 dataset containing a large number of
extreme head poses in both the reference images and driving sequences, enabling a comprehensive evaluation of model performance. We
also show surface normals generated by our video generation model.
Method
Self-reenactment
Cross-reenactment
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
JOD ↑
CSIM ↑
JOD ↑
GAGAvatar [7]
17.550
0.789
0.229
0.714
6.244
0.588
5.081
Portrait4D-v2 [10]
13.689
0.701
0.310
0.702
4.933
0.608
4.656
LAM [23]
16.354
0.759
0.251
0.608
5.772
0.516
5.079
CAP4D [44]
19.295
0.811
0.195
0.719
6.561
0.655
5.064
Our VGM
21.586
0.831
0.174
0.754
7.127
0.671
5.066
GeoDiff4D
19.951
0.822
0.195
0.721
6.720
0.656
5.178
Table 1. Quantitative results of self-reenactment on the NeRSem-
blev2 dataset and cross-reenactment on a mixture of NeRSem-
blev2 subset and in-the-wild motion data. Bold and underlined
indicate the best and second-best results, respectively.
Driving sequences are randomly sampled from a mixture
of NeRSemblev2 and in-the-wild motion data, covering a
wide range of expressions and motions.
As reported in Table 1, while GeoDiff4D and VGM may
not top every quantitative metric, further experiments in
Figure 5 show they deliver superior visual quality under ex-
treme head poses and exaggerated expressions. Additional
results are provided in supplementary material.
5.4. Ablation Study and Extensions
Ablation Study.
We conduct ablation studies on the
NeRSemblev2 self-reenactment set to evaluate the contri-
bution of key components. For the video generation model,
we ablate joint representation learning, the Domain-Spatial
attention module, cross-view pairing, and synthetic data.
For avatar reconstruction, we ablate hierarchical refinement,
surface normal regularization, and the source of normal su-
pervision (generated vs. monocular estimated). Qualitative
results are provided in the supplement.
Quantitative results are shown in Tab. 2. Our full VGM
achieves the best overall performance across nearly all met-
rics. Removing joint representation learning degrades ge-
ometry awareness, causing notable drops in reconstruction
quality. Without domain attention, information exchange
between portrait and geometric representations is weak-
ened, leading to consistent degradation. Cross-view pairing
proves most critical, as its removal causes the largest over-
all performance drop, highlighting the importance of multi-
view geometric constraints. Training without synthetic data
limits identity diversity and reduces generalization.
For head avatar reconstruction, removing either the hi-
erarchical refinement or normal regularization degrades
performance across various metrics.
Replacing video-
generated normals with monocular estimated normals sim-
ilarly shows marginal quantitative differences. However,
qualitative results in the supplementary material reveal
more pronounced distinctions across all ablated variants:
the complete model produces fewer artifacts and superior
temporal stability, while video-generated normals further
contribute to finer facial detail, benefiting from their fine-
grained, temporally coherent nature and better alignment
with RGB inputs.
7

<!-- page 8 -->
Figure 5. Cross-Reenactment results. Evaluated on a mixture of NeRSemblev2 and in-the-wild collections spanning diverse real and
cartoon identities to assess model generalizability.
Method
Ablation
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
JOD ↑
AKD ↓
AED ↓
VGM only
w/o joint rep.
20.809
0.825
0.184
0.757
6.960
4.216
2.489
w/o domain attn.
20.984
0.820
0.179
0.743
7.029
4.195
2.556
w/o cv pairing
19.895
0.804
0.191
0.734
6.859
5.367
3.113
w/o synt. data
20.892
0.820
0.181
0.743
6.978
4.339
2.527
Ours
21.586
0.831
0.174
0.754
7.127
4.016
2.340
Recon
w/o hier. refine
19.816
0.821
0.198
0.736
6.758
4.227
2.603
w/o norm. reg
19.950
0.821
0.196
0.734
6.774
4.291
2.713
w DAViD norm.
19.947
0.822
0.196
0.736
6.782
4.247
2.553
Ours
19.953
0.822
0.195
0.737
6.780
4.248
2.563
Table 2. Ablation study. For video generation model, we ablate
joint representation, Domain-Spatial attention module, cross-view
pairing strategy and synthetic data. For avatar reconstruction, we
ablate hierarchical refinement, normal regularization and gener-
ated normal(vs. normal from DAViD).
Extensions.
We validate several additional aspects of our
framework:
(1) view-consistency of the expression en-
coder, (2) the quality gap between synthetic and monocular-
estimated normals and its benefit to normal prediction,
(3) the contributions of hierarchical refinement and nor-
mal regularization to free-view synthesis, and (4) the
quality-efficiency trade-off across different diffusion sam-
pling steps. Please refer to the supplementary material for
full results and comparisons.
6. Disscusion and Conclusion
6.1. Limitations
While our method produces compelling 3D head avatars
from a single image, several limitations remain. First, we
rely heavily on monocular 3DMM tracking for head pose
estimation, which is inherently challenging due to its ill-
posed nature. Second, although our video generation model
supports tongue motion, the final avatar cannot accurately
reconstruct tongue movements due to FLAME’s structural
limitations and lack of fine-grained tongue parameteriza-
tion. Finally, like other diffusion-based methods, our ap-
proach suffers from relatively slow sampling times com-
pared to feedforward alternatives, hindering real-time de-
ployment. Addressing these limitations and improving the
balance between reconstruction quality and inference effi-
ciency are important directions for future work.
6.2. Conclusion
We present GeoDiff4D, a novel framework for high-fidelity
4D head avatar reconstruction from a single portrait im-
age.
By integrating a pose-free expression encoder, a
joint appearance-geometry diffusion model, and 3D Gaus-
sian Splatting-based reconstruction, our method generates
photorealistic and animatable avatars with strong geome-
try priors and consistent identity-expression details. Exten-
sive experiments demonstrate superior performance in iden-
tity preservation, expression fidelity, and cross-view consis-
tency across diverse subjects and challenging poses. Our
approach effectively bridges diffusion-based generation and
3D avatar reconstruction, advancing high-quality digital hu-
man creation.
Social Impact. As with other generative avatar technolo-
gies, our approach may introduce risks related to identity
misuse. We advocate for responsible deployment, includ-
ing consent-aware data use and provenance tracking.
8

<!-- page 9 -->
Acknowledgments
This work was supported by the National Natural Sci-
ence Foundation of China (NSFC) under Grant 62125107.
References
[1] ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli
Shechtman, and Zhixin Shu. Rignerf: Fully controllable neu-
ral 3d portraits. In Computer Vision and Pattern Recognition
(CVPR), 2022. 2
[2] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xi-
aohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mu-
tual self-attention control for consistent image synthesis and
editing, 2023. 3
[3] Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu,
and Gordon Wetzstein. pi-gan: Periodic implicit generative
adversarial networks for 3d-aware image synthesis, 2021. 3
[4] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki
Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo,
Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero
Karras, and Gordon Wetzstein. Efficient geometry-aware 3d
generative adversarial networks, 2022. 3
[5] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam
Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin.
VideoJAM: Joint appearance-motion representations for en-
hanced motion generation in video models. In Forty-second
International Conference on Machine Learning, 2025. 3
[6] Yufan Chen, Lizhen Wang, Qijing Li, Hongjiang Xiao,
Shengping Zhang, Hongxun Yao, and Yebin Liu. Monogaus-
sianavatar: Monocular gaussian point-based head avatar. In
ACM SIGGRAPH 2024 Conference Papers, pages 1–9, 2024.
2
[7] Xuangeng Chu and Tatsuya Harada. Generalizable and an-
imatable gaussian head avatar.
In The Thirty-eighth An-
nual Conference on Neural Information Processing Systems,
2024. 2, 6, 7
[8] Xuangeng Chu, Yu Li, Ailing Zeng, Tianyu Yang, Lijian Lin,
Yunfei Liu, and Tatsuya Harada. Gpavatar: Generalizable
and precise head avatar from image(s), 2024. 2
[9] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kot-
sia, and Stefanos Zafeiriou. Arcface: Additive angular mar-
gin loss for deep face recognition. IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 44(10):5962–5979,
2022. 6
[10] Yu Deng, Duomin Wang, and baoyuan Wang. Portrait4d-v2:
Pseudo multi-view data creates better 4d head synthesizer.
arXiv, 2024. 2, 3, 6, 7
[11] Nikita Drobyshev, Jenya Chelishev, Taras Khakhulin, Alek-
sei Ivakhnenko, Victor Lempitsky, and Egor Zakharov.
Megaportraits:
One-shot megapixel neural head avatars,
2023.
[12] Nikita Drobyshev, Antoni Bigata Casademunt, Konstantinos
Vougioukas, Zoe Landgraf, Stavros Petridis, and Maja Pan-
tic. Emoportraits: Emotion-enhanced multimodal one-shot
head avatars, 2024. 3
[13] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping
Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowiz-
ard: Unleashing the diffusion priors for 3d geometry estima-
tion from a single image, 2024. 3
[14] Guy Gafni, Justus Thies, Michael Zollh¨ofer, and Matthias
Nießner. Dynamic neural radiance fields for monocular 4d
facial avatar reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 8649–8658, 2021. 2
[15] Yue Gao, Yuan Zhou, Jinglu Wang, Xiao Li, Xiang Ming,
and Yan Lu.
High-fidelity and freely controllable talking
head video generation, 2023. 3
[16] Thomas Gerig, Andreas Morel-Forster, Clemens Blumer,
Bernhard Egger, Marcel L¨uthi, Sandro Sch¨onborn, and
Thomas Vetter. Morphable face models - an open frame-
work, 2017. 2
[17] Simon Giebenhain, Tobias Kirschstein, Martin R¨unz, Lour-
des Agapito, and Matthias Nießner. Pixel3dmm: Versatile
screen-space priors for single-image 3d face reconstruction,
2025. 5, 6
[18] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial networks, 2014. 3
[19] Philip-William Grassal,
Malte Prinzler,
Titus Leistner,
Carsten Rother, Matthias Nießner, and Justus Thies. Neu-
ral head avatars from monocular rgb videos. arXiv preprint
arXiv:2112.01554, 2021. 2
[20] Chen Guo, Zhuo Su, Jian Wang, Shuang Li, Xu Chang,
Zhaohu Li, Yang Zhao, Guidong Wang, and Ruqi Huang.
Sega: Drivable 3d gaussian head avatar from a single image,
2025. 2
[21] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou
Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Livepor-
trait: Efficient portrait animation with stitching and retarget-
ing control. arXiv preprint arXiv:2407.03168, 2024. 3, 6
[22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and
Bo Dai. Animatediff: Animate your personalized text-to-
image diffusion models without specific tuning, 2024. 3
[23] Yisheng He, Xiaodong Gu, Xiaodan Ye, Chao Xu, Zhengyi
Zhao, Yuan Dong, Weihao Yuan, Zilong Dong, and Liefeng
Bo. Lam: Large avatar model for one-shot animatable gaus-
sian head. In Proceedings of the Special Interest Group on
Computer Graphics and Interactive Techniques Conference
Conference Papers, pages 1–13, 2025. 2, 6, 7
[24] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaox-
iao Long, Wenping Wang, and Yuexin Ma. Gaussianshader:
3d gaussian splatting with shading functions for reflective
surfaces. arXiv preprint arXiv:2311.17977, 2023. 6
[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 5
[26] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim
Walter, and Matthias Nießner. Nersemble: Multi-view ra-
diance field reconstruction of human heads.
ACM Trans.
Graph., 42(4), 2023. 6
9

<!-- page 10 -->
[27] Jaeseong Lee, Taewoong Kang, Marcel Buehler, Min-Jung
Kim, Sungwon Hwang, Junha Hyung, Hyojin Jang, and
Jaegul Choo.
Surfhead: Affine rig blending for geomet-
rically accurate 2d gaussian surfel head avatars.
In The
Thirteenth International Conference on Learning Represen-
tations, 2025. 2
[28] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and
Javier Romero. Learning a model of facial shape and ex-
pression from 4D scans. ACM Transactions on Graphics,
(Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017. 2
[29] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu,
Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang,
Marc Habermann, Christian Theobalt, and Wenping Wang.
Wonder3d: Single image to 3d using cross-domain diffusion,
2023. 3
[30] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao
Zhang, Tianshu Hu, and Yongming Zhu. Dreamactor-m1:
Holistic, expressive and robust human image animation with
hybrid guidance, 2025. 2, 3
[31] Shugao Ma, Tomas Simon, Jason Saragih, Dawei Wang,
Yuecheng Li, Fernando De La Torre, and Yaser Sheikh. Pixel
codec avatars, 2021. 2
[32] Rafał K. Mantiuk, Gyorgy Denes, Alexandre Chapiro, Anton
Kaplanyan, Gizem Rufo, Romain Bachy, Trisha Lian, and
Anjul Patney. Fovvideovdp: A visible difference predictor
for wide field-of-view video, 2021. 6
[33] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2
[34] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei
Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo
Dai, Ziwei Liu, Chen Change Loy, Chen Qian, Wayne Wu,
Dahua Lin, and Kwan-Yee Lin. Renderme-360: A large dig-
ital asset library and benchmarks towards high-fidelity head
avatars.
Advances in Neural Information Processing Sys-
tems, 36, 2024. 6
[35] Shenhan Qian. Vhap: Versatile head alignment with adaptive
appearance priors, 2024. 5, 6
[36] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide
Davoli, Simon Giebenhain, and Matthias Nießner.
Gaus-
sianavatars: Photorealistic head avatars with rigged 3d gaus-
sians. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20299–20309,
2024. 2, 5, 6
[37] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models, 2021. 3
[38] Fatemeh Saleh, Sadegh Aliakbarian, Charlie Hewitt, Lohit
Petikam, Xiao-Xian, Antonio Criminisi, Thomas J. Cash-
man, and Tadas Baltruˇsaitis. DAViD: Data-efficient and ac-
curate vision models from synthetic data, 2025. 5, 6
[39] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang,
Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang.
SplattingAvatar: Realistic Real-Time Human Avatars with
Mesh-Embedded Gaussian Splatting.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2024. 2
[40] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov,
Elisa Ricci, and Nicu Sebe. First order motion model for
image animation, 2020. 3
[41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denois-
ing diffusion implicit models, 2022. 6
[42] Jingxiang Sun, Xuan Wang, Lizhen Wang, Xiaoyu Li, Yong
Zhang, Hongwen Zhang, and Yebin Liu. Next3d: Genera-
tive neural texture rasterization for 3d-aware head avatars. In
CVPR, 2023. 3
[43] Felix Taubner, Ruihang Zhang, Mathieu Tuli, Sherwin Bah-
mani, and David B. Lindell. MVP4D: Multi-view portrait
video diffusion for animatable 4D avatars, 2025. 2
[44] Felix Taubner, Ruihang Zhang, Mathieu Tuli, and David B.
Lindell.
CAP4D: Creating animatable 4D portrait avatars
with morphable multi-view diffusion models. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 5318–5330, 2025. 2, 5,
6, 7
[45] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo:
Emote portrait alive – generating expressive portrait videos
with audio2video diffusion model under weak conditions,
2024. 3
[46] Phong Tran, Egor Zakharov, Long-Nhat Ho, Liwen Hu,
Adilbek Karmanov, Aviral Agarwal, McLean Goldwhite,
Ariana Bermudez Venegas, Anh Tuan Tran, and Hao Li.
Voodoo xp: Expressive one-shot head reenactment for vr
telepresence, 2024. 6
[47] Phong Tran, Egor Zakharov, Long-Nhat Ho, Anh Tuan Tran,
Liwen Hu, and Hao Li. Voodoo 3d: Volumetric portrait dis-
entanglement for one-shot 3d head reenactment. Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024. 3
[48] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-
iao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jin-
gren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao,
Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang,
Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei
Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui,
Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang,
Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,
Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu
Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yi-
tong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun
Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi
Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open
and advanced large-scale video generative models.
arXiv
preprint arXiv:2503.20314, 2025. 6
[49] Duomin Wang, Yu Deng, Zixin Yin, Heung-Yeung Shum,
and Baoyuan Wang. Progressive disentangled representation
learning for fine-grained controllable talking head synthesis,
2022. 3
[50] Lizhen Wang, Zhiyua Chen, Tao Yu, Chenguang Ma, Liang
Li, and Yebin Liu.
Faceverse: a fine-grained and detail-
controllable 3d face morphable model from a hybrid dataset.
In IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR2022), 2022. 2
10

<!-- page 11 -->
[51] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. One-shot
free-view neural talking-head synthesis for video conferenc-
ing, 2021. 3
[52] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 6
[53] Jun Xiang, Xuan Gao, Yudong Guo, and Juyong Zhang.
Flashavatar: High-fidelity head avatar with efficient gaussian
embedding. In The IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2024. 2
[54] You Xie, Hongyi Xu, Guoxian Song, Chao Wang, Yichun
Shi, and Linjie Luo. X-portrait: Expressive portrait anima-
tion with hierarchical motion attention, 2024. 3
[55] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang,
Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and
Baining Guo.
Vasa-1: Lifelike audio-driven talking faces
generated in real time, 2024. 3
[56] Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang,
Lizhen Wang, Zerong Zheng, and Yebin Liu. Gaussian head
avatar: Ultra high-fidelity head avatar via dynamic gaussians.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024. 2
[57] Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa,
Kurt Keutzer, and Michael W. Mahoney. Adahessian: An
adaptive second order optimizer for machine learning, 2021.
6
[58] Dongbin Zhang, Yunfei Liu, Lijian Lin, Ye Zhu, Kangjie
Chen, Minghan Qin, Yu Li, and Haoqian Wang. Hravatar:
High-quality and relightable gaussian head avatar. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference (CVPR), pages 26285–26296, 2025. 2
[59] Jingyang Zhang, Shiwei Li, Yuanxun Lu, Tian Fang, David
McKinnon, Yanghai Tsin, Long Quan, and Yao Yao. Joint-
net: Extending text-to-image diffusion for dense distribution
modeling, 2023. 3
[60] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric, 2018. 6
[61] Xiaochen Zhao, Hongyi Xu, Guoxian Song, You Xie,
Chenxu Zhang, Xiu Li, Linjie Luo, Jinli Suo, and Yebin Liu.
X-nemo: Expressive neural motion reenactment via disen-
tangled latent attention. arXiv preprint arXiv:2507.23143,
2025. 2, 3, 4, 6
[62] Xiaozheng Zheng, Chao Wen, Zhaohu Li, Weiyi Zhang,
Zhuo Su, Xu Chang, Yang Zhao, Zheng Lv, Xiaoyuan
Zhang, Yongjie Zhang, Guidong Wang, and Lan Xu.
Headgap: Few-shot 3d head avatar via generalizable gaus-
sian priors, 2025. 2
[63] Yufeng Zheng, Victoria Fern´andez Abrevaya, Marcel C.
B¨uhler, Xu Chen, Michael J. Black, and Otmar Hilliges. I
M Avatar: Implicit morphable head avatars from videos. In
Computer Vision and Pattern Recognition (CVPR), 2022. 2
[64] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J.
Black, and Otmar Hilliges. Pointavatar: Deformable point-
based head avatars from videos.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2023. 2
[65] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Instant
volumetric head avatars, 2023. 2
11

<!-- page 12 -->
GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction
Supplementary Material
A. Implementation Details
A.1. Video Generation Model
Noise Schedule
During inference, we do not initialize the
diffusion process from pure random noise. Instead, we in-
ject reference information directly into the initial latent by
adding Gaussian noise to both the reference image and its
corresponding normal map, and then use these noisy signals
as the starting point for DDIM denoising. This reference-
conditioned initialization encourages the model to preserve
identity features, fine-grained facial details, and geomet-
ric structure more faithfully throughout the iterative sam-
pling process, reducing ambiguity compared to fully noise-
based initialization. Concretely, we first construct the two
domain-specific latents (image and normal) independently
and then concatenate them along the domain dimension to
form an initial latent of shape [B × D × C × T × H × W],
following the formulation described in Section 3. By start-
ing from this structurally coherent latent, the model benefits
from stronger reference guidance, resulting in more consis-
tent and identity-preserving outputs across both spatial and
temporal dimensions.
Classifier-Free Guidance
During training, we randomly
drop the expression latent with a probability of 0.1 to facil-
itate classifier-free guidance, enabling the model to learn
to generate outputs with and without explicit expression
conditioning. During inference, classifier-free guidance is
applied simultaneously to both domains to maintain con-
sistent control over the generated image and normal sig-
nals. Specifically, after constructing the initial noisy latents
as described previously, we first reshape them into an im-
age–normal latent of shape [(B×D)×C×T ×H×W]. This
latent is then duplicated along the batch–domain dimension
to form an unconditional–conditional latent pair, which al-
lows the guidance mechanism to differentiate between con-
ditioned and unconditioned signals. All other conditioning
inputs, including expression latents, head pose maps, and
class labels, are processed in the same manner to maintain
alignment across all control signals. Finally, we apply the
standard classifier-free guidance procedure with a guidance
scale of 2.5 for all experiments, ensuring strong adherence
to the desired expressions and head poses while enhancing
overall fidelity and consistency across both spatial and tem-
poral dimensions.
Video Synthesis for Reconstruction
We use the video
generation model to synthesize videos of approximately
200 frames across 12 viewpoints. Specifically, we select
12 out of the 16 available viewpoints from NeRSemblev2
and adopt their camera configuration, keeping the head pose
fixed while animating only the facial expressions of the ref-
erence image. We then concatenate all video clips into a
single sequence, treating it as a monocular video, which is
subsequently used for avatar reconstruction.
A.2. 4D Reconstruction
FLAME Refinement
To refine the initial FLAME track-
ing and compensate for errors from monocular fitting, we
introduce learnable residuals for all FLAME parameters, in-
cluding shape, expression, jaw, neck, eye, and eyelid co-
efficients, as well as global pose (R, t).
These parame-
ters are jointly optimized with the Gaussian attributes using
a three-phase learning-rate schedule (warmup, stable, de-
cay). We divide the FLAME parameters into two groups:
pose- and shape-related parameters are trained with a con-
servative schedule and a peak learning rate of 1e−5, while
expression-related parameters adopt a higher peak learning
rate of 1e−4 to better capture fine-grained facial dynamics.
All parameters start from an extremely low learning rate
of 1e−10 and linearly ramp up during the first 40K itera-
tions to prevent instability in early training. The learning
rate is maintained at its peak from 20K to 80K iterations
to allow sufficient exploration and then exponentially de-
cayed from 80K to 100K iterations to ensure convergence.
This staged schedule, combined with group-specific learn-
ing rates, enables stable joint optimization of tracking pa-
rameters and Gaussian attributes while preserving photo-
metric fidelity and temporal coherence.
Topology-Preserving Remeshing
To ensure geometric
consistency when remeshing the FLAME template into UV
space, we adopt a topology-preserving strategy that vali-
dates the connectivity of newly generated faces. Standard
UV remeshing subdivides each UV grid cell into two tri-
angles to create a dense tessellation, but this approach can
inadvertently produce invalid faces that connect distant or
topologically unrelated regions of the original mesh. Such
invalid connections can lead to visual artifacts and geomet-
ric distortions during rendering and deformation.
To prevent such artifacts, we introduce an adjacency-
based validation mechanism grounded in the original
FLAME topology.
We first construct a face-adjacency
graph encoding connectivity between all FLAME faces,
which is precomputed and cached for efficiency. For each
candidate UV face, we retrieve the FLAME face indices of
its three vertices through the UV rasterization mapping. A
1

<!-- page 13 -->
UV face is retained only if (1) all vertices belong to the same
FLAME face, or (2) the vertices span multiple FLAME
faces that are mutually connected within a bounded hop
distance in the adjacency graph. This multi-hop connectiv-
ity check is performed via breadth-first search (BFS) with a
maximum hop threshold of 5. This mechanism effectively
filters out topologically invalid or distorted triangles while
preserving sufficient mesh density for high-quality Gaus-
sian splatting. The resulting UV mesh maintains the origi-
nal FLAME topology and provides a reliable, deformation-
aware surface for attaching Gaussian attributes.
B. More Ablation Results
B.1. VGM Ablation
Portrait Generation
Qualitative results are presented in
Fig. 6. Our full model shows clear improvements in fine
facial details, such as wrinkles, eyelashes, and teeth, com-
pared to the ablated versions. When joint representation
learning is removed, the model’s ability to transfer expres-
sions is noticeably weakened. We believe this is due to the
absence of 3D consistency provided by the surface normal
domain, which increases the entanglement between iden-
tity, head pose, and expression. Similarly, removing the
domain attention module degrades both expression transfer
and the fidelity of facial details, emphasizing the importance
of cross-domain information exchange between image and
normal features. Ablating the cross-view pairing strategy
leads to obvious identity leakage and degraded driving qual-
ity, where the generated results inappropriately retain char-
acteristics from the driving sequence rather than faithfully
following the reference identity and driving expressions.
The results obtained without synthetic data are closer to the
complete model, indicating that while the inclusion of syn-
thetic data provides additional gains in generation quality,
the contributions of the core modules—joint representation,
domain attention, and cross-view pairing—are essential for
robust and high-fidelity video generation.
Normal Generation
We further evaluate the impact of
synthetic data on the quality of generated surface normals.
As shown in Fig. 7, removing synthetic data results in a
modest but noticeable reduction in high-frequency facial
details, such as wrinkles and fine contours.
Incorporat-
ing synthetic data provides an overall improvement in vi-
sual fidelity, particularly in regions that are challenging to
reconstruct from pseudo-ground-truth normals alone. We
believe this improvement is primarily due to the high ac-
curacy and fine-grained detail of the synthetic normals,
which offer stronger geometric supervision compared to
the limited-fidelity pseudo-ground-truth normals present in
other datasets. This effect is further illustrated in Fig. 8,
where synthetic normals help preserve subtle facial geome-
try that is otherwise lost.
B.2. Head Avatar Ablation
Qualitative results in Fig. 9 show that removing hierarchi-
cal refinement leads to noticeably degraded reconstruction
quality, particularly in sequences with large head rotations
and exaggerated expressions, where errors from monocu-
lar FLAME tracking propagate more severely without hier-
archical correction. Ablating normal regularization further
highlights its critical role in providing geometric guidance
for challenging regions such as the mouth and teeth, and
in mitigating artifacts under extreme head poses. Fig. 10
further demonstrates the contribution of both modules to
free-view rendering quality, where the full model signifi-
cantly reduces artifacts and produces more faithful novel-
view synthesis. As shown in Fig. 11, normals generated by
our model capture finer facial details compared to monoc-
ular estimated normals, enabling more accurate geomet-
ric guidance for Gaussian splatting optimization. Overall,
these results confirm that hierarchical refinement, normal
regularization, and high-quality generated normals are all
essential for producing high-fidelity 3D head avatars.
C. More Results
C.1. Comparisons with More Baselines
We provide additional comparisons with more baselines in
Tab. 3 and Fig. 12, including diffusion-based generative
models X-NeMo and Wan-Animate, as well as LivePor-
trait and VoodooXP. Both qualitative and quantitative re-
sults show that our method achieves the best performance
on most metrics, delivering more accurate expression trans-
fer and remaining robust under large head pose variations.
C.2. Cross-View Animation
The results illustrated in Fig. 13 demonstrate that our pose-
free expression encoder exhibits strong cross-view consis-
tency, producing highly consistent facial expressions across
a wide range of camera viewpoints. Even when the head
pose is held fixed, the encoder effectively disentangles ex-
pression from pose, maintaining detailed and coherent fa-
cial dynamics regardless of the viewing angle. This high-
lights the robustness of our encoder in preserving expres-
sion fidelity across diverse perspectives.
C.3. Computational Efficiency Discussion
As computational efficiency is a practical concern for
diffusion-based models, we evaluate generation quality
against inference cost across different sampling steps
(Tab. 4). We adopt 25 steps as our default, striking a fa-
vorable balance between quality and speed.
2

<!-- page 14 -->
Figure 6. Ablation of portrait generation. We ablate joint representation learning, the Domain-Spatial attention module, cross-view pairing,
and synthetic data.
Figure 7. Ablation of normal generation. We ablate synthetic data to evaluate its contribution to normal generation quality.
3

<!-- page 15 -->
Figure 8. Comparison of ground-truth and pseudo-ground-truth normals. Ground-truth normals from synthetic data are more accurate and
capture finer geometric details, thereby improving the quality of generated surface normals.
Figure 9. Ablation on 4D reconstruction. We ablate hierarchical refinement and normal regularization to evaluate their contributions to
reconstruction quality.
4

<!-- page 16 -->
Figure 10. Ablations on hierarchical refinement and normal regularization for free-view rendering.
Figure 11. Ablations on our normals versus DAViD normals.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
JOD ↑
VOODOOXP
13.111
0.714
0.276
0.598
5.116
LivePortrait
18.047
0.787
0.214
0.736
6.230
Wan-Animate
12.805
0.655
0.354
0.712
4.597
X-NeMo
14.202
0.683
0.311
0.732
4.987
Our VGM
21.586
0.831
0.174
0.754
7.127
GeoDiff4D
19.953
0.822
0.195
0.737
6.780
Table 3. Quantitative comparison with more baselines (best, second-best).
Figure 12. Qualitative comparison with more baselines.
Steps
PSNR ↑
SSIM ↑
LPIPS ↓
CSIM ↑
JOD ↑
AKD ↓
AED ↓
Speed (s/frame) ↓
10
21.177
0.8286
0.1779
0.7419
7.033
4.215
2.503
1.16
25
21.505
0.8294
0.1746
0.7419
7.093
4.242
2.541
2.74
50
21.529
0.8285
0.1740
0.7421
7.097
4.283
2.573
5.39
100
21.545
0.8280
0.1736
0.7412
7.100
4.261
2.550
10.66
Table 4. Ablation on sampling steps (best, second-best).
5

<!-- page 17 -->
Figure 13. Cross-view animation results. We generate images using expression sequences from 12 different camera viewpoints while fixing
the head pose, demonstrating the view consistency of our pose-free expression encoder.
6
