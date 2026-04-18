<!-- page 1 -->
Complete Gaussian Splats from a Single Image with Denoising Diffusion Models
Ziwei Liao1
Mohamed Sayed2
Steven L. Waslander1
Sara Vicente2
Daniyar Turmukhambetov2
Michael Firman2
1University of Toronto
2Niantic Spatial
https://nianticspatial.github.io/completesplat
Abstract
Gaussian splatting typically requires dense observations
of the scene and can fail to reconstruct occluded and unob-
served areas. We propose a latent diffusion model to recon-
struct a complete 3D scene with Gaussian splats, includ-
ing the occluded parts, from only a single image during in-
ference. Completing the unobserved surfaces of a scene is
challenging due to the ambiguity of the plausible surfaces.
Conventional methods use a regression-based formulation
to predict a single “mode” for occluded and out-of-frustum
surfaces, leading to blurriness, implausibility, and failure
to capture multiple possible explanations. Thus, they often
address this problem partially, focusing either on objects
isolated from the background, reconstructing only visible
surfaces, or failing to extrapolate far from the input views.
In contrast, we propose a generative formulation to learn a
distribution of 3D representations of Gaussian splats con-
ditioned on a single input image. To address the lack of
ground-truth training data, we propose a Variational Au-
toReconstructor to learn a latent space only from 2D images
in a self-supervised manner, over which a diffusion model is
trained. Our method generates faithful reconstructions and
diverse samples with the ability to complete the occluded
surfaces for high-quality 360◦renderings.
1. Introduction
Gaussian splatting [34] has demonstrated impressive perfor-
mance in modeling 3D scenes, delivering high-quality ren-
derings with applications in augmented reality, world sim-
ulations, and robotics. Traditionally, dense observations,
typically hundreds of RGB images, are required to accu-
rately capture scene details and constrain the reconstruction
process [34, 46], due to the large number of unknown pa-
rameters. However, such dense observations are not always
available, often resulting in blurriness, artifacts, or empty
regions caused by underconstrained reconstructions in oc-
cluded or invisible areas, such as the backs of objects and
Gaussian 
splat
Gaussian 
splat
Gaussian 
splat
Our Gaussian splats enable 
efficient novel view synthesis
Input image
Our denoising 
diffusion model
We can generate 
multiple samples
Figure 1. We predict full Gaussian scenes from a single RGB
input image. Our diffusion-based model outputs sharper results
than existing methods, and is also able to sample diverse comple-
tion “modes” given a single image as input.
regions outside the camera frustum. In cases where sparse
or even a single view is available, prior knowledge beyond
the observations becomes critical for constraining the re-
construction.
We aim to address the learning of prior knowledge from
large-scale datasets to help constrain the highly ill-posed
problem of reconstructing a complete 3D scene from a sin-
gle RGB image. Researchers have trained neural networks
to reconstruct 3D Gaussian splats from a single image [60,
61] or sparse two-view images [7, 8]. These approaches can
be broadly categorized into two formulations: regression-
based and generative. As regression-based methods, Splat-
ter Image [61] and its follow-up works [7, 8, 60] predict
multiple Gaussians along each camera ray in a pixel grid
using a single forward pass, efficiently modeling scenes
within the image frustum. However, a fundamental limi-
tation exists for regression-based formulations. They pro-
duce unimodal predictions, where only a single output is
predicted for a given input image. This forces the model
to average multiple hypotheses into a single “mode” for oc-
cluded or ambiguous regions, often resulting in blurry or
inaccurate reconstructions. Due to this limitation, Splat-
ter Image demonstrates performance only on isolated fore-
1
arXiv:2508.21542v1  [cs.CV]  29 Aug 2025

<!-- page 2 -->
ground objects. While follow-up works [7, 8, 60] extend
these methods to scene-level reconstruction, they remain
constrained to simpler tasks such as interpolation and ex-
hibit limited extrapolation ability, particularly when gener-
ating parts of the scene that are not directly visible.
Instead, we propose a generative approach to learn a dis-
tribution over 3D Gaussian splats, enabling the sampling of
multiple high-quality hypotheses in ambiguous situations,
as shown in Figure 1. Generative formulations have also
been explored on NeRFs [65] and Gaussian splats [23, 70].
Different from LatentSplat [70], which follows a variational
autoencoder (VAE), we leverage more powerful denoising
diffusion models [25], which have demonstrated a stronger
ability to model complex distributions in images [53] and
3D models [31], to better capture the complex distributions
of Gaussian splatting.
Recently, diffusion models have been used for Gaussian
splatting. Some approaches [9, 36, 63, 77] leverage priors
from 2D image diffusion models to generate 3D Gaussians,
which requires intensive optimization and can suffer from
consistency issues. Limited work exists that learns diffu-
sion models to directly model 3D distributions of splats.
For example, Bolt3D [62] relies on extra regression heads
for generating Gaussians from denoised intermediate out-
puts. Some methods require ground truth 3D data of Gaus-
sian splats [45, 72], which is expensive to obtain and limits
scalability to diverse scenes. DFM [65] trains a denoising
diffusion model over NeRFs [46] from only image supervi-
sion with forward models. Its architecture requires compu-
tationally expensive denoising for each rendered image and
also inherits NeRF’s high computational cost.
Different from the above, we propose a novel generative
method for 3D Gaussian splatting based on a latent diffu-
sion architecture [53] to directly learn the distribution of 3D
Gaussian splats in a latent space from only image supervi-
sion, enabling both real-time computation and high-quality,
diverse samplings. This is not a straightforward process,
and we make the following critical contributions to enable
an effective training pipeline. The first challenge is how to
acquire 3D ground truth Gaussian splat training data. To
address this, we propose a novel method, Variational Au-
toReconstructor, to learn a latent space for 3D Gaussian
splats using self-supervised losses derived solely from im-
ages. To mitigate the loss of high-frequency details in the
latent space, we use skip connections to propagate and pre-
serve fine-grained details. We further propose an approach
based on classifier-free guidance to provide flexible control
that balances faithfulness and randomness in the outputs for
different applications. Our contributions are as follows:
• We propose a generative Gaussian splatting reconstruc-
tion method with a latent diffusion model for generating
complete 3D scenes from a single RGB image, enabling
the completion of occluded surfaces with multi-view con-
sistency and efficient reasoning.
• We present a novel Variational AutoReconstructor to
learn the latent space for Gaussian splats from only 2D
posed images using self-supervised differentiable render-
ing losses.
• We present flexible control over the fidelity of recon-
structed visible surfaces and the diversity of generated
occluded surfaces within a latent diffusion model, using
skip connections and classifier-free guidance.
Our experiments on the Hydrants and TeddyBears cate-
gories from the CO3D [52] dataset, as well as room scenar-
ios from the RealEstate10K [83] dataset, show that our pre-
dictions are sharper and more complete than those of state-
of-the-art methods, and provide new abilities to sample di-
verse 3D outputs.
2. Related Work
There is limited work on training denoising diffusion mod-
els to directly output 3D Gaussian splats. We provide an
overview highlighting the differences with the broader topic
in Table 1, and discuss them in detail below.
Summary
Unlike novel view synthesis methods that pro-
duce only 2D images [6, 20, 27], or videos lacking explicit
3D modeling and consistency [22, 26], or virtual scenes
generated by text-to-3D models that suffer from domain
gaps compared to real scenes [63, 77], our model naturally
learns and outputs distributions of 3D Gaussian splats of
real scenes with guaranteed 3D geometric consistency. Un-
like object-only reconstruction models [2, 73, 79, 84], we
reconstruct complete 3D scenes, including both foreground
objects and background. Different from offline NeRF-based
methods [65], our representation can be rendered in real-
time. And unlike regression-based methods [7, 8, 61], we
use a generative formulation with a denoising diffusion
model, enabling the completion of occluded or invisible
parts and the sampling of diverse scene variations.
2.1. Learning 3D Reconstruction and Generation
Object-centric Models [2, 41, 58, 73, 75, 76, 79, 84] re-
construct and generate 3D object models given text prompts
or images, but lack awareness of background and scene
context.
Due to the domain gap in large-scale object
datasets [14], the generated outputs often appear synthetic
and virtual, with noticeable discrepancies from real-world
textures and geometry. Notably, DMV3D [76] denoises a
triplane NeRF representation, whereas ours denoises Gaus-
sian splatting in a latent space, enabling much faster in-
ference and training. Most importantly, it remains unclear
how well these methods scale to larger and more complex
scene-level reconstruction tasks, especially in the absence
of large-scale, ground-truth 3D scene data.
Scene-level “Regression” Models predict a single “mode”
of the 3D representation from partial input data.
How-
2

<!-- page 3 -->
Scale
Methods
Outputs
Inputs
Formulations
Architectures
3D Representations
Computation
Objects
Objects Models [2, 73, 79, 84]
3D
/
/
/
GS, NeRF
/
Novel View Sythesis [6, 20, 27]
Image
/
/
/
/
Video Generation [37, 44]
Video
/
/
/
None
/
SplatterImage [7, 8, 60, 61]
3D
1 or 2-views
Regression-based
/
GS
Real-time
LatentSplat [70]
2-views
VAE
GS
Real-time
CAT3D [19]
1-view
2D Diffusion
GS
Offline
DFM [65]
1-view
3D Diffusion
NeRF
Offline
Scenes
Ours
3D
1-view
Generative
3D Diffusion
GS
Real-time
Table 1. Comparison to Related Works. The table highlights the main differences from closely related baselines. We propose a generative
method with diffusion models to reconstruct 3D scenes with Gaussian splats in real time from a single image.
ever, in ambiguous situations, such as occluded areas, the
network tends to learn an “average” mode, which often
leads to blurry results.
Early works focused on voxel
grids [11–13, 17, 59, 68, 71] and occupancy fields [29],
which model geometry only, without capturing texture.
Dust3R and its variants [66, 67] show promising end-to-
end learning for camera localization and reconstruction with
point clouds, which is orthogonal to ours, where we focus
on representing 3D scenes. To synthesize 360◦views with
NeRFs [46], PixelNeRF [78] predicts novel views condi-
tioned on features extracted from single (or few) input im-
ages. NViST [30] further extends this with a Transformer
to predict novel views. However, these methods are limited
by the slow rendering process of NeRF, which is mitigated
by Gaussian splats [34]. Splatter Image [61] and follow-
up works [15, 28] predict multiple Gaussians along camera
rays in a pixel grid, resulting in efficient methods to model
scenes within image frustums. Multi-view inputs [7], depth
priors [43, 60, 74], and structural constraints [8] have been
further explored to regularize the outputs.
Scene-level “Generative” Models can predict a “distribu-
tion” of 3D scene representations to model complex and
ambiguous situations. LatentSplat [70] follows a variational
autoencoder (VAE) design to infer the distribution of Gaus-
sian splats. We leverage more powerful denoising diffusion
models [25], which have demonstrated a stronger ability to
model complex distributions in images [53] and 3D mod-
els [31], to better capture the complex distributions of Gaus-
sians. DFM [65] trains a denoising diffusion model to learn
a distribution of NeRFs conditioned on a single input im-
age. However, it requires running denoising steps for each
novel view, requiring extremely intensive computation.
Novel View Synthesis and Video Generation Models out-
put images or videos [37, 44] conditioned on camera poses.
NerfDiff [20], GeNVS [6], and ViewDiff [27] train diffu-
sion models to generate novel views. However, they cannot
directly output a coherent 3D representation. Instead, they
need to run the intensive denoising process separately for
novel views to generate images or videos, which increases
computational cost and, most importantly, limits multi-view
consistency without 3D geometry.
2.2. Diffusion Models for Gaussian Splats
2D Diffusion: Lifting Image Diffusion Models for 3D
2D image generation models [53] contain priors that can
be used to optimize 3D representations using Score Distil-
lation Sampling (SDS) from rendered multi-view images of
NeRFs [50, 55] or Gaussian splats [9, 36, 63, 77]. Some ap-
proaches first synthesize multiple novel view images and
then optimize the 3D scene [16, 19, 42].
However, a
large number of novel views are required to fully constrain
3D representations during optimization, where identity and
multi-view consistency are challenging, requiring finetun-
ing of the large diffusion model (e.g. Zero 1-to-3 [42] and
Cat3D [19]). Some methods [57, 82] use video diffusion
models as priors. In principle, the priors of image and video
generation models operate in 2D image space rather than di-
rectly on the 3D representation. As a result, these methods
suffer from long optimization times and geometric consis-
tency issues, such as the multi-face problem [50].
3D Diffusion: Direct Modelling of 3D Gaussian Splats
Distributions
There is limited work exploring diffusion
models to directly learn the distribution of 3D Gaussian
splats, with the biggest challenge being the lack of large-
scale 3D ground-truth Gaussian splat datasets for real
scenes. Some works focus on objects [31, 38, 45, 47] rather
than entire scenes. Some works finetune diffusion models
originally designed for image generation [39, 45, 72], tailor-
ing the output to align with Gaussian splat modalities. How-
ever, [39, 45, 72] require ground truth 3D Gaussian splats in
order to learn the latent space, making these methods com-
putationally intensive due to the need to build a dataset in
advance. In contrast, we learn the latent space directly from
images without requiring ground truth splats, with the po-
tential to scale up with the availability of large-scale im-
age datasets. Bolt3D [62] uses a multiview diffusion model
to generate images and requires an additional intermediate
model to produce a splat representation. Some works [49]
rely on an extra regression-based model as a teacher to su-
pervise the diffusion model, complicating the process. Most
similar to ours, Henderson et al. [23] train a denoising dif-
fusion model to generate Gaussian splats directly. Their
3

<!-- page 4 -->
model is trained to encode multi-view input images into a
multi-view latent space. However, during inference, it re-
quires knowledge of multiple camera positions, which lim-
its its applicability in real-world scenarios, whereas ours
only requires single-view input.
3. Method
In this section, we first discuss the challenges of training de-
noising diffusion models on 3D Gaussian splats in the ab-
sence of ground-truth 3D data. We then propose a novel ar-
chitecture, the Variational AutoReconstructor, which learns
a latent space using forward models (differentiable ren-
derings), along with critical modifications for preserving
high-frequency details. These components together enable
the training of a diffusion model to generate 3D Gaussian
splats.
3.1. Training Diffusion Models without Ground
Truth Samples
Conventional denoising diffusion models [25, 53] require
ground-truth samples to which noise is progressively added,
and from which the denoiser learns to remove noise. We
follow Splatter Image [61] to model 3D Gaussian splats
within an image frustum by predicting multiple Gaussians
along each camera ray. In order to train a diffusion model
to learn a distribution P(X) over the Splatter Image X, we
need access to a large-scale dataset of X, i.e., ground-truth
Splatter Images. However, unlike images [53], videos [26],
or 3D objects [31], no large-scale scene-level datasets ex-
ist for 3D Splatter Image representations, which are costly
and time-consuming to construct. One approach might be to
pre-optimize Splatter Images from multi-view inputs [72].
However, aside from the computational and scalability lim-
itations, our early experiments with this method revealed
sparsity and continuity issues hinder the effective learning
of neural networks. Therefore, we propose a more scalable
method that directly learns from image datasets [10, 52, 83],
without relying on intermediate modules.
Training Diffusion Models over Gaussian splats from
Images Supervision
The feasibility of training diffu-
sion models from low-dimensional observations has been
scarcely explored in the literature. Theoretically, our goal
is to learn a diffusion model that approximates the distri-
bution P(X) without access to ground truth samples of X,
but only to indirect projections f(X) of X. Here, f is a
differentiable function—referred to as a “forward model”
or “observation model”, which outputs a lower-dimensional
partial projection of the high-dimensional variable X. In
our case, the projections are rendered images from Gaus-
sian splats. This links our image datasets of scenes to the
potential unknown Gaussian representations that can model
those scenes. Instead of directly training on Splatter Im-
ages, we follow a latent diffusion architecture, which mod-
1
(a) Variational Autoencoders (VAE)
𝑥
Encoder
𝑞𝜙(𝑧|𝑥)
z
Decoder
𝑝𝜃(𝑥|𝑧)
𝑥′
(b) Variational AutoReconstructor (VAR)
Encoder
𝑞𝜙(𝑧|𝑚)
z
Recon-
structor
𝑝𝜃(𝑥|𝑧)
𝑥′
𝑚1′
…
𝑚2′
𝑚3′
𝑚1
…
𝑚2
𝑚3
𝑚1
𝑥
Loss
L
o
s
s
𝑓
Figure 2. Learning a latent space for 3D representations using
only images, without ground-truth 3D data. (a) Variational Au-
toencoders require groundtruth samples of high-dimensional vari-
ables x to learn a latent space; (b) We propose the Variational
AutoReconstructor, which learns a latent space for x using super-
vision from only their projections {m = f(x)}.
els the distribution over the latent space. The latent space
learns and compresses the 3D structure of Splatter Images
and reduces their high-dimensional parameters, enabling
much more computationally efficient training of the diffu-
sion model.
3.2. Learning a Latent Space of 3D Splatter-Images
from 2D Images
We propose a novel architecture, Variational AutoRecon-
structor, to learn the latent space in a self-supervised man-
ner from only images, over which a denoising diffusion
model is then trained. We discuss alternative methods that
did not work in our early experiments in the Sup. Mat. A.3.
Variational AutoEncoder (VAE)
The conventional latent
diffusion pipeline [53] involves learning a latent space with
a Variational Autoencoder (VAE) [35], as shown in Fig-
ure 2(a), where an encoder qϕ takes a ground-truth sample
X as input and maps it to a lower-dimensional latent distri-
bution qϕ(Z|X). A latent is sampled and decoded back to
X′ using a decoder pθ. A loss is computed between X and
X′ to ensure faithful reconstruction of the data, and this
loss is used to train both the encoder qϕ and the decoder
pθ. However, the conventional latent space learning is not
applicable in our case, because we do not have access to
the ground-truth Splatter Images X, but only to the images
m = f(X), which are projections of X.
Variational AutoReconstructor (VAR)
To address this
gap, we propose a new architecture called the Variational
AutoReconstructor (VAR), as shown in Figure 2 (b). This
4

<!-- page 5 -->
encodes a 2D image m = f(X)—a projection of X—into
a latent distribution qϕ(Z|m), and then reconstructs a sam-
ple back to a high-dimensional representation X′, which
corresponds to the unknown 3D Splatter Image represen-
tation. We supervise X′ using posed multi-view images
{mi}, each of which is a projection of X, available in
large-scale image datasets. The differentiable function f(·)
makes the entire pipeline end-to-end trainable.
Figure 3 gives an overview of the Variational AutoRe-
constructor training pipeline. We assume access to train-
ing datasets containing multiple image sequences capturing
different object instances or scenarios, such as CO3D [52]
and RealEstate10K [83]. Each training example comprises
a reference image Iref, a set of target images I = {Itgt
i
| i =
0, . . . , t}, and the corresponding camera poses and intrin-
sics. At training time, the reference image Iref is input to
the encoder, which produces a Gaussian distribution repre-
sented by a mean hµ and variance hσ. The latent code h is
sampled from the Gaussian distribution N(hµ, hσI) using
the reparameterization trick [35]:
h = hµ + ϵhσ,
where ϵ ∼N(0, I).
This sampled latent code is passed to the reconstructor to
obtain the Splatter Image representation corresponding to
the reference view. The predicted Splatter Image is then
converted into a Gaussian splat representation of the scene
and rendered into the target views using the known camera
intrinsics and poses for each view Ii in I, denoted as ˆIi. In
practice, we also render the reference view.
Preserving High-frequency Details with Skip connec-
tions
We observe that a Splatter Image directly recon-
structed from the low-dimensional latent space captures the
geometry but loses high-frequency texture details. To over-
come this, we introduce a skip connection [54], which al-
lows high-frequency information to flow directly from the
encoder’s first-layer features to the higher-resolution layers
of the reconstructor, as demonstrated in Figure 4. At test
time, we gain flexibility by modifying the weight of skip
connection features to control the faithfulness of the recon-
struction to the input images, potentially balancing the di-
versity of reconstructions. More results are in the supple-
mentary material section D.2.
Losses
The reconstruction loss Lrec for a single training
example is:
Lrec =
X
I∈I+
λ1||I −ˆI||2+λ2SSIM(I, ˆI)+λ3LPIPS(I, ˆI)
(1)
where I+ = {Iref} ∪I includes the reference and tar-
get images and λ∗are weights for the different loss terms.
SSIM [69] encourages similarities in structure and con-
trast.
LPIPS [81] matches neural network encodings of
image patches, which encourages photorealism. Following
VAE [35], we also use a KL divergence loss LKL on the
latent space z to regularize it to follow a prior Gaussian dis-
tribution:
LKL = KL (N(hµ, hσI), N(0, I)) .
(2)
3.3. Training the Latent Diffusion Model
Once the Variational AutoReconstructor is trained, we en-
code all training images into latent code samples. A denois-
ing diffusion model is then trained to learn the distribution
of these latent codes conditioned on the input. The overall
pipeline is illustrated in the supplement Fig. A1.
In general, we follow previous latent diffusion training
pipelines [33, 53]. The forward diffusion process for latent
code h at time t is
ht = √¯αth +
√
1 −¯αtϵ,
(3)
where ϵ ∼N(0, I) is the sampled Gaussian noise, ¯αt =
Qt
s=1 (1 −βs) is the noise variance and β1, β2, . . . , βT is
the variance schedule [25]. During training we minimize
the denoising loss:
Eh,ϵ∼N (0,I),t∼U(T )||ϵ −ϵθ(ht, ϕref, t)||2
2,
(4)
where ϵθ(ht, ϕref, t) is the denoising diffusion model that
takes the noised latent code ht, input image features ϕref
and time step t as input.
Conditioning Diffusion Models on Input Images
Con-
ventional diffusion models [53] take text prompts as condi-
tions, which are encoded into feature vectors. Instead, we
condition on the input reference image Iref, preserving the
image structure and correspondence, to generate a Splat-
ter Image. Following Marigold [33], we concatenate fea-
tures ϕref = Ffeat(Iref) extracted using a feature encoder
Ffeat(.) from the input image Iref with the noised latent
codes as input to the denoiser. There are several options for
the feature encoder Ffeat(·). We directly use the pre-trained
encoder from the Stable Diffusion VAE model [53] to ob-
tain the conditional features ϕref. As future work, we plan
to explore other embeddings such as CLIP [51] to condition
on text prompts.
Controlling Diversity with Classifier-Free Guidance
To
increase the diversity of samples for generation tasks, we
train with classifier-free guidance [53], where conditioning
features are zeroed out 20% of the time (akin to uncon-
ditional generation). This reduces overfitting to the scene
appearance visible in the conditioning view. During infer-
ence, it provides the option to increase the guidance weight
to generate more diverse samples to balance between faith-
fulness and diversity.
3.4. Diffusion Inference
During inference, our method takes a single image as input
and outputs a Gaussian splat represented as a Splatter Image
5

<!-- page 6 -->
Encoder
Latent space of
Splatter Images
Reconstructor
Reference Image 
(H x W x 3)
Renders
Ground truth
Sample
Skip connections
Targets
Targets
…
…
Reference
Reference
Gaussian splat
Predicted Splatter 
Image (H x W x MN)
Figure 3. Learning a latent space for Splatter Images. Our encoder predicts the parameters of a normal distribution over latents. We
reconstruct a sampled latent into H × W × MN Splatter Image representations. We render the Gaussian splats from the viewpoints of the
target training images and optimize reprojection losses between the rendered and ground-truth RGB images. Skip connections are critical
to preserving the high-frequency details of the predictions, as shown in Figure 4.
Connection
Input 
view
Novel 
view
GT
No skips
With skips
Figure. Skip Connections.
Our encoder-
decoder
Our encoder-decoder 
w/ skip connections
Figure 4.
Including skip connections helps preserve high-
frequency details from the input view in the AutoReconstructor,
improving the faithfulness of appearance.
(Fig. A2, in Sup. Mat). The input image is first encoded by
a feature encoder to generate conditioning features. A ran-
domly initialized latent code is concatenated with these fea-
tures and fed into the denoising diffusion model for 50 dif-
fusion steps. The resulting denoised latent code is then de-
coded into a Splatter Image using the Reconstructor trained
in the first stage. This Splatter Image is subsequently back-
projected to 3D to produce the final Gaussian splat.
Controlling Faithfulness and Diversity
As a generative
method, our model enables sampling multiple outputs by
denoising different randomly initialized latent codes. The
trade-off between fidelity to the input image and diversity
of the generated results can be controlled by adjusting the
weights of classifier-free guidance and skip connections.
4. Experiments
We introduce the basic experimental settings below and pro-
vide implementation details in the Supplementary Material.
4.1. Experimental Settings
Datasets
We train and evaluate our model on the challeng-
ing CO3D dataset [52] and the RealEstate10K dataset [83].
The CO3D dataset comprises 360-degree video captures of
various objects in both indoor and outdoor real-world con-
texts, with annotated camera poses obtained via Structure-
from-Motion (SfM) [56]. Following prior works [47, 61,
70], we train and evaluate on the most common categories,
Hydrants and Teddy Bears, which include 723 and 1329
scenes, respectively. The RealEstate10K dataset [83] con-
tains real-world videos of residential indoor and outdoor en-
vironments, with camera poses also derived using SfM. We
use the provided official training and testing splits.
Baselines
We compare our method with several state-
of-the-art baselines, including: (1) Original Splatter Im-
age [61], a forward method using Gaussian splatting, ap-
plied only on objects; (2) Splatter Image (Full Images)⋆,
which we train on full images including the background;
(3) PixelNeRF [78], a forward method based on NeRF;
(4) DFM [65], a diffusion-based NeRF model; (5) Ze-
roNVS [55], a novel view synthesis method; (6) SparseFu-
sion [84], which optimizes a scene using score distillation
for each view; and (7) LGM [64]: A large-scale object re-
construction model based on Gaussian splats. We introduce
the baselines in detail in the supplementary material.
Metrics
Following prior work [40, 61, 78], we report
PSNR and LPIPS scores, where PSNR quantifies pixel-
wise reconstruction fidelity and LPIPS measures percep-
tual similarity based on deep features between rendered and
ground-truth images. Following [27], we report FID [24]
and KID [1] scores for generative performance, which mea-
sure image similarity at the distribution level by compar-
ing statistics of deep image features over multiple images,
thereby evaluating visual realism. We further report scores
separately on Full Images, which computes metrics on all
pixels of reference views, and Objects Only, which com-
putes metrics only on object pixels following [61], for a fair
comparison with prior object-centric work.
4.2. Reconstruction Performance
CO3D Datasets
We present quantitative results on full
images in Table 2. The results show that our method out-
performs real-time baselines on most metrics.
Although
6

<!-- page 7 -->
Rendering
Methods
Time Cost / Sequence
Hydrants
Teddybears
Speed
Inference
Scene Render
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
Offline
PixelNeRF [78] †
5.0 sec
4.1 min
17.93
0.54
180.20
0.14
-
-
-
-
DFM [65] †
2 min
8 min
17.47
0.42
84.63
0.05
-
-
-
Splatter Image (Full Images) ⋆
49.4 ms
109.5 ms
17.37
0.492
155.4
0.127
17.44
0.478
102.8
0.069
Ours - AutoReconstructor ⋆
15.7 ms
29.1 ms
17.59
0.471
133.0
0.111
17.49
0.477
95.1
0.058
Ours - Diffusion single sample ⋆
2.7 sec
29.1 ms
17.40
0.473
133.8
0.114
16.77
0.480
97.8
0.061
Real-time
Ours - Diffusion 20-best oracle ⋆
54.0 sec
29.1 ms
17.74
0.466
132.3
0.114
17.08
0.474
96.6
0.062
Table 2. Quantitative results on full images from CO3D. We compare with other methods that also perform single-image-input novel
view synthesis on full images, rather than only on masked objects. Our model outperforms other real-time rendering baselines. †: cited
from [65]; ⋆: trained by us. The reported inference times account for the complete pipeline.
Figure 5. Qualitative results on the “Hydrants” category from the
CO3D dataset. We produce significantly sharper results than Pix-
elNeRF [78], and comparable or better performance on object ar-
eas compared to DFM [65], while being significantly faster. Com-
putation times are reported in Table 2.
DFM [65] outperforms our method on some metrics, it is
considerably slower due to its NeRF representation and per-
frame diffusion design. PixelNeRF [78] achieves slightly
higher PSNR than our method but performs significantly
worse in LPIPS, likely because it produces generic smooth
details in unobserved regions.
As shown in Fig. 5 and
Fig. 6, our results appear considerably sharper than those
of PixelNeRF and SplatterImage, and are comparable to
DFM, while being much faster to compute (3 seconds ver-
sus 10 minutes). We output real-world scenes with both
foreground and background, in comparison to object-only
models such as LGM [64]. We provide a detailed quantita-
tive evaluation of object reconstruction methods in Table D1
in the Supplementary Materials.
RealEstate10K Datasets
We present quantitative results
on the RealEstate10K dataset in Table 3. To fairly compare
with baselines that follow different evaluation settings, we
follow PixelNeRF [78], SparseFusion [84], and DFM [65]
by evaluating on 100 scenes at a resolution of 128×128. We
also follow ZeroNVS [55] to evaluate on all 6,473 scenes
and report results at both 128×128 resolution and 256×256,
with the latter obtained by upsampling our rendered images.
Our method outperforms SparseFusion, PixelNeRF, and Ze-
roNVS by a significant margin, as we generate a consis-
tent 3D representation. In contrast, SparseFusion and Ze-
roNVS suffer from view inconsistency due to their reliance
Input
Target
LGM
SplatterImage
(full image)
Ours
SplatterImage
Figure 6. Qualitative results on the “TeddyBears” category from
the CO3D dataset.
Our model produces sharper results with
higher-quality details compared to the baselines LGM [64] and
SplatterImage [61], especially in occluded areas.
Input
Target
DFM
Ours
Figure 7. Qualitative Results on the RealEstate10K Dataset.
Our method achieves comparable performance to DFM [65], a
diffusion-based NeRF model, and performs better in some chal-
lenging regions (highlighted with dotted boxes), while being sig-
nificantly faster at inference time.
on multi-view optimization, and the intensive computation.
We also achieve performance comparable to DFM while be-
ing significantly faster at inference time. We show qualita-
tive results compared to DFM in Figure 7, where we achieve
better performance in some challenging regions.
7

<!-- page 8 -->
Methods
Rendering speed
# Scenes
Resolution
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
pixelNeRF [78] †
Offline
100
128
-
-
195.4
0.14
SparseFusion [84] †
Offline
100
128
-
-
99.44
0.04
DFM [65]
Offline
100
128
-
-
42.84
0.01
Ours Diffusion
Real-time
100
128
19.9
0.238
49.83
0.01
ZeroNVS [55]
Offline
6,473
256
13.5
0.414
-
-
Ours Diffusion
Real-time
6,473
128
20.2
0.253
15.53
0.01
Ours Diffusion
Real-time
6,473
256
19.6
0.365
33.44
0.03
Table 3. RealEstate10K Results. We report the number of test scenes and resolution following the settings used in prior work to ensure
fair comparison. The baseline results are cited from the corresponding published papers. †: cited from [65]; “–”: Results are not reported.
Source
Input
1st Sample
2nd Sample
Input View
Novel View
Source
Input
Source
Input
Figure 8. 3D Generative Performance.
Our diffusion model
demonstrates the ability to (1) sample diverse outputs in ambigu-
ous situations, and (2) fill in missing areas using 3D priors learned
from large datasets with multi-view consistency. Note the model
is trained purely from only 2D images.
4.3. Generative Performance
Compared with regression-based baselines, our method can
generate diverse samples with varying color, structure, and
style for unobserved and uncertain parts, as shown in Fig-
ure 1. We demonstrate this ability in detail below.
Qualitative Results
As shown in Figure 8, our diffusion
model is able to sample diverse results in ambiguous situa-
tions where parts of objects are occluded with black boxes.
The filled-in areas exhibit reasonable 3D structure consis-
tent with the visible parts and remain consistent as part
of a holistic 3D representation when viewed from novel
perspectives. This demonstrates that the model effectively
learns 3D priors purely from 2D images. We further demon-
strate this in Figure D2 in the Supplementary Materials,
where, compared with DFM, we are able to control the di-
versity in the occluded back of objects, showing increas-
ingly strong differences to balance fidelity to the input im-
age against output diversity.
Quantitative: k-best oracle evaluation
Generative mod-
els are capable of producing multiple possible predictions.
Oracle k-best evaluation [3, 21, 23] allows the model to pro-
duce k = 20 samples for each input, and the score of the
best sample is used to evaluate that input image. This metric
shows an upper bound reconstruction performance, assum-
Encoder-Reconstructor Variants
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
Splatter Image
17.37
0.492
155.4
0.127
No skip connections
16.63
0.551
167.9
0.143
Skips from first 2 layers
16.86
0.554
163.3
0.135
One Gaussian splat per pixel
17.43
0.491
140.1
0.115
No variational sampling
8.30
0.726
290.1
0.293
No SSIM
17.34
0.505
181.1
0.162
No LPIPS
17.51
0.502
149.7
0.118
No Color Augmentation
17.20
0.486
132.4
0.108
Our AutoReconstructor
17.59
0.471
133.0
0.111
Table 4. Ablation studies on the Hydrants scenes from the CO3D
dataset evaluated on Full Images.
ing some manual effort can be made to select the best result.
It also reflects the generative model’s ability to capture the
distribution and produce diverse samples. The results of our
diffusion model on the CO3D datasets are shown in Table 2
and the Sup. Mat.. We observe an improvement in the re-
ported metrics compared to single samples, which demon-
strates that our model generates diverse outputs rather than
collapsing to a single mode.
4.4. Ablation Studies
We ablate the key modules and hyperparameters of our Au-
toReconstructor to validate our contributions in Table 4.
Our method scores highest or second highest across all met-
rics, validating our design decisions. The “No Augmenta-
tion” row shows that our augmentations boost PSNR and
LPIPS at the expense of a small drop in FID and KID scores.
More discussions are given in the Supplementary Materials.
5. Conclusion
We presented a training pipeline for a denoising diffusion
model that reconstructs complete 3D scenes using Gaus-
sian splats from a single RGB image as input. By propos-
ing a Variational AutoReconstructor architecture, we are
able to efficiently learn a latent space for Splatter Im-
ages using only 2D images, from which a denoising dif-
fusion model can be trained by conditioning on input im-
age features. Experiments on the CO3D and RealEstate10K
datasets demonstrate that our approach achieves state-of-
the-art results in scene reconstruction, particularly in oc-
cluded regions. Furthermore, our method enables diverse
sampling of Gaussian splats with controllable trade-offs be-
tween faithfulness to the input and generative diversity.
8

<!-- page 9 -->
References
[1] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and
Arthur Gretton. Demystifying MMD GANs. In ICLR, 2018.
6
[2] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun
Jampani.
SF3D: Stable fast 3D mesh reconstruction with
uv-unwrapping and illumination disentanglement. In CVPR,
2024. 2, 3
[3] Mai Bui, Tolga Birdal, Haowen Deng, Shadi Albarqouni,
Leonidas Guibas, Slobodan Ilic, and Nassir Navab. 6D cam-
era relocalization in ambiguous scenes via continuous multi-
modal inference. In ECCV, 2020. 8
[4] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and
Gordon Wetzstein. pi-GAN: Periodic implicit generative ad-
versarial networks for 3D-aware image synthesis. In CVPR,
2021. 16
[5] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki
Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo,
Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero
Karras, and Gordon Wetzstein. Efficient geometry-aware 3D
generative adversarial networks. In CVPR, 2022. 16
[6] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexan-
der W. Bergman, Jeong Joon Park, Axel Levy, Miika Ait-
tala, Shalini De Mello, Tero Karras, and Gordon Wetzstein.
GeNVS: Generative novel view synthesis with 3D-aware dif-
fusion models. In arXiv, 2023. 2, 3, 12
[7] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelSplat: 3D gaussian splats from im-
age pairs for scalable generalizable 3D reconstruction. In
CVPR, 2024. 1, 2, 3, 12
[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. MVSplat: Efficient 3D gaussian splatting from sparse
multi-view images. In ECCV, 2024. 1, 2, 3
[9] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu.
Text-to-3D using gaussian splatting. In CVPR, 2024. 2, 3
[10] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
CVPR, 2017. 4
[11] Angela Dai, Charles Ruizhongtai Qi, and Matthias Nießner.
Shape completion using 3D-encoder-predictor cnns and
shape synthesis. In CVPR, 2017. 3
[12] Angela Dai, Daniel Ritchie, Martin Bokeloh, Scott Reed,
J¨urgen Sturm, and Matthias Nießner. ScanComplete: Large-
scale scene completion and semantic segmentation for 3D
scans. In CVPR, 2018.
[13] Angela Dai, Christian Diller, and Matthias Nießner. SG-NN:
Sparse generative neural networks for self-supervised scene
completion of RGB-D scans. In CVPR, 2020. 3
[14] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
A universe of annotated 3D objects. In CVPR, 2023. 2
[15] Xin Fei, Wenzhao Zheng, Yueqi Duan, Wei Zhan, Masayoshi
Tomizuka, Kurt Keutzer, and Jiwen Lu. PixelGaussian: gen-
eralizable 3d gaussian reconstruction from arbitrary views.
arXiv preprint arXiv:2410.18979, 2024. 3
[16] Qijun Feng, Zhen Xing, Zuxuan Wu, and Yu-Gang Jiang.
FDGaussian: Fast gaussian splatting from single image via
geometric-aware diffusion model. arXiv, 2024. 3
[17] Juan D Galvis, Xingxing Zuo, Simon Schaefer, and Stefan
Leutengger. SC-Diff: 3D shape completion with latent dif-
fusion models. arXiv, 2024. 3
[18] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen,
Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja
Fidler. GET3D: A generative model of high quality 3D tex-
tured shapes learned from images. In NeurIPS, 2022. 16
[19] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur
Brussee,
Ricardo
Martin-Brualla,
Pratul
Srinivasan,
Jonathan T Barron, and Ben Poole. Cat3D: Create anything
in 3D with multi-view diffusion models. In NeurIPS, 2024.
3
[20] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind,
Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi.
NerfDiff: Single-image view synthesis with nerf-guided dis-
tillation from 3D-aware diffusion. In ICML, 2023. 2, 3
[21] Abner Guzman-Rivera, Dhruv Batra, and Pushmeet Kohli.
Multiple choice learning:
Learning to produce multiple
structured outputs. NeurIPS, 25, 2012. 8
[22] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang.
CameraCtrl: En-
abling camera control for video diffusion models. In ICLR,
2025. 2
[23] Paul Henderson, Melonie de Almeida, Daniela Ivanova, and
Titas Anciukeviˇcius. Sampling 3D gaussian scenes in sec-
onds with latent diffusion models. arXiv, 2024. 2, 3, 8, 14
[24] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. GANs trained by
a two time-scale update rule converge to a local nash equi-
librium. NeurIPS, 2017. 6
[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models. NeurIPS, 2020. 2, 3, 4, 5
[26] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang,
Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben
Poole, Mohammad Norouzi, David J Fleet, and Tim Sali-
mans. Imagen video: High definition video generation with
diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2,
4
[27] Lukas H¨ollein, Aljaˇz Boˇziˇc, Norman M¨uller, David Novotny,
Hung-Yu Tseng, Christian Richardt, Michael Zollh¨ofer, and
Matthias Nießner. ViewDiff: 3D-consistent image genera-
tion with text-to-image models. In CVPR, 2024. 2, 3, 6
[28] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han,
Jiaolong Yang, Chong Luo, and Seungryong Kim. PF3plat:
Pose-free feed-forward 3d gaussian splatting. arXiv preprint
arXiv:2410.22128, 2024. 3
[29] Zixuan Huang, Stefan Stojanov, Anh Thai, Varun Jampani,
and James M Rehg. ZeroShape: Regression-based zero-shot
shape reconstruction. In CVPR, 2024. 3
[30] Wonbong Jang and Lourdes Agapito. NViST: In the wild
new view synthesis from a single image with transformers.
In CVPR, 2024. 3
9

<!-- page 10 -->
[31] Heewoo Jun and Alex Nichol. Shap-E: Generating condi-
tional 3D implicit functions. arXiv, 2023. 2, 3, 4
[32] Animesh Karnewar, Andrea Vedaldi, David Novotny, and
Niloy Mitra. HoloDiffusion: Training a 3D diffusion model
using 2D images. In CVPR, 2023. 16
[33] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Met-
zger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-
ing diffusion-based image generators for monocular depth
estimation. In CVPR, 2024. 5, 13
[34] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
Radiance Field Rendering. ToG, 2023. 1, 3, 12
[35] Diederik P Kingma and Max Welling. Auto-encoding varia-
tional bayes. In ICLR, 2014. 4, 5
[36] Xinhai Li, Huaibin Wang, and Kuo-Kun Tseng.
Gaus-
sianDiffusion: 3D gaussian splatting for denoising diffusion
probabilistic models with structured noise. arXiv, 2023. 2, 3
[37] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian,
Sergei Korolev, Demetri Terzopoulos, Konstantinos N Pla-
taniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navi-
gating 3d scenes from a single image. In CVPR, 2025. 3
[38] Ziwei Liao, Binbin Xu, and Steven L Waslander. Toward
general object-level mapping from sparse views with 3D dif-
fusion priors. In CoRL, 2024. 3
[39] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li,
and Yadong Mu.
Diffsplat: Repurposing image diffusion
models for scalable gaussian splat generation.
In ICLR,
2025. 3, 14
[40] Kai-En Lin, Yen-Chen Lin, Wei-Sheng Lai, Tsung-Yi Lin,
Yi-Chang Shih, and Ravi Ramamoorthi. Vision transformer
for nerf-based view synthesis from a single input image. In
WACV, 2023. 6
[41] Jinpeng Liu, Jiale Xu, Weihao Cheng, Yiming Gao, Xintao
Wang, Ying Shan, and Yansong Tang.
NovelGS: Consis-
tent novel-view denoising via large gaussian reconstruction
model. arXiv preprint arXiv:2411.16779, 2024. 2
[42] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tok-
makov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3:
Zero-shot one image to 3D object. In ICCV, 2023. 3
[43] Yifan Liu, Keyu Fan, Weihao Yu, Chenxin Li, Hao Lu, and
Yixuan Yuan. MonoSplat: Generalizable 3d gaussian splat-
ting from monocular depth foundation models. In CVPR,
2025. 3
[44] Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo,
Tiejun Huang, Lulu Tang, and Xinlong Wang. You see it,
you got it: Learning 3d creation on pose-free videos at scale.
In CVPR, pages 2016–2029, 2025. 3
[45] Xuyi Meng, Chen Wang, Jiahui Lei, Kostas Daniilidis, Ji-
atao Gu, and Lingjie Liu. Zero-1-to-G: Taming pretrained
2d diffusion model for direct 3d generation. arXiv preprint
arXiv:2501.05427, 2025. 2, 3, 14
[46] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2, 3
[47] Yuxuan Mu, Xinxin Zuo, Chuan Guo, Yilin Wang, Juwei Lu,
Xiaofeng Wu, Songcen Xu, Peng Dai, Youliang Yan, and Li
Cheng. GSD: View-guided gaussian splatting diffusion for
3D reconstruction. In ECCV, 2024. 3, 6
[48] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove. Deepsdf: Learning con-
tinuous signed distance functions for shape representation.
In CVPR, 2019. 12
[49] Chensheng Peng, Ido Sobol, Masayoshi Tomizuka, Kurt
Keutzer, Chenfeng Xu, and Or Litany. A lesson in splats:
Teacher-guided diffusion for 3d gaussian splats generation
with 2d supervision. In ICCV, 2025. 3
[50] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. DreamFusion: Text-to-3d using 2d diffusion. In ICLR,
2023. 3, 14
[51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision. In ICML, 2021.
5
[52] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,
Luca Sbordone, Patrick Labatut, and David Novotny. Com-
mon objects in 3D: Large-scale learning and evaluation of
real-life 3D category reconstruction. In ICCV, 2021. 2, 4, 5,
6
[53] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, 2022. 2, 3, 4,
5, 12
[54] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net:
Convolutional networks for biomedical image segmentation.
In MICCAI, 2015. 5
[55] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann,
Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry La-
gun, Li Fei-Fei, Deqing Sun, and Jiajun Wu.
ZeroNVS:
Zero-shot 360-degree view synthesis from a single image.
In CVPR, 2024. 3, 6, 7, 8, 14
[56] Johannes
Lutz
Sch¨onberger
and
Jan-Michael
Frahm.
Structure-from-motion revisited. In CVPR, 2016. 6
[57] Katja Schwarz, Norman Mueller, and Peter Kontschieder.
Generative gaussian splatting: Generating 3d scenes with
video diffusion priors. In ICCV, 2025. 3
[58] Qiuhong Shen, Zike Wu, Xuanyu Yi, Pan Zhou, Hanwang
Zhang, Shuicheng Yan, and Xinchao Wang. Gamba: Marry
gaussian splatting with mamba for single-view 3d recon-
struction. TPAMI, 2025. 2
[59] Shuran Song, Fisher Yu, Andy Zeng, Angel X Chang, Mano-
lis Savva, and Thomas Funkhouser. Semantic scene comple-
tion from a single depth image. In CVPR, 2017. 3
[60] Stanislaw Szymanowicz,
Eldar Insafutdinov,
Chuanxia
Zheng, Dylan Campbell, Jo˜ao F Henriques, Christian Rup-
precht, and Andrea Vedaldi. Flash3D: Feed-forward gener-
alisable 3D scene reconstruction from a single image. arXiv,
2024. 1, 2, 3, 12
[61] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea
Vedaldi. Splatter image: Ultra-fast single-view 3D recon-
struction. In CVPR, 2024. 1, 2, 3, 4, 6, 7, 12, 13, 14, 15,
16
10

<!-- page 11 -->
[62] Stanislaw Szymanowicz, Jason Y Zhang, Pratul Srinivasan,
Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo
Martin-Brualla, Jonathan T Barron, and Philipp Henzler.
Bolt3d: Generating 3d scenes in seconds. In ICCV, 2025.
2, 3, 14
[63] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. DreamGaussian: Generative gaussian splatting for ef-
ficient 3D content creation. In ICLR, 2023. 2, 3
[64] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. LGM: Large multi-view gaus-
sian model for high-resolution 3d content creation. In ECCV,
2024. 6, 7, 14, 15
[65] Ayush Tewari, Tianwei Yin, George Cazenavette, Semon
Rezchikov, Josh Tenenbaum, Fr´edo Durand, Bill Freeman,
and Vincent Sitzmann. Diffusion with forward models: Solv-
ing stochastic inverse problems without direct supervision.
NeurIPS, 2023. 2, 3, 6, 7, 8, 12, 14, 15, 16
[66] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. VGGT:
Visual geometry grounded transformer. In CVPR, 2025. 3
[67] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3d
vision made easy. In CVPR, 2024. 3
[68] Weiyue Wang, Qiangui Huang, Suya You, Chao Yang, and
Ulrich Neumann. Shape inpainting using 3D generative ad-
versarial network and recurrent convolutional networks. In
ICCV, 2017. 3
[69] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. IEEE TIP, 2004. 5
[70] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and
Jan Eric Lenssen.
latentSplat: Autoencoding variational
gaussians for fast generalizable 3D reconstruction. In ECCV,
2024. 2, 3, 6
[71] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Lin-
guang Zhang, Xiaoou Tang, and Jianxiong Xiao.
3D
ShapeNets: A deep representation for volumetric shapes. In
CVPR, 2015. 3
[72] Tiange Xiang, Kai Li, Chengjiang Long, Christian H¨ane,
Peihong Guo, Scott Delp, Ehsan Adeli, and Li Fei-Fei. Re-
purposing 2d diffusion models with gaussian atlas for 3d
generation. In ICCV, 2025. 2, 3, 4, 14
[73] Dejia Xu, Ye Yuan, Morteza Mardani, Sifei Liu, Jiaming
Song, Zhangyang Wang, and Arash Vahdat. AGG: Amor-
tized generative 3D gaussians for single image to 3D. TMLR,
2024. 2, 3
[74] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
DepthSplat: Connecting gaussian splatting and depth.
In
CVPR, 2025. 3
[75] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen,
Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wet-
zstein. GRM: Large gaussian reconstruction model for ef-
ficient 3d reconstruction and generation. In ECCV, 2024. 2,
14, 15
[76] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Ji-
ahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein,
Zexiang Xu, and Kai Zhang.
DMV3D: Denoising multi-
view diffusion using 3d large reconstruction model. In ICLR,
2024. 2
[77] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi
Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang
Wang. Gaussiandreamer: Fast generation from text to 3D
gaussians by bridging 2D and 3D diffusion models.
In
CVPR, 2024. 2, 3
[78] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelNeRF: Neural radiance fields from one or few images.
In CVPR, 2021. 3, 6, 7, 8, 14, 16
[79] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu,
and Yansong Tang. GeoLRM: Geometry-aware large recon-
struction model for high-quality 3D gaussian generation. In
NeurIPS, 2024. 2, 3
[80] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: Large recon-
struction model for 3d gaussian splatting. In ECCV, 2024.
14, 15
[81] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 5
[82] Shengjun Zhang, Jinzhao Li, Xin Fei, Hao Liu, and Yueqi
Duan. Scene splatter: Momentum 3d scene generation from
single image with video diffusion model. In CVPR, 2025. 3
[83] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magnification: Learning view syn-
thesis using multiplane images. SIGGRAPH, 2018. 2, 4, 5,
6
[84] Zhizhuo Zhou and Shubham Tulsiani. SparseFusion: Dis-
tilling view-conditioned diffusion for 3D reconstruction. In
CVPR, 2023. 2, 3, 6, 7, 8, 14
11

<!-- page 12 -->
A. Methods Details
A.1. Gaussian Splatting Representation
We represent scenes as a set of Gaussian splats [34], and
follow Splatter Image [61] to represent Gaussians inside a
frustum. In this section, we first introduce the definitions
of Gaussian splatting, then describe how to represent Gaus-
sians within an image frustum, and how a neural network is
used to predict them.
Gaussian Splatting
Following [34], a 3D scene is param-
eterized as a set of Gaussian splats, each of which is associ-
ated with an opacity σ, 3D position µ, scale and rotation pa-
rameters of the covariance matrix, and color c modeled with
spherical harmonics. Through differentiable rendering, all
Gaussians can be projected onto image planes to form RGB
images. Typically, the parameters of the Gaussian splats are
optimized to match the renderings with a set of posed RGB
training images, in order to reconstruct a single scene at a
time.
Splatter Image
The “Splatter Image” [61] formulation
predicts a Gaussian for each pixel ray of a single H × W
input RGB image I, representing both observed and unob-
served scene content inside the image frustum. This results
in a parameter matrix of shape H × W × N, where N is
the number of parameters per Gaussian splat. Specifically,
the Gaussian mean is parameterized by the pixel depth and
a 3D offset. Each Gaussian thus contains a total of N = 15
scalar values: 1 (opacity), 1 (depth), 3 (offset), 3 (scale), 4
(rotation quaternion), and 3 (color). We use only the DC co-
efficients for spherical harmonics. To further encourage the
network to model occluded surfaces, follow-up work [60]
extends the parameter matrix to H × W × MN, where M
denotes the number of Gaussian splats predicted per pixel,
i.e., the number of Gaussian layers.
Training a Network to Predict Splatter Image
Follow-
ing [61], a neural network can be trained to directly pre-
dict the Gaussian splat parameters of a Splatter Image via a
feed-forward process. Given a single input image, the net-
work (e.g., a U-Net) outputs the complete parameter matrix
corresponding to the Gaussian splats, while preserving the
spatial structure of the image. However, this network is con-
ventionally regression-based and can only model a single
possibility [7, 60, 61]. In contrast, we extend this formula-
tion to a distribution using a diffusion model, enabling the
modeling of multiple plausible outputs.
A.2. Diffusion Pipelines
To better illustrate our diffusion architecture, we present fig-
ures for diffusion training over the learned latent space in
Figure A1, and the diffusion inference procedure in Fig-
ure A2.
A.3. Alternative Approaches for Learning a Latent
Space for Splatter Images
Our Variational AutoReconstructor learns a latent space of
Splatter Images in a simple and effective way. We further
describe alternative approaches we explored in the early
stages of the project that did not yield satisfactory results.
Reconstructor-only without the Encoder
Since we only
need latents for diffusion training, we try removing the en-
coder. We assign each image an optimizable latent code and
optimize both the latent code and the reconstructor weights
to generate Splatter Images, similar to the autodecoder ap-
proach for SDF representations [48]. However, we observe
that this leads to low-quality reconstructions because the
optimization gets stuck in the early stages and fails to cap-
ture the geometry correctly. We assume the encoder is im-
portant for initializing the latents or capturing correlations
that are crucial to the input images.
Pre-optimizing 3D Gaussian splats and projecting into
Splatter Images
This approach optimizes 3D Gaussian
splats for each scene using all available images, then
projects the Gaussians onto each image plane to form Splat-
ter Images. However, rasterization errors occur when pro-
jecting Gaussians to achieve pixel alignment, and the 3D
Gaussians are not distributed uniformly across the images.
The resulting Splatter Images are extremely sparse, with
80% of the pixels lacking a corresponding Gaussian. We
find it difficult to train a VAE to encode each modalities of
such sparse splatter images and reconstruct them with neg-
ligible error, which breaks the geometric and texture con-
sistency.
B. Implementation Details
B.1. Network architecture
We implement the AutoReconstructor architecture follow-
ing the VAE from Stable Diffusion [53]. The encoder re-
mains unchanged and predicts a distribution over the latent
representation at 1
8 the resolution of the input images, with
4 + 4 channels (4 for hµ and 4 for hσ), from which a 4-
channel latent can be sampled. We extend the decoder into
a reconstructor by modifying its output layers: instead of
producing 3 channels for RGB images, it now outputs MN
channels to represent Gaussians for each pixel. We predict
M = 2 Gaussians for each pixel, and set N = 15 as de-
scribed in Section A.1. We train the AutoReconstructor by
fine-tuning it based on the original VAE parameters. We fol-
low the same denoiser architecture as Stable Diffusion [53]
and train it from scratch.
B.2. Data Preprocessing
For a fair comparison with prior work [6, 61, 65], all im-
ages in the datasets are cropped around the principal point
12

<!-- page 13 -->
Stable 
diffusion
encoder
Denoising network
Random noise
(H/8 x W/8 x 4)
Reference image conditioning
(H/8 x W/8 x 4)
🔒
🔒
+
Latent 
encoder
Reference Image 
(H x W x 3)
Concat
Loss
Figure A1. Training a denoising diffusion model over the learned latent space. The network learns to convert a corrupted noisy latent
code back to a ground-truth latent code representing a Splatter Image, by conditioning on a single input image.
Latent
reconstructor
Stable 
diffusion
encoder
Reference Image 
(H x W x 3)
Skip connections
Predicted Splatter 
Image (H x W x MN)
Denoising network
Random noise
(H/8 x W/8 x 4)
Reference image 
conditioning
(H/8 x W/8 x 4)
Concat
Repeat T times
Latent 
encoder
🔒
🔒
🔒
🔒
Figure A2. Diffusion Inference Pipeline. We first compute input image features using the Stable Diffusion encoder. A random latent code
and the input image features are concatenated and passed through R steps of the denoising diffusion process. The denoised latent code,
together with skip connections from the encoder, is then passed through the reconstructor to produce a Splatter Image representation. This
representation is subsequently backprojected to create 3D Gaussian splats.
of the camera (using the largest possible crop) and rescaled
to 128 × 128 pixels before training.
Scaling of scenes
We follow the approach outlined in the
Splatter Image [61] supplementary material (Section B.2)
to address varying scene scales. In practice, this involves
rescaling the predicted Gaussian depths (at both training
and test time) to lie between znear and zfar, where znear and
zfar are computed using the known depth to the object in the
scene.
B.3. Training Details
Encoder-Reconstructor
We first train the AutoRecon-
structor. We use batch size 32 and Adam optimizer. We
train for 6000 epochs with L2 and SSIM losses (λ1 =
0.8, λ2 = 0.2) and learning rate of 5e-5. Fine-tuning is
done with another 8000 epochs with L2, SSIM and LPIPS
losses (λ1 = 0.8, λ2 = 0.2, λ3 = 0.01) and learning rate
of 5e-6. Using LPIPS loss only during finetuning is a com-
mon practice [61] to prevent the LPIPS loss from disrupting
geometric structure learning in the early stages of training.
We apply the same random color augmentation to both the
reference and target views to increase the diversity of colors
that the latent codes can represent.
Skip connection
When using the skip connection, we ob-
served that the model can “cheat” by retaining most of the
information about the scene in the skip connection features
instead of latent codes. To limit this effect, during training
we randomly zero out all the values of the skip features, en-
couraging the network to encapsulate scene information in
the latent codes.
Denoising Diffusion Model
We train with Adam Opti-
mizer and batch size 32. We train for 1, 000K steps and a
probability of 20% for unconditional generation (classifier-
free guidance). We finetune for 100K steps with a probabil-
ity of 50% for randomly masking the input images. Learn-
ing rate is set to 1e-4 with exponential learning rate sched-
uler, similar to Marigold [33]. We also sample latent codes
from the encoder network as ground truth latent codes dur-
ing diffusion trainings.
C. Experiments Details
C.1. Evaluation Protocol
After training on the CO3D or RealEstate10K datasets, we
evaluate on the unseen test sets. For each sequence, we
use the reference view to predict Gaussian splats and ren-
der them from both the reference viewpoint and several
novel target viewpoints. We then compare the rendered im-
ages with the corresponding ground-truth RGB images cap-
tured from the same viewpoints. We design two settings for
comprehensive evaluations: Full Images and Objects Only.
Full Images, computes the metrics on all pixels of reference
13

<!-- page 14 -->
Methods
Open-source Code
Baselines Used
DFM [65]
✓
✓
SplatterImage [61]
✓
✓
pixelNeRF [78]
✓
✓
SparseFusion [84]
✓
✓
ZeroNVS [55]
✓
✓
LGM [64]
✓
✓
Diffsplat [39] *
✓
×
SampleSplat [23]
×
×
Bolt3D [62]
×
×
GRM [75]
×
×
GS-LRM [80] †
×
×
Zero-1-to-G [45]
×
×
Xiang et al. [72]
×
×
Table C1.
Availability of an open-source implementation
for different related methods. We choose our baselines from
the closest competitors that provide open-source code and sup-
port scene-level reconstruction. *: Diffsplat is an object-centric
method; †: Only an incomplete unofficial version is available.
Code availability checked on August 15, 2025.
views. The Objects Only evaluation computes the metrics
only on object pixels, following the protocol of Splatter Im-
age [61], and is reported here for a fair comparison with
prior work. Similar to our method, DFM [65] and Splat-
ter Image (Full Images)⋆always predict background pixels.
For the Objects Only evaluation, we use the ground truth
object mask to set the background pixels of both the pre-
dicted and ground-truth images to black before computing
the metrics.
C.2. Baselines
Open-source methods
We present the open-source status
of the literature methods in Table C1, from which we select
our main competitors in the experiments for evaluation.
Chosen Baselines
We introduce in detail the baselines to
which we compare below:
(1) Splatter Image [61] is the closest model to ours, em-
ploying a regression-based formulation. We use the pub-
licly available code and models for our experiments. This
model was trained only on foreground objects by masking
out the background.
(2) Splatter Image (Full Images)⋆: To obtain a base-
line more comparable to our method, which models the full
scene, we introduce a modified and retrained version of the
original Splatter Image [61] that predicts both objects and
background by training on full images. The ⋆indicates that
this model was trained by us. As a regression-based model,
it produces uni-modal outputs and does not support sam-
pling.
(3) PixelNeRF [78] is a NeRF-based regression method
for novel view prediction conditioned on one or more im-
ages.
(4) DFM [65] is a diffusion-based method built on NeRF
that can generate high-quality renderings. However, the av-
erage scene takes approximately 10 minutes to reconstruct
and render. We use the official codebase to produce outputs
for evaluation. It includes models trained on the Hydrants
and RealEstate10K datasets.
(5) ZeroNVS [55] employs diffusion and Score Distilla-
tion Sampling with anchoring to optimize a NeRF, but it
faces 3D consistency issues such as Multi-Face Janus arti-
facts [50].
(6) SparseFusion [84] outputs novel views of 3D scenes
using a generative model to provide score distillation for
each view, requiring intensive computation during opti-
mization.
We compare our method with ZeroNVS and
SparseFusion on the RealEstate10K dataset.
(7) LGM [64]: A large-scale object reconstruction model
based on Gaussian splats. We demonstrate our object re-
construction performance, while additionally reconstruct-
ing background regions.
C.3. Ablation Studies Analysis
The results of the ablation study are shown in Table 4, and
we discuss them in detail below:
Skip connections
No skip connections: We ablate the
variant that does not use skip connections, which also qual-
itatively shows degradation in Fig. 4. Skips from first 2 lay-
ers: We further ablate by using only two skip connections
from the first and second layers of the encoder, which may
reduce the scene information learned in the latent codes.
Number of Gaussians per Pixel
One Gaussian splat per
pixel: Our method trained to predict only a single Gaussian
splat per pixel, which lacks sufficient Gaussians to represent
scene details.
Variational Sampling
No variational sampling: Disables
variational sampling and directly uses the 4-dimensional
code regressed by the encoder. We observe that training
gets stuck and does not improve in the early stages.
Training Losses
We ablate training with No SSIM or No
LPIPS, which results in critical performance drops.
Data Augmentation
No Color Augmentation: Does not
apply color augmentation to the input and target images dur-
ing training.
C.4. 3D Generative Experiments
We present additional details of the 3D generative experi-
ments in Figure 8. We train the diffusion model using ran-
dom masking, while keeping the Variational AutoRecon-
structor model fixed. The masked input image is passed
through the diffusion model to generate a latent code, which
is then reconstructed into a Splatter Image using the pre-
trained AutoReconstructor. To focus on the generative ca-
pacity of the diffusion model, we disable skip connections
during these experiments. Our goal is to demonstrate the
14

<!-- page 15 -->
GT
LGM
Ours
Condition
Novel Views
Condition
Novel Views
Figure D1. Comparison to LGM [64], a large-scale object recon-
struction model using Gaussian splatting, evaluated on the CO3D
dataset. Our method outputs high-quality scene-level results with
correct geometry and textures, even in occluded areas.
model’s ability to produce diverse outputs from its latent
space.
To further enhance high-frequency details, tech-
niques such as finetuning the AutoReconstructor with ran-
dom masking could be explored, which we leave as future
work.
C.5. Experiments Notes
Cited Baseline Scores
In the quantitative results, we cite
the reported scores from the papers of the baselines and en-
sure that the settings are consistent for a fair comparison.
The main baseline scores are reported in Splatter Image [61]
and DFM [65]. Some combinations of training data and
methods are not publicly available for citation or evalua-
tion, e.g., the performance of the DFM model trained on
TeddyBears is not reported, and the corresponding model
is not publicly available. Thus, we leave the corresponding
entries blank in the table.
D. Additional Experiments and Analysis
D.1. Objects-level Evaluations
Results for Objects Only in Table D1 provide a broader
comparison to methods focused solely on objects, where
our method outperforms almost all of them. Splatter Image
slightly outperforms on TeddyBear LPIPS and KID metrics,
partly due to its object-oriented design, while our approach
focuses on the whole scene. We present a qualitative evalua-
tion on TeddyBears in Figure 6, where our method achieves
more sensible reconstruction, especially in occluded areas.
We further compare our method with representative
large-scale object reconstruction approaches. Some meth-
ods [75, 80] do not have official code, pretrained models,
or output results available for comparison, as also noted in
Table C1. We qualitatively compare with LGM [64] in Fig-
ure 6 and Figure D1, where our method demonstrates su-
perior reconstruction quality, especially in preserving back-
ground context.
Ours vs D
(Portrait)
Input
Target
Samples from DFM
Samples from Ours
Figure D2. Diverse samples from our diffusion model on Hy-
drants in the CO3D dataset. We intentionally show three sam-
ples with increasing diversity from left to right by controlling the
classifier-free guidance and skip connection weights. The samples
transition from faithful reconstruction of the input image to di-
verse generations exhibiting variations in texture, shape, and style.
In contrast, the baseline DFM [65] exhibits only small texture vari-
ations on object areas.
D.2. Additional Qualitative Results for Diverse
Samplings
We show more diverse sampling results on Hydrants in Fig-
ure D2. For each input image, we show three generated
reconstructions rendered from two different target views.
From left to right, the diversity increases by manually in-
creasing the classifier-free guidance weight and decreasing
the skip connection weight. Specifically, the three samples
are generated with skip connection weights of (1.0, 0.5, 0.0)
and classifier-free guidance weights of (0.2, 0.5, 0.5), re-
spectively. This controllability allows tailoring the output to
specific application needs, favoring high fidelity for recon-
struction tasks, and greater diversity for applications like
data augmentation or creative generation.
In the figures, we also compare with DFM [65], which
is a denoising diffusion model based on the NeRF repre-
sentation, and show that our samples are much more di-
verse, especially in the occluded object areas. DFM shows
small variations in the texture of objects, for example, some
rust, while ours is able to add a valve structure and change
the color. Furthermore, DFM denoises 2D images view by
view, whereas ours outputs the Gaussian splats in a single
diffusion process and is much faster.
15

<!-- page 16 -->
Methods
Hydrants
Teddybears
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
PSNR ↑
LPIPS ↓
FID ↓
KID ↓
pi-GAN [4] †
-
-
92.1
0.080
-
-
125.8
0.118
EG3D [5] †
-
-
229.5
0.253
-
-
236.1
0.239
GET3D [18] †
-
-
303.3
0.380
-
-
244.5
0.280
HoloDiffusion (no bootstrap) [32] †
-
-
277.9
0.305
-
-
222.1
0.217
HoloDiffusion [32] †
-
-
100.5
0.079
-
-
109.2
0.106
pixelNeRF [78] †
21.76
0.203
-
-
19.38
0.290
-
-
Splatter Image (Objects Only) [61] †
21.80
0.150
-
-
19.44
0.231
-
-
Splatter Image (Full Images) ⋆
24.00
0.141
75.22
0.045
22.80
0.172
44.93
0.026
DFM [65]
23.37
0.133
60.1
0.029
-
-
-
-
Our AutoReconstructor ⋆
24.18
0.128
49.45
0.025
22.88
0.176
47.54
0.026
Ours Diffusion single sample ⋆
23.87
0.127
49.34
0.024
22.01
0.179
48.17
0.026
Ours Diffusion 20-best oracle ⋆
24.36
0.124
48.12
0.022
22.42
0.176
47.62
0.027
Table D1. Quantitative results on masked objects from the CO3D dataset. To compare against baselines which only predict novel
views for the masked foreground object, as shown in the “Splatter Image” row in Fig. D3, we masked the background areas and compare
metrics. †: cited from respective papers or [65]; ⋆: trained by us.
D.3. Additional Qualitative Results for Reconstruc-
tion
• Figure D3, Figures D4 and D5 show more reconstruction
results on hydrants and teddybears.
• Figure D6 further shows renders of depth maps from our
Gaussian splats, demonstrating that our samples are di-
verse in both geometry and appearance.
D.4. Computation Analysis
On each dataset, the AutoReconstructor takes 6 days and the
diffusion model takes 2 days to train, both on 1 A100 GPU.
Compared with the closest diffusion model DFM, which re-
quires 7 days of training on 8 A100 GPUs, our method is
significantly more efficient. During inference, as shown in
Table 2, generating a Splatter Image and rendering novel
view images for one sequence takes only 3 seconds, while
DFM requires 10 minutes. If using only the AutoRecon-
structor, the inference time can further decrease to 0.15 sec-
onds. Overall, our pipeline is quite efficient in both train-
ing and inference, with strong potential to scale up to larger
datasets and be used in online downstream applications.
E. Limitations and Future Work
We make a step forward in training generative models on
3D scenes using Gaussian splatting representations. How-
ever, there are still limitations that suggest promising di-
rections for future work.
First, we model scenes using
Splatter Images, which are largely constrained to the im-
age frustum of a single view. Extending this representa-
tion to large-scale scenarios, either by exploring multiple
Splatter Images or alternative representations that cover en-
tire scenes, would be valuable. Second, large-scale training
across diverse datasets, including both indoor and outdoor
scenes and a wide range of object categories, could lead to a
generalizable large reconstruction model. Finally, while we
currently model static 3D scenes, extending the framework
to 4D Gaussians to capture scene dynamics would enable
broader applications in dynamic scene understanding and
synthesis.
16

<!-- page 17 -->
GT
DFM
Splatter
Image
(full 
image)
Ours
Condition
Novel Views
Condition
Novel Views
Condition
Novel Views
Condition
Novel Views
Splatter
Image
Not available
Not available
Figure D3. Qualitative results on the “Hydrants” and “TeddyBears” categories from the CO3D dataset. Our model produces sharper
results with higher-quality details compared to the baselines, especially in occluded areas.
Figure D4. Additional qualitative results on “TeddyBears”.
17

<!-- page 18 -->
Figure D5. Additional qualitative results on “Hydrants”.
Input
Target
RGB Samples from Ours
Depth Renders from Ours
Input
Target
RGB Samples from Ours
Depth Renders from Ours
Figure D6. This image shows that our samples are diverse in both RGB and depth.
18
