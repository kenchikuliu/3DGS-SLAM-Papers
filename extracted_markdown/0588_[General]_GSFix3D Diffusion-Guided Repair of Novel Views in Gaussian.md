<!-- page 1 -->
GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting
Jiaxin Wei1
Stefan Leutenegger1,2
Simon Schaefer1
1Technical University of Munich
2ETH Zurich
Abstract
Recent developments in 3D Gaussian Splatting have sig-
nificantly enhanced novel view synthesis, yet generating
high-quality renderings from extreme novel viewpoints or
partially observed regions remains challenging.
Mean-
while, diffusion models exhibit strong generative capabili-
ties, but their reliance on text prompts and lack of aware-
ness of specific scene information hinder accurate 3D re-
construction tasks. To address these limitations, we intro-
duce GSFix3D, a novel framework that improves the vi-
sual fidelity in under-constrained regions by distilling prior
knowledge from diffusion models into 3D representations,
while preserving consistency with observed scene details.
At its core is GSFixer, a latent diffusion model obtained
via our customized fine-tuning protocol that can leverage
both mesh and 3D Gaussians to adapt pretrained gen-
erative models to a variety of environments and artifact
types from different reconstruction methods, enabling ro-
bust novel view repair for unseen camera poses.
More-
over, we propose a random mask augmentation strategy that
empowers GSFixer to plausibly inpaint missing regions.
Experiments on challenging benchmarks demonstrate that
our GSFix3D and GSFixer achieve state-of-the-art perfor-
mance, requiring only minimal scene-specific fine-tuning
on captured data. Real-world test further confirms its re-
silience to potential pose errors. Our code and data will be
made publicly available. Project page: gsfix3d.github.io.
1. Introduction
3D Gaussian Splatting (3DGS) [9] has recently emerged as
an efficient and expressive explicit representation that mod-
els scenes using a set of 3D Gaussian primitives and enables
photorealistic rendering through differentiable rasterization.
Compared to previous Neural Radiance Fields (NeRF) [16]
approaches, it achieves faster convergence and significantly
higher rendering speeds. However, a key limitation persists
in those optimization-based representations as they heav-
ily rely on meticulously curated and densely sampled in-
put views to achieve high visual fidelity near the training
camera poses. In regions with sparse observations or from
viewpoints that deviate substantially from the training data,
3DGS struggles to infer plausible geometry and appearance,
often producing artifacts such as incomplete surfaces, un-
natural geometry, or visible holes that severely degrade im-
age quality. Moreover, obtaining sufficient coverage and
accurate measurements often requires labor-intensive data
collection, costly high-end 3D scanners, and skilled opera-
tors, which largely limits the accessibility of such methods
for casual users with only mobile devices.
In parallel, text-to-image generative models based on
latent-space denoising diffusion, such as Stable Diffu-
sion [25], have shown the remarkable ability to synthe-
size diverse and photorealistic images. Trained on large-
scale, captioned images from the internet, those models
effectively gain a deep understanding of 2D visual con-
cepts. To obtain greater control over diffusion model out-
puts, a variety of techniques, such as ControlNet [40], T2I-
Adapters [17], and LoRA [5], have been proposed. Though
powerful, these methods are primarily designed for image
generation rather than repairing, and thus often lack input-
output consistency, making them unsuitable for direct inte-
gration into 3D reconstruction pipelines where spatial and
visual fidelity are critical.
To combine the strengths of diffusion models with ex-
isting 3D reconstructions, we introduce a novel view re-
pair framework, GSFix3D, tailored for 3D Gaussian Splat-
ting. Our method renders novel view images from initial
reconstructions and refines them using a scene-adapted la-
tent diffusion model by removing rendering artifacts and
completing missing content. These enhanced images are
then treated as pseudo-inputs and lifted back into 3D space
to improve the underlying reconstruction. The key to our
pipeline is a dedicated fine-tuning strategy that enables the
pretrained diffusion model to internalize scene-specific pri-
ors, model artifact patterns, and develop inpainting capa-
bilities using our proposed random mask augmentation. In
contrast to DIFIX [35], which relies on large-scale curated
real image pairs for training yet still lacks inpainting ca-
pabilities and struggles with unseen artifacts, our method
requires only a one-time pretraining on two small synthetic
datasets [1, 24] to obtain a general base model. This base
model can then be efficiently fine-tuned on the same cap-
1
arXiv:2508.14717v1  [cs.CV]  20 Aug 2025

<!-- page 2 -->
tured data used for initial reconstruction, enabling adap-
tation to diverse scenes. The resulting module, GSFixer,
acts as a plug-and-play image enhancer, transforming im-
perfect renderings into high-quality, photorealistic images.
Our main contributions are as follows:
• We propose GSFix3D, a new pipeline for repairing novel
views in 3DGS reconstructions that leverages the diffu-
sion model, GSFixer, to enhance under-constrained re-
gions.
We exploit the complementary properties be-
tween 3DGS and traditional mesh representations to fur-
ther boost repairing performance.
• We introduce a customized fine-tuning protocol for pre-
trained diffusion models tailored to the novel view re-
pair task. This protocol efficiently adapts the model to
diverse scenes and reconstruction pipelines and enables
it to internalize scene-specific priors, learn artifact pat-
terns, and develop strong inpainting capabilities through
our proposed random mask augmentation.
• Experiments on challenging benchmarks demonstrate
state-of-the-art performance under extreme novel view-
points, with only a few hours of fine-tuning on the same
captured data used for reconstruction using a single con-
sumer GPU. Additional tests on self-collected real-world
data further validate its robustness to pose inaccuracies.
We will release the real-world data and selected extreme
novel views from the Replica dataset [29].
2. Related Work
2.1. 3D Reconstruction and Mapping
Traditional dense reconstruction methods, such as Kinect-
Fusion [18], fuse per-frame depth maps into a volumetric
grid. Follow-up work improves scalability by using effi-
cient data structures like octrees [28, 31] and voxel hash-
ing [20, 21]. Though the reconstructed geometry suffices
for robotics tasks such as navigation, it often lacks re-
alism in visualization.
Recently, NeRF [16] represents
scenes as implicit neural functions. Several NeRF-based
SLAM systems combine tracking and mapping within this
framework [30, 43]. Despite producing high-quality ren-
derings, NeRF methods are computationally expensive and
struggle with real-time applications. 3DGS [9] addresses
these limitations by representing scenes with explicit, dif-
ferentiable Gaussian primitives, enabling faster rendering
and optimization.
This has led to several 3DGS-based
SLAM systems: GS-SLAM [37] uses opacity thresholds
to drive adaptive Gaussian insertion, SplaTAM [8] em-
ploys a densification mask based on rendered silhouettes
and depth, while MonoGS [15] relies on monocular depth
estimates with variable uncertainty. To improve efficiency,
RTG-SLAM [23] categorizes Gaussians as either opaque
or transparent and updates only unstable ones, whereas
GSFusion [33] integrates Truncated Signed Distance Field
(TSDF) [2] and 3DGS in a hybrid framework and employs
a quadtree-based image segmentation strategy to reduce re-
dundant splats. Despite these advances, challenges persist
in handling under-constrained areas and achieving artifact-
free reconstruction. We build our approach on 3DGS recon-
structions due to their real-time performance, photorealistic
rendering, and full differentiability, which make them par-
ticularly suitable for downstream repair tasks.
2.2. Novel View Repair
Although dense-view reconstruction has become increas-
ingly reliable, novel view rendering remains susceptible
to artifacts, especially in under-constrained regions. Prior
work has largely focused on sparse-view settings, where
such degradation is more obvious. [13] introduces a decep-
tive diffusion model that refines novel views rendered from
few-view reconstructions and uses an uncertainty measure
to improve consistency. RI3D [22] uses two separate dif-
fusion models for repairing visible regions and inpainting
missing areas, whereas ours integrates these tasks into a sin-
gle model. To improve temporal coherence, several meth-
ods leverage video diffusion models. 3DGS-Enhancer [14]
is the first to train a video diffusion model on a large-
scale dataset created with pairs of low and high-quality im-
ages. GenFusion [36] fine-tunes a video diffusion model on
artifact-prone RGB-D videos using a masking strategy that
simulates common view-dependent artifacts for content-
aware outpainting, while [42] uses training-free scene-
grounding guidance to steer the video diffusion model to-
ward temporally consistent synthesis. Despite promising
results, these methods rely heavily on customized prepro-
cessing steps to bootstrap initial reconstructions and care-
fully curated datasets to train diffusion models effectively.
In this paper, we focus on novel view repair for recon-
structions where artifacts still persist despite extensive cov-
erage. SGD [39] introduces a tailored diffusion pipeline
for autonomous driving scenarios, using adjacent frames as
conditioning inputs and leveraging LiDAR point cloud to
train a ControlNet for explicit depth control. DIFIX [35]
takes a step toward general view repair by training a single-
step diffusion model on a large curated dataset of real
noisy–clean image pairs, created via handcrafted corrup-
tion strategies. However, its performance drops when ex-
posed to unseen artifacts and it struggles with inpainting.
In contrast, our GSFixer is obtained through a lightweight
fine-tuning protocol. With minimal pretraining on synthetic
data and fine-tuning on captured reconstruction data, GS-
Fixer achieves robust artifact removal, adapts to diverse
pipelines and scenes, and exhibits strong inpainting capa-
bilities, all within a single model that runs efficiently on
consumer hardware.
2

<!-- page 3 -->
Novel View
Mesh
3DGS
Ray casting
VAE
Encoder
VAE
Encoder
Noise
𝒩(0, 𝐼)
C
VAE
Decoder
U-Net
𝘵
Denoise
𝘵
𝘵=0
𝘵
C
𝒛"#$
Denoising timestep
Frozen module
Concatenate
GSFixer
Gradient flow
Photometric loss
𝛼-blending
Initial 3D Reconstructions
Figure 1. System overview of the proposed GSFix3D framework for novel view repair. Given initial 3D reconstructions in the form of mesh
and 3DGS, we render novel views and use them as conditional inputs to GSFixer. Through a reverse diffusion process, GSFixer generates
repaired images with artifacts removed and missing regions inpainted. These outputs are then distilled back into 3D by optimizing the
3DGS representation using photometric loss.
3. Method
Our goal is to enhance the photorealism of novel views in
reconstructed 3DGS scenes, especially for viewpoints dis-
tant from the original camera trajectories and suffer from
limited observations. We present a customized fine-tuning
protocol to adapt a pretrained diffusion model for artifact re-
moval and view inpainting (Sec. 3.1). We then describe our
inference scheme (Sec. 3.2), and how the fine-tuned model
integrates into the full pipeline to improve the visual quality
of novel views (Sec. 3.3). An overview of our method is
illustrated in Fig. 1.
3.1. Fine-Tuning Protocol
Given a reconstructed 3DGS scene, we formulate the image
repair task as a conditional generation problem and fine-
tune a pretrained latent diffusion model, i.e. Stable Diffu-
sion v2 [25], to learn the conditional distribution p(Igt|Ic)
where Igt ∈RH×W ×3 denotes the ground truth RGB im-
age and Ic ∈RH×W ×3 is the condition image rendered
from the imperfect reconstruction.
In our approach, we further extend the conditioning input
to two rendered images: one from the 3D Gaussian Splat-
ting representation (Igs) and another from a mesh repre-
sentation (Imesh). Thus, the actual conditional distribution
becomes p(Igt|Imesh, Igs). This dual-conditioning strat-
egy is motivated by the complementary strengths of 3DGS
and traditional mesh-based reconstructions. 3DGS, as an
optimization-based method, tends to suffer in regions with
sparse observations, often leading to visible artifacts such
as holes or incomplete geometry.
Mesh reconstructions,
though usually less photorealistic at lower resolutions, of-
fer more coherent geometry and stronger spatial priors in
under-constrained areas. By jointly leveraging both rep-
resentations, we aim to provide the diffusion model with
richer appearance cues for image refinement.
To ensure
that the mesh input remains geometrically consistent yet in-
dependent from the 3DGS optimization process, we obtain
the mesh and the 3DGS map simultaneously using GSFu-
sion [33], an online RGB-D mapping system. This avoids
directly extracting the mesh from the 3DGS representation,
as done in prior works [3, 6], which could introduce corre-
lated artifacts. The overall fine-tuning protocol is presented
in Fig. 2. We conduct an ablation study in Sec. 4.4 com-
paring the performance of using both inputs versus 3DGS
alone, validating the effectiveness of our design choice.
3.1.1
Network Architecture
Diffusion models [19, 25, 27] are a class of generative
frameworks that generate data by learning to invert a pro-
gressively noised process. We use a frozen Variational Au-
toencoder (VAE) [10] to encode all images into a latent
space, enabling diffusion-based learning in a more compact
domain. For a given image I, its latent code is obtained
via the encoder E : z = E(I). This results in a latent
triplet (zmesh, zgs, zgt). To train the denoising model, we
follow the standard Denoising Diffusion Probabilistic Mod-
els (DDPM) [4] formulation and incrementally add standard
Gaussian noise ϵ ∼N(0, I) to the clean ground-truth latent
z0 := zgt over T discrete timesteps, producing a sequence
{zt}T
t=1. The noisy latent at timestep t is then given by:
zt = √¯αtz0 +
√
1 −¯αtϵ,
(1)
where ¯αt denotes the cumulative product of noise sched-
ule coefficients [4, 26]. Following [7], we repurpose the
U-Net backbone from the pretrained diffusion model into
a conditional denoiser for image repair. We concatenate
the latent codes along the feature dimension to form the
input ¯zt = concat(zmesh, zgs, zt). To accommodate the
increased channel count, we expand the first layer of the
U-Net by duplicating the original weight tensor and divid-
ing its values by three. This design choice maintains the
original weight distribution and prevents excessive activa-
tion scaling, allowing us to preserve the initialization be-
havior of the pretrained model while enabling conditional
3

<!-- page 4 -->
VAE
Encoder
VAE
Encoder
Noise
𝒩(0, 𝐼)
C
U-Net
𝘵
𝘵
C
𝜖
Random timestep
Trainable module
Concatenate
VAE
Encoder
2
̂𝜖
Add Noise
𝘵
Rendered image (mesh)
Rendered image (3DGS)
Captured image
Training View
Frozen module
Random Mask Augmentation
Figure 2. Illustration of the customized fine-tuning protocol for
adapting a pretrained diffusion model into GSFixer, enabling it to
handle diverse artifact types and missing regions.
inputs. The conditional U-Net ϵθ is then trained to predict
the added noise by minimizing a standard DDPM objective:
L = Ez0,ϵ∼N (0,I),t∼U[1,T ]
h
∥ϵ −ˆϵ∥2i
,
(2)
where ˆϵ = ϵθ(¯zt, t) is the predicted noise.
3.1.2
Data Augmentation
To construct our training set, we render each captured view
using both the mesh and the 3DGS map, resulting in paired
triplets (Imesh, Igs, Igt), where Igt is the original captured
RGB image, Igs is the image rendered from 3DGS via
α-blending, and Imesh is obtained via ray-casting on the
mesh. This process requires no additional data beyond the
original captured RGB images, their corresponding camera
poses, and the reconstructed maps.
Direct fine-tuning on these triplets can already help the
diffusion model adapt to the scene and learn to remove spe-
cific artifacts in the 3DGS rendering. However, one major
challenge remains: the model’s ability to inpaint missing
regions, which usually appear as black holes in novel views
due to under-constrained geometry or occlusions.
Since
all training images are rendered from the original captured
viewpoints, they are mostly complete in appearance and fail
to expose the model to such corner cases.
To explicitly train the model to handle incomplete ren-
derings, we introduce a masking-based data augmentation
scheme. For each training triplet, we randomly select a se-
mantic mask from a set of annotated real-image masks [32].
The key intuition is to leverage the diverse mask shapes de-
rived from real-world object semantics, which not only en-
hances the realism of the masked regions but also eliminates
the need for manually designing complex rules to simulate
missing areas caused by various factors such as occlusion
or under-constrained observations. This mask is applied in
two distinct ways: (1) the same mask is overlaid on both
Imesh and Igs, simulating occlusions that might occur in
novel views; and (2) an additional, independent mask is ap-
plied solely to Igs to simulate the common degradation of
3DGS renderings in regions with limited observations. To
better approximate the soft boundaries in 3DGS renderings,
we further apply a small amount of Gaussian blur to the
mask used on Igs. We evaluate the impact of this augmenta-
tion strategy in Sec. 4.4, where we compare models trained
with and without random masks and show its importance in
improving the inpainting ability for novel views.
3.2. Inference with GSFixer
At inference time, we freeze the fine-tuned U-Net param-
eters and apply the model to novel views, as illustrated in
Fig. 1. We begin by encoding the conditional inputs, i.e.,
the rendered images from novel viewpoints, into the latent
space using the frozen VAE encoder. The latent for the
target image to be generated, zt, is initialized as standard
Gaussian noise. We then concatenate these latent codes in
the same order used during fine-tuning to form the diffu-
sion model input: ¯zt = concat(zmesh, zgs, zt). To generate
the fixed image, we iteratively denoise zt using the deter-
ministic Denoising Diffusion Implicit Model (DDIM) [26]
schedule to perform efficient non-Markovian sampling. The
update at each timestep is as follows:
zt−1 = √¯αt−1ˆz0 +
p
1 −¯αt−1ϵθ(¯zt, t),
(3)
where the clean latent ˆz0 is estimated as:
ˆz0 =
1
√¯αt
 zt −
√
1 −¯αtϵθ(¯zt, t)

,
(4)
derived directly from the forward diffusion formulation in
Eq. (1). After completing the denoising process, the final
fixed image is obtained by decoding the predicted clean la-
tent using the VAE decoder D : ˆIfixed = D(z0).
3.3. GSFix3D: Diffusion-Guided Novel View Repair
The final stage of our GSFix3D framework lifts the output
of the diffusion model, i.e., GSFixer, back into the 3D rep-
resentation. Thanks to the full differentiability of 3DGS, we
can continue optimizing the parameters of the initial 3DGS
reconstruction by minimizing a photometric loss between
the fixed image ˆIfixed and the rendered image Igs:
Lpho = (1 −λ)∥ˆIfixed −Igs∥1 + λLSSIM(ˆIfixed, Igs),
where λ is a weighting factor, and LSSIM denotes the Struc-
tural Similarity loss. We also enable adaptive density con-
trol during optimization, following [9], to fill in previously
empty or under-populated regions.
To reduce inconsistencies in the repaired images and
improve global coherence, we further append the repaired
views and their corresponding poses to the original captured
datasets and then optimize over this augmented dataset
for several iterations.
Note that we use a sparse set of
keyframes recorded during the initial reconstruction phase
instead of the full dataset to avoid redundant and time-
consuming optimization.
4

<!-- page 5 -->
Method
ScanNet++
Replica
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
SplaTAM
23.03
0.791
0.311
23.82
0.833
0.267
SplaTAM + DIFIX
23.06
0.789
0.220
22.97
0.790
0.262
SplaTAM + DIFIX-ref
22.79
0.799
0.203
22.97
0.830
0.217
SplaTAM + GSFixer
25.11
0.831
0.188
25.67
0.839
0.215
RTG-SLAM
19.54
0.777
0.341
25.00
0.860
0.247
RTG-SLAM + DIFIX
19.43
0.762
0.245
24.02
0.811
0.214
RTG-SLAM + DIFIX-ref
19.29
0.769
0.223
23.89
0.834
0.193
RTG-SLAM + GSFixer
24.80
0.824
0.204
26.27
0.843
0.228
GSFusion (gs)
24.58
0.838
0.308
22.10
0.844
0.296
GSFusion (gs) + DIFIX
24.34
0.818
0.193
21.81
0.772
0.273
GSFusion (gs) + DIFIX-ref
23.83
0.822
0.184
21.91
0.821
0.224
GSFusion (gs) + GSFixer
24.79
0.833
0.196
23.87
0.830
0.251
GSFusion (mesh+gs) + GSFixer
25.30
0.837
0.183
25.98
0.845
0.219
Table 1. Comparisons of diffusion-based repair methods on the ScanNet++ and Replica datasets. The best result is highlighted in bold,
and the second-best is underlined. The text inside ( ) indicates the format of the reconstruction used.
4. Experiments
4.1. Experimental Setup
Evaluation Datasets and Metrics.
We compare dif-
ferent methods on two challenging benchmark datasets:
ScanNet++[38] and Replica[29]. ScanNet++ is a real-world
indoor dataset containing high-quality RGB-D data. Each
scene includes two separate camera trajectories for train-
ing and evaluation, respectively. Following [33], we se-
lect four scenes from ScanNet++: 8b5caf3398, 39f36da05b,
b20a261fdf, and f34d532901.
The Replica dataset con-
sists of photorealistic synthetic indoor scenes with accurate
RGB-D imagery. We use consistent trajectories from [43]
for reconstruction and fine-tuning. To enable the quanti-
tative assessment of novel views and to evaluate inpaint-
ing capabilities, we manually render ground truth novel
views from extreme viewpoints with large unobserved re-
gions (see Sec. 6.1 in the supplementary). We use three
common metrics to measure rendering quality and fidelity:
PSNR, SSIM, and LPIPS. All reported results are averaged
over scenes within each dataset.
Baselines. We compare our GSFixer against two variants
from [35]: DIFIX and DIFIX-ref. DIFIX is a single-step
image diffusion model trained on 80k noisy-clean image
pairs curated from real-world datasets. DIFIX-ref extends
this setup by incorporating an additional reference view as
input, introducing multi-view constraints to enhance per-
formance. In addition to GSFusion, we also include two
recent Gaussian SLAM methods, SplaTAM[8] and RTG-
SLAM[23], as alternative sources of 3D reconstructions,
each exhibiting distinct artifact patterns due to differences
in initialization and optimization strategies. We apply the
above image repair models to novel view renderings pro-
duced by each of these reconstruction methods.
Implementation Details. We adopt Stable Diffusion v2
as our base latent diffusion model, disabling text prompt
and applying the fine-tuning protocol described in Sec. 3.1.
During training, we use the DDPM noise scheduler with
1000 diffusion steps. For inference, we follow the DDIM
scheduler with only 4 steps for accelerated sampling. Con-
sidering the difficulty of collecting large-scale real-world
training pairs for this task, we first pretrain the modi-
fied U-Net (see Sec. 3.1.1) for 6k iterations on two syn-
thetic datasets: Hypersim[24] for indoor scenes and Virtual
KITTI [1] for outdoor street environments. We use a batch
size of 2 and accumulate gradients over 16 steps to stabilize
training with the Adam optimizer. The learning rate is set
to 3 × 10−5. We acquire the geometrically aligned mesh
and 3DGS map by running GSFusion and fine-tune the pre-
trained model separately for each scene. For real scenes
from ScanNet++, we fine-tune for 800 iterations. For syn-
thetic scenes from Replica, we fine-tune for 400 iterations.
The fine-tuning process typically takes 4 hours for Scan-
Net++ and 2 hours for Replica. As for 3DGS optimization
in GSFix3D, we perform 20 iterations for each repaired im-
age and 50 iterations over the augmented dataset. All exper-
iments are conducted on a single NVIDIA RTX 4500 Ada
GPU with 24GB VRAM.
4.2. Results
Table 1 reports quantitative results on ScanNet++ and
Replica. For SplaTAM and RTG-SLAM, which output only
3DGS maps, we fine-tune GSFixer exclusively on rendered
images from their reconstructions. Despite this constraint,
GSFixer consistently outperforms DIFIX and DIFIX-ref
across all metrics on ScanNet++, with over 5 dB PSNR
gain in the RTG-SLAM+GSFixer setting. Qualitative re-
sults in Fig. 3 show that, in the RTG-SLAM example, DI-
FIX and DIFIX-ref leave a large black hole where a win-
dow is missing, while GSFixer fills it with plausible content.
In the SplaTAM example, baselines leave colorful floaters,
whereas GSFixer learns their patterns and removes them.
For GSFusion, which provides both a mesh and a
3DGS map, we introduce a dual-input setting, GSFu-
5

<!-- page 6 -->
GT
DIFIX
DIFIX-ref
GSFixer
Input (gs)
SplaTAM
RTG-SLAM
GSFusion
SplaTAM
RTG-SLAM
GSFusion
ScanNet++
Replica
Figure 3. Qualitative comparisons of diffusion-based repair methods on the ScanNet++ and Replica datasets. All examples use only 3DGS
reconstructions as the input source. Our GSFixer effectively removes artifacts and fills in large holes, where both DIFIX and DIFIX-ref
fail to produce satisfactory results.
GT
Input (mesh)
Input (gs)
GSFixer
ScanNet++
Replica
GSFix3D
Figure 4. Qualitative comparison between GSFixer and GSFix3D on the ScanNet++ and Replica datasets. Both mesh and 3DGS recon-
structions from GSFusion are used as input sources. The 2D visual improvements from GSFixer are effectively distilled into the 3D space
by GSFix3D.
6

<!-- page 7 -->
Reference GT
Input (gs)
GSFixer
GSFix3D
Figure 5. Novel view repair on self-collected ship data. Our method is robust to pose errors, effectively removing shadow-like floaters.
sion(mesh+gs)+GSFixer, that further boosts performance
over the single-input variant. We analyze this effect in detail
in Sec. 4.4. On the more challenging Replica dataset, where
we evaluate on manually selected extreme novel viewpoints
with large unobserved regions, GSFixer again outperforms
baselines in PSNR and remains competitive in SSIM. The
strong inpainting ability of GSFixer is visually evident on
the Replica dataset in Fig. 3.
Interestingly, DIFIX and
DIFIX-ref achieve lower LPIPS scores in some cases, which
we attribute to their sharp visual details. This is likely due to
their training on 80k noisy-clean image pairs curated from
real-world datasets (though the dataset is not publicly avail-
able), whereas GSFixer is only pretrained on two synthetic
datasets and fine-tuned on a limited amount of clean cap-
tured data. We explore additional comparisons and results
in Sec. 7.1 in the supplementary.
The overall performance of our GSFix3D framework is
reported in Tab. 2. Compared to the direct outputs from
GSFixer, lifting the repaired images back into the 3D rep-
resentation leads to improved perceptual quality thanks to
multi-view constraints, as evidenced by higher PSNR and
SSIM scores. However, due to the optimization character-
istics of the 3DGS representation, the final renderings tend
to be less smooth than the 2D generative results, which ac-
counts for the slightly higher LPIPS values. Qualitative ex-
amples are provided in Fig. 4. We further apply the full
GSFix3D framework to SplaTAM and RTG-SLAM recon-
structions, demonstrating its effectiveness in Sec. 7.3 of the
supplementary material.
4.3. Real-World Evaluation in the Wild
We collect a stereo sequence inside a ship structure using an
Intel RealSense D455 camera. We compute depth maps for
the left camera using FoundationStereo [34] for improved
quality, and estimate camera poses with OKVIS2 [12].
Since no ground truth is available, the estimated poses may
contain errors. Those post-processed data are then fed into
GSFusion to obtain an initial 3DGS reconstruction.
We
fine-tune a GSFixer model using 3DGS renderings as input.
Figure 5 shows a novel view example where shadow-like
floaters appear near the ladder due to inaccurate poses. Our
method effectively removes these artifacts in 2D and dis-
Dataset
Method
PSNR↑
SSIM↑
LPIPS↓
ScanNet++
GSFusion (gs)
24.58
0.838
0.308
GSFusion (mesh+gs) + GSFixer
25.30
0.837
0.183
GSFusion (mesh+gs) + GSFix3D
25.63
0.845
0.238
Replica
GSFusion (gs)
22.10
0.844
0.296
GSFusion (mesh+gs) + GSFixer
25.98
0.845
0.219
GSFusion (mesh+gs) + GSFix3D
26.49
0.864
0.252
Table 2. Comparisons of GSFixer and GSFix3D on the ScanNet++
and Replica datasets.
tills the correction back into the 3D representation, demon-
strating robustness to common pose errors in real-world
data collection, particularly in uncontrolled settings with-
out high-end equipment or precise calibration. Additional
real-world results are provided in Sec. 7.4 of the supple-
mentary, including a test on an outdoor scene [41] using
a LiDAR-Inertial-Camera Gaussian Splatting SLAM sys-
tem [11], which further demonstrates the practical adapt-
ability of our method.
4.4. Ablation Studies
Image Conditions. To analyze the impact of different in-
put image conditions, we evaluate GSFixer under three in-
put configurations: mesh-only, 3DGS-only, and dual-input
(mesh+3DGS), with results in Tab. 3.
On the synthetic
Replica, which provides highly accurate measurements,
mesh-based renderings tend to be of higher quality than
their 3DGS counterparts from novel viewpoints. As a result,
the GSFusion(mesh)+GSFixer setting achieves better ren-
dering performance than GSFusion(gs)+GSFixer. In con-
trast, on the real-world ScanNet++ dataset, 3DGS recon-
structions outperform mesh renderings due to noisy depth,
making GSFusion(gs)+GSFixer the better choice. When
both images are used together as input, we observe comple-
mentary advantages: the dual-input setup leads to improved
performance on ScanNet++ and a modest gain on Replica.
Qualitative results in Fig. 6 further highlight this bene-
fit. For example, in a ScanNet++ scene, the mesh-rendered
image suffers from geometric inaccuracies along the table
edge, while the 3DGS-rendered image shows visual gaps
on the table surface. When both are used to condition GS-
Fixer, these issues are effectively mitigated. Similarly, in a
7

<!-- page 8 -->
GT
Input (mesh)
GSFixer (mesh)
Input (gs)
GSFixer (gs)
GSFixer (mesh+gs)
ScanNet++
Replica
Figure 6. Qualitative ablation of input image conditions on the ScanNet++ and Replica datasets. We compare GSFixer results using three
types of inputs rendered from GSFusion: mesh-only, 3DGS-only, and dual-input. The artifacts (highlighted by green and yellow boxes)
present in the single-input settings are effectively mitigated with the dual-input configuration.
GT
Input (mesh)
Input (gs)
GSFixer (w/o mask)
GSFixer (w mask)
Figure 7. Qualitative ablation of random mask augmentation on the Replica dataset. We compare GSFixer results fine-tuned with and
without our proposed augmentation strategy. The differences in inpainting quality highlight the improved ability to fill large missing
regions when augmentation is used.
Replica scene, the mesh-rendered image exhibits blurry tex-
tures on the pillow, and the 3DGS-rendered image contains
visible holes on the floor. Combining both inputs allows
GSFixer to resolve these artifacts by leveraging strengths
from each source. Additional experiments on SplaTAM and
RTG-SLAM are presented in Sec. 7.2 of the supplementary.
Random Mask Augmentation. To validate the effective-
ness of our proposed data augmentation strategy in improv-
ing inpainting capability for novel view repair, we con-
duct an ablation study by disabling the random mask aug-
mentation during fine-tuning on the Replica dataset. We
choose Replica for this evaluation due to its challenging
novel views with extensive unobserved regions and visible
holes. As shown in Tab. 4, GSFixer fine-tuned with ran-
dom mask augmentation consistently outperforms the vari-
ant without augmentation across all metrics. It is also evi-
dent in Fig. 7. The 3DGS-rendered image contains a large
missing region on the whiteboard. Without random mask
augmentation, GSFixer struggles to inpaint the hole even
when given an additional mesh-rendered image as a condi-
tion. In contrast, our full model with augmentation success-
fully fills in the missing region with coherent and realistic
textures, demonstrating its generalization to real occlusions.
5. Conclusion
GSFix3D raises the bar for novel view repair in 3DGS re-
constructions, requiring no massive real data curation or
costly pertaining, only minimal fine-tuning on a small set of
captured views. By coupling this efficient fine-tuning pro-
Dataset
Method
PSNR↑
SSIM↑
LPIPS↓
ScanNet++
GSFusion (mesh)
17.87
0.750
0.358
GSFusion (mesh) + GSFixer
24.64
0.823
0.198
GSFusion (gs)
24.58
0.838
0.308
GSFusion (gs) + GSFixer
24.79
0.833
0.196
GSFusion (mesh+gs) + GSFixer
25.30
0.837
0.183
Replica
GSFusion (mesh)
23.20
0.849
0.217
GSFusion (mesh) + GSFixer
26.61
0.846
0.200
GSFusion (gs)
22.10
0.844
0.296
GSFusion (gs) + GSFixer
23.87
0.830
0.251
GSFusion (mesh+gs) + GSFixer
25.98
0.845
0.219
Table 3.
Ablation of image conditions on the ScanNet++ and
Replica datasets.
GSFusion (mesh+gs)
PSNR↑
SSIM↑
LPIPS↓
+ GSFixer (w/o mask)
23.54
0.830
0.231
+ GSFixer (w mask)
25.98
0.845
0.219
Table 4. Ablation of random mask augmentation on the Replica
dataset.
tocol with a dual-input design that fuses mesh and 3DGS
cues, and empowering it with random mask augmentation
as the key to strong inpainting performance, the resulting
diffusion model, GSFixer, removes artifacts, fills missing
regions with plausible detail, and adapts seamlessly to dif-
ferent scenes and reconstruction pipelines. Across diverse
and challenging benchmarks, our method consistently out-
performs prior diffusion-based approaches, validating its ef-
fectiveness, adaptability, and robustness even under pose in-
accuracies, underscoring its practicality for a wide range of
3D reconstruction scenarios.
8

<!-- page 9 -->
Acknowledgement.
The
authors
gratefully
acknowl-
edge
support
from
the
EU
project
AUTOASSESS
(Grant 101120732).
We also thank Jaehyung Jung
and
Sebasti´an
Barbas
Laina
for
their
assistance
with ship data collection and processing, and Helen
Oleynikova for her valuable feedback on the manuscript.
References
[1] Yohann Cabon, Naila Murray, and Martin Humenberger. Vir-
tual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 1, 5
[2] Brian Curless and Marc Levoy. A volumetric method for
building complex models from range images. In Proceedings
of the 23rd annual conference on Computer graphics and
interactive techniques, pages 303–312, 1996. 2
[3] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 3
[4] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. Advances in neural information
processing systems, 33:6840–6851, 2020. 3
[5] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models. ICLR,
1(2):3, 2022. 1
[6] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 3
[7] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger,
Shengyu Huang, Bo Li, Anton Obukhov, and Konrad
Schindler.
Marigold: Affordable adaptation of diffusion-
based image generators for image analysis. arXiv preprint
arXiv:2505.09358, 2025. 3
[8] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 2, 5
[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 2, 4
[10] Diederik P Kingma, Max Welling, et al. Auto-encoding vari-
ational bayes, 2013. 3
[11] Xiaolei Lang, Laijian Li, Chenming Wu, Chen Zhao, Lina
Liu, Yong Liu, Jiajun Lv, and Xingxing Zuo.
Gaussian-
lic:
Real-time photo-realistic slam with gaussian splat-
ting and lidar-inertial-camera fusion.
arXiv preprint
arXiv:2404.06926, 2024. 7, 3, 5
[12] Stefan
Leutenegger.
Okvis2:
Realtime
scalable
visual-inertial slam with loop closure.
arXiv preprint
arXiv:2202.09199, 2022. 7, 1
[13] Xinhang Liu, Jiaben Chen, Shiu-Hong Kao, Yu-Wing Tai,
and Chi-Keung Tang.
Deceptive-nerf/3dgs:
Diffusion-
generated pseudo-observations for high-quality sparse-view
reconstruction. In European Conference on Computer Vi-
sion, pages 337–355. Springer, 2024. 2
[14] Xi Liu, Chaoyi Zhou, and Siyu Huang.
3dgs-enhancer:
Enhancing unbounded 3d gaussian splatting with view-
consistent 2d diffusion priors. Advances in Neural Informa-
tion Processing Systems, 37:133305–133327, 2024. 2
[15] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 18039–18048, 2024. 2
[16] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[17] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian
Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning
adapters to dig out more controllable ability for text-to-image
diffusion models. In Proceedings of the AAAI conference on
artificial intelligence, pages 4296–4304, 2024. 1
[18] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Pushmeet
Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon.
Kinectfusion: Real-time dense surface mapping and track-
ing. In 2011 10th IEEE international symposium on mixed
and augmented reality, pages 127–136. Ieee, 2011. 2
[19] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav
Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and
Mark Chen. Glide: Towards photorealistic image generation
and editing with text-guided diffusion models. arXiv preprint
arXiv:2112.10741, 2021. 3
[20] Matthias Nießner, Michael Zollh¨ofer, Shahram Izadi, and
Marc Stamminger. Real-time 3d reconstruction at scale us-
ing voxel hashing. ACM Transactions on Graphics (ToG), 32
(6):1–11, 2013. 2
[21] Helen Oleynikova, Zachary Taylor, Marius Fehr, Roland
Siegwart, and Juan Nieto.
Voxblox: Incremental 3d eu-
clidean signed distance fields for on-board mav planning.
In 2017 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 1366–1373. IEEE, 2017.
2
[22] Avinash Paliwal, Xilong Zhou, Wei Ye, Jinhui Xiong,
Rakesh Ranjan, and Nima Khademi Kalantari. Ri3d: Few-
shot gaussian splatting with repair and inpainting diffusion
priors. 2025. 2
[23] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang,
Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d re-
construction at scale using gaussian splatting. In ACM SIG-
GRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 5
[24] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit
Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb,
and Joshua M Susskind. Hypersim: A photorealistic syn-
thetic dataset for holistic indoor scene understanding.
In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 10912–10922, 2021. 1, 5
9

<!-- page 10 -->
[25] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 1, 3
[26] Jiaming
Song,
Chenlin
Meng,
and
Stefano
Ermon.
Denoising diffusion implicit models.
arXiv preprint
arXiv:2010.02502, 2020. 3, 4
[27] Yang Song and Stefano Ermon. Generative modeling by esti-
mating gradients of the data distribution. Advances in neural
information processing systems, 32, 2019. 3
[28] Frank Steinbrucker, Christian Kerl, and Daniel Cremers.
Large-scale multi-resolution surface reconstruction from
rgb-d sequences. In Proceedings of the IEEE International
Conference on Computer Vision, pages 3264–3271, 2013. 2
[29] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The replica dataset: A digital
replica of indoor spaces. arXiv preprint arXiv:1906.05797,
2019. 2, 5, 1
[30] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davi-
son. imap: Implicit mapping and positioning in real-time. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 6229–6238, 2021. 2
[31] Emanuele Vespa, Nikolay Nikolov, Marius Grimm, Luigi
Nardi, Paul HJ Kelly, and Stefan Leutenegger.
Efficient
octree-based volumetric slam supporting signed-distance
and occupancy mapping.
IEEE Robotics and Automation
Letters, 3(2):1144–1151, 2018. 2
[32] Navve Wasserman, Noam Rotstein, Roy Ganz, and Ron
Kimmel. Paint by inpaint: Learning to add image objects by
removing them first. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 18313–18324,
2025. 4
[33] Jiaxin Wei and Stefan Leutenegger. Gsfusion: Online rgb-d
mapping where gaussian splatting meets tsdf fusion. IEEE
Robotics and Automation Letters, 2024. 2, 3, 5, 1
[34] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz,
Orazio Gallo, and Stan Birchfield. Foundationstereo: Zero-
shot stereo matching. In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 5249–5260,
2025. 7, 1
[35] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Goj-
cic, and Huan Ling.
Difix3d+: Improving 3d reconstruc-
tions with single-step diffusion models. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 26024–26035, 2025. 1, 2, 5
[36] Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger, and
Anpei Chen. Genfusion: Closing the loop between recon-
struction and generation via videos. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
6078–6088, 2025. 2
[37] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19595–19604, 2024. 2
[38] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d in-
door scenes. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 12–22, 2023. 5
[39] Zhongrui Yu, Haoran Wang, Jinze Yang, Hanzhang Wang,
Jiale Cao, Zhong Ji, and Mingming Sun. Sgd: Street view
synthesis with gaussian splatting and diffusion prior. In 2025
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 3812–3822. IEEE, 2025. 2
[40] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding
conditional control to text-to-image diffusion models.
In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 3836–3847, 2023. 1
[41] Chunran Zheng, Qingyan Zhu, Wei Xu, Xiyuan Liu,
Qizhi Guo, and Fu Zhang.
Fast-livo: Fast and tightly-
coupled sparse-direct lidar-inertial-visual odometry. In 2022
IEEE/RSJ international conference on intelligent robots and
systems (IROS), pages 4003–4009. IEEE, 2022. 7, 3, 5
[42] Yingji Zhong, Zhihao Li, Dave Zhenyu Chen, Lanqing
Hong, and Dan Xu. Taming video diffusion prior with scene-
grounding guidance for 3d gaussian splatting from sparse in-
puts.
In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 6133–6143, 2025. 2
[43] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 12786–12796, 2022. 2,
5, 1
10

<!-- page 11 -->
GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting
Supplementary Material
6. Data Preparation
6.1. Novel View Selection for Replica
The Replica dataset [29] contains high-quality reconstruc-
tions of diverse indoor scenes, featuring clean dense ge-
ometry and high-resolution textures. We leverage the pro-
vided 3D models and the official Replica SDK to render
novel view images, which serve as ground truth for quan-
titative evaluation in Sec. 4.2. We adopt the same camera
intrinsics and image resolution (1200 × 680) as the train-
ing views provided by [43] when generating novel views.
Since these trajectories do not cover the entire scene, cer-
tain areas in the reconstructed map remain unobserved or
under-constrained, often exhibiting artifacts. To assess in-
painting performance, we deliberately select novel views
that include such missing or artifact-prone regions, where
existing pipelines (e.g., SplaTAM, RTG-SLAM, GSFusion)
typically fail. This results in an intentionally challenging
novel view repair task. Examples of selected novel views
from the Replica dataset are shown in Fig. 3, Fig. 4, Fig. 6,
Fig. 7, Fig. 8, Fig. 9 and Fig. 10. We will release the curated
Replica novel view dataset to support reproducible research.
6.2. Real-World Data Collection
We collect a stereo image sequence inside a ship’s ballast
water tank using an Intel RealSense D455 camera.
The
dataset contains 1,068 grayscale images at a resolution of
640 × 480 for both the left and right stereo cameras. Stereo
images and recorded IMU data are fed into OKVIS2 [12]
to estimate camera poses for each left stereo image, though
the estimated poses may contain errors. Leveraging recent
advancements in depth estimation, we employ Foundation-
Stereo [34] to generate smoother and higher-quality depth
maps (corresponding to the left stereo images) from the
stereo pairs. The post-processed data, including left stereo
images, depth maps, and camera poses, are then used as in-
put to GSFusion [33] for the initial 3DGS reconstruction.
Since no ground truth is available for this self-collected
dataset, we randomly select five novel views in the recon-
structed scene where shadow-like floaters caused by pose
inaccuracies are prominent. For each, we use a nearby cap-
tured training view as a reference “ground truth” to quali-
tatively assess our method’s performance. Examples of the
captured data and selected novel views are shown in Fig. 5
and Fig. 11. This real-world dataset will be released to sup-
port reproducible research.
Method
PSNR↑
SSIM↑
LPIPS↓
SplaTAM
23.03
0.791
0.311
SplaTAM + DIFIX
23.06
0.789
0.220
SplaTAM + DIFIX-finetune†
24.67
0.829
0.133
SplaTAM + GSFixer
25.11
0.831
0.188
RTG-SLAM
19.54
0.777
0.341
RTG-SLAM + DIFIX
19.43
0.762
0.245
RTG-SLAM + DIFIX-finetune†
24.44
0.812
0.156
RTG-SLAM + GSFixer
24.80
0.824
0.204
GSFusion (gs)
24.58
0.838
0.308
GSFusion (gs) + DIFIX
24.34
0.818
0.193
GSFusion (gs) + DIFIX-finetune†
24.87
0.834
0.142
GSFusion (gs) + GSFixer
24.79
0.833
0.196
GSFusion (mesh+gs) + GSFixer
25.30
0.837
0.183
Table 5. More comparisons of diffusion-based repair methods on
the ScanNet++ dataset. † indicates that we fine-tuned this model
based on the original DIFIX model. The best result is highlighted
in bold, and the second-best is underlined. The text inside ( )
indicates the format of the reconstruction used.
Method
PSNR↑
SSIM↑
LPIPS↓
SplaTAM
23.82
0.833
0.267
SplaTAM + DIFIX
22.97
0.790
0.262
SplaTAM + DIFIX-finetune†
25.45
0.867
0.149
SplaTAM + GSFixer
25.67
0.839
0.215
RTG-SLAM
25.00
0.860
0.247
RTG-SLAM + DIFIX
24.02
0.811
0.214
RTG-SLAM + DIFIX-finetune†
25.47
0.858
0.156
RTG-SLAM + GSFixer
26.27
0.843
0.228
GSFusion (gs)
22.10
0.844
0.296
GSFusion (gs) + DIFIX
21.81
0.772
0.273
GSFusion (gs) + DIFIX-finetune†
23.12
0.847
0.185
GSFusion (gs) + GSFixer
23.87
0.830
0.251
GSFusion (mesh+gs) + GSFixer
25.98
0.845
0.219
Table 6. More comparisons of diffusion-based repair methods on
the Replcia dataset. † indicates that we fine-tuned this model based
on the original DIFIX model. The best result is highlighted in
bold, and the second-best is underlined. The text inside ( ) indi-
cates the format of the reconstruction used.
7. Additional Results
7.1. More DIFIX Variants
DIFIX and DIFIX-ref [35] are diffusion models pretrained
on 80k noisy-clean real image pairs created using their pro-
posed dataset curation strategies, whereas our GSFixer is
only pretrained on two synthetic datasets with randomly
added Gaussian noise and blur, followed by fine-tuning on
a small amount of clean captured data. To demonstrate the
effectiveness and efficiency of our training strategy, we also
1

<!-- page 12 -->
GT
Input (gs)
DIFIX
DIFIX-finetune†
GSFixer
SplaTAM
RTG-SLAM
GSFusion
SplaTAM
RTG-SLAM
GSFusion
ScanNet++
Replica
Figure 8. More qualitative comparisons of diffusion-based repair methods on the ScanNet++ and Replica datasets. All examples use only
3DGS reconstructions as the input source. Zoom in to better observe how GSFixer effectively removes artifacts and fills in large holes,
where both DIFIX and DIFIX-finetune† fail to produce satisfactory results.
fine-tune the original DIFIX model on the same scene data
for 800 iterations on the ScanNet++ dataset and 400 itera-
tions on the Replica dataset.
Due to the higher GPU memory demands of training DI-
FIX, we used an NVIDIA A40 GPU with 48GB VRAM
to get DIFIX-finetune, while our GSFixer is trained on a
24GB NVIDIA RTX 4500 Ada GPU. Results are presented
in Tab. 5 and Tab. 6. As expected, DIFIX-finetune shows
improved performance over the original DIFIX on both
datasets. However, it still falls short of GSFixer, particu-
larly in PSNR, which correlates with lower visual quality
and is clearly visible in Fig. 8. For instance, in the Scan-
Net++ dataset, the colorful floaters in the SplaTAM ex-
ample are slightly suppressed by DIFIX-finetune, but not
fully removed (also seen in the GSFusion example).
In
the RTG-SLAM example, DIFIX-finetune fills in the miss-
ing window area with content, but the result lacks the tex-
ture consistency achieved by GSFixer.
Similarly, in the
Replica dataset, although DIFIX-finetune reduces some ar-
tifacts compared to the original DIFIX, it still fails to inpaint
large visible holes, which is an essential capability for novel
view repair.
In conclusion, our fine-tuning protocol not only delivers
superior performance in challenging scenarios but also re-
quires significantly fewer computational resources and min-
imal dataset curation, making it both effective and efficient.
7.2. More Ablation Studies on Image Conditions
Although SplaTAM and RTG-SLAM do not produce
meshes, we reuse the mesh extracted from GSFusion to
render conditional images for fine-tuning and inference of
GSFixer.
As shown in Tab. 7, GSFixer conditioned on
dual-input consistently outperforms the single-input vari-
ant (conditioned only on 3DGS) across both datasets for
2

<!-- page 13 -->
Dataset
Method
PSNR↑
SSIM↑
LPIPS↓
ScanNet++
SplaTAM
23.03
0.791
0.311
SplaTAM (gs) + GSFixer
25.11
0.831
0.188
SplaTAM (mesh∗+gs) + GSFixer
25.12
0.832
0.185
RTG-SLAM
19.54
0.777
0.341
RTG-SLAM (gs) + GSFixer
24.80
0.824
0.204
RTG-SLAM (mesh∗+gs) + GSFixer
25.05
0.827
0.191
Replica
SplaTAM
23.82
0.833
0.267
SplaTAM (gs) + GSFixer
25.67
0.839
0.215
SplaTAM (mesh∗+gs) + GSFixer
26.49
0.845
0.198
RTG-SLAM
25.00
0.860
0.247
RTG-SLAM (gs) + GSFixer
26.27
0.843
0.228
RTG-SLAM (mesh∗+gs) + GSFixer
26.53
0.848
0.212
Table 7. More ablations of input image conditions on the Scan-
Net++ and Replica datasets. ∗denotes that the mesh reconstruc-
tion used for both SplaTAM and RTG-SLAM comparisons is bor-
rowed from the GSFusion method.
Dataset
Method
PSNR↑
SSIM↑
LPIPS↓
ScanNet++
SplaTAM
23.03
0.791
0.311
SplaTAM (mesh∗+gs) + GSFixer
25.12
0.832
0.185
SplaTAM (mesh∗+gs) + GSFix3D
25.21
0.836
0.218
RTG-SLAM
19.54
0.777
0.341
RTG-SLAM (mesh∗+gs) + GSFixer
25.05
0.827
0.191
RTG-SLAM (mesh∗+gs) + GSFix3D
25.39
0.837
0.233
Replica
SplaTAM
23.82
0.833
0.267
SplaTAM (mesh∗+gs) + GSFixer
26.49
0.845
0.198
SplaTAM (mesh∗+gs) + GSFix3D
27.07
0.862
0.218
RTG-SLAM
25.00
0.860
0.247
RTG-SLAM (mesh∗+gs) + GSFixer
26.53
0.848
0.212
RTG-SLAM (mesh∗+gs) + GSFix3D
27.18
0.868
0.236
Table 8. More comparisons of GSFixer and GSFix3D on the Scan-
Net++ and Replica datasets. ∗denotes that the mesh reconstruc-
tion used for both SplaTAM and RTG-SLAM comparisons is bor-
rowed from the GSFusion method.
SplaTAM and RTG-SLAM. These results reinforce the
trends observed in Sec. 4.4. For instance, in the ScanNet++
dataset (see Fig. 9), floaters persist in the SplaTAM exam-
ple and missing thin structures in the RTG-SLAM exam-
ple are successfully corrected when mesh input is included.
Similarly, in the Replica dataset (see Fig. 9), carpet textures
in the SplaTAM example and the shape of the vase in the
RTG-SLAM example are better preserved with the help of
complementary information from the mesh input, which is
otherwise lost in the 3DGS-only setting.
7.3. More GSFix3D Comparisons
To further demonstrate the flexibility of the GSFix3D
framework, we incorporate reconstructions from SplaTAM
and RTG-SLAM into our pipeline for novel view repair in
3D space. We reuse the extracted mesh from the GSFu-
sion method to enable the dual-input setting, allowing us
to fully leverage GSFixer’s potential. The results in Tab. 8
align with our main experiments on GSFusion presented in
Sec. 4.2.
The improvement in GSFix3D is attributed to
the multi-view constraints applied during the optimization
of 3D representations. Additional qualitative examples are
provided in Fig. 10.
7.4. More Real-World Tests
7.4.1
Self-collected Ship Data
In Fig. 11, we present additional novel view repair results
on our self-collected ship dataset. Since 3DGS is highly
sensitive to pose inaccuracies, erroneous poses from multi-
ple views can introduce shadow-like floaters in the scene,
resulting in more severe artifacts in novel views. Despite
this challenge, our methods, GSFixer and GSFix3D, suc-
cessfully learn the artifact distribution from the captured
training data through our proposed fine-tuning protocol, en-
abling effective removal in both the 2D image space and the
3D scene representation. This in-the-wild test further high-
lights the robustness of our approach.
7.4.2
Outdoor Scenario
To further demonstrate adaptability across scenes and re-
construction pipelines, we select a challenging real-world
outdoor scene from the FAST-LIVO dataset [41], covering
building exteriors and archway corridors. The data is cap-
tured with a hard-synchronized LiDAR and camera setup,
and we reconstruct the 3DGS scene using a LiDAR-Inertial-
Camera Gaussian Splatting SLAM system [11] (Gaussian-
LIC). Note that pose errors may still occur, producing inac-
curate maps and broken geometries. We fine-tune GSFixer
on the captured RGB data and the 3DGS renderings from
Gaussian-LIC. As shown in Fig. 12, our method manages
to fill in visual holes on the brick ground, recover distant
buildings and sky, and correct broken structures such as the
arch wall and deep corridor, further validating GSFix3D’s
robustness in challenging real-world scenarios.
3

<!-- page 14 -->
GT
Input (mesh*)
Input (gs)
GSFixer (gs)
GSFixer (mesh*+gs)
SplaTAM
RTG-SLAM
ScanNet++
Replica
SplaTAM
RTG-SLAM
Figure 9. More qualitative ablations of input image conditions on the ScanNet++ and Replica datasets. The mesh reconstruction used for
both SplaTAM and RTG-SLAM comparisons is borrowed from the GSFusion method. Zoom in to better observe how artifacts (highlighted
by yellow boxes) present in the single-input settings are effectively mitigated with the dual-input configuration.
GT
Input (mesh*)
Input (gs)
GSFixer
GSFix3D
SplaTAM
RTG-SLAM
ScanNet++
Replica
SplaTAM
RTG-SLAM
Figure 10. More qualitative comparisons between GSFixer and GSFix3D on the ScanNet++ and Replica dataset. Both mesh and 3DGS
reconstructions are used as input sources. Zoom in to better observe how the 2D visual improvements from GSFixer are effectively distilled
into the 3D space by GSFix3D.
4

<!-- page 15 -->
Reference GT
Input (gs)
GSFixer
GSFix3D
Figure 11. Novel view repair on self-collected ship data. Our method is robust to pose errors, effectively removing shadow-like floaters.
Reference GT
Input (gs)
GSFixer
GSFix3D
Figure 12. Novel view repair on a challenging outdoor scene from the FAST-LIVO dataset [41]. The initial reconstruction is generated
by a LiDAR-Inertial-Camera Gaussian Splatting SLAM system [11], which may contain pose errors and produce inaccurate maps. Our
method manages to repair those broken geometries to some extent.
5
