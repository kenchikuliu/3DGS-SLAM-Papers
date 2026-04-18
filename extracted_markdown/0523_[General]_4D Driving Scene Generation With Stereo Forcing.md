<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
4D Driving Scene Generation With Stereo
Forcing
Hao Lu∗, Zhuang Ma∗, Guangfeng Jiang, Wenhang Ge, Bohan Li,
Yuzhan Cai, Wenzhao Zheng, Yunpeng Zhang, Yingcong Chen†
Abstract—Current generative models struggle to synthesize dynamic 4D driving scenes that simultaneously support temporal
extrapolation and spatial novel view synthesis (NVS) without per-scene optimization. Bridging generation and novel view synthesis
remains a major challenge. We present PhiGenesis, a unified framework for 4D scene generation that extends video generation
techniques with geometric and temporal consistency. Given multi-view image sequences and camera parameters, PhiGenesis
produces temporally continuous 4D Gaussian splatting representations along target 3D trajectories. In its first stage, PhiGenesis
leverages a pre-trained video VAE with a novel range-view adapter to enable feed-forward 4D reconstruction from multi-view images.
This architecture supports single-frame or video inputs and outputs complete 4D scenes including geometry, semantics, and motion. In
the second stage, PhiGenesis introduces a geometric-guided video diffusion model, using rendered historical 4D scenes as priors to
generate future views conditioned on trajectories. To address geometric exposure bias in novel views, we propose Stereo Forcing, a
novel conditioning strategy that integrates geometric uncertainty during denoising. This method enhances temporal coherence by
dynamically adjusting generative influence based on uncertainty-aware perturbations. Our experimental results demonstrate that our
method achieves state-of-the-art performance in both appearance and geometric reconstruction, temporal generation and novel view
synthesis (NVS) tasks, while simultaneously delivering competitive performance in downstream evaluations. Homepage is at
PhiGensis.
Index Terms—4D Driving Generation, Novel View, Stereo Forcing
✦
1
INTRODUCTION
Building robust autonomous driving systems requires ac-
curate perception, prediction, and decision-making in dy-
namic environments [1]–[4]. To develop and evaluate such
systems at scale, driving simulators that can synthesize
diverse, realistic, and temporally consistent 4D scenes are
becoming increasingly essential. However, collecting and
annotating large-scale real-world driving datasets is both
expensive and time-consuming, often limited by sensor
coverage, weather, occlusion, and scene diversity. To ad-
dress these limitations, generative models have emerged
as a promising alternative, enabling the controllable and
scalable creation of synthetic driving scenes for perception
and planning tasks.
In particular, diffusion-based models have achieved re-
markable success in high-fidelity image and video gen-
eration, and are being actively explored for urban scene
simulation. Building on this success, several works [5]–[9]
have adapted diffusion models to urban street-view video
generation for autonomous driving, conditioning genera-
Hao Lu, Wenhang Ge, Yingcong Chen are with the Hong Kong Uni-
versity of Science and Technology (Guangzhou), Guangzhou, China (e-
mail: hlu585@connect.hkust-gz.edu.cn; gewenhang01@gmail.com; yingcon-
gchen@ust.hk).
Guangfeng Jiang is with the University of Science and Technology of China
(e-mail: jgf1998@mail.ustc.edu.cn).
Bohan Li is with Shanghai Jiao Tong University, Shanghai, China (e-mail:
bohan.li@sjtu.edu.cn).
Wenzhao Zheng is with the University of California, Berkeley (e-mail:
wenzhao@berkeley.edu).
Hao Lu and Zhuang Ma contributed equally to this work.
Corresponding author: Yingcong Chen.
tion on inputs such as text prompts, BEV maps, and object
bounding boxes. While effective at producing photorealistic
videos, these methods are tightly coupled to predefined
ego trajectories and restricted to fixed camera perspectives.
Consequently, they fail to support novel view synthesis
(NVS)—a capability critical for autonomous driving plan-
ning and simulation, which requires flexibility across di-
verse viewpoints.
While NeRF [10], [11] and 3D Gaussian Splatting
(3DGS)-based [12]–[16] methods enable novel view render-
ing, they inherently lack generative capability for new scene
content. To combine the advantages of both paradigms,
hybrid methods [8], [17] like DreamDrive [8] and Magic-
Drive3D [17] jointly model scene reconstruction and gen-
eration. Nevertheless, these methods still require per-scene
optimization—severely limiting their scalability and ability
to generalize across unseen driving scenes.
To overcome the limitations of per-scene reconstruction,
recent works [18]–[21] propose frameworks that leverage
depth information as a bridge, achieving both scene genera-
tion and novel view synthesis without explicit scene-specific
optimization. For instance, Stage-1 [18] and Gen3C [21]
utilize predicted depth to enable 4D-consistent scene syn-
thesis. InfiniCube [19] adopts a sequential multi-module
design, but suffers from accumulated errors across stages.
DiST-4D [20] aggregates LiDAR point clouds and applies
diffusion-based completion to generate depth-supervised
reconstructions. However, during inference, the quality of
predicted depth degrades—severely undermining the spa-
tiotemporal consistency of generated 4D scenes. These lim-
itations highlight the need for a more unified and robust
arXiv:2509.20251v1  [cs.CV]  24 Sep 2025

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
solution that ensures geometric fidelity throughout the gen-
erative process.
To address these limitations, we propose PhiGenesis,
a unified and efficient framework for geometry-aware 4D
generation that bridges video generation and 3D scene rep-
resentation in an end-to-end manner. Unlike prior methods
that rely on depth prediction or per-scene optimization,
PhiGenesis directly generates temporally continuous 4D
Gaussian splatting representations conditioned on historical
observations and future 3D trajectories. This is achieved
through a two-stage design that integrates the strengths of
pre-trained video diffusion models with feed-forward 4D
reconstruction pipelines.
In stage 1, we extend a pre-trained video variational
autoencoder (VAE) to support 4D scene reconstruction
by introducing a range-view adapter. This adapter fuses
multi-view features from the decoder layers to produce
4D Gaussian representations capturing geometry, seman-
tics, and motion—without any per-scene tuning. By doing
so, PhiGenesis effectively enables real-time, generalist 4D
reconstruction from single frames or video clips, support-
ing both monocular and multi-view inputs. In stage 2,
PhiGenesis incorporates a geometry-guided video diffusion
model that synthesizes future 4D scenes along a given tra-
jectory, conditioned on the reconstructed history. To enforce
temporal and geometric consistency in unseen views, we
introduce a novel technique called Stereo Forcing, which
dynamically modulates the generative denoising process us-
ing uncertainty-aware perturbations of the historical latent
features. This mechanism corrects geometric inconsistencies
by leveraging the implicit priors of the diffusion model
while preserving the physical coherence of the underlying
3D structure.
We validate our approach across multiple challeng-
ing benchmarks for autonomous driving scene generation,
including 4D reconstruction, novel view synthesis, and
trajectory-conditioned simulation. Our results demonstrate
that PhiGenesis achieves state-of-the-art performance across
tasks, with strong generalization and scalability. Our contri-
butions can be summarized as follows:
• We propose PhiGenesis, an end-to-end framework for
4D driving scene generation.
• We propose Stereo Forcing, a technique to enhance
geometric consistency by encouraging the model to
focus on depth-ambiguous regions during training.
• Extensive
experimental
evaluations
conducted
on
large-scale autonomous driving datasets confirm that
PhiGenesis attains state-of-the-art performance across a
diverse set of benchmarks.
2
RELATED WORK
2.1
Driving Generation
Recently, diffusion-based methods [22]–[24] have become
the dominant paradigm in image and video generation.
Building on this success, several works [5], [6], [8], [9], [18],
[25]–[28] have extended diffusion models to the generation
and reconstruction of autonomous driving scenes.
Video Driving Generation. The MagicDrive [5] generates
high-quality street-view videos by encoding multiple 3D
geometric signals as conditional inputs and employing a
cross-view attention mechanism to enhance frame-to-frame
consistency. Similarly, Panacea [6] and DriveDreamer [7]
leverage cross-frame modeling to improve temporal coher-
ence. However, these methods lack accurate and coherent
3D spatial representations.
4D Driving Generation. To address the need for 4D-aware
scene generation, MagicDrive3D [17] combines video gen-
eration with reconstruction pipelines to synthesize 4D un-
bounded scenarios. InfiniCube [19] introduces voxel-based
intermediate representations, while UniScene [29] leverages
occupancy grids to generate both LiDAR point clouds and
video frames, achieving multimodal consistency through
point cloud reprojection and diffusion-based completion.
DiST-4D [20] further improves novel view reconstruction
by generating scene clouds via depth maps, though it still
suffers from depth degradation during inference, limiting
4D consistency and realism.
To overcome these challenges, our PhiGenesis decodes
3D geometry using a dedicated 3DGS Adaptor and in-
troduces Stereo Forcing to explicitly regularize depth con-
sistency across frames. Unlike prior methods that rely on
voxel or occupancy-based intermediates and suffer from
depth quality degradation, PhiGenesis directly enhances
the geometric coherence of dynamic scenes, enabling high-
fidelity and consistent 4D generation without per-scene
optimization.
2.2
Scene Reconstruction
3DGS [30] has recently demonstrated impressive perfor-
mance in novel view synthesis and real-time rendering.
In autonomous driving, several methods [12]–[16] apply
3DGS for dynamic scene reconstruction. To further improve
generalization to unseen views, recent approaches [31]–[34]
incorporate generative priors. DriveDreamer4D [31] uses a
world model to generate new trajectory videos for joint
training with reconstruction models. ReconDreamer [32]
builds a restoration dataset from degraded reconstructions
and real sensor images to enhance novel view repair and
improve synthetic-to-real consistency. ReconDreamer++ [33]
further introduces Novel Trajectory Deformation Networks
(NTDNet) to bridge domain gaps through learnable spatial
deformations.
Despite these advances, existing methods still rely on
per-scene optimization and suffer from geometric incon-
sistencies in world model predictions. To overcome these
limitations, we propose a generative 4D driving scene
framework that synthesizes dynamic scenes without scene-
specific optimization, leveraging a 3D cache mechanism to
enhance both temporal and spatial coherence.
2.3
Diffusion Forcing
Traditional diffusion models [22]–[24] apply uniform noise
across all tokens. Diffusion Forcing (DF) [35] introduces per-
frame noise scheduling within causal state-space models,
enabling flexible denoising and unifying next-token pre-
diction with diffusion. CausVid [36] extends DF to causal
transformers, mitigating error accumulation in autoregres-
sive video generation via causal attention, and improving
efficiency and stability for long-form video synthesis. More
recently, Song et al. [37] propose the non-causal Diffusion

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
Forcing Transformer (DFoT), which removes causal con-
straints to allow arbitrary-length history conditioning. By
introducing History Guidance, DFoT enhances temporal
consistency and supports robust long-sequence generation,
outperforming prior approaches in handling variable-length
contexts and complex temporal dynamics.
While these advances have shown strong potential in
video modeling, the idea of adaptive noise scheduling has
not yet been explored for 4D scene generation. In this work,
we adapt the principles of DF to the 4D setting: by injecting
different levels of noise into different depth regions, we
leverage the generative model’s geometric completion abil-
ity to recover consistent and accurate depth across frames,
thereby improving 4D spatial-temporal coherence.
3
PRELIMINARY
Our PhiGensis is based on the video diffusion model,
generalized Gaussian reconstruction and diffusion forcing
technology.
Video Diffusion Models (VDMs) are a subset of gen-
erative models tailored for video generation. Their core
architecture involves three components: an encoder maps
raw video data D to latent representations z(0) = E(I);
a diffusion model iteratively denoises N(0, I) noise; and
a decoder reconstructs generated videos ˆV
= D(z(0)).
To enhance VDM training stability and sampling speed,
Rectified Flow is adopted. It defines a linear sample path
between data distribution p0 and noise distribution p1:
z(t) = (1 −t)z(0) + tϵ
(ϵ ∼N(0, I)),
(1)
where z(t) denotes the sample state at time step t ∈[0, 1].
A velocity prediction loss optimizes the network Θ (param-
eterizing velocity vΘ):
LRF = Ez(0),ϵ,t ∥vΘ(z(t), t) −(z(0) −ϵ)∥2 .
(2)
Generalizable Guassian Model. Our method builds on
the success of generalizable Gaussian models (GGM) [38]–
[40]. GGM accepts a set of images and directly outputs
a 3D or 4D representation of the video, especially, a set
of Gaussian points P. Each Gaussian is represented by 14
parameters, including a center u ∈R3, a scaling factor
s ∈R3, a quaternion rotation q ∈R4, an opacity α ∈R, and
a color feature c ∈R3. GGM presents a novel opportunity
for real-time scene reconstruction in autonomous driving.
However, end-to-end 4D generation for driving has rarely
been explored.
Diffusion Forcing. Traditionally, diffusion models are
trained using uniform noise levels across all tokens. DF [35]
proposes training sequence diffusion models with indepen-
dently varied noise levels per frame. Although DF provides
theoretical and empirical support for this approach, their
work focuses on causal, state-space models. CausVid [36]
builds on DF by scaling it to a causal transformer, creating
an autoregressive video foundation model. History guided
forcing [41] extends the flexibility of DF by developing both
the theory and architecture for non-causal, state-free mod-
els, enabling new, unexplored capabilities in video genera-
tion. Self forcing [42] trains autoregressive video diffusion
models by simulating the inference process during train-
ing, performing autoregressive rollout with KV caching.
TABLE 1: Summary of Mathematical Symbols in the
Manuscript
Symbol
Mathematical Form
Definition
P
Set of Gaussian points
A set of 3D Gaussian points output by the Generalizable
Gaussian Model (GGM), representing the core 3D/4D
scene structure.
I
I = {Iv,tar | v ∈V, t ∈Tobs}
Input set of multi-view images, where Iv,tar denotes the
image from view v at observation time t.
V
Index set of camera views
Set
of
indices
for
distinct
camera
views
(e.g.,
front/left/right cameras in autonomous driving).
Tobs
Index set of observation times
Set of indices for time steps with observed input images
(historical time steps for 4D generation).
Tfuture
Index set of future times
Set of indices for time steps where 4D scenes are gener-
ated (future time steps).
Kv
Camera intrinsic matrix
Intrinsic parameters of camera v (focal length, principal
point, skew coefficients).
Rv,t, Tv,t
Rotation matrix and translation vector (the camera extrinsics)
Time-varying extrinsics of camera v at time t: Rv,t (3×3
rotation) and Tv,t (3D translation) define camera pose
in world coordinates.
Iv,ren
Rendered multi-view video frame
Image rendered from 4D Gaussian set G for view v (used
as geometric guidance).
E
Pre-trained video VAE encoder
Encoder of the video VAE (maps 2D images to latent
representations).
D
Pre-trained video VAE decoder
Decoder of the frozen video VAE (maps latent represen-
tations to 2D images).
z
z ∈Rh×w×c×v×t
Latent tensor from the pre-trained video VAE encoder
(dimensions: height h, width w, channel c, view v, time
t).
zv,ren, zv,tar
Latent vectors
zv,ren = E(Iv,ren): Latent of rendered image Iv,ren;
zv,tar = E(Iv,tar): Latent of target image Iv,tar.
zn
v,tar
Noisy targeted latent vectors
Noise latent of zv,tar: zn
v,tar = (1 −t) · zv,tar + t · ϵ (ϵ
is standard Gaussian noise).
ϵ
Gaussian noise sample
Random noise sampled from N(0, I) (used to corrupt
latents in diffusion training).
LRF
Flow Matching loss
Loss for training the multi-view video diffusion model.
t
Noise level
the diffusion noise level (t ∈(0, 1); t = 0 = clean, t = 1
= full noise).
u
Uncertainty map
2D map encoding geometric uncertainty (options: ran-
dom noise, classification entropy, localization potential).
ω
Stereo forcing scale
The stereo forcing scale of weighting geometric contri-
bution.
SF(·)
Stereo Forcing function
Function
integrating
uncertainty
into
latents:
SF(u, z) = u·z (balances clean latents and uncertainty).
Although these methods alleviate the exposure bias in long
video generation, they ignore spatial geometric uncertainty.
4
METHOD
We present PhiGenesis, a novel framework that extends
video generation techniques to 4D generation. Given a set
of input images I = {Iv,tar | v ∈V, t ∈Tobs} captured from
views V over observation times Tobs, along with camera
intrinsics Kv and time-varying extrinsics [Rv,t|Tv,t], our
framework supports both single and multi-frame inputs.
Conditioned on future 3D trajectories Γfuture = {γt}T
t=0,
we generate a 4D scene representation G = {Gt}T
t=0 using
Gaussian Splatting. This allows for efficient, continuous ren-
dering of dynamic scenes across time. In Sec. 4.1, we explain
how to train the video generation model to generate 4D
Gaussian representation, ultimately enabling 4D generation,
termed as PhiGenesis. Further, PhiGenesis is improved by
Stereo Forcing is proposed to solve the geometric exposure
bias, which will be discussed in Sec. 4.2.
4.1
PhiGenesis
PhiGenesis is designed to generate high-quality 4D scenes
that encapsulates both the geometry and the appearance
information. To achieve this, we need to solve two problems:
(1) how to extend the video generation model to 4D genera-
tion, and (2) how to utilize historical geometric information
to improve temporal consistency.
4.1.1
Generalist 4D Reconstruction
Most 4D generation methods first generate some videos
and then optimize per scene, which is the time consump-
tion at the hour level [8], [17], [43]. Fortunately, the feed-
forward reconstruction method can directly convert multi-
view images to 4D Gaussian representation, which provided
the opportunity for real-time 4D reconstruction [39], [40],
[44]. Based on this, PhiGenesis try to extend the video VAE

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
Fig. 1: Overall framework of the proposed PhiGenesis. (1) Stage 1 aims to train a 4D reconstrucion generalist. Multi-view
images are first fed into a fixed, pre-trained video VAE. The multi-scale features extracted from the decoder of the video
VAE are then passed through a range-view adapter to reconstruct the complete 4D scene (including optical flow, etc.). (2)
The objective of Stage 2 is to enhance geometric consistency generation. We project the 4D scenes reconstructed based on
history onto the future trajectory perspective. The rendered video is denoised according to geometric uncertainty by stereo
forcing and then sent to the pre-trained encoder to obtain the rendered multi-view latent. The rendered multi-view latent
and noise map are fed into the multi-view video diffusion model to generate the latent of the multi-view video of the target
trajectory. The latent of multi-view video is fed into the pre-trained video decoder and the GS adapter of range-view to
generate the 4D scene corresponding to the target trajectory.
Multi-Level Features
View 2 Latent 
Features
View n Latent 
Features
......
View 1 Latent 
Features
Multi-Level Features
Multi-Level Features
Range-View Features
xyz
rgb
alpha
scale
rotation
∆xyz
∆rotation
segmentaion
Range-View Adapter
Multi-Level 
Feature Fusion
VAE Decoder
Fig. 2: The framework of the range-view adapter. The
multi-view latent is fed into the decoder of the pre-trained
video. The multi-scale features of the decoder are concate-
nated in the form of range views and fused through a feature
pyramid form to predict the Gaussian representation.
model to 4D reconstruction ability, which can generate 4D
geometry and semantic information simultaneously.
Specifically, PhiGenesis feed single-image or video se-
quences (I = {Iv,tar | v ∈V, t ∈Tobs}) captured from
multi-views V into a fixed video VAE [45] to get latent
feature e ∈h × w × c × v × t. Here, v and t refer to the
number of perspectives and the temporal dimension after
compression. It is worth noting that the pre-trained video
VAE supports image or video input, adapting to diverse
temporal contexts [45]. Then the multi-view latent feature
e ∈h×w ×c×v ×t is fed into the decoder of video VAE D.
The native decoder reconstructs each view separately, which
causing the lack of multi-view information. Therefore, the
range-view representation [46], widely used in autonomous
driving, is selected as the way to fuse multi-view images to
predict the 4D Gaussian.
As shown in Fig. 2, the range-view adapter integrates
features from different view at different layers, and then
fused the features of different layers to obtain the final
feature. The range-view adapter employs two convolutional
blocks to convert features into segmentation c ∈RC, depth
regression dr ∈R1, alpha a ∈R1, scale r ∈R3, rotation
r ∈R3 and optical flow [∆x, ∆y, ∆z]. The activation func-
tions for RGB color, alpha, scale, and rotation are consistent
with those in [39]. Through the range-view adapter, we can
fully fuse multiview images and output Gaussian repre-
sentations and their corresponding semantic information.
During this training process, the video VAE is frozen, and
only the range-view adapter is trained. During the training
process, RGB rendering, depth and segmentation supervi-
sion Lrender will be used. In addition, the dynamic and
static separation strategy is also employed [39].
Discussion: Many research have studied the feed-
forward 4D reconstruction algorithm in driving tasks [8],
[17], [43]. Unlike them, our work focuses on extending
the 4D reconstruction capabilities of the video generation
models. Furthermore, the principle of our design in this part
is the simplest and necessary way to integrate multi-view
videos to predict 4D scenes. There are also many papers on
4D generation in the driving field here: (1) DiST-4D only
predicts the depth map simultaneously during generation
rather than the complete 3D representation (specifically,
the Gaussian representation) [20]. (2) InfiniCube trains the
video diffusion model and feed-forward method separately,
and then converts the single-view video into a Gaussian
representation in a multi-stage concatenation manner [19].
In contrast, our approach extends the decoder’s capability

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
to directly generate the 3D Gaussian representation in an
end-to-end manner.
4.1.2
Geometric-Guided Generation
Based on the above steps, PhiGensis can convert the given
multi-view images or videos into 4D scenes. PhiGensis
further uses the existing 4D scene guidance to generate long
time 4D scenes, termed Geometric-Guided Generation. From
another perspective, PhiGensis utilize the powerful painting
prior of the video diffusion model to expand the existing 3D
scenes to generate future scenes.
Initially, given a set of future 3D trajectories Γfuture =
{γt}T
t=0, we can render the multi-view video Iv,ren from the
4D scene representation G = {Gt}T
t=0. Then, we employ
the frozen VAE encoder E to encode the rendered multi-
view video Iv,ren and the multi-view target video Iv,tar,
resulting in the latent representation zv,ren = E(Iv,ren) and
zv,tar = E(Iv,tar). The rendered multi-view features zv,ren
is regarded as the geometric condition of the generative
model. For training, we concatenate the noisy target video
latent zn
v,ren (it began with a complete Gaussian noise)
and the noisy rendered multi-view features zn
v,ren along
the channel dimension, and feed these into the multi-view
video diffusion model [9]. Here, the noisy target video latent
zn
v,tar = (1 −t) ∗zv,tar + t ∗ϵ is constructed by adding noise
ϵ sampled from a Gaussian distribution, which began with a
pure Gaussian noise map. The rendered multi-view features
zsf
v,ren = SF(zv,ren) is processed by stereo forcing SF, i.e.,
zsf
v,ren = SF(zv,ren). The stereo forcing will be disscussed
in Sec. 4.2. For our model, we adopt the VAE architecture
from HunyuanVideo [45] and leverage OpenSora V21 for
diffusion pre-training. Additionally, the multi-view cross-
attention fusion mechanism integrated into the diffusion
model draws inspiration from MagicDirveV2 [9]. Our train-
ing pipeline use flow matching loss LRF to generate the
multi-view videos. This method ensures the integration
of geometric representation and latent representation, ulti-
mately promoting the generation of consistent and visually
coherent predictive 3D scenes.
Discussion: Many concurrent research paper also stud-
ied the 3D-Guided Generation [21], [47]. Our method has
three advanced aspects compared with these two concurrent
works: (1) These methods rely on existing geometric recon-
struction tools such as MonST3R [48], while our algorithm
can directly reconstruct historical 3D information. (2) Our
method supports multi-view videos, while these methods
only support monocular videos.
4.2
PhiGenesis with Stereo Forcing
Although geometry-guided history can effectively retains
historical information, the biggest problem is that its depth
is inaccurate in unseen scene, leading to poor geometric
consistency in the generated 4D scene, termed the geometric
exposure bias as shown in Fig. 3. Inspired by DF [35],
[37], we attempt to explore a 4D geometrical relationship
of the denoising process. To achieve this, we propose Stereo
Forcing, which enhances the quality of 4D generation while
maintaining consistency. The core idea is to let the denosing
1. https://github.com/hpcaitech/Open-Sora
shift
Rendered 
Image
Ground Truth
Image
shift
Fig. 3: The geometric exposure bias. Due to inaccurate geo-
metric estimation, the rendered view is not strictly accurate.
Such inaccurate renderings used as geometry guidance will
lead to the degradation of consistency. This requires addi-
tional information to further guide the Diffusion process to
correct inconsistencies.
process know the geometry uncertainty, and then fix the
inaccurate parts of the projection based on the generation
prior. Specifically, the score is given by:
∇p(zv,tar)+ω
∇p
 zv,tar|SF(u, zv,ren)
−∇p(zv,tar)
, (3)
where u ∈(0, 1) is uncertainty map. This approach differs
from conventional CFG and HG [37] in two ways: (1)
The condition zk
v,ren is rendered from the 4D Gaussian,
which retain geometric information to alleviate generative
degradation. (2) Stereo forcing function SF introduces ad-
ditional uncertain information. The further question is how
to determine the geometric uncertainty to determine stereo
forcing function SF. Here, we have attempted three com-
mon uncertain maps u: random noise, entropy of depth
classification, and the localization potential [49].
The localization potential comprehensively and theoret-
ically analyzes and quantitatively presents the uncertain-
ties of temporal fusion, multi-view fusion, and the camera
parameters of the camera on geometric estimation [49].
Inspired by his outstanding effect in perception, we also
applied this uncertain map to the denosing process of
4D generation. The uncertain indicator can also be added
to stereo forcing function SF(u, zv,ren) = u · zv,ren. We
emphasize that the form of uncertainty and the form of
stereo forcing can be further enhanced, and here we only
choose the most concise way to prove the effectiveness of
our method.
4.3
Training and Reasoning
Training. Generating high-quality 4D scenes is non-trival.
We adopted a two-stage training strategy: (1) Building 4D
reconstruction capability. (2) Building Generative capacity. It
is worth mentioning that to achieve high-quality 4D genera-
tion, we adopted a mixed-resolution training process [9]. For
the construction of 4D reconstruction capabilities, we have
implemented a hybrid strategy of high-resolution single-
frame and low-resolution multi-frame. For the construction
of generation capabilities, we first adopt a process from low
resolution to high resolution to accelerate convergence [9].
The details of the training stage are shown in Tab. 2.

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
TABLE 2: Model Training Stages. For a fair comparison, we conducted comparisons using different resolutions on different
datasets. The resolutions of R1 at waymo and nuScene are 160 × 240 and 224 × 400 respectively. The resolutions of R2 at
waymo and nuScene are 576 × 1024 and 424 × 800 respectively.
Training Configuration
Stage 1
Stage 2.1
Stage 2.2
Resolution
R1, R2
R1
R2
Fine-Tuned Model
Range-View Adapter
Multi-View Video Diffusion
Multi-View Video Diffusion
Constraint Loss
Lrender
LRF
LRF
Training Steps
10,000
50,000
50,000
Learning Rate
2 × 10−4
1 × 10−4
5 × 10−5
LR Scheduler
Cosine Annealing
Cosine Annealing with Restarts
Cosine Annealing with Restarts
Weight Decay
1 × 10−4
5 × 10−5
5 × 10−5
Gradient Norm Clip
1.0
1.0
1.0
Optimizer
AdamW
AdamW
AdamW
TABLE 3: Comparison with state-of-the-art methods on the Waymo and nuScenes datasets. Metrics reported include
Peak Signal-to-Noise Ratio ( PSNR), Structural Similarity Index Measure (SSIM), Depth RMSE (D-RMSE), Learned
Perceptual Image Patch Similarity (LPIPS), and Pearson Correlation Coefficient (PCC). Higher values are better for
PSNR, SSIM, and PCC; lower values are better for D-RMSE and LPIPS.
Method
Waymo
nuScenes
PSNR↑
SSIM↑
D-RMSE↓
PSNR↑
SSIM↑
LPIPS↓
PCC↑
Per-Scene Optimization Methods
EmerNeRF [11]
24.51
0.738
33.99
18.45
0.582
0.502
0.061
3DGS [50]
25.13
0.741
19.68
19.67
0.603
0.436
0.094
PVG [51]
22.38
0.661
13.01
18.98
0.567
0.481
0.072
DeformableGS [52]
25.29
0.761
14.79
20.12
0.622
0.422
0.105
Generalizable Feed-Forward Methods
LGM [53]
23.59
0.691
8.02
22.15
0.672
0.318
0.342
GS-LRM [54]
25.18
0.753
7.94
23.41
0.703
0.273
0.598
pixelSplat [55]
22.65
0.684
11.03
21.51
0.616
0.372
0.001
MVSplat [56]
23.42
0.701
9.88
21.61
0.658
0.295
0.181
SCube [57]
25.72
0.783
5.62
23.85
0.721
0.258
0.651
DrivingForward [38]
26.32
0.774
5.79
24.32
0.732
0.229
0.766
Omni-Scene [44]
26.46
0.786
5.66
24.27
0.736
0.237
0.804
STORM [40]
26.38
0.794
5.48
24.56
0.752
0.217
0.788
PhiGensis
27.52
0.833
5.14
25.92
0.801
0.189
0.847
Inference. Our model supports single-frame or multi-
frame input to generate 4D scenes. Specifically, both single-
frame and multi-frame videos can be reconstructed into 3D
scenes, and geometric conditions can be rendered based
on future trajectories as guidance. Moreover, the generated
4D scene can be used to predict and render subsequent
geometric states, enabling a rollout-style generation process.
5
DATASET
Our experimental evaluations were performed on two
benchmark datasets: the Open Waymo Dataset (hereafter
referred to as OWD) [58] and the nuScenes dataset [59].
For OWD, we utilized its dedicated training set for model
training and its validation set for performance testing,
following standard evaluation protocols. Regarding the
nuScenes dataset, we adjusted the annotation frequency of
keyframes—originally provided at 2Hz—to 12Hz through
interpolation, consistent with the approach adopted in prior
research [5], [60]. We strictly followed the official data
partitioning scheme for nuScenes, using 700 videos as the
training corpus and 150 videos for validation. For the gen-
eration of semantic map ground truth, we first retrained
the SegFormer model [61] and then applied the fine-tuned
model to infer pseudo semantic maps for each individual
frame. To acquire dense depth ground truth, we aggregated
LiDAR point clouds of static scene components across the
entire scene, followed by projecting these temporally fused
point clouds onto each frame; this process yielded sparse
LiDAR depth maps. In line with conventional depth com-
pletion methodologies, we employed the DepthLab tool [62]
to densify these sparse depth maps.
6
EXPERIMENT
It is difficult to directly and quantitatively evaluate the
ability of 4D generation without the 4D label. In this section,
we attempt to evaluate our algorithm from two aspects: (1)
Is it efficient to endow the video pre-trained VAE with 4D
reconstruction capabilities? (2) Can this method generate
higher-quality scenes with controllable trajectories?
Apparent reconstruction ability. We verified the 4D
reconstruction performance through two dimensions: (1)
The model’s ability to reconstruct input data after the first

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
RGB
Segmentation
Normal
Depth
Fig. 4: Qualitative comparison on reconstruction ability. We visualized the reconstructed RGB images, segmentation,
normal and depth maps, which demonstrated extremely high quality.
TABLE 4: Quantitative Evaluation of Generated Depth
Maps. We compare our method with state-of-the-art sur-
round depth estimation approaches [20], [63] on the Waymo
dataset.
Method
Abs. Rel. ↓
RMSE ↓
δ < 1.25 ↑
δ < 1.252 ↑
SD [63]
0.25 / 0.26
5.98 / 6.06
0.72 / 0.68
0.89 / 0.87
DiST-4D [20]
0.18 / 0.23
4.90 / 5.13
0.82 / 0.79
0.93 / 0.92
PhiGensis (Ours)
0.16 / 0.19
4.58 / 4.81
0.86 / 0.82
0.95 / 0.96
TABLE 5: Quantitative comparison of novel view synthe-
sis. We evaluate FID and FVD across different viewpoint
shifts (±1m, ±2m, ±4m).
Method
Shift ±$1m
Shift ±$2m
Shift ±$4m
FID ↓
FVD ↓
FID ↓
FVD ↓
FID ↓
FVD ↓
PVG [64]
48.15
246.74
60.44
356.23
84.50
501.16
EmerNeRF [11]
37.57
171.47
52.03
294.55
76.11
497.85
StreetGaussian [14]
32.12
153.45
43.24
256.91
67.44
429.98
OmniRe [15]
31.48
152.01
43.31
254.52
67.36
428.20
FreeVS [65]
51.26
431.99
62.04
497.37
77.14
556.14
DiST-4D
10.12
45.14
12.97
68.80
17.57
105.29
Ours
9.80
43.80
11.71
67.54
15.51
103.13
Point Cloud
Multi-View RGB
Fig. 5: Scene editing. Various 3D assets can be inserted into
the generated scene.
training stage; (2) Its reconstructed generation capability
following the second training stage. For the first stage,
we compared our method against two broad categories
of baselines: per-scene optimization methods and gener-
alizable feed-forward models. For per-scene optimization,
we evaluated against both NeRF-based and 3DGS-based
approaches, including EmerNeRF [11], 3DGS [50], PVG [51],
and DeformableGS [52]. Since LiDAR data is unavailable
during testing in our experimental setup, these baselines
were also run without LiDAR supervision to ensure a fair
comparison. For the generalizable feed-forward category,
we compared with state-of-the-art large-scale reconstruc-
tion models, such as LGM [53], GS-LRM [54], SCube [57],
DrivingForward [38], and STORM [40]. The rendered image
resolutions were set to 160×240 for the Waymo dataset
and 224×400 for the nuScenes dataset, consistent with the
settings in STORM [40] and OmniScene [44].
Quantitative results are presented in Table 3. On the
Waymo dataset, our method (“PhiGensis”) achieves a PSNR
of 27.52, which is 0.86 higher than STORM (26.38) and 0.46
higher than Omni-Scene (26.46)—the two top-performing
feed-forward baselines. In terms of depth accuracy (D-
RMSE), our method’s score of 5.14 is 0.34 lower than
STORM’s 5.48, reflecting more precise geometric recon-
struction. On the nuScenes dataset, PhiGensis maintains its
lead: its PSNR of 25.92 is 1.36 higher than STORM (24.56),
and its LPIPS score of 0.189 is 0.028 lower than STORM’s
0.217—demonstrating superior visual fidelity. A key obser-
vation is that generalizable feed-forward models achieve
performance comparable to per-scene optimization methods
in terms of photorealism, geometric accuracy, and inference
speed—both in dynamic regions and full images. Notably,
our method outperforms other generalizable feed-forward
models in modeling scene dynamics and processing multi-
timestep, multi-view data. This superior reconstruction per-
formance can be attributed to our utilization of a pre-trained
video VAE, which provides robust spatiotemporal feature
priors that simpler feed-forward architectures lack. With
its outstanding reconstruction performance, we can insert
objects into scenes, as shown in the Fig. 5.
Geometry reconstruction ability. Following the evalua-
tion protocol in [20], we further report the geometric predic-
tion performance on the Waymo dataset. As shown in Table
4, the depth maps generated by our method (“PhiGensis
(Ours)”) achieve performance comparable to—and in most
cases superior to—state-of-the-art surround depth estima-
tion approaches like SurroundDepth [63] and DiST-4D [20].
For the Absolute Relative Error (Abs. Rel.), our method
achieves 0.16/0.19, which is 0.02/0.04 lower than DiST-
4D (0.18/0.23) and 0.09/0.07 lower than SurroundDepth
(0.25/0.26), indicating smaller erros between predicted and

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
Original GT
Ours
DiST-4D
EmerNeRF
OmniRe
PVG
StreetGS
Fig. 6: Qualitative comparison on spatial NVS. Comparison between Ours and state-of-the-art methods on spatial
NVS. Please zoom in for the best observation. The red part indicates that our method has outstanding advantages in
reconstructing object details, lane lines and distant views.
ground-truth depth. This confirms that the 4D generalist
model trained in our first stage possesses excellent geo-
metric reconstruction ability. As visualized in Figure 4, our
model produces clear, geometrically consistent reconstruc-
tions, further validating its strong reconstruction capability.
New view synthesis ability. Adopting the evalua-
tion methodology from [20], [65], we focus on assess-
ing our model’s performance on novel trajectories using
Fr´echet Inception Distance (FID) and Fr´echet Video Dis-
tance (FVD). Specifically, we apply lateral offsets of τ ∈
{±1m, ±2m, ±4m} to the camera viewpoint and compute
FID/FVD between the synthesized RGB images of the novel
trajectory and the ground-truth images of the original trajec-
tory. Table 5 compares the NVS results of our method with
existing approaches under these shifted viewpoints.
At a τ ∈{±1m} shift (small viewpoint change), our FID
is 9.80 and FVD is 43.80—slightly outperforming DiST-4D
(10.12/45.14) and far surpassing traditional per-scene meth-
ods like PVG (48.15/246.74) and EmerNeRF (37.57/171.47)
by a large margin. For the larger τ ∈{±4m} shift (more
challenging viewpoint), our FID of 15.51 and FVD of
103.13 remain the lowest among all methods, outperform-
ing DiST-4D (17.57/105.29) and exceeding StreetGaussian
(67.44/429.98) by over 50 points in FID and 300 points in

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
Left 2m
Right 2m
Original
Left 2m
Right 2m
Original
Time
Fig. 7: New view synthesis for long videos. In waymo’s dataset, we visualized the three views generated from long-time
series videos. Under long video generation, our method still maintains high quality after shifting the observation view
(please zoom in for the best view).
TABLE 6: Quantitative Comparison of Cross-Temporal
Rendering on the Waymo Dataset. Metrics are calculated
for synthesized views at frames T + 5 and T + 10 given
frame T as input. Higher values are better for PSNR and
SSIM; lower values are better for LPIPS.
T + 5
T + 10
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PixelNeRF [66]
15.21
0.52
0.64
14.61
0.49
0.66
PixelSplat [55]
20.11
0.70
0.60
18.77
0.66
0.62
DUSt3R [67]
17.08
0.62
0.56
16.08
0.58
0.60
MVSplat [56]
20.14
0.71
0.48
18.78
0.69
0.52
MVSGaussian [68]
16.49
0.70
0.60
16.42
0.60
0.59
SCube [57]
19.90
0.72
0.47
18.78
0.70
0.49
Infinicube [19]
20.80
0.73
0.42
19.93
0.72
0.45
PhiGensis
21.41
0.75
0.38
20.12
0.74
0.42
FVD. The quantitative results demonstrate that our method
achieves substantial improvements in both FID and FVD
metrics. When using conditions generated from real images
as inputs, ours achieves several times better performance in
these metrics, highlighting its strong capability in NVS tasks
based on real data. As illustrated in Figure 6, our proposed
method significantly outperforms previous reconstruction-
based models in synthesizing novel spatial viewpoints for
RGB images—with results nearly free of visual degradation
and artifacts like blurring or geometric distortion.
For the second training stage, we adopt the experimental
setup from SCube [19], [57] to assess reconstruction quality:
given 3 front-view inputs from frame T, we synthesize novel
views at frames T + 5 and T + 10, then evaluate perfor-
mance using PSNR, SSIM, and LPIPS. Quantitative results
are presented in Table 6, where our method (“PhiGensis”)
outperforms all baseline approaches across all metrics and
both time steps. Notably, our method not only leverages the
prior knowledge generated during training but also benefits
from the “Stereo Forcing” mechanism, which contributes to
synthesizing higher-quality videos with stronger temporal
consistency—explaining its consistent superiority over base-
lines, especially at the more distant frame T + 10.
Generation ability. To assess the performance of our
model in long video generation, we follow the experimental
paradigm proposed in [19]: we take the initial frames of
test sequences from the Waymo Open Dataset as inputs to
various video generation models, and compute the FID for
generated video frames at different temporal indices. This
metric is employed to quantify the degradation of video
quality over extended generation horizons.
TABLE 7: FID Values of Different Methods Across Various
Video Frame Lengths. Lower FID values indicate higher
consistency between generated videos and real-world visual
distributions.
Method
FID@50
FID@100
FID@150
FID@200
Vista [26]
130.2
160.4
195.1
224.8
Panacea [69]
109.7
140.3
169.8
201.5
InfiniCube [19]
85.5
95.1
105.3
115.7
Dist-4D [20]
72.3
84.2
98.3
102.7
Ours wo SF
60.4
73.8
92.6
91.2
Ours
55.2
68.6
82.8
88.3
As presented in Table 7, our full model (“Ours”) con-
sistently achieves the lowest FID scores across all evaluated
frame lengths (FID@50 to FID@200), demonstrating superior
and sustained visual quality over long generation horizons.
In contrast, baseline methods exhibit pronounced quality
degradation as the video length extends beyond 100 frames:
for instance, Vista’s FID increases by 72.5 % from 130.2
(FID@50) to 224.8 (FID@200), while Panacea’s FID rises by
83.6% over the same range. Even state-of-the-art methods
like InfiniCube and Dist-4D show a 35.3% and 42.1% FID
increase, respectively, from FID@50 to FID@200. This su-
perior performance underscores the efficacy of our world-
guided video generation strategy and guidance buffer de-
sign, which effectively mitigates the cumulative autoregres-
sive errors that typically plague long video generation tasks.
Among the buffer components, the semantic buffer plays a
pivotal role in preserving high-level video quality, while the
coordinate buffer resolves fine-grained ambiguities arising
from motion-induced scene transformations.
Ablation Experiments. PhiGensis makes two key con-
tributions: (1) 4D Generation Pipeline: As validated in
the previous chapter through comparisons with the SOTA
methods; (2) Stereo Forcing (SF). To verify the effectiveness
of our method in long video generation, we first present
quantitative results in Tab. 7 and further provide qualitative
visualizations of video generation results with and without
SF in Figure 3. Notably, Stereo Forcing enables diffusion
models to better leverage geometric priors for learning
consistency by predefining specific uncertainty metrics. We

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
Time
w/. Stereo Forcing
w/o. Stereo Forcing
Fig. 8: Qualitative ablation on stereo forcing. The stereo forcing can effectively reduce generative degradation and improve
geometric consistency.
TABLE 8: Ablation Study on Different Uncertainty Metrics for Stereo Forcing. Lower FID values indicate better video
generation quality and consistency.
Uncertainty Metric
FID@50
FID@100
FID@150
FID@200
Baseline
60.4
73.8
92.6
91.2
Random Noise
58.7
70.2
89.5
90.1
Entropy
57.1
69.3
86.4
89.2
Localization Potential [49]
55.2
68.6
82.8
88.3
TABLE 9: Quantitative Evaluation of Video Generation for
Perception and Planning. We use UniAD [2] to assess object
detection, BEV segmentation, and L2 open-loop planning
errors on generated videos.
Method
Detection ↑
BEV Segmentation ↑
L2 ↓
NDS
mAP
Lan.
Dri.
Div.
Cro.
1.0
2.0
3.0
Ori. GT
49.85
37.98
31.31
69.14
25.93
14.36
0.51
0.98
1.65
MD [5]
28.36
12.92
21.95
51.46
17.10
5.25
0.57
1.14
1.95
DA [70]
30.03
16.06
26.14
59.37
20.79
8.92
0.56
1.10
1.89
Dist-4D [20]
32.44
15.63
26.80
60.32
21.69
10.99
0.56
1.11
1.91
Ours [20]
34.44
18.06
28.80
62.32
23.69
12.99
0.55
1.09
1.85
have tested three types of such metrics in our experiments:
random noise, entropy of depth classification, and localiza-
tion potential [49], with the corresponding ablation results
detailed in Table 8. PhiGensis demonstrates that incorporat-
ing additional geometric uncertainty information helps mit-
igate the degradation tendency of diffusion models, which
confirms the high effectiveness of our chosen metric—the
localization potential, as it has been shown to capture
the impacts of temporal fusion, multi-view fusion, and
camera parameters on geometric estimation. Furthermore,
our method theoretically supports multiple approaches to
integrating indicators of geometric adequacy and geometric
uncertainty; here, we focus on validating the feasibility of
this integration framework.
Downstream applications. Beyond evaluating the vi-
sual quality of generated images and videos, we fur-
ther assess their utility in downstream autonomous driv-
ing tasks—specifically perception and open-loop plan-
ning—using the UniAD framework [2]. Quantitative results
in Table 9 demonstrate that the high-fidelity videos gen-
erated by our method achieve performance that is well-
aligned with original ground truth data, highlighting the
potential of synthetic data to support practical downstream
applications. A comprehensive analysis of Table 9 reveals
our method’s superiority across perception, segmentation,
and planning tasks. Our method maintains the lowest L2
errors at all horizons. This outperforms baselines like DA
(1.89 at 3.0m) and Dist-4D (1.91 at 3.0m), validating that
synthetic videos preserve meaningful motion and scene
dynamics for planning.
To verify our algorithm’s capability in 3D scene recon-
struction, its ability to synthesize novel view. We focus on
the task of adapting to novel vehicle models. Introducing a
new vehicle model often alters camera parameters, includ-
ing intrinsic parameters (e.g., camera type, focal length) and
extrinsic parameters (e.g., camera placement, orientation)
[71], [72]. A robust 4D reconstruction model should render
images with diverse camera parameters to mitigate overfit-
ting to specific vehicle-mounted camera configurations.
On the Waymo dataset, we rendered images with ran-
domly sampled intrinsic parameters and synthesized novel
views with random extrinsic variations, treating these ren-
dered outputs as augmented data. Following the protocols
in [71], [72], we trained the BEVDepth model on a com-
bined dataset of original Waymo data and our rendered
augmented data, using PD-BEV 2 as the baseline framework.
Notably, the rendered images also underwent standard aug-
mentation pipelines (e.g., resizing, cropping).
Table 10 presents the domain generalization results when
transferring from Waymo to nuScenes. Our method achieves
the highest performance across all metrics on the target
nuScenes domain. A critical insight is that joint augmen-
tation of intrinsic and extrinsic parameters drives this im-
provement, whereas augmenting only intrinsic parameters
yields limited gains. This is because virtual depth (derived
from our 4D reconstruction) already addresses intrinsic pa-
rameter variations effectively, while extrinsic augmentation
further enables the model to learn robust stereo relation-
2. https://github.com/EnVision-Research/Generalizable-BEV

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
TABLE 10: Comparison of different approaches on domain generalization protocols.
Waymo →nuScenes
Target Domain (nuScenes)
Method
mAP↑
mATE↓
mASE↓
mAOE↓
NDS* ↑
Oracle
0.475
0.577
0.177
0.147
0.587
DG-BEV
0.303
0.689
0.218
0.171
0.472
PD-BEV
0.311
0.686
0.216
0.170
0.478
Ours
0.331
0.665
0.26
0.161
0.498
ships between cameras, which is the key to generalizing
across novel vehicle.
7
CONCLUSIONS
This paper introduces PhiGenesis, a unified framework for
geometry-aware 4D scene generation that addresses the
limitations of existing simulation methods for autonomous
driving. While prior approaches either rely on predefined
ego trajectories or require per-scene optimization, PhiGe-
nesis enables scalable and temporally consistent 4D scene
synthesis by combining a feed-forward reconstruction stage
with a trajectory-conditioned video diffusion model. A
novel range-view adapter extends a frozen video VAE to
reconstruct 4D Gaussian representations from monocular or
multi-view inputs without scene-specific tuning. To ensure
geometric fidelity during generation, the framework incor-
porates Stereo Forcing, an uncertainty-guided conditioning
technique that corrects inconsistencies arising from depth
prediction errors. Together, these innovations allow Phi-
Genesis to produce realistic, controllable, and structurally
coherent 4D driving scenes, advancing the capabilities of
simulation for autonomous driving systems. By supporting
trajectory-conditioned generation, multi-view inputs and
outputs, and uncertainty-aware refinement, PhiGenesis of-
fers a scalable solution that bridges the gap between gener-
ative video models and 3D-aware scene synthesis. Exper-
imental results demonstrate that PhiGenesis outperforms
existing methods in terms of visual quality, temporal consis-
tency, and geometric accuracy, making it a strong candidate
for realistic data generation in safety-critical autonomous
driving applications.
REFERENCES
[1]
L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-to-
end autonomous driving: Challenges and frontiers,” TPAMI, 2024.
[2]
Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,
T. Lin, W. Wang et al., “Planning-oriented autonomous driving,”
in CVPR, 2023.
[3]
Y. Ma, T. Wang, X. Bai, H. Yang, Y. Hou, Y. Wang, Y. Qiao, R. Yang,
and X. Zhu, “Vision-centric bev perception: A survey,” TPAMI,
2024.
[4]
L. Kong, X. Xu, J. Ren, W. Zhang, L. Pan, K. Chen, W. T. Ooi,
and Z. Liu, “Multi-modal data-efficient 3d scene understanding
for autonomous driving,” TPAMI, 2025.
[5]
R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu,
“Magicdrive: Street view generation with diverse 3d geometry
control,” in ICLR, 2024.
[6]
Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang,
X. Sun, and X. Zhang, “Panacea: Panoramic and controllable video
generation for autonomous driving,” in CVPR, 2024.
[7]
X. Wang, Z. Zhu, G. Huang, X. Chen, and J. Lu, “Drivedreamer:
Towards real-world-driven world models for autonomous driv-
ing,” ECCV, 2024.
[8]
J. Mao, B. Li, B. Ivanovic, Y. Chen, Y. Wang, Y. You, C. Xiao,
D. Xu, M. Pavone, and Y. Wang, “Dreamdrive: Generative 4d scene
modeling from street view images,” ICRA, 2025.
[9]
R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu, “Magic-
drivedit: High-resolution long video generation for autonomous
driving with adaptive control,” ICCV, 2025.
[10] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou,
H. Lou, Y. Chen, R. Yang, Y. Huang, X. Ye, Z. Yan, Y. Shi, Y. Liao,
and H. Zhao, “Mars: An instance-aware, modular and realistic
simulator for autonomous driving,” CICAI, 2023.
[11] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che,
D. Xu, S. Fidler, M. Pavone, and Y. Wang, “Emernerf: Emergent
spatial-temporal scene decomposition via self-supervision,” 2023.
[12] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibra-
tion gaussian: Dynamic urban scene reconstruction and real-time
rendering,” arXiv preprint arXiv:2311.18561, 2023.
[13] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang,
“Drivinggaussian: Composite gaussian splatting for surrounding
dynamic autonomous driving scenes,” in CVPR, 2024, pp. 21634–
21643.
[14] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang,
X. Zhou, and S. Peng, “Street gaussians: Modeling dynamic urban
scenes with gaussian splatting,” in ECCV.
Springer, 2024, pp.
156–173.
[15] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic,
O. Litany, Z. Gojcic, S. Fidler, M. Pavone et al., “Omnire: Omni
urban scene reconstruction,” arXiv preprint arXiv:2408.16760, 2024.
[16] Y. Huo, G. Jiang, H. Wei, J. Liu, S. Zhang, H. Liu, X. Huang, M. Lu,
J. Peng, D. Li et al., “Egsral: An enhanced 3d gaussian splatting
based renderer with automated labeling for large-scale driving
scene,” in AAAI, 2025, pp. 3860–3867.
[17] R. Gao, K. Chen, Z. Li, L. Hong, Z. Li, and Q. Xu, “Magic-
drive3d: Controllable 3d generation for any-view rendering in
street scenes,” arXiv preprint arXiv:2405.14475, 2024.
[18] L. Wang, W. Zheng, D. Du, Y. Zhang, Y. Ren, H. Jiang, Z. Cui,
H. Yu, J. Zhou, J. Lu et al., “Stag-1: Towards realistic 4d driv-
ing simulation with video generation model,” arXiv preprint
arXiv:2412.05280, 2024.
[19] Y. Lu, X. Ren, J. Yang, T. Shen, Z. Wu, J. Gao, Y. Wang, S. Chen,
M. Chen, S. Fidler et al., “Infinicube: Unbounded and controllable
dynamic 3d driving scene generation with world-guided video
models,” arXiv preprint arXiv:2412.03934, 2024.
[20] J. Guo, Y. Ding, X. Chen, S. Chen, B. Li, Y. Zou, X. Lyu, F. Tan,
X. Qi, Z. Li et al., “Dist-4d: Disentangled spatiotemporal diffusion
with metric depth for 4d driving scene generation,” arXiv preprint
arXiv:2503.15208, 2025.
[21] X. Ren, T. Shen, J. Huang, H. Ling, Y. Lu, M. Nimier-David,
T. M¨uller, A. Keller, S. Fidler, and J. Gao, “Gen3c: 3d-informed
world-consistent video generation with precise camera control,”
arXiv preprint arXiv:2503.03751, 2025.
[22] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic
models,” NeurIPS, 2020.
[23] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli,
“Deep unsupervised learning using nonequilibrium thermody-
namics,” in ICML.
PMLR, 2015.
[24] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and
B. Poole, “Score-based generative modeling through stochastic
differential equations,” arXiv preprint arXiv:2011.13456, 2020.
[25] B. Li, J. Deng, W. Zhang, Z. Liang, D. Du, X. Jin, and W. Zeng,
“Hierarchical temporal context learning for camera-based seman-
tic scene completion,” in ECCV.
Springer, 2024, pp. 131–148.
[26] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang,
and H. Li, “Vista: A generalizable driving world model with high
fidelity and versatile controllability,” NeurIPS, vol. 37, pp. 91560–
91596, 2025.
[27] B. Li, Y. Sun, Z. Liang, D. Du, Z. Zhang, X. Wang, Y. Wang, X. Jin,
and W. Zeng, “Bridging stereo geometry and bev representation
with reliable mutual interaction for semantic scene completion,”
arXiv preprint arXiv:2303.13959, 2023.

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
[28] Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang, “Driving into
the future: Multiview visual forecasting and planning with world
model for autonomous driving,” in CVPR, 2024, pp. 14749–14759.
[29] B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan,
C. Zhang, T. Wang et al., “Uniscene: Unified occupancy-centric
driving scene generation,” in CVPR, 2025, pp. 11971–11981.
[30] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaus-
sian splatting for real-time radiance field rendering.” ACM Trans.
Graph., vol. 42, no. 4, pp. 139–1, 2023.
[31] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G. Huang,
X. Chen, B. Wang, Y. Zhang et al., “Drivedreamer4d: World models
are effective data machines for 4d driving scene representation,”
arXiv preprint arXiv:2410.13571, 2024.
[32] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, G. Huang, C. Liu,
Y. Chen, Y. Wang, X. Zhang et al., “Recondreamer: Crafting world
models for driving scene reconstruction via online restoration,” in
CVPR, 2025, pp. 1559–1569.
[33] G. Zhao, X. Wang, C. Ni, Z. Zhu, W. Qin, G. Huang, and
X. Wang, “Recondreamer++: Harmonizing generative and recon-
structive models for driving scene representation,” arXiv preprint
arXiv:2503.18438, 2025.
[34] L. Fan, H. Zhang, Q. Wang, H. Li, and Z. Zhang, “Freesim: To-
ward free-viewpoint camera simulation in driving scenes,” arXiv
preprint arXiv:2412.03566, 2024.
[35] B. Chen, D. Mart´ı Mons´o, Y. Du, M. Simchowitz, R. Tedrake, and
V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-
sequence diffusion,” NeurIPS, vol. 37, pp. 24081–24125, 2024.
[36] T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shecht-
man, and X. Huang, “From slow bidirectional to fast causal video
generators,” arXiv preprint arXiv:2412.07772, 2024.
[37] K. Song, B. Chen, M. Simchowitz, Y. Du, R. Tedrake, and
V. Sitzmann, “History-guided video diffusion,” arXiv preprint
arXiv:2502.06764, 2025.
[38] Q. Tian, X. Tan, Y. Xie, and L. Ma, “Drivingforward: Feed-forward
3d gaussian splatting for driving scene reconstruction from flexi-
ble surround-view input,” arXiv preprint arXiv:2409.12753, 2024.
[39] H. Lu, T. Xu, W. Zheng, Y. Zhang, W. Zhan, D. Du, M. Tomizuka,
K. Keutzer, and Y. Chen, “Drivingrecon: Large 4d gaussian
reconstruction model for autonomous driving,” arXiv preprint
arXiv:2412.09043, 2024.
[40] J. Yang, J. Huang, Y. Chen, Y. Wang, B. Li, Y. You, A. Sharma,
M. Igl, P. Karkus, D. Xu et al., “Storm: Spatio-temporal recon-
struction model for large-scale outdoor scenes,” arXiv preprint
arXiv:2501.00602, 2024.
[41] K. Song, B. Chen, M. Simchowitz, Y. Du, R. Tedrake, and
V. Sitzmann, “History-guided video diffusion,” 2025. [Online].
Available: https://arxiv.org/abs/2502.06764
[42] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing:
Bridging the train-test gap in autoregressive video diffusion,”
arXiv preprint arXiv:2506.08009, 2025.
[43] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G. Huang,
X. Chen, B. Wang, Y. Zhang et al., “Drivedreamer4d: World models
are effective data machines for 4d driving scene representation,”
in CVPR, 2025, pp. 12015–12026.
[44] D. Wei, Z. Li, and P. Liu, “Omni-scene: Omni-gaussian repre-
sentation for ego-centric sparse-view scene reconstruction,” in
Proceedings of the Computer Vision and Pattern Recognition Conference,
2025, pp. 22317–22327.
[45] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong,
X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic
framework for large video generative models,” arXiv preprint
arXiv:2412.03603, 2024.
[46] L. Kong, Y. Liu, R. Chen, Y. Ma, X. Zhu, Y. Li, Y. Hou, Y. Qiao, and
Z. Liu, “Rethinking range view representation for lidar segmen-
tation,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision, 2023, pp. 228–240.
[47] A. Chen, W. Zheng, Y. Wang, X. Zhang, K. Zhan, P. Jia, K. Keutzer,
and S. Zhang, “Geodrive: 3d geometry-informed driving world
model with precise action control,” arXiv preprint arXiv:2505.22421,
2025.
[48] J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole,
D. Sun, and M.-H. Yang, “Monst3r: A simple approach for esti-
mating geometry in the presence of motion,” ICLR, 2025.
[49] J. Park, C. Xu, S. Yang, K. Keutzer, K. Kitani, M. Tomizuka, and
W. Zhan, “Time will tell: New outlooks and a baseline for temporal
multi-view 3d object detection,” 2023.
[50] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaus-
sian splatting for real-time radiance field rendering,” in ACM
Trans. Graph., vol. 42, no. 4, 2023, pp. 1–14.
[51] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibra-
tion gaussian: Dynamic urban scene reconstruction and real-time
rendering,” in ArXiv, 2023.
[52] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “De-
formable 3d gaussians for high-fidelity monocular dynamic scene
reconstruction,” in ArXiv, vol. abs/2309.13101, 2023.
[53] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, “Lgm:
Large multi-view gaussian model for high-resolution 3d content
creation,” in Proc. ECCV.
Springer, 2024, pp. 1–18.
[54] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli,
and Z. Xu, “Gs-lrm: Large reconstruction model for 3d gaussian
splatting,” in arXiv preprint arXiv:2404.19702, 2024.
[55] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “Pixelsplat:
3d gaussian splats from image pairs for scalable generalizable 3d
reconstruction,” in CVPR, 2024.
[56] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J.
Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from
sparse multi-view images,” in ECCV, 2024.
[57] X. Ren, Y. Lu, J. Z. Wu, H. Ling, M. Chen, S. Fidler, F. Williams,
J. Huang et al., “Scube: Instant large-scale scene reconstruction
using voxsplats,” in NeurIPS, 2024.
[58] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik,
P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine et al., “Scalability in
perception for autonomous driving: Waymo open dataset,” in
CVPR, 2020, pp. 2446–2454.
[59] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu,
A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A
multimodal dataset for autonomous driving,” in CVPR, 2020.
[60] X. Wang, Z. Zhu, Y. Zhang, G. Huang, Y. Ye, W. Xu, Z. Chen,
and X. Wang, “Are we ready for vision-centric driving streaming
perception? the asap benchmark,” arXiv preprint arXiv:2212.08914,
2022.
[61] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo,
“Segformer: Simple and efficient design for semantic segmentation
with transformers,” NeurIPS, vol. 34, pp. 12077–12090, 2021.
[62] Z. Liu, K. L. Cheng, Q. Wang, S. Wang, H. Ouyang, B. Tan,
K. Zhu, Y. Shen, Q. Chen, and P. Luo, “Depthlab: From partial
to complete,” arXiv preprint arXiv:2412.18153, 2024.
[63] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, Y. Rao, G. Huang, J. Lu, and
J. Zhou, “Surrounddepth: Entangling surrounding views for self-
supervised multi-camera depth estimation,” in Conference on robot
learning.
PMLR, 2023, pp. 539–549.
[64] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibra-
tion gaussian: Dynamic urban scene reconstruction and real-time
rendering,” arXiv preprint arXiv:2311.18561, 2023.
[65] Q. Wang, L. Fan, Y. Wang, Y. Chen, and Z. Zhang, “Freevs: Gen-
erative view synthesis on free driving trajectory,” arXiv preprint
arXiv:2410.18079, 2024.
[66] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural
radiance fields from one or few images,” in CVPR, 2021, pp. 4578–
4587.
[67] S.
Wang,
V.
Leroy,
Y.
Cabon,
B.
Chidlovskii,
and
J.
Re-
vaud, “Dust3r: Geometric 3d vision made easy,” arXiv preprint
arXiv:2312.14132, 2023.
[68] T. Liu, G. Wang, S. Hu, L. Shen, X. Ye, Y. Zang, Z. Cao, W. Li, and
Z. Liu, “Mvsgaussian: Fast generalizable gaussian splatting recon-
struction from multi-view stereo,” arXiv preprint arXiv:2405.12218,
vol. 2, 2024.
[69] Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang,
X. Sun, and X. Zhang, “Panacea: Panoramic and controllable video
generation for autonomous driving,” 2023.
[70] X. Yang, L. Wen, Y. Ma, J. Mei, X. Li, T. Wei, W. Lei, D. Fu,
P. Cai, M. Dou et al., “Drivearena: A closed-loop generative
simulation platform for autonomous driving,” arXiv preprint
arXiv:2408.00415, 2024.
[71] S. Wang, X. Zhao, H.-M. Xu, Z. Chen, D. Yu, J. Chang, Z. Yang, and
F. Zhao, “Towards domain generalization for multi-view 3d object
detection in bird-eye-view,” in CVPR, 2023, pp. 13333–13342.
[72] H. Lu, Y. Zhang, G. Wang, Q. Lian, D. Du, and Y.-C. Chen,
“Towards generalizable multi-camera 3d object detection via per-
spective rendering,” in AAAI, vol. 39, no. 6, 2025, pp. 5811–5819.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
Hao Lu is a PhD candidate in Computer Science
at the Hong Kong University of Science and
Technology, Guangzhou, specializing in com-
puter vision and artificial intelligence, with a par-
ticular focus on autonomous driving technology.
Prior to pursuing his PhD, he obtained a master’s
degree from the Chinese Academy of Sciences.
He has published several papers in journals and
conferences such as TPAMI, TIP, CVPR, ICCV,
and ECCV etc.
Zhuang Ma received the B.E. degree from the
University of Plymouth, UK. He received the MSc
degree from the University of Birmingham, UK.
He is currently an engineer at PhiGent. China.
His current research interests include 2D and 3D
visual perception, robotics, and multi-modality
content generation.
Guangfeng Jiang received the B.S. degree in
communication engineering from the Shandong
University, Jinan, China, in 2021. He is currently
pursuing the Ph.D. degree in information and
communication engineering with the University
of Science and Technology of China, Hefei. His
research interests include autonomous driving
and 3D computer vision.
Wenhang Ge is a PhD candidate in Computer
Science at the Hong Kong University of Sci-
ence and Technology, Guangzhou, specializing
in computer vision and artificial intelligence, with
a particular focus on 3D vision and AIGC. Prior
to pursuing his PhD, he obtained a master’s de-
gree from Sun Yat-sen University. He has pub-
lished several papers in journals and confer-
ences such as TPAMI, ICCV, CVPR, and ECCV
etc.
Bohan Li (Student Member, IEEE) received
the B.E. degree from the School of Con-
trol Engineering, Northeastern University (NEU),
Shenyang, China, in 2019. He received the M.E.
degree from the School of Control Science and
Engineering, South China University of Technol-
ogy (SCUT), Guangzhou, China, in 2022. He is
currently pursuing the Ph.D. degree in Shanghai
Jiao Tong University (SJTU) and Eastern Insti-
tute of Technology (EIT). His research interests
include 3D visual perception, robotics, and multi-
modality content generation.
Wenzhao Zheng received the B.S. degree
and the Ph.D. degree from the Department of
Physics and Department of Automation, Ts-
inghua University, China, in 2018 and 2023, re-
spectively. His current research interests include
computer vision, deep learning, and represen-
tation learning. He has authored more than 30
papers in IEEE Transactions on Pattern Analy-
sis and Machine Intelligence, IEEE Transactions
on Image Processing, CVPR, ICCV, and ECCV.
He serves as a regular reviewer member for a
number of journals and conferences, e.g., IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, IEEE Transactions on Image
Processing, IEEE Transactions on Biometrics, Behavior, and Identity
Science, ACM Transactions on Intelligent Systems and Technology,
CVPR, ICCV, ECCV, IJCAI, ICME, and ICIP.
Yunpeng Zhang received the B.Sc. and M.Sc.
degrees in Automation from Tsinghua University
in 2019 and 2022, respectively. He is currently
an algorithm engineer at PhiGent Robotics Co.,
LTD. His main research interests include monoc-
ular 3D object detection, multi-camera based 3D
object detection, vision-based occupancy pre-
diction, and end-to-end autonomous driving.
Yingcong Chen is currently an assistant pro-
fessor at the Artificial Intelligence Trust, the
Hong Kong University of Science and Technol-
ogy (Guangzhou). He received his PhD degree
from the Chinese University of Hong Kong. He
was a postdoctoral associate at the Computer
Science and Artificial Intelligence Lab (CSAIL),
Massachusetts Institute of Technology in 2022.
He obtained the Hong Kong PhD Fellowship in
2016. He is currently an assistant professor at
the Artificial Intelligence Trust, the Hong Kong
University of Science and Technology (Guangzhou). He serves as a
reviewer for TPAMI, IJCV, TIP, CVPR, ICCV, ECCV, BMVC, IJCAI, AAAI,
etc. His research interest includes deep learning, image generation and
editing, generative adversarial networks, etc.
