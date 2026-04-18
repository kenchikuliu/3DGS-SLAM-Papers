<!-- page 1 -->
ArtiFixer: Enhancing and Extending 3D Reconstruction with
Auto-Regressive Diffusion Models
RICCARDO DE LUTIO∗, NVIDIA, USA
TOBIAS FISCHER, NVIDIA, Switzerland and ETHZ, Switzerland
YEN-YU CHANG, NVIDIA, USA and Cornell University, USA
YUXUAN ZHANG, NVIDIA, USA
JAY ZHANGJIE WU, NVIDIA, Canada
XUANCHI REN, NVIDIA, Canada, University of Toronto, Canada, and Vector Institute, Canada
TIANCHANG SHEN, NVIDIA, Canada, University of Toronto, Canada, and Vector Institute, Canada
KATARINA TOTHOVA, NVIDIA, Switzerland
ZAN GOJCIC, NVIDIA, Switzerland
HAITHEM TURKI∗, NVIDIA, USA
Fig. 1. ArtiFixer enhances and extends existing 3D reconstructions in a highly efficient and scalable manner. Given an initial reconstruction, set of reference
views, and an optional text prompt, it auto-regressively generates novel content that maintains a high degree of consistency with existing observations.
ArtiFixer can directly produce hundreds of novel views in a single inference pass or serve as pseudo-supervision to improve the underlying 3D reconstruction.
Video results: artifixer2026.github.io
Per-scene optimization methods such as 3D Gaussian Splatting provide
state-of-the-art novel view synthesis quality but extrapolate poorly to under-
observed areas. Methods that leverage generative priors to correct artifacts
in these areas hold promise but currently suffer from two shortcomings.
The first is scalability, as existing methods use image diffusion models or
bidirectional video models that are limited in the number of views they
can generate in a single pass (and thus require a costly iterative distillation
process for consistency). The second is quality itself, as generators used in
prior work tend to produce outputs that are inconsistent with existing scene
content and fail entirely in completely unobserved regions. To solve these, we
propose a two-stage pipeline that leverages two key insights. First, we train a
∗Equal contribution.
Authors’ Contact Information: Riccardo de Lutio, NVIDIA, Santa Clara, USA, rdelutio@
nvidia.com; Tobias Fischer, NVIDIA, Zurich, Switzerland and ETHZ, Zurich, Switzer-
land, tobiasf@nvidia.com; Yen-Yu Chang, NVIDIA, Santa Clara, USA and Cornell Uni-
versity, Ithaca, USA, yc2463@cornell.edu; Yuxuan Zhang, NVIDIA, New York, USA,
alezhang@nvidia.com; Jay Zhangjie Wu, NVIDIA, Toronto, Canada, wjay@nvidia.com;
Xuanchi Ren, NVIDIA, Toronto, Canada and University of Toronto, Toronto, Canada and
Vector Institute, Toronto, Canada, xuanchir@nvidia.com; Tianchang Shen, NVIDIA,
Toronto, Canada and University of Toronto, Toronto, Canada and Vector Institute,
Toronto, Canada, frshen@nvidia.com; Katarina Tothova, NVIDIA, Zurich, Switzerland,
ktothova@nvidia.com; Zan Gojcic, NVIDIA, Zurich, Switzerland, zgojcic@nvidia.com;
Haithem Turki, NVIDIA, Seattle, USA, hturki@nvidia.com.
powerful bidirectional generative model with a novel opacity mixing strategy
that encourages consistency with existing observations while retaining the
model’s ability to extrapolate novel content in unseen areas. Second, we
distill it into a causal auto-regressive model that generates hundreds of
frames in a single pass. This model can directly produce novel views or serve
as pseudo-supervision to improve the underlying 3D representation in a
simple and highly efficient manner. We evaluate our method extensively
and demonstrate that it can generate plausible reconstructions in scenarios
where existing approaches fail completely. When measured on commonly
benchmarked datasets, we outperform existing all existing baselines by a
wide margin, exceeding prior state-of-the-art methods by 1-3 dB PSNR.
CCS Concepts: • Computing methodologies →Computer vision; Ren-
dering.
Additional Key Words and Phrases: Image & Video Generative AI, Deep
Image/Video Synthesis, Neural Rendering, Multi-View & 3D, Deep Learning,
Machine Learning, Artificial Intelligence
1
Introduction
High-quality novel view synthesis is essential for applications in vir-
tual and augmented reality and closed-loop simulation for physical
AI. These use cases require photorealistic rendering and the ability
arXiv:2603.00492v1  [cs.CV]  28 Feb 2026

<!-- page 2 -->
2
•
de Lutio, R. et al
to navigate complex environments under unconstrained camera
motion. In recent years, two paradigms have emerged as dominant
approaches to novel view synthesis: explicit 3D neural reconstruc-
tion [Kerbl et al. 2023; Mildenhall et al. 2020], and camera-controlled
image or video generation [Ren et al. 2025; Zhou et al. 2025].
Neural reconstruction methods have matured significantly and
now enable real-time rendering and high visual fidelity when trained
from dense image collections with accurate camera poses. However,
in the most widely used per-scene optimization setting, their per-
formance remains fundamentally limited by the completeness and
quality of the input observations. Regions that are sparsely observed
or entirely missing during capture are poorly reconstructed, leading
to artifacts, holes, or implausible geometry. While such deficiencies
remain hidden near the training views, they are inevitably exposed
during free navigation of the scene.
Conversely, recent video generative models have demonstrated
the ability to synthesize photorealistic and temporally coherent con-
tent that is often indistinguishable from real-world videos [Google
DeepMind 2024; NVIDIA et al. 2025; OpenAI 2024]. Despite this
progress, precise camera control over extended sequences, long-
term temporal consistency, and the accumulation of drift and hal-
lucinations remain open challenges, limiting their applicability to
interactive view synthesis.
Instead of treating reconstruction and generation as standalone
alternatives, we aim to combine their complementary strengths:
generative models serve as powerful priors to repair and complete
imperfect reconstructions, while the explicit—albeit noisy and par-
tial—3D representation provides a strong conditioning signal that
grounds generation, mitigates long-term drift, and suppresses hallu-
cinations. Recent methods have taken initial steps in this direction by
training generative models to map degraded novel-view renderings
to clean images and distilling the resulting improvements back into
an underlying 3D representation [Fischer et al. 2025; Gao* et al. 2024;
Wu et al. 2025c; Yu et al. 2024]. However, these approaches must
navigate two fundamental trade-offs. First, they must balance tem-
poral consistency and efficiency: some employ large bidirectional
video generative models that provide strong temporal coherence
but incur high computational cost [Fischer et al. 2025; Gao* et al.
2024; Wu et al. 2025b], while others rely on (multi-view) image-
based generative models that are more efficient but limit temporal
consistency and require progressive distillation strategies [Wu et al.
2025c, 2024]. Second, they face the trade-off between conditioning
strength and generative capacity. Approaches [Wu et al. 2025b; Yu
et al. 2024] that condition generation on corrupted renderings via
concatenation or cross-attention risk altering the observed scene
content, whereas methods [Fischer et al. 2025; Wu et al. 2025c]
trained to directly map corrupted renderings to clean images are
incapable of synthesizing missing content, due to the mode collapse
in fully unobserved regions where all input pixels are black.
In our work, we follow this line of research by adapting a pre-
trained bidirectional video diffusion model into a camera-controllable
generator that maps corrupted renderings to clean images. To over-
come the aforementioned limitations, we introduce two key con-
tributions: (i) we propose an opacity-aware noise mixing strategy
that injects Gaussian noise into low-opacity regions, preventing
mode collapse and preserving generative capacity in unobserved
areas; (ii) we distill the bidirectional model into a few-step causal
auto-regressive generator capable of producing arbitrarily long,
temporally consistent videos while approaching the efficiency of
prior image-based methods. In doing so, we demonstrate that even
highly degraded 3D reconstructions provide sufficient conditioning
signals to significantly simplify the distillation process. Our result-
ing framework enables efficient improvement of the underlying 3D
reconstruction and greatly outperforms a wide range of baselines
across multiple benchmarks. To our knowledge, this work is the
first to explore the intersection of explicit 3D reconstruction and
auto-regressive video generation, and to demonstrate the mutual
benefits that arise from tightly coupling these two paradigms.
2
Related Work
Novel view synthesis from 3D representations. Neural Radiance
Fields (NeRFs) [Mildenhall et al. 2020] and, more recently, 3D Gauss-
ian Splatting (3DGS) [Kerbl et al. 2023] have revolutionized the field
of novel view synthesis by distilling sensor information (usually
overlapping photos of a scene) into a 3D representation that can
then be queried from arbitrary camera viewpoints. Because these
representations are optimized on a per-scene basis, their ability
to extrapolate beyond observed views is inherently limited, and
they fail to render plausible content in sparsely observed or missing
regions.
A large body of work seeks to mitigate these limitations through
handcrafted geometric priors [Niemeyer et al. 2022; Somraj et al.
2023; Yang et al. 2023], pretrained depth [Deng et al. 2022; Roessle
et al. 2022; Wang et al. 2023; Zhu et al. 2024] and normal [Yu et al.
2022] estimators, and adversarial networks [Roessle et al. 2023].
However, these approaches are sensitive to noise, difficult to balance
with data terms, and yield only marginal improvements in denser
captures. An alternative line of work trains feed-forward networks
on large multi-scene datasets, which are used to enhance a scene-
optimized NeRF/3DGS [Zhou et al. 2023] or directly predict novel
views [Chen et al. 2021; Lu et al. 2025; Ren et al. 2024; Yu et al. 2021].
While these deterministic methods perform well near reference
views, they often produce blurry results in ambiguous regions where
the distribution of possible renderings is inherently multi-modal.
Diffusion models for novel view synthesis. An alternative strat-
egy is to leverage the priors learned by generative diffusion models
trained on internet-scale data to enhance novel view synthesis. Early
works [Poole et al. 2023; Sargent et al. 2024; Wu et al. 2024] use a dif-
fusion model as a learned critic during reconstruction optimization,
but this incurs substantial computational overhead. More recent
approaches [Fischer et al. 2025; Gao* et al. 2024; Liu et al. 2024,
2022; Wu et al. 2025c,b] directly generate multi-view–consistent
images that can be consumed by a downstream 3D reconstruction
pipeline. While this strategy substantially improves training effi-
ciency, it typically relies on iterative generation and distillation,
in which newly synthesized views are progressively distilled back
into the 3D representation to satisfy computational and consistency
constraints. Building on the rapid progress of video generative mod-
els [Blattmann et al. 2023; Wan et al. 2025], recent work reverses
this paradigm. Rather than distilling generative outputs into a 3D

<!-- page 3 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
3
Phase I: Bidirectional training
Phase II: Causal Distillation
Opacity Mixing
Text Prompt
Reference Views
Opacity
Input Video
Observed Views & Initial 3D Reconstruction
Generated Views & Distilled 3D Reconstruction
Ray Maps
Noise
Clean Output Video
Clean Output Video
Self Forcing DMD
Causal Autoregressive Model
Inputs
Bidirectional Teacher Model
Fig. 2. Method overview. We first train a bidirectional flow matching model that transports degraded RGB renderings into clean outputs. We encode the
input RGB into latent space and mix with Gaussian noise using the rendered opacity maps to avoid modal collapse in unseen regions. We inject fine-grained
opacity information and camera control along with clean reference views and an optional text prompt. In the second phase of our pipeline, we distill the
teacher into an auto-regressive causal model via Self Forcing-style DMD distillation [Huang et al. 2025], which can be directly used to render novel views or
used as pseudo-supervision to distill back into the underlying 3D representation.
representation, these methods treat the 3D representation as a con-
ditioning signal for a generative model that directly synthesizes
novel views [Kong et al. 2025; Ren et al. 2025]. Although this ap-
proach can improve the perceptual realism of novel views, it inherits
limitations of the underlying generative models, including temporal
inconsistencies, hallucinations, and imperfect camera control.
Auto-regressive video generation. While bidirectional video gener-
ation models synthesize all frames jointly, auto-regressive models
generate frames sequentially using block-causal attention. Auto-
regressive generation improves scalability and generation efficiency
compared to bidirectional models, but often suffers from quality
degradation over time, as each frame is conditioned on previously
generated outputs, causing errors to accumulate [Yin et al. 2025b].
Several methods try to address the issue by better aligning the train-
ing scheme of these models with inference-time conditions, thereby
reducing exposure bias [Cui et al. 2025; Huang et al. 2025; Liu et al.
2025]. A complementary line of research focuses on improving gen-
eration speed and controllability by exploiting temporal and spatial
cues to select per-frame context [Kong et al. 2025; Shin et al. 2025;
Yang et al. 2025], enabling interactive auto-regressive world mod-
els [Hong et al. 2025]. Despite these advances, auto-regressive video
models still lag behind explicit 3D representations in terms of spatial
consistency, camera controllability, and rendering efficiency.
3
Preliminaries
3D Gaussian Splatting. 3DGS [Kerbl et al. 2023] represents a
scene as a collection of anisotropic 3D Gaussian primitives G =
{G𝑗}𝑀
𝑗=1, each parameterized by a mean 𝝁𝑗∈R3, an opacity 𝜎𝑗,
view-dependent color c𝑗∈R3 modeled with spherical harmonics
coefficients, and an anisotropic covariance matrix 𝚺𝑗= R𝑗S𝑗S⊤
𝑗U⊤
𝑗,
where R𝑗∈𝑆𝑂(3) and S𝑗∈R3×3 are the matrix representation of
the unit quaternion q𝑗∈R4 and scaling vector s𝑗∈R3
+.
To render a novel view, the 3D Gaussians are projected onto the
target camera image plane and sorted by depth. The final pixel color
is obtained via front-to-back alpha compositing,
C(p) =
𝑁
∑︁
𝑖=1
𝛼𝑖c𝑖
𝑖−1
Ö
𝑗
(1 −𝛼𝑗),
(1)
where the opacity contribution of the 𝑖-th Gaussian at image-space
location p is
𝛼𝑖= 𝜎𝑗exp −1
2 (p −𝝁𝑖)⊤𝚺−1
𝑖(p −𝝁𝑖) .
(2)
Video diffusion models. Diffusion models (DM) define a continuous-
time forward noising process that transports samples from an ar-
bitrary data distribution 𝑝𝑑𝑎𝑡𝑎(x) to a fixed Gaussian prior N (0, I),
and learn the corresponding reverse-time dynamics that transports
samples from N (0, I) back to 𝑝𝑑𝑎𝑡𝑎(x) [Ho et al. 2020; Song et al.
2020]. Most video diffusion models (VDMs) [Blattmann et al. 2023],
encode video samples into a lower-dimensional latent space to im-
prove computational efficiency. During training, noise is added to
the latent representation according to a predefined schedule. At
timestep 𝜏≤𝑇, the noised latent is given by z𝜏:= 𝛼𝜏z0 +𝜎𝜏𝝐, where
𝝐∼N (0, I) and 𝛼𝜏and 𝜎𝜏are scalar coefficients determined by
the noise schedule, such that z𝑇approaches N (0, I). The model is
trained to approximate the score s𝜃(z𝜏,𝜏) ≈∇z𝜏log𝑝𝜏(z𝜏), com-
monly via a denoising objective min𝜃E𝜏,z0,𝝐
s𝜃(z𝜏,𝜏) −𝝐
2
2. At
inference, the learned score is used to simulate the reverse-time
SDE starting from z𝑇∼N (0, I) to obtain z0.

<!-- page 4 -->
4
•
de Lutio, R. et al
Layer Norm
Self-Attention
Cross-Attention
V-Tokens
Reference Views
PRoPE
T-Tokens
Prompt
Timestep
MLP
Linear
Linear
Camera Rays
Opacity
N x
Layer Norm
Layer Norm
FFN
V-Tokens
+
+
Fig. 3. Transformer block. We start from a pretrained text-to-video
model [Wan et al. 2025] and inject camera and opacity information into
each transformer block via linear layers after applying self-attention and
layer normalization. We patchify reference views into visual tokens, apply
relative camera conditioning via PRoPe [Li et al. 2025], and add 𝐾𝑛and 𝑉𝑛
projections to the cross-attention operation. We zero-initialize 𝑓𝑐, 𝑓𝑜, and
𝑉𝑛to ensure compatibility with the pretrained initialization.
Flow matching [Lipman et al. 2023a; Liu et al. 2023] builds on the
same continuous-time transport viewpoint as score-based diffusion,
but generalizes it to learn an ODE flow between two arbitrary end-
point distributions 𝑝𝑠𝑟𝑐and 𝑝𝑡𝑔𝑡by fitting a time-dependent vector
field v𝜃(z𝑡,𝑡) whose induced probability path {𝑝𝑡}𝑡∈[0,1] satisfies
𝑝0 = 𝑝𝑠𝑟𝑐and 𝑝1 = 𝑝𝑡𝑔𝑡. During training, we sample endpoint la-
tents z0 ∼𝑝𝑠𝑟𝑐and z1 ∼𝑝𝑡𝑔𝑡and a time 𝑡∈[0, 1], construct an
intermediate latent via z𝑡:= (1 −𝑡)z0 + 𝑡z1 with target velocity
v𝑡:= 𝑑z𝑡
𝑑𝑡= z1 −z0, and fit the vector field using the conditional
flow matching objective min𝜃E𝑡,z0,z1
v𝜃(z𝑡,𝑡) −v𝑡
2
2. At inference,
we draw z0 ∼𝑝𝑠𝑟𝑐and numerically integrate the learned ODE from
𝑡= 0 to 𝑡= 1 to obtain z1 as a sample from 𝑝𝑡𝑔𝑡.
4
Method
Given an initial 3D reconstruction of a scene created from a sparse
set of images, our goal is to generate artifact-free renderings from ar-
bitrary camera viewpoints, including regions unobserved by input
images, at interactive rates. Our solution is a controllable auto-
regressive video model that can either directly render arbitrary long
novel-view renderings or provide pseudo-supervision to improve
the underlying 3D reconstruction. We describe how to adapt a pre-
trained video diffusion model to serve as a bidirectional teacher in
Sec. 4.1. We discuss causal distillation and the capabilities of the
resulting model in Sec. 4.2. Fig. 2 illustrates our approach.
4.1
Bidirectional Training
Architecture. We start from a pretrained text-to-video model (Wan
2.1 T2V-14B [Wan et al. 2025]), freeze its VAE and text encoder, and
finetune the remaining components. We guide where to generate
scene content through rendered opacity maps O and enable cam-
era control in completely unobserved areas via Plücker raymaps
R. We downscale the spatial dimensions of O and R to match the
spatial compression factor of the VAE via the PixelUnshuffle opera-
tion [Paszke et al. 2019], encode them via per-block linear layers 𝑓𝑜
and 𝑓𝑟(Fig. 3), and add the embeddings to the visual tokens:
𝑇𝑟:= 𝑇𝑠+ 𝑓𝑟(PixelUnshuffle(R))
(3)
𝑇𝑜:= 𝑇𝑟+ 𝑓𝑜(PixelUnshuffle(O)),
(4)
where 𝑇𝑠denotes the token set after applying self-attention and
layer-normalization. We found this strategy to be more computa-
tionally efficient than alternatives such as VAE encoding R and O
while providing camera control even when the input rendering is
entirely empty (see supplemental videos for an example). To provide
additional scene context, we encode clean reference views with the
frozen VAE, tokenize them via a learned patchifier shared across
blocks, inject them with relative camera information via PRoPe [Li
et al. 2025], and apply cross-attention via additional 𝐾𝑛and 𝑉𝑛pro-
jections. 𝑓𝑐, 𝑓𝑜, and 𝑉𝑛are all zero-initialized to ensure compability
with the pretrained initialization.
Opacity mixing. Most generative models start from Gaussian
noise 𝜖∼N (0, I) which is then iteratively transformed into a la-
tent video representation z. Prior work tends to similarly start from
such noise, conditioning the generation process on the initial de-
graded rendering latent z𝑑𝑒𝑔via channel-concatenation [Wu et al.
2025b; Yin et al. 2025a] or classifier-free guidance [Liu et al. 2022].
Although the resulting latent z𝑒𝑛ℎtends to be semantically similar
to its degraded counterpart, notable inconsistencies remain, espe-
cially in high-artifact regions (Fig. 4). Several methods directly start
from z𝑑𝑒𝑔instead of noise [Fischer et al. 2025; Wu et al. 2025c],
which encourages stronger consistency guarantees, but suffers from
mode collapse in completely unseen areas, hindering its ability to
extrapolate high-quality renderings (Fig. 4). To address this, we mix
Gaussian noise into low-opacity regions by downscaling O into
O𝑧through max pooling to match z𝑑𝑒𝑔’s spatial dimensions (we
retain fine-grained opacity information via Eq. (4)) and deriving
z𝑚𝑖𝑥= O𝑧z𝑑𝑒𝑔+ (1 −O𝑧)𝜖as the source distribution for our model.
Since no source information is lost from the max-pooling, this ap-
proach preserves the consistency benefits of starting from z𝑑𝑒𝑔while
retaining the generative capabilities of the model (as z𝑚𝑖𝑥becomes
𝜖in entirely novel regions).
Data curation. Our goal is to not only correct artifacts in under-
observed areas as in prior work [Fischer et al. 2025; Wu et al. 2025c]
but also generate plausible content in entirely unseen areas. To do
so, we generate paired reconstruction-ground truth samples from
DL3DV-10K [Ling et al. 2024] with a camera selection strategy that
encourages highly sparse reconstructions with large empty regions
that the model must learn to inpaint. Given a set of camera poses
with rotations R𝑖and translations t𝑖, we first measure the camera
pose distance function 𝑑= ||R𝑖−R𝑗||𝐹+ ||t𝑖−t𝑗||2, find the camera
pair (𝑃1, 𝑃2) with the largest distance, and seed groups 𝐺1 and 𝐺2.
We assign the remaining cameras to𝐺1 or𝐺2 based on their distance
to 𝑃1 and 𝑃2, and then sample 2-12 cameras with the largest inter-
camera distance within each group to generate reconstructions
of differing sparsity. We roughly align the camera scales of each

<!-- page 5 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
5
Inputs
Channel Concatenation
Direct Input
Opacity Mixing (Ours)
Ground Truth
Fig. 4. Opacity mixing. Given a degraded rendering, set of reference views, and an optional text prompt (left), we predict an artifact-free rendering at a
target viewpoint. Starting from Gaussian noise and and channel concatenating the degraded rendering as in prior work [Wu et al. 2025b; Yin et al. 2025a]
produces renderings that are semantically similar to the reference views, but with notable inconsistencies (such as the table in the top row). Directly starting
from the degraded rendering instead of Gaussian noise improves consistency, but degrades quality noticeably when extrapolating to areas outside those
covered by the degraded renderings (bottom row). Instead, we mix Gaussian noise into the rendering based on its opacity map. The resulting input retains
the consistency benefits of the original while enabling a strong generative capability in entirely novel regions.
reconstruction with a pretrained metric depth estimator [Wang et al.
2025] and prompt a vision-language model [Bai et al. 2025] for scene
descriptions. We provide more details in Sec. B of the supplement.
Optimization. Given an initial latent-encoded rendering z𝑑𝑒𝑔, which
we transform into z𝑚𝑖𝑥, we train our model to predict its enhanced
counterpart z𝑒𝑛ℎvia conditional flow matching loss L𝑐𝑓𝑚[Lipman
et al. 2023b]. We construct batches of paired reconstruction-ground
truth data by sampling 𝑁= 81 frames along with the corresponding
camera poses, text prompt (dropped with 10% probability), and a
uniformly varying number of reference views (0-12). To enhance the
model’s generative abilities and viewpoint controllability, we drop
the last 𝐾≤𝑁frames of the input reconstruction (𝐾is randomly
chosen) so that the model must rebuild the ground truth from the
prompt, reference views, and camera conditions alone.
4.2
Causal Distillation
Initialization. We initialize the causal model from the weights of
the bidirectional teacher. To stabilize training, we follow a simpler
strategy than the ODE initialization protocol of prior work [Huang
et al. 2025; Shin et al. 2025; Yin et al. 2025b], which requires gener-
ating a dataset of ODE trajectories from the teacher model. Instead,
we simply apply a block-causal mask, perturb each input frame with
differing noise levels as in Diffusion Forcing [Chen et al. 2025], and
otherwise use the same inputs and training protocol as in Sec. 4.1.
Autoregressive rollout. After initialization, we adopt a training
strategy similar to Self Forcing [Huang et al. 2025], where we gener-
ate video chunks sequentially and condition on previously generated
chunks via KV caching, except that we continue applying dropout
as in Sec. 4.1 as camera control and generation from pure noise
otherwise degrade. We apply Distribution Matching Distillation
(DMD) [Yin et al. 2024] to convert the model into a few-step gen-
erator (𝑁= 4 in our experiments, although, outside of entirely
novel regions, this can often be reduced to fewer steps with little
noticeable difference as discussed in Sec. A of the supplement).
Long video generation. Existing methods rely on long-horizon
training [Hong et al. 2025; Yang et al. 2025] to minimize error accu-
mulation in long video rollouts. Although these strategies can be
applied to our method, in practice we find our conditioning signals
(notably the degraded rendering and reference views) sufficient to
prevent error accumulation. We thus train with the same number
of frames as in Sec. 4.1 and use a rolling KV cache during inference.
Although simple, this approach accelerates training convergence
(due to training on a more diverse set of shorter videos for a given
computational budget) and generalizes to arbitrary length videos,
as shown in our experiments.
3D distillation. Prior work distills diffusion model outputs into
3D representations [Kerbl et al. 2023] for consistency purposes, as
they otherwise exhibit temporal instability [Wu et al. 2025c] or
are limited by number of frames bidirectional models can gener-
ate in a single pass [Fischer et al. 2025; Wu et al. 2025b]. As our
auto-regressive model can sequentially generate arbitrary-length
renderings, we are not limited by these constraints. However, 3D dis-
tillation is still sometimes desirable from an efficiency perspective,
as these representations render orders of magnitude faster. To do so,
existing methods require a progressive distillation process that al-
ternates between view generation and 3D reconstruction, incurring
significant training time overhead. In our case, as we can generate
an arbitrary number of frames in a consistent manner, we adopt
a more efficient approach by simply generating all desired novel
views in a single pass before applying standard 3D reconstruction.
5
Experiments
We evaluate three variants of our method: ArtiFixer, which directly
renders novel views from the auto-regressive generator, ArtiFixer

<!-- page 6 -->
6
•
de Lutio, R. et al
3D, which distills its outputs back into the underlying 3D represen-
tation, and ArtiFixer 3D+, which re-applies the auto-regressive
model as a post-procesessing step on top of ArtiFixer 3D (as in
[Wu et al. 2025c]). We assess their ability to enhance in-the-wild
captures against a wide range of prior work in Sec. 5.2 and their
capacity to synthesize unobserved regions on a more challenging
dataset split against a smaller set of relevant baselines in Sec. 5.3.
We validate the contribution of individual components in Sec. 5.4.
5.1
Implementation
We implement our method in PyTorch [Paszke et al. 2019] and train
it on 128 H100 GPUs, using a batch size of one per GPU (128 total).
We use FlashAttention-3 [Shah et al. 2024] for acceleration. In our
main experiments, we finetune the bidirectional model described in
Sec. 4.1 for 15,000 iterations using AdamW [Loshchilov and Hutter
2019] with a learning rate of 1 × 10−5. We then initialize the causal
model for 5,000 iterations with the same learning rate, followed by
2,000 iterations of auto-regressive rollout and DMD training, using
learning rates of 2 × 10−6 for the generator and 4 × 10−7 for the fake
score function. For the ablations, we use a truncated schedule of
10,000 + 2,000 + 600 iterations to reduce computational cost. We use
3DGUT [Wu et al. 2025a] with MCMC densification [Kheradmand
et al. 2024] for the initial reconstructions used by our model.
5.2
Enhancing In-the-Wild Captures
Datasets. We run comparisons on the Nerfbusters dataset [War-
burg et al. 2023] and DL3DV [Ling et al. 2024] using the splits pro-
vided by [Wu et al. 2025c] and on the Mip-NeRF 360 dataset [Barron
et al. 2022] with the splits proposed by [Wu et al. 2024] and used in
subsequent work [Gao* et al. 2024; Wu et al. 2025b].
Baselines. We compare ArtiFixer to an extensive set of baselines,
including the original 3DGS [Kerbl et al. 2023] and 2DGS [Huang
et al. 2024], NeRF variants [Barron et al. 2023; Tancik et al. 2023], non-
generative sparse reconstruction methods [Somraj et al. 2023; Yang
et al. 2023; Zhu et al. 2024], and other diffusion-based work [Gao*
et al. 2024; Sargent et al. 2024; Warburg et al. 2023; Wu et al. 2025c,
2024, 2025b; Wynn and Turmukhambetov 2023; Yin et al. 2025a].
Metrics. We calculate PSNR, SSIM [Wang et al. 2004], LPIPS [Zhang
et al. 2018], and FID score [Heusel et al. 2017] on Nerfbusters and
DL3DV using the exact same protocol and metric implementations
as Difix3D+ [Wu et al. 2025c]. On Mip-NeRF 360, we calculate PSNR,
SSIM, and LPIPS across the 3, 6, and 9 view splits using the same
implementations as GenFusion [Wu et al. 2025b].
Results. We present quantitative results for Nerfbusters and DL3DV
in Table 1 and Mip-NeRF 360 in Table 2. We provide visual com-
parisons in Fig. 8 and Fig. 9. All ArtiFixer variants outperform all
baselines by a substantial margin. Although the different variants
produce similar renderings, ArtiFixer’s are slightly sharper, while
ArtiFixer 3D’s are even more consistent with the source images
at the cost of some blurriness due to its explicit 3D representation,
leading to minor increase in PSNR and SSIM and a small degra-
dation in LPIPS and FID in Table 1. Re-applying the generator to
the improved 3D reconstruction (ArtiFixer 3D+) restores some of
this sharpness, leading to renderings are crisper than ArtiFixer 3D
Reference Views
Prediction
Ground Truth
Fig. 5. Reference views. Without the initial rendering condition, ArtiFixer
can generate predictions from the reference views. Although fidelity drops
somewhat, the high-level structure of the scene remains intact.
and slightly more consistent than ArtiFixer. We provide a video
comparison in the supplement.
5.3
Novel Content Generation
Dataset. We evaluate novel content generation by following the
sparse reconstruction protocol described in Sec. B on scenes from
DL3DV, resulting in numerous “holes" that must be corrected in a
manner consistent with existing observations.
Baselines. We compare to a smaller set of baselines most relevant
to our work, notably 3DGUT [Wu et al. 2025a] as the base represen-
tation we provide as initial renderings to our method, image-based
diffusion methods via Difix3D+ [Wu et al. 2025c] and Fixer [NVIDIA
2025], and approaches that build upon bidirectional video mod-
els [Ren et al. 2025; Wu et al. 2025b], instead of an auto-regressive
one such as ours.
Results. We present quantitative results, using the same metrics
as Table 1, in Table 3. We provide qualitative results in Fig. 7 and
comparison videos in the supplement. All ArtiFixer variants out-
perform the next-best method (GenFusion [Wu et al. 2025b]) by
almost 3 dB in PSNR. Gen3C [Ren et al. 2025] gives the next-best
visually appealing results, but its conditioning often does not re-
spect the source content, and its quality is upper-bounded by the
depth estimator it uses to generate its 3D cache (in contrast to
our purely data-driven approach). Difix3D+ [Wu et al. 2025c] and
Fixer [NVIDIA 2025] generally fail to inpaint plausible context due
to their deterministic conditioning.
5.4
Diagnostics
Ablations. We ablate the effectiveness of our opacity mixing strat-
egy by comparing it to variants that instead use channel concatena-
tion or omit the opacity mixing. We also measure the impact of th
causal model weight initialization described in Sec. 4.2. We report re-
sults on Mip-NeRF 360 dataset averaged over all splits in Table 4 and
show that our design choice of starting from the initial rendering
instead of conditioning on it via channel concatenation is essential

<!-- page 7 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
7
Nerfbusters [Warburg et al. 2023]
DL3DV [Ling et al. 2024]
Method
PSNR↑
SSIM↑
LPIPS↓
FID↓
PSNR↑
SSIM↑
LPIPS↓
FID↓
Nerfacto [Tancik et al. 2023]
17.29
0.621
0.402
134.65
17.16
0.581
0.430
112.30
3DGS [Kerbl et al. 2023]
17.66
0.678
0.327
113.84
17.18
0.588
0.384
107.23
Nerfbusters [Warburg et al. 2023]
17.72
0.647
0.352
116.83
17.45
0.606
0.370
96.61
GANeRF [Roessle et al. 2023]
17.42
0.611
0.354
115.60
17.54
0.610
0.342
81.44
NeRFLiX [Zhou et al. 2023]
17.91
0.656
0.346
113.59
17.56
0.610
0.359
80.65
Difix3D (Nerfacto) [Wu et al. 2025c]
18.08
0.653
0.328
63.77
17.80
0.596
0.327
50.79
Difix3D (3DGS) [Wu et al. 2025c]
18.14
0.682
0.287
51.34
17.80
0.598
0.314
50.45
Difix3D+ (Nerfacto) [Wu et al. 2025c]
18.32
0.662
0.279
49.44
17.82
0.613
0.283
41.77
Difix3D+ (3DGS) [Wu et al. 2025c]
18.51
0.686
0.264
41.77
17.99
0.602
0.293
40.86
ArtiFixer
19.83
0.701
0.254
37.78
19.73
0.672
0.231
20.85
ArtiFixer 3D
20.24
0.729
0.267
39.67
20.14
0.705
0.256
24.27
ArtiFixer 3D+
20.12
0.713
0.264
41.17
20.06
0.686
0.242
22.61
Table 1. Artifact removal on Nerfbusters and DL3DV. All ArtiFixer variants outperform prior methods by a considerable margin, improving PSNR by 2 dB.
PSNR ↑
SSIM ↑
LPIPS ↓
Method
3-view
6-view
9-view
3-view
6-view
9-view
3-view
6-view
9-view
Zip-NeRF [Barron et al. 2023]
12.77
13.61
14.30
0.271
0.284
0.312
0.705
0.663
0.633
3DGS [Kerbl et al. 2023]
13.06
14.96
16.79
0.251
0.355
0.447
0.576
0.505
0.446
2DGS [Huang et al. 2024]
13.07
15.02
16.67
0.243
0.338
0.423
0.580
0.506
0.449
FSGS [Zhu et al. 2024]
14.17
16.12
17.94
0.318
0.415
0.492
0.578
0.517
0.468
FreeNeRF [Yang et al. 2023]
12.87
13.35
14.59
0.260
0.283
0.319
0.715
0.717
0.695
SimpleNeRF [Somraj et al. 2023]
13.27
13.67
15.15
0.283
0.312
0.354
0.741
0.721
0.676
DiffusioNeRF [Wynn and Turmukhambetov 2023]
11.05
12.55
13.37
0.189
0.255
0.267
0.735
0.692
0.680
ZeroNVS [Sargent et al. 2024]
14.44
15.51
15.99
0.316
0.337
0.350
0.680
0.663
0.655
ReconFusion [Wu et al. 2024]
15.50
16.93
18.19
0.358
0.401
0.432
0.585
0.544
0.511
GenFusion [Wu et al. 2025b]
15.29
17.16
18.36
0.369
0.447
0.496
0.585
0.500
0.465
GSFixer [Yin et al. 2025a]
15.61
17.27
18.63
0.370
0.426
0.481
0.559
0.478
0.420
CAT3D [Gao* et al. 2024]
16.62
17.72
18.67
0.377
0.425
0.460
0.515
0.482
0.460
ArtiFixer
17.06
18.64
19.96
0.420
0.476
0.518
0.437
0.390
0.353
ArtiFixer 3D
17.29
18.95
20.24
0.451
0.526
0.598
0.440
0.382
0.327
ArtiFixer 3D+
17.51
18.95
20.16
0.444
0.498
0.537
0.441
0.396
0.359
Table 2. Sparse view reconstruction methods on the Mip-NeRF360 dataset. We exceed existing work by a wide margin across every metric.
Method
PSNR↑
SSIM↑
LPIPS↓
FID↓
3DGUT [Wu et al. 2025a]
16.12
0.537
0.445
92.94
Difix3D (Nerfacto) [Wu et al. 2025c]
14.16
0.453
0.545
74.59
Difix3D (3DGS) [Wu et al. 2025c]
16.60
0.599
0.405
52.70
Difix3D+ (Nerfacto) [Wu et al. 2025c]
13.74
0.434
0.483
30.07
Difix3D+ (3DGS) [Wu et al. 2025c]
16.34
0.564
0.382
21.77
Fixer (offline) [NVIDIA 2025]
13.09
0.355
0.584
135.43
Fixer (online) [NVIDIA 2025]
13.93
0.443
0.535
79.44
Gen3C [Ren et al. 2025]
15.50
0.491
0.476
68.36
GenFusion [Wu et al. 2025b]
17.03
0.624
0.392
132.91
ArtiFixer
19.75
0.643
0.303
12.22
ArtiFixer 3D
19.92
0.673
0.306
16.28
ArtiFixer 3D+
20.15
0.662
0.307
13.91
Table 3. Novel content generation. We reconstruct DL3DV scenes follow-
ing a protocol that creates large areas unobserved by training views. We
outperform the next-best method (GenFusion) by almost 3 dB in PSNR.
to rendering consistently with the source imagery. Our causal ini-
tialization method is not essential as the model still converges to a
competitive level of quality, but provides a modest boost.
Conditioning. To ablate our model’s generative ability, we first
drop the initial rendering condition, forcing the model to rely on
No Opacity Mixing
Opacity Mixing
Base Model
Prompt: A cozy autumn-themed display in a retail store, featuring a variety of fall decorations arranged on
white tables and shelves. The scene includes pumpkins, gourds, and decorative pillows in warm orange,
cream, and brown tones. A sign reading 'Hello Fall' is prominently displayed above the arrangement.
Fig. 6. Text-to-video generation. To illustrate our model’s generative
ability, we generate videos from text prompts alone. With opacity mixing, it
retains similar quality to its base model [Wan et al. 2025]
the reference views, and then exclude all conditioning except for
the text prompt as in the original text-to-video model. Although
fidelity drops somewhat, the model still successfully recreates the
high-level structure of the scene. (Fig. 5), and the model retains a
similar text-to-video synthesis quality as its base model (Fig. 6).

<!-- page 8 -->
8
•
de Lutio, R. et al
Method
Direct
Input
Opacity
Mixing
Diffusion
Forcing
PSNR↑
SSIM↑
LPIPS↓
FID↓
Channel Concatenation
✗
✗
✓
14.52
0.391
0.490
87.551
w/o Opacity Mixing
✓
✗
✓
17.34
0.440
0.429
87.058
w/o Initialization
✓
✓
✗
17.58
0.450
0.416
74.924
Full Method
✓
✓
✓
17.99
0.461
0.408
69.43
Table 4. Diagnostics. We evaluate reconstruction quality on Mip-NeRF
360. Denoising input renderings instead of conditioning via channel con-
catenation is crucial to producing outputs consistent with source images.
6
Conclusion
Neural reconstruction and camera-controlled video generation pro-
vide complementary strengths for novel view synthesis. In this work,
we introduced ArtiFixer, an auto-regressive video diffusion model
that seeks to combine the advantages of both paradigms. ArtiFixer
transforms corrupted renderings of reconstructed scenes into clean,
temporally consistent frames, while retaining sufficient generative
capacity to inpaint unobserved regions and the efficiency required
for interactive use. The strong conditioning signal from the recon-
structed scene significantly simplifies distillation and conversion
to an auto-regressive formulation, enabling ArtiFixer to generate
long video sequences with lesser quality degradation.
Limitations. While ArtiFixer represents an important step to-
ward real-time and unconstrained exploration of reconstructed en-
vironments, limitations remain. The current inference speed, while
reaching interactive rates, is still significantly slower than direct
rendering from neural scene representations. Moreover, ArtiFixer
decodes frames in temporal chunks, which can introduce unde-
sirable latency in downstream applications such as embodied AI.
Promising directions for future work include improving latency by
further reducing the number of denoising steps, as well as enabling
single frame decoding while maintaining temporal coherence.
7
Acknowledgments
We thank Zian Wang and Nicholas Sharp for their helpful advice
and feedback throughout this project.
References
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Liang-
hao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong
Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Cheng-
long Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv,
Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong
Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue
Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang,
Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi
Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi
Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 [cs.CV]
https://arxiv.org/abs/2511.21631
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.
2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In CVPR.
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.
2023. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. In ICCV.
Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian,
Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023.
Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv
preprint arXiv:2311.15127 (2023).
Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu,
and Hao Su. 2021. Mvsnerf: Fast generalizable radiance field reconstruction from
multi-view stereo. In ICCV. 14124–14133.
Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vin-
cent Sitzmann. 2025. Diffusion forcing: Next-token prediction meets full-sequence
diffusion. NeurIPS 37 (2025), 24081–24125.
Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao
Ban, and Cho-Jui Hsieh. 2025. Self-Forcing++: Towards Minute-Scale High-Quality
Video Generation. arXiv preprint arXiv:2510.02283 (2025).
Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. 2022. Depth-supervised
nerf: Fewer views and faster training for free. In CVPR. 12882–12891.
Tobias Fischer, Samuel Rota Bulò, Yung-Hsu Yang, Nikhil Keetha, Lorenzo Porzi, Nor-
man Müller, Katja Schwarz, Jonathon Luiten, Marc Pollefeys, and Peter Kontschieder.
2025. FlowR: Flowing from Sparse to Dense 3D Reconstructions. In ICCV.
Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur Brussee, Ricardo Martin-
Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole*. 2024. CAT3D:
Create Anything in 3D with Multi-View Diffusion Models. NeurIPS.
Google DeepMind. 2024. Veo: A Generative Model for High-Quality Video. https:
//deepmind.google/technologies/veo/. Accessed: 2025.
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local
nash equilibrium. NeurIPS 30 (2017).
Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic
models. NeurIPS (2020).
Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-
Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng
Liu, Zhengqi Li, and Hao Tan. 2025. RELIC: Interactive Video World Model with
Long-Horizon Memory. arXiv:2512.04040 [cs.CV] https://arxiv.org/abs/2512.04040
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2D
Gaussian Splatting for Geometrically Accurate Radiance Fields. In SIGGRAPH Asia.
Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. 2025. Self
Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. NeurIPS.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on
Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng,
Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 2024. 3d
gaussian splatting as markov chain monte carlo. Advances in Neural Information
Processing Systems 37 (2024), 80965–80986.
Hanyang Kong, Xingyi Yang, Xiaoxu Zheng, and Xinchao Wang. 2025. WorldWarp:
Propagating 3D Geometry with Asynchronous Video Diffusion. arXiv preprint
arXiv:2512.19678 (2025).
Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. 2025.
Cameras as Relative Positional Encoding. NeurIPS.
Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu
Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for
deep learning-based 3d vision. In CVPR. 22160–22169.
Yaron Lipman, {Ricky T.Q.} Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.
2023a. Flow Matching for Generative Modeling. In ICLR.
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le.
2023b. Flow Matching for Generative Modeling. In ICLR.
Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. 2025. Rolling forcing:
Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161
(2025).
Xinhang Liu, Jiaben Chen, Shiu-hong Kao, Yu-Wing Tai, and Chi-Keung Tang. 2024.
Deceptive-nerf: Enhancing nerf reconstruction using pseudo-observations from
diffusion models. ECCV.
Xingchao Liu, Chengyue Gong, and Qiang Liu. 2023. Flow Straight and Fast: Learning
to Generate and Transfer Data with Rectified Flow. In ICLR.
Xi Liu, Chaoyi Zhou, and Siyu Huang. 2022. 3DGS-Enhancer: Enhancing Unbounded
3D Gaussian Splatting with View-consistent 2D Diffusion Priors. NeurIPS.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In
ICLR.
Yifan Lu, Xuanchi Ren, Jiawei Yang, Tianchang Shen, Zhangjie Wu, Jun Gao, Yue
Wang, Siheng Chen, Mike Chen, Sanja Fidler, et al. 2025. InfiniCube: Unbounded
and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video
Models. In ICCV.
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ra-
mamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields
for View Synthesis. In ECCV.
Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas
Geiger, and Noha Radwan. 2022. RegNeRF: Regularizing Neural Radiance Fields for
View Synthesis from Sparse Inputs. In CVPR.
NVIDIA. 2025. NVIDIA Fixer. https://huggingface.co/nvidia/Fixer. Accessed: 2026-01-
26.
NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai,
Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, Daniel Dworakowski,
Jiaojiao Fan, Michele Fenzi, Francesco Ferroni, Sanja Fidler, Dieter Fox, Songwei
Ge, Yunhao Ge, Jinwei Gu, Siddharth Gururani, Ethan He, Jiahui Huang, Jacob

<!-- page 9 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
9
Huffman, Pooya Jannaty, Jingyi Jin, Seung Wook Kim, Gergely Klár, Grace Lam,
Shiyi Lan, Laura Leal-Taixe, Anqi Li, Zhaoshuo Li, Chen-Hsuan Lin, Tsung-Yi Lin,
Huan Ling, Ming-Yu Liu, Xian Liu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun
Mo, Arsalan Mousavian, Seungjun Nah, Sriharsha Niverty, David Page, Despoina
Paschalidou, Zeeshan Patel, Lindsey Pavao, Morteza Ramezanali, Fitsum Reda,
Xiaowei Ren, Vasanth Rao Naik Sabavat, Ed Schmerling, Stella Shi, Bartosz Stefaniak,
Shitao Tang, Lyne Tchapmi, Przemek Tredak, Wei-Cheng Tseng, Jibin Varghese,
Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Xinyue
Wei, Jay Zhangjie Wu, Jiashu Xu, Wei Yang, Lin Yen-Chen, Xiaohui Zeng, Yu Zeng,
Jing Zhang, Qinsheng Zhang, Yuxuan Zhang, Qingqing Zhao, and Artur Zolkowski.
2025. Cosmos World Foundation Model Platform for Physical AI. https://arxiv.org/
abs/2501.03575
OpenAI. 2024. Sora: Creating Video from Text. https://openai.com/sora. Accessed:
2025.
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019.
Pytorch: An imperative style, high-performance deep learning library. Advances in
neural information processing systems 32 (2019).
Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion:
Text-to-3D using 2D Diffusion. In ICLR.
Xuanchi Ren, Yifan Lu, Hanxue Liang, Jay Zhangjie Wu, Huan Ling, Mike Chen, Francis
Fidler, Sanja annd Williams, and Jiahui Huang. 2024. SCube: Instant Large-Scale
Scene Reconstruction using VoxSplats. In NeurIPS.
Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-
David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. 2025. GEN3C:
3D-Informed World-Consistent Video Generation with Precise Camera Control. In
CVPR.
Barbara Roessle, Jonathan T Barron, Ben Mildenhall, Pratul P Srinivasan, and Matthias
Nießner. 2022. Dense depth priors for neural radiance fields from sparse input views.
In CVPR. 12892–12901.
Barbara Roessle, Norman Müller, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder,
and Matthias Nießner. 2023. Ganerf: Leveraging discriminators to optimize neural
radiance fields. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–14.
Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi
Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, and Jiajun Wu. 2024.
ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image. In CVPR.
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri
Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony and
Low-precision. arXiv:2407.08608 [cs.LG] https://arxiv.org/abs/2407.08608
Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman,
and Xun Huang. 2025. MotionStream: Real-Time Video Generation with Interactive
Motion Controls. arXiv preprint arXiv:2511.01266 (2025).
Nagabhushan Somraj, Adithyan Karanayil, and Rajiv Soundararajan. 2023. SimpleN-
eRF: Regularizing Sparse Input Neural Radiance Fields with Simpler Solutions. In
SIGGRAPH Asia. doi:10.1145/3610548.3618188
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon,
and Ben Poole. 2020. Score-based generative modeling through stochastic differential
equations. arXiv preprint arXiv:2011.13456 (2020).
Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexan-
der Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. 2023. Nerfstudio:
A modular framework for neural radiance field development. In ACM SIGGRAPH
2023 Conference Proceedings. 1–12.
Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu
Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang,
Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua
Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili
Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu
Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang,
Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou,
Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong
Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng,
Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced
Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).
Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. 2023. Sparsenerf:
Distilling depth ranking for few-shot novel view synthesis. In ICCV. 9065–9076.
Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv,
Guangzhong Sun, Xin Tong, and Jiaolong Yang. 2025. MoGe-2: Accurate Monocular
Geometry with Metric Scale and Sharp Details. In CVPR.
Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. Image quality as-
sessment: from error visibility to structural similarity. IEEE Transactions on Image
Processing 13, 4 (2004), 600–612. doi:10.1109/TIP.2003.819861
Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, and Angjoo
Kanazawa. 2023. Nerfbusters: Removing ghostly artifacts from casually captured
nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision.
18120–18130.
Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng
Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. 2025c. DIFIX3D+: Improving 3D
Reconstructions with Single-Step Diffusion Models. In CVPR. 26024–26035.
Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan
Gojcic. 2025a. 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian
Splatting. In CVPR.
Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson,
Pratul P. Srinivasan, Dor Verbin, Jonathan T. Barron, Ben Poole, and Aleksander
Holynski. 2024. ReconFusion: 3D Reconstruction with Diffusion Priors. In CVPR.
Sibo Wu, Congrong Xu, Binbin Huang, Geiger Andreas, and Anpei Chen. 2025b. CVPR.
Jamie Wynn and Daniyar Turmukhambetov. 2023. DiffusioNeRF: Regularizing Neural
Radiance Fields with Denoising Diffusion Models. In CVPR.
Jiawei Yang, Marco Pavone, and Yue Wang. 2023. FreeNeRF: Improving Few-shot
Neural Rendering with Free Frequency Regularization. In CVPR.
Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang
Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, and Song Hanand Yukang
Chen. 2025.
LongLive: Real-time Interactive Long Video Generation.
(2025).
arXiv:2509.22622 [cs.CV]
Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T
Freeman, and Taesung Park. 2024. One-step Diffusion with Distribution Matching
Distillation. In CVPR.
Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli
Shechtman, and Xun Huang. 2025b. From Slow Bidirectional to Fast Autoregressive
Video Diffusion Models. CVPR.
Xingyilang Yin, Qi Zhang, Jiahao Chang, Ying Feng, Qingnan Fan, Xi Yang, Chi-Man
Pun, Huaqi Zhang, and Xiaodong Cun. 2025a. GSFixer: Improving 3D Gaussian
Splatting with Reference-Guided Video Diffusion Priors. arXiv:2508.09667 [cs.CV]
https://arxiv.org/abs/2508.09667
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021. pixelNeRF: Neural
Radiance Fields from One or Few Images. In CVPR.
Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun
Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024. Viewcrafter: Tam-
ing video diffusion models for high-fidelity novel view synthesis. arXiv preprint
arXiv:2409.02048 (2024).
Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger.
2022. Monosdf: Exploring monocular geometric cues for neural implicit surface
reconstruction. NeurIPS 35, 25018–25032.
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018.
The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.
Jensen (Jinghao) Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao,
Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. 2025. Stable Virtual
Camera: Generative View Synthesis with Diffusion Models. arXiv preprint (2025).
Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, and Jiangbo Lu.
2023. NeRFLix: High-quality neural view synthesis by learning a degradation-driven
inter-viewpoint mixer. In CVPR. 12363–12374.
Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. 2024. FSGS: Real-Time
Few-Shot View Synthesis using Gaussian Splatting. In ECCV.

<!-- page 10 -->
10
•
de Lutio, R. et al
Scene 15ff83
3DGUT
GenFusion
Gen3C
Artifixer3D+ (Ours)
Ground Truth
Scene eb4cf5
Scene a40146
Scene 493816
Scene 946f49
Scene 8fdc51
Scene 35317e
3DGUT
Difix 3D+
Fixer (Offline)
Artifixer3D+ (Ours)
Ground Truth
Scene c07692
Scene d3812a
Fig. 7. DL3DV results. We compare ArtiFixer 3D+ to its initial 3DGUT [Wu et al. 2025a] input, two baselines that build upon bidirectional video diffusion
models (top rows), and two that leverage image models (bottom rows). GenFusion [Wu et al. 2025b]’s video model generates 16 frames at a time, requiring a
iterative distillation process that leads to blurry results, especially in empty areas. Gen3C [Ren et al. 2025]’s renderings are sharper but often do not respect the
source content (background in top row), have incorrect geometry (second row), and exhibit color shift (sixth row). Methods that directly take renderings
as input without opacity mixing [NVIDIA 2025; Wu et al. 2025c] fail to reconstruct empty regions. Our method can reconstruct plausible and consistent
geometry even when the initial rendering is highly degraded. Please refer to the supplement for comparison videos.

<!-- page 11 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
11
Bonsai
3DGUT
GenFusion
Artifixer3D+ (Ours)
Ground Truth
Flowers
Garden
Stump
Fig. 8. Mip-NeRF 360 results. We present visualizations of Mip-NeRF’s most challenging split (3-view). Our results far exceed all prior work both quantitatively
and qualitatively. Our method is able to recover the correct geometry from the reference views even in scenarios where the input rendering is completely
inaccurate (table in third row). We render videos of every scene in the supplement.
Aloe
3DGUT
Nerfbusters
NeRFLiX
Difix 3D+
Artifixer3D+ (Ours) Ground Truth
Garbage
Fig. 9. Nerfbusters results. As with the other datasets, our method is the only to generate plausible visuals in unseen regions while preserving the fidelity of
the original content.

<!-- page 12 -->
12
•
de Lutio, R. et al
A
Denoising Steps
As ArtiFixer starts from renderings instead of pure noise, it is
able to generate plausible visuals in fewer than four steps in most
cases, although sharpness and temporal consistency suffer some-
what in empty areas (Fig. 10). This problem is largely mitigated
when revisiting previously explored areas in our ArtiFixer 3D
and ArtiFixer 3D+ variants, as the 3D distillation process provides
strong conditioning for subsequent generations (please see videos
for an example).
B
Sparse Reconstruction
Camera Sampling. We describe our camera sampling strategy in
Algorithm 1. Given a set of camera poses P, we define the pairwise
distance between two poses as 𝑑= ||R𝑖−R𝑗||𝐹+ ||t𝑖−t𝑗||2. We
initialize the clustering process by identifying the pair (𝑃1, 𝑃2) find
the camera pair (𝑃1, 𝑃2) that maximizes this distance and using them
as seeds for groups 𝐺1 and 𝐺2. with the largest distance, and seed
groups 𝐺1 and 𝐺2. The remaining cameras are assigned to the group
of their nearest seed. Finally, to evaluate varying levels of sparsity,
we apply farthest point sampling within each group to select subsets
of size 𝐾= {2, · · · , 12}.
ALGORITHM 1: CameraSampling
Input: Camera poses P, Selection count 𝐾, Distance function 𝑑
Output: Selected subsets S1 ⊂𝐺1 and S2 ⊂𝐺2
/* 1. Find global farthest camera pair
*/
(𝑃1, 𝑃2) ←argmax𝑃𝑖,𝑃𝑗∈P 𝑑(𝑃𝑖, 𝑃𝑗);
/* 2. Cluster: Assign cameras to nearest seed camera */
𝐺1 ←{𝑃∈P | 𝐷(𝑃, 𝑃1) ≤𝐷(𝑃, 𝑃2)};
𝐺2 ←P \ 𝐺1;
/* 3. Select Top-K points in EACH group
*/
foreach 𝑖∈{1, 2} do
S𝑖←{𝑃𝑖} ;
// Start with the seed camera
while |S𝑖| < 𝐾and |S𝑖| < |𝐺𝑖| do
/* Find pose maximizing distance to current
selection
*/
𝑃𝑛𝑒𝑥𝑡←argmax𝑃∈𝐺𝑖\S𝑖
 min𝑠∈S𝑖𝐷(𝑃,𝑠);
S𝑖←S𝑖∪{𝑃𝑛𝑒𝑥𝑡};
end
end
return S1, S2
Reconstruction. We generate the initial reconstructions we pass to
the ArtiFixer model using the official 3DGUT implementation [Wu
et al. 2025a] with MCMC [Kheradmand et al. 2024] sampling. We
run each reconstruction for 10,000 iterations, taking slightly less
than 10 minutes per reconstruction.
Captioning. We generate captions for each DL3DV scene from
Qwen3-VL-30B-A3B-Instruct [Bai et al. 2025] on different frame
subsets to encourage prompt diversity. Similar to [Hong et al. 2025],
we suppress descriptions of ego-camera movement to avoid en-
tanglement with camera ray conditioning. We use the following
prompt:
You are a video captioning specialist whose goal is to generate high-
quality English prompts by referring to the details of the user’s input videos.
Your task is to carefully analyze the content, context, and actions within the
video, and produce a complete, expressive, and natural-sounding caption
that accurately conveys the scene. The caption should preserve the original
intent and meaning of the video while enhancing its clarity and descriptive
richness. Strictly adhere to the formatting of the examples provided.
Task Requirements: 1. You need to describe the main subject of the video in
detail, including their appearance, actions, expressions, and the surrounding
environment. 2. You should never describe any details about the camera
movement or camera angles. 3. Your output should convey natural movement
attributes, incorporating natural actions related to the described subject
category, using simple and direct verbs as much as possible. 4. You should
reference the detailed information in the video, such as character actions,
clothing, backgrounds, and emphasize the details in the photo. 5. Control
the output prompt to around 80-100 words. 6. No matter what language the
user inputs, you must always output in English.
Example of the English prompt: 1. A Japanese fresh film-style photo of a
young East Asian girl with double braids sitting by the boat. The girl wears
a white square collar puff sleeve dress, decorated with pleats and buttons.
She has fair skin, delicate features, and slightly melancholic eyes, staring
directly at the camera. Her hair falls naturally, with bangs covering part of
her forehead. She rests her hands on the boat, appearing natural and relaxed.
The background features a blurred outdoor scene, with hints of blue sky,
mountains, and some dry plants. The photo has a vintage film texture. A
medium shot of a seated portrait. 2. An anime illustration in vibrant thick
painting style of a white girl with cat ears holding a folder, showing a slightly
dissatisfied expression. She has long dark purple hair and red eyes, wearing
a dark gray skirt and a light gray top with a white waist tie and a name
tag in bold Chinese characters. The background has a light yellow indoor
tone, with faint outlines of some furniture visible. A pink halo hovers above
her head, in a smooth Japanese cel-shading style. A close-up shot from a
slightly elevated perspective. 3. CG game concept digital art featuring a huge
crocodile with its mouth wide open, with trees and thorns growing on its
back. The crocodile’s skin is rough and grayish-white, resembling stone or
wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With
its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The
background features a dusk sky with some distant trees, giving the overall
scene a dark and cold atmosphere. A close-up from a low angle. 4. In the
style of an American drama promotional poster, Walter White sits in a metal
folding chair wearing a yellow protective suit, with the words "Breaking Bad"
written in sans-serif English above him, surrounded by piles of dollar bills
and blue plastic storage boxes. He wears glasses, staring forward, dressed
in a yellow jumpsuit, with his hands resting on his knees, exuding a calm
and confident demeanor. The background shows an abandoned, dim factory
with light filtering through the windows. There’s a noticeable grainy texture.
A medium shot with a straight-on close-up of the character.
Directly output the English text.

<!-- page 13 -->
ArtiFixer : Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models
•
13
Input Rendering
1 Step
2 Steps
3 Steps
4 Steps
Fig. 10. Denoising steps. We vary the number of denoising steps when beginning from the initial degraded rendering. ArtiFixer can render plausible content
in as little as 1 step, although sharpness and temporal consistency suffer somewhat in empty areas.
