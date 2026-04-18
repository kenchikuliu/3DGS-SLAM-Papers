<!-- page 1 -->
DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic
Simulation with Online Diffusion Enhancer
Yuxuan Zhang1*
Katar´ına T´othov´a1*
Zian Wang1,2
Kangxue Yin1
Haithem Turki1
Riccardo de Lutio1
Yen-Yu Chang1,3
Or Litany1,4
Sanja Fidler1,2
Zan Gojcic1
1NVIDIA
2University of Toronto
3Cornell University
4Technion
{alezhang, ktothova, zgojcic}@nvidia.com
Input Video Frames
Our Enhancement Results
Our Enhancement Result
Input Image
Input Image
Our Enhancement Result
Figure 1. DiffusionHarmonizer on Driving Scenes. Our method transforms artifact-prone neural-rendered frames into temporally coherent
simulations, improving their realism by jointly correcting shadows, lighting, appearance discrepancies and reconstruction artifacts.
Abstract
Simulation is essential to the development and evaluation
of autonomous robots such as self-driving vehicles. Neural
reconstruction is emerging as a promising solution as it en-
ables simulating a wide variety of scenarios from real-world
data alone in an automated and scalable way. However,
while methods such as NeRF and 3D Gaussian Splatting
can produce visually compelling results, they often exhibit
artifacts particularly when rendering novel views, and fail
to realistically integrate inserted dynamic objects, especially
when they were captured from different scenes. To over-
come these limitations, we introduce DiffusionHarmonizer,
an online generative enhancement framework that trans-
∗Equal contribution.
forms renderings from such imperfect scenes into temporally
consistent outputs while improving their realism. At its core
is a single-step temporally-conditioned enhancer that is con-
verted from a pretrained multi-step image diffusion model,
capable of running in online simulators on a single GPU.
The key to training it effectively is a custom data curation
pipeline that constructs synthetic–real pairs emphasizing
appearance harmonization, artifact correction, and light-
ing realism. Experiments show that DiffusionHarmonizer
substantially improves perceptual realism, being chosen by
84.28% of users in our comparative study over the second-
best method. Furthermore, it matches the temporal coher-
ence of state-of-the-art video models while maintaining the
inference efficiency of single-step image models, offering
a scalable and practical solution for simulation in both re-
search and production settings.
1
arXiv:2602.24096v2  [cs.CV]  5 Mar 2026

<!-- page 2 -->
1. Introduction
Recent advances in neural reconstruction have made it possi-
ble to recover high-fidelity simulation environments directly
from real-world sensor data, paving the way for scalable
and photorealistic simulation in autonomous driving and
robotics [5, 8, 39, 52]. These methods typically decompose
a scene into a static background and a set of foreground
assets (e.g. cars and people) whose positions and trajecto-
ries can be manipulated during simulation. The extracted
foreground assets can be stored in a databank and flexibly
combined with different backgrounds or scenes.
Despite this progress, two fundamental challenges remain:
(1) Novel-view artifacts: while current methods achieve high
visual quality near training viewpoints, they often produce
spurious geometry, missing regions, and other artifacts when
rendered from sparsely observed or extrapolated viewpoints
that significantly deviate from the training trajectory. Sim-
ilar artifacts arise when the position or trajectory of the
foreground assets is manipulated; and (2) Object insertion
artifacts: when (dynamic) foreground objects, whether syn-
thetic assets or reconstructions from separate captures, are
inserted into reconstructed scenes, the resulting composites
frequently contain tone discrepancies, missing shadows, or
lighting mismatches (see Fig. 1).
In this work, we aim to improve the photorealism of neu-
rally reconstructed simulation environments by leveraging
generative models as post-rendering enhancers to address
the above challenges. We formulate this problem as an
image-to-image translation task: given artifact-prone or visu-
ally inconsistent frames rendered from a reconstructed scene,
our goal is to generate temporally consistent and harmonized
video frames that preserve the underlying scene structure.
Although image-to-image translation and video editing
have been extensively studied, existing models fail to meet
the requirements of online simulation: Video-based gen-
erative models are computationally expensive and cannot
operate in online simulation under practical resource bud-
gets (e.g., a single H100 GPU), while image-based models
lack temporal consistency, leading to flickering and unsta-
ble dynamics. Both approaches often struggle to reliably
model lighting such as cast shadows and may distort existing
scene geometry and appearance, which is undesirable for
physically grounded simulation.
To address these challenges, we convert a pretrained non-
distilled image diffusion model into a single-step, tempo-
rally conditioned enhancement model suitable for online
simulation. This requires a targeted training strategy that
preserves scene structure and mitigates artifacts arising from
the noise-trajectory mismatch which emerges due to the dis-
crepancy between the multi-step noise schedule used during
pretraining and the single-step mode at inference time. In-
deed, naively fine-tuning a pretrained multi-step diffusion
model in a single denoising step introduces high-frequency
checkerboard artifacts. To address this issue, we introduce
a multi-scale perceptual loss that stabilizes high-frequency
behavior and effectively suppresses artifacts induced by the
mismatched denoising trajectories.
To overcome the lack of high-quality paired supervision,
we introduce a scalable data curation pipeline that synthe-
sizes training pairs capturing harmonization, shadow correc-
tion, and artifact correction. Specifically, we combine (i)
artifact-corrupted renderings generated using the four degra-
dation modes from DIFIX3D+ [44]: sparse reconstruction,
cycle reconstruction, cross-referencing, and underfitting; (ii)
appearance-varying data created by randomized modifica-
tions of ISP parameters such as tone mapping, exposure, and
white balance, etc.; (iii) pairs of images with varying illumi-
nation synthesized using a generative relighting model [19];
(iv) physically based shadow data obtained by rendering
synthetic scenes under varying environment-map and light-
source configurations; and (v) asset re-insertion composites
where reconstructed objects are reinserted without shadows
to create realistic, supervision-rich pairs for both harmo-
nization, artifact correction, and shadow synthesis. These
components jointly supply the complementary signals re-
quired to learn robust appearance harmonization, artifact
correction, and physically plausible shadow generation.
By combining this data generation pipeline with our tar-
geted training strategy, we obtain DiffusionHarmonizer, a
unified simulation harmonizer that jointly (i) corrects novel-
view reconstruction artifacts, (ii) harmonizes foreground and
background appearance, and (iii) synthesizes realistic shad-
ows for inserted objects. The resulting model is efficient to
fine-tune using only a small curated dataset, and substan-
tially improves the photorealism of frames rendered from
neural simulators.
2. Related Work
Image and Video Harmonization aims to adjust the ap-
pearance of the inserted foreground objects so that they
blend naturally with the background scene.
Early ap-
proaches [6, 12, 21, 38] typically formulated harmonization
as a frame-to-frame regression problem implemented using
autoencoders. Ke et al. [17] introduced a white-box frame-
work that learns interpretable filters aligned with human
perception and preference. Xu et al. [48] proposed to first
convert the input image into a linear color space, perform
harmonization, and then transform it back to sRGB. More
recently, with the advent of powerful generative models,
diffusion-based approaches were explored, either through
fine-tuning large pretrained diffusion models [4, 32] or via
training-free formulations [25]. Ljungbergh et al. [24] de-
sign a frame-wise enhancer trained on renders of recon-
structed driving scenes with real dynamic actors replaced
by recreated 3D assets. Unlike in our work, their target is
only shadow casting and relighting, omitting artifacts that
2

<!-- page 3 -->
are common in neural reconstruction.
Extending the harmonization to video introduces the chal-
lenge of temporal coherence. Flow-based constraints [15]
and local color mapping [26] reduce flickering but still oper-
ate independently on each frame. More recent works process
videos holistically: VHTT [13] introduces a Video Triplet
Transformer modeling both global and dynamic temporal
variations, while GenCompositor [49] employs a latent video
diffusion model to harmonize inserted objects. However,
existing models primarily address foreground harmoniza-
tion and do not handle reconstruction artifacts or synthesize
physically plausible shadows—which are essential for har-
monizing frames rendered from neural reconstructed scenes
used in simulation.
Neural Reconstruction and Generative Priors. Neural
reconstruction methods often fail in the sparse-view regime
and struggle to generalize to novel viewpoints that are far
from the training views, primarily due to the lack of strong
data priors. To better constrain the optimization, early works
introduced various geometric [7, 42, 50], photometric [40],
and regularization-based priors [29].
More recent works have leveraged generative models to
incorporate appearance priors and thus improve reconstruc-
tion quality. DiffusioNeRF [47] leverages a learned scene
geometry and appearance prior by training a diffusion model
on RGB-D patches from synthetic data. GANeRF [33] lever-
ages an adversarial formulation to refine a NeRF [28] using
a conditional generative network. Nerfbusters [43] incorpo-
rate the prior from a pretrained 3D diffusion model into the
scene by using a density score distillation sampling loss [30].
Several works rely on generating new views of the scene
to use as supervision. ReconFusion [46] proposes a diffu-
sion model that is conditioned on all input views through
feature maps rendered by a PixelNeRF [51] model. 3DGS-
Enhancer [22] employs a video diffusion model to generate
novel views, Cat3D [11] uses a multi-view latent diffusion
model to generate novel views of the scene.
In addition, DIFIX3D+ [44] proposes to use an image-to-
image translation model to remove artifacts. Beyond usage
to generate novel views to lift into 3D, the efficiency of the
single-step model enables use at rendering time. Flowr [10]
proposes a multi-view flow matching model that learns to
turn novel view renderings into clean images.
3. Method
Our goal is to develop an online harmonization model
that transforms artifact-prone rendered frames from recon-
structed scenes into temporally consistent outputs with im-
proved photorealism. This requires: (i) a lightweight archi-
tecture that enhances streaming frames while maintaining
temporal stability, and (ii) a curated paired dataset that su-
pervises color harmonization, illumination consistency, and
shadow synthesis. We first describe the model architecture
and training objectives, then our data curation pipeline.
3.1. Online Frame-to-Frame Enhancer
Pretrained diffusion models contain strong generative priors
for image translation. We adapt an image-based diffusion
model into a deterministic, single-step enhancer suitable for
online simulation.
Network Architecture. We formulate harmonization as an
image-to-image translation task. Given a degraded frame It
at time t, the enhancer predicts an improved frame ˆIt:
ˆIt = Dϕ
 Fθ
 Eη(It)

,
(1)
where Eη and Dϕ denote the pretrained latent encoder and
decoder, which are kept frozen, and Fθ is the diffusion back-
bone. We fine-tune only Fθ to adapt the pretrained diffusion
model to the harmonization task.
In standard diffusion models, the backbone Fθ is trained
as a denoiser operating on stochastic noisy latents across
many timesteps, with conditioning that encodes the noise
level or diffusion time. In contrast, we repurpose Fθ as a
deterministic single-step enhancer. Specifically, we feed the
clean latent Eη(It) directly into the network without injecting
noise and fix the timestep and text-conditioning tokens to
constant “null” values during both training and inference.
This produces a stable and deterministic mapping from input
latent to enhanced latent and improves frame-wise structural
consistency.
Temporal Conditioning. For online simulation, it is crucial
that per-frame enhancements remain temporally stable. We
therefore extend the backbone Fθ to accept a short temporal
context of previous frames.
Let K denote the context length (we use K = 4 in prac-
tice). At time t, we encode the current degraded frame and
up to K previously enhanced frames:
Zt =
h
Eη(It), Eη(ˆIt−1), . . . , Eη(ˆIt−K)
i
,
(2)
and feed Zt into the backbone with temporal attention layers
interleaved with spatial attention, following video diffusion
architectures. For the first few frames, when fewer than
K previous frames are available, we condition only on the
frames that exist. This design allows the enhancer to use
historical context when it is beneficial, while preserving
frame-wise structure and preventing drift.
3.2. Data Curation Strategy
Training our model requires paired data consisting of artifact-
prone renderings and high-quality photorealistic images that
capture diverse factors such as appearance harmonization,
lighting realism, reconstruction robustness, and shadow con-
sistency. Such supervision is scarce in real-world datasets
and unavailable in existing public sources.
3

<!-- page 4 -->
𝓓𝝓
𝓔𝜼
…
!𝐼!
𝐼!
𝐼!"#,…,!"& 
Data Curation
DiffusionHarmonizer Model
𝓕𝜽
Single-Step
PBR Shadow 
Simulation
Relighting
ISP 
Modification
Assert 
Re-insertion
Artifacts 
Correction
Figure 2. Overview of the data curation pipeline (top) and model architecture (bottom) of DiffusionHarmonizer. We use a single-step
temporally conditioned enhancement model, that is converted from a pretrained multi-step image diffusion model. To train it effectively, we
develop a data curation pipeline to construct synthetic–real pairs emphasizing harmonization, artifact correction and lighting realism.
To address this gap, we design a scalable data curation
pipeline that synthesizes paired supervision data tailored for
our enhancement objectives. The pipeline comprises five
complementary components: Novel-view artifacts correc-
tion, ISP modification, Relighting, Shadow simulation, and
Asset re-insertion, with each targeting a specific visual factor.
An overview is shown in Fig. 2.
Novel-View Artifacts Correction. To handle reconstruc-
tion artifacts in challenging novel view synthesis, we follow
the data strategy of DIFIX3D+ [44]. Specifically, we gen-
erate degraded renderings using four procedures: sparse
reconstruction, cycle reconstruction, cross-referencing, and
deliberate model underfitting. These operations produce
frames with blurred details, missing regions, ghosting, and
spurious geometry. Every degraded frame is paired with its
corresponding clean rendering, providing explicit supervi-
sion for correcting novel-view synthesis artifacts.
ISP Modification. Object captures from different devices
often exhibit image signal processing (ISP) induced tone and
color inconsistencies, leading to foreground–background
appearance mismatches after composition. To simulate such
mismatches, we implement a software ISP that re-renders
captured images with randomized parameters (e.g., tone
mapping, exposure, white balance, etc.). Given an original
capture Iorig, we generate an alternative image IISP using
resampled ISP parameters. Using a mask M (obtained from
SAM2 [31]) that segments the foreground region, we create
a composite:
Imix = M ⊙IISP + (1 −M) ⊙Iorig.
(3)
Here, Imix is used as the input and Iorig as the target, enabling
the model to learn foreground–background color and tone
harmonization.
Relighting.
To model illumination mismatches, we use
an image relighting diffusion model [19] to regenerate se-
lected regions under randomly sampled lighting conditions
while preserving scene geometry and texture. In practice, we
relight only the foreground object, intentionally creating in-
puts where local illumination is inconsistent with the global
scene lighting. These paired examples supervise the model
to resolve lighting discrepancies and to synthesize spatially
consistent illumination across the frame.
Physically Based Shadow Simulation. Accurate cast shad-
ows are critical for realism but are difficult to annotate in real
data. To provide explicit supervision for shadow reasoning,
we use a physically based renderer to synthesize cast shad-
ows under controllable light configurations. We randomly
vary the environment maps in synthetic scenes to modify
the direction, softness, and intensity of the light source, and
generate paired samples with and without shadows. These
pairs provide precise pixel-level cues for the network to learn
physically plausible shadow casting and attenuation.
Asset Re-Insertion. While PBR-based shadow data offer
precise supervision signals, they may not fully match real-
world statistics. To reduce the domain gap, we additionally
leverage dynamic scene reconstruction. Specifically, we re-
construct the static background with 3DGUT [45], extract
dynamic foreground objects with an in-house solution, and
then re-insert the foreground objects into the reconstructed
background scene without casting shadows. The resulting
composites mimic the realistic object insertions but inten-
tionally lack proper shadows and harmonization. Paired
with the original sequences that contain correct shadows and
coherent appearance, they offer rich supervision for learn-
ing realistic shadow synthesis and foreground–background
4

<!-- page 5 -->
harmonization.
Together, these components yield a diverse paired dataset
that captures reconstruction artifacts, ISP-induced appear-
ance mismatches, illumination inconsistencies, and missing
shadows commonly seen in simulated renderings. This cu-
rated data is a key enabler for training our online harmonizer
under limited real-world supervision.
3.3. Training
Stabilizing Single-Step Training. Using a multi-step pre-
trained model in a single-step manner introduces a noise-
trajectory mismatch: the original model is trained to operate
across a full diffusion trajectory, whereas at fine-tuning we
apply it only once on a clean latent. Directly fine-tuning as a
deterministic one-step model often leads to high-frequency
artifacts such as checkerboard patterns.
To stabilize training, we introduce a multi-scale percep-
tual loss computed on randomly sampled squared patches of
varying sizes. Given a predicted frame ˆIt and a ground-truth
target frame Igt
t , we sample square patches ˆP (k)
t
and P (k)
gt
of
side length k ∈[128, 512] at random locations, and define
the multi-scale perceptual loss as:
Lperc = Ek
"X
l
λl
ϕl( ˆP (k)
t
) −ϕl(P (k)
gt )
2
2
#
,
(4)
where ϕl(·) denotes features from the l-th layer of a VGG
network and λl are layer-wise weights.
Sampling multi-scale patches perturbs patch boundaries
relative to the model’s receptive field, emphasizing high-
frequency inconsistencies and suppressing periodic alias-
ing. Empirically, this loss significantly suppresses checker-
board artifacts arising from the noise-trajectory mismatch
(see Sec. 4.3).
Temporal Warping Loss. To further encourage temporal
consistency, we incorporate a warping-based temporal loss.
Given consecutive ground-truth frames Igt
t−1, Igt
t , we esti-
mate the optical flow Ft→t−1 using RAFT [37]. We then
warp the enhanced frame at time t−1 into time t and enforce
consistency in the visible pixels:
Ltemp = 1
|Ω|
X
x∈Ω
h
ˆIt(x)−Warp(ˆIt−1, Ft→t−1)(x)
i2
, (5)
where Ωis the set of pixels for which the warping pro-
duces valid correspondences (i.e., not occluded or leaving
the frame). This loss is computationally tractable benefiting
from our single-step formulation: we only need one for-
ward pass per frame to obtain RGB outputs for supervision,
avoiding the memory overhead of backpropagating through
multi-step diffusion trajectories.
Mixed Temporal Training. Our curated dataset contains
both short video sequences and standalone images. The lat-
ter arise from components such as single-image relighting,
for which high-quality temporal variants are difficult to syn-
thesize. To leverage all available data and avoid overfitting
to strong temporal cues, we train with mixed temporal and
non-temporal batches. The overall training objective is:
Ltotal = λl2 Ll2 + λperc Lperc + λtemp Ltemp,
(6)
where Ll2 is a per-pixel loss between ˆIt and Igt
t , and λtemp =
1 for temporal batches and 0 otherwise.
We first pretrain on paired image data to learn robust per-
frame enhancement, then alternate between temporal and
non-temporal batches. This schedule prevents the model
from depending excessively on the nearby frames and im-
proves robustness when temporal conditioning is weak,
noisy, or partially unavailable.
4. Experiments
We conduct extensive experiments across multiple datasets
and evaluation settings to validate the effectiveness of our
proposed framework. While tested on automotive scenarios,
the method is domain-agnostic and readily applicable to
other settings without modification.
Training Details. Our model is built upon the Cosmos 0.6B
text-to-image diffusion model [1], which contains 0.6B pa-
rameters in the diffusion backbone and 0.14B parameters
in the VAE tokenizer. We freeze the VAE tokenizer dur-
ing training and only fine-tune the diffusion backbone. The
model is trained for 10k iterations for non-temporal pretrain-
ing and additional 4k for temporal training. We train the
model at a resolution of 1024 × 576 with bf16 precision. We
set λl2 = 1, λperc = 1. Additional dataset statistics and
visualizations are provided in the Supplement.
Baselines. We compare our method against both general-
purpose diffusion-based image and video editing models as
well as task-specific harmonization approaches.
• General editing baselines: We evaluate SDEdit [27],
InstructPix2Pix [3], and Video-to-Video Editing (V2V)
[16]. For SDEdit, we replace the original model with
the img2img variant of Stable Diffusion 3 [9] to ensure
state-of-the-art performance. For V2V, we use the VACE
video-to-video editing model based on WAN 2.1 [41].
• Specialized video harmonization methods: We compare
our method with Ke et al. [17] and VHTT [13]. Both
methods were evaluated using official implementations
and pre-trained checkpoints provided by the authors.
Evaluation Protocol. We evaluate our model on simulated
videos from three distinct settings. In all settings, we eval-
uate using 10-second video sequences rendered at 30FPS
with resolution of 1024×576:
• Novel Trajectory Simulation. We reconstruct both static
scenes and dynamic objects (e.g., pedestrians and vehi-
cles), then render novel views by laterally shifting the ego
trajectory by 2 m. Evaluation is performed on 13 holdout
scenes from our internal driving dataset (in-domain test).
5

<!-- page 6 -->
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Figure 3. Comparison with Image and Video Editing Baselines on Out-of-Domain Testing Data. Our method harmonizes color tone and
synthesizes realistic lighting and shadows, while editing baselines often fail to produce physically plausible shadowing. Although both can
reduce reconstruction artifacts, baselines tend to hallucinate inconsistent content and over-edit well-reconstructed regions, whereas our
method preserves scene geometry and input structure. Moreover, image-editing baselines introduce frame-to-frame jitter, whereas our model
maintains strong temporal coherence.
• Dynamic Object Insertion. We insert foreground objects
(e.g., cars) into a pre-reconstructed scene and rendered
in a novel trajectory shifted by 3 meters. Evaluation is
performed on 68 scenes from the public Waymo driving
dataset (out-of-domain test).
• Holdout Datasets with Ground-Truth labels. We further
evaluate our method on holdout datasets for ISP modifica-
tion, relighting, and PBR-based shadow simulation. These
three datasets have accurate ground-truth labels, allowing
the computation of GT-dependent evaluation metrics such
as PSNR, SSIM, and LPIPS.
Metrics. An ideal enhancement model should (1) improve
perceptual realism, (2) preserve geometric and structural
fidelity, and (3) maintain temporal consistency. We evaluate
perceptual quality using FID and FVD, structural preser-
vation using DINO-Struct-Dist, which measures feature-
space similarity between input and output, and temporal
smoothness using the temporal flickering score measured by
VBench++. Because lighting, shadows, and general photore-
alism are difficult to quantify without ground-truth supervi-
sion, we additionally conduct the user study and, following
recent practice [2, 18, 20, 23], employ a pretrained vision-
language model (VLM) to assess overall quality. For holdout
datasets with available ground-truth labels, we further cal-
6

<!-- page 7 -->
Ours
Ke et al.
Simulation
Input
VHTT
Figure 4. Comparison with Harmonization Baselines Meth-
ods. While both our method and harmonization baselines adjust
foreground appearance, the baselines fail to synthesize realistic
shadows, resulting in less coherent composites.
culate PSNR, SSIM, and LPIPS on the region of interest to
assess the alignment with the ground-truth label. We also
report inference speed measured on a single H100 GPU.
4.1. Qualitative Comparison
Fig. 3 presents a qualitative comparison between our method
and state-of-the-art image and video editing baselines, on
out-of-domain testing dataset. Our approach effectively har-
monizes color tone and synthesizes realistic lighting and
shadows, whereas baseline editing models often fail to gen-
erate physically plausible shadowing for inserted objects.
Although both our method and existing editing models can
mitigate reconstruction artifacts such as missing details or
blurry regions, baseline outputs frequently hallucinate con-
tent inconsistent with the underlying input scene structure
and also modify well-reconstructed regions that should re-
main unchanged. In contrast, our method faithfully preserves
scene geometry and underlying input structures. Further-
more, while image-editing baselines can maintain coarse
scene consistency under a fixed random seed, they typically
exhibit frame-to-frame variation in fine details, resulting
in temporal jitter. Our model achieves substantially better
temporal coherence across adjacent frames.
We additionally compare against specialized harmoniza-
tion methods in Fig. 4. While both our method and baseline
harmonization approaches adjust foreground color tone to
match the background, these baselines fail to synthesize real-
istic shadows, leading to less realistic composites. Moreover,
harmonization methods do not correct reconstruction arti-
facts by their design, limiting their applicability in neural
simulation pipelines. Finally, we present more examples
and also further visualize our model’s predictions on hold-
out datasets with ground-truth label in the Supplementary
materials.
4.2. Quantitative Evaluation
We split our quantitative evaluation into two segments: (1)
image and video editing baseline comparison, and (2) video
harmonization baseline comparison.
Image and Video Editing Baselines. As shown in Tab. 1,
our method consistently outperforms all editing baselines
in perceptual quality, achieving better FID and FVD scores.
Compared to the baselines, our model also preserves scene
structure more faithfully, as reflected by markedly lower
DINO-Struct-Dist scores. For temporal consistency, our ap-
proach significantly outperforms image-editing baselines and
achieves performance comparable to video diffusion models,
evidenced by the marginal difference in the VBench tempo-
ral score relative to WAN V2V. On holdout datasets with
ground-truth label (Tab. 2), our predictions exhibit much
closer alignment, achieving large improvements over all
baselines in PSNR, SSIM, and LPIPS. Moreover, our method
is at least 1.8× faster than image-editing baselines and 10×
faster than video-editing baselines, enabling online deploy-
ment.
Video Harmonization Baselines.
We compare against
harmonization baselines on the ISP modification dataset
(where segmentation masks are available, required by the
harmonization baselines), and report the results in Tab. 3.
Our method consistently outperforms VHTT [13] and Ke et
al. [17] across all image-quality metrics.
User Study. We conduct a user study to assess the over-
all quality. For each comparison, human evaluators are
shown: (1) the input image as reference, and (2) a pair
of predictions—our output and a baseline output—and asked
to select the better result. The order of the predictions is ran-
domized to avoid systematic bias. We recruit 45 evaluators
(15 per baseline), each completing 50 pairwise comparisons.
We report more configuration details in the supplementary
materials. In Tab. 4, we report the mean preference rate and
standard deviation across evaluators. As shown in the results,
human evaluators consistently prefer our method over all
baselines.
4.3. Ablation Study
We analyze the effectiveness of our model design and data
curation strategy through extensive ablation studies.
Multi-Scale Perceptual Loss. We assess the impact of our
multi-scale perceptual loss introduced in Eq. (4). As shown
in Fig. 5, removing perceptual supervision leads to over-
smoothed outputs, while using a conventional LPIPS loss
produces high-frequency artifacts. Our multi-scale formula-
tion mitigates these artifacts and yields perceptually better
results.
Temporal Components.
We evaluate the effect of tem-
poral conditioning and temporal loss design by comparing
models trained without the temporal modules and temporal
7

<!-- page 8 -->
Novel Trajectory Simulation (In-domain)
Object Insertion Simulation (Out-of-domain)
Method
Inference Speed ↓
FID↓
FVD↓
DINO Struct.↑
Temporal Cons.↑
FID↓
FVD↓
DINO Struct. ↑
Temporal Cons. ↑
SDEdit (SD 3)
398ms
129.92
753.39
0.7075
0.9661
106.17
506.91
0.6345
0.9524
InstructPix2Pix
555ms
153.94
680.92
0.6339
0.9203
128.72
573.20
0.5793
0.9085
Wan-Video V2V
2827ms
134.98
506.86
0.8289
0.9828
104.42
474.96
0.8226
0.9675
Our Model
212ms
120.23
470.11
0.9215
0.9827
101.27
453.17
0.9096
0.9670
Table 1. Quantitative Comparison on Novel Trajectory Simulation (In-domain Test) and Object Insertion Simulation (Out-of-domain
Test) Datasets. Our method outperforms all editing baselines in perceptual quality (lower FID/FVD) and preserves scene structure more
faithfully (lower DINO-Struct-Dist). It also achieves strong temporal consistency (measured by VBench++ temporal flickering score),
surpassing image-editing methods and matching video diffusion models, with only a marginal gap to WAN V2V.
Relighting Data (Images)
PBR Shadow Data (Videos)
ISP Modification Data (Videos)
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
SDEdit (SD 3)
15.17
0.9803
0.0098
15.65
0.9891
0.0095
17.46
0.9850
0.0112
InstructPix2Pix
15.35
0.9812
0.0112
16.46
0.9905
0.0083
16.71
0.9856
0.0105
Wan-Video V2V
N/A
N/A
N/A
14.21
0.9867
0.098
15.81
0.9814
0.1335
Our Model
23.93
0.9938
0.0042
26.31
0.9966
0.0028
28.10
0.9974
0.0020
Table 2. Quantitative Results on Relighting, PBR Shadow, and
ISP Modification Holdout Sets. Our method achieves substan-
tially better PSNR, SSIM, and LPIPS, closely matching real-world
references.
Method
ISP Modification Data
Speed
PSNR
SSIM
LPIPS
FID
FVD
Temporal Cons.
VHTT
63ms
20.96
0.9921
0.0049
46.23
168.83
0.9838
Ke et al. [17]
10ms
25.98
0.9956
0.0037
61.51
170.16
0.9872
Ours
212ms
28.58
0.9971
0.0021
42.03
158.27
0.9805
Table 3. Quantitative Comparison with Harmonization Base-
lines. Evaluated on the harmonization subset (ISP Modification),
our method outperforms all baselines. Inference speed reported at
1024×576 resolution for Ours and Ke et al. [17], and at 576×320
resolution for VHTT.
Evaluation
Baselines
SDEdit (SD 3)
InstructPix2Pix
Wan-Video V2V
Human Eval. (%)
84.28% ± 10.92%
90.10% ± 14.54%
90.11% ± 14.13%
VLM Eval. (%)
79.18%
75.41%
88.57%
Table 4. User Study. Both human participants and VLM evaluators
were asked to compare our results against each individual baseline
and select the better one. We report the percentage of samples
where our method is preferred over the baselines. A preference rate
above 50% indicates that our method is preferred.
Model Variant
Temporal Consistency Score
In-domain Test
Out-of-domain Test
Our Model
0.9827
0.9670
Our Model w/o Temporal Loss
0.9806
0.9618
Our Model w/o Temporal Modules
0.9714
0.9502
Table 5. Ablation on Temporal Components. Adding temporal
loss and temporal modules effectively improves temporal consis-
tency.
loss Ltemp and show results in Table 5. The inclusion of
temporal components improves temporal consistency met-
rics. Qualitatively, this manifests as reduced flickering and
smoother transitions across frames.
Data Curation Strategy. We further ablate the contribution
of different data sources from our curation strategy. Tab. 6
quantifies individual contributions of separate data curation
streams to the overall model performance confirming that all
Raw Simulation (Input)
Model w/ LPIPS Perceptual loss
Model w/ Stochastic Multi-scale Perceptual loss 
Model w/o Perceptual loss
Figure 5. Ablation on Loss Design. Removing perceptual super-
vision leads to oversmoothed outputs, while using a conventional
LPIPS loss produces high-frequency artifacts. Our multi-scale for-
mulation mitigates these artifacts and yields perceptually better
results.
Metric
Full
w/o ISP
w/o
w/o Shadow
w/o Asset
w/o Artifacts
Model
Modif.
Relight.
Simul.
Re-ins.
Correc.
FID ↓
101.27
104.63
102.15
104.28
103.42
105.29
FVD ↓
453.17
465.7
457.92
462.31
459.34
476.82
Table 6. Quantitative ablation on curated data sources. Model
trained on all data sources provides the best performance.
curated data sources contribute complementary supervision
signals critical for generalization. Further qualitative results
can be found in Fig. 6 in the Supplementary materials.
5. Conclusion
We propose DiffusionHarmonizer, an online generative en-
hancement framework that transforms artifact-prone neural-
rendered frames into photorealistic and temporally coherent
simulations. By converting a pretrained image diffusion
model into a single-step temporally conditioned enhancer,
and by introducing a comprehensive data-curation pipeline
together with a tailored training objective, our method effec-
tively addresses reconstruction artifacts, appearance incon-
sistencies, illumination mismatches, and missing shadows.
Extensive experiments demonstrate substantial gains over
both image/video editing and harmonization baselines in
perceptual realism, structural fidelity, and temporal stability,
while operating efficiently on a single GPU. We believe Dif-
fusionHarmonizer offers a practical and scalable solution for
high-fidelity simulation in autonomous driving and robotics,
and opens new avenues for integrating generative priors into
real-time simulation pipelines.
8

<!-- page 9 -->
References
[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik
Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen,
Yin Cui, Yifan Ding, et al. Cosmos world foundation model
platform for physical ai. arXiv preprint arXiv:2501.03575,
2025. 5
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report. arXiv preprint
arXiv:2502.13923, 2025. 6
[3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. In-
structpix2pix: Learning to follow image editing instructions.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 18392–18402, 2023. 5
[4] Jianqi Chen, Yilan Zhang, Zhengxia Zou, Keyan Chen, and
Zhenwei Shi. Zero-shot image harmonization with generative
model prior. IEEE Transactions on Multimedia, 2025. 2
[5] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio,
Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Go-
jcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang.
Omnire: Omni urban scene reconstruction. In The Thirteenth
International Conference on Learning Representations, 2025.
2
[6] Xiaodong Cun and Chi-Man Pun. Improving the harmony
of the composite image by spatial-separated attention mod-
ule. IEEE Transactions on Image Processing, 29:4759–4771,
2020. 2
[7] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan.
Depth-supervised NeRF: Fewer views and faster training for
free. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2022. 3
[8] Alejandro Escontrela, Justin Kerr, Arthur Allshire, Jonas Frey,
Rocky Duan, Carmelo Sferrazza, and Pieter Abbeel. Gauss-
gym: An open-source real-to-sim framework for learning
locomotion from pixels. CoRR, 2025. 2
[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim En-
tezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz,
Axel Sauer, Frederic Boesel, et al. Scaling rectified flow trans-
formers for high-resolution image synthesis. In Forty-first
international conference on machine learning, 2024. 5
[10] Tobias Fischer, Samuel Rota Bul`o, Yung-Hsu Yang, Nikhil
Keetha, Lorenzo Porzi, Norman M¨uller, Katja Schwarz,
Jonathon Luiten, Marc Pollefeys, and Peter Kontschieder.
FlowR: Flowing from sparse to dense 3d reconstructions. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2025. 3
[11] Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur
Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan,
Jonathan T. Barron, and Ben Poole*. Cat3d: Create any-
thing in 3d with multi-view diffusion models. Advances in
Neural Information Processing Systems, 2024. 3
[12] Zonghui Guo, Haiyong Zheng, Yufeng Jiang, Zhaorui Gu,
and Bing Zheng. Intrinsic image harmonization. In CVPR,
pages 16367–16376, 2021. 2
[13] Zonghui Guo, Xinyu Han, Jie Zhang, Shiguang Shan, and
Haiyong Zheng. Video harmonization with triplet spatio-
temporal variation patterns. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 19177–19186, 2024. 3, 5, 7, 1
[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models. In Advances in Neural Information
Processing Systems, pages 6840–6851. Curran Associates,
Inc., 2020. 1
[15] Hao-Zhi Huang, Sen-Zhe Xu, Jun-Xiong Cai, Wei Liu, and
Shi-Min Hu. Temporally coherent video harmonization using
adversarial networks. IEEE Transactions on Image Process-
ing, 29:214–224, 2019. 3
[16] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin
Pan, and Yu Liu. Vace: All-in-one video creation and editing.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 17191–17202, 2025. 5
[17] Zhanghan Ke, Chunyi Sun, Lei Zhu, Ke Xu, and Rynson WH
Lau. Harmonizer: Learning to perform white-box image and
video harmonization. In European conference on computer
vision, pages 690–706. Springer, 2022. 2, 5, 7, 8, 1
[18] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Ma-
tiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset
of user preferences for text-to-image generation. Advances
in neural information processing systems, 36:36652–36663,
2023. 6
[19] Ruofan Liang, Zan Gojcic, Huan Ling, Jacob Munkberg, Jon
Hasselgren, Zhi-Hao Lin, Jun Gao, Alexander Keller, Nandita
Vijaykumar, Sanja Fidler, and Zian Wang. Diffusionrenderer:
Neural inverse and forward rendering with video diffusion
models. In The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2025. 2, 4
[20] Chih-Hao Lin, Zian Wang, Ruofan Liang, Yuxuan Zhang,
Sanja Fidler, Shenlong Wang, and Zan Gojcic. Controllable
weather synthesis and removal with video diffusion mod-
els. IEEE/CVF International Conference on Computer Vision
(ICCV), 2025. 6
[21] Jun Ling, Han Xue, Li Song, Rong Xie, and Xiao Gu. Region-
aware adaptive instance normalization for image harmoniza-
tion. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 9361–9370, 2021.
2
[22] Xi Liu, Chaoyi Zhou, and Siyu Huang.
3dgs-enhancer:
Enhancing unbounded 3d gaussian splatting with view-
consistent 2d diffusion priors. In Advances in Neural Infor-
mation Processing Systems, pages 133305–133327. Curran
Associates, Inc., 2024. 3
[23] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong
Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond
Chan, and Ying Shan. Evalcrafter: Benchmarking and eval-
uating large video generation models. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 22139–22149, 2024. 6
[24] William Ljungbergh, Bernardo Taveira, Wenzhao Zheng,
Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christof-
fer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi
Tomizuka, et al. R3d2: Realistic 3d asset insertion via dif-
fusion for autonomous driving simulation. arXiv preprint
arXiv:2506.07826, 2025. 2
[25] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon:
Diffusion-based training-free cross-domain image composi-
9

<!-- page 10 -->
tion. In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision, pages 2294–2305, 2023. 2
[26] Xinyuan Lu, Shengyuan Huang, Li Niu, Wenyan Cong, and
Liqing Zhang. Deep video harmonization with color mapping
consistency. In Proceedings of the Thirty-First International
Joint Conference on Artificial Intelligence, IJCAI-22, pages
1232–1238. International Joint Conferences on Artificial In-
telligence Organization, 2022. 3
[27] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun
Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image
synthesis and editing with stochastic differential equations. In
International Conference on Learning Representations, 2022.
5
[28] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view synthe-
sis. In ECCV, 2020. 3
[29] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall,
Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan.
Regnerf: Regularizing neural radiance fields for view syn-
thesis from sparse inputs. In Proc. IEEE Conf. on Computer
Vision and Pattern Recognition (CVPR), 2022. 3
[30] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall.
Dreamfusion: Text-to-3d using 2d diffusion. arXiv, 2022. 3
[31] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu,
Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle,
Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment any-
thing in images and videos. arXiv preprint arXiv:2408.00714,
2024. 4
[32] Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu,
Jianming Zhang, HyunJoon Jung, Guido Gerig, and He
Zhang. Relightful harmonization: Lighting-aware portrait
background replacement. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 6452–6462, 2024. 2
[33] Barbara
Roessle,
Norman
M¨uller,
Lorenzo
Porzi,
Samuel Rota Bul`o, Peter Kontschieder, and Matthias
Nießner.
Ganerf: Leveraging discriminators to optimize
neural radiance fields. ACM Trans. Graph., 42(6), 2023. 3
[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
synthesis with latent diffusion models. In cvpr, 2022. 1
[35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan,
and Surya Ganguli.
Deep unsupervised learning using
nonequilibrium thermodynamics. In Proceedings of the 32nd
International Conference on Machine Learning, pages 2256–
2265, Lille, France, 2015. PMLR. 1
[36] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equations.
In ICLR, 2021. 1
[37] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow. In European conference on com-
puter vision, pages 402–419. Springer, 2020. 5
[38] Yi-Hsuan Tsai, Xiaohui Shen, Zhe Lin, Kalyan Sunkavalli,
Xin Lu, and Ming-Hsuan Yang. Deep image harmonization.
In Proceedings of the IEEE conference on computer vision
and pattern recognition, pages 3789–3797, 2017. 2
[39] Haithem Turki, Qi Wu, Xin Kang, Janick Martinez Esturo,
Shengyu Huang, Ruilong Li, Zan Gojcic, and Riccardo de
Lutio. Simuli: Real-time lidar and camera simulation with
unscented transforms. Preprint, 2025. 2
[40] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler,
Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF:
Structured view-dependent appearance for neural radiance
fields. CVPR, 2022. 3
[41] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao
Yang, et al.
Wan: Open and advanced large-scale video
generative models. arXiv preprint arXiv:2503.20314, 2025.
5, 1
[42] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. Neus: Learning neural implicit
surfaces by volume rendering for multi-view reconstruction.
In Advances in Neural Information Processing Systems, pages
27171–27183. Curran Associates, Inc., 2021. 3
[43] Frederik Warburg*, Ethan Weber*, Matthew Tancik, Alek-
sander Hoły´nski, and Angjoo Kanazawa. Nerfbusters: Re-
moving ghostly artifacts from casually captured nerfs. In
International Conference on Computer Vision (ICCV), 2023.
3
[44] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic,
and Huan Ling. Difix3d+: Improving 3d reconstructions with
single-step diffusion models. In Proceedings of the Computer
Vision and Pattern Recognition Conference (CVPR), pages
26024–26035, 2025. 2, 3, 4
[45] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas
Moenne-Loccoz, and Zan Gojcic. 3dgut: Enabling distorted
cameras and secondary rays in gaussian splatting. In CVPR,
2025. 4
[46] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park,
Ruiqi Gao, Daniel Watson, Pratul P. Srinivasan, Dor Verbin,
Jonathan T. Barron, Ben Poole, and Aleksander Holynski.
Reconfusion: 3d reconstruction with diffusion priors. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024. 3
[47] Jamie Wynn and Daniyar Turmukhambetov. Diffusionerf:
Regularizing neural radiance fields with denoising diffusion
models. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
4180–4189, 2023. 3
[48] Ke Xu, Gerhard Petrus Hancke, and Rynson WH Lau. Learn-
ing image harmonization in the linear color space. In Proceed-
ings of the IEEE/CVF International Conference on Computer
Vision, pages 12570–12579, 2023. 2
[49] Shuzhou Yang, Xiaoyu Li, Xiaodong Cun, Guangzhi Wang,
Lingen Li, Ying Shan, and Jian Zhang. Gencompositor: gen-
erative video compositing with diffusion transformer. arXiv
preprint arXiv:2509.02460, 2025. 3
[50] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Vol-
ume rendering of neural implicit surfaces. In Thirty-Fifth
Conference on Neural Information Processing Systems, 2021.
3
[51] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
10

<!-- page 11 -->
pixelNeRF: Neural radiance fields from one or few images.
In CVPR, 2021. 3
[52] Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu,
Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger,
and Yiyi Liao. Hugsim: A real-time, photo-realistic and
closed-loop simulator for autonomous driving. arXiv preprint
arXiv:2412.01718, 2024. 2
11

<!-- page 12 -->
DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic
Simulation with Online Diffusion Enhancer
Supplementary Material
We provide preliminary background information in Sec. 6,
additional experimental details in Sec. 7, and further qual-
itative results in Sec. 8. We provide additional video com-
parisons in the attached video files in the supplementary
materials.
6. Background Information
Diffusion Models
are generative models that learn to trans-
form samples from one source distribution, typically Gaus-
sian noise, to the target data distribution pdata(x) [14, 35, 36].
They consist of two main processes: a forward process where
noise ϵ ∼N(0, I) is progressively added to samples x from
the data distribution, and a reverse process which iteratively
removes noise from a sample from the source distribution to
obtain a sample of the data distribution. The intermediate
latent variables can be expressed as xτ = ατx + στϵ where
ατ and στ are determined by the selected noise schedule and
the timestep τ represents how far along the forward process
we are (higher τ indicates more noise). The reverse process
is achieved by a denoising model Fθ with learnable parame-
ters θ. These are optimized by learning to predict the added
noise; concretely they minimize the following objective:
Ex∼pdata,τ∼pτ ,ϵ∼N (0,I)

w(τ)∥ϵ −Fθ(xτ; c, τ)∥2
2

,
(7)
where c represents the condition (i.e., text prompt or refer-
ence image) used to control the generation, w(τ) is a time-
dependent weighting function. p(τ) is a chosen timestep
distribution, typically a uniform distribution over a set of in-
tegers, i.e. pτ ∼U(0, 1000) [14]. Latent Diffusion Models
(LDMs) [34] greatly improve computational and memory
efficiency by operating on a lower dimensional latent space.
The dimensionality reduction is achieved by an encoder-
decoder model, where the encoder E maps the data samples
into a latent space Z and the decoder D achieves the inverse
operation: D (E(x)) ≈x. LDMs [34] effectively treat Z as
the data set, therefore x can be replaced by z := E(x) in all
equations above.
7. Additional Experiment Details
Training Dataset Details: We visualize in Fig. 7 repre-
sentative paired samples from the five components of our
data-curation pipeline: Relighting, ISP Modification, As-
set Re-insertion, Artifacts Correction, and PBR Simulation.
Each group shows the training input and its corresponding
label, illustrating the types of appearance changes, lighting
variations, reconstruction degradations, and shadow differ-
ences used to supervise DiffusionHarmonizer. The total
training set includes 46K frames for Relighting, 88K frames
for ISP Modification, 21K frames for Asset Re-insertion,
118K frames for Artifacts Correction, and 77K frames for
PBR Simulation. These pairs collectively provide diverse
and complementary signals for harmonization, artifact cor-
rection, and lighting consistency.
General Image/Video Editing Baselines: SDEdit and In-
structPix2Pix require user-provided prompts and selections
of hyperparameters (e.g., the magnitude of injected noise).
To minimize human-induced prompting bias, we automati-
cally generated all image-editing prompts using ChatGPT-5
and adopted the default hyperparameters provided by the of-
ficial HuggingFace implementations. The generated prompt
we use is:
”Translate this image into a high-quality camera
frame captured by an autonomous vehicle, adding
realistic shadows while correcting artifacts and
harmonizing lighting and appearance between the
foreground and background.”
The corresponding negative prompt is:
”blurry, low quality, distorted, artifacts”
For video-to-video editing, we employ the VACE model
built on WAN 2.1 [41]. Among the supported tasks, we select
the grayscale-to-RGB translation setting as it is the closest
analogue to our problem. During inference, we supply the
model with (i) a prompt describing the video content and
(ii) the grayscale frames as input, and request the model
to synthesize the corresponding RGB sequence. All video
prompts are generated using Qwen3-VL.
Specialized Video Harmonization Methods: VHTT im-
plements a design based on joint processing of video frames
and stacked short- and long-term context transformer mod-
ules, which imposes limits on the resolution and length of
the processed videos. VHTT video evaluations were thus
carried out on batches of 20 frames at 576 × 320 resolution.
All results were upsampled to 1024 × 576 resolution for
quantitative evaluation. As VHTT [13] and Ke et al. [17]
only alter pixels of inserted foreground objects and require
input segmentation masks, the metrics reported in Tab. 3
were computed on pixels within the foreground regions only.
As Tab. 3 shows, our method significantly outperforms both
1

<!-- page 13 -->
baselines in PSNR, SSIM, LPIPS, FID, and FVD, highlight-
ing its power as an image and video harmonizer. Note that
both methods assume the presence of a single foreground ob-
ject, which may lead to degraded performance when applied
to images with multiple inserted objects. This requirement is
highly impractical in real-world scenarios, where one often
deals with complex, multi-object scenes without access to
instance segmentation masks. In contrast, our method op-
erates in a mask-free setup, automatically identifying and
harmonizing dissonant regions while having access to the
full image, allowing it to enhance the entire scene. This is
also reflected in inference speed: while our method gener-
ates a full image, Ke et al. [17] only predicts parameters
of six image filters that are then sequentially applied to the
foreground area. Figures 11 and 12 show additional qualita-
tive comparisons. Notice how our method provides superior
results even in a scene with only a single inserted object in
Fig. 12 (bottom).
User Study & VLM-Based Evaluation: We visualize our
study instructions and the user-study interface in Fig. 8. Dur-
ing the study, human evaluators are shown (i) the input im-
age as a reference and (ii) two predictions—our output and
a baseline—and are asked to choose the result they perceive
as more realistic. To mitigate ordering bias, the left–right
placement of the two predictions is randomized for every
comparison. Participants are recruited through the Prolific
platform, which provides access to a diverse, globally dis-
tributed pool of evaluators. In total, we gather responses
from 45 participants (15 per baseline), with each participant
completing 50 pairwise comparisons.
To complement the human study and provide a scalable,
automated comparison, we additionally evaluate model pref-
erence using a vision–language model (VLM), specifically
the LLaVA-1.5-7b model. Following the same protocol as
the user study, we present the VLM with the input image
as a reference along with two predictions—our output and
a baseline—with their order randomly shuffled. The model
is asked to choose the result that better preserves realism
and consistency relative to the reference. To further reduce
prompt-design bias, all evaluation prompts are automatically
generated using ChatGPT-5. Results in Tab. 4 show a high
level of agreement between the VLM-based judgments and
human preferences, reinforcing the superior quality of our
method.
8. Additional Qualitative Results
8.1. Qualitative Ablation of Curated Data Sources
Fig. 6 highlights the qualitative impacts of ablating individ-
ual training data subsets: excluding Artifacts-Correction data
prevents the model from correcting reconstruction errors; re-
moving Shadow Data (PBR Shadow and Asset Re-Insertion)
prevents synthesizing physically plausible shadows; and re-
Full Model
Model w/o Shadow Data
Model w/o Appearance Data
Model w/o Artifacts-correction Data
Input
Input
Input
Input
Figure 6. Ablation on Curated Data Sources. Removing any cu-
rated data source degrades performance: without artifact-correction
data the model fails to fix reconstruction errors; without shadow
data it cannot synthesize plausible shadows; and without appear-
ance data it produces color-tone inconsistencies. Each data source
provides complementary and essential supervision.
moving the appearance dataset (ISP Modification and Re-
lighting data) results in color-tone discontinuities between
foreground and background. This result together with the
quantitative analysis in Tab. 6 underscores the importance of
all components for the model performance.
8.2. Additional Qualitative Comparison with Base-
lines
We provide additional qualitative comparisons with baselines
in Figures 9 and 10, on out-of-domain Waymo scenes. Our
model reliably harmonizes color tone and synthesizes scene-
consistent lighting and shadows, whereas state-of-the-art
image and video editing baselines often fail to generate phys-
ically plausible shadowing for inserted objects. Although
these baselines can partially reduce reconstruction artifacts,
they frequently hallucinate content or modify regions that
should remain intact. In contrast, our approach preserves
scene geometry and underlying input structures while main-
taining strong temporal consistency across adjacent frames.
In Figures 11 and 12, we further provide additional com-
parisons with specialized harmonization methods. While
these approaches adjust foreground color appearance, they
do not synthesize realistic shadows and cannot address re-
construction artifacts by design, limiting their applicability
in simulation-enhancement pipelines.
8.3.
Additional
Qualitative
Comparison
with
Ground-Truth Labels
We additionally present qualitative results on our holdout
datasets and compare the predicted outputs against the cor-
responding ground-truth labels. As shown in Fig. 13, our
model’s predictions closely match real-world scenarios, il-
lustrating that it produces physically faithful enhancements
suitable for online deployment in simulation pipelines.
2

<!-- page 14 -->
Training Label
Training Input
Training Label
Training Input
Artifacts 
Correction
Relighting
ISP 
Modification
Assert 
Re-insertion
PBR 
Simulation
Figure 7. Visualization of Curated Training Dataset. We show representative paired samples from our curated training data: Relighting, ISP Modification,
Asset Re-insertion, Artifacts Correction, and PBR Simulation.
3

<!-- page 15 -->
Figure 8. User Study Interface. We show our study instructions and interface. Evaluators are shown the input image and two predictions (ours and a
baseline) and asked to select the more realistic result, with prediction order randomized to avoid bias.
4

<!-- page 16 -->
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Figure 9. Additional Comparison with Image and Video Editing Baselines on Out-of-Domain Testing Data (Part 1 of 2). Our method harmonizes color
tone and synthesizes realistic lighting and shadows, while editing baselines often fail to produce physically plausible shadowing. Although both can reduce
reconstruction artifacts, baselines tend to hallucinate inconsistent content and over-edit well-reconstructed regions, whereas our method preserves scene
geometry and input structure. Moreover, image-editing baselines introduce frame-to-frame jitter, whereas our model maintains strong temporal coherence.
5

<!-- page 17 -->
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Simulation
Input
Ours
Instruct
Pix2Pix
Wan-video
V2V
SDEdit
Stable 
Diffusion3
Figure 10. Additional Comparison with Image and Video Editing Baselines on Out-of-Domain Testing Data (Part 2 of 2). Our method harmonizes
color tone and synthesizes realistic lighting and shadows, while editing baselines often fail to produce physically plausible shadowing. Although both can
reduce reconstruction artifacts, baselines tend to hallucinate inconsistent content and over-edit well-reconstructed regions, whereas our method preserves scene
geometry and input structure. Moreover, image-editing baselines introduce frame-to-frame jitter, whereas our model maintains strong temporal coherence.
6

<!-- page 18 -->
Simulation
Input
Ours
Ke et al.
VHTT
Ground
Truth
Simulation
Input
Ours
Ke et al.
VHTT
Ground
Truth
Figure 11. Additional Comparison with Video Harmonization Baselines on ISP modification held out test set.
7

<!-- page 19 -->
Simulation
Input
Ours
Ke et al.
VHTT
Ground
Truth
Simulation
Input
Ours
Ke et al.
VHTT
Ground
Truth
Figure 12. Additional Comparison with Video Harmonization Baselines on ISP modification held out test set.
8

<!-- page 20 -->
Input
Our Prediction
Ground Truth
Figure 13. Comparison with Ground Truth on Holdout Datasets. Our model’s predictions closely match the ground-truth real-world captures, producing
faithful, physically plausible results suitable for online simulation systems.
9
