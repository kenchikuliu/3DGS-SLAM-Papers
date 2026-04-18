<!-- page 1 -->
Zip-GS: Extreme Compression of 3D Gaussians
with Diffusion Priors
Jiaqi Chen1,2,*, Xinhao Ji2,*, Yuanyuan Gao1,2,*, Hao Li1, Yuning Gong2, Yifei
Liu2, Dingwen Zhang1,†, Dan Xu4, Xiao Sun2, and Zhihang Zhong3,†
1 Northwestern Polytechnical University
2 Shanghai Artificial Intelligence Laboratory
3 Shanghai Jiao Tong University
4 Hong Kong University of Science and Technology
Ours
GT
LightGS*
Standard Compression
Extreme Compression
3.33x(106 MB)
3.33x(183 MB)
3.33x(106 MB)
39.6x(8.95 MB)
41.3x(14.82 MB)
37.7x(9.41 MB)
63.9x(5.55 MB)
65.1x(9.40 MB)
56.22x(6.31 MB)
119x(2.98 MB)
114x(5.35 MB)
111x(3.20 MB)
354.77MB
611.84MB
354.77MB
Fig. 1: Qualitative comparison under different compression ratios. LightGS*
indicates the optimized version of LightGS [6]. LightGS* achieves only limited com-
pression (around 3×) and produces severe visual artifacts. In contrast, our method
realizes orders-of-magnitude higher compression (exceeding 100× in some cases) while
preserving scene geometry and appearance realism. Even at aggressive compression
ratios, our reconstructions remain close to ground-truth quality, demonstrating the ro-
bustness of the proposed framework.
Abstract. As world models increasingly produce Internet-scale 3D con-
tent, efficient delivery of neural assets becomes a critical systems bot-
tleneck. Neural scene representations, especially 3D Gaussian Splatting
(3DGS), provide excellent rendering quality and speed, but their memory
footprint remains too large for bandwidth-limited storage and streaming.
* denotes equal contribution. This work was done during their internship at Shanghai
Artificial Intelligence Laboratory. † denotes corresponding author.
arXiv:2509.24758v5  [cs.CV]  9 Mar 2026

<!-- page 2 -->
2
J. Chen et al.
Prior 3DGS compression pipelines face a persistent trade-off: optimization-
based methods, which require ground-truth images and per-scene fine-
tuning, are accurate yet slow and often inapplicable to Internet-scale 3D
assets without paired supervision, while lightweight pruning/quantization
schemes are efficient but fragile at very high compression levels. To ad-
dress this gap, we present ZipGS, a fully feed-forward framework for
extreme 3DGS compression and restoration. ZipGS combines Universal
Gaussian Compression (UGC) and GaussPainter in a two-stage design.
UGC removes redundant Gaussians without re-optimization, keeping
a compact set of structurally important primitives. GaussPainter then
recovers visual quality from severely compressed renderings via mask-
guided diffusion priors, while also refining visible regions rather than
only completing missing areas. With a lightweight VAE and one-step
diffusion sampling, our method supports practical real-time restoration.
Across indoor and outdoor benchmarks, ZipGS consistently improves
perceptual and distortion metrics over strong compression and genera-
tive baselines, achieving compression beyond 100× and up to 300× in
challenging settings. These results demonstrate that diffusion priors can
effectively reconcile extreme compression with high-fidelity neural ren-
dering.
Keywords: 3D Gaussian Splatting · Extreme Compression · Diffusion
Priors · Neural Rendering · World Model
1
Introduction
In recent years, rapid progress in world models and generative 3D content cre-
ation has enabled the Internet-scale sharing of 3D assets, often without explicit
per-view ground-truth supervision. These assets are quickly becoming core in-
frastructure for immersive content ecosystems and the emerging 3D Internet,
underpinning applications such as augmented/virtual reality, interactive visu-
alization, remote collaboration, and digital twin systems. As a result, efficient
storage, transmission, and streaming of high-quality 3D representations is criti-
cal, yet they remain bulky and expensive, posing a major obstacle in bandwidth-
constrained or mobile scenarios.
Among recent representations, 3D Gaussian Splatting (3DGS) [10] has emerged
as a compelling choice for real-time neural rendering and novel view synthesis,
explicitly representing scenes with millions of anisotropic Gaussian primitives
rendered efficiently in screen space. However, the sheer number of Gaussians
still leads to substantial storage and communication overhead, with per-scene
sizes reaching hundreds of megabytes or even gigabytes, which becomes a major
bottleneck for Internet-scale 3D asset transmission.
To resolve the redundancy in Gaussian primitives, existing methods can be
broadly categorized into two groups: optimization-based approaches [17,22] that
fine-tune Gaussian parameters, and training-free approaches [6, 28] that rely
on quantization or heuristic pruning. Optimization-based pipelines typically as-
sume access to multi-view ground-truth images and perform costly per-scene

<!-- page 3 -->
Zip-GS
3
optimization, which limits their applicability to Internet-scale 3D assets syn-
thesized by generative models, where only the neural representation is available
and no paired images exist. Training-free methods are lightweight and scalable
but struggle under aggressive compression ratios, leading to degraded rendering
quality and a persistent trade-off between compression ratio and fidelity.
Motivated by this challenge, we propose ZipGS, an extreme 3DGS compres-
sion framework tailored for large-scale 3D asset delivery. ZipGS leverages gen-
erative priors to push compression to very high ratios while maintaining strong
visual fidelity at render time, enabling practical storage and transmission of
3DGS assets over the Internet. Concretely, ZipGS consists of two complemen-
tary modules: Universal Gaussian Compression (UGC) and GaussPainter. UGC
performs extreme compression without per-scene re-optimization, retaining only
essential information and maximizing the benefit of generative priors. Gauss-
Painter then exploits the strong priors of diffusion models with a mask-guided
refinement strategy to restore high-quality renderings from heavily pruned and
incomplete Gaussian scenes.
By combining UGC and GaussPainter, our framework achieves orders of mag-
nitude compression, often exceeding 100×, while still enabling real-time render-
ing and high-quality restoration. In contrast, our method maintains geometric
fidelity and perceptual realism even under much stronger compression, demon-
strating its robustness and clear advantage over existing baselines. Our contri-
butions are threefold:
(1) We demonstrate, for the first time, that a data-driven paradigm can drive
generative 3DGS compression, moving beyond heuristic pruning or optimization-
based tuning by leveraging powerful generative priors for faithful restoration
under extreme compression.
(2) We propose two complementary modules: UGC for re-optimization-free
pruning and GaussPainter for diffusion-based refinement, and demonstrate that
their synergy enables compact representations while preserving high visual fi-
delity.
(3) Extensive experiments on public datasets demonstrate that ZipGS achieves
state-of-the-art performance, maintaining real-time rendering and robust quality
even under a storage size compression ratio of up to 100×.
2
Related Work
2.1
Efficient Neural Rendering.
3D Gaussian Splatting (3DGS) has rapidly emerged as a powerful representa-
tion for real-time scene rendering, yet its efficiency is often constrained by the
large number of Gaussian primitives. To improve scalability, prior works have
explored compression and pruning strategies. Codebook-based methods [6,12,21]
reduce the storage footprint by quantizing Gaussian parameters, while pruning
approaches [6, 28] directly eliminate redundant primitives to achieve compact

<!-- page 4 -->
4
J. Chen et al.
representations. Beyond pure reduction, densification techniques such as Mini-
Splatting [7] and Taming-3DGS [19] regenerate new primitives in a spatially ef-
ficient manner, whereas Scaffold-GS [18] anchors Gaussians on structured grids
for better distribution.
Although these approaches have demonstrated strong effectiveness, they of-
ten rely on retraining or fixed pruning ratios, limiting flexibility under dif-
ferent compression requirements. In contrast, our method improves upon the
LightGS [6] scoring mechanism and further integrates generative priors, enabling
controllable compression while preserving rendering quality even at extreme ra-
tios.
2.2
Diffusion Priors for 3D Reconstruction.
Diffusion models have recently shown strong potential in addressing degradation
in 3D reconstruction tasks. Traditional dense methods, such as NeRF-based
pipelines, require abundant multi-view inputs, while sparse-view methods [5,29]
attempt to compensate with structural priors but still suffer from incomplete
reconstructions. Diffusion models extend this line of work by providing powerful
generative priors that can restore missing details and refine degraded regions.
Several recent works integrate diffusion with Gaussian-based or NeRF-based
reconstructions. For instance, Difix3D+ [30] leverages diffusion to denoise target
views using reference information, yielding improved fidelity. Generative Gaus-
sian Splatting (GGS) [25] integrates video diffusion priors with Gaussian splat-
ting to enhance view consistency, and GSFixer [33] employs reference-guided
diffusion to correct artifacts in sparse inputs. Similarly, GSD [20] combines
Gaussian splatting and diffusion priors for single-view 3D reconstruction, while
Single-Stage Diffusion NeRF (SSDNeRF) [3] jointly learns diffusion priors and
neural fields from sparse inputs. Diffusion has also been explored for human re-
construction, e.g., DiHuR [4], which uses diffusion priors to enhance geometry
and texture fidelity under sparse or degraded observations.
Different from these approaches, our method directly applies diffusion priors
to compressed Gaussian renderings, without requiring extra views or heavy su-
pervision. By combining mask-guided conditioning with efficient one-step diffu-
sion and latent alignment, it is capable of handling diverse degradation sources
such as sparse inputs, severe pruning, and structural inconsistencies, thereby
producing more consistent and high-quality reconstructions.
3
Preliminaries
3.1
3D Gaussian Splatting
3D Gaussian Splatting [10] explicitly represents 3D scenes with a collection of
anisotropic Gaussian primitives. Each Gaussian is parameterized by its mean
position µ ∈R3, covariance matrix Σ ∈R3×3, opacity σ ∈(0, 1], and color
(or feature) c ∈[0, 1]3. During rendering, the color C of a pixel is obtained by
blending N ordered Gaussians that overlap with the pixel using alpha blending:

<!-- page 5 -->
Zip-GS
5
  
C
 = 
\sum
 _{
i
 \i
n N } \mathbf {c}_i \alpha _i \prod _{j=1}^{i-1} (1 - \alpha _j), \label {eq:3dgs-blend} 
(1)
where αi denotes the opacity contribution of the i-th Gaussian at the pixel,
typically computed by evaluating the projected 2D elliptical footprint defined
by Σi and scaling with its opacity σi. This design enables differentiable and
efficient rendering in real time, making 3DGS a strong alternative to volumetric
ray marching methods.
3.2
Diffusion Models
Diffusion models [8, 26] generate data by adding Gaussian noise in a forward
Markov chain and learning to reverse it. In the forward process, noise is added
step by step:
  q(\m a thb f
 
{
z}_
{
1
:T}
 
\ m id  \mat hbf
 
{z}_0) = \prod _{t=1}^{T} \mathcal {N}\!\left (\mathbf {z}_t;\, \sqrt {1-\beta _t}\,\mathbf {z}_{t-1},\, \beta _t \mathbf {I}\right ), \label {eq:forward-step} 
(2)
which has the closed form
  q( \ mat h b
f
 {z }_t \mi d \ mathb
f
 {z}_0) = \mathcal {N}\!\left (\mathbf {z}_t;\, \sqrt {\bar {\alpha }_t}\,\mathbf {z}_0,\, (1-\bar {\alpha }_t)\mathbf {I}\right ), 
(3)
where αt = 1 −βt and ¯αt = Qt
i=1 αi.
Reverse process. Sampling starts from Gaussian noise zT ∼N(0, I) and
reconstructs z0 via
  q(\m a thb f { z }
_
{t-1} \mid \ math bf
 {
z
}_t, \mathbf {z}_0) = \mathcal {N}\!\left (\mathbf {z}_{t-1};\, \mu _t(\mathbf {z}_t,\mathbf {z}_0),\, \sigma _t^2 \mathbf {I}\right ), 
(4)
with mean
  \mu _t( \
m
ath
b
f { z } _t
,\ m ath
b
f
 {z}_0) = \frac {1}{\sqrt {\alpha _t}}\Big (\mathbf {z}_t - \frac {1-\alpha _t}{\sqrt {1-\bar {\alpha }_t}}\,\boldsymbol {\epsilon }\Big ), 
(5)
where ϵ is the injected noise. A neural network ϵθ(zt, t, c) predicts ϵ, optionally
conditioned on c.
The training objective is the noise-prediction loss:
  \ m athcal 
{L} _ {\
text {D M
}
}  = \ ma t
hbb 
{ E}_{\mathbf {z}_0,t,\boldsymbol {\epsilon }}\, \left \|\boldsymbol {\epsilon } - \boldsymbol {\epsilon }_\theta \!\left (\sqrt {\bar {\alpha }_t}\mathbf {z}_0 + \sqrt {1-\bar {\alpha }_t}\,\boldsymbol {\epsilon },t\right )\right \|_2^2. 
(6)
4
Leveraging generative model priors for extreme
Gaussian compression
Recent feed-forward world models and reconstruction systems can produce large
volumes of 3D assets, but these assets are often too heavy to be stored, transmit-
ted, and shared efficiently at scale. Our goal is to establish a practical compres-
sion paradigm for 3D assets that makes them truly deliverable over the Internet
by leveraging strong generative priors.

<!-- page 6 -->
6
J. Chen et al.
Detail-Preserving Pruning
Geometry-Preserving Pruning 
Universal Gaussian Compression
Rendering
Score-based 
Filtering
Voxelization
Refined GS
MLP
Rendered Masks
Generated Images
VAE
Encoder
GaussPainter：A One-Step Diffusion Model
Rendered Images
GT Images
Attention
Attention
Attention
…
𝑧̂
𝑧!"
VAE
Encoder
VAE
Decoder
𝐿𝑜𝑠𝑠!"#$%# 
ℒ𝑜𝑠𝑠
UNet
Global Significance Score
Prune
prune ratio
sort
Fig. 2: Overview of the proposed Zip-GS framework. The pipeline contains two
stages: Universal Gaussian Compression (UGC), which aggressively removes redun-
dancy while preserving key geometric and appearance cues, and GaussPainter, which
restores high-fidelity renderings from compressed Gaussian scenes using mask-guided
diffusion priors.
To this end, we propose a two-stage framework, illustrated in Fig. 2. In the
first stage, we exploit the structural properties of 3D Gaussian representations
to derive an extremely compact, transmission-friendly asset file (Sec. 4.1). In
the second stage, we render the compressed asset efficiently and enhance the
resulting images using generative priors, compensating for the information lost
during extreme compression (Sec. 4.2).
4.1
Universal Gaussian Compression
Given a 3DGS asset without ground-truth images, our compression method aims
to produce a representation with as few Gaussians as possible while preserving
the information most beneficial for subsequent diffusion-based generation. Be-
cause the generative model must recover missing information, our compression
method prunes Gaussians along two complementary axes to reduce the count
while preserving scene content and minimizing hallucinations: Detail-Preserving
Pruning (DPP), which preserves fine rendering details, and Geometry-Preserving
Pruning (GPP), which retains the overall geometric structure.

<!-- page 7 -->
Zip-GS
7
Detail-Preserving Pruning. To quantify the contribution of each Gaussian
primitive and guide pruning, we follow LightGS [6] and define a global signifi-
cance score GSj that accounts for visibility, opacity, and ray transmittance:
  G S
_j 
=
 \s
u
m
 _{i=1 }^
{
M HW }  \mathbf {1}\big (G(\mathbf {X}_j), r_i\big )\,\cdot \, \sigma _j \,\cdot \, T_j, \label {eq:gs} 
(7)
where i indexes all pixels across M views with image resolution H × W.
The indicator function 1(·) returns 1 if the j-th Gaussian G(Xj) intersects the
camera ray ri, and 0 otherwise. σj denotes the opacity of the j-th Gaussian, and
Tj represents the accumulated transmittance of the ray before reaching G(Xj),
defined as:
  T
_j 
=
 \p
ro d  _{i=1}^{j-1} (1 - \sigma _i). \label {eq:transmittance} 
(8)
In a Gaussian-based 3D asset, feed-forward reconstruction often converts
many pixels into primitives, producing substantial redundancy with limited im-
pact on rendering quality. In DPP, our goal is to reduce the Gaussian count
while retaining fine appearance details and view-consistent textures. We there-
fore compute GSj for each Gaussian using either randomly sampled or provided
camera poses, rank the primitives by their scores, and prune those with low im-
portance. This scoring-based pruning preserves visually salient structures while
removing redundant primitives, providing a compact yet detail-preserving input
for the subsequent generative refinement.
(a)
(b)
(c)
Fig. 3: Detail-preserving scoring and
geometry-aware correction. (a) Ren-
dered
views
of
Gaussian
scenes.
(b)
Heatmaps of the global significance score
used in DPP, which emphasize fine tex-
tures but under-cover sparse regions. (c)
The voxel-based supplementation in GPP
rebalances the distribution, yielding more
uniform coverage and improved preserva-
tion of structure and details.
Geometry-Preserving Pruning. We
observe that the importance score is of-
ten influenced by the viewing direction
and the geometric position of Gaus-
sians, as illustrated in Fig. 3. While
it preserves color details, for complete
3D assets it can severely damage geom-
etry, leading to holes in rendered im-
ages. To better leverage diffusion pri-
ors, we introduce Geometry-Preserving
Pruning, which complements the scor-
ing function. With the same number
of Gaussians, this sampling approach
more completely preserves both global
geometric structures and fine-grained
local details.
The significance score does not en-
sure spatial uniformity: dense regions
may retain redundancy, while sparse
ones may lose coverage. We partition the space into voxels of size v, group

<!-- page 8 -->
8
J. Chen et al.
Gaussians by voxel, and ensure each voxel with sufficient points contributes at
least one selected Gaussian. For a voxel V containing Gaussians {gj} with cor-
responding significance scores {sj}, we select k Gaussians, denoted by SV, that
maximize the total significance score within the voxel:
  \ mat hca
l {
S}_\m
a
thca
l {V} = \arg \max _{\substack {\mathcal {S} \subset \mathcal {V} \\ |\mathcal {S}| = k}} \sum _{g_j \in \mathcal {S}} s_j, 
(9)
where k = ⌊|V| · ρ⌋. The final set combines globally selected Gaussians with
voxel-based supplements, balancing global importance and spatial coverage.
However, the Gaussians supplemented by this strategy are often small; al-
though geometric details are retained, these Gaussians may occupy only tiny
regions in the rendered image and thus provide insufficient cues for the diffusion
model. To address this, we add an MLP to predict the scales of the Gaussians
retained after Geometry-Preserving Pruning. This MLP is trained jointly with
the diffusion model, enabling the Gaussians to adapt their scales to parameters
that are most beneficial for diffusion-based completion.
4.2
GaussPainter: Efficient Mask-Guided Diffusion for Gaussian
Scenes
The generative module is designed to leverage the strong priors of diffusion
models to address the information loss introduced by aggressive Gaussian com-
pression. While diffusion excels at synthesizing realistic content, it often suffers
from two limitations in this setting: (i) hallucination of non-existent structures,
and (ii) prohibitive time costs due to iterative sampling.
Mitigating Hallucinations. To mitigate hallucination in compressed Gaus-
sian renderings, we employ two complementary strategies: latent supervision
and mask guidance. For latent supervision, relying only on pixel-level losses
fails to reliably separate regions that should be preserved from those requiring
completion, as corrupted or blackened areas introduced by Gaussian compression
are often misread as valid content (see Fig. 4). Inspired by deblurring models [16],
we therefore impose supervision directly in the latent space: the ground-truth
image y and the degraded rendering x are encoded by the VAE as zhq=E(y) and
zlq=E(x), and we minimize
  \math c al { L }_{\t
ext {latent}}=\|z_{\mathrm {lq}}-z_{\mathrm {hq}}\|_2^2, 
encouraging the degraded latent to approach the manifold of its high-quality
counterpart and helping the model decide what to preserve versus plausibly
complete.
While this latent alignment improves structural completion, large holes or
low-texture regions may still exhibit color drift (see Fig. 5). To address this, we
integrate mask guidance. A visibility mask derived from accumulated opac-
ity in 3DGS rendering (Eq. 1) is normalized, flattened, globally pooled, and

<!-- page 9 -->
Zip-GS
9
projected into the text-conditioning embedding space. The resulting mask em-
bedding is then added element-wise to the caption features, which down-weights
already reliable areas and steers the denoiser toward pixels likely to be missing or
corrupted. This improves boundary handling around holes and reduces the ten-
dency
to
propagate
artifacts
originating
from
compression.
w/o Latent Loss
w/ Latent Loss
GT
Fig. 4: Comparison of pixel-level su-
pervision
and
latent
supervision.
From left to right: ground-truth image,
compressed rendering, pixel-level supervi-
sion, and latent supervision. Latent su-
pervision better distinguishes regions to
preserve from those to complete, reduc-
ing artifacts in heavily degraded areas.
At inference, the latent of the incom-
plete rendering is denoised at a fixed
timestep t=199 in a single forward
pass of the U-Net, where the injected
mask embedding supplies spatial cues
for where to inpaint versus preserve.
This lightweight integration adds negli-
gible overhead yet substantially reduces
hallucinations and stabilizes color and
illumination. Consistent with our over-
all design of directly applying diffusion
priors to compressed Gaussian render-
ings without extra views or heavy su-
pervision, combining mask-guided con-
ditioning with efficient one-step diffu-
sion and latent alignment enables Gaus-
sPainter to handle diverse degradation
sources such as severe pruning, sparse
inputs, and structural inconsistencies,
thereby producing reconstructions that are structurally complete and visually
faithful.
Real-Time Rendering. A key advantage of our framework is its ability to
operate in near real-time, making it suitable for practical deployment in interac-
tive applications. To achieve this, we replace the standard VAE with TAESD [2],
a lightweight autoencoder designed for fast encoding and decoding while main-
taining visual fidelity. In addition, we adopt a one-step diffusion schedule, where
the denoiser directly predicts the clean latent in a single forward pass at a fixed
timestep, bypassing the expensive iterative sampling typical of diffusion models.
Together, these two design choices drastically reduce inference latency, enabling
high-quality restoration of compressed Gaussian renderings at interactive frame
rates.
For efficiency, we measure inference speed on an NVIDIA A100 GPU. After
two warm-up iterations, we run 10 forward passes at 5122 resolution and report
the mean runtime and standard deviation (Tab. 5b). Our method requires 2.13 s
for compression and 0.23 s for decompression, demonstrating favorable compu-
tational cost compared to optimization-based approaches.

<!-- page 10 -->
10
J. Chen et al.
5
Experiment
5.1
Experimental Setups
Training Strategy and Data. We initialize our model from the official pre-
trained weights of SD-Turbo and TASED. During training, we fine-tune the
full U-Net while applying a LoRA module (rank=4) to the VAE decoder for
parameter-efficient adaptation. Training is conducted on DL3DV and the offi-
cial training split of ScanNet++. The DL3DV set includes all available video
sequences.
Evaluation Protocol. We evaluate our method on 50 official ScanNet++ test
scenes [32]. Across multiple compression ratios, we report PSNR, SSIM, LPIPS,
together with the corresponding storage size. To assess out-of-domain general-
ization, we further conduct experiments on three unseen datasets: Replica [27],
Mip-NeRF360 [1], and Tanks & Temples [11].
Baselines. We compare our framework with both compression-based and gen-
erative methods. For compression, we include LightGaussian (LightGS) [6] and
FlexGaussian [28]. For fair comparison, all compression baselines are evaluated
without access to ground-truth images. Accordingly, we adopt the GT-free vari-
ant of LightGS. Moreover, our experiments are conducted under substantially
more aggressive compression ratios than those commonly reported in prior work,
resulting in a significantly more challenging evaluation setting. Additionally, we
remove high-order SH components in our method for more extreme compres-
sion; accordingly, we report results for LightGS both with and without SH. In
addition, we evaluate the original optimization-based LightGS by running its
official optimization routine for 1,000 steps on the random views. For genera-
tive baselines, we evaluate PrefPaint [15] and the inpainting variant of Stable
Diffusion 2 [24]. We also compare with Difix3D [30], which applies single-step dif-
fusion to suppress residual artifacts. This setup ensures a fair comparison across
compression-only, optimization-based, and generative approaches.
Implementation Details. To preserve fine details under heavy compression
while maintaining efficiency, we enhance the VAE decoder with lightweight skip
adapters and integrate LoRA modules for parameter-efficient adaptation. During
training, the full UNet denoiser is optimized, while only the low-rank adapters
on the VAE are updated, balancing flexibility and stability. The overall loss is
  \ma t hc al {L}  = \m athcal {L}_{\text {L2}} + \lambda _{p}\,\mathcal {L}_{\text {LPIPS}} + \lambda _{\text {lat}}\,\mathcal {L}_{\text {latent}}, 
where λp and λlat are weighting factors. UGC and GaussPainter are trained
jointly under this unified objective to ensure consistency. Further implementation
details are provided in Supplement.

<!-- page 11 -->
Zip-GS
11
Table 1: Quantitative results on ScanNet++ under different compression
ratios of Gaussian counts, reporting LPIPS↓, SSIM↑, PSNR↑, and com-
pression ratio relative to the source file (404.73 MB). LightGS* indicates the
optimized version of LightGS. Bold indicates the best performance, and underline de-
notes the second-best.
Method
0.1
0.2
0.3
LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB)
LightGS (w/o SH)
0.4583
0.6130
15.56
36.5× (11.10)
0.3336
0.7577
19.71
18.2× (22.19)
0.2667
0.8270
22.45
12.2× (33.29)
LightGS (with SH) 0.4539
0.6186
15.64
10.0× (40.47)
0.3214
0.7750
20.20
5.0× (80.95)
0.2498
0.8510
23.45
3.3× (121.42)
LightGS*
0.4062
0.6842
18.87
10.0× (40.47)
0.3374
0.7393
20.59
5.0× (80.93)
0.2962
0.7774
21.78
3.3× (121.41)
FlexGaussian
0.4887
0.6833
15.62
60.3× (6.71)
0.4300
0.7899
20.09
31.3× (12.92)
0.3704
0.8408
23.23
21.2× (19.12)
Difix3D
0.4449
0.6037
15.60
–
0.3188
0.7528
19.67
–
0.2535
0.8228
22.33
–
PrefPaint
0.5518
0.5263
13.91
–
0.4547
0.6323
15.28
–
0.4086
0.6837
16.76
–
SD2-Inpaint
0.5893
0.4691
12.64
–
0.4988
0.5691
13.63
–
0.4583
0.6130
14.07
–
Ours
0.2321 0.8122 22.93
109.1× (3.71) 0.1948 0.8387 24.29
60.4× (6.70)
0.1718 0.8603 25.25
37.9× (10.68)
Table 2: Quantitative comparison of different methods under various com-
pression ratios of Gaussian counts on Replica. We report LPIPS↓, SSIM↑,
PSNR↑, and the compression ratio relative to the source file (354.77 MB). LightGS*
indicates the optimized version of LightGS. Bold indicates the best performance, and
underline denotes the second-best.
Method
0.1
0.2
0.3
LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB)
LightGS (w/o SH)
0.4158
0.7000
18.51
36.5× (9.73)
0.2645
0.8402
23.07
18.2× (19.46)
0.2130
0.8812
25.44
12.2× (29.18)
LightGS (with SH) 0.3871
0.7150
18.93
10.0× (35.48)
0.2393
0.8564
23.94
5.0× (70.95)
0.1950 0.8969 26.58
3.3× (106.43)
LightGS*
0.3242
0.8122
23.79
10.0× (35.47)
0.2843
0.8421
25.34
5.0× (70.93)
0.2669
0.8576
25.95
3.3× (106.43)
FlexGaussian
0.4451
0.7237
18.91
59.0× (6.01)
0.3496
0.8200
23.76
30.8× (11.52)
0.3125
0.8510
26.21
20.8× (17.03)
Difix3D
0.4264
0.6785
17.94
–
0.2821
0.8210
22.30
–
0.2358
0.8615
24.65
–
PrefPaint
0.5201
0.5701
13.76
–
0.4112
0.6688
15.43
–
0.3776
0.6928
16.02
–
SD2-Inpaint
0.5451
0.5299
12.74
–
0.4404
0.6233
14.05
–
0.4073
0.6482
14.50
–
Ours
0.2197 0.8298 24.32
107.2× (3.31)
0.1652 0.8717 26.81
53.3× (6.65)
0.1478 0.8844
27.22
35.9× (9.89)
5.2
Comparisons
Comparison on Indoor Datasets. In the indoor datasets ScanNet++ [32]
and Replica [27], we compare different compression and generation methods
under three compression ratios, where the ratios refer to the proportion of
Gaussians retained. As shown in Tab. 1, our method achieves the best over-
all performance across all settings. On ScanNet++, at the most aggressive ratio
(0.1) our approach delivers significant gains in perceptual and structural qual-
ity while reaching over 100× compression, clearly surpassing other methods,
with a PSNR improvement of about 4 dB over the second-best baseline. Similar
improvements are observed at ratios 0.2 and 0.3, where our method further en-
hances fidelity while maintaining fast inference speed. On Replica, our framework
consistently outperforms alternatives, sustaining PSNR above 26 dB and SSIM
around 0.88 even at higher compression levels. These results confirm that our
approach not only reduces artifacts and structural distortions but also preserves
fine details and scene fidelity under extreme compression. For completeness, de-
tailed rate–distortion curves on ScanNet++ and Replica are provided in the
appendix, which visually corroborate the compression trends discussed above.

<!-- page 12 -->
12
J. Chen et al.
GT
Rendered Mask
w/o Mask
w/ Mask
Fig. 5: Comparison of diffusion gen-
eration with and without mask guid-
ance. From left to right: ground-truth im-
age, rendered opacity mask from 3DGS
blending, generation without mask guid-
ance, and generation with mask guidance.
The mask effectively constrains the dif-
fusion model to complete missing regions
while preserving visible structures, lead-
ing to more faithful and realistic recon-
structions.
Comparison on Outdoor Datasets.
Tab. 4 reports results on the Mip-
NeRF360 [1] benchmark. Our method
again
surpasses
both
compression-
based and generative baselines across
all ratios. Even at an aggressive ratio of
0.03, our method significantly outper-
forms the second-best approach, achiev-
ing a 3.76 dB gain in PSNR and a 0.06
reduction in LPIPS, while still reach-
ing over 300× compression. At moder-
ate ratios (0.05 and 0.1), the framework
maintains superior perceptual metrics
with much lower distortion, highlight-
ing its robustness to extreme compres-
sion. Our method also comprehensively
outperforms other methods on Tanks
& Temples dataset [11]. Due to space
limitations, we place the results on the
Tanks & Temples dataset [11] in the ap-
pendix.
Compression on Generated 3D Assets. To further evaluate real-world ap-
plicability to automatically generated 3D content, we conduct an additional ex-
periment on four representative 3D assets synthesized by several world mod-
els [9,31] and converted into Gaussian representations. The intermediate multi-
view renderings produced during asset generation are used as reference images,
against which we compress each asset with different methods, render all view-
points, and compute LPIPS, SSIM, and PSNR over the full-view set. As sum-
marized in Tab. 3, ZipGS consistently achieves the best perceptual and distor-
tion metrics at substantially higher compression ratios than strong compression
baselines such as FlexGaussian, indicating that our framework transfers well to
generated 3D assets and remains well-suited for real-world streaming and de-
ployment scenarios highlighted in the introduction.
5.3
Ablation Studies
Effect of Pipeline Modules. To evaluate the impact of the compression stage,
we conduct ablations using only Global Significance Score pruning as the base-
line. As shown in Tab. 5a, adding UGC on top of this baseline improves structural
similarity and reconstruction fidelity, while further integrating GaussPainter en-
hances perceptual quality. When the two modules are combined, the system
achieves the best overall performance, reducing LPIPS to 0.1948 and increasing

<!-- page 13 -->
Zip-GS
13
Table 3: Quantitative comparison on generated 3D assets. We report LPIPS↓,
SSIM↑, PSNR↑(evaluated over all viewpoints), and compression ratio. Bold indicates
the best performance, and underline denotes the second-best.
Method
LPIPS↓SSIM↑PSNR↑Compression Ratio
LightGS (w/o SH)
0.4523
0.6821
19.45
38.2× (12.45)
LightGS (with SH) 0.4238
0.7145
20.12
10.5× (45.23)
LightGS*
0.4856
0.6654
18.93
10.5× (45.23)
FlexGaussian
0.4387
0.6987
19.78
55.3× (9.87)
Difix3D
0.4652
0.6512
18.34
–
PrefPaint
0.5723
0.5432
15.67
–
SD2-Inpaint
0.5891
0.5124
14.89
–
Ours
0.3124 0.7823 22.56
98.7× (5.61)
Table 4: Quantitative comparison of novel view synthesis on Mip-NeRF360.
We report LPIPS↓, SSIM↑, PSNR↑, and compression ratio relative to the source file
(658.91MB). LightGS* indicates the optimized version of LightGS. Bold indicates the
best performance, and underline denotes the second-best.
Method
0.03
0.05
0.1
LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB) LPIPS↓SSIM↑PSNR↑Ratio (Size, MB)
LightGS (w/o SH)
0.5868
0.4481
15.71
121.6× (5.42)
0.5313
0.5072
16.72
72.9× (9.03)
0.4585
0.5936
17.73
36.5× (18.07)
LightGS (with SH) 0.5802
0.4605
15.76
33.3× (19.77)
0.5224
0.5253
16.83
20.0× (32.95)
0.4466 0.6187 17.99
10.0× (65.89)
LightGS*
0.6840
0.4104
15.40
33.3× (19.77)
0.6286
0.4487
15.95
20.0× (32.95)
0.5491
0.5134
16.82
10.0× (65.89)
FlexGaussian
0.5626
0.4522
14.81
189.9× (3.47)
0.5182
0.5035
15.91
120.0× (5.49)
0.4464
0.5968
17.97
62.27× (10.58)
Difix3D
0.5217
0.4402
15.94
–
0.4690
0.4973
16.97
–
0.3999
0.5885
18.74
–
PrefPaint
0.6574
0.3619
14.27
–
0.6081
0.3983
14.84
–
0.5670
0.4135
14.73
–
SD2-Inpaint
0.6623
0.3428
12.95
–
0.6248
0.3716
13.46
–
0.5851
0.3930
13.39
–
Ours
0.4595 0.4832 19.70
352.3× (1.87) 0.3934 0.5375 21.01
227.2× (2.90) 0.3303 0.6054
22.13 102.15× (6.45)
SSIM/PSNR to 0.8387 and 24.29 dB. This experiment is conducted on Scan-
Net++ [32] with a 20% pruning ratio. Since GaussPainter is the core gener-
ation module, we further analyze its internal components: mask guidance and
latent supervision (Tab. 5c). This ablation is carried out on the ScanNet++
test set under a 10% compression ratio. Incorporating mask guidance improves
perceptual quality by helping the model focus on relevant regions, while adding
latent supervision yields the most significant gains, demonstrating that latent-
level consistency provides a stronger learning signal and substantially improves
perceptual fidelity.
Compatibility with Other Compression Methods. To evaluate the gener-
ality of GaussPainter, we apply it to scenes compressed by LightGS [6] and
MaskGaussian [17] on ScanNet++ at 30% Gaussian retention. As shown in
Tab. 6, GaussPainter consistently improves all metrics across different prun-
ing backbones, demonstrating that it is backbone-agnostic and can serve as a
plug-and-play refinement module for existing Gaussian compression strategies.

<!-- page 14 -->
14
J. Chen et al.
Table 5: Ablation studies of the proposed framework on ScanNet++ [32]. (a)
Effect of UGC and GaussPainter. (b) Effect of mask guidance and latent supervision.
(c) Inference time comparison on an NVIDIA A100 GPU.
(a) Effect of pipeline components
UGC GaussPainter LPIPS↓SSIM↑PSNR↑
×
×
0.3336
0.7577
19.71
✓
×
0.2985
0.7794
21.73
×
✓
0.2059
0.8336
23.24
✓
✓
0.1948 0.8387 24.29
(b) Inference time (A100)
Method
Time (ms)
Difix3D
230.36 ± 16.47
PrefPaint
2724.02 ± 24.32
SD2-Inpaint 2683.14 ± 61.76
Ours
65.85 ± 0.63
(c) Effect of design choices
Method
LPIPS↓SSIM↑PSNR↑
Baseline
0.3464
0.7480
21.05
+ Mask guidance
0.3305
0.7676
21.73
+ Latent supervision 0.2321 0.8122 22.93
Table 6: Compatibility with different compression methods on ScanNet++.
Results are reported at 30% Gaussian retention using PSNR↑, LPIPS↓, and SSIM↑.
Adding GaussPainter consistently improves both LightGS and MaskGaussian.
Method
PSNR ↑LPIPS ↓SSIM ↑Method
PSNR ↑LPIPS ↓SSIM ↑
LightGS
27.55
0.1615
0.9039 MaskGaussian
25.87
0.3078
0.8198
LightGS + GaussPainter 28.31
0.1271 0.9164 MaskGaussian + GaussPainter 26.56
0.1340 0.8770
6
Conclusion
In this work, we introduced ZipGS, a feed-forward framework for extreme com-
pression of 3D Gaussian representations that reconciles aggressive pruning with
high-fidelity rendering. By decoupling compression and restoration into Univer-
sal Gaussian Compression (UGC) and GaussPainter, ZipGS removes redundant
Gaussians without per-scene optimization and leverages mask-guided, latent-
space diffusion to recover missing structures while refining visible regions. Ex-
periments on ScanNet++, Replica, Mip-NeRF360, Tanks & Temples, and auto-
matically generated 3D assets show that ZipGS consistently outperforms strong
compression and generative baselines across a wide range of ratios while remain-
ing compatible with existing pruning backbones, highlighting diffusion priors as
a practical tool for streaming and deploying Internet-scale 3D Gaussian assets.

<!-- page 15 -->
Zip-GS
15
References
1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance
fields. In: Proceedings of the IEEE/CVF international conference on computer
vision. pp. 5855–5864 (2021)
2. Bohan, O.B.: Taesd: Tiny autoencoder for stable diffusion. https://github.com/
madebyollin/taesd (2023), accessed: 2025-09-17
3. Chen, H., Gu, J., Chen, A., Tian, W., Tu, Z., Liu, L., Su, H.: Single-stage diffusion
nerf: A unified approach to 3d generation and reconstruction. In: Proceedings of
the IEEE/CVF international conference on computer vision. pp. 2416–2425 (2023)
4. Chen, J., Li, C., Lee, G.H.: Dihur: Diffusion-guided generalizable human recon-
struction. arXiv preprint arXiv:2411.11903 (2024)
5. Chen, Z., Wang, C., Guo, Y.C., Zhang, S.H.: Structnerf: Neural radiance fields for
indoor scenes with structural hints. IEEE Transactions on Pattern Analysis and
Machine Intelligence 45(12), 15694–15705 (2023)
6. Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., et al.: Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances
in neural information processing systems 37, 140138–140158 (2024)
7. Fang, G., Wang, B.: Mini-splatting: Representing scenes with a constrained number
of gaussians. In: European Conference on Computer Vision. pp. 165–181. Springer
(2024)
8. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in
neural information processing systems 33, 6840–6851 (2020)
9. HunyuanWorld, T.: Hunyuanworld 1.0: Generating immersive, explorable, and in-
teractive 3d worlds from words or pixels. arXiv preprint (2025)
10. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
11. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking
large-scale scene reconstruction. ACM Transactions on Graphics 36(4) (2017)
12. Lee, J.C., Rho, D., Sun, X., Ko, J.H., Park, E.: Compact 3d gaussian representation
for radiance field. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 21719–21728 (2024)
13. Li, Y., Ma, Q., Yang, R., Li, H., Ma, M., Ren, B., Popovic, N., Sebe, N., Konukoglu,
E., Gevers, T., et al.: Scenesplat: Gaussian splatting-based scene understanding
with vision-language pretraining. arXiv preprint arXiv:2503.18052 (2025)
14. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu,
Z., Lu, Y., et al.: Dl3dv-10k: A large-scale scene dataset for deep learning-based
3d vision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 22160–22169 (2024)
15. Liu, K., Zhu, Z., Li, C., Liu, H., Zeng, H., Hou, J.: Prefpaint: Aligning image
inpainting diffusion model with human preference. Advances in Neural Information
Processing Systems 37, 30554–30589 (2024)
16. Liu, X., Wang, Y., Chen, Z., Cao, J., Zhang, H., Zhang, Y., Yang, X.: One-step dif-
fusion model for image motion-deblurring. arXiv preprint arXiv:2503.06537 (2025)
17. Liu, Y., Zhong, Z., Zhan, Y., Xu, S., Sun, X.: Maskgaussian: Adaptive 3d gaussian
representation from probabilistic masks. In: Proceedings of the Computer Vision
and Pattern Recognition Conference. pp. 681–690 (2025)
18. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Struc-
tured 3d gaussians for view-adaptive rendering. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 20654–20664 (2024)

<!-- page 16 -->
16
J. Chen et al.
19. Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F.V., De La Torre, F.:
Taming 3dgs: High-quality radiance fields with limited resources. In: SIGGRAPH
Asia 2024 Conference Papers. pp. 1–11 (2024)
20. Mu, Y., Zuo, X., Guo, C., Wang, Y., Lu, J., Wu, X., Xu, S., Dai, P., Yan, Y.,
Cheng, L.: Gsd: View-guided gaussian splatting diffusion for 3d reconstruction.
arXiv preprint arXiv:2407.04237 (2024)
21. Navaneet, K., Pourahmadi Meibodi, K., Abbasi Koohpayegani, S., Pirsiavash, H.:
Compgs: Smaller and faster gaussian splatting with vector quantization. In: Euro-
pean Conference on Computer Vision. pp. 330–349. Springer (2024)
22. Niedermayr, S., Stumpfegger, J., Westermann, R.: Compressed 3d gaussian splat-
ting for accelerated novel view synthesis. In: Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition. pp. 10349–10358 (2024)
23. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International conference on machine learning. pp.
8748–8763. PMLR (2021)
24. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution
image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684–
10695 (June 2022)
25. Schwarz, K., Mueller, N., Kontschieder, P.: Generative gaussian splatting: Gener-
ating 3d scenes with video diffusion priors. arXiv preprint arXiv:2503.13272 (2025)
26. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-
based generative modeling through stochastic differential equations. arXiv preprint
arXiv:2011.13456 (2020)
27. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-
Artal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan,
X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler,
E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M.,
Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor
spaces. arXiv preprint arXiv:1906.05797 (2019)
28. Tian, B., Gao, Q., Xianyu, S., Cui, X., Zhang, M.: Flexgaussian: Flexible and
cost-effective training-free compression for 3d gaussian splatting. arXiv preprint
arXiv:2507.06671 (2025)
29. Wang, G., Chen, Z., Loy, C.C., Liu, Z.: Sparsenerf: Distilling depth ranking for
few-shot novel view synthesis. In: Proceedings of the IEEE/CVF international
conference on computer vision. pp. 9065–9076 (2023)
30. Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z.,
Ling, H.: Difix3d+: Improving 3d reconstructions with single-step diffusion models.
In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp.
26024–26035 (2025)
31. Yang, Z., Ge, W., Li, Y., Chen, J., Li, H., An, M., Kang, F., Xue, H., Xu, B., Yin,
Y., Li, E., Liu, Y., Wang, Y., Guo, H.X., Zhou, Y.: Matrix-3d: Omnidirectional
explorable 3d world generation. arXiv preprint arXiv:2508.08086 (2025)
32. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset
of 3d indoor scenes. In: Proceedings of the International Conference on Computer
Vision (ICCV) (2023)
33. Yin, X., Zhang, Q., Chang, J., Feng, Y., Fan, Q., Yang, X., Pun, C.M., Zhang,
H., Cun, X.: Gsfixer: Improving 3d gaussian splatting with reference-guided video
diffusion priors. arXiv preprint arXiv:2508.09667 (2025)

<!-- page 17 -->
Zip-GS
17
7
Supplement
7.1
Training Details
All experiments are conducted on NVIDIA A100 GPUs with mixed-precision
training. We train for up to 10,000 steps with a learning rate of 2 × 10−5. The
backbone components—tokenizer, text encoder, VAE, and U-Net—are initialized
from the pre-trained sd-turbo model [23, 24]. We adopt a one-step diffusion
scheduler implemented via DDPMScheduler and apply mask conditioning to guide
the denoising process.
During training, UGC and GaussPainter are jointly optimized with a per-
GPU batch size of 1. Each forward pass processes a single scene: Gaussian prim-
itives are compressed by UGC, and a randomly sampled view is rendered into
an image that GaussPainter uses for reconstruction.
The variational autoencoder (VAE) is augmented with LoRA (rank 4). LoRA
adapters are inserted into all Conv2d layers of the VAE decoder, with Gaussian-
initialized LoRA weights to ensure stable optimization.
7.2
Additional Experimental Results
This section complements the main paper with results under additional com-
pression ratios, extended qualitative visualizations, and further ablations.
Indoor benchmarks at 20% and 30% ratios. Fig. 6 and 7 present quali-
tative comparisons on ScanNet++ and Replica at pruning ratios r=0.20 and
r=0.30, respectively. For each scene, we show the input rendering after compres-
sion, our reconstruction, and zoom-in crops. These visualizations corroborate the
quantitative trends reported in the main paper: as the pruning ratio increases,
UGC preserves structural fidelity while GaussPainter effectively restores texture
and color consistency, yielding perceptually sharper details and fewer artifacts.
Outdoor benchmarks at 3% and 5% ratios. Fig. 8 shows additional results
on Mip-NeRF360 at extreme ratios r=0.03 and r=0.05. Despite aggressive com-
pression, the combination of mask-guided conditioning and one-step diffusion
maintains coherent geometry and suppresses color drift in low-texture regions,
demonstrating robustness in challenging outdoor scenes.
7.3
Details of Data Sources and Training Details
The Gaussian scenes used in our experiments are derived from publicly avail-
able datasets. For indoor benchmarks, ScanNet++ and Replica Gaussian scenes
are obtained from the SceneSplat repository [13]. For outdoor scenes on Mip-
NeRF360 [1] and DL3DV [14], Gaussian scenes are generated using the 3DGS
pipeline with 30,000 optimization steps; densification and other hyperparameters
follow the default 3DGS settings unless otherwise stated.

<!-- page 18 -->
18
J. Chen et al.
LightGS*
FlexGaussian
Difix3D
PrefPaint
SD2-Inpaint
Ours
GT
ScanNet++                                   
Replica
Fig. 6: Indoor results at r=0.20 (20% pruning). Qualitative comparisons on Scan-
Net++ and Replica. Here r denotes the fraction of Gaussians retained after compres-
sion; for example, r=0.20 means 20% of the original Gaussians are preserved. For each
scene we show (left to right): compressed input rendering, our reconstruction, and
zoom-in patches. Our method preserves structure while recovering fine textures.
We train our models using two data sources: DL3DV and ScanNet++. The
training set includes all video sequences from DL3DV and the official training
split of ScanNet++. For validation, we randomly render 100 novel views from the
training scenes. For testing, we use the 50 official test scenes from ScanNet++
and all scenes from Replica and Mip-NeRF360.
7.4
Generalization to Unseen Scenes and Performance Trends.
To further evaluate how well our method generalizes beyond the training dis-
tribution, we conduct experiments on two scenes from the Tanks & Temples
dataset, the Mip-NeRF360 dataset, and the Replica dataset. All these scenes
are not included in our training data, ensuring that the model has never ob-
served any images from these environments, and thus Tanks & Temples, Mip-
NeRF360, and Replica all serve as out-of-domain data for testing generalization
performance.
The Fig. 11 reports the PSNR, SSIM, and LPIPS curves under different
compression rates. Across the three metrics, UGC + GaussPainter performs
consistently better or comparable to UGC across most compression levels. The
gains are most notable under aggressive compression, where the Gaussian repre-
sentation becomes extremely sparse and GaussPainter effectively recovers miss-

<!-- page 19 -->
Zip-GS
19
LightGS*
FlexGaussian
Difix3D
PrefPaint
SD2-Inpaint
Ours
GT
ScanNet++                                   
Replica
Fig. 7: Indoor results at r=0.30 (30% pruning). Additional qualitative compar-
isons on ScanNet++ and Replica. Here r=0.30 means that 30% of the original Gaus-
sians are retained. Even under this stronger compression, the proposed pipeline miti-
gates artifacts and maintains perceptual quality.
ing structures. At higher compression budgets, UGC alone occasionally achieves
slightly better scores; we believe this is because the Tanks & Temples scenes
contain complex outdoor geometry and fine textures, where the generative re-
finement may introduce subtle artifacts when the underlying Gaussian represen-
tation is already of high quality. As summarized in Tab. 2, the Replica results
are reported in this table. The quantitative comparisons on the Mip-NeRF360
dataset are reported in Tab. 4.
Overall, these results show that our approach generalizes well to unseen
scenes and provides clear benefits under challenging compression settings, while
maintaining competitive performance in high-quality regimes.

<!-- page 20 -->
20
J. Chen et al.
LightGS*
FlexGaussian
Difix3D
PrefPaint
SD2-Inpaint
Ours
GT
Mip-Nerf360(Ratio=0.03)
Mip-Nerf360(Ratio=0.05)
Mip-nerf360(Ratio=0.1)
Fig. 8: Outdoor results on Mip-NeRF360 at r=0.03 and r=0.05. Here r=0.03
and r=0.05 indicate that only 3% and 5% of the original Gaussians are retained, respec-
tively. Despite such extreme compression, our approach preserves coherent geometry
and reduces color drift, particularly in low-texture or large-hole regions.
Fig. 9: Quantitative results on ScanNet++ under different compression ra-
tios of Gaussian counts, reporting LPIPS↓, SSIM↑, PSNR↑.
Fig. 10: Quantitative results on Replica under different compression ratios
of Gaussian counts, reporting LPIPS↓, SSIM↑, PSNR↑.

<!-- page 21 -->
Zip-GS
21
Fig. 11: Outdoor results on Tanks & Temples datasets This figure presents
three curves (PSNR, SSIM, LPIPS) of UGC + GaussPainter and UGC under different
compression rates on the unseen scenes of Tanks & Temples dataset. Across metrics,
UGC + GaussPainter performs better or comparable to UGC in most cases—its gains
are prominent under aggressive compression.
Fig. 12: Quantitative results on ScanNet++ under different compression
ratios of Gaussian counts, comparing the performance of LightGS and
LightGS+GaussPainter across PSNR, SSIM, and LPIPS metrics.
