<!-- page 1 -->
Style4D-Bench: A Benchmark Suite for 4D Stylization
Beiqi Chen1,2∗
Shuai Shao2∗
Haitang Feng3,2
Jianhuang Lai4
Jianlou Si5B
Guangcong Wang2B
1Harbin Institute of Technology,
2Vision, Graphics, and X Group, Great Bay University,
3Nanjing University,
4Sun Yat-Sen University,
5Alibaba Group
Project page: https://becky-catherine.github.io/Style4D/
Abstract
We introduce Style4D-Bench, the first benchmark suite
specifically designed for 4D stylization, with the goal of
standardizing evaluation and facilitating progress in this
emerging area. Style4D-Bench comprises: 1) a compre-
hensive evaluation protocol measuring spatial fidelity, tem-
poral coherence, and multi-view consistency through both
perceptual and quantitative metrics, 2) a strong baseline
that make an initial attempt for 4D stylization, and 3) a
curated collection of high-resolution dynamic 4D scenes
with diverse motions and complex backgrounds. To estab-
lish a strong baseline, we present Style4D, a novel frame-
work built upon 4D Gaussian Splatting.
It consists of
three key components: a basic 4DGS scene representation
to capture reliable geometry, a Style Gaussian Represen-
tation that leverages lightweight per-Gaussian MLPs for
temporally and spatially aware appearance control, and
a Holistic Geometry-Preserved Style Transfer module de-
signed to enhance spatio-temporal consistency via con-
trastive coherence learning and structural content preser-
vation. Extensive experiments on Style4D-Bench demon-
strate that Style4D achieves state-of-the-art performance in
4D stylization, producing fine-grained stylistic details with
stable temporal dynamics and consistent multi-view render-
ing. We expect Style4D-Bench to become a valuable re-
source for benchmarking and advancing research in stylized
rendering of dynamic 3D scenes.
1. Introduction
Recent advances in 4D scene representations [12, 38, 44,
54, 58] have enabled high-fidelity modeling of dynamic en-
vironments, unlocking new possibilities for immersive con-
tent creation in virtual reality and film production. As appli-
cations become more demanding, there is a growing need
not only for accurate reconstructions but also for control-
lable appearance and stylization of dynamic 4D content.
Users may wish to stylize a 4D scene to reflect specific artis-
tic intents, emotional tones, or narrative contexts—while
maintaining both temporal coherence and multi-view con-
sistency.
Despite recent progress in related areas like 2D and 3D
stylization, the field of 4D stylization remains largely under-
explored, with no standardized datasets, evaluation met-
rics, or task definitions.
Existing methods adapted from
2D [17, 18, 21, 26, 31], 3D [9, 34, 41, 45, 51, 63, 64],
or video [10, 28] stylization often fail to meet the unique
challenges of 4D settings, such as jointly preserving spa-
tial fidelity, temporal stability, and multi-view consistency
under complex motion and occlusion.
Moreover, recent
dynamic scene representations like 4D Gaussian Splatting
(4DGS) [54] offer strong rendering capabilities, but have
not yet been explored in the context of stylization.
To fill this gap, we introduce Style4D-Bench—the
first benchmark suite dedicated to 4D stylization.
Our
benchmark provides: 1) A curated set of high-resolution,
complex-background dynamic 4D scenes exhibiting diverse
motions, deformations, and view-dependent effects; 2) A
comprehensive evaluation protocol, including both percep-
tual and quantitative metrics, to assess spatial fidelity, tem-
poral coherence, and cross-view consistency.
Style4D-
Bench is designed to facilitate fair, reproducible, and scal-
able evaluation of future 4D stylization approaches.
To establish a strong baseline within our benchmark, we
also propose Style4D, a novel 4D stylization framework
based on 4D Gaussian Splatting. Style4D consists of three
key components: a basic 4DGS scene representation for
geometry modeling, a Style Gaussian Representation that
incorporates lightweight per-Gaussian MLPs for time- and
depth-aware stylization, and a Holistic Geometry-Preserved
Style Transfer module to enhance spatio-temporal consis-
tency through contrastive learning and content-aware regu-
larization.
1
arXiv:2508.19243v1  [cs.CV]  26 Aug 2025

<!-- page 2 -->
Our contributions are summarized as follows: 1) We
present Style4D-Bench, the first benchmark suite for 4D
scene stylization, offering standardized datasets, tasks, and
evaluation metrics to drive progress in this emerging area.
2) We define core evaluation challenges in 4D stylization,
including spatial fidelity, temporal stability, and multi-view
consistency, and design comprehensive protocols to quan-
tify them.
3) We propose Style4D as a strong baseline
method, which leverages 4D Gaussian Splatting with a
style-aware representation and a holistic spatio-temporal
transfer module.
4) Extensive experiments on Style4D-
Bench demonstrate the effectiveness of our method and re-
veal insights into current limitations and future opportuni-
ties in 4D stylization.
2. Related Works
3D Representations. Neural Radiance Fields (NeRFs) [1,
39] have transformed 3D scene representation by model-
ing scenes as continuous volumetric functions with MLPs,
enabling high-quality novel view synthesis. Efficiency and
quality have been further improved via compact represen-
tations such as decomposed tensors [4, 5, 16], hash ta-
bles [40], and voxel grids [15, 48]. Recently, 3D Gaus-
sian Splatting (3DGS) [25] has emerged as a compelling
alternative, representing scenes with explicit 3D Gaussians
and enabling real-time rasterization-based rendering. Its ex-
plicit structure is particularly suitable for editing [7, 19, 33,
50, 68], but 3DGS remains limited to static scenes.
4D Representations. To extend NeRF to dynamic settings,
early works [42–44] modeled temporal variations directly,
while later methods [2, 13, 37, 46, 60] improved efficiency
via voxel-based decompositions. Despite these advances,
achieving real-time rendering and preserving fine-grained
geometry under motion remains challenging. 4D Gaussian
Splatting (4DGS) [12, 54] addresses dynamic scenes by ex-
tending Gaussians into the temporal domain. Unlike Dy-
namic3DGS [38], which stores per-frame parameters with
linear memory growth, 4DGS uses a deformation network
to model temporal changes compactly and efficiently. How-
ever, current 4D representations lack mechanisms for styl-
ization and temporal consistency, motivating our bench-
mark and method for 4D scene stylization.
2D Style Transfer.
Neural style transfer, pioneered by
[17], demonstrated that CNNs can effectively separate and
recombine content and style from images.
The key in-
sight that second-order statistics of VGG features capture
style information led to numerous improvements.
Feed-
forward methods like AdaIN [21] significantly accelerated
stylization by aligning feature statistics, while recent works
have focused on improving semantic consistency and tex-
ture preservation [18, 26]. To maintain temporal coherence
across frames, MCCNet [10] proposes a Multi-Channel
Correlation network. These foundational techniques form
the basis for our 4D stylization component, which we en-
hance with temporal consistency mechanisms to handle dy-
namic scenes.
3D and Video Stylization. Extending style transfer to 3D
has attracted significant attention.
NeRF-based methods
achieve multi-view consistent stylization through optimiza-
tion [41, 51, 64] or feed-forward networks [9, 36]. With
the advent of 3D Gaussian Splatting, StyleGaussian [34]
demonstrates instant style transfer by embedding VGG fea-
tures into Gaussians and aligning their statistics with style
images. GSS [45] and StylizedGS [63] also leverage 3DGS
but require per-style optimization. While these methods ex-
cel at static scenes, they fail to handle temporal dynam-
ics inherent in 4D content, such as motion, deformation,
and occlusion, leading to view inconsistencies and tempo-
ral flickering. Existing video stylization methods [8, 22, 51]
primarily focus on temporal coherence in 2D space without
incorporating 3D scene understanding, whereas 3D styliza-
tion methods typically disregard temporal dynamics alto-
gether.
4D Stylization and Benchmark. Despite notable progress
in 2D and 3D stylization, 4D scene stylization remains
largely unexplored.
The core challenge lies in simulta-
neously ensuring multi-view consistency across different
viewpoints and temporal stability across frames.
Recent
concurrent efforts such as 4DStyleGaussian [30] make early
steps toward 4D stylization, yet they still face limitations
in content preservation, geometric fidelity, and temporally
adaptive style control. Our strong baseline addresses these
challenges by introducing per-Gaussian MLPs to enable
fine-grained, temporally-aware appearance modulation, and
by integrating an enhanced 2D stylization module designed
to preserve spatial-temporal consistency throughout the 4D
stylization process.
Designing a benchmark for 4D stylization poses funda-
mental challenges. The absence of ground truth and the
subjective nature of style make quantitative evaluation in-
herently difficult.
Moreover, measuring temporal coher-
ence and multi-view consistency is non-trivial due to com-
plex spatial-temporal dynamics and the lack of reference
sequences. Existing evaluation protocols from 2D, 3D, or
video stylization are insufficient for capturing these aspects,
highlighting the need for dedicated datasets and tailored
metrics.
3. Style4D-Bench
Existing 4D stylization tasks lack a unified quantitative
evaluation standard. Some studies [30] measure consistency
by calculating frame-wise PSNR, MSE, and other metrics,
using style loss to assess stylization degree. In contrast, oth-
ers [36] overly rely on user studies. While these metrics
simplify the evaluation process, they introduce several is-
sues. First, simple style losses like MSE overly focus on
2

<!-- page 3 -->
pixel-level differences between stylized and reference im-
ages, failing to accurately reflect overall similarity in se-
mantic and textural aspects. Second, 4D stylization is a
complex and multidimensional concept where individual
preferences may prioritize different aspects. For instance,
some emphasize stylization degree, considering blurriness
of characters or objects as part of style, while others pri-
oritize preserving the structure and details of the original
scene. To address these challenges, we propose a decompo-
sition method that breaks down the evaluation of 4D styl-
ization into multiple dimensions for a more comprehensive
and nuanced assessment.
We divide the evaluation of 4D stylization into two as-
pects: 4D stylization quality and 4D stylization consistency.
For 4D stylization quality, we focus on assessing the frame-
by-frame quality of stylized videos rendered at arbitrary
times and angles, without considering similarity to a ref-
erence style. Regarding 4D stylization consistency, we dis-
tinguish between Temporal Quality and Stylization Quality.
Our evaluation method encompasses a total of six specific
dimensions, comprising 12 detailed metrics in total.
3.1. 4D Stylization Quality
Frame-Wise Quality - Imaging Quality. Imaging quality
refers to the distortions (e.g., blurring, high noise, overexpo-
sure) present in each frame after 4D stylization. We assess
this quality using three metrics: UIQM, Clipiqa+[52], and
Musiq[24].
• UIQM: Integrates three dimensions of image qual-
ity—color, sharpness, and contrast—and computes the
overall image quality through a weighted averaging ap-
proach, aligning with human visual perception.
• Clipiqa+: A CLIP-based image quality assessor fine-
tuned using CoOp [67], designed to evaluate overall im-
age quality.
• Musiq: A multi-scale image quality assessor trained on
the KonIQ dataset [20], capable of capturing and evaluat-
ing image quality at various granularities.
Frame-Wise Quality - Aesthetic Quality. Aesthetic qual-
ity refers to the artistic and aesthetic value of each video
frame, reflecting aspects such as layout, color richness and
harmony, photorealism, naturalness, and artistic quality. We
assess aesthetic quality using the Qalign[55] and Musiq-
paq2piq[24] metrics.
• Qalign: A pre-trained large multimodal model (LLM) ca-
pable of simultaneously assessing both image quality and
aesthetic performance.
• Musiq-paq2piq: A multi-scale image aesthetic assessor
trained on the PaQ-2-PiQ dataset[61], capable of captur-
ing image quality and performing aesthetic evaluation at
various granularities.
3.2. 4D Stylization Consistency
Temporal Quality - Spatiotemporal Consistency. Tem-
poral consistency refers to maintaining continuity between
frames and consistency across multiple viewpoints as time
and perspectives change. We evaluate multiple helical tra-
jectory videos rendered for each scene using Dists[11] and
Warp loss metrics to assess cross-frame consistency.
• Dists: Comprehensively measures both the structural and
textural similarity between two images to evaluate their
overall quality and perceptual consistency.
• Warp loss: Using RAFT[49] to compute optical flow,
mapping the next frame to the current frame, and cal-
culating the L1 error between the current frame and the
mapped frame effectively evaluates spatial and motion
consistency of images in a temporal sequence
Temporal Quality - Subject Consistency.
For stylized
videos rendered from a fixed viewpoint, we assess whether
the appearance of a subject (e.g., a person, curtains, etc.)
remains consistent throughout the entire video. To achieve
this, we compute the inter-frame DINO[3] feature similar-
ity.
Stylization Quality - Style Consistency.Style consistency
refers to the degree of stylization in 4D stylization, evalu-
ated using CKDN[66] and LPIPS[65] metrics to assess the
similarity between video frames and style images, repre-
senting the level of stylization.
• CKDN: Utilizes learned representations from degraded
images to assess style similarity, enabling comprehensive
evaluation of both image quality and style similarity.
• LPIPS: A metric based on features extracted from VGG,
measuring perceptual similarity between images, effec-
tively assessing similarity to style images.
Stylization Quality - Content Consistency. Content Con-
sistency refers to the semantic and content similarity be-
tween stylized 4D scenes and their original counterparts.
We employ SSIM and LPIPS metrics to compute the simi-
larity of each frame to the original scene.
3.3. User Study Design
To thoroughly evaluate our method, we conducted a user
study with a total of 34 participants. The study consists of
two parts. The first part focuses on 4D stylization, where we
presented five pairs of videos—each approximately 10 sec-
onds long and containing 300 frames—from test viewpoints
and arbitrary transformed viewpoints, along with four se-
lected frames extracted for detailed qualitative assessment.
The second part aims to validate the effectiveness of our
proposed HGST method in video stylization, featuring two
pairs of 10-second videos (300 frames each) as well as six
individually selected frames for fine-grained evaluation.
Metric. For the extracted single frames, we evaluated
two metrics: stylization quality and image quality. Styliza-
tion quality measures the extent to which edges and textures
3

<!-- page 4 -->
are transformed to reflect the target style without compro-
mising the original image structure. Image quality assesses
the clarity of object boundaries and facial details.
For the continuous long videos, we adopted three met-
rics: stylization quality, spatiotemporal consistency, and
video quality. Stylization quality evaluates how well the
video’s style matches the reference while preserving the
original structure.
Spatiotemporal consistency measures
the coherence of the video across temporal progression and
viewpoint changes. Video quality assesses the clarity of fine
details and textures, as well as overall visual preference.
4. Style4D: A Strong Baseline
Overview of Style4D. In 4D stylization, the direct use of
style transformation leads to low multi-view consistency
and significant blurry artifacts. To address these issues, we
propose Style4D, a new dynamic scene stylization frame-
work, based on the decoupling of geometry learning and
style learning.
The framework of Style4D is illustrated
in Figure 1.
Style4D consists of three key components,
a basic 4DGS representation, a Style Gaussian Represen-
tation, and a Holistic Geometry-preserved Style Transfer.
We first train a basic 4DGS representation given multi-
ple views Icontent to obtain the static Gaussian sequence
Gi = {µi, ri, si, oi, csh
i } and the corresponding Gaussian
Deformation Field Network Fdef, capturing the the geom-
etry of a dynamic scene.
To stylize a 4D scene repre-
sented by 4DGS, we propose a Style Gaussian Represen-
tation method. It is a novel type of Gaussian ellipsoid, with
attributes defined as Gi =
n
µi, ri, si, f style
i
o
. We design
a tiny MLP as part of the style attribute f style
i
. The design
provides pixel-level stylization mapping, and thus achieves
finer color expression while balancing local and global con-
sistency, which significantly improves multi-view consis-
tency. Finally, we introduce a geometry-preserving style
transfer approach that integrates an attention-guided 2D
stylization module with contrastive coherence learning, en-
abling the generation of high-quality and temporally con-
sistent training frames.
Style Gaussian Representation. Inspired by SuperGaus-
sians [57], which enhances 2DGS using bilinear interpo-
lation and spatially varying features, we introduce a Style
Gaussian Representation for 4D scenes.
Extending ker-
nel functions to four-dimensional interpolation does not ef-
fectively capture the four-dimensional variations in scenes.
Therefore, we introduce Gaussian MLP features.
It is
worth noting that directly mapping intersection coordinates
to color and opacity can lead to overfitting. Thus, we map
color and opacity variations based on intersection depth.
Specifically, each Gaussian is assigned a tiny MLP and a
style code f style to modulate color and opacity over space
and time. Given the camera pose M = [R, T], we compute
the ray-ellipsoid intersection point pt from pixel p and time
t, following [62]. The color is then rendered as:
c(p) =
X
i∈N (p)
(ci + Fc(pt, t)) · F i
α
i−1
Y
j=1
(1 −αj(p)),
(1)
where ci is the view-dependent base color, Fc(pt, t) is the
style-driven color increment from the MLP, and αj(p) de-
notes opacity.
All Fc and αj terms are predicted per-
Gaussian via MLPs, with pt as the ray-Gaussian intersec-
tion depth. This formulation preserves 3D geometry, allows
precise temporal control, and enhances multi-view consis-
tency.
Holistic Geometry-preserved Style Transfer.
Long
video stylization remains challenging, especially for high-
resolution sequences. Diffusion-based methods [14, 23, 59]
often suffer from structural distortions and temporal incon-
sistency. Optical flow-based constraints [6, 35] improve co-
herence but are computationally expensive and scale poorly.
Self-supervised approaches [27, 56] reduce flickering but
may introduce artifacts such as hollow textures and sharp
pixel boundaries due to lack of semantic guidance.
To address these issues, we propose a Holistic Geometry-
preserved Style Transfer (HGST) module based on an
encoder-transformer-decoder architecture [27].
We fuse
style and content features using Multichannel Correlation to
ensure global consistency, and decode stylized outputs with
better structural integrity. However, the encoder-decoder
pipeline still leads to temporal instability, particularly on
large unseen frames. Inspired by [56], we introduce a dual
constraint: an attention-guided local contrastive loss Llcl
and a global feature consistency loss Lcontent, which jointly
enhance local coherence and global structure, effectively
mitigating flickering and spatial artifacts. To enhance lo-
cal and global coherence, we extract multi-scale features
from Ics, Icontent, and Istyle via an encoder, denoted as
f cs
i , f c
i , and f s
i for layers i = 1 to 5. For i = 3 to 5, we
apply CBAM [53] to enhance salient features and randomly
sample N locations Gx and their 8-nearest neighbors Gx,y
with small perturbations. Local differences are defined as
dg
x,y = Gf cs
i
x
−Gf cs
i
x,y, dc
x,y = Gf c
i
x −Gf c
i
x,y. We employ
a contrastive loss to maximize similarity between aligned
local differences:
Llcl =
8N
X
m=1
−log
exp(dg
m · dc
m/τ)
P8N
n=1 exp(dg
m · dcn/τ)
,
τ = 0.07.
(2)
To mitigate artifacts and preserve global structure, we
introduce a global content loss:
Lcontent = 1
N
N
X
i=1
∥fcsi −fci∥2
2.
(3)
4

<!-- page 5 -->
Figure 1. Overview of Style4D. Style4D consists of three key components, a basic 4DGS representation, a Style Gaussian Representation,
and a Holistic Geometry-preserved Style Transfer. We first train a basic 4DGS representation with the content image to obtain 4D scene
geometry. Then we propose a new Style Gaussian Representation for 4D stylization. We also introduce a Holistic Geometry-preserved
Style Transfer module to improve consistency and quality of stylization.
Method
Dataset
Imaging Quality
Aesthetic Quality
Spatiotemporal Consistency
Subject Consistency
Style Consistency
Content Consistency
UIQM↑
Clipiqa+↑
Musiq↑
Qalign↑
Musiq-paq2piq↑
Dists↓
Warp Loss↓
DINO Score↑
CKDN↑
LPIPS↓
SSIM↑
LPIPS↓
4DGS(AdaIN)
cook spinach
1.2995
0.4209
49.4095
2.8665
60.2165
0.0114
0.0091
0.9309
0.2084
0.6913
0.4763
0.4898
4DGS(AdaAttN)
1.7240
0.3770
36.1325
2.1298
50.7028
0.0215
0.0117
0.9078
0.2173
0.6904
0.6444
0.2841
4DStyleGaussian
1.0834
0.4267
44.2200
2.7019
55.2280
0.0121
0.0053
0.9403
0.1978
0.7106
0.7646
0.2159
Style4D(Ours)
1.9290
0.4437
53.4681
3.2072
65.8520
0.0112
0.0058
0.9395
0.2290
0.6866
0.7771
0.1834
4DGS(AdaIN)
flame salmon 1
1.6918
0.3336
41.3119
2.8497
51.6363
0.0171
0.0141
0.9267
0.2193
0.7605
0.5201
0.4030
4DGS(AdaAttN)
1.2928
0.3120
39.9117
2.2357
54.5593
0.0271
0.0169
0.9072
0.1966
0.7861
0.6054
0.2944
4DStyleGaussian
1.5488
0.3218
51.3875
3.2182
61.4087
0.0140
0.0074
0.9402
0.1897
0.7701
0.5081
0.6051
Style4D(Ours)
1.7529
0.3962
55.2302
3.6030
63.5178
0.0138
0.0067
0.9415
0.2354
0.7602
0.6963
0.2704
4DGS(AdaIN)
sear steak
1.3544
0.3430
51.6330
2.6115
62.2037
0.0129
0.0078
0.9463
0.2352
0.7114
0.6000
0.2987
4DGS(AdaAttN)
1.1910
0.3996
43.6474
2.0062
63.0785
0.0234
0.0126
0.9021
0.3204
0.7050
0.5153
0.4768
4DStyleGaussian
1.3843
0.3613
42.1204
2.5841
56.5093
0.0131
0.0050
0.9530
0.2722
0.7161
0.6557
0.4819
Style4D(Ours)
1.6818
0.4176
53.3443
2.8820
68.0488
0.0108
0.0066
0.9564
0.3239
0.7014
0.7503
0.2146
Table 1. Quantitative comparisons of our proposed Style4D against state-of-the-art methods on Style4D-Bench.
The final consistency loss combines both terms:
Lconsistency = Llcl + Lcontent.
(4)
Training Objective. We first train a 4D Gaussian Splat-
ting model using multi-view content images Icontent to re-
construct the scene geometry, yielding a static Gaussian se-
quence Gi and a deformation field network Fdef. We then
train a holistic geometry-preserved style transfer module
with the following overall objective:
Ltotal = λconsistencyLconsistency + λstyleLstyle
+ λidLid + λillumLillum + λinsLins,
(5)
where Lconsistency ensures spatio-temporal coherence (see
Eq. 4), and the remaining terms follow conventional styliza-
tion objectives [27]: Lstyle for perceptual style alignment,
Lid for content preservation, Lillum for illumination stabil-
ity, and Lins for intra-channel coherence. See supplemen-
tary material for definitions. After obtaining high-quality
stylized frames, we train a Style Gaussian Representation
supervised by these frames. The optimization objective is:
L =
ˆI −St(I)

1 + Ltv,
(6)
where ˆI is the rendered image, St(I) is the corresponding
stylized frame, and Ltv denotes total variation regulariza-
tion for spatial smoothness.
5. Experiments
Datasets. We evaluate our model on the real-world Neu3D
dataset [29] to benchmark its performance in realistic sce-
narios.
Neu3D comprises six dynamic scenes, each ob-
served by 15–20 static cameras distributed in space. The
dataset features long video sequences (300 frames), with
complex scene dynamics and nontrivial viewpoint varia-
tions. The videos are recorded at a resolution of 1352 ×
1014 pixels, with 300 frames per sequence.
Experiment settings.
We build our implementation of
Style4D based on the publicly available 4DGS codebase
[54]. During training, we adopt the Adam optimizer with
5

<!-- page 6 -->
Figure 2. 4D Stylization Comparison: (a) Original scene image, (b) 4DGS with AdaIN, (c) 4DGS with AdaAttN, (d) 4DStyleGaussian, (e)
Style4D (Ours). Please refer to supplementary material for rendered videos.
hyperparameters following those used in 4DGS. The batch
size is set to 2, and each lightweight per-Gaussian MLP con-
sists of two layers. For each scene, we train the model for
up to 14,000 iterations. Both training and inference are per-
formed on a single NVIDIA A40 GPU with 48GB of mem-
ory. In practice, training a single scene takes approximately
2 hours, with peak GPU memory usage around 10GB.
5.1. Evaluation on Style4D-Bench
Qualitative results.
We compare our method with ex-
isting 4D stylization methods including 4DStyleGaussian,
as well as baseline 4DGS models trained with AdaIN
and AdaAttN stylized images, to assess stylization quality.
As shown in Figure 2, our method demonstrates stronger
temporal consistency compared to 4DGS with AdaIN and
AdaAttN, exhibiting fewer artifacts and blurriness on back-
ground objects.
Moreover, compared to 4DStyleGaus-
sian, our approach significantly enhances stylization ef-
fects while maintaining consistency and effectively preserv-
ing structural details of the original scenes without exces-
sive smoothing. Due to space limitations, we only present
results from fixed test viewpoints here.
Stylization re-
sults from spiral viewpoints are provided in the appendix,
where our method demonstrates stronger consistency and
enhanced stylization quality.
Meanwhile, we also compare our proposed Holistic
Geometry-preserved Style Transfer model (HGST), with
several state-of-the-art 2D stylization methods. As shown
in Figure 3, our method outperforms AdaIN, AdaAttN, and
MCCNet by improving temporal consistency while main-
taining stylization quality, without exhibiting the blocky
pixel artifacts observed in CCPL.
Quantitative Evaluation and Analysis.
As shown in
Table 5, We conduct comprehensive quantitative com-
parisons across three dynamic 4D scenes (cook spinach,
flame salmon 1, and sear steak), evaluating multiple as-
pects of stylization performance, including imaging qual-
ity, aesthetic perception, spatial-temporal consistency, and
fidelity to both content and style.
Across all datasets, Style4D consistently achieves top
performance in most metrics. For imaging quality, we ob-
serve clear improvements in UIQM, Clipiqa+, and Musiq
scores, suggesting that our method preserves structural
clarity and visual quality more effectively than all base-
lines. In particular, Style4D achieves a UIQM of 1.9290
on cook spinach and 1.7529 on flame salmon 1, surpassing
prior methods such as 4DGS(AdaIN) and 4DStyleGaussian
by a large margin.
In terms of aesthetic quality, our method attains the high-
est Qalign and Musiq-PAQ2PIQ scores across all scenes,
6

<!-- page 7 -->
Figure 3. Visualization of HGST stylization method (Ours) compared with other 2D style transfer approaches: (a) Content image; (b)
AdaIN; (c) AdaAttN; (d) CCPL; (e) MCCNet; (f) HGST (Ours). Please refer to supplementary material for rendered videos.
indicating better perceptual stylization aligned with artistic
intent. For example, on flame salmon 1, Style4D yields a
Qalign of 3.6030 and a Musiq-PAQ2PIQ of 63.5178, out-
performing the second-best baseline by significant margins.
Regarding spatial-temporal consistency,
our model
shows robust performance with the lowest DISTS and Warp
Loss values, confirming its ability to produce temporally
stable and coherent results. On sear steak, Style4D reduces
Dists to 0.0108 and Warp Loss to 0.0066, improving both
perceptual smoothness and geometric coherence.
Finally, Style4D also excels in content and style con-
sistency. It achieves the best DINO scores, CKDN accu-
racy, and LPIPS/SSIM metrics across scenes. These results
demonstrate our model’s ability to retain scene semantics
while applying stylization, balancing content preservation
and stylized appearance.
For instance, on cook spinach,
it attains a SSIM of 0.7771 and a content LPIPS of
0.1834—clearly outperforming other methods.
Tables 2 and 3 present the results of our user study. It
can be observed that our method is consistently preferred
in terms of both image quality and overall video quality,
Method
Frame
Stylization Quality
Image Quality
4DGS(AdaIN)
11.76%
2.94%
4DGS(AdattN)
14.70%
8.82%
4DStyleGaussian
11.76%
17.64%
Style4D(Ours)
61.76%
70.58%
Table 2. Single-frame image performance: results of user study
voting
Method
Video
Stylization Quality
Video Quality
Spatiotemporal Consistency
4DStyleGaussian
23.52%
30.12%
44.11%
Style4D(Ours)
76.47%
69.87%
55.88%
4DGS(AdaIN)
20.58%
11.76%
14.70%
Style4D(Ours)
79.41%
88.23%
85.29%
4DGS(AdaAttN)
14.70%
17.64%
14.70%
Style4D(Ours)
85.29%
82.35%
85.29%
Table 3. User study results for video performance evaluation
while simultaneously maintaining strong stylization effects.
These findings align well with the quantitative results re-
7

<!-- page 8 -->
ported in Table 5.
These results collectively confirm that Style4D delivers a
strong balance of style fidelity, spatial-temporal coherence,
and content preservation, establishing new state-of-the-art
performance in 4D stylization.
Further Analyses and Ablation Studies.
We conduct
comprehensive ablation studies to evaluate the contribution
of each key component in our Style4D framework, validat-
ing the necessity and effectiveness of our design. Please
refer to the supplementary material for detail.
6. Conclusion
We present Style4D-Bench, the first benchmark for 4D styl-
ization, featuring: 1) a strong baseline method, 2) a unified
evaluation protocol covering spatial fidelity, temporal co-
herence, and multi-view consistency, and 3) a curated set of
high-resolution dynamic scenes. To establish the baseline,
we propose Style4D, a novel 4D stylization framework that
combines reliable scene representation, per-Gaussian MLPs
for appearance control, and a geometry-preserved styliza-
tion module for spatio-temporal consistency.
Extensive
experiments demonstrate that Style4D outperforms exist-
ing methods, delivering high-quality stylization with stable
dynamics and coherent multi-view rendering. We expect
Style4D-Bench to promote future research in 4D stylized
scene synthesis. In future work, we plan to further improve
the quality of stylization and broader user-controllable style
manipulation for interactive applications.
Limitation.
Although Style4D achieves high-quality
stylization with spatial-temporal consistency, it still faces
limitations. The training process can be time-consuming
due to the multi-stage design and per-Gaussian MLPs.
Additionally, the current framework focuses on a fixed
style per scene; supporting rapid style switching or region-
specific stylization remains an open challenge.
Acknowledgement
The computational resources are supported by SongShan
Lake HPC Center (SSL-HPC) in Great Bay University. This
work was also supported by Guangdong Research Team
for Communication and Sensing Integrated with Intelligent
Computing (Project No. 2024KCXTD047).
References
[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 5855–5864,
2021. 2
[2] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 130–141, 2023. 2
[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers. In Pro-
ceedings of the IEEE/CVF international conference on com-
puter vision, pages 9650–9660, 2021. 3
[4] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano,
Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J
Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient
geometry-aware 3d generative adversarial networks. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 16123–16133, 2022. 2
[5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In European con-
ference on computer vision, pages 333–350. Springer, 2022.
2
[6] Dongdong Chen, Jing Liao, Lu Yuan, Nenghai Yu, and Gang
Hua. Coherent online video style transfer. In Proceedings
of the IEEE International Conference on Computer Vision,
pages 1105–1114, 2017. 4
[7] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21476–21485, 2024. 2
[8] Yaosen Chen, Qi Yuan, Zhiqiang Li, Yuegen Liu, Wei Wang,
Chaoping Xie, Xuming Wen, and Qien Yu. Upst-nerf: Uni-
versal photorealistic style transfer of neural radiance fields
for 3d scene. IEEE Transactions on Visualization and Com-
puter Graphics, 2024. 2
[9] Pei-Ze Chiang, Meng-Shiun Tsai, Hung-Yu Tseng, Wei-
Sheng Lai, and Wei-Chen Chiu. Stylizing 3d scene via im-
plicit representation and hypernetwork. In Proceedings of the
IEEE/CVF winter conference on applications of computer vi-
sion, pages 1475–1484, 2022. 1, 2
[10] Yingying Deng, Fan Tang, Weiming Dong, Haibin Huang,
Chongyang Ma, and Changsheng Xu. Arbitrary video style
transfer via multi-channel correlation. In Proceedings of the
AAAI conference on artificial intelligence, pages 1210–1217,
2021. 1, 2
[11] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli.
Image quality assessment: Unifying structure and texture
similarity. CoRR, abs/2004.07728, 2020. 3
[12] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
towards efficient novel view synthesis for dynamic scenes.
In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,
2024. 1, 2
[13] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural vox-
els. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9,
2022. 2
[14] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan,
Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo.
8

<!-- page 9 -->
Ccedit: Creative and controllable video editing via diffu-
sion models. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6712–
6722, 2024. 4
[15] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5501–5510, 2022. 2
[16] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 12479–12488, 2023. 2
[17] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Im-
age style transfer using convolutional neural networks. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2414–2423, 2016. 1, 2
[18] Shuyang Gu, Congliang Chen, Jing Liao, and Lu Yuan. Ar-
bitrary style transfer with deep feature reshuffle. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 8222–8231, 2018. 1, 2
[19] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander
Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Edit-
ing 3d scenes with instructions.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 19740–19750, 2023. 2
[20] Vlad Hosu, Hanhe Lin, Tamas Sziranyi, and Dietmar Saupe.
Koniq-10k: An ecologically valid database for deep learning
of blind image quality assessment. IEEE Transactions on
Image Processing, 29:4041–4056, 2020. 3
[21] Xun Huang and Serge Belongie. Arbitrary style transfer in
real-time with adaptive instance normalization. In Proceed-
ings of the IEEE international conference on computer vi-
sion, pages 1501–1510, 2017. 1, 2, 14
[22] Yi-Hua Huang, Yue He, Yu-Jie Yuan, Yu-Kun Lai, and Lin
Gao. Stylizednerf: consistent 3d scene stylization as styl-
ized nerf via 2d-3d mutual learning.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18342–18352, 2022. 2
[23] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M
Rehg, and Pinar Yanardag. Rave: Randomized noise shuf-
fling for fast and consistent video editing with diffusion mod-
els. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 6507–6516,
2024. 4
[24] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and
Feng Yang. Musiq: Multi-scale image quality transformer.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 5148–5157, 2021. 3
[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[26] Nicholas
Kolkin,
Jason
Salavon,
and
Gregory
Shakhnarovich. Style transfer by relaxed optimal transport
and self-similarity. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
10051–10060, 2019. 1, 2
[27] Xiaoyu Kong, Yingying Deng, Fan Tang, Weiming Dong,
Chongyang Ma, Yongyong Chen, Zhenyu He, and Chang-
sheng Xu. Exploring the temporal consistency of arbitrary
style transfer: A channelwise perspective.
IEEE Trans-
actions on Neural Networks and Learning Systems, 35(6):
8482–8496, 2024. 4, 5
[28] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman,
Ersin Yumer, and Ming-Hsuan Yang. Learning blind video
temporal consistency. In Proceedings of the European con-
ference on computer vision (ECCV), pages 170–185, 2018.
1
[29] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 5521–5531, 2022. 5, 12
[30] Wanlin Liang, Hongbin Xu, Weitao Chen, Feng Xiao,
and Wenxiong Kang.
4dstylegaussian:
Zero-shot 4d
style transfer with gaussian splatting.
arXiv preprint
arXiv:2410.10412, 2024. 2
[31] Jing Liao, Yuan Yao, Lu Yuan, Gang Hua, and Sing Bing
Kang. Visual attribute transfer through deep image analogy.
arXiv preprint arXiv:1705.01088, 2017. 1
[32] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
Computer vision–ECCV 2014: 13th European conference,
zurich, Switzerland, September 6-12, 2014, proceedings,
part v 13, pages 740–755. Springer, 2014. 12
[33] Kunhao Liu, Fangneng Zhan, Yiwen Chen, Jiahui Zhang,
Yingchen Yu, Abdulmotaleb El Saddik, Shijian Lu, and
Eric P Xing. Stylerf: Zero-shot 3d style transfer of neural
radiance fields. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8338–
8348, 2023. 2
[34] Kunhao Liu, Fangneng Zhan, Muyu Xu, Christian Theobalt,
Ling Shao, and Shijian Lu. Stylegaussian: Instant 3d style
transfer with gaussian splatting. In SIGGRAPH Asia 2024
Technical Communications, pages 1–4. 2024. 1, 2
[35] Shiguang Liu and Ting Zhu. Structure-guided arbitrary style
transfer for artistic image and video. IEEE Transactions on
Multimedia, 24:1299–1312, 2021. 4
[36] Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling
Wang, Xin Li, Zhengxing Sun, Qian Li, and Errui Ding.
Adaattn: Revisit attention mechanism in arbitrary neural
style transfer. In Proceedings of the IEEE/CVF international
conference on computer vision, pages 6649–6658, 2021. 2,
14
[37] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu
Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Jo-
hannes Kopf, and Jia-Bin Huang. Robust dynamic radiance
fields. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 13–23, 2023. 2
[38] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
9

<!-- page 10 -->
sistent dynamic view synthesis. In 2024 International Con-
ference on 3D Vision (3DV), pages 800–809. IEEE, 2024. 1,
2
[39] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[40] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 2
[41] Thu Nguyen-Phuoc, Feng Liu, and Lei Xiao. Snerf: stylized
neural implicit representations for 3d scenes. arXiv preprint
arXiv:2207.02363, 2022. 1, 2
[42] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 5865–5874, 2021. 2
[43] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228, 2021.
[44] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
10318–10327, 2021. 1, 2
[45] Abhishek Saroha, Mariia Gladkova, Cecilia Curreli, Do-
minik Muhle, Tarun Yenamandra, and Daniel Cremers.
Gaussian splatting in style. In DAGM German Conference
on Pattern Recognition, pages 234–251. Springer, 2024. 1, 2
[46] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural
4d decomposition for high-fidelity dynamic reconstruction
and rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 16632–
16642, 2023. 2
[47] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition. arXiv
preprint arXiv:1409.1556, 2014. 12
[48] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 5459–
5469, 2022. 2
[49] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow.
In Computer Vision–ECCV
2020: 16th European Conference, Glasgow, UK, August 23–
28, 2020, Proceedings, Part II 16, pages 402–419. Springer,
2020. 3, 14
[50] Can Wang, Menglei Chai, Mingming He, Dongdong Chen,
and Jing Liao.
Clip-nerf:
Text-and-image driven ma-
nipulation of neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 3835–3844, 2022. 2
[51] Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He,
Dongdong Chen, and Jing Liao. Nerf-art: Text-driven neural
radiance fields stylization. IEEE Transactions on Visualiza-
tion and Computer Graphics, 2023. 1, 2
[52] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Ex-
ploring clip for assessing the look and feel of images. In
AAAI, 2023. 3
[53] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So
Kweon. Cbam: Convolutional block attention module. In
Computer Vision – ECCV 2018, pages 3–19, Cham, 2018.
Springer International Publishing. 4
[54] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 1,
2, 5, 12
[55] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng
Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang,
Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtai Zhai,
and Weisi Lin.
Q-align:
Teaching lmms for visual
scoring via discrete text-defined levels.
arXiv preprint
arXiv:2312.17090, 2023. Equal Contribution by Wu, Haon-
ing and Zhang, Zicheng. Project Lead by Wu, Haoning. Cor-
responding Authors: Zhai, Guangtai and Lin, Weisi. 3
[56] Zijie Wu, Zhen Zhu, Junping Du, and Xiang Bai. Ccpl: Con-
trastive coherence preserving loss for versatile style transfer.
In European conference on computer vision, pages 189–206.
Springer, 2022. 4
[57] Rui Xu, Wenyue Chen, Jiepeng Wang, Yuan Liu, Peng
Wang, Lin Gao, Shiqing Xin, Taku Komura, Xin Li, and
Wenping Wang. Supergaussians: Enhancing gaussian splat-
ting using primitives with spatially varying colors, 2024. 4
[58] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang.
Real-time photorealistic dynamic scene representation and
rendering with 4d gaussian splatting.
arXiv preprint
arXiv:2310.10642, 2023. 1
[59] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu
Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiao-
han Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video
diffusion models with an expert transformer. arXiv preprint
arXiv:2408.06072, 2024. 4
[60] Taoran Yi, Jiemin Fang, Xinggang Wang, and Wenyu Liu.
Generalizable neural voxels for fast human radiance fields.
arXiv preprint arXiv:2303.15387, 2023. 2
[61] Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Maha-
jan, Deepti Ghadiyaram, and Alan Bovik. From patches to
pictures (paq-2-piq): Mapping the perceptual space of pic-
ture quality. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 3575–3585,
2020. 3
[62] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (TOG),
43(6):1–13, 2024. 4
[63] Dingxi Zhang, Yu-Jie Yuan, Zhuoxun Chen, Fang-Lue
Zhang, Zhenliang He, Shiguang Shan, and Lin Gao. Styl-
10

<!-- page 11 -->
izedgs: Controllable stylization for 3d gaussian splatting.
arXiv preprint arXiv:2404.05220, 2024. 1, 2
[64] Kai Zhang, Nick Kolkin, Sai Bi, Fujun Luan, Zexiang Xu,
Eli Shechtman, and Noah Snavely. Arf: Artistic radiance
fields. In European Conference on Computer Vision, pages
717–733. Springer, 2022. 1, 2
[65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 3
[66] Heliang Zheng, Jianlong Fu, Yanhong Zeng, Zheng-Jun Zha,
and Jiebo Luo.
Learning conditional knowledge distilla-
tion for degraded-reference image quality assessment. ICCV,
2021. 3
[67] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free
dense labels from clip. In European conference on computer
vision, pages 696–712. Springer, 2022. 3
[68] Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and
Guanbin Li. Dreameditor: Text-driven 3d scene editing with
neural fields. In SIGGRAPH Asia 2023 Conference Papers,
pages 1–10, 2023. 2
11

<!-- page 12 -->
A. Implementation and Network Details
4DGS Representation.
We follow the training configura-
tions described in the 4DGS [54] paper for this component.
Stylization experiments are conducted on the real-world
Neu3D [29] dataset, which contains six dynamic scenes.
Each input image has a resolution of 1352 × 1014, and
the reconstruction is performed at the same resolution, with
each video comprising 300 frames. The initial point cloud
for each scene is generated using the official initialization
code provided by 4DGS 1. After initialization, each scene is
downsampled to approximately 3–4k Gaussian points.
Style Gaussian Representation.
Each tiny MLP receives
the time step t and intersection depth as inputs. The network
consists of four hidden units and outputs RGB deltas along
with opacity values (4-dimensional output). The learning
rate is scheduled with lrinit = 0.0001, lrfinal = 0.00001, and
a delay multiplier lrdelay mult = 0.02.
Holistic Geometry-preserved Style Transfer Module.
For this module, we use images from the COCO2014
dataset [32] as content inputs and style images sampled
from WikiArt. The model is evaluated on Neu3D scenes
to demonstrate strong generalization ability. During train-
ing, images are cropped to 256×256 resolution. The overall
loss function is defined as:
Ltotal = λconsistencyLconsistency + λstyleLstyle + λidLid
+λillumLillum + λinsLins,
(7)
where the weights are set as follows: λconsistency = 3,
λstyle = 18, λid = 7, λillum = 10−5, λins = 1.
1) The style perceptual loss Lstyle minimizes the style
differences between the generated image Ics and the style
image Is by comparing the mean and variance of features
extracted from each layer of a pre-trained VGG19. For-
mally, the loss is expressed as:
Lstyle =
X
l
 ∥µl(Ics) −µl(Is)∥2
2
+∥σl(Ics) −σl(Is)∥2
2

,
(8)
where µl(·) and σl(·) denote the mean and standard de-
viation computed over the spatial dimensions of the feature
map at layer l. Given the feature map Fl(I) ∈RCl×Hl×Wl,
these statistics are defined as:
1https://github.com/hustvl/4DGaussians
µl(I) =
1
HlWl
Hl
X
h=1
Wl
X
w=1
Fl(I):,h,w,
σl(I) =
v
u
u
t
1
HlWl
Hl
X
h=1
Wl
X
w=1
(Fl(I):,h,w −µl(I))2,
(9)
where the operations are computed channel-wise.
2) The identity loss Lid helps to preserve the content
structure while maintaining the richness of the style pat-
terns:
Lid = ∥Icc −Ic∥2 + ∥Iss −Is∥2
(10)
where Icc and Iss are the generated results using natural im-
ages and paintings as content and style images, respectively.
3) Illumination Loss Lillum: Illumination loss addresses
the flickering effect caused by illumination variations in
video sequences. It is defined as:
Lillum = ∥G(Ic, Is) −G(Ic + ϵ, Is)∥2
(11)
where G(·) is the generation function, and ϵ ∼N(0, σ2I)
represents random Gaussian noise.
4) Inner Channel Similarity Loss Lins:
This loss
strengthens the consistency of generated features within
each channel, ensuring that there are no disharmonious ar-
eas:
Lins =
C
X
c=1
Innerc,i
(12)
where Innerc,i is the inner similarity defined as:
Innerc,i = arg min
i
h×w
X
j=1

1 −fi · fj
∥f∥2
2

(13)
where f represents the generated features, and h × w is
their resolution.
B. Additional comparative experiments
B.1. Quantitative Results
Metrics. The goal of stylization is to transform object edges
and texture details to match the style of a reference im-
age while preserving the overall structural integrity of the
original image. To evaluate the structural similarity with
the original image, we employ SSIM and LPIPS metrics.
To quantitatively assess the degree of stylization, we use
a style loss computed as follows: features are extracted
from five layers of a pretrained VGG network [47] for both
the stylized image Ics and the reference style image Is.
The style loss is defined as the sum of mean squared er-
rors (MSE) between the Gram matrices of corresponding
feature layers of Ics and Is. This accumulated style loss
12

<!-- page 13 -->
Figure 4. Comparison of stylization results by different methods: (a) Original images (before style transfer), (b) 4DGS with AdaIN, (c)
4DGS with AdaAttN, (d) Style4D(Ours).
Figure 5. Visualization of HGST stylization method (Ours) compared with other 2D style transfer approaches: (a) Content image; (b)
AdaIN; (c) AdaAttN; (d) CCPL; (e) MCCNet; (f) HGST (Ours).
serves as a measure of the stylistic discrepancy between
the generated image and the target style. The Gram ma-
13

<!-- page 14 -->
Method
AdaIN
AdaAttN
CCPL
MCCNet
HGST(Ours)
Warp Loss↓
0.045824
0.031671
0.013301
0.021858
0.021068
Table 4. Comparison of Warp Loss for Video Stylization Results
Across Different Methods.
Method
Dataset
SSIM↑
LPIPS↓
4DGS(AdaIN)
sear steak
0.6000
0.2987
4DGS(AdaAttN)
0.5153
0.4768
4DStyleGaussian
0.6557
0.4819
Style4D(Ours)
0.7503
0.2146
4DGS(AdaIN)
cook spinach
0.4763
0.4898
4DGS(AdaAttN)
0.6444
0.2841
4DStyleGaussian
0.7646
0.2159
Style4D(Ours)
0.7771
0.1834
4DGS(AdaIN)
flame salmon 1
0.5201
0.4030
4DGS(AdaAttN)
0.6054
0.2944
4DStyleGaussian
0.5081
0.6051
Style4D(Ours)
0.6963
0.2704
Table 5. Quantitative comparisons of Style4D against other meth-
ods.
trix is computed as follows: Given the feature map at a
certain layer F ∈RC×H×W , where C denotes the num-
ber of channels, and H and W denote the height and width
of the feature map, respectively. We first reshape F into a
matrix F ′ ∈RC×(H×W ), then compute the Gram matrix
G ∈RC×C as
Gij =
1
C × H × W
H
X
k=1
W
X
l=1
Fi,k,l × Fj,k,l,
(14)
where Gij represents the correlation between the i-th and j-
th channel features, capturing the style information encoded
in that layer.
Table 5 presents a quantitative comparison between our
method and other stylization approaches on rendered styl-
ized videos from novel test viewpoints. The results demon-
strate that our method effectively preserves the underlying
scene structure while achieving strong stylization perfor-
mance.
Since both our method and 4DStyleGaussian exhibit
strong visual consistency, we employ style loss as a quan-
titative metric to better evaluate stylization quality. Table 6
presents the style loss between the stylized results of our
method, 4DStyleGaussian, and the reference style image.
The results indicate that our method achieves closer adher-
ence to the reference style while effectively preserving the
original scene structure.
Holistic Geometry-preserved Style Transfer(HGST)
To quantitatively evaluate the temporal consistency of dif-
ferent stylization methods, we employ the warp loss metric.
We estimate the optical flow between consecutive frames
Method
Dataset
Style Loss↓
4DStyleGaussian
sear steak
0.006816
Style4D(Ours)
0.005687
4DStyleGaussian
cook spinach
0.008329
Style4D(Ours)
0.006123
4DStyleGaussian
flame salmon 1
0.007387
Style4D(Ours)
0.006364
Table 6. Quantitative Comparison of Stylization Performance.
using the RAFT method [49]. Let It and It+1 denote image
frames at time steps t and t+1, respectively, and Ft→t+1 be
the optical flow field mapping pixel coordinates from frame
t to frame t + 1. Using this flow, the subsequent frame
It+1 is spatially warped to produce a transformed image
ˆIt+1 aligned with It:
ˆIt+1(x) = It+1
 x + Ft→t+1(x)

,
(15)
where x denotes pixel coordinates. The warp loss be-
tween two frames is then defined as the pixel-wise differ-
ence between ˆIt+1 and It:
Lt
warp =
1
HW
X
x
ˆIt+1(x) −It(x)
 ,
(16)
where H and W are the height and width of the image,
respectively. The overall warp loss for a video sequence is
computed as the average over all consecutive frame pairs:
Lwarp =
1
T −1
T −1
X
t=1
Lt
warp,
(17)
where T is the total number of frames in the video. This
metric reflects the temporal coherence across video frames.
Table 4 presents a comparison of temporal consistency
among our proposed HGST method and other stylization
approaches in the video stylization task.
Our method
demonstrates superior consistency compared to most base-
lines, slightly trailing behind CCPL. However, CCPL
achieves this consistency improvement at the cost of sac-
rificing local detail fidelity.
B.2. Qualitative results
Figure 4 presents qualitative comparisons of rendered im-
ages from novel test viewpoints between our method and
several baselines. Although AdaIN [21] and AdaAttN [36]
achieve stronger stylization effects, they substantially dis-
tort the original scene structure and introduce significant ar-
tifacts and noise. In contrast, our method (d) achieves a high
degree of style transfer while faithfully preserving the scene
structure and fine facial details.
Holistic Geometry-preserved Style Transfer(HGST)
Due to space constraints in the main text, we provide here
14

<!-- page 15 -->
Figure 6. Ablation study: (a) Original scene images, (b) Stylized images, (c) 4DGS trained with stylized images, (d) 4DGS trained under
our two-stage scheme without lightweight MLPs for gaussians, (e) Style4D (Ours). Our method better preserves fine-grained details such
as facial structures and accessories while minimizing the artifacts.
Figure 7. Ablation study with moving views: (a) 4DGS trained with stylized images, (b) 4DGS trained under our two-stage scheme without
lightweight MLPs for gaussians, (c) Style4D (Ours). Our method achieves a better multi-view consistencys.
a detailed comparison of our 2D stylization method HGST
against other state-of-the-art 2D stylization approaches, as
well as a corrected visualization of our method’s results. As
shown in Figure 5, while AdaIN (b) and AdaAttN (c) ex-
hibit strong stylization effects, they suffer from poor tem-
poral consistency in video sequences.
MCCNet (e) in-
adequately preserves local detail features, resulting in no-
ticeable background flickering and blurred shadows be-
tween subjects and backgrounds. Although CCPL main-
tains strong temporal consistency, it compromises local de-
tail, leading to heavily blurred facial regions and numer-
ous hole-like artifacts in the background. In contrast, our
method effectively balances stylization quality with preser-
vation of original image structure and temporal coherence.
15

<!-- page 16 -->
Method
Dataset
flame steak
sear steak
flame salmon 1
coffee martini
cook spinach
cut roasted beef
4DGS
32.02
31.75
28.69
28.68
32.06
32.55
ours
32.85
32.60
28.89
28.87
32.28
32.81
Table 7.
Comparison of PSNR between Our Framework and
4DGS on Various Datasets.
Figure 8. Ablation study of HGST.
C. Ablation Studies
We conduct comprehensive ablation studies to evaluate the
contribution of each key component in our Style4D frame-
work, validating the necessity and effectiveness of our de-
sign.
Compared with Direct 4DGS Training on Stylized
Images. We first investigate the baseline of directly training
4D Gaussian Splatting (4DGS) on stylized images, without
decoupling geometry reconstruction from stylization and
lightweight MLPs for stylization. As illustrated in the row
c in Figure 6 , this baseline produces substantially degraded
results characterized by temporal flickering and spatial arti-
facts. Such outcomes confirm the necessity of our proposed
two-stage training pipeline, which disentangles geometry
learning (coarse and fine stages) from stylization, and en-
hances the representation ability for each gaussian, thereby
ensuring a stable 4D scene structure and enabling flexible
and high-quality style transfer.
Effectiveness of Per-Gaussian MLPs. To assess the im-
pact of the per-Gaussian MLPs introduced in the style stage,
we compare our full model against a variant that directly
optimizes Gaussian parameters without the use of MLPs.
As shown in the row d in Figure 6, removing the MLPs re-
sults in noticeable degradation of overall stylization quality
and a loss of fine-grained texture details. The per-Gaussian
MLPs facilitate spatial-temporal modulation of appearance,
providing precise control over style evolution across both
time and viewpoint, which is critical for achieving smooth
temporal transitions and multi-view consistency.
To further validate the effectiveness of our proposed style
Gaussian representation, we conduct training and evalua-
tion on the original datasets and compare with 4DGS. As
shown in Table 7 and Figure 9, 10, our Gaussian representa-
tion outperforms 4DGS, significantly reducing artifacts and
blurriness in the synthesized novel views.
Effectiveness of Geometry-guided Initialization. We
further evaluate if the learned geometry from earlier stages
could help achieving a better view-consistent stylization re-
sult by comparing training 4DGS with stylized images with
and without geometry-informed priors. Figure 6 demon-
strates that training stylized 4DGS from scratch will lead
to inferior visual fidelity and structural artifacts, confirming
that geometry-guided initialization provides essential struc-
tural priors.
Effectiveness of Holistic Geometry-preserved Style
Transfer. As illustrated in Figure 8, our proposed HGST
model significantly improves spatial-temporal consistency
compared to the original MCCNet while maintaining styl-
ization quality. Flickering artifacts in the video are substan-
tially reduced.
The full ablation study of the proposed HGST module is
illustrated in Figure 11. Although (a) MCCNet, our base-
line, preserves spatiotemporal consistency during styliza-
tion, it still suffers from severe flickering and exhibits poor
detail consistency, especially on high-resolution frames. To
evaluate the effectiveness of our proposed local consistency
loss (LCL) and content loss, we conducted the following
experiments: (b) incorporates the CCPL contrastive loss
into MCCNet. While it introduces some structural consis-
tency, it also leads to noticeable hollow artifacts around the
face and adjacent regions. (c) integrates the content loss
into MCCNet, which improves overall consistency, but re-
sults in blurring of physical shapes and boundaries.
(d)
adds our LCL loss on top of MCCNet, which enhances
local consistency and alleviates hollow artifacts to some
extent; however, some artifacts remain, and flickering is
still apparent in the background curtain region. Finally, (e)
presents our complete Holistic Geometry-preserved Style
Transfer (HGST) module. Compared to all prior variants,
our method achieves significantly better global and local
consistency, while substantially reducing temporal flicker-
ing artifacts.
16

<!-- page 17 -->
Figure 9. Reconstruction Results under Moving Viewpoints on flame steak, coffee martini, and flame salmon 1 Datasets: (a) 4DGS (b)
Ours.
17

<!-- page 18 -->
Figure 10. Reconstruction Results under Moving Viewpoints on sear steak, cut roasted beef, and cook spinach Datasets: (a) 4DGS (b)
Ours.
18

<!-- page 19 -->
Figure 11. Ablation Study of the Holistic Geometry-preserved Style Transfer Module: (a) MCCNet (b) MCCNet with CCPL Loss (c)
MCCNet with Content Loss (d) MCCNet with LCL Loss (e) HGST (Ours).
19
