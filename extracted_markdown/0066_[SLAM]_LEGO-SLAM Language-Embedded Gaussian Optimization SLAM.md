<!-- page 1 -->
LEGO-SLAM: Language-Embedded Gaussian
Optimization SLAM
Sibaek Lee , Seongbo Ha , Kyeongsu Kang , Joonyeol Choi , Seungjun
Tak , and Hyeonwoo Yu
Sungkyunkwan University, South Korea
{lmjlss,sobo3607,thithin0821,joonyeol99,tmdwns8840,hwyu}@g.skku.edu
Abstract. Recent advances in 3D Gaussian Splatting (3DGS) have en-
abled Simultaneous Localization and Mapping (SLAM) systems to build
photorealistic maps. However, these maps lack the open-vocabulary se-
mantic understanding required for advanced robotic interaction. Inte-
grating language features into SLAM remains a significant challenge, as
storing high-dimensional features demands excessive memory and ren-
dering overhead, while existing methods with static models lack adapt-
ability for novel environments. To address these limitations, we propose
LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the
first framework to achieve real-time, open-vocabulary mapping within a
3DGS-based SLAM system. At the core of our method is a scene-adaptive
encoder-decoder that distills high-dimensional language embeddings into
a compact 16-dimensional feature space. This design reduces the memory
per Gaussian and accelerates rendering, enabling real-time performance.
Unlike static approaches, our encoder adapts online to unseen scenes.
These compact features also enable a language-guided pruning strat-
egy that identifies semantic redundancy, reducing the map’s Gaussian
count by over 60% while maintaining rendering quality. Furthermore, we
introduce a language-based loop detection approach that reuses these
mapping features, eliminating the need for a separate detection model.
Extensive experiments demonstrate that LEGO-SLAM achieves compet-
itive mapping quality and tracking accuracy, all while providing open-
vocabulary capabilities at 15 FPS. Our project page is at: https://lab-
of-ai-and-robotics.github.io/LEGO-SLAM/
Keywords: Open-Vocabulary SLAM · 3D Gaussian Splatting
1
Introduction
While the foundational objective of Simultaneous Localization and Mapping
(SLAM) has long been to jointly construct a map and estimate an agent’s lo-
cation, technological advancements have elevated the ambition towards generat-
ing rich and scalable environmental representations. In response, recent break-
throughs in 3D reconstruction have enhanced the mapping component of SLAM,
with Neural Radiance Fields (NeRF) [23] and 3D Gaussian Splatting (3DGS) [12]
emerging as prominent solutions for creating photo-realistic maps [37]. NeRF
arXiv:2511.16144v1  [cs.CV]  20 Nov 2025

<!-- page 2 -->
2
S. Lee et al.
Fig. 1: LEGO-SLAM: real-time, open-vocabulary 3DGS-SLAM. (Left) A
large-scale real-world scene reconstructed by our system, with colored boxes highlight-
ing specific regions. (Middle) Relevancy maps showing the 3D localization for corre-
sponding text queries. (Right) Graphs on the ScanNet dataset show LEGO-SLAM
operates at 15 FPS while maintaining competitive performance.
provided memory-efficient implicit mapping but failed to meet the real-time de-
mands of SLAM due to slow rendering. This limitation was overcome by 3DGS,
which delivers high-fidelity mapping at real-time. However, such photorealistic
maps lack the semantic understanding required for enabling embodied AI to
perform diverse downstream tasks [10, 15, 32, 33]. The evolution of these meth-
ods has progressed beyond simple RGB representation to semantic mapping,
assigning categorical labels to the reconstructed geometry [17,18,29,46,47]. Yet,
these semantic approaches have relied on a closed-set paradigm, limiting them to
predefined object labels. While the 3D reconstruction field is rapidly advancing
towards open-vocabulary representations that can interpret arbitrary language
queries [13,19,26,44], integrating these capabilities into SLAM remains a chal-
lenge, due to critical limitations in real-time performance and model adaptability.
To address these limitations, we propose LEGO-SLAM (Language-Embedded
Gaussian Optimization SLAM), the first framework that achieves real-time,
open-vocabulary mapping within a Gaussian Splatting-based SLAM system.
Approaches with open-vocabulary 3D reconstruction that store high-dimensional
language features on each Gaussian are incompatible with the demands of SLAM.
They suffer from rendering latency and high memory overhead, making them im-
practical for real-time applications [13,19,44]. Furthermore, methods that rely on
pretrained models to extract fixed, low-dimensional features lack the adaptability
required for SLAM systems, which must continuously learn from new environ-
ments [26]. Our core contribution overcomes these limitations by introducing a
scene-adaptive autoencoder that distills high-dimensional language embeddings
into a compact 16-dimensional feature space. This encoder trains in real-time
to map visual information to a compact feature space, enabling highly efficient
open-vocabulary 3D object localization. This is because instead of decoding the
entire map back to a high-dimensional space for querying, we leverage the en-
coder to project the text query down into the map’s compact space for direct and

<!-- page 3 -->
Abbreviated paper title
3
fast comparison. Specifically, we reduce the language feature dimension to 16, a
choice that provides the optimal balance, as validated in Sec. 4.3, between mini-
mizing memory per Gaussian and accelerating rendering. However, this compact
16-dimensional representation introduces an initialization challenge, as these ab-
stract features cannot be derived from input RGB-D. We solve this by leveraging
a pretrained encoder as a powerful prior, ensuring the rapid feature convergence
essential for real-time SLAM.
Furthermore, the benefits of this compact 16-dimensional representation ex-
tend beyond mapping, allowing these features to be efficiently leveraged for other
SLAM components. For instance, they enable an effective language-guided prun-
ing strategy. By operating on the compact features, our system identifies seman-
tically redundant Gaussians with negligible computational overhead, addressing
the trade-offs of purely geometric pruning [7, 11, 12, 22, 45] without sacrificing
important detail. In addition, these same mapping features are reused for loop
detection. This integrated approach, which performs place recognition directly
on our compact features, eliminates the need for separate, computationally ex-
pensive detection models [1, 24]. This ensures robust long-term drift correction
and maintains tracking accuracy competitive with leading systems [20, 45], all
without incurring additional overhead. We demonstrate the effectiveness of our
framework on standard benchmarks including Replica [35], TUM-RGBD [36]
and ScanNet [4], and our main contributions are as follows:
– We propose LEGO-SLAM, the real-time, open-vocabulary 3DGS SLAM
framework for simultaneous high-fidelity mapping and tracking.
– A lightweight, scene-adaptive autoencoder that learns a compact 16-dim
language representation, using a pretrained prior for rapid convergence, en-
abling real-time low-overhead mapping and efficient 3D object localization
directly in the compact space.
– A semantic-guided Gaussian pruning strategy enhancing memory efficiency
by removing semantically redundant Gaussians without sacrificing structural
detail.
– An efficient, language-based loop detection reusing mapping features, avoid-
ing a separate model while ensuring robust long-term tracking accuracy.
2
Related work
Neural Rendering SLAM. Recent advancements in differentiable rendering
have given rise to a new paradigm in SLAM, often termed Neural Render-
ing SLAM. These methods move beyond traditional geometric representations
to build high-fidelity, photorealistic maps by optimizing a scene representation
against input images [37]. Early works in this domain integrated NeRF [23] into
the SLAM pipeline, demonstrating the ability to construct continuous implicit
map representations from RGB images [9, 20, 30, 42, 48]. While they produced
maps of unprecedented visual quality, their high computational cost for per-ray
rendering made them unsuitable for live SLAM operations. To address this bot-
tleneck, 3DGS [12] marked a significant breakthrough. Its explicit representation

<!-- page 4 -->
4
S. Lee et al.
enables real-time rendering, leading to its rapid adoption in SLAM [11,22,38,41].
Efforts have also aimed to improve long-term robustness in large-scale environ-
ments by incorporating loop closure into these systems [45] While visually im-
pressive, these photorealistic maps lack the semantic context required for mean-
ingful robotic interaction. Consequently, the ambition of these systems has nat-
urally evolved beyond photorealism towards semantic understanding. H0owever,
these efforts have predominantly operated within a closed-set paradigm, capable
of recognizing only a predefined list of object categories [17,18,29,46,47].
Open Vocabulary Scene Representation. The transition to an interactive
open-vocabulary map remains a significant hurdle for SLAM. Embedding the
required high-dimensional language features introduces substantial memory and
computational overhead [13,19,44]. Furthermore, reliance on static pre-trained
models fundamentally conflict with the need for SLAM systems to continuously
adapt to novel environments [26]. The foundation for this open-vocabulary shift
was laid by powerful vision-language models, most notably CLIP [27], which
learned a shared embedding space for images and text. Concurrently, power-
ful foundation models like DINO [2, 25] and the Segment Anything Model [14]
emerged, demonstrating remarkable capabilities in tasks such as segmentation
and tracking [3,21,28,39], though they lack inherent open-vocabulary semantic
recognition. Subsequent methods unlocked the ability to extract dense, per-pixel
language features [6, 16], enabling a new class of open-vocabulary downstream
tasks. These pixel-level language descriptors, combined with CLIP’s global un-
derstanding, have enabled lifting these features into 3D, leading to impressive
open-vocabulary 3D reconstruction methods. However, they were not designed
for the strict real-time constraints of SLAM, as distilling these high-dimensional
features per-frame remains a significant bottleneck [13,19,26,44]. In contrast, our
work aims to bridge this gap by learning a compact and scene-adaptive language
representation directly within a real-time SLAM framework.
3
Method
Our work, LEGO-SLAM, is the real-time system that constructs a 3D Gaussian
Splatting map enriched with open-vocabulary language features. As illustrated
in Fig. 2, our system integrates a Tracking module (Sec. 3.1) and the Mapping
module (Sec. 3.2), which is responsible for optimizing the language-embedded
map and enables downstream tasks such as 3D Object Localization. These are
complemented by a Language Pruning strategy for efficient map management
(Sec. 3.3) and a Loop Detection mechanism for long-term consistency (Sec. 3.4).
3.1
Tracking
The tracking module estimates the camera pose Tk ∈SE(3) for each incoming
RGB-D frame Ik. Instead of costly photometric alignment, we adopt a direct
3D-to-3D geometric approach using the G-ICP algorithm [31]. A key advantage

<!-- page 5 -->
Abbreviated paper title
5
Fig. 2: System Overview. LEGO-SLAM architecture, where the Tracking module
estimates pose and the Mapping module optimizes the 3D Gaussian Map via language
distillation. This map is refined by Language Pruning and Loop Detection, enabling
3D Object Localization.
is that we directly track against our unified 3D Gaussian map, eliminating the
need for a separate tracking map as seen in decoupled systems [24].
Pose Estimation. We estimate the camera pose Tk using G-ICP refinement
[31], initializing the current pose with the previous optimized pose Tk−1. Source
Gaussians are generated from the current depth image, while target Gaussians
are sampled directly from our language-embedded 3DGS map. A key efficiency of
our system is that the source covariances computed during tracking are reused to
initialize new Gaussians in the mapping stage, avoiding redundant computations.
Keyframe Selection. After optimizing the pose, we select a frame as a keyframe
if the proportion of inlier correspondences from the G-ICP alignment drops be-
low a predefined threshold. This ensures new keyframes are added only when
significant new information is observed.
3.2
Mapping
Language-Embedded 3D Gaussians. To enable open-vocabulary capabil-
ities, we extend the standard 3D Gaussian representation. Each Gaussian is
defined by a set of optimizable attributes Θ = {p, q, s, α, c, f}, which includes
position p, rotation q, scale s, opacity α, SH color coefficients c, and crucially,
our compact language feature f ∈R16. Following the standard 3DGS pipeline,
these attributes are rendered to produce both an RGB image ˆI and a compact
16-dimensional feature map Frender.
Gaussian Initialization. When a new keyframe is received, we generate new
3D Gaussians from its depth data, deriving their geometric attributes of position,
scale, and rotation from the 3D point cloud. Crucially, to maximize efficiency,
we reuse the covariance matrices computed during the G-ICP tracking stage
as the initial scale and rotation for these new Gaussians, avoiding redundant
calculations.

<!-- page 6 -->
6
S. Lee et al.
While geometric properties are directly initialized, the abstract 16-dimensional
language features present a unique challenge. To address this, we leverage our
scene-adaptive encoder. We first use a pretrained 2D foundation model to ex-
tract a high-dimensional feature map from the keyframe’s RGB, which is then
passed through our encoder to produce a strong prior for the new 16D features.
This encoder-based initialization is critical for achieving the rapid feature con-
vergence essential for real-time SLAM, as the encoder continuously adapts to
the scene, providing a scene-specific prior for new Gaussians.
Map Optimization via Feature Distillation. Following initialization, we
optimize the map over several iterations. In each iteration, we randomly select
a keyframe from the active window to render its view and minimize a joint loss
function Ltotal:
  \lab e l {e q :total_loss}  \mathcal {L}_{\text {total}} = \mathcal {L}_{\text {rgb}} + w_{\text {depth}}\mathcal {L}_{\text {depth}} + w_{\text {feat}}\mathcal {L}_{\text {feat}} 
(1)
where wdepth and wfeat are weighting coefficients. The photometric loss Lrgb
combines an L1 and a D-SSIM term between the rendered ˆI and ground-truth
I. The geometric loss Ldepth is an L1 loss between the rendered ˆD and ground-
truth D. The feature distillation loss Lfeat is the core of our language feature
learning:
  \la b el {eq:feat_loss } \mathcal {L}_{\text {feat}} = \mathcal {L}_{\text {L1}}(D_{\theta }(F_{\text {render}}), F_{gt}) 
(2)
To compute this, we render the compact 16-dimensional feature map Frender from
our 3D Gaussians. This map is passed through our lightweight convolutional
decoder Dθ to upsample it to the high dimension, such as 512D, of the frozen
guidance model’s features Fgt for L1 comparison. This distillation process allows
our 3D Gaussians to learn a compact and scene-adaptive language representation
while maintaining real-time performance.
Scene-Adaptive Encoder Optimization. We found that jointly optimizing
the encoder Eϕ with the map introduces training instability, especially if adap-
tation starts before the Gaussian features f converge. Therefore, we adopt a
decoupled strategy, freezing Eϕ during the initial map optimization to allow f
to stabilize first. After feature convergence, we periodically adapt Eϕ by freezing
the map and decoder Dθ and minimizing only Lenc:
  \l a bel {eq:enc_ loss} \mathcal {L}_{\text {enc}} = \mathcal {L}_{\text {L1}}(E_{\phi }(F_{gt}), F_{\text {render}}) 
(3)
This trains Eϕ to map the high-dimensional Fgt to the map’s learned Frender,
ensuring stable convergence. This adaptive encoder is critical for real-time 3D
object localization, as shown in Sec. 4.3.
3.3
Language-Guided Gaussian Pruning
To maintain map efficiency and compactness, we periodically prune Gaussians.
Conventional pruning in 3DGS-based systems [7, 11, 12, 45] relies on geometric
heuristics, such as low opacity or large scale. However, these methods strug-
gle to distinguish redundant Gaussians, and aggressive geometric pruning often
degrades map quality by removing structurally important primitives.

<!-- page 7 -->
Abbreviated paper title
7
To overcome this, we introduce a language-guided pruning strategy that aug-
ments the geometric criteria with a semantic one, identifying redundant Gaus-
sians that are both spatially proximate and semantically similar. This check is
highly efficient as it operates directly on the compact 16-dimensional language
features f already stored on each Gaussian. This eliminates the need for addi-
tional feature extraction, in contrast to other semantic pruning methods that
require separate models or significant overhead [8,40,43].
Specifically, our method identifies redundant primitives by evaluating each
Gaussian Gi against its local neighborhood. We first find its K-nearest neighbors
in 3D space. Then, for each neighbor Gj, we evaluate its redundancy based on
both its Euclidean distance d(pi, pj) and the cosine similarity of their language
features, sim(fi, fj). A neighbor Gj is marked as redundant if it satisfies the
following condition:
  \lab el { eq:pru n ing_cond iti o n} (d(\mathbf {p}_i, \mathbf {p}_j) < \tau _{dist}) \land (\text {sim}(\mathbf {f}_i, \mathbf {f}_j) > \tau _{sim}) 
(4)
where τdist and τsim are predefined distance and similarity thresholds. The final
set of Gaussians to be pruned is the union of those identified by our language-
based redundancy check and those filtered by the traditional geometric criteria.
3.4
Language-Based Loop Closure
To correct the inevitable drift accumulated during long-term operation, we in-
corporate a language-based loop closure module. Our approach is designed for
high efficiency by reusing the features already computed for mapping, avoiding
separate loop detection models [1,24].
Language-Based Candidate Detection. Our approach performs place recog-
nition by representing each keyframe with a compact semantic signature. This
uses a language codebook, generated offline by applying k-means clustering to
high-dimensional language features. This vocabulary allows us to create a nor-
malized histogram for each new keyframe by assigning its pixel features to the
closest clusters in the codebook. To find potential loop candidates, we compute
the cosine similarity between the current keyframe’s histogram and those of spa-
tially proximate past keyframes. Keyframes with a similarity above a threshold
τsim are selected as potential loop candidates.
Geometric Verification and Global Optimization. Each loop candidate
undergoes a rigorous geometric verification using G-ICP against a local submap.
If a match is confirmed based on overlap and RMSE thresholds, its relative pose
transformation Tij is added as a constraint to a pose graph. We then employ
GSTAM [5] to optimize the entire trajectory for global consistency, propagating
the corrected poses back to our 3D Gaussian map.
4
Experiments
4.1
Experiments Setup
Datasets and Metrics. We evaluate our framework on three datasets, includ-
ing the synthetic Replica dataset [35], and the real-world TUM-RGBD [36] and

<!-- page 8 -->
8
S. Lee et al.
ScanNet [4] datasets. For tables that report a single metric per dataset, the value
represents the average performance across all evaluated scenes. Tracking Accu-
racy is evaluated using Absolute Trajectory Error (ATE RMSE [cm]). Mapping
quality is assessed via PSNR, SSIM, and LPIPS. For open-vocabulary perfor-
mance, we measure semantic understanding using mean Intersection-over-Union
(mIoU) and pixel-wise Accuracy, following the protocol of Feature 3DGS [44].
Implementation Details. All experiments are conducted on a desktop with
a Ryzen 7 7800x3d CPU, 32GB RAM, and an NVIDIA RTX 4090 GPU. The
ground-truth feature maps for distillation are pre-distilled by extracting per-
pixel language embeddings using LSeg [16]. Our scene-adaptive encoder and
decoder, composed of lightweight 1x1 convolutional layers, are pretrained as an
autoencoder to learn the compression to our 16-dimensional space. The language
codebook for loop closure is generated offline via k-means clustering (k=64) on
millions of feature vectors. During the SLAM process, our language-guided prun-
ing is performed every 200 iterations, and for loop detection, we select the top
two candidates with a cosine similarity above 0.7 for geometric verification. Cru-
cially, to reflect true online performance, all evaluations are performed on the
final map generated during the SLAM process, without any post-run optimiza-
tion iterations.
Baseline. We evaluate LEGO-SLAM against baselines, with all methods run
on the same hardware. For open-vocabulary performance, we compare against
LeRF [13], LangSplat [26], and Feature 3DGS [44]. For core SLAM capabilities,
we benchmark against NeRF-based systems (NICE-SLAM [48], Point-SLAM
[30], Loopy-SLAM [20]) and 3DGS-based systems (MonoGS [22], SplaTAM [11],
LoopSplat [45]), including those that incorporate loop closure mechanisms.
4.2
Quantitative Evaluation
Table 1: Open-Vocabulary Segmentation on 3 datasets. LEGO-SLAM uses
estimated poses while baselines use GT poses. Feature 3DGS with 512-dim features
failed on several scenes due to memory constraints, unlike our compact 16-dim features.
Method
Metrics
Replica TUM-RGBD ScanNet
LeRF [13]
Accuracy ↑
0.617
0.490
0.261
mIoU ↑
0.276
0.263
0.066
LangSplat [26]
Accuracy ↑
0.614
0.544
0.429
mIoU ↑
0.263
0.229
0.160
Feature 3DGS [44] Accuracy ↑
0.902
0.835
fail
mIoU ↑
0.691
0.633
fail
LEGO-SLAM
Accuracy ↑
0.882
0.834
0.791
mIoU ↑
0.674
0.650
0.519

<!-- page 9 -->
Abbreviated paper title
9
Open-Vocabulary Performance. We evaluate the open-vocabulary mapping
performance of LEGO-SLAM against leading 3D reconstruction methods, with
results summarized in Tab. 1. For this evaluation, we follow the protocol of [44]
and use the per-pixel language embeddings extracted by LSeg [16] as the GT
features for all methods. To ensure a fair comparison, we trained all baseline
reconstruction methods for 3000 iterations, aligning with the optimization steps
available to our mapping module. Feature 3DGS [44] achieves high accuracy on
Replica by distilling full 512-dimensional features but suffers from significant
memory overhead, causing it to fail on larger-scale ScanNet scenes. Conversely,
LangSplat [26] compresses features to only 3 dimensions, which substantially
reduces memory but also leads to significantly lower semantic accuracy. LEGO-
SLAM achieves a strong balance, delivering performance competitive with Fea-
ture 3DGS on Replica while successfully scaling to all large-scale ScanNet scenes
where memory-intensive methods fail. Notably, these results are achieved even
though LEGO-SLAM learns concurrently from estimated camera poses, a more
challenging and realistic setting compared to the ground-truth poses used by all
baseline methods.
Tracking Accuracy. We evaluate the tracking accuracy of LEGO-SLAM against
prominent NeRF-based and 3DGS-based SLAM systems (Tabs. 2 to 4). On the
Replica dataset, LEGO-SLAM achieves the lowest average ATE (0.20 cm). It
remains competitive on the challenging real-world TUM-RGBD sequences (2.30
cm) and large-scale ScanNet (8.68 cm). This ScanNet performance, which lever-
ages our efficient language-based loop closure, is comparable to specialized loop-
closure systems like LoopSplat and Loopy-SLAM, while achieving higher accu-
racy than non-loop-closure baselines. Crucially, this high accuracy is achieved
at a consistent 15 FPS (Tab. 5), operating faster than competitors. Notably,
this competitive tracking performance is achieved while LEGO-SLAM concur-
rently builds a rich, open-vocabulary map, whereas the compared baselines only
perform geometric and RGB mapping.
Table 2: Tracking Performance on Replica [35] (ATE RMSE [cm] ↓). The best
results are highlighted as first , second , and third .
Method
R0
R1
R2
O0
O1
O2
O3
O4
Avg.
NICE-SLAM [48]
0.97
1.31
1.07
0.88
1.00
1.06
1.10
1.13
1.06
Point-SLAM [30]
0.54
0.41
0.23
0.32
0.45
0.48
0.56
0.68
0.47
Loopy-SLAM [20]
0.30
0.47
0.30
0.25
0.21
0.31
0.32
0.40
0.32
MonoGS [22]
0.30
0.22
0.28
0.36
0.21
0.24
0.12
0.77
0.31
SplaTAM [11]
0.31
0.39
0.27
0.49
0.23
0.30
0.32
0.60
0.36
LoopSplat [45]
0.27
0.24
0.16
0.23
0.17
0.36
0.21
0.36
0.25
LEGO-SLAM
0.15
0.21
0.16
0.20
0.16
0.22
0.28
0.22
0.20

<!-- page 10 -->
10
S. Lee et al.
Table 3: Tracking Performance on TUM-RGBD [36] (ATE RMSE [cm] ↓).
Method
fr1-desk
fr2-xyz
fr3-office
Avg.
NICE-SLAM [48]
2.80
2.10
7.20
4.00
Point-SLAM [30]
2.73
1.30
3.51
2.51
Loopy-SLAM [20]
3.74
1.90
3.12
2.92
MonoGS [22]
1.47
1.57
1.51
1.51
SplaTAM [11]
3.31
1.35
5.13
3.26
LoopSplat [45]
2.35
1.41
3.90
2.55
LEGO-SLAM
2.53
1.72
2.64
2.30
Table 4: Tracking Performance on ScanNet [4] (ATE RMSE [cm] ↓).
Method
00
59
106
169
181
207
Avg.
NICE-SLAM [48]
12.0
14.0
7.9
10.9
13.4
6.2
10.73
Point-SLAM [30]
9.60
7.24
8.44
20.92
14.41
4.56
10.86
Loopy-SLAM [20]
4.63
7.82
8.55
7.97
11.65
6.59
7.87
MonoGS [22]
30.34
17.03
11.30
21.53
20.51
8.07
18.13
SplaTAM [11]
12.19
10.01
18.05
12.37
12.78
7.74
12.19
LoopSplat [45]
5.30
7.09
6.54
10.76
7.93
6.07
7.28
LEGO-SLAM
5.78
13.43
7.58
6.47
12.68
6.11
8.68
Mapping Quality and FPS. We evaluate the mapping quality and system
speed in Tab. 5. LEGO-SLAM achieves high-fidelity mapping quality while build-
ing a rich, open-vocabulary map at 15 FPS. Across all datasets, our method
demonstrates leading performance in mapping quality results against all base-
lines. All metrics are from the online SLAM process without post-run optimiza-
tion. This 15 FPS operation is significantly faster than all competing Neural
Rendering SLAM systems in the table, which operate at non-real-time rates.
Fig. 3: Qualitative Mapping Comparison. We compare the rendered maps of
LEGO-SLAM against baselines on the TUM-RGBD, and ScanNet datasets. All maps
shown are captured directly from the online SLAM process without any post-run op-
timization.

<!-- page 11 -->
Abbreviated paper title
11
Table 5: Rendering Performance and FPS on 3 datasets. All metrics are from
the online SLAM process without post-run optimization. LEGO-SLAM achieves high-
fidelity rendering quality at 15 FPS, operating faster than all baselines.
Method
Metrics
Replica
TUM-RGBD
ScanNet
Point-SLAM [30]
PSNR[dB] ↑
35.56
21.33
23.31
SSIM ↑
0.977
0.733
0.753
LPIPS ↓
0.118
0.453
0.509
FPS ↑
0.415
0.252
0.233
Loopy-SLAM [20]
PSNR[dB] ↑
19.28
14.32
12.46
SSIM ↑
0.662
0.512
0.495
LPIPS ↓
0.506
0.470
0.574
FPS ↑
0.374
0.222
0.357
MonoGS [22]
PSNR[dB] ↑
35.33
17.82
16.23
SSIM ↑
0.943
0.714
0.599
LPIPS ↓
0.122
0.327
0.588
FPS ↑
0.679
2.52
2.01
SplaTAM [11]
PSNR[dB] ↑
34.19
23.53
18.82
SSIM ↑
0.970
0.908
0.699
LPIPS ↓
0.094
0.166
0.370
FPS ↑
0.212
0.407
0.544
LoopSplat [45]
PSNR[dB] ↑
14.13
12.85
14.20
SSIM ↑
0.748
0.511
0.571
LPIPS ↓
0.584
0.746
0.708
FPS ↑
0.651
0.58
0.445
LEGO-SLAM
PSNR[dB] ↑
36.38
23.86
19.44
SSIM ↑
0.957
0.858
0.758
LPIPS ↓
0.075
0.138
0.286
FPS ↑
15.0
15.0
15.0
4.3
Ablation Study
Analysis of the Scene-Adaptive Encoder. We validate our encoder-based
initialization strategy, which is essential for real-time SLAM systems that must
continuously adapt to new environments. As shown in Tab. 6, training from
scratch (Ours w/o Init) requires a high number of iterations for the abstract
feature loss to converge. In contrast, by using our pretrained encoder to provide
a strong prior for new Gaussians (Ours w/ Init), the feature convergence is
considerably accelerated.
Furthermore, our scene-adaptive encoder design provides a second, equally
important advantage by enabling efficient 3D object localization. Rather than
decoding the entire map back to a high-dimensional space for querying, we lever-
age the learned encoder to project the text query directly into the map’s compact
16-dimensional space for fast, direct comparison. While pretraining provides a
strong start, continuous adaptation is critical in SLAM, where systems must

<!-- page 12 -->
12
S. Lee et al.
learn from new environments, unlike methods that rely on static feature extrac-
tors. To demonstrate this necessity, we compare our adaptive encoder against
a baseline where the pretrained encoder is kept frozen. As visualized in Fig. 4,
the frozen encoder fails to produce meaningful localization results, as its static
features cannot adapt to the specific scene. Our scene-adaptive encoder, how-
ever, successfully localizes the object, proving that continuous online learning is
critical for our system’s emergent open-vocabulary capabilities.
Table 6: Ablation on Initialization. Convergence speed in iterations comparing
pretrained initialization against training from scratch (Convergence thresholds: Total
< 0.1, RGB < 0.05, Feature < 0.05).
Method
Conv. Steps
Replica
TUM-RGBD
ScanNet
Ours (w/o Init)
RGB
214
238
288
Feature
228
230
228
Total
219
232
247
Ours (w/ Init)
RGB
214
240
277
Feature
76
54
67
Total
179
145
154
Fig. 4: Scene-Adaptive Encoder Adaptation. Our Adaptive, online-tuned encoder
generates accurate relevancy maps for 3D object queries, while the Frozen baseline fails.
Impact of Feature Dimension. We investigate the impact of our compact
feature dimension (d) on mapping quality, semantic accuracy, and memory us-
age in Tab. 7. Selecting the feature dimension involves a critical trade-off. A
larger d increases memory and rendering costs, which in turn reduces the map-
ping optimization iterations available in real-time SLAM. This trade-off is clear
at the extremes. At d = 8, the system is most efficient, yielding the highest
PSNR as geometry converges more fully. However, its semantic accuracy is low-
est, as 8 dimensions are insufficient for feature reconstruction. Conversely, at

<!-- page 13 -->
Abbreviated paper title
13
Table 7: Ablation study on the impact of feature dimension (d).
Method
Metrics
Replica
TUM-RGBD
ScanNet
Ours (d=8)
PSNR [dB] ↑
36.61
24.07
19.49
Accuracy ↑
0.854
0.784
0.746
Memory [MB] ↓
61.9
71.5
194.6
Ours (d=16)
PSNR [dB] ↑
36.38
23.86
19.44
Accuracy ↑
0.882
0.834
0.791
Memory [MB] ↓
81.8
92.9
256.6
Ours (d=32)
PSNR [dB] ↑
35.84
23.38
19.24
Accuracy ↑
0.861
0.816
0.776
Memory [MB] ↓
123.7
146.1
385.6
Ours (d=64)
PSNR [dB] ↑
34.93
23.00
19.16
Accuracy ↑
0.861
0.810
0.778
Memory [MB] ↓
123.7
248.1
636.6
Ours (d=128)
PSNR [dB] ↑
32.86
22.34
fail
Accuracy ↑
0.771
0.795
fail
Memory [MB] ↓
371.1
452.7
fail
d ≥32, semantic accuracy also drops because slower rendering reduces mapping
iterations, preventing feature convergence. At d = 128, the memory overhead is
prohibitive, failing on ScanNet. Based on this analysis, we select d = 16. This
choice represents an intentional trade-off for robust language alignment. As vali-
dated in Tab. 7, we accept a minor sacrifice in geometric quality, seen as a slight
PSNR drop compared to d = 8, to gain a significant improvement in semantic
accuracy. This trade-off allows us to balance the crucial demands of language
representation power with the low memory essential for real-time SLAM.
Analysis of Language-Guided Pruning. We analyze the effectiveness of our
language-guided pruning against the conventional geometry-based pruning used
in systems [11,45]. To test robustness, we compare their performance as the prun-
ing aggressiveness is increased. As visualized in Fig. 5, the geometric approach’s
rendering quality degrades sharply even at low pruning ratios, leading to a severe
Fig. 5: Pruning Performance Comparison. As the pruning ratio increases, our
language-guided method shows significantly less degradation in rendering quality com-
pared to the geometric approach on the Replica Room0 scene.

<!-- page 14 -->
14
S. Lee et al.
degradation as aggressiveness increases. In contrast, our language-guided method
remains robust, maintaining high rendering quality across a much wider range
of pruning ratios. This stability is confirmed by Tab. 8, which shows that at a
highly aggressive pruning level, our method maintains high PSNR and semantic
accuracy where the geometric-only strategy fails. This effectiveness is achieved
with negligible computational overhead, as our semantic check efficiently reuses
the compact 16-dimensional features already stored on the Gaussians.
Table 8: Pruning Ablation Study. We compare our language-guided strategy
against a geometric-only baseline under a highly aggressive setting.
Method
Metrics
Replica
TUM-RGBD
ScanNet
Baseline
(No Pruning)
PSNR ↑
36.14
23.80
19.25
SSIM ↑
0.956
0.857
0.753
LPIPS ↓
0.077
0.137
0.294
Acc ↑
0.880
0.833
0.788
IoU ↑
0.649
0.647
0.515
# of GS (K) ↓
645
761
2063
Baseline [11,45]
(Geo Pruning)
PSNR ↑
24.20
17.39
16.56
SSIM ↑
0.833
0.725
0.655
LPIPS ↓
0.232
0.281
0.377
Acc ↑
0.834
0.772
0.725
IoU ↑
0.574
0.570
0.451
# of GS (K) ↓
400
232
951
Ours
(Lang Pruning)
PSNR ↑
35.21
22.34
19.16
SSIM ↑
0.949
0.820
0.753
LPIPS ↓
0.098
0.191
0.298
Acc ↑
0.880
0.819
0.786
IoU ↑
0.649
0.632
0.516
# of GS (K) ↓
382
218
928
Effectiveness of Language-Based Loop Detection. We validate our inte-
grated language-based loop closure module. It is efficient as it directly reuses
language features computed during the mapping process, avoiding the computa-
tional overhead of separate models [1,24] that would compromise our real-time
objective. To demonstrate this efficiency does not sacrifice accuracy, we there-
Table 9: Loop Detection Comparison. Our language-based method achieves lower
tracking error (ATE RMSE [cm] ↓) than the position-based approach.
Method
Replica
TUM-RGBD
ScanNet
Position-based [34]
0.276
3.06
10.15
Language-based
0.206
2.30
8.68

<!-- page 15 -->
Abbreviated paper title
15
fore compare its performance against a similarly lightweight, real-time position-
based baseline [34]. As shown in Tab. 9, our language-based method consistently
achieves a lower tracking error across all datasets. This confirms it offers more
robust place recognition than the position-based approach.
5
Conclusion
We proposed LEGO-SLAM, the first real-time, open-vocabulary 3DGS SLAM
system. Our core contribution is a lightweight, scene-adaptive autoencoder that
distills language features into a compact 16-dimensional space, considerably re-
ducing memory and rendering overhead to enable real-time performance. These
features also enhance the SLAM pipeline by enabling a language-guided pruning
strategy and an efficient feature-reuse loop closure. Experiments demonstrated
that LEGO-SLAM achieves competitive mapping quality and tracking accuracy
at real-time speeds.
References
1. Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: Netvlad: Cnn ar-
chitecture for weakly supervised place recognition. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 5297–5307 (2016)
2. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin,
A.: Emerging properties in self-supervised vision transformers. In: Proceedings of
the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)
3. Cheng, H.K., Oh, S.W., Price, B., Schwing, A., Lee, J.Y.: Tracking anything with
decoupled video segmentation. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. pp. 1316–1326 (2023)
4. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet:
Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 5828–5839 (2017)
5. Dellaert, F.: Factor graphs and gtsam: A hands-on introduction. Georgia Institute
of Technology, Tech. Rep 2(4) (2012)
6. Ghiasi, G., Gu, X., Cui, Y., Lin, T.Y.: Scaling open-vocabulary image segmentation
with image-level labels. In: European conference on computer vision. pp. 540–557.
Springer (2022)
7. Ha, S., Yeon, J., Yu, H.: Rgbd gs-icp slam. In: European Conference on Computer
Vision. pp. 180–197. Springer (2024)
8. Hanson, A., Tu, A., Singla, V., Jayawardhana, M., Zwicker, M., Goldstein, T.: Pup
3d-gs: Principled uncertainty pruning for 3d gaussian splatting. In: Proceedings of
the Computer Vision and Pattern Recognition Conference. pp. 5949–5958 (2025)
9. Johari, M.M., Carta, C., Fleuret, F.: Eslam: Efficient dense slam system based on
hybrid representation of signed distance fields. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 17408–17419 (2023)
10. Kant, Y., Ramachandran, A., Yenamandra, S., Gilitschenski, I., Batra, D., Szot, A.,
Agrawal, H.: Housekeep: Tidying virtual households using commonsense reasoning.
In: European Conference on Computer Vision. pp. 355–373. Springer (2022)

<!-- page 16 -->
16
S. Lee et al.
11. Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan,
D., Luiten, J.: Splatam: Splat track & map 3d gaussians for dense rgb-d slam.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 21357–21366 (2024)
12. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)
13. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: Lerf: Language em-
bedded radiance fields. In: Proceedings of the IEEE/CVF international conference
on computer vision. pp. 19729–19739 (2023)
14. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of
the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)
15. Lee, S., Yu, H., Kim, G., Choi, S.: Lamp: Implicit language map for robot naviga-
tion. IEEE Robotics and Automation Letters (2025)
16. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven
semantic segmentation. arXiv preprint arXiv:2201.03546 (2022)
17. Li, K., Niemeyer, M., Navab, N., Tombari, F.: Dns-slam: Dense neural semantic-
informed slam. In: 2024 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS). pp. 7839–7846. IEEE (2024)
18. Li, M., Liu, S., Zhou, H., Zhu, G., Cheng, N., Deng, T., Wang, H.: Sgs-slam:
Semantic gaussian splatting for neural dense slam. In: European Conference on
Computer Vision. pp. 163–179. Springer (2024)
19. Liao, G., Zhou, K., Bao, Z., Liu, K., Li, Q.: Ov-nerf: Open-vocabulary neural
radiance fields with vision and language foundation models for 3d semantic un-
derstanding. IEEE Transactions on Circuits and Systems for Video Technology
(2024)
20. Liso, L., Sandström, E., Yugay, V., Van Gool, L., Oswald, M.R.: Loopy-slam: Dense
neural slam with loop closures. In: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. pp. 20363–20373 (2024)
21. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang,
J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for
open-set object detection. In: European conference on computer vision. pp. 38–55.
Springer (2024)
22. Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.: Gaussian splatting slam. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition. pp. 18039–18048 (2024)
23. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commu-
nications of the ACM 65(1), 99–106 (2021)
24. Mur-Artal, R., Tardós, J.D.: Orb-slam2: An open-source slam system for monoc-
ular, stereo, and rgb-d cameras. IEEE transactions on robotics 33(5), 1255–1262
(2017)
25. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V.,
Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust
visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
26. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian
splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 20051–20060 (2024)
27. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from

<!-- page 17 -->
Abbreviated paper title
17
natural language supervision. In: International conference on machine learning. pp.
8748–8763. PmLR (2021)
28. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction.
In: Proceedings of the IEEE/CVF international conference on computer vision.
pp. 12179–12188 (2021)
29. Rosinol, A., Abate, M., Chang, Y., Carlone, L.: Kimera: an open-source library for
real-time metric-semantic localization and mapping. In: 2020 IEEE International
Conference on Robotics and Automation (ICRA). pp. 1689–1696. IEEE (2020)
30. Sandström, E., Li, Y., Van Gool, L., Oswald, M.R.: Point-slam: Dense neural point
cloud-based slam. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision. pp. 18433–18444 (2023)
31. Segal, A., Haehnel, D., Thrun, S.: Generalized-icp. In: Robotics: science and sys-
tems. vol. 2, p. 435. Seattle, WA (2009)
32. Shafiullah, N.M.M., Paxton, C., Pinto, L., Chintala, S., Szlam, A.: Clip-
fields: Weakly supervised semantic fields for robotic memory. arXiv preprint
arXiv:2210.05663 (2022)
33. Shah, D., Osiński, B., Levine, S., et al.: Lm-nav: Robotic navigation with large pre-
trained models of language, vision, and action. In: Conference on robot learning.
pp. 492–504. PMLR (2023)
34. Shan, T., Englot, B., Meyers, D., Wang, W., Ratti, C., Rus, D.: Lio-sam: Tightly-
coupled lidar inertial odometry via smoothing and mapping. In: 2020 IEEE/RSJ
international conference on intelligent robots and systems (IROS). pp. 5135–5142.
IEEE (2020)
35. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J.,
Mur-Artal, R., Ren, C., Verma, S., et al.: The replica dataset: A digital replica of
indoor spaces. arXiv preprint arXiv:1906.05797 (2019)
36. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for
the evaluation of rgb-d slam systems. In: 2012 IEEE/RSJ international conference
on intelligent robots and systems. pp. 573–580. IEEE (2012)
37. Tosi, F., Zhang, Y., Gong, Z., Sandström, E., Mattoccia, S., Oswald, M.R., Poggi,
M.: How nerfs and 3d gaussian splatting are reshaping slam: a survey. arXiv
preprint arXiv:2402.13255 4, 1 (2024)
38. Yan, C., Qu, D., Xu, D., Zhao, B., Wang, Z., Wang, D., Li, X.: Gs-slam: Dense vi-
sual slam with 3d gaussian splatting. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 19595–19604 (2024)
39. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth anything: Un-
leashing the power of large-scale unlabeled data. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 10371–10381 (2024)
40. Ye, Z., Wan, C., Li, C., Hong, J., Li, S., Li, L., Zhang, Y., Lin, Y.C.: 3d gaus-
sian rendering can be sparser: Efficient rendering via learned fragment pruning.
Advances in Neural Information Processing Systems 37, 5850–5869 (2024)
41. Yugay, V., Li, Y., Gevers, T., Oswald, M.R.: Gaussian-slam: Photo-realistic dense
slam with gaussian splatting. arXiv preprint arXiv:2312.10070 (2023)
42. Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: Go-slam: Global optimization for con-
sistent 3d instant reconstruction. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. pp. 3727–3737 (2023)
43. Zhang, Z., Song, T., Lee, Y., Yang, L., Peng, C., Chellappa, R., Fan, D.: Lp-
3dgs: Learning to prune 3d gaussian splatting. Advances in Neural Information
Processing Systems 37, 122434–122457 (2024)

<!-- page 18 -->
18
S. Lee et al.
44. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z.,
Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled
feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. pp. 21676–21685 (2024)
45. Zhu, L., Li, Y., Sandström, E., Huang, S., Schindler, K., Armeni, I.: Loopsplat:
Loop closure by registering 3d gaussian splats. In: 2025 International Conference
on 3D Vision (3DV). pp. 156–167. IEEE (2025)
46. Zhu, S., Qin, R., Wang, G., Liu, J., Wang, H.: Semgauss-slam: Dense semantic
gaussian splatting slam. arXiv preprint arXiv:2403.07494 (2024)
47. Zhu, S., Wang, G., Blum, H., Liu, J., Song, L., Pollefeys, M., Wang, H.: Sni-slam:
Semantic neural implicit slam. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. pp. 21167–21177 (2024)
48. Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys,
M.: Nice-slam: Neural implicit scalable encoding for slam. In: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. pp. 12786–
12796 (2022)
