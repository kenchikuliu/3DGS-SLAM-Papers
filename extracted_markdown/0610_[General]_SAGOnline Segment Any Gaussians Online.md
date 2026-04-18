<!-- page 1 -->
1
SAGOnline: Segment Any Gaussians Online
Wentao Sun, Quanyun Wu, Hanqing Xu, Kyle Gao, Member, IEEE, Zhengsen Xu, Yiping Chen, Senior
Member, IEEE, Dedong Zhang, Member, IEEE, Lingfei Ma, Member, IEEE, John S. Zelek, Member, IEEE,
Jonathan Li, Fellow, IEEE
Abstract—3D Gaussian Splatting has emerged as a power-
ful paradigm for explicit 3D scene representation, yet achiev-
ing efficient and consistent 3D segmentation remains challeng-
ing. Existing segmentation approaches typically rely on high-
dimensional feature lifting, which causes costly optimization,
implicit semantics, and task-specific constraints. We present
Segment Any Gaussians Online (SAGOnline), a unified, zero-
shot framework that achieves real-time, cross-view consistent
segmentation without scene-specific training. SAGOnline decou-
ples the monolithic segmentation problem into lightweight sub-
tasks. By integrating video foundation models (e.g., SAM 2), we
first generate temporally consistent 2D masks across rendered
views. Crucially, instead of learning continuous feature fields, we
introduce a Rasterization-aware Geometric Consensus mecha-
nism that leverages the traceability of the Gaussian rasterization
pipeline. This allows us to deterministically map 2D predictions
to explicit, discrete 3D primitive labels in real-time. This discrete
representation eliminates the memory and computational burden
of feature distillation, enabling instant inference. Extensive eval-
uations on NVOS and SPIn-NeRF benchmarks demonstrate that
SAGOnline achieves state-of-the-art accuracy (92.7% and 95.2%
mIoU) while operating at the fastest speed at 27 ms per frame.
By providing a flexible interface for diverse foundation models,
our framework supports instant prompt, instance, and semantic
segmentation, paving the way for interactive 3D understanding
in AR/VR and robotics.
Index Terms—2D Gaussian Splatting, 3D Segmentation, Gaus-
sian Segmentation, Foundation Model
I. INTRODUCTION
T
HE
recent
emergence
of
3D
Gaussian
Splatting
(3DGS) [1] has reshaped real-time 3D scene reconstruc-
tion by offering an explicit, compact, and efficient scene
representation. Unlike earlier implicit radiance-field formula-
tions, 3DGS models a scene as a set of anisotropic Gaussian
primitives, enabling real-time rendering while preserving high-
fidelity geometry and appearance. Because this representation
is discrete and explicitly structured, each Gaussian can be
enriched with additional attributes such as semantic labels or
feature vectors, making 3DGS well suited for downstream 3D
understanding.
This ability to attach semantics to individual Gaussians is
especially appealing for tasks that require online and cross-
Corresponding author: Yiping Chen
W. Sun, Q. Wu, K. Gao, D. Zhang, J. S. Zelek, and J. Li are
with the University of Waterloo, Waterloo, ON N2L 3G1, Canada (e-
mails: w27sun@uwaterloo.ca; q34wu@uwaterloo.ca; y56gao@uwaterloo.ca;
dedong.zhang@uwaterloo.ca; jzelek@uwaterloo.ca; junli@uwaterloo.ca).
H. Xu and L. Ma are with East China Normal University, Shanghai 200050,
China (e-mails: 51273901140@stu.ecnu.edu.cn; l53ma@cufe.edu.cn).
Z. Xu is with the University of Calgary, Calgary Alberta T2N 1N4, Canada
(e-mail: zhengsen.xu@ucalgary.ca)
Y. Chen is with Sun Yat-Sen University, Zhuhai 519082, China (e-mail:
chenyp79@mail.sysu.edu.cn).
view consistent segmentation. Such capabilities are increas-
ingly crucial for applications in augmented and virtual real-
ity [2]–[4], robotic manipulation and mapping [5]–[7], and
autonomous systems [8], [9], where agents must parse the
environment from continuously changing viewpoints. Conse-
quently, a growing body of work aims to embed semantic
information into 3D Gaussians so that the representation
itself supports view-consistent segmentation from novel view-
points [10]–[12].
Recent advancements in 3DGS-based segmentation can be
broadly categorized by their mechanism for semantic injection.
Distillation-based methods transfer features from 2D founda-
tion models into 3D primitives, enabling early demonstrations
of zero-shot 3D segmentation once the features are aligned
with the Gaussian representation [11], [13]. Contrastive frame-
works, conversely, take potentially inconsistent multi-view 2D
masks as input and enforce cross-view consistency in the 3D
embedding space, which in turn enables consistent novel-view
segmentation [12], [14]. Interaction-based approaches operate
directly on Gaussians to support click- or prompt-driven tasks
and benefit from the geometry-aware structure of the Gaussian
scene [10].
Despite these advancements, online 3DGS segmentation is
still difficult due to three main challenges: First, inconsistent
2D observations. In an online system, 2D foundation mod-
els often produce unstable results as the camera viewpoint
changes. Maintaining a stable 3D identity without slow, per-
scene optimization is a major challenge. Second, complex
occlusions. 3DGS consists of many overlapping primitives. In
crowded scenes, it is hard to map 2D masks to the correct
3D depth, especially when objects are partially hidden from
certain angles. Third, diverse task requirements. Different
applications need different types of segmentation, such as in-
stance, semantic, or prompt-based masks. A single, unified 3D
representation that handles all these tasks without retraining is
still missing.
Existing methods only partially address these challenges and
introduce their own bottlenecks. First, to fix 2D inconsisten-
cies, many methods [11], [13] require long optimization times
(e.g., tens of minutes). This makes real-time use impossible.
Second, most approaches rely on large feature vectors instead
of simple labels [11], [12]. These features use a lot of memory.
More importantly, they are ”implicit,” meaning the 3D masks
are not directly visible. This makes it hard to handle complex
occlusions because the system cannot accurately map labels
to the correct depth. Finally, current pipelines [14] are often
rigid and designed for only one task. They lack a unified way
to handle instance, semantic, and prompt segmentation at the
same time.
arXiv:2508.08219v2  [cs.CV]  6 Jan 2026

<!-- page 2 -->
2
To address these challenges, we propose Segment Any
Gaussians Online (SAGOnline), a decoupled and optimization-
free framework. Since 3DGS can render novel viewpoints at
real-time frame rates, we can treat the continuous render-
ing stream as a synchronized video sequence. This physical
characteristic allows us to decompose the monolithic 3D
segmentation task into three lightweight, modular sub-tasks:
(1) novel-view rendering, (2) multi-view consistent 2D video
segmentation, and (3) global 3D consensus fusion. By bridging
the domain gap between 3D scenes and 2D videos, we can
directly deploy powerful video segmenters without any 3D
fine-tuning or per-scene optimization.
To address the consistency challenge (Challenge 1), we
leverage the high-speed rendering nature of 3DGS by treat-
ing its continuous rendering stream as a synchronized video
sequence. This perspective allows us to bridge the domain
gap between static 3D scenes and dynamic 2D observations,
enabling the integration of temporal priors from state-of-the-
art video segmenters to stabilize multi-view observations. By
decoupling the segmentation task into a video-like processing
pipeline, SAGOnline effectively mitigates the 2D inconsis-
tency problem without requiring expensive per-scene optimiza-
tion.
To overcome the occlusion and mapping ambiguity (Chal-
lenge 2), we introduce a Rasterization-aware Geometric Con-
sensus (RGC) mechanism. By exploiting the deterministic
traceability inherent in the 3DGS rasterizer, RGC establishes
a robust voting scheme that fuses 2D labels into explicit,
discrete 3D attributes for each primitive. This mechanism nat-
urally handles complex occlusions by selectively aggregating
semantic information only from the visible surface of the
objects during the fusion process. Unlike prior implicit feature-
based methods that suffer from ”black-box” decoding, our
explicit representation ensures high-fidelity mask projection
and rigorous geometric interpretability, even in cluttered or
densely occluded environments.
Furthermore, to ensure cross-task generalization (challenge
3), we design a generalizable interface for integrating founda-
tion models. Because SAGOnline cleanly decouples rendering,
2D segmentation, and 3D fusion, various foundation models
(e.g., SAM2 [15], SAM3 [16], SEEM [17], and video seg-
menters) can be plugged in interchangeably as modular com-
ponents. Unlike prior methods that treat foundation models
merely as offline mask generators, SAGOnline incorporates
them as interactive components, enabling prompt segmenta-
tion, semantic segmentation, and instance segmentation within
a unified pipeline.
Quantitative evaluations on the NVOS and SPIn-NeRF
benchmarks demonstrate that SAGOnline outperforms existing
baselines (e.g., Feature3DGS [11], SA3D [18]) with state-of-
the-art mIoUs of 92.7% and 95.2%, respectively. Regarding
speed, our framework achieves immediate usability with the
fastest novel-view segmentation rate of 27 ms per frame
(960 × 540 resolution on an NVIDIA RTX 4090). Beyond
standard benchmarks, we validate the method’s versatility
on the KITTI-360 [19], UDD [20], etc. datasets, proving
its efficacy across diverse scenarios, including autonomous
driving, drone-based urban views, and complex indoor scenes.
Our main contributions are summarized as follows:
• A
Novel
Optimization-free
Online
Segmentation
Framework: We propose SAGOnline, a decoupled archi-
tecture that reinterprets the continuous rendering stream
of 3DGS as a synchronized video sequence. By lever-
aging temporal priors from video foundation models,
SAGOnline achieves robust cross-view consistency and
eliminates the need for expensive per-scene optimization,
enabling immediate and real-time 3D segmentation.
• The Rasterization-aware Geometric Consensus (RGC)
Mechanism: We introduce RGC, a deterministic fusion
scheme that bridges the gap between 2D observations
and 3D primitives. By exploiting the traceability of the
3DGS rasterizer, RGC maps 2D predictions into explicit,
discrete 3D attributes. This mechanism effectively re-
solves occlusion-induced ambiguities and depth-mapping
inaccuracies inherent in implicit feature-based represen-
tations.
• A Unified and Extensible Interface for Multi-task Seg-
mentation: Through its modular design, SAGOnline pro-
vides a task-agnostic interface that seamlessly integrates
various foundation models (e.g., SAM2, SEEM). This
unified pipeline supports instance, semantic, and prompt-
based segmentation within a single framework without
retraining, demonstrating superior versatility across di-
verse indoor and outdoor scenarios.
• State-of-the-Art Performance and Efficiency: Quanti-
tative evaluations on NVOS and SPIn-NeRF benchmarks
show that SAGOnline achieves record-breaking mIoU
scores while maintaining a high-speed inference rate
of 27 ms per frame. Its robustness is further validated
on large-scale datasets including KITTI-360 and UDD,
underscoring its potential for practical deployment in
autonomous systems.
II. RELATED WORK
A. Geometry-Aware 3D Gaussian Representations
The fidelity of 3D segmentation is intrinsically linked to the
quality of the underlying geometric representation. While the
original 3D Gaussian Splatting (3DGS) excels in photoreal-
istic rendering, its optimization often results in unstructured
volumetric ”clouds” or ”floaters” that lack physical surface
fidelity [21]. Such artifacts are catastrophic for segmentation,
as semantic boundaries cannot be clearly defined on fuzzy
geometries.
To address this, recent research has pivoted towards
geometry-aware optimization. SuGaR [21] regularizes Gaus-
sians to align with the zero-level set of a Signed Dis-
tance Function (SDF), enabling explicit mesh extraction.
More significantly for segmentation, 2D Gaussian Splatting
(2DGS) [22] and its successors [23] replace 3D ellipsoids
with 2D oriented disks. This ”planar shift” provides explicit
normal vectors and enforces surface constraints, which prevent
semantic labels from bleeding through the object volume. Our
work builds upon these geometric insights, leveraging surface-
aligned primitives to ensure that segmentation masks respect
physical boundaries.

<!-- page 3 -->
3
B. Semantic Distillation and Consistency
Bridging the gap between 2D foundation models (e.g., SAM
[24], CLIP [25]) and 3D space is primarily achieved through
feature distillation. Early attempts like Feature3DGS [11]
lift 2D semantic features into high-dimensional Gaussian
attributes. However, these methods suffer from severe multi-
view inconsistency, where conflicting 2D predictions lead to
fragmented 3D boundaries. Recent studies, such as Feature-
Homogenized GS (FHGS) [26], attribute this issue to the
conflict between the anisotropic nature of 3DGS (optimized for
view-dependent radiance) and the isotropic nature of semantic
identity (invariant to view).
To mitigate inconsistency, contrastive learning frameworks
like OmniSeg3D-GS [13] and GLS [27] employ cross-view
clustering or joint optimization with geometric priors (e.g.,
TSDF). While these methods improve coherence, they typ-
ically operate in an offline manner, requiring expensive pre-
computation or global optimization (e.g., Linear Programming
in FlashSplat [28]) that precludes real-time user interaction.
C. Interactive and Online Segmentation
The transition from static annotation to interactive editing
has driven the development of human-in-the-loop systems.
Early interactive models like SA3D [18] relied on iterative
inverse rendering, resulting in high latency (seconds to minutes
per prompt).
To achieve real-time performance, recent approaches such
as SAGA [12] and Click-Gaussian [10] pre-cache seman-
tic features, allowing for millisecond-level inference. SAGA
specifically introduces scale-aware embeddings to disentangle
semantic granularity from geometric scale. However, these
methods are often limited by the temporal inconsistency of
the underlying 2D segmentation models (SAM 1.0). With the
advent of video foundation models like SAM 2 [15], new
paradigms such as Seg-Wild [29] and WildSeg3D [30] have
begun to exploit temporal memory for better consistency. Yet,
effectively integrating these video-based priors into a truly on-
line, geometry-aware 3DGS framework—without succumbing
to ”spiky” artifacts or requiring offline retraining—remains an
open challenge.
D. Summary
In summary, existing paradigms largely treat geometry,
semantics, and interaction as separate optimization targets.
Geometry-focused methods (SuGaR, 2DGS) lack semantic
flexibility; Distillation-based methods (Feature3DGS) struggle
with consistency; and Interactive frameworks (SA3D, SAGA)
often trade geometric precision for speed. Our proposed
method addresses this tripartite gap by unifying geometry-
aware regularization with an online, interaction-driven mech-
anism enabled by foundation video models.
III. METHOD
The proposed SAGOnline framework aims to achieve real-
time 3D semantic segmentation through two stages: Stage
I: Rasterization-aware Geometric Consensus and Stage II:
Self-supervised Refinement. Our key insight is that while
foundational video segmenters provide consistent 2D masks,
they are too heavy for real-time interaction. Therefore, we
utilize them only as ”teachers” to initialize 3D primitive labels
and supervise a lightweight ”student” network for instant
inference.
A. Problem Formulation
Given a pre-trained Gaussian Splatting model (trained by
2DGS [22]) G = {gi}N
i=1, where each Gaussian gi is param-
eterized by its position µ, rotation q, scale s, opacity α, and
color c. Our objective is twofold:
1) 3D Labeling: Assign a discrete semantic label li ∈
{1, . . . , K} to each primitive gi, resulting in a semantic
field S = {li}N
i=1.
2) Real-time Segmentation: Given an arbitrary camera
pose θ, efficiently render a high-quality semantic mask
ˆ
Mθ ∈{1, . . . , K}H×W .
The challenge lies in achieving this without high-dimensional
feature lifting while maintaining temporal consistency and
real-time inference speeds.
B. Stage I: Rasterization-aware Geometric Consensus
This stage establishes a deterministic mapping between 2D
multi-view semantics and 3D Gaussian primitives. By leverag-
ing the inherent traceability of the 3DGS rasterization pipeline,
we avoid the memory-intensive requirements of feature-based
lifting. This module operates in two sequential steps: 2D
temporal labeling and 3D geometric consensus.
1) 2D Temporal Labeling: To generate reliable 2D supervi-
sion, we exploit the explicit nature of 3DGS, where rendering
along a smooth camera trajectory naturally yields temporally
coherent image sequences. Let V = {v1, v2, . . . , vT } denote a
sequence of camera poses. We first render the corresponding
RGB images I using the 3DGS rasterizer:
It = R(G, vt),
(1)
where R(·) denotes the rasterization operator. To ensure cross-
view consistency, we employ a foundational video segmenter
fm (e.g., SAM 2 [15]) to process the rendered sequence I,
producing a set of consistent 2D masks:
L2D = fm(I1, . . . , IT ).
(2)
This video-based approach significantly mitigates the incon-
sistent issues common in frame-wise segmentation, providing
a robust foundation for 3D label lifting.
2) 3D Geometric Consensus Mechanism: Once the consis-
tent 2D masks L2D are obtained, we propose a Rasterization-
aware Geometric Consensus (RGC) mechanism to lift these
labels into 3D space. During the 3DGS forward pass for a
pixel p, the rendered color is computed via alpha-blending:
C(p) =
X
i∈N p
ciαiTi,
Ti =
i−1
Y
j=1
(1 −αj),
(3)
where Np denotes the sorted set of primitives overlapping
pixel p, and Ti represents the accumulated transmittance.

<!-- page 4 -->
4
Continuous
3DGS Renders
Video Foundation
Model
(e.g., SAM 2)
Consistent
2D Masks
RAGC Mechanism
Geometric Alignment
Accumulated
Transmittance (T)
Num. of Gaussians
Visibility Criterion
Majority
Voting
Sparse Semantic 
3D Gaussians
A. Rasterization-aware Geometric Consensus (Initialization)
Novel View
Sparse Semantic 
3D Gaussians
Counts
3DGS
Renderer
Rasterizer
RGB Image
Coarse Mask
(with holes)
B. Self-supervised Refinemnet (Real-time Inference)
Training
Views
3D Gaussians
Dual-Branch Refinement Network
Image
Encoder
(ResNet-50)
Mask
Encoder
Final Upsample
Decoder Block
Decoder Block
Decoder Block
Channel Attention Fusion
Online Distillation
(Pseudo-GT)
Pixel Ray
Dense Refine Mask
Fig. 1.
The architecture of SAGOnline. The framework comprises an initialization stage (Left) and a real-time inference stage (Right). (A) We propose
a Rasterization-aware Geometric Consensus (RAGC) module to resolve semantic ambiguities. By analyzing the pixel ray, RAGC identifies valid Gaussians
within the surface crust (green zone) that are geometrically aligned with the pixel center, aggregating 2D semantics via majority voting. (B) To achieve
real-time segmentation, the learned sparse semantic Gaussians are projected to form a coarse mask. This coarse prior is fused with the photorealistic RGB
render in a Dual-Branch Refinement Network, employing an encoder-decoder structure with channel attention to recover fine-grained details, supervised by
online distillation.
To ensure that labels are only assigned to primitives that
accurately represent the scene geometry, we enforce two strict
filtering criteria:
• Surface Visibility: A primitive contributes to the seman-
tic consensus only if it lies on the visible surface crust.
We traverse the ray front-to-back and evaluate the post-
primitive transmittance Ti+1 = Ti(1−αi). A primitive Gi
is considered a valid surface component if the ray remains
visibly potent upon reaching it (Ti > ϵ). The traversal
is terminated immediately when the accumulated opacity
saturates, i.e., when Ti+1 drops below a threshold ϵ
(experimentally set to 0.01). This ensures we strictly label
the visible surface layer while discarding fully occluded
internal primitives.
• Geometric Alignment: Due to the anisotropic nature of
3DGS, a primitive may intercept a ray with its elongated
”tail” rather than its core, leading to loose semantic
associations. To enforce strict geometric alignment, we
require the projected center µ2D
i
of primitive Gi to be
strictly aligned with the pixel p: ∥µ2D
i
−p∥2 < δ (δ
experimentally set to 2). This rejects primitives that only
intersect the ray with their Gaussian tails, ensuring that
the semantic label is assigned to the primitive’s core.
Based on these criteria, we maintain a voting histogram Hi for
each Gaussian primitive Gi. For every pixel p in the training
views, if Gi satisfies both the visibility and alignment criteria,
it receives a vote from the corresponding 2D mask label
L2D(p). The final 3D label li is determined by the majority
vote:
li = arg max
k
Hi(k)
(4)
Primitives that fail to accumulate sufficient votes or ambiguous
votes are left unlabeled. This strict filtering results in a sparse
but highly accurate explicit 3D segmentation, effectively cre-
ating a ”semantic point cloud” on the object surface. This
sparsity motivates the need for our subsequent self-supervised
refinement stage.
C. Stage II: Self-supervised Refinement
Stage II is designed to operate concurrently with Stage I to
overcome three critical limitations. First, the efficiency bottle-
neck: the segmentation speed in Stage I is strictly bounded
by the video segmenter. Second, the viewpoint constraint:
reliance on temporal continuity prevents the system from
handling wide-baseline camera transitions. Third, the sparsity
issue: although the Stage I: Rasterization-aware Geometric
Consensus effectively lifts 2D semantics to 3D, our rigorous
filtering (to ensure precision) results in sparse 3D labels. Con-
sequently, direct rasterization produces coarse masks (M 2D
coarse)
with structural holes and aliasing.
To achieve more rapid and robust Gaussian segmentation,
we propose a Self-supervised Refinement Module. This mod-

<!-- page 5 -->
5
ule employs a lightweight student network, Nϕ, to distill
knowledge from two sources: the ”accurate but slow” founda-
tion model and the ”consistent but sparse” 3D projections. By
doing so, Nϕ transforms these inputs into a ”fast and dense”
inference engine, enabling high-quality segmentation across
arbitrary viewpoints. The operational flow of this module is
formulated as follows:
M 2D
coarse = R(S, vt),
It = R(G, vt)
(5)
M 2D
refine = Nϕ(M 2D
coarse, It)
(6)
where S denotes the semantic field obtained from Stage I, vt
represents the camera viewpoint, R(·) signifies the Gaussian
Splatting rasterizer, and G is the Gaussian model. It corre-
sponds to the rendered RGB image at viewpoint vt. M 2D
refine
corresponds to the refined 2D mask.
1) Network Architecture:
As illustrated in Fig. 1, the
refinement network employs a dual-branch encoder-decoder
architecture specifically designed to fuse appearance cues with
coarse semantic guidance.
Dual-Branch Encoder: The network processes two inputs:
the rendered RGB image and the coarse rasterized mask.
• Image Encoder: We employ a ResNet-50 backbone ini-
tialized with ImageNet pre-trained weights to extract
multi-scale appearance features. We strategically select
this robust architecture to leverage its rich feature hi-
erarchy, which significantly accelerates the convergence
of our online distillation process compared to training a
lightweight encoder from scratch. This branch captures
high-fidelity texture and boundary details essential for
completing the coarse mask.
• Mask Encoder: A lightweight convolutional branch pro-
cesses the coarse mask M 2D
coarse. This serves as a strong
spatial prompt, guiding the network to focus on specific
regions of interest.
Feature Fusion and Decoding: High-level semantic features
from the Mask Encoder are fused with deep appearance
features from the Image Encoder. To effectively integrate these
modalities, we employ a Channel Attention (CA) mechanism
before concatenation. The decoder follows a U-Net-like struc-
ture with skip connections, progressively upsampling features
to recover fine-grained details.
2) Online Distillation Strategy: Unlike traditional super-
vised learning requiring manual annotation, we adopt an online
distillation strategy. We leverage the temporally consistent
masks generated by the foundation model on the rendered
views as pseudo-ground truth. Since the network focuses
on the simplified task of refining boundaries and filling
holes based on specific scene textures, rather than learning
generalized semantics, it converges rapidly. Empirically, this
adaptation requires approximately 4 minutes of fine-tuning per
scene.
Formally, the network is optimized via a pixel-wise multi-
class cross-entropy loss over the training views V:
Lrefine = −
1
|V| · |Ω|
X
v∈V
X
p∈Ω
C
X
c=1
⊮[M GT
v (p) = c] log ˆ
Mv(p, c)
(7)
where Ωdenotes the pixel coordinates space of the image, and
C represents the total number of semantic classes. The term
⊮[·] is the indicator function which equals 1 if the condition is
true and 0 otherwise. Specifically, M GT
v (p) denotes the pseudo-
ground truth label index for pixel p in view v, while ˆ
Mv(p, c)
represents the predicted probability that pixel p belongs to
class c.
Once trained, this module replaces the foundation model
entirely, enabling high-speed inference (approx. 27 ms/frame)
for arbitrary, non-continuous viewpoints while maintaining
high segmentation quality.
IV. EXPERIMENTS
A. Datasets
Our evaluation utilizes two primary datasets: NVOS [31]
and the SPIn-NeRF benchmark [32]. The NVOS dataset
extends the LLFF dataset [33] by providing instance-level
annotations for foreground objects across diverse scenes. The
SPIn-NeRF benchmark is a multiview segmentation dataset
that offers unified instance segmentation labels across multiple
domains. Both datasets are specifically designed for NeRF-
based and Gaussian-based segmentation tasks, ensuring direct
compatibility with our methods.
To comprehensively evaluate cross-domain performance,
we conduct additional experiments on the KITTI-360 [19],
UDD [20], NeRDS-360 [34], Mip-360 [35], LERF-Mask
dataset [14], DesktopObjects-360 [36] scenes. This extended
evaluation enables a systematic analysis of our method’s
segmentation quality (in both 2D and 3D) and computational
efficiency across: 1) indoor/outdoor environments, 2) varying
object densities, and 3) different illumination conditions.
B. Experimental Settings
All experiments were conducted on a workstation equipped
with an NVIDIA RTX 4090 GPU and an Intel i5-11400F CPU,
running PyTorch 2.1.0 with CUDA 11.8 on a Linux operating
system. Unless otherwise specified, all Gaussian scenes used
for evaluation were trained for 30,000 iterations. Because
SAM2 requires manually provided prompts to segment the
target of interest, we simulate this behavior during initializa-
tion by randomly sampling clicks on the ground-truth object
to form the prompt. No additional user inputs are required
throughout inference.
For datasets with sequential image organization, we adopt a
filename-based pseudo-temporal ordering to facilitate initial-
ization. Specifically, for the Spin-NeRF dataset, the training
images are lexicographically sorted by filename to form the
pseudo-temporal sequence. Likewise, for the NVOS dataset,
we establish the same ordering strategy and initialize the target
region through random clicks on the ground-truth object.
C. Quantitative Segmentation Performance
We evaluate our method on two challenging benchmarks:
the NVOS dataset [31] and the SPIn-NeRF dataset [32]. We
compare SAGOnline against a comprehensive set of baselines,
including NVOS [31], ISRF [37], MVSeg [32], SA3D [18],

<!-- page 6 -->
6
TABLE I
QUANTITATIVE COMPARISON ON NVOS AND SPIN-NERF DATASETS.
OUR SAGONLINE ACHIEVES STATE-OF-THE-ART PERFORMANCE ON
BOTH BENCHMARKS. “-” INDICATES THE METHOD IS NOT APPLICABLE OR
NOT REPORTED FOR THAT DATASET.
Method
NVOS Dataset
SPIn-NeRF Dataset
mIoU
mAcc
mIoU
mAcc
NVOS [31]
70.1
92.0
-
-
ISRF [37]
83.8
96.4
-
-
MVSeg [32]
-
-
91.0
98.9
SA3D [18]
90.3
98.2
92.4
98.9
OmniSeg3D [13]
91.7
98.4
95.2
99.2
SA3D-GS [38]
92.2
98.5
93.2
99.1
SAGA [12]
92.6
98.6
93.4
99.2
Click-Gaussian [10]
-
-
94.0
-
SAGOnline (Ours)
92.7
98.7
95.2
99.3
OmniSeg3D-GS [13], SA3D-GS [38], SAGA [12], and Click-
Gaussian [10]. Notably, OmniSeg3D-GS represents the pre-
vious top performer on SPIn-NeRF, while SAGA holds the
leading position on NVOS.
Performance
on
NVOS.
As
presented
in
Table
I,
SAGOnline achieves a mIoU of 92.7% and mAcc of 98.7%,
effectively matching and slightly surpassing the previous state-
of-the-art method, SAGA. It is crucial to note that SAGA
relies on a heavier offline optimization process. The fact that
SAGOnline achieves a slight edge (+0.1% mIoU) confirms
that our streamlined online framework does not compromise
segmentation quality.
Performance on SPIn-NeRF. Table I shows the results on
the SPIn-NeRF dataset. Our method attains 95.2% mIoU and
99.3% mAcc. While the mIoU is tied with the previous best
performer (OmniSeg3D), our method shows a slight improve-
ment in segmentation consistency (mAcc). More importantly,
these results indicate that SAGOnline effectively utilizes the
priors from foundation models.
Analysis of Performance Saturation. As observed in
Table I, the performance gap between recent methods (e.g.,
SA3D-GS, SAGA, OmniSeg3D) and ours is relatively narrow.
This convergence suggests that current 3D segmentation meth-
ods are approaching the upper bound defined by the quality
of the underlying 2D foundation models (e.g., SAM series).
In this context, SAGOnline achieves high accuracy within
an efficient online framework. It bridges 2D priors and 3D
Gaussians without the heavy computational overhead of prior
offline methods.
Unlike baselines that require extensive offline training or
complex distillation to preserve foundation model priors, our
method leverages the temporal coherence of SAM 2 to directly
and rapidly propagate segmentation in 3D. The results demon-
strate that SAGOnline effectively bridges the gap between
2D foundation models and 3D Gaussian Splatting, achieving
state-of-the-art accuracy without the computational overhead
of prior methods.
Qualitative results are illustrated in Fig. 2. Our method
exhibits robust performance across diverse scenarios. For
instance, in the Fork scene, SAGOnline captures fine structural
details that are occasionally missed by ground truth annota-
TABLE II
TIME EFFICIENCY COMPARISON ON MULTI-OBJECT SCENES. UNLIKE
OFFLINE METHODS THAT REQUIRE SIGNIFICANT PRE-COMPUTATION
(BLOCKING), OUR METHOD LEVERAGES A FOUNDATION MODEL TO
OFFER INSTANT INTERACTION (0 S), WITH RAPID 3D MASK
AGGREGATION (1.47S FOR ∼200 FRAMES).
Method
Time-to-First-Mask
Inference
Mode
(Start-up Latency)
(Speed)
SA3D [18]
0 s
>50 s
Online
GARField [39]
45 min
3 s
Offline (Blocking)
Feature3DGS [11]
35 min
510 ms
Offline (Blocking)
OmniSeg3D-GS [13]
37 min
463 ms
Offline (Blocking)
SAGA [12]
32 min
31 ms
Offline (Blocking)
SAGOnline (Initial)
0 s†
97 ms
Online (Immediate)
SAGOnline (Refined)
∼5 min∗
27 ms
Online (Background)
† Instant response via Foundation Model. Full 3D aggregation for the
sequence takes approx. 1.47s.
∗The refinement process runs asynchronously in the background. Users can
interact immediately using the Initial mode without waiting.
tions, further validating the effectiveness of our initialization
and refinement strategy.
D. Time Efficiency
The temporal efficiency of 3D segmentation algorithms
is a critical factor for their deployment in latency-sensitive
applications, such as AR/VR systems and interactive scene
editing. We evaluate the efficiency of SAGOnline against state-
of-the-art baselines on DesktopObjects-360 dataset, focusing
on two key metrics:
• Time-to-First-Mask (Start-up Latency): The waiting
time required before the user can visualize and interact
with the first valid segmentation result.
• Inference Speed: The rendering frame rate during the
interactive segmentation session.
To ensure a fair comparison, all baseline methods were
executed on the same hardware configuration (NVIDIA RTX
4090). The quantitative comparisons are summarized in Ta-
ble II.
Immediate Availability vs. Blocking Pre-computation. As
shown in Table II, traditional offline methods (e.g., GARField,
SAGA, Feature3DGS) suffer from a significant ”cold-start”
problem. These approaches are blocking: users must wait for
a mandatory training period of 30 to 45 minutes before any
segmentation result is produced.
In contrast, SAGOnline is designed as a non-blocking,
online framework. By leveraging a vision Foundation Model
(e.g., SAM2), our method instantly generates a 2D mask,
achieving a Time-to-First-Mask of 0 s. Following this inter-
action, the system rapidly aggregates the generated 2D masks
(e.g., across 207 frames) to construct the initial 3D mask.
This aggregation process completes in merely 1.47 seconds,
which is negligible compared to the tens of minutes of training
required by offline baselines. This capability matches the
responsiveness of online methods like SA3D but operates at a
significantly higher frame rate (97 ms vs. > 50s).
Asynchronous Self-supervised Refinement. To achieve
real-time rendering performance suitable for AR/VR (e.g.,
> 30 FPS), our framework incorporates a Self-supervised
Refinement Module. This module operates as a background

<!-- page 7 -->
7
Scene
Segmentation Results  Across Views
Render 0
Render 1
Render 2
Fork
Fortress
Horns
Truck
Scene
Segmentation Results  Across Views
Render 0
Render 1
Render 2
Fig. 2. Qualitative segmentation results demonstrating multi-view consistency. We present target object extraction results for four diverse scenes: Fork, Fortress,
Horns, and Truck. For each block, the input scene is shown in the 1st column, followed by the extracted binary masks rendered from three distinct viewpoints
(Render 0-2). The results highlight our method’s ability to maintain precise object boundaries and geometric coherence across varying camera poses.
process to distill the coarse 3D masks and optimize Gaussian
attributes, which takes approximately 5 minutes. Crucially, this
refinement phase is transparent to the user. The user can
continue to interact with the scene using the Initial mode (97
ms/frame) while the optimization runs in parallel. Once the
refinement converges, the system seamlessly transitions to the
Refined mode, boosting the inference speed to 27 ms/frame
(37 FPS)—a 13% improvement over SAGA and 17× faster
than OmniSeg3D-GS.
This ”As-You-Go” paradigm ensures that SAGOnline pro-
vides the best of both worlds: the instant responsiveness of
online methods and the high-performance rendering of offline
baked models, without imposing mandatory waiting periods
on the user.
E. Multi-Object Segmentation & Cross-View Consistency
While current benchmarks such as NVOS and Spin-NeRF
provide valuable evaluation platforms for 3D-aware segmen-
tation, they exhibit two major limitations: (1) scenes in these
datasets generally contain only a single dominant object of
interest, limiting the evaluation of instance-level separation
in cluttered environments; and (2) their evaluation protocols
primarily rely on 2D projection-based metrics (e.g., IoU and
mIoU on rendered masks), which cannot comprehensively cap-
ture true 3D spatial consistency or inter-object boundary pre-
cision. To overcome these limitations and comprehensively as-
sess the generalization capability of our method, we addition-
ally evaluate on the Mip-360, LERF-Mask, DesktopObjects-
360, and NeRds360 datasets. These datasets feature complex
indoor, tabletop, and outdoor scenes characterized by multiple
interacting instances, severe occlusions, and varying object
scales.
Fig. 3 presents representative qualitative results of our
method across these datasets. As shown in the second column,
our approach successfully distinguishes individual object in-
stances from a 3D perspective, achieving accurate separation
even in scenes with severe occlusion and background clutter.
The third and fourth columns further demonstrate consistent
segmentation across drastically different viewpoints. Although
the scenes are static, the significant camera movement creates
a challenging scenario akin to object tracking. In this context,
our method ensures that each object maintains a stable instance
ID throughout the trajectory. This multi-view consistency
highlights our framework’s capacity for reliable cross-view
instance association and tracking-like continuity.
The robustness of our framework stems from the first-
stage generation of global 3D instance masks, which estab-
lish explicit 3D coordinate correspondences for every object.
These masks serve as spatial anchors that enable stable object
localization and identity preservation under arbitrary viewpoint
transformations. Leveraging the Gaussian Splatting rendering
pipeline, our system projects 3D segmentation results onto the
2D image plane while estimating per-pixel opacity. This en-
ables accurate reasoning about inter-object occlusion and visi-
bility, allowing the method to maintain object continuity even
in cluttered scenes. Consequently, our method achieves high-
fidelity multi-object segmentation with temporal-like consis-
tency across views, offering a practical foundation for down-
stream applications such as 3D scene editing, object-centric
reconstruction, and Gaussian field compositing.
F. Different Segmentation Tasks
Our framework enables seamless integration with various
foundation models, allowing it to support multiple segmenta-
tion tasks across different domains. This design also ensures
that our method remains compatible with future algorithmic
upgrades. For example, by incorporating YOLO [40], we
can perform fast object detection and automatic instance
segmentation. We evaluate this capability on the widely used
autonomous-driving dataset KITTI-360, as illustrated in Fig. 4.
Our method effectively leverages YOLO to accomplish in-
stance segmentation of vehicles.
Moreover, our framework naturally supports language-
guided segmentation tasks through direct integration with
vision-language models. For instance, by using the recent
Segment Anything Model 3 (SAM 3) [16], we can achieve
open-vocabulary segmentation. We conduct experiments on
the UAV-based UDD dataset, and the visualizations show
that our method can accurately segment buildings, roads, and
vegetation in complex urban scenes.
G. Effectiveness of Refinement Module
The Refinement algorithm is designed to refine the rendered
segmentation masks into spatially coherent and accurate re-
sults. To verify the effectiveness of our refinement, we perform

<!-- page 8 -->
8
Scene
3D Mask
Render 0
Render 1
ID: 1
ID: 4
ID: 7
ID: 6
ID: 5
ID: 2
ID: 3
ID: 3
ID: 2
ID: 7
ID: 4
ID: 1
ID: 6
ID: 5
Desktop
Segmentation Across Views
ID: 1
ID: 1
ID: 2
ID: 2
ID: 3
ID: 3
Cars
ID: 1
ID: 2
ID: 3
ID: 4
ID: 5
ID: 6
ID: 7
ID: 8
ID: 1
ID: 2
ID: 3
ID: 4
ID: 5
ID: 6
ID: 7
ID: 8
Teatime
Counter
ID: 1
ID: 2
ID: 3
ID: 4
ID: 1
ID: 2
ID: 3
ID: 4
ID: 5
ID: 6
ID: 6
ID: 7
ID: 7
ID: 8
ID: 9
ID: 9
ID: 10
ID: 10
ID: 11
Fig. 3. Qualitative results of multi-object segmentation across diverse scenes. The figure is organized into four scene blocks: Desktop, Cars, Teatime, and
Counter. For each block, the 1st column displays the reference scene; the 2nd column visualizes the explicit 3D masks formed by the Gaussian primitive
means; and the subsequent columns (3rd-4th) present the segmentation masks rendered from different viewpoints. Consistent colors across views indicate that
our method maintains robust instance identity.
experiments comparing results before and after the Refinement
Module.
As illustrated in Fig. 5, the directly rendered masks con-
tain scattered and fragmented regions. After the Refinement
Module, these sparse points are aggregated into continuous
and well-defined instance masks. This refinement significantly
enhances both mask completeness and boundary precision,
yielding more stable multi-view consistent segmentation re-
sults.
H. Robustness Analysis
Since our algorithm reconstructs 3D masks from multi-view
2D segmentation results, it is essential to examine how the
number of available 2D inputs affects the quality and stability
of the generated 3D masks. To this end, we conducted a
robustness analysis by randomly sampling different numbers
of 2D masks as input for the 3D mask generation process. As
the number of input masks increased, we measured both the
quality of the reconstructed 3D masks and the corresponding
computation time. This experiment provides a quantitative un-
derstanding of the trade-off between accuracy and efficiency,
and offers insights into the scalability of our method under
varying data availability.
As illustrated in Fig. 6, when only a few 2D masks are
provided, the resulting 3D masks still preserve clear object
boundaries and instance separability. As more 2D views are
incorporated, the spatial completeness and surface smoothness
of the 3D masks further improve, with more Gaussian primi-
tives being assigned consistent instance IDs. Remarkably, even
with as few as six input masks, our algorithm is capable
of accurately reconstructing multiple objects with minimal
identity confusion, indicating strong robustness against sparse
input views.
In terms of efficiency, Table III reports the average time
required to generate 3D masks under different input set-
tings. The results show a near-linear relationship between the

<!-- page 9 -->
9
Text Prompt: 'building'
Open-Vocabulary Segmentation
Text Prompt: 'tree'
Text Prompt: 'street'
Object Detection + Automatic Instance Segmentation
Fig. 4.
Qualitative results on diverse segmentation tasks. The top row demonstrates automatic vehicle instance segmentation on the KITTI-360 dataset
utilizing YOLO. The bottom row showcases open-vocabulary segmentation on the UDD dataset driven by SAM 3 with text prompts. These results highlight
our framework’s versatility, capable of leveraging distinct backbone models to handle various segmentation paradigms within 3D Gaussian Splatting scenes.
(a) Without Refinement
(b) With Refinement
Fig. 5.
Visual comparison of the Dual-Branch Refinement Network. (a)
Segmentation results derived directly from Sparse Semantic 3D Gaussians
exhibit noticeable sparsity and noise artifacts. (b) After applying our refine-
ment module, the segmentation masks become significantly denser, smoother,
and spatially coherent, demonstrating the network’s effectiveness in enhancing
mask quality.
TABLE III
TIME CONSUMPTION OF 3D MASK GENERATION. ”AVG.” DENOTES THE
AVERAGE TIME COST PER MASK (MS).
# Masks
Time (ms) Avg. (ms) # Masks Time (ms) Avg. (ms)
Full (206)
1466
7.1
12
162
13.5
100
571
5.7
6
140
23.3
50
384
7.7
3
114
38.0
25
224
9.0
1
111
111.0
number of input masks and computation time: the full 206-
view configuration requires 1466 ms, while processing only a
single mask reduces the time to 111 ms. This demonstrates
that our approach achieves efficient and scalable 3D mask
reconstruction, maintaining real-time performance even under
limited input conditions. The combination of high robustness
and low latency makes the method suitable for practical multi-
view perception and online scene understanding tasks.
I. Fidelity Preservation in Refinement Stage
To evaluate the impact of the Refinement Module on seg-
mentation fidelity, we analyzed the performance shift between
the Initial and Refined stages. The Refined stage maintains an
overall accuracy of 99.89% and mIoU of 98.47% relative to
the Initial stage outputs. This high correlation confirms that
our background refinement process successfully optimizes the
Gaussian attributes for rendering speed (from 97ms to 27ms)
without degrading the geometric integrity or semantic accuracy
established in the initialization phase.
V. CONCLUSION
In this work, we introduced SAGOnline, an optimization-
free framework that enables real-time, multi-view-consistent
segmentation for 3D Gaussian Splatting scenes. Departing
from prior approaches that rely on heavy feature distillation,
continuous semantic embeddings, or lengthy offline optimiza-
tion, SAGOnline reformulates Gaussian segmentation as three
lightweight sub-tasks: novel-view rendering, multi-view 2D
segmentation, and global 3D fusion. Central to our framework
is the Rasterization-aware Geometric Consensus mechanism.
It exploits 3DGS rasterizer traceability to deterministically lift
2D predictions into explicit, discrete 3D semantics. This ex-
plicit formulation allows immediate mask extraction from any
viewpoint without feature decoding, while our self-supervised
refinement module restores fine-grained boundaries and en-
ables high-speed rendering suitable for interactive applications.
Extensive experiments across NVOS, SPIn-NeRF, KITTI-
360, UDD, DesktopObjects-360, and other datasets demon-
strate that SAGOnline achieves state-of-the-art segmentation
performance while offering the fastest initialization and infer-
ence speeds among existing 3DGS-based methods. The frame-
work supports diverse segmentation paradigms—instance, se-
mantic, and prompt-guided segmentation—through a modular
interface that integrates a wide range of foundation models,
including SAM2, SAM3, YOLO, and video segmenters. These
results show that SAGOnline not only preserves the accuracy
of powerful 2D foundation models in the 3D domain, but
also delivers the responsiveness required for real-time scene
understanding, interactive 3D editing, and immersive AR/VR
applications.
Overall, SAGOnline demonstrates that explicit, geometry-
aware fusion combined with foundation-model priors provides
a practical and scalable route toward high-fidelity 3D segmen-
tation in Gaussian-splatting–based representations, paving the
way for more general and interactive 3D perception systems.

<!-- page 10 -->
10
Outdoor Scene
Indoor Scene
5% 2D Mask
10% 2D Mask
25% 2D Mask
0.5% 2D Mask
3% 2D Mask
12% 2D Mask
25% 2D Mask
50% 2D Mask
95% 2D Mask
Fig. 6.
Qualitative evaluation of 3D segmentation robustness under varying 2D mask supervision. The top row illustrates the outdoor scene results with
2D mask usage ranging from 5% to 25%, while the bottom row depicts the indoor scene from 0.5% to 95%. Observe that our method maintains consistent
semantic structures even with extremely sparse 2D mask inputs (e.g., 5% in the outdoor scene).
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3D gaussian
splatting for real-time radiance field rendering.” ACM Transactions on
Graphics, vol. 42, no. 4, pp. 139–1, 2023.
[2] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3D
gaussian splatting: Survey, technologies, challenges, and opportunities,”
IEEE Transactions on Circuits and Systems for Video Technology, pp.
1–1, 2025.
[3] Q. Herau, M. Bennehar, A. Moreau, N. Piasco, L. Rold˜ao, D. Tsishkou,
C. Migniot, P. Vasseur, and C. Demonceaux, “3DGS-Calib: 3d gaussian
splatting for multimodal spatiotemporal calibration,” in 2024 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS),
2024, pp. 8315–8321.
[4] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024, pp. 20 331–20 341.
[5] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “GS-
SLAM: Dense visual slam with 3d gaussian splatting,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024, pp. 19 595–19 604.
[6] H. Zhai, G. Huang, Q. Hu, G. Li, H. Bao, and G. Zhang, “NIS-SLAM:
Neural implicit semantic rgb-d slam for 3d consistent scene under-
standing,” IEEE Transactions on Visualization and Computer Graphics,
vol. 30, no. 11, pp. 7129–7139, 2024.
[7] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer,
D. Ramanan, and J. Luiten, “SplaTAM: Splat, track & map 3d gaussians
for dense rgb-d slam,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 357–
21 366.
[8] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Driv-
inggaussian: Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 634–
21 643.
[9] R. Xiao, W. Liu, Y. Chen, and L. Hu, “LiV-GS: Lidar-vision integration
for 3d gaussian splatting slam in outdoor environments,” IEEE Robotics
and Automation Letters, vol. 10, no. 1, pp. 421–428, 2025.
[10] S. Choi, H. Song, J. Kim, T. Kim, and H. Do, “Click-Gaussian:
Interactive segmentation to any 3D gaussians,” in Proceedings of the
European Conference on Computer Vision (ECCV), 2024, p. 289–305.
[11] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari,
S. You, Z. Wang, and A. Kadambi, “Feature 3DGS: Supercharging 3D
gaussian splatting to enable distilled feature fields,” in Proceedings of

<!-- page 11 -->
11
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024, pp. 21 676–21 685.
[12] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen, and Q. Tian,
“Segment any 3d gaussians,” Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 39, no. 2, pp. 1971–1979, Apr. 2025.
[13] H. Ying, Y. Yin, J. Zhang, F. Wang, T. Yu, R. Huang, and L. Fang,
“OmniSeg3D: Omniversal 3D segmentation via hierarchical contrastive
learning,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024, pp. 20 612–20 622.
[14] M. Ye, M. Danelljan, F. Yu, and L. Ke, “Gaussian Grouping: Segment
and edit anything in 3D scenes,” in Proceedings of the European
Conference on Computer Vision (ECCV), 2025, pp. 162–179.
[15] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr,
R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala,
N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, and C. Feichtenhofer,
“SAM 2: Segment anything in images and videos,” arXiv preprint
arXiv:2408.00714, 2024.
[16] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali,
K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything
with concepts,” arXiv preprint arXiv:2511.16719, 2025.
[17] X. Zou, J. Yang, H. Zhang, F. Li, L. Li, J. Wang, L. Wang,
J. Gao, and Y. J. Lee, “Segment everything everywhere all at
once,” in Advances in Neural Information Processing Systems, A. Oh,
T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
Eds., vol. 36.
Curran Associates, Inc., 2023, pp. 19 769–19 782.
[Online].
Available:
https://proceedings.neurips.cc/paper files/paper/
2023/file/3ef61f7e4afacf9a2c5b71c726172b86-Paper-Conference.pdf
[18] J. Cen, Z. Zhou, J. Fang, c. yang, W. Shen, L. Xie, D. Jiang, X. ZHANG,
and Q. Tian, “Segment anything in 3D with NeRFs,” in Advances in
Neural Information Processing Systems, vol. 36, 2023, pp. 25 971–
25 990.
[19] Y. Liao, J. Xie, and A. Geiger, “Kitti-360: A novel dataset and bench-
marks for urban scene understanding in 2d and 3d,” IEEE Transactions
on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3292–
3310, 2023.
[20] Y. Chen, Y. Wang, P. Lu, Y. Chen, and G. Wang, “Large-scale structure
from motion with semantic constraints of aerial images,” in Chinese
Conference on Pattern Recognition and Computer Vision (PRCV).
Springer, 2018, pp. 347–359.
[21] A. Gu´edon and V. Lepetit, “SuGaR: Surface-aligned gaussian splatting
for efficient 3D mesh reconstruction and high-quality mesh rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024, pp. 5354–5363.
[22] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2D gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
Conference Papers, 2024.
[23] Y. Zhang, A. Chen, Y. Wan, Z. Song, J. Yu, Y. Luo, and W. Yang, “Ref-
gs: Directional factorization for 2d gaussian splatting,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
26 483–26 492.
[24] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,
“Segment anything,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), October 2023, pp. 4015–4026.
[25] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable
visual models from natural language supervision,” in International
conference on machine learning.
PmLR, 2021, pp. 8748–8763.
[26] Q. G. Duan, B. Zhao, M. Han, Y. Huang, and B. M. Chen, “Fhgs:
Feature-homogenized gaussian splatting for 3d scene understanding with
multi-view consistency,” in Advances in Neural Information Processing
Systems, 2025.
[27] J. Qiu, L. Liu, X. Wang, T. Lin, W. Sui, and Z. Su, “Gls: Geometry-
aware 3d language gaussian splatting,” arXiv preprint arXiv:2411.18066,
2024.
[28] Q. Shen, X. Yang, and X. Wang, “Flashsplat: 2d to 3d gaussian splatting
segmentation solved optimally,” in Computer Vision – ECCV 2024: 18th
European Conference, Milan, Italy, September 29 – October 4, 2024,
Proceedings, Part XXII.
Berlin, Heidelberg: Springer-Verlag, 2024, p.
456–472.
[29] Y. Bao, C. Tang, Y. Wang, and H. Li, “Seg-wild: Interactive
segmentation based on 3d gaussian splatting for unconstrained image
collections,” in Proceedings of the 33rd ACM International Conference
on Multimedia, ser. MM ’25.
New York, NY, USA: Association
for Computing Machinery, 2025, p. 8567–8576. [Online]. Available:
https://doi.org/10.1145/3746027.3755567
[30] Y. Guo, J. Hu, Y. Qu, and L. Cao, “Wildseg3d: Segment any 3d objects in
the wild from 2d images,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), October 2025, pp. 5166–5176.
[31] Z. Ren, A. Agarwala, B. Russell, A. G. Schwing, and O. Wang,
“Neural volumetric object selection,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2022,
pp. 6123–6132.
[32] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A.
Brubaker, I. Gilitschenski, and A. Levinshtein, “SPIn-NeRF: Multiview
segmentation and perceptual inpainting with neural radiance fields,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2023, pp. 20 669–20 679.
[33] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ra-
mamoorthi, R. Ng, and A. Kar, “Local light field fusion: practical view
synthesis with prescriptive sampling guidelines,” ACM Transactions on
Graphics, vol. 38, no. 4, Jul. 2019.
[34] M. Z. Irshad, S. Zakharov, K. Liu, V. Guizilini, T. Kollar, A. Gaidon,
Z. Kira, and R. Ambrus, “Neo 360: Neural fields for sparse view syn-
thesis of outdoor scenes,” in 2023 IEEE/CVF International Conference
on Computer Vision (ICCV), 2023, pp. 9153–9164.
[35] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2022, pp. 5460–5469.
[36] W. Sun, H. Xu, Q. Wu, D. Zhang, Y. Chen, L. Ma, J. S. Zelek, and J. Li,
“Pointgauss: Point cloud-guided multi-object segmentation for gaussian
splatting,” 2025.
[37] R. Goel, D. Sirikonda, S. Saini, and P. J. Narayanan, “Interactive
segmentation of radiance fields,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2023,
pp. 4201–4211.
[38] J. Cen, J. Fang, Z. Zhou, C. Yang, L. Xie, X. Zhang, W. Shen, and
Q. Tian, “Segment anything in 3d with radiance fields,” arXiv preprint
arXiv:2304.12308, 2024.
[39] C. M. Kim, M. Wu, J. Kerr, K. Goldberg, M. Tancik, and A. Kanazawa,
“GARField: Group anything with radiance fields,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024, pp. 21 530–21 539.
[40] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024. [Online]. Available:
https://github.com/ultralytics/ultralytics
