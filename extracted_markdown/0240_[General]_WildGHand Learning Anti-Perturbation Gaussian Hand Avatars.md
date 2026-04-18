<!-- page 1 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
1
WildGHand: Learning Anti-Perturbation Gaussian
Hand Avatars from Monocular In-the-Wild Videos
Hanhui Li∗, Xuan Huang∗, Wanquan Liu, Senior Member, IEEE, Yuhao Cheng, Long Chen, Yiqiang Yan,
Xiaodan Liang, Senior Member, IEEE, Chenqiang Gao†
Abstract—Despite recent progress in 3D hand reconstruction
from monocular videos, most existing methods rely on data
captured in well-controlled environments and therefore degrade
in real-world settings with severe perturbations, such as hand-
object interactions, extreme poses, illumination changes, and
motion blur. To tackle these issues, we introduce WildGHand,
an optimization-based framework that enables self-adaptive 3D
Gaussian splatting on in-the-wild videos and produces high-
fidelity hand avatars. WildGHand incorporates two key compo-
nents: (i) a dynamic perturbation disentanglement module that
explicitly represents perturbations as time-varying biases on 3D
Gaussian attributes during optimization, and (ii) a perturbation-
aware optimization strategy that generates per-frame anisotropic
weighted masks to guide optimization. Together, these compo-
nents allow the framework to identify and suppress perturba-
tions across both spatial and temporal dimensions. We further
curate a dataset of monocular hand videos captured under
diverse perturbations to benchmark in-the-wild hand avatar
reconstruction. Extensive experiments on this dataset and two
public datasets demonstrate that WildGHand achieves state-of-
the-art performance and substantially improves over its base
model across multiple metrics (e.g., up to a 15.8% relative
gain in PSNR and a 23.1% relative reduction in LPIPS). Our
implementation and dataset are available at https://github.com/
XuanHuang0/WildGHand.
Index Terms—Hand avatar, 3D Gaussian splatting, perturba-
tion modeling, in-the-wild video.
I. INTRODUCTION
H
ANDS play a central role in human interaction in
both the physical world and immersive digital expe-
riences. With the rapid development of virtual/augmented
reality and embodied intelligence, the demand for realistic,
personalized hand avatars is increasing, and recent advances
in differentiable rendering [1], [2] have substantially improved
the fidelity of reconstructed avatars. Existing methods [3]–
[6] leverage large-scale data captured in carefully controlled
environments (e.g., studios with hundreds of calibrated cam-
eras [7]) and have demonstrated impressive performance in
generating high-quality hand avatars.
However, the high data requirements and limited data di-
versity of existing approaches severely limit their usability
and generalization. To alleviate these issues, several recent
studies exploit short monocular videos [8], [9] or even single
Hanhui Li, Xuan Huang, Wanquan Liu, Xiaodan Liang, Chenqiang Gao
are with the School of Intelligent Systems Engineering, Shenzhen Campus
of Sun Yat-sen University, P.R. China, 518107. Yuhao Cheng, Long Chen,
and Yiqiang Yan are with the Lenovo Research Group, Shenzhen, P.R. China,
518038.
∗Hanhui Li and Xuan Huang contribute equally.
† Chenqiang Gao is the corresponding author.
images [3], [10]. To simplify the reconstruction problem,
these methods typically assume that both the hands and their
surrounding environments remain static. While this assumption
reduces the need for large-scale captures, it breaks down in
uncontrolled in-the-wild scenarios, where hand observations
are often affected by occlusions and extreme poses, or the
captured images may further degrade due to motion blur and
illumination variations.
A complementary line of research focuses on dynamic or
distraction-aware reconstruction [11]–[16], which leverages
geometric, appearance, or semantic consistency across frames
to improve robustness. For example, [15] combines an un-
supervised classifier, a segmentation model, and an object
tracker to identify transient objects, while [16] leverages DINO
features [17] with uncertainty-aware optimization to better
handle appearance changes. Overall, these methods improve
the reconstruction quality of predominantly static scenes in
the presence of transient objects, underscoring the importance
of explicitly modeling perturbations.
Nevertheless, these dynamic models are not directly appli-
cable to in-the-wild hand avatar reconstruction for two key
reasons. First, most of them are designed to handle transient
distractions, e.g., moving objects or short-term occluders that
can be separated from the target using auxiliary cues such as
segmentation or tracking. In contrast, perturbations in hand-
centric in-the-wild videos are often global (e.g., illumination
changes affecting the entire image) and persistent (lasting
for extended time intervals). Therefore, perturbations in our
task cannot be treated as sparse, short-lived outliers. Second,
these methods are primarily developed for rigid or mildly
deformable scenes, whereas hands exhibit highly articulated
motions, frequent self-occlusions or hand-object occlusions,
and rapid pose changes. These properties make hand avatar
reconstruction particularly sensitive to corrupted supervision.
As a result, naive optimization can either underfit the true
hand appearance or geometry, or overfit to perturbations,
leading to an underfitting-overfitting dilemma under real-world
perturbations.
To address the above limitations and enable high-fidelity
hand avatar reconstruction from monocular in-the-wild videos
under perturbations, we propose a novel framework, dubbed
WildGHand. The key idea of WildGHand is to explicitly
disentangle perturbations from the underlying hand con-
tent during optimization. Concretely, WildGHand extends the
optimization-based 3D Gaussian splatting (3DGS) framework
[2] with two components: a dynamic perturbation disentan-
glement (DPD) module and a perturbation-aware optimiza-
arXiv:2602.20556v1  [cs.CV]  24 Feb 2026

<!-- page 2 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
2
Input Video
Rendered Images
Hand-Object Interaction
Input Video
Rendered Images
Complex Pose
Input Video
Rendered Images
Illumination Variation
Input Video
Rendered Images
Motion Blur
Fig. 1. We present WildGHand, a novel Gaussian splatting framework for generating realistic hand avatars from short monocular videos exhibiting challenging
perturbations, including hand-object interactions, complex poses, illumination variations, and motion blur.
tion (PAO) strategy. The DPD module adopts a lightweight
multilayer perceptron to model perturbations as temporally
weighted biases of 3D Gaussian attributes. These biases are
learned on training frames and removed at inference, thereby
reducing the risk of overfitting to perturbations. Meanwhile,
the PAO strategy leverages semantic hand structures and
reconstruction errors to localize perturbed regions and generate
anisotropic weighted masks that provide reliable supervision
for model optimization. These two components enable our
framework to generate robust and realistic hand renderings
under various perturbations, as shown in Figure 1.
Moreover, since existing benchmarks are either collected
in well-controlled environments [7], [18], [19] or contain
only limited perturbation diversity [8], [20]–[22], they are
insufficient for evaluating hand avatar reconstruction under
realistic in-the-wild environments. To bridge this gap, we
systematically categorize common perturbations and curate a
challenging dataset accordingly. As summarized in Table I, our
hand with perturbation (HWP) dataset covers four represen-
tative challenging factors in both single-hand and interacting-
hand scenarios, including hand-object interactions, complex
poses, illumination variations, and motion blur. Importantly,
each sequence contains multiple perturbation types to reflect
real-world capture conditions, while we additionally provide
a clean test clip to enable reliable and fair evaluation. The
dataset further includes diverse hand gestures and daily inter-
actions, such as shuffling cards, spinning pens, and applying
hand cream. Experiments on this dataset and two public bench-
marks demonstrate that WildGHand produces high-quality
personalized hand avatars from complex in-the-wild captures
and achieves state-of-the-art performance.
In summary, the contributions of this paper can be summa-
rized as follows:
• We introduce WildGHand, an optimization-based 3D
Gaussian splatting framework for reconstructing high-fidelity
hand avatars from short monocular in-the-wild videos under
severe perturbations.
• We propose a dynamic perturbation disentanglement
(DPD) module that models perturbations as temporally
weighted attribute biases during optimization and removes
these biases at inference to mitigate overfitting to corruptions.
• We develop a perturbation-aware optimization (PAO)
strategy that generates anisotropic per-frame weighted masks
to down-weight unreliable regions, thereby improving robust-
ness to both spatial and temporal perturbations.
• We curate a challenging in-the-wild hand video dataset
with diverse perturbations to benchmark robust hand avatar
reconstruction.
II. RELATED WORK
A. Differentiable Rendering
3D reconstruction via differentiable rendering has attracted
extensive attention due to its ability to jointly optimize geom-
etry and appearance from image observations. Representative
approaches include neural radiance fields (NeRF) [1] and
3D Gaussian splatting (3DGS) [2], which have demonstrated
high-fidelity results in a variety of downstream applications
such as novel view synthesis [23]–[25] and human avatar
creation [26]–[30]. From the perspective of object and scene
representation, existing differentiable renderers can be broadly
categorized into (i) implicit methods that parameterize targets
as continuous functions (e.g., NeRF and its variants), and
(ii) explicit methods that represent a target using discrete
primitives with learnable attributes (e.g., Gaussian points in
3DGS). Implicit representations are expressive and naturally
support continuous, arbitrarily high resolution. However, they
are often computationally expensive at rendering time. In
contrast, explicit representations, particularly 3DGS, achieve
efficient rasterization-based rendering with competitive visual
quality, and have thus become increasingly popular in practice.
Specifically, owing to the advantages of function-based rep-
resentations that provide a unified modeling and optimization
paradigm, recent NeRF-based methods have been extended to
incorporate additional inductive biases and signals, such as
explicit physical models [31], [32], higher-level relational or
functional reasoning [33], [34], and complementary modal-
ities [35]. For instance, [31] introduces a general radiative
physical formulation that combines emission, absorption, and
scattering, which can handle various media like underwater
and low-light scenes. [33] proposes a framework that can infer

<!-- page 3 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
3
TABLE I
DETAILED COMPARISON BETWEEN THE PROPOSED HWP DATASET AND EXISTING HAND VIDEO DATASETS.
Dataset
Env.
Clean Testset
Single Hand
Int. Hand
Hand-Object Int.
Complex Poses
Illum. Var.
Motion Blur
InterHand2.6M [7]
Lab
✓
✓
✓
HARP [20]
Wild
✓
✓
UHM [8]
Wild
✓
✓
DexYCB [18]
Lab
✓
✓
✓
EpicKitchen [48]
Wild
✓
✓
✓
✓
✓
HInt [21]
Wild
✓
✓
✓
✓
✓
ARCTIC [19]
Lab
✓
✓
✓
Ego4D [22]
Wild
✓
✓
✓
✓
✓
HWP (Ours)
Wild
✓
✓
✓
✓
✓
✓
✓
inter-object relationships from NeRFs. [35] leverages scene
priors from video diffusion models to improve consistency of
reconstructed results.
As for 3DGS, recent research has focused on improving its
representational capacity to better handle complex appearance
and motion patterns, such as specular surfaces with strong
reflections [36], articulated objects [37], parameterized objects
[38], and fine-grained human details (e.g., faces and hair) [39].
These extensions are often enabled by introducing stronger
priors and additional optimization objectives. In particular,
video diffusion priors have attracted growing interest [40]–
[43], as they can encode rich geometry and appearance statis-
tics and encourage cross-view consistency. Beyond video dif-
fusion models, several works incorporate semantic priors from
multimodal large language models [44], [45] or texture priors
from image diffusion models [46], [47]. While such large-
scale priors can improve robustness and generalization under
challenging capture conditions, they typically introduce sub-
stantial computational and engineering overhead, which may
limit their applicability in per-scene optimization pipelines. In
contrast, our method aims to improve robustness to in-the-wild
perturbations without relying on heavy external generative
models. Specifically, we employ a lightweight MLP to model
perturbations as attribute-level biases with temporal weights,
and combine it with a perturbation-aware optimization strategy
to reduce the influence of unreliable regions during training.
This design achieves a practical balance between efficiency
and effectiveness, making it well-suited for robust hand avatar
reconstruction from monocular in-the-wild videos.
B. Dynamic Modeling
Despite the success of differentiable rendering techniques,
most existing methods implicitly assume that the underlying
scene or subject is view- and time-consistent, i.e., the observa-
tions are not substantially affected by environmental changes
such as moving distractors, illumination variations, or motion
blur. However, this assumption is often violated in real-world
capture conditions.
To improve robustness, a line of work on NeRFs identifies
and treats distractors as outliers and reduces their influence
by reweighting or truncating the reconstruction loss [11], [12],
[49], [50]. RobustNeRF [12], for example, detects outliers by
ranking pixel residuals and applying a blur kernel to exploit
their spatial smoothness. NeRF-W [11] relaxes the consistency
assumption by learning per-image appearance embeddings
and decomposing scenes into static and transient components.
NeRF-HuGS [49] further combines heuristics with segmenta-
tion models to enhance robust hand identification and mitigate
interference from hand-irrelevant regions.
More recently, several methods have investigated robust
training of 3DGS for wild-captured data [13], [51]–[54].
Similar to RobustNeRF, SpotlessSplats [13] performs unsuper-
vised outlier detection in a learned feature space to identify
distractors from rendering errors. Robust-3DGS [14] integrates
Segment Anything [51] with a neural classifier to refine
segmentation masks.
Several methods [55]–[58] make their efforts to extend
3DGS to 4DGS by using time-varying Gaussian attributes
or deformation fields. For example, RigGS [57] proposes
to animate canonical 3D Gaussians by sparse 3D skeletons
with learnable skin deformations and pose-dependent detailed
deformations. Except for skeletons, sparse motion controls like
Hermite splines [59] have been used to drive 3D Gaussians.
MoSca [60] introduces the concept of 4D motion scaffolds,
which are structured graphs that Gaussians can be anchored
on and manipulated with priors extracted by vision foundation
models. 7DGS [61] even includes 3D viewing directions
into the 4D Gaussian representation, which show superior
performance compared to 4DGS.
Nevertheless, most of these approaches focus on short-lived,
transient distractors (e.g., moving objects), whereas in-the-wild
hand avatar reconstruction often involves broader and longer-
lasting perturbations (e.g., global illumination changes), which
remain challenging to handle.
C. 3D Hand Avatars
Numerous works [3]–[6], [62] have achieved impres-
sive results in creating hand avatars from studio-captured
datasets. LiveHand [4], built on NeRF, enables real-time
rendering through low-resolution training and 3D-consistent
super-resolution. HandAvatar [5] introduces a high-resolution
MANO-HD model and separates hand albedo and illumina-
tion based on pose. RelightableHands [62] presents a novel
relighting approach for hand rendering.
However, these methods rely on deliberately collected data
with expensive cameras, tracking devices, and annotation
tools, which limits their real-world applications. To address

<!-- page 4 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
4
these challenges, recent research has focused on creating per-
sonalized hand avatars from short monocular videos [8], [20]
or even a single image [9], [10], [63]. HARP [20] optimizes
hand avatars from mobile phone sequences. Besides, instead of
neural implicit models, it uses a mesh-based parametric hand
model with explicit texture representation for better robustness.
UHM [8] offers a universal solution for representing hand
meshes across arbitrary identities and adapting to individuals
with a phone scan. By integrating tracking and modeling
into a single stage, UHM [8] overcomes the error accu-
mulation problem, delivering more accurate results. Creating
hand avatars from a single image with limited information
requires strong prior knowledge. Handy [63], trained on a
diverse dataset of over 1,200 subjects, uses a GAN-based
approach to accurately reconstruct 3D hand shape, pose, and
textures. OHTA [9] separates hand representation into identity-
specific albedo and transferable geometry for realistic one-shot
reconstruction. InterGaussianHand [10] combines learning-
based features for cross-subject generalization and identity
maps for optimization, with an interaction-aware attention
module and Gaussian refinement for improved interacting hand
rendering.
Moreover, from the perspective of perturbation modeling,
current methods mainly focus on hand-object interactions
[64]–[66], while other challenging factors like illumination
changes remain relatively under-explored. Besides, methods
of these type usually leverage image or video diffusion mod-
els to synthesize animated hands, rather than optimizing an
explicit 3D representation via differentiable rendering. For
example, [67] trains a hand image generation model using
approximately 10M images. While such data-driven generative
pipelines can produce visually plausible results, they typically
incur substantial training costs and may suffer from inconsis-
tency or the Janus issue due to the lack of explicit geometric
constraints.
Despite the above significant progress in monocular hand
reconstruction, a substantial gap remains between current
methodologies and their applicability in real-world environ-
ments, as they assume near-ideal capturing conditions. On
the other hand, we aim at addressing scenarios where diverse
perturbations are present, thereby facilitating more robust and
real-life applications.
III. METHODOLOGY
In this section, we present the details of the proposed
WildGHand framework (Sec. III-B) for reconstructing hand
avatars from monocular videos that exhibit challenging factors,
such as hand-object obstructions, complex poses, motion blur,
and illumination variations. Specifically, WildGHand intro-
duces two core components to improve the robustness of the
inverse renderings of hands against perturbations caused by
the above factors: (i) a dynamic perturbation disentanglement
paradigm that explicitly represents perturbations as tempo-
rally weighted biases of 3D Gaussians (Sec. III-C); and (ii)
a perturbation-aware optimization strategy (Sec. III-D) that
provides anisotropic per-frame weighted masks for fitting 3D
Gaussians to videos. These two components enable our method
to achieve a better balance between underfitting dynamic
hands and overfitting to perturbations, thereby facilitating the
generation of high-fidelity personalized hand avatars.
A. Preliminary
We adopt 3D Gaussian splatting (3DGS) [2] to implement
our differentiable renderer, which is an efficient rasterization-
based rendering method that projects a set of 3D Gaussian
spheres onto the image plane. Each Gaussian sphere g =
[p, o, q, s, c] is governed by multiple attributes, i.e., a center
position p ∈R3, an opacity value o ∈R, a quaternion q ∈R4
representing rotation, an anisotropic scaling vector s ∈R3,
and a color value c ∈R3. Conventional methods [10], [26],
[68] determine the number and the attributes of 3D Gaussians
by fitting the input video directly, which are inevitably con-
taminated by noise and hand-irrelevant information. Therefore,
instead of adopting this naive fitting strategy, we propose a
self-adaptive optimization paradigm in WildGHand to handle
this challenge.
B. Overall Framework
Task formulation. Given a short monocular hand video,
our goal is to reconstruct an animatable hand avatar capable
of rendering high-quality images under arbitrary poses and
viewpoints. To this end, we adopt the MANO-HD hand model
[5] that parameterizes the SE(3) transformations of a 3D
hand mesh template by a hand pose vector θ ∈R16×3, a
hand shape vector β ∈R10, and a translation vector t ∈R3
describing the global position of the hand. Let h = [θ, β, t]
denote the concatenation of these parameters. With h and
a camera parameter vector d ∈R25 describing the intrinsic
and extrinsic camera properties [69], the hand avatar can be
formulated within the differentiable rendering framework, i.e.,
R(h, d) = I, where I ∈RH×W ×3 denotes the rendered
RGB image with resolution H × W. For clarity, here we
assume the given video contains only a single hand; however,
our framework can be seamlessly extended to scenarios of
interacting hands by independently processing each hand.
Unlike existing methods [3], [8]–[10] that are designed for
videos captured in clean and static environments, we aim at in-
the-wild hand videos with unknown perturbations that hinder
the reconstruction process.
The overall framework of WildGHand is shown in Fig. 2,
which adopts the per-scene optimization paradigm following
[9], [10]. Given a training frame Il sampled from a video
comprising L frames, l ∈{1, . . . , L}, we employ an off-
the-shelf estimator [70] to obtain the hand parameters h and
the camera parameter vector d. The hand parameters are
used to deform the 3D hand mesh template of MANO-HD
[5] and generate a posed hand mesh. Let V, V ∈RN×3
denote the N = 49,281 vertices of the template and the
posed mesh, respectively. We initialize a set of 3D Gaussians
G = {gn | n = 1, . . . , N} centered at V, and estimate their
attributes iteratively by jointly optimizing a latent identity map
and a Gaussian prediction network. This optimization ensures
that the rendered image R(h, d) approximates Il. The latent
identity map m ∈RH×W ×D has D channels and is used

<!-- page 5 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
5
Sampled Frame
Parameterized 
Mesh Estimator
Hand Mesh
Identity Map
Identity Encoder 
& Decoder
Geometry Encoder 
& Decoder
Texture Features
Geometry Features
Gaussian 
Decoder
3D Gaussians
[ ', , , , 
]
g
p
o q s c x
=
⋅
Gaussian Attribute Estimation Network
3D Gaussians Biases  
Frame Index l
MLP
γ(l)
Temporal 
Embedding
MLP
MLP
lw .
Temporal 
Weight

Dynamic Perturbation Disentanglement
g
D
Sampled Frame
Calculate Loss 
Assign Anisotropic
Loss Weights
Weighted Mask
Rendered Frame
Perturbation-aware Optimization
Training and Inference Flow
Training-only Flow

Element-wise Add
γ Encoding
( )⋅
γ
Splatting
Fig. 2. The proposed WildGHand framework. Given a monocular video affected by perturbations, WildGHand introduces two key components to achieve the
robust estimation of 3D Gaussians, including a lightweight dynamic perturbation disentanglement (DPD) module and a perturbation-aware optimization (PAO)
strategy. The DPD module represents potential perturbations by biases of Gaussian attributes, which are optimized guided by the weighted masks predicted
by the PAO strategy. During inference, the optimized biases are removed to render perturbation-free images.
to encode the characteristics of the target hand, while the
Gaussian prediction network f is pretrained on the large-scale
InterHand2.6M dataset [7] to leverage cross-subject priors.
Specifically, f takes as input geometry features extracted
from V and texture features extracted from m, and predicts
per-Gaussian attributes. f is implemented by a dual-branch
encoder–decoder backbone followed by four MLP-based pre-
diction heads. The backbone architecture is identical to that
proposed in our previous work [10], which separately extracts
geometry features xg from V and texture features xt from m.
The four prediction heads are designed for distinct purposes:
(i) the first head predicts the Gaussian attributes g based on
xg and xt; (ii) the second head predicts a shadow coefficient
ξ ∈(0, 1) from xg to modulate the color attribute in g as
c · ξ, which improves the disentanglement of shadow and
albedo for rendering; (iii) the third head combines xt with
positional encoding γ [1] to predict offsets for the template
V, allowing more flexible deformations to fit various hands;
and (iv) the fourth head refines the hand parameters h by
estimating the corresponding biases ∆h conditioned on xg and
xt. Subsequently, we update the center of each 3D Gaussian
for the next optimization step as p′ = pg+v′, where pg denotes
the center position predicted by the first head and v′ denotes
the corresponding mesh vertex derived from the predictions of
the third and fourth heads.
To model potential perturbations in the input video, we
assume that they can be represented as additive biases to the
predicted Gaussian attributes. Specifically, the per-Gaussian
attributes for a training frame are defined as
˜g = g + ∆g,
(1)
where ∆g denotes the biases. To estimate ∆g effectively, we
introduce a lightweight module in parallel to the Gaussian
prediction network and devise a perturbation-aware optimiza-
tion strategy to guide its optimization. The details of these
two components are elaborated in the subsequent sections.
Our design is based on the observation that mesh-based hand
representations are relatively stable across frames, while per-
turbations are abrupt and noisy. Hence, the Gaussian prediction
network, with greater modeling capacity, is better suited to
capture the canonical component g, while the lightweight mod-
ule focuses on modeling ∆g. Consequently, we can optimize ˜g
during per-scene optimization while employing g for inference,
thereby mitigating the trade-off between overfitting to pertur-
bations and underfitting the underlying hand appearance.
C. Dynamic Perturbation Disentanglement
Considering that perturbations are time-varying and chal-
lenging to model using static representations, we introduce
a dynamic perturbation disentanglement (DPD) module that
integrates temporal weights into the estimation of Gaussian at-
tribute biases. The DPD module adopts a shallow architecture
consisting of three MLP-based blocks, thereby preventing it
from overpowering the Gaussian attribute prediction network.
Formally, inspired by the positional encoding mechanism
introduced in [1], the DPD module first computes a high-
dimensional positional encoding γ(l) of the frame index l
to facilitate fine-grained temporal modeling. An MLP-based
encoder is then applied to obtain the temporal embedding zl
as follows:
zl = ϕ(γ(l)),
(2)
γ(l) =

sin(20πl/L), cos(20πl/L), . . . ,
sin(2Kπl/L), cos(2Kπl/L)

,
(3)
where K is the user-specified encoding length, and we set
K = 9 in this paper. The temporal embedding zl serves two
purposes. First, zl is concatenated with the predicted Gaussian
attributes gl (with the Gaussian/vertex index in gl omitted for
brevity) to predict the biases. Second, zl is used to predict a
global scaling factor ωl ∈(−1, 1) that controls the influence
of the predicted biases. This can be formulated as
∆gl = ωl · φ([zl, gl]),
ωl = 2σ(ψ(zl)) −1,
(4)

<!-- page 6 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
6
Target Image
Semantic Mask
Backgroud Mask
Hand Area
Loss Weight
0.0000
-0.0066
0.0155
0.0247
0.0334
Temporal Weight
0.0000
0.0029
-0.0056
-0.0064
-0.0087
0.0000
0.0000
-0.0093
-0.0101
-0.0017
Fig. 3. Illustration of the proposed PAO and DPD modules. Left: Our perturbation-aware optimization (PAO) strategy segments the hand regions and leverages
reconstruction error to generate weighted masks to guide the optimization of 3DGS. Right: The temporal weights estimated by our dynamic perturbation
disentanglement (DPD) module that reflect the strengths of perturbations. Partial perturbations (e.g., occlusions) tends to have smaller weights (labeled in
green), while holistic perturbations (e.g., motion blur) have larger weights (labeled in red).
where φ and ψ are each implemented via an MLP, and σ
denotes the sigmoid function. Furthermore, we randomly set
ωl to 0 with a probability of 30% so that the Gaussian attribute
prediction network can fit each frame independently, which
encourages the network to capture the principal components
of the Gaussian attributes.
Discussion. Compared with existing methods for deformable
3D Gaussians and 4D Gaussians [71], [72], the role of our
temporally weighted Gaussian attribute biases is fundamen-
tally different. Most existing methods utilize biases to facilitate
more flexible deformations; that is, larger bias magnitudes
are often beneficial. In contrast, our approach uses biases
to explicitly model and mitigate perturbations, and therefore
favors smaller bias magnitudes. This behavior is supported
by Fig. 3, which visualizes the learned temporal weights for
frames affected by occlusions and motion blur. We observe
that the temporal weights are proportional to perturbation
strengths. For example, at the bottom of Fig. 3, the weight
of the frame with the most severe perturbations (last column)
is approximately five times larger than that of a clean frame
(second column). Moreover, the absolute values of these
temporal weights remain small (ranging from 0.0066 to 0.034),
which is reasonable because the temporal weights are used to
scale the bias terms rather than govern the main components
of the Gaussian attributes.
D. Perturbation-aware Optimization
In addition to the DPD module, which facilitates temporal
perturbation modeling, we propose a perturbation-aware opti-
mization (PAO) strategy that spatially identifies intra-frame
perturbations, thereby enabling comprehensive suppression
across both spatial and temporal dimensions. The core in-
tuition behind PAO is that regions affected by perturbations
are inherently more difficult to synthesize; hence, areas with
low rendering quality are more likely to be corrupted by
perturbations. Therefore, we adaptively reduce the loss weights
assigned to these regions during optimization to mitigate their
adverse impact.
Specifically, we utilize the Segment Anything Model (SAM)
[51] to segment the input image Il into U regions, and we let
Y = {yu | u = 1, . . . , U} denote the corresponding binary
masks. One advantage of SAM is that it is promptable, which
allows us to use the hand root as a prompt to conveniently
locate the hand. Let yh denote the mask of the target hand.
We assign an anisotropic loss weight to each region as
λu = α · κ(TE −Eu) · κ(µu −Tµ) · (1/(1 + β · ωl)).
(5)
Here, α and β are user-defined scaling factors. κ denotes the
ReLU function, where κ(x) = x if x > 0 and 0 otherwise.
TE and Tµ are learnable thresholds. Eu = ∥(Il −I) ⊙yu∥1
is the mean ℓ1 loss over the u-th region, where ⊙denotes
the element-wise product. µu is the fraction of region u
that belongs to the hand foreground, computed as µu =
∥yu ⊙yh∥0 / ∥yh∥0. Therefore, Eq. (5) can be viewed as a
piecewise function that selects hand regions (via κ(µu −Tµ))
with acceptable rendering quality (via κ(TE −Eu)). Addition-
ally, we incorporate the temporal weight ωl of the given frame
into Eq. (5) to modulate λu.
Consequently, the overall weighted mask that combines all
regions is defined as
W =
U
X
u=1
λu · yu + (1 −yh) · κ(Tb −∥Il −I∥1),
(6)
where the second term in Eq. (6) controls the weight of
the background, and Tb is also a learnable threshold. Fig. 3
shows a scene with hand occlusions, where the proposed PAO
strategy assigns higher weights to visible hand regions.
E. Optimization Objectives
We employ the weighted mask W computed by the PAO
strategy to optimize WildGHand by minimizing the following
loss function:
L = W ⊙Lrec(Il, I) + Lreg,
(7)

<!-- page 7 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
7
TABLE II
DETAILED DESCRIPTION OF THE PROPOSED HWP DATASET, COVERING HAND TYPE, PRIMARY ACTION, RECORDING DEVICE, NUMBER OF FRAMES, AND
THE PRESENCE OF PERTURBATIONS.
Capture
Hand
Action
Capture Device #Frames
Perturbations
Interaction Complex Poses Illumination Changes Motion Blur
0
Single
Grabbing blocks
Mobile phone
901
✓
✓
1
Single
Multiple gestures
Mobile phone
742
✓
✓
2
Single
Multiple gestures
Mobile phone
828
✓
3
Single
Spinning a pen
Mobile phone
1086
✓
✓
4
Interacting
Spinning a pen
Webcam
1662
✓
✓
5
Interacting
Sticking stickers
Webcam
1216
✓
✓
6
Interacting
Applying lotion
Webcam
1524
✓
✓
7
Interacting
Multiple gestures
Webcam
1360
✓
✓
8
Interacting
Putting on a watch
Webcam
1444
✓
✓
✓
9
Interacting Scrolling on a phone Mobile phone
1533
✓
✓
✓
10
Interacting
Shuffling cards
Mobile phone
1525
✓
✓
✓
where Lrec denotes the reconstruction loss [10] between the in-
put frame Il and the rendered image I. Lreg is a regularization
term consisting of the following components:
Lreg =
N
X
n=1
λbias∥∆gn∥1 + λξ∥ξn −1∥2 + λo∥on −1∥2
+ λLapLLap −λW∥W∥1.
(8)
Here, λbias, λξ, λo, λLap, and λW are user-defined loss
weights. The first three terms of Eq. (8) impose point-wise
regularization on the Gaussian attribute bias ∆g, the shadow
coefficient ξ, and the opacity o, respectively. LLap is a Lapla-
cian regularizer [26] that improves the connectivity between
vertices. Finally, we incorporate the term ∥W∥1 into the
objective to prevent trivial weight assignments (e.g., all region
weights being set to zero). With these weighted objectives, we
can dynamically adjust the contribution of each region during
optimization and thereby ensure accurate reconstruction of the
input video.
During training on interacting-hand sequences, we introduce
an additional ℓ1 regularization term that enforces consistency
between the texture features of the left and right hands:
Lcross = ∥Tleft −Tright∥1,
(9)
where Tleft and Tright denote the extracted texture features
of the left and right hands, respectively. This term leverages
the inherent bilateral symmetry of human hands, allowing
visible regions of one hand to provide informative cues for
reconstructing occluded regions on the other. Consequently,
even without explicit joint modeling, the system can robustly
infer missing textures and maintain appearance consistency
under inter-hand occlusions.
For clarity, we primarily focus on the single-hand setting in
the main formulation, and the cross-hand consistency term is
included only during training on interacting-hand sequences.
IV. HWP DATASET
As shown in Table I, most existing hand video datasets
exhibit notable limitations, such as restricted interaction di-
versity, inadequate coverage of real-world perturbations, or
the absence of clean and reliable test subsets. Therefore, we
Fig. 4. Visual examples of our HWP dataset, which covers diverse challenging
scenes like motion blur, hand-object interactions, illumination variations, and
complex poses (from top left to bottom right).
introduce the HWP dataset, a monocular hand video dataset
captured in unconstrained environments to overcome these
issues.
Specifically, videos on HWP are recorded with either a fixed
single camera or a handheld mobile phone. We target four
dominant perturbation types: (i) hand–object interaction, (ii)
complex poses, (iii) illumination change, and (iv) motion blur.
To increase realism and interaction diversity, we consider daily
actions and gestures such as spinning a pen or applying hand
cream. In total, our dataset contains four single-hand and seven
interacting-hand sequences, with over 13.8K frames.
Table II summarizes the composition of the proposed HWP
dataset, including hand type (single or interacting), primary
action category, recording device (fixed camera or handheld
mobile phone), number of frames, and associated perturbation
types. Representative visual samples from different sequences
are shown in Figure 4.
In this way, the proposed HWP dataset encompasses a
broad spectrum of systematically categorized perturbations
under clearly categorized conditions while maintaining an
isolated, clean test subset for objective benchmarking. This
design enables both fine-grained evaluation under specific
challenges and fair cross-method comparison in realistic, in-
the-wild scenarios.
V. EXPERIMENTS
In this section, we validate the effectiveness of the proposed
WildGHand framework through extensive experiments. We
conduct evaluations on publicly available datasets as well as

<!-- page 8 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
8
TABLE III
COMPARISON WITH STATE-OF-THE-ART METHODS ON THE THREE DATASET, INCLUDING WILDGHAND, INTERHAND2.6M AND ANCHORCRAFTER.
“SINGLE” AND “INTER.” DENOTE SINGLE-HAND AND INTERACTING-HAND SCENARIOS, RESPECTIVELY.
Method
WildGHand (Single)
WildGHand (Inter.)
InterHand2.6M (Inter.)
AnchorCrafter (Inter.)
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
UHM [8]
20.24
0.867
0.203
–
–
–
–
–
–
–
–
–
Handy [63]
22.99
0.874
0.127
24.10
0.864
0.141
24.34
0.867
0.198
25.88
0.912
0.074
InterGaussianHand [10]
23.06
0.858
0.147
24.69
0.851
0.148
26.70
0.861
0.173
24.28
0.912
0.098
WildGHand (Ours)
26.24
0.916
0.116
26.29
0.893
0.122
28.25
0.894
0.151
27.68
0.948
0.072
TABLE IV
COMPARISON ON ONLINE VIDEOS.
Method
PSNR↑
SSIM↑
LPIPS↓
Handy [63]
21.46
0.809
0.191
InterGaussianHand [10]
22.95
0.820
0.157
WildGHand (Ours)
27.39
0.920
0.080
self-collected and online videos that cover a wide range of
in-the-wild perturbations.
A. Datasets
We evaluate the proposed method on our HWP dataset,
InterHand2.6M [7], the AnchorCrafter dataset [73], and in-the-
wild online videos. For each video, we use 80% of the frames
for optimization, 10% for validation, and 10% for testing.
Details of the three datasets other than HWP are summarized
below:
InterHand2.6M.
InterHand2.6M
[7]
is
a
widely
used
in-lab
benchmark
for
hand–hand
interactions.
We
evaluate
on
three
sequences:
Capture21/0390_dh_touchROM/cam400015,
Capture21/0390_dh_touchROM/cam400016,
and
Capture21/0286_handscratch/cam400016.
AnchorCrafter dataset. The AnchorCrafter dataset [73]
provides in-the-wild human–object interaction videos. We
use the sequence tune/1_0 to assess performance on
hand–object interaction scenarios.
Online videos. To further evaluate generalization, we test
on hand videos collected from online platforms, which feature
diverse lighting conditions, camera motions, backgrounds, and
occlusions.
B. Setup
Implementation Details. Following [10], we pretrain the
Gaussian attribute prediction network on the large-scale Inter-
Hand2.6M dataset [7] (licensed under CC BY-NC 4.0). For
each video, we optimize our model on two NVIDIA A6000
GPUs with a batch size of 4 for 50 epochs using the Adam
optimizer [74]. The initial learning rate is set to 1 × 10−6
for the learnable thresholds and to 1×10−4 for the remaining
parameters; both rates are decayed by a factor of 0.5 every five
epochs. The feature dimension of the latent identity map D is
set to 33. The loss weights in Eq. (8) are set to λbias = 0.1,
λξ = 0.001, λo = 0.1, λLap = 0.1, and λW = 0.5. The
initial learnable thresholds for interacting-hand videos are set
to TE = 1.0, Tµ = 0.3, and Tb = 10.0, while those for single-
hand videos are set to TE = 1.0, Tµ = 0.3, and Tb = 3.0.
Baselines and Metrics. We compare WildGHand with three
pioneering hand avatar methods, including UHM [8], Handy
[63], and InterGaussianHand [10]. All methods are trained
and evaluated under the same experimental setting based on
their open-source implementations. Among these methods,
both UHM and Handy adopt their own hand models. However,
UHM only provides its right-hand model fitted on its own
dataset, which is not compatible with MANO-HD and hard to
convert. Hence, we report only the single-hand performance
for UHM. As for Handy, it is compatible with MANO-HD, and
hence we convert it accordingly and optimize the latent vectors
of its texture model using differentiable rendering. Following
[8], [10], [20], [63], we evaluate the quality of rendered images
using PSNR [75], LPIPS [76], and SSIM [77].
C. Comparison with State-of-the-art Methods
Quantitative Comparison. Table III reports quantitative
results of our method and state-of-the-art approaches on HWP,
InterHand2.6M, and AnchorCrafter. The proposed WildGHand
framework consistently outperforms all baselines across all
metrics in both single-hand and interacting-hand settings,
demonstrating its effectiveness and robustness. Table IV fur-
ther reports results on Internet videos, highlighting the strong
generalization ability of our method under real-world condi-
tions.
Our previous InterGaussianHand is designed for one-shot
reconstruction and relies on well-captured images and accurate
hand mesh estimation. As a result, it is highly susceptible to
perturbations in target frames, which leads to a substantial
performance drop. Similarly, Handy integrates explicit hand
meshes with implicit texture optimization and is therefore sen-
sitive to perturbations; moreover, its texture optimization tends
to overfit to irrelevant details, further degrading performance.
Because UHM is incompatible with MANO-HD, it relies
on off-the-shelf estimators to obtain foreground masks and
joint coordinates and then optimizes its animation objective.
However, this process is often compromised by perturbations,
resulting in unstable fitting and degraded performance. In
contrast, by incorporating the DPD module and the PAO
strategy, WildGHand effectively mitigates these issues and
produces high-quality results.
Qualitative Comparison. Fig. 5 shows qualitative compar-
isons between our approach and the baseline methods on both

<!-- page 9 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
9
GT
WildGHand InterGaussianHand
Handy
UHM
Input Video
GT
WildGHand InterGaussianHand
Handy
UHM
Input Video
GT
WildGHand
InterGaussianHand
GT
WildGHand
InterGaussianHand
Handy
Handy
Input Video
Input Video
Fig. 5.
Qualitative comparisons between our proposed WildGHand model with state-of-the-art methods on interacting-hand videos (top) and single-hand
videos (bottom).
TABLE V
COMPARISON WITH STATE-OF-THE-ART METHODS ON THE INTERACTING-HAND VIDEOS OF THE HWP DATASET, EVALUATED ACROSS FOUR SCENARIOS:
HAND-OBJECT INTERACTIONS, COMPLEX POSES, ILLUMINATION VARIATIONS, AND MOTION BLUR.
Method
Hand-object Interactions
Complex Poses
Illumination Variations
Motion Blur
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Handy
24.67
0.872
0.138
24.14
0.863
0.138
20.65
0.814
0.156
24.33
0.872
0.139
InterGaussianHand
25.11
0.858
0.146
24.51
0.848
0.142
22.18
0.805
0.158
24.71
0.860
0.145
WildGHand (ours)
26.87
0.902
0.118
25.29
0.881
0.124
22.81
0.839
0.148
26.68
0.901
0.116
interacting-hand and single-hand videos. InterGaussianHand
fails to generate accurate hand geometry and appearance in
the presence of perturbations. Without a dedicated mecha-
nism for handling interference and occlusions, it is heavily
influenced by moving objects and motion blur, leading to
suboptimal results. Handy, while capable of recovering basic
hand geometry and a reasonable appearance, fails to capture
fine-grained identity cues. Although its texture optimization in
the pretrained latent space is relatively stable, it is still misled
by perturbations, compromising the final results. For instance,
the skin tones synthesized by Handy deviate significantly from
the ground truth, particularly in interacting-hand scenarios.
UHM is affected by unreliable foreground masks and joint
coordinates, producing distorted and unnatural hand structures
(e.g., the first row of the single-hand results). In contrast,
WildGHand successfully reconstructs intricate hand details,
such as nails, wrinkles, and veins, across diverse views and
poses. These qualitative comparisons further validate the ro-
bustness and strong performance of our approach in generating
accurate and realistic hand avatars.

<!-- page 10 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
10
GT
Full
w/ DPD
Baseline
GT
Full
Weight Mask
w/o PAO
Input
Fig. 6. Visual example of the ablation study on the proposed modules. Left: Weight masks learned by the proposed PAO strategy, light hand regions indicate
higher weights. Right: Results of the variants of the proposed method. The full model obtains the best results consistently on various videos.
TABLE VI
ABLATION STUDY ON THE PROPOSED MODULES.
DPD PAO
Single Hand
Interacting Hands
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
22.65
0.862
0.151
24.86
0.867
0.123
✓
23.00
0.869
0.138
25.01
0.868
0.122
✓
✓
26.24
0.916
0.116
26.29
0.893
0.122
D. Ablation Study
We conduct ablation studies on our dataset to validate
the effectiveness of the proposed DPD module and the PAO
strategy. We consider the Gaussian attribute prediction network
as the baseline.
Effectiveness of DPD. Our lightweight DPD module has
75k learnable parameters, roughly 0.018% of the whole model
(400M). Despite its tiny capacity, DPD still boosts the perfor-
mance of the baseline (Table VI) and addresses a few artifacts
like floaters and broken structures (Figure 6).
Effectiveness of PAO. As reported in Table VI, the per-
formance gains achieved with PAO are more substantial than
those by DPD. This is reasonable, as merely increasing net-
work capacity without a proper optimization goal does not
necessarily guarantee enhanced robustness against perturba-
tions. Figure 6 also shows the weighted masks predicted by
PAO, which clearly indicate the areas of artifacts generated
by the baseline. Guided by these weighted masks, the full
WildGHand model avoids overfitting to perturbations and
successfully eliminates artifacts in the rendered images. We
also notice that the performance gains in single-hand scenarios
are significantly greater than those in interacting-hand cases
(e.g., a 23.1% relative reduction in the LPIPS score is observed
in the former). This is caused by the increased difficulty
of preserving hand information in single-hand videos when
perturbations occur, which the proposed method manages to
tackle.
Robustness to perturbation. Table V compares WildG-
Hand with two state-of-the-art methods, Handy and Inter-
GaussianHand, under four challenging scenarios: hand-object
interactions, complex poses, illumination variations, and mo-
tion blur. WildGHand consistently outperforms both baselines,
achieving the best PSNR, SSIM, and lowest LPIPS across all
settings. These results demonstrate WildGHand’s robustness
and effectiveness across diverse real-world perturbations.
Binary mask vs. PAO. We compare our perturbation-aware
occlusion (PAO) strategy with a SAM-based binary mask. As
shown in Table VII, binary masks lack geometric constraints
and fail to reliably distinguish occluders from the background
under challenging in-the-wild conditions. In contrast, PAO
adaptively down-weights unreliable regions and preserves in-
formative hand areas, resulting in improved reconstruction
quality.
Effectiveness of prediction heads. Our network employs
four prediction heads (Sec. III-B): (i)–(ii) estimate the core
Gaussian attributes (e.g., color, position, shading), which are
standard in 3D Gaussian Splatting models; (iii) predicts vertex
offsets to capture shape and pose variations; and (iv) refines

<!-- page 11 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
11
TABLE VII
COMPARISON BETWEEN PAO AND SAM-BASED BINARY MASK.
Method
PSNR↑
SSIM↑
LPIPS↓
WildGHand (ours)
26.44
0.871
0.179
WildGHand w/ binary mask
25.36
0.844
0.184
TABLE VIII
ABLATION ON PREDICTION HEADS.
Method
PSNR↑
SSIM↑
LPIPS↓
WildGHand (ours)
26.44
0.871
0.179
w/o Heads (iii) & (iv)
26.20
0.864
0.177
hand-specific parameters via residual correction. As shown in
Table VIII, removing heads (iii) and (iv) leads to measurable
degradation in both reconstruction accuracy and perceptual
quality.
E. Bias strength and interpretation.
As stated in Sec. III-C, our biases model frame-specific
perturbations. Consequently, larger bias magnitudes imply
stronger perturbations and are not favored. This can be val-
idated by Figure 3, where temporal weights correlate with
visual disturbance. We further report average temporal weights
across captures of different noise levels in Table IX. It is clear
that higher perturbation levels lead to larger weights and lower
image quality, confirming our model’s adaptive behavior.
VI. CONCLUSION
In this paper, we present WildGHand for reconstructing
high-fidelity hand avatars from monocular in-the-wild videos
under diverse perturbations. WildGHand introduces a dynamic
perturbation disentanglement module that models perturba-
tions as temporally weighted biases of 3D Gaussian attributes
during optimization, and removes these biases at inference
to mitigate the risk of overfitting to perturbations. Further-
more, WildGHand introduces a perturbation-aware optimiza-
tion strategy that generates anisotropic weighted masks to un-
reliable regions and provides robust supervision. We evaluate
WildGHand on two public datasets and our collected dataset
covering both single-hand and interacting-hand scenarios. Re-
sults demonstrate that WildGHand consistently produces high-
quality hand avatars and outperforms state-of-the-art methods
across multiple qualitative and quantitative metrics.
REFERENCES
[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[2] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM TOG, vol. 42,
no. 4, pp. 139–1, 2023.
[3] X. Huang, H. Li, Z. Yang, Z. Wang, and X. Liang, “3d visibility-aware
generalizable neural radiance fields for interacting hands,” in AAAI,
vol. 38, no. 3, 2024, pp. 2400–2408.
TABLE IX
AVERAGE TEMPORAL WEIGHT VS. PERTURBATION LEVEL. HIGHER
PERTURBATION CORRELATES WITH LARGER WEIGHTS AND DEGRADED
QUALITY. WEIGHTS ARE SCALED BY 10−3 FOR READABILITY.
Capture
Disturbance
PSNR ↑
SSIM ↑
LPIPS ↓
Avg. T. Wt. (×10−3) ↑
4
Low
29.59
0.931
0.115
0.234
5
Medium
28.00
0.915
0.116
1.581
6
High
27.83
0.907
0.123
7.572
TABLE X
MEAN AND STANDARD DEVIATION OF THE PERFORMANCE OF
INTERGAUSSIANHAND AND WILDGHAND.
Method
PSNR↑
SSIM↑
LPIPS↓
InterGaussianHand
26.2518±0.0770
0.8723±0.0009
0.1582±0.0012
WildGHand (ours)
29.5859±0.1028
0.9313±0.0015
0.1146±0.0029
[4] A. Mundra, J. Wang, M. Habermann, C. Theobalt, M. Elgharib et al.,
“Livehand: Real-time and photorealistic neural hand rendering,” in
ICCV, 2023, pp. 18 035–18 045.
[5] X. Chen, B. Wang, and H.-Y. Shum, “Hand avatar: Free-pose hand
animation and rendering from monocular video,” in CVPR, 2023, pp.
8683–8693.
[6] Z. Guo, W. Zhou, M. Wang, L. Li, and H. Li, “Handnerf: Neural radiance
fields for animatable interacting hands,” in CVPR, 2023, pp. 21 078–
21 087.
[7] G. Moon, S.-I. Yu, H. Wen, T. Shiratori, and K. M. Lee, “Interhand2.
6m: A dataset and baseline for 3d interacting hand pose estimation from
a single rgb image,” in ECCV.
Springer, 2020, pp. 548–564.
[8] G. Moon, W. Xu, R. Joshi, C. Wu, and T. Shiratori, “Authentic hand
avatar from a phone scan via universal hand model,” in CVPR, 2024,
pp. 2029–2038.
[9] X. Zheng, C. Wen, Z. Su, Z. Xu, Z. Li, Y. Zhao, and Z. Xue, “Ohta:
One-shot hand avatar via data-driven implicit priors,” in CVPR, 2024,
pp. 799–810.
[10] X. Huang, H. Li, W. Liu, X. Liang, Y. Yan, Y. Cheng, and C. GAO,
“Learning interaction-aware 3d gaussian splatting for one-shot hand
avatars,” in NeurIPS, 2024.
[11] R. Martin-Brualla, N. Radwan, M. S. M. Sajjadi, J. T. Barron, A. Doso-
vitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields for
unconstrained photo collections,” in CVPR, Jun 2021.
[12] S. Sabour, S. Vora, D. Duckworth, I. Krasin, D. Fleet, and A. Tagliasac-
chi, “Robustnerf: Ignoring distractors with robust losses,” in CVPR,
2023, pp. 20 626–20 636.
[13] S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas,
A. Jacobson, D. J. Fleet, and A. Tagliasacchi, “Spotlesssplats: Ignoring
distractors in 3d gaussian splatting,” ACM TOG, vol. 44, no. 2, pp. 1–11,
2025.
[14] P. Ungermann, A. Ettenhofer, M. Nießner, and B. Roessle, “Robust 3d
gaussian splatting for novel view synthesis in presence of distractors,”
in DAGM German Conference on Pattern Recognition.
Springer, 2024,
pp. 153–167.
[15] A. Markin, V. Pryadilshchikov, A. Komarichev, R. Rakhimov, P. Wonka,
and E. Burnaev, “T-3dgs: Removing transient objects for 3d scene
reconstruction,” arXiv preprint arXiv:2412.00155, 2024.
[16] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler,
“Wildgaussians: 3d gaussian splatting in the wild,” in NeurIPS, 2024.
[17] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov,
P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2:
Learning robust visual features without supervision,” Transactions on
Machine Learning Research Journal, pp. 1–31, 2024.
[18] Y.-W. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay,
Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield et al., “Dexycb: A
benchmark for capturing hand grasping of objects,” in CVPR, 2021, pp.
9044–9053.
[19] Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black,
and O. Hilliges, “ARCTIC: A dataset for dexterous bimanual hand-
object manipulation,” in Proceedings IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), 2023.
[20] K. Karunratanakul, S. Prokudin, O. Hilliges, and S. Tang, “Harp:

<!-- page 12 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
12
Personalized hand reconstruction from a monocular rgb video,” in CVPR,
2023, pp. 12 802–12 813.
[21] G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and
J. Malik, “Reconstructing hands in 3D with transformers,” in CVPR,
2024.
[22] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar,
J. Hamburger, H. Jiang, M. Liu, X. Liu et al., “Ego4d: Around the
world in 3,000 hours of egocentric video,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2022,
pp. 18 995–19 012.
[23] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu,
H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for
efficient and high-fidelity surface reconstruction,” IEEE TVCG, 2024.
[24] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu,
“Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-
time human novel view synthesis,” in CVPR, 2024, pp. 19 680–19 690.
[25] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in CVPR, 2024,
pp. 10 349–10 358.
[26] G. Moon, T. Shiratori, and S. Saito, “Expressive whole-body 3d gaussian
avatar,” in ECCV.
Springer, 2024, pp. 19–35.
[27] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, and A. Ranjan, “Neuman:
Neural human radiance field from a single video,” in ECCV.
Springer,
2022, pp. 402–418.
[28] C. Guo, T. Jiang, X. Chen, J. Song, and O. Hilliges, “Vid2avatar: 3d
avatar reconstruction from videos in the wild via self-supervised scene
decomposition,” in CVPR, 2023, pp. 12 858–12 868.
[29] T. Jiang, X. Chen, J. Song, and O. Hilliges, “Instantavatar: Learning
avatars from monocular video in 60 seconds,” in CVPR, 2023, pp.
16 922–16 932.
[30] K. Shen, C. Guo, M. Kaufmann, J. J. Zarate, J. Valentin, J. Song, and
O. Hilliges, “X-avatar: Expressive human avatars,” in CVPR, 2023, pp.
16 911–16 921.
[31] S. Liu, L. Gu, Z. Cui, X. Chu, and T. Harada, “I2-nerf: Learning
neural radiance fields under physically-grounded media interactions,”
in NeurIPS, 2025.
[32] Z. Li, D. Wang, K. Chen, Z. Lv, T. Nguyen-Phuoc, M. Lee, J.-B.
Huang, L. Xiao, Y. Zhu, C. S. Marshall et al., “Lirm: Large inverse
rendering model for progressive reconstruction of shape, materials and
view-dependent radiance fields,” in CVPR, 2025, pp. 505–517.
[33] S. Koch, J. Wald, M. Colosi, N. Vaskevicius, P. Hermosilla, F. Tombari,
and T. Ropinski, “Relationfield: Relate anything in radiance fields,” in
CVPR, 2025, pp. 21 706–21 716.
[34] L. Weijler, S. Koch, F. Poiesi, T. Ropinski, and P. Hermosilla, “Open-
hype: Hyperbolic embeddings for hierarchical open-vocabulary radiance
fields,” in NeurIPS, 2025.
[35] J. Cao, H. Wu, Z. Feng, H. Bao, X. Zhou, and S. Peng, “Universe:
Unleashing the scene prior of video diffusion models for robust radiance
field reconstruction,” in ICCV, 2025, pp. 27 031–27 041.
[36] J. Tang, F. Fei, Z. Li, X. Tang, S. Liu, Y. Chen, B. Huang, Z. Chen,
X. Wu, and B. Shi, “Spectre-gs: Modeling highly specular surfaces with
reflected nearby objects by tracing rays in 3d gaussian splatting,” in
CVPR, 2025, pp. 16 133–16 142.
[37] J. Guo, Y. Xin, G. Liu, K. Xu, L. Liu, and R. Hu, “Articulatedgs: Self-
supervised digital twin modeling of articulated objects using 3d gaussian
splatting,” in CVPR, 2025, pp. 27 144–27 153.
[38] Z. Gao, R. Yi, Y. Dai, X. Zhu, W. Chen, C. Zhu, and K. Xu, “Curve-
aware gaussian splatting for 3d parametric curve reconstruction,” in
ICCV, 2025, pp. 27 531–27 541.
[39] B. Kim, S. Saito, G. Nam, T. Simon, J. Saragih, H. Joo, and J. Li,
“Haircup: Hair compositional universal prior for 3d gaussian avatars,”
in ICCV, 2025, pp. 9966–9976.
[40] Y. Zhong, Z. Li, D. Z. Chen, L. Hong, and D. Xu, “Taming video
diffusion prior with scene-grounding guidance for 3d gaussian splatting
from sparse inputs,” in CVPR, 2025, pp. 6133–6143.
[41] K. Schwarz, N. Mueller, and P. Kontschieder, “Generative gaussian
splatting: Generating 3d scenes with video diffusion priors,” in ICCV,
2025, pp. 27 510–27 520.
[42] J. Li, Z. Song, and B. Yang, “Trace: Learning 3d gaussian physical
dynamics from multi-view videos,” in ICCV, 2025, pp. 8820–8829.
[43] H. Go, B. Park, H. Nam, B.-H. Kim, H. Chung, and C. Kim, “Vide-
orfsplat: Direct scene-level text-to-3d gaussian splatting generation with
flexible pose and multi-view joint modeling,” in ICCV, 2025, pp. 26 706–
26 717.
[44] H. Zhao, H. Wang, X. Zhao, H. Fei, H. Wang, C. Long, and H. Zou,
“Physsplat: Efficient physics simulation for 3d scenes via mllm-guided
gaussian splatting,” in ICCV, 2025, pp. 5242–5252.
[45] W. Li, R. Zhou, J. Zhou, Y. Song, J. Herter, M. Qin, G. Huang, and
H. Pfister, “4d langsplat: 4d language gaussian splatting via multimodal
large language models,” in CVPR, 2025, pp. 22 001–22 011.
[46] W. Zhang, J. Zhou, H. Geng, W. Zhang, and Y.-S. Liu, “Gap: Gaussian-
ize any point clouds with text guidance,” in ICCV, 2025, pp. 25 627–
25 638.
[47] Z. Ma, H. Li, Z. Xie, X. Luo, M. Kampffmeyer, F. Gao, and X. Liang,
“Ergo: Excess-risk-guided optimization for high-fidelity monocular 3d
gaussian splatting,” arXiv preprint arXiv:2602.10278, 2026.
[48] D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kaza-
kos, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray, “Scaling
egocentric vision: The epic-kitchens dataset,” in European Conference
on Computer Vision (ECCV), 2018.
[49] J. Chen, Y. Qin, L. Liu, J. Lu, and G. Li, “Nerf-hugs: Improved neural
radiance fields in non-static scenes using heuristics-guided segmenta-
tion,” in CVPR, 2024, pp. 19 436–19 446.
[50] T. Wu, F. Zhong, A. Tagliasacchi, F. Cole, and G. Research, “D 2
nerf: Self-supervised decoupling of dynamic and static objects from a
monocular video,” NeurIPS, vol. 35, pp. 32 653–32 666, 2022.
[51] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick,
“Segment anything,” in ICCV, 2023, pp. 4015–4026.
[52] Y. Wang, J. Wang, and Y. Qi, “We-gs: An in-the-wild efficient 3d gaus-
sian representation for unconstrained photo collections,” arXiv preprint
arXiv:2406.02407, 2024.
[53] J. Xu, Y. Mei, and V. M. Patel, “Wild-gs: Real-time novel view synthesis
from unconstrained photo collections,” NeurIPS, vol. 37, pp. 103 334–
103 355, 2024.
[54] D. Zhang, C. Wang, W. Wang, P. Li, M. Qin, and H. Wang, “Gaussian
in the wild: 3d gaussian splatting for unconstrained image collections,”
in ECCV.
Springer, 2024, pp. 341–359.
[55] S. Sun, C. Zhao, Z. Sun, Y. V. Chen, and M. Chen, “Splatflow: Self-
supervised dynamic gaussian splatting in neural motion flow field for
autonomous driving,” in CVPR, 2025, pp. 27 487–27 496.
[56] J. Kwon, H. Cho, and J. Kim, “Efficient dynamic scene editing via 4d
gaussian-based static-dynamic separation,” in CVPR, 2025, pp. 26 855–
26 865.
[57] Y. Yao, Z. Deng, and J. Hou, “Riggs: Rigging of 3d gaussians for
modeling articulated objects in videos,” in CVPR, 2025, pp. 5592–5601.
[58] Z. Luo, H. Ran, and L. Lu, “Instant4d: 4d gaussian splatting in minutes,”
in NeurIPS, 2025.
[59] J. Park, M.-Q. V. Bui, J. L. G. Bello, J. Moon, J. Oh, and M. Kim,
“Splinegs: Robust motion-adaptive spline for real-time dynamic 3d
gaussians from monocular video,” in CVPR, 2025, pp. 26 866–26 875.
[60] J. Lei, Y. Weng, A. W. Harley, L. Guibas, and K. Daniilidis, “Mosca:
Dynamic gaussian fusion from casual videos via 4d motion scaffolds,”
in CVPR, 2025, pp. 6165–6177.
[61] Z. Gao, B. Planche, M. Zheng, A. Choudhuri, T. Chen, and Z. Wu,
“7dgs: Unified spatial-temporal-angular gaussian splatting,” in ICCV,
2025, pp. 26 316–26 325.
[62] S. Iwase, S. Saito, T. Simon, S. Lombardi, T. Bagautdinov, R. Joshi,
F. Prada, T. Shiratori, Y. Sheikh, and J. Saragih, “Relightablehands:
Efficient neural relighting of articulated hand models,” in CVPR, 2023,
pp. 16 663–16 673.
[63] R. A. Potamias, S. Ploumpis, S. Moschoglou, V. Triantafyllou, and
S. Zafeiriou, “Handy: Towards a high fidelity 3d hand shape and
appearance model,” in CVPR, 2023, pp. 4670–4680.
[64] Y. Pang, R. Shao, J. Zhang, H. Tu, Y. Liu, B. Zhou, H. Zhang, and Y. Liu,
“Manivideo: Generating hand-object manipulation video with dexterous
and generalizable grasping,” in CVPR, 2025, pp. 12 209–12 219.
[65] Y. Fan, Q. Yang, K. Wang, H. Zhou, Y. Li, H. Feng, E. Ding, Y. Wu,
and J. Wang, “Re-hold: Video hand object interaction reenactment via
adaptive layout-instructed diffusion model,” in CVPR, 2025, pp. 17 550–
17 560.
[66] L. Wang, Z. Xia, T. Hu, P. Wang, P. Wei, Z. Zheng, M. Zhou, Y. Zhang,
and M. Gao, “Dreamactor-h1: High-fidelity human-product demonstra-
tion video generation via motion-designed diffusion transformers,” arXiv
preprint arXiv:2506.10568, 2025.
[67] K. Chen, C. Min, L. Zhang, S. Hampali, C. Keskin, and S. Sridhar,
“Foundhand: Large-scale domain-specific learning for controllable hand
image generation,” in CVPR, 2025, pp. 17 448–17 460.
[68] L. Zhao, X. Lu, R. Fan, S. K. Im, and L. Wang, “Gaussianhand: Real-
time 3d gaussian rendering for hand avatar animation,” IEEE TVCG,
2024.
[69] R. Hartley and A. Zisserman, Multiple view geometry in computer vision.
Cambridge university press, 2003.

<!-- page 13 -->
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021
13
[70] D. Lin, Y. Zhang, M. Li, Y. Liu, W. Jing, Q. Yan, Q. Wang, and
H. Zhang, “Omnihands: Towards robust 4d hand mesh recovery via a
versatile transformer,” arXiv preprint arXiv:2405.20330, 2024.
[71] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in CVPR, 2024, pp. 20 310–20 320.
[72] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,”
in CVPR, 2024, pp. 20 331–20 341.
[73] Z. Xu, Z. Huang, J. Cao, Y. Zhang, X. Cun, Q. Shuai, Y. Wang, L. Bao,
J. Li, and F. Tang, “Anchorcrafter: Animate cyberanchors saling your
products via human-object interacting video generation,” arXiv preprint
arXiv:2411.17383, 2024.
[74] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
arXiv preprint arXiv:1412.6980, 2014.
[75] U. Sara, M. Akter, and M. S. Uddin, “Image quality assessment through
fsim, ssim, mse and psnr—a comparative study,” Journal of Computer
and Communications, vol. 7, no. 3, pp. 8–18, 2019.
[76] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
CVPR, 2018, pp. 586–595.
[77] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
TIP, vol. 13, no. 4, pp. 600–612, 2004.
Hanhui Li received the Ph.D. degree in computer
software and theory from Sun Yat-sen University,
Guangzhou, China, in 2018, where he also received
the B.S. degree in computer science and technology
in 2012. He is now a research associate professor in
Shenzhen Campus of Sun Yat-sen University, China.
Before that, he was a research fellow in Nanyang
Technological University, Singapore, from 2019 to
2021. His research interests include image process-
ing, 3D content creation, and artificial intelligence
generated content.
Xuan Huang received the B.E. degree in Intel-
ligent Engineering from the School of Intelligent
Systems Engineering, Sun Yat-sen University, Shen-
zhen, Guangdong, China, in 2023. She is currently
pursuing the M.E. degree in Intelligent Engineering
at the same institution. She has published papers in
top-tier conferences, including NeurIPS and AAAI.
Her research interests focus on 3D reconstruction,
video generation, and their applications in digital
humans.
Wanquan Liu received the B.S. degree in Applied
Mathematics from Qufu Normal University, China,
in 1985, the M.S. degree in Control Theory and
Operation Research from Chinese Academy of Sci-
ence in 1988, and the Ph.D. degree in Electrical
Engineering from Shanghai Jiaotong University, in
1993. He once held ARC, U2000 and JSPS Fellow-
ships and secured over $2.4 million in research funds
from various sources. He is a Full Professor at the
School of Intelligent Systems Engineering, SYSU,
Guangzhou, China. His research interests include
large-scale pattern recognition, machine learning, and control systems.
Yuhao Cheng received the B.E. degree from the
Beijing University of Posts and Telecommunica-
tions, Beijing, China, in 2018, the First-Class Honor
B.E. degree from Queen Mary University of London,
London, U.K. in 2018, and the Master’s degree
from Beijing University of Posts and Telecommu-
nications, Beijing, China, in 2021. He is currently
working in Lenovo Research. His current research
interests include the Large Language Model, human-
centric computer vision, video representation, and so
on.
Long Chen received his B.S. degree from North-
eastern University, Shenyang, China, in 2016, and
the Master’s degree from Tianjin University, Tian-
jin, China, in 2019. He is currently the advisory
researcher in Lenovo Research. His research inter-
ests include computer vision, image generation, and
pattern recognition.
Yiqiang Yan is the Distinguished Researcher of
Lenovo Group, and has been engaged in the devel-
opment of smart devices for a long time. He has
led the development of many industry-first devices,
including wireless display devices, smart TVs, flex-
ible devices, smart retail, and tablet/laptop two-in-
one devices. These products he led to develop have
won more than 120 awards at CES and MWC, and
won the CCF Technology Invention Award in 2021.
Currently, he leads the PC innovation of comput-
ing architecture and human-computer interaction for
large language models. He has published 78 invention patents, including 19
U.S. patents.
Xiaodan Liang is currently a Professor at Sun Yat-
sen University. She was a postdoc researcher in the
machine learning department at Carnegie Mellon
University, working with Prof. Eric Xing, from 2016
to 2018. She received her Ph.D. degree from Sun
Yat-sen University in 2016. She has published sev-
eral cutting-edge projects on human-related analysis,
including human parsing, pedestrian detection, and
instance segmentation, 2D/3D human pose estima-
tion, and activity recognition.
Chenqiang Gao received the B.S. degree in com-
puter science from the China University of Geo-
sciences, Wuhan, China, in 2004 and the Ph.D.
degree in control science and engineering from the
Huazhong University of Science and Technology,
Wuhan, China, in 2009. In August 2009, he joined
the School of Communications and Information
Engineering, Chongqing University of Posts and
Telecommunications (CQUPT), Chongqing, China.
In September 2012, he joined the Informedia Group
with the School of Computer Science, Carnegie Mel-
lon University, Pittsburgh, PA, USA, working on multimedia event detection
(MED) and surveillance event detection (SED) until March 2014, when he
returned to CQUPT. In September 2023, he joined the School of Intelligent
Systems Engineering, Sun Yat-sen University, Shenzhen, Guangdong, China.
His research interests include image processing, infrared target detection,
action recognition, and event detection.
