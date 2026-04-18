<!-- page 1 -->
1
Adaptive 3D Gaussian Splatting Video Streaming:
Visual Saliency-Aware Tiling and
Meta-Learning-Based Bitrate Adaptation
Han Gong, Qiyue Li, Senior Member, IEEE, Jie Li, Member, IEEE, Zhi Liu, Senior Member, IEEE
Abstract—3D Gaussian splatting video (3DGS) streaming has
recently emerged as a research hotspot in both academia and
industry, owing to its impressive ability to deliver immersive
3D video experiences. However, research in this area is still
in its early stages, and several fundamental challenges—such
as tiling, quality assessment, and bitrate adaptation—require
further investigation. In this paper, we tackle these challenges
by proposing a comprehensive set of solutions. Specifically, we
propose an adaptive 3DGS tiling technique guided by saliency
analysis, which integrates both spatial and temporal features.
Each tile is encoded into versions possessing dedicated defor-
mation fields and multiple quality levels for adaptive selection.
We also introduce a novel quality assessment framework for
3DGS video that jointly evaluates spatial-domain degradation
in 3DGS representations during streaming and the quality of the
resulting 2D rendered images. Additionally, we develop a meta-
learning-based adaptive bitrate algorithm specifically tailored for
3DGS video streaming, achieving optimal performance across
varying network conditions. Extensive experiments demonstrate
that our proposed approaches significantly outperform state-of-
the-art methods. The source code will be made publicly available
upon acceptance of the paper.
Index Terms—Gaussian Splatting, video streaming, tiling,
saliency, meta-learning
I. INTRODUCTION
With the advancement of multimedia and communication
technologies, volumetric video, delivered through VR/AR/MR,
has seen rapid development and been widely adopted in
multiple fields. For example, in telemedicine, volumetric video
provides surgeons with high-quality visual feedback, allowing
remote execution of complex and urgent medical procedures
while also expanding to auditory and haptic feedback [1],
[2]. In 3D video conferencing, it facilitates true immersive
communication, allowing users to interact with one another in
virtual environments as they would in the real world [3].
Volumetric video representations are primarily divided into
explicit forms (such as point clouds and meshes) and im-
plicit representations (exemplified by Neural Radiance Fields,
NeRF) [4]. Explicit representations demonstrate superior ed-
itability, facilitating the implementation of various interactive
Han Gong and Qiyue Li is with the School of Electrical Engineering
and Automation, Hefei University of Technology, Hefei, China, and also
with the Engineering Technology Research Center of Industrial Automa-
tion of Anhui Province, Hefei, China (e-mail: han gong@mail.hfut.edu.cn,
liqiyue@mail.ustc.edu.cn)
Jie Li is with the School of Computer Science and Information Engineering,
Hefei University of Technology, Hefei, China (e-mail: lijie@hfut.edu.cn).
Zhi Liu is with The University of Electro-Communications, Tokyo, Japan
(e-mail: liu@ieee.org).
functions for volumetric video. However, both point clouds
and meshes exhibit significant visual quality limitations: point
clouds lose substantial textural and geometric details due to
their discrete nature, while meshes can only represent smooth
surfaces and fail to accurately render sharp or complex struc-
tures. In comparison, NeRF achieves photorealistic rendering
quality, but its implicit radiance field representation imposes
severe constraints on editing efficiency, rendering it impracti-
cal for real-time applications [5]. As an innovative volumetric
video representation, 3D Gaussian Splatting (3DGS) combines
NeRF-level scene realism with the editing flexibility inherent
in explicit representations [6]. By leveraging adaptive 3D
Gaussian primitives with anisotropic covariance, this approach
achieves real-time rendering through differentiable splatting
while maintaining geometric fidelity. The explicit yet param-
eterized representation allows selective manipulation of scene
elements at varying granularities, overcoming the opacity of
implicit neural representations. This hybrid paradigm not only
preserves high-frequency texture details comparable to NeRF’s
volumetric rendering, but also enables efficient geometric
transformations akin to point cloud operations [7]. For diverse
volumetric video applications, 3DGS-based volumetric video
undoubtedly represents the most promising research focus cur-
rently and its efficient streaming is one fundamental research
challenge in these volumetric video applications.
However, conventional volumetric video transmission meth-
ods cannot be directly applied to 3DGS videos due to fun-
damental incompatibilities between the unique data charac-
teristics of 3DGS and existing technical frameworks[8]. The
unstructured Gaussian distribution in 3DGS disrupts the struc-
tured spatial assumptions inherent to traditional point clouds
or meshes, while the high-dimensional attributes of each
Gaussian primitive far exceed the geometric color dimensions
of conventional volumetric videos [9]. Additionally, due to
the unique rendering mechanism of 3DGS, different Gaus-
sian primitives exhibit varying rendering weights, with those
possessing higher rendering weights significantly impacting
visual quality [10]. Compounded by novel distortion types
exemplified through Gaussian overlap artifacts and spherical
harmonic discontinuities arising from anisotropic rendering
characteristics that cannot be effectively quantified by con-
ventional quality metrics these fundamental discrepancies col-
lectively demand a dedicated compression transmission and
quality evaluation framework tailored to 3DGS video [11].
To enable specialized streaming for 3DGS video, re-
searchers have conducted studies and proposed several novel
arXiv:2507.14454v1  [cs.CV]  19 Jul 2025

<!-- page 2 -->
2
transmission systems [12] [13] [14]. Nevertheless, these sys-
tems lack sufficient consideration of the intrinsic characteris-
tics of 3DGS video, particularly in terms of effective visual
feature extraction, video quality assessment, and adaptive
bitrate (ABR) algorithm design. Quality assessment for 3DGS
must jointly evaluate geometric fidelity and rendered percep-
tual quality. Existing metrics focus on single representations,
failing to capture their combined impact [12] [8]. While recent
works have made progress in streaming static 3DGS models
through attribute quantization and Level of Detail (LOD)
control, these methods prove inadequate for dynamic 3DGS
video scenarios [15] [16]. The temporal coherence of Gaussian
attributes introduces unique compression dynamics that static
optimization frameworks cannot capture.
Conventional volumetric video transmission methods fail to
address the prohibitive per-frame data volumes when applied
to 3DGS video [8]. Although dynamic 3DGS reconstruction
techniques significantly compress frame-level data, decoding
latency constraints cause client-side playback stuttering [12].
The standard dual-mode transmission strategy—delivering ei-
ther pre-reconstructed models to circumvent decoding delays
or encoded data to reduce bandwidth—creates extreme dispar-
ities between minimum and maximum transmission demands
for 3DGS streaming [17]. This implementation gap arises
because 3DGS videos exhibit per-frame data volumes substan-
tially larger than traditional volumetric formats exemplified
by point cloud videos, fundamentally constraining the design
of network-agnostic ABR algorithms for 3DGS streaming
systems. Although meta-learning has demonstrated potential
for network adaptation in conventional video streaming [18]
[19], its application to 3DGS remains fundamentally limited
by the inherent complexity of Gaussian representations: The
dynamic interdependence between geometric attributes and
perceptual quality obstructs effective meta-knowledge trans-
fer, while sparse 3DGS training data prevents models from
capturing cross-scenario adaptation patterns.
The current challenges in 3DGS video streaming can be
summarized as follows:
• Visual Saliency Extraction in 3DGS Video: Similar
to conventional volumetric video, viewers consistently
prefer to focus on the most visually salient content.
However, as an emerging video format, 3DGS video
presents unique challenges for visual saliency extraction
due to its complex attribute composition and distinctive
rendering mechanism. Currently, there is a lack of effec-
tive methods specifically designed for saliency detection
in 3DGS content. Consequently, traditional approaches
commonly employed in volumetric video streaming -
such as viewpoint prediction and salient region segmen-
tation - prove inadequate when directly applied to 3DGS
video streaming scenarios.
• Quality Assessment Criteria for 3DGS Video: As
a novel video representation technology, 3DGS funda-
mentally differs from conventional formats in its scene
representation paradigm. Unlike point clouds, voxels, or
meshes that rely solely on 3D model geometry, or NeRF
that depends exclusively on rendered outputs without ex-
plicit models, 3DGS employs a hybrid approach. It com-
bines explicit Gaussian primitives with their splatting-
based rendered results to achieve comprehensive scene
representation. This unique characteristic necessitates a
dual evaluation framework that simultaneously considers
both the geometric fidelity of the underlying Gaussian
model and the perceptual quality of the final rendered
output. Consequently, establishing a dedicated quality
assessment standard becomes imperative for 3DGS video
transmission systems.
• Adaptive Bitrate Selection for 3DGS Video: Cur-
rent 3DGS streaming research predominantly focuses
on constructing transmission-optimized video sources,
while ABR algorithms specifically designed for 3DGS
content remain insufficiently investigated [20]. Although
LTS proposed a 3DGS-compatible ABR framework [8],
it lacks comprehensive user experience evaluation and
imposes excessive bandwidth requirements that limit gen-
eralization in common scenarios. Furthermore, reinforce-
ment learning-based ABR algorithms face fundamental
limitations when applied to 3DGS streaming: the sig-
nificant variation between minimum and maximum data
volumes in dual-version transmission systems creates
adaptation barriers, while inadequate video source data
severely compromises trained models’ generalization ca-
pabilities across diverse network conditions and deploy-
ment scenarios.
To address the aforementioned challenges in 3DGS video
streaming, we have designed a comprehensive 3DGS video
streaming system that effectively resolves these difficulties
and enables practical deployment of 3DGS video streaming
applications. Our system fundamentally advances the state-
of-the-art by simultaneously addressing three critical aspects
of 3DGS streaming: attention-aware content delivery, hybrid
quality evaluation, and network-adaptive transmission. The
principal contributions of our work can be summarized as
follows:
• We propose an adaptive tiling method for 3DGS video.
A saliency feature extraction network tailored for 3DGS
content is designed to estimate the visual importance
of different regions. Based on saliency variations across
uniform tiles, we adaptively merge them into new tiles
that better align with users’ visual attention.
• We develop a tile-based dynamic 3DGS encoding method
compatible with adaptive tiling. By analyzing motion
patterns across GoFs, tiles are categorized into static,
low-dynamic, and high-dynamic types, enabling efficient
deformation-based reconstruction. In parallel, we intro-
duce a saliency-aware quality tiering strategy that applies
Gaussian pruning with variable rates, ensuring high visual
quality under constrained bandwidth.
• We introduce a meta-reinforcement learning-based ABR
scheme for 3DGS video, supported by a hierarchical
perceptual Quality of Experience (QoE) model that re-
flects holistic perceptual quality. This approach enables
robust adaptation across diverse network conditions while
efficiently optimizing user experience.
• We conduct extensive simulations across diverse network

<!-- page 3 -->
3
conditions and 3DGS video datasets, and the results
verify the performance of the proposed schemes.
The rest of the paper is organized as follows: Section 2
critically analyzes related work in 3DGS volumetric video
streaming, and adaptive bitrate algorithms. Section 3 presents
our end-to-end streaming framework, integrating three inno-
vations: saliency-driven adaptive tiling, tile-based dynamic
3DGS encoding with multi-quality tiering, and meta-learning-
based QoE optimization. Each segmentation phase within our
framework is meticulously detailed in Section 4. Finally, Sec-
tion 5 validates the framework through extensive experiments
across diverse network conditions and 3DGS video datasets.
II. RELATED WORK
This section describes related work on 3DGS content
streaming and ABR schemes for volumetric video.
A. 3DGS Content Streaming
The high visual quality and inherent editability of 3DGS
content have made its efficient streaming a critical research
focus. Current 3DGS streaming approaches primarily bifur-
cate into two directions: static 3DGS content streaming and
dynamic content streaming.
In static 3DGS streaming research, L3GS [15] introduces
a customized training pipeline that generates controllable-
scale 3DGS models through iterative pruning and hierarchi-
cal freezing strategies. By segmenting scenes into seman-
tic objects and constructing a base layer plus enhancement
layer architecture, it achieves progressive transmission and
viewport-adaptive scheduling. PCGS [16] proposes a progres-
sive compression framework that incrementally decodes new
anchors via progressive masking while refining existing anchor
attributes through tri-plane quantization. Enhanced by context
modeling, this approach boosts entropy coding efficiency. Its
joint optimization of anchor quantity and quality achieves
compression performance comparable to single-rate methods
like HAC++, while supporting progressive enhancement under
dynamic network bandwidth. StreamGS [21] enables online
generalized reconstruction from unposed image streams by
predicting pixel-aligned 3D Gaussians frame-by-frame and
dynamically aggregating Gaussian sets using reliable inter-
frame matching information, eliminating dependencies on SfM
preprocessing or depth priors.
Dynamic 3DGS content has attracted significant research
attention due to its broader application prospects [20]. Among
existing dynamic 3DGS transmission frameworks, 3DGStream
[12] employs a Neural Transformation Cache (NTC) to model
motion attributes of 3D Gaussians, combining dynamic Gaus-
sian adaptive generation strategies to handle newly added
objects in scenes, achieving real-time online construction
of dynamic free-viewpoint videos. Dynamic 3DGS Stream-
ing [13] proposes a multi-frame bitrate allocation method
for dynamic 3DGS streaming, designing a model-driven al-
gorithm (MGA) and its adaptive variant (MGAA) through
rate-distortion optimization of geometric, spherical harmonic,
opacity, and transformation attributes. This approach maxi-
mizes rendering quality under dynamic network bandwidth
constraints. V3 [14] encodes dynamic 3D Gaussian attributes
into 2D video streams, leveraging hardware video codecs
for efficient compression and real-time mobile rendering. Its
innovations include a temporally consistent training strategy,
residual entropy loss, and temporal loss, significantly reducing
storage requirements. TGH [22] introduces a Temporal Gaus-
sian Hierarchy (TGH) for efficient long-form volumetric video
representation. By analyzing temporal redundancy across dy-
namic scene regions, it constructs hierarchical 4D Gaussian
primitives to hierarchically share static or slowly varying re-
gions, substantially reducing storage and computational costs.
IGS [23] proposes an Anchor-Driven Gaussian Motion Net-
work (AGM-Net), decoding inter-frame Gaussian deforma-
tions via multi-view optical flow feature projection and anchor
neighborhood interpolation. With key frame optimization and
maximum point count constraints, it effectively suppresses
error accumulation, enabling single-inference frame interpola-
tion. EvolvingGS [24] enhances dynamic scene reconstruction
quality through evolving Gaussian representations and efficient
compression strategies. By integrating local adjustments with
global optimization, it balances temporal coherence and detail
preservation, achieving significant improvements in dynamic
3DGS encoding efficiency.
While recent research has yielded preliminary investiga-
tions into 3DGS video streaming, the overwhelming majority
of efforts remain concentrated on reconstructing streamable
3DGS video sources. Critical aspects essential to traditional
volumetric video streaming—specifically, source processing
and optimized transmission—have received scant attention.
This work shifts focus toward efficiently transmitting exist-
ing 3DGS video content, thereby establishing complementary
advances to current methodologies and offering novel path-
ways for practical implementation of 3DGS video streaming
systems.
B. ABR Schemes for Volumetric Video
ABR algorithms for volumetric video inherit principles
from 2D video streaming but face amplified challenges due
to massive data volumes, 6-degree-of-freedom (6DoF) view-
ing constraints, and hybrid quality evaluation requirements.
Traditional ABR algorithms of 2D video relies on discrete
resolution-quality pairs, whereas volumetric video streaming
demands sophisticated adaptation strategies.
Point cloud video, as the most exemplary volumetric video
format, has received extensive research attention, with ABR
algorithms constituting a major research focus within this
domain. In particular, Wang et al. [25] proposed a novel
perspective-projection-based QoE model that holistically in-
tegrates view frustum, distance, occlusion, and screen reso-
lution, and designs a greedy-based rate adaptation algorithm
that transforms the optimization problem into a submodular
maximization task, achieving near-optimal QoE with low
complexity for tile-based point cloud video streaming. Li et
al. [26] studied rolling prediction-optimization-transmission
framework that leverages short-window prediction to mit-
igate long-term bandwidth/FoV prediction errors and em-
ploys a serial DRL solver SC-DDQN for real-time ABR

<!-- page 4 -->
4
Building 3DGS
Video
Fine-Grained
Even Tiling
Deformation
Field
Key Frame
GoF K
Get 2D Photos
Multiple GoFs
by Time
 Saliency
Extraction
Server
Adaptive
Tiling
Reference
Gaussian
Ellipsoid
Encoded Tiles
Counting All
Tiles
HTTP Inferface
Required
Tiles
Rendering
Weights
Target Frame
 Encode Different
Quality Level
Figure 1: Server-side architecture for 3DGS video streaming.
decisions, significantly improving QoE in point cloud video
streaming. Zhang et al. [27] designed a QoE-driven joint
network-computation adaptation framework that dynamically
adjusts per-patch download quality and super-resolution ratios
to balance bandwidth consumption, computational load, and
visual consistency in volumetric video streaming. Huang et
al. [28] proposed an AI-native DRL-based adaptive stream-
ing framework that dynamically selects lightweight encoder-
decoder models to balance network conditions, device com-
putation, and QoE. Liu et al. [29] proposed a client-cache-
assisted viewport adaptive streaming framework that leverages
long/mid/short-term viewport trajectory prediction to prioritize
caching of temporally repetitive tiles. A progressive frame
patching scheme with KKT-optimized tile rate allocation for
FoV-adaptive point cloud video streaming was proposed by
Zong et al. [30]. Shi et al. [31] studied a QoE-based viewpoint-
aware tiling and adaptive bitrate allocation scheme for V-PCC-
encoded volumetric video streaming.
Although 3DGS employs an explicit spatial representa-
tion analogous to point clouds, its underlying data structure
and rendering mechanisms are fundamentally distinct. Conse-
quently, quality assessment methodologies based on geometric
loss metrics prove inadequate for evaluating the perceptual
experience of 3DGS video. To address this critical disconnect,
we develop a dedicated ABR algorithm tailored for 3DGS
video streaming, establishing a viewer-centric QoE standard
that accurately reflects authentic user viewing experiences.
Emerging research has explored ABR algorithms for neural
volumetric video. For example, V²NeRF [32] unified the
implicit representation properties of NeRF with conventional
ABR mechanisms to address computational resource con-
tention between NeRF rendering and point cloud projec-
tion on GPUs. The study proposed a two-stage decoupled
ABR algorithm that independently optimized distinct resource
dimensions. LTS [8] introduced a novel ABR framework
specifically designed for 3DGS video, extending the static
LapisGS framework [33] to dynamic scenarios. Through joint
optimization of layer-tile-segment decision dimensions, the
implementation of ABR algorithms for practical 3DGS video
streaming was achieved.
While researchers had integrated conventional volumetric
video ABR methods into 3DGS streaming systems, these
approaches failed to account for user viewing behavior patterns
and the inherent visual characteristics of 3DGS represen-
tations. Furthermore, the prohibitively large per-frame data
volumes required for transmission imposed stringent band-
width requirements that severely limited practical deployment
in real-world scenarios. In contrast, our solution prioritizes
authentic viewing experience while ensuring broader applica-
bility under common network conditions.
III. OVERVIEW
This study proposes a novel streaming framework specifi-
cally designed for 3DGS video. Unlike prior 3DGS stream-
ing research primarily focused on reconstructing streamable
3DGS video sources, our work centers on optimizing the
transmission efficiency of existing 3DGS video content. Figure
1 illustrates the server-side architecture of our 3DGS video
streaming system, encompassing all processing stages from
source preparation to optimized delivery, thereby providing
a practical foundation for real-world 3DGS video streaming
applications. The system architecture is composed of three
major functional stages: (1) Saliency-driven Adaptive Tiling,
(2) Tile-based Dynamic 3DGS Encoding with Multi-quality
Tiering, and (3) Meta-learning-based QoE Optimization.
We temporally partition the video source into multiple
Groups of Frames (GoFs). Within each GoF, visual saliency
extraction guides the processing of video content into adap-
tive tiles aligned with users’ visual attention patterns. For
every tile, Gaussian deformation fields are selectively applied
according to its motion-intensity class determined by dis-
placement magnitude thresholds, while multi-quality tiering
is implemented based on rendering significance and saliency
importance. To bridge conventional volumetric video ABR

<!-- page 5 -->
5
algorithms with 3DGS streaming, we establish a dedicated
QoE metric that quantifies authentic viewing experiences dur-
ing 3DGS video playback. This framework integrates meta-
reinforcement learning to enhance cross-environment gener-
alizability. By strategically selecting tiles and their corre-
sponding quality levels to maximize QoE, we achieve optimal
viewing experiences under real-world conditions.
The following section elaborates in detail on each compo-
nent of our proposed 3DGS video streaming framework.
IV. DESIGN OF SYSTEM
A. Saliency-driven Adaptive Tiling
Volumetric video streaming often requires transmitting only
a subset of data due to its massive volume. A common
approach is to partition the model into smaller independent
tiles. During transmission, tiles are selected based on the
user’s FoV. This tiling strategy also applies to 3DGS videos.
However, uniform tiling faces a trade-off in tile count: too
few tiles reduce FoV alignment accuracy, leading to excessive
transmission beyond the user’s viewing range, while too many
tiles increase decoding overhead and encoding inefficiency.
Dynamically adjusting tile shapes based on video content
and user behavior can maximize content-FoV alignment while
balancing computational and communication resources. In this
paper, we propose a saliency-driven adaptive tiling method,
with the detailed workflow illustrated in Figure 2.
This method first partitions the initial 3DGS model into
small tiles per GoF, denoted as tj,t for the j-th tile in GoF t.
Within each tile tj,t, Gaussian primitives are sampled based
on rendering weights. The sampling probability psample (i) is
defined as:
psample (i) =
wi
PN
n=1 wn
,
(1)
wi = σi ·
p
det (Σi).
(2)
The parameter wi represents the rendering weight of the i-
th Gaussian primitive within tile tj,t, quantifying its relative
importance for the tile’s visual synthesis; σi denotes the
opacity of this primitive, governing its visibility contribution
during rendering; while Σi constitutes the covariance matrix
characterizing the primitive’s 3D spatial configuration through
shape anisotropy and orientation. The term
p
det(Σi) approx-
imates the effective volume of the primitive. We define the
visual importance of each primitive as wi = σi
p
det(Σi),
which jointly considers both opacity and spatial extent. Based
on this importance metric, we prioritize sampling primitives
with higher wi values. The resulting sampled tile is denoted
as Dj,t. For each Dj,t, we further design a spatial saliency
detection model and a temporal saliency detection model to
extract spatial and temporal saliency cues, respectively.
Spatial Saliency Detection Model. Each Gaussian primi-
tive generates view-dependent color through high-order spher-
ical harmonics (SH) coefficients. However, due to the contin-
uously changing user FoVs during viewing, it is impractical
to determine a primitive’s color from a specific FoV for
feature extraction. Therefore, we approximate the color of
each Gaussian primitive using its zero-order SH coefficients.
Specifically, c0
i,t,R, c0
i,t,G, and c0
i,t,B represent the zero-order
SH coefficients corresponding to the red, green, and blue color
channels, respectively, and are used as an approximation of the
primitive’s RGB color. Thus, the position and color attributes
of the i-th primitive in Dj,t are:
pi,t = {xi,t, yi,t, zi,t} ,
(3)
ai,t =

c0
i,t,R, c0
i,t,G, c0
i,t,B
	
.
(4)
Initial features for primitive i are extracted via a fully con-
nected (FC) layer:
fi,t = FC (pi,t ⊕ai,t) ,
(5)
where ⊕denotes concatenation. The Local Discrepancy
Catcher (LDC) module integrates spatial and color discrep-
ancies. Specifically, the neighborhood coding unit explicitly
encodes coordinate and color differences between Gaussian
primitive i and its neighboring primitive k in Dj,t. Grayscale
conversion is applied to zero-order SH coefficients to align
with human visual sensitivity:
di,t = 0.299 · c0
i,t,R + 0.587 · c0
i,t,G + 0.114 · c0
i,t,B.
(6)
The discrepancy between primitives i and k is then encoded
as:
dpk
i,t =MLP

pi,t ⊕pk
i,t ⊕
 pi,t −pk
i,t

⊕
pi,t −pk
i,t

⊕di,t ⊕dk
i,t ⊕
 di,t −dk
i,t

⊕
di,t −dk
i,t

, (7)
where pk
i,t and dk
i,t are the position and grayscale values of
neighbor k, respectively. The encoded discrepancy dpk
i,t is
concatenated with the neighbor’s initial feature f k
i,t, generating
an enhanced feature ˆ
f k
i,t.
ˆ
f k
i,t = dpk
i,t ⊕f k
i,t.
(8)
For Gaussian primitive i, all K neighboring primitives are
encoded, forming an enhanced feature set:
Ai,t =
n ˆ
f 1
i,t, ˆ
f 2
i,t, . . . , ˆ
f K
i,t
o
.
(9)
A shared Multilayer Perceptron (MLP) generates raw attention
scores for each enhanced neighbor feature ˆ
f k
i,t:
Sk
i,t = γ
 ˆ
f k
i,t, W

,
(10)
where W is shared across neighbors, and γ() denotes the
shared MLP function. Scores are normalized via Softmax:
σ
 Sk
i,t

=
exp
 Sk
i,t

PK
k=1 exp
 Sk
i,t
).
(11)
Neighboring features are aggregated via weighted summation:
ˆ
fi,t =
K
X
i=1
h ˆ
f k
i,t ∗σ
 Sk
i,t
i
.
(12)
To expand the receptive field, dilated residual blocks itera-
tively apply neighborhood coding and attention pooling twice,
enabling each primitive to capture contextual information from
up to K2 neighbors. This process is formalized as:
ˆ
F c
j,t = Rc
n
LDCc
h
ˆ
F c−1
j,t
i
, θu
io
,
(13)

<!-- page 6 -->
6
GoF t KeyFrame
Sampling
Sampling
Probability
GoF t+1 KeyFrame
Spatial Saliency
Feature
Sampling
Spatial Saliency
Feature
Spatial Saliency
Detection Model
Spatial Saliency
Detection Model
Temporal Saliency
Detection Model
Temporal Saliency
Feature
Feature Fusion
Co-located
Tile
Tile‘s Saliency
Score
Clustering
Similar Saliency
Tiles 
Adaptive Tiling
Figure 2: Saliency extraction and adaptive tiling.
ˆ
F 0
j,t = L1 (Fj,t) ,
(14)
where
ˆ
F c
j,t is the input feature from the (c −1)-th layer
and
ˆ
F c
j,t is the output feature from the c-th dilated residual
block, Rc denotes downsampling, and θu represents learnable
weights. Finally, encoded features are decoded through MLP,
upsampling, and FC layers to produce spatial saliency feature
F (t)
S .
Temporal
Saliency
Detection
Model.
For
temporal
saliency detection, we combine the temporal contrast layer
(TC) with the LDC module to capture dynamic changes
between consecutive GoFs. The TC layer extracts feature dif-
ferences between co-located tiles in adjacent GoFs, reflecting
the degree of variation from Dj,t−1 to Dj,t. The process begins
by extracting global feature Qc
t from corresponding tiles using
max pooling:
Qc
t = M (T c
t ) ,
(15)
Qc
t−1 = M
 T c
t−1

,
(16)
where T c
t−1 and T c
t represent inputs to the c-th TC layer for
Dj,t−1 and Dj,t, initialized as
ˆ
F 1
j,t−1 and ˆ
F 1
j,t outputs from the
first LDC layer, and M() denotes the max pooling operation.
The similarity between these global features is computed via
a shared MLP:
Ssim = γ
 Qc
t ⊕Qc
t−1

,
(17)
and converted to a saliency intensity score Os:
Os =
1
1 + exp(Ssim) + 1.
(18)
This score dynamically weights the input features of the
current frame, amplifying regions with significant temporal
changes:
ˆ
T c
t = Os ∗T c
t .
(19)
The weighted features
ˆ
TCc+1
t
are then processed through the
LDC module and downsampled via Rc:
ˆ
T c+1
t
= Rc
n
LDCc
h
ˆ
T c
t
i
, θu
io
.
(20)
Through iterative TC layers, dynamically salient regions are
progressively highlighted. Finally, the encoded temporal fea-
tures are decoded using MLP, upsampling, and FC layers,
generating temporal saliency feature F (t)
T .
Feature Fusion and Adaptive Tiling. Through the afore-
mentioned feature extraction modules, we obtain the spatial
saliency feature F (t)
S
and temporal saliency feature F (t)
T
for
GoF t. A shared MLP followed by Softmax activation com-
putes attention scores A(t)
S
and A(t)
T :
A(t)
S = σ
h
γ

F (t)
S , W1
i
,
(21)
A(t)
T = σ
h
γ

F (t)
T , W2
i
,
(22)
where σ() denotes the Softmax function, γ() represents the
shared MLP operation, and W1, W2 are learnable weight
matrices.
The comprehensive saliency feature F (t)
C
is derived via
attention-weighted fusion:
F (t)
C
= A(t)
S ⊙F (t)
S
+ A(t)
T ⊙F (t)
T ,
(23)
with ⊙indicating element-wise multiplication. The compre-
hensive saliency feature for tile tj,t is computed as:
¯F (t)
C,j = 1
Nj
X
i∈Bj
F (t)
C,i,
(24)
where Bj denotes the set of Gaussian primitives within tile
tj,t and Nj is the primitive count.
The aggregated comprehensive saliency feature
¯F (t)
C,j is
mapped to a saliency score via a two-layer perceptron:
S(t)
j
= Wb
h
ReLU

Wa ¯F (t)
C,j + ba
i
+ bb
(25)

<!-- page 7 -->
7
where Wa, Wb are learnable weights, and ba, bb are biases.
We optimize the model using a Smooth L1 loss between
the predicted score S(t)
j
and the ground truth saliency score
Score(t)
j :
L = 1
M
M
X
j=1
ℓ

S(t)
j
−Score(t)
j

(26)
ℓ(x) =
(
0.5x2
if |x| < 1
|x| −0.5
otherwise
(27)
The ground truth Score(t)
j
is computed from static saliency
detection and dynamic motion estimation as in [17].
After obtaining the saliency scores for all tiles tj,t, we
employ a clustering algorithm [34] to regroup tiles with similar
saliency values. High-saliency tiles in spatial proximity are
prioritized for aggregation into larger tiles. The clustering
process initiates by treating each tile as an individual cluster.
Subsequently, the similarity between every pair of clusters
is computed, and the two clusters exhibiting the highest
similarity are merged into a new cluster. This iterative merging
continues until the number of remaining clusters reaches a
predetermined threshold, resulting in adaptively aggregated
tiles optimized for saliency coherence and spatial continuity.
B. Tile-based Dynamic 3DGS Encoding and Multi-quality
Tiering
Existing point cloud tiling techniques can be applied to
frame-wise 3DGS video streaming due to its explicit repre-
sentation. However, the prohibitively large per-frame data vol-
ume severely limits widespread adoption, imposing stringent
network requirements [8]. For 3DGS video, dynamic 3DGS
techniques commonly model temporal evolution by varying
Gaussian attributes (position, shape, color) over time to capture
scene motion or deformation. Yet all current dynamic 3DGS
methods operate on entire 3DGS models and are incompatible
with tiled streaming [12]. To address this challenge, we design
a dynamic 3DGS tile encoding method.
Following the acquisition of saliency scores S(t)
j
for each
tile tj,t via the feature extraction network, aggregated adap-
tive tiles ta
m,t derive their saliency scores S(t)
m by averaging
S(t)
j
values across their constituent pre-aggregation tiles. For
temporal coherence analysis, each aggregated tile ta
m,t within
GoF t is matched to its corresponding tile ta
m,t+1 in GoF t+1
by selecting the tile with the closest saliency score. Displace-
ment vectors dm between matched tile pairs are computed,
and tiles are categorized into three motion classes based on
the magnitude ∥dm∥: Static tiles exhibit no motion across
GoFs and are treated as background components requiring no
deformation fields; low-dynamic tiles demonstrate coordinated
surface motions and undergo shared deformation field encod-
ing leveraging local motion consistency; high-dynamic tiles
preserve complex independent motions and utilize dedicated
deformation fields for optimal reconstruction.
For low-dynamic tiles Tlow , spatially adjacent tiles with
similar motion patterns are grouped into shared sets Sk ⊆Tlow
using cosine similarity:
s(m, n) =
⟨vm, vn⟩
∥vm∥∥vn∥,
(28)
vm = ∇S(t)
m ,
(29)
Deformation fields are constructed following 3DGStream-
ing’s methodology [12], where multi-resolution hash grid
features for Gaussian positions µi ∈ta
m,t are generated as:
h (µi) = Concat (h (µi; 0) , h (µi; 1) , . . . , h (µi; L −1)) ,
(30)
with L denoting hash resolution levels and h (µi; ℓ) rep-
resenting level-specific encoded features. This hierarchical
encoding preserves spatial dependencies while maintaining
computational efficiency.
A shallow MLP generates transformation parameters for
Gaussian primitives:
dµi, dqi = MLPm (h (µi)) ,
(31)
where dµi and dqi denote predicted positional and rotational
variations. Parameter reduction is achieved through shared
MLPs for low-dynamic tile groups, while high-dynamic tiles
undergo independent full-parameter optimization via dedicated
MLPs. The updated Gaussian primitive positions and rotations
are computed as:
µ′
i = µi + dµi,
(32)
q′
i = norm (qi) × norm (dqi) .
(33)
Here, norm denotes quaternion normalization and × repre-
sents quaternion multiplication.
We partition all frames within each GoF into key frames
and target frames. Key frames are designated as the first frame
of each GoF, with only their encoded tiles and corresponding
deformation fields transmitted. Target frames are subsequently
reconstructed client-side using these deformation fields.
To accommodate diverse network conditions, multi-quality
versions are generated for each tile. Our tile quality partition-
ing scheme is illustrated in Figure 3.
Primitives with low opacity and small ellipsoidal dimen-
sions contribute minimally to rendering quality compared to
high-opacity, large-scale Gaussians. Our core objective is to
identify and filter these non-essential primitives across quality
levels. Tile saliency further inform hierarchical construction.
The sampling probability psample (i) for each primitive in a
tile follows our established definition:
psample (i) =
wi
PN
n=1 wn
,
(34)
wi = σi ·
p
det (Σi).
(35)
Pruning rates are optimized using tile saliency weights.
High-saliency regions receive reduced pruning to align with
viewer attention patterns, while low-saliency areas undergo
more aggressive compression. Normalization is applied as
follows:
˜S(t)
m = S(t)
m −Smin
Smax −Smin
.
(36)

<!-- page 8 -->
8
Rendering
Weight
Tile of Different
Quality Levels
Counting All
Tiles
Saliency
Weight
Level 2
Level 3
Level 4
Level 5
Figure 3: Quality level partitioning.
The adjusted pruning rate per tile is computed as:
padj = pbase ·

1 −˜S(t)
m · α

,
(37)
where pbase denotes the base pruning rate and α ∈[0, 1]
regulates minimum pruning. Empirical results show that prun-
ing 30% of Gaussians via psample (i) preserves high visual
quality [14], whereas exceeding 50% causes severe degra-
dation [15]. We therefore implement a base pruning rate of
15% per layer as pbase, ensuring satisfactory quality for the
top three levels. The lowest quality level maintains at most
50% pruning to prevent abrupt quality collapse. We enforce a
minimum pruning rate of 8% across all tiles, ensuring overall
pruning remains below 30% at the lowest quality level. This
guarantees acceptable visual fidelity even for high-saliency
tiles at minimal quality settings. The pruning coefficient α
is derived inversely to satisfy this constraint.
C. Meta-learning-based QoE Maximization Scheme
In this section, we propose a meta-learning-based optimiza-
tion module for 3DGS video streaming. We first provide a
quantitative description of the QoE metric for 3DGS video
and formulate the ABR problem as a deep reinforcement
learning task solvable through meta-reinforcement learning
frameworks.
To address the limitations of existing QoE models in evalu-
ating 3DGS video, we redefine the user experience assessment
framework. Unlike conventional point cloud video where
geometric fidelity predominantly determines perceived qual-
ity—as rendered images derive directly from RGB attributes,
3DGS video rendering quality arises from the combined pa-
rameters of spherical harmonics, Gaussian positions and other
attributes, meaning that both geometric loss and rendering loss
must be jointly considered.
To address the limitations of existing QoE models in eval-
uating 3DGS video, we propose a layered perceptual QoE
model. For GoF t, the QoE is computed as:
QoE = λ
 αQgeo
t
+ (1 −α)Qrender
t

−µ · P d
t −σ · P f
t , (38)
where Qgeo
t
represents geometric quality, Qrender
t
denotes ren-
dering quality, P d
t indicates stall time, and P f
t specifies stall
frequency. λ, µ and σ are nonnegative weighting parameters
corresponding to the average video quality, stall time and stall
frequency, respectively. α is weighting factor determining the
contribution ratio of geometric quality to rendering quality.
The geometric quality Qgeo
t
is computed as:
Qgeo
t
=
Kt
X
k=1
R
X
r=1
[PSNRt,k,r · Φt,k] · xt,k,r,
(39)
where Kt represents the total number of tiles in the current
GoF t. The term PSNRt,k,r denotes the Peak Signal-to-Noise
Ratio (PSNR) between the rendered tile k at quality level
r and its reference version [35], while Φt,k represents the
spatial influence factor of tile k in GoF t, quantifying the
spatial importance weight of each tile in 3DGS video. Its core
objective is to dynamically adjust the contribution weights
of different tiles in QoE computation by integrating visual
salience and viewpoint visibility, thereby accurately reflecting
how spatial positions in 3DGS rendering impact perceptual
quality. This mechanism ensures that tiles with higher visibility
or stronger visual prominence are assigned greater weights in
the QoE calculation, while occluded or peripheral tiles are
downweighted. The Φt,k is calculated as:
Φt,k =
1
1 + e−γ·st,k · vt,k.
(40)
Here, st,k reflects the salience weight of tile k, determined by
its visual prominence in the viewport, and vt,k represents its
visibility ratio, computed as the intersection area between the
viewport and tile k normalized by the tile’s total area. The

<!-- page 9 -->
9
parameter γ scales the salience weight, and xt,k,r is a binary
selection variable constrained by:
R
X
r=1
xt,k,r = 1, xt,k,r ∈[0, 1],
(41)
where xt,k,r = 1 indicates the selection of the r quality level
for tile k.
By predicting the user’s viewport in advance, we can obtain
the next frame’s viewpoint and render the corresponding tiles.
Since neural rendering cannot precisely define the perceptual
quality of individual tiles, the final rendered image results from
the combined contributions of multiple overlapping tiles. After
calculating the rendering loss across all stacked tiles, we define
the perceptual quality of a single tile through occlusion weight.
For rendering quality Qrender
t
, we define it as:
Qrender
t
=
Kt
X
k=1
R
X
r=1
[SSIMfov · Φt,k · Ψt,k] · xt,k,r,
(42)
where SSIMfov measures the structural similarity of the
rendered image within the field of view. The occlusion weight
Ψt,k quantifies the contribution of tile k in GoF t based
on its visibility and overlap with other tiles. The occlusion
attenuation factor Ψt,k is computed as:
Ψt,k =
1
Nt,k
X
i∈Ct,k
αi · e−β·
dt,k
dmax ,
(43)
with Nt,k being the number of Gaussian primitives in tile k,
Ct,k the set of all Gaussian primitives in tile k, and dt,k the
Euclidean distance from center of tile k to the viewpoint. The
parameter dmax denotes the maximum distance of tile in the
current viewport, and β is attenuation coefficient.
While encoding with deformation fields can significantly
reduce the excessive data volume caused by frame-by-frame
transmission of 3DGS models, even the current state-of-the-
art dynamic Gaussian reconstruction methods require 33 ms
to decode a single tile per target frame, severely impacting
user viewing experiences in video streaming. However, pre-
transmitting all frames is impractical because each Gaussian
primitive in 3DGS requires as many as 48 parameters for color
representation, whereas each point in point clouds inherently
contains only 6 parameters (3 for position and 3 for RGB).
Consequently, the per-frame size of 3DGS video far exceeds
that of point cloud video, making frame-by-frame transmission
infeasible. Therefore, we define two transmission modes for
each tile: (1) Encoded tiles: Tiles encoded using deformation
fields and decoded client-side after transmission. (2) Re-
constructed tiles: Tiles bypass client-side processing entirely
by pre-reconstructing geometric details through deformation
fields and transmitting only finalized decoded data. For each
GoF t, we formalize its decoding time T D
t
as:
T D
t
=
PKt
k et,k × PR
r (φt,k,r × xt,k,r × ft,k)
C
,
(44)
where et,k ∈{0, 1} indicates the transmission mode of tile
k (1: encoded mode, 0: reconstructed mode). φt,k,r denotes
decoding time per tile, linearly proportional to its Gaussian
primitive count. ft,k = I (vt,k > 0) acts as a viewport indicator
function. C represents the client’s CPU core count enabling
parallel decoding.
The total transmission time T S
t
comprises encoded tile
transmission T E
t
and reconstructed tile transmission T R
t :
T S
t = T R
t + T E
t ,
(45)
with each component calculated as:
T R
t =
PKt
k=1
h
(1 −et,k) × PR
r=1

SE
t,k,r × xt,k,r × ft,k
i
Bt
,
(46)
T E
t =
PKt
k=1
h
et,k × PR
r=1

SR
t,k,r × xt,k,r × ft,k
i
Bt
, (47)
where SE
t,k,r
and SR
t,k,r
denote the data sizes of en-
coded/reconstructed tiles respectively, and Bt is the available
network bandwidth.
The end-to-end latency from client request to buffer readi-
ness is:
T U
t = T E
t + max
 T R
t , T D
t

.
(48)
This formulation reflects the pipelined transmission strategy:
encoded tiles are transmitted first while reconstructed tiles and
decoding operations proceed concurrently.
The stall time P d
t and stall frequency P f
t are derived from
playback buffer dynamics:
P d
t =
 T U
t −T I −Lt−1

+ ,
(49)
P f
t =

1, T U
t −T I −Lt−1 > 0
0, else
,
(50)
where T I is the GoF playback duration and Lt−1 represents
the previous buffer occupancy.
The dynamic nature of 3DGS video streaming, character-
ized by viewport-dependent rendering, heterogeneous network
conditions, and computational constraints, necessitates a prin-
cipled framework for joint bitrate adaptation and transmis-
sion mode optimization. We formulate this challenge as a
meta-reinforcement learning (meta-RL) task, where an agent
iteratively interacts with the streaming environment to learn
a policy that maximizes the expected long-term QoE. At
each decision epoch, the agent observes a state encoding
viewport dynamics, network bandwidth variability, and client-
side resource utilization. It then selects actions that jointly de-
termine two critical parameters: the encoding mode (encoded
or reconstructed) and the quality level for each spatial tile.
The state vector integrates spatiotemporal viewport dynam-
ics, network conditions, and content characteristics to guide
adaptive decision-making. It is formally defined as:
st =

⃗vt,⃗st, ⃗Bt, Lt−1, Ct,
X
et,kNt,k

,
(51)
where ⃗vt captures the 3D trajectory of the viewport center over
five consecutive frames, directly linked to the tile visibility
ratio vt,k. The saliency vector ⃗st provides per-tile visual
prominence weights st,k, which modulate the spatial influence
factor Φt,k in the QoE model. Network status −→
Bt characterizes
bandwidth availability through its instantaneous value and

<!-- page 10 -->
10
temporal variance, governing the denominator in transmission
time equations T E
t
and T R
t . Buffer occupancy Lt−1 serves as
a critical constraint for stall time calculation P d
t , while the
content descriptor Ct encodes scene-specific dynamics affect-
ing Gaussian primitive density Nt,k. The decoding load term
P et,kNt,k quantifies computational overhead by aggregating
Gaussian primitives in encoded tiles, directly influencing T D
t .
The action space comprises two interdependent decisions:
transmission mode selection and quality level allocation. Each
tile’s transmission mode et,k ∈{0, 1} determines whether
it is encoded (1) or reconstructed (0), subject to the global
bandwidth constraint:
X 
et,kSE
t,k,r + (1 −et,k) SR
t,k,r

xt,k,r ≤BtT I.
(52)
Concurrently, the quality selection variable xt,k,r ∈{0, 1}
adheres to the exclusivity constraint PR
r=1 xt,k,r = 1, ensuring
exactly one quality level is chosen per tile. This dual-action
structure enables joint optimization of bandwidth utilization
and rendering quality.
To address quality fluctuations between Groups of Frames
(GoFs) caused by keyframe-based 3DGS model transitions,
we enhance the QoE-driven reward function with a temporal
smoothness penalty. The revised reward formulation integrates
geometric consistency and rendering continuity across consec-
utive GoFs:
rt = λ (αQgeo + (1 −α)Qrender) −µP d
t −σP f
t −ηSt, (53)
St = δ ·
Qt
geo −Qt−1
geo

2 + (1 −δ) ·
Qt
render −Qt−1
render

2 .
(54)
The smoothness term St penalizes abrupt quality variations
through two components, Qt
geo and Qt
render denote the geomet-
ric and rendering quality metrics of GoF t, respectively. The
balance factor δ ∈[0, 1] adapts to scene dynamics, prioritizing
geometric stability for fast-motion sequences and rendering
consistency for viewport rotations. The unified formulation
preserves the core QoE optimization objectives while explic-
itly mitigating inter-GoF artifacts inherent to 3DGS streaming
architectures.
To address the scarcity of high-quality 3DGS training data
and enable robust generalization across diverse streaming sce-
narios, we integrate model-agnostic meta-learning (MAML)
into the reinforcement learning framework. During meta-
training, the agent is exposed to a distribution of tasks Ti,
each defined by a unique combination of bandwidth profiles
and content dynamics. For each task, the policy learns to
optimize the composite reward rt while adapting to two critical
dimensions: (1) Bandwidth variability: Simulated throughput
fluctuations mimic real-world network conditions, requiring
dynamic trade-offs between encoded and reconstructed tiles.
(2) Content Heterogeneity: Scene-specific Gaussian distribu-
tions demand adaptive spatial weighting of Φt,k and Ψt,k.
The meta-learning objective trains an initial policy πθ that
can rapidly adapt to unseen tasks with minimal fine-tuning
episodes. This is achieved through bi-level optimization:
Inner loop: For task Ti, perform gradient updates on a
support set of streaming trajectories to minimize:
Li
inner = −E(s,a,r)∼Dsupp
" H
X
τ=0
γτrτ
#
+ξ·KL (πθ′||πθ) , (55)
where γ is the discount factor and ξ regularizes policy diver-
gence.
Outer loop: Update the meta-policy θ by evaluating adapted
policies πθ′ on query sets from all tasks:
θ∗= arg min
θ
X
Ti
Li
outer (πθ′) ,
(56)
θ′ = θ −α∇θLi
inner .
(57)
To encode task-specific characteristics, we introduce a learn-
able embedding zT
= Enc

Ct ⊕⃗Bt

, where Ct captures
scene dynamics via Gaussian centroid displacements obtained
from each tile’s deformation field, and ⃗Bt encodes band-
width statistics. This embedding modulates the policy network
through feature-wise linear modulation (FiLM) layers:
FiLM(h) = β (zT ) ⊙h + γ (zT ) ,
(58)
where h denotes hidden layer activations, and β, γ are
generated by multi-layer perceptrons. Crucially, the QoE
weighting parameters λ, µ, σ, η are dynamically generated as
MLP (zT ) rather than fixed, allowing automatic prioritization
of geometric fidelity, rendering quality, stall avoidance, and
temporal smoothness based on current task requirements. For
instance, in bandwidth-constrained scenarios, the meta-policy
learns to increase µ (stall penalty weight) while decreasing λ
(quality emphasis), whereas viewport-unstable tasks upweight
η (smoothness penalty).
V. EXPERIMENT
A. Experimental Setup
Video Source: The 3DGS video sources in our experi-
ments comprise selected sequences from the publicly available
HiFi4G [36] and DNA-Randering [37] datasets. To enhance
the cross-scenario generalizability of the meta-learning algo-
rithm and improve controllability over user viewing behavior
by establishing more consistent head trajectories, we cap-
tured additional video sequences featuring complex scenar-
ios including multi-person interactions (Figures 4a and 4b),
occlusion situations (Figure 4c), and narrative-rich environ-
ments with extended durations (Figure 4d). All sequences
undergo preprocessing through our proposed reconstruction
and tiling framework. Videos were divided into constrained
(CS)/unconstrained (UC) sequences based on viewing perspec-
tive restrictions, as this classification directly impacts view-
point prediction accuracy and transmission decision-making.
Additionally, sequences were labeled as high-dynamic (HD)
or low-dynamic (LD) according to subject motion amplitude
, as dynamic sequences exhibit more pronounced quality
fluctuations across Group-of-Frames (GoFs) and increased risk
of rendering artifacts. All videos were standardized to 30
Hz frame rate and encoded into five quality versions via

<!-- page 11 -->
11
(a)
(b)
(c)
(d)
Figure 4: Partial sequence demonstrations from our dataset.
multi-quality encoding, alongside two transmission versions
(encoded vs. reconstructed tiles).
FoV Trace: The processed 3DGS video sequences were
imported into Meta Quest 3 headsets via the Unity environ-
ment, with 50 participants recruited to view these sequences
through VR devices while their head movement trajectories
were systematically recorded. Building upon this empirical
dataset, we refined our prior viewpoint prediction framework
[38] to align with the unique characteristics of 3DGS video
content. This adaptation yielded predicted FoV traces for each
sequence, encompassing spatial viewpoint coordinates (x, y,
z) and detailed head orientation parameters (pitch, yaw, roll).
These comprehensive viewpoint metrics serve as critical inputs
for optimizing tile selection and quality-level allocation during
the streaming process, ensuring prioritized transmission of
regions within predicted visual attention zones.
Bandwidth Trace: To rigorously evaluate the meta-learning
algorithm’s adaptability across heterogeneous network envi-
ronments, we employed diverse real-world bandwidth pat-
terns sampled from established 4G network trace datasets
[39][40] and 5G network trace dataset [41]. These traces
were systematically categorized into four distinct communi-
cation scenarios based on their peak bandwidth values and
temporal stability characteristics: Standard 4G environment
(Std4G) with sustained bandwidth ranging from 35 Mbps to
90 Mbps, Extreme 4G environment (Ext4G) exhibiting volatile
fluctuations between 0 Mbps and 150 Mbps, Standard 5G
environment (Std5G) maintaining stable connections from 150
Mbps to 600 Mbps, and Extreme 5G environment (Ext5G)
demonstrating highly erratic patterns spanning 0 Mbps to 1200
Mbps.
level 1
level 2
level 3
level 4
level 5
Quality Levels
20
22
24
26
28
30
32
PSNR (dB)
PSNR Comparison for Different Sampling Methods
Sampling Methods
Ours
URS
VS
FPS
LPS
IDIS
GS
RS
(a)
0.7
0.5
0.3
0.1
0.05
DaCVV
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
IFMI
IFMI Comparison for Different Sampling Methods
Sampling Methods
Ours
URS
VS
FPS
LPS
IDIS
GS
RS
(b)
Figure 5: Comparison of different sampling and multi-quality
compression methods.
B. Experimental Results
Sampling and multi-quality compression method: To
validate the effectiveness of our rendering-weight-aware sam-
pling and compression scheme in preserving perceptually
critical Gaussian primitives, comprehensive comparisons were
conducted against multiple baseline sampling methods through
quantitative evaluation of video quality and inter-frame map-
ping intensity (IFMI) [38]. Video quality assessment employs
the PSNR between rendered 2D frames and corresponding
ground truth training images under identical viewpoints. For
IFMI analysis, temporal coherence across GoFs was quantified
by measuring sequence consistency under varying distance
and color variation value (DaCVV) thresholds. Evaluated
baselines encompass uniform random sampling (URS) [42],
farthest point sampling (FPS) [43], local details preservation
sampling (LPS) [38], voxel sampling (VS) [44], inverse den-
sity importance sampling (IDIS) [45], geometric sampling
(GS), and random sampling (RS). Figure 5a demonstrates
the PSNR degradation across different quality levels of tile
compression, where each quality level corresponds to fixed
compression ratios. Our method maintains superior rendering
quality preservation during compression, particularly under ag-
gressive compression ratios. This resilience stems from our se-
lective retention of Gaussian primitives with higher rendering
significance, effectively mitigating abrupt visual degradation
when primitive counts fall below critical thresholds required
for coherent scene representation. Figure 5b illustrates the
IFMI variations under different DaCVV thresholds across sam-
pling methods. Our approach achieves substantially improved

<!-- page 12 -->
12
CS
UC
HD
LD
Video Characteristics
3.0
3.5
4.0
4.5
5.0
QoE Values
QoE Comparison for Different Tiling Methods
Tiling Strategy
Ours
AT
45T
30T
16T
NT
(a)
CS
UC
HD
LD
Video Characteristics
0.3
0.4
0.5
0.6
0.7
0.8
0.9
FoV Matching Rates
Matching Rates of User Observation Area for Different Tiling Methods
Tiling Strategy
Ours
AT
45T
30T
16T
NT
(b)
Figure 6: Comparison of different tiling methods.
Std4G
Ext4G
Std5G
Ext5G
Network Scenarios
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
QoE Values
QoE Comparison for Different ABR Algorithm using Hybrid Training Dataset
Algorithm
Ours
POT
QDA
HST
RLA
(a)
Std4G
Ext4G
Std5G
Ext5G
Network Scenarios
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
QoE Values
QoE Comparison for Different ABR Algorithm using Only Std5G Dataset
Algorithm
Ours
POT
QDA
HST
RLA
(b)
Std4G
Ext4G
Std5G
Ext5G
Network Scenarios
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
QoE Values
QoE Comparison for Different ABR Algorithm using 20% Training Dataset
Algorithm
Ours
POT
QDA
HST
RLA
(c)
Std4G
Ext4G
Std5G
Ext5G
Network Scenarios
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
QoE Values
QoE Comparison for Different ABR Algorithm using 50% Training Dataset
Algorithm
Ours
POT
QDA
HST
RLA
(d)
Figure 7: Comparison of different ABR algorithms.
temporal coherence compared to alternatives, attributable to
the inherent characteristics of 3DGS. Unlike point clouds that
maintain geometrically consistent distributions, 3DGS primi-
tives exhibit stochastic spatial arrangements optimized purely
for rendering fidelity. High-weight Gaussians critical for view
synthesis recur more persistently across frames, enabling our
method to maintain superior GoF consistency through targeted
primitive retention. This mechanism proves particularly crucial
given 3DGS’s rendering-oriented primitive distribution, where
conventional geometry-preserving sampling strategies fail to
capture perceptually vital elements.
Tiling method: To validate the efficacy of our adaptive
tiling scheme for 3DGS video streaming, we compare our
method with the adaptive tiling approach (AT) proposed in
[17]. The baseline AT method employs fast point feature
histograms (FPFH) to evaluate spatial saliency across tiles,
clustering regions based on fused saliency scores. Additionally,
we benchmark against uniform tiling configurations, including
45T (dividing the video into 45 uniform tiles), 30T (30 uniform
tiles), 16T (16 uniform tiles), and NT (No Tiling, transmitting
the entire video). Evaluations are conducted under Standard
5G (Std5G) network conditions, with QoE metrics and FoV
matching rates as key performance indicators. As illustrated
in Figure 6, our method achieves superior QoE and FoV
alignment across all datasets. Excessive tiling granularity (e.g.,
45T) introduces significant computational overhead, while
insufficient partitioning (e.g., 16T or NT) increases redundant
data transmission. Unlike AT, which predominantly relies on
Gaussian spatial distributions for saliency estimation, our tiling
scheme incorporates comprehensive video feature analysis

<!-- page 13 -->
13
during the partitioning process. This enables precise adaptation
to real-world user FoV patterns, ensuring optimal resource
allocation and perceptual quality.
Adaptive bitrate algorithm: To validate the superior gener-
alization capability and few-shot learning performance of our
meta-learning-based ABR algorithm, we compare against four
state-of-the-art point cloud video ABR baselines: POT[26],
QDA[46], HST[17], and RLA[47]. These baselines were
adapted with modified configurations to accommodate 3DGS
video streaming. Figure 7a demonstrates that our model
achieves the highest QoE performance when generalized to in-
dividual network environments using a hybrid training dataset
(Std4G + Ext4G + Std5G + Ext5G). Our method maintains
superior QoE across disparate network conditions, attributable
to meta-learning’s ability to extract cross-environment univer-
sal policies while avoiding overfitting to task-specific noise.
To evaluate cross-environment knowledge transferability, we
trained the model exclusively on Std5G data and tested its per-
formance under other network conditions (Figure 7b). While
performance degrades in drastically different 4G environ-
ments, our method still outperforms all baselines, confirming
meta-learning’s efficacy in knowledge distillation. Figures 7c
and 7d illustrate performance under limited training data.
Compared to severe performance degradation observed in
baseline methods with insufficient data, our approach retains
robust effectiveness in few-shot scenarios. For instance, using
only 20% of the Std5G training data achieves 84.9% of the
full-data performance, while 50% training data reaches 94.3%
efficacy—significantly surpassing all comparative methods.
This underscores our algorithm’s exceptional data efficiency
and adaptability to resource-constrained scenarios.
VI. CONCLUSION
This study proposes a novel 3DGS video streaming frame-
work that unlocks the practical potential of volumetric video
transmission through three integrated innovations: a saliency-
driven adaptive tiling mechanism employing spatiotemporal
feature fusion and clustering algorithms to aggregate primi-
tive blocks into viewport-optimized irregular tiles; a motion-
categorized dynamic encoding scheme classifying tiles into
static, low-dynamic, and high-dynamic types with correspond-
ing shared or dedicated deformation fields, coupled with
saliency-weighted multi-quality generation via adaptive Gaus-
sian pruning; and a meta-reinforcement learning ABR con-
troller incorporating 3DGS-specific QoE modeling for cross-
environment generalization. Collectively, these components
establish an end-to-end pipeline from source processing to op-
timized delivery, resolving critical barriers in practical 3DGS
video deployment. The experimental results demonstrate the
superiority of our proposal over existing schemes.
REFERENCES
[1] M. R. Desselle, R. A. Brown, A. R. James, M. J. Midwinter, S. K.
Powell, and M. A. Woodruff, “Augmented and virtual reality in surgery,”
Computing in Science & Engineering, vol. 22, no. 3, pp. 18–26, 2020.
[2] Z. Liu, Q. Li, X. Chen, C. Wu, S. Ishihara, J. Li, and Y. Ji, “Point cloud
video streaming: Challenges and solutions,” IEEE Network, vol. 35,
no. 5, pp. 202–209, 2021.
[3] J. Jansen, S. Subramanyam, R. Bouqueau, G. Cernigliaro, M. M. Cabr´e,
F. P´erez, and P. Cesar, “A pipeline for multiparty volumetric video
conferencing: transmission of point clouds over low latency dash,” in
Proceedings of the 11th ACM Multimedia Systems Conference, 2020,
pp. 341–344.
[4] Y. Jin, K. Hu, J. Liu, F. Wang, and X. Liu, “From capture to display: A
survey on volumetric video,” arXiv preprint arXiv:2309.05658, 2023.
[5] Y.-J. Yuan, Y.-T. Sun, Y.-K. Lai, Y. Ma, R. Jia, and L. Gao, “Nerf-
editing: geometry editing of neural radiance fields,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 18 353–18 364.
[6] M. Ye, M. Danelljan, F. Yu, and L. Ke, “Gaussian grouping: Segment
and edit anything in 3d scenes,” in European Conference on Computer
Vision.
Springer, 2024, pp. 162–179.
[7] Y. Jiang, Z. Shen, Y. Hong, C. Guo, Y. Wu, Y. Zhang, J. Yu, and L. Xu,
“Robust dual gaussian splatting for immersive human-centric volumetric
videos,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1–15,
2024.
[8] Y.-C. Sun, Y. Shi, C.-T. Lee, M. Zhu, W. T. Ooi, Y. Liu, C.-Y. Huang, and
C.-H. Hsu, “Lts: A dash streaming system for dynamic multi-layer 3d
gaussian splatting scenes,” in Proceedings of the 16th ACM Multimedia
Systems Conference, 2025, pp. 136–147.
[9] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and
L. Gao, “Recent advances in 3d gaussian splatting,” Computational
Visual Media, pp. 1–30, 2024.
[10] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 719–21 728.
[11] P. Martin, A. Rodrigues, J. Ascenso, and M. P. Queluz, “Gs-qa: Com-
prehensive quality assessment benchmark for gaussian splatting view
synthesis,” arXiv preprint arXiv:2502.13196, 2025.
[12] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2024, pp. 20 675–20 685.
[13] Y.-C. Sun, Y. Shi, W. T. Ooi, C.-Y. Huang, and C.-H. Hsu, “Multi-
frame bitrate allocation of dynamic 3d gaussian splatting streaming over
dynamic networks,” in Proceedings of the 2024 SIGCOMM Workshop
on Emerging Multimedia Systems, 2024, pp. 1–7.
[14] P. Wang, Z. Zhang, L. Wang, K. Yao, S. Xie, J. Yu, M. Wu, and L. Xu,
“Vˆ 3: Viewing volumetric videos on mobiles via streamable 2d dynamic
gaussians,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp.
1–13, 2024.
[15] Y.-Z. Tsai, X. Zhang, Z. Li, and J. Chen, “L3gs: Layered 3d gaussian
splats for efficient 3d scene delivery,” arXiv preprint arXiv:2504.05517,
2025.
[16] Y. Chen, M. Li, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Pcgs:
Progressive compression of 3d gaussian splatting,” arXiv preprint
arXiv:2503.08511, 2025.
[17] J. Li, C. Zhang, Z. Liu, R. Hong, and H. Hu, “Optimal volumetric
video streaming with hybrid saliency based tiling,” IEEE Transactions
on Multimedia, 2022.
[18] W. Li, X. Li, Y. Xu, Y. Yang, and S. Lu, “Metaabr: A meta-learning
approach on adaptative bitrate selection for video streaming,” IEEE
Transactions on Mobile Computing, vol. 23, no. 3, pp. 2422–2437, 2023.
[19] A. Bentaleb, M. Lim, M. N. Akcay, A. C. Begen, and R. Zimmermann,
“Bitrate adaptation and guidance with meta reinforcement learning,”
IEEE Transactions on Mobile Computing, 2024.
[20] J. Zhu and H. Tang, “Dynamic scene reconstruction: Recent advance in
real-time rendering and streaming,” arXiv preprint arXiv:2503.08166,
2025.
[21] Y. Li, J. Wang, L. Chu, X. Li, S.-h. Kao, Y.-C. Chen, and Y. Lu,
“Streamgs: Online generalizable gaussian splatting reconstruction for
unposed image streams,” arXiv preprint arXiv:2503.06235, 2025.
[22] Z. Xu, Y. Xu, Z. Yu, S. Peng, J. Sun, H. Bao, and X. Zhou, “Repre-
senting long volumetric video with temporal gaussian hierarchy,” ACM
Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1–18, 2024.
[23] J. Yan, R. Peng, Z. Wang, L. Tang, J. Yang, J. Liang, J. Wu, and
R. Wang, “Instant gaussian stream: Fast and generalizable streaming
of dynamic scene reconstruction via gaussian splatting,” arXiv preprint
arXiv:2503.16979, 2025.
[24] C. Zhang, Y. Zhou, S. Wang, W. Li, D. Wang, Y. Xu, and S. Jiao,
“Evolvinggs: High-fidelity streamable volumetric video via evolving 3d
gaussian representation,” arXiv preprint arXiv:2503.05162, 2025.

<!-- page 14 -->
14
[25] L. Wang, C. Li, W. Dai, S. Li, J. Zou, and H. Xiong, “Qoe-driven
adaptive streaming for point clouds,” IEEE Transactions on Multimedia,
2022.
[26] J. Li, H. Wang, Z. Liu, P. Zhou, X. Chen, Q. Li, and R. Hong, “Toward
optimal real-time volumetric video streaming: A rolling optimization
and deep reinforcement learning based approach,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7870–
7883, 2023.
[27] A. Zhang, C. Wang, B. Han, and F. Qian, “Efficient volumetric video
streaming through super resolution,” in Proceedings of the 22nd Interna-
tional Workshop on Mobile Computing Systems and Applications, 2021,
pp. 106–111.
[28] Y. Huang, Y. Zhu, X. Qiao, X. Su, S. Dustdar, and P. Zhang, “Toward
holographic video communications: a promising ai-driven solution,”
IEEE Communications Magazine, vol. 60, no. 11, pp. 82–88, 2022.
[29] J. Liu, B. Zhu, F. Wang, Y. Jin, W. Zhang, Z. Xu, and S. Cui, “Cav3:
Cache-assisted viewport adaptive volumetric video streaming,” in 2023
IEEE Conference Virtual Reality and 3D User Interfaces (VR).
IEEE,
2023, pp. 173–183.
[30] T. Zong, Y. Mao, C. Li, Y. Liu, and Y. Wang, “Progressive frame
patching for fov-based point cloud video streaming,” IEEE Transactions
on Multimedia, 2025.
[31] Y. Shi, B. Clement, and W. T. Ooi, “Qv4: Qoe-based viewpoint-aware
v-pcc-encoded volumetric video streaming,” in Proceedings of the 15th
ACM Multimedia Systems Conference, 2024, pp. 144–154.
[32] J. Shi, M. Zhang, L. Shen, J. Liu, Y. Zhang, L. Pu, and J. Xu,
“Towards full-scene volumetric video streaming via spatially layered
representation and nerf generation,” in Proceedings of the 34th edition
of the Workshop on Network and Operating System Support for Digital
Audio and Video, 2024, pp. 22–28.
[33] Y. Shi, G. Morin, S. Gasparini, and W. T. Ooi, “Lapisgs: Layered
progressive 3d gaussian splatting for adaptive streaming,” arXiv preprint
arXiv:2408.14823, 2024.
[34] S. Pasupathi, V. Shanmuganathan, K. Madasamy, H. R. Yesudhas, and
M. Kim, “Trend analysis using agglomerative hierarchical clustering
approach for time series big data,” The Journal of Supercomputing,
vol. 77, no. 7, pp. 6505–6524, 2021.
[35] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A. Chou,
R. A. Cohen, M. Krivoku´ca, S. Lasserre, Z. Li et al., “Emerging mpeg
standards for point cloud compression,” IEEE Journal on Emerging and
Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 133–148,
2018.
[36] Y. Jiang, Z. Shen, P. Wang, Z. Su, Y. Hong, Y. Zhang, J. Yu, and
L. Xu, “Hifi4g: High-fidelity human performance rendering via compact
gaussian splatting,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2024, pp. 19 734–19 745.
[37] W. Cheng, R. Chen, S. Fan, W. Yin, K. Chen, Z. Cai, J. Wang, Y. Gao,
Z. Yu, Z. Lin et al., “Dna-rendering: A diverse neural actor repository for
high-fidelity human-centric rendering,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 19 982–19 993.
[38] J. Li, Z. Zhao, Q. Li, Z. Li, P. Y. Zhou, Z. Liu, H. Zhou, and Z. Li,
“Vpformer: Leveraging transformer with voxel integration for viewport
prediction in volumetric video,” ACM Transactions on Multimedia
Computing, Communications and Applications.
[39] D. Raca, J. J. Quinlan, A. H. Zahran, and C. J. Sreenan, “Beyond
throughput: A 4g lte dataset with channel and context metrics,” in
Proceedings of the 9th ACM multimedia systems conference, 2018, pp.
460–465.
[40] L. Mei, R. Hu, H. Cao, Y. Liu, Z. Han, F. Li, and J. Li, “Realtime
mobile bandwidth prediction using lstm neural network and bayesian
fusion,” Computer Networks, vol. 182, p. 107515, 2020.
[41] D. Raca, D. Leahy, C. J. Sreenan, and J. J. Quinlan, “Beyond throughput,
the next generation: A 5g dataset with channel and context metrics,” in
Proceedings of the 11th ACM multimedia systems conference, 2020, pp.
303–308.
[42] J. Li, Z. Li, Z. Liu, P. Zhou, R. Hong, Q. Li, and H. Hu, “Viewport
prediction for volumetric video streaming by exploring video saliency
and trajectory information,” IEEE Transactions on Circuits and Systems
for Video Technology, 2025.
[43] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical
feature learning on point sets in a metric space,” Advances in neural
information processing systems, vol. 30, 2017.
[44] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point cloud
based 3d object detection,” in Proceedings of the IEEE conference on
computer vision and pattern recognition, 2018, pp. 4490–4499.
[45] F. Groh, P. Wieschollek, and H. P. Lensch, “Flex-convolution: Million-
scale point-cloud learning beyond grid-worlds,” in Asian Conference on
Computer Vision.
Springer, 2018, pp. 105–122.
[46] L. Yu, T. Tillo, and J. Xiao, “Qoe-driven dynamic adaptive video
streaming strategy with future information,” IEEE Transactions on
Broadcasting, vol. 63, no. 3, pp. 523–534, 2017.
[47] N. T. Nguyen, L. Luu, P. L. Vo, S. T. T. Nguyen, C. T. Do, and N.-
T. Nguyen, “Reinforcement learning-based adaptation and scheduling
methods for multi-source dash,” Computer Science and Information
Systems, vol. 20, no. 1, pp. 157–173, 2023.
